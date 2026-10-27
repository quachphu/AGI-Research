        for term in Add.make_args(p.removeO()):
            co1, e1 = coeff_exp(term, t)
            pterms[e1] = pterms.get(e1, S.Zero) + co1

        k = S.One
        terms = {}
        pk = pterms

        while k*d < n:
            coeff = -S.NegativeOne**k/k
            for ex in pk:
                terms[ex] = terms.get(ex, S.Zero) + coeff*pk[ex]
            pk = mul(pk, pterms)
            k += S.One

        res = log(a) - b*log(cdir) + b*logx
        for ex in terms:
            res += terms[ex].cancel()*t**(ex)

        if a.is_negative and im(z) != 0:
            from sympy.functions.special.delta_functions import Heaviside
            for i, term in enumerate(z.lseries(t)):
                if not term.is_real or i == 5:
                    break
            if i < 5:
                coeff, _ = term.as_coeff_exponent(t)
                res += -2*I*pi*Heaviside(-im(coeff), 0)

        res = res.subs(t, x/cdir)
        return res + Order(x**n, x)

    def _eval_as_leading_term(self, x, logx, cdir):
        # NOTE
        # Refer https://github.com/sympy/sympy/pull/23592 for more information
        # on each of the following steps involved in this method.
        arg0 = self.args[0].together()

        # STEP 1
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg0.subs(x, cdir*t)

        # STEP 2
        try:
            c, e = z.leadterm(t, logx=logx, cdir=1)
        except ValueError:
            arg = arg0.as_leading_term(x, logx=logx, cdir=cdir)
            return log(arg)
        if c.has(t):
            c = c.subs(t, x/cdir)
            if e != 0:
                raise PoleError("Cannot expand %s around 0" % (self))
            return log(c)

        # STEP 3
        if c == S.One and e == S.Zero:
            return (arg0 - S.One).as_leading_term(x, logx=logx)

        # STEP 4
        res = log(c) - e*log(cdir)
        logx = log(x) if logx is None else logx
        res += e*logx

        # STEP 5
        if c.is_negative and im(z) != 0:
            from sympy.functions.special.delta_functions import Heaviside
            for i, term in enumerate(z.lseries(t)):
                if not term.is_real or i == 5:
                    break
            if i < 5:
                coeff, _ = term.as_coeff_exponent(t)
                res += -2*I*pi*Heaviside(-im(coeff), 0)
        return res


class LambertW(DefinedFunction):
    r"""
    The Lambert W function $W(z)$ is defined as the inverse
    function of $w \exp(w)$ [1]_.

    Explanation
    ===========

    In other words, the value of $W(z)$ is such that $z = W(z) \exp(W(z))$
    for any complex number $z$.  The Lambert W function is a multivalued
    function with infinitely many branches $W_k(z)$, indexed by
    $k \in \mathbb{Z}$.  Each branch gives a different solution $w$
    of the equation $z = w \exp(w)$.

    The Lambert W function has two partially real branches: the
    principal branch ($k = 0$) is real for real $z > -1/e$, and the
    $k = -1$ branch is real for $-1/e < z < 0$. All branches except
    $k = 0$ have a logarithmic singularity at $z = 0$.

    Examples
    ========

    >>> from sympy import LambertW
    >>> LambertW(1.2)
    0.635564016364870
    >>> LambertW(1.2, -1).n()
    -1.34747534407696 - 4.41624341514535*I
    >>> LambertW(-1).is_real
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lambert_W_function
    """
    _singularities = (-Pow(S.Exp1, -1, evaluate=False), S.ComplexInfinity)

    @classmethod
    def eval(cls, x, k=None):
        if k == S.Zero:
            return cls(x)
        elif k is None:
            k = S.Zero

        if k.is_zero:
            if x.is_zero:
                return S.Zero
            if x is S.Exp1:
                return S.One
            if x == -1/S.Exp1:
                return S.NegativeOne
            if x == -log(2)/2:
                return -log(2)
            if x == 2*log(2):
                return log(2)
            if x == -pi/2:
                return I*pi/2
            if x == exp(1 + S.Exp1):
                return S.Exp1
            if x is S.Infinity:
                return S.Infinity

        if fuzzy_not(k.is_zero):
            if x.is_zero:
                return S.NegativeInfinity
        if k is S.NegativeOne:
            if x == -pi/2:
                return -I*pi/2
            elif x == -1/S.Exp1:
                return S.NegativeOne
            elif x == -2*exp(-2):
                return -Integer(2)

    def fdiff(self, argindex=1):
        """
        Return the first derivative of this function.
        """
        x = self.args[0]

        if len(self.args) == 1:
            if argindex == 1:
                return LambertW(x)/(x*(1 + LambertW(x)))
        else:
            k = self.args[1]
            if argindex == 1:
                return LambertW(x, k)/(x*(1 + LambertW(x, k)))

        raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        x = self.args[0]
        if len(self.args) == 1:
            k = S.Zero
        else:
            k = self.args[1]
        if k.is_zero:
            if (x + 1/S.Exp1).is_positive:
                return True
            elif (x + 1/S.Exp1).is_nonpositive:
                return False
        elif (k + 1).is_zero:
            if x.is_negative and (x + 1/S.Exp1).is_positive:
                return True
            elif x.is_nonpositive or (x + 1/S.Exp1).is_nonnegative:
                return False
        elif fuzzy_not(k.is_zero) and fuzzy_not((k + 1).is_zero):
            if x.is_extended_real:
                return False

    def _eval_is_finite(self):
        return self.args[0].is_finite

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
        else:
            return s.is_algebraic

    def _eval_as_leading_term(self, x, logx, cdir):
        if len(self.args) == 1:
            arg = self.args[0]
            arg0 = arg.subs(x, 0).cancel()
            if not arg0.is_zero:
                return self.func(arg0)
            return arg.as_leading_term(x)

    def _eval_nseries(self, x, n, logx, cdir=0):
        if len(self.args) == 1:
            from sympy.functions.elementary.integers import ceiling
            from sympy.series.order import Order
            arg = self.args[0].nseries(x, n=n, logx=logx)
            lt = arg.as_leading_term(x, logx=logx)
            lte = 1
            if lt.is_Pow:
                lte = lt.exp
            if ceiling(n/lte) >= 1:
                s = Add(*[(-S.One)**(k - 1)*Integer(k)**(k - 2)/
                          factorial(k - 1)*arg**k for k in range(1, ceiling(n/lte))])
                s = expand_multinomial(s)
            else:
                s = S.Zero

            return s + Order(x**n, x)
        return super()._eval_nseries(x, n, logx)

    def _eval_is_zero(self):
        x = self.args[0]
        if len(self.args) == 1:
            return x.is_zero
        else:
            return fuzzy_and([x.is_zero, self.args[1].is_zero])


@cacheit
def _log_atan_table():
    return {
        # first quadrant only
        sqrt(3): pi / 3,
        1: pi / 4,
        sqrt(5 - 2 * sqrt(5)): pi / 5,
        sqrt(2) * sqrt(5 - sqrt(5)) / (1 + sqrt(5)): pi / 5,
        sqrt(5 + 2 * sqrt(5)): pi * Rational(2, 5),
        sqrt(2) * sqrt(sqrt(5) + 5) / (-1 + sqrt(5)): pi * Rational(2, 5),
        sqrt(3) / 3: pi / 6,
        sqrt(2) - 1: pi / 8,
        sqrt(2 - sqrt(2)) / sqrt(sqrt(2) + 2): pi / 8,
        sqrt(2) + 1: pi * Rational(3, 8),
        sqrt(sqrt(2) + 2) / sqrt(2 - sqrt(2)): pi * Rational(3, 8),
        sqrt(1 - 2 * sqrt(5) / 5): pi / 10,
        (-sqrt(2) + sqrt(10)) / (2 * sqrt(sqrt(5) + 5)): pi / 10,
        sqrt(1 + 2 * sqrt(5) / 5): pi * Rational(3, 10),
        (sqrt(2) + sqrt(10)) / (2 * sqrt(5 - sqrt(5))): pi * Rational(3, 10),
        2 - sqrt(3): pi / 12,
        (-1 + sqrt(3)) / (1 + sqrt(3)): pi / 12,
        2 + sqrt(3): pi * Rational(5, 12),
        (1 + sqrt(3)) / (-1 + sqrt(3)): pi * Rational(5, 12)
    }
<|endoftext|>from __future__ import annotations

from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (DefinedFunction, Derivative, ArgumentIndexError,
    AppliedUndef, expand_mul, PoleError)
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise

###############################################################################
######################### REAL and IMAGINARY PARTS ############################
###############################################################################


class re(DefinedFunction):
    """
    Returns real part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly
    more complicated expressions. If completely simplified result
    is needed then use ``Basic.as_real_imag()`` or perform complex
    expansion on instance of this function.

    Examples
    ========

    >>> from sympy import re, im, I, E, symbols
    >>> x, y = symbols('x y', real=True)
    >>> re(2*E)
    2*E
    >>> re(2*I + 17)
    17
    >>> re(2*I)
    0
    >>> re(im(x) + x*I + 2)
    2
    >>> re(5 + I + 2)
    7

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Real part of expression.

    See Also
    ========

    im
    """

    args: tuple[Expr]

    is_extended_real = True
    unbranched = True  # implicitly works on the projection to C
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return arg
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return S.Zero
        elif arg.is_Matrix:
            return arg.as_real_imag()[0]
        elif arg.is_Function and isinstance(arg, conjugate):
            return re(arg.args[0])
        else:

            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                elif not term.has(I) and term.is_extended_real:
                    excluded.append(term)
                else:
                    # Try to do some advanced expansion.  If
                    # impossible, don't try to do re(arg) again
                    # (because this is what we are trying to do now).
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[0])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) - im(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Returns the real number with a zero imaginary part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return re(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * im(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_im(self, arg, **kwargs):
        return self.args[0] - I*im(self.args[0])

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        # is_imaginary implies nonzero
        return fuzzy_or([self.args[0].is_imaginary, self.args[0].is_zero])

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True


class im(DefinedFunction):
    """
    Returns imaginary part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly more
    complicated expressions. If completely simplified result is needed then
    use ``Basic.as_real_imag()`` or perform complex expansion on instance of
    this function.

    Examples
    ========

    >>> from sympy import re, im, E, I
    >>> from sympy.abc import x, y
    >>> im(2*E)
    0
    >>> im(2*I + 17)
    2
    >>> im(x*I)
    re(x)
    >>> im(re(x) + y)
    im(y)
    >>> im(2 + 3*I)
    3

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Imaginary part of expression.

    See Also
    ========

    re
    """

    args: tuple[Expr]

    is_extended_real = True
    unbranched = True  # implicitly works on the projection to C
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return S.Zero
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return -I * arg
        elif arg.is_Matrix:
            return arg.as_real_imag()[1]
        elif arg.is_Function and isinstance(arg, conjugate):
            return -im(arg.args[0])
        else:
            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                    else:
                        excluded.append(coeff)
                elif term.has(I) or not term.is_extended_real:
                    # Try to do some advanced expansion.  If
                    # impossible, don't try to do im(arg) again
                    # (because this is what we are trying to do now).
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[1])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) + re(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Return the imaginary part with a zero real part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return im(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * re(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_re(self, arg, **kwargs):
        return -I*(self.args[0] - re(self.args[0]))

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True

###############################################################################
############### SIGN, ABSOLUTE VALUE, ARGUMENT and CONJUGATION ################
###############################################################################

class sign(DefinedFunction):
    """
    Returns the complex sign of an expression:

    Explanation
    ===========

    If the expression is real the sign will be:

        * $1$ if expression is positive
        * $0$ if expression is equal to zero
        * $-1$ if expression is negative

    If the expression is imaginary the sign will be:

        * $I$ if im(expression) is positive
        * $-I$ if im(expression) is negative

    Otherwise an unevaluated expression will be returned. When evaluated, the
    result (in general) will be ``cos(arg(expr)) + I*sin(arg(expr))``.

    Examples
    ========

    >>> from sympy import sign, I

    >>> sign(-1)
    -1
    >>> sign(0)
    0
    >>> sign(-3*I)
    -I
    >>> sign(1 + I)
    sign(1 + I)
    >>> _.evalf()
    0.707106781186548 + 0.707106781186548*I

    Parameters
    ==========

    arg : Expr
        Real or imaginary expression.

    Returns
    =======

    expr : Expr
        Complex sign of expression.

    See Also
    ========

    Abs, conjugate
    """

    is_complex = True
    _singularities = True

    def doit(self, **hints):
        s = super().doit()
        if s == self and self.args[0].is_zero is False:
            return self.args[0] / Abs(self.args[0])
        return s

    @classmethod
    def eval(cls, arg):
        # handle what we can
        if arg.is_Mul:
            c, args = arg.as_coeff_mul()
            unk = []
            s = sign(c)
            for a in args:
                if a.is_extended_negative:
                    s = -s
                elif a.is_extended_positive:
                    pass
                else:
                    if a.is_imaginary:
                        ai = im(a)
                        if ai.is_comparable:  # i.e. a = I*real
                            s *= I
                            if ai.is_extended_negative:
                                # can't use sign(ai) here since ai might not be
                                # a Number
                                s = -s
                        else:
                            unk.append(a)
                    else:
                        unk.append(a)
            if c is S.One and len(unk) == len(args):
                return None
            return s * cls(arg._new_rawargs(*unk))
        if arg is S.NaN:
            return S.NaN
        if arg.is_zero:  # it may be an Expr that is zero
            return S.Zero
        if arg.is_extended_positive:
            return S.One
        if arg.is_extended_negative:
            return S.NegativeOne
        if arg.is_Function:
            if isinstance(arg, sign):
                return arg
        if arg.is_imaginary:
            if arg.is_Pow and arg.exp is S.Half:
                # we catch this because non-trivial sqrt args are not expanded
                # e.g. sqrt(1-sqrt(2)) --x-->  to I*sqrt(sqrt(2) - 1)
                return I
            arg2 = -I * arg
            if arg2.is_extended_positive:
                return I
            if arg2.is_extended_negative:
                return -I

    def _eval_Abs(self):
        if fuzzy_not(self.args[0].is_zero):
            return S.One

    def _eval_conjugate(self):
        return sign(conjugate(self.args[0]))

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(self.args[0])
        elif self.args[0].is_imaginary:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(-I * self.args[0])

    def _eval_is_nonnegative(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_nonpositive(self):
        if self.args[0].is_nonpositive:
            return True

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_power(self, other):
        if (
            fuzzy_not(self.args[0].is_zero) and
            other.is_integer and
            other.is_even
        ):
            return S.One

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg0 = self.args[0]
        x0 = arg0.subs(x, 0)
        if x0 != 0:
            return self.func(x0)
        if cdir != 0:
            cdir = arg0.dir(x, cdir)
        return -S.One if re(cdir) < 0 else S.One

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((1, arg > 0), (-1, arg < 0), (0, True))

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return Heaviside(arg) * 2 - 1

    def _eval_rewrite_as_Abs(self, arg, **kwargs):
        return Piecewise((0, Eq(arg, 0)), (arg / Abs(arg), True))

    def _eval_simplify(self, **kwargs):
        return self.func(factor_terms(self.args[0]))  # XXX include doit?


class Abs(DefinedFunction):
    """
    Return the absolute value of the argument.

    Explanation
    ===========

    This is an extension of the built-in function ``abs()`` to accept symbolic
    values.  If you pass a SymPy expression to the built-in ``abs()``, it will
    pass it automatically to ``Abs()``.

    Examples
    ========

    >>> from sympy import Abs, Symbol, S, I
    >>> Abs(-1)
    1
    >>> x = Symbol('x', real=True)
    >>> Abs(-x)
    Abs(x)
    >>> Abs(x**2)
    x**2
    >>> abs(-x) # The Python built-in
    Abs(x)
    >>> Abs(3*x + 2*I)
    sqrt(9*x**2 + 4)
    >>> Abs(8*I)
    8

    Note that the Python built-in will return either an Expr or int depending on
    the argument::

        >>> type(abs(-1))
        <... 'int'>
        >>> type(abs(S.NegativeOne))
        <class 'sympy.core.numbers.One'>

    Abs will always return a SymPy object.

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Absolute value returned can be an expression or integer depending on
        input arg.

    See Also
    ========

    sign, conjugate
    """

    args: tuple[Expr]

    is_extended_real = True
    is_extended_negative = False
    is_extended_nonnegative = True
    unbranched = True
    _singularities = True  # non-holomorphic

    def fdiff(self, argindex=1):
        """
        Get the first derivative of the argument to Abs().

        """
        if argindex == 1:
            return sign(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.simplify.simplify import signsimp

        if hasattr(arg, '_eval_Abs'):
            obj = arg._eval_Abs()
            if obj is not None:
                return obj
        if not isinstance(arg, Expr):
            raise TypeError("Bad argument type for Abs(): %s" % type(arg))

        # handle what we can
        arg = signsimp(arg, evaluate=False)
        n, d = arg.as_numer_denom()
        if d.free_symbols and not n.free_symbols:
            return cls(n)/cls(d)

        if arg.is_Mul:
            known = []
            unk = []
            for t in arg.args:
                if t.is_Pow and t.exp.is_integer and t.exp.is_negative:
                    bnew = cls(t.base)
                    if isinstance(bnew, cls):
                        unk.append(t)
                    else:
                        known.append(Pow(bnew, t.exp))
                else:
                    tnew = cls(t)
                    if isinstance(tnew, cls):
                        unk.append(t)
                    else:
                        known.append(tnew)
            known = Mul(*known)
            unk = cls(Mul(*unk), evaluate=False) if unk else S.One
            return known*unk
        if arg is S.NaN:
            return S.NaN
        if arg is S.ComplexInfinity:
            return oo
        from sympy.functions.elementary.exponential import exp, log

        if arg.is_Pow:
            base, exponent = arg.as_base_exp()
            if base.is_extended_real:
                if exponent.is_integer:
                    if exponent.is_even:
                        return arg
                    if base is S.NegativeOne:
                        return S.One
                    return Abs(base)**exponent
                if base.is_extended_nonnegative:
                    return base**re(exponent)
                if base.is_extended_negative:
                    return (-base)**re(exponent)*exp(-pi*im(exponent))
                return
            elif not base.has(Symbol): # complex base
                # express base**exponent as exp(exponent*log(base))
                a, b = log(base).as_real_imag()
                z = a + I*b
                return exp(re(exponent*z))
        if isinstance(arg, exp):
            return exp(re(arg.args[0]))
        if isinstance(arg, AppliedUndef):
            if arg.is_positive:
                return arg
            elif arg.is_negative:
                return -arg
            return
        if arg.is_Add and arg.has(oo, S.NegativeInfinity):
            if any(a.is_infinite for a in arg.as_real_imag()):
                return oo
        if arg.is_zero:
            return S.Zero
        if arg.is_extended_nonnegative:
            return arg
        if arg.is_extended_nonpositive:
            return -arg
        if arg.is_imaginary:
            arg2 = -I * arg
            if arg2.is_extended_nonnegative:
                return arg2
        if arg.is_extended_real:
            return
        # reject result if all new conjugates are just wrappers around
        # an expression that was already in the arg
        conj = signsimp(arg.conjugate(), evaluate=False)
        new_conj = conj.atoms(conjugate) - arg.atoms(conjugate)
        if new_conj and all(arg.has(i.args[0]) for i in new_conj):
            return
        if arg != conj and arg != -conj:
            ignore = arg.atoms(Abs)
            abs_free_arg = arg.xreplace({i: Dummy(real=True) for i in ignore})
            unk = [a for a in abs_free_arg.free_symbols if a.is_extended_real is None]
            if not unk or not all(conj.has(conjugate(u)) for u in unk):
                return sqrt(expand_mul(arg*conj))

    def _eval_is_real(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_integer(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_integer

    def _eval_is_extended_nonzero(self):
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_zero(self):
        return self._args[0].is_zero

    def _eval_is_extended_positive(self):
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_rational(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_rational

    def _eval_is_even(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_even

    def _eval_is_odd(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_odd

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_power(self, exponent):
        if self.args[0].is_extended_real and exponent.is_integer:
            if exponent.is_even:
                return self.args[0]**exponent
            elif exponent is not S.NegativeOne and exponent.is_Integer:
                return self.args[0]**(exponent - 1)*self
        return

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.exponential import log
        direction = self.args[0].leadterm(x)[0]
        if direction.has(log(x)):
            direction = direction.subs(log(x), logx)
        s = self.args[0]._eval_nseries(x, n=n, logx=logx)
        return (sign(direction)*s).expand()

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return Derivative(self.args[0], x, evaluate=True) \
                * sign(conjugate(self.args[0]))
        rv = (re(self.args[0]) * Derivative(re(self.args[0]), x,
            evaluate=True) + im(self.args[0]) * Derivative(im(self.args[0]),
                x, evaluate=True)) / Abs(self.args[0])
        return rv.rewrite(sign)

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        # Note this only holds for real arg (since Heaviside is not defined
        # for complex arguments).
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return arg*(Heaviside(arg) - Heaviside(-arg))

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((arg, arg >= 0), (-arg, True))
        elif arg.is_imaginary:
            return Piecewise((I*arg, I*arg >= 0), (-I*arg, True))

    def _eval_rewrite_as_sign(self, arg, **kwargs):
        return arg/sign(arg)

    def _eval_rewrite_as_conjugate(self, arg, **kwargs):
        return sqrt(arg*conjugate(arg))


class arg(DefinedFunction):
    r"""
    Returns the argument (in radians) of a complex number. The argument is
    evaluated in consistent convention with ``atan2`` where the branch-cut is
    taken along the negative real axis and ``arg(z)`` is in the interval
    $(-\pi,\pi]$. For a positive number, the argument is always 0; the
    argument of a negative number is $\pi$; and the argument of 0
    is undefined and returns ``nan``. So the ``arg`` function will never nest
    greater than 3 levels since at the 4th application, the result must be
    nan; for a real number, nan is returned on the 3rd application.

    Examples
    ========

    >>> from sympy import arg, I, sqrt, Dummy
    >>> from sympy.abc import x
    >>> arg(2.0)
    0
    >>> arg(I)
    pi/2
    >>> arg(sqrt(2) + I*sqrt(2))
    pi/4
    >>> arg(sqrt(3)/2 + I/2)
    pi/6
    >>> arg(4 + 3*I)
    atan(3/4)
    >>> arg(0.8 + 0.6*I)
    0.643501108793284
    >>> arg(arg(arg(arg(x))))
    nan
    >>> real = Dummy(real=True)
    >>> arg(arg(arg(real)))
    nan

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    value : Expr
        Returns arc tangent of arg measured in radians.

    """

    is_extended_real = True
    is_real = True
    is_finite = True
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        a = arg
        for i in range(3):
            if isinstance(a, cls):
                a = a.args[0]
            else:
                if i == 2 and a.is_extended_real:
                    return S.NaN
                break
        else:
            return S.NaN
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(arg, exp_polar):
            return periodic_argument(arg, oo)
        elif isinstance(arg, exp):
            i_ = im(arg.args[0])
            if i_.is_comparable:
                i_ %= 2*S.Pi
                if i_ > S.Pi:
                    i_ -= 2*S.Pi
                return i_

        if not arg.is_Atom:
            c, arg_ = factor_terms(arg).as_coeff_Mul()
            if arg_.is_Mul:
                arg_ = Mul(*[a if (sign(a) not in (-1, 1)) else
                    sign(a) for a in arg_.args])
            arg_ = sign(c)*arg_
        else:
            arg_ = arg
        if any(i.is_extended_positive is None for i in arg_.atoms(AppliedUndef)):
            return
        from sympy.functions.elementary.trigonometric import atan2
        x, y = arg_.as_real_imag()
        rv = atan2(y, x)
        if rv.is_number:
            return rv
        if arg_ != arg:
            return cls(arg_, evaluate=False)

    def _eval_derivative(self, t):
        x, y = self.args[0].as_real_imag()
        return (x * Derivative(y, t, evaluate=True) - y *
                    Derivative(x, t, evaluate=True)) / (x**2 + y**2)

    def _eval_rewrite_as_atan2(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import atan2
        x, y = self.args[0].as_real_imag()
        return atan2(y, x)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg0 = self.args[0]
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg0.subs(x, cdir*t)
        if z.is_positive:
            return S.Zero
        elif z.is_negative:
            return S.Pi
        else:
            raise PoleError("Cannot expand %s around 0" % (self))

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        if n <= 0:
            return Order(1)
        return self._eval_as_leading_term(x, logx=logx, cdir=cdir)


class conjugate(DefinedFunction):
    """
    Returns the *complex conjugate* [1]_ of an argument.
    In mathematics, the complex conjugate of a complex number
    is given by changing the sign of the imaginary part.

    Thus, the conjugate of the complex number
    :math:`a + ib` (where $a$ and $b$ are real numbers) is :math:`a - ib`

    Examples
    ========

    >>> from sympy import conjugate, I
    >>> conjugate(2)
    2
    >>> conjugate(I)
    -I
    >>> conjugate(3 + 2*I)
    3 - 2*I
    >>> conjugate(5 - I)
    5 + I

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    arg : Expr
        Complex conjugate of arg as real, imaginary or mixed expression.

    See Also
    ========

    sign, Abs

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_conjugation
    """
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_conjugate()
        if obj is not None:
            return obj

    def inverse(self):
        return conjugate

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)

    def _eval_adjoint(self):
        return transpose(self.args[0])

    def _eval_conjugate(self):
        return self.args[0]

    def _eval_derivative(self, x):
        if x.is_real:
            return conjugate(Derivative(self.args[0], x, evaluate=True))
        elif x.is_imaginary:
            return -conjugate(Derivative(self.args[0], x, evaluate=True))

    def _eval_transpose(self):
        return adjoint(self.args[0])

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic


class transpose(DefinedFunction):
    """
    Linear map transposition.

    Examples
    ========

    >>> from sympy import transpose, Matrix, MatrixSymbol
    >>> A = MatrixSymbol('A', 25, 9)
    >>> transpose(A)
    A.T
    >>> B = MatrixSymbol('B', 9, 22)
    >>> transpose(B)
    B.T
    >>> transpose(A*B)
    B.T*A.T
    >>> M = Matrix([[4, 5], [2, 1], [90, 12]])
    >>> M
    Matrix([
    [ 4,  5],
    [ 2,  1],
    [90, 12]])
    >>> transpose(M)
    Matrix([
    [4, 2, 90],
    [5, 1, 12]])

    Parameters
    ==========

    arg : Matrix
         Matrix or matrix expression to take the transpose of.

    Returns
    =======

    value : Matrix
        Transpose of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_transpose()
        if obj is not None:
            return obj

    def _eval_adjoint(self):
        return conjugate(self.args[0])

    def _eval_conjugate(self):
        return adjoint(self.args[0])

    def _eval_transpose(self):
        return self.args[0]


class adjoint(DefinedFunction):
    """
    Conjugate transpose or Hermite conjugation.

    Examples
    ========

    >>> from sympy import adjoint, MatrixSymbol
    >>> A = MatrixSymbol('A', 10, 5)
    >>> adjoint(A)
    Adjoint(A)

    Parameters
    ==========

    arg : Matrix
        Matrix or matrix expression to take the adjoint of.

    Returns
    =======

    value : Matrix
        Represents the conjugate transpose or Hermite
        conjugation of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_adjoint()
        if obj is not None:
            return obj
        obj = arg._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def _eval_adjoint(self):
        return self.args[0]

    def _eval_conjugate(self):
        return transpose(self.args[0])

    def _eval_transpose(self):
        return conjugate(self.args[0])

    def _latex(self, printer, exp=None, *args):
        arg = printer._print(self.args[0])
        tex = r'%s^{\dagger}' % arg
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if printer._use_unicode:
            pform = pform**prettyForm('\N{DAGGER}')
        else:
            pform = pform**prettyForm('+')
        return pform

###############################################################################
############### HANDLING OF POLAR NUMBERS #####################################
###############################################################################


class polar_lift(DefinedFunction):
    """
    Lift argument to the Riemann surface of the logarithm, using the
    standard branch.

    Examples
    ========

    >>> from sympy import Symbol, polar_lift, I
    >>> p = Symbol('p', polar=True)
    >>> x = Symbol('x')
    >>> polar_lift(4)
    4*exp_polar(0)
    >>> polar_lift(-4)
    4*exp_polar(I*pi)
    >>> polar_lift(-I)
    exp_polar(-I*pi/2)
    >>> polar_lift(I + 2)
    polar_lift(2 + I)

    >>> polar_lift(4*x)
    4*polar_lift(x)
    >>> polar_lift(4*p)
    4*p

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    periodic_argument
    """

    is_polar = True
    is_comparable = False  # Cannot be evalf'd.

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.complexes import arg as argument
        if arg.is_number:
            ar = argument(arg)
            # In general we want to affirm that something is known,
            # e.g. `not ar.has(argument) and not ar.has(atan)`
            # but for now we will just be more restrictive and
            # see that it has evaluated to one of the known values.
            if ar in (0, pi/2, -pi/2, pi):
                from sympy.functions.elementary.exponential import exp_polar
                return exp_polar(I*ar)*abs(arg)

        if arg.is_Mul:
            args = arg.args
        else:
            args = [arg]
        included = []
        excluded = []
        positive = []
        for arg in args:
            if arg.is_polar:
                included += [arg]
            elif arg.is_positive:
                positive += [arg]
            else:
                excluded += [arg]
        if len(excluded) < len(args):
            if excluded:
                return Mul(*(included + positive))*polar_lift(Mul(*excluded))
            elif included:
                return Mul(*(included + positive))
            else:
                from sympy.functions.elementary.exponential import exp_polar
                return Mul(*positive)*exp_polar(0)

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        return self.args[0]._eval_evalf(prec)

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)


class periodic_argument(DefinedFunction):
    r"""
    Represent the argument on a quotient of the Riemann surface of the
    logarithm. That is, given a period $P$, always return a value in
    $(-P/2, P/2]$, by using $\exp(PI) = 1$.

    Examples
    ========

    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(10*I*pi), 2*pi)
    0
    >>> periodic_argument(exp_polar(5*I*pi), 4*pi)
    pi
    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(5*I*pi), 2*pi)
    pi
    >>> periodic_argument(exp_polar(5*I*pi), 3*pi)
    -pi
    >>> periodic_argument(exp_polar(5*I*pi), pi)
    0

    Parameters
    ==========

    ar : Expr
        A polar number.

    period : Expr
        The period $P$.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    principal_branch
    """

    @classmethod
    def _getunbranched(cls, ar):
        from sympy.functions.elementary.exponential import exp_polar, log
        if ar.is_Mul:
            args = ar.args
        else:
            args = [ar]
        unbranched = 0
        for a in args:
            if not a.is_polar:
                unbranched += arg(a)
            elif isinstance(a, exp_polar):
                unbranched += a.exp.as_real_imag()[1]
            elif a.is_Pow:
                re, im = a.exp.as_real_imag()
                unbranched += re*unbranched_argument(
                    a.base) + im*log(abs(a.base))
            elif isinstance(a, polar_lift):
                unbranched += arg(a.args[0])
            else:
                return None
        return unbranched

    @classmethod
    def eval(cls, ar, period):
        # Our strategy is to evaluate the argument on the Riemann surface of the
        # logarithm, and then reduce.
        # NOTE evidently this means it is a rather bad idea to use this with
        # period != 2*pi and non-polar numbers.
        if not period.is_extended_positive:
            return None
        if period == oo and isinstance(ar, principal_branch):
            return periodic_argument(*ar.args)
        if isinstance(ar, polar_lift) and period >= 2*pi:
            return periodic_argument(ar.args[0], period)
        if ar.is_Mul:
            newargs = [x for x in ar.args if not x.is_positive]
            if len(newargs) != len(ar.args):
                return periodic_argument(Mul(*newargs), period)
        unbranched = cls._getunbranched(ar)
        if unbranched is None:
            return None
        from sympy.functions.elementary.trigonometric import atan, atan2
        if unbranched.has(periodic_argument, atan2, atan):
            return None
        if period == oo:
            return unbranched
        if period != oo:
            from sympy.functions.elementary.integers import ceiling
            n = ceiling(unbranched/period - S.Half)*period
            if not n.has(ceiling):
                return unbranched - n

    def _eval_evalf(self, prec):
        z, period = self.args
        if period == oo:
            unbranched = periodic_argument._getunbranched(z)
            if unbranched is None:
                return self
            return unbranched._eval_evalf(prec)
        ub = periodic_argument(z, oo)._eval_evalf(prec)
        from sympy.functions.elementary.integers import ceiling
        return (ub - ceiling(ub/period - S.Half)*period)._eval_evalf(prec)


def unbranched_argument(arg):
    '''
    Returns periodic argument of arg with period as infinity.

    Examples
    ========

    >>> from sympy import exp_polar, unbranched_argument
    >>> from sympy import I, pi
    >>> unbranched_argument(exp_polar(15*I*pi))
    15*pi
    >>> unbranched_argument(exp_polar(7*I*pi))
    7*pi

    See also
    ========

    periodic_argument
    '''
    return periodic_argument(arg, oo)


class principal_branch(DefinedFunction):
    """
    Represent a polar number reduced to its principal branch on a quotient
    of the Riemann surface of the logarithm.

    Explanation
    ===========

    This is a function of two arguments. The first argument is a polar
    number `z`, and the second one a positive real number or infinity, `p`.
    The result is ``z mod exp_polar(I*p)``.

    Examples
    ========

    >>> from sympy import exp_polar, principal_branch, oo, I, pi
    >>> from sympy.abc import z
    >>> principal_branch(z, oo)
    z
    >>> principal_branch(exp_polar(2*pi*I)*3, 2*pi)
    3*exp_polar(0)
    >>> principal_branch(exp_polar(2*pi*I)*3*z, 2*pi)
    3*principal_branch(z, 2*pi)

    Parameters
    ==========

    x : Expr
        A polar number.

    period : Expr
        Positive real number or infinity.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    periodic_argument
    """

    is_polar = True
    is_comparable = False  # cannot always be evalf'd

    @classmethod
    def eval(self, x, period):
        from sympy.functions.elementary.exponential import exp_polar
        if isinstance(x, polar_lift):
            return principal_branch(x.args[0], period)
        if period == oo:
            return x
        ub = periodic_argument(x, oo)
        barg = periodic_argument(x, period)
        if ub != barg and not ub.has(periodic_argument) \
                and not barg.has(periodic_argument):
            pl = polar_lift(x)

            def mr(expr):
                if not isinstance(expr, Symbol):
                    return polar_lift(expr)
                return expr
            pl = pl.replace(polar_lift, mr)
            # Recompute unbranched argument
            ub = periodic_argument(pl, oo)
            if not pl.has(polar_lift):
                if ub != barg:
                    res = exp_polar(I*(barg - ub))*pl
                else:
                    res = pl
                if not res.is_polar and not res.has(exp_polar):
                    res *= exp_polar(0)
                return res

        if not x.free_symbols:
            c, m = x, ()
        else:
            c, m = x.as_coeff_mul(*x.free_symbols)
        others = []
        for y in m:
            if y.is_positive:
                c *= y
            else:
                others += [y]
        m = tuple(others)
        arg = periodic_argument(c, period)
        if arg.has(periodic_argument):
            return None
        if arg.is_number and (unbranched_argument(c) != arg or
                              (arg == 0 and m != () and c != 1)):
            if arg == 0:
                return abs(c)*principal_branch(Mul(*m), period)
            return principal_branch(exp_polar(I*arg)*Mul(*m), period)*abs(c)
        if arg.is_number and ((abs(arg) < period/2) == True or arg == period/2) \
                and m == ():
            return exp_polar(arg*I)*abs(c)

    def _eval_evalf(self, prec):
        z, period = self.args
        p = periodic_argument(z, period)._eval_evalf(prec)
        if abs(p) > pi or p == -pi:
            return self  # Cannot evalf for this argument.
        from sympy.functions.elementary.exponential import exp
        return (abs(z)*exp(I*p))._eval_evalf(prec)


def _polarify(eq, lift, pause=False):
    from sympy.integrals.integrals import Integral
    if eq.is_polar:
        return eq
    if eq.is_number and not pause:
        return polar_lift(eq)
    if isinstance(eq, Symbol) and not pause and lift:
        return polar_lift(eq)
    elif eq.is_Atom:
        return eq
    elif eq.is_Add:
        r = eq.func(*[_polarify(arg, lift, pause=True) for arg in eq.args])
        if lift:
            return polar_lift(r)
        return r
    elif eq.is_Pow and eq.base == S.Exp1:
        return eq.func(S.Exp1, _polarify(eq.exp, lift, pause=False))
    elif eq.is_Function:
        return eq.func(*[_polarify(arg, lift, pause=False) for arg in eq.args])
    elif isinstance(eq, Integral):
        # Don't lift the integration variable
        func = _polarify(eq.function, lift, pause=pause)
        limits = []
        for limit in eq.args[1:]:
            var = _polarify(limit[0], lift=False, pause=pause)
            rest = _polarify(limit[1:], lift=lift, pause=pause)
            limits.append((var,) + rest)
        return Integral(*((func,) + tuple(limits)))
    else:
        return eq.func(*[_polarify(arg, lift, pause=pause)
                         if isinstance(arg, Expr) else arg for arg in eq.args])


def polarify(eq, subs=True, lift=False):
    """
    Turn all numbers in eq into their polar equivalents (under the standard
    choice of argument).

    Note that no attempt is made to guess a formal convention of adding
    polar numbers, expressions like $1 + x$ will generally not be altered.

    Note also that this function does not promote ``exp(x)`` to ``exp_polar(x)``.

    If ``subs`` is ``True``, all symbols which are not already polar will be
    substituted for polar dummies; in this case the function behaves much
    like :func:`~.posify`.

    If ``lift`` is ``True``, both addition statements and non-polar symbols are
    changed to their ``polar_lift()``ed versions.
    Note that ``lift=True`` implies ``subs=False``.

    Examples
    ========

    >>> from sympy import polarify, sin, I
    >>> from sympy.abc import x, y
    >>> expr = (-x)**y
    >>> expr.expand()
    (-x)**y
    >>> polarify(expr)
    ((_x*exp_polar(I*pi))**_y, {_x: x, _y: y})
    >>> polarify(expr)[0].expand()
    _x**_y*exp_polar(_y*I*pi)
    >>> polarify(x, lift=True)
    polar_lift(x)
    >>> polarify(x*(1+y), lift=True)
    polar_lift(x)*polar_lift(y + 1)

    Adds are treated carefully:

    >>> polarify(1 + sin((1 + I)*x))
    (sin(_x*polar_lift(1 + I)) + 1, {_x: x})
    """
    if lift:
        subs = False
    eq = _polarify(sympify(eq), lift)
    if not subs:
        return eq
    reps = {s: Dummy(s.name, polar=True) for s in eq.free_symbols}
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}


def _unpolarify(eq, exponents_only, pause=False):
    if not isinstance(eq, Basic) or eq.is_Atom:
        return eq

    if not pause:
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(eq, exp_polar):
            return exp(_unpolarify(eq.exp, exponents_only))
        if isinstance(eq, principal_branch) and eq.args[1] == 2*pi:
            return _unpolarify(eq.args[0], exponents_only)
        if (
            eq.is_Add or eq.is_Mul or eq.is_Boolean or
            eq.is_Relational and (
                eq.rel_op in ('==', '!=') and 0 in eq.args or
                eq.rel_op not in ('==', '!='))
        ):
            return eq.func(*[_unpolarify(x, exponents_only) for x in eq.args])
        if isinstance(eq, polar_lift):
            return _unpolarify(eq.args[0], exponents_only)

    if eq.is_Pow:
        expo = _unpolarify(eq.exp, exponents_only)
        base = _unpolarify(eq.base, exponents_only,
            not (expo.is_integer and not pause))
        return base**expo

    if eq.is_Function and getattr(eq.func, 'unbranched', False):
        return eq.func(*[_unpolarify(x, exponents_only, exponents_only)
            for x in eq.args])

    return eq.func(*[_unpolarify(x, exponents_only, True) for x in eq.args])


def unpolarify(eq, subs=None, exponents_only=False):
    """
    If `p` denotes the projection from the Riemann surface of the logarithm to
    the complex line, return a simplified version `eq'` of `eq` such that
    `p(eq') = p(eq)`.
    Also apply the substitution subs in the end. (This is a convenience, since
    ``unpolarify``, in a certain sense, undoes :func:`polarify`.)

    Examples
    ========

    >>> from sympy import unpolarify, polar_lift, sin, I
    >>> unpolarify(polar_lift(I + 2))
    2 + I
    >>> unpolarify(sin(polar_lift(I + 7)))
    sin(7 + I)
    """
    if isinstance(eq, bool):
        return eq

    eq = sympify(eq)
    if subs is not None:
        return unpolarify(eq.subs(subs))
    changed = True
    pause = False
    if exponents_only:
        pause = True
    while changed:
        changed = False
        res = _unpolarify(eq, exponents_only, pause)
        if res != eq:
            changed = True
            eq = res
        if isinstance(res, bool):
            return res
    # Finally, replacing Exp(0) by 1 is always correct.
    # So is polar_lift(0) -> 0.
    from sympy.functions.elementary.exponential import exp_polar
    return res.subs({exp_polar(0): 1, polar_lift(0): 0})
<|endoftext|>from sympy.core import S, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.function import DefinedFunction
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
    _canonical, _canonical_coeff)
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name

from itertools import product

Undefined = S.NaN  # Piecewise()

class ExprCondPair(Tuple):
    """Represents an expression, condition pair."""

    def __new__(cls, expr, cond):
        expr = as_Basic(expr)
        if cond == True:
            return Tuple.__new__(cls, expr, true)
        elif cond == False:
            return Tuple.__new__(cls, expr, false)
        elif isinstance(cond, Basic) and cond.has(Piecewise):
            cond = piecewise_fold(cond)
            if isinstance(cond, Piecewise):
                cond = cond.rewrite(ITE)

        if not isinstance(cond, Boolean):
            raise TypeError(filldedent('''
                Second argument must be a Boolean,
                not `%s`''' % func_name(cond)))
        return Tuple.__new__(cls, expr, cond)

    @property
    def expr(self):
        """
        Returns the expression of this pair.
        """
        return self.args[0]

    @property
    def cond(self):
        """
        Returns the condition of this pair.
        """
        return self.args[1]

    @property
    def is_commutative(self):
        return self.expr.is_commutative

    def __iter__(self):
        yield self.expr
        yield self.cond

    def _eval_simplify(self, **kwargs):
        return self.func(*[a.simplify(**kwargs) for a in self.args])


class Piecewise(DefinedFunction):
    """
    Represents a piecewise function.

    Usage:

      Piecewise( (expr,cond), (expr,cond), ... )
        - Each argument is a 2-tuple defining an expression and condition
        - The conds are evaluated in turn returning the first that is True.
          If any of the evaluated conds are not explicitly False,
          e.g. ``x < 1``, the function is returned in symbolic form.
        - If the function is evaluated at a place where all conditions are False,
          nan will be returned.
        - Pairs where the cond is explicitly False, will be removed and no pair
          appearing after a True condition will ever be retained. If a single
          pair with a True condition remains, it will be returned, even when
          evaluation is False.

    Examples
    ========

    >>> from sympy import Piecewise, log, piecewise_fold
    >>> from sympy.abc import x, y
    >>> f = x**2
    >>> g = log(x)
    >>> p = Piecewise((0, x < -1), (f, x <= 1), (g, True))
    >>> p.subs(x,1)
    1
    >>> p.subs(x,5)
    log(5)

    Booleans can contain Piecewise elements:

    >>> cond = (x < y).subs(x, Piecewise((2, x < 0), (3, True))); cond
    Piecewise((2, x < 0), (3, True)) < y

    The folded version of this results in a Piecewise whose
    expressions are Booleans:

    >>> folded_cond = piecewise_fold(cond); folded_cond
    Piecewise((2 < y, x < 0), (3 < y, True))

    When a Boolean containing Piecewise (like cond) or a Piecewise
    with Boolean expressions (like folded_cond) is used as a condition,
    it is converted to an equivalent :class:`~.ITE` object:

    >>> Piecewise((1, folded_cond))
    Piecewise((1, ITE(x < 0, y > 2, y > 3)))

    When a condition is an ``ITE``, it will be converted to a simplified
    Boolean expression:

    >>> piecewise_fold(_)
    Piecewise((1, ((x >= 0) | (y > 2)) & ((y > 3) | (x < 0))))

    See Also
    ========

    piecewise_fold
    piecewise_exclusive
    ITE
    """

    nargs = None
    is_Piecewise = True

    def __new__(cls, *args, **options):
        if len(args) == 0:
            raise TypeError("At least one (expr, cond) pair expected.")
        # (Try to) sympify args first
        newargs = []
        for ec in args:
            # ec could be a ExprCondPair or a tuple
            pair = ExprCondPair(*getattr(ec, 'args', ec))
            cond = pair.cond
            if cond is false:
                continue
            newargs.append(pair)
            if cond is true:
                break

        eval = options.pop('evaluate', global_parameters.evaluate)
        if eval:
            r = cls.eval(*newargs)
            if r is not None:
                return r
        elif len(newargs) == 1 and newargs[0].cond == True:
            return newargs[0].expr

        return Basic.__new__(cls, *newargs, **options)

    @classmethod
    def eval(cls, *_args):
        """Either return a modified version of the args or, if no
        modifications were made, return None.

        Modifications that are made here:

        1. relationals are made canonical
        2. any False conditions are dropped
        3. any repeat of a previous condition is ignored
        4. any args past one with a true condition are dropped

        If there are no args left, nan will be returned.
        If there is a single arg with a True condition, its
        corresponding expression will be returned.

        EXAMPLES
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> cond = -x < -1
        >>> args = [(1, cond), (4, cond), (3, False), (2, True), (5, x < 1)]
        >>> Piecewise(*args, evaluate=False)
        Piecewise((1, -x < -1), (4, -x < -1), (2, True))
        >>> Piecewise(*args)
        Piecewise((1, x > 1), (2, True))
        """
        if not _args:
            return Undefined

        if len(_args) == 1 and _args[0][-1] == True:
            return _args[0][0]

        newargs = _piecewise_collapse_arguments(_args)

        # some conditions may have been redundant
        missing = len(newargs) != len(_args)
        # some conditions may have changed
        same = all(a == b for a, b in zip(newargs, _args))
        # if either change happened we return the expr with the
        # updated args
        if not newargs:
            raise ValueError(filldedent('''
                There are no conditions (or none that
                are not trivially false) to define an
                expression.'''))
        if missing or not same:
            return cls(*newargs)

    def doit(self, **hints):
        """
        Evaluate this piecewise function.
        """
        newargs = []
        for e, c in self.args:
            if hints.get('deep', True):
                if isinstance(e, Basic):
                    newe = e.doit(**hints)
                    if newe != self:
                        e = newe
                if isinstance(c, Basic):
                    c = c.doit(**hints)
            newargs.append((e, c))
        return self.func(*newargs)

    def _eval_simplify(self, **kwargs):
        return piecewise_simplify(self, **kwargs)

    def _eval_as_leading_term(self, x, logx, cdir):
        for e, c in self.args:
            if c == True or c.subs(x, 0) == True:
                return e.as_leading_term(x)

    def _eval_adjoint(self):
        return self.func(*[(e.adjoint(), c) for e, c in self.args])

    def _eval_conjugate(self):
        return self.func(*[(e.conjugate(), c) for e, c in self.args])

    def _eval_derivative(self, x):
        return self.func(*[(diff(e, x), c) for e, c in self.args])

    def _eval_evalf(self, prec):
        return self.func(*[(e._evalf(prec), c) for e, c in self.args])

    def _eval_is_meromorphic(self, x, a):
        # Conditions often implicitly assume that the argument is real.
        # Hence, there needs to be some check for as_set.
        if not a.is_real:
            return None

        # Then, scan ExprCondPairs in the given order to find a piece that would contain a,
        # possibly as a boundary point.
        for e, c in self.args:
            cond = c.subs(x, a)

            if cond.is_Relational:
                return None
            if a in c.as_set().boundary:
                return None
            # Apply expression if a is an interior point of the domain of e.
            if cond:
                return e._eval_is_meromorphic(x, a)

    def piecewise_integrate(self, x, **kwargs):
        """Return the Piecewise with each expression being
        replaced with its antiderivative. To obtain a continuous
        antiderivative, use the :func:`~.integrate` function or method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        Note that this does not give a continuous function, e.g.
        at x = 1 the 3rd condition applies and the antiderivative
        there is 2*x so the value of the antiderivative is 2:

        >>> anti = _
        >>> anti.subs(x, 1)
        2

        The continuous derivative accounts for the integral *up to*
        the point of interest, however:

        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> _.subs(x, 1)
        1

        See Also
        ========
        Piecewise._eval_integral
        """
        from sympy.integrals import integrate
        return self.func(*[(integrate(e, x, **kwargs), c) for e, c in self.args])

    def _handle_irel(self, x, handler):
        """Return either None (if the conditions of self depend only on x) else
        a Piecewise expression whose expressions (handled by the handler that
        was passed) are paired with the governing x-independent relationals,
        e.g. Piecewise((A, a(x) & b(y)), (B, c(x) | c(y)) ->
        Piecewise(
            (handler(Piecewise((A, a(x) & True), (B, c(x) | True)), b(y) & c(y)),
            (handler(Piecewise((A, a(x) & True), (B, c(x) | False)), b(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | True)), c(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | False)), True))
        """
        # identify governing relationals
        rel = self.atoms(Relational)
        irel = list(ordered([r for r in rel if x not in r.free_symbols
            and r not in (S.true, S.false)]))
        if irel:
            args = {}
            exprinorder = []
            for truth in product((1, 0), repeat=len(irel)):
                reps = dict(zip(irel, truth))
                # only store the true conditions since the false are implied
                # when they appear lower in the Piecewise args
                if 1 not in truth:
                    cond = None  # flag this one so it doesn't get combined
                else:
                    andargs = Tuple(*[i for i in reps if reps[i]])
                    free = list(andargs.free_symbols)
                    if len(free) == 1:
                        from sympy.solvers.inequalities import (
                            reduce_inequalities, _solve_inequality)
                        try:
                            t = reduce_inequalities(andargs, free[0])
                            # ValueError when there are potentially
                            # nonvanishing imaginary parts
                        except (ValueError, NotImplementedError):
                            # at least isolate free symbol on left
                            t = And(*[_solve_inequality(
                                a, free[0], linear=True)
                                for a in andargs])
                    else:
                        t = And(*andargs)
                    if t is S.false:
                        continue  # an impossible combination
                    cond = t
                expr = handler(self.xreplace(reps))
                if isinstance(expr, self.func) and len(expr.args) == 1:
                    expr, econd = expr.args[0]
                    cond = And(econd, True if cond is None else cond)
                # the ec pairs are being collected since all possibilities
                # are being enumerated, but don't put the last one in since
                # its expr might match a previous expression and it
                # must appear last in the args
                if cond is not None:
                    args.setdefault(expr, []).append(cond)
                    # but since we only store the true conditions we must maintain
                    # the order so that the expression with the most true values
                    # comes first
                    exprinorder.append(expr)
            # convert collected conditions as args of Or
            for k in args:
                args[k] = Or(*args[k])
            # take them in the order obtained
            args = [(e, args[e]) for e in uniq(exprinorder)]
            # add in the last arg
            args.append((expr, True))
            return Piecewise(*args)

    def _eval_integral(self, x, _first=True, **kwargs):
        """Return the indefinite integral of the
        Piecewise such that subsequent substitution of x with a
        value will give the value of the integral (not including
        the constant of integration) up to that point. To only
        integrate the individual parts of Piecewise, use the
        ``piecewise_integrate`` method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        See Also
        ========
        Piecewise.piecewise_integrate
        """
        from sympy.integrals.integrals import integrate

        if _first:
            def handler(ipw):
                if isinstance(ipw, self.func):
                    return ipw._eval_integral(x, _first=False, **kwargs)
                else:
                    return ipw.integrate(x, **kwargs)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv

        # handle a Piecewise from -oo to oo with and no x-independent relationals
        # -----------------------------------------------------------------------
        ok, abei = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            return Integral(self, x)  # unevaluated

        pieces = [(a, b) for a, b, _, _ in abei]
        oo = S.Infinity
        done = [(-oo, oo, -1)]
        for k, p in enumerate(pieces):
            if p == (-oo, oo):
                # all undone intervals will get this key
                for j, (a, b, i) in enumerate(done):
                    if i == -1:
                        done[j] = a, b, k
                break  # nothing else to consider
            N = len(done) - 1
            for j, (a, b, i) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j: j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for a, b, i in done if a != b]

        # append an arg if there is a hole so a reference to
        # argument -1 will give Undefined
        if any(i == -1 for (a, b, i) in done):
            abei.append((-oo, oo, Undefined, -1))

        # return the sum of the intervals
        args = []
        sum = None
        for a, b, i in done:
            anti = integrate(abei[i][-2], x, **kwargs)
            if sum is None:
                sum = anti
            else:
                sum = sum.subs(x, a)
                e = anti._eval_interval(x, a, x)
                if sum.has(*_illegal) or e.has(*_illegal):
                    sum = anti
                else:
                    sum += e
            # see if we know whether b is contained in original
            # condition
            if b is S.Infinity:
                cond = True
            elif self.args[abei[i][-1]].cond.subs(x, b) == False:
                cond = (x < b)
            else:
                cond = (x <= b)
            args.append((sum, cond))
        return Piecewise(*args)

    def _eval_interval(self, sym, a, b, _first=True):
        """Evaluates the function along the sym in a given interval [a, b]"""
        # FIXME: Currently complex intervals are not supported.  A possible
        # replacement algorithm, discussed in issue 5227, can be found in the
        # following papers;
        #     http://portal.acm.org/citation.cfm?id=281649
        #     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.4127&rep=rep1&type=pdf

        if a is None or b is None:
            # In this case, it is just simple substitution
            return super()._eval_interval(sym, a, b)
        else:
            x, lo, hi = map(as_Basic, (sym, a, b))

        if _first:  # get only x-dependent relationals
            def handler(ipw):
                if isinstance(ipw, self.func):
                    return ipw._eval_interval(x, lo, hi, _first=None)
                else:
                    return ipw._eval_interval(x, lo, hi)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv

            if (lo < hi) is S.false or (
                    lo is S.Infinity or hi is S.NegativeInfinity):
                rv = self._eval_interval(x, hi, lo, _first=False)
                if isinstance(rv, Piecewise):
                    rv = Piecewise(*[(-e, c) for e, c in rv.args])
                else:
                    rv = -rv
                return rv

            if (lo < hi) is S.true or (
                    hi is S.Infinity or lo is S.NegativeInfinity):
                pass
            else:
                _a = Dummy('lo')
                _b = Dummy('hi')
                a = lo if lo.is_comparable else _a
                b = hi if hi.is_comparable else _b
                pos = self._eval_interval(x, a, b, _first=False)
                if a == _a and b == _b:
                    # it's purely symbolic so just swap lo and hi and
                    # change the sign to get the value for when lo > hi
                    neg, pos = (-pos.xreplace({_a: hi, _b: lo}),
                        pos.xreplace({_a: lo, _b: hi}))
                else:
                    # at least one of the bounds was comparable, so allow
                    # _eval_interval to use that information when computing
                    # the interval with lo and hi reversed
                    neg, pos = (-self._eval_interval(x, hi, lo, _first=False),
                        pos.xreplace({_a: lo, _b: hi}))

                # allow simplification based on ordering of lo and hi
                p = Dummy('', positive=True)
                if lo.is_Symbol:
                    pos = pos.xreplace({lo: hi - p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({lo: hi + p}).xreplace({p: lo - hi})
                elif hi.is_Symbol:
                    pos = pos.xreplace({hi: lo + p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({hi: lo - p}).xreplace({p: lo - hi})
                # evaluate limits that may have unevaluate Min/Max
                touch = lambda _: _.replace(
                    lambda x: isinstance(x, (Min, Max)),
                    lambda x: x.func(*x.args))
                neg = touch(neg)
                pos = touch(pos)
                # assemble return expression; make the first condition be Lt
                # b/c then the first expression will look the same whether
                # the lo or hi limit is symbolic
                if a == _a:  # the lower limit was symbolic
                    rv = Piecewise(
                        (pos,
                            lo < hi),
                        (neg,
                            True))
                else:
                    rv = Piecewise(
                        (neg,
                            hi < lo),
                        (pos,
                            True))

                if rv == Undefined:
                    raise ValueError("Can't integrate across undefined region.")
                if any(isinstance(i, Piecewise) for i in (pos, neg)):
                    rv = piecewise_fold(rv)
                return rv

        # handle a Piecewise with lo <= hi and no x-independent relationals
        # -----------------------------------------------------------------
        ok, abei = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            # not being able to do the interval of f(x) can
            # be stated as not being able to do the integral
            # of f'(x) over the same range
            return Integral(self.diff(x), (x, lo, hi))  # unevaluated

        pieces = [(a, b) for a, b, _, _ in abei]
        done = [(lo, hi, -1)]
        oo = S.Infinity
        for k, p in enumerate(pieces):
            if p[:2] == (-oo, oo):
                # all undone intervals will get this key
                for j, (a, b, i) in enumerate(done):
                    if i == -1:
                        done[j] = a, b, k
                break  # nothing else to consider
            N = len(done) - 1
            for j, (a, b, i) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j: j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for a, b, i in done if a != b]

        # return the sum of the intervals
        sum = S.Zero
        upto = None
        for a, b, i in done:
            if i == -1:
                if upto is None:
                    return Undefined
                # TODO simplify hi <= upto
                return Piecewise((sum, hi <= upto), (Undefined, True))
            sum += abei[i][-2]._eval_interval(x, a, b)
            upto = b
        return sum

    def _intervals(self, sym, err_on_Eq=False):
        r"""Return a bool and a message (when bool is False), else a
        list of unique tuples, (a, b, e, i), where a and b
        are the lower and upper bounds in which the expression e of
        argument i in self is defined and $a < b$ (when involving
        numbers) or $a \le b$ when involving symbols.

        If there are any relationals not involving sym, or any
        relational cannot be solved for sym, the bool will be False
        a message be given as the second return value. The calling
        routine should have removed such relationals before calling
        this routine.

        The evaluated conditions will be returned as ranges.
        Discontinuous ranges will be returned separately with
        identical expressions. The first condition that evaluates to
        True will be returned as the last tuple with a, b = -oo, oo.
        """
        from sympy.solvers.inequalities import _solve_inequality

        assert isinstance(self, Piecewise)

        def nonsymfail(cond):
            return False, filldedent('''
                A condition not involving
                %s appeared: %s''' % (sym, cond))

        def _solve_relational(r):
            if sym not in r.free_symbols:
                return nonsymfail(r)
            try:
                rv = _solve_inequality(r, sym)
            except NotImplementedError:
                return False, 'Unable to solve relational %s for %s.' % (r, sym)
            if isinstance(rv, Relational):
                free = rv.args[1].free_symbols
                if rv.args[0] != sym or sym in free:
                    return False, 'Unable to solve relational %s for %s.' % (r, sym)
                if rv.rel_op == '==':
                    # this equality has been affirmed to have the form
                    # Eq(sym, rhs) where rhs is sym-free; it represents
                    # a zero-width interval which will be ignored
                    # whether it is an isolated condition or contained
                    # within an And or an Or
                    rv = S.false
                elif rv.rel_op == '!=':
                    try:
                        rv = Or(sym < rv.rhs, sym > rv.rhs)
                    except TypeError:
                        # e.g. x != I ==> all real x satisfy
                        rv = S.true
            elif rv == (S.NegativeInfinity < sym) & (sym < S.Infinity):
                rv = S.true
            return True, rv

        args = list(self.args)
        # make self canonical wrt Relationals
        keys = self.atoms(Relational)
        reps = {}
        for r in keys:
            ok, s = _solve_relational(r)
            if ok != True:
                return False, ok
            reps[r] = s
        # process args individually so if any evaluate, their position
        # in the original Piecewise will be known
        args = [i.xreplace(reps) for i in self.args]

        # precondition args
        expr_cond = []
        default = idefault = None
        for i, (expr, cond) in enumerate(args):
            if cond is S.false:
                continue
            if cond is S.true:
                default = expr
                idefault = i
                break
            if isinstance(cond, Eq):
                # unanticipated condition, but it is here in case a
                # replacement caused an Eq to appear
                if err_on_Eq:
                    return False, 'encountered Eq condition: %s' % cond
                continue  # zero width interval

            cond = to_cnf(cond)
            if isinstance(cond, And):
                cond = distribute_or_over_and(cond)

            if isinstance(cond, Or):
                expr_cond.extend(
                    [(i, expr, o) for o in cond.args
                    if not isinstance(o, Eq)])
            elif cond is not S.false:
                expr_cond.append((i, expr, cond))
            elif cond is S.true:
                default = expr
                idefault = i
                break

        # determine intervals represented by conditions
        int_expr = []
        for iarg, expr, cond in expr_cond:
            if isinstance(cond, And):
                lower = S.NegativeInfinity
                upper = S.Infinity
                exclude = []
                for cond2 in cond.args:
                    if not isinstance(cond2, Relational):
                        return False, 'expecting only Relationals'
                    if isinstance(cond2, Eq):
                        lower = upper  # ignore
                        if err_on_Eq:
                            return False, 'encountered secondary Eq condition'
                        break
                    elif isinstance(cond2, Ne):
                        l, r = cond2.args
                        if l == sym:
                            exclude.append(r)
                        elif r == sym:
                            exclude.append(l)
                        else:
                            return nonsymfail(cond2)
                        continue
                    elif cond2.lts == sym:
                        upper = Min(cond2.gts, upper)
                    elif cond2.gts == sym:
                        lower = Max(cond2.lts, lower)
                    else:
                        return nonsymfail(cond2)  # should never get here
                if exclude:
                    exclude = list(ordered(exclude))
                    newcond = []
                    for i, e in enumerate(exclude):
                        if e < lower == True or e > upper == True:
                            continue
                        if not newcond:
                            newcond.append((None, lower))  # add a primer
                        newcond.append((newcond[-1][1], e))
                    newcond.append((newcond[-1][1], upper))
                    newcond.pop(0)  # remove the primer
                    expr_cond.extend([(iarg, expr, And(i[0] < sym, sym < i[1])) for i in newcond])
                    continue
            elif isinstance(cond, Relational) and cond.rel_op != '!=':
                lower, upper = cond.lts, cond.gts  # part 1: initialize with givens
                if cond.lts == sym:                # part 1a: expand the side ...
                    lower = S.NegativeInfinity   # e.g. x <= 0 ---> -oo <= 0
                elif cond.gts == sym:            # part 1a: ... that can be expanded
                    upper = S.Infinity           # e.g. x >= 0 --->  oo >= 0
                else:
                    return nonsymfail(cond)
            else:
                return False, 'unrecognized condition: %s' % cond

            upper = Max(lower, upper)
            if err_on_Eq and lower == upper:
                return False, 'encountered Eq condition'
            if (lower >= upper) is not S.true:
                int_expr.append((lower, upper, expr, iarg))

        if default is not None:
            int_expr.append(
                (S.NegativeInfinity, S.Infinity, default, idefault))

        return True, list(uniq(int_expr))

    def _eval_nseries(self, x, n, logx, cdir=0):
        args = [(ec.expr._eval_nseries(x, n, logx), ec.cond) for ec in self.args]
        return self.func(*args)

    def _eval_power(self, s):
        return self.func(*[(e**s, c) for e, c in self.args])

    def _eval_subs(self, old, new):
        # this is strictly not necessary, but we can keep track
        # of whether True or False conditions arise and be
        # somewhat more efficient by avoiding other substitutions
        # and avoiding invalid conditions that appear after a
        # True condition
        args = list(self.args)
        args_exist = False
        for i, (e, c) in enumerate(args):
            c = c._subs(old, new)
            if c != False:
                args_exist = True
                e = e._subs(old, new)
            args[i] = (e, c)
            if c == True:
                break
        if not args_exist:
            args = ((Undefined, True),)
        return self.func(*args)

    def _eval_transpose(self):
        return self.func(*[(e.transpose(), c) for e, c in self.args])

    def _eval_template_is_attr(self, is_attr):
        b = None
        for expr, _ in self.args:
            a = getattr(expr, is_attr)
            if a is None:
                return
            if b is None:
                b = a
            elif b is not a:
                return
        return b

    _eval_is_finite = lambda self: self._eval_template_is_attr(
        'is_finite')
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    _eval_is_imaginary = lambda self: self._eval_template_is_attr(
        'is_imaginary')
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    _eval_is_irrational = lambda self: self._eval_template_is_attr(
        'is_irrational')
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr(
        'is_nonnegative')
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr(
        'is_nonpositive')
    _eval_is_nonzero = lambda self: self._eval_template_is_attr(
        'is_nonzero')
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    _eval_is_extended_real = lambda self: self._eval_template_is_attr(
            'is_extended_real')
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr(
            'is_extended_positive')
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr(
            'is_extended_negative')
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr(
            'is_extended_nonzero')
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr(
            'is_extended_nonpositive')
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr(
            'is_extended_nonnegative')
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    _eval_is_zero = lambda self: self._eval_template_is_attr(
        'is_zero')

    @classmethod
    def __eval_cond(cls, cond):
        """Return the truth value of the condition."""
        if cond == True:
            return True
        if isinstance(cond, Eq):
            try:
                diff = cond.lhs - cond.rhs
                if diff.is_commutative:
                    return diff.is_zero
            except TypeError:
                pass

    def as_expr_set_pairs(self, domain=None):
        """Return tuples for each argument of self that give
        the expression and the interval in which it is valid
        which is contained within the given domain.
        If a condition cannot be converted to a set, an error
        will be raised. The variable of the conditions is
        assumed to be real; sets of real values are returned.

        Examples
        ========

        >>> from sympy import Piecewise, Interval
        >>> from sympy.abc import x
        >>> p = Piecewise(
        ...     (1, x < 2),
        ...     (2,(x > 0) & (x < 4)),
        ...     (3, True))
        >>> p.as_expr_set_pairs()
        [(1, Interval.open(-oo, 2)),
         (2, Interval.Ropen(2, 4)),
         (3, Interval(4, oo))]
        >>> p.as_expr_set_pairs(Interval(0, 3))
        [(1, Interval.Ropen(0, 2)),
         (2, Interval(2, 3))]
        """
        if domain is None:
            domain = S.Reals
        exp_sets = []
        U = domain
        complex = not domain.is_subset(S.Reals)
        cond_free = set()
        for expr, cond in self.args:
            cond_free |= cond.free_symbols
            if len(cond_free) > 1:
                raise NotImplementedError(filldedent('''
                    multivariate conditions are not handled.'''))
            if complex:
                for i in cond.atoms(Relational):
                    if not isinstance(i, (Eq, Ne)):
                        raise ValueError(filldedent('''
                            Inequalities in the complex domain are
                            not supported. Try the real domain by
                            setting domain=S.Reals'''))
            cond_int = U.intersect(cond.as_set())
            U = U - cond_int
            if cond_int != S.EmptySet:
                exp_sets.append((expr, cond_int))
        return exp_sets

    def _eval_rewrite_as_ITE(self, *args, **kwargs):
        byfree = {}
        args = list(args)
        default = any(c == True for b, c in args)
        for i, (b, c) in enumerate(args):
            if not isinstance(b, Boolean) and b != True:
                raise TypeError(filldedent('''
                    Expecting Boolean or bool but got `%s`
                    ''' % func_name(b)))
            if c == True:
                break
            # loop over independent conditions for this b
            for c in c.args if isinstance(c, Or) else [c]:
                free = c.free_symbols
                x = free.pop()
                try:
                    byfree[x] = byfree.setdefault(
                        x, S.EmptySet).union(c.as_set())
                except NotImplementedError:
                    if not default:
                        raise NotImplementedError(filldedent('''
                            A method to determine whether a multivariate
                            conditional is consistent with a complete coverage
                            of all variables has not been implemented so the
                            rewrite is being stopped after encountering `%s`.
                            This error would not occur if a default expression
                            like `(foo, True)` were given.
                            ''' % c))
                if byfree[x] in (S.UniversalSet, S.Reals):
                    # collapse the ith condition to True and break
                    args[i] = list(args[i])
                    c = args[i][1] = True
                    break
            if c == True:
                break
        if c != True:
            raise ValueError(filldedent('''
                Conditions must cover all reals or a final default
                condition `(foo, True)` must be given.
                '''))
        last, _ = args[i]  # ignore all past ith arg
        for a, c in reversed(args[:i]):
            last = ITE(c, a, last)
        return _canonical(last)

    def _eval_rewrite_as_KroneckerDelta(self, *args, **kwargs):
        from sympy.functions.special.tensor_functions import KroneckerDelta

        rules = {
            And: [False, False],
            Or: [True, True],
            Not: [True, False],
            Eq: [None, None],
            Ne: [None, None]
        }

        class UnrecognizedCondition(Exception):
            pass

        def rewrite(cond):
            if isinstance(cond, Eq):
                return KroneckerDelta(*cond.args)
            if isinstance(cond, Ne):
                return 1 - KroneckerDelta(*cond.args)

            cls, args = type(cond), cond.args
            if cls not in rules:
                raise UnrecognizedCondition(cls)

            b1, b2 = rules[cls]
            k = Mul(*[1 - rewrite(c) for c in args]) if b1 else Mul(*[rewrite(c) for c in args])

            if b2:
                return 1 - k
            return k

        conditions = []
        true_value = None
        for value, cond in args:
            if type(cond) in rules:
                conditions.append((value, cond))
            elif cond is S.true:
                if true_value is None:
                    true_value = value
            else:
                return

        if true_value is not None:
            result = true_value

            for value, cond in conditions[::-1]:
                try:
                    k = rewrite(cond)
                    result = k * value + (1 - k) * result
                except UnrecognizedCondition:
                    return

            return result


def piecewise_fold(expr, evaluate=True):
    """
    Takes an expression containing a piecewise function and returns the
    expression in piecewise form. In addition, any ITE conditions are
    rewritten in negation normal form and simplified.

    The final Piecewise is evaluated (default) but if the raw form
    is desired, send ``evaluate=False``; if trivial evaluation is
    desired, send ``evaluate=None`` and duplicate conditions and
    processing of True and False will be handled.

    Examples
    ========

    >>> from sympy import Piecewise, piecewise_fold, S
    >>> from sympy.abc import x
    >>> p = Piecewise((x, x < 1), (1, S(1) <= x))
    >>> piecewise_fold(x*p)
    Piecewise((x**2, x < 1), (x, True))

    See Also
    ========

    Piecewise
    piecewise_exclusive
    """
    if not isinstance(expr, Basic) or not expr.has(Piecewise):
        return expr

    new_args = []
    if isinstance(expr, (ExprCondPair, Piecewise)):
        for e, c in expr.args:
            if not isinstance(e, Piecewise):
                e = piecewise_fold(e)
            # we don't keep Piecewise in condition because
            # it has to be checked to see that it's complete
            # and we convert it to ITE at that time
            assert not c.has(Piecewise)  # pragma: no cover
            if isinstance(c, ITE):
                c = c.to_nnf()
                c = simplify_logic(c, form='cnf')
            if isinstance(e, Piecewise):
                new_args.extend([(piecewise_fold(ei), And(ci, c))
                    for ei, ci in e.args])
            else:
                new_args.append((e, c))
    else:
        # Given
        #     P1 = Piecewise((e11, c1), (e12, c2), A)
        #     P2 = Piecewise((e21, c1), (e22, c2), B)
        #     ...
        # the folding of f(P1, P2) is trivially
        # Piecewise(
        #   (f(e11, e21), c1),
        #   (f(e12, e22), c2),
        #   (f(Piecewise(A), Piecewise(B)), True))
        # Certain objects end up rewriting themselves as thus, so
        # we do that grouping before the more generic folding.
        # The following applies this idea when f = Add or f = Mul
        # (and the expression is commutative).
        if expr.is_Add or expr.is_Mul and expr.is_commutative:
            p, args = sift(expr.args, lambda x: x.is_Piecewise, binary=True)
            pc = sift(p, lambda x: tuple([c for e,c in x.args]))
            for c in list(ordered(pc)):
                if len(pc[c]) > 1:
                    pargs = [list(i.args) for i in pc[c]]
                    # the first one is the same; there may be more
                    com = common_prefix(*[
                        [i.cond for i in j] for j in pargs])
                    n = len(com)
                    collected = []
                    for i in range(n):
                        collected.append((
                            expr.func(*[ai[i].expr for ai in pargs]),
                            com[i]))
                    remains = []
                    for a in pargs:
                        if n == len(a):  # no more args
                            continue
                        if a[n].cond == True:  # no longer Piecewise
                            remains.append(a[n].expr)
                        else:  # restore the remaining Piecewise
                            remains.append(
                                Piecewise(*a[n:], evaluate=False))
                    if remains:
                        collected.append((expr.func(*remains), True))
                    args.append(Piecewise(*collected, evaluate=False))
                    continue
                args.extend(pc[c])
        else:
            args = expr.args
        # fold
        folded = list(map(piecewise_fold, args))
        for ec in product(*[
                (i.args if isinstance(i, Piecewise) else
                 [(i, true)]) for i in folded]):
            e, c = zip(*ec)
            new_args.append((expr.func(*e), And(*c)))

    if evaluate is None:
        # don't return duplicate conditions, otherwise don't evaluate
        new_args = list(reversed([(e, c) for c, e in {
            c: e for e, c in reversed(new_args)}.items()]))
    rv = Piecewise(*new_args, evaluate=evaluate)
    if evaluate is None and len(rv.args) == 1 and rv.args[0].cond == True:
        return rv.args[0].expr
    if any(s.expr.has(Piecewise) for p in rv.atoms(Piecewise) for s in p.args):
        return piecewise_fold(rv)
    return rv


def _clip(A, B, k):
    """Return interval B as intervals that are covered by A (keyed
    to k) and all other intervals of B not covered by A keyed to -1.

    The reference point of each interval is the rhs; if the lhs is
    greater than the rhs then an interval of zero width interval will
    result, e.g. (4, 1) is treated like (1, 1).

    Examples
    ========

    >>> from sympy.functions.elementary.piecewise import _clip
    >>> from sympy import Tuple
    >>> A = Tuple(1, 3)
    >>> B = Tuple(2, 4)
    >>> _clip(A, B, 0)
    [(2, 3, 0), (3, 4, -1)]

    Interpretation: interval portion (2, 3) of interval (2, 4) is
    covered by interval (1, 3) and is keyed to 0 as requested;
    interval (3, 4) was not covered by (1, 3) and is keyed to -1.
    """
    a, b = B
    c, d = A
    c, d = Min(Max(c, a), b), Min(Max(d, a), b)
    a = Min(a, b)
    p = []
    if a != c:
        p.append((a, c, -1))
    else:
        pass
    if c != d:
        p.append((c, d, k))
    else:
        pass
    if b != d:
        if d == c and p and p[-1][-1] == -1:
            p[-1] = p[-1][0], b, -1
        else:
            p.append((d, b, -1))
    else:
        pass

    return p


def piecewise_simplify_arguments(expr, **kwargs):
    from sympy.simplify.simplify import simplify

    # simplify conditions
    f1 = expr.args[0].cond.free_symbols
    args = None
    if len(f1) == 1 and not expr.atoms(Eq):
        x = f1.pop()
        # this won't return intervals involving Eq
        # and it won't handle symbols treated as
        # booleans
        ok, abe_ = expr._intervals(x, err_on_Eq=True)
        def include(c, x, a):
            "return True if c.subs(x, a) is True, else False"
            try:
                return c.subs(x, a) == True
            except TypeError:
                return False
        if ok:
            args = []
            covered = S.EmptySet
            from sympy.sets.sets import Interval
            for a, b, e, i in abe_:
                c = expr.args[i].cond
                incl_a = include(c, x, a)
                incl_b = include(c, x, b)
                iv = Interval(a, b, not incl_a, not incl_b)
                cset = iv - covered
                if not cset:
                    continue
                try:
                    a = cset.inf
                except NotImplementedError:
                    pass # continue with the given `a`
                else:
                    incl_a = include(c, x, a)
                if incl_a and incl_b:
                    if a.is_infinite and b.is_infinite:
                        c = S.true
                    elif b.is_infinite:
                        c = (x > a) if a in covered else (x >= a)
                    elif a.is_infinite:
                        c = (x <= b)
                    elif a in covered:
                        c = And(a < x, x <= b)
                    else:
                        c = And(a <= x, x <= b)
                elif incl_a:
                    if a.is_infinite:
                        c = (x < b)
                    elif a in covered:
                        c = And(a < x, x < b)
                    else:
                        c = And(a <= x, x < b)
                elif incl_b:
                    if b.is_infinite:
                        c = (x > a)
                    else:
                        c = And(a < x, x <= b)
                else:
                    if a in covered:
                        c = (x < b)
                    else:
                        c = And(a < x, x < b)
                covered |= iv
                if a is S.NegativeInfinity and incl_a:
                    covered |= {S.NegativeInfinity}
                if b is S.Infinity and incl_b:
                    covered |= {S.Infinity}
                args.append((e, c))
            if not S.Reals.is_subset(covered):
                args.append((Undefined, True))
    if args is None:
        args = list(expr.args)
        for i in range(len(args)):
            e, c  = args[i]
            if isinstance(c, Basic):
                c = simplify(c, **kwargs)
            args[i] = (e, c)

    # simplify expressions
    doit = kwargs.pop('doit', None)
    for i in range(len(args)):
        e, c  = args[i]
        if isinstance(e, Basic):
            # Skip doit to avoid growth at every call for some integrals
            # and sums, see sympy/sympy#17165
            newe = simplify(e, doit=False, **kwargs)
            if newe != e:
                e = newe
        args[i] = (e, c)

    # restore kwargs flag
    if doit is not None:
        kwargs['doit'] = doit

    return Piecewise(*args)


def _piecewise_collapse_arguments(_args):
    newargs = []  # the unevaluated conditions
    current_cond = set()  # the conditions up to a given e, c pair
    for expr, cond in _args:
        cond = cond.replace(
            lambda _: _.is_Relational, _canonical_coeff)
        # Check here if expr is a Piecewise and collapse if one of
        # the conds in expr matches cond. This allows the collapsing
        # of Piecewise((Piecewise((x,x<0)),x<0)) to Piecewise((x,x<0)).
        # This is important when using piecewise_fold to simplify
        # multiple Piecewise instances having the same conds.
        # Eventually, this code should be able to collapse Piecewise's
        # having different intervals, but this will probably require
        # using the new assumptions.
        if isinstance(expr, Piecewise):
            unmatching = []
            for i, (e, c) in enumerate(expr.args):
                if c in current_cond:
                    # this would already have triggered
                    continue
                if c == cond:
                    if c != True:
                        # nothing past this condition will ever
                        # trigger and only those args before this
                        # that didn't match a previous condition
                        # could possibly trigger
                        if unmatching:
                            expr = Piecewise(*(
                                unmatching + [(e, c)]))
                        else:
                            expr = e
                    break
                else:
                    unmatching.append((e, c))

        # check for condition repeats
        got = False
        # -- if an And contains a condition that was
        #    already encountered, then the And will be
        #    False: if the previous condition was False
        #    then the And will be False and if the previous
        #    condition is True then then we wouldn't get to
        #    this point. In either case, we can skip this condition.
        for i in ([cond] +
                  (list(cond.args) if isinstance(cond, And) else
                  [])):
            if i in current_cond:
                got = True
                break
        if got:
            continue

        # -- if not(c) is already in current_cond then c is
        #    a redundant condition in an And. This does not
        #    apply to Or, however: (e1, c), (e2, Or(~c, d))
        #    is not (e1, c), (e2, d) because if c and d are
        #    both False this would give no results when the
        #    true answer should be (e2, True)
        if isinstance(cond, And):
            nonredundant = []
            for c in cond.args:
                if isinstance(c, Relational):
                    if c.negated.canonical in current_cond:
                        continue
                    # if a strict inequality appears after
                    # a non-strict one, then the condition is
                    # redundant
                    if isinstance(c, (Lt, Gt)) and (
                        c.weak in current_cond):
                        cond = False
                        break
                nonredundant.append(c)
            else:
                cond = cond.func(*nonredundant)
        elif isinstance(cond, Relational):
            if cond.negated.canonical in current_cond:
                cond = S.true

        current_cond.add(cond)

        # collect successive e,c pairs when exprs or cond match
        if newargs:
            if newargs[-1].expr == expr:
                orcond = Or(cond, newargs[-1].cond)
                if isinstance(orcond, (And, Or)):
                    orcond = distribute_and_over_or(orcond)
                newargs[-1] = ExprCondPair(expr, orcond)
                continue
            elif newargs[-1].cond == cond:
                continue
        newargs.append(ExprCondPair(expr, cond))
    return newargs


_blessed = lambda e: getattr(e.lhs, '_diff_wrt', False) and (
    getattr(e.rhs, '_diff_wrt', None) or
    isinstance(e.rhs, (Rational, NumberSymbol)))


def piecewise_simplify(expr, **kwargs):
    expr = piecewise_simplify_arguments(expr, **kwargs)
    if not isinstance(expr, Piecewise):
        return expr
    args = list(expr.args)

    args = _piecewise_simplify_eq_and(args)
    args = _piecewise_simplify_equal_to_next_segment(args)
    return Piecewise(*args)


def _piecewise_simplify_equal_to_next_segment(args):
    """
    See if expressions valid for an Equal expression happens to evaluate
    to the same function as in the next piecewise segment, see:
    https://github.com/sympy/sympy/issues/8458
    """
    prevexpr = None
    for i, (expr, cond) in reversed(list(enumerate(args))):
        if prevexpr is not None:
            if isinstance(cond, And):
                eqs, other = sift(cond.args,
                                  lambda i: isinstance(i, Eq), binary=True)
            elif isinstance(cond, Eq):
                eqs, other = [cond], []
            else:
                eqs = other = []
            _prevexpr = prevexpr
            _expr = expr
            if eqs and not other:
                eqs = list(ordered(eqs))
                for e in eqs:
                    # allow 2 args to collapse into 1 for any e
                    # otherwise limit simplification to only simple-arg
                    # Eq instances
                    if len(args) == 2 or _blessed(e):
                        _prevexpr = _prevexpr.subs(*e.args)
                        _expr = _expr.subs(*e.args)
            # Did it evaluate to the same?
            if _prevexpr == _expr:
                # Set the expression for the Not equal section to the same
                # as the next. These will be merged when creating the new
                # Piecewise
                args[i] = args[i].func(args[i + 1][0], cond)
            else:
                # Update the expression that we compare against
                prevexpr = expr
        else:
            prevexpr = expr
    return args


def _piecewise_simplify_eq_and(args):
    """
    Try to simplify conditions and the expression for
    equalities that are part of the condition, e.g.
    Piecewise((n, And(Eq(n,0), Eq(n + m, 0))), (1, True))
    -> Piecewise((0, And(Eq(n, 0), Eq(m, 0))), (1, True))
    """
    for i, (expr, cond) in enumerate(args):
        if isinstance(cond, And):
            eqs, other = sift(cond.args,
                              lambda i: isinstance(i, Eq), binary=True)
        elif isinstance(cond, Eq):
            eqs, other = [cond], []
        else:
            eqs = other = []
        if eqs:
            eqs = list(ordered(eqs))
            for j, e in enumerate(eqs):
                # these blessed lhs objects behave like Symbols
                # and the rhs are simple replacements for the "symbols"
                if _blessed(e):
                    expr = expr.subs(*e.args)
                    eqs[j + 1:] = [ei.subs(*e.args) for ei in eqs[j + 1:]]
                    other = [ei.subs(*e.args) for ei in other]
            cond = And(*(eqs + other))
            args[i] = args[i].func(expr, cond)
    return args


def piecewise_exclusive(expr, *, skip_nan=False, deep=True):
    """
    Rewrite :class:`Piecewise` with mutually exclusive conditions.

    Explanation
    ===========

    SymPy represents the conditions of a :class:`Piecewise` in an
    "if-elif"-fashion, allowing more than one condition to be simultaneously
    True. The interpretation is that the first condition that is True is the
    case that holds. While this is a useful representation computationally it
    is not how a piecewise formula is typically shown in a mathematical text.
    The :func:`piecewise_exclusive` function can be used to rewrite any
    :class:`Piecewise` with more typical mutually exclusive conditions.

    Note that further manipulation of the resulting :class:`Piecewise`, e.g.
    simplifying it, will most likely make it non-exclusive. Hence, this is
    primarily a function to be used in conjunction with printing the Piecewise
    or if one would like to reorder the expression-condition pairs.

    If it is not possible to determine that all possibilities are covered by
    the different cases of the :class:`Piecewise` then a final
    :class:`~sympy.core.numbers.NaN` case will be included explicitly. This
    can be prevented by passing ``skip_nan=True``.

    Examples
    ========

    >>> from sympy import piecewise_exclusive, Symbol, Piecewise, S
    >>> x = Symbol('x', real=True)
    >>> p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))
    >>> piecewise_exclusive(p)
    Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, x > 0))
    >>> piecewise_exclusive(Piecewise((2, x > 1)))
    Piecewise((2, x > 1), (nan, x <= 1))
    >>> piecewise_exclusive(Piecewise((2, x > 1)), skip_nan=True)
    Piecewise((2, x > 1))

    Parameters
    ==========

    expr: a SymPy expression.
        Any :class:`Piecewise` in the expression will be rewritten.
    skip_nan: ``bool`` (default ``False``)
        If ``skip_nan`` is set to ``True`` then a final
        :class:`~sympy.core.numbers.NaN` case will not be included.
    deep:  ``bool`` (default ``True``)
        If ``deep`` is ``True`` then :func:`piecewise_exclusive` will rewrite
        any :class:`Piecewise` subexpressions in ``expr`` rather than just
        rewriting ``expr`` itself.

    Returns
    =======

    An expression equivalent to ``expr`` but where all :class:`Piecewise` have
    been rewritten with mutually exclusive conditions.

    See Also
    ========

    Piecewise
    piecewise_fold
    """

    def make_exclusive(*pwargs):

        cumcond = false
        newargs = []

        # Handle the first n-1 cases
        for expr_i, cond_i in pwargs[:-1]:
            cancond = And(cond_i, Not(cumcond)).simplify()
            cumcond = Or(cond_i, cumcond).simplify()
            newargs.append((expr_i, cancond))

        # For the nth case defer simplification of cumcond
        expr_n, cond_n = pwargs[-1]
        cancond_n = And(cond_n, Not(cumcond)).simplify()
        newargs.append((expr_n, cancond_n))

        if not skip_nan:
            cumcond = Or(cond_n, cumcond).simplify()
            if cumcond is not true:
                newargs.append((Undefined, Not(cumcond).simplify()))

        return Piecewise(*newargs, evaluate=False)

    if deep:
        return expr.replace(Piecewise, make_exclusive)
    elif isinstance(expr, Piecewise):
        return make_exclusive(*expr.args)
    else:
        return expr
<|endoftext|>from __future__ import annotations

from sympy.core.basic import Basic
from sympy.core.expr import Expr

from sympy.core import Add, S
from sympy.core.evalf import get_integer_part, PrecisionExhausted
from sympy.core.function import DefinedFunction
from sympy.core.logic import fuzzy_or, fuzzy_and
from sympy.core.numbers import Integer, int_valued
from sympy.core.relational import Gt, Lt, Ge, Le, Relational, is_eq, is_le, is_lt
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import im, re
from sympy.multipledispatch import dispatch

###############################################################################
######################### FLOOR and CEILING FUNCTIONS #########################
###############################################################################


class RoundFunction(DefinedFunction):
    """Abstract base class for rounding functions."""

    args: tuple[Expr]

    @classmethod
    def eval(cls, arg):
        if (v := cls._eval_number(arg)) is not None:
            return v
        if (v := cls._eval_const_number(arg)) is not None:
            return v

        if arg.is_integer or arg.is_finite is False:
            return arg
        if arg.is_imaginary or (S.ImaginaryUnit*arg).is_real:
            i = im(arg)
            if not i.has(S.ImaginaryUnit):
                return cls(i)*S.ImaginaryUnit
            return cls(arg, evaluate=False)

        # Integral, numerical, symbolic part
        ipart = npart = spart = S.Zero

        # Extract integral (or complex integral) terms
        intof = lambda x: int(x) if int_valued(x) else (
            x if x.is_integer else None)
        for t in Add.make_args(arg):
            if t.is_imaginary and (i := intof(im(t))) is not None:
                ipart += i*S.ImaginaryUnit
            elif (i := intof(t)) is not None:
                ipart += i
            elif t.is_number:
                npart += t
            else:
                spart += t

        if not (npart or spart):
            return ipart

        # Evaluate npart numerically if independent of spart
        if npart and (
            not spart or
            npart.is_real and (spart.is_imaginary or (S.ImaginaryUnit*spart).is_real) or
                npart.is_imaginary and spart.is_real):
            try:
                r, i = get_integer_part(
                    npart, cls._dir, {}, return_ints=True)
                ipart += Integer(r) + Integer(i)*S.ImaginaryUnit
                npart = S.Zero
            except (PrecisionExhausted, NotImplementedError):
                pass

        spart += npart
        if not spart:
            return ipart
        elif spart.is_imaginary or (S.ImaginaryUnit*spart).is_real:
            return ipart + cls(im(spart), evaluate=False)*S.ImaginaryUnit
        elif isinstance(spart, (floor, ceiling)):
            return ipart + spart
        else:
            return ipart + cls(spart, evaluate=False)

    @classmethod
    def _eval_number(cls, arg):
        raise NotImplementedError()

    def _eval_is_finite(self):
        return self.args[0].is_finite

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_integer(self):
        return self.args[0].is_real


class floor(RoundFunction):
    """
    Floor is a univariate function which returns the largest integer
    value not greater than its argument. This implementation
    generalizes floor to complex numbers by taking the floor of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import floor, E, I, S, Float, Rational
    >>> floor(17)
    17
    >>> floor(Rational(23, 10))
    2
    >>> floor(2*E)
    5
    >>> floor(-Float(0.567))
    -1
    >>> floor(-I/2)
    -I
    >>> floor(S(5)/2 + 5*I/2)
    2 + 2*I

    See Also
    ========

    sympy.functions.elementary.integers.ceiling

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] https://mathworld.wolfram.com/FloorFunction.html

    """
    _dir = -1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.floor()
        if any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[0]

    @classmethod
    def _eval_const_number(cls, arg):
        if arg.is_real:
            if arg.is_zero:
                return S.Zero
            if arg.is_positive:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # 0 <= num/den < 1 -> 0
                if is_lt(num, den):
                    return S.Zero
                # 1 <= num/den < 2 -> 1
                if fuzzy_and([is_le(den, num), is_lt(num, 2*den)]):
                    return S.One
            if arg.is_negative:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # -1 <= num/den < 0 -> -1
                if is_le(-den, num):
                    return S.NegativeOne
                # -2 <= num/den < -1 -> -2
                if fuzzy_and([is_le(-2*den, num), is_lt(num, -den)]):
                    return Integer(-2)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        if arg0.is_finite:
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
                if ndir.is_negative:
                    return r - 1
                elif ndir.is_positive:
                    return r
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ndir)
            else:
                return r
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(-1, 0)
            return s + o
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            if ndir.is_negative:
                return r - 1
            elif ndir.is_positive:
                return r
            else:
                raise NotImplementedError("Not sure of sign of %s" % ndir)
        else:
            return r

    def _eval_is_negative(self):
        return self.args[0].is_negative

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return -ceiling(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg - frac(arg)

    def __le__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] < other + 1
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.Infinity and self.is_finite:
            return S.true

        return Le(self, other, evaluate=False)

    def __ge__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] >= other
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] >= other + 1
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Gt(self, other, evaluate=False)

    def __lt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] < other
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.true
        if other is S.Infinity and self.is_finite:
            return S.true

        return Lt(self, other, evaluate=False)


@dispatch(floor, Expr)
def _eval_is_eq(lhs, rhs): # noqa:F811
    return is_eq(lhs.rewrite(ceiling), rhs) or \
        is_eq(lhs.rewrite(frac),rhs)


class ceiling(RoundFunction):
    """
    Ceiling is a univariate function which returns the smallest integer
    value not less than its argument. This implementation
    generalizes ceiling to complex numbers by taking the ceiling of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import ceiling, E, I, S, Float, Rational
    >>> ceiling(17)
    17
    >>> ceiling(Rational(23, 10))
    3
    >>> ceiling(2*E)
    6
    >>> ceiling(-Float(0.567))
    0
    >>> ceiling(I/2)
    I
    >>> ceiling(S(5)/2 + 5*I/2)
    3 + 3*I

    See Also
    ========

    sympy.functions.elementary.integers.floor

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] https://mathworld.wolfram.com/CeilingFunction.html

    """
    _dir = 1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.ceiling()
        if any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[1]

    @classmethod
    def _eval_const_number(cls, arg):
        if arg.is_real:
            if arg.is_zero:
                return S.Zero
            if arg.is_positive:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # 0 < num/den <= 1 -> 1
                if is_le(num, den):
                    return S.One
                # 1 < num/den <= 2 -> 2
                if fuzzy_and([is_lt(den, num), is_le(num, 2*den)]):
                    return Integer(2)
            if arg.is_negative:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # -1 < num/den <= 0 -> 0
                if is_lt(-den, num):
                    return S.Zero
                # -2 < num/den <= -1 -> -1
                if fuzzy_and([is_lt(-2*den, num), is_le(num, -den)]):
                    return S.NegativeOne

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = ceiling(arg0)
        if arg0.is_finite:
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
                if ndir.is_negative:
                    return r
                elif ndir.is_positive:
                    return r + 1
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ndir)
            else:
                return r
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = ceiling(arg0)
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1)
            return s + o
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            if ndir.is_negative:
                return r
            elif ndir.is_positive:
                return r + 1
            else:
                raise NotImplementedError("Not sure of sign of %s" % ndir)
        else:
            return r

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return -floor(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg + frac(-arg)

    def _eval_is_positive(self):
        return self.args[0].is_positive

    def _eval_is_nonpositive(self):
        return self.args[0].is_nonpositive

    def __lt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] <= other - 1
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Lt(self, other, evaluate=False)

    def __gt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] > other
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.true
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Gt(self, other, evaluate=False)

    def __ge__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] > other - 1
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Ge(self, other, evaluate=False)

    def __le__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] <= other
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Le(self, other, evaluate=False)


@dispatch(ceiling, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    return is_eq(lhs.rewrite(floor), rhs) or is_eq(lhs.rewrite(frac),rhs)


class frac(DefinedFunction):
    r"""Represents the fractional part of x

    For real numbers it is defined [1]_ as

    .. math::
        x - \left\lfloor{x}\right\rfloor

    Examples
    ========

    >>> from sympy import Symbol, frac, Rational, floor, I
    >>> frac(Rational(4, 3))
    1/3
    >>> frac(-Rational(4, 3))
    2/3

    returns zero for integer arguments

    >>> n = Symbol('n', integer=True)
    >>> frac(n)
    0

    rewrite as floor

    >>> x = Symbol('x')
    >>> frac(x).rewrite(floor)
    x - floor(x)

    for complex arguments

    >>> r = Symbol('r', real=True)
    >>> t = Symbol('t', real=True)
    >>> frac(t + I*r)
    I*frac(r) + frac(t)

    See Also
    ========

    sympy.functions.elementary.integers.floor
    sympy.functions.elementary.integers.ceiling

    References
    ===========

    .. [1] https://en.wikipedia.org/wiki/Fractional_part
    .. [2] https://mathworld.wolfram.com/FractionalPart.html

    """
    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds

        def _eval(arg):
            if arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(0, 1)
            if arg.is_integer:
                return S.Zero
            if arg.is_number:
                if arg is S.NaN:
                    return S.NaN
                elif arg is S.ComplexInfinity:
                    return S.NaN
                else:
                    return arg - floor(arg)
            return cls(arg, evaluate=False)

        real, imag = S.Zero, S.Zero
        for t in Add.make_args(arg):
            # Two checks are needed for complex arguments
            # see issue-7649 for details
            if t.is_imaginary or (S.ImaginaryUnit*t).is_real:
                i = im(t)
                if not i.has(S.ImaginaryUnit):
                    imag += i
                else:
                    real += t
            else:
                real += t

        real = _eval(real)
        imag = _eval(imag)
        return real + S.ImaginaryUnit*imag

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    def _eval_is_finite(self):
        return True

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_integer

    def _eval_is_zero(self):
        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])

    def _eval_is_negative(self):
        return False

    def __ge__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.true
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
            # Check if other >= 1
            if other.is_extended_negative:
                return S.true
        return Gt(self, other, evaluate=False)

    def __le__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            if other.is_extended_negative:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Le(self, other, evaluate=False)

    def __lt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Lt(self, other, evaluate=False)

    def _value_one_or_more(self, other):
        if other.is_extended_real:
            if other.is_number:
                res = other >= 1
                if res and not isinstance(res, Relational):
                    return S.true
            if other.is_integer and other.is_positive:
                return S.true

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)

        if arg0.is_finite:
            if r.is_zero:
                ndir = arg.dir(x, cdir=cdir)
                if ndir.is_negative:
                    return S.One
                return (arg - arg0).as_leading_term(x, logx=logx, cdir=cdir)
            else:
                return r
        elif arg0 in (S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
            return AccumBounds(0, 1)
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)

        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1) + Order(x**n, (x, 0))
            return o
        else:
            res = (arg - arg0)._eval_nseries(x, n, logx=logx, cdir=cdir)
            if r.is_zero:
                ndir = arg.dir(x, cdir=cdir)
                res += S.One if ndir.is_negative else S.Zero
            else:
                res += r
            return res


@dispatch(frac, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    if (lhs.rewrite(floor) == rhs) or \
        (lhs.rewrite(ceiling) == rhs):
        return True
    # Check if other < 0
    if rhs.is_extended_negative:
        return False
    # Check if other >= 1
    res = lhs._value_one_or_more(rhs)
    if res is not None:
        return False
<|endoftext|>from sympy.printing.c import C99CodePrinter

def render_as_source_file(content, Printer=C99CodePrinter, settings=None):
    """ Renders a C source file (with required #include statements) """
    printer = Printer(settings or {})
    code_str = printer.doprint(content)
    includes = '\n'.join(['#include <%s>' % h for h in printer.headers])
    return includes + '\n\n' + code_str
<|endoftext|>"""
Classes and functions useful for rewriting expressions for optimized code
generation. Some languages (or standards thereof), e.g. C99, offer specialized
math functions for better performance and/or precision.

Using the ``optimize`` function in this module, together with a collection of
rules (represented as instances of ``Optimization``), one can rewrite the
expressions for this purpose::

    >>> from sympy import Symbol, exp, log
    >>> from sympy.codegen.rewriting import optimize, optims_c99
    >>> x = Symbol('x')
    >>> optimize(3*exp(2*x) - 3, optims_c99)
    3*expm1(2*x)
    >>> optimize(exp(2*x) - 1 - exp(-33), optims_c99)
    expm1(2*x) - exp(-33)
    >>> optimize(log(3*x + 3), optims_c99)
    log1p(x) + log(3)
    >>> optimize(log(2*x + 3), optims_c99)
    log(2*x + 3)

The ``optims_c99`` imported above is tuple containing the following instances
(which may be imported from ``sympy.codegen.rewriting``):

- ``expm1_opt``
- ``log1p_opt``
- ``exp2_opt``
- ``log2_opt``
- ``log2const_opt``


"""
from sympy.core.function import expand_log
from sympy.core.singleton import S
from sympy.core.symbol import Wild
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.assumptions import Q, ask
from sympy.codegen.cfunctions import log1p, log2, exp2, expm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.expr import UnevaluatedExpr
from sympy.core.power import Pow
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.core.mul import Mul
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.utilities.iterables import sift


class Optimization:
    """ Abstract base class for rewriting optimization.

    Subclasses should implement ``__call__`` taking an expression
    as argument.

    Parameters
    ==========
    cost_function : callable returning number
    priority : number

    """
    def __init__(self, cost_function=None, priority=1):
        self.cost_function = cost_function
        self.priority=priority

    def cheapest(self, *args):
        return min(args, key=self.cost_function)


class ReplaceOptim(Optimization):
    """ Rewriting optimization calling replace on expressions.

    Explanation
    ===========

    The instance can be used as a function on expressions for which
    it will apply the ``replace`` method (see
    :meth:`sympy.core.basic.Basic.replace`).

    Parameters
    ==========

    query :
        First argument passed to replace.
    value :
        Second argument passed to replace.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.rewriting import ReplaceOptim
    >>> from sympy.codegen.cfunctions import exp2
    >>> x = Symbol('x')
    >>> exp2_opt = ReplaceOptim(lambda p: p.is_Pow and p.base == 2,
    ...     lambda p: exp2(p.exp))
    >>> exp2_opt(2**x)
    exp2(x)

    """

    def __init__(self, query, value, **kwargs):
        super().__init__(**kwargs)
        self.query = query
        self.value = value

    def __call__(self, expr):
        return expr.replace(self.query, self.value)


def optimize(expr, optimizations):
    """ Apply optimizations to an expression.

    Parameters
    ==========

    expr : expression
    optimizations : iterable of ``Optimization`` instances
        The optimizations will be sorted with respect to ``priority`` (highest first).

    Examples
    ========

    >>> from sympy import log, Symbol
    >>> from sympy.codegen.rewriting import optims_c99, optimize
    >>> x = Symbol('x')
    >>> optimize(log(x+3)/log(2) + log(x**2 + 1), optims_c99)
    log1p(x**2) + log2(x + 3)

    """

    for optim in sorted(optimizations, key=lambda opt: opt.priority, reverse=True):
        new_expr = optim(expr)
        if optim.cost_function is None:
            expr = new_expr
        else:
            expr = optim.cheapest(expr, new_expr)
    return expr


exp2_opt = ReplaceOptim(
    lambda p: p.is_Pow and p.base == 2,
    lambda p: exp2(p.exp)
)


_d = Wild('d', properties=[lambda x: x.is_Dummy])
_u = Wild('u', properties=[lambda x: not x.is_number and not x.is_Add])
_v = Wild('v')
_w = Wild('w')
_n = Wild('n', properties=[lambda x: x.is_number])

sinc_opt1 = ReplaceOptim(
    sin(_w)/_w, sinc(_w)
)
sinc_opt2 = ReplaceOptim(
    sin(_n*_w)/_w, _n*sinc(_n*_w)
)
sinc_opts = (sinc_opt1, sinc_opt2)

log2_opt = ReplaceOptim(_v*log(_w)/log(2), _v*log2(_w), cost_function=lambda expr: expr.count(
    lambda e: (  # division & eval of transcendentals are expensive floating point operations...
        e.is_Pow and e.exp.is_negative  # division
        or (isinstance(e, (log, log2)) and not e.args[0].is_number))  # transcendental
    )
)

log2const_opt = ReplaceOptim(log(2)*log2(_w), log(_w))

logsumexp_2terms_opt = ReplaceOptim(
    lambda l: (isinstance(l, log)
               and l.args[0].is_Add
               and len(l.args[0].args) == 2
               and all(isinstance(t, exp) for t in l.args[0].args)),
    lambda l: (
        Max(*[e.args[0] for e in l.args[0].args]) +
        log1p(exp(Min(*[e.args[0] for e in l.args[0].args])))
    )
)


class FuncMinusOneOptim(ReplaceOptim):
    """Specialization of ReplaceOptim for functions evaluating "f(x) - 1".

    Explanation
    ===========

    Numerical functions which go toward one as x go toward zero is often best
    implemented by a dedicated function in order to avoid catastrophic
    cancellation. One such example is ``expm1(x)`` in the C standard library
    which evaluates ``exp(x) - 1``. Such functions preserves many more
    significant digits when its argument is much smaller than one, compared
    to subtracting one afterwards.

    Parameters
    ==========

    func :
        The function which is subtracted by one.
    func_m_1 :
        The specialized function evaluating ``func(x) - 1``.
    opportunistic : bool
        When ``True``, apply the transformation as long as the magnitude of the
        remaining number terms decreases. When ``False``, only apply the
        transformation if it completely eliminates the number term.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.codegen.rewriting import FuncMinusOneOptim
    >>> from sympy.codegen.cfunctions import expm1
    >>> x, y = symbols('x y')
    >>> expm1_opt = FuncMinusOneOptim(exp, expm1)
    >>> expm1_opt(exp(x) + 2*exp(5*y) - 3)
    expm1(x) + 2*expm1(5*y)


    """

    def __init__(self, func, func_m_1, opportunistic=True):
        weight = 10  # <-- this is an arbitrary number (heuristic)
        super().__init__(lambda e: e.is_Add, self.replace_in_Add,
                         cost_function=lambda expr: expr.count_ops() - weight*expr.count(func_m_1))
        self.func = func
        self.func_m_1 = func_m_1
        self.opportunistic = opportunistic

    def _group_Add_terms(self, add):
        numbers, non_num = sift(add.args, lambda arg: arg.is_number, binary=True)
        numsum = sum(numbers)
        terms_with_func, other = sift(non_num, lambda arg: arg.has(self.func), binary=True)
        return numsum, terms_with_func, other

    def replace_in_Add(self, e):
        """ passed as second argument to Basic.replace(...) """
        numsum, terms_with_func, other_non_num_terms = self._group_Add_terms(e)
        if numsum == 0:
            return e
        substituted, untouched = [], []
        for with_func in terms_with_func:
            if with_func.is_Mul:
                func, coeff = sift(with_func.args, lambda arg: arg.func == self.func, binary=True)
                if len(func) == 1 and len(coeff) == 1:
                    func, coeff = func[0], coeff[0]
                else:
                    coeff = None
            elif with_func.func == self.func:
                func, coeff = with_func, S.One
            else:
                coeff = None

            if coeff is not None and coeff.is_number and sign(coeff) == -sign(numsum):
                if self.opportunistic:
                    do_substitute = abs(coeff+numsum) < abs(numsum)
                else:
                    do_substitute = coeff+numsum == 0

                if do_substitute:  # advantageous substitution
                    numsum += coeff
                    substituted.append(coeff*self.func_m_1(*func.args))
                    continue
            untouched.append(with_func)

        return e.func(numsum, *substituted, *untouched, *other_non_num_terms)

    def __call__(self, expr):
        alt1 = super().__call__(expr)
        alt2 = super().__call__(expr.factor())
        return self.cheapest(alt1, alt2)


expm1_opt = FuncMinusOneOptim(exp, expm1)
cosm1_opt = FuncMinusOneOptim(cos, cosm1)
powm1_opt = FuncMinusOneOptim(Pow, powm1)

log1p_opt = ReplaceOptim(
    lambda e: isinstance(e, log),
    lambda l: expand_log(l.replace(
        log, lambda arg: log(arg.factor())
    )).replace(log(_u+1), log1p(_u))
)

def create_expand_pow_optimization(limit, *, base_req=lambda b: b.is_symbol):
    """ Creates an instance of :class:`ReplaceOptim` for expanding ``Pow``.

    Explanation
    ===========

    The requirements for expansions are that the base needs to be a symbol
    and the exponent needs to be an Integer (and be less than or equal to
    ``limit``).

    Parameters
    ==========

    limit : int
         The highest power which is expanded into multiplication.
    base_req : function returning bool
         Requirement on base for expansion to happen, default is to return
         the ``is_symbol`` attribute of the base.

    Examples
    ========

    >>> from sympy import Symbol, sin
    >>> from sympy.codegen.rewriting import create_expand_pow_optimization
    >>> x = Symbol('x')
    >>> expand_opt = create_expand_pow_optimization(3)
    >>> expand_opt(x**5 + x**3)
    x**5 + x*x*x
    >>> expand_opt(x**5 + x**3 + sin(x)**3)
    x**5 + sin(x)**3 + x*x*x
    >>> opt2 = create_expand_pow_optimization(3, base_req=lambda b: not b.is_Function)
    >>> opt2((x+1)**2 + sin(x)**2)
    sin(x)**2 + (x + 1)*(x + 1)

    """
    return ReplaceOptim(
        lambda e: e.is_Pow and base_req(e.base) and e.exp.is_Integer and abs(e.exp) <= limit,
        lambda p: (
            UnevaluatedExpr(Mul(*([p.base]*+p.exp), evaluate=False)) if p.exp > 0 else
            1/UnevaluatedExpr(Mul(*([p.base]*-p.exp), evaluate=False))
        ))

# Optimization procedures for turning A**(-1) * x into MatrixSolve(A, x)
def _matinv_predicate(expr):
    # TODO: We should be able to support more than 2 elements
    if expr.is_MatMul and len(expr.args) == 2:
        left, right = expr.args
        if left.is_Inverse and right.shape[1] == 1:
            inv_arg = left.arg
            if isinstance(inv_arg, MatrixSymbol):
                return bool(ask(Q.fullrank(left.arg)))

    return False

def _matinv_transform(expr):
    left, right = expr.args
    inv_arg = left.arg
    return MatrixSolve(inv_arg, right)


matinv_opt = ReplaceOptim(_matinv_predicate, _matinv_transform)


logaddexp_opt = ReplaceOptim(log(exp(_v)+exp(_w)), logaddexp(_v, _w))
logaddexp2_opt = ReplaceOptim(log(Pow(2, _v)+Pow(2, _w)), logaddexp2(_v, _w)*log(2))

# Collections of optimizations:
optims_c99 = (expm1_opt, log1p_opt, exp2_opt, log2_opt, log2const_opt)

optims_numpy = optims_c99 + (logaddexp_opt, logaddexp2_opt,) + sinc_opts

optims_scipy = (cosm1_opt, powm1_opt)
<|endoftext|>from .abstract_nodes import List as AbstractList
from .ast import Token


class List(AbstractList):
    pass


class NumExprEvaluate(Token):
    """represents a call to :class:`numexpr`s :func:`evaluate`"""
    __slots__ = _fields = ('expr',)
<|endoftext|>"""This module provides containers for python objects that are valid
printing targets but are not a subclass of SymPy's Printable.
"""


from sympy.core.containers import Tuple


class List(Tuple):
    """Represents a (frozen) (Python) list (for code printing purposes)."""
    def __eq__(self, other):
        if isinstance(other, list):
            return self == List(*other)
        else:
            return self.args == other

    def __hash__(self):
        return super().__hash__()
<|endoftext|>"""
Additional AST nodes for operations on matrices. The nodes in this module
are meant to represent optimization of matrix expressions within codegen's
target languages that cannot be represented by SymPy expressions.

As an example, we can use :meth:`sympy.codegen.rewriting.optimize` and the
``matin_opt`` optimization provided in :mod:`sympy.codegen.rewriting` to
transform matrix multiplication under certain assumptions:

    >>> from sympy import symbols, MatrixSymbol
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> expr = A**(-1) * x
    >>> from sympy import assuming, Q
    >>> from sympy.codegen.rewriting import matinv_opt, optimize
    >>> with assuming(Q.fullrank(A)):
    ...     optimize(expr, [matinv_opt])
    MatrixSolve(A, vector=x)
"""

from .ast import Token
from sympy.matrices import MatrixExpr
from sympy.core.sympify import sympify


class MatrixSolve(Token, MatrixExpr):
    """Represents an operation to solve a linear matrix equation.

    Parameters
    ==========

    matrix : MatrixSymbol

      Matrix representing the coefficients of variables in the linear
      equation. This matrix must be square and full-rank (i.e. all columns must
      be linearly independent) for the solving operation to be valid.

    vector : MatrixSymbol

      One-column matrix representing the solutions to the equations
      represented in ``matrix``.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol
    >>> from sympy.codegen.matrix_nodes import MatrixSolve
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> from sympy.printing.numpy import NumPyPrinter
    >>> NumPyPrinter().doprint(MatrixSolve(A, x))
    'numpy.linalg.solve(A, x)'
    >>> from sympy import octave_code
    >>> octave_code(MatrixSolve(A, x))
    'A \\\\ x'

    """
    __slots__ = _fields = ('matrix', 'vector')

    _construct_matrix = staticmethod(sympify)
    _construct_vector = staticmethod(sympify)

    @property
    def shape(self):
        return self.vector.shape

    def _eval_derivative(self, x):
        A, b = self.matrix, self.vector
        return MatrixSolve(A, b.diff(x) - A.diff(x) * MatrixSolve(A, b))
<|endoftext|>"""
Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.


AST Type Tree
-------------
::

  *Basic*
       |
       |
   CodegenAST
       |
       |--->AssignmentBase
       |             |--->Assignment
       |             |--->AugmentedAssignment
       |                                    |--->AddAugmentedAssignment
       |                                    |--->SubAugmentedAssignment
       |                                    |--->MulAugmentedAssignment
       |                                    |--->DivAugmentedAssignment
       |                                    |--->ModAugmentedAssignment
       |
       |--->CodeBlock
       |
       |
       |--->Token
                |--->Attribute
                |--->For
                |--->String
                |       |--->QuotedString
                |       |--->Comment
                |--->Type
                |       |--->IntBaseType
                |       |              |--->_SizedIntType
                |       |                               |--->SignedIntType
                |       |                               |--->UnsignedIntType
                |       |--->FloatBaseType
                |                        |--->FloatType
                |                        |--->ComplexBaseType
                |                                           |--->ComplexType
                |--->Node
                |       |--->Variable
                |       |           |---> Pointer
                |       |--->FunctionPrototype
                |                            |--->FunctionDefinition
                |--->Element
                |--->Declaration
                |--->While
                |--->Scope
                |--->Stream
                |--->Print
                |--->FunctionCall
                |--->BreakToken
                |--->ContinueToken
                |--->NoneToken
                |--->Return


Predefined types
----------------

A number of ``Type`` instances are provided in the ``sympy.codegen.ast`` module
for convenience. Perhaps the two most common ones for code-generation (of numeric
codes) are ``float32`` and ``float64`` (known as single and double precision respectively).
There are also precision generic versions of Types (for which the codeprinters selects the
underlying data type at time of printing): ``real``, ``integer``, ``complex_``, ``bool_``.

The other ``Type`` instances defined are:

- ``intc``: Integer type used by C's "int".
- ``intp``: Integer type used by C's "unsigned".
- ``int8``, ``int16``, ``int32``, ``int64``: n-bit integers.
- ``uint8``, ``uint16``, ``uint32``, ``uint64``: n-bit unsigned integers.
- ``float80``: known as "extended precision" on modern x86/amd64 hardware.
- ``complex64``: Complex number represented by two ``float32`` numbers
- ``complex128``: Complex number represented by two ``float64`` numbers

Using the nodes
---------------

It is possible to construct simple algorithms using the AST nodes. Let's construct a loop applying
Newton's method::

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import While, Assignment, aug_assign, Print, QuotedString
    >>> t, dx, x = symbols('tol delta val')
    >>> expr = cos(x) - x**3
    >>> whl = While(abs(dx) > t, [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx),
    ...     Print([x])
    ... ])
    >>> from sympy import pycode
    >>> py_str = pycode(whl)
    >>> print(py_str)
    while (abs(delta) > tol):
        delta = (val**3 - math.cos(val))/(-3*val**2 - math.sin(val))
        val += delta
        print(val)
    >>> import math
    >>> tol, val, delta = 1e-5, 0.5, float('inf')
    >>> exec(py_str)
    1.1121416371
    0.909672693737
    0.867263818209
    0.865477135298
    0.865474033111
    >>> print('%3.1g' % (math.cos(val) - val**3))
    -3e-11

If we want to generate Fortran code for the same while loop we simple call ``fcode``::

    >>> from sympy import fcode
    >>> print(fcode(whl, standard=2003, source_format='free'))
    do while (abs(delta) > tol)
       delta = (val**3 - cos(val))/(-3*val**2 - sin(val))
       val = val + delta
       print *, val
    end do

There is a function constructing a loop (or a complete function) like this in
:mod:`sympy.codegen.algorithms`.

"""

from __future__ import annotations
from typing import Any

from collections import defaultdict

from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
                                       numbered_symbols, filter_symbols)


def _mk_Tuple(args):
    """
    Create a SymPy Tuple object from an iterable, converting Python strings to
    AST strings.

    Parameters
    ==========

    args: iterable
        Arguments to :class:`sympy.Tuple`.

    Returns
    =======

    sympy.Tuple
    """
    args = [String(arg) if isinstance(arg, str) else arg for arg in args]
    return Tuple(*args)


class CodegenAST(Basic):
    __slots__ = ()


class Token(CodegenAST):
    """ Base class for the AST types.

    Explanation
    ===========

    Defining fields are set in ``_fields``. Attributes (defined in _fields)
    are only allowed to contain instances of Basic (unless atomic, see
    ``String``). The arguments to ``__new__()`` correspond to the attributes in
    the order defined in ``_fields`. The ``defaults`` class attribute is a
    dictionary mapping attribute names to their default values.

    Subclasses should not need to override the ``__new__()`` method. They may
    define a class or static method named ``_construct_<attr>`` for each
    attribute to process the value passed to ``__new__()``. Attributes listed
    in the class attribute ``not_in_args`` are not passed to :class:`~.Basic`.
    """

    __slots__: tuple[str, ...] = ()
    _fields = __slots__
    defaults: dict[str, Any] = {}
    not_in_args: list[str] = []
    indented_args = ['body']

    @property
    def is_Atom(self):
        return len(self._fields) == 0

    @classmethod
    def _get_constructor(cls, attr):
        """ Get the constructor function for an attribute by name. """
        return getattr(cls, '_construct_%s' % attr, lambda x: x)

    @classmethod
    def _construct(cls, attr, arg):
        """ Construct an attribute value from argument passed to ``__new__()``. """
        # arg may be ``NoneToken()``, so comparison is done using == instead of ``is`` operator
        if arg == None:
            return cls.defaults.get(attr, none)
        else:
            if isinstance(arg, Dummy):  # SymPy's replace uses Dummy instances
                return arg
            else:
                return cls._get_constructor(attr)(arg)

    def __new__(cls, *args, **kwargs):
        # Pass through existing instances when given as sole argument
        if len(args) == 1 and not kwargs and isinstance(args[0], cls):
            return args[0]

        if len(args) > len(cls._fields):
            raise ValueError("Too many arguments (%d), expected at most %d" % (len(args), len(cls._fields)))

        attrvals = []

        # Process positional arguments
        for attrname, argval in zip(cls._fields, args):
            if attrname in kwargs:
                raise TypeError('Got multiple values for attribute %r' % attrname)

            attrvals.append(cls._construct(attrname, argval))

        # Process keyword arguments
        for attrname in cls._fields[len(args):]:
            if attrname in kwargs:
                argval = kwargs.pop(attrname)

            elif attrname in cls.defaults:
                argval = cls.defaults[attrname]

            else:
                raise TypeError('No value for %r given and attribute has no default' % attrname)

            attrvals.append(cls._construct(attrname, argval))

        if kwargs:
            raise ValueError("Unknown keyword arguments: %s" % ' '.join(kwargs))

        # Parent constructor
        basic_args = [
            val for attr, val in zip(cls._fields, attrvals)
            if attr not in cls.not_in_args
        ]
        obj = CodegenAST.__new__(cls, *basic_args)

        # Set attributes
        for attr, arg in zip(cls._fields, attrvals):
            setattr(obj, attr, arg)

        return obj

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for attr in self._fields:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _hashable_content(self):
        return tuple([getattr(self, attr) for attr in self._fields])

    def __hash__(self):
        return super().__hash__()

    def _joiner(self, k, indent_level):
        return (',\n' + ' '*indent_level) if k in self.indented_args else ', '

    def _indented(self, printer, k, v, *args, **kwargs):
        il = printer._context['indent_level']
        def _print(arg):
            if isinstance(arg, Token):
                return printer._print(arg, *args, joiner=self._joiner(k, il), **kwargs)
            else:
                return printer._print(arg, *args, **kwargs)

        if isinstance(v, Tuple):
            joined = self._joiner(k, il).join([_print(arg) for arg in v.args])
            if k in self.indented_args:
                return '(\n' + ' '*il + joined + ',\n' + ' '*(il - 4) + ')'
            else:
                return ('({0},)' if len(v.args) == 1 else '({0})').format(joined)
        else:
            return _print(v)

    def _sympyrepr(self, printer, *args, joiner=', ', **kwargs):
        from sympy.printing.printer import printer_context
        exclude = kwargs.get('exclude', ())
        values = [getattr(self, k) for k in self._fields]
        indent_level = printer._context.get('indent_level', 0)

        arg_reprs = []

        for i, (attr, value) in enumerate(zip(self._fields, values)):
            if attr in exclude:
                continue

            # Skip attributes which have the default value
            if attr in self.defaults and value == self.defaults[attr]:
                continue

            ilvl = indent_level + 4 if attr in self.indented_args else 0
            with printer_context(printer, indent_level=ilvl):
                indented = self._indented(printer, attr, value, *args, **kwargs)
            arg_reprs.append(('{1}' if i == 0 else '{0}={1}').format(attr, indented.lstrip()))

        return "{}({})".format(self.__class__.__name__, joiner.join(arg_reprs))

    _sympystr = _sympyrepr

    def __repr__(self):  # sympy.core.Basic.__repr__ uses sstr
        from sympy.printing import srepr
        return srepr(self)

    def kwargs(self, exclude=(), apply=None):
        """ Get instance's attributes as dict of keyword arguments.

        Parameters
        ==========

        exclude : collection of str
            Collection of keywords to exclude.

        apply : callable, optional
            Function to apply to all values.
        """
        kwargs = {k: getattr(self, k) for k in self._fields if k not in exclude}
        if apply is not None:
            return {k: apply(v) for k, v in kwargs.items()}
        else:
            return kwargs

class BreakToken(Token):
    """ Represents 'break' in C/Python ('exit' in Fortran).

    Use the premade instance ``break_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import break_
    >>> ccode(break_)
    'break'
    >>> fcode(break_, source_format='free')
    'exit'
    """

break_ = BreakToken()


class ContinueToken(Token):
    """ Represents 'continue' in C/Python ('cycle' in Fortran)

    Use the premade instance ``continue_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import continue_
    >>> ccode(continue_)
    'continue'
    >>> fcode(continue_, source_format='free')
    'cycle'
    """

continue_ = ContinueToken()

class NoneToken(Token):
    """ The AST equivalence of Python's NoneType

    The corresponding instance of Python's ``None`` is ``none``.

    Examples
    ========

    >>> from sympy.codegen.ast import none, Variable
    >>> from sympy import pycode
    >>> print(pycode(Variable('x').as_Declaration(value=none)))
    x = None

    """
    def __eq__(self, other):
        return other is None or isinstance(other, NoneToken)

    def _hashable_content(self):
        return ()

    def __hash__(self):
        return super().__hash__()


none = NoneToken()


class AssignmentBase(CodegenAST):
    """ Abstract base class for Assignment and AugmentedAssignment.

    Attributes:
    ===========

    op : str
        Symbol for assignment operator, e.g. "=", "+=", etc.
    """

    def __new__(cls, lhs, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)

        cls._check_args(lhs, rhs)

        return super().__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    @classmethod
    def _check_args(cls, lhs, rhs):
        """ Check arguments to __new__ and raise exception if any problems found.

        Derived classes may wish to override this.
        """
        from sympy.matrices.expressions.matexpr import (
            MatrixElement, MatrixSymbol)
        from sympy.tensor.indexed import Indexed
        from sympy.tensor.array.expressions import ArrayElement

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable,
                ArrayElement)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        # Indexed types implement shape, but don't define it until later. This
        # causes issues in assignment validation. For now, matrices are defined
        # as anything with a shape that is not an Indexed
        lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
        rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)

        # If lhs and rhs have same structure, then this assignment is ok
        if lhs_is_mat:
            if not rhs_is_mat:
                raise ValueError("Cannot assign a scalar to a matrix.")
            elif lhs.shape != rhs.shape:
                raise ValueError("Dimensions of lhs and rhs do not align.")
        elif rhs_is_mat and not lhs_is_mat:
            raise ValueError("Cannot assign a matrix to a scalar.")


class Assignment(AssignmentBase):
    """
    Represents variable assignment for code generation.

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.codegen.ast import Assignment
    >>> x, y, z = symbols('x, y, z')
    >>> Assignment(x, y)
    Assignment(x, y)
    >>> Assignment(x, 0)
    Assignment(x, 0)
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assignment(A, mat)
    Assignment(A, Matrix([[x, y, z]]))
    >>> Assignment(A[0, 1], x)
    Assignment(A[0, 1], x)
    """

    op = ':='


class AugmentedAssignment(AssignmentBase):
    """
    Base class for augmented assignments.

    Attributes:
    ===========

    binop : str
       Symbol for binary operation being applied in the assignment, such as "+",
       "*", etc.
    """
    binop: str | None

    @property
    def op(self):
        return self.binop + '='


class AddAugmentedAssignment(AugmentedAssignment):
    binop = '+'


class SubAugmentedAssignment(AugmentedAssignment):
    binop = '-'


class MulAugmentedAssignment(AugmentedAssignment):
    binop = '*'


class DivAugmentedAssignment(AugmentedAssignment):
    binop = '/'


class ModAugmentedAssignment(AugmentedAssignment):
    binop = '%'


# Mapping from binary op strings to AugmentedAssignment subclasses
augassign_classes = {
    cls.binop: cls for cls in [
        AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
        DivAugmentedAssignment, ModAugmentedAssignment
    ]
}


def aug_assign(lhs, op, rhs):
    """
    Create 'lhs op= rhs'.

    Explanation
    ===========

    Represents augmented variable assignment for code generation. This is a
    convenience function. You can also use the AugmentedAssignment classes
    directly, like AddAugmentedAssignment(x, y).

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : str
        Operator (+, -, /, \\*, %).

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.codegen.ast import aug_assign
    >>> x, y = symbols('x, y')
    >>> aug_assign(x, '+', y)
    AddAugmentedAssignment(x, y)
    """
    if op not in augassign_classes:
        raise ValueError("Unrecognized operator %s" % op)
    return augassign_classes[op](lhs, rhs)


class CodeBlock(CodegenAST):
    """
    Represents a block of code.

    Explanation
    ===========

    For now only assignments are supported. This restriction will be lifted in
    the future.

    Useful attributes on this object are:

    ``left_hand_sides``:
        Tuple of left-hand sides of assignments, in order.
    ``left_hand_sides``:
        Tuple of right-hand sides of assignments, in order.
    ``free_symbols``: Free symbols of the expressions in the right-hand sides
        which do not appear in the left-hand side of an assignment.

    Useful methods on this object are:

    ``topological_sort``:
        Class method. Return a CodeBlock with assignments
        sorted so that variables are assigned before they
        are used.
    ``cse``:
        Return a new CodeBlock with common subexpressions eliminated and
        pulled out as assignments.

    Examples
    ========

    >>> from sympy import symbols, ccode
    >>> from sympy.codegen.ast import CodeBlock, Assignment
    >>> x, y = symbols('x y')
    >>> c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    >>> print(ccode(c))
    x = 1;
    y = x + 1;

    """
    def __new__(cls, *args):
        left_hand_sides = []
        right_hand_sides = []
        for i in args:
            if isinstance(i, Assignment):
                lhs, rhs = i.args
                left_hand_sides.append(lhs)
                right_hand_sides.append(rhs)

        obj = CodegenAST.__new__(cls, *args)

        obj.left_hand_sides = Tuple(*left_hand_sides)
        obj.right_hand_sides = Tuple(*right_hand_sides)
        return obj

    def __iter__(self):
        return iter(self.args)

    def _sympyrepr(self, printer, *args, **kwargs):
        il = printer._context.get('indent_level', 0)
        joiner = ',\n' + ' '*il
        joined = joiner.join(map(printer._print, self.args))
        return ('{}(\n'.format(' '*(il-4) + self.__class__.__name__,) +
                ' '*il + joined + '\n' + ' '*(il - 4) + ')')

    _sympystr = _sympyrepr

    @property
    def free_symbols(self):
        return super().free_symbols - set(self.left_hand_sides)

    @classmethod
    def topological_sort(cls, assignments):
        """
        Return a CodeBlock with topologically sorted assignments so that
        variables are assigned before they are used.

        Examples
        ========

        The existing order of assignments is preserved as much as possible.

        This function assumes that variables are assigned to only once.

        This is a class constructor so that the default constructor for
        CodeBlock can error when variables are used before they are assigned.

        >>> from sympy import symbols
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> assignments = [
        ...     Assignment(x, y + z),
        ...     Assignment(y, z + 1),
        ...     Assignment(z, 2),
        ... ]
        >>> CodeBlock.topological_sort(assignments)
        CodeBlock(
            Assignment(z, 2),
            Assignment(y, z + 1),
            Assignment(x, y + z)
        )

        """

        if not all(isinstance(i, Assignment) for i in assignments):
            # Will support more things later
            raise NotImplementedError("CodeBlock.topological_sort only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in assignments):
            raise NotImplementedError("CodeBlock.topological_sort does not yet work with AugmentedAssignments")

        # Create a graph where the nodes are assignments and there is a directed edge
        # between nodes that use a variable and nodes that assign that
        # variable, like

        # [(x := 1, y := x + 1), (x := 1, z := y + z), (y := x + 1, z := y + z)]

        # If we then topologically sort these nodes, they will be in
        # assignment order, like

        # x := 1
        # y := x + 1
        # z := y + z

        # A = The nodes
        #
        # enumerate keeps nodes in the same order they are already in if
        # possible. It will also allow us to handle duplicate assignments to
        # the same variable when those are implemented.
        A = list(enumerate(assignments))

        # var_map = {variable: [nodes for which this variable is assigned to]}
        # like {x: [(1, x := y + z), (4, x := 2 * w)], ...}
        var_map = defaultdict(list)
        for node in A:
            i, a = node
            var_map[a.lhs].append(node)

        # E = Edges in the graph
        E = []
        for dst_node in A:
            i, a = dst_node
            for s in a.rhs.free_symbols:
                for src_node in var_map[s]:
                    E.append((src_node, dst_node))

        ordered_assignments = topological_sort([A, E])

        # De-enumerate the result
        return cls(*[a for i, a in ordered_assignments])

    def cse(self, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
        """
        Return a new code block with common subexpressions eliminated.

        Explanation
        ===========

        See the docstring of :func:`sympy.simplify.cse_main.cse` for more
        information.

        Examples
        ========

        >>> from sympy import symbols, sin
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> c = CodeBlock(
        ...     Assignment(x, 1),
        ...     Assignment(y, sin(x) + 1),
        ...     Assignment(z, sin(x) - 1),
        ... )
        ...
        >>> c.cse()
        CodeBlock(
            Assignment(x, 1),
            Assignment(x0, sin(x)),
            Assignment(y, x0 + 1),
            Assignment(z, x0 - 1)
        )

        """
        from sympy.simplify.cse_main import cse

        # Check that the CodeBlock only contains assignments to unique variables
        if not all(isinstance(i, Assignment) for i in self.args):
            # Will support more things later
            raise NotImplementedError("CodeBlock.cse only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in self.args):
            raise NotImplementedError("CodeBlock.cse does not yet work with AugmentedAssignments")

        for i, lhs in enumerate(self.left_hand_sides):
            if lhs in self.left_hand_sides[:i]:
                raise NotImplementedError("Duplicate assignments to the same "
                    "variable are not yet supported (%s)" % lhs)

        # Ensure new symbols for subexpressions do not conflict with existing
        existing_symbols = self.atoms(Symbol)
        if symbols is None:
            symbols = numbered_symbols()
        symbols = filter_symbols(symbols, existing_symbols)

        replacements, reduced_exprs = cse(list(self.right_hand_sides),
            symbols=symbols, optimizations=optimizations, postprocess=postprocess,
            order=order)

        new_block = [Assignment(var, expr) for var, expr in
            zip(self.left_hand_sides, reduced_exprs)]
        new_assignments = [Assignment(var, expr) for var, expr in replacements]
        return self.topological_sort(new_assignments + new_block)


class For(Token):
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ==========

    target : symbol
    iter : iterable
    body : CodeBlock or iterable
!        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Range
    >>> from sympy.codegen.ast import aug_assign, For
    >>> x, i, j, k = symbols('x i j k')
    >>> for_i = For(i, Range(10), [aug_assign(x, '+', i*j*k)])
    >>> for_i  # doctest: -NORMALIZE_WHITESPACE
    For(i, iterable=Range(0, 10, 1), body=CodeBlock(
        AddAugmentedAssignment(x, i*j*k)
    ))
    >>> for_ji = For(j, Range(7), [for_i])
    >>> for_ji  # doctest: -NORMALIZE_WHITESPACE
    For(j, iterable=Range(0, 7, 1), body=CodeBlock(
        For(i, iterable=Range(0, 10, 1), body=CodeBlock(
            AddAugmentedAssignment(x, i*j*k)
        ))
    ))
    >>> for_kji =For(k, Range(5), [for_ji])
    >>> for_kji  # doctest: -NORMALIZE_WHITESPACE
    For(k, iterable=Range(0, 5, 1), body=CodeBlock(
        For(j, iterable=Range(0, 7, 1), body=CodeBlock(
            For(i, iterable=Range(0, 10, 1), body=CodeBlock(
                AddAugmentedAssignment(x, i*j*k)
            ))
        ))
    ))
    """
    __slots__ = _fields = ('target', 'iterable', 'body')
    _construct_target = staticmethod(_sympify)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def _construct_iterable(cls, itr):
        if not iterable(itr):
            raise TypeError("iterable must be an iterable")
        if isinstance(itr, list):  # _sympify errors on lists because they are mutable
            itr = tuple(itr)
        return _sympify(itr)


class String(Atom, Token):
    """ SymPy object representing a string.

    Atomic object which is not an expression (as opposed to Symbol).

    Parameters
    ==========

    text : str

    Examples
    ========

    >>> from sympy.codegen.ast import String
    >>> f = String('foo')
    >>> f
    foo
    >>> str(f)
    'foo'
    >>> f.text
    'foo'
    >>> print(repr(f))
    String('foo')

    """
    __slots__ = _fields = ('text',)
    not_in_args = ['text']
    is_Atom = True

    @classmethod
    def _construct_text(cls, text):
        if not isinstance(text, str):
            raise TypeError("Argument text is not a string type.")
        return text

    def _sympystr(self, printer, *args, **kwargs):
        return self.text

    def kwargs(self, exclude = (), apply = None):
        return {}

    #to be removed when Atom is given a suitable func
    @property
    def func(self):
        return lambda: self

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        return r'\texttt{{"{}"}}'.format(latex_escape(self.text))

class QuotedString(String):
    """ Represents a string which should be printed with quotes. """

class Comment(String):
    """ Represents a comment. """

class Node(Token):
    """ Subclass of Token, carrying the attribute 'attrs' (Tuple)

    Examples
    ========

    >>> from sympy.codegen.ast import Node, value_const, pointer_const
    >>> n1 = Node([value_const])
    >>> n1.attr_params('value_const')  # get the parameters of attribute (by name)
    ()
    >>> from sympy.codegen.fnodes import dimension
    >>> n2 = Node([value_const, dimension(5, 3)])
    >>> n2.attr_params(value_const)  # get the parameters of attribute (by Attribute instance)
    ()
    >>> n2.attr_params('dimension')  # get the parameters of attribute (by name)
    (5, 3)
    >>> n2.attr_params(pointer_const) is None
    True

    """

    __slots__: tuple[str, ...] = ('attrs',)
    _fields = __slots__

    defaults: dict[str, Any] = {'attrs': Tuple()}

    _construct_attrs = staticmethod(_mk_Tuple)

    def attr_params(self, looking_for):
        """ Returns the parameters of the Attribute with name ``looking_for`` in self.attrs """
        for attr in self.attrs:
            if str(attr.name) == str(looking_for):
                return attr.parameters


class Type(Token):
    """ Represents a type.

    Explanation
    ===========

    The naming is a super-set of NumPy naming. Type has a classmethod
    ``from_expr`` which offer type deduction. It also has a method
    ``cast_check`` which casts the argument to its type, possibly raising an
    exception if rounding error is not within tolerances, or if the value is not
    representable by the underlying data type (e.g. unsigned integers).

    Parameters
    ==========

    name : str
        Name of the type, e.g. ``object``, ``int16``, ``float16`` (where the latter two
        would use the ``Type`` sub-classes ``IntType`` and ``FloatType`` respectively).
        If a ``Type`` instance is given, the said instance is returned.

    Examples
    ========

    >>> from sympy.codegen.ast import Type
    >>> t = Type.from_expr(42)
    >>> t
    integer
    >>> print(repr(t))
    IntBaseType(String('integer'))
    >>> from sympy.codegen.ast import uint8
    >>> uint8.cast_check(-1)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Minimum value for data type bigger than new value.
    >>> from sympy.codegen.ast import float32
    >>> v6 = 0.123456
    >>> float32.cast_check(v6)
    0.123456
    >>> v10 = 12345.67894
    >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Casting gives a significantly different value.
    >>> boost_mp50 = Type('boost::multiprecision::cpp_dec_float_50')
    >>> from sympy import cxxcode
    >>> from sympy.codegen.ast import Declaration, Variable
    >>> cxxcode(Declaration(Variable('x', type=boost_mp50)))
    'boost::multiprecision::cpp_dec_float_50 x'

    References
    ==========

    .. [1] https://numpy.org/doc/stable/user/basics.types.html

    """
    __slots__: tuple[str, ...] = ('name',)
    _fields = __slots__

    _construct_name = String

    def _sympystr(self, printer, *args, **kwargs):
        return str(self.name)

    @classmethod
    def from_expr(cls, expr):
        """ Deduces type from an expression or a ``Symbol``.

        Parameters
        ==========

        expr : number or SymPy object
            The type will be deduced from type or properties.

        Examples
        ========

        >>> from sympy.codegen.ast import Type, integer, complex_
        >>> Type.from_expr(2) == integer
        True
        >>> from sympy import Symbol
        >>> Type.from_expr(Symbol('z', complex=True)) == complex_
        True
        >>> Type.from_expr(sum)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Could not deduce type from expr.

        Raises
        ======

        ValueError when type deduction fails.

        """
        if isinstance(expr, (float, Float)):
            return real
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return integer
        if getattr(expr, 'is_real', False):
            return real
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return complex_
        if isinstance(expr, bool) or getattr(expr, 'is_Relational', False):
            return bool_
        else:
            raise ValueError("Could not deduce type from expr.")

    def _check(self, value):
        pass

    def cast_check(self, value, rtol=None, atol=0, precision_targets=None):
        """ Casts a value to the data type of the instance.

        Parameters
        ==========

        value : number
        rtol : floating point number
            Relative tolerance. (will be deduced if not given).
        atol : floating point number
            Absolute tolerance (in addition to ``rtol``).
        type_aliases : dict
            Maps substitutions for Type, e.g. {integer: int64, real: float32}

        Examples
        ========

        >>> from sympy.codegen.ast import integer, float32, int8
        >>> integer.cast_check(3.0) == 3
        True
        >>> float32.cast_check(1e-40)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Minimum value for data type bigger than new value.
        >>> int8.cast_check(256)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Maximum value for data type smaller than new value.
        >>> v10 = 12345.67894
        >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float64
        >>> float64.cast_check(v10)
        12345.67894
        >>> from sympy import Float
        >>> v18 = Float('0.123456789012345646')
        >>> float64.cast_check(v18)
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float80
        >>> float80.cast_check(v18)
        0.123456789012345649

        """
        val = sympify(value)

        ten = Integer(10)
        exp10 = getattr(self, 'decimal_dig', None)

        if rtol is None:
            rtol = 1e-15 if exp10 is None else 2.0*ten**(-exp10)

        def tol(num):
            return atol + rtol*abs(num)

        new_val = self.cast_nocheck(value)
        self._check(new_val)

        delta = new_val - val
        if abs(delta) > tol(val):  # rounding, e.g. int(3.5) != 3.5
            raise ValueError("Casting gives a significantly different value.")

        return new_val

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        type_name = latex_escape(self.__class__.__name__)
        name = latex_escape(self.name.text)
        return r"\text{{{}}}\left(\texttt{{{}}}\right)".format(type_name, name)


class IntBaseType(Type):
    """ Integer base type, contains no size information. """
    __slots__ = ()
    cast_nocheck = lambda self, i: Integer(int(i))


class _SizedIntType(IntBaseType):
    __slots__ = ('nbits',)
    _fields = Type._fields + __slots__

    _construct_nbits = Integer

    def _check(self, value):
        if value < self.min:
            raise ValueError("Value is too small: %d < %d" % (value, self.min))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))


class SignedIntType(_SizedIntType):
    """ Represents a signed integer type. """
    __slots__ = ()
    @property
    def min(self):
        return -2**(self.nbits-1)

    @property
    def max(self):
        return 2**(self.nbits-1) - 1


class UnsignedIntType(_SizedIntType):
    """ Represents an unsigned integer type. """
    __slots__ = ()
    @property
    def min(self):
        return 0

    @property
    def max(self):
        return 2**self.nbits - 1

two = Integer(2)

class FloatBaseType(Type):
    """ Represents a floating point number type. """
    __slots__ = ()
    cast_nocheck = Float

class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the mantissa.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """

    __slots__ = ('nbits', 'nmant', 'nexp',)
    _fields = Type._fields + __slots__

    _construct_nbits = _construct_nmant = _construct_nexp = Integer


    @property
    def max_exponent(self):
        """ The largest positive number n, such that 2**(n - 1) is a representable finite value. """
        # cf. C++'s ``std::numeric_limits::max_exponent``
        return two**(self.nexp - 1)

    @property
    def min_exponent(self):
        """ The lowest negative number n, such that 2**(n - 1) is a valid normalized number. """
        # cf. C++'s ``std::numeric_limits::min_exponent``
        return 3 - self.max_exponent

    @property
    def max(self):
        """ Maximum value representable. """
        return (1 - two**-(self.nmant+1))*two**self.max_exponent

    @property
    def tiny(self):
        """ The minimum positive normalized value. """
        # See C macros: FLT_MIN, DBL_MIN, LDBL_MIN
        # or C++'s ``std::numeric_limits::min``
        # or numpy.finfo(dtype).tiny
        return two**(self.min_exponent - 1)


    @property
    def eps(self):
        """ Difference between 1.0 and the next representable value. """
        return two**(-self.nmant)

    @property
    def dig(self):
        """ Number of decimal digits that are guaranteed to be preserved in text.

        When converting text -> float -> text, you are guaranteed that at least ``dig``
        number of digits are preserved with respect to rounding or overflow.
        """
        from sympy.functions import floor, log
        return floor(self.nmant * log(2)/log(10))

    @property
    def decimal_dig(self):
        """ Number of digits needed to store & load without loss.

        Explanation
        ===========

        Number of decimal digits needed to guarantee that two consecutive conversions
        (float -> text -> float) to be idempotent. This is useful when one do not want
        to loose precision due to rounding errors when storing a floating point value
        as text.
        """
        from sympy.functions import ceiling, log
        return ceiling((self.nmant + 1) * log(2)/log(10) + 1)

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        if value == oo:  # float(oo) or oo
            return float(oo)
        elif value == -oo:  # float(-oo) or -oo
            return float(-oo)
        return Float(str(sympify(value).evalf(self.decimal_dig)), self.decimal_dig)

    def _check(self, value):
        if value < -self.max:
            raise ValueError("Value is too small: %d < %d" % (value, -self.max))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))
        if abs(value) < self.tiny:
            raise ValueError("Smallest (absolute) value for data type bigger than new value.")

class ComplexBaseType(FloatBaseType):

    __slots__ = ()

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        from sympy.functions import re, im
        return (
            super().cast_nocheck(re(value)) +
            super().cast_nocheck(im(value))*1j
        )

    def _check(self, value):
        from sympy.functions import re, im
        super()._check(re(value))
        super()._check(im(value))


class ComplexType(ComplexBaseType, FloatType):
    """ Represents a complex floating point number. """
    __slots__ = ()


# NumPy types:
intc = IntBaseType('intc')
intp = IntBaseType('intp')
int8 = SignedIntType('int8', 8)
int16 = SignedIntType('int16', 16)
int32 = SignedIntType('int32', 32)
int64 = SignedIntType('int64', 64)
uint8 = UnsignedIntType('uint8', 8)
uint16 = UnsignedIntType('uint16', 16)
uint32 = UnsignedIntType('uint32', 32)
uint64 = UnsignedIntType('uint64', 64)
float16 = FloatType('float16', 16, nexp=5, nmant=10)  # IEEE 754 binary16, Half precision
float32 = FloatType('float32', 32, nexp=8, nmant=23)  # IEEE 754 binary32, Single precision
float64 = FloatType('float64', 64, nexp=11, nmant=52)  # IEEE 754 binary64, Double precision
float80 = FloatType('float80', 80, nexp=15, nmant=63)  # x86 extended precision (1 integer part bit), "long double"
float128 = FloatType('float128', 128, nexp=15, nmant=112)  # IEEE 754 binary128, Quadruple precision
float256 = FloatType('float256', 256, nexp=19, nmant=236)  # IEEE 754 binary256, Octuple precision

complex64 = ComplexType('complex64', nbits=64, **float32.kwargs(exclude=('name', 'nbits')))
complex128 = ComplexType('complex128', nbits=128, **float64.kwargs(exclude=('name', 'nbits')))

# Generic types (precision may be chosen by code printers):
untyped = Type('untyped')
real = FloatBaseType('real')
integer = IntBaseType('integer')
complex_ = ComplexBaseType('complex')
bool_ = Type('bool')


class Attribute(Token):
    """ Attribute (possibly parametrized)

    For use with :class:`sympy.codegen.ast.Node` (which takes instances of
    ``Attribute`` as ``attrs``).

    Parameters
    ==========

    name : str
    parameters : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import Attribute
    >>> volatile = Attribute('volatile')
    >>> volatile
    volatile
    >>> print(repr(volatile))
    Attribute(String('volatile'))
    >>> a = Attribute('foo', [1, 2, 3])
    >>> a
    foo(1, 2, 3)
    >>> a.parameters == (1, 2, 3)
    True
    """
    __slots__ = _fields = ('name', 'parameters')
    defaults = {'parameters': Tuple()}

    _construct_name = String
    _construct_parameters = staticmethod(_mk_Tuple)

    def _sympystr(self, printer, *args, **kwargs):
        result = str(self.name)
        if self.parameters:
            result += '(%s)' % ', '.join((printer._print(
                arg, *args, **kwargs) for arg in self.parameters))
        return result

value_const = Attribute('value_const')
pointer_const = Attribute('pointer_const')


class Variable(Node):
    """ Represents a variable.

    Parameters
    ==========

    symbol : Symbol
    type : Type (optional)
        Type of the variable.
    attrs : iterable of Attribute instances
        Will be stored as a Tuple.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Variable, float32, integer
    >>> x = Symbol('x')
    >>> v = Variable(x, type=float32)
    >>> v.attrs
    ()
    >>> v == Variable('x')
    False
    >>> v == Variable('x', type=float32)
    True
    >>> v
    Variable(x, type=float32)

    One may also construct a ``Variable`` instance with the type deduced from
    assumptions about the symbol using the ``deduced`` classmethod:

    >>> i = Symbol('i', integer=True)
    >>> v = Variable.deduced(i)
    >>> v.type == integer
    True
    >>> v == Variable('i')
    False
    >>> from sympy.codegen.ast import value_const
    >>> value_const in v.attrs
    False
    >>> w = Variable('w', attrs=[value_const])
    >>> w
    Variable(w, attrs=(value_const,))
    >>> value_const in w.attrs
    True
    >>> w.as_Declaration(value=42)
    Declaration(Variable(w, value=42, attrs=(value_const,)))

    """

    __slots__ = ('symbol', 'type', 'value')
    _fields = __slots__ + Node._fields

    defaults = Node.defaults.copy()
    defaults.update({'type': untyped, 'value': none})

    _construct_symbol = staticmethod(sympify)
    _construct_value = staticmethod(sympify)

    @classmethod
    def deduced(cls, symbol, value=None, attrs=Tuple(), cast_check=True):
        """ Alt. constructor with type deduction from ``Type.from_expr``.

        Deduces type primarily from ``symbol``, secondarily from ``value``.

        Parameters
        ==========

        symbol : Symbol
        value : expr
            (optional) value of the variable.
        attrs : iterable of Attribute instances
        cast_check : bool
            Whether to apply ``Type.cast_check`` on ``value``.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.codegen.ast import Variable, complex_
        >>> n = Symbol('n', integer=True)
        >>> str(Variable.deduced(n).type)
        'integer'
        >>> x = Symbol('x', real=True)
        >>> v = Variable.deduced(x)
        >>> v.type
        real
        >>> z = Symbol('z', complex=True)
        >>> Variable.deduced(z).type == complex_
        True

        """
        if isinstance(symbol, Variable):
            return symbol

        try:
            type_ = Type.from_expr(symbol)
        except ValueError:
            type_ = Type.from_expr(value)

        if value is not None and cast_check:
            value = type_.cast_check(value)
        return cls(symbol, type=type_, value=value, attrs=attrs)

    def as_Declaration(self, **kwargs):
        """ Convenience method for creating a Declaration instance.

        Explanation
        ===========

        If the variable of the Declaration need to wrap a modified
        variable keyword arguments may be passed (overriding e.g.
        the ``value`` of the Variable instance).

        Examples
        ========

        >>> from sympy.codegen.ast import Variable, NoneToken
        >>> x = Variable('x')
        >>> decl1 = x.as_Declaration()
        >>> # value is special NoneToken() which must be tested with == operator
        >>> decl1.variable.value is None  # won't work
        False
        >>> decl1.variable.value == None  # not PEP-8 compliant
        True
        >>> decl1.variable.value == NoneToken()  # OK
        True
        >>> decl2 = x.as_Declaration(value=42.0)
        >>> decl2.variable.value == 42.0
        True

        """
        kw = self.kwargs()
        kw.update(kwargs)
        return Declaration(self.func(**kw))

    def _relation(self, rhs, op):
        try:
            rhs = _sympify(rhs)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, rhs))
        return op(self, rhs, evaluate=False)

    __lt__ = lambda self, other: self._relation(other, Lt)
    __le__ = lambda self, other: self._relation(other, Le)
    __ge__ = lambda self, other: self._relation(other, Ge)
    __gt__ = lambda self, other: self._relation(other, Gt)

class Pointer(Variable):
    """ Represents a pointer. See ``Variable``.

    Examples
    ========

    Can create instances of ``Element``:

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Pointer
    >>> i = Symbol('i', integer=True)
    >>> p = Pointer('x')
    >>> p[i+1]
    Element(x, indices=(i + 1,))

    """
    __slots__ = ()

    def __getitem__(self, key):
        try:
            return Element(self.symbol, key)
        except TypeError:
            return Element(self.symbol, (key,))


class Element(Token):
    """ Element in (a possibly N-dimensional) array.

    Examples
    ========

    >>> from sympy.codegen.ast import Element
    >>> elem = Element('x', 'ijk')
    >>> elem.symbol.name == 'x'
    True
    >>> elem.indices
    (i, j, k)
    >>> from sympy import ccode
    >>> ccode(elem)
    'x[i][j][k]'
    >>> ccode(Element('x', 'ijk', strides='lmn', offset='o'))
    'x[i*l + j*m + k*n + o]'

    """
    __slots__ = _fields = ('symbol', 'indices', 'strides', 'offset')
    defaults = {'strides': none, 'offset': none}
    _construct_symbol = staticmethod(sympify)
    _construct_indices = staticmethod(lambda arg: Tuple(*arg))
    _construct_strides = staticmethod(lambda arg: Tuple(*arg))
    _construct_offset = staticmethod(sympify)


class Declaration(Token):
    """ Represents a variable declaration

    Parameters
    ==========

    variable : Variable

    Examples
    ========

    >>> from sympy.codegen.ast import Declaration, NoneToken, untyped
    >>> z = Declaration('z')
    >>> z.variable.type == untyped
    True
    >>> # value is special NoneToken() which must be tested with == operator
    >>> z.variable.value is None  # won't work
    False
    >>> z.variable.value == None  # not PEP-8 compliant
    True
    >>> z.variable.value == NoneToken()  # OK
    True
    """
    __slots__ = _fields = ('variable',)
    _construct_variable = Variable


class While(Token):
    """ Represents a 'for-loop' in the code.

    Expressions are of the form:
        "while condition:
             body..."

    Parameters
    ==========

    condition : expression convertible to Boolean
    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Gt, Abs
    >>> from sympy.codegen import aug_assign, Assignment, While
    >>> x, dx = symbols('x dx')
    >>> expr = 1 - x**2
    >>> whl = While(Gt(Abs(dx), 1e-9), [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx)
    ... ])

    """
    __slots__ = _fields = ('condition', 'body')
    _construct_condition = staticmethod(lambda cond: _sympify(cond))

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class Scope(Token):
    """ Represents a scope in the code.

    Parameters
    ==========

    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    """
    __slots__ = _fields = ('body',)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class Stream(Token):
    """ Represents a stream.

    There are two predefined Stream instances ``stdout`` & ``stderr``.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import pycode, Symbol
    >>> from sympy.codegen.ast import Print, stderr, QuotedString
    >>> print(pycode(Print(['x'], file=stderr)))
    print(x, file=sys.stderr)
    >>> x = Symbol('x')
    >>> print(pycode(Print([QuotedString('x')], file=stderr)))  # print literally "x"
