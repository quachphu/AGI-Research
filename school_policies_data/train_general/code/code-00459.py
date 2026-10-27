        when Heaviside(0) is undefined.

        Examples
        ========

        >>> from sympy import Heaviside, Symbol, sign, nan
        >>> x = Symbol('x', real=True)
        >>> y = Symbol('y')

        >>> Heaviside(x).rewrite(sign)
        sign(x)/2 + 1/2

        >>> Heaviside(x, 0).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (0, True))

        >>> Heaviside(x, nan).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (nan, True))

        >>> Heaviside(x - 2).rewrite(sign)
        sign(x - 2)/2 + 1/2

        >>> Heaviside(x**2 - 2*x + 1).rewrite(sign)
        sign(x**2 - 2*x + 1)/2 + 1/2

        >>> Heaviside(y).rewrite(sign)
        Heaviside(y)

        >>> Heaviside(y**2 - 2*y + 1).rewrite(sign)
        Heaviside(y**2 - 2*y + 1)

        See Also
        ========

        sign

        """
        if arg.is_extended_real:
            pw1 = Piecewise(
                ((sign(arg) + 1)/2, Ne(arg, 0)),
                (Heaviside(0, H0=H0), True))
            pw2 = Piecewise(
                ((sign(arg) + 1)/2, Eq(Heaviside(0, H0=H0), S.Half)),
                (pw1, True))
            return pw2

    def _eval_rewrite_as_SingularityFunction(self, args, H0=S.Half, **kwargs):
        """
        Returns the Heaviside expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == Heaviside(0):
            return SingularityFunction(0, 0, 0)
        free = self.free_symbols
        if len(free) == 1:
            x = (free.pop())
            return SingularityFunction(x, solve(args, x)[0], 0)
            # TODO
            # ((x - 5)**3*Heaviside(x - 5)).rewrite(SingularityFunction) should output
            # SingularityFunction(x, 5, 0) instead of (x - 5)**3*SingularityFunction(x, 5, 0)
        else:
            # I don't know how to handle the case for Heaviside expressions
            # having arguments with more than one variable.
            raise TypeError(filldedent('''
                rewrite(SingularityFunction) does not
                support arguments with more that one variable.'''))
<|endoftext|>"""Hypergeometric and Meijer G-functions"""
from collections import Counter

from sympy.core import S, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, Derivative, ArgumentIndexError

from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.parameters import global_parameters
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy

from sympy.external.gmpy import lcm
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
        sinh, cosh, asinh, acosh, atanh, acoth)
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy import ordered


class TupleArg(Tuple):

    # This method is only needed because hyper._eval_as_leading_term falls back
    # (via super()) on using Function._eval_as_leading_term, which in turn
    # calls as_leading_term on the args of the hyper. Ideally hyper should just
    # have an _eval_as_leading_term method that handles all cases and this
    # method should be removed because leading terms of tuples don't make
    # sense.
    def as_leading_term(self, *x, logx=None, cdir=0):
        return TupleArg(*[f.as_leading_term(*x, logx=logx, cdir=cdir) for f in self.args])

    def limit(self, x, xlim, dir='+'):
        """ Compute limit x->xlim.
        """
        from sympy.series.limits import limit
        return TupleArg(*[limit(f, x, xlim, dir) for f in self.args])


# TODO should __new__ accept **options?
# TODO should constructors should check if parameters are sensible?


def _prep_tuple(v):
    """
    Turn an iterable argument *v* into a tuple and unpolarify, since both
    hypergeometric and meijer g-functions are unbranched in their parameters.

    Examples
    ========

    >>> from sympy.functions.special.hyper import _prep_tuple
    >>> _prep_tuple([1, 2, 3])
    (1, 2, 3)
    >>> _prep_tuple((4, 5))
    (4, 5)
    >>> _prep_tuple((7, 8, 9))
    (7, 8, 9)

    """
    return TupleArg(*[unpolarify(x) for x in v])


class TupleParametersBase(DefinedFunction):
    """ Base class that takes care of differentiation, when some of
        the arguments are actually tuples. """
    # This is not deduced automatically since there are Tuples as arguments.
    is_commutative = True

    def _eval_derivative(self, s):
        try:
            res = 0
            if self.args[0].has(s) or self.args[1].has(s):
                for i, p in enumerate(self._diffargs):
                    m = self._diffargs[i].diff(s)
                    if m != 0:
                        res += self.fdiff((1, i))*m
            return res + self.fdiff(3)*self.args[2].diff(s)
        except (ArgumentIndexError, NotImplementedError):
            return Derivative(self, s)


class hyper(TupleParametersBase):
    r"""
    The generalized hypergeometric function is defined by a series where
    the ratios of successive terms are a rational function of the summation
    index. When convergent, it is continued analytically to the largest
    possible domain.

    Explanation
    ===========

    The hypergeometric function depends on two vectors of parameters, called
    the numerator parameters $a_p$, and the denominator parameters
    $b_q$. It also has an argument $z$. The series definition is

    .. math ::
        {}_pF_q\left(\begin{matrix} a_1, \cdots, a_p \\ b_1, \cdots, b_q \end{matrix}
                     \middle| z \right)
        = \sum_{n=0}^\infty \frac{(a_1)_n \cdots (a_p)_n}{(b_1)_n \cdots (b_q)_n}
                            \frac{z^n}{n!},

    where $(a)_n = (a)(a+1)\cdots(a+n-1)$ denotes the rising factorial.

    If one of the $b_q$ is a non-positive integer then the series is
    undefined unless one of the $a_p$ is a larger (i.e., smaller in
    magnitude) non-positive integer. If none of the $b_q$ is a
    non-positive integer and one of the $a_p$ is a non-positive
    integer, then the series reduces to a polynomial. To simplify the
    following discussion, we assume that none of the $a_p$ or
    $b_q$ is a non-positive integer. For more details, see the
    references.

    The series converges for all $z$ if $p \le q$, and thus
    defines an entire single-valued function in this case. If $p =
    q+1$ the series converges for $|z| < 1$, and can be continued
    analytically into a half-plane. If $p > q+1$ the series is
    divergent for all $z$.

    Please note the hypergeometric function constructor currently does *not*
    check if the parameters actually yield a well-defined function.

    Examples
    ========

    The parameters $a_p$ and $b_q$ can be passed as arbitrary
    iterables, for example:

    >>> from sympy import hyper
    >>> from sympy.abc import x, n, a
    >>> h = hyper((1, 2, 3), [3, 4], x); h
    hyper((1, 2), (4,), x)
    >>> hyper((3, 1, 2), [3, 4], x, evaluate=False)  # don't remove duplicates
    hyper((1, 2, 3), (3, 4), x)

    There is also pretty printing (it looks better using Unicode):

    >>> from sympy import pprint
    >>> pprint(h, use_unicode=False)
      _
     |_  /1, 2 |  \
     |   |     | x|
    2  1 \  4  |  /

    The parameters must always be iterables, even if they are vectors of
    length one or zero:

    >>> hyper((1, ), [], x)
    hyper((1,), (), x)

    But of course they may be variables (but if they depend on $x$ then you
    should not expect much implemented functionality):

    >>> hyper((n, a), (n**2,), x)
    hyper((a, n), (n**2,), x)

    The hypergeometric function generalizes many named special functions.
    The function ``hyperexpand()`` tries to express a hypergeometric function
    using named special functions. For example:

    >>> from sympy import hyperexpand
    >>> hyperexpand(hyper([], [], x))
    exp(x)

    You can also use ``expand_func()``:

    >>> from sympy import expand_func
    >>> expand_func(x*hyper([1, 1], [2], -x))
    log(x + 1)

    More examples:

    >>> from sympy import S
    >>> hyperexpand(hyper([], [S(1)/2], -x**2/4))
    cos(x)
    >>> hyperexpand(x*hyper([S(1)/2, S(1)/2], [S(3)/2], x**2))
    asin(x)

    We can also sometimes ``hyperexpand()`` parametric functions:

    >>> from sympy.abc import a
    >>> hyperexpand(hyper([-a], [], x))
    (1 - x)**a

    See Also
    ========

    sympy.simplify.hyperexpand
    gamma
    meijerg

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Generalized_hypergeometric_function

    """


    def __new__(cls, ap, bq, z, **kwargs):
        # TODO should we check convergence conditions?
        if kwargs.pop('evaluate', global_parameters.evaluate):
            ca = Counter(Tuple(*ap))
            cb = Counter(Tuple(*bq))
            common = ca & cb
            arg = ap, bq = [], []
            for i, c in enumerate((ca, cb)):
                c -= common
                for k in ordered(c):
                    arg[i].extend([k]*c[k])
        else:
            ap = list(ordered(ap))
            bq = list(ordered(bq))
        return super().__new__(cls, _prep_tuple(ap), _prep_tuple(bq), z, **kwargs)

    @classmethod
    def eval(cls, ap, bq, z):
        if len(ap) <= len(bq) or (len(ap) == len(bq) + 1 and (Abs(z) <= 1) == True):
            nz = unpolarify(z)
            if z != nz:
                return hyper(ap, bq, nz)

    def fdiff(self, argindex=3):
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        fac = Mul(*self.ap)/Mul(*self.bq)
        return fac*hyper(nap, nbq, self.argument)

    def _eval_expand_func(self, **hints):
        from sympy.functions.special.gamma_functions import gamma
        from sympy.simplify.hyperexpand import hyperexpand
        if len(self.ap) == 2 and len(self.bq) == 1 and self.argument == 1:
            a, b = self.ap
            c = self.bq[0]
            return gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
        return hyperexpand(self)

    def _eval_rewrite_as_Sum(self, ap, bq, z, **kwargs):
        from sympy.concrete.summations import Sum
        n = Dummy("n", integer=True)
        rfap = [RisingFactorial(a, n) for a in ap]
        rfbq = [RisingFactorial(b, n) for b in bq]
        coeff = Mul(*rfap) / Mul(*rfbq)
        return Piecewise((Sum(coeff * z**n / factorial(n), (n, 0, oo)),
                         self.convergence_statement), (self, True))

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[2]
        x0 = arg.subs(x, 0)
        if x0 is S.NaN:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        if x0 is S.Zero:
            return S.One
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):

        from sympy.series.order import Order

        arg = self.args[2]
        x0 = arg.limit(x, 0)
        ap = self.args[0]
        bq = self.args[1]

        if not (arg == x and x0 == 0):
            # It would be better to do something with arg.nseries here, rather
            # than falling back on Function._eval_nseries. The code below
            # though is not sufficient if arg is something like x/(x+1).
            from sympy.simplify.hyperexpand import hyperexpand
            return hyperexpand(super()._eval_nseries(x, n, logx))

        terms = []

        for i in range(n):
            num = Mul(*[RisingFactorial(a, i) for a in ap])
            den = Mul(*[RisingFactorial(b, i) for b in bq])
            terms.append(((num/den) * (arg**i)) / factorial(i))

        return (Add(*terms) + Order(x**n,x))

    @property
    def argument(self):
        """ Argument of the hypergeometric function. """
        return self.args[2]

    @property
    def ap(self):
        """ Numerator parameters of the hypergeometric function. """
        return Tuple(*self.args[0])

    @property
    def bq(self):
        """ Denominator parameters of the hypergeometric function. """
        return Tuple(*self.args[1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def eta(self):
        """ A quantity related to the convergence of the series. """
        return sum(self.ap) - sum(self.bq)

    @property
    def radius_of_convergence(self):
        """
        Compute the radius of convergence of the defining series.

        Explanation
        ===========

        Note that even if this is not ``oo``, the function may still be
        evaluated outside of the radius of convergence by analytic
        continuation. But if this is zero, then the function is not actually
        defined anywhere else.

        Examples
        ========

        >>> from sympy import hyper
        >>> from sympy.abc import z
        >>> hyper((1, 2), [3], z).radius_of_convergence
        1
        >>> hyper((1, 2, 3), [4], z).radius_of_convergence
        0
        >>> hyper((1, 2), (3, 4), z).radius_of_convergence
        oo

        """
        if any(a.is_integer and (a <= 0) == True for a in self.ap + self.bq):
            aints = [a for a in self.ap if a.is_Integer and (a <= 0) == True]
            bints = [a for a in self.bq if a.is_Integer and (a <= 0) == True]
            if len(aints) < len(bints):
                return S.Zero
            popped = False
            for b in bints:
                cancelled = False
                while aints:
                    a = aints.pop()
                    if a >= b:
                        cancelled = True
                        break
                    popped = True
                if not cancelled:
                    return S.Zero
            if aints or popped:
                # There are still non-positive numerator parameters.
                # This is a polynomial.
                return oo
        if len(self.ap) == len(self.bq) + 1:
            return S.One
        elif len(self.ap) <= len(self.bq):
            return oo
        else:
            return S.Zero

    @property
    def convergence_statement(self):
        """ Return a condition on z under which the series converges. """
        R = self.radius_of_convergence
        if R == 0:
            return False
        if R == oo:
            return True
        # The special functions and their approximations, page 44
        e = self.eta
        z = self.argument
        c1 = And(re(e) < 0, abs(z) <= 1)
        c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
        c3 = And(re(e) >= 1, abs(z) < 1)
        return Or(c1, c2, c3)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)


class meijerg(TupleParametersBase):
    r"""
    The Meijer G-function is defined by a Mellin-Barnes type integral that
    resembles an inverse Mellin transform. It generalizes the hypergeometric
    functions.

    Explanation
    ===========

    The Meijer G-function depends on four sets of parameters. There are
    "*numerator parameters*"
    $a_1, \ldots, a_n$ and $a_{n+1}, \ldots, a_p$, and there are
    "*denominator parameters*"
    $b_1, \ldots, b_m$ and $b_{m+1}, \ldots, b_q$.
    Confusingly, it is traditionally denoted as follows (note the position
    of $m$, $n$, $p$, $q$, and how they relate to the lengths of the four
    parameter vectors):

    .. math ::
        G_{p,q}^{m,n} \left(\begin{matrix}a_1, \cdots, a_n & a_{n+1}, \cdots, a_p \\
                                        b_1, \cdots, b_m & b_{m+1}, \cdots, b_q
                          \end{matrix} \middle| z \right).

    However, in SymPy the four parameter vectors are always available
    separately (see examples), so that there is no need to keep track of the
    decorating sub- and super-scripts on the G symbol.

    The G function is defined as the following integral:

    .. math ::
         \frac{1}{2 \pi i} \int_L \frac{\prod_{j=1}^m \Gamma(b_j - s)
         \prod_{j=1}^n \Gamma(1 - a_j + s)}{\prod_{j=m+1}^q \Gamma(1- b_j +s)
         \prod_{j=n+1}^p \Gamma(a_j - s)} z^s \mathrm{d}s,

    where $\Gamma(z)$ is the gamma function. There are three possible
    contours which we will not describe in detail here (see the references).
    If the integral converges along more than one of them, the definitions
    agree. The contours all separate the poles of $\Gamma(1-a_j+s)$
    from the poles of $\Gamma(b_k-s)$, so in particular the G function
    is undefined if $a_j - b_k \in \mathbb{Z}_{>0}$ for some
    $j \le n$ and $k \le m$.

    The conditions under which one of the contours yields a convergent integral
    are complicated and we do not state them here, see the references.

    Please note currently the Meijer G-function constructor does *not* check any
    convergence conditions.

    Examples
    ========

    You can pass the parameters either as four separate vectors:

    >>> from sympy import meijerg, Tuple, pprint
    >>> from sympy.abc import x, a
    >>> pprint(meijerg((1, 2), (a, 4), (5,), [], x), use_unicode=False)
     __1, 2 /1, 2  4, a |  \
    /__     |           | x|
    \_|4, 1 \ 5         |  /

    Or as two nested vectors:

    >>> pprint(meijerg([(1, 2), (3, 4)], ([5], Tuple()), x), use_unicode=False)
     __1, 2 /1, 2  3, 4 |  \
    /__     |           | x|
    \_|4, 1 \ 5         |  /

    As with the hypergeometric function, the parameters may be passed as
    arbitrary iterables. Vectors of length zero and one also have to be
    passed as iterables. The parameters need not be constants, but if they
    depend on the argument then not much implemented functionality should be
    expected.

    All the subvectors of parameters are available:

    >>> from sympy import pprint
    >>> g = meijerg([1], [2], [3], [4], x)
    >>> pprint(g, use_unicode=False)
     __1, 1 /1  2 |  \
    /__     |     | x|
    \_|2, 2 \3  4 |  /
    >>> g.an
    (1,)
    >>> g.ap
    (1, 2)
    >>> g.aother
    (2,)
    >>> g.bm
    (3,)
    >>> g.bq
    (3, 4)
    >>> g.bother
    (4,)

    The Meijer G-function generalizes the hypergeometric functions.
    In some cases it can be expressed in terms of hypergeometric functions,
    using Slater's theorem. For example:

    >>> from sympy import hyperexpand
    >>> from sympy.abc import a, b, c
    >>> hyperexpand(meijerg([a], [], [c], [b], x), allow_hyper=True)
    x**c*gamma(-a + c + 1)*hyper((-a + c + 1,),
                                 (-b + c + 1,), -x)/gamma(-b + c + 1)

    Thus the Meijer G-function also subsumes many named functions as special
    cases. You can use ``expand_func()`` or ``hyperexpand()`` to (try to)
    rewrite a Meijer G-function in terms of named special functions. For
    example:

    >>> from sympy import expand_func, S
    >>> expand_func(meijerg([[],[]], [[0],[]], -x))
    exp(x)
    >>> hyperexpand(meijerg([[],[]], [[S(1)/2],[0]], (x/2)**2))
    sin(x)/sqrt(pi)

    See Also
    ========

    hyper
    sympy.simplify.hyperexpand

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Meijer_G-function

    """


    def __new__(cls, *args, **kwargs):
        if len(args) == 5:
            args = [(args[0], args[1]), (args[2], args[3]), args[4]]
        if len(args) != 3:
            raise TypeError("args must be either as, as', bs, bs', z or "
                            "as, bs, z")

        def tr(p):
            if len(p) != 2:
                raise TypeError("wrong argument")
            p = [list(ordered(i)) for i in p]
            return TupleArg(_prep_tuple(p[0]), _prep_tuple(p[1]))

        arg0, arg1 = tr(args[0]), tr(args[1])
        if Tuple(arg0, arg1).has(oo, zoo, -oo):
            raise ValueError("G-function parameters must be finite")
        if any((a - b).is_Integer and a - b > 0
               for a in arg0[0] for b in arg1[0]):
            raise ValueError("no parameter a1, ..., an may differ from "
                         "any b1, ..., bm by a positive integer")

        # TODO should we check convergence conditions?
        return super().__new__(cls, arg0, arg1, args[2], **kwargs)

    def fdiff(self, argindex=3):
        if argindex != 3:
            return self._diff_wrt_parameter(argindex[1])
        if len(self.an) >= 1:
            a = list(self.an)
            a[0] -= 1
            G = meijerg(a, self.aother, self.bm, self.bother, self.argument)
            return 1/self.argument * ((self.an[0] - 1)*self + G)
        elif len(self.bm) >= 1:
            b = list(self.bm)
            b[0] += 1
            G = meijerg(self.an, self.aother, b, self.bother, self.argument)
            return 1/self.argument * (self.bm[0]*self - G)
        else:
            return S.Zero

    def _diff_wrt_parameter(self, idx):
        # Differentiation wrt a parameter can only be done in very special
        # cases. In particular, if we want to differentiate with respect to
        # `a`, all other gamma factors have to reduce to rational functions.
        #
        # Let MT denote mellin transform. Suppose T(-s) is the gamma factor
        # appearing in the definition of G. Then
        #
        #   MT(log(z)G(z)) = d/ds T(s) = d/da T(s) + ...
        #
        # Thus d/da G(z) = log(z)G(z) - ...
        # The ... can be evaluated as a G function under the above conditions,
        # the formula being most easily derived by using
        #
        # d  Gamma(s + n)    Gamma(s + n) / 1    1                1     \
        # -- ------------ =  ------------ | - + ----  + ... + --------- |
        # ds Gamma(s)        Gamma(s)     \ s   s + 1         s + n - 1 /
        #
        # which follows from the difference equation of the digamma function.
        # (There is a similar equation for -n instead of +n).

        # We first figure out how to pair the parameters.
        an = list(self.an)
        ap = list(self.aother)
        bm = list(self.bm)
        bq = list(self.bother)
        if idx < len(an):
            an.pop(idx)
        else:
            idx -= len(an)
            if idx < len(ap):
                ap.pop(idx)
            else:
                idx -= len(ap)
                if idx < len(bm):
                    bm.pop(idx)
                else:
                    bq.pop(idx - len(bm))
        pairs1 = []
        pairs2 = []
        for l1, l2, pairs in [(an, bq, pairs1), (ap, bm, pairs2)]:
            while l1:
                x = l1.pop()
                found = None
                for i, y in enumerate(l2):
                    if not Mod((x - y).simplify(), 1):
                        found = i
                        break
                if found is None:
                    raise NotImplementedError('Derivative not expressible '
                                              'as G-function?')
                y = l2[i]
                l2.pop(i)
                pairs.append((x, y))

        # Now build the result.
        res = log(self.argument)*self

        for a, b in pairs1:
            sign = 1
            n = a - b
            base = b
            if n < 0:
                sign = -1
                n = b - a
                base = a
            for k in range(n):
                res -= sign*meijerg(self.an + (base + k + 1,), self.aother,
                                    self.bm, self.bother + (base + k + 0,),
                                    self.argument)

        for a, b in pairs2:
            sign = 1
            n = b - a
            base = a
            if n < 0:
                sign = -1
                n = a - b
                base = b
            for k in range(n):
                res -= sign*meijerg(self.an, self.aother + (base + k + 1,),
                                    self.bm + (base + k + 0,), self.bother,
                                    self.argument)

        return res

    def get_period(self):
        """
        Return a number $P$ such that $G(x*exp(I*P)) == G(x)$.

        Examples
        ========

        >>> from sympy import meijerg, pi, S
        >>> from sympy.abc import z

        >>> meijerg([1], [], [], [], z).get_period()
        2*pi
        >>> meijerg([pi], [], [], [], z).get_period()
        oo
        >>> meijerg([1, 2], [], [], [], z).get_period()
        oo
        >>> meijerg([1,1], [2], [1, S(1)/2, S(1)/3], [1], z).get_period()
        12*pi

        """
        # This follows from slater's theorem.
        def compute(l):
            # first check that no two differ by an integer
            for i, b in enumerate(l):
                if not b.is_Rational:
                    return oo
                for j in range(i + 1, len(l)):
                    if not Mod((b - l[j]).simplify(), 1):
                        return oo
            return lcm(*(x.q for x in l))
        beta = compute(self.bm)
        alpha = compute(self.an)
        p, q = len(self.ap), len(self.bq)
        if p == q:
            if oo in (alpha, beta):
                return oo
            return 2*pi*lcm(alpha, beta)
        elif p < q:
            return 2*pi*beta
        else:
            return 2*pi*alpha

    def _eval_expand_func(self, **hints):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)

    def _eval_evalf(self, prec):
        # The default code is insufficient for polar arguments.
        # mpmath provides an optional argument "r", which evaluates
        # G(z**(1/r)). I am not sure what its intended use is, but we hijack it
        # here in the following way: to evaluate at a number z of |argument|
        # less than (say) n*pi, we put r=1/n, compute z' = root(z, n)
        # (carefully so as not to loose the branch information), and evaluate
        # G(z'**(1/r)) = G(z'**n) = G(z).
        import mpmath
        znum = self.argument._eval_evalf(prec)
        if znum.has(exp_polar):
            znum, branch = znum.as_coeff_mul(exp_polar)
            if len(branch) != 1:
                return
            branch = branch[0].args[0]/I
        else:
            branch = S.Zero
        n = ceiling(abs(branch/pi)) + 1
        znum = znum**(S.One/n)*exp(I*branch / n)

        # Convert all args to mpf or mpc
        try:
            [z, r, ap, bq] = [arg._to_mpmath(prec)
                    for arg in [znum, 1/n, self.args[0], self.args[1]]]
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = mpmath.meijerg(ap, bq, z, r)

        return Expr._from_mpmath(v, prec)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self).as_leading_term(x, logx=logx, cdir=cdir)

    def integrand(self, s):
        """ Get the defining integrand D(s). """
        from sympy.functions.special.gamma_functions import gamma
        return self.argument**s \
            * Mul(*(gamma(b - s) for b in self.bm)) \
            * Mul(*(gamma(1 - a + s) for a in self.an)) \
            / Mul(*(gamma(1 - b + s) for b in self.bother)) \
            / Mul(*(gamma(a - s) for a in self.aother))

    @property
    def argument(self):
        """ Argument of the Meijer G-function. """
        return self.args[2]

    @property
    def an(self):
        """ First set of numerator parameters. """
        return Tuple(*self.args[0][0])

    @property
    def ap(self):
        """ Combined numerator parameters. """
        return Tuple(*(self.args[0][0] + self.args[0][1]))

    @property
    def aother(self):
        """ Second set of numerator parameters. """
        return Tuple(*self.args[0][1])

    @property
    def bm(self):
        """ First set of denominator parameters. """
        return Tuple(*self.args[1][0])

    @property
    def bq(self):
        """ Combined denominator parameters. """
        return Tuple(*(self.args[1][0] + self.args[1][1]))

    @property
    def bother(self):
        """ Second set of denominator parameters. """
        return Tuple(*self.args[1][1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def nu(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        return sum(self.bq) - sum(self.ap)

    @property
    def delta(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        return len(self.bm) + len(self.an) - S(len(self.ap) + len(self.bq))/2

    @property
    def is_number(self):
        """ Returns true if expression has numeric data only. """
        return not self.free_symbols


class HyperRep(DefinedFunction):
    """
    A base class for "hyper representation functions".

    This is used exclusively in ``hyperexpand()``, but fits more logically here.

    pFq is branched at 1 if p == q+1. For use with slater-expansion, we want
    define an "analytic continuation" to all polar numbers, which is
    continuous on circles and on the ray t*exp_polar(I*pi). Moreover, we want
    a "nice" expression for the various cases.

    This base class contains the core logic, concrete derived classes only
    supply the actual functions.

    """


    @classmethod
    def eval(cls, *args):
        newargs = tuple(map(unpolarify, args[:-1])) + args[-1:]
        if args != newargs:
            return cls(*newargs)

    @classmethod
    def _expr_small(cls, x):
        """ An expression for F(x) which holds for |x| < 1. """
        raise NotImplementedError

    @classmethod
    def _expr_small_minus(cls, x):
        """ An expression for F(-x) which holds for |x| < 1. """
        raise NotImplementedError

    @classmethod
    def _expr_big(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n)*x), |x| > 1. """
        raise NotImplementedError

    @classmethod
    def _expr_big_minus(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n + pi*I)*x), |x| > 1. """
        raise NotImplementedError

    def _eval_rewrite_as_nonrep(self, *args, **kwargs):
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        minus = False
        newargs = self.args[:-1] + (x,)
        if not n.is_Integer:
            minus = True
            n -= S.Half
        newerargs = newargs + (n,)
        if minus:
            small = self._expr_small_minus(*newargs)
            big = self._expr_big_minus(*newerargs)
        else:
            small = self._expr_small(*newargs)
            big = self._expr_big(*newerargs)

        if big == small:
            return small
        return Piecewise((big, abs(x) > 1), (small, True))

    def _eval_rewrite_as_nonrepsmall(self, *args, **kwargs):
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        args = self.args[:-1] + (x,)
        if not n.is_Integer:
            return self._expr_small_minus(*args)
        return self._expr_small(*args)


class HyperRep_power1(HyperRep):
    """ Return a representative for hyper([-a], [], z) == (1 - z)**a. """

    @classmethod
    def _expr_small(cls, a, x):
        return (1 - x)**a

    @classmethod
    def _expr_small_minus(cls, a, x):
        return (1 + x)**a

    @classmethod
    def _expr_big(cls, a, x, n):
        if a.is_integer:
            return cls._expr_small(a, x)
        return (x - 1)**a*exp((2*n - 1)*pi*I*a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        if a.is_integer:
            return cls._expr_small_minus(a, x)
        return (1 + x)**a*exp(2*n*pi*I*a)


class HyperRep_power2(HyperRep):
    """ Return a representative for hyper([a, a - 1/2], [2*a], z). """

    @classmethod
    def _expr_small(cls, a, x):
        return 2**(2*a - 1)*(1 + sqrt(1 - x))**(1 - 2*a)

    @classmethod
    def _expr_small_minus(cls, a, x):
        return 2**(2*a - 1)*(1 + sqrt(1 + x))**(1 - 2*a)

    @classmethod
    def _expr_big(cls, a, x, n):
        sgn = -1
        if n.is_odd:
            sgn = 1
            n -= 1
        return 2**(2*a - 1)*(1 + sgn*I*sqrt(x - 1))**(1 - 2*a) \
            *exp(-2*n*pi*I*a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        sgn = 1
        if n.is_odd:
            sgn = -1
        return sgn*2**(2*a - 1)*(sqrt(1 + x) + sgn)**(1 - 2*a)*exp(-2*pi*I*a*n)


class HyperRep_log1(HyperRep):
    """ Represent -z*hyper([1, 1], [2], z) == log(1 - z). """
    @classmethod
    def _expr_small(cls, x):
        return log(1 - x)

    @classmethod
    def _expr_small_minus(cls, x):
        return log(1 + x)

    @classmethod
    def _expr_big(cls, x, n):
        return log(x - 1) + (2*n - 1)*pi*I

    @classmethod
    def _expr_big_minus(cls, x, n):
        return log(1 + x) + 2*n*pi*I


class HyperRep_atanh(HyperRep):
    """ Represent hyper([1/2, 1], [3/2], z) == atanh(sqrt(z))/sqrt(z). """
    @classmethod
    def _expr_small(cls, x):
        return atanh(sqrt(x))/sqrt(x)

    def _expr_small_minus(cls, x):
        return atan(sqrt(x))/sqrt(x)

    def _expr_big(cls, x, n):
        if n.is_even:
            return (acoth(sqrt(x)) + I*pi/2)/sqrt(x)
        else:
            return (acoth(sqrt(x)) - I*pi/2)/sqrt(x)

    def _expr_big_minus(cls, x, n):
        if n.is_even:
            return atan(sqrt(x))/sqrt(x)
        else:
            return (atan(sqrt(x)) - pi)/sqrt(x)


class HyperRep_asin1(HyperRep):
    """ Represent hyper([1/2, 1/2], [3/2], z) == asin(sqrt(z))/sqrt(z). """
    @classmethod
    def _expr_small(cls, z):
        return asin(sqrt(z))/sqrt(z)

    @classmethod
    def _expr_small_minus(cls, z):
        return asinh(sqrt(z))/sqrt(z)

    @classmethod
    def _expr_big(cls, z, n):
        return S.NegativeOne**n*((S.Half - n)*pi/sqrt(z) + I*acosh(sqrt(z))/sqrt(z))

    @classmethod
    def _expr_big_minus(cls, z, n):
        return S.NegativeOne**n*(asinh(sqrt(z))/sqrt(z) + n*pi*I/sqrt(z))


class HyperRep_asin2(HyperRep):
    """ Represent hyper([1, 1], [3/2], z) == asin(sqrt(z))/sqrt(z)/sqrt(1-z). """
    # TODO this can be nicer
    @classmethod
    def _expr_small(cls, z):
        return HyperRep_asin1._expr_small(z) \
            /HyperRep_power1._expr_small(S.Half, z)

    @classmethod
    def _expr_small_minus(cls, z):
        return HyperRep_asin1._expr_small_minus(z) \
            /HyperRep_power1._expr_small_minus(S.Half, z)

    @classmethod
    def _expr_big(cls, z, n):
        return HyperRep_asin1._expr_big(z, n) \
            /HyperRep_power1._expr_big(S.Half, z, n)

    @classmethod
    def _expr_big_minus(cls, z, n):
        return HyperRep_asin1._expr_big_minus(z, n) \
            /HyperRep_power1._expr_big_minus(S.Half, z, n)


class HyperRep_sqrts1(HyperRep):
    """ Return a representative for hyper([-a, 1/2 - a], [1/2], z). """

    @classmethod
    def _expr_small(cls, a, z):
        return ((1 - sqrt(z))**(2*a) + (1 + sqrt(z))**(2*a))/2

    @classmethod
    def _expr_small_minus(cls, a, z):
        return (1 + z)**a*cos(2*a*atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        if n.is_even:
            return ((sqrt(z) + 1)**(2*a)*exp(2*pi*I*n*a) +
                    (sqrt(z) - 1)**(2*a)*exp(2*pi*I*(n - 1)*a))/2
        else:
            n -= 1
            return ((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) +
                    (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))/2

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        if n.is_even:
            return (1 + z)**a*exp(2*pi*I*n*a)*cos(2*a*atan(sqrt(z)))
        else:
            return (1 + z)**a*exp(2*pi*I*n*a)*cos(2*a*atan(sqrt(z)) - 2*pi*a)


class HyperRep_sqrts2(HyperRep):
    """ Return a representative for
          sqrt(z)/2*[(1-sqrt(z))**2a - (1 + sqrt(z))**2a]
          == -2*z/(2*a+1) d/dz hyper([-a - 1/2, -a], [1/2], z)"""

    @classmethod
    def _expr_small(cls, a, z):
        return sqrt(z)*((1 - sqrt(z))**(2*a) - (1 + sqrt(z))**(2*a))/2

    @classmethod
    def _expr_small_minus(cls, a, z):
        return sqrt(z)*(1 + z)**a*sin(2*a*atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        if n.is_even:
            return sqrt(z)/2*((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n - 1)) -
                              (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))
        else:
            n -= 1
            return sqrt(z)/2*((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) -
                              (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))

    def _expr_big_minus(cls, a, z, n):
        if n.is_even:
            return (1 + z)**a*exp(2*pi*I*n*a)*sqrt(z)*sin(2*a*atan(sqrt(z)))
        else:
            return (1 + z)**a*exp(2*pi*I*n*a)*sqrt(z) \
                *sin(2*a*atan(sqrt(z)) - 2*pi*a)


class HyperRep_log2(HyperRep):
    """ Represent log(1/2 + sqrt(1 - z)/2) == -z/4*hyper([3/2, 1, 1], [2, 2], z) """

    @classmethod
    def _expr_small(cls, z):
        return log(S.Half + sqrt(1 - z)/2)

    @classmethod
    def _expr_small_minus(cls, z):
        return log(S.Half + sqrt(1 + z)/2)

    @classmethod
    def _expr_big(cls, z, n):
        if n.is_even:
            return (n - S.Half)*pi*I + log(sqrt(z)/2) + I*asin(1/sqrt(z))
        else:
            return (n - S.Half)*pi*I + log(sqrt(z)/2) - I*asin(1/sqrt(z))

    def _expr_big_minus(cls, z, n):
        if n.is_even:
            return pi*I*n + log(S.Half + sqrt(1 + z)/2)
        else:
            return pi*I*n + log(sqrt(1 + z)/2 - S.Half)


class HyperRep_cosasin(HyperRep):
    """ Represent hyper([a, -a], [1/2], z) == cos(2*a*asin(sqrt(z))). """
    # Note there are many alternative expressions, e.g. as powers of a sum of
    # square roots.

    @classmethod
    def _expr_small(cls, a, z):
        return cos(2*a*asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        return cosh(2*a*asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        return cosh(2*a*acosh(sqrt(z)) + a*pi*I*(2*n - 1))

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        return cosh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n)


class HyperRep_sinasin(HyperRep):
    """ Represent 2*a*z*hyper([1 - a, 1 + a], [3/2], z)
        == sqrt(z)/sqrt(1-z)*sin(2*a*asin(sqrt(z))) """

    @classmethod
    def _expr_small(cls, a, z):
        return sqrt(z)/sqrt(1 - z)*sin(2*a*asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        return -sqrt(z)/sqrt(1 + z)*sinh(2*a*asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        return -1/sqrt(1 - 1/z)*sinh(2*a*acosh(sqrt(z)) + a*pi*I*(2*n - 1))

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        return -1/sqrt(1 + 1/z)*sinh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n)

class appellf1(DefinedFunction):
    r"""
    This is the Appell hypergeometric function of two variables as:

    .. math ::
        F_1(a,b_1,b_2,c,x,y) = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
        \frac{(a)_{m+n} (b_1)_m (b_2)_n}{(c)_{m+n}}
        \frac{x^m y^n}{m! n!}.

    Examples
    ========

    >>> from sympy import appellf1, symbols
    >>> x, y, a, b1, b2, c = symbols('x y a b1 b2 c')
    >>> appellf1(2., 1., 6., 4., 5., 6.)
    0.0063339426292673
    >>> appellf1(12., 12., 6., 4., 0.5, 0.12)
    172870711.659936
    >>> appellf1(40, 2, 6, 4, 15, 60)
    appellf1(40, 2, 6, 4, 15, 60)
    >>> appellf1(20., 12., 10., 3., 0.5, 0.12)
    15605338197184.4
    >>> appellf1(40, 2, 6, 4, x, y)
    appellf1(40, 2, 6, 4, x, y)
    >>> appellf1(a, b1, b2, c, x, y)
    appellf1(a, b1, b2, c, x, y)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Appell_series
    .. [2] https://functions.wolfram.com/HypergeometricFunctions/AppellF1/

    """

    @classmethod
    def eval(cls, a, b1, b2, c, x, y):
        if default_sort_key(b1) > default_sort_key(b2):
            b1, b2 = b2, b1
            x, y = y, x
            return cls(a, b1, b2, c, x, y)
        elif b1 == b2 and default_sort_key(x) > default_sort_key(y):
            x, y = y, x
            return cls(a, b1, b2, c, x, y)
        if x == 0 and y == 0:
            return S.One

    def fdiff(self, argindex=5):
        a, b1, b2, c, x, y = self.args
        if argindex == 5:
            return (a*b1/c)*appellf1(a + 1, b1 + 1, b2, c + 1, x, y)
        elif argindex == 6:
            return (a*b2/c)*appellf1(a + 1, b1, b2 + 1, c + 1, x, y)
        elif argindex in (1, 2, 3, 4):
            return Derivative(self, self.args[argindex-1])
        else:
            raise ArgumentIndexError(self, argindex)
<|endoftext|>""" This module contains various functions that are special cases
    of incomplete gamma functions. It should probably be renamed. """

from sympy.core import EulerGamma # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import DefinedFunction, ArgumentIndexError, expand_mul
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import I, pi, Rational, Integer
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg

# TODO series expansions
# TODO see the "Note:" in Ei

# Helper function
def real_to_real_as_real_imag(self, deep=True, **hints):
    if self.args[0].is_extended_real:
        if deep:
            hints['complex'] = False
            return (self.expand(deep, **hints), S.Zero)
        else:
            return (self, S.Zero)
    if deep:
        x, y = self.args[0].expand(deep, **hints).as_real_imag()
    else:
        x, y = self.args[0].as_real_imag()
    re = (self.func(x + I*y) + self.func(x - I*y))/2
    im = (self.func(x + I*y) - self.func(x - I*y))/(2*I)
    return (re, im)


###############################################################################
################################ ERROR FUNCTION ###############################
###############################################################################


class erf(DefinedFunction):
    r"""
    The Gauss error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \mathrm{d}t.

    Examples
    ========

    >>> from sympy import I, oo, erf
    >>> from sympy.abc import z

    Several special values are known:

    >>> erf(0)
    0
    >>> erf(oo)
    1
    >>> erf(-oo)
    -1
    >>> erf(I*oo)
    oo*I
    >>> erf(-I*oo)
    -oo*I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erf(-z)
    -erf(z)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf(z))
    erf(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erf(z), z)
    2*exp(-z**2)/sqrt(pi)

    We can numerically evaluate the error function to arbitrary precision
    on the whole complex plane:

    >>> erf(4).evalf(30)
    0.999999984582742099719981147840

    >>> erf(-4*I).evalf(30)
    -1296959.73071763923152794095062*I

    See Also
    ========

    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erf.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erf

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfinv

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

        if isinstance(arg, erfinv):
            return arg.args[0]

        if isinstance(arg, erfcinv):
            return S.One - arg.args[0]

        if arg.is_zero:
            return S.Zero

        # Only happens with unevaluated erf2inv
        if isinstance(arg, erf2inv) and arg.args[0].is_zero:
            return arg.args[1]

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return 2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        if self.args[0].is_extended_real is True:
            return True
        # There are cases where erf(z) becomes a real number
        # even if z is a complex number

    def _eval_is_imaginary(self):
        if self.args[0].is_imaginary is True:
            return True

    def _eval_is_finite(self):
        z = self.args[0]
        return fuzzy_or([z.is_finite, z.is_extended_real])

    def _eval_is_zero(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_zero

    def _eval_is_positive(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_extended_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_extended_negative

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        from sympy.series.limits import limit
        if limitvar:
            lim = limit(z, limitvar, S.Infinity)
            if lim is S.NegativeInfinity:
                return S.NegativeOne + _erfs(-z)*exp(-z**2)
        return S.One - _erfs(z)*exp(-z**2)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return S.One - erfc(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return -I*erfi(I*z)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if x in arg.free_symbols and arg0.is_zero:
            return 2*arg/sqrt(pi)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in [S.Infinity, S.NegativeInfinity]:
            z = self.args[0]

            try:
                _, ex = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            ex = -ex # as x->1/x for aseries
            if ex.is_positive:
                newn = ceiling(n/ex)
                s = [S.NegativeOne**k * factorial2(2*k - 1) / (z**(2*k + 1) * 2**k)
                     for k in range(newn)] + [Order(1/z**newn, x)]
                return S.One - (exp(-z**2)/sqrt(pi)) * Add(*s)

        return super(erf, self)._eval_aseries(n, args0, x, logx)

    as_real_imag = real_to_real_as_real_imag


class erfc(DefinedFunction):
    r"""
    Complementary Error Function.

    Explanation
    ===========

    The function is defined as:

    .. math ::
        \mathrm{erfc}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfc
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfc(0)
    1
    >>> erfc(oo)
    0
    >>> erfc(-oo)
    2
    >>> erfc(I*oo)
    -oo*I
    >>> erfc(-I*oo)
    oo*I

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erfc(z))
    erfc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfc(z), z)
    -2*exp(-z**2)/sqrt(pi)

    It also follows

    >>> erfc(-z)
    2 - erfc(z)

    We can numerically evaluate the complementary error function to arbitrary
    precision on the whole complex plane:

    >>> erfc(4).evalf(30)
    0.0000000154172579002800188521596734869

    >>> erfc(4*I).evalf(30)
    1.0 - 1296959.73071763923152794095062*I

    See Also
    ========

    erf: Gaussian error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erfc.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erfc

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfcinv

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg.is_zero:
                return S.One

        if isinstance(arg, erfinv):
            return S.One - arg.args[0]

        if isinstance(arg, erfcinv):
            return arg.args[0]

        if arg.is_zero:
            return S.One

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return -arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return 2 - cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return S.One
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return -2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        if self.args[0].is_extended_real is True:
            return True
        if self.args[0].is_imaginary is True:
            return False

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return S.One - erf(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return S.One + I*erfi(I*z)

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One-I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return S.One - z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return S.One - 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return S.One - sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if arg0.is_zero:
            return S.One
        else:
            return self.func(arg0)

    as_real_imag = real_to_real_as_real_imag

    def _eval_aseries(self, n, args0, x, logx):
        return S.One - erf(*self.args)._eval_aseries(n, args0, x, logx)


class erfi(DefinedFunction):
    r"""
    Imaginary error function.

    Explanation
    ===========

    The function erfi is defined as:

    .. math ::
        \mathrm{erfi}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfi
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfi(0)
    0
    >>> erfi(oo)
    oo
    >>> erfi(-oo)
    -oo
    >>> erfi(I*oo)
    I
    >>> erfi(-I*oo)
    -I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erfi(-z)
    -erfi(z)

    >>> from sympy import conjugate
    >>> conjugate(erfi(z))
    erfi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfi(z), z)
    2*exp(z**2)/sqrt(pi)

    We can numerically evaluate the imaginary error function to arbitrary
    precision on the whole complex plane:

    >>> erfi(2).evalf(30)
    18.5648024145755525987042919132

    >>> erfi(-2*I).evalf(30)
    -0.995322265018952734162069256367*I

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://mathworld.wolfram.com/Erfi.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/Erfi

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, z):
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z.is_zero:
                return S.Zero
            elif z is S.Infinity:
                return S.Infinity

        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(-z)

        # Try to pull out factors of I
        nz = z.extract_multiplicatively(I)
        if nz is not None:
            if nz is S.Infinity:
                return I
            if isinstance(nz, erfinv):
                return I*nz.args[0]
            if isinstance(nz, erfcinv):
                return I*(S.One - nz.args[0])
            # Only happens with unevaluated erf2inv
            if isinstance(nz, erf2inv) and nz.args[0].is_zero:
                return I*nz.args[1]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return 2 * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return -I*erf(I*z)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return I*erfc(I*z) - I

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(-z**2)/z*(uppergamma(S.Half, -z**2)/sqrt(pi) - S.One)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    as_real_imag = real_to_real_as_real_imag

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if x in arg.free_symbols and arg0.is_zero:
            return 2*arg/sqrt(pi)
        elif arg0.is_finite:
            return self.func(arg0)
        return self.func(arg)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point is S.Infinity:
            z = self.args[0]
            s = [factorial2(2*k - 1) / (2**k * z**(2*k + 1))
                    for k in range(n)] + [Order(1/z**n, x)]
            return -I + (exp(z**2)/sqrt(pi)) * Add(*s)

        return super(erfi, self)._eval_aseries(n, args0, x, logx)


class erf2(DefinedFunction):
    r"""
    Two-argument error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf2}(x, y) = \frac{2}{\sqrt{\pi}} \int_x^y e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import oo, erf2
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2(0, 0)
    0
    >>> erf2(x, x)
    0
    >>> erf2(x, oo)
    1 - erf(x)
    >>> erf2(x, -oo)
    -erf(x) - 1
    >>> erf2(oo, y)
    erf(y) - 1
    >>> erf2(-oo, y)
    erf(y) + 1

    In general one can pull out factors of -1:

    >>> erf2(-x, -y)
    -erf2(x, y)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf2(x, y))
    erf2(conjugate(x), conjugate(y))

    Differentiation with respect to $x$, $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2(x, y), x)
    -2*exp(-x**2)/sqrt(pi)
    >>> diff(erf2(x, y), y)
    2*exp(-y**2)/sqrt(pi)

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/Erf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return -2*exp(-x**2)/sqrt(pi)
        elif argindex == 2:
            return 2*exp(-y**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        chk = (S.Infinity, S.NegativeInfinity, S.Zero)
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x == y:
            return S.Zero
        elif x in chk or y in chk:
            return erf(y) - erf(x)

        if isinstance(y, erf2inv) and y.args[0] == x:
            return y.args[1]

        if x.is_zero or y.is_zero or x.is_extended_real and x.is_infinite or \
                y.is_extended_real and y.is_infinite:
            return erf(y) - erf(x)

        #Try to pull out -1 factor
        sign_x = x.could_extract_minus_sign()
        sign_y = y.could_extract_minus_sign()
        if (sign_x and sign_y):
            return -cls(-x, -y)
        elif (sign_x or sign_y):
            return erf(y)-erf(x)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_rewrite_as_erf(self, x, y, **kwargs):
        return erf(y) - erf(x)

    def _eval_rewrite_as_erfc(self, x, y, **kwargs):
        return erfc(x) - erfc(y)

    def _eval_rewrite_as_erfi(self, x, y, **kwargs):
        return I*(erfi(I*x)-erfi(I*y))

    def _eval_rewrite_as_fresnels(self, x, y, **kwargs):
        return erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)

    def _eval_rewrite_as_fresnelc(self, x, y, **kwargs):
        return erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)

    def _eval_rewrite_as_meijerg(self, x, y, **kwargs):
        return erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)

    def _eval_rewrite_as_hyper(self, x, y, **kwargs):
        return erf(y).rewrite(hyper) - erf(x).rewrite(hyper)

    def _eval_rewrite_as_uppergamma(self, x, y, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (sqrt(y**2)/y*(S.One - uppergamma(S.Half, y**2)/sqrt(pi)) -
            sqrt(x**2)/x*(S.One - uppergamma(S.Half, x**2)/sqrt(pi)))

    def _eval_rewrite_as_expint(self, x, y, **kwargs):
        return erf(y).rewrite(expint) - erf(x).rewrite(expint)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_is_zero(self):
        return is_eq(*self.args)

class erfinv(DefinedFunction):
    r"""
    Inverse Error Function. The erfinv function is defined as:

    .. math ::
        \mathrm{erf}(x) = y \quad \Rightarrow \quad \mathrm{erfinv}(y) = x

    Examples
    ========

    >>> from sympy import erfinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfinv(0)
    0
    >>> erfinv(1)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfinv(x), x)
    sqrt(pi)*exp(erfinv(x)**2)/2

    We can numerically evaluate the inverse error function to arbitrary
    precision on [-1, 1]:

    >>> erfinv(0.2).evalf(30)
    0.179143454621291692285822705344

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErf/

    """


    def fdiff(self, argindex =1):
        if argindex == 1:
            return sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else :
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erf

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z is S.NegativeOne:
            return S.NegativeInfinity
        elif z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.Infinity

        if isinstance(z, erf) and z.args[0].is_extended_real:
            return z.args[0]

        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1
        nz = z.extract_multiplicatively(-1)
        if nz is not None and (isinstance(nz, erf) and (nz.args[0]).is_extended_real):
            return -nz.args[0]

    def _eval_rewrite_as_erfcinv(self, z, **kwargs):
        return erfcinv(1-z)

    def _eval_is_zero(self):
        return self.args[0].is_zero


class erfcinv (DefinedFunction):
    r"""
    Inverse Complementary Error Function. The erfcinv function is defined as:

    .. math ::
        \mathrm{erfc}(x) = y \quad \Rightarrow \quad \mathrm{erfcinv}(y) = x

    Examples
    ========

    >>> from sympy import erfcinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfcinv(1)
    0
    >>> erfcinv(0)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfcinv(x), x)
    -sqrt(pi)*exp(erfcinv(x)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErfc/

    """


    def fdiff(self, argindex =1):
        if argindex == 1:
            return -sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfc

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z.is_zero:
            return S.Infinity
        elif z is S.One:
            return S.Zero
        elif z == 2:
            return S.NegativeInfinity

        if z.is_zero:
            return S.Infinity

    def _eval_rewrite_as_erfinv(self, z, **kwargs):
        return erfinv(1-z)

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_infinite(self):
        z = self.args[0]
        return fuzzy_or([z.is_zero, is_eq(z, Integer(2))])


class erf2inv(DefinedFunction):
    r"""
    Two-argument Inverse error function. The erf2inv function is defined as:

    .. math ::
        \mathrm{erf2}(x, w) = y \quad \Rightarrow \quad \mathrm{erf2inv}(x, y) = w

    Examples
    ========

    >>> from sympy import erf2inv, oo
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2inv(0, 0)
    0
    >>> erf2inv(1, 0)
    1
    >>> erf2inv(0, 1)
    oo
    >>> erf2inv(0, y)
    erfinv(y)
    >>> erf2inv(oo, y)
    erfcinv(-y)

    Differentiation with respect to $x$ and $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2inv(x, y), x)
    exp(-x**2 + erf2inv(x, y)**2)
    >>> diff(erf2inv(x, y), y)
    sqrt(pi)*exp(erf2inv(x, y)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse complementary error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/InverseErf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return exp(self.func(x,y)**2-x**2)
        elif argindex == 2:
            return sqrt(pi)*S.Half*exp(self.func(x,y)**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x.is_zero and y.is_zero:
            return S.Zero
        elif x.is_zero and y is S.One:
            return S.Infinity
        elif x is S.One and y.is_zero:
            return S.One
        elif x.is_zero:
            return erfinv(y)
        elif x is S.Infinity:
            return erfcinv(-y)
        elif y.is_zero:
            return x
        elif y is S.Infinity:
            return erfinv(x)

        if x.is_zero:
            if y.is_zero:
                return S.Zero
            else:
                return erfinv(y)
        if y.is_zero:
            return x

    def _eval_is_zero(self):
        x, y = self.args
        if x.is_zero and y.is_zero:
            return True

###############################################################################
#################### EXPONENTIAL INTEGRALS ####################################
###############################################################################

class Ei(DefinedFunction):
    r"""
    The classical exponential integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Ei}(x) = \sum_{n=1}^\infty \frac{x^n}{n\, n!}
                                     + \log(x) + \gamma,

    where $\gamma$ is the Euler-Mascheroni constant.

    If $x$ is a polar number, this defines an analytic function on the
    Riemann surface of the logarithm. Otherwise this defines an analytic
    function in the cut plane $\mathbb{C} \setminus (-\infty, 0]$.

    **Background**

    The name exponential integral comes from the following statement:

    .. math:: \operatorname{Ei}(x) = \int_{-\infty}^x \frac{e^t}{t} \mathrm{d}t

    If the integral is interpreted as a Cauchy principal value, this statement
    holds for $x > 0$ and $\operatorname{Ei}(x)$ as defined above.

    Examples
    ========

    >>> from sympy import Ei, polar_lift, exp_polar, I, pi
    >>> from sympy.abc import x

    >>> Ei(-1)
    Ei(-1)

    This yields a real value:

    >>> Ei(-1).n(chop=True)
    -0.219383934395520

    On the other hand the analytic continuation is not real:

    >>> Ei(polar_lift(-1)).n(chop=True)
    -0.21938393439552 + 3.14159265358979*I

    The exponential integral has a logarithmic branch point at the origin:

    >>> Ei(x*exp_polar(2*I*pi))
    Ei(x) + 2*I*pi

    Differentiation is supported:

    >>> Ei(x).diff(x)
    exp(x)/x

    The exponential integral is related to many other special functions.
    For example:

    >>> from sympy import expint, Shi
    >>> Ei(x).rewrite(expint)
    -expint(1, x*exp_polar(I*pi)) - I*pi
    >>> Ei(x).rewrite(Shi)
    Chi(x) + Shi(x)

    See Also
    ========

    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma: Upper incomplete gamma function.

    References
    ==========

    .. [1] https://dlmf.nist.gov/6.6
    .. [2] https://en.wikipedia.org/wiki/Exponential_integral
    .. [3] Abramowitz & Stegun, section 5: https://web.archive.org/web/20201128173312/http://people.math.sfu.ca/~cbm/aands/page_228.htm

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        elif z is S.NegativeInfinity:
            return S.Zero

        if z.is_zero:
            return S.NegativeInfinity

        nz, n = z.extract_branch_factor()
        if n:
            return Ei(nz) + 2*I*pi*n

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return exp(arg)/arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if (self.args[0]/polar_lift(-1)).is_positive:
            return super()._eval_evalf(prec) + (I*pi)._eval_evalf(prec)
        return super()._eval_evalf(prec)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        # XXX this does not currently work usefully because uppergamma
        #     immediately turns into expint
        return -uppergamma(0, polar_lift(-1)*z) - I*pi

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -expint(1, polar_lift(-1)*z) - I*pi

    def _eval_rewrite_as_li(self, z, **kwargs):
        if isinstance(z, log):
            return li(z.args[0])
        # TODO:
        # Actually it only holds that:
        #  Ei(z) = li(exp(z))
        # for -pi < imag(z) <= pi
        return li(exp(z))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        if z.is_negative:
            return Shi(z) + Chi(z) - I*pi
        else:
            return Shi(z) + Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(z) * _eis(z)

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(S.Exp1**t/t, (t, S.NegativeInfinity, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy import re
        x0 = self.args[0].limit(x, 0)
        arg = self.args[0].as_leading_term(x, cdir=cdir)
        cdir = arg.dir(x, cdir)
        if x0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma - (
                I*pi if re(cdir).is_negative else S.Zero)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_Si(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in (S.Infinity, S.NegativeInfinity):
            z = self.args[0]
            s = [factorial(k) / (z)**k for k in range(n)] + \
                    [Order(1/z**n, x)]
            return (exp(z)/z) * Add(*s)

        return super(Ei, self)._eval_aseries(n, args0, x, logx)


class expint(DefinedFunction):
    r"""
    Generalized exponential integral.

    Explanation
    ===========

    This function is defined as

    .. math:: \operatorname{E}_\nu(z) = z^{\nu - 1} \Gamma(1 - \nu, z),

    where $\Gamma(1 - \nu, z)$ is the upper incomplete gamma function
    (``uppergamma``).

    Hence for $z$ with positive real part we have

    .. math:: \operatorname{E}_\nu(z)
              =   \int_1^\infty \frac{e^{-zt}}{t^\nu} \mathrm{d}t,

    which explains the name.

    The representation as an incomplete gamma function provides an analytic
    continuation for $\operatorname{E}_\nu(z)$. If $\nu$ is a
    non-positive integer, the exponential integral is thus an unbranched
    function of $z$, otherwise there is a branch point at the origin.
    Refer to the incomplete gamma function documentation for details of the
    branching behavior.

    Examples
    ========

    >>> from sympy import expint, S
    >>> from sympy.abc import nu, z

    Differentiation is supported. Differentiation with respect to $z$ further
    explains the name: for integral orders, the exponential integral is an
    iterated integral of the exponential function.

    >>> expint(nu, z).diff(z)
    -expint(nu - 1, z)

    Differentiation with respect to $\nu$ has no classical expression:

    >>> expint(nu, z).diff(nu)
    -z**(nu - 1)*meijerg(((), (1, 1)), ((0, 0, 1 - nu), ()), z)

    At non-postive integer orders, the exponential integral reduces to the
    exponential function:

    >>> expint(0, z)
    exp(-z)/z
    >>> expint(-1, z)
    exp(-z)/z + exp(-z)/z**2

    At half-integers it reduces to error functions:

    >>> expint(S(1)/2, z)
    sqrt(pi)*erfc(sqrt(z))/sqrt(z)

    At positive integer orders it can be rewritten in terms of exponentials
    and ``expint(1, z)``. Use ``expand_func()`` to do this:

    >>> from sympy import expand_func
    >>> expand_func(expint(5, z))
    z**4*expint(1, z)/24 + (-z**3 + z**2 - 2*z + 6)*exp(-z)/24

    The generalised exponential integral is essentially equivalent to the
    incomplete gamma function:

    >>> from sympy import uppergamma
    >>> expint(nu, z).rewrite(uppergamma)
    z**(nu - 1)*uppergamma(1 - nu, z)

    As such it is branched at the origin:

    >>> from sympy import exp_polar, pi, I
    >>> expint(4, z*exp_polar(2*pi*I))
    I*pi*z**3/3 + expint(4, z)
    >>> expint(nu, z*exp_polar(2*pi*I))
    z**(nu - 1)*(exp(2*I*pi*nu) - 1)*gamma(1 - nu) + expint(nu, z)

    See Also
    ========

    Ei: Another related function called exponential integral.
    E1: The classical case, returns expint(1, z).
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma

    References
    ==========

    .. [1] https://dlmf.nist.gov/8.19
    .. [2] https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/
    .. [3] https://en.wikipedia.org/wiki/Exponential_integral

    """


    @classmethod
    def eval(cls, nu, z):
        from sympy.functions.special.gamma_functions import (gamma, uppergamma)
        nu2 = unpolarify(nu)
        if nu != nu2:
            return expint(nu2, z)
        if nu.is_Integer and nu <= 0 or (not nu.is_Integer and (2*nu).is_Integer):
            return unpolarify(expand_mul(z**(nu - 1)*uppergamma(1 - nu, z)))

        # Extract branching information. This can be deduced from what is
        # explained in lowergamma.eval().
        z, n = z.extract_branch_factor()
        if n is S.Zero:
            return
        if nu.is_integer:
            if not nu > 0:
                return
            return expint(nu, z) \
                - 2*pi*I*n*S.NegativeOne**(nu - 1)/factorial(nu - 1)*unpolarify(z)**(nu - 1)
        else:
            return (exp(2*I*pi*nu*n) - 1)*z**(nu - 1)*gamma(1 - nu) + expint(nu, z)

    def fdiff(self, argindex):
        nu, z = self.args
        if argindex == 1:
            return -z**(nu - 1)*meijerg([], [1, 1], [0, 0, 1 - nu], [], z)
        elif argindex == 2:
            return -expint(nu - 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_uppergamma(self, nu, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return z**(nu - 1)*uppergamma(1 - nu, z)

    def _eval_rewrite_as_Ei(self, nu, z, **kwargs):
        if nu == 1:
            return -Ei(z*exp_polar(-I*pi)) - I*pi
        elif nu.is_Integer and nu > 1:
            # DLMF, 8.19.7
            x = -unpolarify(z)
            return x**(nu - 1)/factorial(nu - 1)*E1(z).rewrite(Ei) + \
                exp(x)/factorial(nu - 1) * \
                Add(*[factorial(nu - k - 2)*x**k for k in range(nu - 1)])
        else:
            return self

    def _eval_expand_func(self, **hints):
        return self.rewrite(Ei).rewrite(expint, **hints)

    def _eval_rewrite_as_Si(self, nu, z, **kwargs):
        if nu != 1:
            return self
        return Shi(z) - Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_nseries(self, x, n, logx, cdir=0):
        if not self.args[0].has(x):
            nu = self.args[0]
            if nu == 1:
                f = self._eval_rewrite_as_Si(*self.args)
                return f._eval_nseries(x, n, logx)
            elif nu.is_Integer and nu > 1:
                f = self._eval_rewrite_as_Ei(*self.args)
                return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[1]
        nu = self.args[0]

        if point is S.Infinity:
            z = self.args[1]
            s = [S.NegativeOne**k * RisingFactorial(nu, k) / z**k for k in range(n)] + [Order(1/z**n, x)]
            return (exp(-z)/z) * Add(*s)

        return super(expint, self)._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        n, x = self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(t**-n * exp(-t*x), (t, 1, S.Infinity))


def E1(z):
    """
    Classical case of the generalized exponential integral.

    Explanation
    ===========

    This is equivalent to ``expint(1, z)``.

    Examples
    ========

    >>> from sympy import E1
    >>> E1(0)
    expint(1, 0)

    >>> E1(5)
    expint(1, 5)

    See Also
    ========

    Ei: Exponential integral.
    expint: Generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    """
    return expint(1, z)


class li(DefinedFunction):
    r"""
    The classical logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{li}(x) = \int_0^x \frac{1}{\log(t)} \mathrm{d}t \,.

    Examples
    ========

    >>> from sympy import I, oo, li
    >>> from sympy.abc import z

    Several special values are known:

    >>> li(0)
    0
    >>> li(1)
    -oo
    >>> li(oo)
    oo

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(li(z), z)
    1/log(z)

    Defining the ``li`` function via an integral:
    >>> from sympy import integrate
    >>> integrate(li(z))
    z*li(z) - Ei(2*log(z))

    >>> integrate(li(z),z)
    z*li(z) - Ei(2*log(z))


    The logarithmic integral can also be defined in terms of ``Ei``:

    >>> from sympy import Ei
    >>> li(z).rewrite(Ei)
    Ei(log(z))
    >>> diff(li(z).rewrite(Ei), z)
    1/log(z)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> li(2).evalf(30)
    1.04516378011749278484458888919

    >>> li(2*I).evalf(30)
    1.0652795784357498247001125598 + 3.08346052231061726610939702133*I

    We can even compute Soldner's constant by the help of mpmath:

    >>> from mpmath import findroot
    >>> findroot(li, 2)
    1.45136923488338

    Further transformations include rewriting ``li`` in terms of
    the trigonometric integrals ``Si``, ``Ci``, ``Shi`` and ``Chi``:

    >>> from sympy import Si, Ci, Shi, Chi
    >>> li(z).rewrite(Si)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Ci)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Shi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))
    >>> li(z).rewrite(Chi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))

    See Also
    ========

    Li: Offset logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6
    .. [4] https://mathworld.wolfram.com/SoldnersConstant.html

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        if z.is_zero:
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z = self.args[0]
        # Exclude values on the branch cut (-oo, 0)
        if not z.is_extended_negative:
            return self.func(z.conjugate())

    def _eval_rewrite_as_Li(self, z, **kwargs):
        return Li(z) + li(2)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return Ei(log(z))

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (-uppergamma(0, -log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) - log(-log(z)))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        return (Ci(I*log(z)) - I*Si(I*log(z)) -
                S.Half*(log(S.One/log(z)) - log(log(z))) - log(I*log(z)))

    _eval_rewrite_as_Ci = _eval_rewrite_as_Si

    def _eval_rewrite_as_Shi(self, z, **kwargs):
        return (Chi(log(z)) - Shi(log(z)) - S.Half*(log(S.One/log(z)) - log(log(z))))

    _eval_rewrite_as_Chi = _eval_rewrite_as_Shi

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return (log(z)*hyper((1, 1), (2, 2), log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) + EulerGamma)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (-log(-log(z)) - S.Half*(log(S.One/log(z)) - log(log(z)))
                - meijerg(((), (1,)), ((0, 0), ()), -log(z)))

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return z * _eis(log(z))

    def _eval_nseries(self, x, n, logx, cdir=0):
        z = self.args[0]
        s = [(log(z))**k / (factorial(k) * k) for k in range(1, n)]
        return EulerGamma + log(log(z)) + Add(*s)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True

class Li(DefinedFunction):
    r"""
    The offset logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Li}(x) = \operatorname{li}(x) - \operatorname{li}(2)

    Examples
    ========

    >>> from sympy import Li
    >>> from sympy.abc import z

    The following special value is known:

    >>> Li(2)
    0

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(Li(z), z)
    1/log(z)

    The shifted logarithmic integral can be written in terms of $li(z)$:

    >>> from sympy import li
    >>> Li(z).rewrite(li)
    li(z) - li(2)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> Li(2).evalf(30)
    0

    >>> Li(4).evalf(30)
    1.92242131492155809316615998938

    See Also
    ========

    li: Logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6

    """


    @classmethod
    def eval(cls, z):
        if z is S.Infinity:
            return S.Infinity
        elif z == S(2):
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        return self.rewrite(li).evalf(prec)

    def _eval_rewrite_as_li(self, z, **kwargs):
        return li(z) - li(2)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(li).rewrite("tractable", deep=True)

    def _eval_nseries(self, x, n, logx, cdir=0):
        f = self._eval_rewrite_as_li(*self.args)
        return f._eval_nseries(x, n, logx)

###############################################################################
#################### TRIGONOMETRIC INTEGRALS ##################################
###############################################################################

class TrigonometricIntegral(DefinedFunction):
    """ Base class for trigonometric integrals. """


    @classmethod
    def eval(cls, z):
        if z is S.Zero:
            return cls._atzero
        elif z is S.Infinity:
            return cls._atinf()
        elif z is S.NegativeInfinity:
            return cls._atneginf()

        if z.is_zero:
            return cls._atzero

        nz = z.extract_multiplicatively(polar_lift(I))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(I)
        if nz is not None:
            return cls._Ifactor(nz, 1)
        nz = z.extract_multiplicatively(polar_lift(-I))
        if nz is not None:
            return cls._Ifactor(nz, -1)

        nz = z.extract_multiplicatively(polar_lift(-1))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(-1)
        if nz is not None:
            return cls._minusfactor(nz)

        nz, n = z.extract_branch_factor()
        if n == 0 and nz == z:
            return
        return 2*pi*I*n*cls._trigfunc(0) + cls(nz)

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return self._trigfunc(arg)/arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return self._eval_rewrite_as_expint(z).rewrite(Ei)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return self._eval_rewrite_as_expint(z).rewrite(uppergamma)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE this is fairly inefficient
        if self.args[0].subs(x, 0) != 0:
            return super()._eval_nseries(x, n, logx)
        baseseries = self._trigfunc(x)._eval_nseries(x, n, logx)
        if self._trigfunc(0) != 0:
            baseseries -= 1
        baseseries = baseseries.replace(Pow, lambda t, n: t**n/n, simultaneous=False)
        if self._trigfunc(0) != 0:
            baseseries += EulerGamma + log(x)
        return baseseries.subs(x, self.args[0])._eval_nseries(x, n, logx)


class Si(TrigonometricIntegral):
    r"""
    Sine integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Si}(z) = \int_0^z \frac{\sin{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Si
    >>> from sympy.abc import z

    The sine integral is an antiderivative of $sin(z)/z$:

    >>> Si(z).diff(z)
    sin(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Si(z*exp_polar(2*I*pi))
    Si(z)

    Sine integral behaves much like ordinary sine under multiplication by ``I``:

    >>> Si(I*z)
    I*Shi(z)
    >>> Si(-z)
    -Si(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Si(z).rewrite(expint)
    -I*(-expint(1, z*exp_polar(-I*pi/2))/2 +
         expint(1, z*exp_polar(I*pi/2))/2) + pi/2

    It can be rewritten in the form of sinc function (by definition):

    >>> from sympy import sinc
    >>> Si(z).rewrite(sinc)
    Integral(sinc(_t), (_t, 0, z))

    See Also
    ========

    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    sinc: unnormalized sinc function
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sin
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return pi*S.Half

    @classmethod
    def _atneginf(cls):
        return -pi*S.Half

    @classmethod
    def _minusfactor(cls, z):
        return -Si(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I*Shi(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # XXX should we polarify z?
        return pi/2 + (E1(polar_lift(I)*z) - E1(polar_lift(-I)*z))/2/I

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sinc(t), (t, 0, z))

    _eval_rewrite_as_sinc =  _eval_rewrite_as_Integral

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point is S.Infinity:
            z = self.args[0]
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            return pi/2 - cos(z)*Add(*p) - sin(z)*Add(*q)

        # All other points are not handled
        return super(Si, self)._eval_aseries(n, args0, x, logx)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True


class Ci(TrigonometricIntegral):
    r"""
    Cosine integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Ci}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cos{t} - 1}{t} \mathrm{d}t
           = -\int_x^\infty \frac{\cos{t}}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Ci}(z) =
        -\frac{\operatorname{E}_1\left(e^{i\pi/2} z\right)
               + \operatorname{E}_1\left(e^{-i \pi/2} z\right)}{2}

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.

    The formula also holds as stated
    for $z \in \mathbb{C}$ with $\Re(z) > 0$.
    By lifting to the principal branch, we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Ci
    >>> from sympy.abc import z

    The cosine integral is a primitive of $\cos(z)/z$:

    >>> Ci(z).diff(z)
    cos(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Ci(z*exp_polar(2*I*pi))
    Ci(z) + 2*I*pi

    The cosine integral behaves somewhat like ordinary $\cos$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Ci(polar_lift(I)*z)
    Chi(z) + I*pi/2
    >>> Ci(polar_lift(-1)*z)
    Ci(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Ci(z).rewrite(expint)
    -expint(1, z*exp_polar(-I*pi/2))/2 - expint(1, z*exp_polar(I*pi/2))/2

    See Also
    ========

    Si: Sine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cos
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Zero

    @classmethod
    def _atneginf(cls):
        return I*pi

    @classmethod
    def _minusfactor(cls, z):
        return Ci(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Chi(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -(E1(polar_lift(I)*z) + E1(polar_lift(-I)*z))/2

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return S.EulerGamma + log(z) - Integral((1-cos(t))/t, (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in (S.Infinity, S.NegativeInfinity):
            z = self.args[0]
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            result = sin(z)*(Add(*p)) - cos(z)*(Add(*q))

            if point is S.NegativeInfinity:
                result += I*pi
            return result

        return super(Ci, self)._eval_aseries(n, args0, x, logx)

class Shi(TrigonometricIntegral):
    r"""
    Sinh integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Shi}(z) = \int_0^z \frac{\sinh{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Shi
    >>> from sympy.abc import z

    The Sinh integral is a primitive of $\sinh(z)/z$:

    >>> Shi(z).diff(z)
    sinh(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Shi(z*exp_polar(2*I*pi))
    Shi(z)

    The $\sinh$ integral behaves much like ordinary $\sinh$ under
    multiplication by $i$:

    >>> Shi(I*z)
    I*Si(z)
    >>> Shi(-z)
    -Shi(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Shi(z).rewrite(expint)
    expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sinh
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.NegativeInfinity

    @classmethod
    def _minusfactor(cls, z):
        return -Shi(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I*Si(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # XXX should we polarify z?
        return (E1(z) - E1(exp_polar(I*pi)*z))/2 - I*pi/2

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self


class Chi(TrigonometricIntegral):
    r"""
    Cosh integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Chi}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cosh{t} - 1}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Chi}(z) = \operatorname{Ci}\left(e^{i \pi/2}z\right)
                         - i\frac{\pi}{2},

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Chi
    >>> from sympy.abc import z

    The $\cosh$ integral is a primitive of $\cosh(z)/z$:

    >>> Chi(z).diff(z)
    cosh(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Chi(z*exp_polar(2*I*pi))
    Chi(z) + 2*I*pi

    The $\cosh$ integral behaves somewhat like ordinary $\cosh$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Chi(polar_lift(I)*z)
    Ci(z) + I*pi/2
    >>> Chi(polar_lift(-1)*z)
    Chi(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Chi(z).rewrite(expint)
    -expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cosh
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.Infinity

    @classmethod
    def _minusfactor(cls, z):
        return Chi(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Ci(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -I*pi/2 - (E1(z) + E1(exp_polar(I*pi)*z))/2

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self


###############################################################################
#################### FRESNEL INTEGRALS ########################################
###############################################################################

class FresnelIntegral(DefinedFunction):
    """ Base class for the Fresnel integrals."""

    unbranched = True

    @classmethod
    def eval(cls, z):
        # Values at positive infinities signs
        # if any were extracted automatically
        if z is S.Infinity:
            return S.Half

        # Value at zero
        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1 and I
        prefact = S.One
        newarg = z
        changed = False

        nz = newarg.extract_multiplicatively(-1)
        if nz is not None:
            prefact = -prefact
            newarg = nz
            changed = True

        nz = newarg.extract_multiplicatively(I)
        if nz is not None:
            prefact = cls._sign*I*prefact
            newarg = nz
            changed = True

        if changed:
            return prefact*cls(newarg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self._trigfunc(S.Half*pi*self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    _eval_is_finite = _eval_is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    as_real_imag = real_to_real_as_real_imag


class fresnels(FresnelIntegral):
    r"""
    Fresnel integral S.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{S}(z) = \int_0^z \sin{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnels
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnels(0)
    0
    >>> fresnels(oo)
    1/2
    >>> fresnels(-oo)
    -1/2
    >>> fresnels(I*oo)
    -I/2
    >>> fresnels(-I*oo)
    I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnels(-z)
    -fresnels(z)
    >>> fresnels(I*z)
    -I*fresnels(z)

    The Fresnel S integral obeys the mirror symmetry
    $\overline{S(z)} = S(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnels(z))
    fresnels(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnels(z), z)
    sin(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, sin, expand_func
    >>> integrate(sin(pi*z**2/2), z)
    3*fresnels(z)*gamma(3/4)/(4*gamma(7/4))
    >>> expand_func(integrate(sin(pi*z**2/2), z))
    fresnels(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnels(2).evalf(30)
    0.343415678363698242195300815958

    >>> fresnels(-2*I).evalf(30)
    0.343415678363698242195300815958*I

    See Also
    ========

    fresnelc: Fresnel cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelS
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = sin
    _sign = -S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (-pi**2*x**4*(4*n - 1)/(8*n*(2*n + 1)*(4*n + 3))) * p
            else:
                return x**3 * (-x**4)**n * (S(2)**(-2*n - 1)*pi**(2*n + 1)) / ((4*n + 3)*factorial(2*n + 1))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One + I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(9, 4) / (sqrt(2)*(z**2)**Rational(3, 4)*(-z)**Rational(3, 4))
                * meijerg([], [1], [Rational(3, 4)], [Rational(1, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sin(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return pi*arg**3/6
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo and -oo
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # expansion of S(x) = S1(x*sqrt(pi/2)), see reference[5] page 1-8
            # as only real infinities are dealt with, sin and cos are O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(0, n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            p = [-sqrt(2/pi)*t for t in p]
            q = [-sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # The expansion at oo is 1/2 + some odd powers of z
            # To get the expansion at -oo, replace z by -z and flip the sign
            # The result -1/2 + the same odd powers of z as before.
            return s*S.Half + (sin(z**2)*Add(*p) + cos(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)


class fresnelc(FresnelIntegral):
    r"""
    Fresnel integral C.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{C}(z) = \int_0^z \cos{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnelc
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnelc(0)
    0
    >>> fresnelc(oo)
    1/2
    >>> fresnelc(-oo)
    -1/2
    >>> fresnelc(I*oo)
    I/2
    >>> fresnelc(-I*oo)
    -I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnelc(-z)
    -fresnelc(z)
    >>> fresnelc(I*z)
    I*fresnelc(z)

    The Fresnel C integral obeys the mirror symmetry
    $\overline{C(z)} = C(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnelc(z))
    fresnelc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnelc(z), z)
    cos(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, cos, expand_func
    >>> integrate(cos(pi*z**2/2), z)
    fresnelc(z)*gamma(1/4)/(4*gamma(5/4))
    >>> expand_func(integrate(cos(pi*z**2/2), z))
    fresnelc(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnelc(2).evalf(30)
    0.488253406075340754500223503357

    >>> fresnelc(-2*I).evalf(30)
    -0.488253406075340754500223503357*I

    See Also
    ========

    fresnels: Fresnel sine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelC
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = cos
    _sign = S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (-pi**2*x**4*(4*n - 3)/(8*n*(2*n - 1)*(4*n + 1))) * p
            else:
                return x * (-x**4)**n * (S(2)**(-2*n)*pi**(2*n)) / ((4*n + 1)*factorial(2*n))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One - I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(3, 4) / (sqrt(2)*root(z**2, 4)*root(-z, 4))
                * meijerg([], [1], [Rational(1, 4)], [Rational(3, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(cos(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # expansion of C(x) = C1(x*sqrt(pi/2)), see reference[5] page 1-8
            # as only real infinities are dealt with, sin and cos are O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            p = [-sqrt(2/pi)*t for t in p]
            q = [ sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # The expansion at oo is 1/2 + some odd powers of z
            # To get the expansion at -oo, replace z by -z and flip the sign
            # The result -1/2 + the same odd powers of z as before.
            return s*S.Half + (cos(z**2)*Add(*p) + sin(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)


###############################################################################
#################### HELPER FUNCTIONS #########################################
###############################################################################


class _erfs(DefinedFunction):
    """
    Helper function to make the $\\mathrm{erf}(z)$ function
    tractable for the Gruntz algorithm.

    """
    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point is S.Infinity:
            z = self.args[0]
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # It is very inefficient to first add the order and then do the nseries
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # Expansion at I*oo
        t = point.extract_multiplicatively(I)
        if t is S.Infinity:
            z = self.args[0]
            # TODO: is the series really correct?
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # It is very inefficient to first add the order and then do the nseries
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -2/sqrt(pi) + 2*z*_erfs(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return (S.One - erf(z))*exp(z**2)


class _eis(DefinedFunction):
    """
    Helper function to make the $\\mathrm{Ei}(z)$ and $\\mathrm{li}(z)$
    functions tractable for the Gruntz algorithm.

    """


    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] not in (S.Infinity, S.NegativeInfinity):
            return super()._eval_aseries(n, args0, x, logx)

        z = self.args[0]
        l = [factorial(k) * (1/z)**(k + 1) for k in range(n)]
        o = Order(1/z**(n + 1), x)
        # It is very inefficient to first add the order and then do the nseries
        return (Add(*l))._eval_nseries(x, n, logx) + o


    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return S.One / z - _eis(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return exp(-z)*Ei(z)

    def _eval_as_leading_term(self, x, logx, cdir):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)
<|endoftext|>""" Riemann zeta and related function. """

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, DefinedFunction
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly

###############################################################################
###################### LERCH TRANSCENDENT #####################################
###############################################################################


class lerchphi(DefinedFunction):
    r"""
    Lerch transcendent (Lerch phi function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$, $|z| < 1$ and $s \in \mathbb{C}$, the
    Lerch transcendent is defined as

    .. math :: \Phi(z, s, a) = \sum_{n=0}^\infty \frac{z^n}{(n + a)^s},

    where the standard branch of the argument is used for $n + a$,
    and by analytic continuation for other values of the parameters.

    A commonly used related function is the Lerch zeta function, defined by

    .. math:: L(q, s, a) = \Phi(e^{2\pi i q}, s, a).

    **Analytic Continuation and Branching Behavior**

    It can be shown that

    .. math:: \Phi(z, s, a) = z\Phi(z, s, a+1) + a^{-s}.

    This provides the analytic continuation to $\operatorname{Re}(a) \le 0$.

    Assume now $\operatorname{Re}(a) > 0$. The integral representation

    .. math:: \Phi_0(z, s, a) = \int_0^\infty \frac{t^{s-1} e^{-at}}{1 - ze^{-t}}
                                \frac{\mathrm{d}t}{\Gamma(s)}

    provides an analytic continuation to $\mathbb{C} - [1, \infty)$.
    Finally, for $x \in (1, \infty)$ we find

    .. math:: \lim_{\epsilon \to 0^+} \Phi_0(x + i\epsilon, s, a)
             -\lim_{\epsilon \to 0^+} \Phi_0(x - i\epsilon, s, a)
             = \frac{2\pi i \log^{s-1}{x}}{x^a \Gamma(s)},

    using the standard branch for both $\log{x}$ and
    $\log{\log{x}}$ (a branch of $\log{\log{x}}$ is needed to
    evaluate $\log{x}^{s-1}$).
    This concludes the analytic continuation. The Lerch transcendent is thus
    branched at $z \in \{0, 1, \infty\}$ and
    $a \in \mathbb{Z}_{\le 0}$. For fixed $z, a$ outside these
    branch points, it is an entire function of $s$.

    Examples
    ========

    The Lerch transcendent is a fairly general function, for this reason it does
    not automatically evaluate to simpler functions. Use ``expand_func()`` to
    achieve this.

    If $z=1$, the Lerch transcendent reduces to the Hurwitz zeta function:

    >>> from sympy import lerchphi, expand_func
    >>> from sympy.abc import z, s, a
    >>> expand_func(lerchphi(1, s, a))
    zeta(s, a)

    More generally, if $z$ is a root of unity, the Lerch transcendent
    reduces to a sum of Hurwitz zeta functions:

    >>> expand_func(lerchphi(-1, s, a))
    zeta(s, a/2)/2**s - zeta(s, a/2 + 1/2)/2**s

    If $a=1$, the Lerch transcendent reduces to the polylogarithm:

    >>> expand_func(lerchphi(z, s, 1))
    polylog(s, z)/z

    More generally, if $a$ is rational, the Lerch transcendent reduces
    to a sum of polylogarithms:

    >>> from sympy import S
    >>> expand_func(lerchphi(z, s, S(1)/2))
    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))
    >>> expand_func(lerchphi(z, s, S(3)/2))
    -2**s/z + 2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                          polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))/z

    The derivatives with respect to $z$ and $a$ can be computed in
    closed form:

    >>> lerchphi(z, s, a).diff(z)
    (-a*lerchphi(z, s, a) + lerchphi(z, s - 1, a))/z
    >>> lerchphi(z, s, a).diff(a)
    -s*lerchphi(z, s + 1, a)

    See Also
    ========

    polylog, zeta

    References
    ==========

    .. [1] Bateman, H.; Erdelyi, A. (1953), Higher Transcendental Functions,
           Vol. I, New York: McGraw-Hill. Section 1.11.
    .. [2] https://dlmf.nist.gov/25.14
    .. [3] https://en.wikipedia.org/wiki/Lerch_transcendent

    """

    def _eval_expand_func(self, **hints):
        z, s, a = self.args
        if z == 1:
            return zeta(s, a)
        if s.is_Integer and s <= 0:
            t = Dummy('t')
            p = Poly((t + a)**(-s), t)
            start = 1/(1 - t)
            res = S.Zero
            for c in reversed(p.all_coeffs()):
                res += c*start
                start = t*start.diff(t)
            return res.subs(t, z)

        if a.is_Rational:
            # See section 18 of
            #   Kelly B. Roach.  Hypergeometric Function Representations.
            #   In: Proceedings of the 1997 International Symposium on Symbolic and
            #   Algebraic Computation, pages 205-211, New York, 1997. ACM.
            # TODO should something be polarified here?
            add = S.Zero
            mul = S.One
            # First reduce a to the interaval (0, 1]
            if a > 1:
                n = floor(a)
                if n == a:
                    n -= 1
                a -= n
                mul = z**(-n)
                add = Add(*[-z**(k - n)/(a + k)**s for k in range(n)])
            elif a <= 0:
                n = floor(-a) + 1
                a += n
                mul = z**n
                add = Add(*[z**(n - 1 - k)/(a - k - 1)**s for k in range(n)])

            m, n = S([a.p, a.q])
            zet = exp_polar(2*pi*I/n)
            root = z**(1/n)
            up_zet = unpolarify(zet)
            addargs = []
            for k in range(n):
                p = polylog(s, zet**k*root)
                if isinstance(p, polylog):
                    p = p._eval_expand_func(**hints)
                addargs.append(p/(up_zet**k*root)**m)
            return add + mul*n**(s - 1)*Add(*addargs)

        # TODO use minpoly instead of ad-hoc methods when issue 5888 is fixed
        if isinstance(z, exp) and (z.args[0]/(pi*I)).is_Rational or z in [-1, I, -I]:
            # TODO reference?
            if z == -1:
                p, q = S([1, 2])
            elif z == I:
                p, q = S([1, 4])
            elif z == -I:
                p, q = S([-1, 4])
            else:
                arg = z.args[0]/(2*pi*I)
                p, q = S([arg.p, arg.q])
            return Add(*[exp(2*pi*I*k*p/q)/q**s*zeta(s, (k + a)/q)
                         for k in range(q)])

        return lerchphi(z, s, a)

    def fdiff(self, argindex=1):
        z, s, a = self.args
        if argindex == 3:
            return -s*lerchphi(z, s + 1, a)
        elif argindex == 1:
            return (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
        else:
            raise ArgumentIndexError

    def _eval_rewrite_helper(self, target):
        res = self._eval_expand_func()
        if res.has(target):
            return res
        else:
            return self

    def _eval_rewrite_as_zeta(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(zeta)

    def _eval_rewrite_as_polylog(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(polylog)

###############################################################################
###################### POLYLOGARITHM ##########################################
###############################################################################


class polylog(DefinedFunction):
    r"""
    Polylogarithm function.

    Explanation
    ===========

    For $|z| < 1$ and $s \in \mathbb{C}$, the polylogarithm is
    defined by

    .. math:: \operatorname{Li}_s(z) = \sum_{n=1}^\infty \frac{z^n}{n^s},

    where the standard branch of the argument is used for $n$. It admits
    an analytic continuation which is branched at $z=1$ (notably not on the
    sheet of initial definition), $z=0$ and $z=\infty$.

    The name polylogarithm comes from the fact that for $s=1$, the
    polylogarithm is related to the ordinary logarithm (see examples), and that

    .. math:: \operatorname{Li}_{s+1}(z) =
                    \int_0^z \frac{\operatorname{Li}_s(t)}{t} \mathrm{d}t.

    The polylogarithm is a special case of the Lerch transcendent:

    .. math:: \operatorname{Li}_{s}(z) = z \Phi(z, s, 1).

    Examples
    ========

    For $z \in \{0, 1, -1\}$, the polylogarithm is automatically expressed
    using other functions:

    >>> from sympy import polylog
    >>> from sympy.abc import s
    >>> polylog(s, 0)
    0
    >>> polylog(s, 1)
    zeta(s)
    >>> polylog(s, -1)
    -dirichlet_eta(s)

    If $s$ is a negative integer, $0$ or $1$, the polylogarithm can be
    expressed using elementary functions. This can be done using
    ``expand_func()``:

    >>> from sympy import expand_func
    >>> from sympy.abc import z
    >>> expand_func(polylog(1, z))
    -log(1 - z)
    >>> expand_func(polylog(0, z))
    z/(1 - z)

    The derivative with respect to $z$ can be computed in closed form:

    >>> polylog(s, z).diff(z)
    polylog(s - 1, z)/z

    The polylogarithm can be expressed in terms of the lerch transcendent:

    >>> from sympy import lerchphi
    >>> polylog(s, z).rewrite(lerchphi)
    z*lerchphi(z, s, 1)

    See Also
    ========

    zeta, lerchphi

    """

    @classmethod
    def eval(cls, s, z):
        if z.is_number:
            if z is S.One:
                return zeta(s)
            elif z is S.NegativeOne:
                return -dirichlet_eta(s)
            elif z is S.Zero:
                return S.Zero
            elif s == 2:
                dilogtable = _dilogtable()
                if z in dilogtable:
                    return dilogtable[z]

        if z.is_zero:
            return S.Zero

        # Make an effort to determine if z is 1 to avoid replacing into
        # expression with singularity
        zone = z.equals(S.One)

        if zone:
            return zeta(s)
        elif zone is False:
            # For s = 0 or -1 use explicit formulas to evaluate, but
            # automatically expanding polylog(1, z) to -log(1-z) seems
            # undesirable for summation methods based on hypergeometric
            # functions
            if s is S.Zero:
                return z/(1 - z)
            elif s is S.NegativeOne:
                return z/(1 - z)**2
            if s.is_zero:
                return z/(1 - z)

        # polylog is branched, but not over the unit disk
        if z.has(exp_polar, polar_lift) and (zone or (Abs(z) <= S.One) == True):
            return cls(s, unpolarify(z))

    def fdiff(self, argindex=1):
        s, z = self.args
        if argindex == 2:
            return polylog(s - 1, z)/z
        raise ArgumentIndexError

    def _eval_rewrite_as_lerchphi(self, s, z, **kwargs):
        return z*lerchphi(z, s, 1)

    def _eval_expand_func(self, **hints):
        s, z = self.args
        if s == 1:
            return -log(1 - z)
        if s.is_Integer and s <= 0:
            u = Dummy('u')
            start = u/(1 - u)
            for _ in range(-s):
                start = u*start.diff(u)
            return expand_mul(start).subs(u, z)
        return polylog(s, z)

    def _eval_is_zero(self):
        z = self.args[1]
        if z.is_zero:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args

        z0 = z.subs(x, 0)
        if z0 is S.NaN:
            z0 = z.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        if z0.is_zero:
            # In case of powers less than 1, number of terms need to be computed
            # separately to avoid repeated callings of _eval_nseries with wrong n
            try:
                _, exp = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            if exp.is_positive:
                newn = ceiling(n/exp)
                o = Order(x**n, x)
                r = z._eval_nseries(x, n, logx, cdir).removeO()
                if r is S.Zero:
                    return o

                term = r
                s = [term]
                for k in range(2, newn):
                    term *= r
                    s.append(term/k**nu)
                return Add(*s) + o

        return super(polylog, self)._eval_nseries(x, n, logx, cdir)

###############################################################################
###################### HURWITZ GENERALIZED ZETA FUNCTION ######################
###############################################################################


class zeta(DefinedFunction):
    r"""
    Hurwitz zeta function (or Riemann zeta function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$ and $\operatorname{Re}(s) > 1$, this
    function is defined as

    .. math:: \zeta(s, a) = \sum_{n=0}^\infty \frac{1}{(n + a)^s},

    where the standard choice of argument for $n + a$ is used. For fixed
    $a$ not a nonpositive integer the Hurwitz zeta function admits a
    meromorphic continuation to all of $\mathbb{C}$; it is an unbranched
    function with a simple pole at $s = 1$.

    The Hurwitz zeta function is a special case of the Lerch transcendent:

    .. math:: \zeta(s, a) = \Phi(1, s, a).

    This formula defines an analytic continuation for all possible values of
    $s$ and $a$ (also $\operatorname{Re}(a) < 0$), see the documentation of
    :class:`lerchphi` for a description of the branching behavior.

    If no value is passed for $a$ a default value of $a = 1$ is assumed,
    yielding the Riemann zeta function.

    Examples
    ========

    For $a = 1$ the Hurwitz zeta function reduces to the famous Riemann
    zeta function:

    .. math:: \zeta(s, 1) = \zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}.

    >>> from sympy import zeta
    >>> from sympy.abc import s
    >>> zeta(s, 1)
    zeta(s)
    >>> zeta(s)
    zeta(s)

    The Riemann zeta function can also be expressed using the Dirichlet eta
    function:

    >>> from sympy import dirichlet_eta
    >>> zeta(s).rewrite(dirichlet_eta)
    dirichlet_eta(s)/(1 - 2**(1 - s))

    The Riemann zeta function at nonnegative even and negative integer
    values is related to the Bernoulli numbers and polynomials:

    >>> zeta(2)
    pi**2/6
    >>> zeta(4)
    pi**4/90
    >>> zeta(0)
    -1/2
    >>> zeta(-1)
    -1/12
    >>> zeta(-4)
    0

    The specific formulae are:

    .. math:: \zeta(2n) = -\frac{(2\pi i)^{2n} B_{2n}}{2(2n)!}
    .. math:: \zeta(-n,a) = -\frac{B_{n+1}(a)}{n+1}

    No closed-form expressions are known at positive odd integers, but
    numerical evaluation is possible:

    >>> zeta(3).n()
    1.20205690315959

    The derivative of $\zeta(s, a)$ with respect to $a$ can be computed:

    >>> from sympy.abc import a
    >>> zeta(s, a).diff(a)
    -s*zeta(s + 1, a)

    However the derivative with respect to $s$ has no useful closed form
    expression:

    >>> zeta(s, a).diff(s)
    Derivative(zeta(s, a), s)

    The Hurwitz zeta function can be expressed in terms of the Lerch
    transcendent, :class:`~.lerchphi`:

    >>> from sympy import lerchphi
    >>> zeta(s, a).rewrite(lerchphi)
    lerchphi(1, s, a)

    See Also
    ========

    dirichlet_eta, lerchphi, polylog

    References
    ==========

    .. [1] https://dlmf.nist.gov/25.11
    .. [2] https://en.wikipedia.org/wiki/Hurwitz_zeta_function

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        elif s is S.NaN or a is S.NaN:
            return S.NaN
        elif s is S.One:
            return S.ComplexInfinity
        elif s is S.Infinity:
            return S.One
        elif a is S.Infinity:
            return S.Zero

        sint = s.is_Integer
        if a is None:
            a = S.One
        if sint and s.is_nonpositive:
            return bernoulli(1-s, a) / (s-1)
        elif a is S.One:
            if sint and s.is_even:
                return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        elif sint and a.is_Integer and a.is_positive:
            return cls(s) - harmonic(a-1, s)
        elif a.is_Integer and a.is_nonpositive and \
                (s.is_integer is False or s.is_nonpositive is False):
            return S.NaN

    def _eval_rewrite_as_bernoulli(self, s, a=1, **kwargs):
        if a == 1 and s.is_integer and s.is_nonnegative and s.is_even:
            return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        return bernoulli(1-s, a) / (s-1)

    def _eval_rewrite_as_dirichlet_eta(self, s, a=1, **kwargs):
        if a != 1:
            return self
        s = self.args[0]
        return dirichlet_eta(s)/(1 - 2**(1 - s))

    def _eval_rewrite_as_lerchphi(self, s, a=1, **kwargs):
        return lerchphi(1, s, a)

    def _eval_is_finite(self):
        return fuzzy_not((self.args[0] - 1).is_zero)

    def _eval_expand_func(self, **hints):
        s = self.args[0]
        a = self.args[1] if len(self.args) > 1 else S.One
        if a.is_integer:
            if a.is_positive:
                return zeta(s) - harmonic(a-1, s)
            if a.is_nonpositive and (s.is_integer is False or
                    s.is_nonpositive is False):
                return S.NaN
        return self

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (1,)
        if argindex == 2:
            return -s*zeta(s + 1, a)
        else:
            raise ArgumentIndexError

    def _eval_as_leading_term(self, x, logx, cdir):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (S.One,)

        try:
            c, e = a.leadterm(x)
        except NotImplementedError:
            return self

        if e.is_negative and not s.is_positive:
            raise NotImplementedError

        return super(zeta, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)


class dirichlet_eta(DefinedFunction):
    r"""
    Dirichlet eta function.

    Explanation
    ===========

    For $\operatorname{Re}(s) > 0$ and $0 < x \le 1$, this function is defined as

    .. math:: \eta(s, a) = \sum_{n=0}^\infty \frac{(-1)^n}{(n+a)^s}.

    It admits a unique analytic continuation to all of $\mathbb{C}$ for any
    fixed $a$ not a nonpositive integer. It is an entire, unbranched function.

    It can be expressed using the Hurwitz zeta function as

    .. math:: \eta(s, a) = \zeta(s,a) - 2^{1-s} \zeta\left(s, \frac{a+1}{2}\right)

    and using the generalized Genocchi function as

    .. math:: \eta(s, a) = \frac{G(1-s, a)}{2(s-1)}.

    In both cases the limiting value of $\log2 - \psi(a) + \psi\left(\frac{a+1}{2}\right)$
    is used when $s = 1$.

    Examples
    ========

    >>> from sympy import dirichlet_eta, zeta
    >>> from sympy.abc import s
    >>> dirichlet_eta(s).rewrite(zeta)
    Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1 - s))*zeta(s), True))

    See Also
    ========

    zeta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_eta_function
    .. [2] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        if a is None:
            if s == 1:
                return log(2)
            z = zeta(s)
            if not z.has(zeta):
                return (1 - 2**(1-s)) * z
            return
        elif s == 1:
            from sympy.functions.special.gamma_functions import digamma
            return log(2) - digamma(a) + digamma((a+1)/2)
        z1 = zeta(s, a)
        z2 = zeta(s, (a+1)/2)
        if not z1.has(zeta) and not z2.has(zeta):
            return z1 - 2**(1-s) * z2

    def _eval_rewrite_as_zeta(self, s, a=1, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        if a == 1:
            return Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1-s)) * zeta(s), True))
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (zeta(s, a) - 2**(1-s) * zeta(s, (a+1)/2), True))

    def _eval_rewrite_as_genocchi(self, s, a=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (genocchi(1-s, a) / (2 * (s-1)), True))

    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(zeta)._eval_evalf(prec)


class riemann_xi(DefinedFunction):
    r"""
    Riemann Xi function.

    Examples
    ========

    The Riemann Xi function is closely related to the Riemann zeta function.
    The zeros of Riemann Xi function are precisely the non-trivial zeros
    of the zeta function.

    >>> from sympy import riemann_xi, zeta
    >>> from sympy.abc import s
    >>> riemann_xi(s).rewrite(zeta)
    s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Riemann_Xi_function

    """


    @classmethod
    def eval(cls, s):
        from sympy.functions.special.gamma_functions import gamma
        z = zeta(s)
        if s in (S.Zero, S.One):
            return S.Half

        if not isinstance(z, zeta):
            return s*(s - 1)*gamma(s/2)*z/(2*pi**(s/2))

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))


class stieltjes(DefinedFunction):
    r"""
    Represents Stieltjes constants, $\gamma_{k}$ that occur in
    Laurent Series expansion of the Riemann zeta function.

    Examples
    ========

    >>> from sympy import stieltjes
    >>> from sympy.abc import n, m
    >>> stieltjes(n)
    stieltjes(n)

    The zero'th stieltjes constant:

    >>> stieltjes(0)
    EulerGamma
    >>> stieltjes(0, 1)
    EulerGamma

    For generalized stieltjes constants:

    >>> stieltjes(n, m)
    stieltjes(n, m)

    Constants are only defined for integers >= 0:

    >>> stieltjes(-1)
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stieltjes_constants

    """

    @classmethod
    def eval(cls, n, a=None):
        if a is not None:
            a = sympify(a)
            if a is S.NaN:
                return S.NaN
            if a.is_Integer and a.is_nonpositive:
                return S.ComplexInfinity

        if n.is_Number:
            if n is S.NaN:
                return S.NaN
            elif n < 0:
                return S.ComplexInfinity
            elif not n.is_Integer:
                return S.ComplexInfinity
            elif n is S.Zero and a in [None, 1]:
                return S.EulerGamma

        if n.is_extended_negative:
            return S.ComplexInfinity

        if n.is_zero and a in [None, 1]:
            return S.EulerGamma

        if n.is_integer == False:
            return S.ComplexInfinity


@cacheit
def _dilogtable():
    return {
        S.Half: pi**2/12 - log(2)**2/2,
        Integer(2) : pi**2/4 - I*pi*log(2),
        -(sqrt(5) - 1)/2 : -pi**2/15 + log((sqrt(5)-1)/2)**2/2,
        -(sqrt(5) + 1)/2 : -pi**2/10 - log((sqrt(5)+1)/2)**2,
        (3 - sqrt(5))/2 : pi**2/15 - log((sqrt(5)-1)/2)**2,
        (sqrt(5) - 1)/2 : pi**2/10 - log((sqrt(5)-1)/2)**2,
        I : I*S.Catalan - pi**2/48,
        -I : -I*S.Catalan - pi**2/48,
        1 - I : pi**2/16 - I*S.Catalan - pi*I/4*log(2),
        1 + I : pi**2/16 + I*S.Catalan + pi*I/4*log(2),
        (1 - I)/2 : -log(2)**2/8 + pi*I*log(2)/8 + 5*pi**2/96 - I*S.Catalan
    }
<|endoftext|>from sympy.core import S, sympify
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise, piecewise_fold
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval

from functools import lru_cache


def _ivl(cond, x):
    """return the interval corresponding to the condition

    Conditions in spline's Piecewise give the range over
    which an expression is valid like (lo <= x) & (x <= hi).
    This function returns (lo, hi).
    """
    if isinstance(cond, And) and len(cond.args) == 2:
        a, b = cond.args
        if a.lts == x:
            a, b = b, a
        return a.lts, b.gts
    raise TypeError('unexpected cond type: %s' % cond)


def _add_splines(c, b1, d, b2, x):
    """Construct c*b1 + d*b2."""

    if S.Zero in (b1, c):
        rv = piecewise_fold(d * b2)
    elif S.Zero in (b2, d):
        rv = piecewise_fold(c * b1)
    else:
        new_args = []
        # Just combining the Piecewise without any fancy optimization
        p1 = piecewise_fold(c * b1)
        p2 = piecewise_fold(d * b2)

        # Search all Piecewise arguments except (0, True)
        p2args = list(p2.args[:-1])

        # This merging algorithm assumes the conditions in
        # p1 and p2 are sorted
        for arg in p1.args[:-1]:
            expr = arg.expr
            cond = arg.cond

            lower = _ivl(cond, x)[0]

            # Check p2 for matching conditions that can be merged
            for i, arg2 in enumerate(p2args):
                expr2 = arg2.expr
                cond2 = arg2.cond

                lower_2, upper_2 = _ivl(cond2, x)
                if cond2 == cond:
                    # Conditions match, join expressions
                    expr += expr2
                    # Remove matching element
                    del p2args[i]
                    # No need to check the rest
                    break
                elif lower_2 < lower and upper_2 <= lower:
                    # Check if arg2 condition smaller than arg1,
                    # add to new_args by itself (no match expected
                    # in p1)
                    new_args.append(arg2)
                    del p2args[i]
                    break

            # Checked all, add expr and cond
            new_args.append((expr, cond))

        # Add remaining items from p2args
        new_args.extend(p2args)

        # Add final (0, True)
        new_args.append((0, True))

        rv = Piecewise(*new_args, evaluate=False)

    return rv.expand()


@lru_cache(maxsize=128)
def bspline_basis(d, knots, n, x):
    """
    The $n$-th B-spline at $x$ of degree $d$ with knots.

    Explanation
    ===========

    B-Splines are piecewise polynomials of degree $d$. They are defined on a
    set of knots, which is a sequence of integers or floats.

    Examples
    ========

    The 0th degree splines have a value of 1 on a single interval:

        >>> from sympy import bspline_basis
        >>> from sympy.abc import x
        >>> d = 0
        >>> knots = tuple(range(5))
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1, (x >= 0) & (x <= 1)), (0, True))

    For a given ``(d, knots)`` there are ``len(knots)-d-1`` B-splines
    defined, that are indexed by ``n`` (starting at 0).

    Here is an example of a cubic B-spline:

        >>> bspline_basis(3, tuple(range(5)), 0, x)
        Piecewise((x**3/6, (x >= 0) & (x <= 1)),
                  (-x**3/2 + 2*x**2 - 2*x + 2/3,
                  (x >= 1) & (x <= 2)),
                  (x**3/2 - 4*x**2 + 10*x - 22/3,
                  (x >= 2) & (x <= 3)),
                  (-x**3/6 + 2*x**2 - 8*x + 32/3,
                  (x >= 3) & (x <= 4)),
                  (0, True))

    By repeating knot points, you can introduce discontinuities in the
    B-splines and their derivatives:

        >>> d = 1
        >>> knots = (0, 0, 2, 3, 4)
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1 - x/2, (x >= 0) & (x <= 2)), (0, True))

    It is quite time consuming to construct and evaluate B-splines. If
    you need to evaluate a B-spline many times, it is best to lambdify them
    first:

        >>> from sympy import lambdify
        >>> d = 3
        >>> knots = tuple(range(10))
        >>> b0 = bspline_basis(d, knots, 0, x)
        >>> f = lambdify(x, b0)
        >>> y = f(0.5)

    Parameters
    ==========

    d : integer
        degree of bspline

    knots : list of integer values
        list of knots points of bspline

    n : integer
        $n$-th B-spline

    x : symbol

    See Also
    ========

    bspline_basis_set

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/B-spline

    """
    # make sure x has no assumptions so conditions don't evaluate
    xvar = x
    x = Dummy()

    knots = tuple(sympify(k) for k in knots)
    d = int(d)
    n = int(n)
    n_knots = len(knots)
    n_intervals = n_knots - 1
    if n + d + 1 > n_intervals:
        raise ValueError("n + d + 1 must not exceed len(knots) - 1")
    if d == 0:
        result = Piecewise(
            (S.One, Interval(knots[n], knots[n + 1]).contains(x)), (0, True)
        )
    elif d > 0:
        denom = knots[n + d + 1] - knots[n + 1]
        if denom != S.Zero:
            B = (knots[n + d + 1] - x) / denom
            b2 = bspline_basis(d - 1, knots, n + 1, x)
        else:
            b2 = B = S.Zero

        denom = knots[n + d] - knots[n]
        if denom != S.Zero:
            A = (x - knots[n]) / denom
            b1 = bspline_basis(d - 1, knots, n, x)
        else:
            b1 = A = S.Zero

        result = _add_splines(A, b1, B, b2, x)
    else:
        raise ValueError("degree must be non-negative: %r" % n)

    # return result with user-given x
    return result.xreplace({x: xvar})


def bspline_basis_set(d, knots, x):
    """
    Return the ``len(knots)-d-1`` B-splines at *x* of degree *d*
    with *knots*.

    Explanation
    ===========

    This function returns a list of piecewise polynomials that are the
    ``len(knots)-d-1`` B-splines of degree *d* for the given knots.
    This function calls ``bspline_basis(d, knots, n, x)`` for different
    values of *n*.

    Examples
    ========

    >>> from sympy import bspline_basis_set
    >>> from sympy.abc import x
    >>> d = 2
    >>> knots = range(5)
    >>> splines = bspline_basis_set(d, knots, x)
    >>> splines
    [Piecewise((x**2/2, (x >= 0) & (x <= 1)),
               (-x**2 + 3*x - 3/2, (x >= 1) & (x <= 2)),
               (x**2/2 - 3*x + 9/2, (x >= 2) & (x <= 3)),
               (0, True)),
    Piecewise((x**2/2 - x + 1/2, (x >= 1) & (x <= 2)),
              (-x**2 + 5*x - 11/2, (x >= 2) & (x <= 3)),
              (x**2/2 - 4*x + 8, (x >= 3) & (x <= 4)),
              (0, True))]

    Parameters
    ==========

    d : integer
        degree of bspline

    knots : list of integers
        list of knots points of bspline

    x : symbol

    See Also
    ========

    bspline_basis

    """
    n_splines = len(knots) - d - 1
    return [bspline_basis(d, tuple(knots), i, x) for i in range(n_splines)]


def interpolating_spline(d, x, X, Y):
    """
    Return spline of degree *d*, passing through the given *X*
    and *Y* values.

    Explanation
    ===========

    This function returns a piecewise function such that each part is
    a polynomial of degree not greater than *d*. The value of *d*
    must be 1 or greater and the values of *X* must be strictly
    increasing.

    Examples
    ========

    >>> from sympy import interpolating_spline
    >>> from sympy.abc import x
    >>> interpolating_spline(1, x, [1, 2, 4, 7], [3, 6, 5, 7])
    Piecewise((3*x, (x >= 1) & (x <= 2)),
            (7 - x/2, (x >= 2) & (x <= 4)),
            (2*x/3 + 7/3, (x >= 4) & (x <= 7)))
    >>> interpolating_spline(3, x, [-2, 0, 1, 3, 4], [4, 2, 1, 1, 3])
    Piecewise((7*x**3/117 + 7*x**2/117 - 131*x/117 + 2, (x >= -2) & (x <= 1)),
            (10*x**3/117 - 2*x**2/117 - 122*x/117 + 77/39, (x >= 1) & (x <= 4)))

    Parameters
    ==========

    d : integer
        Degree of Bspline strictly greater than equal to one

    x : symbol

    X : list of strictly increasing real values
        list of X coordinates through which the spline passes

    Y : list of real values
        list of corresponding Y coordinates through which the spline passes

    See Also
    ========

    bspline_basis_set, interpolating_poly

    """
    from sympy.solvers.solveset import linsolve
    from sympy.matrices.dense import Matrix

    # Input sanitization
    d = sympify(d)
    if not (d.is_Integer and d.is_positive):
        raise ValueError("Spline degree must be a positive integer, not %s." % d)
    if len(X) != len(Y):
        raise ValueError("Number of X and Y coordinates must be the same.")
    if len(X) < d + 1:
        raise ValueError("Degree must be less than the number of control points.")
    if not all(a < b for a, b in zip(X, X[1:])):
        raise ValueError("The x-coordinates must be strictly increasing.")
    X = [sympify(i) for i in X]

    # Evaluating knots value
    if d.is_odd:
        j = (d + 1) // 2
        interior_knots = X[j:-j]
    else:
        j = d // 2
        interior_knots = [
            (a + b)/2 for a, b in zip(X[j : -j - 1], X[j + 1 : -j])
        ]

    knots = [X[0]] * (d + 1) + list(interior_knots) + [X[-1]] * (d + 1)

    basis = bspline_basis_set(d, knots, x)

    A = [[b.subs(x, v) for b in basis] for v in X]

    coeff = linsolve((Matrix(A), Matrix(Y)), symbols("c0:{}".format(len(X)), cls=Dummy))
    coeff = list(coeff)[0]
    intervals = {c for b in basis for (e, c) in b.args if c != True}

    # Sorting the intervals
    #  ival contains the end-points of each interval
    intervals = sorted(intervals, key=lambda c: _ivl(c, x))

    basis_dicts = [{c: e for (e, c) in b.args} for b in basis]
    spline = []
    for i in intervals:
        piece = sum(
            [c * d.get(i, S.Zero) for (c, d) in zip(coeff, basis_dicts)], S.Zero
        )
        spline.append((piece, i))
    return Piecewise(*spline)
<|endoftext|>from math import prod

from sympy.core import S, Integer
from sympy.core.function import DefinedFunction
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.iterables import has_dups

###############################################################################
###################### Kronecker Delta, Levi-Civita etc. ######################
###############################################################################


def Eijk(*args, **kwargs):
    """
    Represent the Levi-Civita symbol.

    This is a compatibility wrapper to ``LeviCivita()``.

    See Also
    ========

    LeviCivita

    """
    return LeviCivita(*args, **kwargs)


def eval_levicivita(*args):
    """Evaluate Levi-Civita symbol."""
    n = len(args)
    return prod(
        prod(args[j] - args[i] for j in range(i + 1, n))
        / factorial(i) for i in range(n))
    # converting factorial(i) to int is slightly faster


class LeviCivita(DefinedFunction):
    """
    Represent the Levi-Civita symbol.

    Explanation
    ===========

    For even permutations of indices it returns 1, for odd permutations -1, and
    for everything else (a repeated index) it returns 0.

    Thus it represents an alternating pseudotensor.

    Examples
    ========

    >>> from sympy import LeviCivita
    >>> from sympy.abc import i, j, k
    >>> LeviCivita(1, 2, 3)
    1
    >>> LeviCivita(1, 3, 2)
    -1
    >>> LeviCivita(1, 2, 2)
    0
    >>> LeviCivita(i, j, k)
    LeviCivita(i, j, k)
    >>> LeviCivita(i, j, i)
    0

    See Also
    ========

    Eijk

    """

    is_integer = True

    @classmethod
    def eval(cls, *args):
        if all(isinstance(a, (SYMPY_INTS, Integer)) for a in args):
            return eval_levicivita(*args)
        if has_dups(args):
            return S.Zero

    def doit(self, **hints):
        return eval_levicivita(*self.args)


class KroneckerDelta(DefinedFunction):
    """
    The discrete, or Kronecker, delta function.

    Explanation
    ===========

    A function that takes in two integers $i$ and $j$. It returns $0$ if $i$
    and $j$ are not equal, or it returns $1$ if $i$ and $j$ are equal.

    Examples
    ========

    An example with integer indices:

        >>> from sympy import KroneckerDelta
        >>> KroneckerDelta(1, 2)
        0
        >>> KroneckerDelta(3, 3)
        1

    Symbolic indices:

        >>> from sympy.abc import i, j, k
        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

    Parameters
    ==========

    i : Number, Symbol
        The first index of the delta function.
    j : Number, Symbol
        The second index of the delta function.

    See Also
    ========

    eval
    DiracDelta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_delta

    """

    is_integer = True

    @classmethod
    def eval(cls, i, j, delta_range=None):
        """
        Evaluates the discrete delta function.

        Examples
        ========

        >>> from sympy import KroneckerDelta
        >>> from sympy.abc import i, j, k

        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

        # indirect doctest

        """

        if delta_range is not None:
            dinf, dsup = delta_range
            if (dinf - i > 0) == True:
                return S.Zero
            if (dinf - j > 0) == True:
                return S.Zero
            if (dsup - i < 0) == True:
                return S.Zero
            if (dsup - j < 0) == True:
                return S.Zero

        diff = i - j
        if diff.is_zero:
            return S.One
        elif fuzzy_not(diff.is_zero):
            return S.Zero

        if i.assumptions0.get("below_fermi") and \
                j.assumptions0.get("above_fermi"):
            return S.Zero
        if j.assumptions0.get("below_fermi") and \
                i.assumptions0.get("above_fermi"):
            return S.Zero
        # to make KroneckerDelta canonical
        # following lines will check if inputs are in order
        # if not, will return KroneckerDelta with correct order
        if default_sort_key(j) < default_sort_key(i):
            if delta_range:
                return cls(j, i, delta_range)
            else:
                return cls(j, i)

    @property
    def delta_range(self):
        if len(self.args) > 2:
            return self.args[2]

    def _eval_power(self, expt):
        if expt.is_positive:
            return self
        if expt.is_negative and expt is not S.NegativeOne:
            return 1/self

    @property
    def is_above_fermi(self):
        """
        True if Delta can be non-zero above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_above_fermi
        True
        >>> KroneckerDelta(p, i).is_above_fermi
        False
        >>> KroneckerDelta(p, q).is_above_fermi
        True

        See Also
        ========

        is_below_fermi, is_only_below_fermi, is_only_above_fermi

        """
        if self.args[0].assumptions0.get("below_fermi"):
            return False
        if self.args[1].assumptions0.get("below_fermi"):
            return False
        return True

    @property
    def is_below_fermi(self):
        """
        True if Delta can be non-zero below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_below_fermi
        False
        >>> KroneckerDelta(p, i).is_below_fermi
        True
        >>> KroneckerDelta(p, q).is_below_fermi
        True

        See Also
        ========

        is_above_fermi, is_only_above_fermi, is_only_below_fermi

        """
        if self.args[0].assumptions0.get("above_fermi"):
            return False
        if self.args[1].assumptions0.get("above_fermi"):
            return False
        return True

    @property
    def is_only_above_fermi(self):
        """
        True if Delta is restricted to above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_only_above_fermi
        True
        >>> KroneckerDelta(p, q).is_only_above_fermi
        False
        >>> KroneckerDelta(p, i).is_only_above_fermi
        False

        See Also
        ========

        is_above_fermi, is_below_fermi, is_only_below_fermi

        """
        return ( self.args[0].assumptions0.get("above_fermi")
                or
                self.args[1].assumptions0.get("above_fermi")
                ) or False

    @property
    def is_only_below_fermi(self):
        """
        True if Delta is restricted to below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, i).is_only_below_fermi
        True
        >>> KroneckerDelta(p, q).is_only_below_fermi
        False
        >>> KroneckerDelta(p, a).is_only_below_fermi
        False

        See Also
        ========

        is_above_fermi, is_below_fermi, is_only_above_fermi

        """
        return ( self.args[0].assumptions0.get("below_fermi")
                or
                self.args[1].assumptions0.get("below_fermi")
                ) or False

    @property
    def indices_contain_equal_information(self):
        """
        Returns True if indices are either both above or below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, q).indices_contain_equal_information
        True
        >>> KroneckerDelta(p, q+1).indices_contain_equal_information
        True
        >>> KroneckerDelta(i, p).indices_contain_equal_information
        False

        """
        if (self.args[0].assumptions0.get("below_fermi") and
                self.args[1].assumptions0.get("below_fermi")):
            return True
        if (self.args[0].assumptions0.get("above_fermi")
                and self.args[1].assumptions0.get("above_fermi")):
            return True

        # if both indices are general we are True, else false
        return self.is_below_fermi and self.is_above_fermi

    @property
    def preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        Explanation
        ===========

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, 'a' is preferred before
        'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).preferred_index
        i
        >>> KroneckerDelta(p, a).preferred_index
        a
        >>> KroneckerDelta(i, j).preferred_index
        i

        See Also
        ========

        killable_index

        """
        if self._get_preferred_index():
            return self.args[1]
        else:
            return self.args[0]

    @property
    def killable_index(self):
        """
        Returns the index which is preferred to substitute in the final
        expression.

        Explanation
        ===========

        The index to substitute is the index with less information regarding
        fermi level. If indices contain the same information, 'a' is preferred
        before 'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).killable_index
        p
        >>> KroneckerDelta(p, a).killable_index
        p
        >>> KroneckerDelta(i, j).killable_index
        j

        See Also
        ========

        preferred_index

        """
        if self._get_preferred_index():
            return self.args[0]
        else:
            return self.args[1]

    def _get_preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, index 0 is returned.

        """
        if not self.is_above_fermi:
            if self.args[0].assumptions0.get("below_fermi"):
                return 0
            else:
                return 1
        elif not self.is_below_fermi:
            if self.args[0].assumptions0.get("above_fermi"):
                return 0
            else:
                return 1
        else:
            return 0

    @property
    def indices(self):
        return self.args[0:2]

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        i, j = args
        return Piecewise((0, Ne(i, j)), (1, True))
<|endoftext|>"""
This module mainly implements special orthogonal polynomials.

See also functions.combinatorial.numbers which contains some
combinatorial polynomials.

"""

from sympy.core import Rational
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (chebyshevt_poly, chebyshevu_poly,
                                    gegenbauer_poly, hermite_poly, hermite_prob_poly,
                                    jacobi_poly, laguerre_poly, legendre_poly)

_x = Dummy('x')


class OrthogonalPolynomial(DefinedFunction):
    """Base class for orthogonal polynomials.
    """

    @classmethod
    def _eval_at_order(cls, n, x):
        if n.is_integer and n >= 0:
            return cls._ortho_poly(int(n), _x).subs(_x, x)

    def _eval_conjugate(self):
        return self.func(self.args[0], self.args[1].conjugate())

#----------------------------------------------------------------------------
# Jacobi polynomials
#


class jacobi(OrthogonalPolynomial):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi(n, alpha, beta, x)`` gives the $n$th Jacobi polynomial
    in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    Examples
    ========

    >>> from sympy import jacobi, S, conjugate, diff
    >>> from sympy.abc import a, b, n, x

    >>> jacobi(0, a, b, x)
    1
    >>> jacobi(1, a, b, x)
    a/2 - b/2 + x*(a/2 + b/2 + 1)
    >>> jacobi(2, a, b, x)
    a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 + x**2*(a**2/8 + a*b/4 + 7*a/8 + b**2/8 + 7*b/8 + 3/2) + x*(a**2/4 + 3*a/4 - b**2/4 - 3*b/4) - 1/2

    >>> jacobi(n, a, b, x)
    jacobi(n, a, b, x)

    >>> jacobi(n, a, a, x)
    RisingFactorial(a + 1, n)*gegenbauer(n,
        a + 1/2, x)/RisingFactorial(2*a + 1, n)

    >>> jacobi(n, 0, 0, x)
    legendre(n, x)

    >>> jacobi(n, S(1)/2, S(1)/2, x)
    RisingFactorial(3/2, n)*chebyshevu(n, x)/factorial(n + 1)

    >>> jacobi(n, -S(1)/2, -S(1)/2, x)
    RisingFactorial(1/2, n)*chebyshevt(n, x)/factorial(n)

    >>> jacobi(n, a, b, -x)
    (-1)**n*jacobi(n, b, a, x)

    >>> jacobi(n, a, b, 0)
    gamma(a + n + 1)*hyper((-n, -b - n), (a + 1,), -1)/(2**n*factorial(n)*gamma(a + 1))
    >>> jacobi(n, a, b, 1)
    RisingFactorial(a + 1, n)/factorial(n)

    >>> conjugate(jacobi(n, a, b, x))
    jacobi(n, conjugate(a), conjugate(b), conjugate(x))

    >>> diff(jacobi(n,a,b,x), x)
    (a/2 + b/2 + n/2 + 1/2)*jacobi(n - 1, a + 1, b + 1, x)

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """

    @classmethod
    def eval(cls, n, a, b, x):
        # Simplify to other polynomials
        # P^{a, a}_n(x)
        if a == b:
            if a == Rational(-1, 2):
                return RisingFactorial(S.Half, n) / factorial(n) * chebyshevt(n, x)
            elif a.is_zero:
                return legendre(n, x)
            elif a == S.Half:
                return RisingFactorial(3*S.Half, n) / factorial(n + 1) * chebyshevu(n, x)
            else:
                return RisingFactorial(a + 1, n) / RisingFactorial(2*a + 1, n) * gegenbauer(n, a + S.Half, x)
        elif b == -a:
            # P^{a, -a}_n(x)
            return gamma(n + a + 1) / gamma(n + 1) * (1 + x)**(a/2) / (1 - x)**(a/2) * assoc_legendre(n, -a, x)

        if not n.is_Number:
            # Symbolic result P^{a,b}_n(x)
            # P^{a,b}_n(-x)  --->  (-1)**n * P^{b,a}_n(-x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * jacobi(n, b, a, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return (2**(-n) * gamma(a + n + 1) / (gamma(a + 1) * factorial(n)) *
                        hyper([-b - n, -n], [a + 1], -1))
            if x == S.One:
                return RisingFactorial(a + 1, n) / factorial(n)
            elif x is S.Infinity:
                if n.is_positive:
                    # Make sure a+b+2*n \notin Z
                    if (a + b + 2*n).is_integer:
                        raise ValueError("Error. a + b + 2*n should not be an integer.")
                    return RisingFactorial(a + b + n + 1, n) * S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            return jacobi_poly(n, a, b, x)

    def fdiff(self, argindex=4):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt a
            n, a, b, x = self.args
            k = Dummy("k")
            f1 = 1 / (a + b + n + k + 1)
            f2 = ((a + b + 2*k + 1) * RisingFactorial(b + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt b
            n, a, b, x = self.args
            k = Dummy("k")
            f1 = 1 / (a + b + n + k + 1)
            f2 = (-1)**(n - k) * ((a + b + 2*k + 1) * RisingFactorial(a + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 4:
            # Diff wrt x
            n, a, b, x = self.args
            return S.Half * (a + b + n + 1) * jacobi(n - 1, a + 1, b + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, b, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = (RisingFactorial(-n, k) * RisingFactorial(a + b + n + 1, k) * RisingFactorial(a + k + 1, n - k) /
                factorial(k) * ((1 - x)/2)**k)
        return 1 / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, a, b, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, a, b, x, **kwargs)

    def _eval_conjugate(self):
        n, a, b, x = self.args
        return self.func(n, a.conjugate(), b.conjugate(), x.conjugate())


def jacobi_normalized(n, a, b, x):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi_normalized(n, alpha, beta, x)`` gives the $n$th
    Jacobi polynomial in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    This functions returns the polynomials normilzed:

    .. math::

        \int_{-1}^{1}
          P_m^{\left(\alpha, \beta\right)}(x)
          P_n^{\left(\alpha, \beta\right)}(x)
          (1-x)^{\alpha} (1+x)^{\beta} \mathrm{d}x
        = \delta_{m,n}

    Examples
    ========

    >>> from sympy import jacobi_normalized
    >>> from sympy.abc import n,a,b,x

    >>> jacobi_normalized(n, a, b, x)
    jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)/((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1)))

    Parameters
    ==========

    n : integer degree of polynomial

    a : alpha value

    b : beta value

    x : symbol

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """
    nfactor = (S(2)**(a + b + 1) * (gamma(n + a + 1) * gamma(n + b + 1))
               / (2*n + a + b + 1) / (factorial(n) * gamma(n + a + b + 1)))

    return jacobi(n, a, b, x) / sqrt(nfactor)


#----------------------------------------------------------------------------
# Gegenbauer polynomials
#


class gegenbauer(OrthogonalPolynomial):
    r"""
    Gegenbauer polynomial $C_n^{\left(\alpha\right)}(x)$.

    Explanation
    ===========

    ``gegenbauer(n, alpha, x)`` gives the $n$th Gegenbauer polynomial
    in $x$, $C_n^{\left(\alpha\right)}(x)$.

    The Gegenbauer polynomials are orthogonal on $[-1, 1]$ with
    respect to the weight $\left(1-x^2\right)^{\alpha-\frac{1}{2}}$.

    Examples
    ========

    >>> from sympy import gegenbauer, conjugate, diff
    >>> from sympy.abc import n,a,x
    >>> gegenbauer(0, a, x)
    1
    >>> gegenbauer(1, a, x)
    2*a*x
    >>> gegenbauer(2, a, x)
    -a + x**2*(2*a**2 + 2*a)
    >>> gegenbauer(3, a, x)
    x**3*(4*a**3/3 + 4*a**2 + 8*a/3) + x*(-2*a**2 - 2*a)

    >>> gegenbauer(n, a, x)
    gegenbauer(n, a, x)
    >>> gegenbauer(n, a, -x)
    (-1)**n*gegenbauer(n, a, x)

    >>> gegenbauer(n, a, 0)
    2**n*sqrt(pi)*gamma(a + n/2)/(gamma(a)*gamma(1/2 - n/2)*gamma(n + 1))
    >>> gegenbauer(n, a, 1)
    gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    >>> conjugate(gegenbauer(n, a, x))
    gegenbauer(n, conjugate(a), conjugate(x))

    >>> diff(gegenbauer(n, a, x), x)
    2*a*gegenbauer(n - 1, a + 1, x)

    See Also
    ========

    jacobi,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials
    .. [2] https://mathworld.wolfram.com/GegenbauerPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/GegenbauerC3/

    """

    @classmethod
    def eval(cls, n, a, x):
        # For negative n the polynomials vanish
        # See https://functions.wolfram.com/Polynomials/GegenbauerC3/03/01/03/0012/
        if n.is_negative:
            return S.Zero

        # Some special values for fixed a
        if a == S.Half:
            return legendre(n, x)
        elif a == S.One:
            return chebyshevu(n, x)
        elif a == S.NegativeOne:
            return S.Zero

        if not n.is_Number:
            # Handle this before the general sign extraction rule
            if x == S.NegativeOne:
                if (re(a) > S.Half) == True:
                    return S.ComplexInfinity
                else:
                    return (cos(S.Pi*(a+n)) * sec(S.Pi*a) * gamma(2*a+n) /
                                (gamma(2*a) * gamma(n+1)))

            # Symbolic result C^a_n(x)
            # C^a_n(-x)  --->  (-1)**n * C^a_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * gegenbauer(n, a, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return (2**n * sqrt(S.Pi) * gamma(a + S.Half*n) /
                        (gamma((1 - n)/2) * gamma(n + 1) * gamma(a)) )
            if x == S.One:
                return gamma(2*a + n) / (gamma(2*a) * gamma(n + 1))
            elif x is S.Infinity:
                if n.is_positive:
                    return RisingFactorial(a, n) * S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            return gegenbauer_poly(n, a, x)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt a
            n, a, x = self.args
            k = Dummy("k")
            factor1 = 2 * (1 + (-1)**(n - k)) * (k + a) / ((k +
                           n + 2*a) * (n - k))
            factor2 = 2*(k + 1) / ((k + 2*a) * (2*k + 2*a + 1)) + \
                2 / (k + n + 2*a)
            kern = factor1*gegenbauer(k, a, x) + factor2*gegenbauer(n, a, x)
            return Sum(kern, (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt x
            n, a, x = self.args
            return 2*a*gegenbauer(n - 1, a + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = ((-1)**k * RisingFactorial(a, n - k) * (2*x)**(n - 2*k) /
                (factorial(k) * factorial(n - 2*k)))
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, a, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, a, x, **kwargs)

    def _eval_conjugate(self):
        n, a, x = self.args
        return self.func(n, a.conjugate(), x.conjugate())

#----------------------------------------------------------------------------
# Chebyshev polynomials of first and second kind
#


class chebyshevt(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the first kind, $T_n(x)$.

    Explanation
    ===========

    ``chebyshevt(n, x)`` gives the $n$th Chebyshev polynomial (of the first
    kind) in $x$, $T_n(x)$.

    The Chebyshev polynomials of the first kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\frac{1}{\sqrt{1-x^2}}$.

    Examples
    ========

    >>> from sympy import chebyshevt, diff
    >>> from sympy.abc import n,x
    >>> chebyshevt(0, x)
    1
    >>> chebyshevt(1, x)
    x
    >>> chebyshevt(2, x)
    2*x**2 - 1

    >>> chebyshevt(n, x)
    chebyshevt(n, x)
    >>> chebyshevt(n, -x)
    (-1)**n*chebyshevt(n, x)
    >>> chebyshevt(-n, x)
    chebyshevt(n, x)

    >>> chebyshevt(n, 0)
    cos(pi*n/2)
    >>> chebyshevt(n, -1)
    (-1)**n

    >>> diff(chebyshevt(n, x), x)
    n*chebyshevu(n - 1, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    _ortho_poly = staticmethod(chebyshevt_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result T_n(x)
            # T_n(-x)  --->  (-1)**n * T_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevt(n, -x)
            # T_{-n}(x)  --->  T_n(x)
            if n.could_extract_minus_sign():
                return chebyshevt(-n, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                # T_{-n}(x) == T_n(x)
                return cls._eval_at_order(-n, x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return n * chebyshevu(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = binomial(n, 2*k) * (x**2 - 1)**k * x**(n - 2*k)
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class chebyshevu(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the second kind, $U_n(x)$.

    Explanation
    ===========

    ``chebyshevu(n, x)`` gives the $n$th Chebyshev polynomial of the second
    kind in x, $U_n(x)$.

    The Chebyshev polynomials of the second kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\sqrt{1-x^2}$.

    Examples
    ========

    >>> from sympy import chebyshevu, diff
    >>> from sympy.abc import n,x
    >>> chebyshevu(0, x)
    1
    >>> chebyshevu(1, x)
    2*x
    >>> chebyshevu(2, x)
    4*x**2 - 1

    >>> chebyshevu(n, x)
    chebyshevu(n, x)
    >>> chebyshevu(n, -x)
    (-1)**n*chebyshevu(n, x)
    >>> chebyshevu(-n, x)
    -chebyshevu(n - 2, x)

    >>> chebyshevu(n, 0)
    cos(pi*n/2)
    >>> chebyshevu(n, 1)
    n + 1

    >>> diff(chebyshevu(n, x), x)
    (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    _ortho_poly = staticmethod(chebyshevu_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result U_n(x)
            # U_n(-x)  --->  (-1)**n * U_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevu(n, -x)
            # U_{-n}(x)  --->  -U_{n-2}(x)
            if n.could_extract_minus_sign():
                if n == S.NegativeOne:
                    # n can not be -1 here
                    return S.Zero
                elif not (-n - 2).could_extract_minus_sign():
                    return -chebyshevu(-n - 2, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One + n
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                # U_{-n}(x)  --->  -U_{n-2}(x)
                if n == S.NegativeOne:
                    return S.Zero
                else:
                    return -cls._eval_at_order(-n - 2, x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return ((n + 1) * chebyshevt(n + 1, x) - x * chebyshevu(n, x)) / (x**2 - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k * factorial(
            n - k) * (2*x)**(n - 2*k) / (factorial(k) * factorial(n - 2*k))
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class chebyshevt_root(DefinedFunction):
    r"""
    ``chebyshev_root(n, k)`` returns the $k$th root (indexed from zero) of
    the $n$th Chebyshev polynomial of the first kind; that is, if
    $0 \le k < n$, ``chebyshevt(n, chebyshevt_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevt, chebyshevt_root
    >>> chebyshevt_root(3, 2)
    -sqrt(3)/2
    >>> chebyshevt(3, chebyshevt_root(3, 2))
    0

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """

    @classmethod
    def eval(cls, n, k):
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        return cos(S.Pi*(2*k + 1)/(2*n))


class chebyshevu_root(DefinedFunction):
    r"""
    ``chebyshevu_root(n, k)`` returns the $k$th root (indexed from zero) of the
    $n$th Chebyshev polynomial of the second kind; that is, if $0 \le k < n$,
    ``chebyshevu(n, chebyshevu_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevu, chebyshevu_root
    >>> chebyshevu_root(3, 2)
    -sqrt(2)/2
    >>> chebyshevu(3, chebyshevu_root(3, 2))
    0

    See Also
    ========

    chebyshevt, chebyshevt_root, chebyshevu,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """


    @classmethod
    def eval(cls, n, k):
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        return cos(S.Pi*(k + 1)/(n + 1))

#----------------------------------------------------------------------------
# Legendre polynomials and Associated Legendre polynomials
#


class legendre(OrthogonalPolynomial):
    r"""
    ``legendre(n, x)`` gives the $n$th Legendre polynomial of $x$, $P_n(x)$

    Explanation
    ===========

    The Legendre polynomials are orthogonal on $[-1, 1]$ with respect to
    the constant weight 1. They satisfy $P_n(1) = 1$ for all $n$; further,
    $P_n$ is odd for odd $n$ and even for even $n$.

    Examples
    ========

    >>> from sympy import legendre, diff
    >>> from sympy.abc import x, n
    >>> legendre(0, x)
    1
    >>> legendre(1, x)
    x
    >>> legendre(2, x)
    3*x**2/2 - 1/2
    >>> legendre(n, x)
    legendre(n, x)
    >>> diff(legendre(n,x), x)
    n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Legendre_polynomial
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    _ortho_poly = staticmethod(legendre_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_n(-x)  --->  (-1)**n * L_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * legendre(n, -x)
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return legendre(-n - S.One, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return sqrt(S.Pi)/(gamma(S.Half - n/2)*gamma(S.One + n/2))
            elif x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial;
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.is_negative:
                n = -n - S.One
            return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = +/-1
            # https://www.autodiff.org/ad16/Oral/Buecker_Legendre.pdf says
            # at x = 1:
            #    n*(n + 1)/2            , m = 0
            #    oo                     , m = 1
            #    -(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                      , m = 3, 4, ..., n
            #
            # at x = -1
            #    (-1)**(n+1)*n*(n + 1)/2       , m = 0
            #    (-1)**n*oo                    , m = 1
            #    (-1)**n*(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                             , m = 3, 4, ..., n
            n, x = self.args
            return n/(x**2 - 1)*(x*legendre(n, x) - legendre(n - 1, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k*binomial(n, k)**2*((1 + x)/2)**(n - k)*((1 - x)/2)**k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class assoc_legendre(DefinedFunction):
    r"""
    ``assoc_legendre(n, m, x)`` gives $P_n^m(x)$, where $n$ and $m$ are
    the degree and order or an expression which is related to the nth
    order Legendre polynomial, $P_n(x)$ in the following manner:

    .. math::
        P_n^m(x) = (-1)^m (1 - x^2)^{\frac{m}{2}}
                   \frac{\mathrm{d}^m P_n(x)}{\mathrm{d} x^m}

    Explanation
    ===========

    Associated Legendre polynomials are orthogonal on $[-1, 1]$ with:

    - weight $= 1$            for the same $m$ and different $n$.
    - weight $= \frac{1}{1-x^2}$   for the same $n$ and different $m$.

    Examples
    ========

    >>> from sympy import assoc_legendre
    >>> from sympy.abc import x, m, n
    >>> assoc_legendre(0,0, x)
    1
    >>> assoc_legendre(1,0, x)
    x
    >>> assoc_legendre(1,1, x)
    -sqrt(1 - x**2)
    >>> assoc_legendre(n,m,x)
    assoc_legendre(n, m, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    @classmethod
    def _eval_at_order(cls, n, m):
        P = legendre_poly(n, _x, polys=True).diff((_x, m))
        return S.NegativeOne**m * (1 - _x**2)**Rational(m, 2) * P.as_expr()

    @classmethod
    def eval(cls, n, m, x):
        if m.could_extract_minus_sign():
            # P^{-m}_n  --->  F * P^m_n
            return S.NegativeOne**(-m) * (factorial(m + n)/factorial(n - m)) * assoc_legendre(n, -m, x)
        if m == 0:
            # P^0_n  --->  L_n
            return legendre(n, x)
        if x == 0:
            return 2**m*sqrt(S.Pi) / (gamma((1 - m - n)/2)*gamma(1 - (m - n)/2))
        if n.is_Number and m.is_Number and n.is_integer and m.is_integer:
            if n.is_negative:
                raise ValueError("%s : 1st index must be nonnegative integer (got %r)" % (cls, n))
            if abs(m) > n:
                raise ValueError("%s : abs('2nd index') must be <= '1st index' (got %r, %r)" % (cls, n, m))
            return cls._eval_at_order(int(n), abs(int(m))).subs(_x, x)

    def fdiff(self, argindex=3):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt m
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = 1
            n, m, x = self.args
            return 1/(x**2 - 1)*(x*n*assoc_legendre(n, m, x) - (m + n)*assoc_legendre(n - 1, m, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, m, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = factorial(2*n - 2*k)/(2**n*factorial(n - k)*factorial(
            k)*factorial(n - 2*k - m))*S.NegativeOne**k*x**(n - m - 2*k)
        return (1 - x**2)**(m/2) * Sum(kern, (k, 0, floor((n - m)*S.Half)))

    def _eval_rewrite_as_polynomial(self, n, m, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, m, x, **kwargs)

    def _eval_conjugate(self):
        n, m, x = self.args
        return self.func(n, m.conjugate(), x.conjugate())

