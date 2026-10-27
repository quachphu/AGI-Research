                    reval = mpf_abs(reval)
                real.append(reval)
        s = mpf_sum(real, prec, rnd, absolute)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def fdot(ctx, A, B=None, conjugate=False):
        r"""
        Computes the dot product of the iterables `A` and `B`,

        .. math ::

            \sum_{k=0} A_k B_k.

        Alternatively, :func:`~mpmath.fdot` accepts a single iterable of pairs.
        In other words, ``fdot(A,B)`` and ``fdot(zip(A,B))`` are equivalent.
        The elements are automatically converted to mpmath numbers.

        With ``conjugate=True``, the elements in the second vector
        will be conjugated:

        .. math ::

            \sum_{k=0} A_k \overline{B_k}

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = [2, 1.5, 3]
            >>> B = [1, -1, 2]
            >>> fdot(A, B)
            mpf('6.5')
            >>> list(zip(A, B))
            [(2, 1), (1.5, -1), (3, 2)]
            >>> fdot(_)
            mpf('6.5')
            >>> A = [2, 1.5, 3j]
            >>> B = [1+j, 3, -1-j]
            >>> fdot(A, B)
            mpc(real='9.5', imag='-1.0')
            >>> fdot(A, B, conjugate=True)
            mpc(real='3.5', imag='-5.0')

        """
        if B is not None:
            A = zip(A, B)
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        hasattr_ = hasattr
        types = (ctx.mpf, ctx.mpc)
        for a, b in A:
            if type(a) not in types: a = ctx.convert(a)
            if type(b) not in types: b = ctx.convert(b)
            a_real = hasattr_(a, "_mpf_")
            b_real = hasattr_(b, "_mpf_")
            if a_real and b_real:
                real.append(mpf_mul(a._mpf_, b._mpf_))
                continue
            a_complex = hasattr_(a, "_mpc_")
            b_complex = hasattr_(b, "_mpc_")
            if a_real and b_complex:
                aval = a._mpf_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(aval, bre))
                imag.append(mpf_mul(aval, bim))
            elif b_real and a_complex:
                are, aim = a._mpc_
                bval = b._mpf_
                real.append(mpf_mul(are, bval))
                imag.append(mpf_mul(aim, bval))
            elif a_complex and b_complex:
                #re, im = mpc_mul(a._mpc_, b._mpc_, prec+20)
                are, aim = a._mpc_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(are, bre))
                real.append(mpf_neg(mpf_mul(aim, bim)))
                imag.append(mpf_mul(are, bim))
                imag.append(mpf_mul(aim, bre))
            else:
                raise NotImplementedError
        s = mpf_sum(real, prec, rnd)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def _wrap_libmp_function(ctx, mpf_f, mpc_f=None, mpi_f=None, doc="<no doc>"):
        """
        Given a low-level mpf_ function, and optionally similar functions
        for mpc_ and mpi_, defines the function as a context method.

        It is assumed that the return type is the same as that of
        the input; the exception is that propagation from mpf to mpc is possible
        by raising ComplexResult.

        """
        def f(x, **kwargs):
            if type(x) not in ctx.types:
                x = ctx.convert(x)
            prec, rounding = ctx._prec_rounding
            if kwargs:
                prec = kwargs.get('prec', prec)
                if 'dps' in kwargs:
                    prec = dps_to_prec(kwargs['dps'])
                rounding = kwargs.get('rounding', rounding)
            if hasattr(x, '_mpf_'):
                try:
                    return ctx.make_mpf(mpf_f(x._mpf_, prec, rounding))
                except ComplexResult:
                    # Handle propagation to complex
                    if ctx.trap_complex:
                        raise
                    return ctx.make_mpc(mpc_f((x._mpf_, fzero), prec, rounding))
            elif hasattr(x, '_mpc_'):
                return ctx.make_mpc(mpc_f(x._mpc_, prec, rounding))
            raise NotImplementedError("%s of a %s" % (name, type(x)))
        name = mpf_f.__name__[4:]
        f.__doc__ = function_docs.__dict__.get(name, "Computes the %s of x" % doc)
        return f

    # Called by SpecialFunctions.__init__()
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                prec = ctx.prec
                try:
                    ctx.prec += 10
                    retval = f(ctx, *args, **kwargs)
                finally:
                    ctx.prec = prec
                return +retval
        else:
            f_wrapped = f
        f_wrapped.__doc__ = function_docs.__dict__.get(name, f.__doc__)
        setattr(cls, name, f_wrapped)

    def _convert_param(ctx, x):
        if hasattr(x, "_mpc_"):
            v, im = x._mpc_
            if im != fzero:
                return x, 'C'
        elif hasattr(x, "_mpf_"):
            v = x._mpf_
        else:
            if type(x) in int_types:
                return int(x), 'Z'
            p = None
            if isinstance(x, tuple):
                p, q = x
            elif hasattr(x, '_mpq_'):
                p, q = x._mpq_
            elif isinstance(x, basestring) and '/' in x:
                p, q = x.split('/')
                p = int(p)
                q = int(q)
            if p is not None:
                if not p % q:
                    return p // q, 'Z'
                return ctx.mpq(p,q), 'Q'
            x = ctx.convert(x)
            if hasattr(x, "_mpc_"):
                v, im = x._mpc_
                if im != fzero:
                    return x, 'C'
            elif hasattr(x, "_mpf_"):
                v = x._mpf_
            else:
                return x, 'U'
        sign, man, exp, bc = v
        if man:
            if exp >= -4:
                if sign:
                    man = -man
                if exp >= 0:
                    return int(man) << exp, 'Z'
                if exp >= -4:
                    p, q = int(man), (1<<(-exp))
                    return ctx.mpq(p,q), 'Q'
            x = ctx.make_mpf(v)
            return x, 'R'
        elif not exp:
            return 0, 'Z'
        else:
            return x, 'U'

    def _mpf_mag(ctx, x):
        sign, man, exp, bc = x
        if man:
            return exp+bc
        if x == fzero:
            return ctx.ninf
        if x == finf or x == fninf:
            return ctx.inf
        return ctx.nan

    def mag(ctx, x):
        """
        Quick logarithmic magnitude estimate of a number. Returns an
        integer or infinity `m` such that `|x| <= 2^m`. It is not
        guaranteed that `m` is an optimal bound, but it will never
        be too large by more than 2 (and probably not more than 1).

        **Examples**

            >>> from mpmath import *
            >>> mp.pretty = True
            >>> mag(10), mag(10.0), mag(mpf(10)), int(ceil(log(10,2)))
            (4, 4, 4, 4)
            >>> mag(10j), mag(10+10j)
            (4, 5)
            >>> mag(0.01), int(ceil(log(0.01,2)))
            (-6, -6)
            >>> mag(0), mag(inf), mag(-inf), mag(nan)
            (-inf, +inf, +inf, nan)

        """
        if hasattr(x, "_mpf_"):
            return ctx._mpf_mag(x._mpf_)
        elif hasattr(x, "_mpc_"):
            r, i = x._mpc_
            if r == fzero:
                return ctx._mpf_mag(i)
            if i == fzero:
                return ctx._mpf_mag(r)
            return 1+max(ctx._mpf_mag(r), ctx._mpf_mag(i))
        elif isinstance(x, int_types):
            if x:
                return bitcount(abs(x))
            return ctx.ninf
        elif isinstance(x, rational.mpq):
            p, q = x._mpq_
            if p:
                return 1 + bitcount(abs(p)) - bitcount(q)
            return ctx.ninf
        else:
            x = ctx.convert(x)
            if hasattr(x, "_mpf_") or hasattr(x, "_mpc_"):
                return ctx.mag(x)
            else:
                raise TypeError("requires an mpf/mpc")


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Complex.register(_mpc)
    numbers.Real.register(_mpf)
except ImportError:
    pass
<|endoftext|>
def monitor(f, input='print', output='print'):
    """
    Returns a wrapped copy of *f* that monitors evaluation by calling
    *input* with every input (*args*, *kwargs*) passed to *f* and
    *output* with every value returned from *f*. The default action
    (specify using the special string value ``'print'``) is to print
    inputs and outputs to stdout, along with the total evaluation
    count::

        >>> from mpmath import *
        >>> mp.dps = 5; mp.pretty = False
        >>> diff(monitor(exp), 1)   # diff will eval f(x-h) and f(x+h)
        in  0 (mpf('0.99999999906867742538452148'),) {}
        out 0 mpf('2.7182818259274480055282064')
        in  1 (mpf('1.0000000009313225746154785'),) {}
        out 1 mpf('2.7182818309906424675501024')
        mpf('2.7182808')

    To disable either the input or the output handler, you may
    pass *None* as argument.

    Custom input and output handlers may be used e.g. to store
    results for later analysis::

        >>> mp.dps = 15
        >>> input = []
        >>> output = []
        >>> findroot(monitor(sin, input.append, output.append), 3.0)
        mpf('3.1415926535897932')
        >>> len(input)  # Count number of evaluations
        9
        >>> print(input[3]); print(output[3])
        ((mpf('3.1415076583334066'),), {})
        8.49952562843408e-5
        >>> print(input[4]); print(output[4])
        ((mpf('3.1415928201669122'),), {})
        -1.66577118985331e-7

    """
    if not input:
        input = lambda v: None
    elif input == 'print':
        incount = [0]
        def input(value):
            args, kwargs = value
            print("in  %s %r %r" % (incount[0], args, kwargs))
            incount[0] += 1
    if not output:
        output = lambda v: None
    elif output == 'print':
        outcount = [0]
        def output(value):
            print("out %s %r" % (outcount[0], value))
            outcount[0] += 1
    def f_monitored(*args, **kwargs):
        input((args, kwargs))
        v = f(*args, **kwargs)
        output(v)
        return v
    return f_monitored

def timing(f, *args, **kwargs):
    """
    Returns time elapsed for evaluating ``f()``. Optionally arguments
    may be passed to time the execution of ``f(*args, **kwargs)``.

    If the first call is very quick, ``f`` is called
    repeatedly and the best time is returned.
    """
    once = kwargs.get('once')
    if 'once' in kwargs:
        del kwargs['once']
    if args or kwargs:
        if len(args) == 1 and not kwargs:
            arg = args[0]
            g = lambda: f(arg)
        else:
            g = lambda: f(*args, **kwargs)
    else:
        g = f
    from timeit import default_timer as clock
    t1=clock(); v=g(); t2=clock(); t=t2-t1
    if t > 0.05 or once:
        return t
    for i in range(3):
        t1=clock();
        # Evaluate multiple times because the timer function
        # has a significant overhead
        g();g();g();g();g();g();g();g();g();g()
        t2=clock()
        t=min(t,(t2-t1)/10)
    return t
<|endoftext|>import operator
import sys
from .libmp import int_types, mpf_hash, bitcount, from_man_exp, HASH_MODULUS

new = object.__new__

def create_reduced(p, q, _cache={}):
    key = p, q
    if key in _cache:
        return _cache[key]
    x, y = p, q
    while y:
        x, y = y, x % y
    if x != 1:
        p //= x
        q //= x
    v = new(mpq)
    v._mpq_ = p, q
    # Speedup integers, half-integers and other small fractions
    if q <= 4 and abs(key[0]) < 100:
        _cache[key] = v
    return v

class mpq(object):
    """
    Exact rational type, currently only intended for internal use.
    """

    __slots__ = ["_mpq_"]

    def __new__(cls, p, q=1):
        if type(p) is tuple:
            p, q = p
        elif hasattr(p, '_mpq_'):
            p, q = p._mpq_
        return create_reduced(p, q)

    def __repr__(s):
        return "mpq(%s,%s)" % s._mpq_

    def __str__(s):
        return "(%s/%s)" % s._mpq_

    def __int__(s):
        a, b = s._mpq_
        return a // b

    def __nonzero__(s):
        return bool(s._mpq_[0])

    __bool__ = __nonzero__

    def __hash__(s):
        a, b = s._mpq_
        if sys.version_info >= (3, 2):
            inverse = pow(b, HASH_MODULUS-2, HASH_MODULUS)
            if not inverse:
                h = sys.hash_info.inf
            else:
                h = (abs(a) * inverse) % HASH_MODULUS
            if a < 0: h = -h
            if h == -1: h = -2
            return h
        else:
            if b == 1:
                return hash(a)
            # Power of two: mpf compatible hash
            if not (b & (b-1)):
                return mpf_hash(from_man_exp(a, 1-bitcount(b)))
            return hash((a,b))

    def __eq__(s, t):
        ttype = type(t)
        if ttype is mpq:
            return s._mpq_ == t._mpq_
        if ttype in int_types:
            a, b = s._mpq_
            if b != 1:
                return False
            return a == t
        return NotImplemented

    def __ne__(s, t):
        ttype = type(t)
        if ttype is mpq:
            return s._mpq_ != t._mpq_
        if ttype in int_types:
            a, b = s._mpq_
            if b != 1:
                return True
            return a != t
        return NotImplemented

    def _cmp(s, t, op):
        ttype = type(t)
        if ttype in int_types:
            a, b = s._mpq_
            return op(a, t*b)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return op(a*d, b*c)
        return NotImplementedError

    def __lt__(s, t): return s._cmp(t, operator.lt)
    def __le__(s, t): return s._cmp(t, operator.le)
    def __gt__(s, t): return s._cmp(t, operator.gt)
    def __ge__(s, t): return s._cmp(t, operator.ge)

    def __abs__(s):
        a, b = s._mpq_
        if a >= 0:
            return s
        v = new(mpq)
        v._mpq_ = -a, b
        return v

    def __neg__(s):
        a, b = s._mpq_
        v = new(mpq)
        v._mpq_ = -a, b
        return v

    def __pos__(s):
        return s

    def __add__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a*d+b*c, b*d)
        if ttype in int_types:
            a, b = s._mpq_
            v = new(mpq)
            v._mpq_ = a+b*t, b
            return v
        return NotImplemented

    __radd__ = __add__

    def __sub__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a*d-b*c, b*d)
        if ttype in int_types:
            a, b = s._mpq_
            v = new(mpq)
            v._mpq_ = a-b*t, b
            return v
        return NotImplemented

    def __rsub__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(b*c-a*d, b*d)
        if ttype in int_types:
            a, b = s._mpq_
            v = new(mpq)
            v._mpq_ = b*t-a, b
            return v
        return NotImplemented

    def __mul__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a*c, b*d)
        if ttype in int_types:
            a, b = s._mpq_
            return create_reduced(a*t, b)
        return NotImplemented

    __rmul__ = __mul__

    def __div__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a*d, b*c)
        if ttype in int_types:
            a, b = s._mpq_
            return create_reduced(a, b*t)
        return NotImplemented

    def __rdiv__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(b*c, a*d)
        if ttype in int_types:
            a, b = s._mpq_
            return create_reduced(b*t, a)
        return NotImplemented

    def __pow__(s, t):
        ttype = type(t)
        if ttype in int_types:
            a, b = s._mpq_
            if t:
                if t < 0:
                    a, b, t = b, a, -t
                v = new(mpq)
                v._mpq_ = a**t, b**t
                return v
            raise ZeroDivisionError
        return NotImplemented


mpq_1 = mpq((1,1))
mpq_0 = mpq((0,1))
mpq_1_2 = mpq((1,2))
mpq_3_2 = mpq((3,2))
mpq_1_4 = mpq((1,4))
mpq_1_16 = mpq((1,16))
mpq_3_16 = mpq((3,16))
mpq_5_2 = mpq((5,2))
mpq_3_4 = mpq((3,4))
mpq_7_4 = mpq((7,4))
mpq_5_4 = mpq((5,4))


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Rational.register(mpq)
except ImportError:
    pass
<|endoftext|># contributed to mpmath by Kristopher L. Kuhlman, February 2017
# contributed to mpmath by Guillermo Navas-Palencia, February 2022

class InverseLaplaceTransform(object):
    r"""
    Inverse Laplace transform methods are implemented using this
    class, in order to simplify the code and provide a common
    infrastructure.

    Implement a custom inverse Laplace transform algorithm by
    subclassing :class:`InverseLaplaceTransform` and implementing the
    appropriate methods. The subclass can then be used by
    :func:`~mpmath.invertlaplace` by passing it as the *method*
    argument.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def calc_laplace_parameter(self, t, **kwargs):
        r"""
        Determine the vector of Laplace parameter values needed for an
        algorithm, this will depend on the choice of algorithm (de
        Hoog is default), the algorithm-specific parameters passed (or
        default ones), and desired time.
        """
        raise NotImplementedError

    def calc_time_domain_solution(self, fp):
        r"""
        Compute the time domain solution, after computing the
        Laplace-space function evaluations at the abscissa required
        for the algorithm. Abscissa computed for one algorithm are
        typically not useful for another algorithm.
        """
        raise NotImplementedError


class FixedTalbot(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""The "fixed" Talbot method deforms the Bromwich contour towards
        `-\infty` in the shape of a parabola. Traditionally the Talbot
        algorithm has adjustable parameters, but the "fixed" version
        does not. The `r` parameter could be passed in as a parameter,
        if you want to override the default given by (Abate & Valko,
        2004).

        The Laplace parameter is sampled along a parabola opening
        along the negative imaginary axis, with the base of the
        parabola along the real axis at
        `p=\frac{r}{t_\mathrm{max}}`. As the number of terms used in
        the approximation (degree) grows, the abscissa required for
        function evaluation tend towards `-\infty`, requiring high
        precision to prevent overflow.  If any poles, branch cuts or
        other singularities exist such that the deformed Bromwich
        contour lies to the left of the singularity, the method will
        fail.

        **Optional arguments**

        :class:`~mpmath.calculus.inverselaplace.FixedTalbot.calc_laplace_parameter`
        recognizes the following keywords

        *tmax*
            maximum time associated with vector of times
            (typically just the time requested)
        *degree*
            integer order of approximation (M = number of terms)
        *r*
            abscissa for `p_0` (otherwise computed using rule
            of thumb `2M/5`)

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        .. math ::

            p_0=\frac{r}{t}

        .. math ::

            p_i=\frac{i r \pi}{Mt_\mathrm{max}}\left[\cot\left(
            \frac{i\pi}{M}\right) + j \right] \qquad 1\le i <M

        where `j=\sqrt{-1}`, `r=2M/5`, and `t_\mathrm{max}` is the
        maximum specified time.

        """

        # required
        # ------------------------------
        # time of desired approximation
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------
        # maximum time desired (used for scaling) default is requested
        # time.
        self.tmax = self.ctx.convert(kwargs.get('tmax', self.t))

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = self.degree
        else:
            self.dps_goal = int(1.72*self.ctx.dps)
            self.degree = max(12, int(1.38*self.dps_goal))

        M = self.degree

        # this is adjusting the dps of the calling context hopefully
        # the caller doesn't monkey around with it between calling
        # this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        # Abate & Valko rule of thumb for r parameter
        self.r = kwargs.get('r', self.ctx.fraction(2, 5)*M)

        self.theta = self.ctx.linspace(0.0, self.ctx.pi, M+1)

        self.cot_theta = self.ctx.matrix(M, 1)
        self.cot_theta[0] = 0  # not used

        # all but time-dependent part of p
        self.delta = self.ctx.matrix(M, 1)
        self.delta[0] = self.r

        for i in range(1, M):
            self.cot_theta[i] = self.ctx.cot(self.theta[i])
            self.delta[i] = self.r*self.theta[i]*(self.cot_theta[i] + 1j)

        self.p = self.ctx.matrix(M, 1)
        self.p = self.delta/self.tmax

        # NB: p is complex (mpc)

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""The fixed Talbot time-domain solution is computed from the
        Laplace-space function evaluations using

        .. math ::

            f(t,M)=\frac{2}{5t}\sum_{k=0}^{M-1}\Re \left[
            \gamma_k \bar{f}(p_k)\right]

        where

        .. math ::

            \gamma_0 = \frac{1}{2}e^{r}\bar{f}(p_0)

        .. math ::

            \gamma_k = e^{tp_k}\left\lbrace 1 + \frac{jk\pi}{M}\left[1 +
            \cot \left( \frac{k \pi}{M} \right)^2 \right] - j\cot\left(
            \frac{k \pi}{M}\right)\right \rbrace \qquad 1\le k<M.

        Again, `j=\sqrt{-1}`.

        Before calling this function, call
        :class:`~mpmath.calculus.inverselaplace.FixedTalbot.calc_laplace_parameter`
        to set the parameters and compute the required coefficients.

        **References**

        1. Abate, J., P. Valko (2004). Multi-precision Laplace
           transform inversion. *International Journal for Numerical
           Methods in Engineering* 60:979-993,
           http://dx.doi.org/10.1002/nme.995
        2. Talbot, A. (1979). The accurate numerical inversion of
           Laplace transforms. *IMA Journal of Applied Mathematics*
           23(1):97, http://dx.doi.org/10.1093/imamat/23.1.97
        """

        # required
        # ------------------------------
        self.t = self.ctx.convert(t)

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter(), so is already a list or matrix of
        # mpmath 'mpc' types

        # these were computed in previous call to
        # calc_laplace_parameter()
        theta = self.theta
        delta = self.delta
        M = self.degree
        p = self.p
        r = self.r

        ans = self.ctx.matrix(M, 1)
        ans[0] = self.ctx.exp(delta[0])*fp[0]/2

        for i in range(1, M):
            ans[i] = self.ctx.exp(delta[i])*fp[i]*(
                1 + 1j*theta[i]*(1 + self.cot_theta[i]**2) -
                1j*self.cot_theta[i])

        result = self.ctx.fraction(2, 5)*self.ctx.fsum(ans)/self.t

        # setting dps back to value when calc_laplace_parameter was
        # called, unless flag is set.
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        return result.real


# ****************************************

class Stehfest(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""
        The Gaver-Stehfest method is a discrete approximation of the
        Widder-Post inversion algorithm, rather than a direct
        approximation of the Bromwich contour integral.

        The method abscissa along the real axis, and therefore has
        issues inverting oscillatory functions (which have poles in
        pairs away from the real axis).

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        .. math ::

            p_k = \frac{k \log 2}{t} \qquad 1 \le k \le M
        """

        # required
        # ------------------------------
        # time of desired approximation
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.38*self.degree)
        else:
            self.dps_goal = int(2.93*self.ctx.dps)
            self.degree = max(16, self.dps_goal)

        # _coeff routine requires even degree
        if self.degree % 2 > 0:
            self.degree += 1

        M = self.degree

        # this is adjusting the dps of the calling context
        # hopefully the caller doesn't monkey around with it
        # between calling this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        self.V = self._coeff()
        self.p = self.ctx.matrix(self.ctx.arange(1, M+1))*self.ctx.ln2/self.t

        # NB: p is real (mpf)

    def _coeff(self):
        r"""Salzer summation weights (aka, "Stehfest coefficients")
        only depend on the approximation order (M) and the precision"""

        M = self.degree
        M2 = int(M/2)  # checked earlier that M is even

        V = self.ctx.matrix(M, 1)

        # Salzer summation weights
        # get very large in magnitude and oscillate in sign,
        # if the precision is not high enough, there will be
        # catastrophic cancellation
        for k in range(1, M+1):
            z = self.ctx.matrix(min(k, M2)+1, 1)
            for j in range(int((k+1)/2), min(k, M2)+1):
                z[j] = (self.ctx.power(j, M2)*self.ctx.fac(2*j)/
                        (self.ctx.fac(M2-j)*self.ctx.fac(j)*
                         self.ctx.fac(j-1)*self.ctx.fac(k-j)*
                         self.ctx.fac(2*j-k)))
            V[k-1] = self.ctx.power(-1, k+M2)*self.ctx.fsum(z)

        return V

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""Compute time-domain Stehfest algorithm solution.

        .. math ::

            f(t,M) = \frac{\log 2}{t} \sum_{k=1}^{M} V_k \bar{f}\left(
            p_k \right)

        where

        .. math ::

            V_k = (-1)^{k + N/2} \sum^{\min(k,N/2)}_{i=\lfloor(k+1)/2 \rfloor}
            \frac{i^{\frac{N}{2}}(2i)!}{\left(\frac{N}{2}-i \right)! \, i! \,
            \left(i-1 \right)! \, \left(k-i\right)! \, \left(2i-k \right)!}

        As the degree increases, the abscissa (`p_k`) only increase
        linearly towards `\infty`, but the Stehfest coefficients
        (`V_k`) alternate in sign and increase rapidly in sign,
        requiring high precision to prevent overflow or loss of
        significance when evaluating the sum.

        **References**

        1. Widder, D. (1941). *The Laplace Transform*. Princeton.
        2. Stehfest, H. (1970). Algorithm 368: numerical inversion of
           Laplace transforms. *Communications of the ACM* 13(1):47-49,
           http://dx.doi.org/10.1145/361953.361969

        """

        # required
        self.t = self.ctx.convert(t)

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter(), so is already
        # a list or matrix of mpmath 'mpf' types

        result = self.ctx.fdot(self.V, fp)*self.ctx.ln2/self.t

        # setting dps back to value when calc_laplace_parameter was called
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        # ignore any small imaginary part
        return result.real


# ****************************************

class deHoog(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""the de Hoog, Knight & Stokes algorithm is an
        accelerated form of the Fourier series numerical
        inverse Laplace transform algorithms.

        .. math ::

            p_k = \gamma + \frac{jk}{T} \qquad 0 \le k < 2M+1

        where

        .. math ::

            \gamma = \alpha - \frac{\log \mathrm{tol}}{2T},

        `j=\sqrt{-1}`, `T = 2t_\mathrm{max}` is a scaled time,
        `\alpha=10^{-\mathrm{dps\_goal}}` is the real part of the
        rightmost pole or singularity, which is chosen based on the
        desired accuracy (assuming the rightmost singularity is 0),
        and `\mathrm{tol}=10\alpha` is the desired tolerance, which is
        chosen in relation to `\alpha`.`

        When increasing the degree, the abscissa increase towards
        `j\infty`, but more slowly than the fixed Talbot
        algorithm. The de Hoog et al. algorithm typically does better
        with oscillatory functions of time, and less well-behaved
        functions. The method tends to be slower than the Talbot and
        Stehfest algorithsm, especially so at very high precision
        (e.g., `>500` digits precision).

        """

        # required
        # ------------------------------
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------
        self.tmax = kwargs.get('tmax', self.t)

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.38*self.degree)
        else:
            self.dps_goal = int(self.ctx.dps*1.36)
            self.degree = max(10, self.dps_goal)

        # 2*M+1 terms in approximation
        M = self.degree

        # adjust alpha component of abscissa of convergence for higher
        # precision
        tmp = self.ctx.power(10.0, -self.dps_goal)
        self.alpha = self.ctx.convert(kwargs.get('alpha', tmp))

        # desired tolerance (here simply related to alpha)
        self.tol = self.ctx.convert(kwargs.get('tol', self.alpha*10.0))
        self.np = 2*self.degree+1  # number of terms in approximation

        # this is adjusting the dps of the calling context
        # hopefully the caller doesn't monkey around with it
        # between calling this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        # scaling factor (likely tun-able, but 2 is typical)
        self.scale = kwargs.get('scale', 2)
        self.T = self.ctx.convert(kwargs.get('T', self.scale*self.tmax))

        self.p = self.ctx.matrix(2*M+1, 1)
        self.gamma = self.alpha - self.ctx.log(self.tol)/(self.scale*self.T)
        self.p = (self.gamma + self.ctx.pi*
                  self.ctx.matrix(self.ctx.arange(self.np))/self.T*1j)

        # NB: p is complex (mpc)

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""Calculate time-domain solution for
        de Hoog, Knight & Stokes algorithm.

        The un-accelerated Fourier series approach is:

        .. math ::

            f(t,2M+1) = \frac{e^{\gamma t}}{T} \sum_{k=0}^{2M}{}^{'}
            \Re\left[\bar{f}\left( p_k \right)
            e^{i\pi t/T} \right],

        where the prime on the summation indicates the first term is halved.

        This simplistic approach requires so many function evaluations
        that it is not practical. Non-linear acceleration is
        accomplished via Pade-approximation and an analytic expression
        for the remainder of the continued fraction. See the original
        paper (reference 2 below) a detailed description of the
        numerical approach.

        **References**

        1. Davies, B. (2005). *Integral Transforms and their
           Applications*, Third Edition. Springer.
        2. de Hoog, F., J. Knight, A. Stokes (1982). An improved
           method for numerical inversion of Laplace transforms. *SIAM
           Journal of Scientific and Statistical Computing* 3:357-366,
           http://dx.doi.org/10.1137/0903022

        """

        M = self.degree
        np = self.np
        T = self.T

        self.t = self.ctx.convert(t)

        # would it be useful to try re-using
        # space between e&q and A&B?
        e = self.ctx.zeros(np, M+1)
        q = self.ctx.matrix(2*M, M)
        d = self.ctx.matrix(np, 1)
        A = self.ctx.zeros(np+1, 1)
        B = self.ctx.ones(np+1, 1)

        # initialize Q-D table
        e[:, 0] = 0.0 + 0j
        q[0, 0] = fp[1]/(fp[0]/2)
        for i in range(1, 2*M):
            q[i, 0] = fp[i+1]/fp[i]

        # rhombus rule for filling triangular Q-D table (e & q)
        for r in range(1, M+1):
            # start with e, column 1, 0:2*M-2
            mr = 2*(M-r) + 1
            e[0:mr, r] = q[1:mr+1, r-1] - q[0:mr, r-1] + e[1:mr+1, r-1]
            if not r == M:
                rq = r+1
                mr = 2*(M-rq)+1 + 2
                for i in range(mr):
                    q[i, rq-1] = q[i+1, rq-2]*e[i+1, rq-1]/e[i, rq-1]

        # build up continued fraction coefficients (d)
        d[0] = fp[0]/2
        for r in range(1, M+1):
            d[2*r-1] = -q[0, r-1]  # even terms
            d[2*r]   = -e[0, r]    # odd terms

        # seed A and B for recurrence
        A[0] = 0.0 + 0.0j
        A[1] = d[0]
        B[0:2] = 1.0 + 0.0j

        # base of the power series
        z = self.ctx.expjpi(self.t/T)  # i*pi is already in fcn

        # coefficients of Pade approximation (A & B)
        # using recurrence for all but last term
        for i in range(1, 2*M):
            A[i+1] = A[i] + d[i]*A[i-1]*z
            B[i+1] = B[i] + d[i]*B[i-1]*z

        # "improved remainder" to continued fraction
        brem = (1 + (d[2*M-1] - d[2*M])*z)/2
        # powm1(x,y) computes x^y - 1 more accurately near zero
        rem = brem*self.ctx.powm1(1 + d[2*M]*z/brem,
                                  self.ctx.fraction(1, 2))

        # last term of recurrence using new remainder
        A[np] = A[2*M] + rem*A[2*M-1]
        B[np] = B[2*M] + rem*B[2*M-1]

        # diagonal Pade approximation
        # F=A/B represents accelerated trapezoid rule
        result = self.ctx.exp(self.gamma*self.t)/T*(A[np]/B[np]).real

        # setting dps back to value when calc_laplace_parameter was called
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        return result


# ****************************************

class Cohen(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""The Cohen algorithm accelerates the convergence of the nearly
        alternating series resulting from the application of the trapezoidal
        rule to the Bromwich contour inversion integral.

        .. math ::

            p_k = \frac{\gamma}{2 t} + \frac{\pi i k}{t} \qquad 0 \le k < M

        where

        .. math ::

            \gamma = \frac{2}{3} (d + \log(10) + \log(2 t)),

        `d = \mathrm{dps\_goal}`, which is chosen based on the desired
        accuracy using the method developed in [1] to improve numerical
        stability. The Cohen algorithm shows robustness similar to the de Hoog
        et al. algorithm, but it is faster than the fixed Talbot algorithm.

        **Optional arguments**

        *degree*
            integer order of the approximation (M = number of terms)
        *alpha*
            abscissa for `p_0` (controls the discretization error)

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        **References**

        1. P. Glasserman, J. Ruiz-Mata (2006). Computing the credit loss
        distribution in the Gaussian copula model: a comparison of methods.
        *Journal of Credit Risk* 2(4):33-66, 10.21314/JCR.2006.057

        """
        self.t = self.ctx.convert(t)

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.5 * self.degree)
        else:
            self.dps_goal = int(self.ctx.dps * 1.74)
            self.degree = max(22, int(1.31 * self.dps_goal))

        M = self.degree + 1

        # this is adjusting the dps of the calling context hopefully
        # the caller doesn't monkey around with it between calling
        # this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        ttwo = 2 * self.t
        tmp = self.ctx.dps * self.ctx.log(10) + self.ctx.log(ttwo)
        tmp = self.ctx.fraction(2, 3) * tmp
        self.alpha = self.ctx.convert(kwargs.get('alpha', tmp))

        # all but time-dependent part of p
        a_t = self.alpha / ttwo
        p_t = self.ctx.pi * 1j / self.t

        self.p = self.ctx.matrix(M, 1)
        self.p[0] = a_t

        for i in range(1, M):
            self.p[i] = a_t + i * p_t

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""Calculate time-domain solution for Cohen algorithm.

        The accelerated nearly alternating series is:

        .. math ::

            f(t, M) = \frac{e^{\gamma / 2}}{t} \left[\frac{1}{2}
            \Re\left(\bar{f}\left(\frac{\gamma}{2t}\right) \right) -
            \sum_{k=0}^{M-1}\frac{c_{M,k}}{d_M}\Re\left(\bar{f}
            \left(\frac{\gamma + 2(k+1) \pi i}{2t}\right)\right)\right],

        where coefficients `\frac{c_{M, k}}{d_M}` are described in [1].

        1. H. Cohen, F. Rodriguez Villegas, D. Zagier (2000). Convergence
        acceleration of alternating series. *Experiment. Math* 9(1):3-12

        """
        self.t = self.ctx.convert(t)

        n = self.degree
        M = n + 1

        A = self.ctx.matrix(M, 1)
        for i in range(M):
            A[i] = fp[i].real

        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = -self.ctx.one
        c = -d
        s = 0

        for k in range(n):
            c = b - c
            s = s + c * A[k + 1]
            b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + self.ctx.one))

        result = self.ctx.exp(self.alpha / 2) / self.t * (A[0] / 2 - s / d)

        # setting dps back to value when calc_laplace_parameter was
        # called, unless flag is set.
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        return result


# ****************************************

class LaplaceTransformInversionMethods(object):
    def __init__(ctx, *args, **kwargs):
        ctx._fixed_talbot = FixedTalbot(ctx)
        ctx._stehfest = Stehfest(ctx)
        ctx._de_hoog = deHoog(ctx)
        ctx._cohen = Cohen(ctx)

    def invertlaplace(ctx, f, t, **kwargs):
        r"""Computes the numerical inverse Laplace transform for a
        Laplace-space function at a given time.  The function being
        evaluated is assumed to be a real-valued function of time.

        The user must supply a Laplace-space function `\bar{f}(p)`,
        and a desired time at which to estimate the time-domain
        solution `f(t)`.

        A few basic examples of Laplace-space functions with known
        inverses (see references [1,2]) :

        .. math ::

            \mathcal{L}\left\lbrace f(t) \right\rbrace=\bar{f}(p)

        .. math ::

            \mathcal{L}^{-1}\left\lbrace \bar{f}(p) \right\rbrace = f(t)

        .. math ::

            \bar{f}(p) = \frac{1}{(p+1)^2}

        .. math ::

            f(t) = t e^{-t}

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> tt = [0.001, 0.01, 0.1, 1, 10]
        >>> fp = lambda p: 1/(p+1)**2
        >>> ft = lambda t: t*exp(-t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='talbot')
        (0.000999000499833375, 8.57923043561212e-20)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='talbot')
        (0.00990049833749168, 3.27007646698047e-19)
        >>> ft(tt[2]),ft(tt[2])-invertlaplace(fp,tt[2],method='talbot')
        (0.090483741803596, -1.75215800052168e-18)
        >>> ft(tt[3]),ft(tt[3])-invertlaplace(fp,tt[3],method='talbot')
        (0.367879441171442, 1.2428864009344e-17)
        >>> ft(tt[4]),ft(tt[4])-invertlaplace(fp,tt[4],method='talbot')
        (0.000453999297624849, 4.04513489306658e-20)

        The methods also work for higher precision:

        >>> mp.dps = 100; mp.pretty = True
        >>> nstr(ft(tt[0]),15),nstr(ft(tt[0])-invertlaplace(fp,tt[0],method='talbot'),15)
        ('0.000999000499833375', '-4.96868310693356e-105')
        >>> nstr(ft(tt[1]),15),nstr(ft(tt[1])-invertlaplace(fp,tt[1],method='talbot'),15)
        ('0.00990049833749168', '1.23032291513122e-104')

        .. math ::

            \bar{f}(p) = \frac{1}{p^2+1}

        .. math ::

            f(t) = \mathrm{J}_0(t)

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: 1/sqrt(p*p + 1)
        >>> ft = lambda t: besselj(0,t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='dehoog')
        (0.999999750000016, -6.09717765032273e-18)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='dehoog')
        (0.99997500015625, -5.61756281076169e-17)

        .. math ::

            \bar{f}(p) = \frac{\log p}{p}

        .. math ::

            f(t) = -\gamma -\log t

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: log(p)/p
        >>> ft = lambda t: -euler-log(t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='stehfest')
        (6.3305396140806, -1.92126634837863e-16)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='stehfest')
        (4.02795452108656, -4.81486093200704e-16)

        **Options**

        :func:`~mpmath.invertlaplace` recognizes the following optional
        keywords valid for all methods:

        *method*
            Chooses numerical inverse Laplace transform algorithm
            (described below).
        *degree*
            Number of terms used in the approximation

        **Algorithms**

        Mpmath implements four numerical inverse Laplace transform
        algorithms, attributed to: Talbot, Stehfest, and de Hoog,
        Knight and Stokes. These can be selected by using
        *method='talbot'*, *method='stehfest'*, *method='dehoog'* or
        *method='cohen'* or by passing the classes *method=FixedTalbot*,
        *method=Stehfest*, *method=deHoog*, or *method=Cohen*. The functions
        :func:`~mpmath.invlaptalbot`, :func:`~mpmath.invlapstehfest`,
        :func:`~mpmath.invlapdehoog`, and :func:`~mpmath.invlapcohen`
        are also available as shortcuts.

        All four algorithms implement a heuristic balance between the
        requested precision and the precision used internally for the
        calculations. This has been tuned for a typical exponentially
        decaying function and precision up to few hundred decimal
        digits.

        The Laplace transform converts the variable time (i.e., along
        a line) into a parameter given by the right half of the
        complex `p`-plane.  Singularities, poles, and branch cuts in
        the complex `p`-plane contain all the information regarding
        the time behavior of the corresponding function. Any numerical
        method must therefore sample `p`-plane "close enough" to the
        singularities to accurately characterize them, while not
        getting too close to have catastrophic cancellation, overflow,
        or underflow issues. Most significantly, if one or more of the
        singularities in the `p`-plane is not on the left side of the
        Bromwich contour, its effects will be left out of the computed
        solution, and the answer will be completely wrong.

        *Talbot*

        The fixed Talbot method is high accuracy and fast, but the
        method can catastrophically fail for certain classes of time-domain
        behavior, including a Heaviside step function for positive
        time (e.g., `H(t-2)`), or some oscillatory behaviors. The
        Talbot method usually has adjustable parameters, but the
        "fixed" variety implemented here does not. This method
        deforms the Bromwich integral contour in the shape of a
        parabola towards `-\infty`, which leads to problems
        when the solution has a decaying exponential in it (e.g., a
        Heaviside step function is equivalent to multiplying by a
        decaying exponential in Laplace space).

        *Stehfest*

        The Stehfest algorithm only uses abscissa along the real axis
        of the complex `p`-plane to estimate the time-domain
        function. Oscillatory time-domain functions have poles away
        from the real axis, so this method does not work well with
        oscillatory functions, especially high-frequency ones. This
        method also depends on summation of terms in a series that
        grows very large, and will have catastrophic cancellation
        during summation if the working precision is too low.

        *de Hoog et al.*

        The de Hoog, Knight, and Stokes method is essentially a
        Fourier-series quadrature-type approximation to the Bromwich
        contour integral, with non-linear series acceleration and an
        analytical expression for the remainder term. This method is
        typically one of the most robust. This method also involves the
        greatest amount of overhead, so it is typically the slowest of the
        four methods at high precision.

        *Cohen*

        The Cohen method is a trapezoidal rule approximation to the Bromwich
        contour integral, with linear acceleration for alternating
        series. This method is as robust as the de Hoog et al method and the
        fastest of the four methods at high precision, and is therefore the
        default method.

        **Singularities**

        All numerical inverse Laplace transform methods have problems
        at large time when the Laplace-space function has poles,
        singularities, or branch cuts to the right of the origin in
        the complex plane. For simple poles in `\bar{f}(p)` at the
        `p`-plane origin, the time function is constant in time (e.g.,
        `\mathcal{L}\left\lbrace 1 \right\rbrace=1/p` has a pole at
        `p=0`). A pole in `\bar{f}(p)` to the left of the origin is a
        decreasing function of time (e.g., `\mathcal{L}\left\lbrace
        e^{-t/2} \right\rbrace=1/(p+1/2)` has a pole at `p=-1/2`), and
        a pole to the right of the origin leads to an increasing
        function in time (e.g., `\mathcal{L}\left\lbrace t e^{t/4}
        \right\rbrace = 1/(p-1/4)^2` has a pole at `p=1/4`).  When
        singularities occur off the real `p` axis, the time-domain
        function is oscillatory. For example `\mathcal{L}\left\lbrace
        \mathrm{J}_0(t) \right\rbrace=1/\sqrt{p^2+1}` has a branch cut
        starting at `p=j=\sqrt{-1}` and is a decaying oscillatory
        function, This range of behaviors is illustrated in Duffy [3]
        Figure 4.10.4, p. 228.

        In general as `p \rightarrow \infty` `t \rightarrow 0` and
        vice-versa. All numerical inverse Laplace transform methods
        require their abscissa to shift closer to the origin for
        larger times. If the abscissa shift left of the rightmost
        singularity in the Laplace domain, the answer will be
        completely wrong (the effect of singularities to the right of
        the Bromwich contour are not included in the results).

        For example, the following exponentially growing function has
        a pole at `p=3`:

        .. math ::

            \bar{f}(p)=\frac{1}{p^2-9}

        .. math ::

            f(t)=\frac{1}{3}\sinh 3t

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: 1/(p*p-9)
        >>> ft = lambda t: sinh(3*t)/3
        >>> tt = [0.01,0.1,1.0,10.0]
        >>> ft(tt[0]),invertlaplace(fp,tt[0],method='talbot')
        (0.0100015000675014, 0.0100015000675014)
        >>> ft(tt[1]),invertlaplace(fp,tt[1],method='talbot')
        (0.101506764482381, 0.101506764482381)
        >>> ft(tt[2]),invertlaplace(fp,tt[2],method='talbot')
        (3.33929164246997, 3.33929164246997)
        >>> ft(tt[3]),invertlaplace(fp,tt[3],method='talbot')
        (1781079096920.74, -1.61331069624091e-14)

        **References**

        1. [DLMF]_ section 1.14 (http://dlmf.nist.gov/1.14T4)
        2. Cohen, A.M. (2007). Numerical Methods for Laplace Transform
           Inversion, Springer.
        3. Duffy, D.G. (1998). Advanced Engineering Mathematics, CRC Press.

        **Numerical Inverse Laplace Transform Reviews**

        1. Bellman, R., R.E. Kalaba, J.A. Lockett (1966). *Numerical
           inversion of the Laplace transform: Applications to Biology,
           Economics, Engineering, and Physics*. Elsevier.
        2. Davies, B., B. Martin (1979). Numerical inversion of the
           Laplace transform: a survey and comparison of methods. *Journal
           of Computational Physics* 33:1-32,
           http://dx.doi.org/10.1016/0021-9991(79)90025-1
        3. Duffy, D.G. (1993). On the numerical inversion of Laplace
           transforms: Comparison of three new methods on characteristic
           problems from applications. *ACM Transactions on Mathematical
           Software* 19(3):333-359, http://dx.doi.org/10.1145/155743.155788
        4. Kuhlman, K.L., (2013). Review of Inverse Laplace Transform
           Algorithms for Laplace-Space Numerical Approaches, *Numerical
           Algorithms*, 63(2):339-355.
           http://dx.doi.org/10.1007/s11075-012-9625-3

        """

        rule = kwargs.get('method', 'cohen')
        if type(rule) is str:
            lrule = rule.lower()
            if lrule == 'talbot':
                rule = ctx._fixed_talbot
            elif lrule == 'stehfest':
                rule = ctx._stehfest
            elif lrule == 'dehoog':
                rule = ctx._de_hoog
            elif rule == 'cohen':
                rule = ctx._cohen
            else:
                raise ValueError("unknown invlap algorithm: %s" % rule)
        else:
            rule = rule(ctx)

        # determine the vector of Laplace-space parameter
        # needed for the requested method and desired time
        rule.calc_laplace_parameter(t, **kwargs)

        # compute the Laplace-space function evalutations
        # at the required abscissa.
        fp = [f(p) for p in rule.p]

        # compute the time-domain solution from the
        # Laplace-space function evaluations
        return rule.calc_time_domain_solution(fp, t)

    # shortcuts for the above function for specific methods
    def invlaptalbot(ctx, *args, **kwargs):
        kwargs['method'] = 'talbot'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapstehfest(ctx, *args, **kwargs):
        kwargs['method'] = 'stehfest'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapdehoog(ctx, *args, **kwargs):
        kwargs['method'] = 'dehoog'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapcohen(ctx, *args, **kwargs):
        kwargs['method'] = 'cohen'
        return ctx.invertlaplace(*args, **kwargs)


# ****************************************

if __name__ == '__main__':
    import doctest
    doctest.testmod()
<|endoftext|>try:
    from itertools import izip
except ImportError:
    izip = zip

from ..libmp.backend import xrange
from .calculus import defun

try:
    next = next
except NameError:
    next = lambda _: _.next()

@defun
def richardson(ctx, seq):
    r"""
    Given a list ``seq`` of the first `N` elements of a slowly convergent
    infinite sequence, :func:`~mpmath.richardson` computes the `N`-term
    Richardson extrapolate for the limit.

    :func:`~mpmath.richardson` returns `(v, c)` where `v` is the estimated
    limit and `c` is the magnitude of the largest weight used during the
    computation. The weight provides an estimate of the precision
    lost to cancellation. Due to cancellation effects, the sequence must
    be typically be computed at a much higher precision than the target
    accuracy of the extrapolation.

    **Applicability and issues**

    The `N`-step Richardson extrapolation algorithm used by
    :func:`~mpmath.richardson` is described in [1].

    Richardson extrapolation only works for a specific type of sequence,
    namely one converging like partial sums of
    `P(1)/Q(1) + P(2)/Q(2) + \ldots` where `P` and `Q` are polynomials.
    When the sequence does not convergence at such a rate
    :func:`~mpmath.richardson` generally produces garbage.

    Richardson extrapolation has the advantage of being fast: the `N`-term
    extrapolate requires only `O(N)` arithmetic operations, and usually
    produces an estimate that is accurate to `O(N)` digits. Contrast with
    the Shanks transformation (see :func:`~mpmath.shanks`), which requires
    `O(N^2)` operations.

    :func:`~mpmath.richardson` is unable to produce an estimate for the
    approximation error. One way to estimate the error is to perform
    two extrapolations with slightly different `N` and comparing the
    results.

    Richardson extrapolation does not work for oscillating sequences.
    As a simple workaround, :func:`~mpmath.richardson` detects if the last
    three elements do not differ monotonically, and in that case
    applies extrapolation only to the even-index elements.

    **Example**

    Applying Richardson extrapolation to the Leibniz series for `\pi`::

        >>> from mpmath import *
        >>> mp.dps = 30; mp.pretty = True
        >>> S = [4*sum(mpf(-1)**n/(2*n+1) for n in range(m))
        ...     for m in range(1,30)]
        >>> v, c = richardson(S[:10])
        >>> v
        3.2126984126984126984126984127
        >>> nprint([v-pi, c])
        [0.0711058, 2.0]

        >>> v, c = richardson(S[:30])
        >>> v
        3.14159265468624052829954206226
        >>> nprint([v-pi, c])
        [1.09645e-9, 20833.3]

    **References**

    1. [BenderOrszag]_ pp. 375-376

    """
    if len(seq) < 3:
        raise ValueError("seq should be of minimum length 3")
    if ctx.sign(seq[-1]-seq[-2]) != ctx.sign(seq[-2]-seq[-3]):
        seq = seq[::2]
    N = len(seq)//2-1
    s = ctx.zero
    # The general weight is c[k] = (N+k)**N * (-1)**(k+N) / k! / (N-k)!
    # To avoid repeated factorials, we simplify the quotient
    # of successive weights to obtain a recurrence relation
    c = (-1)**N * N**N / ctx.mpf(ctx._ifac(N))
    maxc = 1
    for k in xrange(N+1):
        s += c * seq[N+k]
        maxc = max(abs(c), maxc)
        c *= (k-N)*ctx.mpf(k+N+1)**N
        c /= ((1+k)*ctx.mpf(k+N)**N)
    return s, maxc

@defun
def shanks(ctx, seq, table=None, randomized=False):
    r"""
    Given a list ``seq`` of the first `N` elements of a slowly
    convergent infinite sequence `(A_k)`, :func:`~mpmath.shanks` computes the iterated
    Shanks transformation `S(A), S(S(A)), \ldots, S^{N/2}(A)`. The Shanks
    transformation often provides strong convergence acceleration,
    especially if the sequence is oscillating.

    The iterated Shanks transformation is computed using the Wynn
    epsilon algorithm (see [1]). :func:`~mpmath.shanks` returns the full
    epsilon table generated by Wynn's algorithm, which can be read
    off as follows:

    * The table is a list of lists forming a lower triangular matrix,
      where higher row and column indices correspond to more accurate
      values.
    * The columns with even index hold dummy entries (required for the
      computation) and the columns with odd index hold the actual
      extrapolates.
    * The last element in the last row is typically the most
      accurate estimate of the limit.
    * The difference to the third last element in the last row
      provides an estimate of the approximation error.
    * The magnitude of the second last element provides an estimate
      of the numerical accuracy lost to cancellation.

    For convenience, so the extrapolation is stopped at an odd index
    so that ``shanks(seq)[-1][-1]`` always gives an estimate of the
    limit.

    Optionally, an existing table can be passed to :func:`~mpmath.shanks`.
    This can be used to efficiently extend a previous computation after
    new elements have been appended to the sequence. The table will
    then be updated in-place.

    **The Shanks transformation**

    The Shanks transformation is defined as follows (see [2]): given
    the input sequence `(A_0, A_1, \ldots)`, the transformed sequence is
    given by

    .. math ::

        S(A_k) = \frac{A_{k+1}A_{k-1}-A_k^2}{A_{k+1}+A_{k-1}-2 A_k}

    The Shanks transformation gives the exact limit `A_{\infty}` in a
    single step if `A_k = A + a q^k`. Note in particular that it
    extrapolates the exact sum of a geometric series in a single step.

    Applying the Shanks transformation once often improves convergence
    substantially for an arbitrary sequence, but the optimal effect is
    obtained by applying it iteratively:
    `S(S(A_k)), S(S(S(A_k))), \ldots`.

    Wynn's epsilon algorithm provides an efficient way to generate
    the table of iterated Shanks transformations. It reduces the
    computation of each element to essentially a single division, at
    the cost of requiring dummy elements in the table. See [1] for
    details.

    **Precision issues**

    Due to cancellation effects, the sequence must be typically be
    computed at a much higher precision than the target accuracy
    of the extrapolation.

    If the Shanks transformation converges to the exact limit (such
    as if the sequence is a geometric series), then a division by
    zero occurs. By default, :func:`~mpmath.shanks` handles this case by
    terminating the iteration and returning the table it has
    generated so far. With *randomized=True*, it will instead
    replace the zero by a pseudorandom number close to zero.
    (TODO: find a better solution to this problem.)

    **Examples**

    We illustrate by applying Shanks transformation to the Leibniz
    series for `\pi`::

        >>> from mpmath import *
        >>> mp.dps = 50
        >>> S = [4*sum(mpf(-1)**n/(2*n+1) for n in range(m))
        ...     for m in range(1,30)]
        >>>
        >>> T = shanks(S[:7])
        >>> for row in T:
        ...     nprint(row)
        ...
        [-0.75]
        [1.25, 3.16667]
        [-1.75, 3.13333, -28.75]
        [2.25, 3.14524, 82.25, 3.14234]
        [-2.75, 3.13968, -177.75, 3.14139, -969.937]
        [3.25, 3.14271, 327.25, 3.14166, 3515.06, 3.14161]

    The extrapolated accuracy is about 4 digits, and about 4 digits
    may have been lost due to cancellation::

        >>> L = T[-1]
        >>> nprint([abs(L[-1] - pi), abs(L[-1] - L[-3]), abs(L[-2])])
        [2.22532e-5, 4.78309e-5, 3515.06]

    Now we extend the computation::

        >>> T = shanks(S[:25], T)
        >>> L = T[-1]
        >>> nprint([abs(L[-1] - pi), abs(L[-1] - L[-3]), abs(L[-2])])
        [3.75527e-19, 1.48478e-19, 2.96014e+17]

    The value for pi is now accurate to 18 digits. About 18 digits may
    also have been lost to cancellation.

    Here is an example with a geometric series, where the convergence
    is immediate (the sum is exactly 1)::

        >>> mp.dps = 15
        >>> for row in shanks([0.5, 0.75, 0.875, 0.9375, 0.96875]):
        ...     nprint(row)
        [4.0]
        [8.0, 1.0]

    **References**

    1. [GravesMorris]_

    2. [BenderOrszag]_ pp. 368-375

    """
    if len(seq) < 2:
        raise ValueError("seq should be of minimum length 2")
    if table:
        START = len(table)
    else:
        START = 0
        table = []
    STOP = len(seq) - 1
    if STOP & 1:
        STOP -= 1
    one = ctx.one
    eps = +ctx.eps
    if randomized:
        from random import Random
        rnd = Random()
        rnd.seed(START)
    for i in xrange(START, STOP):
        row = []
        for j in xrange(i+1):
            if j == 0:
                a, b = 0, seq[i+1]-seq[i]
            else:
                if j == 1:
                    a = seq[i]
                else:
                    a = table[i-1][j-2]
                b = row[j-1] - table[i-1][j-1]
            if not b:
                if randomized:
                    b = (1 + rnd.getrandbits(10))*eps
                elif i & 1:
                    return table[:-1]
                else:
                    return table
            row.append(a + one/b)
        table.append(row)
    return table


class levin_class:
    # levin: Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
    r"""
    This interface implements Levin's (nonlinear) sequence transformation for
    convergence acceleration and summation of divergent series. It performs
    better than the Shanks/Wynn-epsilon algorithm for logarithmic convergent
    or alternating divergent series.

    Let *A* be the series we want to sum:

    .. math ::

        A = \sum_{k=0}^{\infty} a_k

    Attention: all `a_k` must be non-zero!

    Let `s_n` be the partial sums of this series:

    .. math ::

        s_n = \sum_{k=0}^n a_k.

    **Methods**

    Calling ``levin`` returns an object with the following methods.

    ``update(...)`` works with the list of individual terms `a_k` of *A*, and
    ``update_step(...)`` works with the list of partial sums `s_k` of *A*:

    .. code ::

        v, e = ...update([a_0, a_1,..., a_k])
        v, e = ...update_psum([s_0, s_1,..., s_k])

    ``step(...)`` works with the individual terms `a_k` and ``step_psum(...)``
    works with the partial sums `s_k`:

    .. code ::

        v, e = ...step(a_k)
        v, e = ...step_psum(s_k)

    *v* is the current estimate for *A*, and *e* is an error estimate which is
    simply the difference between the current estimate and the last estimate.
    One should not mix ``update``, ``update_psum``, ``step`` and ``step_psum``.

    **A word of caution**

    One can only hope for good results (i.e. convergence acceleration or
    resummation) if the `s_n` have some well defind asymptotic behavior for
    large `n` and are not erratic or random. Furthermore one usually needs very
    high working precision because of the numerical cancellation. If the working
    precision is insufficient, levin may produce silently numerical garbage.
    Furthermore even if the Levin-transformation converges, in the general case
    there is no proof that the result is mathematically sound. Only for very
    special classes of problems one can prove that the Levin-transformation
    converges to the expected result (for example Stieltjes-type integrals).
    Furthermore the Levin-transform is quite expensive (i.e. slow) in comparison
    to Shanks/Wynn-epsilon, Richardson & co.
    In summary one can say that the Levin-transformation is powerful but
    unreliable and that it may need a copious amount of working precision.

    The Levin transform has several variants differing in the choice of weights.
    Some variants are better suited for the possible flavours of convergence
    behaviour of *A* than other variants:

    .. code ::

       convergence behaviour   levin-u   levin-t   levin-v   shanks/wynn-epsilon

       logarithmic               +         -         +           -
       linear                    +         +         +           +
       alternating divergent     +         +         +           +

         "+" means the variant is suitable,"-" means the variant is not suitable;
         for comparison the Shanks/Wynn-epsilon transform is listed, too.

    The variant is controlled though the variant keyword (i.e. ``variant="u"``,
    ``variant="t"`` or ``variant="v"``). Overall "u" is probably the best choice.

    Finally it is possible to use the Sidi-S transform instead of the Levin transform
    by using the keyword ``method='sidi'``. The Sidi-S transform works better than the
    Levin transformation for some divergent series (see the examples).

    Parameters:

    .. code ::

       method      "levin" or "sidi" chooses either the Levin or the Sidi-S transformation
       variant     "u","t" or "v" chooses the weight variant.

    The Levin transform is also accessible through the nsum interface.
    ``method="l"`` or ``method="levin"`` select the normal Levin transform while
    ``method="sidi"``
    selects the Sidi-S transform. The variant is in both cases selected through the
    levin_variant keyword. The stepsize in :func:`~mpmath.nsum` must not be chosen too large, otherwise
    it will miss the point where the Levin transform converges resulting in numerical
    overflow/garbage. For highly divergent series a copious amount of working precision
    must be chosen.

    **Examples**

    First we sum the zeta function::

        >>> from mpmath import mp
        >>> mp.prec = 53
        >>> eps = mp.mpf(mp.eps)
        >>> with mp.extraprec(2 * mp.prec): # levin needs a high working precision
        ...     L = mp.levin(method = "levin", variant = "u")
        ...     S, s, n = [], 0, 1
        ...     while 1:
        ...         s += mp.one / (n * n)
        ...         n += 1
        ...         S.append(s)
        ...         v, e = L.update_psum(S)
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - mp.pi ** 2 / 6))
        0.0
        >>> w = mp.nsum(lambda n: 1 / (n*n), [1, mp.inf], method = "levin", levin_variant = "u")
        >>> print(mp.chop(v - w))
        0.0

    Now we sum the zeta function outside its range of convergence
    (attention: This does not work at the negative integers!)::

        >>> eps = mp.mpf(mp.eps)
        >>> with mp.extraprec(2 * mp.prec): # levin needs a high working precision
        ...     L = mp.levin(method = "levin", variant = "v")
        ...     A, n = [], 1
        ...     while 1:
        ...         s = mp.mpf(n) ** (2 + 3j)
        ...         n += 1
        ...         A.append(s)
        ...         v, e = L.update(A)
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - mp.zeta(-2-3j)))
        0.0
        >>> w = mp.nsum(lambda n: n ** (2 + 3j), [1, mp.inf], method = "levin", levin_variant = "v")
        >>> print(mp.chop(v - w))
        0.0

    Now we sum the divergent asymptotic expansion of an integral related to the
    exponential integral (see also [2] p.373). The Sidi-S transform works best here::

        >>> z = mp.mpf(10)
        >>> exact = mp.quad(lambda x: mp.exp(-x)/(1+x/z),[0,mp.inf])
        >>> # exact = z * mp.exp(z) * mp.expint(1,z) # this is the symbolic expression for the integral
        >>> eps = mp.mpf(mp.eps)
        >>> with mp.extraprec(2 * mp.prec): # high working precisions are mandatory for divergent resummation
        ...     L = mp.levin(method = "sidi", variant = "t")
        ...     n = 0
        ...     while 1:
        ...         s = (-1)**n * mp.fac(n) * z ** (-n)
        ...         v, e = L.step(s)
        ...         n += 1
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - exact))
        0.0
        >>> w = mp.nsum(lambda n: (-1) ** n * mp.fac(n) * z ** (-n), [0, mp.inf], method = "sidi", levin_variant = "t")
        >>> print(mp.chop(v - w))
        0.0

    Another highly divergent integral is also summable::

        >>> z = mp.mpf(2)
        >>> eps = mp.mpf(mp.eps)
        >>> exact = mp.quad(lambda x: mp.exp( -x * x / 2 - z * x ** 4), [0,mp.inf]) * 2 / mp.sqrt(2 * mp.pi)
        >>> # exact = mp.exp(mp.one / (32 * z)) * mp.besselk(mp.one / 4, mp.one / (32 * z)) / (4 * mp.sqrt(z * mp.pi)) # this is the symbolic expression for the integral
        >>> with mp.extraprec(7 * mp.prec):  # we need copious amount of precision to sum this highly divergent series
        ...     L = mp.levin(method = "levin", variant = "t")
        ...     n, s = 0, 0
        ...     while 1:
        ...         s += (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n))
        ...         n += 1
        ...         v, e = L.step_psum(s)
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - exact))
        0.0
        >>> w = mp.nsum(lambda n: (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n)),
        ...   [0, mp.inf], method = "levin", levin_variant = "t", workprec = 8*mp.prec, steps = [2] + [1 for x in xrange(1000)])
        >>> print(mp.chop(v - w))
        0.0

    These examples run with 15-20 decimal digits precision. For higher precision the
    working precision must be raised.

    **Examples for nsum**

    Here we calculate Euler's constant as the constant term in the Laurent
    expansion of `\zeta(s)` at `s=1`. This sum converges extremly slowly because of
    the logarithmic convergence behaviour of the Dirichlet series for zeta::

        >>> mp.dps = 30
        >>> z = mp.mpf(10) ** (-10)
        >>> a = mp.nsum(lambda n: n**(-(1+z)), [1, mp.inf], method = "l") - 1 / z
        >>> print(mp.chop(a - mp.euler, tol = 1e-10))
        0.0

    The Sidi-S transform performs excellently for the alternating series of `\log(2)`::

        >>> a = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "sidi")
        >>> print(mp.chop(a - mp.log(2)))
        0.0

    Hypergeometric series can also be summed outside their range of convergence.
    The stepsize in :func:`~mpmath.nsum` must not be chosen too large, otherwise it will miss the
    point where the Levin transform converges resulting in numerical overflow/garbage::

        >>> z = 2 + 1j
        >>> exact = mp.hyp2f1(2 / mp.mpf(3), 4 / mp.mpf(3), 1 / mp.mpf(3), z)
        >>> f = lambda n: mp.rf(2 / mp.mpf(3), n) * mp.rf(4 / mp.mpf(3), n) * z**n / (mp.rf(1 / mp.mpf(3), n) * mp.fac(n))
        >>> v = mp.nsum(f, [0, mp.inf], method = "levin", steps = [10 for x in xrange(1000)])
        >>> print(mp.chop(exact-v))
        0.0

    References:

      [1] E.J. Weniger - "Nonlinear Sequence Transformations for the Acceleration of
          Convergence and the Summation of Divergent Series" arXiv:math/0306302

      [2] A. Sidi - "Pratical Extrapolation Methods"

      [3] H.H.H. Homeier - "Scalar Levin-Type Sequence Transformations" arXiv:math/0005209

    """

    def __init__(self, method = "levin", variant = "u"):
        self.variant = variant
        self.n = 0
        self.a0 = 0
        self.theta = 1
        self.A = []
        self.B = []
        self.last = 0
        self.last_s = False

        if method == "levin":
            self.factor = self.factor_levin
        elif method == "sidi":
            self.factor = self.factor_sidi
        else:
            raise ValueError("levin: unknown method \"%s\"" % method)

    def factor_levin(self, i):
        # original levin
        # [1] p.50,e.7.5-7 (with n-j replaced by i)
        return (self.theta + i) * (self.theta + self.n - 1) ** (self.n - i - 2) / self.ctx.mpf(self.theta + self.n) ** (self.n - i - 1)

    def factor_sidi(self, i):
        # sidi analogon to levin (factorial series)
        # [1] p.59,e.8.3-16 (with n-j replaced by i)
        return (self.theta + self.n - 1) * (self.theta + self.n - 2) / self.ctx.mpf((self.theta + 2 * self.n - i - 2) * (self.theta + 2 * self.n - i - 3))

    def run(self, s, a0, a1 = 0):
        if self.variant=="t":
            # levin t
            w=a0
        elif self.variant=="u":
            # levin u
            w=a0*(self.theta+self.n)
        elif self.variant=="v":
            # levin v
            w=a0*a1/(a0-a1)
        else:
            assert False, "unknown variant"

        if w==0:
            raise ValueError("levin: zero weight")

        self.A.append(s/w)
        self.B.append(1/w)

        for i in range(self.n-1,-1,-1):
            if i==self.n-1:
                f=1
            else:
                f=self.factor(i)

            self.A[i]=self.A[i+1]-f*self.A[i]
            self.B[i]=self.B[i+1]-f*self.B[i]

        self.n+=1

    ###########################################################################

    def update_psum(self,S):
        """
        This routine applies the convergence acceleration to the list of partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k, k = 0..n)

        v, e = ...update_psum([s_0, s_1,..., s_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """

        if self.variant!="v":
            if self.n==0:
                self.run(S[0],S[0])
            while self.n<len(S):
                self.run(S[self.n],S[self.n]-S[self.n-1])
        else:
            if len(S)==1:
                self.last=0
                return S[0],abs(S[0])

            if self.n==0:
                self.a1=S[1]-S[0]
                self.run(S[0],S[0],self.a1)

            while self.n<len(S)-1:
                na1=S[self.n+1]-S[self.n]
                self.run(S[self.n],self.a1,na1)
                self.a1=na1

        value=self.A[0]/self.B[0]
        err=abs(value-self.last)
        self.last=value

        return value,err

    def update(self,X):
        """
        This routine applies the convergence acceleration to the list of individual terms.

        A = sum(a_k, k = 0..infinity)

        v, e = ...update([a_0, a_1,..., a_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """

        if self.variant!="v":
            if self.n==0:
                self.s=X[0]
                self.run(self.s,X[0])
            while self.n<len(X):
                self.s+=X[self.n]
                self.run(self.s,X[self.n])
        else:
            if len(X)==1:
                self.last=0
                return X[0],abs(X[0])

            if self.n==0:
                self.s=X[0]
                self.run(self.s,X[0],X[1])

            while self.n<len(X)-1:
                self.s+=X[self.n]
                self.run(self.s,X[self.n],X[self.n+1])

        value=self.A[0]/self.B[0]
        err=abs(value-self.last)
        self.last=value

        return value,err

    ###########################################################################

    def step_psum(self,s):
        """
        This routine applies the convergence acceleration to the partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k, k = 0..n)

        v, e = ...step_psum(s_k)

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """

        if self.variant!="v":
            if self.n==0:
                self.last_s=s
                self.run(s,s)
            else:
                self.run(s,s-self.last_s)
                self.last_s=s
        else:
            if isinstance(self.last_s,bool):
                self.last_s=s
                self.last_w=s
                self.last=0
                return s,abs(s)

            na1=s-self.last_s
            self.run(self.last_s,self.last_w,na1)
            self.last_w=na1
            self.last_s=s

        value=self.A[0]/self.B[0]
        err=abs(value-self.last)
        self.last=value

        return value,err

    def step(self,x):
        """
        This routine applies the convergence acceleration to the individual terms.

        A = sum(a_k, k = 0..infinity)

        v, e = ...step(a_k)

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """

        if self.variant!="v":
            if self.n==0:
                self.s=x
                self.run(self.s,x)
            else:
                self.s+=x
                self.run(self.s,x)
        else:
            if isinstance(self.last_s,bool):
                self.last_s=x
                self.s=0
                self.last=0
                return x,abs(x)

            self.s+=self.last_s
            self.run(self.s,self.last_s,x)
            self.last_s=x

        value=self.A[0]/self.B[0]
        err=abs(value-self.last)
        self.last=value

        return value,err

def levin(ctx, method = "levin", variant = "u"):
    L = levin_class(method = method, variant = variant)
    L.ctx = ctx
    return L

levin.__doc__ = levin_class.__doc__
defun(levin)


class cohen_alt_class:
    # cohen_alt: Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
    r"""
    This interface implements the convergence acceleration of alternating series
    as described in H. Cohen, F.R. Villegas, D. Zagier - "Convergence Acceleration
    of Alternating Series". This series transformation works only well if the
    individual terms of the series have an alternating sign. It belongs to the
    class of linear series transformations (in contrast to the Shanks/Wynn-epsilon
    or Levin transform). This series transformation is also able to sum some types
    of divergent series. See the paper under which conditions this resummation is
    mathematical sound.

    Let *A* be the series we want to sum:

    .. math ::

        A = \sum_{k=0}^{\infty} a_k

    Let `s_n` be the partial sums of this series:

    .. math ::

        s_n = \sum_{k=0}^n a_k.


    **Interface**

    Calling ``cohen_alt`` returns an object with the following methods.

    Then ``update(...)`` works with the list of individual terms `a_k` and
    ``update_psum(...)`` works with the list of partial sums `s_k`:

    .. code ::

        v, e = ...update([a_0, a_1,..., a_k])
        v, e = ...update_psum([s_0, s_1,..., s_k])

    *v* is the current estimate for *A*, and *e* is an error estimate which is
    simply the difference between the current estimate and the last estimate.

    **Examples**

    Here we compute the alternating zeta function using ``update_psum``::

        >>> from mpmath import mp
        >>> AC = mp.cohen_alt()
        >>> S, s, n = [], 0, 1
        >>> while 1:
        ...     s += -((-1) ** n) * mp.one / (n * n)
        ...     n += 1
        ...     S.append(s)
        ...     v, e = AC.update_psum(S)
        ...     if e < mp.eps:
        ...         break
        ...     if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - mp.pi ** 2 / 12))
        0.0

    Here we compute the product `\prod_{n=1}^{\infty} \Gamma(1+1/(2n-1)) / \Gamma(1+1/(2n))`::

        >>> A = []
        >>> AC = mp.cohen_alt()
        >>> n = 1
        >>> while 1:
        ...     A.append( mp.loggamma(1 + mp.one / (2 * n - 1)))
        ...     A.append(-mp.loggamma(1 + mp.one / (2 * n)))
        ...     n += 1
        ...     v, e = AC.update(A)
        ...     if e < mp.eps:
        ...         break
        ...     if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> v = mp.exp(v)
        >>> print(mp.chop(v - 1.06215090557106, tol = 1e-12))
        0.0

    ``cohen_alt`` is also accessible through the :func:`~mpmath.nsum` interface::

        >>> v = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "a")
        >>> print(mp.chop(v - mp.log(2)))
        0.0
        >>> v = mp.nsum(lambda n: (-1)**n / (2 * n + 1), [0, mp.inf], method = "a")
        >>> print(mp.chop(v - mp.pi / 4))
        0.0
        >>> v = mp.nsum(lambda n: (-1)**n * mp.log(n) * n, [1, mp.inf], method = "a")
        >>> print(mp.chop(v - mp.diff(lambda s: mp.altzeta(s), -1)))
        0.0

    """

    def __init__(self):
        self.last=0

    def update(self, A):
        """
        This routine applies the convergence acceleration to the list of individual terms.

        A    = sum(a_k, k = 0..infinity)

        v, e = ...update([a_0, a_1,..., a_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """

        n = len(A)
        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = -self.ctx.one
        c = -d
        s = 0

        for k in xrange(n):
            c = b - c
            if k % 2 == 0:
                s = s + c * A[k]
            else:
                s = s - c * A[k]
            b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + self.ctx.one))

        value = s / d

        err = abs(value - self.last)
        self.last = value

        return value, err

    def update_psum(self, S):
        """
        This routine applies the convergence acceleration to the list of partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k ,k = 0..n)

        v, e = ...update_psum([s_0, s_1,..., s_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """

        n = len(S)
        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = self.ctx.one
        s = 0

        for k in xrange(n):
            b = 2 * (n + k) * (n - k) * b / ((2 * k + 1) * (k + self.ctx.one))
            s += b * S[k]

        value = s / d

        err = abs(value - self.last)
        self.last = value

        return value, err

def cohen_alt(ctx):
    L = cohen_alt_class()
    L.ctx = ctx
    return L

cohen_alt.__doc__ = cohen_alt_class.__doc__
defun(cohen_alt)


@defun
def sumap(ctx, f, interval, integral=None, error=False):
    r"""
    Evaluates an infinite series of an analytic summand *f* using the
    Abel-Plana formula

    .. math ::

        \sum_{k=0}^{\infty} f(k) = \int_0^{\infty} f(t) dt + \frac{1}{2} f(0) +
            i \int_0^{\infty} \frac{f(it)-f(-it)}{e^{2\pi t}-1} dt.

    Unlike the Euler-Maclaurin formula (see :func:`~mpmath.sumem`),
    the Abel-Plana formula does not require derivatives. However,
    it only works when `|f(it)-f(-it)|` does not
    increase too rapidly with `t`.

    **Examples**

    The Abel-Plana formula is particularly useful when the summand
    decreases like a power of `k`; for example when the sum is a pure
    zeta function::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> sumap(lambda k: 1/k**2.5, [1,inf])
        1.34148725725091717975677
        >>> zeta(2.5)
        1.34148725725091717975677
        >>> sumap(lambda k: 1/(k+1j)**(2.5+2.5j), [1,inf])
        (-3.385361068546473342286084 - 0.7432082105196321803869551j)
        >>> zeta(2.5+2.5j, 1+1j)
        (-3.385361068546473342286084 - 0.7432082105196321803869551j)

    If the series is alternating, numerical quadrature along the real
    line is likely to give poor results, so it is better to evaluate
    the first term symbolically whenever possible:

        >>> n=3; z=-0.75
        >>> I = expint(n,-log(z))
        >>> chop(sumap(lambda k: z**k / k**n, [1,inf], integral=I))
        -0.6917036036904594510141448
        >>> polylog(n,z)
        -0.6917036036904594510141448

    """
    prec = ctx.prec
    try:
        ctx.prec += 10
        a, b = interval
        if  b != ctx.inf:
            raise ValueError("b should be equal to ctx.inf")
        g = lambda x: f(x+a)
        if integral is None:
            i1, err1 = ctx.quad(g, [0,ctx.inf], error=True)
        else:
            i1, err1 = integral, 0
        j = ctx.j
        p = ctx.pi * 2
        if ctx._is_real_type(i1):
            h = lambda t: -2 * ctx.im(g(j*t)) / ctx.expm1(p*t)
        else:
            h = lambda t: j*(g(j*t)-g(-j*t)) / ctx.expm1(p*t)
        i2, err2 = ctx.quad(h, [0,ctx.inf], error=True)
        err = err1+err2
        v = i1+i2+0.5*g(ctx.mpf(0))
    finally:
        ctx.prec = prec
    if error:
        return +v, err
    return +v


@defun
def sumem(ctx, f, interval, tol=None, reject=10, integral=None,
    adiffs=None, bdiffs=None, verbose=False, error=False,
    _fast_abort=False):
    r"""
    Uses the Euler-Maclaurin formula to compute an approximation accurate
    to within ``tol`` (which defaults to the present epsilon) of the sum

    .. math ::

        S = \sum_{k=a}^b f(k)

    where `(a,b)` are given by ``interval`` and `a` or `b` may be
    infinite. The approximation is

    .. math ::

        S \sim \int_a^b f(x) \,dx + \frac{f(a)+f(b)}{2} +
        \sum_{k=1}^{\infty} \frac{B_{2k}}{(2k)!}
        \left(f^{(2k-1)}(b)-f^{(2k-1)}(a)\right).

    The last sum in the Euler-Maclaurin formula is not generally
    convergent (a notable exception is if `f` is a polynomial, in
    which case Euler-Maclaurin actually gives an exact result).

    The summation is stopped as soon as the quotient between two
    consecutive terms falls below *reject*. That is, by default
    (*reject* = 10), the summation is continued as long as each
    term adds at least one decimal.

    Although not convergent, convergence to a given tolerance can
    often be "forced" if `b = \infty` by summing up to `a+N` and then
    applying the Euler-Maclaurin formula to the sum over the range
    `(a+N+1, \ldots, \infty)`. This procedure is implemented by
    :func:`~mpmath.nsum`.

    By default numerical quadrature and differentiation is used.
    If the symbolic values of the integral and endpoint derivatives
    are known, it is more efficient to pass the value of the
    integral explicitly as ``integral`` and the derivatives
    explicitly as ``adiffs`` and ``bdiffs``. The derivatives
    should be given as iterables that yield
    `f(a), f'(a), f''(a), \ldots` (and the equivalent for `b`).

    **Examples**

    Summation of an infinite series, with automatic and symbolic
    integral and derivative values (the second should be much faster)::

        >>> from mpmath import *
        >>> mp.dps = 50; mp.pretty = True
        >>> sumem(lambda n: 1/n**2, [32, inf])
        0.03174336652030209012658168043874142714132886413417
        >>> I = mpf(1)/32
        >>> D = adiffs=((-1)**n*fac(n+1)*32**(-2-n) for n in range(999))
        >>> sumem(lambda n: 1/n**2, [32, inf], integral=I, adiffs=D)
        0.03174336652030209012658168043874142714132886413417

    An exact evaluation of a finite polynomial sum::

        >>> sumem(lambda n: n**5-12*n**2+3*n, [-100000, 200000])
        10500155000624963999742499550000.0
        >>> print(sum(n**5-12*n**2+3*n for n in range(-100000, 200001)))
        10500155000624963999742499550000

    """
    tol = tol or +ctx.eps
    interval = ctx._as_points(interval)
    a = ctx.convert(interval[0])
    b = ctx.convert(interval[-1])
    err = ctx.zero
    prev = 0
    M = 10000
    if a == ctx.ninf: adiffs = (0 for n in xrange(M))
    else:             adiffs = adiffs or ctx.diffs(f, a)
    if b == ctx.inf:  bdiffs = (0 for n in xrange(M))
    else:             bdiffs = bdiffs or ctx.diffs(f, b)
    orig = ctx.prec
    #verbose = 1
    try:
        ctx.prec += 10
        s = ctx.zero
        for k, (da, db) in enumerate(izip(adiffs, bdiffs)):
            if k & 1:
                term = (db-da) * ctx.bernoulli(k+1) / ctx.factorial(k+1)
                mag = abs(term)
                if verbose:
                    print("term", k, "magnitude =", ctx.nstr(mag))
                if k > 4 and mag < tol:
                    s += term
                    break
                elif k > 4 and abs(prev) / mag < reject:
                    err += mag
                    if _fast_abort:
                        return [s, (s, err)][error]
                    if verbose:
                        print("Failed to converge")
                    break
                else:
                    s += term
                prev = term
        # Endpoint correction
        if a != ctx.ninf: s += f(a)/2
        if b != ctx.inf: s += f(b)/2
        # Tail integral
        if verbose:
            print("Integrating f(x) from x = %s to %s" % (ctx.nstr(a), ctx.nstr(b)))
        if integral:
            s += integral
        else:
            integral, ierr = ctx.quad(f, interval, error=True)
            if verbose:
                print("Integration error:", ierr)
            s += integral
            err += ierr
    finally:
        ctx.prec = orig
    if error:
        return s, err
    else:
        return s

@defun
def adaptive_extrapolation(ctx, update, emfun, kwargs):
    option = kwargs.get
    if ctx._fixed_precision:
        tol = option('tol', ctx.eps*2**10)
    else:
        tol = option('tol', ctx.eps/2**10)
    verbose = option('verbose', False)
    maxterms = option('maxterms', ctx.dps*10)
    method = set(option('method', 'r+s').split('+'))
    skip = option('skip', 0)
    steps = iter(option('steps', xrange(10, 10**9, 10)))
    strict = option('strict')
    #steps = (10 for i in xrange(1000))
    summer=[]
    if 'd' in method or 'direct' in method:
        TRY_RICHARDSON = TRY_SHANKS = TRY_EULER_MACLAURIN = False
    else:
        TRY_RICHARDSON = ('r' in method) or ('richardson' in method)
        TRY_SHANKS = ('s' in method) or ('shanks' in method)
        TRY_EULER_MACLAURIN = ('e' in method) or \
            ('euler-maclaurin' in method)

        def init_levin(m):
            variant = kwargs.get("levin_variant", "u")
            if isinstance(variant, str):
                if variant == "all":
                    variant = ["u", "v", "t"]
                else:
                    variant = [variant]
            for s in variant:
                L = levin_class(method = m, variant = s)
                L.ctx = ctx
                L.name = m + "(" + s + ")"
                summer.append(L)

        if ('l' in method) or ('levin' in method):
            init_levin("levin")

        if ('sidi' in method):
            init_levin("sidi")

        if ('a' in method) or ('alternating' in method):
            L = cohen_alt_class()
            L.ctx = ctx
            L.name = "alternating"
            summer.append(L)

    last_richardson_value = 0
    shanks_table = []
    index = 0
    step = 10
    partial = []
    best = ctx.zero
    orig = ctx.prec
    try:
        if 'workprec' in kwargs:
            ctx.prec = kwargs['workprec']
        elif TRY_RICHARDSON or TRY_SHANKS or len(summer)!=0:
            ctx.prec = (ctx.prec+10) * 4
        else:
            ctx.prec += 30
        while 1:
            if index >= maxterms:
                break

            # Get new batch of terms
            try:
                step = next(steps)
            except StopIteration:
                pass
            if verbose:
                print("-"*70)
                print("Adding terms #%i-#%i" % (index, index+step))
            update(partial, xrange(index, index+step))
            index += step

            # Check direct error
            best = partial[-1]
            error = abs(best - partial[-2])
            if verbose:
                print("Direct error: %s" % ctx.nstr(error))
            if error <= tol:
                return best

            # Check each extrapolation method
            if TRY_RICHARDSON:
                value, maxc = ctx.richardson(partial)
                # Convergence
                richardson_error = abs(value - last_richardson_value)
                if verbose:
                    print("Richardson error: %s" % ctx.nstr(richardson_error))
                # Convergence
                if richardson_error <= tol:
                    return value
                last_richardson_value = value
                # Unreliable due to cancellation
                if ctx.eps*maxc > tol:
                    if verbose:
                        print("Ran out of precision for Richardson")
                    TRY_RICHARDSON = False
                if richardson_error < error:
                    error = richardson_error
                    best = value
            if TRY_SHANKS:
                shanks_table = ctx.shanks(partial, shanks_table, randomized=True)
                row = shanks_table[-1]
                if len(row) == 2:
                    est1 = row[-1]
                    shanks_error = 0
                else:
                    est1, maxc, est2 = row[-1], abs(row[-2]), row[-3]
                    shanks_error = abs(est1-est2)
                if verbose:
                    print("Shanks error: %s" % ctx.nstr(shanks_error))
                if shanks_error <= tol:
                    return est1
                if ctx.eps*maxc > tol:
                    if verbose:
                        print("Ran out of precision for Shanks")
                    TRY_SHANKS = False
                if shanks_error < error:
                    error = shanks_error
                    best = est1
            for L in summer:
                est, lerror = L.update_psum(partial)
                if verbose:
                    print("%s error: %s" % (L.name, ctx.nstr(lerror)))
                if lerror <= tol:
                    return est
                if lerror < error:
                    error = lerror
                    best = est
            if TRY_EULER_MACLAURIN:
                if ctx.almosteq(ctx.mpc(ctx.sign(partial[-1]) / ctx.sign(partial[-2])), -1):
                    if verbose:
                        print ("NOT using Euler-Maclaurin: the series appears"
                            " to be alternating, so numerical\n quadrature"
                            " will most likely fail")
                    TRY_EULER_MACLAURIN = False
                else:
                    value, em_error = emfun(index, tol)
                    value += partial[-1]
                    if verbose:
                        print("Euler-Maclaurin error: %s" % ctx.nstr(em_error))
                    if em_error <= tol:
                        return value
                    if em_error < error:
                        best = value
    finally:
        ctx.prec = orig
    if strict:
        raise ctx.NoConvergence
    if verbose:
        print("Warning: failed to converge to target accuracy")
    return best

@defun
def nsum(ctx, f, *intervals, **options):
    r"""
    Computes the sum

    .. math :: S = \sum_{k=a}^b f(k)

    where `(a, b)` = *interval*, and where `a = -\infty` and/or
    `b = \infty` are allowed, or more generally

    .. math :: S = \sum_{k_1=a_1}^{b_1} \cdots
                   \sum_{k_n=a_n}^{b_n} f(k_1,\ldots,k_n)

    if multiple intervals are given.

    Two examples of infinite series that can be summed by :func:`~mpmath.nsum`,
    where the first converges rapidly and the second converges slowly,
    are::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nsum(lambda n: 1/fac(n), [0, inf])
        2.71828182845905
        >>> nsum(lambda n: 1/n**2, [1, inf])
        1.64493406684823

    When appropriate, :func:`~mpmath.nsum` applies convergence acceleration to
    accurately estimate the sums of slowly convergent series. If the series is
    finite, :func:`~mpmath.nsum` currently does not attempt to perform any
    extrapolation, and simply calls :func:`~mpmath.fsum`.

    Multidimensional infinite series are reduced to a single-dimensional
    series over expanding hypercubes; if both infinite and finite dimensions
    are present, the finite ranges are moved innermost. For more advanced
    control over the summation order, use nested calls to :func:`~mpmath.nsum`,
    or manually rewrite the sum as a single-dimensional series.

    **Options**

    *tol*
        Desired maximum final error. Defaults roughly to the
        epsilon of the working precision.

    *method*
        Which summation algorithm to use (described below).
        Default: ``'richardson+shanks'``.

    *maxterms*
        Cancel after at most this many terms. Default: 10*dps.

    *steps*
        An iterable giving the number of terms to add between
        each extrapolation attempt. The default sequence is
        [10, 20, 30, 40, ...]. For example, if you know that
        approximately 100 terms will be required, efficiency might be
        improved by setting this to [100, 10]. Then the first
        extrapolation will be performed after 100 terms, the second
        after 110, etc.

    *verbose*
        Print details about progress.

    *ignore*
        If enabled, any term that raises ``ArithmeticError``
        or ``ValueError`` (e.g. through division by zero) is replaced
        by a zero. This is convenient for lattice sums with
        a singular term near the origin.

    **Methods**

    Unfortunately, an algorithm that can efficiently sum any infinite
    series does not exist. :func:`~mpmath.nsum` implements several different
    algorithms that each work well in different cases. The *method*
    keyword argument selects a method.

    The default method is ``'r+s'``, i.e. both Richardson extrapolation
    and Shanks transformation is attempted. A slower method that
    handles more cases is ``'r+s+e'``. For very high precision
    summation, or if the summation needs to be fast (for example if
    multiple sums need to be evaluated), it is a good idea to
    investigate which one method works best and only use that.

    ``'richardson'`` / ``'r'``:
        Uses Richardson extrapolation. Provides useful extrapolation
        when `f(k) \sim P(k)/Q(k)` or when `f(k) \sim (-1)^k P(k)/Q(k)`
        for polynomials `P` and `Q`. See :func:`~mpmath.richardson` for
        additional information.

    ``'shanks'`` / ``'s'``:
        Uses Shanks transformation. Typically provides useful
        extrapolation when `f(k) \sim c^k` or when successive terms
        alternate signs. Is able to sum some divergent series.
        See :func:`~mpmath.shanks` for additional information.

    ``'levin'`` / ``'l'``:
        Uses the Levin transformation. It performs better than the Shanks
        transformation for logarithmic convergent or alternating divergent
        series. The ``'levin_variant'``-keyword selects the variant. Valid
        choices are "u", "t", "v" and "all" whereby "all" uses all three
        u,t and v simultanously (This is good for performance comparison in
        conjunction with "verbose=True"). Instead of the Levin transform one can
        also use the Sidi-S transform by selecting the method ``'sidi'``.
        See :func:`~mpmath.levin` for additional details.

    ``'alternating'`` / ``'a'``:
        This is the convergence acceleration of alternating series developped
        by Cohen, Villegras and Zagier.
        See :func:`~mpmath.cohen_alt` for additional details.

    ``'euler-maclaurin'`` / ``'e'``:
        Uses the Euler-Maclaurin summation formula to approximate
        the remainder sum by an integral. This requires high-order
        numerical derivatives and numerical integration. The advantage
        of this algorithm is that it works regardless of the
        decay rate of `f`, as long as `f` is sufficiently smooth.
        See :func:`~mpmath.sumem` for additional information.

    ``'direct'`` / ``'d'``:
        Does not perform any extrapolation. This can be used
        (and should only be used for) rapidly convergent series.
        The summation automatically stops when the terms
        decrease below the target tolerance.

    **Basic examples**

    A finite sum::

        >>> nsum(lambda k: 1/k, [1, 6])
        2.45

    Summation of a series going to negative infinity and a doubly
    infinite series::

        >>> nsum(lambda k: 1/k**2, [-inf, -1])
        1.64493406684823
        >>> nsum(lambda k: 1/(1+k**2), [-inf, inf])
        3.15334809493716

    :func:`~mpmath.nsum` handles sums of complex numbers::

        >>> nsum(lambda k: (0.5+0.25j)**k, [0, inf])
        (1.6 + 0.8j)

    The following sum converges very rapidly, so it is most
    efficient to sum it by disabling convergence acceleration::

        >>> mp.dps = 1000
        >>> a = nsum(lambda k: -(-1)**k * k**2 / fac(2*k), [1, inf],
        ...     method='direct')
        >>> b = (cos(1)+sin(1))/4
        >>> abs(a-b) < mpf('1e-998')
        True

    **Examples with Richardson extrapolation**

    Richardson extrapolation works well for sums over rational
    functions, as well as their alternating counterparts::

        >>> mp.dps = 50
        >>> nsum(lambda k: 1 / k**3, [1, inf],
        ...     method='richardson')
        1.2020569031595942853997381615114499907649862923405
        >>> zeta(3)
        1.2020569031595942853997381615114499907649862923405

        >>> nsum(lambda n: (n + 3)/(n**3 + n**2), [1, inf],
        ...     method='richardson')
        2.9348022005446793094172454999380755676568497036204
        >>> pi**2/2-2
        2.9348022005446793094172454999380755676568497036204

        >>> nsum(lambda k: (-1)**k / k**3, [1, inf],
        ...     method='richardson')
        -0.90154267736969571404980362113358749307373971925537
        >>> -3*zeta(3)/4
        -0.90154267736969571404980362113358749307373971925538

    **Examples with Shanks transformation**

    The Shanks transformation works well for geometric series
    and typically provides excellent acceleration for Taylor
    series near the border of their disk of convergence.
    Here we apply it to a series for `\log(2)`, which can be
    seen as the Taylor series for `\log(1+x)` with `x = 1`::

        >>> nsum(lambda k: -(-1)**k/k, [1, inf],
        ...     method='shanks')
        0.69314718055994530941723212145817656807550013436025
        >>> log(2)
        0.69314718055994530941723212145817656807550013436025

    Here we apply it to a slowly convergent geometric series::

        >>> nsum(lambda k: mpf('0.995')**k, [0, inf],
        ...     method='shanks')
        200.0

    Finally, Shanks' method works very well for alternating series
    where `f(k) = (-1)^k g(k)`, and often does so regardless of
    the exact decay rate of `g(k)`::

        >>> mp.dps = 15
        >>> nsum(lambda k: (-1)**(k+1) / k**1.5, [1, inf],
        ...     method='shanks')
        0.765147024625408
        >>> (2-sqrt(2))*zeta(1.5)/2
        0.765147024625408

    The following slowly convergent alternating series has no known
    closed-form value. Evaluating the sum a second time at higher
    precision indicates that the value is probably correct::

        >>> nsum(lambda k: (-1)**k / log(k), [2, inf],
        ...     method='shanks')
        0.924299897222939
        >>> mp.dps = 30
        >>> nsum(lambda k: (-1)**k / log(k), [2, inf],
        ...     method='shanks')
        0.92429989722293885595957018136

    **Examples with Levin transformation**

    The following example calculates Euler's constant as the constant term in
    the Laurent expansion of zeta(s) at s=1. This sum converges extremly slow
    because of the logarithmic convergence behaviour of the Dirichlet series
    for zeta.

      >>> mp.dps = 30
      >>> z = mp.mpf(10) ** (-10)
      >>> a = mp.nsum(lambda n: n**(-(1+z)), [1, mp.inf], method = "levin") - 1 / z
      >>> print(mp.chop(a - mp.euler, tol = 1e-10))
      0.0

    Now we sum the zeta function outside its range of convergence
    (attention: This does not work at the negative integers!):

      >>> mp.dps = 15
      >>> w = mp.nsum(lambda n: n ** (2 + 3j), [1, mp.inf], method = "levin", levin_variant = "v")
      >>> print(mp.chop(w - mp.zeta(-2-3j)))
      0.0

    The next example resummates an asymptotic series expansion of an integral
    related to the exponential integral.

      >>> mp.dps = 15
      >>> z = mp.mpf(10)
      >>> # exact = mp.quad(lambda x: mp.exp(-x)/(1+x/z),[0,mp.inf])
      >>> exact = z * mp.exp(z) * mp.expint(1,z) # this is the symbolic expression for the integral
      >>> w = mp.nsum(lambda n: (-1) ** n * mp.fac(n) * z ** (-n), [0, mp.inf], method = "sidi", levin_variant = "t")
      >>> print(mp.chop(w - exact))
      0.0

    Following highly divergent asymptotic expansion needs some care. Firstly we
    need copious amount of working precision. Secondly the stepsize must not be
    chosen to large, otherwise nsum may miss the point where the Levin transform
    converges and reach the point where only numerical garbage is produced due to
    numerical cancellation.

      >>> mp.dps = 15
      >>> z = mp.mpf(2)
      >>> # exact = mp.quad(lambda x: mp.exp( -x * x / 2 - z * x ** 4), [0,mp.inf]) * 2 / mp.sqrt(2 * mp.pi)
      >>> exact = mp.exp(mp.one / (32 * z)) * mp.besselk(mp.one / 4, mp.one / (32 * z)) / (4 * mp.sqrt(z * mp.pi)) # this is the symbolic expression for the integral
      >>> w = mp.nsum(lambda n: (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n)),
      ...   [0, mp.inf], method = "levin", levin_variant = "t", workprec = 8*mp.prec, steps = [2] + [1 for x in xrange(1000)])
      >>> print(mp.chop(w - exact))
      0.0

    The hypergeoemtric function can also be summed outside its range of convergence:

      >>> mp.dps = 15
      >>> z = 2 + 1j
      >>> exact = mp.hyp2f1(2 / mp.mpf(3), 4 / mp.mpf(3), 1 / mp.mpf(3), z)
      >>> f = lambda n: mp.rf(2 / mp.mpf(3), n) * mp.rf(4 / mp.mpf(3), n) * z**n / (mp.rf(1 / mp.mpf(3), n) * mp.fac(n))
      >>> v = mp.nsum(f, [0, mp.inf], method = "levin", steps = [10 for x in xrange(1000)])
      >>> print(mp.chop(exact-v))
      0.0

    **Examples with Cohen's alternating series resummation**

      The next example sums the alternating zeta function:

      >>> v = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "a")
      >>> print(mp.chop(v - mp.log(2)))
      0.0

      The derivate of the alternating zeta function outside its range of
      convergence:

      >>> v = mp.nsum(lambda n: (-1)**n * mp.log(n) * n, [1, mp.inf], method = "a")
      >>> print(mp.chop(v - mp.diff(lambda s: mp.altzeta(s), -1)))
      0.0

    **Examples with Euler-Maclaurin summation**

    The sum in the following example has the wrong rate of convergence
    for either Richardson or Shanks to be effective.

        >>> f = lambda k: log(k)/k**2.5
        >>> mp.dps = 15
        >>> nsum(f, [1, inf], method='euler-maclaurin')
        0.38734195032621
        >>> -diff(zeta, 2.5)
        0.38734195032621

    Increasing ``steps`` improves speed at higher precision::

        >>> mp.dps = 50
        >>> nsum(f, [1, inf], method='euler-maclaurin', steps=[250])
        0.38734195032620997271199237593105101319948228874688
        >>> -diff(zeta, 2.5)
        0.38734195032620997271199237593105101319948228874688

    **Divergent series**

    The Shanks transformation is able to sum some *divergent*
    series. In particular, it is often able to sum Taylor series
    beyond their radius of convergence (this is due to a relation
    between the Shanks transformation and Pade approximations;
    see :func:`~mpmath.pade` for an alternative way to evaluate divergent
    Taylor series). Furthermore the Levin-transform examples above
    contain some divergent series resummation.

    Here we apply it to `\log(1+x)` far outside the region of
    convergence::

        >>> mp.dps = 50
        >>> nsum(lambda k: -(-9)**k/k, [1, inf],
        ...     method='shanks')
        2.3025850929940456840179914546843642076011014886288
        >>> log(10)
        2.3025850929940456840179914546843642076011014886288

    A particular type of divergent series that can be summed
    using the Shanks transformation is geometric series.
    The result is the same as using the closed-form formula
    for an infinite geometric series::

        >>> mp.dps = 15
        >>> for n in range(-8, 8):
        ...     if n == 1:
        ...         continue
        ...     print("%s %s %s" % (mpf(n), mpf(1)/(1-n),
        ...         nsum(lambda k: n**k, [0, inf], method='shanks')))
        ...
        -8.0 0.111111111111111 0.111111111111111
        -7.0 0.125 0.125
        -6.0 0.142857142857143 0.142857142857143
        -5.0 0.166666666666667 0.166666666666667
        -4.0 0.2 0.2
        -3.0 0.25 0.25
        -2.0 0.333333333333333 0.333333333333333
        -1.0 0.5 0.5
        0.0 1.0 1.0
        2.0 -1.0 -1.0
        3.0 -0.5 -0.5
        4.0 -0.333333333333333 -0.333333333333333
        5.0 -0.25 -0.25
        6.0 -0.2 -0.2
        7.0 -0.166666666666667 -0.166666666666667

    **Multidimensional sums**

    Any combination of finite and infinite ranges is allowed for the
    summation indices::

        >>> mp.dps = 15
        >>> nsum(lambda x,y: x+y, [2,3], [4,5])
        28.0
        >>> nsum(lambda x,y: x/2**y, [1,3], [1,inf])
        6.0
        >>> nsum(lambda x,y: y/2**x, [1,inf], [1,3])
        6.0
        >>> nsum(lambda x,y,z: z/(2**x*2**y), [1,inf], [1,inf], [3,4])
        7.0
        >>> nsum(lambda x,y,z: y/(2**x*2**z), [1,inf], [3,4], [1,inf])
        7.0
        >>> nsum(lambda x,y,z: x/(2**z*2**y), [3,4], [1,inf], [1,inf])
        7.0

    Some nice examples of double series with analytic solutions or
    reductions to single-dimensional series (see [1])::

        >>> nsum(lambda m, n: 1/2**(m*n), [1,inf], [1,inf])
        1.60669515241529
        >>> nsum(lambda n: 1/(2**n-1), [1,inf])
        1.60669515241529

        >>> nsum(lambda i,j: (-1)**(i+j)/(i**2+j**2), [1,inf], [1,inf])
        0.278070510848213
        >>> pi*(pi-3*ln2)/12
        0.278070510848213

        >>> nsum(lambda i,j: (-1)**(i+j)/(i+j)**2, [1,inf], [1,inf])
        0.129319852864168
        >>> altzeta(2) - altzeta(1)
        0.129319852864168

        >>> nsum(lambda i,j: (-1)**(i+j)/(i+j)**3, [1,inf], [1,inf])
        0.0790756439455825
        >>> altzeta(3) - altzeta(2)
        0.0790756439455825

        >>> nsum(lambda m,n: m**2*n/(3**m*(n*3**m+m*3**n)),
        ...     [1,inf], [1,inf])
        0.28125
        >>> mpf(9)/32
        0.28125

        >>> nsum(lambda i,j: fac(i-1)*fac(j-1)/fac(i+j),
        ...     [1,inf], [1,inf], workprec=400)
        1.64493406684823
        >>> zeta(2)
        1.64493406684823

    A hard example of a multidimensional sum is the Madelung constant
    in three dimensions (see [2]). The defining sum converges very
    slowly and only conditionally, so :func:`~mpmath.nsum` is lucky to
    obtain an accurate value through convergence acceleration. The
    second evaluation below uses a much more efficient, rapidly
    convergent 2D sum::

        >>> nsum(lambda x,y,z: (-1)**(x+y+z)/(x*x+y*y+z*z)**0.5,
        ...     [-inf,inf], [-inf,inf], [-inf,inf], ignore=True)
        -1.74756459463318
        >>> nsum(lambda x,y: -12*pi*sech(0.5*pi * \
        ...     sqrt((2*x+1)**2+(2*y+1)**2))**2, [0,inf], [0,inf])
        -1.74756459463318

    Another example of a lattice sum in 2D::

        >>> nsum(lambda x,y: (-1)**(x+y) / (x**2+y**2), [-inf,inf],
        ...     [-inf,inf], ignore=True)
        -2.1775860903036
        >>> -pi*ln2
        -2.1775860903036

    An example of an Eisenstein series::

        >>> nsum(lambda m,n: (m+n*1j)**(-4), [-inf,inf], [-inf,inf],
        ...     ignore=True)
        (3.1512120021539 + 0.0j)

    **References**

    1. [Weisstein]_ http://mathworld.wolfram.com/DoubleSeries.html,
    2. [Weisstein]_ http://mathworld.wolfram.com/MadelungConstants.html

    """
    infinite, g = standardize(ctx, f, intervals, options)
    if not infinite:
        return +g()

    def update(partial_sums, indices):
        if partial_sums:
            psum = partial_sums[-1]
        else:
            psum = ctx.zero
        for k in indices:
            psum = psum + g(ctx.mpf(k))
            partial_sums.append(psum)

    prec = ctx.prec

    def emfun(point, tol):
        workprec = ctx.prec
        ctx.prec = prec + 10
        v = ctx.sumem(g, [point, ctx.inf], tol, error=1)
        ctx.prec = workprec
        return v

    return +ctx.adaptive_extrapolation(update, emfun, options)


def wrapsafe(f):
    def g(*args):
        try:
            return f(*args)
        except (ArithmeticError, ValueError):
            return 0
    return g

def standardize(ctx, f, intervals, options):
    if options.get("ignore"):
        f = wrapsafe(f)
    finite = []
    infinite = []
    for k, points in enumerate(intervals):
        a, b = ctx._as_points(points)
        if b < a:
            return False, (lambda: ctx.zero)
        if a == ctx.ninf or b == ctx.inf:
            infinite.append((k, (a,b)))
        else:
            finite.append((k, (int(a), int(b))))
    if finite:
        f = fold_finite(ctx, f, finite)
        if not infinite:
            return False, lambda: f(*([0]*len(intervals)))
    if infinite:
        f = standardize_infinite(ctx, f, infinite)
        f = fold_infinite(ctx, f, infinite)
        args = [0] * len(intervals)
        d = infinite[0][0]
        def g(k):
            args[d] = k
            return f(*args)
        return True, g

# backwards compatible itertools.product
def cartesian_product(args):
    pools = map(tuple, args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def fold_finite(ctx, f, intervals):
    if not intervals:
        return f
    indices = [v[0] for v in intervals]
    points = [v[1] for v in intervals]
    ranges = [xrange(a, b+1) for (a,b) in points]
    def g(*args):
        args = list(args)
        s = ctx.zero
        for xs in cartesian_product(ranges):
            for dim, x in zip(indices, xs):
                args[dim] = ctx.mpf(x)
            s += f(*args)
        return s
    #print "Folded finite", indices
    return g

# Standardize each interval to [0,inf]
def standardize_infinite(ctx, f, intervals):
    if not intervals:
        return f
    dim, [a,b] = intervals[-1]
    if a == ctx.ninf:
        if b == ctx.inf:
            def g(*args):
                args = list(args)
                k = args[dim]
                if k:
                    s = f(*args)
                    args[dim] = -k
                    s += f(*args)
                    return s
                else:
                    return f(*args)
        else:
            def g(*args):
                args = list(args)
                args[dim] = b - args[dim]
                return f(*args)
    else:
        def g(*args):
            args = list(args)
            args[dim] += a
            return f(*args)
    #print "Standardized infinity along dimension", dim, a, b
    return standardize_infinite(ctx, g, intervals[:-1])

def fold_infinite(ctx, f, intervals):
    if len(intervals) < 2:
        return f
    dim1 = intervals[-2][0]
    dim2 = intervals[-1][0]
    # Assume intervals are [0,inf] x [0,inf] x ...
    def g(*args):
        args = list(args)
        #args.insert(dim2, None)
        n = int(args[dim1])
        s = ctx.zero
        #y = ctx.mpf(n)
        args[dim2] = ctx.mpf(n) #y
        for x in xrange(n+1):
            args[dim1] = ctx.mpf(x)
            s += f(*args)
        args[dim1] = ctx.mpf(n) #ctx.mpf(n)
        for y in xrange(n):
            args[dim2] = ctx.mpf(y)
            s += f(*args)
        return s
    #print "Folded infinite from", len(intervals), "to", (len(intervals)-1)
    return fold_infinite(ctx, g, intervals[:-1])

@defun
def nprod(ctx, f, interval, nsum=False, **kwargs):
    r"""
    Computes the product

    .. math ::

        P = \prod_{k=a}^b f(k)

    where `(a, b)` = *interval*, and where `a = -\infty` and/or
    `b = \infty` are allowed.

    By default, :func:`~mpmath.nprod` uses the same extrapolation methods as
    :func:`~mpmath.nsum`, except applied to the partial products rather than
    partial sums, and the same keyword options as for :func:`~mpmath.nsum` are
    supported. If ``nsum=True``, the product is instead computed via
    :func:`~mpmath.nsum` as

    .. math ::

        P = \exp\left( \sum_{k=a}^b \log(f(k)) \right).

    This is slower, but can sometimes yield better results. It is
    also required (and used automatically) when Euler-Maclaurin
    summation is requested.

    **Examples**

    A simple finite product::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> nprod(lambda k: k, [1, 4])
        24.0

    A large number of infinite products have known exact values,
    and can therefore be used as a reference. Most of the following
    examples are taken from MathWorld [1].

    A few infinite products with simple values are::

        >>> 2*nprod(lambda k: (4*k**2)/(4*k**2-1), [1, inf])
        3.141592653589793238462643
        >>> nprod(lambda k: (1+1/k)**2/(1+2/k), [1, inf])
        2.0
        >>> nprod(lambda k: (k**3-1)/(k**3+1), [2, inf])
        0.6666666666666666666666667
        >>> nprod(lambda k: (1-1/k**2), [2, inf])
        0.5

    Next, several more infinite products with more complicated
    values::

        >>> nprod(lambda k: exp(1/k**2), [1, inf]); exp(pi**2/6)
        5.180668317897115748416626
        5.180668317897115748416626

        >>> nprod(lambda k: (k**2-1)/(k**2+1), [2, inf]); pi*csch(pi)
        0.2720290549821331629502366
        0.2720290549821331629502366

        >>> nprod(lambda k: (k**4-1)/(k**4+1), [2, inf])
        0.8480540493529003921296502
        >>> pi*sinh(pi)/(cosh(sqrt(2)*pi)-cos(sqrt(2)*pi))
        0.8480540493529003921296502

        >>> nprod(lambda k: (1+1/k+1/k**2)**2/(1+2/k+3/k**2), [1, inf])
        1.848936182858244485224927
        >>> 3*sqrt(2)*cosh(pi*sqrt(3)/2)**2*csch(pi*sqrt(2))/pi
        1.848936182858244485224927

        >>> nprod(lambda k: (1-1/k**4), [2, inf]); sinh(pi)/(4*pi)
        0.9190194775937444301739244
        0.9190194775937444301739244

        >>> nprod(lambda k: (1-1/k**6), [2, inf])
        0.9826842777421925183244759
        >>> (1+cosh(pi*sqrt(3)))/(12*pi**2)
        0.9826842777421925183244759

        >>> nprod(lambda k: (1+1/k**2), [2, inf]); sinh(pi)/(2*pi)
        1.838038955187488860347849
        1.838038955187488860347849

        >>> nprod(lambda n: (1+1/n)**n * exp(1/(2*n)-1), [1, inf])
        1.447255926890365298959138
        >>> exp(1+euler/2)/sqrt(2*pi)
        1.447255926890365298959138

    The following two products are equivalent and can be evaluated in
    terms of a Jacobi theta function. Pi can be replaced by any value
    (as long as convergence is preserved)::

        >>> nprod(lambda k: (1-pi**-k)/(1+pi**-k), [1, inf])
        0.3838451207481672404778686
        >>> nprod(lambda k: tanh(k*log(pi)/2), [1, inf])
        0.3838451207481672404778686
        >>> jtheta(4,0,1/pi)
        0.3838451207481672404778686

    This product does not have a known closed form value::

        >>> nprod(lambda k: (1-1/2**k), [1, inf])
        0.2887880950866024212788997

    A product taken from `-\infty`::

        >>> nprod(lambda k: 1-k**(-3), [-inf,-2])
        0.8093965973662901095786805
        >>> cosh(pi*sqrt(3)/2)/(3*pi)
        0.8093965973662901095786805

    A doubly infinite product::

        >>> nprod(lambda k: exp(1/(1+k**2)), [-inf, inf])
        23.41432688231864337420035
        >>> exp(pi/tanh(pi))
        23.41432688231864337420035

    A product requiring the use of Euler-Maclaurin summation to compute
    an accurate value::

        >>> nprod(lambda k: (1-1/k**2.5), [2, inf], method='e')
        0.696155111336231052898125

    **References**

    1. [Weisstein]_ http://mathworld.wolfram.com/InfiniteProduct.html

    """
    if nsum or ('e' in kwargs.get('method', '')):
        orig = ctx.prec
        try:
            # TODO: we are evaluating log(1+eps) -> eps, which is
            # inaccurate. This currently works because nsum greatly
            # increases the working precision. But we should be
            # more intelligent and handle the precision here.
            ctx.prec += 10
            v = ctx.nsum(lambda n: ctx.ln(f(n)), interval, **kwargs)
        finally:
            ctx.prec = orig
        return +ctx.exp(v)

    a, b = ctx._as_points(interval)
    if a == ctx.ninf:
        if b == ctx.inf:
            return f(0) * ctx.nprod(lambda k: f(-k) * f(k), [1, ctx.inf], **kwargs)
        return ctx.nprod(f, [-b, ctx.inf], **kwargs)
    elif b != ctx.inf:
        return ctx.fprod(f(ctx.mpf(k)) for k in xrange(int(a), int(b)+1))

    a = int(a)

    def update(partial_products, indices):
        if partial_products:
            pprod = partial_products[-1]
        else:
            pprod = ctx.one
        for k in indices:
            pprod = pprod * f(a + ctx.mpf(k))
            partial_products.append(pprod)

    return +ctx.adaptive_extrapolation(update, None, kwargs)


@defun
def limit(ctx, f, x, direction=1, exp=False, **kwargs):
    r"""
    Computes an estimate of the limit

    .. math ::

        \lim_{t \to x} f(t)

    where `x` may be finite or infinite.

    For finite `x`, :func:`~mpmath.limit` evaluates `f(x + d/n)` for
    consecutive integer values of `n`, where the approach direction
    `d` may be specified using the *direction* keyword argument.
    For infinite `x`, :func:`~mpmath.limit` evaluates values of
    `f(\mathrm{sign}(x) \cdot n)`.

    If the approach to the limit is not sufficiently fast to give
    an accurate estimate directly, :func:`~mpmath.limit` attempts to find
    the limit using Richardson extrapolation or the Shanks
    transformation. You can select between these methods using
    the *method* keyword (see documentation of :func:`~mpmath.nsum` for
    more information).

    **Options**

    The following options are available with essentially the
    same meaning as for :func:`~mpmath.nsum`: *tol*, *method*, *maxterms*,
    *steps*, *verbose*.

    If the option *exp=True* is set, `f` will be
    sampled at exponentially spaced points `n = 2^1, 2^2, 2^3, \ldots`
    instead of the linearly spaced points `n = 1, 2, 3, \ldots`.
    This can sometimes improve the rate of convergence so that
    :func:`~mpmath.limit` may return a more accurate answer (and faster).
    However, do note that this can only be used if `f`
    supports fast and accurate evaluation for arguments that
    are extremely close to the limit point (or if infinite,
    very large arguments).

    **Examples**

    A basic evaluation of a removable singularity::

        >>> from mpmath import *
        >>> mp.dps = 30; mp.pretty = True
        >>> limit(lambda x: (x-sin(x))/x**3, 0)
        0.166666666666666666666666666667

    Computing the exponential function using its limit definition::

        >>> limit(lambda n: (1+3/n)**n, inf)
        20.0855369231876677409285296546
        >>> exp(3)
        20.0855369231876677409285296546

    A limit for `\pi`::

        >>> f = lambda n: 2**(4*n+1)*fac(n)**4/(2*n+1)/fac(2*n)**2
        >>> limit(f, inf)
        3.14159265358979323846264338328

    Calculating the coefficient in Stirling's formula::

        >>> limit(lambda n: fac(n) / (sqrt(n)*(n/e)**n), inf)
        2.50662827463100050241576528481
        >>> sqrt(2*pi)
        2.50662827463100050241576528481

    Evaluating Euler's constant `\gamma` using the limit representation

    .. math ::

        \gamma = \lim_{n \rightarrow \infty } \left[ \left(
        \sum_{k=1}^n \frac{1}{k} \right) - \log(n) \right]

    (which converges notoriously slowly)::

        >>> f = lambda n: sum([mpf(1)/k for k in range(1,int(n)+1)]) - log(n)
        >>> limit(f, inf)
        0.577215664901532860606512090082
        >>> +euler
        0.577215664901532860606512090082

    With default settings, the following limit converges too slowly
    to be evaluated accurately. Changing to exponential sampling
    however gives a perfect result::

        >>> f = lambda x: sqrt(x**3+x**2)/(sqrt(x**3)+x)
        >>> limit(f, inf)
        0.992831158558330281129249686491
        >>> limit(f, inf, exp=True)
        1.0

    """

    if ctx.isinf(x):
        direction = ctx.sign(x)
        g = lambda k: f(ctx.mpf(k+1)*direction)
    else:
        direction *= ctx.one
        g = lambda k: f(x + direction/(k+1))
    if exp:
        h = g
        g = lambda k: h(2**k)

    def update(values, indices):
        for k in indices:
            values.append(g(k+1))

    # XXX: steps used by nsum don't work well
    if not 'steps' in kwargs:
        kwargs['steps'] = [10]

    return +ctx.adaptive_extrapolation(update, None, kwargs)
<|endoftext|>import math

from ..libmp.backend import xrange

class QuadratureRule(object):
    """
    Quadrature rules are implemented using this class, in order to
    simplify the code and provide a common infrastructure
    for tasks such as error estimation and node caching.

    You can implement a custom quadrature rule by subclassing
    :class:`QuadratureRule` and implementing the appropriate
    methods. The subclass can then be used by :func:`~mpmath.quad` by
    passing it as the *method* argument.

    :class:`QuadratureRule` instances are supposed to be singletons.
    :class:`QuadratureRule` therefore implements instance caching
    in :func:`~mpmath.__new__`.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def clear(self):
        """
        Delete cached node data.
        """
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        Compute nodes for the standard interval `[-1, 1]`. Subclasses
        should probably implement only this method, and use
        :func:`~mpmath.get_nodes` method to retrieve the nodes.
        """
        raise NotImplementedError

    def get_nodes(self, a, b, degree, prec, verbose=False):
        """
        Return nodes for given interval, degree and precision. The
        nodes are retrieved from a cache if already computed;
        otherwise they are computed by calling :func:`~mpmath.calc_nodes`
        and are then cached.

        Subclasses should probably not implement this method,
        but just implement :func:`~mpmath.calc_nodes` for the actual
        node computation.
        """
        key = (a, b, degree, prec)
        if key in self.transformed_cache:
            return self.transformed_cache[key]
        orig = self.ctx.prec
        try:
            self.ctx.prec = prec+20
            # Get nodes on standard interval
            if (degree, prec) in self.standard_cache:
                nodes = self.standard_cache[degree, prec]
            else:
                nodes = self.calc_nodes(degree, prec, verbose)
                self.standard_cache[degree, prec] = nodes
            # Transform to general interval
            nodes = self.transform_nodes(nodes, a, b, verbose)
            if key in self.interval_count:
                self.transformed_cache[key] = nodes
            else:
                self.interval_count[key] = True
        finally:
            self.ctx.prec = orig
        return nodes

    def transform_nodes(self, nodes, a, b, verbose=False):
        r"""
        Rescale standardized nodes (for `[-1, 1]`) to a general
        interval `[a, b]`. For a finite interval, a simple linear
        change of variables is used. Otherwise, the following
        transformations are used:

        .. math ::

            \lbrack a, \infty \rbrack : t = \frac{1}{x} + (a-1)

            \lbrack -\infty, b \rbrack : t = (b+1) - \frac{1}{x}

            \lbrack -\infty, \infty \rbrack : t = \frac{x}{\sqrt{1-x^2}}

        """
        ctx = self.ctx
        a = ctx.convert(a)
        b = ctx.convert(b)
        one = ctx.one
        if (a, b) == (-one, one):
            return nodes
        half = ctx.mpf(0.5)
        new_nodes = []
        if ctx.isinf(a) or ctx.isinf(b):
            if (a, b) == (ctx.ninf, ctx.inf):
                p05 = -half
                for x, w in nodes:
                    x2 = x*x
                    px1 = one-x2
                    spx1 = px1**p05
                    x = x*spx1
                    w *= spx1/px1
                    new_nodes.append((x, w))
            elif a == ctx.ninf:
                b1 = b+1
                for x, w in nodes:
                    u = 2/(x+one)
                    x = b1-u
                    w *= half*u**2
                    new_nodes.append((x, w))
            elif b == ctx.inf:
                a1 = a-1
                for x, w in nodes:
                    u = 2/(x+one)
                    x = a1+u
                    w *= half*u**2
                    new_nodes.append((x, w))
            elif a == ctx.inf or b == ctx.ninf:
                return [(x,-w) for (x,w) in self.transform_nodes(nodes, b, a, verbose)]
            else:
                raise NotImplementedError
        else:
            # Simple linear change of variables
            C = (b-a)/2
            D = (b+a)/2
            for x, w in nodes:
                new_nodes.append((D+C*x, C*w))
        return new_nodes

    def guess_degree(self, prec):
        """
        Given a desired precision `p` in bits, estimate the degree `m`
        of the quadrature required to accomplish full accuracy for
        typical integrals. By default, :func:`~mpmath.quad` will perform up
        to `m` iterations. The value of `m` should be a slight
        overestimate, so that "slightly bad" integrals can be dealt
        with automatically using a few extra iterations. On the
        other hand, it should not be too big, so :func:`~mpmath.quad` can
        quit within a reasonable amount of time when it is given
        an "unsolvable" integral.

        The default formula used by :func:`~mpmath.guess_degree` is tuned
        for both :class:`TanhSinh` and :class:`GaussLegendre`.
        The output is roughly as follows:

            +---------+---------+
            | `p`     | `m`     |
            +=========+=========+
            | 50      | 6       |
            +---------+---------+
            | 100     | 7       |
            +---------+---------+
            | 500     | 10      |
            +---------+---------+
            | 3000    | 12      |
            +---------+---------+

        This formula is based purely on a limited amount of
        experimentation and will sometimes be wrong.
        """
        # Expected degree
        # XXX: use mag
        g = int(4 + max(0, self.ctx.log(prec/30.0, 2)))
        # Reasonable "worst case"
        g += 2
        return g

    def estimate_error(self, results, prec, epsilon):
        r"""
        Given results from integrations `[I_1, I_2, \ldots, I_k]` done
        with a quadrature of rule of degree `1, 2, \ldots, k`, estimate
        the error of `I_k`.

        For `k = 2`, we estimate  `|I_{\infty}-I_2|` as `|I_2-I_1|`.

        For `k > 2`, we extrapolate `|I_{\infty}-I_k| \approx |I_{k+1}-I_k|`
        from `|I_k-I_{k-1}|` and `|I_k-I_{k-2}|` under the assumption
        that each degree increment roughly doubles the accuracy of
        the quadrature rule (this is true for both :class:`TanhSinh`
        and :class:`GaussLegendre`). The extrapolation formula is given
        by Borwein, Bailey & Girgensohn. Although not very conservative,
        this method seems to be very robust in practice.
        """
        if len(results) == 2:
            return abs(results[0]-results[1])
        try:
            if results[-1] == results[-2] == results[-3]:
                return self.ctx.zero
            D1 = self.ctx.log(abs(results[-1]-results[-2]), 10)
            D2 = self.ctx.log(abs(results[-1]-results[-3]), 10)
        except ValueError:
            return epsilon
        D3 = -prec
        D4 = min(0, max(D1**2/D2, 2*D1, D3))
        return self.ctx.mpf(10) ** int(D4)

    def summation(self, f, points, prec, epsilon, max_degree, verbose=False):
        """
        Main integration function. Computes the 1D integral over
        the interval specified by *points*. For each subinterval,
        performs quadrature of degree from 1 up to *max_degree*
        until :func:`~mpmath.estimate_error` signals convergence.

        :func:`~mpmath.summation` transforms each subintegration to
        the standard interval and then calls :func:`~mpmath.sum_next`.
        """
        ctx = self.ctx
        I = total_err = ctx.zero
        for i in xrange(len(points)-1):
            a, b = points[i], points[i+1]
            if a == b:
                continue
            # XXX: we could use a single variable transformation,
            # but this is not good in practice. We get better accuracy
            # by having 0 as an endpoint.
            if (a, b) == (ctx.ninf, ctx.inf):
                _f = f
                f = lambda x: _f(-x) + _f(x)
                a, b = (ctx.zero, ctx.inf)
            results = []
            err = ctx.zero
            for degree in xrange(1, max_degree+1):
                nodes = self.get_nodes(a, b, degree, prec, verbose)
                if verbose:
                    print("Integrating from %s to %s (degree %s of %s)" % \
                        (ctx.nstr(a), ctx.nstr(b), degree, max_degree))
                result = self.sum_next(f, nodes, degree, prec, results, verbose)
                results.append(result)
                if degree > 1:
                    err = self.estimate_error(results, prec, epsilon)
                    if verbose:
                        print("Estimated error:", ctx.nstr(err), " epsilon:", ctx.nstr(epsilon), " result: ", ctx.nstr(result))
                    if err <= epsilon:
                        break
            I += results[-1]
            total_err += err
        if total_err > epsilon:
            if verbose:
                print("Failed to reach full accuracy. Estimated error:", ctx.nstr(total_err))
        return I, total_err

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        r"""
        Evaluates the step sum `\sum w_k f(x_k)` where the *nodes* list
        contains the `(w_k, x_k)` pairs.

        :func:`~mpmath.summation` will supply the list *results* of
        values computed by :func:`~mpmath.sum_next` at previous degrees, in
        case the quadrature rule is able to reuse them.
        """
        return self.ctx.fdot((w, f(x)) for (x,w) in nodes)


class TanhSinh(QuadratureRule):
    r"""
    This class implements "tanh-sinh" or "doubly exponential"
    quadrature. This quadrature rule is based on the Euler-Maclaurin
    integral formula. By performing a change of variables involving
    nested exponentials / hyperbolic functions (hence the name), the
    derivatives at the endpoints vanish rapidly. Since the error term
    in the Euler-Maclaurin formula depends on the derivatives at the
    endpoints, a simple step sum becomes extremely accurate. In
    practice, this means that doubling the number of evaluation
    points roughly doubles the number of accurate digits.

    Comparison to Gauss-Legendre:
      * Initial computation of nodes is usually faster
      * Handles endpoint singularities better
      * Handles infinite integration intervals better
      * Is slower for smooth integrands once nodes have been computed

    The implementation of the tanh-sinh algorithm is based on the
    description given in Borwein, Bailey & Girgensohn, "Experimentation
    in Mathematics - Computational Paths to Discovery", A K Peters,
    2003, pages 312-313. In the present implementation, a few
    improvements have been made:

      * A more efficient scheme is used to compute nodes (exploiting
        recurrence for the exponential function)
      * The nodes are computed successively instead of all at once

    **References**

    * [Bailey]_
    * http://users.cs.dal.ca/~jborwein/tanh-sinh.pdf

    """

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        """
        Step sum for tanh-sinh quadrature of degree `m`. We exploit the
        fact that half of the abscissas at degree `m` are precisely the
        abscissas from degree `m-1`. Thus reusing the result from
        the previous level allows a 2x speedup.
        """
        h = self.ctx.mpf(2)**(-degree)
        # Abscissas overlap, so reusing saves half of the time
        if previous:
            S = previous[-1]/(h*2)
        else:
            S = self.ctx.zero
        S += self.ctx.fdot((w,f(x)) for (x,w) in nodes)
        return h*S

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        The abscissas and weights for tanh-sinh quadrature of degree
        `m` are given by

        .. math::

            x_k = \tanh(\pi/2 \sinh(t_k))

            w_k = \pi/2 \cosh(t_k) / \cosh(\pi/2 \sinh(t_k))^2

        where `t_k = t_0 + hk` for a step length `h \sim 2^{-m}`. The
        list of nodes is actually infinite, but the weights die off so
        rapidly that only a few are needed.
        """
        ctx = self.ctx
        nodes = []

        extra = 20
        ctx.prec += extra
        tol = ctx.ldexp(1, -prec-10)
        pi4 = ctx.pi/4

        # For simplicity, we work in steps h = 1/2^n, with the first point
        # offset so that we can reuse the sum from the previous degree

        # We define degree 1 to include the "degree 0" steps, including
        # the point x = 0. (It doesn't work well otherwise; not sure why.)
        t0 = ctx.ldexp(1, -degree)
        if degree == 1:
            #nodes.append((mpf(0), pi4))
            #nodes.append((-mpf(0), pi4))
            nodes.append((ctx.zero, ctx.pi/2))
            h = t0
        else:
            h = t0*2

        # Since h is fixed, we can compute the next exponential
        # by simply multiplying by exp(h)
        expt0 = ctx.exp(t0)
        a = pi4 * expt0
        b = pi4 / expt0
        udelta = ctx.exp(h)
        urdelta = 1/udelta

        for k in xrange(0, 20*2**degree+1):
            # Reference implementation:
            # t = t0 + k*h
            # x = tanh(pi/2 * sinh(t))
            # w = pi/2 * cosh(t) / cosh(pi/2 * sinh(t))**2

            # Fast implementation. Note that c = exp(pi/2 * sinh(t))
            c = ctx.exp(a-b)
            d = 1/c
            co = (c+d)/2
            si = (c-d)/2
            x = si / co
            w = (a+b) / co**2
            diff = abs(x-1)
            if diff <= tol:
                break

            nodes.append((x, w))
            nodes.append((-x, w))

            a *= udelta
            b *= urdelta

            if verbose and k % 300 == 150:
                # Note: the number displayed is rather arbitrary. Should
                # figure out how to print something that looks more like a
                # percentage
                print("Calculating nodes:", ctx.nstr(-ctx.log(diff, 10) / prec))

        ctx.prec -= extra
        return nodes


class GaussLegendre(QuadratureRule):
    r"""
    This class implements Gauss-Legendre quadrature, which is
    exceptionally efficient for polynomials and polynomial-like (i.e.
    very smooth) integrands.

    The abscissas and weights are given by roots and values of
    Legendre polynomials, which are the orthogonal polynomials
    on `[-1, 1]` with respect to the unit weight
    (see :func:`~mpmath.legendre`).

    In this implementation, we take the "degree" `m` of the quadrature
    to denote a Gauss-Legendre rule of degree `3 \cdot 2^m` (following
    Borwein, Bailey & Girgensohn). This way we get quadratic, rather
    than linear, convergence as the degree is incremented.

    Comparison to tanh-sinh quadrature:
      * Is faster for smooth integrands once nodes have been computed
      * Initial computation of nodes is usually slower
      * Handles endpoint singularities worse
      * Handles infinite integration intervals worse

    """

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        Calculates the abscissas and weights for Gauss-Legendre
        quadrature of degree of given degree (actually `3 \cdot 2^m`).
        """
        ctx = self.ctx
        # It is important that the epsilon is set lower than the
        # "real" epsilon
        epsilon = ctx.ldexp(1, -prec-8)
        # Fairly high precision might be required for accurate
        # evaluation of the roots
        orig = ctx.prec
        ctx.prec = int(prec*1.5)
        if degree == 1:
            x = ctx.sqrt(ctx.mpf(3)/5)
            w = ctx.mpf(5)/9
            nodes = [(-x,w),(ctx.zero,ctx.mpf(8)/9),(x,w)]
            ctx.prec = orig
            return nodes
        nodes = []
        n = 3*2**(degree-1)
        upto = n//2 + 1
        for j in xrange(1, upto):
            # Asymptotic formula for the roots
            r = ctx.mpf(math.cos(math.pi*(j-0.25)/(n+0.5)))
            # Newton iteration
            while 1:
                t1, t2 = 1, 0
                # Evaluates the Legendre polynomial using its defining
                # recurrence relation
                for j1 in xrange(1,n+1):
                    t3, t2, t1 = t2, t1, ((2*j1-1)*r*t1 - (j1-1)*t2)/j1
                t4 = n*(r*t1-t2)/(r**2-1)
                a = t1/t4
                r = r - a
                if abs(a) < epsilon:
                    break
            x = r
            w = 2/((1-r**2)*t4**2)
            if verbose  and j % 30 == 15:
                print("Computing nodes (%i of %i)" % (j, upto))
            nodes.append((x, w))
            nodes.append((-x, w))
        ctx.prec = orig
        return nodes

class QuadratureMethods(object):

    def __init__(ctx, *args, **kwargs):
        ctx._gauss_legendre = GaussLegendre(ctx)
        ctx._tanh_sinh = TanhSinh(ctx)

    def quad(ctx, f, *points, **kwargs):
        r"""
        Computes a single, double or triple integral over a given
        1D interval, 2D rectangle, or 3D cuboid. A basic example::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(sin, [0, pi])
            2.0

        A basic 2D integral::

            >>> f = lambda x, y: cos(x+y/2)
            >>> quad(f, [-pi/2, pi/2], [0, pi])
            4.0

        **Interval format**

        The integration range for each dimension may be specified
        using a list or tuple. Arguments are interpreted as follows:

        ``quad(f, [x1, x2])`` -- calculates
        `\int_{x_1}^{x_2} f(x) \, dx`

        ``quad(f, [x1, x2], [y1, y2])`` -- calculates
        `\int_{x_1}^{x_2} \int_{y_1}^{y_2} f(x,y) \, dy \, dx`

        ``quad(f, [x1, x2], [y1, y2], [z1, z2])`` -- calculates
        `\int_{x_1}^{x_2} \int_{y_1}^{y_2} \int_{z_1}^{z_2} f(x,y,z)
        \, dz \, dy \, dx`

        Endpoints may be finite or infinite. An interval descriptor
        may also contain more than two points. In this
        case, the integration is split into subintervals, between
        each pair of consecutive points. This is useful for
        dealing with mid-interval discontinuities, or integrating
        over large intervals where the function is irregular or
        oscillates.

        **Options**

        :func:`~mpmath.quad` recognizes the following keyword arguments:

        *method*
            Chooses integration algorithm (described below).
        *error*
            If set to true, :func:`~mpmath.quad` returns `(v, e)` where `v` is the
            integral and `e` is the estimated error.
        *maxdegree*
            Maximum degree of the quadrature rule to try before
            quitting.
        *verbose*
            Print details about progress.

        **Algorithms**

        Mpmath presently implements two integration algorithms: tanh-sinh
        quadrature and Gauss-Legendre quadrature. These can be selected
        using *method='tanh-sinh'* or *method='gauss-legendre'* or by
        passing the classes *method=TanhSinh*, *method=GaussLegendre*.
        The functions :func:`~mpmath.quadts` and :func:`~mpmath.quadgl` are also available
        as shortcuts.

        Both algorithms have the property that doubling the number of
        evaluation points roughly doubles the accuracy, so both are ideal
        for high precision quadrature (hundreds or thousands of digits).

        At high precision, computing the nodes and weights for the
        integration can be expensive (more expensive than computing the
        function values). To make repeated integrations fast, nodes
        are automatically cached.

        The advantages of the tanh-sinh algorithm are that it tends to
        handle endpoint singularities well, and that the nodes are cheap
        to compute on the first run. For these reasons, it is used by
        :func:`~mpmath.quad` as the default algorithm.

        Gauss-Legendre quadrature often requires fewer function
        evaluations, and is therefore often faster for repeated use, but
        the algorithm does not handle endpoint singularities as well and
        the nodes are more expensive to compute. Gauss-Legendre quadrature
        can be a better choice if the integrand is smooth and repeated
        integrations are required (e.g. for multiple integrals).

        See the documentation for :class:`TanhSinh` and
        :class:`GaussLegendre` for additional details.

        **Examples of 1D integrals**

        Intervals may be infinite or half-infinite. The following two
        examples evaluate the limits of the inverse tangent function
        (`\int 1/(1+x^2) = \tan^{-1} x`), and the Gaussian integral
        `\int_{\infty}^{\infty} \exp(-x^2)\,dx = \sqrt{\pi}`::

            >>> mp.dps = 15
            >>> quad(lambda x: 2/(x**2+1), [0, inf])
            3.14159265358979
            >>> quad(lambda x: exp(-x**2), [-inf, inf])**2
            3.14159265358979

        Integrals can typically be resolved to high precision.
        The following computes 50 digits of `\pi` by integrating the
        area of the half-circle defined by `x^2 + y^2 \le 1`,
        `-1 \le x \le 1`, `y \ge 0`::

            >>> mp.dps = 50
            >>> 2*quad(lambda x: sqrt(1-x**2), [-1, 1])
            3.1415926535897932384626433832795028841971693993751

        One can just as well compute 1000 digits (output truncated)::

            >>> mp.dps = 1000
            >>> 2*quad(lambda x: sqrt(1-x**2), [-1, 1])  #doctest:+ELLIPSIS
            3.141592653589793238462643383279502884...216420199

        Complex integrals are supported. The following computes
        a residue at `z = 0` by integrating counterclockwise along the
        diamond-shaped path from `1` to `+i` to `-1` to `-i` to `1`::

            >>> mp.dps = 15
            >>> chop(quad(lambda z: 1/z, [1,j,-1,-j,1]))
            (0.0 + 6.28318530717959j)

        **Examples of 2D and 3D integrals**

        Here are several nice examples of analytically solvable
        2D integrals (taken from MathWorld [1]) that can be evaluated
        to high precision fairly rapidly by :func:`~mpmath.quad`::

            >>> mp.dps = 30
            >>> f = lambda x, y: (x-1)/((1-x*y)*log(x*y))
            >>> quad(f, [0, 1], [0, 1])
            0.577215664901532860606512090082
            >>> +euler
            0.577215664901532860606512090082

            >>> f = lambda x, y: 1/sqrt(1+x**2+y**2)
            >>> quad(f, [-1, 1], [-1, 1])
            3.17343648530607134219175646705
            >>> 4*log(2+sqrt(3))-2*pi/3
            3.17343648530607134219175646705

            >>> f = lambda x, y: 1/(1-x**2 * y**2)
            >>> quad(f, [0, 1], [0, 1])
            1.23370055013616982735431137498
            >>> pi**2 / 8
            1.23370055013616982735431137498

            >>> quad(lambda x, y: 1/(1-x*y), [0, 1], [0, 1])
            1.64493406684822643647241516665
            >>> pi**2 / 6
            1.64493406684822643647241516665

        Multiple integrals may be done over infinite ranges::

            >>> mp.dps = 15
            >>> print(quad(lambda x,y: exp(-x-y), [0, inf], [1, inf]))
            0.367879441171442
            >>> print(1/e)
            0.367879441171442

        For nonrectangular areas, one can call :func:`~mpmath.quad` recursively.
        For example, we can replicate the earlier example of calculating
        `\pi` by integrating over the unit-circle, and actually use double
        quadrature to actually measure the area circle::

            >>> f = lambda x: quad(lambda y: 1, [-sqrt(1-x**2), sqrt(1-x**2)])
            >>> quad(f, [-1, 1])
            3.14159265358979

        Here is a simple triple integral::

            >>> mp.dps = 15
            >>> f = lambda x,y,z: x*y/(1+z)
            >>> quad(f, [0,1], [0,1], [1,2], method='gauss-legendre')
            0.101366277027041
            >>> (log(3)-log(2))/4
            0.101366277027041

        **Singularities**

        Both tanh-sinh and Gauss-Legendre quadrature are designed to
        integrate smooth (infinitely differentiable) functions. Neither
        algorithm copes well with mid-interval singularities (such as
        mid-interval discontinuities in `f(x)` or `f'(x)`).
        The best solution is to split the integral into parts::

            >>> mp.dps = 15
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])   # Bad
            3.99900894176779
            >>> quad(lambda x: abs(sin(x)), [0, pi, 2*pi])  # Good
            4.0

        The tanh-sinh rule often works well for integrands having a
        singularity at one or both endpoints::

            >>> mp.dps = 15
            >>> quad(log, [0, 1], method='tanh-sinh')  # Good
            -1.0
            >>> quad(log, [0, 1], method='gauss-legendre')  # Bad
            -0.999932197413801

        However, the result may still be inaccurate for some functions::

            >>> quad(lambda x: 1/sqrt(x), [0, 1], method='tanh-sinh')
            1.99999999946942

        This problem is not due to the quadrature rule per se, but to
        numerical amplification of errors in the nodes. The problem can be
        circumvented by temporarily increasing the precision::

            >>> mp.dps = 30
            >>> a = quad(lambda x: 1/sqrt(x), [0, 1], method='tanh-sinh')
            >>> mp.dps = 15
            >>> +a
            2.0

        **Highly variable functions**

        For functions that are smooth (in the sense of being infinitely
        differentiable) but contain sharp mid-interval peaks or many
        "bumps", :func:`~mpmath.quad` may fail to provide full accuracy. For
        example, with default settings, :func:`~mpmath.quad` is able to integrate
        `\sin(x)` accurately over an interval of length 100 but not over
        length 1000::

            >>> quad(sin, [0, 100]); 1-cos(100)   # Good
            0.137681127712316
            0.137681127712316
            >>> quad(sin, [0, 1000]); 1-cos(1000)   # Bad
            -37.8587612408485
            0.437620923709297

        One solution is to break the integration into 10 intervals of
        length 100::

            >>> quad(sin, linspace(0, 1000, 10))   # Good
            0.437620923709297

        Another is to increase the degree of the quadrature::

            >>> quad(sin, [0, 1000], maxdegree=10)   # Also good
            0.437620923709297

        Whether splitting the interval or increasing the degree is
        more efficient differs from case to case. Another example is the
        function `1/(1+x^2)`, which has a sharp peak centered around
        `x = 0`::

            >>> f = lambda x: 1/(1+x**2)
            >>> quad(f, [-100, 100])   # Bad
            3.64804647105268
            >>> quad(f, [-100, 100], maxdegree=10)   # Good
            3.12159332021646
            >>> quad(f, [-100, 0, 100])   # Also good
            3.12159332021646

        **References**

        1. http://mathworld.wolfram.com/DoubleIntegral.html

        """
        rule = kwargs.get('method', 'tanh-sinh')
        if type(rule) is str:
            if rule == 'tanh-sinh':
                rule = ctx._tanh_sinh
            elif rule == 'gauss-legendre':
                rule = ctx._gauss_legendre
            else:
                raise ValueError("unknown quadrature rule: %s" % rule)
        else:
            rule = rule(ctx)
        verbose = kwargs.get('verbose')
        dim = len(points)
        orig = prec = ctx.prec
        epsilon = ctx.eps/8
        m = kwargs.get('maxdegree') or rule.guess_degree(prec)
        points = [ctx._as_points(p) for p in points]
        try:
            ctx.prec += 20
            if dim == 1:
                v, err = rule.summation(f, points[0], prec, epsilon, m, verbose)
            elif dim == 2:
                v, err = rule.summation(lambda x: \
                        rule.summation(lambda y: f(x,y), \
                        points[1], prec, epsilon, m)[0],
                    points[0], prec, epsilon, m, verbose)
            elif dim == 3:
                v, err = rule.summation(lambda x: \
                        rule.summation(lambda y: \
                            rule.summation(lambda z: f(x,y,z), \
                            points[2], prec, epsilon, m)[0],
                        points[1], prec, epsilon, m)[0],
                    points[0], prec, epsilon, m, verbose)
            else:
                raise NotImplementedError("quadrature must have dim 1, 2 or 3")
        finally:
            ctx.prec = orig
        if kwargs.get("error"):
            return +v, err
        return +v

    def quadts(ctx, *args, **kwargs):
        """
        Performs tanh-sinh quadrature. The call

            quadts(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=TanhSinh)

        For example, a single integral and a double integral:

            quadts(lambda x: exp(cos(x)), [0, 1])
            quadts(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
        kwargs['method'] = 'tanh-sinh'
        return ctx.quad(*args, **kwargs)

    def quadgl(ctx, *args, **kwargs):
        """
        Performs Gauss-Legendre quadrature. The call

            quadgl(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=GaussLegendre)

        For example, a single integral and a double integral:

            quadgl(lambda x: exp(cos(x)), [0, 1])
            quadgl(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
        kwargs['method'] = 'gauss-legendre'
        return ctx.quad(*args, **kwargs)

    def quadosc(ctx, f, interval, omega=None, period=None, zeros=None):
        r"""
        Calculates

        .. math ::

            I = \int_a^b f(x) dx

        where at least one of `a` and `b` is infinite and where
        `f(x) = g(x) \cos(\omega x  + \phi)` for some slowly
        decreasing function `g(x)`. With proper input, :func:`~mpmath.quadosc`
        can also handle oscillatory integrals where the oscillation
        rate is different from a pure sine or cosine wave.

        In the standard case when `|a| < \infty, b = \infty`,
        :func:`~mpmath.quadosc` works by evaluating the infinite series

        .. math ::

            I = \int_a^{x_1} f(x) dx +
            \sum_{k=1}^{\infty} \int_{x_k}^{x_{k+1}} f(x) dx

        where `x_k` are consecutive zeros (alternatively
        some other periodic reference point) of `f(x)`.
        Accordingly, :func:`~mpmath.quadosc` requires information about the
        zeros of `f(x)`. For a periodic function, you can specify
        the zeros by either providing the angular frequency `\omega`
        (*omega*) or the *period* `2 \pi/\omega`. In general, you can
        specify the `n`-th zero by providing the *zeros* arguments.
        Below is an example of each::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = lambda x: sin(3*x)/(x**2+1)
            >>> quadosc(f, [0,inf], omega=3)
            0.37833007080198
            >>> quadosc(f, [0,inf], period=2*pi/3)
            0.37833007080198
            >>> quadosc(f, [0,inf], zeros=lambda n: pi*n/3)
            0.37833007080198
            >>> (ei(3)*exp(-3)-exp(3)*ei(-3))/2  # Computed by Mathematica
            0.37833007080198

        Note that *zeros* was specified to multiply `n` by the
        *half-period*, not the full period. In theory, it does not matter
        whether each partial integral is done over a half period or a full
        period. However, if done over half-periods, the infinite series
        passed to :func:`~mpmath.nsum` becomes an *alternating series* and this
        typically makes the extrapolation much more efficient.

        Here is an example of an integration over the entire real line,
        and a half-infinite integration starting at `-\infty`::

            >>> quadosc(lambda x: cos(x)/(1+x**2), [-inf, inf], omega=1)
            1.15572734979092
            >>> pi/e
            1.15572734979092
            >>> quadosc(lambda x: cos(x)/x**2, [-inf, -1], period=2*pi)
            -0.0844109505595739
            >>> cos(1)+si(1)-pi/2
            -0.0844109505595738

        Of course, the integrand may contain a complex exponential just as
        well as a real sine or cosine::

            >>> quadosc(lambda x: exp(3*j*x)/(1+x**2), [-inf,inf], omega=3)
            (0.156410688228254 + 0.0j)
            >>> pi/e**3
            0.156410688228254
            >>> quadosc(lambda x: exp(3*j*x)/(2+x+x**2), [-inf,inf], omega=3)
            (0.00317486988463794 - 0.0447701735209082j)
            >>> 2*pi/sqrt(7)/exp(3*(j+sqrt(7))/2)
            (0.00317486988463794 - 0.0447701735209082j)

        **Non-periodic functions**

        If `f(x) = g(x) h(x)` for some function `h(x)` that is not
        strictly periodic, *omega* or *period* might not work, and it might
        be necessary to use *zeros*.

        A notable exception can be made for Bessel functions which, though not
        periodic, are "asymptotically periodic" in a sufficiently strong sense
        that the sum extrapolation will work out::

            >>> quadosc(j0, [0, inf], period=2*pi)
            1.0
            >>> quadosc(j1, [0, inf], period=2*pi)
            1.0

        More properly, one should provide the exact Bessel function zeros::

            >>> j0zero = lambda n: findroot(j0, pi*(n-0.25))
            >>> quadosc(j0, [0, inf], zeros=j0zero)
            1.0

        For an example where *zeros* becomes necessary, consider the
        complete Fresnel integrals

        .. math ::

            \int_0^{\infty} \cos x^2\,dx = \int_0^{\infty} \sin x^2\,dx
            = \sqrt{\frac{\pi}{8}}.

        Although the integrands do not decrease in magnitude as
        `x \to \infty`, the integrals are convergent since the oscillation
        rate increases (causing consecutive periods to asymptotically
        cancel out). These integrals are virtually impossible to calculate
        to any kind of accuracy using standard quadrature rules. However,
        if one provides the correct asymptotic distribution of zeros
        (`x_n \sim \sqrt{n}`), :func:`~mpmath.quadosc` works::

            >>> mp.dps = 30
            >>> f = lambda x: cos(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> f = lambda x: sin(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> sqrt(pi/8)
            0.626657068657750125603941321203

        (Interestingly, these integrals can still be evaluated if one
        places some other constant than `\pi` in the square root sign.)

        In general, if `f(x) \sim g(x) \cos(h(x))`, the zeros follow
        the inverse-function distribution `h^{-1}(x)`::

            >>> mp.dps = 15
            >>> f = lambda x: sin(exp(x))
            >>> quadosc(f, [1,inf], zeros=lambda n: log(n))
            -0.25024394235267
            >>> pi/2-si(e)
            -0.250243942352671

        **Non-alternating functions**

        If the integrand oscillates around a positive value, without
        alternating signs, the extrapolation might fail. A simple trick
        that sometimes works is to multiply or divide the frequency by 2::

            >>> f = lambda x: 1/x**2+sin(x)/x**4
            >>> quadosc(f, [1,inf], omega=1)  # Bad
            1.28642190869861
            >>> quadosc(f, [1,inf], omega=0.5)  # Perfect
            1.28652953559617
            >>> 1+(cos(1)+ci(1)+sin(1))/6
            1.28652953559617

        **Fast decay**

        :func:`~mpmath.quadosc` is primarily useful for slowly decaying
        integrands. If the integrand decreases exponentially or faster,
        :func:`~mpmath.quad` will likely handle it without trouble (and generally be
        much faster than :func:`~mpmath.quadosc`)::

            >>> quadosc(lambda x: cos(x)/exp(x), [0, inf], omega=1)
            0.5
            >>> quad(lambda x: cos(x)/exp(x), [0, inf])
            0.5

        """
        a, b = ctx._as_points(interval)
        a = ctx.convert(a)
        b = ctx.convert(b)
        if [omega, period, zeros].count(None) != 2:
            raise ValueError( \
                "must specify exactly one of omega, period, zeros")
        if a == ctx.ninf and b == ctx.inf:
            s1 = ctx.quadosc(f, [a, 0], omega=omega, zeros=zeros, period=period)
            s2 = ctx.quadosc(f, [0, b], omega=omega, zeros=zeros, period=period)
            return s1 + s2
        if a == ctx.ninf:
            if zeros:
                return ctx.quadosc(lambda x:f(-x), [-b,-a], lambda n: zeros(-n))
            else:
                return ctx.quadosc(lambda x:f(-x), [-b,-a], omega=omega, period=period)
        if b != ctx.inf:
            raise ValueError("quadosc requires an infinite integration interval")
        if not zeros:
            if omega:
                period = 2*ctx.pi/omega
            zeros = lambda n: n*period/2
        #for n in range(1,10):
        #    p = zeros(n)
        #    if p > a:
        #        break
        #if n >= 9:
        #    raise ValueError("zeros do not appear to be correctly indexed")
        n = 1
        s = ctx.quadgl(f, [a, zeros(n)])
        def term(k):
            return ctx.quadgl(f, [zeros(k), zeros(k+1)])
        s += ctx.nsum(term, [n, ctx.inf])
        return s

    def quadsubdiv(ctx, f, interval, tol=None, maxintervals=None, **kwargs):
        """
        Computes the integral of *f* over the interval or path specified
        by *interval*, using :func:`~mpmath.quad` together with adaptive
        subdivision of the interval.

        This function gives an accurate answer for some integrals where
        :func:`~mpmath.quad` fails::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])
            3.99900894176779
            >>> quadsubdiv(lambda x: abs(sin(x)), [0, 2*pi])
            4.0
            >>> quadsubdiv(sin, [0, 1000])
            0.437620923709297
            >>> quadsubdiv(lambda x: 1/(1+x**2), [-100, 100])
            3.12159332021646
            >>> quadsubdiv(lambda x: ceil(x), [0, 100])
            5050.0
            >>> quadsubdiv(lambda x: sin(x+exp(x)), [0,8])
            0.347400172657248

        The argument *maxintervals* can be set to limit the permissible
        subdivision::

            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=5, error=True)
            (-5.40487904307774, 5.011)
            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=100, error=True)
            (0.631417921866934, 1.10101120134116e-17)

        Subdivision does not guarantee a correct answer since, the error
        estimate on subintervals may be inaccurate::

            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.210802735500549, 1.0001111101e-17)
            >>> mp.dps = 20
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.21080273550054927738, 2.200000001e-24)

        The second answer is correct. We can get an accurate result at lower
        precision by forcing a finer initial subdivision::

            >>> mp.dps = 15
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, linspace(0,1,5))
            0.210802735500549

        The following integral is too oscillatory for convergence, but we can get a
        reasonable estimate::

            >>> v, err = fp.quadsubdiv(lambda x: fp.sin(1/x), [0,1], error=True)
            >>> round(v, 6), round(err, 6)
            (0.504067, 1e-06)
            >>> sin(1) - ci(1)
            0.504067061906928

        """
        queue = []
        for i in range(len(interval)-1):
            queue.append((interval[i], interval[i+1]))
        total = ctx.zero
        total_error = ctx.zero
        if maxintervals is None:
            maxintervals = 10 * ctx.prec
        count = 0
        quad_args = kwargs.copy()
        quad_args["verbose"] = False
        quad_args["error"] = True
        if tol is None:
            tol = +ctx.eps
        orig = ctx.prec
        try:
            ctx.prec += 5
            while queue:
                a, b = queue.pop()
                s, err = ctx.quad(f, [a, b], **quad_args)
                if kwargs.get("verbose"):
                    print("subinterval", count, a, b, err)
                if err < tol or count > maxintervals:
                    total += s
                    total_error += err
                else:
                    count += 1
                    if count == maxintervals and kwargs.get("verbose"):
                        print("warning: number of intervals exceeded maxintervals")
                    if a == -ctx.inf and b == ctx.inf:
                        m = 0
                    elif a == -ctx.inf:
                        m = min(b-1, 2*b)
                    elif b == ctx.inf:
                        m = max(a+1, 2*a)
                    else:
                        m = a + (b - a) / 2
                    queue.append((a, m))
                    queue.append((m, b))
        finally:
            ctx.prec = orig
        if kwargs.get("error"):
            return +total, +total_error
        else:
            return +total

if __name__ == '__main__':
    import doctest
    doctest.testmod()
<|endoftext|>from __future__ import print_function

from copy import copy

from ..libmp.backend import xrange

class OptimizationMethods(object):
    def __init__(ctx):
        pass

##############
# 1D-SOLVERS #
##############

class Newton:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0 close to the root.

    Pro:

    * converges fast
    * sometimes more robust than secant with bad second starting point

    Contra:

    * converges slowly for multiple roots
    * needs first derivative
    * 2 function evaluations per iteration
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
        else:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df

    def __iter__(self):
        f = self.f
        df = self.df
        x0 = self.x0
        while True:
            x1 = x0 - f(x0) / df(x0)
            error = abs(x1 - x0)
            x0 = x1
            yield (x1, error)

class Secant:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0 and x1 close to the root.
    x1 defaults to x0 + 0.25.

    Pro:

    * converges fast

    Contra:

    * converges slowly for multiple roots
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
            self.x1 = self.x0 + 0.25
        elif len(x0) == 2:
            self.x0 = x0[0]
            self.x1 = x0[1]
        else:
            raise ValueError('expected 1 or 2 starting points, got %i' % len(x0))
        self.f = f

    def __iter__(self):
        f = self.f
        x0 = self.x0
        x1 = self.x1
        f0 = f(x0)
        while True:
            f1 = f(x1)
            l = x1 - x0
            if not l:
                break
            s = (f1 - f0) / l
            if not s:
                break
            x0, x1 = x1, x1 - f1/s
            f0 = f1
            yield x1, abs(l)

class MNewton:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting point x0 close to the root.
    Uses modified Newton's method that converges fast regardless of the
    multiplicity of the root.

    Pro:

    * converges fast for multiple roots

    Contra:

    * needs first and second derivative of f
    * 3 function evaluations per iteration
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        if not 'd2f' in kwargs:
            def d2f(x):
                return self.ctx.diff(df, x)
        else:
            d2f = kwargs['df']
        self.d2f = d2f

    def __iter__(self):
        x = self.x0
        f = self.f
        df = self.df
        d2f = self.d2f
        while True:
            prevx = x
            fx = f(x)
            if fx == 0:
                break
            dfx = df(x)
            d2fx = d2f(x)
            # x = x - F(x)/F'(x) with F(x) = f(x)/f'(x)
            x -= fx / (dfx - fx * d2fx / dfx)
            error = abs(x - prevx)
            yield x, error

class Halley:
    """
    1d-solver generating pairs of approximative root and error.

    Needs a starting point x0 close to the root.
    Uses Halley's method with cubic convergence rate.

    Pro:

    * converges even faster the Newton's method
    * useful when computing with *many* digits

    Contra:

    * needs first and second derivative of f
    * 3 function evaluations per iteration
    * converges slowly for multiple roots
    """

    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        if not 'd2f' in kwargs:
            def d2f(x):
                return self.ctx.diff(df, x)
        else:
            d2f = kwargs['df']
        self.d2f = d2f

    def __iter__(self):
        x = self.x0
        f = self.f
        df = self.df
        d2f = self.d2f
        while True:
            prevx = x
            fx = f(x)
            dfx = df(x)
            d2fx = d2f(x)
            x -=  2*fx*dfx / (2*dfx**2 - fx*d2fx)
            error = abs(x - prevx)
            yield x, error

class Muller:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0, x1 and x2 close to the root.
    x1 defaults to x0 + 0.25; x2 to x1 + 0.25.
    Uses Muller's method that converges towards complex roots.

    Pro:

    * converges fast (somewhat faster than secant)
    * can find complex roots

    Contra:

    * converges slowly for multiple roots
    * may have complex values for real starting points and real roots

    http://en.wikipedia.org/wiki/Muller's_method
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
            self.x1 = self.x0 + 0.25
            self.x2 = self.x1 + 0.25
        elif len(x0) == 2:
            self.x0 = x0[0]
            self.x1 = x0[1]
            self.x2 = self.x1 + 0.25
        elif len(x0) == 3:
            self.x0 = x0[0]
            self.x1 = x0[1]
            self.x2 = x0[2]
        else:
            raise ValueError('expected 1, 2 or 3 starting points, got %i'
                             % len(x0))
        self.f = f
        self.verbose = kwargs['verbose']

    def __iter__(self):
        f = self.f
        x0 = self.x0
        x1 = self.x1
        x2 = self.x2
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
        while True:
            # TODO: maybe refactoring with function for divided differences
            # calculate divided differences
            fx2x1 = (fx1 - fx2) / (x1 - x2)
            fx2x0 = (fx0 - fx2) / (x0 - x2)
            fx1x0 = (fx0 - fx1) / (x0 - x1)
            w = fx2x1 + fx2x0 - fx1x0
            fx2x1x0 = (fx1x0 - fx2x1) / (x0 - x2)
            if w == 0 and fx2x1x0 == 0:
                if self.verbose:
                    print('canceled with')
                    print('x0 =', x0, ', x1 =', x1, 'and x2 =', x2)
                break
            x0 = x1
            fx0 = fx1
            x1 = x2
            fx1 = fx2
            # denominator should be as large as possible => choose sign
            r = self.ctx.sqrt(w**2 - 4*fx2*fx2x1x0)
            if abs(w - r) > abs(w + r):
                r = -r
            x2 -= 2*fx2 / (w + r)
            fx2 = f(x2)
            error = abs(x2 - x1)
            yield x2, error

# TODO: consider raising a ValueError when there's no sign change in a and b
class Bisection:
    """
    1d-solver generating pairs of approximative root and error.

    Uses bisection method to find a root of f in [a, b].
    Might fail for multiple roots (needs sign change).

    Pro:

    * robust and reliable

    Contra:

    * converges slowly
    * needs sign change
    """
    maxsteps = 100

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.f = f
        self.a = x0[0]
        self.b = x0[1]

    def __iter__(self):
        f = self.f
        a = self.a
        b = self.b
        l = b - a
        fb = f(b)
        while True:
            m = self.ctx.ldexp(a + b, -1)
            fm = f(m)
            sign = fm * fb
            if sign < 0:
                a = m
            elif sign > 0:
                b = m
                fb = fm
            else:
                yield m, self.ctx.zero
            l /= 2
            yield (a + b)/2, abs(l)

def _getm(method):
    """
    Return a function to calculate m for Illinois-like methods.
    """
    if method == 'illinois':
        def getm(fz, fb):
            return 0.5
    elif method == 'pegasus':
        def getm(fz, fb):
            return fb/(fb + fz)
    elif method == 'anderson':
        def getm(fz, fb):
            m = 1 - fz/fb
            if m > 0:
                return m
            else:
                return 0.5
    else:
        raise ValueError("method '%s' not recognized" % method)
    return getm

class Illinois:
    """
    1d-solver generating pairs of approximative root and error.

    Uses Illinois method or similar to find a root of f in [a, b].
    Might fail for multiple roots (needs sign change).
    Combines bisect with secant (improved regula falsi).

    The only difference between the methods is the scaling factor m, which is
    used to ensure convergence (you can choose one using the 'method' keyword):

    Illinois method ('illinois'):
        m = 0.5

    Pegasus method ('pegasus'):
        m = fb/(fb + fz)

    Anderson-Bjoerk method ('anderson'):
        m = 1 - fz/fb if positive else 0.5

    Pro:

    * converges very fast

    Contra:

    * has problems with multiple roots
    * needs sign change
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.a = x0[0]
        self.b = x0[1]
        self.f = f
        self.tol = kwargs['tol']
        self.verbose = kwargs['verbose']
        self.method = kwargs.get('method', 'illinois')
        self.getm = _getm(self.method)
        if self.verbose:
            print('using %s method' % self.method)

    def __iter__(self):
        method = self.method
        f = self.f
        a = self.a
        b = self.b
        fa = f(a)
        fb = f(b)
        m = None
        while True:
            l = b - a
            if l == 0:
                break
            s = (fb - fa) / l
            z = a - fa/s
            fz = f(z)
            if abs(fz) < self.tol:
                # TODO: better condition (when f is very flat)
                if self.verbose:
                    print('canceled with z =', z)
                yield z, l
                break
            if fz * fb < 0: # root in [z, b]
                a = b
                fa = fb
                b = z
                fb = fz
            else: # root in [a, z]
                m = self.getm(fz, fb)
                b = z
                fb = fz
                fa = m*fa # scale down to ensure convergence
            if self.verbose and m and not method == 'illinois':
                print('m:', m)
            yield (a + b)/2, abs(l)

def Pegasus(*args, **kwargs):
    """
    1d-solver generating pairs of approximative root and error.

    Uses Pegasus method to find a root of f in [a, b].
    Wrapper for illinois to use method='pegasus'.
    """
    kwargs['method'] = 'pegasus'
    return Illinois(*args, **kwargs)

def Anderson(*args, **kwargs):
    """
    1d-solver generating pairs of approximative root and error.

    Uses Anderson-Bjoerk method to find a root of f in [a, b].
    Wrapper for illinois to use method='pegasus'.
    """
    kwargs['method'] = 'anderson'
    return Illinois(*args, **kwargs)

# TODO: check whether it's possible to combine it with Illinois stuff
class Ridder:
    """
    1d-solver generating pairs of approximative root and error.

    Ridders' method to find a root of f in [a, b].
    Is told to perform as well as Brent's method while being simpler.

    Pro:

    * very fast
    * simpler than Brent's method

    Contra:

    * two function evaluations per step
    * has problems with multiple roots
    * needs sign change

    http://en.wikipedia.org/wiki/Ridders'_method
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        self.f = f
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.x1 = x0[0]
        self.x2 = x0[1]
        self.verbose = kwargs['verbose']
        self.tol = kwargs['tol']

    def __iter__(self):
        ctx = self.ctx
        f = self.f
        x1 = self.x1
        fx1 = f(x1)
        x2 = self.x2
        fx2 = f(x2)
        while True:
            x3 = 0.5*(x1 + x2)
            fx3 = f(x3)
            x4 = x3 + (x3 - x1) * ctx.sign(fx1 - fx2) * fx3 / ctx.sqrt(fx3**2 - fx1*fx2)
            fx4 = f(x4)
            if abs(fx4) < self.tol:
                # TODO: better condition (when f is very flat)
                if self.verbose:
                    print('canceled with f(x4) =', fx4)
                yield x4, abs(x1 - x2)
                break
            if fx4 * fx2 < 0: # root in [x4, x2]
                x1 = x4
                fx1 = fx4
            else: # root in [x1, x4]
                x2 = x4
                fx2 = fx4
            error = abs(x1 - x2)
            yield (x1 + x2)/2, error

class ANewton:
    """
    EXPERIMENTAL 1d-solver generating pairs of approximative root and error.

    Uses Newton's method modified to use Steffensens method when convergence is
    slow. (I.e. for multiple roots.)
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        def phi(x):
            return x - f(x) / df(x)
        self.phi = phi
        self.verbose = kwargs['verbose']

    def __iter__(self):
        x0 = self.x0
        f = self.f
        df = self.df
        phi = self.phi
        error = 0
        counter = 0
        while True:
            prevx = x0
            try:
                x0 = phi(x0)
            except ZeroDivisionError:
                if self.verbose:
                    print('ZeroDivisionError: canceled with x =', x0)
                break
            preverror = error
            error = abs(prevx - x0)
            # TODO: decide not to use convergence acceleration
            if error and abs(error - preverror) / error < 1:
                if self.verbose:
                    print('converging slowly')
                counter += 1
            if counter >= 3:
                # accelerate convergence
                phi = steffensen(phi)
                counter = 0
                if self.verbose:
                    print('accelerating convergence')
            yield x0, error

# TODO: add Brent

############################
# MULTIDIMENSIONAL SOLVERS #
############################

def jacobian(ctx, f, x):
    """
    Calculate the Jacobian matrix of a function at the point x0.

    This is the first derivative of a vectorial function:

        f : R^m -> R^n with m >= n
    """
    x = ctx.matrix(x)
    h = ctx.sqrt(ctx.eps)
    fx = ctx.matrix(f(*x))
    m = len(fx)
    n = len(x)
    J = ctx.matrix(m, n)
    for j in xrange(n):
        xj = x.copy()
        xj[j] += h
        Jj = (ctx.matrix(f(*xj)) - fx) / h
        for i in xrange(m):
            J[i,j] = Jj[i]
    return J

# TODO: test with user-specified jacobian matrix
class MDNewton:
    """
    Find the root of a vector function numerically using Newton's method.

    f is a vector function representing a nonlinear equation system.

    x0 is the starting point close to the root.

    J is a function returning the Jacobian matrix for a point.

    Supports overdetermined systems.

    Use the 'norm' keyword to specify which norm to use. Defaults to max-norm.
    The function to calculate the Jacobian matrix can be given using the
    keyword 'J'. Otherwise it will be calculated numerically.

    Please note that this method converges only locally. Especially for high-
    dimensional systems it is not trivial to find a good starting point being
    close enough to the root.

    It is recommended to use a faster, low-precision solver from SciPy [1] or
    OpenOpt [2] to get an initial guess. Afterwards you can use this method for
    root-polishing to any precision.

    [1] http://scipy.org

    [2] http://openopt.org/Welcome
    """
    maxsteps = 10

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        self.f = f
        if isinstance(x0, (tuple, list)):
            x0 = ctx.matrix(x0)
        assert x0.cols == 1, 'need a vector'
        self.x0 = x0
        if 'J' in kwargs:
            self.J = kwargs['J']
        else:
            def J(*x):
                return ctx.jacobian(f, x)
            self.J = J
        self.norm = kwargs['norm']
        self.verbose = kwargs['verbose']

    def __iter__(self):
        f = self.f
        x0 = self.x0
        norm = self.norm
        J = self.J
        fx = self.ctx.matrix(f(*x0))
        fxnorm = norm(fx)
        cancel = False
        while not cancel:
            # get direction of descent
            fxn = -fx
            Jx = J(*x0)
            s = self.ctx.lu_solve(Jx, fxn)
            if self.verbose:
                print('Jx:')
                print(Jx)
                print('s:', s)
            # damping step size TODO: better strategy (hard task)
            l = self.ctx.one
            x1 = x0 + s
            while True:
                if x1 == x0:
                    if self.verbose:
                        print("canceled, won't get more excact")
                    cancel = True
                    break
                fx = self.ctx.matrix(f(*x1))
                newnorm = norm(fx)
                if newnorm < fxnorm:
                    # new x accepted
                    fxnorm = newnorm
                    x0 = x1
                    break
                l /= 2
                x1 = x0 + l*s
            yield (x0, fxnorm)

#############
# UTILITIES #
#############

str2solver = {'newton':Newton, 'secant':Secant, 'mnewton':MNewton,
              'halley':Halley, 'muller':Muller, 'bisect':Bisection,
              'illinois':Illinois, 'pegasus':Pegasus, 'anderson':Anderson,
              'ridder':Ridder, 'anewton':ANewton, 'mdnewton':MDNewton}

def findroot(ctx, f, x0, solver='secant', tol=None, verbose=False, verify=True, **kwargs):
    r"""
    Find an approximate solution to `f(x) = 0`, using *x0* as starting point or
    interval for *x*.

    Multidimensional overdetermined systems are supported.
    You can specify them using a function or a list of functions.

    Mathematically speaking, this function returns `x` such that
    `|f(x)|^2 \leq \mathrm{tol}` is true within the current working precision.
    If the computed value does not meet this criterion, an exception is raised.
    This exception can be disabled with *verify=False*.

    For interval arithmetic (``iv.findroot()``), please note that
    the returned interval ``x`` is not guaranteed to contain `f(x)=0`!
    It is only some `x` for which `|f(x)|^2 \leq \mathrm{tol}` certainly holds
    regardless of numerical error. This may be improved in the future.

    **Arguments**

    *f*
        one dimensional function
    *x0*
        starting point, several starting points or interval (depends on solver)
    *tol*
        the returned solution has an error smaller than this
    *verbose*
        print additional information for each iteration if true
    *verify*
        verify the solution and raise a ValueError if `|f(x)|^2 > \mathrm{tol}`
    *solver*
        a generator for *f* and *x0* returning approximative solution and error
    *maxsteps*
        after how many steps the solver will cancel
    *df*
        first derivative of *f* (used by some solvers)
    *d2f*
        second derivative of *f* (used by some solvers)
    *multidimensional*
        force multidimensional solving
    *J*
        Jacobian matrix of *f* (used by multidimensional solvers)
    *norm*
        used vector norm (used by multidimensional solvers)

    solver has to be callable with ``(f, x0, **kwargs)`` and return an generator
    yielding pairs of approximative solution and estimated error (which is
    expected to be positive).
    You can use the following string aliases:
    'secant', 'mnewton', 'halley', 'muller', 'illinois', 'pegasus', 'anderson',
    'ridder', 'anewton', 'bisect'

    See mpmath.calculus.optimization for their documentation.

    **Examples**

    The function :func:`~mpmath.findroot` locates a root of a given function using the
    secant method by default. A simple example use of the secant method is to
    compute `\pi` as the root of `\sin x` closest to `x_0 = 3`::

        >>> from mpmath import *
        >>> mp.dps = 30; mp.pretty = True
        >>> findroot(sin, 3)
        3.14159265358979323846264338328

    The secant method can be used to find complex roots of analytic functions,
    although it must in that case generally be given a nonreal starting value
    (or else it will never leave the real line)::

        >>> mp.dps = 15
        >>> findroot(lambda x: x**3 + 2*x + 1, j)
        (0.226698825758202 + 1.46771150871022j)

    A nice application is to compute nontrivial roots of the Riemann zeta
    function with many digits (good initial values are needed for convergence)::

        >>> mp.dps = 30
        >>> findroot(zeta, 0.5+14j)
        (0.5 + 14.1347251417346937904572519836j)

    The secant method can also be used as an optimization algorithm, by passing
    it a derivative of a function. The following example locates the positive
    minimum of the gamma function::

        >>> mp.dps = 20
        >>> findroot(lambda x: diff(gamma, x), 1)
        1.4616321449683623413

    Finally, a useful application is to compute inverse functions, such as the
    Lambert W function which is the inverse of `w e^w`, given the first
    term of the solution's asymptotic expansion as the initial value. In basic
    cases, this gives identical results to mpmath's built-in ``lambertw``
    function::

        >>> def lambert(x):
        ...     return findroot(lambda w: w*exp(w) - x, log(1+x))
        ...
        >>> mp.dps = 15
        >>> lambert(1); lambertw(1)
        0.567143290409784
        0.567143290409784
        >>> lambert(1000); lambert(1000)
        5.2496028524016
        5.2496028524016

    Multidimensional functions are also supported::

        >>> f = [lambda x1, x2: x1**2 + x2,
        ...      lambda x1, x2: 5*x1**2 - 3*x1 + 2*x2 - 3]
        >>> findroot(f, (0, 0))
        [-0.618033988749895]
        [-0.381966011250105]
        >>> findroot(f, (10, 10))
        [ 1.61803398874989]
        [-2.61803398874989]

    You can verify this by solving the system manually.

    Please note that the following (more general) syntax also works::

        >>> def f(x1, x2):
        ...     return x1**2 + x2, 5*x1**2 - 3*x1 + 2*x2 - 3
        ...
        >>> findroot(f, (0, 0))
        [-0.618033988749895]
        [-0.381966011250105]


    **Multiple roots**

    For multiple roots all methods of the Newtonian family (including secant)
    converge slowly. Consider this example::

        >>> f = lambda x: (x - 1)**99
        >>> findroot(f, 0.9, verify=False)
        0.918073542444929

    Even for a very close starting point the secant method converges very
    slowly. Use ``verbose=True`` to illustrate this.

    It is possible to modify Newton's method to make it converge regardless of
    the root's multiplicity::

        >>> findroot(f, -10, solver='mnewton')
        1.0

    This variant uses the first and second derivative of the function, which is
    not very efficient.

    Alternatively you can use an experimental Newtonian solver that keeps track
    of the speed of convergence and accelerates it using Steffensen's method if
    necessary::

        >>> findroot(f, -10, solver='anewton', verbose=True)
        x:     -9.88888888888888888889
        error: 0.111111111111111111111
        converging slowly
        x:     -9.77890011223344556678
        error: 0.10998877665544332211
        converging slowly
        x:     -9.67002233332199662166
        error: 0.108877778911448945119
        converging slowly
        accelerating convergence
        x:     -9.5622443299551077669
        error: 0.107778003366888854764
        converging slowly
        x:     0.99999999999999999214
        error: 10.562244329955107759
        x:     1.0
        error: 7.8598304758094664213e-18
        ZeroDivisionError: canceled with x = 1.0
        1.0

    **Complex roots**

    For complex roots it's recommended to use Muller's method as it converges
    even for real starting points very fast::

        >>> findroot(lambda x: x**4 + x + 1, (0, 1, 2), solver='muller')
        (0.727136084491197 + 0.934099289460529j)


    **Intersection methods**

    When you need to find a root in a known interval, it's highly recommended to
    use an intersection-based solver like ``'anderson'`` or ``'ridder'``.
    Usually they converge faster and more reliable. They have however problems
    with multiple roots and usually need a sign change to find a root::

        >>> findroot(lambda x: x**3, (-1, 1), solver='anderson')
        0.0

    Be careful with symmetric functions::

        >>> findroot(lambda x: x**2, (-1, 1), solver='anderson') #doctest:+ELLIPSIS
        Traceback (most recent call last):
          ...
        ZeroDivisionError

    It fails even for better starting points, because there is no sign change::

        >>> findroot(lambda x: x**2, (-1, .5), solver='anderson')
        Traceback (most recent call last):
          ...
        ValueError: Could not find root within given tolerance. (1.0 > 2.16840434497100886801e-19)
        Try another starting point or tweak arguments.

    """
    prec = ctx.prec
    try:
        ctx.prec += 20

        # initialize arguments
        if tol is None:
            tol = ctx.eps * 2**10

        kwargs['verbose'] = kwargs.get('verbose', verbose)

        if 'd1f' in kwargs:
            kwargs['df'] = kwargs['d1f']

        kwargs['tol'] = tol
        if isinstance(x0, (list, tuple)):
            x0 = [ctx.convert(x) for x in x0]
        else:
            x0 = [ctx.convert(x0)]

        if isinstance(solver, str):
            try:
                solver = str2solver[solver]
            except KeyError:
                raise ValueError('could not recognize solver')

        # accept list of functions
        if isinstance(f, (list, tuple)):
            f2 = copy(f)
            def tmp(*args):
                return [fn(*args) for fn in f2]
            f = tmp

        # detect multidimensional functions
        try:
            fx = f(*x0)
            multidimensional = isinstance(fx, (list, tuple, ctx.matrix))
        except TypeError:
            fx = f(x0[0])
            multidimensional = False
        if 'multidimensional' in kwargs:
            multidimensional = kwargs['multidimensional']
        if multidimensional:
            # only one multidimensional solver available at the moment
            solver = MDNewton
            if not 'norm' in kwargs:
                norm = lambda x: ctx.norm(x, 'inf')
                kwargs['norm'] = norm
            else:
                norm = kwargs['norm']
        else:
            norm = abs

        # happily return starting point if it's a root
        if norm(fx) == 0:
            if multidimensional:
                return ctx.matrix(x0)
            else:
                return x0[0]

        # use solver
        iterations = solver(ctx, f, x0, **kwargs)
        if 'maxsteps' in kwargs:
            maxsteps = kwargs['maxsteps']
        else:
            maxsteps = iterations.maxsteps
        i = 0
        for x, error in iterations:
            if verbose:
                print('x:    ', x)
                print('error:', error)
            i += 1
            if error < tol * max(1, norm(x)) or i >= maxsteps:
                break
        else:
            if not i:
                raise ValueError('Could not find root using the given solver.\n'
                                 'Try another starting point or tweak arguments.')
        if not isinstance(x, (list, tuple, ctx.matrix)):
            xl = [x]
        else:
            xl = x
        if verify and norm(f(*xl))**2 > tol: # TODO: better condition?
            raise ValueError('Could not find root within given tolerance. '
                             '(%s > %s)\n'
                             'Try another starting point or tweak arguments.'
                             % (norm(f(*xl))**2, tol))
        return x
    finally:
        ctx.prec = prec


def multiplicity(ctx, f, root, tol=None, maxsteps=10, **kwargs):
    """
    Return the multiplicity of a given root of f.

    Internally, numerical derivatives are used. This might be inefficient for
    higher order derviatives. Due to this, ``multiplicity`` cancels after
    evaluating 10 derivatives by default. You can be specify the n-th derivative
    using the dnf keyword.

    >>> from mpmath import *
    >>> multiplicity(lambda x: sin(x) - 1, pi/2)
    2

    """
    if tol is None:
        tol = ctx.eps ** 0.8
    kwargs['d0f'] = f
    for i in xrange(maxsteps):
        dfstr = 'd' + str(i) + 'f'
        if dfstr in kwargs:
            df = kwargs[dfstr]
        else:
            df = lambda x: ctx.diff(f, x, i)
        if not abs(df(root)) < tol:
            break
    return i

def steffensen(f):
    """
    linear convergent function -> quadratic convergent function

