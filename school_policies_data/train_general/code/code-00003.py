    Steffensen's method for quadratic convergence of a linear converging
    sequence.
    Don not use it for higher rates of convergence.
    It may even work for divergent sequences.

    Definition:
    F(x) = (x*f(f(x)) - f(x)**2) / (f(f(x)) - 2*f(x) + x)

    Example
    .......

    You can use Steffensen's method to accelerate a fixpoint iteration of linear
    (or less) convergence.

    x* is a fixpoint of the iteration x_{k+1} = phi(x_k) if x* = phi(x*). For
    phi(x) = x**2 there are two fixpoints: 0 and 1.

    Let's try Steffensen's method:

    >>> f = lambda x: x**2
    >>> from mpmath.calculus.optimization import steffensen
    >>> F = steffensen(f)
    >>> for x in [0.5, 0.9, 2.0]:
    ...     fx = Fx = x
    ...     for i in xrange(9):
    ...         try:
    ...             fx = f(fx)
    ...         except OverflowError:
    ...             pass
    ...         try:
    ...             Fx = F(Fx)
    ...         except ZeroDivisionError:
    ...             pass
    ...         print('%20g  %20g' % (fx, Fx))
                    0.25                  -0.5
                  0.0625                   0.1
              0.00390625            -0.0011236
             1.52588e-05           1.41691e-09
             2.32831e-10          -2.84465e-27
             5.42101e-20           2.30189e-80
             2.93874e-39          -1.2197e-239
             8.63617e-78                     0
            7.45834e-155                     0
                    0.81               1.02676
                  0.6561               1.00134
                0.430467                     1
                0.185302                     1
               0.0343368                     1
              0.00117902                     1
             1.39008e-06                     1
             1.93233e-12                     1
             3.73392e-24                     1
                       4                   1.6
                      16                1.2962
                     256               1.10194
                   65536               1.01659
             4.29497e+09               1.00053
             1.84467e+19                     1
             3.40282e+38                     1
             1.15792e+77                     1
            1.34078e+154                     1

    Unmodified, the iteration converges only towards 0. Modified it converges
    not only much faster, it converges even to the repelling fixpoint 1.
    """
    def F(x):
        fx = f(x)
        ffx = f(fx)
        return (x*ffx - fx**2) / (ffx - 2*fx + x)
    return F

OptimizationMethods.jacobian = jacobian
OptimizationMethods.findroot = findroot
OptimizationMethods.multiplicity = multiplicity

if __name__ == '__main__':
    import doctest
    doctest.testmod()
<|endoftext|>from ..libmp.backend import xrange
from .calculus import defun

#----------------------------------------------------------------------------#
#                              Approximation methods                         #
#----------------------------------------------------------------------------#

# The Chebyshev approximation formula is given at:
# http://mathworld.wolfram.com/ChebyshevApproximationFormula.html

# The only major changes in the following code is that we return the
# expanded polynomial coefficients instead of Chebyshev coefficients,
# and that we automatically transform [a,b] -> [-1,1] and back
# for convenience.

# Coefficient in Chebyshev approximation
def chebcoeff(ctx,f,a,b,j,N):
    s = ctx.mpf(0)
    h = ctx.mpf(0.5)
    for k in range(1, N+1):
        t = ctx.cospi((k-h)/N)
        s += f(t*(b-a)*h + (b+a)*h) * ctx.cospi(j*(k-h)/N)
    return 2*s/N

# Generate Chebyshev polynomials T_n(ax+b) in expanded form
def chebT(ctx, a=1, b=0):
    Tb = [1]
    yield Tb
    Ta = [b, a]
    while 1:
        yield Ta
        # Recurrence: T[n+1](ax+b) = 2*(ax+b)*T[n](ax+b) - T[n-1](ax+b)
        Tmp = [0] + [2*a*t for t in Ta]
        for i, c in enumerate(Ta): Tmp[i] += 2*b*c
        for i, c in enumerate(Tb): Tmp[i] -= c
        Ta, Tb = Tmp, Ta

@defun
def chebyfit(ctx, f, interval, N, error=False):
    r"""
    Computes a polynomial of degree `N-1` that approximates the
    given function `f` on the interval `[a, b]`. With ``error=True``,
    :func:`~mpmath.chebyfit` also returns an accurate estimate of the
    maximum absolute error; that is, the maximum value of
    `|f(x) - P(x)|` for `x \in [a, b]`.

    :func:`~mpmath.chebyfit` uses the Chebyshev approximation formula,
    which gives a nearly optimal solution: that is, the maximum
    error of the approximating polynomial is very close to
    the smallest possible for any polynomial of the same degree.

    Chebyshev approximation is very useful if one needs repeated
    evaluation of an expensive function, such as function defined
    implicitly by an integral or a differential equation. (For
    example, it could be used to turn a slow mpmath function
    into a fast machine-precision version of the same.)

    **Examples**

    Here we use :func:`~mpmath.chebyfit` to generate a low-degree approximation
    of `f(x) = \cos(x)`, valid on the interval `[1, 2]`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> poly, err = chebyfit(cos, [1, 2], 5, error=True)
        >>> nprint(poly)
        [0.00291682, 0.146166, -0.732491, 0.174141, 0.949553]
        >>> nprint(err, 12)
        1.61351758081e-5

    The polynomial can be evaluated using ``polyval``::

        >>> nprint(polyval(poly, 1.6), 12)
        -0.0291858904138
        >>> nprint(cos(1.6), 12)
        -0.0291995223013

    Sampling the true error at 1000 points shows that the error
    estimate generated by ``chebyfit`` is remarkably good::

        >>> error = lambda x: abs(cos(x) - polyval(poly, x))
        >>> nprint(max([error(1+n/1000.) for n in range(1000)]), 12)
        1.61349954245e-5

    **Choice of degree**

    The degree `N` can be set arbitrarily high, to obtain an
    arbitrarily good approximation. As a rule of thumb, an
    `N`-term Chebyshev approximation is good to `N/(b-a)` decimal
    places on a unit interval (although this depends on how
    well-behaved `f` is). The cost grows accordingly: ``chebyfit``
    evaluates the function `(N^2)/2` times to compute the
    coefficients and an additional `N` times to estimate the error.

    **Possible issues**

    One should be careful to use a sufficiently high working
    precision both when calling ``chebyfit`` and when evaluating
    the resulting polynomial, as the polynomial is sometimes
    ill-conditioned. It is for example difficult to reach
    15-digit accuracy when evaluating the polynomial using
    machine precision floats, no matter the theoretical
    accuracy of the polynomial. (The option to return the
    coefficients in Chebyshev form should be made available
    in the future.)

    It is important to note the Chebyshev approximation works
    poorly if `f` is not smooth. A function containing singularities,
    rapid oscillation, etc can be approximated more effectively by
    multiplying it by a weight function that cancels out the
    nonsmooth features, or by dividing the interval into several
    segments.
    """
    a, b = ctx._as_points(interval)
    orig = ctx.prec
    try:
        ctx.prec = orig + int(N**0.5) + 20
        c = [chebcoeff(ctx,f,a,b,k,N) for k in range(N)]
        d = [ctx.zero] * N
        d[0] = -c[0]/2
        h = ctx.mpf(0.5)
        T = chebT(ctx, ctx.mpf(2)/(b-a), ctx.mpf(-1)*(b+a)/(b-a))
        for (k, Tk) in zip(range(N), T):
            for i in range(len(Tk)):
                d[i] += c[k]*Tk[i]
        d = d[::-1]
        # Estimate maximum error
        err = ctx.zero
        for k in range(N):
            x = ctx.cos(ctx.pi*k/N) * (b-a)*h + (b+a)*h
            err = max(err, abs(f(x) - ctx.polyval(d, x)))
    finally:
        ctx.prec = orig
    if error:
        return d, +err
    else:
        return d

@defun
def fourier(ctx, f, interval, N):
    r"""
    Computes the Fourier series of degree `N` of the given function
    on the interval `[a, b]`. More precisely, :func:`~mpmath.fourier` returns
    two lists `(c, s)` of coefficients (the cosine series and sine
    series, respectively), such that

    .. math ::

        f(x) \sim \sum_{k=0}^N
            c_k \cos(k m x) + s_k \sin(k m x)

    where `m = 2 \pi / (b-a)`.

    Note that many texts define the first coefficient as `2 c_0` instead
    of `c_0`. The easiest way to evaluate the computed series correctly
    is to pass it to :func:`~mpmath.fourierval`.

    **Examples**

    The function `f(x) = x` has a simple Fourier series on the standard
    interval `[-\pi, \pi]`. The cosine coefficients are all zero (because
    the function has odd symmetry), and the sine coefficients are
    rational numbers::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> c, s = fourier(lambda x: x, [-pi, pi], 5)
        >>> nprint(c)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> nprint(s)
        [0.0, 2.0, -1.0, 0.666667, -0.5, 0.4]

    This computes a Fourier series of a nonsymmetric function on
    a nonstandard interval::

        >>> I = [-1, 1.5]
        >>> f = lambda x: x**2 - 4*x + 1
        >>> cs = fourier(f, I, 4)
        >>> nprint(cs[0])
        [0.583333, 1.12479, -1.27552, 0.904708, -0.441296]
        >>> nprint(cs[1])
        [0.0, -2.6255, 0.580905, 0.219974, -0.540057]

    It is instructive to plot a function along with its truncated
    Fourier series::

        >>> plot([f, lambda x: fourierval(cs, I, x)], I) #doctest: +SKIP

    Fourier series generally converge slowly (and may not converge
    pointwise). For example, if `f(x) = \cosh(x)`, a 10-term Fourier
    series gives an `L^2` error corresponding to 2-digit accuracy::

        >>> I = [-1, 1]
        >>> cs = fourier(cosh, I, 9)
        >>> g = lambda x: (cosh(x) - fourierval(cs, I, x))**2
        >>> nprint(sqrt(quad(g, I)))
        0.00467963

    :func:`~mpmath.fourier` uses numerical quadrature. For nonsmooth functions,
    the accuracy (and speed) can be improved by including all singular
    points in the interval specification::

        >>> nprint(fourier(abs, [-1, 1], 0), 10)
        ([0.5000441648], [0.0])
        >>> nprint(fourier(abs, [-1, 0, 1], 0), 10)
        ([0.5], [0.0])

    """
    interval = ctx._as_points(interval)
    a = interval[0]
    b = interval[-1]
    L = b-a
    cos_series = []
    sin_series = []
    cutoff = ctx.eps*10
    for n in xrange(N+1):
        m = 2*n*ctx.pi/L
        an = 2*ctx.quadgl(lambda t: f(t)*ctx.cos(m*t), interval)/L
        bn = 2*ctx.quadgl(lambda t: f(t)*ctx.sin(m*t), interval)/L
        if n == 0:
            an /= 2
        if abs(an) < cutoff: an = ctx.zero
        if abs(bn) < cutoff: bn = ctx.zero
        cos_series.append(an)
        sin_series.append(bn)
    return cos_series, sin_series

@defun
def fourierval(ctx, series, interval, x):
    """
    Evaluates a Fourier series (in the format computed by
    by :func:`~mpmath.fourier` for the given interval) at the point `x`.

    The series should be a pair `(c, s)` where `c` is the
    cosine series and `s` is the sine series. The two lists
    need not have the same length.
    """
    cs, ss = series
    ab = ctx._as_points(interval)
    a = interval[0]
    b = interval[-1]
    m = 2*ctx.pi/(ab[-1]-ab[0])
    s = ctx.zero
    s += ctx.fsum(cs[n]*ctx.cos(m*n*x) for n in xrange(len(cs)) if cs[n])
    s += ctx.fsum(ss[n]*ctx.sin(m*n*x) for n in xrange(len(ss)) if ss[n])
    return s
<|endoftext|>from ..libmp.backend import xrange
from .calculus import defun

#----------------------------------------------------------------------------#
#                                Polynomials                                 #
#----------------------------------------------------------------------------#

# XXX: extra precision
@defun
def polyval(ctx, coeffs, x, derivative=False):
    r"""
    Given coefficients `[c_n, \ldots, c_2, c_1, c_0]` and a number `x`,
    :func:`~mpmath.polyval` evaluates the polynomial

    .. math ::

        P(x) = c_n x^n + \ldots + c_2 x^2 + c_1 x + c_0.

    If *derivative=True* is set, :func:`~mpmath.polyval` simultaneously
    evaluates `P(x)` with the derivative, `P'(x)`, and returns the
    tuple `(P(x), P'(x))`.

        >>> from mpmath import *
        >>> mp.pretty = True
        >>> polyval([3, 0, 2], 0.5)
        2.75
        >>> polyval([3, 0, 2], 0.5, derivative=True)
        (2.75, 3.0)

    The coefficients and the evaluation point may be any combination
    of real or complex numbers.
    """
    if not coeffs:
        return ctx.zero
    p = ctx.convert(coeffs[0])
    q = ctx.zero
    for c in coeffs[1:]:
        if derivative:
            q = p + x*q
        p = c + x*p
    if derivative:
        return p, q
    else:
        return p

@defun
def polyroots(ctx, coeffs, maxsteps=50, cleanup=True, extraprec=10,
        error=False, roots_init=None):
    """
    Computes all roots (real or complex) of a given polynomial.

    The roots are returned as a sorted list, where real roots appear first
    followed by complex conjugate roots as adjacent elements. The polynomial
    should be given as a list of coefficients, in the format used by
    :func:`~mpmath.polyval`. The leading coefficient must be nonzero.

    With *error=True*, :func:`~mpmath.polyroots` returns a tuple *(roots, err)*
    where *err* is an estimate of the maximum error among the computed roots.

    **Examples**

    Finding the three real roots of `x^3 - x^2 - 14x + 24`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nprint(polyroots([1,-1,-14,24]), 4)
        [-4.0, 2.0, 3.0]

    Finding the two complex conjugate roots of `4x^2 + 3x + 2`, with an
    error estimate::

        >>> roots, err = polyroots([4,3,2], error=True)
        >>> for r in roots:
        ...     print(r)
        ...
        (-0.375 + 0.59947894041409j)
        (-0.375 - 0.59947894041409j)
        >>>
        >>> err
        2.22044604925031e-16
        >>>
        >>> polyval([4,3,2], roots[0])
        (2.22044604925031e-16 + 0.0j)
        >>> polyval([4,3,2], roots[1])
        (2.22044604925031e-16 + 0.0j)

    The following example computes all the 5th roots of unity; that is,
    the roots of `x^5 - 1`::

        >>> mp.dps = 20
        >>> for r in polyroots([1, 0, 0, 0, 0, -1]):
        ...     print(r)
        ...
        1.0
        (-0.8090169943749474241 + 0.58778525229247312917j)
        (-0.8090169943749474241 - 0.58778525229247312917j)
        (0.3090169943749474241 + 0.95105651629515357212j)
        (0.3090169943749474241 - 0.95105651629515357212j)

    **Precision and conditioning**

    The roots are computed to the current working precision accuracy. If this
    accuracy cannot be achieved in ``maxsteps`` steps, then a
    ``NoConvergence`` exception is raised. The algorithm internally is using
    the current working precision extended by ``extraprec``. If
    ``NoConvergence`` was raised, that is caused either by not having enough
    extra precision to achieve convergence (in which case increasing
    ``extraprec`` should fix the problem) or too low ``maxsteps`` (in which
    case increasing ``maxsteps`` should fix the problem), or a combination of
    both.

    The user should always do a convergence study with regards to
    ``extraprec`` to ensure accurate results. It is possible to get
    convergence to a wrong answer with too low ``extraprec``.

    Provided there are no repeated roots, :func:`~mpmath.polyroots` can
    typically compute all roots of an arbitrary polynomial to high precision::

        >>> mp.dps = 60
        >>> for r in polyroots([1, 0, -10, 0, 1]):
        ...     print(r)
        ...
        -3.14626436994197234232913506571557044551247712918732870123249
        -0.317837245195782244725757617296174288373133378433432554879127
        0.317837245195782244725757617296174288373133378433432554879127
        3.14626436994197234232913506571557044551247712918732870123249
        >>>
        >>> sqrt(3) + sqrt(2)
        3.14626436994197234232913506571557044551247712918732870123249
        >>> sqrt(3) - sqrt(2)
        0.317837245195782244725757617296174288373133378433432554879127

    **Algorithm**

    :func:`~mpmath.polyroots` implements the Durand-Kerner method [1], which
    uses complex arithmetic to locate all roots simultaneously.
    The Durand-Kerner method can be viewed as approximately performing
    simultaneous Newton iteration for all the roots. In particular,
    the convergence to simple roots is quadratic, just like Newton's
    method.

    Although all roots are internally calculated using complex arithmetic, any
    root found to have an imaginary part smaller than the estimated numerical
    error is truncated to a real number (small real parts are also chopped).
    Real roots are placed first in the returned list, sorted by value. The
    remaining complex roots are sorted by their real parts so that conjugate
    roots end up next to each other.

    **References**

    1. http://en.wikipedia.org/wiki/Durand-Kerner_method

    """
    if len(coeffs) <= 1:
        if not coeffs or not coeffs[0]:
            raise ValueError("Input to polyroots must not be the zero polynomial")
        # Constant polynomial with no roots
        return []

    orig = ctx.prec
    tol = +ctx.eps
    with ctx.extraprec(extraprec):
        deg = len(coeffs) - 1
        # Must be monic
        lead = ctx.convert(coeffs[0])
        if lead == 1:
            coeffs = [ctx.convert(c) for c in coeffs]
        else:
            coeffs = [c/lead for c in coeffs]
        f = lambda x: ctx.polyval(coeffs, x)
        if roots_init is None:
            roots = [ctx.mpc((0.4+0.9j)**n) for n in xrange(deg)]
        else:
            roots = [None]*deg;
            deg_init = min(deg, len(roots_init))
            roots[:deg_init] = list(roots_init[:deg_init])
            roots[deg_init:] = [ctx.mpc((0.4+0.9j)**n) for n
                                in xrange(deg_init,deg)]
        err = [ctx.one for n in xrange(deg)]
        # Durand-Kerner iteration until convergence
        for step in xrange(maxsteps):
            if abs(max(err)) < tol:
                break
            for i in xrange(deg):
                p = roots[i]
                x = f(p)
                for j in range(deg):
                    if i != j:
                        try:
                            x /= (p-roots[j])
                        except ZeroDivisionError:
                            continue
                roots[i] = p - x
                err[i] = abs(x)
        if abs(max(err)) >= tol:
            raise ctx.NoConvergence("Didn't converge in maxsteps=%d steps." \
                    % maxsteps)
        # Remove small real or imaginary parts
        if cleanup:
            for i in xrange(deg):
                if abs(roots[i]) < tol:
                    roots[i] = ctx.zero
                elif abs(ctx._im(roots[i])) < tol:
                    roots[i] = roots[i].real
                elif abs(ctx._re(roots[i])) < tol:
                    roots[i] = roots[i].imag * 1j
        roots.sort(key=lambda x: (abs(ctx._im(x)), ctx._re(x)))
    if error:
        err = max(err)
        err = max(err, ctx.ldexp(1, -orig+1))
        return [+r for r in roots], +err
    else:
        return [+r for r in roots]
<|endoftext|>from bisect import bisect
from ..libmp.backend import xrange

class ODEMethods(object):
    pass

def ode_taylor(ctx, derivs, x0, y0, tol_prec, n):
    h = tol = ctx.ldexp(1, -tol_prec)
    dim = len(y0)
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    orig = ctx.prec
    try:
        ctx.prec = orig*(1+n)
        # Use n steps with Euler's method to get
        # evaluation points for derivatives
        for i in range(n):
            fxy = derivs(x, y)
            y = [y[i]+h*fxy[i] for i in xrange(len(y))]
            x += h
            xs.append(x)
            ys.append(y)
        # Compute derivatives
        ser = [[] for d in range(dim)]
        for j in range(n+1):
            s = [0]*dim
            b = (-1) ** (j & 1)
            k = 1
            for i in range(j+1):
                for d in range(dim):
                    s[d] += b * ys[i][d]
                b = (b * (j-k+1)) // (-k)
                k += 1
            scale = h**(-j) / ctx.fac(j)
            for d in range(dim):
                s[d] = s[d] * scale
                ser[d].append(s[d])
    finally:
        ctx.prec = orig
    # Estimate radius for which we can get full accuracy.
    # XXX: do this right for zeros
    radius = ctx.one
    for ts in ser:
        if ts[-1]:
            radius = min(radius, ctx.nthroot(tol/abs(ts[-1]), n))
    radius /= 2  # XXX
    return ser, x0+radius

def odefun(ctx, F, x0, y0, tol=None, degree=None, method='taylor', verbose=False):
    r"""
    Returns a function `y(x) = [y_0(x), y_1(x), \ldots, y_n(x)]`
    that is a numerical solution of the `n+1`-dimensional first-order
    ordinary differential equation (ODE) system

    .. math ::

        y_0'(x) = F_0(x, [y_0(x), y_1(x), \ldots, y_n(x)])

        y_1'(x) = F_1(x, [y_0(x), y_1(x), \ldots, y_n(x)])

        \vdots

        y_n'(x) = F_n(x, [y_0(x), y_1(x), \ldots, y_n(x)])

    The derivatives are specified by the vector-valued function
    *F* that evaluates
    `[y_0', \ldots, y_n'] = F(x, [y_0, \ldots, y_n])`.
    The initial point `x_0` is specified by the scalar argument *x0*,
    and the initial value `y(x_0) =  [y_0(x_0), \ldots, y_n(x_0)]` is
    specified by the vector argument *y0*.

    For convenience, if the system is one-dimensional, you may optionally
    provide just a scalar value for *y0*. In this case, *F* should accept
    a scalar *y* argument and return a scalar. The solution function
    *y* will return scalar values instead of length-1 vectors.

    Evaluation of the solution function `y(x)` is permitted
    for any `x \ge x_0`.

    A high-order ODE can be solved by transforming it into first-order
    vector form. This transformation is described in standard texts
    on ODEs. Examples will also be given below.

    **Options, speed and accuracy**

    By default, :func:`~mpmath.odefun` uses a high-order Taylor series
    method. For reasonably well-behaved problems, the solution will
    be fully accurate to within the working precision. Note that
    *F* must be possible to evaluate to very high precision
    for the generation of Taylor series to work.

    To get a faster but less accurate solution, you can set a large
    value for *tol* (which defaults roughly to *eps*). If you just
    want to plot the solution or perform a basic simulation,
    *tol = 0.01* is likely sufficient.

    The *degree* argument controls the degree of the solver (with
    *method='taylor'*, this is the degree of the Taylor series
    expansion). A higher degree means that a longer step can be taken
    before a new local solution must be generated from *F*,
    meaning that fewer steps are required to get from `x_0` to a given
    `x_1`. On the other hand, a higher degree also means that each
    local solution becomes more expensive (i.e., more evaluations of
    *F* are required per step, and at higher precision).

    The optimal setting therefore involves a tradeoff. Generally,
    decreasing the *degree* for Taylor series is likely to give faster
    solution at low precision, while increasing is likely to be better
    at higher precision.

    The function
    object returned by :func:`~mpmath.odefun` caches the solutions at all step
    points and uses polynomial interpolation between step points.
    Therefore, once `y(x_1)` has been evaluated for some `x_1`,
    `y(x)` can be evaluated very quickly for any `x_0 \le x \le x_1`.
    and continuing the evaluation up to `x_2 > x_1` is also fast.

    **Examples of first-order ODEs**

    We will solve the standard test problem `y'(x) = y(x), y(0) = 1`
    which has explicit solution `y(x) = \exp(x)`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> f = odefun(lambda x, y: y, 0, 1)
        >>> for x in [0, 1, 2.5]:
        ...     print((f(x), exp(x)))
        ...
        (1.0, 1.0)
        (2.71828182845905, 2.71828182845905)
        (12.1824939607035, 12.1824939607035)

    The solution with high precision::

        >>> mp.dps = 50
        >>> f = odefun(lambda x, y: y, 0, 1)
        >>> f(1)
        2.7182818284590452353602874713526624977572470937
        >>> exp(1)
        2.7182818284590452353602874713526624977572470937

    Using the more general vectorized form, the test problem
    can be input as (note that *f* returns a 1-element vector)::

        >>> mp.dps = 15
        >>> f = odefun(lambda x, y: [y[0]], 0, [1])
        >>> f(1)
        [2.71828182845905]

    :func:`~mpmath.odefun` can solve nonlinear ODEs, which are generally
    impossible (and at best difficult) to solve analytically. As
    an example of a nonlinear ODE, we will solve `y'(x) = x \sin(y(x))`
    for `y(0) = \pi/2`. An exact solution happens to be known
    for this problem, and is given by
    `y(x) = 2 \tan^{-1}\left(\exp\left(x^2/2\right)\right)`::

        >>> f = odefun(lambda x, y: x*sin(y), 0, pi/2)
        >>> for x in [2, 5, 10]:
        ...     print((f(x), 2*atan(exp(mpf(x)**2/2))))
        ...
        (2.87255666284091, 2.87255666284091)
        (3.14158520028345, 3.14158520028345)
        (3.14159265358979, 3.14159265358979)

    If `F` is independent of `y`, an ODE can be solved using direct
    integration. We can therefore obtain a reference solution with
    :func:`~mpmath.quad`::

        >>> f = lambda x: (1+x**2)/(1+x**3)
        >>> g = odefun(lambda x, y: f(x), pi, 0)
        >>> g(2*pi)
        0.72128263801696
        >>> quad(f, [pi, 2*pi])
        0.72128263801696

    **Examples of second-order ODEs**

    We will solve the harmonic oscillator equation `y''(x) + y(x) = 0`.
    To do this, we introduce the helper functions `y_0 = y, y_1 = y_0'`
    whereby the original equation can be written as `y_1' + y_0' = 0`. Put
    together, we get the first-order, two-dimensional vector ODE

    .. math ::

        \begin{cases}
        y_0' = y_1 \\
        y_1' = -y_0
        \end{cases}

    To get a well-defined IVP, we need two initial values. With
    `y(0) = y_0(0) = 1` and `-y'(0) = y_1(0) = 0`, the problem will of
    course be solved by `y(x) = y_0(x) = \cos(x)` and
    `-y'(x) = y_1(x) = \sin(x)`. We check this::

        >>> f = odefun(lambda x, y: [-y[1], y[0]], 0, [1, 0])
        >>> for x in [0, 1, 2.5, 10]:
        ...     nprint(f(x), 15)
        ...     nprint([cos(x), sin(x)], 15)
        ...     print("---")
        ...
        [1.0, 0.0]
        [1.0, 0.0]
        ---
        [0.54030230586814, 0.841470984807897]
        [0.54030230586814, 0.841470984807897]
        ---
        [-0.801143615546934, 0.598472144103957]
        [-0.801143615546934, 0.598472144103957]
        ---
        [-0.839071529076452, -0.54402111088937]
        [-0.839071529076452, -0.54402111088937]
        ---

    Note that we get both the sine and the cosine solutions
    simultaneously.

    **TODO**

    * Better automatic choice of degree and step size
    * Make determination of Taylor series convergence radius
      more robust
    * Allow solution for `x < x_0`
    * Allow solution for complex `x`
    * Test for difficult (ill-conditioned) problems
    * Implement Runge-Kutta and other algorithms

    """
    if tol:
        tol_prec = int(-ctx.log(tol, 2))+10
    else:
        tol_prec = ctx.prec+10
    degree = degree or (3 + int(3*ctx.dps/2.))
    workprec = ctx.prec + 40
    try:
        len(y0)
        return_vector = True
    except TypeError:
        F_ = F
        F = lambda x, y: [F_(x, y[0])]
        y0 = [y0]
        return_vector = False
    ser, xb = ode_taylor(ctx, F, x0, y0, tol_prec, degree)
    series_boundaries = [x0, xb]
    series_data = [(ser, x0, xb)]
    # We will be working with vectors of Taylor series
    def mpolyval(ser, a):
        return [ctx.polyval(s[::-1], a) for s in ser]
    # Find nearest expansion point; compute if necessary
    def get_series(x):
        if x < x0:
            raise ValueError
        n = bisect(series_boundaries, x)
        if n < len(series_boundaries):
            return series_data[n-1]
        while 1:
            ser, xa, xb = series_data[-1]
            if verbose:
                print("Computing Taylor series for [%f, %f]" % (xa, xb))
            y = mpolyval(ser, xb-xa)
            xa = xb
            ser, xb = ode_taylor(ctx, F, xb, y, tol_prec, degree)
            series_boundaries.append(xb)
            series_data.append((ser, xa, xb))
            if x <= xb:
                return series_data[-1]
    # Evaluation function
    def interpolant(x):
        x = ctx.convert(x)
        orig = ctx.prec
        try:
            ctx.prec = workprec
            ser, xa, xb = get_series(x)
            y = mpolyval(ser, x-xa)
        finally:
            ctx.prec = orig
        if return_vector:
            return [+yk for yk in y]
        else:
            return +y[0]
    return interpolant

ODEMethods.odefun = odefun

if __name__ == "__main__":
    import doctest
    doctest.testmod()
<|endoftext|>from ..libmp.backend import xrange
from .calculus import defun

try:
    iteritems = dict.iteritems
except AttributeError:
    iteritems = dict.items

#----------------------------------------------------------------------------#
#                                Differentiation                             #
#----------------------------------------------------------------------------#

@defun
def difference(ctx, s, n):
    r"""
    Given a sequence `(s_k)` containing at least `n+1` items, returns the
    `n`-th forward difference,

    .. math ::

        \Delta^n = \sum_{k=0}^{\infty} (-1)^{k+n} {n \choose k} s_k.
    """
    n = int(n)
    d = ctx.zero
    b = (-1) ** (n & 1)
    for k in xrange(n+1):
        d += b * s[k]
        b = (b * (k-n)) // (k+1)
    return d

def hsteps(ctx, f, x, n, prec, **options):
    singular = options.get('singular')
    addprec = options.get('addprec', 10)
    direction = options.get('direction', 0)
    workprec = (prec+2*addprec) * (n+1)
    orig = ctx.prec
    try:
        ctx.prec = workprec
        h = options.get('h')
        if h is None:
            if options.get('relative'):
                hextramag = int(ctx.mag(x))
            else:
                hextramag = 0
            h = ctx.ldexp(1, -prec-addprec-hextramag)
        else:
            h = ctx.convert(h)
        # Directed: steps x, x+h, ... x+n*h
        direction = options.get('direction', 0)
        if direction:
            h *= ctx.sign(direction)
            steps = xrange(n+1)
            norm = h
        # Central: steps x-n*h, x-(n-2)*h ..., x, ..., x+(n-2)*h, x+n*h
        else:
            steps = xrange(-n, n+1, 2)
            norm = (2*h)
        # Perturb
        if singular:
            x += 0.5*h
        values = [f(x+k*h) for k in steps]
        return values, norm, workprec
    finally:
        ctx.prec = orig


@defun
def diff(ctx, f, x, n=1, **options):
    r"""
    Numerically computes the derivative of `f`, `f'(x)`, or generally for
    an integer `n \ge 0`, the `n`-th derivative `f^{(n)}(x)`.
    A few basic examples are::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> diff(lambda x: x**2 + x, 1.0)
        3.0
        >>> diff(lambda x: x**2 + x, 1.0, 2)
        2.0
        >>> diff(lambda x: x**2 + x, 1.0, 3)
        0.0
        >>> nprint([diff(exp, 3, n) for n in range(5)])   # exp'(x) = exp(x)
        [20.0855, 20.0855, 20.0855, 20.0855, 20.0855]

    Even more generally, given a tuple of arguments `(x_1, \ldots, x_k)`
    and order `(n_1, \ldots, n_k)`, the partial derivative
    `f^{(n_1,\ldots,n_k)}(x_1,\ldots,x_k)` is evaluated. For example::

        >>> diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (0,1))
        2.75
        >>> diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (1,1))
        3.0

    **Options**

    The following optional keyword arguments are recognized:

    ``method``
        Supported methods are ``'step'`` or ``'quad'``: derivatives may be
        computed using either a finite difference with a small step
        size `h` (default), or numerical quadrature.
    ``direction``
        Direction of finite difference: can be -1 for a left
        difference, 0 for a central difference (default), or +1
        for a right difference; more generally can be any complex number.
    ``addprec``
        Extra precision for `h` used to account for the function's
        sensitivity to perturbations (default = 10).
    ``relative``
        Choose `h` relative to the magnitude of `x`, rather than an
        absolute value; useful for large or tiny `x` (default = False).
    ``h``
        As an alternative to ``addprec`` and ``relative``, manually
        select the step size `h`.
    ``singular``
        If True, evaluation exactly at the point `x` is avoided; this is
        useful for differentiating functions with removable singularities.
        Default = False.
    ``radius``
        Radius of integration contour (with ``method = 'quad'``).
        Default = 0.25. A larger radius typically is faster and more
        accurate, but it must be chosen so that `f` has no
        singularities within the radius from the evaluation point.

    A finite difference requires `n+1` function evaluations and must be
    performed at `(n+1)` times the target precision. Accordingly, `f` must
    support fast evaluation at high precision.

    With integration, a larger number of function evaluations is
    required, but not much extra precision is required. For high order
    derivatives, this method may thus be faster if f is very expensive to
    evaluate at high precision.

    **Further examples**

    The direction option is useful for computing left- or right-sided
    derivatives of nonsmooth functions::

        >>> diff(abs, 0, direction=0)
        0.0
        >>> diff(abs, 0, direction=1)
        1.0
        >>> diff(abs, 0, direction=-1)
        -1.0

    More generally, if the direction is nonzero, a right difference
    is computed where the step size is multiplied by sign(direction).
    For example, with direction=+j, the derivative from the positive
    imaginary direction will be computed::

        >>> diff(abs, 0, direction=j)
        (0.0 - 1.0j)

    With integration, the result may have a small imaginary part
    even even if the result is purely real::

        >>> diff(sqrt, 1, method='quad')    # doctest:+ELLIPSIS
        (0.5 - 4.59...e-26j)
        >>> chop(_)
        0.5

    Adding precision to obtain an accurate value::

        >>> diff(cos, 1e-30)
        0.0
        >>> diff(cos, 1e-30, h=0.0001)
        -9.99999998328279e-31
        >>> diff(cos, 1e-30, addprec=100)
        -1.0e-30

    """
    partial = False
    try:
        orders = list(n)
        x = list(x)
        partial = True
    except TypeError:
        pass
    if partial:
        x = [ctx.convert(_) for _ in x]
        return _partial_diff(ctx, f, x, orders, options)
    method = options.get('method', 'step')
    if n == 0 and method != 'quad' and not options.get('singular'):
        return f(ctx.convert(x))
    prec = ctx.prec
    try:
        if method == 'step':
            values, norm, workprec = hsteps(ctx, f, x, n, prec, **options)
            ctx.prec = workprec
            v = ctx.difference(values, n) / norm**n
        elif method == 'quad':
            ctx.prec += 10
            radius = ctx.convert(options.get('radius', 0.25))
            def g(t):
                rei = radius*ctx.expj(t)
                z = x + rei
                return f(z) / rei**n
            d = ctx.quadts(g, [0, 2*ctx.pi])
            v = d * ctx.factorial(n) / (2*ctx.pi)
        else:
            raise ValueError("unknown method: %r" % method)
    finally:
        ctx.prec = prec
    return +v

def _partial_diff(ctx, f, xs, orders, options):
    if not orders:
        return f()
    if not sum(orders):
        return f(*xs)
    i = 0
    for i in range(len(orders)):
        if orders[i]:
            break
    order = orders[i]
    def fdiff_inner(*f_args):
        def inner(t):
            return f(*(f_args[:i] + (t,) + f_args[i+1:]))
        return ctx.diff(inner, f_args[i], order, **options)
    orders[i] = 0
    return _partial_diff(ctx, fdiff_inner, xs, orders, options)

@defun
def diffs(ctx, f, x, n=None, **options):
    r"""
    Returns a generator that yields the sequence of derivatives

    .. math ::

        f(x), f'(x), f''(x), \ldots, f^{(k)}(x), \ldots

    With ``method='step'``, :func:`~mpmath.diffs` uses only `O(k)`
    function evaluations to generate the first `k` derivatives,
    rather than the roughly `O(k^2)` evaluations
    required if one calls :func:`~mpmath.diff` `k` separate times.

    With `n < \infty`, the generator stops as soon as the
    `n`-th derivative has been generated. If the exact number of
    needed derivatives is known in advance, this is further
    slightly more efficient.

    Options are the same as for :func:`~mpmath.diff`.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15
        >>> nprint(list(diffs(cos, 1, 5)))
        [0.540302, -0.841471, -0.540302, 0.841471, 0.540302, -0.841471]
        >>> for i, d in zip(range(6), diffs(cos, 1)):
        ...     print("%s %s" % (i, d))
        ...
        0 0.54030230586814
        1 -0.841470984807897
        2 -0.54030230586814
        3 0.841470984807897
        4 0.54030230586814
        5 -0.841470984807897

    """
    if n is None:
        n = ctx.inf
    else:
        n = int(n)
    if options.get('method', 'step') != 'step':
        k = 0
        while k < n + 1:
            yield ctx.diff(f, x, k, **options)
            k += 1
        return
    singular = options.get('singular')
    if singular:
        yield ctx.diff(f, x, 0, singular=True)
    else:
        yield f(ctx.convert(x))
    if n < 1:
        return
    if n == ctx.inf:
        A, B = 1, 2
    else:
        A, B = 1, n+1
    while 1:
        callprec = ctx.prec
        y, norm, workprec = hsteps(ctx, f, x, B, callprec, **options)
        for k in xrange(A, B):
            try:
                ctx.prec = workprec
                d = ctx.difference(y, k) / norm**k
            finally:
                ctx.prec = callprec
            yield +d
            if k >= n:
                return
        A, B = B, int(A*1.4+1)
        B = min(B, n)

def iterable_to_function(gen):
    gen = iter(gen)
    data = []
    def f(k):
        for i in xrange(len(data), k+1):
            data.append(next(gen))
        return data[k]
    return f

@defun
def diffs_prod(ctx, factors):
    r"""
    Given a list of `N` iterables or generators yielding
    `f_k(x), f'_k(x), f''_k(x), \ldots` for `k = 1, \ldots, N`,
    generate `g(x), g'(x), g''(x), \ldots` where
    `g(x) = f_1(x) f_2(x) \cdots f_N(x)`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of each `f_k(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> f = lambda x: exp(x)*cos(x)*sin(x)
        >>> u = diffs(f, 1)
        >>> v = mp.diffs_prod([diffs(exp,1), diffs(cos,1), diffs(sin,1)])
        >>> next(u); next(v)
        1.23586333600241
        1.23586333600241
        >>> next(u); next(v)
        0.104658952245596
        0.104658952245596
        >>> next(u); next(v)
        -5.96999877552086
        -5.96999877552086
        >>> next(u); next(v)
        -12.4632923122697
        -12.4632923122697

    """
    N = len(factors)
    if N == 1:
        for c in factors[0]:
            yield c
    else:
        u = iterable_to_function(ctx.diffs_prod(factors[:N//2]))
        v = iterable_to_function(ctx.diffs_prod(factors[N//2:]))
        n = 0
        while 1:
            #yield sum(binomial(n,k)*u(n-k)*v(k) for k in xrange(n+1))
            s = u(n) * v(0)
            a = 1
            for k in xrange(1,n+1):
                a = a * (n-k+1) // k
                s += a * u(n-k) * v(k)
            yield s
            n += 1

def dpoly(n, _cache={}):
    """
    nth differentiation polynomial for exp (Faa di Bruno's formula).

    TODO: most exponents are zero, so maybe a sparse representation
    would be better.
    """
    if n in _cache:
        return _cache[n]
    if not _cache:
        _cache[0] = {(0,):1}
    R = dpoly(n-1)
    R = dict((c+(0,),v) for (c,v) in iteritems(R))
    Ra = {}
    for powers, count in iteritems(R):
        powers1 = (powers[0]+1,) + powers[1:]
        if powers1 in Ra:
            Ra[powers1] += count
        else:
            Ra[powers1] = count
    for powers, count in iteritems(R):
        if not sum(powers):
            continue
        for k,p in enumerate(powers):
            if p:
                powers2 = powers[:k] + (p-1,powers[k+1]+1) + powers[k+2:]
                if powers2 in Ra:
                    Ra[powers2] += p*count
                else:
                    Ra[powers2] = p*count
    _cache[n] = Ra
    return _cache[n]

@defun
def diffs_exp(ctx, fdiffs):
    r"""
    Given an iterable or generator yielding `f(x), f'(x), f''(x), \ldots`
    generate `g(x), g'(x), g''(x), \ldots` where `g(x) = \exp(f(x))`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of `f(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

    The derivatives of the gamma function can be computed using
    logarithmic differentiation::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>>
        >>> def diffs_loggamma(x):
        ...     yield loggamma(x)
        ...     i = 0
        ...     while 1:
        ...         yield psi(i,x)
        ...         i += 1
        ...
        >>> u = diffs_exp(diffs_loggamma(3))
        >>> v = diffs(gamma, 3)
        >>> next(u); next(v)
        2.0
        2.0
        >>> next(u); next(v)
        1.84556867019693
        1.84556867019693
        >>> next(u); next(v)
        2.49292999190269
        2.49292999190269
        >>> next(u); next(v)
        3.44996501352367
        3.44996501352367

    """
    fn = iterable_to_function(fdiffs)
    f0 = ctx.exp(fn(0))
    yield f0
    i = 1
    while 1:
        s = ctx.mpf(0)
        for powers, c in iteritems(dpoly(i)):
            s += c*ctx.fprod(fn(k+1)**p for (k,p) in enumerate(powers) if p)
        yield s * f0
        i += 1

@defun
def differint(ctx, f, x, n=1, x0=0):
    r"""
    Calculates the Riemann-Liouville differintegral, or fractional
    derivative, defined by

    .. math ::

        \,_{x_0}{\mathbb{D}}^n_xf(x) = \frac{1}{\Gamma(m-n)} \frac{d^m}{dx^m}
        \int_{x_0}^{x}(x-t)^{m-n-1}f(t)dt

    where `f` is a given (presumably well-behaved) function,
    `x` is the evaluation point, `n` is the order, and `x_0` is
    the reference point of integration (`m` is an arbitrary
    parameter selected automatically).

    With `n = 1`, this is just the standard derivative `f'(x)`; with `n = 2`,
    the second derivative `f''(x)`, etc. With `n = -1`, it gives
    `\int_{x_0}^x f(t) dt`, with `n = -2`
    it gives `\int_{x_0}^x \left( \int_{x_0}^t f(u) du \right) dt`, etc.

    As `n` is permitted to be any number, this operator generalizes
    iterated differentiation and iterated integration to a single
    operator with a continuous order parameter.

    **Examples**

    There is an exact formula for the fractional derivative of a
    monomial `x^p`, which may be used as a reference. For example,
    the following gives a half-derivative (order 0.5)::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> x = mpf(3); p = 2; n = 0.5
        >>> differint(lambda t: t**p, x, n)
        7.81764019044672
        >>> gamma(p+1)/gamma(p-n+1) * x**(p-n)
        7.81764019044672

    Another useful test function is the exponential function, whose
    integration / differentiation formula easy generalizes
    to arbitrary order. Here we first compute a third derivative,
    and then a triply nested integral. (The reference point `x_0`
    is set to `-\infty` to avoid nonzero endpoint terms.)::

        >>> differint(lambda x: exp(pi*x), -1.5, 3)
        0.278538406900792
        >>> exp(pi*-1.5) * pi**3
        0.278538406900792
        >>> differint(lambda x: exp(pi*x), 3.5, -3, -inf)
        1922.50563031149
        >>> exp(pi*3.5) / pi**3
        1922.50563031149

    However, for noninteger `n`, the differentiation formula for the
    exponential function must be modified to give the same result as the
    Riemann-Liouville differintegral::

        >>> x = mpf(3.5)
        >>> c = pi
        >>> n = 1+2*j
        >>> differint(lambda x: exp(c*x), x, n)
        (-123295.005390743 + 140955.117867654j)
        >>> x**(-n) * exp(c)**x * (x*c)**n * gammainc(-n, 0, x*c) / gamma(-n)
        (-123295.005390743 + 140955.117867654j)


    """
    m = max(int(ctx.ceil(ctx.re(n)))+1, 1)
    r = m-n-1
    g = lambda x: ctx.quad(lambda t: (x-t)**r * f(t), [x0, x])
    return ctx.diff(g, x, m) / ctx.gamma(m-n)

@defun
def diffun(ctx, f, n=1, **options):
    r"""
    Given a function `f`, returns a function `g(x)` that evaluates the nth
    derivative `f^{(n)}(x)`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> cos2 = diffun(sin)
        >>> sin2 = diffun(sin, 4)
        >>> cos(1.3), cos2(1.3)
        (0.267498828624587, 0.267498828624587)
        >>> sin(1.3), sin2(1.3)
        (0.963558185417193, 0.963558185417193)

    The function `f` must support arbitrary precision evaluation.
    See :func:`~mpmath.diff` for additional details and supported
    keyword options.
    """
    if n == 0:
        return f
    def g(x):
        return ctx.diff(f, x, n, **options)
    return g

@defun
def taylor(ctx, f, x, n, **options):
    r"""
    Produces a degree-`n` Taylor polynomial around the point `x` of the
    given function `f`. The coefficients are returned as a list.

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nprint(chop(taylor(sin, 0, 5)))
        [0.0, 1.0, 0.0, -0.166667, 0.0, 0.00833333]

    The coefficients are computed using high-order numerical
    differentiation. The function must be possible to evaluate
    to arbitrary precision. See :func:`~mpmath.diff` for additional details
    and supported keyword options.

    Note that to evaluate the Taylor polynomial as an approximation
    of `f`, e.g. with :func:`~mpmath.polyval`, the coefficients must be reversed,
    and the point of the Taylor expansion must be subtracted from
    the argument:

        >>> p = taylor(exp, 2.0, 10)
        >>> polyval(p[::-1], 2.5 - 2.0)
        12.1824939606092
        >>> exp(2.5)
        12.1824939607035

    """
    gen = enumerate(ctx.diffs(f, x, n, **options))
    if options.get("chop", True):
        return [ctx.chop(d)/ctx.factorial(i) for i, d in gen]
    else:
        return [d/ctx.factorial(i) for i, d in gen]

@defun
def pade(ctx, a, L, M):
    r"""
    Computes a Pade approximation of degree `(L, M)` to a function.
    Given at least `L+M+1` Taylor coefficients `a` approximating
    a function `A(x)`, :func:`~mpmath.pade` returns coefficients of
    polynomials `P, Q` satisfying

    .. math ::

        P = \sum_{k=0}^L p_k x^k

        Q = \sum_{k=0}^M q_k x^k

        Q_0 = 1

        A(x) Q(x) = P(x) + O(x^{L+M+1})

    `P(x)/Q(x)` can provide a good approximation to an analytic function
    beyond the radius of convergence of its Taylor series (example
    from G.A. Baker 'Essentials of Pade Approximants' Academic Press,
    Ch.1A)::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> one = mpf(1)
        >>> def f(x):
        ...     return sqrt((one + 2*x)/(one + x))
        ...
        >>> a = taylor(f, 0, 6)
        >>> p, q = pade(a, 3, 3)
        >>> x = 10
        >>> polyval(p[::-1], x)/polyval(q[::-1], x)
        1.38169105566806
        >>> f(x)
        1.38169855941551

    """
    # To determine L+1 coefficients of P and M coefficients of Q
    # L+M+1 coefficients of A must be provided
    if len(a) < L+M+1:
        raise ValueError("L+M+1 Coefficients should be provided")

    if M == 0:
        if L == 0:
            return [ctx.one], [ctx.one]
        else:
            return a[:L+1], [ctx.one]

    # Solve first
    # a[L]*q[1] + ... + a[L-M+1]*q[M] = -a[L+1]
    # ...
    # a[L+M-1]*q[1] + ... + a[L]*q[M] = -a[L+M]
    A = ctx.matrix(M)
    for j in range(M):
        for i in range(min(M, L+j+1)):
            A[j, i] = a[L+j-i]
    v = -ctx.matrix(a[(L+1):(L+M+1)])
    x = ctx.lu_solve(A, v)
    q = [ctx.one] + list(x)
    # compute p
    p = [0]*(L+1)
    for i in range(L+1):
        s = a[i]
        for j in range(1, min(M,i) + 1):
            s += q[j]*a[i-j]
        p[i] = s
    return p, q
<|endoftext|>import os
import sys

#----------------------------------------------------------------------------#
# Support GMPY for high-speed large integer arithmetic.                      #
#                                                                            #
# To allow an external module to handle arithmetic, we need to make sure     #
# that all high-precision variables are declared of the correct type. MPZ    #
# is the constructor for the high-precision type. It defaults to Python's    #
# long type but can be assinged another type, typically gmpy.mpz.            #
#                                                                            #
# MPZ must be used for the mantissa component of an mpf and must be used     #
# for internal fixed-point operations.                                       #
#                                                                            #
# Side-effects                                                               #
# 1) "is" cannot be used to test for special values. Must use "==".          #
# 2) There are bugs in GMPY prior to v1.02 so we must use v1.03 or later.    #
#----------------------------------------------------------------------------#

# So we can import it from this module
gmpy = None
sage = None
sage_utils = None

if sys.version_info[0] < 3:
    python3 = False
else:
    python3 = True

BACKEND = 'python'

if not python3:
    MPZ = long
    xrange = xrange
    basestring = basestring

    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")
else:
    MPZ = int
    xrange = range
    basestring = str

    import builtins
    exec_ = getattr(builtins, "exec")

# Define constants for calculating hash on Python 3.2.
if sys.version_info >= (3, 2):
    HASH_MODULUS = sys.hash_info.modulus
    if sys.hash_info.width == 32:
        HASH_BITS = 31
    else:
        HASH_BITS = 61
else:
    HASH_MODULUS = None
    HASH_BITS = None

if 'MPMATH_NOGMPY' not in os.environ:
    try:
        try:
            import gmpy2 as gmpy
        except ImportError:
            try:
                import gmpy
            except ImportError:
                raise ImportError
        if gmpy.version() >= '1.03':
            BACKEND = 'gmpy'
            MPZ = gmpy.mpz
    except:
        pass

if ('MPMATH_NOSAGE' not in os.environ and 'SAGE_ROOT' in os.environ or
        'MPMATH_SAGE' in os.environ):
    try:
        import sage.all
        import sage.libs.mpmath.utils as _sage_utils
        sage = sage.all
        sage_utils = _sage_utils
        BACKEND = 'sage'
        MPZ = sage.Integer
    except:
        pass

if 'MPMATH_STRICT' in os.environ:
    STRICT = True
else:
    STRICT = False

MPZ_TYPE = type(MPZ(0))
MPZ_ZERO = MPZ(0)
MPZ_ONE = MPZ(1)
MPZ_TWO = MPZ(2)
MPZ_THREE = MPZ(3)
MPZ_FIVE = MPZ(5)

try:
    if BACKEND == 'python':
        int_types = (int, long)
    else:
        int_types = (int, long, MPZ_TYPE)
except NameError:
    if BACKEND == 'python':
        int_types = (int,)
    else:
        int_types = (int, MPZ_TYPE)
<|endoftext|>"""
-----------------------------------------------------------------------
This module implements gamma- and zeta-related functions:

* Bernoulli numbers
* Factorials
* The gamma function
* Polygamma functions
* Harmonic numbers
* The Riemann zeta function
* Constants related to these functions

-----------------------------------------------------------------------
"""

import math
import sys

from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy

from .libintmath import list_primes, ifac, ifac2, moebius

from .libmpf import (\
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast,
    lshift, sqrt_fixed, isqrt_fast,
    fzero, fone, fnone, fhalf, ftwo, finf, fninf, fnan,
    from_int, to_int, to_fixed, from_man_exp, from_rational,
    mpf_pos, mpf_neg, mpf_abs, mpf_add, mpf_sub,
    mpf_mul, mpf_mul_int, mpf_div, mpf_sqrt, mpf_pow_int,
    mpf_rdiv_int,
    mpf_perturb, mpf_le, mpf_lt, mpf_gt, mpf_shift,
    negative_rnd, reciprocal_rnd,
    bitcount, to_float, mpf_floor, mpf_sign, ComplexResult
)

from .libelefun import (\
    constant_memo,
    def_mpf_constant,
    mpf_pi, pi_fixed, ln2_fixed, log_int_fixed, mpf_ln2,
    mpf_exp, mpf_log, mpf_pow, mpf_cosh,
    mpf_cos_sin, mpf_cosh_sinh, mpf_cos_sin_pi, mpf_cos_pi, mpf_sin_pi,
    ln_sqrt2pi_fixed, mpf_ln_sqrt2pi, sqrtpi_fixed, mpf_sqrtpi,
    cos_sin_fixed, exp_fixed
)

from .libmpc import (\
    mpc_zero, mpc_one, mpc_half, mpc_two,
    mpc_abs, mpc_shift, mpc_pos, mpc_neg,
    mpc_add, mpc_sub, mpc_mul, mpc_div,
    mpc_add_mpf, mpc_mul_mpf, mpc_div_mpf, mpc_mpf_div,
    mpc_mul_int, mpc_pow_int,
    mpc_log, mpc_exp, mpc_pow,
    mpc_cos_pi, mpc_sin_pi,
    mpc_reciprocal, mpc_square,
    mpc_sub_mpf
)



# Catalan's constant is computed using Lupas's rapidly convergent series
# (listed on http://mathworld.wolfram.com/CatalansConstant.html)
#            oo
#            ___       n-1  8n     2                   3    2
#        1  \      (-1)    2   (40n  - 24n + 3) [(2n)!] (n!)
#  K =  ---  )     -----------------------------------------
#       64  /___               3               2
#                             n  (2n-1) [(4n)!]
#           n = 1

@constant_memo
def catalan_fixed(prec):
    prec = prec + 20
    a = one = MPZ_ONE << prec
    s, t, n = 0, 1, 1
    while t:
        a *= 32 * n**3 * (2*n-1)
        a //= (3-16*n+16*n**2)**2
        t = a * (-1)**(n-1) * (40*n**2-24*n+3) // (n**3 * (2*n-1))
        s += t
        n += 1
    return s >> (20 + 6)

# Khinchin's constant is relatively difficult to compute. Here
# we use the rational zeta series

#                    oo                2*n-1
#                   ___                ___
#                   \   ` zeta(2*n)-1  \   ` (-1)^(k+1)
#  log(K)*log(2) =   )    ------------  )    ----------
#                   /___.      n       /___.      k
#                   n = 1              k = 1

# which adds half a digit per term. The essential trick for achieving
# reasonable efficiency is to recycle both the values of the zeta
# function (essentially Bernoulli numbers) and the partial terms of
# the inner sum.

# An alternative might be to use K = 2*exp[1/log(2) X] where

#      / 1     1       [ pi*x*(1-x^2) ]
#  X = |    ------ log [ ------------ ].
#      / 0  x(1+x)     [  sin(pi*x)   ]

# and integrate numerically. In practice, this seems to be slightly
# slower than the zeta series at high precision.

@constant_memo
def khinchin_fixed(prec):
    wp = int(prec + prec**0.5 + 15)
    s = MPZ_ZERO
    fac = from_int(4)
    t = ONE = MPZ_ONE << wp
    pi = mpf_pi(wp)
    pipow = twopi2 = mpf_shift(mpf_mul(pi, pi, wp), 2)
    n = 1
    while 1:
        zeta2n = mpf_abs(mpf_bernoulli(2*n, wp))
        zeta2n = mpf_mul(zeta2n, pipow, wp)
        zeta2n = mpf_div(zeta2n, fac, wp)
        zeta2n = to_fixed(zeta2n, wp)
        term = (((zeta2n - ONE) * t) // n) >> wp
        if term < 100:
            break
        #if not n % 10:
        #    print n, math.log(int(abs(term)))
        s += term
        t += ONE//(2*n+1) - ONE//(2*n)
        n += 1
        fac = mpf_mul_int(fac, (2*n)*(2*n-1), wp)
        pipow = mpf_mul(pipow, twopi2, wp)
    s = (s << wp) // ln2_fixed(wp)
    K = mpf_exp(from_man_exp(s, -wp), wp)
    K = to_fixed(K, prec)
    return K


# Glaisher's constant is defined as A = exp(1/2 - zeta'(-1)).
# One way to compute it would be to perform direct numerical
# differentiation, but computing arbitrary Riemann zeta function
# values at high precision is expensive. We instead use the formula

#     A = exp((6 (-zeta'(2))/pi^2 + log 2 pi + gamma)/12)

# and compute zeta'(2) from the series representation

#              oo
#              ___
#             \     log k
#  -zeta'(2) = )    -----
#             /___     2
#                    k
#            k = 2

# This series converges exceptionally slowly, but can be accelerated
# using Euler-Maclaurin formula. The important insight is that the
# E-M integral can be done in closed form and that the high order
# are given by

#    n  /       \
#   d   | log x |   a + b log x
#   --- | ----- | = -----------
#     n |   2   |      2 + n
#   dx  \  x    /     x

# where a and b are integers given by a simple recurrence. Note
# that just one logarithm is needed. However, lots of integer
# logarithms are required for the initial summation.

# This algorithm could possibly be turned into a faster algorithm
# for general evaluation of zeta(s) or zeta'(s); this should be
# looked into.

@constant_memo
def glaisher_fixed(prec):
    wp = prec + 30
    # Number of direct terms to sum before applying the Euler-Maclaurin
    # formula to the tail. TODO: choose more intelligently
    N = int(0.33*prec + 5)
    ONE = MPZ_ONE << wp
    # Euler-Maclaurin, step 1: sum log(k)/k**2 for k from 2 to N-1
    s = MPZ_ZERO
    for k in range(2, N):
        #print k, N
        s += log_int_fixed(k, wp) // k**2
    logN = log_int_fixed(N, wp)
    #logN = to_fixed(mpf_log(from_int(N), wp+20), wp)
    # E-M step 2: integral of log(x)/x**2 from N to inf
    s += (ONE + logN) // N
    # E-M step 3: endpoint correction term f(N)/2
    s += logN // (N**2 * 2)
    # E-M step 4: the series of derivatives
    pN = N**3
    a = 1
    b = -2
    j = 3
    fac = from_int(2)
    k = 1
    while 1:
        # D(2*k-1) * B(2*k) / fac(2*k) [D(n) = nth derivative]
        D = ((a << wp) + b*logN) // pN
        D = from_man_exp(D, -wp)
        B = mpf_bernoulli(2*k, wp)
        term = mpf_mul(B, D, wp)
        term = mpf_div(term, fac, wp)
        term = to_fixed(term, wp)
        if abs(term) < 100:
            break
        #if not k % 10:
        #    print k, math.log(int(abs(term)), 10)
        s -= term
        # Advance derivative twice
        a, b, pN, j = b-a*j, -j*b, pN*N, j+1
        a, b, pN, j = b-a*j, -j*b, pN*N, j+1
        k += 1
        fac = mpf_mul_int(fac, (2*k)*(2*k-1), wp)
    # A = exp((6*s/pi**2 + log(2*pi) + euler)/12)
    pi = pi_fixed(wp)
    s *= 6
    s = (s << wp) // (pi**2 >> wp)
    s += euler_fixed(wp)
    s += to_fixed(mpf_log(from_man_exp(2*pi, -wp), wp), wp)
    s //= 12
    A = mpf_exp(from_man_exp(s, -wp), wp)
    return to_fixed(A, prec)

# Apery's constant can be computed using the very rapidly convergent
# series
#              oo
#              ___              2                      10
#             \         n  205 n  + 250 n + 77     (n!)
#  zeta(3) =   )    (-1)   -------------------  ----------
#             /___               64                      5
#             n = 0                             ((2n+1)!)

@constant_memo
def apery_fixed(prec):
    prec += 20
    d = MPZ_ONE << prec
    term = MPZ(77) << prec
    n = 1
    s = MPZ_ZERO
    while term:
        s += term
        d *= (n**10)
        d //= (((2*n+1)**5) * (2*n)**5)
        term = (-1)**n * (205*(n**2) + 250*n + 77) * d
        n += 1
    return s >> (20 + 6)

"""
Euler's constant (gamma) is computed using the Brent-McMillan formula,
gamma ~= I(n)/J(n) - log(n), where

   I(n) = sum_{k=0,1,2,...} (n**k / k!)**2 * H(k)
   J(n) = sum_{k=0,1,2,...} (n**k / k!)**2
   H(k) = 1 + 1/2 + 1/3 + ... + 1/k

The error is bounded by O(exp(-4n)). Choosing n to be a power
of two, 2**p, the logarithm becomes particularly easy to calculate.[1]

We use the formulation of Algorithm 3.9 in [2] to make the summation
more efficient.

Reference:
[1] Xavier Gourdon & Pascal Sebah, The Euler constant: gamma
http://numbers.computation.free.fr/Constants/Gamma/gamma.pdf

[2] [BorweinBailey]_
"""

@constant_memo
def euler_fixed(prec):
    extra = 30
    prec += extra
    # choose p such that exp(-4*(2**p)) < 2**-n
    p = int(math.log((prec/4) * math.log(2), 2)) + 1
    n = 2**p
    A = U = -p*ln2_fixed(prec)
    B = V = MPZ_ONE << prec
    k = 1
    while 1:
        B = B*n**2//k**2
        A = (A*n**2//k + B)//k
        U += A
        V += B
        if max(abs(A), abs(B)) < 100:
            break
        k += 1
    return (U<<(prec-extra))//V

# Use zeta accelerated formulas for the Mertens and twin
# prime constants; see
# http://mathworld.wolfram.com/MertensConstant.html
# http://mathworld.wolfram.com/TwinPrimesConstant.html

@constant_memo
def mertens_fixed(prec):
    wp = prec + 20
    m = 2
    s = mpf_euler(wp)
    while 1:
        t = mpf_zeta_int(m, wp)
        if t == fone:
            break
        t = mpf_log(t, wp)
        t = mpf_mul_int(t, moebius(m), wp)
        t = mpf_div(t, from_int(m), wp)
        s = mpf_add(s, t)
        m += 1
    return to_fixed(s, prec)

@constant_memo
def twinprime_fixed(prec):
    def I(n):
        return sum(moebius(d)<<(n//d) for d in xrange(1,n+1) if not n%d)//n
    wp = 2*prec + 30
    res = fone
    primes = [from_rational(1,p,wp) for p in [2,3,5,7]]
    ppowers = [mpf_mul(p,p,wp) for p in primes]
    n = 2
    while 1:
        a = mpf_zeta_int(n, wp)
        for i in range(4):
            a = mpf_mul(a, mpf_sub(fone, ppowers[i]), wp)
            ppowers[i] = mpf_mul(ppowers[i], primes[i], wp)
        a = mpf_pow_int(a, -I(n), wp)
        if mpf_pos(a, prec+10, 'n') == fone:
            break
        #from libmpf import to_str
        #print n, to_str(mpf_sub(fone, a), 6)
        res = mpf_mul(res, a, wp)
        n += 1
    res = mpf_mul(res, from_int(3*15*35), wp)
    res = mpf_div(res, from_int(4*16*36), wp)
    return to_fixed(res, prec)


mpf_euler = def_mpf_constant(euler_fixed)
mpf_apery = def_mpf_constant(apery_fixed)
mpf_khinchin = def_mpf_constant(khinchin_fixed)
mpf_glaisher = def_mpf_constant(glaisher_fixed)
mpf_catalan = def_mpf_constant(catalan_fixed)
mpf_mertens = def_mpf_constant(mertens_fixed)
mpf_twinprime = def_mpf_constant(twinprime_fixed)


#-----------------------------------------------------------------------#
#                                                                       #
#                          Bernoulli numbers                            #
#                                                                       #
#-----------------------------------------------------------------------#

MAX_BERNOULLI_CACHE = 3000


r"""
Small Bernoulli numbers and factorials are used in numerous summations,
so it is critical for speed that sequential computation is fast and that
values are cached up to a fairly high threshold.

On the other hand, we also want to support fast computation of isolated
large numbers. Currently, no such acceleration is provided for integer
factorials (though it is for large floating-point factorials, which are
computed via gamma if the precision is low enough).

For sequential computation of Bernoulli numbers, we use Ramanujan's formula

                           / n + 3 \
  B   =  (A(n) - S(n))  /  |       |
   n                       \   n   /

where A(n) = (n+3)/3 when n = 0 or 2 (mod 6), A(n) = -(n+3)/6
when n = 4 (mod 6), and

         [n/6]
          ___
         \      /  n + 3  \
  S(n) =  )     |         | * B
         /___   \ n - 6*k /    n-6*k
         k = 1

For isolated large Bernoulli numbers, we use the Riemann zeta function
to calculate a numerical value for B_n. The von Staudt-Clausen theorem
can then be used to optionally find the exact value of the
numerator and denominator.
"""

bernoulli_cache = {}
f3 = from_int(3)
f6 = from_int(6)

def bernoulli_size(n):
    """Accurately estimate the size of B_n (even n > 2 only)"""
    lgn = math.log(n,2)
    return int(2.326 + 0.5*lgn + n*(lgn - 4.094))

BERNOULLI_PREC_CUTOFF = bernoulli_size(MAX_BERNOULLI_CACHE)

def mpf_bernoulli(n, prec, rnd=None):
    """Computation of Bernoulli numbers (numerically)"""
    if n < 2:
        if n < 0:
            raise ValueError("Bernoulli numbers only defined for n >= 0")
        if n == 0:
            return fone
        if n == 1:
            return mpf_neg(fhalf)
    # For odd n > 1, the Bernoulli numbers are zero
    if n & 1:
        return fzero
    # If precision is extremely high, we can save time by computing
    # the Bernoulli number at a lower precision that is sufficient to
    # obtain the exact fraction, round to the exact fraction, and
    # convert the fraction back to an mpf value at the original precision
    if prec > BERNOULLI_PREC_CUTOFF and prec > bernoulli_size(n)*1.1 + 1000:
        p, q = bernfrac(n)
        return from_rational(p, q, prec, rnd or round_floor)
    if n > MAX_BERNOULLI_CACHE:
        return mpf_bernoulli_huge(n, prec, rnd)
    wp = prec + 30
    # Reuse nearby precisions
    wp += 32 - (prec & 31)
    cached = bernoulli_cache.get(wp)
    if cached:
        numbers, state = cached
        if n in numbers:
            if not rnd:
                return numbers[n]
            return mpf_pos(numbers[n], prec, rnd)
        m, bin, bin1 = state
        if n - m > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
    else:
        if n > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
        numbers = {0:fone}
        m, bin, bin1 = state = [2, MPZ(10), MPZ_ONE]
        bernoulli_cache[wp] = (numbers, state)
    while m <= n:
        #print m
        case = m % 6
        # Accurately estimate size of B_m so we can use
        # fixed point math without using too much precision
        szbm = bernoulli_size(m)
        s = 0
        sexp = max(0, szbm)  - wp
        if m < 6:
            a = MPZ_ZERO
        else:
            a = bin1
        for j in xrange(1, m//6+1):
            usign, uman, uexp, ubc = u = numbers[m-6*j]
            if usign:
                uman = -uman
            s += lshift(a*uman, uexp-sexp)
            # Update inner binomial coefficient
            j6 = 6*j
            a *= ((m-5-j6)*(m-4-j6)*(m-3-j6)*(m-2-j6)*(m-1-j6)*(m-j6))
            a //= ((4+j6)*(5+j6)*(6+j6)*(7+j6)*(8+j6)*(9+j6))
        if case == 0: b = mpf_rdiv_int(m+3, f3, wp)
        if case == 2: b = mpf_rdiv_int(m+3, f3, wp)
        if case == 4: b = mpf_rdiv_int(-m-3, f6, wp)
        s = from_man_exp(s, sexp, wp)
        b = mpf_div(mpf_sub(b, s, wp), from_int(bin), wp)
        numbers[m] = b
        m += 2
        # Update outer binomial coefficient
        bin = bin * ((m+2)*(m+3)) // (m*(m-1))
        if m > 6:
            bin1 = bin1 * ((2+m)*(3+m)) // ((m-7)*(m-6))
        state[:] = [m, bin, bin1]
    return numbers[n]

def mpf_bernoulli_huge(n, prec, rnd=None):
    wp = prec + 10
    piprec = wp + int(math.log(n,2))
    v = mpf_gamma_int(n+1, wp)
    v = mpf_mul(v, mpf_zeta_int(n, wp), wp)
    v = mpf_mul(v, mpf_pow_int(mpf_pi(piprec), -n, wp))
    v = mpf_shift(v, 1-n)
    if not n & 3:
        v = mpf_neg(v)
    return mpf_pos(v, prec, rnd or round_fast)

def bernfrac(n):
    r"""
    Returns a tuple of integers `(p, q)` such that `p/q = B_n` exactly,
    where `B_n` denotes the `n`-th Bernoulli number. The fraction is
    always reduced to lowest terms. Note that for `n > 1` and `n` odd,
    `B_n = 0`, and `(0, 1)` is returned.

    **Examples**

    The first few Bernoulli numbers are exactly::

        >>> from mpmath import *
        >>> for n in range(15):
        ...     p, q = bernfrac(n)
        ...     print("%s %s/%s" % (n, p, q))
        ...
        0 1/1
        1 -1/2
        2 1/6
        3 0/1
        4 -1/30
        5 0/1
        6 1/42
        7 0/1
        8 -1/30
        9 0/1
        10 5/66
        11 0/1
        12 -691/2730
        13 0/1
        14 7/6

    This function works for arbitrarily large `n`::

        >>> p, q = bernfrac(10**4)
        >>> print(q)
        2338224387510
        >>> print(len(str(p)))
        27692
        >>> mp.dps = 15
        >>> print(mpf(p) / q)
        -9.04942396360948e+27677
        >>> print(bernoulli(10**4))
        -9.04942396360948e+27677

    .. note ::

        :func:`~mpmath.bernoulli` computes a floating-point approximation
        directly, without computing the exact fraction first.
        This is much faster for large `n`.

    **Algorithm**

    :func:`~mpmath.bernfrac` works by computing the value of `B_n` numerically
    and then using the von Staudt-Clausen theorem [1] to reconstruct
    the exact fraction. For large `n`, this is significantly faster than
    computing `B_1, B_2, \ldots, B_2` recursively with exact arithmetic.
    The implementation has been tested for `n = 10^m` up to `m = 6`.

    In practice, :func:`~mpmath.bernfrac` appears to be about three times
    slower than the specialized program calcbn.exe [2]

    **References**

    1. MathWorld, von Staudt-Clausen Theorem:
       http://mathworld.wolfram.com/vonStaudt-ClausenTheorem.html

    2. The Bernoulli Number Page:
       http://www.bernoulli.org/

    """
    n = int(n)
    if n < 3:
        return [(1, 1), (-1, 2), (1, 6)][n]
    if n & 1:
        return (0, 1)
    q = 1
    for k in list_primes(n+1):
        if not (n % (k-1)):
            q *= k
    prec = bernoulli_size(n) + int(math.log(q,2)) + 20
    b = mpf_bernoulli(n, prec)
    p = mpf_mul(b, from_int(q))
    pint = to_int(p, round_nearest)
    return (pint, q)


#-----------------------------------------------------------------------#
#                                                                       #
#                         Polygamma functions                           #
#                                                                       #
#-----------------------------------------------------------------------#

r"""
For all polygamma (psi) functions, we use the Euler-Maclaurin summation
formula. It looks slightly different in the m = 0 and m > 0 cases.

For m = 0, we have
                                 oo
                                ___   B
       (0)                1    \       2 k    -2 k
    psi   (z)  ~ log z + --- -  )    ------  z
                         2 z   /___  (2 k)!
                               k = 1

Experiment shows that the minimum term of the asymptotic series
reaches 2^(-p) when Re(z) > 0.11*p. So we simply use the recurrence
for psi (equivalent, in fact, to summing to the first few terms
directly before applying E-M) to obtain z large enough.

Since, very crudely, log z ~= 1 for Re(z) > 1, we can use
fixed-point arithmetic  (if z is extremely large, log(z) itself
is a sufficient approximation, so we can stop there already).

For Re(z) << 0, we could use recurrence, but this is of course
inefficient for large negative z, so there we use the
reflection formula instead.

For m > 0, we have

                  N - 1
                   ___
  ~~~(m)       [  \          1    ]         1            1
  psi   (z)  ~ [   )     -------- ] +  ---------- +  -------- +
               [  /___        m+1 ]           m+1           m
                  k = 1  (z+k)    ]    2 (z+N)       m (z+N)

      oo
     ___    B
    \        2 k   (m+1) (m+2) ... (m+2k-1)
  +  )     ------  ------------------------
    /___   (2 k)!            m + 2 k
    k = 1               (z+N)

where ~~~ denotes the function rescaled by 1/((-1)^(m+1) m!).

Here again N is chosen to make z+N large enough for the minimum
term in the last series to become smaller than eps.

TODO: the current estimation of N for m > 0 is *very suboptimal*.

TODO: implement the reflection formula for m > 0, Re(z) << 0.
It is generally a combination of multiple cotangents. Need to
figure out a reasonably simple way to generate these formulas
on the fly.

TODO: maybe use exact algorithms to compute psi for integral
and certain rational arguments, as this can be much more
efficient. (On the other hand, the availability of these
special values provides a convenient way to test the general
algorithm.)
"""

# Harmonic numbers are just shifted digamma functions
# We should calculate these exactly when x is an integer
# and when doing so is faster.

def mpf_harmonic(x, prec, rnd):
    if x in (fzero, fnan, finf):
        return x
    a = mpf_psi0(mpf_add(fone, x, prec+5), prec)
    return mpf_add(a, mpf_euler(prec+5, rnd), prec, rnd)

def mpc_harmonic(z, prec, rnd):
    if z[1] == fzero:
        return (mpf_harmonic(z[0], prec, rnd), fzero)
    a = mpc_psi0(mpc_add_mpf(z, fone, prec+5), prec)
    return mpc_add_mpf(a, mpf_euler(prec+5, rnd), prec, rnd)

def mpf_psi0(x, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a real argument.
    """
    sign, man, exp, bc = x
    wp = prec + 10
    if not man:
        if x == finf: return x
        if x == fninf or x == fnan: return fnan
    if x == fzero or (exp >= 0 and sign):
        raise ValueError("polygamma pole")
    # Near 0 -- fixed-point arithmetic becomes bad
    if exp+bc < -5:
        v = mpf_psi0(mpf_add(x, fone, prec, rnd), prec, rnd)
        return mpf_sub(v, mpf_div(fone, x, wp, rnd), prec, rnd)
    # Reflection formula
    if sign and exp+bc > 3:
        c, s = mpf_cos_sin_pi(x, wp)
        q = mpf_mul(mpf_div(c, s, wp), mpf_pi(wp), wp)
        p = mpf_psi0(mpf_sub(fone, x, wp), wp)
        return mpf_sub(p, q, prec, rnd)
    # The logarithmic term is accurate enough
    if (not sign) and bc + exp > wp:
        return mpf_log(mpf_sub(x, fone, wp), prec, rnd)
    # Initial recurrence to obtain a large enough x
    m = to_int(x)
    n = int(0.11*wp) + 2
    s = MPZ_ZERO
    x = to_fixed(x, wp)
    one = MPZ_ONE << wp
    if m < n:
        for k in xrange(m, n):
            s -= (one << wp) // x
            x += one
    x -= one
    # Logarithmic term
    s += to_fixed(mpf_log(from_man_exp(x, -wp, wp), wp), wp)
    # Endpoint term in Euler-Maclaurin expansion
    s += (one << wp) // (2*x)
    # Euler-Maclaurin remainder sum
    x2 = (x*x) >> wp
    t = one
    prev = 0
    k = 1
    while 1:
        t = (t*x2) >> wp
        bsign, bman, bexp, bbc = mpf_bernoulli(2*k, wp)
        offset = (bexp + 2*wp)
        if offset >= 0: term = (bman << offset) // (t*(2*k))
        else:           term = (bman >> (-offset)) // (t*(2*k))
        if k & 1: s -= term
        else:     s += term
        if k > 2 and term >= prev:
            break
        prev = term
        k += 1
    return from_man_exp(s, -wp, wp, rnd)

def mpc_psi0(z, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a complex argument.
    """
    re, im = z
    # Fall back to the real case
    if im == fzero:
        return (mpf_psi0(re, prec, rnd), fzero)
    wp = prec + 20
    sign, man, exp, bc = re
    # Reflection formula
    if sign and exp+bc > 3:
        c = mpc_cos_pi(z, wp)
        s = mpc_sin_pi(z, wp)
        q = mpc_mul_mpf(mpc_div(c, s, wp), mpf_pi(wp), wp)
        p = mpc_psi0(mpc_sub(mpc_one, z, wp), wp)
        return mpc_sub(p, q, prec, rnd)
    # Just the logarithmic term
    if (not sign) and bc + exp > wp:
        return mpc_log(mpc_sub(z, mpc_one, wp), prec, rnd)
    # Initial recurrence to obtain a large enough z
    w = to_int(re)
    n = int(0.11*wp) + 2
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            s = mpc_sub(s, mpc_reciprocal(z, wp), wp)
            z = mpc_add_mpf(z, fone, wp)
    z = mpc_sub(z, mpc_one, wp)
    # Logarithmic and endpoint term
    s = mpc_add(s, mpc_log(z, wp), wp)
    s = mpc_add(s, mpc_div(mpc_half, z, wp), wp)
    # Euler-Maclaurin remainder sum
    z2 = mpc_square(z, wp)
    t = mpc_one
    prev = mpc_zero
    szprev = fzero
    k = 1
    eps = mpf_shift(fone, -wp+2)
    while 1:
        t = mpc_mul(t, z2, wp)
        bern = mpf_bernoulli(2*k, wp)
        term = mpc_mpf_div(bern, mpc_mul_int(t, 2*k, wp), wp)
        s = mpc_sub(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and (mpf_le(szterm, eps) or mpf_le(szprev, szterm)):
            break
        prev = term
        szprev = szterm
        k += 1
    return s

# Currently unoptimized
def mpf_psi(m, x, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a real argument x.
    """
    if m == 0:
        return mpf_psi0(x, prec, rnd=round_fast)
    return mpc_psi(m, (x, fzero), prec, rnd)[0]

def mpc_psi(m, z, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a complex argument z.
    """
    if m == 0:
        return mpc_psi0(z, prec, rnd)
    re, im = z
    wp = prec + 20
    sign, man, exp, bc = re
    if not im[1]:
        if im in (finf, fninf, fnan):
            return (fnan, fnan)
    if not man:
        if re == finf and im == fzero:
            return (fzero, fzero)
        if re == fnan:
            return (fnan, fnan)
    # Recurrence
    w = to_int(re)
    n = int(0.4*wp + 4*m)
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            t = mpc_pow_int(z, -m-1, wp)
            s = mpc_add(s, t, wp)
            z = mpc_add_mpf(z, fone, wp)
    zm = mpc_pow_int(z, -m, wp)
    z2 = mpc_pow_int(z, -2, wp)
    # 1/m*(z+N)^m
    integral_term = mpc_div_mpf(zm, from_int(m), wp)
    s = mpc_add(s, integral_term, wp)
    # 1/2*(z+N)^(-(m+1))
    s = mpc_add(s, mpc_mul_mpf(mpc_div(zm, z, wp), fhalf, wp), wp)
    a = m + 1
    b = 2
    k = 1
    # Important: we want to sum up to the *relative* error,
    # not the absolute error, because psi^(m)(z) might be tiny
    magn = mpc_abs(s, 10)
    magn = magn[2]+magn[3]
    eps = mpf_shift(fone, magn-wp+2)
    while 1:
        zm = mpc_mul(zm, z2, wp)
        bern = mpf_bernoulli(2*k, wp)
        scal = mpf_mul_int(bern, a, wp)
        scal = mpf_div(scal, from_int(b), wp)
        term = mpc_mul_mpf(zm, scal, wp)
        s = mpc_add(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and mpf_le(szterm, eps):
            break
        #print k, to_str(szterm, 10), to_str(eps, 10)
        a *= (m+2*k)*(m+2*k+1)
        b *= (2*k+1)*(2*k+2)
        k += 1
    # Scale and sign factor
    v = mpc_mul_mpf(s, mpf_gamma(from_int(m+1), wp), prec, rnd)
    if not (m & 1):
        v = mpf_neg(v[0]), mpf_neg(v[1])
    return v


#-----------------------------------------------------------------------#
#                                                                       #
#                         Riemann zeta function                         #
#                                                                       #
#-----------------------------------------------------------------------#

r"""
We use zeta(s) = eta(s) / (1 - 2**(1-s)) and Borwein's approximation

                  n-1
                  ___       k
             -1  \      (-1)  (d_k - d_n)
  eta(s) ~= ----  )     ------------------
             d_n /___              s
                 k = 0      (k + 1)
where
             k
             ___                i
            \     (n + i - 1)! 4
  d_k  =  n  )    ---------------.
            /___   (n - i)! (2i)!
            i = 0

If s = a + b*I, the absolute error for eta(s) is bounded by

    3 (1 + 2|b|)
    ------------ * exp(|b| pi/2)
               n
    (3+sqrt(8))

Disregarding the linear term, we have approximately,

  log(err) ~= log(exp(1.58*|b|)) - log(5.8**n)
  log(err) ~= 1.58*|b| - log(5.8)*n
  log(err) ~= 1.58*|b| - 1.76*n
  log2(err) ~= 2.28*|b| - 2.54*n

So for p bits, we should choose n > (p + 2.28*|b|) / 2.54.

References:
-----------

Peter Borwein, "An Efficient Algorithm for the Riemann Zeta Function"
http://www.cecm.sfu.ca/personal/pborwein/PAPERS/P117.ps

http://en.wikipedia.org/wiki/Dirichlet_eta_function
"""

borwein_cache = {}

def borwein_coefficients(n):
    if n in borwein_cache:
        return borwein_cache[n]
    ds = [MPZ_ZERO] * (n+1)
    d = MPZ_ONE
    s = ds[0] = MPZ_ONE
    for i in range(1, n+1):
        d = d * 4 * (n+i-1) * (n-i+1)
        d //= ((2*i) * ((2*i)-1))
        s += d
        ds[i] = s
    borwein_cache[n] = ds
    return ds

ZETA_INT_CACHE_MAX_PREC = 1000
zeta_int_cache = {}

def mpf_zeta_int(s, prec, rnd=round_fast):
    """
    Optimized computation of zeta(s) for an integer s.
    """
    wp = prec + 20
    s = int(s)
    if s in zeta_int_cache and zeta_int_cache[s][0] >= wp:
        return mpf_pos(zeta_int_cache[s][1], prec, rnd)
    if s < 2:
        if s == 1:
            raise ValueError("zeta(1) pole")
        if not s:
            return mpf_neg(fhalf)
        return mpf_div(mpf_bernoulli(-s+1, wp), from_int(s-1), prec, rnd)
    # 2^-s term vanishes?
    if s >= wp:
        return mpf_perturb(fone, 0, prec, rnd)
    # 5^-s term vanishes?
    elif s >= wp*0.431:
        t = one = 1 << wp
        t += 1 << (wp - s)
        t += one // (MPZ_THREE ** s)
        t += 1 << max(0, wp - s*2)
        return from_man_exp(t, -wp, prec, rnd)
    else:
        # Fast enough to sum directly?
        # Even better, we use the Euler product (idea stolen from pari)
        m = (float(wp)/(s-1) + 1)
        if m < 30:
            needed_terms = int(2.0**m + 1)
            if needed_terms < int(wp/2.54 + 5) / 10:
                t = fone
                for k in list_primes(needed_terms):
                    #print k, needed_terms
                    powprec = int(wp - s*math.log(k,2))
                    if powprec < 2:
                        break
                    a = mpf_sub(fone, mpf_pow_int(from_int(k), -s, powprec), wp)
                    t = mpf_mul(t, a, wp)
                return mpf_div(fone, t, wp)
    # Use Borwein's algorithm
    n = int(wp/2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    s = MPZ(s)
    for k in xrange(n):
        t += (((-1)**k * (d[k] - d[n])) << wp) // (k+1)**s
    t = (t << wp) // (-d[n])
    t = (t << wp) // ((1 << wp) - (1 << (wp+1-s)))
    if (s in zeta_int_cache and zeta_int_cache[s][0] < wp) or (s not in zeta_int_cache):
        zeta_int_cache[s] = (wp, from_man_exp(t, -wp-wp))
    return from_man_exp(t, -wp-wp, prec, rnd)

def mpf_zeta(s, prec, rnd=round_fast, alt=0):
    sign, man, exp, bc = s
    if not man:
        if s == fzero:
            if alt:
                return fhalf
            else:
                return mpf_neg(fhalf)
        if s == finf:
            return fone
        return fnan
    wp = prec + 20
    # First term vanishes?
    if (not sign) and (exp + bc > (math.log(wp,2) + 2)):
        return mpf_perturb(fone, alt, prec, rnd)
    # Optimize for integer arguments
    elif exp >= 0:
        if alt:
            if s == fone:
                return mpf_ln2(prec, rnd)
            z = mpf_zeta_int(to_int(s), wp, negative_rnd[rnd])
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(z, q, prec, rnd)
        else:
            return mpf_zeta_int(to_int(s), prec, rnd)
    # Negative: use the reflection formula
    # Borwein only proves the accuracy bound for x >= 1/2. However, based on
    # tests, the accuracy without reflection is quite good even some distance
    # to the left of 1/2. XXX: verify this.
    if sign:
        # XXX: could use the separate refl. formula for Dirichlet eta
        if alt:
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(mpf_zeta(s, wp), q, prec, rnd)
        # XXX: -1 should be done exactly
        y = mpf_sub(fone, s, 10*wp)
        a = mpf_gamma(y, wp)
        b = mpf_zeta(y, wp)
        c = mpf_sin_pi(mpf_shift(s, -1), wp)
        wp2 = wp + max(0,exp+bc)
        pi = mpf_pi(wp+wp2)
        d = mpf_div(mpf_pow(mpf_shift(pi, 1), s, wp2), pi, wp2)
        return mpf_mul(a,mpf_mul(b,mpf_mul(c,d,wp),wp),prec,rnd)

    # Near pole
    r = mpf_sub(fone, s, wp)
    asign, aman, aexp, abc = mpf_abs(r)
    pole_dist = -2*(aexp+abc)
    if pole_dist > wp:
        if alt:
            return mpf_ln2(prec, rnd)
        else:
            q = mpf_neg(mpf_div(fone, r, wp))
            return mpf_add(q, mpf_euler(wp), prec, rnd)
    else:
        wp += max(0, pole_dist)

    t = MPZ_ZERO
    #wp += 16 - (prec & 15)
    # Use Borwein's algorithm
    n = int(wp/2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    sf = to_fixed(s, wp)
    ln2 = ln2_fixed(wp)
    for k in xrange(n):
        u = (-sf*log_int_fixed(k+1, wp, ln2)) >> wp
        #esign, eman, eexp, ebc = mpf_exp(u, wp)
        #offset = eexp + wp
        #if offset >= 0:
        #    w = ((d[k] - d[n]) * eman) << offset
        #else:
        #    w = ((d[k] - d[n]) * eman) >> (-offset)
        eman = exp_fixed(u, wp, ln2)
        w = (d[k] - d[n]) * eman
        if k & 1:
            t -= w
        else:
            t += w
    t = t // (-d[n])
    t = from_man_exp(t, -wp, wp)
    if alt:
        return mpf_pos(t, prec, rnd)
    else:
        q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
        return mpf_div(t, q, prec, rnd)

def mpc_zeta(s, prec, rnd=round_fast, alt=0, force=False):
    re, im = s
    if im == fzero:
        return mpf_zeta(re, prec, rnd, alt), fzero

    # slow for large s
    if (not force) and mpf_gt(mpc_abs(s, 10), from_int(prec)):
        raise NotImplementedError

    wp = prec + 20

    # Near pole
    r = mpc_sub(mpc_one, s, wp)
    asign, aman, aexp, abc = mpc_abs(r, 10)
    pole_dist = -2*(aexp+abc)
    if pole_dist > wp:
        if alt:
            q = mpf_ln2(wp)
            y = mpf_mul(q, mpf_euler(wp), wp)
            g = mpf_shift(mpf_mul(q, q, wp), -1)
            g = mpf_sub(y, g)
            z = mpc_mul_mpf(r, mpf_neg(g), wp)
            z = mpc_add_mpf(z, q, wp)
            return mpc_pos(z, prec, rnd)
        else:
            q = mpc_neg(mpc_div(mpc_one, r, wp))
            q = mpc_add_mpf(q, mpf_euler(wp), wp)
            return mpc_pos(q, prec, rnd)
    else:
        wp += max(0, pole_dist)

    # Reflection formula. To be rigorous, we should reflect to the left of
    # re = 1/2 (see comments for mpf_zeta), but this leads to unnecessary
    # slowdown for interesting values of s
    if mpf_lt(re, fzero):
        # XXX: could use the separate refl. formula for Dirichlet eta
        if alt:
            q = mpc_sub(mpc_one, mpc_pow(mpc_two, mpc_sub(mpc_one, s, wp),
                wp), wp)
            return mpc_mul(mpc_zeta(s, wp), q, prec, rnd)
        # XXX: -1 should be done exactly
        y = mpc_sub(mpc_one, s, 10*wp)
        a = mpc_gamma(y, wp)
        b = mpc_zeta(y, wp)
        c = mpc_sin_pi(mpc_shift(s, -1), wp)
        rsign, rman, rexp, rbc = re
        isign, iman, iexp, ibc = im
        mag = max(rexp+rbc, iexp+ibc)
        wp2 = wp + max(0, mag)
        pi = mpf_pi(wp+wp2)
        pi2 = (mpf_shift(pi, 1), fzero)
        d = mpc_div_mpf(mpc_pow(pi2, s, wp2), pi, wp2)
        return mpc_mul(a,mpc_mul(b,mpc_mul(c,d,wp),wp),prec,rnd)
    n = int(wp/2.54 + 5)
    n += int(0.9*abs(to_int(im)))
    d = borwein_coefficients(n)
    ref = to_fixed(re, wp)
    imf = to_fixed(im, wp)
    tre = MPZ_ZERO
    tim = MPZ_ZERO
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)
    critical_line = re == fhalf
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    wp2 = wp+wp
    for k in xrange(n):
        log = log_int_fixed(k+1, wp, ln2)
        # A square root is much cheaper than an exp
        if critical_line:
            w = one_2wp // isqrt_fast((k+1) << wp2)
        else:
            w = exp_fixed((-ref*log) >> wp, wp)
        if k & 1:
            w *= (d[n] - d[k])
        else:
            w *= (d[k] - d[n])
        wre, wim = cos_sin_fixed((-imf*log)>>wp, wp, pi2)
        tre += (w * wre) >> wp
        tim += (w * wim) >> wp
    tre //= (-d[n])
    tim //= (-d[n])
    tre = from_man_exp(tre, -wp, wp)
    tim = from_man_exp(tim, -wp, wp)
    if alt:
        return mpc_pos((tre, tim), prec, rnd)
    else:
        q = mpc_sub(mpc_one, mpc_pow(mpc_two, r, wp), wp)
        return mpc_div((tre, tim), q, prec, rnd)

def mpf_altzeta(s, prec, rnd=round_fast):
    return mpf_zeta(s, prec, rnd, 1)

def mpc_altzeta(s, prec, rnd=round_fast):
    return mpc_zeta(s, prec, rnd, 1)

# Not optimized currently
mpf_zetasum = None


def pow_fixed(x, n, wp):
    if n == 1:
        return x
    y = MPZ_ONE << wp
    while n:
        if n & 1:
            y = (y*x) >> wp
            n -= 1
        x = (x*x) >> wp
        n //= 2
    return y

# TODO: optimize / cleanup interface / unify with list_primes
sieve_cache = []
primes_cache = []
mult_cache = []

def primesieve(n):
    global sieve_cache, primes_cache, mult_cache
    if n < len(sieve_cache):
        sieve = sieve_cache#[:n+1]
        primes = primes_cache[:primes_cache.index(max(sieve))+1]
        mult = mult_cache#[:n+1]
        return sieve, primes, mult
    sieve = [0] * (n+1)
    mult = [0] * (n+1)
    primes = list_primes(n)
    for p in primes:
        #sieve[p::p] = p
        for k in xrange(p,n+1,p):
            sieve[k] = p
    for i, p in enumerate(sieve):
        if i >= 2:
            m = 1
            n = i // p
            while not n % p:
                n //= p
                m += 1
            mult[i] = m
    sieve_cache = sieve
    primes_cache = primes
    mult_cache = mult
    return sieve, primes, mult

def zetasum_sieved(critical_line, sre, sim, a, n, wp):
    if a < 1:
        raise ValueError("a cannot be less than 1")
    sieve, primes, mult = primesieve(a+n)
    basic_powers = {}
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)
    wp2 = wp+wp
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    for p in primes:
        if p*2 > a+n:
            break
        log = log_int_fixed(p, wp, ln2)
        cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(p<<wp2)
        else:
            u = exp_fixed((-sre*log)>>wp, wp)
        pre = (u*cos) >> wp
        pim = (u*sin) >> wp
        basic_powers[p] = [(pre, pim)]
        tre, tim = pre, pim
        for m in range(1,int(math.log(a+n,p)+0.01)+1):
            tre, tim = ((pre*tre-pim*tim)>>wp), ((pim*tre+pre*tim)>>wp)
            basic_powers[p].append((tre,tim))
    xre = MPZ_ZERO
    xim = MPZ_ZERO
    if a == 1:
        xre += one
    aa = max(a,2)
    for k in xrange(aa, a+n+1):
        p = sieve[k]
        if p in basic_powers:
            m = mult[k]
            tre, tim = basic_powers[p][m-1]
            while 1:
                k //= p**m
                if k == 1:
                    break
                p = sieve[k]
                m = mult[k]
                pre, pim = basic_powers[p][m-1]
                tre, tim = ((pre*tre-pim*tim)>>wp), ((pim*tre+pre*tim)>>wp)
        else:
            log = log_int_fixed(k, wp, ln2)
            cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
            if critical_line:
                u = one_2wp // isqrt_fast(k<<wp2)
            else:
                u = exp_fixed((-sre*log)>>wp, wp)
            tre = (u*cos) >> wp
            tim = (u*sin) >> wp
        xre += tre
        xim += tim
    return xre, xim

# Set to something large to disable
ZETASUM_SIEVE_CUTOFF = 10

def mpc_zetasum(s, a, n, derivatives, reflect, prec):
    """
    Fast version of mp._zetasum, assuming s = complex, a = integer.
    """

    wp = prec + 10
    derivatives = list(derivatives)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1

    # parse s
    sre, sim = s
    critical_line = (sre == fhalf)
    sre = to_fixed(sre, wp)
    sim = to_fixed(sim, wp)

    if a > 0 and n > ZETASUM_SIEVE_CUTOFF and not have_derivatives \
            and not reflect and (n < 4e7 or sys.maxsize > 2**32):
        re, im = zetasum_sieved(critical_line, sre, sim, a, n, wp)
        xs = [(from_man_exp(re, -wp, prec, 'n'), from_man_exp(im, -wp, prec, 'n'))]
        return xs, []

    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd+1)

    # x_d = 0, y_d = 0
    xre = [MPZ_ZERO for d in derivatives]
    xim = [MPZ_ZERO for d in derivatives]
    if reflect:
        yre = [MPZ_ZERO for d in derivatives]
        yim = [MPZ_ZERO for d in derivatives]
    else:
        yre = yim = []

    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)

    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    wp2 = wp+wp

    for w in xrange(a, a+n+1):
        log = log_int_fixed(w, wp, ln2)
        cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(w<<wp2)
        else:
            u = exp_fixed((-sre*log)>>wp, wp)
        xterm_re = (u * cos) >> wp
        xterm_im = (u * sin) >> wp
        if reflect:
            reciprocal = (one_2wp // (u*w))
            yterm_re = (reciprocal * cos) >> wp
            yterm_im = (reciprocal * sin) >> wp

        if have_derivatives:
            if have_one_derivative:
                log = pow_fixed(log, maxd, wp)
                xre[0] += (xterm_re * log) >> wp
                xim[0] += (xterm_im * log) >> wp
                if reflect:
                    yre[0] += (yterm_re * log) >> wp
                    yim[0] += (yterm_im * log) >> wp
            else:
                t = MPZ_ONE << wp
                for d in derivatives:
                    xre[d] += (xterm_re * t) >> wp
                    xim[d] += (xterm_im * t) >> wp
                    if reflect:
                        yre[d] += (yterm_re * t) >> wp
                        yim[d] += (yterm_im * t) >> wp
                    t = (t * log) >> wp
        else:
            xre[0] += xterm_re
            xim[0] += xterm_im
            if reflect:
                yre[0] += yterm_re
                yim[0] += yterm_im
    if have_derivatives:
        if have_one_derivative:
            if maxd % 2:
                xre[0] = -xre[0]
                xim[0] = -xim[0]
                if reflect:
                    yre[0] = -yre[0]
                    yim[0] = -yim[0]
        else:
            xre = [(-1)**d * xre[d] for d in derivatives]
            xim = [(-1)**d * xim[d] for d in derivatives]
            if reflect:
                yre = [(-1)**d * yre[d] for d in derivatives]
                yim = [(-1)**d * yim[d] for d in derivatives]
    xs = [(from_man_exp(xa, -wp, prec, 'n'), from_man_exp(xb, -wp, prec, 'n'))
        for (xa, xb) in zip(xre, xim)]
    ys = [(from_man_exp(ya, -wp, prec, 'n'), from_man_exp(yb, -wp, prec, 'n'))
        for (ya, yb) in zip(yre, yim)]
    return xs, ys


#-----------------------------------------------------------------------#
#                                                                       #
#              The gamma function  (NEW IMPLEMENTATION)                 #
#                                                                       #
#-----------------------------------------------------------------------#

# Higher means faster, but more precomputation time
MAX_GAMMA_TAYLOR_PREC = 5000
# Need to derive higher bounds for Taylor series to go higher
assert MAX_GAMMA_TAYLOR_PREC < 15000

# Use Stirling's series if abs(x) > beta*prec
# Important: must be large enough for convergence!
GAMMA_STIRLING_BETA = 0.2

SMALL_FACTORIAL_CACHE_SIZE = 150

gamma_taylor_cache = {}
gamma_stirling_cache = {}

small_factorial_cache = [from_int(ifac(n)) for \
    n in range(SMALL_FACTORIAL_CACHE_SIZE+1)]

def zeta_array(N, prec):
    """
    zeta(n) = A * pi**n / n! + B

    where A is a rational number (A = Bernoulli number
    for n even) and B is an infinite sum over powers of exp(2*pi).
    (B = 0 for n even).

    TODO: this is currently only used for gamma, but could
    be very useful elsewhere.
    """
    extra = 30
    wp = prec+extra
    zeta_values = [MPZ_ZERO] * (N+2)
    pi = pi_fixed(wp)
    # STEP 1:
    one = MPZ_ONE << wp
    zeta_values[0] = -one//2
    f_2pi = mpf_shift(mpf_pi(wp),1)
    exp_2pi_k = exp_2pi = mpf_exp(f_2pi, wp)
    # Compute exponential series
    # Store values of 1/(exp(2*pi*k)-1),
    # exp(2*pi*k)/(exp(2*pi*k)-1)**2, 1/(exp(2*pi*k)-1)**2
    # pi*k*exp(2*pi*k)/(exp(2*pi*k)-1)**2
    exps3 = []
    k = 1
    while 1:
        tp = wp - 9*k
        if tp < 1:
            break
        # 1/(exp(2*pi*k-1)
        q1 = mpf_div(fone, mpf_sub(exp_2pi_k, fone, tp), tp)
        # pi*k*exp(2*pi*k)/(exp(2*pi*k)-1)**2
        q2 = mpf_mul(exp_2pi_k, mpf_mul(q1,q1,tp), tp)
        q1 = to_fixed(q1, wp)
        q2 = to_fixed(q2, wp)
        q2 = (k * q2 * pi) >> wp
        exps3.append((q1, q2))
        # Multiply for next round
        exp_2pi_k = mpf_mul(exp_2pi_k, exp_2pi, wp)
        k += 1
    # Exponential sum
    for n in xrange(3, N+1, 2):
        s = MPZ_ZERO
        k = 1
        for e1, e2 in exps3:
            if n%4 == 3:
                t = e1 // k**n
            else:
                U = (n-1)//4
                t = (e1 + e2//U) // k**n
            if not t:
                break
            s += t
            k += 1
        zeta_values[n] = -2*s
    # Even zeta values
    B = [mpf_abs(mpf_bernoulli(k,wp)) for k in xrange(N+2)]
    pi_pow = fpi = mpf_pow_int(mpf_shift(mpf_pi(wp), 1), 2, wp)
    pi_pow = mpf_div(pi_pow, from_int(4), wp)
    for n in xrange(2,N+2,2):
        z = mpf_mul(B[n], pi_pow, wp)
        zeta_values[n] = to_fixed(z, wp)
        pi_pow = mpf_mul(pi_pow, fpi, wp)
        pi_pow = mpf_div(pi_pow, from_int((n+1)*(n+2)), wp)
    # Zeta sum
    reciprocal_pi = (one << wp) // pi
    for n in xrange(3, N+1, 4):
        U = (n-3)//4
        s = zeta_values[4*U+4]*(4*U+7)//4
        for k in xrange(1, U+1):
            s -= (zeta_values[4*k] * zeta_values[4*U+4-4*k]) >> wp
        zeta_values[n] += (2*s*reciprocal_pi) >> wp
    for n in xrange(5, N+1, 4):
        U = (n-1)//4
        s = zeta_values[4*U+2]*(2*U+1)
        for k in xrange(1, 2*U+1):
            s += ((-1)**k*2*k* zeta_values[2*k] * zeta_values[4*U+2-2*k])>>wp
        zeta_values[n] += ((s*reciprocal_pi)>>wp)//(2*U)
    return [x>>extra for x in zeta_values]

def gamma_taylor_coefficients(inprec):
    """
    Gives the Taylor coefficients of 1/gamma(1+x) as
    a list of fixed-point numbers. Enough coefficients are returned
    to ensure that the series converges to the given precision
    when x is in [0.5, 1.5].
    """
    # Reuse nearby cache values (small case)
    if inprec < 400:
        prec = inprec + (10-(inprec%10))
    elif inprec < 1000:
        prec = inprec + (30-(inprec%30))
    else:
        prec = inprec
    if prec in gamma_taylor_cache:
        return gamma_taylor_cache[prec], prec

    # Experimentally determined bounds
    if prec < 1000:
        N = int(prec**0.76 + 2)
    else:
        # Valid to at least 15000 bits
        N = int(prec**0.787 + 2)

    # Reuse higher precision values
    for cprec in gamma_taylor_cache:
        if cprec > prec:
            coeffs = [x>>(cprec-prec) for x in gamma_taylor_cache[cprec][-N:]]
            if inprec < 1000:
                gamma_taylor_cache[prec] = coeffs
            return coeffs, prec

    # Cache at a higher precision (large case)
    if prec > 1000:
        prec = int(prec * 1.2)

    wp = prec + 20
    A = [0] * N
    A[0] = MPZ_ZERO
    A[1] = MPZ_ONE << wp
    A[2] = euler_fixed(wp)
    # SLOW, reference implementation
    #zeta_values = [0,0]+[to_fixed(mpf_zeta_int(k,wp),wp) for k in xrange(2,N)]
    zeta_values = zeta_array(N, wp)
    for k in xrange(3, N):
        a = (-A[2]*A[k-1])>>wp
        for j in xrange(2,k):
            a += ((-1)**j * zeta_values[j] * A[k-j]) >> wp
        a //= (1-k)
        A[k] = a
    A = [a>>20 for a in A]
    A = A[::-1]
    A = A[:-1]
    gamma_taylor_cache[prec] = A
    #return A, prec
    return gamma_taylor_coefficients(inprec)

def gamma_fixed_taylor(xmpf, x, wp, prec, rnd, type):
    # Determine nearest multiple of N/2
    #n = int(x >> (wp-1))
    #steps = (n-1)>>1
    nearest_int = ((x >> (wp-1)) + MPZ_ONE) >> 1
    one = MPZ_ONE << wp
    coeffs, cwp = gamma_taylor_coefficients(wp)
    if nearest_int > 0:
        r = one
        for i in xrange(nearest_int-1):
            x -= one
            r = (r*x) >> wp
        x -= one
        p = MPZ_ZERO
        for c in coeffs:
            p = c + ((x*p)>>wp)
        p >>= (cwp-wp)
        if type == 0:
            return from_man_exp((r<<wp)//p, -wp, prec, rnd)
        if type == 2:
            return mpf_shift(from_rational(p, (r<<wp), prec, rnd), wp)
        if type == 3:
            return mpf_log(mpf_abs(from_man_exp((r<<wp)//p, -wp)), prec, rnd)
    else:
        r = one
        for i in xrange(-nearest_int):
            r = (r*x) >> wp
            x += one
        p = MPZ_ZERO
        for c in coeffs:
            p = c + ((x*p)>>wp)
        p >>= (cwp-wp)
        if wp - bitcount(abs(x)) > 10:
            # pass very close to 0, so do floating-point multiply
            g = mpf_add(xmpf, from_int(-nearest_int))  # exact
            r = from_man_exp(p*r,-wp-wp)
            r = mpf_mul(r, g, wp)
            if type == 0:
                return mpf_div(fone, r, prec, rnd)
            if type == 2:
                return mpf_pos(r, prec, rnd)
            if type == 3:
                return mpf_log(mpf_abs(mpf_div(fone, r, wp)), prec, rnd)
        else:
            r = from_man_exp(x*p*r,-3*wp)
            if type == 0: return mpf_div(fone, r, prec, rnd)
            if type == 2: return mpf_pos(r, prec, rnd)
            if type == 3: return mpf_neg(mpf_log(mpf_abs(r), prec, rnd))

def stirling_coefficient(n):
    if n in gamma_stirling_cache:
        return gamma_stirling_cache[n]
    p, q = bernfrac(n)
    q *= MPZ(n*(n-1))
    gamma_stirling_cache[n] = p, q, bitcount(abs(p)), bitcount(q)
    return gamma_stirling_cache[n]

def real_stirling_series(x, prec):
    """
    Sums the rational part of Stirling's expansion,

    log(sqrt(2*pi)) - z + 1/(12*z) - 1/(360*z^3) + ...

    """
    t = (MPZ_ONE<<(prec+prec)) // x   # t = 1/x
    u = (t*t)>>prec                  # u = 1/x**2
    s = ln_sqrt2pi_fixed(prec) - x
    # Add initial terms of Stirling's series
    s += t//12;            t = (t*u)>>prec
    s -= t//360;           t = (t*u)>>prec
    s += t//1260;          t = (t*u)>>prec
    s -= t//1680;          t = (t*u)>>prec
    if not t: return s
    s += t//1188;          t = (t*u)>>prec
    s -= 691*t//360360;    t = (t*u)>>prec
    s += t//156;           t = (t*u)>>prec
    if not t: return s
    s -= 3617*t//122400;   t = (t*u)>>prec
    s += 43867*t//244188;  t = (t*u)>>prec
    s -= 174611*t//125400;  t = (t*u)>>prec
    if not t: return s
    k = 22
    # From here on, the coefficients are growing, so we
    # have to keep t at a roughly constant size
    usize = bitcount(abs(u))
    tsize = bitcount(abs(t))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            w = t >> m
            shift -= m
        else:
            w = t
        term = (t*p//q) >> shift
        if not term:
            break
        s += term
        t = (t*u) >> usize
        texp -= (prec - usize)
        k += 2
    return s

def complex_stirling_series(x, y, prec):
    # t = 1/z
    _m = (x*x + y*y) >> prec
    tre = (x << prec) // _m
    tim = (-y << prec) // _m
    # u = 1/z**2
    ure = (tre*tre - tim*tim) >> prec
    uim = tim*tre >> (prec-1)
    # s = log(sqrt(2*pi)) - z
    sre = ln_sqrt2pi_fixed(prec) - x
    sim = -y

    # Add initial terms of Stirling's series
    sre += tre//12; sim += tim//12;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= tre//360; sim -= tim//360;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += tre//1260; sim += tim//1260;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= tre//1680; sim -= tim//1680;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim
    sre += tre//1188; sim += tim//1188;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= 691*tre//360360; sim -= 691*tim//360360;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += tre//156; sim += tim//156;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim
    sre -= 3617*tre//122400; sim -= 3617*tim//122400;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += 43867*tre//244188; sim += 43867*tim//244188;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= 174611*tre//125400; sim -= 174611*tim//125400;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim

    k = 22
    # From here on, the coefficients are growing, so we
    # have to keep t at a roughly constant size
    usize = bitcount(max(abs(ure), abs(uim)))
    tsize = bitcount(max(abs(tre), abs(tim)))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            wre = tre >> m
            wim = tim >> m
            shift -= m
        else:
            wre = tre
            wim = tim
        termre = (tre*p//q) >> shift
        termim = (tim*p//q) >> shift
        if abs(termre) + abs(termim) < 5:
            break
        sre += termre
        sim += termim
        tre, tim = ((tre*ure - tim*uim)>>usize), \
            ((tre*uim + tim*ure)>>usize)
        texp -= (prec - usize)
        k += 2
    return sre, sim


def mpf_gamma(x, prec, rnd='d', type=0):
    """
    This function implements multipurpose evaluation of the gamma
    function, G(x), as well as the following versions of the same:

    type = 0 -- G(x)                    [standard gamma function]
    type = 1 -- G(x+1) = x*G(x+1) = x!  [factorial]
    type = 2 -- 1/G(x)                  [reciprocal gamma function]
    type = 3 -- log(|G(x)|)             [log-gamma function, real part]
    """

    # Specal values
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            if type == 1: return fone
            if type == 2: return fzero
            raise ValueError("gamma function pole")
        if x == finf:
            if type == 2: return fzero
            return finf
        return fnan

    # First of all, for log gamma, numbers can be well beyond the fixed-point
    # range, so we must take care of huge numbers before e.g. trying
    # to convert x to the nearest integer
    if type == 3:
        wp = prec+20
        if exp+bc > wp and not sign:
            return mpf_sub(mpf_mul(x, mpf_log(x, wp), wp), x, prec, rnd)

    # We strongly want to special-case small integers
    is_integer = exp >= 0
    if is_integer:
        # Poles
        if sign:
            if type == 2:
                return fzero
            raise ValueError("gamma function pole")
        # n = x
        n = man << exp
        if n < SMALL_FACTORIAL_CACHE_SIZE:
            if type == 0:
                return mpf_pos(small_factorial_cache[n-1], prec, rnd)
            if type == 1:
                return mpf_pos(small_factorial_cache[n], prec, rnd)
            if type == 2:
                return mpf_div(fone, small_factorial_cache[n-1], prec, rnd)
            if type == 3:
                return mpf_log(small_factorial_cache[n-1], prec, rnd)
    else:
        # floor(abs(x))
        n = int(man >> (-exp))

    # Estimate size and precision
    # Estimate log(gamma(|x|),2) as x*log(x,2)
    mag = exp + bc
    gamma_size = n*mag

    if type == 3:
        wp = prec + 20
    else:
        wp = prec + bitcount(gamma_size) + 20

    # Very close to 0, pole
    if mag < -wp:
        if type == 0:
            return mpf_sub(mpf_div(fone,x, wp),mpf_shift(fone,-wp),prec,rnd)
        if type == 1: return mpf_sub(fone, x, prec, rnd)
        if type == 2: return mpf_add(x, mpf_shift(fone,mag-wp), prec, rnd)
        if type == 3: return mpf_neg(mpf_log(mpf_abs(x), prec, rnd))

    # From now on, we assume having a gamma function
    if type == 1:
        return mpf_gamma(mpf_add(x, fone), prec, rnd, 0)

    # Special case integers (those not small enough to be caught above,
    # but still small enough for an exact factorial to be faster
    # than an approximate algorithm), and half-integers
    if exp >= -1:
        if is_integer:
            if gamma_size < 10*wp:
                if type == 0:
                    return from_int(ifac(n-1), prec, rnd)
                if type == 2:
                    return from_rational(MPZ_ONE, ifac(n-1), prec, rnd)
                if type == 3:
                    return mpf_log(from_int(ifac(n-1)), prec, rnd)
        # half-integer
        if n < 100 or gamma_size < 10*wp:
            if sign:
                w = sqrtpi_fixed(wp)
                if n % 2: f = ifac2(2*n+1)
                else:     f = -ifac2(2*n+1)
                if type == 0:
                    return mpf_shift(from_rational(w, f, prec, rnd), -wp+n+1)
                if type == 2:
                    return mpf_shift(from_rational(f, w, prec, rnd), wp-n-1)
                if type == 3:
                    return mpf_log(mpf_shift(from_rational(w, abs(f),
                        prec, rnd), -wp+n+1), prec, rnd)
            elif n == 0:
                if type == 0: return mpf_sqrtpi(prec, rnd)
                if type == 2: return mpf_div(fone, mpf_sqrtpi(wp), prec, rnd)
                if type == 3: return mpf_log(mpf_sqrtpi(wp), prec, rnd)
            else:
                w = sqrtpi_fixed(wp)
                w = from_man_exp(w * ifac2(2*n-1), -wp-n)
                if type == 0: return mpf_pos(w, prec, rnd)
                if type == 2: return mpf_div(fone, w, prec, rnd)
                if type == 3: return mpf_log(mpf_abs(w), prec, rnd)

    # Convert to fixed point
    offset = exp + wp
    if offset >= 0: absxman = man << offset
    else:           absxman = man >> (-offset)

    # For log gamma, provide accurate evaluation for x = 1+eps and 2+eps
    if type == 3 and not sign:
        one = MPZ_ONE << wp
        one_dist = abs(absxman-one)
        two_dist = abs(absxman-2*one)
        cancellation = (wp - bitcount(min(one_dist, two_dist)))
        if cancellation > 10:
            xsub1 = mpf_sub(fone, x)
            xsub2 = mpf_sub(ftwo, x)
            xsub1mag = xsub1[2]+xsub1[3]
            xsub2mag = xsub2[2]+xsub2[3]
            if xsub1mag < -wp:
                return mpf_mul(mpf_euler(wp), mpf_sub(fone, x), prec, rnd)
            if xsub2mag < -wp:
                return mpf_mul(mpf_sub(fone, mpf_euler(wp)),
                    mpf_sub(x, ftwo), prec, rnd)
            # Proceed but increase precision
            wp += max(-xsub1mag, -xsub2mag)
            offset = exp + wp
            if offset >= 0: absxman = man << offset
            else:           absxman = man >> (-offset)

    # Use Taylor series if appropriate
    n_for_stirling = int(GAMMA_STIRLING_BETA*wp)
    if n < max(100, n_for_stirling) and wp < MAX_GAMMA_TAYLOR_PREC:
        if sign:
            absxman = -absxman
        return gamma_fixed_taylor(x, absxman, wp, prec, rnd, type)

    # Use Stirling's series
    # First ensure that |x| is large enough for rapid convergence
    xorig = x

    # Argument reduction
    r = 0
    if n < n_for_stirling:
        r = one = MPZ_ONE << wp
        d = n_for_stirling - n
        for k in xrange(d):
            r = (r * absxman) >> wp
            absxman += one
        x = xabs = from_man_exp(absxman, -wp)
        if sign:
            x = mpf_neg(x)
    else:
        xabs = mpf_abs(x)

    # Asymptotic series
    y = real_stirling_series(absxman, wp)
    u = to_fixed(mpf_log(xabs, wp), wp)
    u = ((absxman - (MPZ_ONE<<(wp-1))) * u) >> wp
    y += u
    w = from_man_exp(y, -wp)

    # Compute final value
    if sign:
        # Reflection formula
        A = mpf_mul(mpf_sin_pi(xorig, wp), xorig, wp)
        B = mpf_neg(mpf_pi(wp))
        if type == 0 or type == 2:
            A = mpf_mul(A, mpf_exp(w, wp))
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            if type == 0:
                return mpf_div(B, A, prec, rnd)
            if type == 2:
                return mpf_div(A, B, prec, rnd)
        if type == 3:
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            A = mpf_add(mpf_log(mpf_abs(A), wp), w, wp)
            return mpf_sub(mpf_log(mpf_abs(B), wp), A, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpf_div(mpf_exp(w, wp),
                    from_man_exp(r, -wp), prec, rnd)
            return mpf_exp(w, prec, rnd)
        if type == 2:
            if r:
                return mpf_div(from_man_exp(r, -wp),
                    mpf_exp(w, wp), prec, rnd)
            return mpf_exp(mpf_neg(w), prec, rnd)
        if type == 3:
            if r:
                return mpf_sub(w, mpf_log(from_man_exp(r,-wp), wp), prec, rnd)
            return mpf_pos(w, prec, rnd)


def mpc_gamma(z, prec, rnd='d', type=0):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b

    if b == fzero:
        # Imaginary part on negative half-axis for log-gamma function
        if type == 3 and asign:
            re = mpf_gamma(a, prec, rnd, 3)
            n = (-aman) >> (-aexp)
            im = mpf_mul_int(mpf_pi(prec+10), n, prec, rnd)
            return re, im
        return mpf_gamma(a, prec, rnd, type), fzero

    # Some kind of complex inf/nan
    if (not aman and aexp) or (not bman and bexp):
        return (fnan, fnan)

    # Initial working precision
    wp = prec + 20

    amag = aexp+abc
    bmag = bexp+bbc
    if aman:
        mag = max(amag, bmag)
    else:
        mag = bmag

    # Close to 0
    if mag < -8:
        if mag < -wp:
            # 1/gamma(z) = z + euler*z^2 + O(z^3)
            v = mpc_add(z, mpc_mul_mpf(mpc_mul(z,z,wp),mpf_euler(wp),wp), wp)
            if type == 0: return mpc_reciprocal(v, prec, rnd)
            if type == 1: return mpc_div(z, v, prec, rnd)
            if type == 2: return mpc_pos(v, prec, rnd)
            if type == 3: return mpc_log(mpc_reciprocal(v, prec), prec, rnd)
        elif type != 1:
            wp += (-mag)

    # Handle huge log-gamma values; must do this before converting to
    # a fixed-point value. TODO: determine a precise cutoff of validity
    # depending on amag and bmag
    if type == 3 and mag > wp and ((not asign) or (bmag >= amag)):
        return mpc_sub(mpc_mul(z, mpc_log(z, wp), wp), z, prec, rnd)

    # From now on, we assume having a gamma function
    if type == 1:
        return mpc_gamma((mpf_add(a, fone), b), prec, rnd, 0)

    an = abs(to_int(a))
    bn = abs(to_int(b))
    absn = max(an, bn)
    gamma_size = absn*mag
    if type == 3:
        pass
    else:
        wp += bitcount(gamma_size)

    # Reflect to the right half-plane. Note that Stirling's expansion
    # is valid in the left half-plane too, as long as we're not too close
    # to the real axis, but in order to use this argument reduction
    # in the negative direction must be implemented.
    #need_reflection = asign and ((bmag < 0) or (amag-bmag > 4))
    need_reflection = asign
    zorig = z
    if need_reflection:
        z = mpc_neg(z)
        asign, aman, aexp, abc = a = z[0]
        bsign, bman, bexp, bbc = b = z[1]

    # Imaginary part very small compared to real one?
    yfinal = 0
    balance_prec = 0
    if bmag < -10:
        # Check z ~= 1 and z ~= 2 for loggamma
        if type == 3:
            zsub1 = mpc_sub_mpf(z, fone)
            if zsub1[0] == fzero:
                cancel1 = -bmag
            else:
                cancel1 = -max(zsub1[0][2]+zsub1[0][3], bmag)
            if cancel1 > wp:
                pi = mpf_pi(wp)
                x = mpc_mul_mpf(zsub1, pi, wp)
                x = mpc_mul(x, x, wp)
                x = mpc_div_mpf(x, from_int(12), wp)
                y = mpc_mul_mpf(zsub1, mpf_neg(mpf_euler(wp)), wp)
                yfinal = mpc_add(x, y, wp)
                if not need_reflection:
                    return mpc_pos(yfinal, prec, rnd)
            elif cancel1 > 0:
                wp += cancel1
            zsub2 = mpc_sub_mpf(z, ftwo)
            if zsub2[0] == fzero:
                cancel2 = -bmag
            else:
                cancel2 = -max(zsub2[0][2]+zsub2[0][3], bmag)
            if cancel2 > wp:
                pi = mpf_pi(wp)
                t = mpf_sub(mpf_mul(pi, pi), from_int(6))
                x = mpc_mul_mpf(mpc_mul(zsub2, zsub2, wp), t, wp)
                x = mpc_div_mpf(x, from_int(12), wp)
                y = mpc_mul_mpf(zsub2, mpf_sub(fone, mpf_euler(wp)), wp)
                yfinal = mpc_add(x, y, wp)
                if not need_reflection:
                    return mpc_pos(yfinal, prec, rnd)
            elif cancel2 > 0:
                wp += cancel2
        if bmag < -wp:
            # Compute directly from the real gamma function.
            pp = 2*(wp+10)
            aabs = mpf_abs(a)
            eps = mpf_shift(fone, amag-wp)
            x1 = mpf_gamma(aabs, pp, type=type)
            x2 = mpf_gamma(mpf_add(aabs, eps), pp, type=type)
            xprime = mpf_div(mpf_sub(x2, x1, pp), eps, pp)
            y = mpf_mul(b, xprime, prec, rnd)
            yfinal = (x1, y)
            # Note: we still need to use the reflection formula for
            # near-poles, and the correct branch of the log-gamma function
            if not need_reflection:
                return mpc_pos(yfinal, prec, rnd)
        else:
            balance_prec += (-bmag)

    wp += balance_prec
    n_for_stirling = int(GAMMA_STIRLING_BETA*wp)
    need_reduction = absn < n_for_stirling

    afix = to_fixed(a, wp)
    bfix = to_fixed(b, wp)

    r = 0
    if not yfinal:
        zprered = z
        # Argument reduction
        if absn < n_for_stirling:
            absn = complex(an, bn)
            d = int((1 + n_for_stirling**2 - bn**2)**0.5 - an)
            rre = one = MPZ_ONE << wp
            rim = MPZ_ZERO
            for k in xrange(d):
                rre, rim = ((afix*rre-bfix*rim)>>wp), ((afix*rim + bfix*rre)>>wp)
                afix += one
            r = from_man_exp(rre, -wp), from_man_exp(rim, -wp)
            a = from_man_exp(afix, -wp)
            z = a, b

        yre, yim = complex_stirling_series(afix, bfix, wp)
        # (z-1/2)*log(z) + S
        lre, lim = mpc_log(z, wp)
        lre = to_fixed(lre, wp)
        lim = to_fixed(lim, wp)
        yre = ((lre*afix - lim*bfix)>>wp) - (lre>>1) + yre
        yim = ((lre*bfix + lim*afix)>>wp) - (lim>>1) + yim
        y = from_man_exp(yre, -wp), from_man_exp(yim, -wp)

        if r and type == 3:
            # If re(z) > 0 and abs(z) <= 4, the branches of loggamma(z)
            # and log(gamma(z)) coincide. Otherwise, use the zeroth order
            # Stirling expansion to compute the correct imaginary part.
            y = mpc_sub(y, mpc_log(r, wp), wp)
            zfa = to_float(zprered[0])
            zfb = to_float(zprered[1])
            zfabs = math.hypot(zfa,zfb)
            #if not (zfa > 0.0 and zfabs <= 4):
            yfb = to_float(y[1])
            u = math.atan2(zfb, zfa)
            if zfabs <= 0.5:
                gi = 0.577216*zfb - u
            else:
                gi = -zfb - 0.5*u + zfa*u + zfb*math.log(zfabs)
            n = int(math.floor((gi-yfb)/(2*math.pi)+0.5))
            y = (y[0], mpf_add(y[1], mpf_mul_int(mpf_pi(wp), 2*n, wp), wp))

    if need_reflection:
        if type == 0 or type == 2:
            A = mpc_mul(mpc_sin_pi(zorig, wp), zorig, wp)
            B = (mpf_neg(mpf_pi(wp)), fzero)
            if yfinal:
                if type == 2:
                    A = mpc_div(A, yfinal, wp)
                else:
                    A = mpc_mul(A, yfinal, wp)
            else:
                A = mpc_mul(A, mpc_exp(y, wp), wp)
            if r:
                B = mpc_mul(B, r, wp)
            if type == 0: return mpc_div(B, A, prec, rnd)
            if type == 2: return mpc_div(A, B, prec, rnd)

        # Reflection formula for the log-gamma function with correct branch
        # http://functions.wolfram.com/GammaBetaErf/LogGamma/16/01/01/0006/
        # LogGamma[z] == -LogGamma[-z] - Log[-z] +
        # Sign[Im[z]] Floor[Re[z]] Pi I + Log[Pi] -
        #      Log[Sin[Pi (z - Floor[Re[z]])]] -
        # Pi I (1 - Abs[Sign[Im[z]]]) Abs[Floor[Re[z]]]
        if type == 3:
            if yfinal:
                s1 = mpc_neg(yfinal)
            else:
                s1 = mpc_neg(y)
            # s -= log(-z)
            s1 = mpc_sub(s1, mpc_log(mpc_neg(zorig), wp), wp)
            # floor(re(z))
            rezfloor = mpf_floor(zorig[0])
            imzsign = mpf_sign(zorig[1])
            pi = mpf_pi(wp)
            t = mpf_mul(pi, rezfloor)
            t = mpf_mul_int(t, imzsign, wp)
            s1 = (s1[0], mpf_add(s1[1], t, wp))
            s1 = mpc_add_mpf(s1, mpf_log(pi, wp), wp)
            t = mpc_sin_pi(mpc_sub_mpf(zorig, rezfloor), wp)
            t = mpc_log(t, wp)
            s1 = mpc_sub(s1, t, wp)
            # Note: may actually be unused, because we fall back
            # to the mpf_ function for real arguments
            if not imzsign:
                t = mpf_mul(pi, mpf_floor(rezfloor), wp)
                s1 = (s1[0], mpf_sub(s1[1], t, wp))
            return mpc_pos(s1, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpc_div(mpc_exp(y, wp), r, prec, rnd)
            return mpc_exp(y, prec, rnd)
        if type == 2:
            if r:
                return mpc_div(r, mpc_exp(y, wp), prec, rnd)
            return mpc_exp(mpc_neg(y), prec, rnd)
        if type == 3:
            return mpc_pos(y, prec, rnd)

def mpf_factorial(x, prec, rnd='d'):
    return mpf_gamma(x, prec, rnd, 1)

def mpc_factorial(x, prec, rnd='d'):
    return mpc_gamma(x, prec, rnd, 1)

def mpf_rgamma(x, prec, rnd='d'):
    return mpf_gamma(x, prec, rnd, 2)

def mpc_rgamma(x, prec, rnd='d'):
    return mpc_gamma(x, prec, rnd, 2)

def mpf_loggamma(x, prec, rnd='d'):
    sign, man, exp, bc = x
    if sign:
        raise ComplexResult
    return mpf_gamma(x, prec, rnd, 3)

def mpc_loggamma(z, prec, rnd='d'):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero and asign:
        re = mpf_gamma(a, prec, rnd, 3)
        n = (-aman) >> (-aexp)
        im = mpf_mul_int(mpf_pi(prec+10), n, prec, rnd)
        return re, im
    return mpc_gamma(z, prec, rnd, 3)

def mpf_gamma_int(n, prec, rnd=round_fast):
    if n < SMALL_FACTORIAL_CACHE_SIZE:
        return mpf_pos(small_factorial_cache[n-1], prec, rnd)
    return mpf_gamma(from_int(n), prec, rnd)
<|endoftext|>"""
Low-level functions for complex arithmetic.
"""

import sys

from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND

from .libmpf import (\
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast, bitcount,
    bctable, normalize, normalize1, reciprocal_rnd, rshift, lshift, giant_steps,
    negative_rnd,
    to_str, to_fixed, from_man_exp, from_float, to_float, from_int, to_int,
    fzero, fone, ftwo, fhalf, finf, fninf, fnan, fnone,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul,
    mpf_div, mpf_mul_int, mpf_shift, mpf_sqrt, mpf_hypot,
    mpf_rdiv_int, mpf_floor, mpf_ceil, mpf_nint, mpf_frac,
    mpf_sign, mpf_hash,
    ComplexResult
)

from .libelefun import (\
    mpf_pi, mpf_exp, mpf_log, mpf_cos_sin, mpf_cosh_sinh, mpf_tan, mpf_pow_int,
    mpf_log_hypot,
    mpf_cos_sin_pi, mpf_phi,
    mpf_cos, mpf_sin, mpf_cos_pi, mpf_sin_pi,
    mpf_atan, mpf_atan2, mpf_cosh, mpf_sinh, mpf_tanh,
    mpf_asin, mpf_acos, mpf_acosh, mpf_nthroot, mpf_fibonacci
)

# An mpc value is a (real, imag) tuple
mpc_one = fone, fzero
mpc_zero = fzero, fzero
mpc_two = ftwo, fzero
mpc_half = (fhalf, fzero)

_infs = (finf, fninf)
_infs_nan = (finf, fninf, fnan)

def mpc_is_inf(z):
    """Check if either real or imaginary part is infinite"""
    re, im = z
    if re in _infs: return True
    if im in _infs: return True
    return False

def mpc_is_infnan(z):
    """Check if either real or imaginary part is infinite or nan"""
    re, im = z
    if re in _infs_nan: return True
    if im in _infs_nan: return True
    return False

def mpc_to_str(z, dps, **kwargs):
    re, im = z
    rs = to_str(re, dps)
    if im[0]:
        return rs + " - " + to_str(mpf_neg(im), dps, **kwargs) + "j"
    else:
        return rs + " + " + to_str(im, dps, **kwargs) + "j"

def mpc_to_complex(z, strict=False, rnd=round_fast):
    re, im = z
    return complex(to_float(re, strict, rnd), to_float(im, strict, rnd))

def mpc_hash(z):
    if sys.version_info >= (3, 2):
        re, im = z
        h = mpf_hash(re) + sys.hash_info.imag * mpf_hash(im)
        # Need to reduce either module 2^32 or 2^64
        h = h % (2**sys.hash_info.width)
        return int(h)
    else:
        try:
            return hash(mpc_to_complex(z, strict=True))
        except OverflowError:
            return hash(z)

def mpc_conjugate(z, prec, rnd=round_fast):
    re, im = z
    return re, mpf_neg(im, prec, rnd)

def mpc_is_nonzero(z):
    return z != mpc_zero

def mpc_add(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    return mpf_add(a, c, prec, rnd), mpf_add(b, d, prec, rnd)

def mpc_add_mpf(z, x, prec, rnd=round_fast):
    a, b = z
    return mpf_add(a, x, prec, rnd), b

def mpc_sub(z, w, prec=0, rnd=round_fast):
    a, b = z
    c, d = w
    return mpf_sub(a, c, prec, rnd), mpf_sub(b, d, prec, rnd)

def mpc_sub_mpf(z, p, prec=0, rnd=round_fast):
    a, b = z
    return mpf_sub(a, p, prec, rnd), b

def mpc_pos(z, prec, rnd=round_fast):
    a, b = z
    return mpf_pos(a, prec, rnd), mpf_pos(b, prec, rnd)

def mpc_neg(z, prec=None, rnd=round_fast):
    a, b = z
    return mpf_neg(a, prec, rnd), mpf_neg(b, prec, rnd)

def mpc_shift(z, n):
    a, b = z
    return mpf_shift(a, n), mpf_shift(b, n)

def mpc_abs(z, prec, rnd=round_fast):
    """Absolute value of a complex number, |a+bi|.
    Returns an mpf value."""
    a, b = z
    return mpf_hypot(a, b, prec, rnd)

def mpc_arg(z, prec, rnd=round_fast):
    """Argument of a complex number. Returns an mpf value."""
    a, b = z
    return mpf_atan2(b, a, prec, rnd)

def mpc_floor(z, prec, rnd=round_fast):
    a, b = z
    return mpf_floor(a, prec, rnd), mpf_floor(b, prec, rnd)

def mpc_ceil(z, prec, rnd=round_fast):
    a, b = z
    return mpf_ceil(a, prec, rnd), mpf_ceil(b, prec, rnd)

def mpc_nint(z, prec, rnd=round_fast):
    a, b = z
    return mpf_nint(a, prec, rnd), mpf_nint(b, prec, rnd)

def mpc_frac(z, prec, rnd=round_fast):
    a, b = z
    return mpf_frac(a, prec, rnd), mpf_frac(b, prec, rnd)


def mpc_mul(z, w, prec, rnd=round_fast):
    """
    Complex multiplication.

    Returns the real and imaginary part of (a+bi)*(c+di), rounded to
    the specified precision. The rounding mode applies to the real and
    imaginary parts separately.
    """
    a, b = z
    c, d = w
    p = mpf_mul(a, c)
    q = mpf_mul(b, d)
    r = mpf_mul(a, d)
    s = mpf_mul(b, c)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_add(r, s, prec, rnd)
    return re, im

def mpc_square(z, prec, rnd=round_fast):
    # (a+b*I)**2 == a**2 - b**2 + 2*I*a*b
    a, b = z
    p = mpf_mul(a,a)
    q = mpf_mul(b,b)
    r = mpf_mul(a,b, prec, rnd)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_shift(r, 1)
    return re, im

def mpc_mul_mpf(z, p, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul(a, p, prec, rnd)
    im = mpf_mul(b, p, prec, rnd)
    return re, im

def mpc_mul_imag_mpf(z, x, prec, rnd=round_fast):
    """
    Multiply the mpc value z by I*x where x is an mpf value.
    """
    a, b = z
    re = mpf_neg(mpf_mul(b, x, prec, rnd))
    im = mpf_mul(a, x, prec, rnd)
    return re, im

def mpc_mul_int(z, n, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul_int(a, n, prec, rnd)
    im = mpf_mul_int(b, n, prec, rnd)
    return re, im

def mpc_div(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    wp = prec + 10
    # mag = c*c + d*d
    mag = mpf_add(mpf_mul(c, c), mpf_mul(d, d), wp)
    # (a*c+b*d)/mag, (b*c-a*d)/mag
    t = mpf_add(mpf_mul(a,c), mpf_mul(b,d), wp)
    u = mpf_sub(mpf_mul(b,c), mpf_mul(a,d), wp)
    return mpf_div(t,mag,prec,rnd), mpf_div(u,mag,prec,rnd)

def mpc_div_mpf(z, p, prec, rnd=round_fast):
    """Calculate z/p where p is real"""
    a, b = z
    re = mpf_div(a, p, prec, rnd)
    im = mpf_div(b, p, prec, rnd)
    return re, im

def mpc_reciprocal(z, prec, rnd=round_fast):
    """Calculate 1/z efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a,a),mpf_mul(b,b),prec+10)
    re = mpf_div(a, m, prec, rnd)
    im = mpf_neg(mpf_div(b, m, prec, rnd))
    return re, im

def mpc_mpf_div(p, z, prec, rnd=round_fast):
    """Calculate p/z where p is real efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a,a),mpf_mul(b,b), prec+10)
    re = mpf_div(mpf_mul(a,p), m, prec, rnd)
    im = mpf_div(mpf_neg(mpf_mul(b,p)), m, prec, rnd)
    return re, im

def complex_int_pow(a, b, n):
    """Complex integer power: computes (a+b*I)**n exactly for
    nonnegative n (a and b must be Python ints)."""
    wre = 1
    wim = 0
    while n:
        if n & 1:
            wre, wim = wre*a - wim*b, wim*a + wre*b
            n -= 1
        a, b = a*a - b*b, 2*a*b
        n //= 2
    return wre, wim

def mpc_pow(z, w, prec, rnd=round_fast):
    if w[1] == fzero:
        return mpc_pow_mpf(z, w[0], prec, rnd)
    return mpc_exp(mpc_mul(mpc_log(z, prec+10), w, prec+10), prec, rnd)

def mpc_pow_mpf(z, p, prec, rnd=round_fast):
    psign, pman, pexp, pbc = p
    if pexp >= 0:
        return mpc_pow_int(z, (-1)**psign * (pman<<pexp), prec, rnd)
    if pexp == -1:
        sqrtz = mpc_sqrt(z, prec+10)
        return mpc_pow_int(sqrtz, (-1)**psign * pman, prec, rnd)
    return mpc_exp(mpc_mul_mpf(mpc_log(z, prec+10), p, prec+10), prec, rnd)

def mpc_pow_int(z, n, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_pow_int(a, n, prec, rnd), fzero
    if a == fzero:
        v = mpf_pow_int(b, n, prec, rnd)
        n %= 4
        if n == 0:
            return v, fzero
        elif n == 1:
            return fzero, v
        elif n == 2:
            return mpf_neg(v), fzero
        elif n == 3:
            return fzero, mpf_neg(v)
    if n == 0: return mpc_one
    if n == 1: return mpc_pos(z, prec, rnd)
    if n == 2: return mpc_square(z, prec, rnd)
    if n == -1: return mpc_reciprocal(z, prec, rnd)
    if n < 0: return mpc_reciprocal(mpc_pow_int(z, -n, prec+4), prec, rnd)
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign: aman = -aman
    if bsign: bman = -bman
    de = aexp - bexp
    abs_de = abs(de)
    exact_size = n*(abs_de + max(abc, bbc))
    if exact_size < 10000:
        if de > 0:
            aman <<= de
            aexp = bexp
        else:
            bman <<= (-de)
            bexp = aexp
        re, im = complex_int_pow(aman, bman, n)
        re = from_man_exp(re, int(n*aexp), prec, rnd)
        im = from_man_exp(im, int(n*bexp), prec, rnd)
        return re, im
    return mpc_exp(mpc_mul_int(mpc_log(z, prec+10), n, prec+10), prec, rnd)

def mpc_sqrt(z, prec, rnd=round_fast):
    """Complex square root (principal branch).

    We have sqrt(a+bi) = sqrt((r+a)/2) + b/sqrt(2*(r+a))*i where
    r = abs(a+bi), when a+bi is not a negative real number."""
    a, b = z
    if b == fzero:
        if a == fzero:
            return (a, b)
        # When a+bi is a negative real number, we get a real sqrt times i
        if a[0]:
            im = mpf_sqrt(mpf_neg(a), prec, rnd)
            return (fzero, im)
        else:
            re = mpf_sqrt(a, prec, rnd)
            return (re, fzero)
    wp = prec+20
    if not a[0]:                               # case a positive
        t  = mpf_add(mpc_abs((a, b), wp), a, wp)  # t = abs(a+bi) + a
        u = mpf_shift(t, -1)                      # u = t/2
        re = mpf_sqrt(u, prec, rnd)               # re = sqrt(u)
        v = mpf_shift(t, 1)                       # v = 2*t
        w  = mpf_sqrt(v, wp)                      # w = sqrt(v)
        im = mpf_div(b, w, prec, rnd)             # im = b / w
    else:                                      # case a negative
        t = mpf_sub(mpc_abs((a, b), wp), a, wp)   # t = abs(a+bi) - a
        u = mpf_shift(t, -1)                      # u = t/2
        im = mpf_sqrt(u, prec, rnd)               # im = sqrt(u)
        v = mpf_shift(t, 1)                       # v = 2*t
        w  = mpf_sqrt(v, wp)                      # w = sqrt(v)
        re = mpf_div(b, w, prec, rnd)             # re = b/w
        if b[0]:
            re = mpf_neg(re)
            im = mpf_neg(im)
    return re, im

def mpc_nthroot_fixed(a, b, n, prec):
    # a, b signed integers at fixed precision prec
    start = 50
    a1 = int(rshift(a, prec - n*start))
    b1 = int(rshift(b, prec - n*start))
    try:
        r = (a1 + 1j * b1)**(1.0/n)
        re = r.real
        im = r.imag
        re = MPZ(int(re))
        im = MPZ(int(im))
    except OverflowError:
        a1 = from_int(a1, start)
        b1 = from_int(b1, start)
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, start)
        re, im = mpc_pow((a1, b1), (nth, fzero), start)
        re = to_int(re)
        im = to_int(im)
    extra = 10
    prevp = start
    extra1 = n
    for p in giant_steps(start, prec+extra):
        # this is slow for large n, unlike int_pow_fixed
        re2, im2 = complex_int_pow(re, im, n-1)
        re2 = rshift(re2, (n-1)*prevp - p - extra1)
        im2 = rshift(im2, (n-1)*prevp - p - extra1)
        r4 = (re2*re2 + im2*im2) >> (p + extra1)
        ap = rshift(a, prec - p)
        bp = rshift(b, prec - p)
        rec = (ap * re2 + bp * im2) >> p
        imc = (-ap * im2 + bp * re2) >> p
        reb = (rec << p) // r4
        imb = (imc << p) // r4
        re = (reb + (n-1)*lshift(re, p-prevp))//n
        im = (imb + (n-1)*lshift(im, p-prevp))//n
        prevp = p
    return re, im

def mpc_nthroot(z, n, prec, rnd=round_fast):
    """
    Complex n-th root.

    Use Newton method as in the real case when it is faster,
    otherwise use z**(1/n)
    """
    a, b = z
    if a[0] == 0 and b == fzero:
        re = mpf_nthroot(a, n, prec, rnd)
        return (re, fzero)
    if n < 2:
        if n == 0:
            return mpc_one
        if n == 1:
            return mpc_pos((a, b), prec, rnd)
        if n == -1:
            return mpc_div(mpc_one, (a, b), prec, rnd)
        inverse = mpc_nthroot((a, b), -n, prec+5, reciprocal_rnd[rnd])
        return mpc_div(mpc_one, inverse, prec, rnd)
    if n <= 20:
        prec2 = int(1.2 * (prec + 10))
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        pf = mpc_abs((a,b), prec)
        if pf[-2] + pf[-1] > -10  and pf[-2] + pf[-1] < prec:
            af = to_fixed(a, prec2)
            bf = to_fixed(b, prec2)
            re, im = mpc_nthroot_fixed(af, bf, n, prec2)
            extra = 10
            re = from_man_exp(re, -prec2-extra, prec2, rnd)
            im = from_man_exp(im, -prec2-extra, prec2, rnd)
            return re, im
    fn = from_int(n)
    prec2 = prec+10 + 10
    nth = mpf_rdiv_int(1, fn, prec2)
    re, im = mpc_pow((a, b), (nth, fzero), prec2, rnd)
    re = normalize(re[0], re[1], re[2], re[3], prec, rnd)
    im = normalize(im[0], im[1], im[2], im[3], prec, rnd)
    return re, im

def mpc_cbrt(z, prec, rnd=round_fast):
    """
    Complex cubic root.
    """
    return mpc_nthroot(z, 3, prec, rnd)

def mpc_exp(z, prec, rnd=round_fast):
    """
    Complex exponential function.

    We use the direct formula exp(a+bi) = exp(a) * (cos(b) + sin(b)*i)
    for the computation. This formula is very nice because it is
    pefectly stable; since we just do real multiplications, the only
    numerical errors that can creep in are single-ulp rounding errors.

    The formula is efficient since mpmath's real exp is quite fast and
    since we can compute cos and sin simultaneously.

    It is no problem if a and b are large; if the implementations of
    exp/cos/sin are accurate and efficient for all real numbers, then
    so is this function for all complex numbers.
    """
    a, b = z
    if a == fzero:
        return mpf_cos_sin(b, prec, rnd)
    if b == fzero:
        return mpf_exp(a, prec, rnd), fzero
    mag = mpf_exp(a, prec+4, rnd)
    c, s = mpf_cos_sin(b, prec+4, rnd)
    re = mpf_mul(mag, c, prec, rnd)
    im = mpf_mul(mag, s, prec, rnd)
    return re, im

def mpc_log(z, prec, rnd=round_fast):
    re = mpf_log_hypot(z[0], z[1], prec, rnd)
    im = mpc_arg(z, prec, rnd)
    return re, im

def mpc_cos(z, prec, rnd=round_fast):
    """Complex cosine. The formula used is cos(a+bi) = cos(a)*cosh(b) -
    sin(a)*sinh(b)*i.

    The same comments apply as for the complex exp: only real
    multiplications are pewrormed, so no cancellation errors are
    possible. The formula is also efficient since we can compute both
    pairs (cos, sin) and (cosh, sinh) in single stwps."""
    a, b = z
    if b == fzero:
        return mpf_cos(a, prec, rnd), fzero
    if a == fzero:
        return mpf_cosh(b, prec, rnd), fzero
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return re, mpf_neg(im)

def mpc_sin(z, prec, rnd=round_fast):
    """Complex sine. We have sin(a+bi) = sin(a)*cosh(b) +
    cos(a)*sinh(b)*i. See the docstring for mpc_cos for additional
    comments."""
    a, b = z
    if b == fzero:
        return mpf_sin(a, prec, rnd), fzero
    if a == fzero:
        return fzero, mpf_sinh(b, prec, rnd)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return re, im

def mpc_tan(z, prec, rnd=round_fast):
    """Complex tangent. Computed as tan(a+bi) = sin(2a)/M + sinh(2b)/M*i
    where M = cos(2a) + cosh(2b)."""
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero: return mpf_tan(a, prec, rnd), fzero
    if a == fzero: return fzero, mpf_tanh(b, prec, rnd)
    wp = prec + 15
    a = mpf_shift(a, 1)
    b = mpf_shift(b, 1)
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    # TODO: handle cancellation when c ~=  -1 and ch ~= 1
    mag = mpf_add(c, ch, wp)
    re = mpf_div(s, mag, prec, rnd)
    im = mpf_div(sh, mag, prec, rnd)
    return re, im

def mpc_cos_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_cos_pi(a, prec, rnd), fzero
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        return mpf_cosh(b, prec, rnd), fzero
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return re, mpf_neg(im)

def mpc_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_sin_pi(a, prec, rnd), fzero
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        return fzero, mpf_sinh(b, prec, rnd)
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return re, im

def mpc_cos_sin(z, prec, rnd=round_fast):
    a, b = z
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return (ch, fzero), (fzero, sh)
    if b == fzero:
        c, s = mpf_cos_sin(a, prec, rnd)
        return (c, fzero), (s, fzero)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return (cre, mpf_neg(cim)), (sre, sim)

def mpc_cos_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        c, s = mpf_cos_sin_pi(a, prec, rnd)
        return (c, fzero), (s, fzero)
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return (ch, fzero), (fzero, sh)
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return (cre, mpf_neg(cim)), (sre, sim)

def mpc_cosh(z, prec, rnd=round_fast):
    """Complex hyperbolic cosine. Computed as cosh(z) = cos(z*i)."""
    a, b = z
    return mpc_cos((b, mpf_neg(a)), prec, rnd)

def mpc_sinh(z, prec, rnd=round_fast):
    """Complex hyperbolic sine. Computed as sinh(z) = -i*sin(z*i)."""
    a, b = z
    b, a = mpc_sin((b, a), prec, rnd)
    return a, b

def mpc_tanh(z, prec, rnd=round_fast):
    """Complex hyperbolic tangent. Computed as tanh(z) = -i*tan(z*i)."""
    a, b = z
    b, a = mpc_tan((b, a), prec, rnd)
    return a, b

# TODO: avoid loss of accuracy
def mpc_atan(z, prec, rnd=round_fast):
    a, b = z
    # atan(z) = (I/2)*(log(1-I*z) - log(1+I*z))
    # x = 1-I*z = 1 + b - I*a
    # y = 1+I*z = 1 - b + I*a
    wp = prec + 15
    x = mpf_add(fone, b, wp), mpf_neg(a)
    y = mpf_sub(fone, b, wp), a
    l1 = mpc_log(x, wp)
    l2 = mpc_log(y, wp)
    a, b = mpc_sub(l1, l2, prec, rnd)
    # (I/2) * (a+b*I) = (-b/2 + a/2*I)
    v = mpf_neg(mpf_shift(b,-1)), mpf_shift(a,-1)
    # Subtraction at infinity gives correct real part but
    # wrong imaginary part (should be zero)
    if v[1] == fnan and mpc_is_inf(z):
        v = (v[0], fzero)
    return v

beta_crossover = from_float(0.6417)
alpha_crossover = from_float(1.5)

def acos_asin(z, prec, rnd, n):
    """ complex acos for n = 0, asin for n = 1
    The algorithm is described in
    T.E. Hull, T.F. Fairgrieve and P.T.P. Tang
    'Implementing the Complex Arcsine and Arcosine Functions
    using Exception Handling',
    ACM Trans. on Math. Software Vol. 23 (1997), p299
    The complex acos and asin can be defined as
    acos(z) = acos(beta) - I*sign(a)* log(alpha + sqrt(alpha**2 -1))
    asin(z) = asin(beta) + I*sign(a)* log(alpha + sqrt(alpha**2 -1))
    where z = a + I*b
    alpha = (1/2)*(r + s); beta = (1/2)*(r - s) = a/alpha
    r = sqrt((a+1)**2 + y**2); s = sqrt((a-1)**2 + y**2)
    These expressions are rewritten in different ways in different
    regions, delimited by two crossovers alpha_crossover and beta_crossover,
    and by abs(a) <= 1, in order to improve the numerical accuracy.
    """
    a, b = z
    wp = prec + 10
    # special cases with real argument
    if b == fzero:
        am = mpf_sub(fone, mpf_abs(a), wp)
        # case abs(a) <= 1
        if not am[0]:
            if n == 0:
                return mpf_acos(a, prec, rnd), fzero
            else:
                return mpf_asin(a, prec, rnd), fzero
        # cases abs(a) > 1
        else:
            # case a < -1
            if a[0]:
                pi = mpf_pi(prec, rnd)
                c = mpf_acosh(mpf_neg(a), prec, rnd)
                if n == 0:
                    return pi, mpf_neg(c)
                else:
                    return mpf_neg(mpf_shift(pi, -1)), c
            # case a > 1
            else:
                c = mpf_acosh(a, prec, rnd)
                if n == 0:
                    return fzero, c
                else:
                    pi = mpf_pi(prec, rnd)
                    return mpf_shift(pi, -1), mpf_neg(c)
    asign = bsign = 0
    if a[0]:
        a = mpf_neg(a)
        asign = 1
    if b[0]:
        b = mpf_neg(b)
        bsign = 1
    am = mpf_sub(fone, a, wp)
    ap = mpf_add(fone, a, wp)
    r = mpf_hypot(ap, b, wp)
    s = mpf_hypot(am, b, wp)
    alpha = mpf_shift(mpf_add(r, s, wp), -1)
    beta = mpf_div(a, alpha, wp)
    b2 = mpf_mul(b,b, wp)
    # case beta <= beta_crossover
    if not mpf_sub(beta_crossover, beta, wp)[0]:
        if n == 0:
            re = mpf_acos(beta, wp)
        else:
            re = mpf_asin(beta, wp)
    else:
        # to compute the real part in this region use the identity
        # asin(beta) = atan(beta/sqrt(1-beta**2))
        # beta/sqrt(1-beta**2) = (alpha + a) * (alpha - a)
        # alpha + a is numerically accurate; alpha - a can have
        # cancellations leading to numerical inaccuracies, so rewrite
        # it in differente ways according to the region
        Ax = mpf_add(alpha, a, wp)
        # case a <= 1
        if not am[0]:
            # c = b*b/(r + (a+1)); d = (s + (1-a))
            # alpha - a = (1/2)*(c + d)
            # case n=0: re = atan(sqrt((1/2) * Ax * (c + d))/a)
            # case n=1: re = atan(a/sqrt((1/2) * Ax * (c + d)))
            c = mpf_div(b2, mpf_add(r, ap, wp), wp)
            d = mpf_add(s, am, wp)
            re = mpf_shift(mpf_mul(Ax, mpf_add(c, d, wp), wp), -1)
            if n == 0:
                re = mpf_atan(mpf_div(mpf_sqrt(re, wp), a, wp), wp)
            else:
                re = mpf_atan(mpf_div(a, mpf_sqrt(re, wp), wp), wp)
        else:
            # c = Ax/(r + (a+1)); d = Ax/(s - (1-a))
            # alpha - a = (1/2)*(c + d)
            # case n = 0: re = atan(b*sqrt(c + d)/2/a)
            # case n = 1: re = atan(a/(b*sqrt(c + d)/2)
            c = mpf_div(Ax, mpf_add(r, ap, wp), wp)
            d = mpf_div(Ax, mpf_sub(s, am, wp), wp)
            re = mpf_shift(mpf_add(c, d, wp), -1)
            re = mpf_mul(b, mpf_sqrt(re, wp), wp)
            if n == 0:
                re = mpf_atan(mpf_div(re, a, wp), wp)
            else:
                re = mpf_atan(mpf_div(a, re, wp), wp)
    # to compute alpha + sqrt(alpha**2 - 1), if alpha <= alpha_crossover
    # replace it with 1 + Am1 + sqrt(Am1*(alpha+1)))
    # where Am1 = alpha -1
    # if alpha <= alpha_crossover:
    if not mpf_sub(alpha_crossover, alpha, wp)[0]:
        c1 = mpf_div(b2, mpf_add(r, ap, wp), wp)
        # case a < 1
        if mpf_neg(am)[0]:
            # Am1 = (1/2) * (b*b/(r + (a+1)) + b*b/(s + (1-a))
            c2 = mpf_add(s, am, wp)
            c2 = mpf_div(b2, c2, wp)
            Am1 = mpf_shift(mpf_add(c1, c2, wp), -1)
        else:
            # Am1 = (1/2) * (b*b/(r + (a+1)) + (s - (1-a)))
            c2 = mpf_sub(s, am, wp)
            Am1 = mpf_shift(mpf_add(c1, c2, wp), -1)
        # im = log(1 + Am1 + sqrt(Am1*(alpha+1)))
        im = mpf_mul(Am1, mpf_add(alpha, fone, wp), wp)
        im = mpf_log(mpf_add(fone, mpf_add(Am1, mpf_sqrt(im, wp), wp), wp), wp)
    else:
        # im = log(alpha + sqrt(alpha*alpha - 1))
        im = mpf_sqrt(mpf_sub(mpf_mul(alpha, alpha, wp), fone, wp), wp)
        im = mpf_log(mpf_add(alpha, im, wp), wp)
    if asign:
        if n == 0:
            re = mpf_sub(mpf_pi(wp), re, wp)
        else:
            re = mpf_neg(re)
    if not bsign and n == 0:
        im = mpf_neg(im)
    if bsign and n == 1:
        im = mpf_neg(im)
    re = normalize(re[0], re[1], re[2], re[3], prec, rnd)
    im = normalize(im[0], im[1], im[2], im[3], prec, rnd)
    return re, im

def mpc_acos(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 0)

def mpc_asin(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 1)

def mpc_asinh(z, prec, rnd=round_fast):
    # asinh(z) = I * asin(-I z)
    a, b = z
    a, b =  mpc_asin((b, mpf_neg(a)), prec, rnd)
    return mpf_neg(b), a

def mpc_acosh(z, prec, rnd=round_fast):
    # acosh(z) = -I * acos(z)   for Im(acos(z)) <= 0
    #            +I * acos(z)   otherwise
    a, b = mpc_acos(z, prec, rnd)
    if b[0] or b == fzero:
        return mpf_neg(b), a
    else:
        return b, mpf_neg(a)

def mpc_atanh(z, prec, rnd=round_fast):
    # atanh(z) = (log(1+z)-log(1-z))/2
    wp = prec + 15
    a = mpc_add(z, mpc_one, wp)
    b = mpc_sub(mpc_one, z, wp)
    a = mpc_log(a, wp)
    b = mpc_log(b, wp)
    v = mpc_shift(mpc_sub(a, b, wp), -1)
    # Subtraction at infinity gives correct imaginary part but
    # wrong real part (should be zero)
    if v[0] == fnan and mpc_is_inf(z):
        v = (fzero, v[1])
    return v

def mpc_fibonacci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        return (mpf_fibonacci(re, prec, rnd), fzero)
    size = max(abs(re[2]+re[3]), abs(re[2]+re[3]))
    wp = prec + size + 20
    a = mpf_phi(wp)
    b = mpf_add(mpf_shift(a, 1), fnone, wp)
    u = mpc_pow((a, fzero), z, wp)
    v = mpc_cos_pi(z, wp)
    v = mpc_div(v, u, wp)
    u = mpc_sub(u, v, wp)
    u = mpc_div_mpf(u, b, prec, rnd)
    return u

def mpf_expj(x, prec, rnd='f'):
    raise ComplexResult

def mpc_expj(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin(re, prec, rnd)
    if re == fzero:
        return mpf_exp(mpf_neg(im), prec, rnd), fzero
    ey = mpf_exp(mpf_neg(im), prec+10)
    c, s = mpf_cos_sin(re, prec+10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return re, im

def mpf_expjpi(x, prec, rnd='f'):
    raise ComplexResult

def mpc_expjpi(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin_pi(re, prec, rnd)
    sign, man, exp, bc = im
    wp = prec+10
    if man:
        wp += max(0, exp+bc)
    im = mpf_neg(mpf_mul(mpf_pi(wp), im, wp))
    if re == fzero:
        return mpf_exp(im, prec, rnd), fzero
    ey = mpf_exp(im, prec+10)
    c, s = mpf_cos_sin_pi(re, prec+10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return re, im


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as _lbmp
        mpc_exp = _lbmp.mpc_exp
        mpc_sqrt = _lbmp.mpc_sqrt
    except (ImportError, AttributeError):
        print("Warning: Sage imports in libmpc failed")
<|endoftext|>from .libmpf import (prec_to_dps, dps_to_prec, repr_dps,
  round_down, round_up, round_floor, round_ceiling, round_nearest,
  to_pickable, from_pickable, ComplexResult,
  fzero, fnzero, fone, fnone, ftwo, ften, fhalf, fnan, finf, fninf,
  math_float_inf, round_int, normalize, normalize1,
  from_man_exp, from_int, to_man_exp, to_int, mpf_ceil, mpf_floor,
  mpf_nint, mpf_frac,
  from_float, from_npfloat, from_Decimal, to_float, from_rational, to_rational, to_fixed,
  mpf_rand, mpf_eq, mpf_hash, mpf_cmp, mpf_lt, mpf_le, mpf_gt, mpf_ge,
  mpf_pos, mpf_neg, mpf_abs, mpf_sign, mpf_add, mpf_sub, mpf_sum,
  mpf_mul, mpf_mul_int, mpf_shift, mpf_frexp,
  mpf_div, mpf_rdiv_int, mpf_mod, mpf_pow_int,
  mpf_perturb,
  to_digits_exp, to_str, str_to_man_exp, from_str, from_bstr, to_bstr,
  mpf_sqrt, mpf_hypot)

from .libmpc import (mpc_one, mpc_zero, mpc_two, mpc_half,
  mpc_is_inf, mpc_is_infnan, mpc_to_str, mpc_to_complex, mpc_hash,
  mpc_conjugate, mpc_is_nonzero, mpc_add, mpc_add_mpf,
  mpc_sub, mpc_sub_mpf, mpc_pos, mpc_neg, mpc_shift, mpc_abs,
  mpc_arg, mpc_floor, mpc_ceil,  mpc_nint, mpc_frac, mpc_mul, mpc_square,
  mpc_mul_mpf, mpc_mul_imag_mpf, mpc_mul_int,
  mpc_div, mpc_div_mpf, mpc_reciprocal, mpc_mpf_div,
  complex_int_pow, mpc_pow, mpc_pow_mpf, mpc_pow_int,
  mpc_sqrt, mpc_nthroot, mpc_cbrt, mpc_exp, mpc_log, mpc_cos, mpc_sin,
  mpc_tan, mpc_cos_pi, mpc_sin_pi, mpc_cosh, mpc_sinh, mpc_tanh,
  mpc_atan, mpc_acos, mpc_asin, mpc_asinh, mpc_acosh, mpc_atanh,
  mpc_fibonacci, mpf_expj, mpf_expjpi, mpc_expj, mpc_expjpi,
  mpc_cos_sin, mpc_cos_sin_pi)

from .libelefun import (ln2_fixed, mpf_ln2, ln10_fixed, mpf_ln10,
  pi_fixed, mpf_pi, e_fixed, mpf_e, phi_fixed, mpf_phi,
  degree_fixed, mpf_degree,
  mpf_pow, mpf_nthroot, mpf_cbrt, log_int_fixed, agm_fixed,
  mpf_log, mpf_log_hypot, mpf_exp, mpf_cos_sin, mpf_cos, mpf_sin, mpf_tan,
  mpf_cos_sin_pi, mpf_cos_pi, mpf_sin_pi, mpf_cosh_sinh,
  mpf_cosh, mpf_sinh, mpf_tanh, mpf_atan, mpf_atan2, mpf_asin,
  mpf_acos, mpf_asinh, mpf_acosh, mpf_atanh, mpf_fibonacci)

from .libhyper import (NoConvergence, make_hyp_summator,
  mpf_erf, mpf_erfc, mpf_ei, mpc_ei, mpf_e1, mpc_e1, mpf_expint,
  mpf_ci_si, mpf_ci, mpf_si, mpc_ci, mpc_si, mpf_besseljn,
  mpc_besseljn, mpf_agm, mpf_agm1, mpc_agm, mpc_agm1,
  mpf_ellipk, mpc_ellipk, mpf_ellipe, mpc_ellipe)

from .gammazeta import (catalan_fixed, mpf_catalan,
  khinchin_fixed, mpf_khinchin, glaisher_fixed, mpf_glaisher,
  apery_fixed, mpf_apery, euler_fixed, mpf_euler, mertens_fixed,
  mpf_mertens, twinprime_fixed, mpf_twinprime,
  mpf_bernoulli, bernfrac, mpf_gamma_int,
  mpf_factorial, mpc_factorial, mpf_gamma, mpc_gamma,
  mpf_loggamma, mpc_loggamma, mpf_rgamma, mpc_rgamma,
  mpf_harmonic, mpc_harmonic, mpf_psi0, mpc_psi0,
  mpf_psi, mpc_psi, mpf_zeta_int, mpf_zeta, mpc_zeta,
  mpf_altzeta, mpc_altzeta, mpf_zetasum, mpc_zetasum)

from .libmpi import (mpi_str,
  mpi_from_str, mpi_to_str,
  mpi_eq, mpi_ne,
  mpi_lt, mpi_le, mpi_gt, mpi_ge,
  mpi_add, mpi_sub, mpi_delta, mpi_mid,
  mpi_pos, mpi_neg, mpi_abs, mpi_mul, mpi_div, mpi_exp,
  mpi_log, mpi_sqrt, mpi_pow_int, mpi_pow, mpi_cos_sin,
  mpi_cos, mpi_sin, mpi_tan, mpi_cot,
  mpi_atan, mpi_atan2,
  mpci_pos, mpci_neg, mpci_add, mpci_sub, mpci_mul, mpci_div, mpci_pow,
  mpci_abs, mpci_pow, mpci_exp, mpci_log, mpci_cos, mpci_sin,
  mpi_gamma, mpci_gamma, mpi_loggamma, mpci_loggamma,
  mpi_rgamma, mpci_rgamma, mpi_factorial, mpci_factorial)

from .libintmath import (trailing, bitcount, numeral, bin_to_radix,
  isqrt, isqrt_small, isqrt_fast, sqrt_fixed, sqrtrem, ifib, ifac,
  list_primes, isprime, moebius, gcd, eulernum, stirling1, stirling2)

from .backend import (gmpy, sage, BACKEND, STRICT, MPZ, MPZ_TYPE,
  MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_THREE, MPZ_FIVE, int_types,
  HASH_MODULUS, HASH_BITS)
<|endoftext|>"""
Utility functions for integer math.

TODO: rename, cleanup, perhaps move the gmpy wrapper code
here from settings.py

"""

import math
from bisect import bisect

from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO

small_trailing = [0] * 256
for j in range(1,8):
    small_trailing[1<<j::1<<(j+1)] = [j] * (1<<(7-j))

def giant_steps(start, target, n=2):
    """
    Return a list of integers ~=

    [start, n*start, ..., target/n^2, target/n, target]

    but conservatively rounded so that the quotient between two
    successive elements is actually slightly less than n.

    With n = 2, this describes suitable precision steps for a
    quadratically convergent algorithm such as Newton's method;
    with n = 3 steps for cubic convergence (Halley's method), etc.

        >>> giant_steps(50,1000)
        [66, 128, 253, 502, 1000]
        >>> giant_steps(50,1000,4)
        [65, 252, 1000]

    """
    L = [target]
    while L[-1] > start*n:
        L = L + [L[-1]//n + 2]
    return L[::-1]

def rshift(x, n):
    """For an integer x, calculate x >> n with the fastest (floor)
    rounding. Unlike the plain Python expression (x >> n), n is
    allowed to be negative, in which case a left shift is performed."""
    if n >= 0: return x >> n
    else:      return x << (-n)

def lshift(x, n):
    """For an integer x, calculate x << n. Unlike the plain Python
    expression (x << n), n is allowed to be negative, in which case a
    right shift with default (floor) rounding is performed."""
    if n >= 0: return x << n
    else:      return x >> (-n)

if BACKEND == 'sage':
    import operator
    rshift = operator.rshift
    lshift = operator.lshift

def python_trailing(n):
    """Count the number of trailing zero bits in abs(n)."""
    if not n:
        return 0
    low_byte = n & 0xff
    if low_byte:
        return small_trailing[low_byte]
    t = 8
    n >>= 8
    while not n & 0xff:
        n >>= 8
        t += 8
    return t + small_trailing[n & 0xff]

if BACKEND == 'gmpy':
    if gmpy.version() >= '2':
        def gmpy_trailing(n):
            """Count the number of trailing zero bits in abs(n) using gmpy."""
            if n: return MPZ(n).bit_scan1()
            else: return 0
    else:
        def gmpy_trailing(n):
            """Count the number of trailing zero bits in abs(n) using gmpy."""
            if n: return MPZ(n).scan1()
            else: return 0

# Small powers of 2
powers = [1<<_ for _ in range(300)]

def python_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    bc = bisect(powers, n)
    if bc != 300:
        return bc
    bc = int(math.log(n, 2)) - 4
    return bc + bctable[n>>bc]

def gmpy_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    if n: return MPZ(n).numdigits(2)
    else: return 0

#def sage_bitcount(n):
#    if n: return MPZ(n).nbits()
#    else: return 0

def sage_trailing(n):
    return MPZ(n).trailing_zero_bits()

if BACKEND == 'gmpy':
    bitcount = gmpy_bitcount
    trailing = gmpy_trailing
elif BACKEND == 'sage':
    sage_bitcount = sage_utils.bitcount
    bitcount = sage_bitcount
    trailing = sage_trailing
else:
    bitcount = python_bitcount
    trailing = python_trailing

if BACKEND == 'gmpy' and 'bit_length' in dir(gmpy):
    bitcount = gmpy.bit_length

# Used to avoid slow function calls as far as possible
trailtable = [trailing(n) for n in range(256)]
bctable = [bitcount(n) for n in range(1024)]

# TODO: speed up for bases 2, 4, 8, 16, ...

def bin_to_radix(x, xbits, base, bdigits):
    """Changes radix of a fixed-point number; i.e., converts
    x * 2**xbits to floor(x * 10**bdigits)."""
    return x * (MPZ(base)**bdigits) >> xbits

stddigits = '0123456789abcdefghijklmnopqrstuvwxyz'

def small_numeral(n, base=10, digits=stddigits):
    """Return the string numeral of a positive integer in an arbitrary
    base. Most efficient for small input."""
    if base == 10:
        return str(n)
    digs = []
    while n:
        n, digit = divmod(n, base)
        digs.append(digits[digit])
    return "".join(digs[::-1])

def numeral_python(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n <= 0:
        if not n:
            return "0"
        return "-" + numeral(-n, base, size, digits)
    # Fast enough to do directly
    if size < 250:
        return small_numeral(n, base, digits)
    # Divide in half
    half = (size // 2) + (size & 1)
    A, B = divmod(n, base**half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, "0")
    return ad + bd

def numeral_gmpy(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n < 0:
        return "-" + numeral(-n, base, size, digits)
    # gmpy.digits() may cause a segmentation fault when trying to convert
    # extremely large values to a string. The size limit may need to be
    # adjusted on some platforms, but 1500000 works on Windows and Linux.
    if size < 1500000:
        return gmpy.digits(n, base)
    # Divide in half
    half = (size // 2) + (size & 1)
    A, B = divmod(n, MPZ(base)**half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, "0")
    return ad + bd

if BACKEND == "gmpy":
    numeral = numeral_gmpy
else:
    numeral = numeral_python

_1_800 = 1<<800
_1_600 = 1<<600
_1_400 = 1<<400
_1_200 = 1<<200
_1_100 = 1<<100
_1_50 = 1<<50

def isqrt_small_python(x):
    """
    Correctly (floor) rounded integer square root, using
    division. Fast up to ~200 digits.
    """
    if not x:
        return x
    if x < _1_800:
        # Exact with IEEE double precision arithmetic
        if x < _1_50:
            return int(x**0.5)
        # Initial estimate can be any integer >= the true root; round up
        r = int(x**0.5 * 1.00000000000001) + 1
    else:
        bc = bitcount(x)
        n = bc//2
        r = int((x>>(2*n-100))**0.5+2)<<(n-50)  # +2 is to round up
    # The following iteration now precisely computes floor(sqrt(x))
    # See e.g. Crandall & Pomerance, "Prime Numbers: A Computational
    # Perspective"
    while 1:
        y = (r+x//r)>>1
        if y >= r:
            return r
        r = y

def isqrt_fast_python(x):
    """
    Fast approximate integer square root, computed using division-free
    Newton iteration for large x. For random integers the result is almost
    always correct (floor(sqrt(x))), but is 1 ulp too small with a roughly
    0.1% probability. If x is very close to an exact square, the answer is
    1 ulp wrong with high probability.

    With 0 guard bits, the largest error over a set of 10^5 random
    inputs of size 1-10^5 bits was 3 ulp. The use of 10 guard bits
    almost certainly guarantees a max 1 ulp error.
    """
    # Use direct division-based iteration if sqrt(x) < 2^400
    # Assume floating-point square root accurate to within 1 ulp, then:
    # 0 Newton iterations good to 52 bits
    # 1 Newton iterations good to 104 bits
    # 2 Newton iterations good to 208 bits
    # 3 Newton iterations good to 416 bits
    if x < _1_800:
        y = int(x**0.5)
        if x >= _1_100:
            y = (y + x//y) >> 1
            if x >= _1_200:
                y = (y + x//y) >> 1
                if x >= _1_400:
                    y = (y + x//y) >> 1
        return y
    bc = bitcount(x)
    guard_bits = 10
    x <<= 2*guard_bits
    bc += 2*guard_bits
    bc += (bc&1)
    hbc = bc//2
    startprec = min(50, hbc)
    # Newton iteration for 1/sqrt(x), with floating-point starting value
    r = int(2.0**(2*startprec) * (x >> (bc-2*startprec)) ** -0.5)
    pp = startprec
    for p in giant_steps(startprec, hbc):
        # r**2, scaled from real size 2**(-bc) to 2**p
        r2 = (r*r) >> (2*pp - p)
        # x*r**2, scaled from real size ~1.0 to 2**p
        xr2 = ((x >> (bc-p)) * r2) >> p
        # New value of r, scaled from real size 2**(-bc/2) to 2**p
        r = (r * ((3<<p) - xr2)) >> (pp+1)
        pp = p
    # (1/sqrt(x))*x = sqrt(x)
    return (r*(x>>hbc)) >> (p+guard_bits)

def sqrtrem_python(x):
    """Correctly rounded integer (floor) square root with remainder."""
    # to check cutoff:
    # plot(lambda x: timing(isqrt, 2**int(x)), [0,2000])
    if x < _1_600:
        y = isqrt_small_python(x)
        return y, x - y*y
    y = isqrt_fast_python(x) + 1
    rem = x - y*y
    # Correct remainder
    while rem < 0:
        y -= 1
        rem += (1+2*y)
    else:
        if rem:
            while rem > 2*(1+y):
                y += 1
                rem -= (1+2*y)
    return y, rem

def isqrt_python(x):
    """Integer square root with correct (floor) rounding."""
    return sqrtrem_python(x)[0]

def sqrt_fixed(x, prec):
    return isqrt_fast(x<<prec)

sqrt_fixed2 = sqrt_fixed

if BACKEND == 'gmpy':
    if gmpy.version() >= '2':
        isqrt_small = isqrt_fast = isqrt = gmpy.isqrt
        sqrtrem = gmpy.isqrt_rem
    else:
        isqrt_small = isqrt_fast = isqrt = gmpy.sqrt
        sqrtrem = gmpy.sqrtrem
elif BACKEND == 'sage':
    isqrt_small = isqrt_fast = isqrt = \
        getattr(sage_utils, "isqrt", lambda n: MPZ(n).isqrt())
    sqrtrem = lambda n: MPZ(n).sqrtrem()
else:
    isqrt_small = isqrt_small_python
    isqrt_fast = isqrt_fast_python
    isqrt = isqrt_python
    sqrtrem = sqrtrem_python


def ifib(n, _cache={}):
    """Computes the nth Fibonacci number as an integer, for
    integer n."""
    if n < 0:
        return (-1)**(-n+1) * ifib(-n)
    if n in _cache:
        return _cache[n]
    m = n
    # Use Dijkstra's logarithmic algorithm
    # The following implementation is basically equivalent to
    # http://en.literateprograms.org/Fibonacci_numbers_(Scheme)
    a, b, p, q = MPZ_ONE, MPZ_ZERO, MPZ_ZERO, MPZ_ONE
    while n:
        if n & 1:
            aq = a*q
            a, b = b*q+aq+a*p, b*p+aq
            n -= 1
        else:
            qq = q*q
            p, q = p*p+qq, qq+2*p*q
            n >>= 1
    if m < 250:
        _cache[m] = b
    return b

MAX_FACTORIAL_CACHE = 1000

def ifac(n, memo={0:1, 1:1}):
    """Return n factorial (for integers n >= 0 only)."""
    f = memo.get(n)
    if f:
        return f
    k = len(memo)
    p = memo[k-1]
    MAX = MAX_FACTORIAL_CACHE
    while k <= n:
        p *= k
        if k <= MAX:
            memo[k] = p
        k += 1
    return p

def ifac2(n, memo_pair=[{0:1}, {1:1}]):
    """Return n!! (double factorial), integers n >= 0 only."""
    memo = memo_pair[n&1]
    f = memo.get(n)
    if f:
        return f
    k = max(memo)
    p = memo[k]
    MAX = MAX_FACTORIAL_CACHE
    while k < n:
        k += 2
        p *= k
        if k <= MAX:
            memo[k] = p
    return p

if BACKEND == 'gmpy':
    ifac = gmpy.fac
elif BACKEND == 'sage':
    ifac = lambda n: int(sage.factorial(n))
    ifib = sage.fibonacci

def list_primes(n):
    n = n + 1
    sieve = list(xrange(n))
    sieve[:2] = [0, 0]
    for i in xrange(2, int(n**0.5)+1):
        if sieve[i]:
            for j in xrange(i**2, n, i):
                sieve[j] = 0
    return [p for p in sieve if p]

if BACKEND == 'sage':
    # Note: it is *VERY* important for performance that we convert
    # the list to Python ints.
    def list_primes(n):
        return [int(_) for _ in sage.primes(n+1)]

small_odd_primes = (3,5,7,11,13,17,19,23,29,31,37,41,43,47)
small_odd_primes_set = set(small_odd_primes)

def isprime(n):
    """
    Determines whether n is a prime number. A probabilistic test is
    performed if n is very large. No special trick is used for detecting
    perfect powers.

        >>> sum(list_primes(100000))
        454396537
        >>> sum(n*isprime(n) for n in range(100000))
        454396537

    """
    n = int(n)
    if not n & 1:
        return n == 2
    if n < 50:
        return n in small_odd_primes_set
    for p in small_odd_primes:
        if not n % p:
            return False
    m = n-1
    s = trailing(m)
    d = m >> s
    def test(a):
        x = pow(a,d,n)
        if x == 1 or x == m:
            return True
        for r in xrange(1,s):
            x = x**2 % n
            if x == m:
                return True
        return False
    # See http://primes.utm.edu/prove/prove2_3.html
    if n < 1373653:
        witnesses = [2,3]
    elif n < 341550071728321:
        witnesses = [2,3,5,7,11,13,17]
    else:
        witnesses = small_odd_primes
    for a in witnesses:
        if not test(a):
            return False
    return True

def moebius(n):
    """
    Evaluates the Moebius function which is `mu(n) = (-1)^k` if `n`
    is a product of `k` distinct primes and `mu(n) = 0` otherwise.

    TODO: speed up using factorization
    """
    n = abs(int(n))
    if n < 2:
        return n
    factors = []
    for p in xrange(2, n+1):
        if not (n % p):
            if not (n % p**2):
                return 0
            if not sum(p % f for f in factors):
                factors.append(p)
    return (-1)**len(factors)

def gcd(*args):
    a = 0
    for b in args:
        if a:
            while b:
                a, b = b, a % b
        else:
            a = b
    return a


#  Comment by Juan Arias de Reyna:
#
#  I learn this method to compute EulerE[2n] from van de Lune.
#
#  We apply the formula   EulerE[2n] = (-1)^n 2**(-2n) sum_{j=0}^n a(2n,2j+1)
#
#  where the numbers a(n,j) vanish for  j > n+1 or j <= -1  and satisfies
#
#  a(0,-1) = a(0,0) = 0;  a(0,1)= 1; a(0,2) = a(0,3) = 0
#
#  a(n,j) = a(n-1,j)                              when n+j is even
#  a(n,j) = (j-1) a(n-1,j-1) + (j+1) a(n-1,j+1)   when n+j is odd
#
#
#  But we can use only one array unidimensional a(j) since to compute
#  a(n,j) we only need to know a(n-1,k) where k and j are of different parity
#  and we have not to conserve the used values.
#
#  We cached up the values of Euler numbers to sufficiently high order.
#
#  Important Observation: If we pretend to use the numbers
#     EulerE[1], EulerE[2], ... , EulerE[n]
#     it is convenient to compute first EulerE[n], since the algorithm
#     computes first all
#     the previous ones, and keeps them in the CACHE

MAX_EULER_CACHE = 500

def eulernum(m, _cache={0:MPZ_ONE}):
    r"""
    Computes the Euler numbers `E(n)`, which can be defined as
    coefficients of the Taylor expansion of `1/cosh x`:

    .. math ::

        \frac{1}{\cosh x} = \sum_{n=0}^\infty \frac{E_n}{n!} x^n

    Example::

        >>> [int(eulernum(n)) for n in range(11)]
        [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
        >>> [int(eulernum(n)) for n in range(11)]   # test cache
        [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]

    """
    # for odd m > 1, the Euler numbers are zero
    if m & 1:
        return MPZ_ZERO
    f = _cache.get(m)
    if f:
        return f
    MAX = MAX_EULER_CACHE
    n = m
    a = [MPZ(_) for _ in [0,0,1,0,0,0]]
    for  n in range(1, m+1):
        for j in range(n+1, -1, -2):
            a[j+1] = (j-1)*a[j] + (j+1)*a[j+2]
        a.append(0)
        suma = 0
        for k in range(n+1, -1, -2):
            suma += a[k+1]
            if n <= MAX:
                _cache[n] = ((-1)**(n//2))*(suma // 2**n)
        if n == m:
            return ((-1)**(n//2))*suma // 2**n

def stirling1(n, k):
    """
    Stirling number of the first kind.
    """
    if n < 0 or k < 0:
        raise ValueError
    if k >= n:
        return MPZ(n == k)
    if k < 1:
        return MPZ_ZERO
    L = [MPZ_ZERO] * (k+1)
    L[1] = MPZ_ONE
    for m in xrange(2, n+1):
        for j in xrange(min(k, m), 0, -1):
            L[j] = (m-1) * L[j] + L[j-1]
    return (-1)**(n+k) * L[k]

def stirling2(n, k):
    """
    Stirling number of the second kind.
    """
    if n < 0 or k < 0:
        raise ValueError
    if k >= n:
        return MPZ(n == k)
    if k <= 1:
        return MPZ(k == 1)
    s = MPZ_ZERO
    t = MPZ_ONE
    for j in xrange(k+1):
        if (k + j) & 1:
            s -= t * MPZ(j)**n
        else:
            s += t * MPZ(j)**n
        t = t * (k - j) // (j + 1)
    return s // ifac(k)
<|endoftext|>"""
Low-level functions for arbitrary-precision floating-point arithmetic.
"""

__docformat__ = 'plaintext'

import math

from bisect import bisect

import sys

# Importing random is slow
#from random import getrandbits
getrandbits = None

from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
    BACKEND, STRICT, HASH_MODULUS, HASH_BITS, gmpy, sage, sage_utils)

from .libintmath import (giant_steps,
    trailtable, bctable, lshift, rshift, bitcount, trailing,
    sqrt_fixed, numeral, isqrt, isqrt_fast, sqrtrem,
    bin_to_radix)

# We don't pickle tuples directly for the following reasons:
#   1: pickle uses str() for ints, which is inefficient when they are large
#   2: pickle doesn't work for gmpy mpzs
# Both problems are solved by using hex()

if BACKEND == 'sage':
    def to_pickable(x):
        sign, man, exp, bc = x
        return sign, hex(man), exp, bc
else:
    def to_pickable(x):
        sign, man, exp, bc = x
        return sign, hex(man)[2:], exp, bc

def from_pickable(x):
    sign, man, exp, bc = x
    return (sign, MPZ(man, 16), exp, bc)

class ComplexResult(ValueError):
    pass

try:
    intern
except NameError:
    intern = lambda x: x

# All supported rounding modes
round_nearest = intern('n')
round_floor = intern('f')
round_ceiling = intern('c')
round_up = intern('u')
round_down = intern('d')
round_fast = round_down

def prec_to_dps(n):
    """Return number of accurate decimals that can be represented
    with a precision of n bits."""
    return max(1, int(round(int(n)/3.3219280948873626)-1))

def dps_to_prec(n):
    """Return the number of bits required to represent n decimals
    accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))

def repr_dps(n):
    """Return the number of decimal digits required to represent
    a number with n-bit precision so that it can be uniquely
    reconstructed from the representation."""
    dps = prec_to_dps(n)
    if dps == 15:
        return 17
    return dps + 3

#----------------------------------------------------------------------------#
#                    Some commonly needed float values                       #
#----------------------------------------------------------------------------#

# Regular number format:
# (-1)**sign * mantissa * 2**exponent, plus bitcount of mantissa
fzero = (0, MPZ_ZERO, 0, 0)
fnzero = (1, MPZ_ZERO, 0, 0)
fone = (0, MPZ_ONE, 0, 1)
fnone = (1, MPZ_ONE, 0, 1)
ftwo = (0, MPZ_ONE, 1, 1)
ften = (0, MPZ_FIVE, 1, 3)
fhalf = (0, MPZ_ONE, -1, 1)

# Arbitrary encoding for special numbers: zero mantissa, nonzero exponent
fnan = (0, MPZ_ZERO, -123, -1)
finf = (0, MPZ_ZERO, -456, -2)
fninf = (1, MPZ_ZERO, -789, -3)

# Was 1e1000; this is broken in Python 2.4
math_float_inf = 1e300 * 1e300


#----------------------------------------------------------------------------#
#                                  Rounding                                  #
#----------------------------------------------------------------------------#

# This function can be used to round a mantissa generally. However,
# we will try to do most rounding inline for efficiency.
def round_int(x, n, rnd):
    if rnd == round_nearest:
        if x >= 0:
            t = x >> (n-1)
            if t & 1 and ((t & 2) or (x & h_mask[n<300][n])):
                return (t>>1)+1
            else:
                return t>>1
        else:
            return -round_int(-x, n, rnd)
    if rnd == round_floor:
        return x >> n
    if rnd == round_ceiling:
        return -((-x) >> n)
    if rnd == round_down:
        if x >= 0:
            return x >> n
        return -((-x) >> n)
    if rnd == round_up:
        if x >= 0:
            return -((-x) >> n)
        return x >> n

# These masks are used to pick out segments of numbers to determine
# which direction to round when rounding to nearest.
class h_mask_big:
    def __getitem__(self, n):
        return (MPZ_ONE<<(n-1))-1

h_mask_small = [0]+[((MPZ_ONE<<(_-1))-1) for _ in range(1, 300)]
h_mask = [h_mask_big(), h_mask_small]

# The >> operator rounds to floor. shifts_down[rnd][sign]
# tells whether this is the right direction to use, or if the
# number should be negated before shifting
shifts_down = {round_floor:(1,0), round_ceiling:(0,1),
    round_down:(1,1), round_up:(0,0)}


#----------------------------------------------------------------------------#
#                          Normalization of raw mpfs                         #
#----------------------------------------------------------------------------#

# This function is called almost every time an mpf is created.
# It has been optimized accordingly.

def _normalize(sign, man, exp, bc, prec, rnd):
    """
    Create a raw mpf tuple with value (-1)**sign * man * 2**exp and
    normalized mantissa. The mantissa is rounded in the specified
    direction if its size exceeds the precision. Trailing zero bits
    are also stripped from the mantissa to ensure that the
    representation is canonical.

    Conditions on the input:
    * The input must represent a regular (finite) number
    * The sign bit must be 0 or 1
    * The mantissa must be positive
    * The exponent must be an integer
    * The bitcount must be exact

    If these conditions are not met, use from_man_exp, mpf_pos, or any
    of the conversion functions to create normalized raw mpf tuples.
    """
    if not man:
        return fzero
    # Cut mantissa down to size if larger than target precision
    n = bc - prec
    if n > 0:
        if rnd == round_nearest:
            t = man >> (n-1)
            if t & 1 and ((t & 2) or (man & h_mask[n<300][n])):
                man = (t>>1)+1
            else:
                man = t>>1
        elif shifts_down[rnd][sign]:
            man >>= n
        else:
            man = -((-man)>>n)
        exp += n
        bc = prec
    # Strip trailing bits
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    # Bit count can be wrong if the input mantissa was 1 less than
    # a power of 2 and got rounded up, thereby adding an extra bit.
    # With trailing bits removed, all powers of two have mantissa 1,
    # so this is easy to check for.
    if man == 1:
        bc = 1
    return sign, man, exp, bc

def _normalize1(sign, man, exp, bc, prec, rnd):
    """same as normalize, but with the added condition that
       man is odd or zero
    """
    if not man:
        return fzero
    if bc <= prec:
        return sign, man, exp, bc
    n = bc - prec
    if rnd == round_nearest:
        t = man >> (n-1)
        if t & 1 and ((t & 2) or (man & h_mask[n<300][n])):
            man = (t>>1)+1
        else:
            man = t>>1
    elif shifts_down[rnd][sign]:
        man >>= n
    else:
        man = -((-man)>>n)
    exp += n
    bc = prec
    # Strip trailing bits
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    # Bit count can be wrong if the input mantissa was 1 less than
    # a power of 2 and got rounded up, thereby adding an extra bit.
    # With trailing bits removed, all powers of two have mantissa 1,
    # so this is easy to check for.
    if man == 1:
        bc = 1
    return sign, man, exp, bc

try:
    _exp_types = (int, long)
except NameError:
    _exp_types = (int,)

def strict_normalize(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    return _normalize(sign, man, exp, bc, prec, rnd)

def strict_normalize1(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    assert (not man) or (man & 1)
    return _normalize1(sign, man, exp, bc, prec, rnd)

if BACKEND == 'gmpy' and '_mpmath_normalize' in dir(gmpy):
    _normalize = gmpy._mpmath_normalize
    _normalize1 = gmpy._mpmath_normalize

if BACKEND == 'sage':
    _normalize = _normalize1 = sage_utils.normalize

if STRICT:
    normalize = strict_normalize
    normalize1 = strict_normalize1
else:
    normalize = _normalize
    normalize1 = _normalize1

#----------------------------------------------------------------------------#
#                            Conversion functions                            #
#----------------------------------------------------------------------------#

def from_man_exp(man, exp, prec=None, rnd=round_fast):
    """Create raw mpf from (man, exp) pair. The mantissa may be signed.
    If no precision is specified, the mantissa is stored exactly."""
    man = MPZ(man)
    sign = 0
    if man < 0:
        sign = 1
        man = -man
    if man < 1024:
        bc = bctable[int(man)]
    else:
        bc = bitcount(man)
    if not prec:
        if not man:
            return fzero
        if not man & 1:
            if man & 2:
                return (sign, man >> 1, exp + 1, bc - 1)
            t = trailtable[int(man & 255)]
            if not t:
                while not man & 255:
                    man >>= 8
                    exp += 8
                    bc -= 8
                t = trailtable[int(man & 255)]
            man >>= t
            exp += t
            bc -= t
        return (sign, man, exp, bc)
    return normalize(sign, man, exp, bc, prec, rnd)

int_cache = dict((n, from_man_exp(n, 0)) for n in range(-10, 257))

if BACKEND == 'gmpy' and '_mpmath_create' in dir(gmpy):
    from_man_exp = gmpy._mpmath_create

if BACKEND == 'sage':
    from_man_exp = sage_utils.from_man_exp

def from_int(n, prec=0, rnd=round_fast):
    """Create a raw mpf from an integer. If no precision is specified,
    the mantissa is stored exactly."""
    if not prec:
        if n in int_cache:
            return int_cache[n]
    return from_man_exp(n, 0, prec, rnd)

def to_man_exp(s):
    """Return (man, exp) of a raw mpf. Raise an error if inf/nan."""
    sign, man, exp, bc = s
    if (not man) and exp:
        raise ValueError("mantissa and exponent are undefined for %s" % man)
    return man, exp

def to_int(s, rnd=None):
    """Convert a raw mpf to the nearest int. Rounding is done down by
    default (same as int(float) in Python), but can be changed. If the
    input is inf/nan, an exception is raised."""
    sign, man, exp, bc = s
    if (not man) and exp:
        raise ValueError("cannot convert inf or nan to int")
    if exp >= 0:
        if sign:
            return (-man) << exp
        return man << exp
    # Make default rounding fast
    if not rnd:
        if sign:
            return -(man >> (-exp))
        else:
            return man >> (-exp)
    if sign:
        return round_int(-man, -exp, rnd)
    else:
        return round_int(man, -exp, rnd)

def mpf_round_int(s, rnd):
    sign, man, exp, bc = s
    if (not man) and exp:
        return s
    if exp >= 0:
        return s
    mag = exp+bc
    if mag < 1:
        if rnd == round_ceiling:
            if sign: return fzero
            else:    return fone
        elif rnd == round_floor:
            if sign: return fnone
            else:    return fzero
        elif rnd == round_nearest:
            if mag < 0 or man == MPZ_ONE: return fzero
            elif sign: return fnone
            else:      return fone
        else:
            raise NotImplementedError
    return mpf_pos(s, min(bc, mag), rnd)

def mpf_floor(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_floor)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_ceil(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_ceiling)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_nint(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_nearest)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_frac(s, prec=0, rnd=round_fast):
    return mpf_sub(s, mpf_floor(s), prec, rnd)

def from_float(x, prec=53, rnd=round_fast):
    """Create a raw mpf from a Python float, rounding if necessary.
    If prec >= 53, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=53."""
    # frexp only raises an exception for nan on some platforms
    if x != x:
        return fnan
    # in Python2.5 math.frexp gives an exception for float infinity
    # in Python2.6 it returns (float infinity, 0)
    try:
        m, e = math.frexp(x)
    except:
        if x == math_float_inf: return finf
        if x == -math_float_inf: return fninf
        return fnan
    if x == math_float_inf: return finf
    if x == -math_float_inf: return fninf
    return from_man_exp(int(m*(1<<53)), e-53, prec, rnd)

def from_npfloat(x, prec=113, rnd=round_fast):
    """Create a raw mpf from a numpy float, rounding if necessary.
    If prec >= 113, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=113."""
    y = float(x)
    if x == y: # ldexp overflows for float16
        return from_float(y, prec, rnd)
    import numpy as np
    if np.isfinite(x):
        m, e = np.frexp(x)
        return from_man_exp(int(np.ldexp(m, 113)), int(e-113), prec, rnd)
    if np.isposinf(x): return finf
    if np.isneginf(x): return fninf
    return fnan

def from_Decimal(x, prec=None, rnd=round_fast):
    """Create a raw mpf from a Decimal, rounding if necessary.
    If prec is not specified, use the equivalent bit precision
    of the number of significant digits in x."""
    if x.is_nan(): return fnan
    if x.is_infinite(): return fninf if x.is_signed() else finf
    if prec is None:
        prec = int(len(x.as_tuple()[1])*3.3219280948873626)
    return from_str(str(x), prec, rnd)

def to_float(s, strict=False, rnd=round_fast):
    """
    Convert a raw mpf to a Python float. The result is exact if the
    bitcount of s is <= 53 and no underflow/overflow occurs.

    If the number is too large or too small to represent as a regular
    float, it will be converted to inf or 0.0. Setting strict=True
    forces an OverflowError to be raised instead.

    Warning: with a directed rounding mode, the correct nearest representable
    floating-point number in the specified direction might not be computed
    in case of overflow or (gradual) underflow.
    """
    sign, man, exp, bc = s
    if not man:
        if s == fzero: return 0.0
        if s == finf: return math_float_inf
        if s == fninf: return -math_float_inf
        return math_float_inf/math_float_inf
    if bc > 53:
        sign, man, exp, bc = normalize1(sign, man, exp, bc, 53, rnd)
    if sign:
        man = -man
    try:
        return math.ldexp(man, exp)
    except OverflowError:
        if strict:
            raise
        # Overflow to infinity
        if exp + bc > 0:
            if sign:
                return -math_float_inf
            else:
                return math_float_inf
        # Underflow to zero
        return 0.0

def from_rational(p, q, prec, rnd=round_fast):
    """Create a raw mpf from a rational number p/q, round if
    necessary."""
    return mpf_div(from_int(p), from_int(q), prec, rnd)

def to_rational(s):
    """Convert a raw mpf to a rational number. Return integers (p, q)
    such that s = p/q exactly."""
    sign, man, exp, bc = s
    if sign:
        man = -man
    if bc == -1:
        raise ValueError("cannot convert %s to a rational number" % man)
    if exp >= 0:
        return man * (1<<exp), 1
    else:
        return man, 1<<(-exp)

def to_fixed(s, prec):
    """Convert a raw mpf to a fixed-point big integer"""
    sign, man, exp, bc = s
    offset = exp + prec
    if sign:
        if offset >= 0: return (-man) << offset
        else:           return (-man) >> (-offset)
    else:
        if offset >= 0: return man << offset
        else:           return man >> (-offset)


##############################################################################
##############################################################################

#----------------------------------------------------------------------------#
#                       Arithmetic operations, etc.                          #
#----------------------------------------------------------------------------#

def mpf_rand(prec):
    """Return a raw mpf chosen randomly from [0, 1), with prec bits
    in the mantissa."""
    global getrandbits
    if not getrandbits:
        import random
        getrandbits = random.getrandbits
    return from_man_exp(getrandbits(prec), -prec, prec, round_floor)

def mpf_eq(s, t):
    """Test equality of two raw mpfs. This is simply tuple comparison
    unless either number is nan, in which case the result is False."""
    if not s[1] or not t[1]:
        if s == fnan or t == fnan:
            return False
    return s == t

def mpf_hash(s):
    # Duplicate the new hash algorithm introduces in Python 3.2.
    if sys.version_info >= (3, 2):
        ssign, sman, sexp, sbc = s

        # Handle special numbers
        if not sman:
            if s == fnan: return sys.hash_info.nan
            if s == finf: return sys.hash_info.inf
            if s == fninf: return -sys.hash_info.inf
        h = sman % HASH_MODULUS
        if sexp >= 0:
            sexp = sexp % HASH_BITS
        else:
            sexp = HASH_BITS - 1 - ((-1 - sexp) % HASH_BITS)
        h = (h << sexp) % HASH_MODULUS
        if ssign: h = -h
        if h == -1: h = -2
        return int(h)
    else:
        try:
            # Try to be compatible with hash values for floats and ints
            return hash(to_float(s, strict=1))
        except OverflowError:
            # We must unfortunately sacrifice compatibility with ints here.
            # We could do hash(man << exp) when the exponent is positive, but
            # this would cause unreasonable inefficiency for large numbers.
            return hash(s)

def mpf_cmp(s, t):
    """Compare the raw mpfs s and t. Return -1 if s < t, 0 if s == t,
    and 1 if s > t. (Same convention as Python's cmp() function.)"""

    # In principle, a comparison amounts to determining the sign of s-t.
    # A full subtraction is relatively slow, however, so we first try to
    # look at the components.
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t

    # Handle zeros and special numbers
    if not sman or not tman:
        if s == fzero: return -mpf_sign(t)
        if t == fzero: return mpf_sign(s)
        if s == t: return 0
        # Follow same convention as Python's cmp for float nan
        if t == fnan: return 1
        if s == finf: return 1
        if t == fninf: return 1
        return -1
    # Different sides of zero
    if ssign != tsign:
        if not ssign: return 1
        return -1
    # This reduces to direct integer comparison
    if sexp == texp:
        if sman == tman:
            return 0
        if sman > tman:
            if ssign: return -1
            else:     return 1
        else:
            if ssign: return 1
            else:     return -1
    # Check position of the highest set bit in each number. If
    # different, there is certainly an inequality.
    a = sbc + sexp
    b = tbc + texp
    if ssign:
        if a < b: return 1
        if a > b: return -1
    else:
        if a < b: return -1
        if a > b: return 1

    # Both numbers have the same highest bit. Subtract to find
    # how the lower bits compare.
    delta = mpf_sub(s, t, 5, round_floor)
    if delta[0]:
        return -1
    return 1

def mpf_lt(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) < 0

def mpf_le(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) <= 0

def mpf_gt(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) > 0

def mpf_ge(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) >= 0

def mpf_min_max(seq):
    min = max = seq[0]
    for x in seq[1:]:
        if mpf_lt(x, min): min = x
        if mpf_gt(x, max): max = x
    return min, max

def mpf_pos(s, prec=0, rnd=round_fast):
    """Calculate 0+s for a raw mpf (i.e., just round s to the specified
    precision)."""
    if prec:
        sign, man, exp, bc = s
        if (not man) and exp:
            return s
        return normalize1(sign, man, exp, bc, prec, rnd)
    return s

def mpf_neg(s, prec=None, rnd=round_fast):
    """Negate a raw mpf (return -s), rounding the result to the
    specified precision. The prec argument can be omitted to do the
    operation exactly."""
    sign, man, exp, bc = s
    if not man:
        if exp:
            if s == finf: return fninf
            if s == fninf: return finf
        return s
    if not prec:
        return (1-sign, man, exp, bc)
    return normalize1(1-sign, man, exp, bc, prec, rnd)

def mpf_abs(s, prec=None, rnd=round_fast):
    """Return abs(s) of the raw mpf s, rounded to the specified
    precision. The prec argument can be omitted to generate an
    exact result."""
    sign, man, exp, bc = s
    if (not man) and exp:
        if s == fninf:
            return finf
        return s
    if not prec:
        if sign:
            return (0, man, exp, bc)
        return s
    return normalize1(0, man, exp, bc, prec, rnd)

def mpf_sign(s):
    """Return -1, 0, or 1 (as a Python int, not a raw mpf) depending on
    whether s is negative, zero, or positive. (Nan is taken to give 0.)"""
    sign, man, exp, bc = s
    if not man:
        if s == finf: return 1
        if s == fninf: return -1
        return 0
    return (-1) ** sign

def mpf_add(s, t, prec=0, rnd=round_fast, _sub=0):
    """
    Add the two raw mpf values s and t.

    With prec=0, no rounding is performed. Note that this can
    produce a very large mantissa (potentially too large to fit
    in memory) if exponents are far apart.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    tsign ^= _sub
    # Standard case: two nonzero, regular numbers
    if sman and tman:
        offset = sexp - texp
        if offset:
            if offset > 0:
                # Outside precision range; only need to perturb
                if offset > 100 and prec:
                    delta = sbc + sexp - tbc - texp
                    if delta > prec + 4:
                        offset = prec + 4
                        sman <<= offset
                        if tsign == ssign: sman += 1
                        else:              sman -= 1
                        return normalize1(ssign, sman, sexp-offset,
                            bitcount(sman), prec, rnd)
                # Add
                if ssign == tsign:
                    man = tman + (sman << offset)
                # Subtract
                else:
                    if ssign: man = tman - (sman << offset)
                    else:     man = (sman << offset) - tman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, texp, bc, prec or bc, rnd)
            elif offset < 0:
                # Outside precision range; only need to perturb
                if offset < -100 and prec:
                    delta = tbc + texp - sbc - sexp
                    if delta > prec + 4:
                        offset = prec + 4
                        tman <<= offset
                        if ssign == tsign: tman += 1
                        else:              tman -= 1
                        return normalize1(tsign, tman, texp-offset,
                            bitcount(tman), prec, rnd)
                # Add
                if ssign == tsign:
                    man = sman + (tman << -offset)
                # Subtract
                else:
                    if tsign: man = sman - (tman << -offset)
                    else:     man = (tman << -offset) - sman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, sexp, bc, prec or bc, rnd)
        # Equal exponents; no shifting necessary
        if ssign == tsign:
            man = tman + sman
        else:
            if ssign: man = tman - sman
            else:     man = sman - tman
            if man >= 0:
                ssign = 0
            else:
                man = -man
                ssign = 1
        bc = bitcount(man)
        return normalize(ssign, man, texp, bc, prec or bc, rnd)
    # Handle zeros and special numbers
    if _sub:
        t = mpf_neg(t)
    if not sman:
        if sexp:
            if s == t or tman or not texp:
                return s
            return fnan
        if tman:
            return normalize1(tsign, tman, texp, tbc, prec or tbc, rnd)
        return t
    if texp:
        return t
    if sman:
        return normalize1(ssign, sman, sexp, sbc, prec or sbc, rnd)
    return s

def mpf_sub(s, t, prec=0, rnd=round_fast):
    """Return the difference of two raw mpfs, s-t. This function is
    simply a wrapper of mpf_add that changes the sign of t."""
    return mpf_add(s, t, prec, rnd, 1)

def mpf_sum(xs, prec=0, rnd=round_fast, absolute=False):
    """
    Sum a list of mpf values efficiently and accurately
    (typically no temporary roundoff occurs). If prec=0,
    the final result will not be rounded either.

    There may be roundoff error or cancellation if extremely
    large exponent differences occur.

    With absolute=True, sums the absolute values.
    """
    man = 0
    exp = 0
    max_extra_prec = prec*2 or 1000000  # XXX
    special = None
    for x in xs:
        xsign, xman, xexp, xbc = x
        if xman:
            if xsign and not absolute:
                xman = -xman
            delta = xexp - exp
            if xexp >= exp:
                # x much larger than existing sum?
                # first: quick test
                if (delta > max_extra_prec) and \
                    ((not man) or delta-bitcount(abs(man)) > max_extra_prec):
                    man = xman
                    exp = xexp
                else:
                    man += (xman << delta)
            else:
                delta = -delta
                # x much smaller than existing sum?
                if delta-xbc > max_extra_prec:
                    if not man:
                        man, exp = xman, xexp
                else:
                    man = (man << delta) + xman
                    exp = xexp
        elif xexp:
            if absolute:
                x = mpf_abs(x)
            special = mpf_add(special or fzero, x, 1)
    # Will be inf or nan
    if special:
        return special
    return from_man_exp(man, exp, prec, rnd)

def gmpy_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman*tman
    if man:
        bc = bitcount(man)
        if prec:
            return normalize1(sign, man, sexp+texp, bc, prec, rnd)
        else:
            return (sign, man, sexp+texp, bc)
    s_special = (not sman) and sexp
    t_special = (not tman) and texp
    if not s_special and not t_special:
        return fzero
    if fnan in (s, t): return fnan
    if (not tman) and texp: s, t = t, s
    if t == fzero: return fnan
