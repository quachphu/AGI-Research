                    ctx.prec = orig
            else:
                return ctx._gamma_upper_int(z, a)
        except NotImplementedError:
            pass
    # hypercomb is unable to detect the exact zeros, so handle them here
    if z == 2 and a == -1:
        return (z+a)*0
    if z == 3 and (a == -1-1j or a == -1+1j):
        return (z+a)*0
    nega = ctx.fneg(a, exact=True)
    G = [z] * regularized
    # Use 2F0 series when possible; fall back to lower gamma representation
    try:
        def h(z):
            r = z-1
            return [([ctx.exp(nega), a], [1, r], [], G, [1, -r], [], 1/nega)]
        return ctx.hypercomb(h, [z], force_series=True)
    except ctx.NoConvergence:
        def h(z):
            T1 = [], [1, z-1], [z], G, [], [], 0
            T2 = [-ctx.exp(nega), a, z], [1, z, -1], [], G, [1], [1+z], a
            return T1, T2
        return ctx.hypercomb(h, [z])

@defun
def _gamma3(ctx, z, a, b, regularized=False):
    pole = ctx.isnpint(z)
    if regularized and pole:
        return ctx.zero
    try:
        ctx.prec += 15
        # We don't know in advance whether it's better to write as a difference
        # of lower or upper gamma functions, so try both
        T1 = ctx.gammainc(z, a, regularized=regularized)
        T2 = ctx.gammainc(z, b, regularized=regularized)
        R = T1 - T2
        if ctx.mag(R) - max(ctx.mag(T1), ctx.mag(T2)) > -10:
            return R
        if not pole:
            T1 = ctx.gammainc(z, 0, b, regularized=regularized)
            T2 = ctx.gammainc(z, 0, a, regularized=regularized)
            R = T1 - T2
            # May be ok, but should probably at least print a warning
            # about possible cancellation
            if 1: #ctx.mag(R) - max(ctx.mag(T1), ctx.mag(T2)) > -10:
                return R
    finally:
        ctx.prec -= 15
    raise NotImplementedError

@defun_wrapped
def expint(ctx, n, z):
    if ctx.isint(n) and ctx._is_real_type(z):
        try:
            return ctx._expint_int(n, z)
        except NotImplementedError:
            pass
    if ctx.isnan(n) or ctx.isnan(z):
        return z*n
    if z == ctx.inf:
        return 1/z
    if z == 0:
        # integral from 1 to infinity of t^n
        if ctx.re(n) <= 1:
            # TODO: reasonable sign of infinity
            return type(z)(ctx.inf)
        else:
            return ctx.one/(n-1)
    if n == 0:
        return ctx.exp(-z)/z
    if n == -1:
        return ctx.exp(-z)*(z+1)/z**2
    return z**(n-1) * ctx.gammainc(1-n, z)

@defun_wrapped
def li(ctx, z, offset=False):
    if offset:
        if z == 2:
            return ctx.zero
        return ctx.ei(ctx.ln(z)) - ctx.ei(ctx.ln2)
    if not z:
        return z
    if z == 1:
        return ctx.ninf
    return ctx.ei(ctx.ln(z))

@defun
def ei(ctx, z):
    try:
        return ctx._ei(z)
    except NotImplementedError:
        return ctx._ei_generic(z)

@defun_wrapped
def _ei_generic(ctx, z):
    # Note: the following is currently untested because mp and fp
    # both use special-case ei code
    if z == ctx.inf:
        return z
    if z == ctx.ninf:
        return ctx.zero
    if ctx.mag(z) > 1:
        try:
            r = ctx.one/z
            v = ctx.exp(z)*ctx.hyper([1,1],[],r,
                maxterms=ctx.prec, force_series=True)/z
            im = ctx._im(z)
            if im > 0:
                v += ctx.pi*ctx.j
            if im < 0:
                v -= ctx.pi*ctx.j
            return v
        except ctx.NoConvergence:
            pass
    v = z*ctx.hyp2f2(1,1,2,2,z) + ctx.euler
    if ctx._im(z):
        v += 0.5*(ctx.log(z) - ctx.log(ctx.one/z))
    else:
        v += ctx.log(abs(z))
    return v

@defun
def e1(ctx, z):
    try:
        return ctx._e1(z)
    except NotImplementedError:
        return ctx.expint(1, z)

@defun
def ci(ctx, z):
    try:
        return ctx._ci(z)
    except NotImplementedError:
        return ctx._ci_generic(z)

@defun_wrapped
def _ci_generic(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf: return ctx.zero
        if z == ctx.ninf: return ctx.pi*1j
    jz = ctx.fmul(ctx.j,z,exact=True)
    njz = ctx.fneg(jz,exact=True)
    v = 0.5*(ctx.ei(jz) + ctx.ei(njz))
    zreal = ctx._re(z)
    zimag = ctx._im(z)
    if zreal == 0:
        if zimag > 0: v += ctx.pi*0.5j
        if zimag < 0: v -= ctx.pi*0.5j
    if zreal < 0:
        if zimag >= 0: v += ctx.pi*1j
        if zimag <  0: v -= ctx.pi*1j
    if ctx._is_real_type(z) and zreal > 0:
        v = ctx._re(v)
    return v

@defun
def si(ctx, z):
    try:
        return ctx._si(z)
    except NotImplementedError:
        return ctx._si_generic(z)

@defun_wrapped
def _si_generic(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf: return 0.5*ctx.pi
        if z == ctx.ninf: return -0.5*ctx.pi
    # Suffers from cancellation near 0
    if ctx.mag(z) >= -1:
        jz = ctx.fmul(ctx.j,z,exact=True)
        njz = ctx.fneg(jz,exact=True)
        v = (-0.5j)*(ctx.ei(jz) - ctx.ei(njz))
        zreal = ctx._re(z)
        if zreal > 0:
            v -= 0.5*ctx.pi
        if zreal < 0:
            v += 0.5*ctx.pi
        if ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    else:
        return z*ctx.hyp1f2((1,2),(3,2),(3,2),-0.25*z*z)

@defun_wrapped
def chi(ctx, z):
    nz = ctx.fneg(z, exact=True)
    v = 0.5*(ctx.ei(z) + ctx.ei(nz))
    zreal = ctx._re(z)
    zimag = ctx._im(z)
    if zimag > 0:
        v += ctx.pi*0.5j
    elif zimag < 0:
        v -= ctx.pi*0.5j
    elif zreal < 0:
        v += ctx.pi*1j
    return v

@defun_wrapped
def shi(ctx, z):
    # Suffers from cancellation near 0
    if ctx.mag(z) >= -1:
        nz = ctx.fneg(z, exact=True)
        v = 0.5*(ctx.ei(z) - ctx.ei(nz))
        zimag = ctx._im(z)
        if zimag > 0: v -= 0.5j*ctx.pi
        if zimag < 0: v += 0.5j*ctx.pi
        return v
    else:
        return z * ctx.hyp1f2((1,2),(3,2),(3,2),0.25*z*z)

@defun_wrapped
def fresnels(ctx, z):
    if z == ctx.inf:
        return ctx.mpf(0.5)
    if z == ctx.ninf:
        return ctx.mpf(-0.5)
    return ctx.pi*z**3/6*ctx.hyp1f2((3,4),(3,2),(7,4),-ctx.pi**2*z**4/16)

@defun_wrapped
def fresnelc(ctx, z):
    if z == ctx.inf:
        return ctx.mpf(0.5)
    if z == ctx.ninf:
        return ctx.mpf(-0.5)
    return z*ctx.hyp1f2((1,4),(1,2),(5,4),-ctx.pi**2*z**4/16)
<|endoftext|>from . import functions
# Hack to update methods
from . import factorials
from . import hypergeometric
from . import expintegrals
from . import bessel
from . import orthogonal
from . import theta
from . import elliptic
from . import signals
from . import zeta
from . import rszeta
from . import zetazeros
from . import qfunctions
<|endoftext|>from __future__ import print_function

from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static

@defun
def stieltjes(ctx, n, a=1):
    n = ctx.convert(n)
    a = ctx.convert(a)
    if n < 0:
        return ctx.bad_domain("Stieltjes constants defined for n >= 0")
    if hasattr(ctx, "stieltjes_cache"):
        stieltjes_cache = ctx.stieltjes_cache
    else:
        stieltjes_cache = ctx.stieltjes_cache = {}
    if a == 1:
        if n == 0:
            return +ctx.euler
        if n in stieltjes_cache:
            prec, s = stieltjes_cache[n]
            if prec >= ctx.prec:
                return +s
    mag = 1
    def f(x):
        xa = x/a
        v = (xa-ctx.j)*ctx.ln(a-ctx.j*x)**n/(1+xa**2)/(ctx.exp(2*ctx.pi*x)-1)
        return ctx._re(v) / mag
    orig = ctx.prec
    try:
        # Normalize integrand by approx. magnitude to
        # speed up quadrature (which uses absolute error)
        if n > 50:
            ctx.prec = 20
            mag = ctx.quad(f, [0,ctx.inf], maxdegree=3)
        ctx.prec = orig + 10 + int(n**0.5)
        s = ctx.quad(f, [0,ctx.inf], maxdegree=20)
        v = ctx.ln(a)**n/(2*a) - ctx.ln(a)**(n+1)/(n+1) + 2*s/a*mag
    finally:
        ctx.prec = orig
    if a == 1 and ctx.isint(n):
        stieltjes_cache[n] = (ctx.prec, v)
    return +v

@defun_wrapped
def siegeltheta(ctx, t, derivative=0):
    d = int(derivative)
    if  (t == ctx.inf or t == ctx.ninf):
        if d < 2:
            if t == ctx.ninf and d == 0:
                return ctx.ninf
            return ctx.inf
        else:
            return ctx.zero
    if d == 0:
        if ctx._im(t):
            # XXX: cancellation occurs
            a = ctx.loggamma(0.25+0.5j*t)
            b = ctx.loggamma(0.25-0.5j*t)
            return -ctx.ln(ctx.pi)/2*t - 0.5j*(a-b)
        else:
            if ctx.isinf(t):
                return t
            return ctx._im(ctx.loggamma(0.25+0.5j*t)) - ctx.ln(ctx.pi)/2*t
    if d > 0:
        a = (-0.5j)**(d-1)*ctx.polygamma(d-1, 0.25-0.5j*t)
        b = (0.5j)**(d-1)*ctx.polygamma(d-1, 0.25+0.5j*t)
        if ctx._im(t):
            if d == 1:
                return -0.5*ctx.log(ctx.pi)+0.25*(a+b)
            else:
                return 0.25*(a+b)
        else:
            if d == 1:
                return ctx._re(-0.5*ctx.log(ctx.pi)+0.25*(a+b))
            else:
                return ctx._re(0.25*(a+b))

@defun_wrapped
def grampoint(ctx, n):
    # asymptotic expansion, from
    # http://mathworld.wolfram.com/GramPoint.html
    g = 2*ctx.pi*ctx.exp(1+ctx.lambertw((8*n+1)/(8*ctx.e)))
    return ctx.findroot(lambda t: ctx.siegeltheta(t)-ctx.pi*n, g)


@defun_wrapped
def siegelz(ctx, t, **kwargs):
    d = int(kwargs.get("derivative", 0))
    t = ctx.convert(t)
    t1 = ctx._re(t)
    t2 = ctx._im(t)
    prec = ctx.prec
    try:
        if abs(t1) > 500*prec and t2**2 < t1:
            v = ctx.rs_z(t, d)
            if ctx._is_real_type(t):
                return ctx._re(v)
            return v
    except NotImplementedError:
        pass
    ctx.prec += 21
    e1 = ctx.expj(ctx.siegeltheta(t))
    z = ctx.zeta(0.5+ctx.j*t)
    if d == 0:
        v = e1*z
        ctx.prec=prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z1 = ctx.zeta(0.5+ctx.j*t, derivative=1)
    theta1 = ctx.siegeltheta(t, derivative=1)
    if d == 1:
        v =  ctx.j*e1*(z1+z*theta1)
        ctx.prec=prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z2 = ctx.zeta(0.5+ctx.j*t, derivative=2)
    theta2 = ctx.siegeltheta(t, derivative=2)
    comb1 = theta1**2-ctx.j*theta2
    if d == 2:
        def terms():
            return [2*z1*theta1, z2, z*comb1]
        v = ctx.sum_accurately(terms, 1)
        v =  -e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    ctx.prec += 10
    z3 = ctx.zeta(0.5+ctx.j*t, derivative=3)
    theta3 = ctx.siegeltheta(t, derivative=3)
    comb2 = theta1**3-3*ctx.j*theta1*theta2-theta3
    if d == 3:
        def terms():
            return  [3*theta1*z2, 3*z1*comb1, z3+z*comb2]
        v = ctx.sum_accurately(terms, 1)
        v =  -ctx.j*e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z4 = ctx.zeta(0.5+ctx.j*t, derivative=4)
    theta4 = ctx.siegeltheta(t, derivative=4)
    def terms():
        return [theta1**4, -6*ctx.j*theta1**2*theta2, -3*theta2**2,
            -4*theta1*theta3, ctx.j*theta4]
    comb3 = ctx.sum_accurately(terms, 1)
    if d == 4:
        def terms():
            return  [6*theta1**2*z2, -6*ctx.j*z2*theta2, 4*theta1*z3,
                 4*z1*comb2, z4, z*comb3]
        v = ctx.sum_accurately(terms, 1)
        v =  e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    if d > 4:
        h = lambda x: ctx.siegelz(x, derivative=4)
        return ctx.diff(h, t, n=d-4)


_zeta_zeros = [
14.134725142,21.022039639,25.010857580,30.424876126,32.935061588,
37.586178159,40.918719012,43.327073281,48.005150881,49.773832478,
52.970321478,56.446247697,59.347044003,60.831778525,65.112544048,
67.079810529,69.546401711,72.067157674,75.704690699,77.144840069,
79.337375020,82.910380854,84.735492981,87.425274613,88.809111208,
92.491899271,94.651344041,95.870634228,98.831194218,101.317851006,
103.725538040,105.446623052,107.168611184,111.029535543,111.874659177,
114.320220915,116.226680321,118.790782866,121.370125002,122.946829294,
124.256818554,127.516683880,129.578704200,131.087688531,133.497737203,
134.756509753,138.116042055,139.736208952,141.123707404,143.111845808,
146.000982487,147.422765343,150.053520421,150.925257612,153.024693811,
156.112909294,157.597591818,158.849988171,161.188964138,163.030709687,
165.537069188,167.184439978,169.094515416,169.911976479,173.411536520,
174.754191523,176.441434298,178.377407776,179.916484020,182.207078484,
184.874467848,185.598783678,187.228922584,189.416158656,192.026656361,
193.079726604,195.265396680,196.876481841,198.015309676,201.264751944,
202.493594514,204.189671803,205.394697202,207.906258888,209.576509717,
211.690862595,213.347919360,214.547044783,216.169538508,219.067596349,
220.714918839,221.430705555,224.007000255,224.983324670,227.421444280,
229.337413306,231.250188700,231.987235253,233.693404179,236.524229666,
]

def _load_zeta_zeros(url):
    import urllib
    d = urllib.urlopen(url)
    L = [float(x) for x in d.readlines()]
    # Sanity check
    assert round(L[0]) == 14
    _zeta_zeros[:] = L

@defun
def oldzetazero(ctx, n, url='http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1'):
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError("n must be nonzero")
    if n > len(_zeta_zeros) and n <= 100000:
        _load_zeta_zeros(url)
    if n > len(_zeta_zeros):
        raise NotImplementedError("n too large for zetazeros")
    return ctx.mpc(0.5, ctx.findroot(ctx.siegelz, _zeta_zeros[n-1]))

@defun_wrapped
def riemannr(ctx, x):
    if x == 0:
        return ctx.zero
    # Check if a simple asymptotic estimate is accurate enough
    if abs(x) > 1000:
        a = ctx.li(x)
        b = 0.5*ctx.li(ctx.sqrt(x))
        if abs(b) < abs(a)*ctx.eps:
            return a
    if abs(x) < 0.01:
        # XXX
        ctx.prec += int(-ctx.log(abs(x),2))
    # Sum Gram's series
    s = t = ctx.one
    u = ctx.ln(x)
    k = 1
    while abs(t) > abs(s)*ctx.eps:
        t = t * u / k
        s += t / (k * ctx._zeta_int(k+1))
        k += 1
    return s

@defun_static
def primepi(ctx, x):
    x = int(x)
    if x < 2:
        return 0
    return len(ctx.list_primes(x))

# TODO: fix the interface wrt contexts
@defun_wrapped
def primepi2(ctx, x):
    x = int(x)
    if x < 2:
        return ctx._iv.zero
    if x < 2657:
        return ctx._iv.mpf(ctx.primepi(x))
    mid = ctx.li(x)
    # Schoenfeld's estimate for x >= 2657, assuming RH
    err = ctx.sqrt(x,rounding='u')*ctx.ln(x,rounding='u')/8/ctx.pi(rounding='d')
    a = ctx.floor((ctx._iv.mpf(mid)-err).a, rounding='d')
    b = ctx.ceil((ctx._iv.mpf(mid)+err).b, rounding='u')
    return ctx._iv.mpf([a,b])

@defun_wrapped
def primezeta(ctx, s):
    if ctx.isnan(s):
        return s
    if ctx.re(s) <= 0:
        raise ValueError("prime zeta function defined only for re(s) > 0")
    if s == 1:
        return ctx.inf
    if s == 0.5:
        return ctx.mpc(ctx.ninf, ctx.pi)
    r = ctx.re(s)
    if r > ctx.prec:
        return 0.5**s
    else:
        wp = ctx.prec + int(r)
        def terms():
            orig = ctx.prec
            # zeta ~ 1+eps; need to set precision
            # to get logarithm accurately
            k = 0
            while 1:
                k += 1
                u = ctx.moebius(k)
                if not u:
                    continue
                ctx.prec = wp
                t = u*ctx.ln(ctx.zeta(k*s))/k
                if not t:
                    return
                #print ctx.prec, ctx.nstr(t)
                ctx.prec = orig
                yield t
    return ctx.sum_accurately(terms)

# TODO: for bernpoly and eulerpoly, ensure that all exact zeros are covered

@defun_wrapped
def bernpoly(ctx, n, z):
    # Slow implementation:
    #return sum(ctx.binomial(n,k)*ctx.bernoulli(k)*z**(n-k) for k in xrange(0,n+1))
    n = int(n)
    if n < 0:
        raise ValueError("Bernoulli polynomials only defined for n >= 0")
    if z == 0 or (z == 1 and n > 1):
        return ctx.bernoulli(n)
    if z == 0.5:
        return (ctx.ldexp(1,1-n)-1)*ctx.bernoulli(n)
    if n <= 3:
        if n == 0: return z ** 0
        if n == 1: return z - 0.5
        if n == 2: return (6*z*(z-1)+1)/6
        if n == 3: return z*(z*(z-1.5)+0.5)
    if ctx.isinf(z):
        return z ** n
    if ctx.isnan(z):
        return z
    if abs(z) > 2:
        def terms():
            t = ctx.one
            yield t
            r = ctx.one/z
            k = 1
            while k <= n:
                t = t*(n+1-k)/k*r
                if not (k > 2 and k & 1):
                    yield t*ctx.bernoulli(k)
                k += 1
        return ctx.sum_accurately(terms) * z**n
    else:
        def terms():
            yield ctx.bernoulli(n)
            t = ctx.one
            k = 1
            while k <= n:
                t = t*(n+1-k)/k * z
                m = n-k
                if not (m > 2 and m & 1):
                    yield t*ctx.bernoulli(m)
                k += 1
        return ctx.sum_accurately(terms)

@defun_wrapped
def eulerpoly(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError("Euler polynomials only defined for n >= 0")
    if n <= 2:
        if n == 0: return z ** 0
        if n == 1: return z - 0.5
        if n == 2: return z*(z-1)
    if ctx.isinf(z):
        return z**n
    if ctx.isnan(z):
        return z
    m = n+1
    if z == 0:
        return -2*(ctx.ldexp(1,m)-1)*ctx.bernoulli(m)/m * z**0
    if z == 1:
        return 2*(ctx.ldexp(1,m)-1)*ctx.bernoulli(m)/m * z**0
    if z == 0.5:
        if n % 2:
            return ctx.zero
        # Use exact code for Euler numbers
        if n < 100 or n*ctx.mag(0.46839865*n) < ctx.prec*0.25:
            return ctx.ldexp(ctx._eulernum(n), -n)
    # http://functions.wolfram.com/Polynomials/EulerE2/06/01/02/01/0002/
    def terms():
        t = ctx.one
        k = 0
        w = ctx.ldexp(1,n+2)
        while 1:
            v = n-k+1
            if not (v > 2 and v & 1):
                yield (2-w)*ctx.bernoulli(v)*t
            k += 1
            if k > n:
                break
            t = t*z*(n-k+2)/k
            w *= 0.5
    return ctx.sum_accurately(terms) / m

@defun
def eulernum(ctx, n, exact=False):
    n = int(n)
    if exact:
        return int(ctx._eulernum(n))
    if n < 100:
        return ctx.mpf(ctx._eulernum(n))
    if n % 2:
        return ctx.zero
    return ctx.ldexp(ctx.eulerpoly(n,0.5), n)

# TODO: this should be implemented low-level
def polylog_series(ctx, s, z):
    tol = +ctx.eps
    l = ctx.zero
    k = 1
    zk = z
    while 1:
        term = zk / k**s
        l += term
        if abs(term) < tol:
            break
        zk *= z
        k += 1
    return l

def polylog_continuation(ctx, n, z):
    if n < 0:
        return z*0
    twopij = 2j * ctx.pi
    a = -twopij**n/ctx.fac(n) * ctx.bernpoly(n, ctx.ln(z)/twopij)
    if ctx._is_real_type(z) and z < 0:
        a = ctx._re(a)
    if ctx._im(z) < 0 or (ctx._im(z) == 0 and ctx._re(z) >= 1):
        a -= twopij*ctx.ln(z)**(n-1)/ctx.fac(n-1)
    return a

def polylog_unitcircle(ctx, n, z):
    tol = +ctx.eps
    if n > 1:
        l = ctx.zero
        logz = ctx.ln(z)
        logmz = ctx.one
        m = 0
        while 1:
            if (n-m) != 1:
                term = ctx.zeta(n-m) * logmz / ctx.fac(m)
                if term and abs(term) < tol:
                    break
                l += term
            logmz *= logz
            m += 1
        l += ctx.ln(z)**(n-1)/ctx.fac(n-1)*(ctx.harmonic(n-1)-ctx.ln(-ctx.ln(z)))
    elif n < 1:  # else
        l = ctx.fac(-n)*(-ctx.ln(z))**(n-1)
        logz = ctx.ln(z)
        logkz = ctx.one
        k = 0
        while 1:
            b = ctx.bernoulli(k-n+1)
            if b:
                term = b*logkz/(ctx.fac(k)*(k-n+1))
                if abs(term) < tol:
                    break
                l -= term
            logkz *= logz
            k += 1
    else:
        raise ValueError
    if ctx._is_real_type(z) and z < 0:
        l = ctx._re(l)
    return l

def polylog_general(ctx, s, z):
    v = ctx.zero
    u = ctx.ln(z)
    if not abs(u) < 5: # theoretically |u| < 2*pi
        j = ctx.j
        v = 1-s
        y = ctx.ln(-z)/(2*ctx.pi*j)
        return ctx.gamma(v)*(j**v*ctx.zeta(v,0.5+y) + j**-v*ctx.zeta(v,0.5-y))/(2*ctx.pi)**v
    t = 1
    k = 0
    while 1:
        term = ctx.zeta(s-k) * t
        if abs(term) < ctx.eps:
            break
        v += term
        k += 1
        t *= u
        t /= k
    return ctx.gamma(1-s)*(-u)**(s-1) + v

@defun_wrapped
def polylog(ctx, s, z):
    s = ctx.convert(s)
    z = ctx.convert(z)
    if z == 1:
        return ctx.zeta(s)
    if z == -1:
        return -ctx.altzeta(s)
    if s == 0:
        return z/(1-z)
    if s == 1:
        return -ctx.ln(1-z)
    if s == -1:
        return z/(1-z)**2
    if abs(z) <= 0.75 or (not ctx.isint(s) and abs(z) < 0.9):
        return polylog_series(ctx, s, z)
    if abs(z) >= 1.4 and ctx.isint(s):
        return (-1)**(s+1)*polylog_series(ctx, s, 1/z) + polylog_continuation(ctx, int(ctx.re(s)), z)
    if ctx.isint(s):
        return polylog_unitcircle(ctx, int(ctx.re(s)), z)
    return polylog_general(ctx, s, z)

@defun_wrapped
def clsin(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and int(s) % 2 == 1:
        return z*0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.im(ctx.polylog(s,a))
    b = 1/a
    return (-0.5j)*(ctx.polylog(s,a) - ctx.polylog(s,b))

@defun_wrapped
def clcos(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and int(s) % 2 == 0:
        return z*0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.re(ctx.polylog(s,a))
    b = 1/a
    return 0.5*(ctx.polylog(s,a) + ctx.polylog(s,b))

@defun
def altzeta(ctx, s, **kwargs):
    try:
        return ctx._altzeta(s, **kwargs)
    except NotImplementedError:
        return ctx._altzeta_generic(s)

@defun_wrapped
def _altzeta_generic(ctx, s):
    if s == 1:
        return ctx.ln2 + 0*s
    return -ctx.powm1(2, 1-s) * ctx.zeta(s)

@defun
def zeta(ctx, s, a=1, derivative=0, method=None, **kwargs):
    d = int(derivative)
    if a == 1 and not (d or method):
        try:
            return ctx._zeta(s, **kwargs)
        except NotImplementedError:
            pass
    s = ctx.convert(s)
    prec = ctx.prec
    method = kwargs.get('method')
    verbose = kwargs.get('verbose')
    if (not s) and (not derivative):
        return ctx.mpf(0.5) - ctx._convert_param(a)[0]
    if a == 1 and method != 'euler-maclaurin':
        im = abs(ctx._im(s))
        re = abs(ctx._re(s))
        #if (im < prec or method == 'borwein') and not derivative:
        #    try:
        #        if verbose:
        #            print "zeta: Attempting to use the Borwein algorithm"
        #        return ctx._zeta(s, **kwargs)
        #    except NotImplementedError:
        #        if verbose:
        #            print "zeta: Could not use the Borwein algorithm"
        #        pass
        if abs(im) > 500*prec and 10*re < prec and derivative <= 4 or \
            method == 'riemann-siegel':
            try:   #  py2.4 compatible try block
                try:
                    if verbose:
                        print("zeta: Attempting to use the Riemann-Siegel algorithm")
                    return ctx.rs_zeta(s, derivative, **kwargs)
                except NotImplementedError:
                    if verbose:
                        print("zeta: Could not use the Riemann-Siegel algorithm")
                    pass
            finally:
                ctx.prec = prec
    if s == 1:
        return ctx.inf
    abss = abs(s)
    if abss == ctx.inf:
        if ctx.re(s) == ctx.inf:
            if d == 0:
                return ctx.one
            return ctx.zero
        return s*0
    elif ctx.isnan(abss):
        return 1/s
    if ctx.re(s) > 2*ctx.prec and a == 1 and not derivative:
        return ctx.one + ctx.power(2, -s)
    return +ctx._hurwitz(s, a, d, **kwargs)

@defun
def _hurwitz(ctx, s, a=1, d=0, **kwargs):
    prec = ctx.prec
    verbose = kwargs.get('verbose')
    try:
        extraprec = 10
        ctx.prec += extraprec
        # We strongly want to special-case rational a
        a, atype = ctx._convert_param(a)
        if ctx.re(s) < 0:
            if verbose:
                print("zeta: Attempting reflection formula")
            try:
                return _hurwitz_reflection(ctx, s, a, d, atype)
            except NotImplementedError:
                pass
            if verbose:
                print("zeta: Reflection formula failed")
        if verbose:
            print("zeta: Using the Euler-Maclaurin algorithm")
        while 1:
            ctx.prec = prec + extraprec
            T1, T2 = _hurwitz_em(ctx, s, a, d, prec+10, verbose)
            cancellation = ctx.mag(T1) - ctx.mag(T1+T2)
            if verbose:
                print("Term 1:", T1)
                print("Term 2:", T2)
                print("Cancellation:", cancellation, "bits")
            if cancellation < extraprec:
                return T1 + T2
            else:
                extraprec = max(2*extraprec, min(cancellation + 5, 100*prec))
                if extraprec > kwargs.get('maxprec', 100*prec):
                    raise ctx.NoConvergence("zeta: too much cancellation")
    finally:
        ctx.prec = prec

def _hurwitz_reflection(ctx, s, a, d, atype):
    # TODO: implement for derivatives
    if d != 0:
        raise NotImplementedError
    res = ctx.re(s)
    negs = -s
    # Integer reflection formula
    if ctx.isnpint(s):
        n = int(res)
        if n <= 0:
            return ctx.bernpoly(1-n, a) / (n-1)
    if not (atype == 'Q' or atype == 'Z'):
        raise NotImplementedError
    t = 1-s
    # We now require a to be standardized
    v = 0
    shift = 0
    b = a
    while ctx.re(b) > 1:
        b -= 1
        v -= b**negs
        shift -= 1
    while ctx.re(b) <= 0:
        v += b**negs
        b += 1
        shift += 1
    # Rational reflection formula
    try:
        p, q = a._mpq_
    except:
        assert a == int(a)
        p = int(a)
        q = 1
    p += shift*q
    assert 1 <= p <= q
    g = ctx.fsum(ctx.cospi(t/2-2*k*b)*ctx._hurwitz(t,(k,q)) \
        for k in range(1,q+1))
    g *= 2*ctx.gamma(t)/(2*ctx.pi*q)**t
    v += g
    return v

def _hurwitz_em(ctx, s, a, d, prec, verbose):
    # May not be converted at this point
    a = ctx.convert(a)
    tol = -prec
    # Estimate number of terms for Euler-Maclaurin summation; could be improved
    M1 = 0
    M2 = prec // 3
    N = M2
    lsum = 0
    # This speeds up the recurrence for derivatives
    if ctx.isint(s):
        s = int(ctx._re(s))
    s1 = s-1
    while 1:
        # Truncated L-series
        l = ctx._zetasum(s, M1+a, M2-M1-1, [d])[0][0]
        #if d:
        #    l = ctx.fsum((-ctx.ln(n+a))**d * (n+a)**negs for n in range(M1,M2))
        #else:
        #    l = ctx.fsum((n+a)**negs for n in range(M1,M2))
        lsum += l
        M2a = M2+a
        logM2a = ctx.ln(M2a)
        logM2ad = logM2a**d
        logs = [logM2ad]
        logr = 1/logM2a
        rM2a = 1/M2a
        M2as = M2a**(-s)
        if d:
            tailsum = ctx.gammainc(d+1, s1*logM2a) / s1**(d+1)
        else:
            tailsum = 1/((s1)*(M2a)**s1)
        tailsum += 0.5 * logM2ad * M2as
        U = [1]
        r = M2as
        fact = 2
        for j in range(1, N+1):
            # TODO: the following could perhaps be tidied a bit
            j2 = 2*j
            if j == 1:
                upds = [1]
            else:
                upds = [j2-2, j2-1]
            for m in upds:
                D = min(m,d+1)
                if m <= d:
                    logs.append(logs[-1] * logr)
                Un = [0]*(D+1)
                for i in xrange(D): Un[i] = (1-m-s)*U[i]
                for i in xrange(1,D+1): Un[i] += (d-(i-1))*U[i-1]
                U = Un
                r *= rM2a
            t = ctx.fdot(U, logs) * r * ctx.bernoulli(j2)/(-fact)
            tailsum += t
            if ctx.mag(t) < tol:
                return lsum, (-1)**d * tailsum
            fact *= (j2+1)*(j2+2)
        if verbose:
            print("Sum range:", M1, M2, "term magnitude", ctx.mag(t), "tolerance", tol)
        M1, M2 = M2, M2*2
        if ctx.re(s) < 0:
            N += N//2



@defun
def _zetasum(ctx, s, a, n, derivatives=[0], reflect=False):
    """
    Returns [xd0,xd1,...,xdr], [yd0,yd1,...ydr] where

    xdk = D^k     ( 1/a^s     +  1/(a+1)^s      +  ...  +  1/(a+n)^s     )
    ydk = D^k conj( 1/a^(1-s) +  1/(a+1)^(1-s)  +  ...  +  1/(a+n)^(1-s) )

    D^k = kth derivative with respect to s, k ranges over the given list of
    derivatives (which should consist of either a single element
    or a range 0,1,...r). If reflect=False, the ydks are not computed.
    """
    #print "zetasum", s, a, n
    # don't use the fixed-point code if there are large exponentials
    if abs(ctx.re(s)) < 0.5 * ctx.prec:
        try:
            return ctx._zetasum_fast(s, a, n, derivatives, reflect)
        except NotImplementedError:
            pass
    negs = ctx.fneg(s, exact=True)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1
    if not reflect:
        if not have_derivatives:
            return [ctx.fsum((a+k)**negs for k in xrange(n+1))], []
        if have_one_derivative:
            d = derivatives[0]
            x = ctx.fsum(ctx.ln(a+k)**d * (a+k)**negs for k in xrange(n+1))
            return [(-1)**d * x], []
    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd+1)
    xs = [ctx.zero for d in derivatives]
    if reflect:
        ys = [ctx.zero for d in derivatives]
    else:
        ys = []
    for k in xrange(n+1):
        w = a + k
        xterm = w ** negs
        if reflect:
            yterm = ctx.conj(ctx.one / (w * xterm))
        if have_derivatives:
            logw = -ctx.ln(w)
            if have_one_derivative:
                logw = logw ** maxd
                xs[0] += xterm * logw
                if reflect:
                    ys[0] += yterm * logw
            else:
                t = ctx.one
                for d in derivatives:
                    xs[d] += xterm * t
                    if reflect:
                        ys[d] += yterm * t
                    t *= logw
        else:
            xs[0] += xterm
            if reflect:
                ys[0] += yterm
    return xs, ys

@defun
def dirichlet(ctx, s, chi=[1], derivative=0):
    s = ctx.convert(s)
    q = len(chi)
    d = int(derivative)
    if d > 2:
        raise NotImplementedError("arbitrary order derivatives")
    prec = ctx.prec
    try:
        ctx.prec += 10
        if s == 1:
            have_pole = True
            for x in chi:
                if x and x != 1:
                    have_pole = False
                    h = +ctx.eps
                    ctx.prec *= 2*(d+1)
                    s += h
            if have_pole:
                return +ctx.inf
        z = ctx.zero
        for p in range(1,q+1):
            if chi[p%q]:
                if d == 1:
                    z += chi[p%q] * (ctx.zeta(s, (p,q), 1) - \
                        ctx.zeta(s, (p,q))*ctx.log(q))
                else:
                    z += chi[p%q] * ctx.zeta(s, (p,q))
        z /= q**s
    finally:
        ctx.prec = prec
    return +z


def secondzeta_main_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5*s, a*gamm**2, regularized=True)*gamm**(-s)
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        gamm = ctx.im(ctx.zetazero_memoized(n))
        term = f(n)
        mg = abs(term)
    err = 0
    if kwargs.get("error"):
        sg = ctx.re(s)
        err = 0.5*ctx.pi**(-1)*max(1,sg)*a**(sg-0.5)*ctx.log(gamm/(2*ctx.pi))*\
             ctx.gammainc(-0.5, a*gamm**2)/abs(ctx.gamma(s/2))
        err = abs(err)
    return +totsum, err, n

def secondzeta_prime_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5*(1-s),0.25*ctx.log(n)**2 * a**(-1))*\
        ((0.5*ctx.log(n))**(s-1))*ctx.mangoldt(n)/ctx.sqrt(n)/\
        (2*ctx.gamma(0.5*s)*ctx.sqrt(ctx.pi))
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 1
    while mg > tol or n < 9:
        totsum += term
        n += 1
        term = f(n)
        if term == 0:
            mg = ctx.inf
        else:
            mg = abs(term)
    if kwargs.get("error"):
        err = mg
    return +totsum, err, n

def secondzeta_exp_term(ctx, s, a):
    if ctx.isint(s) and ctx.re(s) <= 0:
        m = int(round(ctx.re(s)))
        if not m & 1:
            return ctx.mpf('-0.25')**(-m//2)
    tol = ctx.eps
    f = lambda n: (0.25*a)**n/((n+0.5*s)*ctx.fac(n))
    totsum = ctx.zero
    term = f(0)
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        term = f(n)
        mg = abs(term)
    v = a**(0.5*s)*totsum/ctx.gamma(0.5*s)
    return v

def secondzeta_singular_term(ctx, s, a, **kwargs):
    factor = a**(0.5*(s-1))/(4*ctx.sqrt(ctx.pi)*ctx.gamma(0.5*s))
    extraprec = ctx.mag(factor)
    ctx.prec += extraprec
    factor = a**(0.5*(s-1))/(4*ctx.sqrt(ctx.pi)*ctx.gamma(0.5*s))
    tol = ctx.eps
    f = lambda n: ctx.bernpoly(n,0.75)*(4*ctx.sqrt(a))**n*\
       ctx.gamma(0.5*n)/((s+n-1)*ctx.fac(n))
    totsum = ctx.zero
    mg1 = ctx.inf
    n = 1
    term = f(n)
    mg2 = abs(term)
    while mg2 > tol and mg2 <= mg1:
        totsum += term
        n += 1
        term = f(n)
        totsum += term
        n +=1
        term = f(n)
        mg1 = mg2
        mg2 = abs(term)
    totsum += term
    pole = -2*(s-1)**(-2)+(ctx.euler+ctx.log(16*ctx.pi**2*a))*(s-1)**(-1)
    st = factor*(pole+totsum)
    err = 0
    if kwargs.get("error"):
        if not ((mg2 > tol) and (mg2 <= mg1)):
            if mg2 <= tol:
                err = ctx.mpf(10)**int(ctx.log(abs(factor*tol),10))
            if mg2 > mg1:
                err = ctx.mpf(10)**int(ctx.log(abs(factor*mg1),10))
        err = max(err, ctx.eps*1.)
    ctx.prec -= extraprec
    return +st, err

@defun
def secondzeta(ctx, s, a = 0.015, **kwargs):
    r"""
    Evaluates the secondary zeta function `Z(s)`, defined for
    `\mathrm{Re}(s)>1` by

    .. math ::

        Z(s) = \sum_{n=1}^{\infty} \frac{1}{\tau_n^s}

    where `\frac12+i\tau_n` runs through the zeros of `\zeta(s)` with
    imaginary part positive.

    `Z(s)` extends to a meromorphic function on `\mathbb{C}`  with a
    double pole at `s=1` and  simple poles at the points `-2n` for
    `n=0`,  1, 2, ...

    **Examples**

        >>> from mpmath import *
        >>> mp.pretty = True; mp.dps = 15
        >>> secondzeta(2)
        0.023104993115419
        >>> xi = lambda s: 0.5*s*(s-1)*pi**(-0.5*s)*gamma(0.5*s)*zeta(s)
        >>> Xi = lambda t: xi(0.5+t*j)
        >>> chop(-0.5*diff(Xi,0,n=2)/Xi(0))
        0.023104993115419

    We may ask for an approximate error value::

        >>> secondzeta(0.5+100j, error=True)
        ((-0.216272011276718 - 0.844952708937228j), 2.22044604925031e-16)

    The function has poles at the negative odd integers,
    and dyadic rational values at the negative even integers::

        >>> mp.dps = 30
        >>> secondzeta(-8)
        -0.67236328125
        >>> secondzeta(-7)
        +inf

    **Implementation notes**

    The function is computed as sum of four terms `Z(s)=A(s)-P(s)+E(s)-S(s)`
    respectively main, prime, exponential and singular terms.
    The main term `A(s)` is computed from the zeros of zeta.
    The prime term depends on the von Mangoldt function.
    The singular term is responsible for the poles of the function.

    The four terms depends on a small parameter `a`. We may change the
    value of `a`. Theoretically this has no effect on the sum of the four
    terms, but in practice may be important.

    A smaller value of the parameter `a` makes `A(s)` depend on
    a smaller number of zeros of zeta, but `P(s)`  uses more values of
    von Mangoldt function.

    We may also add a verbose option to obtain data about the
    values of the four terms.

        >>> mp.dps = 10
        >>> secondzeta(0.5 + 40j, error=True, verbose=True)
        main term = (-30190318549.138656312556 - 13964804384.624622876523j)
            computed using 19 zeros of zeta
        prime term = (132717176.89212754625045 + 188980555.17563978290601j)
            computed using 9 values of the von Mangoldt function
        exponential term = (542447428666.07179812536 + 362434922978.80192435203j)
        singular term = (512124392939.98154322355 + 348281138038.65531023921j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

        >>> secondzeta(0.5 + 40j, a=0.04, error=True, verbose=True)
        main term = (-151962888.19606243907725 - 217930683.90210294051982j)
            computed using 9 zeros of zeta
        prime term = (2476659342.3038722372461 + 28711581821.921627163136j)
            computed using 37 values of the von Mangoldt function
        exponential term = (178506047114.7838188264 + 819674143244.45677330576j)
        singular term = (175877424884.22441310708 + 790744630738.28669174871j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

    Notice the great cancellation between the four terms. Changing `a`, the
    four terms are very different numbers but the cancellation gives
    the good value of Z(s).

    **References**

    A. Voros, Zeta functions for the Riemann zeros, Ann. Institute Fourier,
    53, (2003) 665--699.

    A. Voros, Zeta functions over Zeros of Zeta Functions, Lecture Notes
    of the Unione Matematica Italiana, Springer, 2009.
    """
    s = ctx.convert(s)
    a = ctx.convert(a)
    tol = ctx.eps
    if ctx.isint(s) and ctx.re(s) <= 1:
        if abs(s-1) < tol*1000:
            return ctx.inf
        m = int(round(ctx.re(s)))
        if m & 1:
            return ctx.inf
        else:
            return ((-1)**(-m//2)*\
                   ctx.fraction(8-ctx.eulernum(-m,exact=True),2**(-m+3)))
    prec = ctx.prec
    try:
        t3 = secondzeta_exp_term(ctx, s, a)
        extraprec = max(ctx.mag(t3),0)
        ctx.prec += extraprec + 3
        t1, r1, gt = secondzeta_main_term(ctx,s,a,error='True', verbose='True')
        t2, r2, pt = secondzeta_prime_term(ctx,s,a,error='True', verbose='True')
        t4, r4 = secondzeta_singular_term(ctx,s,a,error='True')
        t3 = secondzeta_exp_term(ctx, s, a)
        err = r1+r2+r4
        t = t1-t2+t3-t4
        if kwargs.get("verbose"):
            print('main term =', t1)
            print('    computed using', gt, 'zeros of zeta')
            print('prime term =', t2)
            print('    computed using', pt, 'values of the von Mangoldt function')
            print('exponential term =', t3)
            print('singular term =', t4)
    finally:
        ctx.prec = prec
    if kwargs.get("error"):
        w = max(ctx.mag(abs(t)),0)
        err = max(err*2**w, ctx.eps*1.*2**w)
        return +t, err
    return +t


@defun_wrapped
def lerchphi(ctx, z, s, a):
    r"""
    Gives the Lerch transcendent, defined for `|z| < 1` and
    `\Re{a} > 0` by

    .. math ::

        \Phi(z,s,a) = \sum_{k=0}^{\infty} \frac{z^k}{(a+k)^s}

    and generally by the recurrence `\Phi(z,s,a) = z \Phi(z,s,a+1) + a^{-s}`
    along with the integral representation valid for `\Re{a} > 0`

    .. math ::

        \Phi(z,s,a) = \frac{1}{2 a^s} +
                \int_0^{\infty} \frac{z^t}{(a+t)^s} dt -
                2 \int_0^{\infty} \frac{\sin(t \log z - s
                    \operatorname{arctan}(t/a)}{(a^2 + t^2)^{s/2}
                    (e^{2 \pi t}-1)} dt.

    The Lerch transcendent generalizes the Hurwitz zeta function :func:`zeta`
    (`z = 1`) and the polylogarithm :func:`polylog` (`a = 1`).

    **Examples**

    Several evaluations in terms of simpler functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> lerchphi(-1,2,0.5); 4*catalan
        3.663862376708876060218414
        3.663862376708876060218414
        >>> diff(lerchphi, (-1,-2,1), (0,1,0)); 7*zeta(3)/(4*pi**2)
        0.2131391994087528954617607
        0.2131391994087528954617607
        >>> lerchphi(-4,1,1); log(5)/4
        0.4023594781085250936501898
        0.4023594781085250936501898
        >>> lerchphi(-3+2j,1,0.5); 2*atanh(sqrt(-3+2j))/sqrt(-3+2j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)

    Evaluation works for complex arguments and `|z| \ge 1`::

        >>> lerchphi(1+2j, 3-j, 4+2j)
        (0.002025009957009908600539469 + 0.003327897536813558807438089j)
        >>> lerchphi(-2,2,-2.5)
        -12.28676272353094275265944
        >>> lerchphi(10,10,10)
        (-4.462130727102185701817349e-11 - 1.575172198981096218823481e-12j)
        >>> lerchphi(10,10,-10.5)
        (112658784011940.5605789002 - 498113185.5756221777743631j)

    Some degenerate cases::

        >>> lerchphi(0,1,2)
        0.5
        >>> lerchphi(0,1,-2)
        -0.5

    Reduction to simpler functions::

        >>> lerchphi(1, 4.25+1j, 1)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> zeta(4.25+1j)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> lerchphi(1 - 0.5**10, 4.25+1j, 1)
        (1.044629338021507546737197 - 0.04667768813963388181708101j)
        >>> lerchphi(3, 4, 1)
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> polylog(4, 3) / 3
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> lerchphi(3, 4, 1 - 0.5**10)
        (1.253978063946663945672674 - 0.2316736622836535468765376j)

    **References**

    1. [DLMF]_ section 25.14

    """
    if z == 0:
        return a ** (-s)
    # Faster, but these cases are useful for testing right now
    if z == 1:
        return ctx.zeta(s, a)
    if a == 1:
        return ctx.polylog(s, z) / z
    if ctx.re(a) < 1:
        if ctx.isnpint(a):
            raise ValueError("Lerch transcendent complex infinity")
        m = int(ctx.ceil(1-ctx.re(a)))
        v = ctx.zero
        zpow = ctx.one
        for n in xrange(m):
            v += zpow / (a+n)**s
            zpow *= z
        return zpow * ctx.lerchphi(z,s, a+m) + v
    g = ctx.ln(z)
    v = 1/(2*a**s) + ctx.gammainc(1-s, -a*g) * (-g)**(s-1) / z**a
    h = s / 2
    r = 2*ctx.pi
    f = lambda t: ctx.sin(s*ctx.atan(t/a)-t*g) / \
        ((a**2+t**2)**h * ctx.expm1(r*t))
    v += 2*ctx.quad(f, [0, ctx.inf])
    if not ctx.im(z) and not ctx.im(s) and not ctx.im(a) and ctx.re(z) < 1:
        v = ctx.chop(v)
    return v
<|endoftext|>"""
The function zetazero(n) computes the n-th nontrivial zero of zeta(s).

The general strategy is to locate a block of Gram intervals B where we
know exactly the number of zeros contained and which of those zeros
is that which we search.

If n <= 400 000 000  we know exactly the Rosser exceptions, contained
in a list in this file. Hence for n<=400 000 000 we simply
look at these list of exceptions. If our zero is implicated in one of
these exceptions we have our block B.  In other case we simply locate
the good Rosser block containing our zero.

For n > 400 000 000 we apply the method of Turing, as complemented by
Lehman, Brent and Trudgian  to find a suitable B.
"""

from .functions import defun, defun_wrapped

def find_rosser_block_zero(ctx, n):
    """for n<400 000 000 determines a block were one find our zero"""
    for k in range(len(_ROSSER_EXCEPTIONS)//2):
        a=_ROSSER_EXCEPTIONS[2*k][0]
        b=_ROSSER_EXCEPTIONS[2*k][1]
        if ((a<= n-2) and (n-1 <= b)):
            t0 = ctx.grampoint(a)
            t1 = ctx.grampoint(b)
            v0 = ctx._fp.siegelz(t0)
            v1 = ctx._fp.siegelz(t1)
            my_zero_number = n-a-1
            zero_number_block = b-a
            pattern = _ROSSER_EXCEPTIONS[2*k+1]
            return (my_zero_number, [a,b], [t0,t1], [v0,v1])
    k = n-2
    t,v,b = compute_triple_tvb(ctx, k)
    T = [t]
    V = [v]
    while b < 0:
        k -= 1
        t,v,b = compute_triple_tvb(ctx, k)
        T.insert(0,t)
        V.insert(0,v)
    my_zero_number = n-k-1
    m = n-1
    t,v,b = compute_triple_tvb(ctx, m)
    T.append(t)
    V.append(v)
    while b < 0:
        m += 1
        t,v,b = compute_triple_tvb(ctx, m)
        T.append(t)
        V.append(v)
    return (my_zero_number, [k,m], T, V)

def wpzeros(t):
    """Precision needed to compute higher zeros"""
    wp = 53
    if t > 3*10**8:
        wp = 63
    if t > 10**11:
        wp = 70
    if t > 10**14:
        wp = 83
    return wp

def separate_zeros_in_block(ctx, zero_number_block, T, V, limitloop=None,
    fp_tolerance=None):
    """Separate the zeros contained in the block T, limitloop
    determines how long one must search"""
    if limitloop is None:
        limitloop = ctx.inf
    loopnumber = 0
    variations = count_variations(V)
    while ((variations < zero_number_block) and (loopnumber <limitloop)):
        a = T[0]
        v = V[0]
        newT = [a]
        newV = [v]
        variations = 0
        for n in range(1,len(T)):
            b2 = T[n]
            u = V[n]
            if (u*v>0):
                alpha = ctx.sqrt(u/v)
                b= (alpha*a+b2)/(alpha+1)
            else:
                b = (a+b2)/2
            if fp_tolerance < 10:
                w = ctx._fp.siegelz(b)
                if abs(w)<fp_tolerance:
                    w = ctx.siegelz(b)
            else:
                w=ctx.siegelz(b)
            if v*w<0:
                variations += 1
            newT.append(b)
            newV.append(w)
            u = V[n]
            if u*w <0:
                variations += 1
            newT.append(b2)
            newV.append(u)
            a = b2
            v = u
        T = newT
        V = newV
        loopnumber +=1
        if (limitloop>ITERATION_LIMIT)and(loopnumber>2)and(variations+2==zero_number_block):
            dtMax=0
            dtSec=0
            kMax = 0
            for k1 in range(1,len(T)):
                dt = T[k1]-T[k1-1]
                if dt > dtMax:
                    kMax=k1
                    dtSec = dtMax
                    dtMax = dt
                elif  (dt<dtMax) and(dt >dtSec):
                    dtSec = dt
            if dtMax>3*dtSec:
                f = lambda x: ctx.rs_z(x,derivative=1)
                t0=T[kMax-1]
                t1 = T[kMax]
                t=ctx.findroot(f,  (t0,t1), solver ='illinois',verify=False, verbose=False)
                v = ctx.siegelz(t)
                if (t0<t) and (t<t1) and (v*V[kMax]<0):
                    T.insert(kMax,t)
                    V.insert(kMax,v)
        variations = count_variations(V)
    if variations == zero_number_block:
        separated = True
    else:
        separated = False
    return (T,V, separated)

def separate_my_zero(ctx, my_zero_number, zero_number_block, T, V, prec):
    """If we know which zero of this block is mine,
    the function separates the zero"""
    variations = 0
    v0 = V[0]
    for k in range(1,len(V)):
        v1 = V[k]
        if v0*v1 < 0:
            variations +=1
            if variations == my_zero_number:
                k0 = k
                leftv = v0
                rightv = v1
        v0 = v1
    t1 = T[k0]
    t0 = T[k0-1]
    ctx.prec = prec
    wpz = wpzeros(my_zero_number*ctx.log(my_zero_number))

    guard = 4*ctx.mag(my_zero_number)
    precs = [ctx.prec+4]
    index=0
    while precs[0] > 2*wpz:
        index +=1
        precs = [precs[0] // 2 +3+2*index] + precs
    ctx.prec = precs[0] + guard
    r = ctx.findroot(lambda x:ctx.siegelz(x), (t0,t1), solver ='illinois', verbose=False)
    #print "first step at", ctx.dps, "digits"
    z=ctx.mpc(0.5,r)
    for prec in precs[1:]:
        ctx.prec = prec + guard
        #print "refining to", ctx.dps, "digits"
        znew = z - ctx.zeta(z) / ctx.zeta(z, derivative=1)
        #print "difference", ctx.nstr(abs(z-znew))
        z=ctx.mpc(0.5,ctx.im(znew))
    return ctx.im(z)

def sure_number_block(ctx, n):
    """The number of good Rosser blocks needed to apply
    Turing method
    References:
    R. P. Brent, On the Zeros of the Riemann Zeta Function
    in the Critical Strip, Math. Comp. 33 (1979) 1361--1372
    T. Trudgian, Improvements to Turing Method, Math. Comp."""
    if n < 9*10**5:
        return(2)
    g = ctx.grampoint(n-100)
    lg = ctx._fp.ln(g)
    brent = 0.0061 * lg**2 +0.08*lg
    trudgian = 0.0031 * lg**2 +0.11*lg
    N = ctx.ceil(min(brent,trudgian))
    N = int(N)
    return N

def compute_triple_tvb(ctx, n):
    t = ctx.grampoint(n)
    v = ctx._fp.siegelz(t)
    if ctx.mag(abs(v))<ctx.mag(t)-45:
        v = ctx.siegelz(t)
    b = v*(-1)**n
    return t,v,b



ITERATION_LIMIT = 4

def search_supergood_block(ctx, n, fp_tolerance):
    """To use for n>400 000 000"""
    sb = sure_number_block(ctx, n)
    number_goodblocks = 0
    m2 = n-1
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf = [t]
    Vf = [v]
    while b < 0:
        m2 += 1
        t,v,b = compute_triple_tvb(ctx, m2)
        Tf.append(t)
        Vf.append(v)
    goodpoints = [m2]
    T = [t]
    V = [v]
    while number_goodblocks < 2*sb:
        m2 += 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.append(t)
        V.append(v)
        while b < 0:
            m2 += 1
            t,v,b = compute_triple_tvb(ctx, m2)
            T.append(t)
            V.append(v)
        goodpoints.append(m2)
        zn = len(T)-1
        A, B, separated =\
           separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT,
                fp_tolerance=fp_tolerance)
        Tf.pop()
        Tf.extend(A)
        Vf.pop()
        Vf.extend(B)
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    # Now the same procedure to the left
    number_goodblocks = 0
    m2 = n-2
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf.insert(0,t)
    Vf.insert(0,v)
    while b < 0:
        m2 -= 1
        t,v,b = compute_triple_tvb(ctx, m2)
        Tf.insert(0,t)
        Vf.insert(0,v)
    goodpoints.insert(0,m2)
    T = [t]
    V = [v]
    while number_goodblocks < 2*sb:
        m2 -= 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.insert(0,t)
        V.insert(0,v)
        while b < 0:
            m2 -= 1
            t,v,b = compute_triple_tvb(ctx, m2)
            T.insert(0,t)
            V.insert(0,v)
        goodpoints.insert(0,m2)
        zn = len(T)-1
        A, B, separated =\
           separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
        A.pop()
        Tf = A+Tf
        B.pop()
        Vf = B+Vf
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    r = goodpoints[2*sb]
    lg = len(goodpoints)
    s = goodpoints[lg-2*sb-1]
    tr, vr, br = compute_triple_tvb(ctx, r)
    ar = Tf.index(tr)
    ts, vs, bs = compute_triple_tvb(ctx, s)
    as1 = Tf.index(ts)
    T = Tf[ar:as1+1]
    V = Vf[ar:as1+1]
    zn = s-r
    A, B, separated =\
       separate_zeros_in_block(ctx, zn,T,V,limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
    if separated:
        return (n-r-1,[r,s],A,B)
    q = goodpoints[sb]
    lg = len(goodpoints)
    t = goodpoints[lg-sb-1]
    tq, vq, bq = compute_triple_tvb(ctx, q)
    aq = Tf.index(tq)
    tt, vt, bt = compute_triple_tvb(ctx, t)
    at = Tf.index(tt)
    T = Tf[aq:at+1]
    V = Vf[aq:at+1]
    return (n-q-1,[q,t],T,V)

def count_variations(V):
    count = 0
    vold = V[0]
    for n in range(1, len(V)):
        vnew = V[n]
        if vold*vnew < 0:
            count +=1
        vold = vnew
    return count

def pattern_construct(ctx, block, T, V):
    pattern = '('
    a = block[0]
    b = block[1]
    t0,v0,b0 = compute_triple_tvb(ctx, a)
    k = 0
    k0 = 0
    for n in range(a+1,b+1):
        t1,v1,b1 = compute_triple_tvb(ctx, n)
        lgT =len(T)
        while (k < lgT) and (T[k] <= t1):
            k += 1
        L = V[k0:k]
        L.append(v1)
        L.insert(0,v0)
        count = count_variations(L)
        pattern = pattern + ("%s" % count)
        if b1 > 0:
            pattern = pattern + ')('
        k0 = k
        t0,v0,b0 = t1,v1,b1
    pattern = pattern[:-1]
    return pattern

@defun
def zetazero(ctx, n, info=False, round=True):
    r"""
    Computes the `n`-th nontrivial zero of `\zeta(s)` on the critical line,
    i.e. returns an approximation of the `n`-th largest complex number
    `s = \frac{1}{2} + ti` for which `\zeta(s) = 0`. Equivalently, the
    imaginary part `t` is a zero of the Z-function (:func:`~mpmath.siegelz`).

    **Examples**

    The first few zeros::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> zetazero(1)
        (0.5 + 14.13472514173469379045725j)
        >>> zetazero(2)
        (0.5 + 21.02203963877155499262848j)
        >>> zetazero(20)
        (0.5 + 77.14484006887480537268266j)

    Verifying that the values are zeros::

        >>> for n in range(1,5):
        ...     s = zetazero(n)
        ...     chop(zeta(s)), chop(siegelz(s.imag))
        ...
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)

    Negative indices give the conjugate zeros (`n = 0` is undefined)::

        >>> zetazero(-1)
        (0.5 - 14.13472514173469379045725j)

    :func:`~mpmath.zetazero` supports arbitrarily large `n` and arbitrary precision::

        >>> mp.dps = 15
        >>> zetazero(1234567)
        (0.5 + 727690.906948208j)
        >>> mp.dps = 50
        >>> zetazero(1234567)
        (0.5 + 727690.9069482075392389420041147142092708393819935j)
        >>> chop(zeta(_)/_)
        0.0

    with *info=True*, :func:`~mpmath.zetazero` gives additional information::

        >>> mp.dps = 15
        >>> zetazero(542964976,info=True)
        ((0.5 + 209039046.578535j), [542964969, 542964978], 6, '(013111110)')

    This means that the zero is between Gram points 542964969 and 542964978;
    it is the 6-th zero between them. Finally (01311110) is the pattern
    of zeros in this interval. The numbers indicate the number of zeros
    in each Gram interval (Rosser blocks between parenthesis). In this case
    there is only one Rosser block of length nine.
    """
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError("n must be nonzero")
    wpinitial = ctx.prec
    try:
        wpz, fp_tolerance = comp_fp_tolerance(ctx, n)
        ctx.prec = wpz
        if n < 400000000:
            my_zero_number, block, T, V =\
             find_rosser_block_zero(ctx, n)
        else:
            my_zero_number, block, T, V =\
             search_supergood_block(ctx, n, fp_tolerance)
        zero_number_block = block[1]-block[0]
        T, V, separated = separate_zeros_in_block(ctx, zero_number_block, T, V,
            limitloop=ctx.inf, fp_tolerance=fp_tolerance)
        if info:
            pattern = pattern_construct(ctx,block,T,V)
        prec = max(wpinitial, wpz)
        t = separate_my_zero(ctx, my_zero_number, zero_number_block,T,V,prec)
        v = ctx.mpc(0.5,t)
    finally:
        ctx.prec = wpinitial
    if round:
        v =+v
    if info:
        return (v,block,my_zero_number,pattern)
    else:
        return v

def gram_index(ctx, t):
    if t > 10**13:
        wp = 3*ctx.log(t, 10)
    else:
        wp = 0
    prec = ctx.prec
    try:
        ctx.prec += wp
        h = int(ctx.siegeltheta(t)/ctx.pi)
    finally:
        ctx.prec = prec
    return(h)

def count_to(ctx, t, T, V):
    count = 0
    vold = V[0]
    told = T[0]
    tnew = T[1]
    k = 1
    while tnew < t:
        vnew = V[k]
        if vold*vnew < 0:
            count += 1
        vold = vnew
        k += 1
        tnew = T[k]
    a = ctx.siegelz(t)
    if a*vold < 0:
        count += 1
    return count

def comp_fp_tolerance(ctx, n):
    wpz = wpzeros(n*ctx.log(n))
    if n < 15*10**8:
        fp_tolerance = 0.0005
    elif n <= 10**14:
        fp_tolerance = 0.1
    else:
        fp_tolerance = 100
    return wpz, fp_tolerance

@defun
def nzeros(ctx, t):
    r"""
    Computes the number of zeros of the Riemann zeta function in
    `(0,1) \times (0,t]`, usually denoted by `N(t)`.

    **Examples**

    The first zero has imaginary part between 14 and 15::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nzeros(14)
        0
        >>> nzeros(15)
        1
        >>> zetazero(1)
        (0.5 + 14.1347251417347j)

    Some closely spaced zeros::

        >>> nzeros(10**7)
        21136125
        >>> zetazero(21136125)
        (0.5 + 9999999.32718175j)
        >>> zetazero(21136126)
        (0.5 + 10000000.2400236j)
        >>> nzeros(545439823.215)
        1500000001
        >>> zetazero(1500000001)
        (0.5 + 545439823.201985j)
        >>> zetazero(1500000002)
        (0.5 + 545439823.325697j)

    This confirms the data given by J. van de Lune,
    H. J. J. te Riele and D. T. Winter in 1986.
    """
    if t < 14.1347251417347:
        return 0
    x = gram_index(ctx, t)
    k = int(ctx.floor(x))
    wpinitial = ctx.prec
    wpz, fp_tolerance = comp_fp_tolerance(ctx, k)
    ctx.prec = wpz
    a = ctx.siegelz(t)
    if k == -1 and a < 0:
        return 0
    elif k == -1 and a > 0:
        return 1
    if k+2 < 400000000:
        Rblock = find_rosser_block_zero(ctx, k+2)
    else:
        Rblock = search_supergood_block(ctx, k+2, fp_tolerance)
    n1, n2 = Rblock[1]
    if n2-n1 == 1:
        b = Rblock[3][0]
        if a*b > 0:
            ctx.prec = wpinitial
            return k+1
        else:
            ctx.prec = wpinitial
            return k+2
    my_zero_number,block, T, V = Rblock
    zero_number_block = n2-n1
    T, V, separated = separate_zeros_in_block(ctx,\
                                              zero_number_block, T, V,\
                                              limitloop=ctx.inf,\
                                            fp_tolerance=fp_tolerance)
    n = count_to(ctx, t, T, V)
    ctx.prec = wpinitial
    return n+n1+1

@defun_wrapped
def backlunds(ctx, t):
    r"""
    Computes the function
    `S(t) = \operatorname{arg} \zeta(\frac{1}{2} + it) / \pi`.

    See Titchmarsh Section 9.3 for details of the definition.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> backlunds(217.3)
        0.16302205431184

    Generally, the value is a small number. At Gram points it is an integer,
    frequently equal to 0::

        >>> chop(backlunds(grampoint(200)))
        0.0
        >>> backlunds(extraprec(10)(grampoint)(211))
        1.0
        >>> backlunds(extraprec(10)(grampoint)(232))
        -1.0

    The number of zeros of the Riemann zeta function up to height `t`
    satisfies `N(t) = \theta(t)/\pi + 1 + S(t)` (see :func:nzeros` and
    :func:`siegeltheta`)::

        >>> t = 1234.55
        >>> nzeros(t)
        842
        >>> siegeltheta(t)/pi+1+backlunds(t)
        842.0

    """
    return ctx.nzeros(t)-1-ctx.siegeltheta(t)/ctx.pi


"""
_ROSSER_EXCEPTIONS is a list of all  exceptions to
Rosser's rule for n <= 400 000 000.

Alternately the  entry is of type   [n,m], or a string.
The string is the zero pattern of the Block and the relevant
adjacent.  For example (010)3 corresponds to a block
composed of three Gram intervals, the first ant third without
a zero and the intermediate with a zero. The next Gram interval
contain three zeros. So that in total we have 4 zeros in 4 Gram
blocks. n and m are the indices of the Gram points  of this
interval of four Gram intervals. The Rosser exception is therefore
formed by the three Gram intervals that are signaled between
parenthesis.

We have included also some Rosser's exceptions beyond n=400 000 000
that are noted in the literature by some reason.

The list is composed from the data published in the references:

R. P. Brent, J. van de Lune, H. J. J. te Riele, D. T. Winter,
'On the Zeros of the Riemann Zeta Function in the Critical Strip. II',
Math. Comp. 39 (1982) 681--688.
See also Corrigenda in Math. Comp. 46 (1986) 771.

J. van de Lune, H. J. J. te Riele,
'On the Zeros of the Riemann Zeta Function in the Critical Strip. III',
Math. Comp. 41 (1983) 759--767.
See also  Corrigenda in Math. Comp. 46 (1986) 771.

J. van de Lune,
'Sums of Equal Powers of Positive Integers',
Dissertation,
Vrije Universiteit te Amsterdam, Centrum voor Wiskunde en Informatica,
Amsterdam, 1984.

Thanks to the authors all this papers and those others that have
contributed to make this possible.
"""







_ROSSER_EXCEPTIONS = \
[[13999525, 13999528], '(00)3',
[30783329, 30783332], '(00)3',
[30930926, 30930929], '3(00)',
[37592215, 37592218], '(00)3',
[40870156, 40870159], '(00)3',
[43628107, 43628110], '(00)3',
[46082042, 46082045], '(00)3',
[46875667, 46875670], '(00)3',
[49624540, 49624543], '3(00)',
[50799238, 50799241], '(00)3',
[55221453, 55221456], '3(00)',
[56948779, 56948782], '3(00)',
[60515663, 60515666], '(00)3',
[61331766, 61331770], '(00)40',
[69784843, 69784846], '3(00)',
[75052114, 75052117], '(00)3',
[79545240, 79545243], '3(00)',
[79652247, 79652250], '3(00)',
[83088043, 83088046], '(00)3',
[83689522, 83689525], '3(00)',
[85348958, 85348961], '(00)3',
[86513820, 86513823], '(00)3',
[87947596, 87947599], '3(00)',
[88600095, 88600098], '(00)3',
[93681183, 93681186], '(00)3',
[100316551, 100316554], '3(00)',
[100788444, 100788447], '(00)3',
[106236172, 106236175], '(00)3',
[106941327, 106941330], '3(00)',
[107287955, 107287958], '(00)3',
[107532016, 107532019], '3(00)',
[110571044, 110571047], '(00)3',
[111885253, 111885256], '3(00)',
[113239783, 113239786], '(00)3',
[120159903, 120159906], '(00)3',
[121424391, 121424394], '3(00)',
[121692931, 121692934], '3(00)',
[121934170, 121934173], '3(00)',
[122612848, 122612851], '3(00)',
[126116567, 126116570], '(00)3',
[127936513, 127936516], '(00)3',
[128710277, 128710280], '3(00)',
[129398902, 129398905], '3(00)',
[130461096, 130461099], '3(00)',
[131331947, 131331950], '3(00)',
[137334071, 137334074], '3(00)',
[137832603, 137832606], '(00)3',
[138799471, 138799474], '3(00)',
[139027791, 139027794], '(00)3',
[141617806, 141617809], '(00)3',
[144454931, 144454934], '(00)3',
[145402379, 145402382], '3(00)',
[146130245, 146130248], '3(00)',
[147059770, 147059773], '(00)3',
[147896099, 147896102], '3(00)',
[151097113, 151097116], '(00)3',
[152539438, 152539441], '(00)3',
[152863168, 152863171], '3(00)',
[153522726, 153522729], '3(00)',
[155171524, 155171527], '3(00)',
[155366607, 155366610], '(00)3',
[157260686, 157260689], '3(00)',
[157269224, 157269227], '(00)3',
[157755123, 157755126], '(00)3',
[158298484, 158298487], '3(00)',
[160369050, 160369053], '3(00)',
[162962787, 162962790], '(00)3',
[163724709, 163724712], '(00)3',
[164198113, 164198116], '3(00)',
[164689301, 164689305], '(00)40',
[164880228, 164880231], '3(00)',
[166201932, 166201935], '(00)3',
[168573836, 168573839], '(00)3',
[169750763, 169750766], '(00)3',
[170375507, 170375510], '(00)3',
[170704879, 170704882], '3(00)',
[172000992, 172000995], '3(00)',
[173289941, 173289944], '(00)3',
[173737613, 173737616], '3(00)',
[174102513, 174102516], '(00)3',
[174284990, 174284993], '(00)3',
[174500513, 174500516], '(00)3',
[175710609, 175710612], '(00)3',
[176870843, 176870846], '3(00)',
[177332732, 177332735], '3(00)',
[177902861, 177902864], '3(00)',
[179979095, 179979098], '(00)3',
[181233726, 181233729], '3(00)',
[181625435, 181625438], '(00)3',
[182105255, 182105259], '22(00)',
[182223559, 182223562], '3(00)',
[191116404, 191116407], '3(00)',
[191165599, 191165602], '3(00)',
[191297535, 191297539], '(00)22',
[192485616, 192485619], '(00)3',
[193264634, 193264638], '22(00)',
[194696968, 194696971], '(00)3',
[195876805, 195876808], '(00)3',
[195916548, 195916551], '3(00)',
[196395160, 196395163], '3(00)',
[196676303, 196676306], '(00)3',
[197889882, 197889885], '3(00)',
[198014122, 198014125], '(00)3',
[199235289, 199235292], '(00)3',
[201007375, 201007378], '(00)3',
[201030605, 201030608], '3(00)',
[201184290, 201184293], '3(00)',
[201685414, 201685418], '(00)22',
[202762875, 202762878], '3(00)',
[202860957, 202860960], '3(00)',
[203832577, 203832580], '3(00)',
[205880544, 205880547], '(00)3',
[206357111, 206357114], '(00)3',
[207159767, 207159770], '3(00)',
[207167343, 207167346], '3(00)',
[207482539, 207482543], '3(010)',
[207669540, 207669543], '3(00)',
[208053426, 208053429], '(00)3',
[208110027, 208110030], '3(00)',
[209513826, 209513829], '3(00)',
[212623522, 212623525], '(00)3',
[213841715, 213841718], '(00)3',
[214012333, 214012336], '(00)3',
[214073567, 214073570], '(00)3',
[215170600, 215170603], '3(00)',
[215881039, 215881042], '3(00)',
[216274604, 216274607], '3(00)',
[216957120, 216957123], '3(00)',
[217323208, 217323211], '(00)3',
[218799264, 218799267], '(00)3',
[218803557, 218803560], '3(00)',
[219735146, 219735149], '(00)3',
[219830062, 219830065], '3(00)',
[219897904, 219897907], '(00)3',
[221205545, 221205548], '(00)3',
[223601929, 223601932], '(00)3',
[223907076, 223907079], '3(00)',
[223970397, 223970400], '(00)3',
[224874044, 224874048], '22(00)',
[225291157, 225291160], '(00)3',
[227481734, 227481737], '(00)3',
[228006442, 228006445], '3(00)',
[228357900, 228357903], '(00)3',
[228386399, 228386402], '(00)3',
[228907446, 228907449], '(00)3',
[228984552, 228984555], '3(00)',
[229140285, 229140288], '3(00)',
[231810024, 231810027], '(00)3',
[232838062, 232838065], '3(00)',
[234389088, 234389091], '3(00)',
[235588194, 235588197], '(00)3',
[236645695, 236645698], '(00)3',
[236962876, 236962879], '3(00)',
[237516723, 237516727], '04(00)',
[240004911, 240004914], '(00)3',
[240221306, 240221309], '3(00)',
[241389213, 241389217], '(010)3',
[241549003, 241549006], '(00)3',
[241729717, 241729720], '(00)3',
[241743684, 241743687], '3(00)',
[243780200, 243780203], '3(00)',
[243801317, 243801320], '(00)3',
[244122072, 244122075], '(00)3',
[244691224, 244691227], '3(00)',
[244841577, 244841580], '(00)3',
[245813461, 245813464], '(00)3',
[246299475, 246299478], '(00)3',
[246450176, 246450179], '3(00)',
[249069349, 249069352], '(00)3',
[250076378, 250076381], '(00)3',
[252442157, 252442160], '3(00)',
[252904231, 252904234], '3(00)',
[255145220, 255145223], '(00)3',
[255285971, 255285974], '3(00)',
[256713230, 256713233], '(00)3',
[257992082, 257992085], '(00)3',
[258447955, 258447959], '22(00)',
[259298045, 259298048], '3(00)',
[262141503, 262141506], '(00)3',
[263681743, 263681746], '3(00)',
[266527881, 266527885], '(010)3',
[266617122, 266617125], '(00)3',
[266628044, 266628047], '3(00)',
[267305763, 267305766], '(00)3',
[267388404, 267388407], '3(00)',
[267441672, 267441675], '3(00)',
[267464886, 267464889], '(00)3',
[267554907, 267554910], '3(00)',
[269787480, 269787483], '(00)3',
[270881434, 270881437], '(00)3',
[270997583, 270997586], '3(00)',
[272096378, 272096381], '3(00)',
[272583009, 272583012], '(00)3',
[274190881, 274190884], '3(00)',
[274268747, 274268750], '(00)3',
[275297429, 275297432], '3(00)',
[275545476, 275545479], '3(00)',
[275898479, 275898482], '3(00)',
[275953000, 275953003], '(00)3',
[277117197, 277117201], '(00)22',
[277447310, 277447313], '3(00)',
[279059657, 279059660], '3(00)',
[279259144, 279259147], '3(00)',
[279513636, 279513639], '3(00)',
[279849069, 279849072], '3(00)',
[280291419, 280291422], '(00)3',
[281449425, 281449428], '3(00)',
[281507953, 281507956], '3(00)',
[281825600, 281825603], '(00)3',
[282547093, 282547096], '3(00)',
[283120963, 283120966], '3(00)',
[283323493, 283323496], '(00)3',
[284764535, 284764538], '3(00)',
[286172639, 286172642], '3(00)',
[286688824, 286688827], '(00)3',
[287222172, 287222175], '3(00)',
[287235534, 287235537], '3(00)',
[287304861, 287304864], '3(00)',
[287433571, 287433574], '(00)3',
[287823551, 287823554], '(00)3',
[287872422, 287872425], '3(00)',
[288766615, 288766618], '3(00)',
[290122963, 290122966], '3(00)',
[290450849, 290450853], '(00)22',
[291426141, 291426144], '3(00)',
[292810353, 292810356], '3(00)',
[293109861, 293109864], '3(00)',
[293398054, 293398057], '3(00)',
[294134426, 294134429], '3(00)',
[294216438, 294216441], '(00)3',
[295367141, 295367144], '3(00)',
[297834111, 297834114], '3(00)',
[299099969, 299099972], '3(00)',
[300746958, 300746961], '3(00)',
[301097423, 301097426], '(00)3',
[301834209, 301834212], '(00)3',
[302554791, 302554794], '(00)3',
[303497445, 303497448], '3(00)',
[304165344, 304165347], '3(00)',
[304790218, 304790222], '3(010)',
[305302352, 305302355], '(00)3',
[306785996, 306785999], '3(00)',
[307051443, 307051446], '3(00)',
[307481539, 307481542], '3(00)',
[308605569, 308605572], '3(00)',
[309237610, 309237613], '3(00)',
[310509287, 310509290], '(00)3',
[310554057, 310554060], '3(00)',
[310646345, 310646348], '3(00)',
[311274896, 311274899], '(00)3',
[311894272, 311894275], '3(00)',
[312269470, 312269473], '(00)3',
[312306601, 312306605], '(00)40',
[312683193, 312683196], '3(00)',
[314499804, 314499807], '3(00)',
[314636802, 314636805], '(00)3',
[314689897, 314689900], '3(00)',
[314721319, 314721322], '3(00)',
[316132890, 316132893], '3(00)',
[316217470, 316217474], '(010)3',
[316465705, 316465708], '3(00)',
[316542790, 316542793], '(00)3',
[320822347, 320822350], '3(00)',
[321733242, 321733245], '3(00)',
[324413970, 324413973], '(00)3',
[325950140, 325950143], '(00)3',
[326675884, 326675887], '(00)3',
[326704208, 326704211], '3(00)',
[327596247, 327596250], '3(00)',
[328123172, 328123175], '3(00)',
[328182212, 328182215], '(00)3',
[328257498, 328257501], '3(00)',
[328315836, 328315839], '(00)3',
[328800974, 328800977], '(00)3',
[328998509, 328998512], '3(00)',
[329725370, 329725373], '(00)3',
[332080601, 332080604], '(00)3',
[332221246, 332221249], '(00)3',
[332299899, 332299902], '(00)3',
[332532822, 332532825], '(00)3',
[333334544, 333334548], '(00)22',
[333881266, 333881269], '3(00)',
[334703267, 334703270], '3(00)',
[334875138, 334875141], '3(00)',
[336531451, 336531454], '3(00)',
[336825907, 336825910], '(00)3',
[336993167, 336993170], '(00)3',
[337493998, 337494001], '3(00)',
[337861034, 337861037], '3(00)',
[337899191, 337899194], '(00)3',
[337958123, 337958126], '(00)3',
[342331982, 342331985], '3(00)',
[342676068, 342676071], '3(00)',
[347063781, 347063784], '3(00)',
[347697348, 347697351], '3(00)',
[347954319, 347954322], '3(00)',
[348162775, 348162778], '3(00)',
[349210702, 349210705], '(00)3',
[349212913, 349212916], '3(00)',
[349248650, 349248653], '(00)3',
[349913500, 349913503], '3(00)',
[350891529, 350891532], '3(00)',
[351089323, 351089326], '3(00)',
[351826158, 351826161], '3(00)',
[352228580, 352228583], '(00)3',
[352376244, 352376247], '3(00)',
[352853758, 352853761], '(00)3',
[355110439, 355110442], '(00)3',
[355808090, 355808094], '(00)40',
[355941556, 355941559], '3(00)',
[356360231, 356360234], '(00)3',
[356586657, 356586660], '3(00)',
[356892926, 356892929], '(00)3',
[356908232, 356908235], '3(00)',
[357912730, 357912733], '3(00)',
[358120344, 358120347], '3(00)',
[359044096, 359044099], '(00)3',
[360819357, 360819360], '3(00)',
[361399662, 361399666], '(010)3',
[362361315, 362361318], '(00)3',
[363610112, 363610115], '(00)3',
[363964804, 363964807], '3(00)',
[364527375, 364527378], '(00)3',
[365090327, 365090330], '(00)3',
[365414539, 365414542], '3(00)',
[366738474, 366738477], '3(00)',
[368714778, 368714783], '04(010)',
[368831545, 368831548], '(00)3',
[368902387, 368902390], '(00)3',
[370109769, 370109772], '3(00)',
[370963333, 370963336], '3(00)',
[372541136, 372541140], '3(010)',
[372681562, 372681565], '(00)3',
[373009410, 373009413], '(00)3',
[373458970, 373458973], '3(00)',
[375648658, 375648661], '3(00)',
[376834728, 376834731], '3(00)',
[377119945, 377119948], '(00)3',
[377335703, 377335706], '(00)3',
[378091745, 378091748], '3(00)',
[379139522, 379139525], '3(00)',
[380279160, 380279163], '(00)3',
[380619442, 380619445], '3(00)',
[381244231, 381244234], '3(00)',
[382327446, 382327450], '(010)3',
[382357073, 382357076], '3(00)',
[383545479, 383545482], '3(00)',
[384363766, 384363769], '(00)3',
[384401786, 384401790], '22(00)',
[385198212, 385198215], '3(00)',
[385824476, 385824479], '(00)3',
[385908194, 385908197], '3(00)',
[386946806, 386946809], '3(00)',
[387592175, 387592179], '22(00)',
[388329293, 388329296], '(00)3',
[388679566, 388679569], '3(00)',
[388832142, 388832145], '3(00)',
[390087103, 390087106], '(00)3',
[390190926, 390190930], '(00)22',
[390331207, 390331210], '3(00)',
[391674495, 391674498], '3(00)',
[391937831, 391937834], '3(00)',
[391951632, 391951636], '(00)22',
[392963986, 392963989], '(00)3',
[393007921, 393007924], '3(00)',
[393373210, 393373213], '3(00)',
[393759572, 393759575], '(00)3',
[394036662, 394036665], '(00)3',
[395813866, 395813869], '(00)3',
[395956690, 395956693], '3(00)',
[396031670, 396031673], '3(00)',
[397076433, 397076436], '3(00)',
[397470601, 397470604], '3(00)',
[398289458, 398289461], '3(00)',
#
[368714778, 368714783], '04(010)',
[437953499, 437953504], '04(010)',
[526196233, 526196238], '032(00)',
[744719566, 744719571], '(010)40',
[750375857, 750375862], '032(00)',
[958241932, 958241937], '04(010)',
[983377342, 983377347], '(00)410',
[1003780080, 1003780085], '04(010)',
[1070232754, 1070232759], '(00)230',
[1209834865, 1209834870], '032(00)',
[1257209100, 1257209105], '(00)410',
[1368002233, 1368002238], '(00)230'
]
<|endoftext|>from ..libmp.backend import xrange

class SpecialFunctions(object):
    """
    This class implements special functions using high-level code.

    Elementary and some other functions (e.g. gamma function, basecase
    hypergeometric series) are assumed to be predefined by the context as
    "builtins" or "low-level" functions.
    """
    defined_functions = {}

    # The series for the Jacobi theta functions converge for |q| < 1;
    # in the current implementation they throw a ValueError for
    # abs(q) > THETA_Q_LIM
    THETA_Q_LIM = 1 - 10**-7

    def __init__(self):
        cls = self.__class__
        for name in cls.defined_functions:
            f, wrap = cls.defined_functions[name]
            cls._wrap_specfun(name, f, wrap)

        self.mpq_1 = self._mpq((1,1))
        self.mpq_0 = self._mpq((0,1))
        self.mpq_1_2 = self._mpq((1,2))
        self.mpq_3_2 = self._mpq((3,2))
        self.mpq_1_4 = self._mpq((1,4))
        self.mpq_1_16 = self._mpq((1,16))
        self.mpq_3_16 = self._mpq((3,16))
        self.mpq_5_2 = self._mpq((5,2))
        self.mpq_3_4 = self._mpq((3,4))
        self.mpq_7_4 = self._mpq((7,4))
        self.mpq_5_4 = self._mpq((5,4))
        self.mpq_1_3 = self._mpq((1,3))
        self.mpq_2_3 = self._mpq((2,3))
        self.mpq_4_3 = self._mpq((4,3))
        self.mpq_1_6 = self._mpq((1,6))
        self.mpq_5_6 = self._mpq((5,6))
        self.mpq_5_3 = self._mpq((5,3))

        self._misc_const_cache = {}

        self._aliases.update({
            'phase' : 'arg',
            'conjugate' : 'conj',
            'nthroot' : 'root',
            'polygamma' : 'psi',
            'hurwitz' : 'zeta',
            #'digamma' : 'psi0',
            #'trigamma' : 'psi1',
            #'tetragamma' : 'psi2',
            #'pentagamma' : 'psi3',
            'fibonacci' : 'fib',
            'factorial' : 'fac',
        })

        self.zetazero_memoized = self.memoize(self.zetazero)

    # Default -- do nothing
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        setattr(cls, name, f)

    # Optional fast versions of common functions in common cases.
    # If not overridden, default (generic hypergeometric series)
    # implementations will be used
    def _besselj(ctx, n, z): raise NotImplementedError
    def _erf(ctx, z): raise NotImplementedError
    def _erfc(ctx, z): raise NotImplementedError
    def _gamma_upper_int(ctx, z, a): raise NotImplementedError
    def _expint_int(ctx, n, z): raise NotImplementedError
    def _zeta(ctx, s): raise NotImplementedError
    def _zetasum_fast(ctx, s, a, n, derivatives, reflect): raise NotImplementedError
    def _ei(ctx, z): raise NotImplementedError
    def _e1(ctx, z): raise NotImplementedError
    def _ci(ctx, z): raise NotImplementedError
    def _si(ctx, z): raise NotImplementedError
    def _altzeta(ctx, s): raise NotImplementedError

def defun_wrapped(f):
    SpecialFunctions.defined_functions[f.__name__] = f, True
    return f

def defun(f):
    SpecialFunctions.defined_functions[f.__name__] = f, False
    return f

def defun_static(f):
    setattr(SpecialFunctions, f.__name__, f)
    return f

@defun_wrapped
def cot(ctx, z): return ctx.one / ctx.tan(z)

@defun_wrapped
def sec(ctx, z): return ctx.one / ctx.cos(z)

@defun_wrapped
def csc(ctx, z): return ctx.one / ctx.sin(z)

@defun_wrapped
def coth(ctx, z): return ctx.one / ctx.tanh(z)

@defun_wrapped
def sech(ctx, z): return ctx.one / ctx.cosh(z)

@defun_wrapped
def csch(ctx, z): return ctx.one / ctx.sinh(z)

@defun_wrapped
def acot(ctx, z):
    if not z:
        return ctx.pi * 0.5
    else:
        return ctx.atan(ctx.one / z)

@defun_wrapped
def asec(ctx, z): return ctx.acos(ctx.one / z)

@defun_wrapped
def acsc(ctx, z): return ctx.asin(ctx.one / z)

@defun_wrapped
def acoth(ctx, z):
    if not z:
        return ctx.pi * 0.5j
    else:
        return ctx.atanh(ctx.one / z)


@defun_wrapped
def asech(ctx, z): return ctx.acosh(ctx.one / z)

@defun_wrapped
def acsch(ctx, z): return ctx.asinh(ctx.one / z)

@defun
def sign(ctx, x):
    x = ctx.convert(x)
    if not x or ctx.isnan(x):
        return x
    if ctx._is_real_type(x):
        if x > 0:
            return ctx.one
        else:
            return -ctx.one
    return x / abs(x)

@defun
def agm(ctx, a, b=1):
    if b == 1:
        return ctx.agm1(a)
    a = ctx.convert(a)
    b = ctx.convert(b)
    return ctx._agm(a, b)

@defun_wrapped
def sinc(ctx, x):
    if ctx.isinf(x):
        return 1/x
    if not x:
        return x+1
    return ctx.sin(x)/x

@defun_wrapped
def sincpi(ctx, x):
    if ctx.isinf(x):
        return 1/x
    if not x:
        return x+1
    return ctx.sinpi(x)/(ctx.pi*x)

# TODO: tests; improve implementation
@defun_wrapped
def expm1(ctx, x):
    if not x:
        return ctx.zero
    # exp(x) - 1 ~ x
    if ctx.mag(x) < -ctx.prec:
        return x + 0.5*x**2
    # TODO: accurately eval the smaller of the real/imag parts
    return ctx.sum_accurately(lambda: iter([ctx.exp(x),-1]),1)

@defun_wrapped
def log1p(ctx, x):
    if not x:
        return ctx.zero
    if ctx.mag(x) < -ctx.prec:
        return x - 0.5*x**2
    return ctx.log(ctx.fadd(1, x, prec=2*ctx.prec))

@defun_wrapped
def powm1(ctx, x, y):
    mag = ctx.mag
    one = ctx.one
    w = x**y - one
    M = mag(w)
    # Only moderate cancellation
    if M > -8:
        return w
    # Check for the only possible exact cases
    if not w:
        if (not y) or (x in (1, -1, 1j, -1j) and ctx.isint(y)):
            return w
    x1 = x - one
    magy = mag(y)
    lnx = ctx.ln(x)
    # Small y: x^y - 1 ~ log(x)*y + O(log(x)^2 * y^2)
    if magy + mag(lnx) < -ctx.prec:
        return lnx*y + (lnx*y)**2/2
    # TODO: accurately eval the smaller of the real/imag part
    return ctx.sum_accurately(lambda: iter([x**y, -1]), 1)

@defun
def _rootof1(ctx, k, n):
    k = int(k)
    n = int(n)
    k %= n
    if not k:
        return ctx.one
    elif 2*k == n:
        return -ctx.one
    elif 4*k == n:
        return ctx.j
    elif 4*k == 3*n:
        return -ctx.j
    return ctx.expjpi(2*ctx.mpf(k)/n)

@defun
def root(ctx, x, n, k=0):
    n = int(n)
    x = ctx.convert(x)
    if k:
        # Special case: there is an exact real root
        if (n & 1 and 2*k == n-1) and (not ctx.im(x)) and (ctx.re(x) < 0):
            return -ctx.root(-x, n)
        # Multiply by root of unity
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.root(x, n, 0) * ctx._rootof1(k, n)
        finally:
            ctx.prec = prec
        return +v
    return ctx._nthroot(x, n)

@defun
def unitroots(ctx, n, primitive=False):
    gcd = ctx._gcd
    prec = ctx.prec
    try:
        ctx.prec += 10
        if primitive:
            v = [ctx._rootof1(k,n) for k in range(n) if gcd(k,n) == 1]
        else:
            # TODO: this can be done *much* faster
            v = [ctx._rootof1(k,n) for k in range(n)]
    finally:
        ctx.prec = prec
    return [+x for x in v]

@defun
def arg(ctx, x):
    x = ctx.convert(x)
    re = ctx._re(x)
    im = ctx._im(x)
    return ctx.atan2(im, re)

@defun
def fabs(ctx, x):
    return abs(ctx.convert(x))

@defun
def re(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, "real"):    # py2.5 doesn't have .real/.imag for all numbers
        return x.real
    return x

@defun
def im(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, "imag"):    # py2.5 doesn't have .real/.imag for all numbers
        return x.imag
    return ctx.zero

@defun
def conj(ctx, x):
    x = ctx.convert(x)
    try:
        return x.conjugate()
    except AttributeError:
        return x

@defun
def polar(ctx, z):
    return (ctx.fabs(z), ctx.arg(z))

@defun_wrapped
def rect(ctx, r, phi):
    return r * ctx.mpc(*ctx.cos_sin(phi))

@defun
def log(ctx, x, b=None):
    if b is None:
        return ctx.ln(x)
    wp = ctx.prec + 20
    return ctx.ln(x, prec=wp) / ctx.ln(b, prec=wp)

@defun
def log10(ctx, x):
    return ctx.log(x, 10)

@defun
def fmod(ctx, x, y):
    return ctx.convert(x) % ctx.convert(y)

@defun
def degrees(ctx, x):
    return x / ctx.degree

@defun
def radians(ctx, x):
    return x * ctx.degree

def _lambertw_special(ctx, z, k):
    # W(0,0) = 0; all other branches are singular
    if not z:
        if not k:
            return z
        return ctx.ninf + z
    if z == ctx.inf:
        if k == 0:
            return z
        else:
            return z + 2*k*ctx.pi*ctx.j
    if z == ctx.ninf:
        return (-z) + (2*k+1)*ctx.pi*ctx.j
    # Some kind of nan or complex inf/nan?
    return ctx.ln(z)

import math
import cmath

def _lambertw_approx_hybrid(z, k):
    imag_sign = 0
    if hasattr(z, "imag"):
        x = float(z.real)
        y = z.imag
        if y:
            imag_sign = (-1) ** (y < 0)
        y = float(y)
    else:
        x = float(z)
        y = 0.0
        imag_sign = 0
    # hack to work regardless of whether Python supports -0.0
    if not y:
        y = 0.0
    z = complex(x,y)
    if k == 0:
        if -4.0 < y < 4.0 and -1.0 < x < 2.5:
            if imag_sign:
                # Taylor series in upper/lower half-plane
                if y > 1.00: return (0.876+0.645j) + (0.118-0.174j)*(z-(0.75+2.5j))
                if y > 0.25: return (0.505+0.204j) + (0.375-0.132j)*(z-(0.75+0.5j))
                if y < -1.00: return (0.876-0.645j) + (0.118+0.174j)*(z-(0.75-2.5j))
                if y < -0.25: return (0.505-0.204j) + (0.375+0.132j)*(z-(0.75-0.5j))
            # Taylor series near -1
            if x < -0.5:
                if imag_sign >= 0:
                    return (-0.318+1.34j) + (-0.697-0.593j)*(z+1)
                else:
                    return (-0.318-1.34j) + (-0.697+0.593j)*(z+1)
            # return real type
            r = -0.367879441171442
            if (not imag_sign) and x > r:
                z = x
            # Singularity near -1/e
            if x < -0.2:
                return -1 + 2.33164398159712*(z-r)**0.5 - 1.81218788563936*(z-r)
            # Taylor series near 0
            if x < 0.5: return z
            # Simple linear approximation
            return 0.2 + 0.3*z
        if (not imag_sign) and x > 0.0:
            L1 = math.log(x); L2 = math.log(L1)
        else:
            L1 = cmath.log(z); L2 = cmath.log(L1)
    elif k == -1:
        # return real type
        r = -0.367879441171442
        if (not imag_sign) and r < x < 0.0:
            z = x
        if (imag_sign >= 0) and y < 0.1 and -0.6 < x < -0.2:
            return -1 - 2.33164398159712*(z-r)**0.5 - 1.81218788563936*(z-r)
        if (not imag_sign) and -0.2 <= x < 0.0:
            L1 = math.log(-x)
            return L1 - math.log(-L1)
        else:
            if imag_sign == -1 and (not y) and x < 0.0:
                L1 = cmath.log(z) - 3.1415926535897932j
            else:
                L1 = cmath.log(z) - 6.2831853071795865j
            L2 = cmath.log(L1)
    return L1 - L2 + L2/L1 + L2*(L2-2)/(2*L1**2)

def _lambertw_series(ctx, z, k, tol):
    """
    Return rough approximation for W_k(z) from an asymptotic series,
    sufficiently accurate for the Halley iteration to converge to
    the correct value.
    """
    magz = ctx.mag(z)
    if (-10 < magz < 900) and (-1000 < k < 1000):
        # Near the branch point at -1/e
        if magz < 1 and abs(z+0.36787944117144) < 0.05:
            if k == 0 or (k == -1 and ctx._im(z) >= 0) or \
                         (k == 1  and ctx._im(z) < 0):
                delta = ctx.sum_accurately(lambda: [z, ctx.exp(-1)])
                cancellation = -ctx.mag(delta)
                ctx.prec += cancellation
                # Use series given in Corless et al.
                p = ctx.sqrt(2*(ctx.e*z+1))
                ctx.prec -= cancellation
                u = {0:ctx.mpf(-1), 1:ctx.mpf(1)}
                a = {0:ctx.mpf(2), 1:ctx.mpf(-1)}
                if k != 0:
                    p = -p
                s = ctx.zero
                # The series converges, so we could use it directly, but unless
                # *extremely* close, it is better to just use the first few
                # terms to get a good approximation for the iteration
                for l in xrange(max(2,cancellation)):
                    if l not in u:
                        a[l] = ctx.fsum(u[j]*u[l+1-j] for j in xrange(2,l))
                        u[l] = (l-1)*(u[l-2]/2+a[l-2]/4)/(l+1)-a[l]/2-u[l-1]/(l+1)
                    term = u[l] * p**l
                    s += term
                    if ctx.mag(term) < -tol:
                        return s, True
                    l += 1
                ctx.prec += cancellation//2
                return s, False
        if k == 0 or k == -1:
            return _lambertw_approx_hybrid(z, k), False
    if k == 0:
        if magz < -1:
            return z*(1-z), False
        L1 = ctx.ln(z)
        L2 = ctx.ln(L1)
    elif k == -1 and (not ctx._im(z)) and (-0.36787944117144 < ctx._re(z) < 0):
        L1 = ctx.ln(-z)
        return L1 - ctx.ln(-L1), False
    else:
        # This holds both as z -> 0 and z -> inf.
        # Relative error is O(1/log(z)).
        L1 = ctx.ln(z) + 2j*ctx.pi*k
        L2 = ctx.ln(L1)
    return L1 - L2 + L2/L1 + L2*(L2-2)/(2*L1**2), False

@defun
def lambertw(ctx, z, k=0):
    z = ctx.convert(z)
    k = int(k)
    if not ctx.isnormal(z):
        return _lambertw_special(ctx, z, k)
    prec = ctx.prec
    ctx.prec += 20 + ctx.mag(k or 1)
    wp = ctx.prec
    tol = wp - 5
    w, done = _lambertw_series(ctx, z, k, tol)
    if not done:
        # Use Halley iteration to solve w*exp(w) = z
        two = ctx.mpf(2)
        for i in xrange(100):
            ew = ctx.exp(w)
            wew = w*ew
            wewz = wew-z
            wn = w - wewz/(wew+ew-(w+two)*wewz/(two*w+two))
            if ctx.mag(wn-w) <= ctx.mag(wn) - tol:
                w = wn
                break
            else:
                w = wn
        if i == 100:
            ctx.warn("Lambert W iteration failed to converge for z = %s" % z)
    ctx.prec = prec
    return +w

@defun_wrapped
def bell(ctx, n, x=1):
    x = ctx.convert(x)
    if not n:
        if ctx.isnan(x):
            return x
        return type(x)(1)
    if ctx.isinf(x) or ctx.isinf(n) or ctx.isnan(x) or ctx.isnan(n):
        return x**n
    if n == 1: return x
    if n == 2: return x*(x+1)
    if x == 0: return ctx.sincpi(n)
    return _polyexp(ctx, n, x, True) / ctx.exp(x)

def _polyexp(ctx, n, x, extra=False):
    def _terms():
        if extra:
            yield ctx.sincpi(n)
        t = x
        k = 1
        while 1:
            yield k**n * t
            k += 1
            t = t*x/k
    return ctx.sum_accurately(_terms, check_step=4)

@defun_wrapped
def polyexp(ctx, s, z):
    if ctx.isinf(z) or ctx.isinf(s) or ctx.isnan(z) or ctx.isnan(s):
        return z**s
    if z == 0: return z*s
    if s == 0: return ctx.expm1(z)
    if s == 1: return ctx.exp(z)*z
    if s == 2: return ctx.exp(z)*z*(z+1)
    return _polyexp(ctx, s, z)

@defun_wrapped
def cyclotomic(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError("n cannot be negative")
    p = ctx.one
    if n == 0:
        return p
    if n == 1:
        return z - p
    if n == 2:
        return z + p
    # Use divisor product representation. Unfortunately, this sometimes
    # includes singularities for roots of unity, which we have to cancel out.
    # Matching zeros/poles pairwise, we have (1-z^a)/(1-z^b) ~ a/b + O(z-1).
    a_prod = 1
    b_prod = 1
    num_zeros = 0
    num_poles = 0
    for d in range(1,n+1):
        if not n % d:
            w = ctx.moebius(n//d)
            # Use powm1 because it is important that we get 0 only
            # if it really is exactly 0
            b = -ctx.powm1(z, d)
            if b:
                p *= b**w
            else:
                if w == 1:
                    a_prod *= d
                    num_zeros += 1
                elif w == -1:
                    b_prod *= d
                    num_poles += 1
    #print n, num_zeros, num_poles
    if num_zeros:
        if num_zeros > num_poles:
            p *= 0
        else:
            p *= a_prod
            p /= b_prod
    return p

@defun
def mangoldt(ctx, n):
    r"""
    Evaluates the von Mangoldt function `\Lambda(n) = \log p`
    if `n = p^k` a power of a prime, and `\Lambda(n) = 0` otherwise.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> [mangoldt(n) for n in range(-2,3)]
        [0.0, 0.0, 0.0, 0.0, 0.6931471805599453094172321]
        >>> mangoldt(6)
        0.0
        >>> mangoldt(7)
        1.945910149055313305105353
        >>> mangoldt(8)
        0.6931471805599453094172321
        >>> fsum(mangoldt(n) for n in range(101))
        94.04531122935739224600493
        >>> fsum(mangoldt(n) for n in range(10001))
        10013.39669326311478372032

    """
    n = int(n)
    if n < 2:
        return ctx.zero
    if n % 2 == 0:
        # Must be a power of two
        if n & (n-1) == 0:
            return +ctx.ln2
        else:
            return ctx.zero
    # TODO: the following could be generalized into a perfect
    # power testing function
    # ---
    # Look for a small factor
    for p in (3,5,7,11,13,17,19,23,29,31):
        if not n % p:
            q, r = n // p, 0
            while q > 1:
                q, r = divmod(q, p)
                if r:
                    return ctx.zero
            return ctx.ln(p)
    if ctx.isprime(n):
        return ctx.ln(n)
    # Obviously, we could use arbitrary-precision arithmetic for this...
    if n > 10**30:
        raise NotImplementedError
    k = 2
    while 1:
        p = int(n**(1./k) + 0.5)
        if p < 2:
            return ctx.zero
        if p ** k == n:
            if ctx.isprime(p):
                return ctx.ln(p)
        k += 1

@defun
def stirling1(ctx, n, k, exact=False):
    v = ctx._stirling1(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)

@defun
def stirling2(ctx, n, k, exact=False):
    v = ctx._stirling2(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)
<|endoftext|>from .functions import defun, defun_wrapped

@defun
def j0(ctx, x):
    """Computes the Bessel function `J_0(x)`. See :func:`~mpmath.besselj`."""
    return ctx.besselj(0, x)

@defun
def j1(ctx, x):
    """Computes the Bessel function `J_1(x)`.  See :func:`~mpmath.besselj`."""
    return ctx.besselj(1, x)

@defun
def besselj(ctx, n, z, derivative=0, **kwargs):
    if type(n) is int:
        n_isint = True
    else:
        n = ctx.convert(n)
        n_isint = ctx.isint(n)
        if n_isint:
            n = int(ctx._re(n))
    if n_isint and n < 0:
        return (-1)**n * ctx.besselj(-n, z, derivative, **kwargs)
    z = ctx.convert(z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        # TODO: the integer special-casing shouldn't be necessary.
        # However, the hypergeometric series gets inaccurate for large d
        # because of inaccurate pole cancellation at a pole far from
        # zero (needs to be fixed in hypercomb or hypsum)
        if ctx.isint(d) and d >= 0:
            d = int(d)
            orig = ctx.prec
            try:
                ctx.prec += 15
                v = ctx.fsum((-1)**k * ctx.binomial(d,k) * ctx.besselj(2*k+n-d,z)
                    for k in range(d+1))
            finally:
                ctx.prec = orig
            v *= ctx.mpf(2)**(-d)
        else:
            def h(n,d):
                r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec+M), -0.25, exact=True)
                B = [0.5*(n-d+1), 0.5*(n-d+2)]
                T = [([2,ctx.pi,z],[d-2*n,0.5,n-d],[],B,[(n+1)*0.5,(n+2)*0.5],B+[n+1],r)]
                return T
            v = ctx.hypercomb(h, [n,d], **kwargs)
    else:
        # Fast case: J_n(x), n int, appropriate magnitude for fixed-point calculation
        if (not derivative) and n_isint and abs(M) < 10 and abs(n) < 20:
            try:
                return ctx._besselj(n, z)
            except NotImplementedError:
                pass
        if not z:
            if not n:
                v = ctx.one + n+z
            elif ctx.re(n) > 0:
                v = n*z
            else:
                v = ctx.inf + z + n
        else:
            #v = 0
            orig = ctx.prec
            try:
                # XXX: workaround for accuracy in low level hypergeometric series
                # when alternating, large arguments
                ctx.prec += min(3*abs(M), ctx.prec)
                w = ctx.fmul(z, 0.5, exact=True)
                def h(n):
                    r = ctx.fneg(ctx.fmul(w, w, prec=max(0,ctx.prec+M)), exact=True)
                    return [([w], [n], [], [n+1], [], [n+1], r)]
                v = ctx.hypercomb(h, [n], **kwargs)
            finally:
                ctx.prec = orig
        v = +v
    return v

@defun
def besseli(ctx, n, z, derivative=0, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    if not z:
        if derivative:
            raise ValueError
        if not n:
            # I(0,0) = 1
            return 1+n+z
        if ctx.isint(n):
            return 0*(n+z)
        r = ctx.re(n)
        if r == 0:
            return ctx.nan*(n+z)
        elif r > 0:
            return 0*(n+z)
        else:
            return ctx.inf+(n+z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        def h(n,d):
            r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec+M), 0.25, exact=True)
            B = [0.5*(n-d+1), 0.5*(n-d+2), n+1]
            T = [([2,ctx.pi,z],[d-2*n,0.5,n-d],[n+1],B,[(n+1)*0.5,(n+2)*0.5],B,r)]
            return T
        v = ctx.hypercomb(h, [n,d], **kwargs)
    else:
        def h(n):
            w = ctx.fmul(z, 0.5, exact=True)
            r = ctx.fmul(w, w, prec=max(0,ctx.prec+M))
            return [([w], [n], [], [n+1], [], [n+1], r)]
        v = ctx.hypercomb(h, [n], **kwargs)
    return v

@defun_wrapped
def bessely(ctx, n, z, derivative=0, **kwargs):
    if not z:
        if derivative:
            # Not implemented
            raise ValueError
        if not n:
            # ~ log(z/2)
            return -ctx.inf + (n+z)
        if ctx.im(n):
            return ctx.nan * (n+z)
        r = ctx.re(n)
        q = n+0.5
        if ctx.isint(q):
            if n > 0:
                return -ctx.inf + (n+z)
            else:
                return 0 * (n+z)
        if r < 0 and int(ctx.floor(q)) % 2:
            return ctx.inf + (n+z)
        else:
            return ctx.ninf + (n+z)
    # XXX: use hypercomb
    ctx.prec += 10
    m, d = ctx.nint_distance(n)
    if d < -ctx.prec:
        h = +ctx.eps
        ctx.prec *= 2
        n += h
    elif d < 0:
        ctx.prec -= d
    # TODO: avoid cancellation for imaginary arguments
    cos, sin = ctx.cospi_sinpi(n)
    return (ctx.besselj(n,z,derivative,**kwargs)*cos - \
        ctx.besselj(-n,z,derivative,**kwargs))/sin

@defun_wrapped
def besselk(ctx, n, z, **kwargs):
    if not z:
        return ctx.inf
    M = ctx.mag(z)
    if M < 1:
        # Represent as limit definition
        def h(n):
            r = (z/2)**2
            T1 = [z, 2], [-n, n-1], [n], [], [], [1-n], r
            T2 = [z, 2], [n, -n-1], [-n], [], [], [1+n], r
            return T1, T2
    # We could use the limit definition always, but it leads
    # to very bad cancellation (of exponentially large terms)
    # for large real z
    # Instead represent in terms of 2F0
    else:
        ctx.prec += M
        def h(n):
            return [([ctx.pi/2, z, ctx.exp(-z)], [0.5,-0.5,1], [], [], \
                [n+0.5, 0.5-n], [], -1/(2*z))]
    return ctx.hypercomb(h, [n], **kwargs)

@defun_wrapped
def hankel1(ctx,n,x,**kwargs):
    return ctx.besselj(n,x,**kwargs) + ctx.j*ctx.bessely(n,x,**kwargs)

@defun_wrapped
def hankel2(ctx,n,x,**kwargs):
    return ctx.besselj(n,x,**kwargs) - ctx.j*ctx.bessely(n,x,**kwargs)

@defun_wrapped
def whitm(ctx,k,m,z,**kwargs):
    if z == 0:
        # M(k,m,z) = 0^(1/2+m)
        if ctx.re(m) > -0.5:
            return z
        elif ctx.re(m) < -0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5+m
    return ctx.exp(x) * z**y * ctx.hyp1f1(y-k, 1+2*m, z, **kwargs)

@defun_wrapped
def whitw(ctx,k,m,z,**kwargs):
    if z == 0:
        g = abs(ctx.re(m))
        if g < 0.5:
            return z
        elif g > 0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5+m
    return ctx.exp(x) * z**y * ctx.hyperu(y-k, 1+2*m, z, **kwargs)

@defun
def hyperu(ctx, a, b, z, **kwargs):
    a, atype = ctx._convert_param(a)
    b, btype = ctx._convert_param(b)
    z = ctx.convert(z)
    if not z:
        if ctx.re(b) <= 1:
            return ctx.gammaprod([1-b],[a-b+1])
        else:
            return ctx.inf + z
    bb = 1+a-b
    bb, bbtype = ctx._convert_param(bb)
    try:
        orig = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.hypsum(2, 0, (atype, bbtype), [a, bb], -1/z, maxterms=ctx.prec)
            return v / z**a
        finally:
            ctx.prec = orig
    except ctx.NoConvergence:
        pass
    def h(a,b):
        w = ctx.sinpi(b)
        T1 = ([ctx.pi,w],[1,-1],[],[a-b+1,b],[a],[b],z)
        T2 = ([-ctx.pi,w,z],[1,-1,1-b],[],[a,2-b],[a-b+1],[2-b],z)
        return T1, T2
    return ctx.hypercomb(h, [a,b], **kwargs)

@defun
def struveh(ctx,n,z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/StruveH/26/01/02/
    def h(n):
        return [([z/2, 0.5*ctx.sqrt(ctx.pi)], [n+1, -1], [], [n+1.5], [1], [1.5, n+1.5], -(z/2)**2)]
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def struvel(ctx,n,z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/StruveL/26/01/02/
    def h(n):
        return [([z/2, 0.5*ctx.sqrt(ctx.pi)], [n+1, -1], [], [n+1.5], [1], [1.5, n+1.5], (z/2)**2)]
    return ctx.hypercomb(h, [n], **kwargs)

def _anger(ctx,which,v,z,**kwargs):
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    def h(v):
        b = ctx.mpq_1_2
        u = v*b
        m = b*3
        a1,a2,b1,b2 = m-u, m+u, 1-u, 1+u
        c, s = ctx.cospi_sinpi(u)
        if which == 0:
            A, B = [b*z, s], [c]
        if which == 1:
            A, B = [b*z, -c], [s]
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = A, [1, 1], [], [a1,a2], [1], [a1,a2], w
        T2 = B, [1], [], [b1,b2], [1], [b1,b2], w
        return T1, T2
    return ctx.hypercomb(h, [v], **kwargs)

@defun
def angerj(ctx, v, z, **kwargs):
    return _anger(ctx, 0, v, z, **kwargs)

@defun
def webere(ctx, v, z, **kwargs):
    return _anger(ctx, 1, v, z, **kwargs)

@defun
def lommels1(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    def h(u,v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        return ([u-v+1, u+v+1, z], [-1, -1, u+1], [], [], [1], \
            [b*(u-v+3),b*(u+v+3)], w),
    return ctx.hypercomb(h, [u,v], **kwargs)

@defun
def lommels2(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    # Asymptotic expansion (GR p. 947) -- need to be careful
    # not to use for small arguments
    # def h(u,v):
    #    b = ctx.mpq_1_2
    #    w = -(z/2)**(-2)
    #    return ([z], [u-1], [], [], [b*(1-u+v)], [b*(1-u-v)], w),
    def h(u,v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = [u-v+1, u+v+1, z], [-1, -1, u+1], [], [], [1], [b*(u-v+3),b*(u+v+3)], w
        T2 = [2, z], [u+v-1, -v], [v, b*(u+v+1)], [b*(v-u+1)], [], [1-v], w
        T3 = [2, z], [u-v-1, v], [-v, b*(u-v+1)], [b*(1-u-v)], [], [1+v], w
        #c1 = ctx.cospi((u-v)*b)
        #c2 = ctx.cospi((u+v)*b)
        #s = ctx.sinpi(v)
        #r1 = (u-v+1)*b
        #r2 = (u+v+1)*b
        #T2 = [c1, s, z, 2], [1, -1, -v, v], [], [-v+1], [], [-v+1], w
        #T3 = [-c2, s, z, 2], [1, -1, v, -v], [], [v+1], [], [v+1], w
        #T2 = [c1, s, z, 2], [1, -1, -v, v+u-1], [r1, r2], [-v+1], [], [-v+1], w
        #T3 = [-c2, s, z, 2], [1, -1, v, -v+u-1], [r1, r2], [v+1], [], [v+1], w
        return T1, T2, T3
    return ctx.hypercomb(h, [u,v], **kwargs)

@defun
def ber(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinBer2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos, sin = ctx.cospi_sinpi(-0.75*n)
        T1 = [cos, z/2], [1, n], [], [n+1], [], [0.5, 0.5*(n+1), 0.5*n+1], r
        T2 = [sin, z/2], [1, n+2], [], [n+2], [], [1.5, 0.5*(n+3), 0.5*n+1], r
        return T1, T2
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def bei(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinBei2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos, sin = ctx.cospi_sinpi(0.75*n)
        T1 = [cos, z/2], [1, n+2], [], [n+2], [], [1.5, 0.5*(n+3), 0.5*n+1], r
        T2 = [sin, z/2], [1, n], [], [n+1], [], [0.5, 0.5*(n+1), 0.5*n+1], r
        return T1, T2
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def ker(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinKer2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos1, sin1 = ctx.cospi_sinpi(0.25*n)
        cos2, sin2 = ctx.cospi_sinpi(0.75*n)
        T1 = [2, z, 4*cos1], [-n-3, n, 1], [-n], [], [], [0.5, 0.5*(1+n), 0.5*(n+2)], r
        T2 = [2, z, -sin1], [-n-3, 2+n, 1], [-n-1], [], [], [1.5, 0.5*(3+n), 0.5*(n+2)], r
        T3 = [2, z, 4*cos2], [n-3, -n, 1], [n], [], [], [0.5, 0.5*(1-n), 1-0.5*n], r
        T4 = [2, z, -sin2], [n-3, 2-n, 1], [n-1], [], [], [1.5, 0.5*(3-n), 1-0.5*n], r
        return T1, T2, T3, T4
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def kei(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinKei2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos1, sin1 = ctx.cospi_sinpi(0.75*n)
        cos2, sin2 = ctx.cospi_sinpi(0.25*n)
        T1 = [-cos1, 2, z], [1, n-3, 2-n], [n-1], [], [], [1.5, 0.5*(3-n), 1-0.5*n], r
        T2 = [-sin1, 2, z], [1, n-1, -n], [n], [], [], [0.5, 0.5*(1-n), 1-0.5*n], r
        T3 = [-sin2, 2, z], [1, -n-1, n], [-n], [], [], [0.5, 0.5*(n+1), 0.5*(n+2)], r
        T4 = [-cos2, 2, z], [1, -n-3, n+2], [-n-1], [], [], [1.5, 0.5*(n+3), 0.5*(n+2)], r
        return T1, T2, T3, T4
    return ctx.hypercomb(h, [n], **kwargs)

# TODO: do this more generically?
def c_memo(f):
    name = f.__name__
    def f_wrapped(ctx):
        cache = ctx._misc_const_cache
        prec = ctx.prec
        p,v = cache.get(name, (-1,0))
        if p >= prec:
            return +v
        else:
            cache[name] = (prec, f(ctx))
            return cache[name][1]
    return f_wrapped

@c_memo
def _airyai_C1(ctx):
    return 1 / (ctx.cbrt(9) * ctx.gamma(ctx.mpf(2)/3))

@c_memo
def _airyai_C2(ctx):
    return -1 / (ctx.cbrt(3) * ctx.gamma(ctx.mpf(1)/3))

@c_memo
def _airybi_C1(ctx):
    return 1 / (ctx.nthroot(3,6) * ctx.gamma(ctx.mpf(2)/3))

@c_memo
def _airybi_C2(ctx):
    return ctx.nthroot(3,6) / ctx.gamma(ctx.mpf(1)/3)

def _airybi_n2_inf(ctx):
    prec = ctx.prec
    try:
        v = ctx.power(3,'2/3')*ctx.gamma('2/3')/(2*ctx.pi)
    finally:
        ctx.prec = prec
    return +v

# Derivatives at z = 0
# TODO: could be expressed more elegantly using triple factorials
def _airyderiv_0(ctx, z, n, ntype, which):
    if ntype == 'Z':
        if n < 0:
            return z
        r = ctx.mpq_1_3
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.gamma((n+1)*r) * ctx.power(3,n*r) / ctx.pi
            if which == 0:
                v *= ctx.sinpi(2*(n+1)*r)
                v /= ctx.power(3,'2/3')
            else:
                v *= abs(ctx.sinpi(2*(n+1)*r))
                v /= ctx.power(3,'1/6')
        finally:
            ctx.prec = prec
        return +v + z
    else:
        # singular (does the limit exist?)
        raise NotImplementedError

@defun
def airyai(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    # Values at infinities
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if n == -1:
                if z == ctx.inf:
                    return ctx.mpf(1)/3 + 1/z
                if z == ctx.ninf:
                    return ctx.mpf(-2)/3 + 1/z
            if n < -1:
                if z == ctx.inf:
                    return z
                if z == ctx.ninf:
                    return (-1)**n * (-z)
        if (not n) and z == ctx.inf or z == ctx.ninf:
            return 1/z
        # TODO: limits
        raise ValueError("essential singularity of Ai(z)")
    # Account for exponential scaling
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:
            def h():
                # http://functions.wolfram.com/03.07.06.0005.01
                if ctx._re(z) > 4:
                    ctx.prec += extraprec
                    w = z**1.5; r = -0.75/w; u = -2*w/3
                    ctx.prec -= extraprec
                    C = -ctx.exp(u)/(2*ctx.sqrt(ctx.pi))*ctx.nthroot(z,4)
                    return ([C],[1],[],[],[(-1,6),(7,6)],[],r),
                # http://functions.wolfram.com/03.07.26.0001.01
                else:
                    ctx.prec += extraprec
                    w = z**3 / 9
                    ctx.prec -= extraprec
                    C1 = _airyai_C1(ctx) * 0.5
                    C2 = _airyai_C2(ctx)
                    T1 = [C1,z],[1,2],[],[],[],[ctx.mpq_5_3],w
                    T2 = [C2],[1],[],[],[],[ctx.mpq_1_3],w
                    return T1, T2
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 0)
            # http://functions.wolfram.com/03.05.20.0004.01
            def h(n):
                ctx.prec += extraprec
                w = z**3/9
                ctx.prec -= extraprec
                q13,q23,q43 = ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3
                a1=q13; a2=1; b1=(1-n)*q13; b2=(2-n)*q13; b3=1-n*q13
                T1 = [3, z], [n-q23, -n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                a1=q23; b1=(2-n)*q13; b2=1-n*q13; b3=(4-n)*q13
                T2 = [3, z, -z], [n-q43, -n, 1], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                return T1, T2
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:
        def h():
            if ctx._re(z) > 4:
                # We could use 1F1, but it results in huge cancellation;
                # the following expansion is better.
                # TODO: asymptotic series for derivatives
                ctx.prec += extraprec
                w = z**1.5; r = -0.75/w; u = -2*w/3
                ctx.prec -= extraprec
                C = ctx.exp(u)/(2*ctx.sqrt(ctx.pi)*ctx.nthroot(z,4))
                return ([C],[1],[],[],[(1,6),(5,6)],[],r),
            else:
                ctx.prec += extraprec
                w = z**3 / 9
                ctx.prec -= extraprec
                C1 = _airyai_C1(ctx)
                C2 = _airyai_C2(ctx)
                T1 = [C1],[1],[],[],[],[ctx.mpq_2_3],w
                T2 = [z*C2],[1],[],[],[],[ctx.mpq_4_3],w
                return T1, T2
        return ctx.hypercomb(h, [], **kwargs)

@defun
def airybi(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    # Values at infinities
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if z == ctx.inf:
                return z
            if z == ctx.ninf:
                if n == -1:
                    return 1/z
                if n == -2:
                    return _airybi_n2_inf(ctx)
                if n < -2:
                    return (-1)**n * (-z)
        if not n:
            if z == ctx.inf:
                return z
            if z == ctx.ninf:
                return 1/z
        # TODO: limits
        raise ValueError("essential singularity of Bi(z)")
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:
            # http://functions.wolfram.com/03.08.26.0001.01
            def h():
                ctx.prec += extraprec
                w = z**3 / 9
                ctx.prec -= extraprec
                C1 = _airybi_C1(ctx)*0.5
                C2 = _airybi_C2(ctx)
                T1 = [C1,z],[1,2],[],[],[],[ctx.mpq_5_3],w
                T2 = [C2],[1],[],[],[],[ctx.mpq_1_3],w
                return T1, T2
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 1)
            def h(n):
                ctx.prec += extraprec
                w = z**3/9
                ctx.prec -= extraprec
                q13,q23,q43 = ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3
                q16 = ctx.mpq_1_6
                q56 = ctx.mpq_5_6
                a1=q13; a2=1; b1=(1-n)*q13; b2=(2-n)*q13; b3=1-n*q13
                T1 = [3, z], [n-q16, -n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                a1=q23; b1=(2-n)*q13; b2=1-n*q13; b3=(4-n)*q13
                T2 = [3, z], [n-q56, 1-n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                return T1, T2
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:
        def h():
            ctx.prec += extraprec
            w = z**3 / 9
            ctx.prec -= extraprec
            C1 = _airybi_C1(ctx)
            C2 = _airybi_C2(ctx)
            T1 = [C1],[1],[],[],[],[ctx.mpq_2_3],w
            T2 = [z*C2],[1],[],[],[],[ctx.mpq_4_3],w
            return T1, T2
        return ctx.hypercomb(h, [], **kwargs)

def _airy_zero(ctx, which, k, derivative, complex=False):
    # Asymptotic formulas are given in DLMF section 9.9
    def U(t): return t**(2/3.)*(1-7/(t**2*48))
    def T(t): return t**(2/3.)*(1+5/(t**2*48))
    k = int(k)
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if not derivative in (0,1):
        raise ValueError("Derivative should lie between 0 and 1")
    if which == 0:
        if derivative:
            return ctx.findroot(lambda z: ctx.airyai(z,1),
                -U(3*ctx.pi*(4*k-3)/8))
        return ctx.findroot(ctx.airyai, -T(3*ctx.pi*(4*k-1)/8))
    if which == 1 and complex == False:
        if derivative:
            return ctx.findroot(lambda z: ctx.airybi(z,1),
                -U(3*ctx.pi*(4*k-1)/8))
        return ctx.findroot(ctx.airybi, -T(3*ctx.pi*(4*k-3)/8))
    if which == 1 and complex == True:
        if derivative:
            t = 3*ctx.pi*(4*k-3)/8 + 0.75j*ctx.ln2
            s = ctx.expjpi(ctx.mpf(1)/3) * T(t)
            return ctx.findroot(lambda z: ctx.airybi(z,1), s)
        t = 3*ctx.pi*(4*k-1)/8 + 0.75j*ctx.ln2
        s = ctx.expjpi(ctx.mpf(1)/3) * U(t)
        return ctx.findroot(ctx.airybi, s)

@defun
def airyaizero(ctx, k, derivative=0):
    return _airy_zero(ctx, 0, k, derivative, False)

@defun
def airybizero(ctx, k, derivative=0, complex=False):
    return _airy_zero(ctx, 1, k, derivative, complex)

def _scorer(ctx, z, which, kwargs):
    z = ctx.convert(z)
    if ctx.isinf(z):
        if z == ctx.inf:
            if which == 0: return 1/z
            if which == 1: return z
        if z == ctx.ninf:
            return 1/z
        raise ValueError("essential singularity")
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if kwargs.get('derivative'):
        raise NotImplementedError
    # Direct asymptotic expansions, to avoid
    # exponentially large cancellation
    try:
        if ctx.mag(z) > 3:
            if which == 0 and abs(ctx.arg(z)) < ctx.pi/3 * 0.999:
                def h():
                    return (([ctx.pi,z],[-1,-1],[],[],[(1,3),(2,3),1],[],9/z**3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
            if which == 1 and abs(ctx.arg(-z)) < 2*ctx.pi/3 * 0.999:
                def h():
                    return (([-ctx.pi,z],[-1,-1],[],[],[(1,3),(2,3),1],[],9/z**3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
    except ctx.NoConvergence:
        pass
    def h():
        A = ctx.airybi(z, **kwargs)/3
        B = -2*ctx.pi
        if which == 1:
            A *= 2
            B *= -1
        ctx.prec += extraprec
        w = z**3/9
        ctx.prec -= extraprec
        T1 = [A], [1], [], [], [], [], 0
        T2 = [B,z], [-1,2], [], [], [1], [ctx.mpq_4_3,ctx.mpq_5_3], w
        return T1, T2
    return ctx.hypercomb(h, [], **kwargs)

@defun
def scorergi(ctx, z, **kwargs):
    return _scorer(ctx, z, 0, kwargs)

@defun
def scorerhi(ctx, z, **kwargs):
    return _scorer(ctx, z, 1, kwargs)

@defun_wrapped
def coulombc(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l,eta][0] >= ctx.prec:
        return +_cache[l,eta][1]
    G3 = ctx.loggamma(2*l+2)
    G1 = ctx.loggamma(1+l+ctx.j*eta)
    G2 = ctx.loggamma(1+l-ctx.j*eta)
    v = 2**l * ctx.exp((-ctx.pi*eta+G1+G2)/2 - G3)
    if not (ctx.im(l) or ctx.im(eta)):
        v = ctx.re(v)
    _cache[l,eta] = (ctx.prec, v)
    return v

@defun_wrapped
def coulombf(ctx, l, eta, z, w=1, chop=True, **kwargs):
    # Regular Coulomb wave function
    # Note: w can be either 1 or -1; the other may be better in some cases
    # TODO: check that chop=True chops when and only when it should
    #ctx.prec += 10
    def h(l, eta):
        try:
            jw = ctx.j*w
            jwz = ctx.fmul(jw, z, exact=True)
            jwz2 = ctx.fmul(jwz, -2, exact=True)
            C = ctx.coulombc(l, eta)
            T1 = [C, z, ctx.exp(jwz)], [1, l+1, 1], [], [], [1+l+jw*eta], \
                [2*l+2], jwz2
        except ValueError:
            T1 = [0], [-1], [], [], [], [], 0
        return (T1,)
    v = ctx.hypercomb(h, [l,eta], **kwargs)
    if chop and (not ctx.im(l)) and (not ctx.im(eta)) and (not ctx.im(z)) and \
        (ctx.re(z) >= 0):
        v = ctx.re(v)
    return v

@defun_wrapped
def _coulomb_chi(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l,eta][0] >= ctx.prec:
        return _cache[l,eta][1]
    def terms():
        l2 = -l-1
        jeta = ctx.j*eta
        return [ctx.loggamma(1+l+jeta) * (-0.5j),
            ctx.loggamma(1+l-jeta) * (0.5j),
            ctx.loggamma(1+l2+jeta) * (0.5j),
            ctx.loggamma(1+l2-jeta) * (-0.5j),
            -(l+0.5)*ctx.pi]
    v = ctx.sum_accurately(terms, 1)
    _cache[l,eta] = (ctx.prec, v)
    return v

@defun_wrapped
def coulombg(ctx, l, eta, z, w=1, chop=True, **kwargs):
    # Irregular Coulomb wave function
    # Note: w can be either 1 or -1; the other may be better in some cases
    # TODO: check that chop=True chops when and only when it should
    if not ctx._im(l):
        l = ctx._re(l)  # XXX: for isint
    def h(l, eta):
        # Force perturbation for integers and half-integers
        if ctx.isint(l*2):
            T1 = [0], [-1], [], [], [], [], 0
            return (T1,)
        l2 = -l-1
        try:
            chi = ctx._coulomb_chi(l, eta)
            jw = ctx.j*w
            s = ctx.sin(chi); c = ctx.cos(chi)
            C1 = ctx.coulombc(l,eta)
            C2 = ctx.coulombc(l2,eta)
            u = ctx.exp(jw*z)
            x = -2*jw*z
            T1 = [s, C1, z, u, c], [-1, 1, l+1, 1, 1], [], [], \
                [1+l+jw*eta], [2*l+2], x
            T2 = [-s, C2, z, u],   [-1, 1, l2+1, 1],    [], [], \
                [1+l2+jw*eta], [2*l2+2], x
            return T1, T2
        except ValueError:
            T1 = [0], [-1], [], [], [], [], 0
            return (T1,)
    v = ctx.hypercomb(h, [l,eta], **kwargs)
    if chop and (not ctx._im(l)) and (not ctx._im(eta)) and (not ctx._im(z)) and \
        (ctx._re(z) >= 0):
        v = ctx._re(v)
    return v

def mcmahon(ctx,kind,prime,v,m):
    """
    Computes an estimate for the location of the Bessel function zero
    j_{v,m}, y_{v,m}, j'_{v,m} or y'_{v,m} using McMahon's asymptotic
    expansion (Abramowitz & Stegun 9.5.12-13, DLMF 20.21(vi)).

    Returns (r,err) where r is the estimated location of the root
    and err is a positive number estimating the error of the
    asymptotic expansion.
    """
    u = 4*v**2
    if kind == 1 and not prime: b = (4*m+2*v-1)*ctx.pi/4
    if kind == 2 and not prime: b = (4*m+2*v-3)*ctx.pi/4
    if kind == 1 and prime: b = (4*m+2*v-3)*ctx.pi/4
    if kind == 2 and prime: b = (4*m+2*v-1)*ctx.pi/4
    if not prime:
        s1 = b
        s2 = -(u-1)/(8*b)
        s3 = -4*(u-1)*(7*u-31)/(3*(8*b)**3)
        s4 = -32*(u-1)*(83*u**2-982*u+3779)/(15*(8*b)**5)
        s5 = -64*(u-1)*(6949*u**3-153855*u**2+1585743*u-6277237)/(105*(8*b)**7)
    if prime:
        s1 = b
        s2 = -(u+3)/(8*b)
        s3 = -4*(7*u**2+82*u-9)/(3*(8*b)**3)
        s4 = -32*(83*u**3+2075*u**2-3039*u+3537)/(15*(8*b)**5)
        s5 = -64*(6949*u**4+296492*u**3-1248002*u**2+7414380*u-5853627)/(105*(8*b)**7)
    terms = [s1,s2,s3,s4,s5]
    s = s1
    err = 0.0
    for i in range(1,len(terms)):
        if abs(terms[i]) < abs(terms[i-1]):
            s += terms[i]
        else:
            err = abs(terms[i])
    if i == len(terms)-1:
        err = abs(terms[-1])
    return s, err

def generalized_bisection(ctx,f,a,b,n):
    """
    Given f known to have exactly n simple roots within [a,b],
    return a list of n intervals isolating the roots
    and having opposite signs at the endpoints.

    TODO: this can be optimized, e.g. by reusing evaluation points.
    """
    if n < 1:
        raise ValueError("n cannot be less than 1")
    N = n+1
    points = []
    signs = []
    while 1:
        points = ctx.linspace(a,b,N)
        signs = [ctx.sign(f(x)) for x in points]
        ok_intervals = [(points[i],points[i+1]) for i in range(N-1) \
            if signs[i]*signs[i+1] == -1]
        if len(ok_intervals) == n:
            return ok_intervals
        N = N*2

def find_in_interval(ctx, f, ab):
    return ctx.findroot(f, ab, solver='illinois', verify=False)

def bessel_zero(ctx, kind, prime, v, m, isoltol=0.01, _interval_cache={}):
    prec = ctx.prec
    workprec = max(prec, ctx.mag(v), ctx.mag(m))+10
    try:
        ctx.prec = workprec
        v = ctx.mpf(v)
        m = int(m)
        prime = int(prime)
        if v < 0:
            raise ValueError("v cannot be negative")
        if m < 1:
            raise ValueError("m cannot be less than 1")
        if not prime in (0,1):
            raise ValueError("prime should lie between 0 and 1")
        if kind == 1:
            if prime: f = lambda x: ctx.besselj(v,x,derivative=1)
            else:     f = lambda x: ctx.besselj(v,x)
        if kind == 2:
            if prime: f = lambda x: ctx.bessely(v,x,derivative=1)
            else:     f = lambda x: ctx.bessely(v,x)
        # The first root of J' is very close to 0 for small
        # orders, and this needs to be special-cased
        if kind == 1 and prime and m == 1:
            if v == 0:
                return ctx.zero
            if v <= 1:
                # TODO: use v <= j'_{v,1} < y_{v,1}?
                r = 2*ctx.sqrt(v*(1+v)/(v+2))
                return find_in_interval(ctx, f, (r/10, 2*r))
        if (kind,prime,v,m) in _interval_cache:
            return find_in_interval(ctx, f, _interval_cache[kind,prime,v,m])
        r, err = mcmahon(ctx, kind, prime, v, m)
        if err < isoltol:
            return find_in_interval(ctx, f, (r-isoltol, r+isoltol))
        # An x such that 0 < x < r_{v,1}
        if kind == 1 and not prime: low = 2.4
        if kind == 1 and prime: low = 1.8
        if kind == 2 and not prime: low = 0.8
        if kind == 2 and prime: low = 2.0
        n = m+1
        while 1:
            r1, err = mcmahon(ctx, kind, prime, v, n)
            if err < isoltol:
                r2, err2 = mcmahon(ctx, kind, prime, v, n+1)
                intervals = generalized_bisection(ctx, f, low, 0.5*(r1+r2), n)
                for k, ab in enumerate(intervals):
                    _interval_cache[kind,prime,v,k+1] = ab
                return find_in_interval(ctx, f, intervals[m-1])
            else:
                n = n*2
    finally:
        ctx.prec = prec

@defun
def besseljzero(ctx, v, m, derivative=0):
    r"""
    For a real order `\nu \ge 0` and a positive integer `m`, returns
    `j_{\nu,m}`, the `m`-th positive zero of the Bessel function of the
    first kind `J_{\nu}(z)` (see :func:`~mpmath.besselj`). Alternatively,
    with *derivative=1*, gives the first nonnegative simple zero
    `j'_{\nu,m}` of `J'_{\nu}(z)`.

    The indexing convention is that used by Abramowitz & Stegun
    and the DLMF. Note the special case `j'_{0,1} = 0`, while all other
    zeros are positive. In effect, only simple zeros are counted
    (all zeros of Bessel functions are simple except possibly `z = 0`)
    and `j_{\nu,m}` becomes a monotonic function of both `\nu`
    and `m`.

    The zeros are interlaced according to the inequalities

    .. math ::

        j'_{\nu,k} < j_{\nu,k} < j'_{\nu,k+1}

        j_{\nu,1} < j_{\nu+1,2} < j_{\nu,2} < j_{\nu+1,2} < j_{\nu,3} < \cdots

    **Examples**

    Initial zeros of the Bessel functions `J_0(z), J_1(z), J_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besseljzero(0,1); besseljzero(0,2); besseljzero(0,3)
        2.404825557695772768621632
        5.520078110286310649596604
        8.653727912911012216954199
        >>> besseljzero(1,1); besseljzero(1,2); besseljzero(1,3)
        3.831705970207512315614436
        7.01558666981561875353705
        10.17346813506272207718571
        >>> besseljzero(2,1); besseljzero(2,2); besseljzero(2,3)
        5.135622301840682556301402
        8.417244140399864857783614
        11.61984117214905942709415

    Initial zeros of `J'_0(z), J'_1(z), J'_2(z)`::

        0.0
        3.831705970207512315614436
        7.01558666981561875353705
        >>> besseljzero(1,1,1); besseljzero(1,2,1); besseljzero(1,3,1)
        1.84118378134065930264363
        5.331442773525032636884016
        8.536316366346285834358961
        >>> besseljzero(2,1,1); besseljzero(2,2,1); besseljzero(2,3,1)
        3.054236928227140322755932
        6.706133194158459146634394
        9.969467823087595793179143

    Zeros with large index::

        >>> besseljzero(0,100); besseljzero(0,1000); besseljzero(0,10000)
        313.3742660775278447196902
        3140.807295225078628895545
        31415.14114171350798533666
        >>> besseljzero(5,100); besseljzero(5,1000); besseljzero(5,10000)
        321.1893195676003157339222
        3148.657306813047523500494
        31422.9947255486291798943
        >>> besseljzero(0,100,1); besseljzero(0,1000,1); besseljzero(0,10000,1)
        311.8018681873704508125112
        3139.236339643802482833973
        31413.57032947022399485808

    Zeros of functions with large order::

        >>> besseljzero(50,1)
        57.11689916011917411936228
        >>> besseljzero(50,2)
        62.80769876483536093435393
        >>> besseljzero(50,100)
        388.6936600656058834640981
        >>> besseljzero(50,1,1)
        52.99764038731665010944037
        >>> besseljzero(50,2,1)
        60.02631933279942589882363
        >>> besseljzero(50,100,1)
        387.1083151608726181086283

    Zeros of functions with fractional order::

        >>> besseljzero(0.5,1); besseljzero(1.5,1); besseljzero(2.25,4)
        3.141592653589793238462643
        4.493409457909064175307881
        15.15657692957458622921634

    Both `J_{\nu}(z)` and `J'_{\nu}(z)` can be expressed as infinite
    products over their zeros::

        >>> v,z = 2, mpf(1)
        >>> (z/2)**v/gamma(v+1) * \
        ...     nprod(lambda k: 1-(z/besseljzero(v,k))**2, [1,inf])
        ...
        0.1149034849319004804696469
        >>> besselj(v,z)
        0.1149034849319004804696469
        >>> (z/2)**(v-1)/2/gamma(v) * \
        ...     nprod(lambda k: 1-(z/besseljzero(v,k,1))**2, [1,inf])
        ...
        0.2102436158811325550203884
        >>> besselj(v,z,1)
        0.2102436158811325550203884

    """
    return +bessel_zero(ctx, 1, derivative, v, m)

@defun
def besselyzero(ctx, v, m, derivative=0):
    r"""
    For a real order `\nu \ge 0` and a positive integer `m`, returns
    `y_{\nu,m}`, the `m`-th positive zero of the Bessel function of the
    second kind `Y_{\nu}(z)` (see :func:`~mpmath.bessely`). Alternatively,
    with *derivative=1*, gives the first positive zero `y'_{\nu,m}` of
    `Y'_{\nu}(z)`.

    The zeros are interlaced according to the inequalities

    .. math ::

        y_{\nu,k} < y'_{\nu,k} < y_{\nu,k+1}

        y_{\nu,1} < y_{\nu+1,2} < y_{\nu,2} < y_{\nu+1,2} < y_{\nu,3} < \cdots

    **Examples**

    Initial zeros of the Bessel functions `Y_0(z), Y_1(z), Y_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besselyzero(0,1); besselyzero(0,2); besselyzero(0,3)
        0.8935769662791675215848871
        3.957678419314857868375677
        7.086051060301772697623625
        >>> besselyzero(1,1); besselyzero(1,2); besselyzero(1,3)
        2.197141326031017035149034
        5.429681040794135132772005
        8.596005868331168926429606
        >>> besselyzero(2,1); besselyzero(2,2); besselyzero(2,3)
        3.384241767149593472701426
        6.793807513268267538291167
        10.02347797936003797850539

    Initial zeros of `Y'_0(z), Y'_1(z), Y'_2(z)`::

        >>> besselyzero(0,1,1); besselyzero(0,2,1); besselyzero(0,3,1)
        2.197141326031017035149034
        5.429681040794135132772005
        8.596005868331168926429606
        >>> besselyzero(1,1,1); besselyzero(1,2,1); besselyzero(1,3,1)
        3.683022856585177699898967
        6.941499953654175655751944
        10.12340465543661307978775
        >>> besselyzero(2,1,1); besselyzero(2,2,1); besselyzero(2,3,1)
        5.002582931446063945200176
        8.350724701413079526349714
        11.57419546521764654624265

    Zeros with large index::

        >>> besselyzero(0,100); besselyzero(0,1000); besselyzero(0,10000)
        311.8034717601871549333419
        3139.236498918198006794026
        31413.57034538691205229188
        >>> besselyzero(5,100); besselyzero(5,1000); besselyzero(5,10000)
        319.6183338562782156235062
        3147.086508524556404473186
        31421.42392920214673402828
        >>> besselyzero(0,100,1); besselyzero(0,1000,1); besselyzero(0,10000,1)
        313.3726705426359345050449
        3140.807136030340213610065
        31415.14112579761578220175

    Zeros of functions with large order::

        >>> besselyzero(50,1)
        53.50285882040036394680237
        >>> besselyzero(50,2)
        60.11244442774058114686022
        >>> besselyzero(50,100)
        387.1096509824943957706835
        >>> besselyzero(50,1,1)
        56.96290427516751320063605
        >>> besselyzero(50,2,1)
        62.74888166945933944036623
        >>> besselyzero(50,100,1)
        388.6923300548309258355475

    Zeros of functions with fractional order::

        >>> besselyzero(0.5,1); besselyzero(1.5,1); besselyzero(2.25,4)
        1.570796326794896619231322
        2.798386045783887136720249
        13.56721208770735123376018

    """
    return +bessel_zero(ctx, 2, derivative, v, m)
<|endoftext|>from ..libmp.backend import xrange
from .functions import defun, defun_wrapped

def _check_need_perturb(ctx, terms, prec, discard_known_zeros):
    perturb = recompute = False
    extraprec = 0
    discard = []
    for term_index, term in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term
        have_singular_nongamma_weight = False
        # Avoid division by zero in leading factors (TODO:
        # also check for near division by zero?)
        for k, w in enumerate(w_s):
            if not w:
                if ctx.re(c_s[k]) <= 0 and c_s[k]:
                    perturb = recompute = True
                    have_singular_nongamma_weight = True
        pole_count = [0, 0, 0]
        # Check for gamma and series poles and near-poles
        for data_index, data in enumerate([alpha_s, beta_s, b_s]):
            for i, x in enumerate(data):
                n, d = ctx.nint_distance(x)
                # Poles
                if n > 0:
                    continue
                if d == ctx.ninf:
                    # OK if we have a polynomial
                    # ------------------------------
                    ok = False
                    if data_index == 2:
                        for u in a_s:
                            if ctx.isnpint(u) and u >= int(n):
                                ok = True
                                break
                    if ok:
                        continue
                    pole_count[data_index] += 1
                    # ------------------------------
                    #perturb = recompute = True
                    #return perturb, recompute, extraprec
                elif d < -4:
                    extraprec += -d
                    recompute = True
        if discard_known_zeros and pole_count[1] > pole_count[0] + pole_count[2] \
            and not have_singular_nongamma_weight:
            discard.append(term_index)
        elif sum(pole_count):
            perturb = recompute = True
    return perturb, recompute, extraprec, discard

_hypercomb_msg = """
hypercomb() failed to converge to the requested %i bits of accuracy
using a working precision of %i bits. The function value may be zero or
infinite; try passing zeroprec=N or infprec=M to bound finite values between
2^(-N) and 2^M. Otherwise try a higher maxprec or maxterms.
"""

@defun
def hypercomb(ctx, function, params=[], discard_known_zeros=True, **kwargs):
    orig = ctx.prec
    sumvalue = ctx.zero
    dist = ctx.nint_distance
    ninf = ctx.ninf
    orig_params = params[:]
    verbose = kwargs.get('verbose', False)
    maxprec = kwargs.get('maxprec', ctx._default_hyper_maxprec(orig))
    kwargs['maxprec'] = maxprec   # For calls to hypsum
    zeroprec = kwargs.get('zeroprec')
    infprec = kwargs.get('infprec')
    perturbed_reference_value = None
    hextra = 0
    try:
        while 1:
            ctx.prec += 10
            if ctx.prec > maxprec:
                raise ValueError(_hypercomb_msg % (orig, ctx.prec))
            orig2 = ctx.prec
            params = orig_params[:]
            terms = function(*params)
            if verbose:
                print()
                print("ENTERING hypercomb main loop")
                print("prec =", ctx.prec)
                print("hextra", hextra)
            perturb, recompute, extraprec, discard = \
                _check_need_perturb(ctx, terms, orig, discard_known_zeros)
            ctx.prec += extraprec
            if perturb:
                if "hmag" in kwargs:
                    hmag = kwargs["hmag"]
                elif ctx._fixed_precision:
                    hmag = int(ctx.prec*0.3)
                else:
                    hmag = orig + 10 + hextra
                h = ctx.ldexp(ctx.one, -hmag)
                ctx.prec = orig2 + 10 + hmag + 10
                for k in range(len(params)):
                    params[k] += h
                    # Heuristically ensure that the perturbations
                    # are "independent" so that two perturbations
                    # don't accidentally cancel each other out
                    # in a subtraction.
                    h += h/(k+1)
            if recompute:
                terms = function(*params)
            if discard_known_zeros:
                terms = [term for (i, term) in enumerate(terms) if i not in discard]
            if not terms:
                return ctx.zero
            evaluated_terms = []
            for term_index, term_data in enumerate(terms):
                w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term_data
                if verbose:
                    print()
                    print("  Evaluating term %i/%i : %iF%i" % \
                        (term_index+1, len(terms), len(a_s), len(b_s)))
                    print("    powers", ctx.nstr(w_s), ctx.nstr(c_s))
                    print("    gamma", ctx.nstr(alpha_s), ctx.nstr(beta_s))
                    print("    hyper", ctx.nstr(a_s), ctx.nstr(b_s))
                    print("    z", ctx.nstr(z))
                #v = ctx.hyper(a_s, b_s, z, **kwargs)
                #for a in alpha_s: v *= ctx.gamma(a)
                #for b in beta_s: v *= ctx.rgamma(b)
                #for w, c in zip(w_s, c_s): v *= ctx.power(w, c)
                v = ctx.fprod([ctx.hyper(a_s, b_s, z, **kwargs)] + \
                    [ctx.gamma(a) for a in alpha_s] + \
                    [ctx.rgamma(b) for b in beta_s] + \
                    [ctx.power(w,c) for (w,c) in zip(w_s,c_s)])
                if verbose:
                    print("    Value:", v)
                evaluated_terms.append(v)

            if len(terms) == 1 and (not perturb):
                sumvalue = evaluated_terms[0]
                break

            if ctx._fixed_precision:
                sumvalue = ctx.fsum(evaluated_terms)
                break

            sumvalue = ctx.fsum(evaluated_terms)
            term_magnitudes = [ctx.mag(x) for x in evaluated_terms]
            max_magnitude = max(term_magnitudes)
            sum_magnitude = ctx.mag(sumvalue)
            cancellation = max_magnitude - sum_magnitude
            if verbose:
                print()
                print("  Cancellation:", cancellation, "bits")
                print("  Increased precision:", ctx.prec - orig, "bits")

            precision_ok = cancellation < ctx.prec - orig

            if zeroprec is None:
                zero_ok = False
            else:
                zero_ok = max_magnitude - ctx.prec < -zeroprec
            if infprec is None:
                inf_ok = False
            else:
                inf_ok = max_magnitude > infprec

            if precision_ok and (not perturb) or ctx.isnan(cancellation):
                break
            elif precision_ok:
                if perturbed_reference_value is None:
                    hextra += 20
                    perturbed_reference_value = sumvalue
                    continue
                elif ctx.mag(sumvalue - perturbed_reference_value) <= \
                        ctx.mag(sumvalue) - orig:
                    break
                elif zero_ok:
                    sumvalue = ctx.zero
                    break
                elif inf_ok:
                    sumvalue = ctx.inf
                    break
                elif 'hmag' in kwargs:
                    break
                else:
                    hextra *= 2
                    perturbed_reference_value = sumvalue
            # Increase precision
            else:
                increment = min(max(cancellation, orig//2), max(extraprec,orig))
                ctx.prec += increment
                if verbose:
                    print("  Must start over with increased precision")
                continue
    finally:
        ctx.prec = orig
    return +sumvalue

@defun
def hyper(ctx, a_s, b_s, z, **kwargs):
    """
    Hypergeometric function, general case.
    """
    z = ctx.convert(z)
    p = len(a_s)
    q = len(b_s)
    a_s = [ctx._convert_param(a) for a in a_s]
    b_s = [ctx._convert_param(b) for b in b_s]
    # Reduce degree by eliminating common parameters
    if kwargs.get('eliminate', True):
        elim_nonpositive = kwargs.get('eliminate_all', False)
        i = 0
        while i < q and a_s:
            b = b_s[i]
            if b in a_s and (elim_nonpositive or not ctx.isnpint(b[0])):
                a_s.remove(b)
                b_s.remove(b)
                p -= 1
                q -= 1
            else:
                i += 1
    # Handle special cases
    if p == 0:
        if   q == 1: return ctx._hyp0f1(b_s, z, **kwargs)
        elif q == 0: return ctx.exp(z)
    elif p == 1:
        if   q == 1: return ctx._hyp1f1(a_s, b_s, z, **kwargs)
        elif q == 2: return ctx._hyp1f2(a_s, b_s, z, **kwargs)
        elif q == 0: return ctx._hyp1f0(a_s[0][0], z)
    elif p == 2:
        if   q == 1: return ctx._hyp2f1(a_s, b_s, z, **kwargs)
        elif q == 2: return ctx._hyp2f2(a_s, b_s, z, **kwargs)
        elif q == 3: return ctx._hyp2f3(a_s, b_s, z, **kwargs)
        elif q == 0: return ctx._hyp2f0(a_s, b_s, z, **kwargs)
    elif p == q+1:
        return ctx._hypq1fq(p, q, a_s, b_s, z, **kwargs)
    elif p > q+1 and not kwargs.get('force_series'):
        return ctx._hyp_borel(p, q, a_s, b_s, z, **kwargs)
    coeffs, types = zip(*(a_s+b_s))
    return ctx.hypsum(p, q, types, coeffs, z, **kwargs)

@defun
def hyp0f1(ctx,b,z,**kwargs):
    return ctx.hyper([],[b],z,**kwargs)

@defun
def hyp1f1(ctx,a,b,z,**kwargs):
    return ctx.hyper([a],[b],z,**kwargs)

@defun
def hyp1f2(ctx,a1,b1,b2,z,**kwargs):
    return ctx.hyper([a1],[b1,b2],z,**kwargs)

@defun
def hyp2f1(ctx,a,b,c,z,**kwargs):
    return ctx.hyper([a,b],[c],z,**kwargs)

@defun
def hyp2f2(ctx,a1,a2,b1,b2,z,**kwargs):
    return ctx.hyper([a1,a2],[b1,b2],z,**kwargs)

@defun
def hyp2f3(ctx,a1,a2,b1,b2,b3,z,**kwargs):
    return ctx.hyper([a1,a2],[b1,b2,b3],z,**kwargs)

@defun
def hyp2f0(ctx,a,b,z,**kwargs):
    return ctx.hyper([a,b],[],z,**kwargs)

@defun
def hyp3f2(ctx,a1,a2,a3,b1,b2,z,**kwargs):
    return ctx.hyper([a1,a2,a3],[b1,b2],z,**kwargs)

@defun_wrapped
def _hyp1f0(ctx, a, z):
    return (1-z) ** (-a)

@defun
def _hyp0f1(ctx, b_s, z, **kwargs):
    (b, btype), = b_s
    if z:
        magz = ctx.mag(z)
    else:
        magz = 0
    if magz >= 8 and not kwargs.get('force_series'):
        try:
            # http://functions.wolfram.com/HypergeometricFunctions/
            # Hypergeometric0F1/06/02/03/0004/
            # TODO: handle the all-real case more efficiently!
            # TODO: figure out how much precision is needed (exponential growth)
            orig = ctx.prec
            try:
                ctx.prec += 12 + magz//2
                def h():
                    w = ctx.sqrt(-z)
                    jw = ctx.j*w
                    u = 1/(4*jw)
                    c = ctx.mpq_1_2 - b
                    E = ctx.exp(2*jw)
                    T1 = ([-jw,E], [c,-1], [], [], [b-ctx.mpq_1_2, ctx.mpq_3_2-b], [], -u)
                    T2 = ([jw,E], [c,1], [], [], [b-ctx.mpq_1_2, ctx.mpq_3_2-b], [], u)
                    return T1, T2
                v = ctx.hypercomb(h, [], force_series=True)
                v = ctx.gamma(b)/(2*ctx.sqrt(ctx.pi))*v
            finally:
                ctx.prec = orig
            if ctx._is_real_type(b) and ctx._is_real_type(z):
                v = ctx._re(v)
            return +v
        except ctx.NoConvergence:
            pass
    return ctx.hypsum(0, 1, (btype,), [b], z, **kwargs)

@defun
def _hyp1f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), = a_s
    (b, btype), = b_s
    if not z:
        return ctx.one+z
    magz = ctx.mag(z)
    if magz >= 7 and not (ctx.isint(a) and ctx.re(a) <= 0):
        if ctx.isinf(z):
            if ctx.sign(a) == ctx.sign(b) == ctx.sign(z) == 1:
                return ctx.inf
            return ctx.nan * z
        try:
            try:
                ctx.prec += magz
                sector = ctx._im(z) < 0
                def h(a,b):
                    if sector:
                        E = ctx.expjpi(ctx.fneg(a, exact=True))
                    else:
                        E = ctx.expjpi(a)
                    rz = 1/z
                    T1 = ([E,z], [1,-a], [b], [b-a], [a, 1+a-b], [], -rz)
                    T2 = ([ctx.exp(z),z], [1,a-b], [b], [a], [b-a, 1-a], [], rz)
                    return T1, T2
                v = ctx.hypercomb(h, [a,b], force_series=True)
                if ctx._is_real_type(a) and ctx._is_real_type(b) and ctx._is_real_type(z):
                    v = ctx._re(v)
                return +v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec -= magz
    v = ctx.hypsum(1, 1, (atype, btype), [a, b], z, **kwargs)
    return v

def _hyp2f1_gosper(ctx,a,b,c,z,**kwargs):
    # Use Gosper's recurrence
    # See http://www.math.utexas.edu/pipermail/maxima/2006/000126.html
    _a,_b,_c,_z = a, b, c, z
    orig = ctx.prec
    maxprec = kwargs.get('maxprec', 100*orig)
    extra = 10
    while 1:
        ctx.prec = orig + extra
        #a = ctx.convert(_a)
        #b = ctx.convert(_b)
        #c = ctx.convert(_c)
        z = ctx.convert(_z)
        d = ctx.mpf(0)
        e = ctx.mpf(1)
        f = ctx.mpf(0)
        k = 0
        # Common subexpression elimination, unfortunately making
        # things a bit unreadable. The formula is quite messy to begin
        # with, though...
        abz = a*b*z
        ch = c * ctx.mpq_1_2
        c1h = (c+1) * ctx.mpq_1_2
        nz = 1-z
        g = z/nz
        abg = a*b*g
        cba = c-b-a
        z2 = z-2
        tol = -ctx.prec - 10
        nstr = ctx.nstr
        nprint = ctx.nprint
        mag = ctx.mag
        maxmag = ctx.ninf
        while 1:
            kch = k+ch
            kakbz = (k+a)*(k+b)*z / (4*(k+1)*kch*(k+c1h))
            d1 = kakbz*(e-(k+cba)*d*g)
            e1 = kakbz*(d*abg+(k+c)*e)
            ft = d*(k*(cba*z+k*z2-c)-abz)/(2*kch*nz)
            f1 = f + e - ft
            maxmag = max(maxmag, mag(f1))
            if mag(f1-f) < tol:
                break
            d, e, f = d1, e1, f1
            k += 1
        cancellation = maxmag - mag(f1)
        if cancellation < extra:
            break
        else:
            extra += cancellation
            if extra > maxprec:
                raise ctx.NoConvergence
    return f1

@defun
def _hyp2f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    (c, ctype), = b_s
    if z == 1:
        # TODO: the following logic can be simplified
        convergent = ctx.re(c-a-b) > 0
        finite = (ctx.isint(a) and a <= 0) or (ctx.isint(b) and b <= 0)
        zerodiv = ctx.isint(c) and c <= 0 and not \
            ((ctx.isint(a) and c <= a <= 0) or (ctx.isint(b) and c <= b <= 0))
        #print "bz", a, b, c, z, convergent, finite, zerodiv
        # Gauss's theorem gives the value if convergent
        if (convergent or finite) and not zerodiv:
            return ctx.gammaprod([c, c-a-b], [c-a, c-b], _infsign=True)
        # Otherwise, there is a pole and we take the
        # sign to be that when approaching from below
        # XXX: this evaluation is not necessarily correct in all cases
        return ctx.hyp2f1(a,b,c,1-ctx.eps*2) * ctx.inf

    # Equal to 1 (first term), unless there is a subsequent
    # division by zero
    if not z:
        # Division by zero but power of z is higher than
        # first order so cancels
        if c or a == 0 or b == 0:
            return 1+z
        # Indeterminate
        return ctx.nan

    # Hit zero denominator unless numerator goes to 0 first
    if ctx.isint(c) and c <= 0:
        if (ctx.isint(a) and c <= a <= 0) or \
           (ctx.isint(b) and c <= b <= 0):
            pass
        else:
            # Pole in series
            return ctx.inf

    absz = abs(z)

    # Fast case: standard series converges rapidly,
    # possibly in finitely many terms
    if absz <= 0.8 or (ctx.isint(a) and a <= 0 and a >= -1000) or \
                      (ctx.isint(b) and b <= 0 and b >= -1000):
        return ctx.hypsum(2, 1, (atype, btype, ctype), [a, b, c], z, **kwargs)

    orig = ctx.prec
    try:
        ctx.prec += 10

        # Use 1/z transformation
        if absz >= 1.3:
            def h(a,b):
                t = ctx.mpq_1-c; ab = a-b; rz = 1/z
                T1 = ([-z],[-a], [c,-ab],[b,c-a], [a,t+a],[ctx.mpq_1+ab],  rz)
                T2 = ([-z],[-b], [c,ab],[a,c-b], [b,t+b],[ctx.mpq_1-ab],  rz)
                return T1, T2
            v = ctx.hypercomb(h, [a,b], **kwargs)

        # Use 1-z transformation
        elif abs(1-z) <= 0.75:
            def h(a,b):
                t = c-a-b; ca = c-a; cb = c-b; rz = 1-z
                T1 = [], [], [c,t], [ca,cb], [a,b], [1-t], rz
                T2 = [rz], [t], [c,a+b-c], [a,b], [ca,cb], [1+t], rz
                return T1, T2
            v = ctx.hypercomb(h, [a,b], **kwargs)

        # Use z/(z-1) transformation
        elif abs(z/(z-1)) <= 0.75:
            v = ctx.hyp2f1(a, c-b, c, z/(z-1)) / (1-z)**a

        # Remaining part of unit circle
        else:
            v = _hyp2f1_gosper(ctx,a,b,c,z,**kwargs)

    finally:
        ctx.prec = orig
    return +v

@defun
def _hypq1fq(ctx, p, q, a_s, b_s, z, **kwargs):
    r"""
    Evaluates 3F2, 4F3, 5F4, ...
    """
    a_s, a_types = zip(*a_s)
    b_s, b_types = zip(*b_s)
    a_s = list(a_s)
    b_s = list(b_s)
    absz = abs(z)
    ispoly = False
    for a in a_s:
        if ctx.isint(a) and a <= 0:
            ispoly = True
            break
    # Direct summation
    if absz < 1 or ispoly:
        try:
            return ctx.hypsum(p, q, a_types+b_types, a_s+b_s, z, **kwargs)
        except ctx.NoConvergence:
            if absz > 1.1 or ispoly:
                raise
    # Use expansion at |z-1| -> 0.
    # Reference: Wolfgang Buhring, "Generalized Hypergeometric Functions at
    #   Unit Argument", Proc. Amer. Math. Soc., Vol. 114, No. 1 (Jan. 1992),
    #   pp.145-153
    # The current implementation has several problems:
    # 1. We only implement it for 3F2. The expansion coefficients are
    #    given by extremely messy nested sums in the higher degree cases
    #    (see reference). Is efficient sequential generation of the coefficients
    #    possible in the > 3F2 case?
    # 2. Although the series converges, it may do so slowly, so we need
    #    convergence acceleration. The acceleration implemented by
    #    nsum does not always help, so results returned are sometimes
    #    inaccurate! Can we do better?
    # 3. We should check conditions for convergence, and possibly
    #    do a better job of cancelling out gamma poles if possible.
    if z == 1:
        # XXX: should also check for division by zero in the
        # denominator of the series (cf. hyp2f1)
        S = ctx.re(sum(b_s)-sum(a_s))
        if S <= 0:
            #return ctx.hyper(a_s, b_s, 1-ctx.eps*2, **kwargs) * ctx.inf
            return ctx.hyper(a_s, b_s, 0.9, **kwargs) * ctx.inf
    if (p,q) == (3,2) and abs(z-1) < 0.05:   # and kwargs.get('sum1')
        #print "Using alternate summation (experimental)"
        a1,a2,a3 = a_s
        b1,b2 = b_s
        u = b1+b2-a3
        initial = ctx.gammaprod([b2-a3,b1-a3,a1,a2],[b2-a3,b1-a3,1,u])
        def term(k, _cache={0:initial}):
            u = b1+b2-a3+k
            if k in _cache:
                t = _cache[k]
            else:
                t = _cache[k-1]
                t *= (b1+k-a3-1)*(b2+k-a3-1)
                t /= k*(u-1)
                _cache[k] = t
            return t * ctx.hyp2f1(a1,a2,u,z)
        try:
            S = ctx.nsum(term, [0,ctx.inf], verbose=kwargs.get('verbose'),
                strict=kwargs.get('strict', True))
            return S * ctx.gammaprod([b1,b2],[a1,a2,a3])
        except ctx.NoConvergence:
            pass
    # Try to use convergence acceleration on and close to the unit circle.
    # Problem: the convergence acceleration degenerates as |z-1| -> 0,
    # except for special cases. Everywhere else, the Shanks transformation
    # is very efficient.
    if absz < 1.1 and ctx._re(z) <= 1:

        def term(kk, _cache={0:ctx.one}):
            k = int(kk)
            if k != kk:
                t = z ** ctx.mpf(kk) / ctx.fac(kk)
                for a in a_s: t *= ctx.rf(a,kk)
                for b in b_s: t /= ctx.rf(b,kk)
                return t
            if k in _cache:
                return _cache[k]
            t = term(k-1)
            m = k-1
            for j in xrange(p): t *= (a_s[j]+m)
            for j in xrange(q): t /= (b_s[j]+m)
            t *= z
            t /= k
            _cache[k] = t
            return t

        sum_method = kwargs.get('sum_method', 'r+s+e')

        try:
            return ctx.nsum(term, [0,ctx.inf], verbose=kwargs.get('verbose'),
                strict=kwargs.get('strict', True),
                method=sum_method.replace('e',''))
        except ctx.NoConvergence:
            if 'e' not in sum_method:
                raise
            pass

        if kwargs.get('verbose'):
            print("Attempting Euler-Maclaurin summation")


        """
        Somewhat slower version (one diffs_exp for each factor).
        However, this would be faster with fast direct derivatives
        of the gamma function.

        def power_diffs(k0):
            r = 0
            l = ctx.log(z)
            while 1:
                yield z**ctx.mpf(k0) * l**r
                r += 1

        def loggamma_diffs(x, reciprocal=False):
            sign = (-1) ** reciprocal
            yield sign * ctx.loggamma(x)
            i = 0
            while 1:
                yield sign * ctx.psi(i,x)
                i += 1

        def hyper_diffs(k0):
            b2 = b_s + [1]
            A = [ctx.diffs_exp(loggamma_diffs(a+k0)) for a in a_s]
            B = [ctx.diffs_exp(loggamma_diffs(b+k0,True)) for b in b2]
            Z = [power_diffs(k0)]
            C = ctx.gammaprod([b for b in b2], [a for a in a_s])
            for d in ctx.diffs_prod(A + B + Z):
                v = C * d
                yield v
        """

        def log_diffs(k0):
            b2 = b_s + [1]
            yield sum(ctx.loggamma(a+k0) for a in a_s) - \
                sum(ctx.loggamma(b+k0) for b in b2) + k0*ctx.log(z)
            i = 0
            while 1:
                v = sum(ctx.psi(i,a+k0) for a in a_s) - \
                    sum(ctx.psi(i,b+k0) for b in b2)
                if i == 0:
                    v += ctx.log(z)
                yield v
                i += 1

        def hyper_diffs(k0):
            C = ctx.gammaprod([b for b in b_s], [a for a in a_s])
            for d in ctx.diffs_exp(log_diffs(k0)):
                v = C * d
                yield v

        tol = ctx.eps / 1024
        prec = ctx.prec
        try:
            trunc = 50 * ctx.dps
            ctx.prec += 20
            for i in xrange(5):
                head = ctx.fsum(term(k) for k in xrange(trunc))
                tail, err = ctx.sumem(term, [trunc, ctx.inf], tol=tol,
                    adiffs=hyper_diffs(trunc),
                    verbose=kwargs.get('verbose'),
                    error=True,
                    _fast_abort=True)
                if err < tol:
                    v = head + tail
                    break
                trunc *= 2
                # Need to increase precision because calculation of
                # derivatives may be inaccurate
                ctx.prec += ctx.prec//2
                if i == 4:
                    raise ctx.NoConvergence(\
                        "Euler-Maclaurin summation did not converge")
        finally:
            ctx.prec = prec
        return +v

    # Use 1/z transformation
    # http://functions.wolfram.com/HypergeometricFunctions/
    #   HypergeometricPFQ/06/01/05/02/0004/
    def h(*args):
        a_s = list(args[:p])
        b_s = list(args[p:])
        Ts = []
        recz = ctx.one/z
        negz = ctx.fneg(z, exact=True)
        for k in range(q+1):
            ak = a_s[k]
            C = [negz]
            Cp = [-ak]
            Gn = b_s + [ak] + [a_s[j]-ak for j in range(q+1) if j != k]
            Gd = a_s + [b_s[j]-ak for j in range(q)]
            Fn = [ak] + [ak-b_s[j]+1 for j in range(q)]
            Fd = [1-a_s[j]+ak for j in range(q+1) if j != k]
            Ts.append((C, Cp, Gn, Gd, Fn, Fd, recz))
        return Ts
    return ctx.hypercomb(h, a_s+b_s, **kwargs)

@defun
def _hyp_borel(ctx, p, q, a_s, b_s, z, **kwargs):
    if a_s:
        a_s, a_types = zip(*a_s)
        a_s = list(a_s)
    else:
        a_s, a_types = [], ()
    if b_s:
        b_s, b_types = zip(*b_s)
        b_s = list(b_s)
    else:
        b_s, b_types = [], ()
    kwargs['maxterms'] = kwargs.get('maxterms', ctx.prec)
    try:
        return ctx.hypsum(p, q, a_types+b_types, a_s+b_s, z, **kwargs)
    except ctx.NoConvergence:
        pass
    prec = ctx.prec
    try:
        tol = kwargs.get('asymp_tol', ctx.eps/4)
        ctx.prec += 10
        # hypsum is has a conservative tolerance. So we try again:
        def term(k, cache={0:ctx.one}):
            if k in cache:
                return cache[k]
            t = term(k-1)
            for a in a_s: t *= (a+(k-1))
            for b in b_s: t /= (b+(k-1))
            t *= z
            t /= k
            cache[k] = t
            return t
        s = ctx.one
        for k in xrange(1, ctx.prec):
            t = term(k)
            s += t
            if abs(t) <= tol:
                return s
    finally:
        ctx.prec = prec
    if p <= q+3:
        contour = kwargs.get('contour')
        if not contour:
            if ctx.arg(z) < 0.25:
                u = z / max(1, abs(z))
                if ctx.arg(z) >= 0:
                    contour = [0, 2j, (2j+2)/u, 2/u, ctx.inf]
                else:
                    contour = [0, -2j, (-2j+2)/u, 2/u, ctx.inf]
                #contour = [0, 2j/z, 2/z, ctx.inf]
                #contour = [0, 2j, 2/z, ctx.inf]
                #contour = [0, 2j, ctx.inf]
            else:
                contour = [0, ctx.inf]
        quad_kwargs = kwargs.get('quad_kwargs', {})
        def g(t):
            return ctx.exp(-t)*ctx.hyper(a_s, b_s+[1], t*z)
        I, err = ctx.quad(g, contour, error=True, **quad_kwargs)
        if err <= abs(I)*ctx.eps*8:
            return I
    raise ctx.NoConvergence


@defun
def _hyp2f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec

    # Asymptotic expansion is ~ exp(z)
    asymp_extraprec = magz

    # Asymptotic series is in terms of 3F1
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 3)

    # TODO: much of the following could be shared with 2F3 instead of
    # copypasted
    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric2F2/06/02/02/0002/
                def h(a1,a2,b1,b2):
                    X = a1+a2-b1-b2
                    A2 = a1+a2
                    B2 = b1+b2
                    c = {}
                    c[0] = ctx.one
                    c[1] = (A2-1)*X+b1*b2-a1*a2
                    s1 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = 1-B2+2*a1+a1**2+2*a2+a2**2-A2*B2+a1*a2+b1*b2+(2*B2-3*(A2+1))*k+2*k**2
                            uu2 = (k-A2+b1-1)*(k-A2+b2-1)*(k-X-2)
                            c[k] = ctx.one/k * (uu1*c[k-1]-uu2*c[k-2])
                        t1 = c[k] * z**(-k)
                        if abs(t1) < 0.1*ctx.eps:
                            #print "Convergence :)"
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            #print "No convergence :("
                            raise ctx.NoConvergence
                        s1 += t1
                        tprev = t1
                        k += 1
                    S = ctx.exp(z)*s1
                    T1 = [z,S], [X,1], [b1,b2],[a1,a2],[],[],0
                    T2 = [-z],[-a1],[b1,b2,a2-a1],[a2,b1-a1,b2-a1],[a1,a1-b1+1,a1-b2+1],[a1-a2+1],-1/z
                    T3 = [-z],[-a2],[b1,b2,a1-a2],[a1,b1-a2,b2-a2],[a2,a2-b1+1,a2-b2+1],[-a1+a2+1],-1/z
                    return T1, T2, T3
                v = ctx.hypercomb(h, [a1,a2,b1,b2], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,a2,b1,b2,z]) == 5:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    return ctx.hypsum(2, 2, (a1type, a2type, b1type, b2type), [a1, a2, b1, b2], z, **kwargs)



@defun
def _hyp1f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), = a_s
    (b1, b1type), (b2, b2type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec

    # Asymptotic expansion is ~ exp(sqrt(z))
    asymp_extraprec = z and magz//2

    # Asymptotic series is in terms of 3F0
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 19) and \
        (ctx.sqrt(absz) > 1.5*orig)  # and \
    #   ctx._hyp_check_convergence([a1, a1-b1+1, a1-b2+1], [],
    #                              1/absz, orig+40+asymp_extraprec)

    # TODO: much of the following could be shared with 2F3 instead of
    # copypasted
    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric1F2/06/02/03/
                def h(a1,b1,b2):
                    X = ctx.mpq_1_2*(a1-b1-b2+ctx.mpq_1_2)
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2*(ctx.mpq_1_4*(3*a1+b1+b2-2)*(a1-b1-b2)+b1*b2-ctx.mpq_3_16)
                    c[2] = 2*(b1*b2+ctx.mpq_1_4*(a1-b1-b2)*(3*a1+b1+b2-2)-ctx.mpq_3_16)**2+\
                        ctx.mpq_1_16*(-16*(2*a1-3)*b1*b2 + \
                        4*(a1-b1-b2)*(-8*a1**2+11*a1+b1+b2-2)-3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (3*k**2+(-6*a1+2*b1+2*b2-4)*k + 3*a1**2 - \
                                (b1-b2)**2 - 2*a1*(b1+b2-2) + ctx.mpq_1_4)
                            uu2 = (k-a1+b1-b2-ctx.mpq_1_2)*(k-a1-b1+b2-ctx.mpq_1_2)*\
                                (k-a1+b1+b2-ctx.mpq_5_2)
                            c[k] = ctx.one/(2*k)*(uu1*c[k-1]-uu2*c[k-2])
                        w = c[k] * (-z)**(-0.5*k)
                        t1 = (-ctx.j)**k * ctx.mpf(2)**(-k) * w
                        t2 = ctx.j**k * ctx.mpf(2)**(-k) * w
                        if abs(t1) < 0.1*ctx.eps:
                            #print "Convergence :)"
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            #print "No convergence :("
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi*X+2*ctx.sqrt(-z))*s1 + \
                        ctx.expj(-(ctx.pi*X+2*ctx.sqrt(-z)))*s2
                    T1 = [0.5*S, ctx.pi, -z], [1, -0.5, X], [b1, b2], [a1],\
                        [], [], 0
                    T2 = [-z], [-a1], [b1,b2],[b1-a1,b2-a1], \
                        [a1,a1-b1+1,a1-b2+1], [], 1/z
                    return T1, T2
                v = ctx.hypercomb(h, [a1,b1,b2], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,b1,b2,z]) == 4:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    #print "not using asymp"
    return ctx.hypsum(1, 2, (a1type, b1type, b2type), [a1, b1, b2], z, **kwargs)



@defun
def _hyp2f3(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type), (b3, b3type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)

    # Asymptotic expansion is ~ exp(sqrt(z))
    asymp_extraprec = z and magz//2
    orig = ctx.prec

    # Asymptotic series is in terms of 4F1
    # The square root below empirically provides a plausible criterion
    # for the leading series to converge
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 19) and (ctx.sqrt(absz) > 1.5*orig)

    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric2F3/06/02/03/01/0002/
                def h(a1,a2,b1,b2,b3):
                    X = ctx.mpq_1_2*(a1+a2-b1-b2-b3+ctx.mpq_1_2)
                    A2 = a1+a2
                    B3 = b1+b2+b3
                    A = a1*a2
                    B = b1*b2+b3*b2+b1*b3
                    R = b1*b2*b3
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2*(B - A + ctx.mpq_1_4*(3*A2+B3-2)*(A2-B3) - ctx.mpq_3_16)
                    c[2] = ctx.mpq_1_2*c[1]**2 + ctx.mpq_1_16*(-16*(2*A2-3)*(B-A) + 32*R +\
                        4*(-8*A2**2 + 11*A2 + 8*A + B3 - 2)*(A2-B3)-3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (k-2*X-3)*(k-2*X-2*b1-1)*(k-2*X-2*b2-1)*\
                                (k-2*X-2*b3-1)
                            uu2 = (4*(k-1)**3 - 6*(4*X+B3)*(k-1)**2 + \
                                2*(24*X**2+12*B3*X+4*B+B3-1)*(k-1) - 32*X**3 - \
                                24*B3*X**2 - 4*B - 8*R - 4*(4*B+B3-1)*X + 2*B3-1)
                            uu3 = (5*(k-1)**2+2*(-10*X+A2-3*B3+3)*(k-1)+2*c[1])
                            c[k] = ctx.one/(2*k)*(uu1*c[k-3]-uu2*c[k-2]+uu3*c[k-1])
                        w = c[k] * ctx.power(-z, -0.5*k)
                        t1 = (-ctx.j)**k * ctx.mpf(2)**(-k) * w
                        t2 = ctx.j**k * ctx.mpf(2)**(-k) * w
                        if abs(t1) < 0.1*ctx.eps:
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi*X+2*ctx.sqrt(-z))*s1 + \
                        ctx.expj(-(ctx.pi*X+2*ctx.sqrt(-z)))*s2
                    T1 = [0.5*S, ctx.pi, -z], [1, -0.5, X], [b1, b2, b3], [a1, a2],\
                        [], [], 0
                    T2 = [-z], [-a1], [b1,b2,b3,a2-a1],[a2,b1-a1,b2-a1,b3-a1], \
                        [a1,a1-b1+1,a1-b2+1,a1-b3+1], [a1-a2+1], 1/z
                    T3 = [-z], [-a2], [b1,b2,b3,a1-a2],[a1,b1-a2,b2-a2,b3-a2], \
                        [a2,a2-b1+1,a2-b2+1,a2-b3+1],[-a1+a2+1], 1/z
                    return T1, T2, T3
                v = ctx.hypercomb(h, [a1,a2,b1,b2,b3], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,a2,b1,b2,b3,z]) == 6:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    return ctx.hypsum(2, 3, (a1type, a2type, b1type, b2type, b3type), [a1, a2, b1, b2, b3], z, **kwargs)

@defun
def _hyp2f0(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    # We want to try aggressively to use the asymptotic expansion,
    # and fall back only when absolutely necessary
    try:
        kwargsb = kwargs.copy()
        kwargsb['maxterms'] = kwargsb.get('maxterms', ctx.prec)
        return ctx.hypsum(2, 0, (atype,btype), [a,b], z, **kwargsb)
    except ctx.NoConvergence:
        if kwargs.get('force_series'):
            raise
        pass
    def h(a, b):
        w = ctx.sinpi(b)
        rz = -1/z
        T1 = ([ctx.pi,w,rz],[1,-1,a],[],[a-b+1,b],[a],[b],rz)
        T2 = ([-ctx.pi,w,rz],[1,-1,1+a-b],[],[a,2-b],[a-b+1],[2-b],rz)
        return T1, T2
    return ctx.hypercomb(h, [a, 1+a-b], **kwargs)

@defun
def meijerg(ctx, a_s, b_s, z, r=1, series=None, **kwargs):
    an, ap = a_s
    bm, bq = b_s
    n = len(an)
    p = n + len(ap)
    m = len(bm)
    q = m + len(bq)
    a = an+ap
    b = bm+bq
    a = [ctx.convert(_) for _ in a]
    b = [ctx.convert(_) for _ in b]
    z = ctx.convert(z)
    if series is None:
        if p < q: series = 1
        if p > q: series = 2
        if p == q:
            if m+n == p and abs(z) > 1:
                series = 2
            else:
                series = 1
    if kwargs.get('verbose'):
        print("Meijer G m,n,p,q,series =", m,n,p,q,series)
    if series == 1:
        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(m):
                bases = [z]
                expts = [b[k]/r]
                gn = [b[j]-b[k] for j in range(m) if j != k]
                gn += [1-a[j]+b[k] for j in range(n)]
                gd = [a[j]-b[k] for j in range(n,p)]
                gd += [1-b[j]+b[k] for j in range(m,q)]
                hn = [1-a[j]+b[k] for j in range(p)]
                hd = [1-b[j]+b[k] for j in range(q) if j != k]
                hz = (-ctx.one)**(p-m-n) * z**(ctx.one/r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    else:
        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(n):
                bases = [z]
                if r == 1:
                    expts = [a[k]-1]
                else:
                    expts = [(a[k]-1)/ctx.convert(r)]
                gn = [a[k]-a[j] for j in range(n) if j != k]
                gn += [1-a[k]+b[j] for j in range(m)]
                gd = [a[k]-b[j] for j in range(m,q)]
                gd += [1-a[k]+a[j] for j in range(n,p)]
                hn = [1-a[k]+b[j] for j in range(q)]
                hd = [1+a[j]-a[k] for j in range(p) if j != k]
                hz = (-ctx.one)**(q-m-n) / z**(ctx.one/r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    return ctx.hypercomb(h, a+b, **kwargs)

@defun_wrapped
def appellf1(ctx,a,b1,b2,c,x,y,**kwargs):
    # Assume x smaller
    # We will use x for the outer loop
    if abs(x) > abs(y):
        x, y = y, x
        b1, b2 = b2, b1
    def ok(x):
        return abs(x) < 0.99
    # Finite cases
    if ctx.isnpint(a):
        pass
    elif ctx.isnpint(b1):
        pass
    elif ctx.isnpint(b2):
        x, y, b1, b2 = y, x, b2, b1
    else:
        #print x, y
        # Note: ok if |y| > 1, because
        # 2F1 implements analytic continuation
        if not ok(x):
            u1 = (x-y)/(x-1)
            if not ok(u1):
                raise ValueError("Analytic continuation not implemented")
            #print "Using analytic continuation"
            return (1-x)**(-b1)*(1-y)**(c-a-b2)*\
                ctx.appellf1(c-a,b1,c-b1-b2,c,u1,y,**kwargs)
    return ctx.hyper2d({'m+n':[a],'m':[b1],'n':[b2]}, {'m+n':[c]}, x,y, **kwargs)

@defun
def appellf2(ctx,a,b1,b2,c1,c2,x,y,**kwargs):
    # TODO: continuation
    return ctx.hyper2d({'m+n':[a],'m':[b1],'n':[b2]},
        {'m':[c1],'n':[c2]}, x,y, **kwargs)

@defun
def appellf3(ctx,a1,a2,b1,b2,c,x,y,**kwargs):
    outer_polynomial = ctx.isnpint(a1) or ctx.isnpint(b1)
    inner_polynomial = ctx.isnpint(a2) or ctx.isnpint(b2)
    if not outer_polynomial:
        if inner_polynomial or abs(x) > abs(y):
            x, y = y, x
            a1,a2,b1,b2 = a2,a1,b2,b1
    return ctx.hyper2d({'m':[a1,b1],'n':[a2,b2]}, {'m+n':[c]},x,y,**kwargs)

@defun
def appellf4(ctx,a,b,c1,c2,x,y,**kwargs):
    # TODO: continuation
    return ctx.hyper2d({'m+n':[a,b]}, {'m':[c1],'n':[c2]},x,y,**kwargs)

@defun
def hyper2d(ctx, a, b, x, y, **kwargs):
    r"""
    Sums the generalized 2D hypergeometric series

    .. math ::

        \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
            \frac{P((a),m,n)}{Q((b),m,n)}
            \frac{x^m y^n} {m! n!}

    where `(a) = (a_1,\ldots,a_r)`, `(b) = (b_1,\ldots,b_s)` and where
    `P` and `Q` are products of rising factorials such as `(a_j)_n` or
    `(a_j)_{m+n}`. `P` and `Q` are specified in the form of dicts, with
    the `m` and `n` dependence as keys and parameter lists as values.
    The supported rising factorials are given in the following table
    (note that only a few are supported in `Q`):

    +------------+-------------------+--------+
    | Key        |  Rising factorial | `Q`    |
    +============+===================+========+
    | ``'m'``    |   `(a_j)_m`       | Yes    |
    +------------+-------------------+--------+
    | ``'n'``    |   `(a_j)_n`       | Yes    |
    +------------+-------------------+--------+
    | ``'m+n'``  |   `(a_j)_{m+n}`   | Yes    |
    +------------+-------------------+--------+
    | ``'m-n'``  |   `(a_j)_{m-n}`   | No     |
    +------------+-------------------+--------+
    | ``'n-m'``  |   `(a_j)_{n-m}`   | No     |
    +------------+-------------------+--------+
    | ``'2m+n'`` |   `(a_j)_{2m+n}`  | No     |
    +------------+-------------------+--------+
    | ``'2m-n'`` |   `(a_j)_{2m-n}`  | No     |
    +------------+-------------------+--------+
    | ``'2n-m'`` |   `(a_j)_{2n-m}`  | No     |
    +------------+-------------------+--------+

    For example, the Appell F1 and F4 functions

    .. math ::

        F_1 = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
              \frac{(a)_{m+n} (b)_m (c)_n}{(d)_{m+n}}
              \frac{x^m y^n}{m! n!}

        F_4 = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
              \frac{(a)_{m+n} (b)_{m+n}}{(c)_m (d)_{n}}
              \frac{x^m y^n}{m! n!}

    can be represented respectively as

        ``hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)``

        ``hyper2d({'m+n':[a,b]}, {'m':[c], 'n':[d]}, x, y)``

    More generally, :func:`~mpmath.hyper2d` can evaluate any of the 34 distinct
    convergent second-order (generalized Gaussian) hypergeometric
    series enumerated by Horn, as well as the Kampe de Feriet
    function.

    The series is computed by rewriting it so that the inner
    series (i.e. the series containing `n` and `y`) has the form of an
    ordinary generalized hypergeometric series and thereby can be
    evaluated efficiently using :func:`~mpmath.hyper`. If possible,
    manually swapping `x` and `y` and the corresponding parameters
    can sometimes give better results.

    **Examples**

    Two separable cases: a product of two geometric series, and a
    product of two Gaussian hypergeometric functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> x, y = mpf(0.25), mpf(0.5)
        >>> hyper2d({'m':1,'n':1}, {}, x,y)
        2.666666666666666666666667
        >>> 1/(1-x)/(1-y)
        2.666666666666666666666667
        >>> hyper2d({'m':[1,2],'n':[3,4]}, {'m':[5],'n':[6]}, x,y)
        4.164358531238938319669856
        >>> hyp2f1(1,2,5,x)*hyp2f1(3,4,6,y)
        4.164358531238938319669856

    Some more series that can be done in closed form::

        >>> hyper2d({'m':1,'n':1},{'m+n':1},x,y)
        2.013417124712514809623881
        >>> (exp(x)*x-exp(y)*y)/(x-y)
        2.013417124712514809623881

    Six of the 34 Horn functions, G1-G3 and H1-H3::

        >>> from mpmath import *
        >>> mp.dps = 10; mp.pretty = True
        >>> x, y = 0.0625, 0.125
        >>> a1,a2,b1,b2,c1,c2,d = 1.1,-1.2,-1.3,-1.4,1.5,-1.6,1.7
        >>> hyper2d({'m+n':a1,'n-m':b1,'m-n':b2},{},x,y)  # G1
        1.139090746
        >>> nsum(lambda m,n: rf(a1,m+n)*rf(b1,n-m)*rf(b2,m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.139090746
        >>> hyper2d({'m':a1,'n':a2,'n-m':b1,'m-n':b2},{},x,y)  # G2
        0.9503682696
        >>> nsum(lambda m,n: rf(a1,m)*rf(a2,n)*rf(b1,n-m)*rf(b2,m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.9503682696
        >>> hyper2d({'2n-m':a1,'2m-n':a2},{},x,y)  # G3
        1.029372029
        >>> nsum(lambda m,n: rf(a1,2*n-m)*rf(a2,2*m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.029372029
        >>> hyper2d({'m-n':a1,'m+n':b1,'n':c1},{'m':d},x,y)  # H1
        -1.605331256
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m+n)*rf(c1,n)/rf(d,m)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -1.605331256
        >>> hyper2d({'m-n':a1,'m':b1,'n':[c1,c2]},{'m':d},x,y)  # H2
        -2.35405404
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m)*rf(c1,n)*rf(c2,n)/rf(d,m)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -2.35405404
        >>> hyper2d({'2m+n':a1,'n':b1},{'m+n':c1},x,y)  # H3
        0.974479074
        >>> nsum(lambda m,n: rf(a1,2*m+n)*rf(b1,n)/rf(c1,m+n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.974479074

    **References**

    1. [SrivastavaKarlsson]_
    2. [Weisstein]_ http://mathworld.wolfram.com/HornFunction.html
    3. [Weisstein]_ http://mathworld.wolfram.com/AppellHypergeometricFunction.html

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    def parse(dct, key):
        args = dct.pop(key, [])
        try:
            args = list(args)
        except TypeError:
            args = [args]
        return [ctx.convert(arg) for arg in args]
    a_s = dict(a)
    b_s = dict(b)
    a_m = parse(a, 'm')
    a_n = parse(a, 'n')
    a_m_add_n = parse(a, 'm+n')
    a_m_sub_n = parse(a, 'm-n')
    a_n_sub_m = parse(a, 'n-m')
    a_2m_add_n = parse(a, '2m+n')
    a_2m_sub_n = parse(a, '2m-n')
    a_2n_sub_m = parse(a, '2n-m')
    b_m = parse(b, 'm')
    b_n = parse(b, 'n')
    b_m_add_n = parse(b, 'm+n')
    if a: raise ValueError("unsupported key: %r" % a.keys()[0])
    if b: raise ValueError("unsupported key: %r" % b.keys()[0])
    s = 0
    outer = ctx.one
    m = ctx.mpf(0)
    ok_count = 0
    prec = ctx.prec
    maxterms = kwargs.get('maxterms', 20*prec)
    try:
        ctx.prec += 10
        tol = +ctx.eps
        while 1:
            inner_sign = 1
            outer_sign = 1
            inner_a = list(a_n)
            inner_b = list(b_n)
            outer_a = [a+m for a in a_m]
            outer_b = [b+m for b in b_m]
            # (a)_{m+n} = (a)_m (a+m)_n
            for a in a_m_add_n:
                a = a+m
                inner_a.append(a)
                outer_a.append(a)
            # (b)_{m+n} = (b)_m (b+m)_n
            for b in b_m_add_n:
                b = b+m
                inner_b.append(b)
                outer_b.append(b)
            # (a)_{n-m} = (a-m)_n / (a-m)_m
            for a in a_n_sub_m:
                inner_a.append(a-m)
                outer_b.append(a-m-1)
            # (a)_{m-n} = (-1)^(m+n) (1-a-m)_m / (1-a-m)_n
            for a in a_m_sub_n:
                inner_sign *= (-1)
                outer_sign *= (-1)**(m)
                inner_b.append(1-a-m)
                outer_a.append(-a-m)
            # (a)_{2m+n} = (a)_{2m} (a+2m)_n
            for a in a_2m_add_n:
                inner_a.append(a+2*m)
                outer_a.append((a+2*m)*(1+a+2*m))
            # (a)_{2m-n} = (-1)^(2m+n) (1-a-2m)_{2m} / (1-a-2m)_n
            for a in a_2m_sub_n:
                inner_sign *= (-1)
                inner_b.append(1-a-2*m)
                outer_a.append((a+2*m)*(1+a+2*m))
            # (a)_{2n-m} = 4^n ((a-m)/2)_n ((a-m+1)/2)_n / (a-m)_m
            for a in a_2n_sub_m:
                inner_sign *= 4
                inner_a.append(0.5*(a-m))
                inner_a.append(0.5*(a-m+1))
                outer_b.append(a-m-1)
            inner = ctx.hyper(inner_a, inner_b, inner_sign*y,
                zeroprec=ctx.prec, **kwargs)
            term = outer * inner * outer_sign
            if abs(term) < tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= 3 or not outer:
                break
            s += term
            for a in outer_a: outer *= a
            for b in outer_b: outer /= b
            m += 1
            outer = outer * x / m
            if m > maxterms:
                raise ctx.NoConvergence("maxterms exceeded in hyper2d")
    finally:
        ctx.prec = prec
    return +s

"""
@defun
def kampe_de_feriet(ctx,a,b,c,d,e,f,x,y,**kwargs):
    return ctx.hyper2d({'m+n':a,'m':b,'n':c},
        {'m+n':d,'m':e,'n':f}, x,y, **kwargs)
"""

@defun
def bihyper(ctx, a_s, b_s, z, **kwargs):
    r"""
    Evaluates the bilateral hypergeometric series

    .. math ::

        \,_AH_B(a_1, \ldots, a_k; b_1, \ldots, b_B; z) =
            \sum_{n=-\infty}^{\infty}
            \frac{(a_1)_n \ldots (a_A)_n}
                 {(b_1)_n \ldots (b_B)_n} \, z^n

    where, for direct convergence, `A = B` and `|z| = 1`, although a
    regularized sum exists more generally by considering the
    bilateral series as a sum of two ordinary hypergeometric
    functions. In order for the series to make sense, none of the
    parameters may be integers.

    **Examples**

    The value of `\,_2H_2` at `z = 1` is given by Dougall's formula::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a,b,c,d = 0.5, 1.5, 2.25, 3.25
        >>> bihyper([a,b],[c,d],1)
        -14.49118026212345786148847
        >>> gammaprod([c,d,1-a,1-b,c+d-a-b-1],[c-a,d-a,c-b,d-b])
        -14.49118026212345786148847

    The regularized function `\,_1H_0` can be expressed as the
    sum of one `\,_2F_0` function and one `\,_1F_1` function::

        >>> a = mpf(0.25)
        >>> z = mpf(0.75)
        >>> bihyper([a], [], z)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + (hyper([1],[1-a],-1/z)-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + hyper([1],[2-a],-1/z)/z/(a-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)

    **References**

    1. [Slater]_ (chapter 6: "Bilateral Series", pp. 180-189)
    2. [Wikipedia]_ http://en.wikipedia.org/wiki/Bilateral_hypergeometric_series

    """
    z = ctx.convert(z)
    c_s = a_s + b_s
    p = len(a_s)
    q = len(b_s)
    if (p, q) == (0,0) or (p, q) == (1,1):
        return ctx.zero * z
    neg = (p-q) % 2
    def h(*c_s):
        a_s = list(c_s[:p])
        b_s = list(c_s[p:])
        aa_s = [2-b for b in b_s]
        bb_s = [2-a for a in a_s]
        rp = [(-1)**neg * z] + [1-b for b in b_s] + [1-a for a in a_s]
        rc = [-1] + [1]*len(b_s) + [-1]*len(a_s)
        T1 = [], [], [], [], a_s + [1], b_s, z
        T2 = rp, rc, [], [], aa_s + [1], bb_s, (-1)**neg / z
        return T1, T2
    return ctx.hypercomb(h, c_s, **kwargs)
<|endoftext|>"""
---------------------------------------------------------------------
.. sectionauthor:: Juan Arias de Reyna <arias@us.es>

This module implements zeta-related functions using the Riemann-Siegel
expansion: zeta_offline(s,k=0)

* coef(J, eps): Need in the computation of Rzeta(s,k)

* Rzeta_simul(s, der=0) computes Rzeta^(k)(s) and Rzeta^(k)(1-s) simultaneously
  for  0 <= k <= der. Used by zeta_offline and z_offline

* Rzeta_set(s, derivatives) computes Rzeta^(k)(s) for given derivatives, used by
  z_half(t,k) and zeta_half

* z_offline(w,k): Z(w) and its derivatives of order k <= 4
* z_half(t,k): Z(t) (Riemann Siegel function) and its derivatives of order k <= 4
* zeta_offline(s): zeta(s) and its derivatives of order k<= 4
* zeta_half(1/2+it,k):  zeta(s)  and its derivatives of order k<= 4

* rs_zeta(s,k=0) Computes zeta^(k)(s)   Unifies zeta_half and zeta_offline
* rs_z(w,k=0)    Computes Z^(k)(w)      Unifies z_offline and z_half
----------------------------------------------------------------------

This program uses Riemann-Siegel expansion even to compute
zeta(s) on points s = sigma + i t  with sigma arbitrary not
necessarily equal to 1/2.

It is founded on a new deduction of the formula, with rigorous
and sharp bounds for the  terms and rest of this expansion.

More information on the papers:

 J. Arias de Reyna, High Precision Computation of Riemann's
 Zeta Function by the Riemann-Siegel Formula I, II

 We refer to them as I, II.

 In them we shall find detailed explanation of all the
 procedure.

The program uses Riemann-Siegel expansion.
This  is useful when t is big, ( say  t > 10000 ).
The precision is limited, roughly it can compute zeta(sigma+it)
with an error less than exp(-c t) for some constant c depending
on sigma.  The program gives an error when the Riemann-Siegel
formula can not compute to the wanted precision.

"""

import math

class RSCache(object):
    def __init__(ctx):
        ctx._rs_cache = [0, 10, {}, {}]

from .functions import defun

#-------------------------------------------------------------------------------#
#                                                                               #
#                       coef(ctx, J, eps, _cache=[0, 10, {} ] )                 #
#                                                                               #
#-------------------------------------------------------------------------------#

#  This function computes the coefficients c[n] defined on (I, equation (47))
#  but see also  (II, section 3.14).
#
#  Since these coefficients are very difficult to compute we save the values
#  in a cache. So if we compute several values of the functions Rzeta(s) for
#  near values of s, we do not recompute these coefficients.
#
#  c[n] are the Taylor coefficients of the function:
#
#  F(z):= (exp(pi*j*(z*z/2+3/8))-j* sqrt(2) cos(pi*z/2))/(2*cos(pi *z))
#
#

def _coef(ctx, J, eps):
    r"""
    Computes the coefficients  `c_n`  for `0\le n\le 2J` with error less than eps

    **Definition**

    The coefficients c_n are defined by

    .. math ::

        \begin{equation}
        F(z)=\frac{e^{\pi i
        \bigl(\frac{z^2}{2}+\frac38\bigr)}-i\sqrt{2}\cos\frac{\pi}{2}z}{2\cos\pi
        z}=\sum_{n=0}^\infty c_{2n} z^{2n}
        \end{equation}

    they are computed applying the relation

    .. math ::

        \begin{multline}
        c_{2n}=-\frac{i}{\sqrt{2}}\Bigl(\frac{\pi}{2}\Bigr)^{2n}
        \sum_{k=0}^n\frac{(-1)^k}{(2k)!}
        2^{2n-2k}\frac{(-1)^{n-k}E_{2n-2k}}{(2n-2k)!}+\\
        +e^{3\pi i/8}\sum_{j=0}^n(-1)^j\frac{
        E_{2j}}{(2j)!}\frac{i^{n-j}\pi^{n+j}}{(n-j)!2^{n-j+1}}.
        \end{multline}
    """

    newJ = J+2        # compute more coefficients that are needed
    neweps6 = eps/2.  # compute with a slight more precision that are needed

    #  PREPARATION FOR THE COMPUTATION OF V(N) AND W(N)
    #    See II Section 3.16
    #
    #  Computing the exponent wpvw of the error II equation (81)
    wpvw = max(ctx.mag(10*(newJ+3)), 4*newJ+5-ctx.mag(neweps6))

    #  Preparation of Euler numbers (we need until the 2*RS_NEWJ)
    E = ctx._eulernum(2*newJ)

    #  Now we have in the cache all the needed Euler numbers.
    #
    #  Computing the powers of pi
    #
    # We need to compute the powers pi**n for 1<= n <= 2*J
    # with relative error less than 2**(-wpvw)
    # it is easy to show that this is obtained
    # taking wppi as the least d with
    # 2**d>40*J and 2**d> 4.24 *newJ + 2**wpvw
    # In II Section 3.9 we need also that
    #  wppi > wptcoef[0], and that the powers
    # here computed  0<= k <= 2*newJ are more
    # than those needed there that are 2*L-2.
    # so we need  J >= L this will be checked
    # before computing tcoef[]
    wppi = max(ctx.mag(40*newJ), ctx.mag(newJ)+3 +wpvw)
    ctx.prec = wppi
    pipower = {}
    pipower[0] = ctx.one
    pipower[1] = ctx.pi
    for n in range(2,2*newJ+1):
        pipower[n] = pipower[n-1]*ctx.pi

    # COMPUTING THE COEFFICIENTS v(n) AND w(n)
    #  see II equation (61) and equations (81) and (82)
    ctx.prec = wpvw+2
    v={}
    w={}
    for n in range(0,newJ+1):
        va = (-1)**n * ctx._eulernum(2*n)
        va = ctx.mpf(va)/ctx.fac(2*n)
        v[n]=va*pipower[2*n]
    for n in range(0,2*newJ+1):
        wa = ctx.one/ctx.fac(n)
        wa=wa/(2**n)
        w[n]=wa*pipower[n]

    # COMPUTATION OF THE CONVOLUTIONS RS_P1 AND RS_P2
    #  See II Section 3.16
    ctx.prec = 15
    wpp1a = 9 - ctx.mag(neweps6)
    P1 = {}
    for n in range(0,newJ+1):
        ctx.prec = 15
        wpp1 = max(ctx.mag(10*(n+4)),4*n+wpp1a)
        ctx.prec = wpp1
        sump = 0
        for k in range(0,n+1):
            sump += ((-1)**k) * v[k]*w[2*n-2*k]
        P1[n]=((-1)**(n+1))*ctx.j*sump
    P2={}
    for n in range(0,newJ+1):
        ctx.prec = 15
        wpp2 = max(ctx.mag(10*(n+4)),4*n+wpp1a)
        ctx.prec = wpp2
        sump = 0
        for k in range(0,n+1):
            sump += (ctx.j**(n-k)) * v[k]*w[n-k]
        P2[n]=sump
    # COMPUTING THE COEFFICIENTS c[2n]
    # See II Section 3.14
    ctx.prec = 15
    wpc0 = 5 - ctx.mag(neweps6)
    wpc = max(6,4*newJ+wpc0)
    ctx.prec = wpc
    mu = ctx.sqrt(ctx.mpf('2'))/2
    nu = ctx.expjpi(3./8)/2
    c={}
    for n in range(0,newJ):
        ctx.prec = 15
        wpc = max(6,4*n+wpc0)
        ctx.prec = wpc
        c[2*n] = mu*P1[n]+nu*P2[n]
    for n in range(1,2*newJ,2):
        c[n] = 0
    return [newJ, neweps6, c, pipower]

def coef(ctx, J, eps):
    _cache = ctx._rs_cache
    if J <= _cache[0] and eps >= _cache[1]:
        return _cache[2], _cache[3]
    orig = ctx._mp.prec
    try:
        data = _coef(ctx._mp, J, eps)
    finally:
        ctx._mp.prec = orig
    if ctx is not ctx._mp:
        data[2] = dict((k,ctx.convert(v)) for (k,v) in data[2].items())
        data[3] = dict((k,ctx.convert(v)) for (k,v) in data[3].items())
    ctx._rs_cache[:] = data
    return ctx._rs_cache[2], ctx._rs_cache[3]

#-------------------------------------------------------------------------------#
#                                                                               #
#                          Rzeta_simul(s,k=0)                                   #
#                                                                               #
#-------------------------------------------------------------------------------#
#  This function return a list with the values:
#  Rzeta(sigma+it), conj(Rzeta(1-sigma+it)),Rzeta'(sigma+it), conj(Rzeta'(1-sigma+it)),
#  .... , Rzeta^{(k)}(sigma+it), conj(Rzeta^{(k)}(1-sigma+it))
#
#  Useful to compute  the function zeta(s) and Z(w)  or its derivatives.
#

def aux_M_Fp(ctx, xA, xeps4, a, xB1, xL):
    # COMPUTING M  NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    #  See II Section 3.11  equations (47) and (48)
    aux1 = 126.0657606*xA/xeps4   # 126.06.. = 316/sqrt(2*pi)
    aux1 = ctx.ln(aux1)
    aux2 = (2*ctx.ln(ctx.pi)+ctx.ln(xB1)+ctx.ln(a))/3 -ctx.ln(2*ctx.pi)/2
    m = 3*xL-3
    aux3= (ctx.loggamma(m+1)-ctx.loggamma(m/3.0+2))/2 -ctx.loggamma((m+1)/2.)
    while((aux1 < m*aux2+ aux3)and (m>1)):
        m = m - 1
        aux3 = (ctx.loggamma(m+1)-ctx.loggamma(m/3.0+2))/2 -ctx.loggamma((m+1)/2.)
    xM = m
    return xM

def aux_J_needed(ctx, xA, xeps4, a, xB1, xM):
    #  DETERMINATION OF  J  THE NUMBER OF TERMS NEEDED
    #            IN THE TAYLOR SERIES OF F.
    #  See II Section 3.11 equation (49))
    #  Only determine one
    h1 = xeps4/(632*xA)
    h2 = xB1*a * 126.31337419529260248  # = pi^2*e^2*sqrt(3)
    h2 = h1 * ctx.power((h2/xM**2),(xM-1)/3) / xM
    h3 = min(h1,h2)
    return h3

def Rzeta_simul(ctx, s, der=0):
    # First we take the value of ctx.prec
    wpinitial = ctx.prec

    # INITIALIZATION
    # Take the real and imaginary part of s
    t = ctx._im(s)
    xsigma = ctx._re(s)
    ysigma = 1 - xsigma

    # Now compute several parameter that appear on the program
    ctx.prec = 15
    a = ctx.sqrt(t/(2*ctx.pi))
    xasigma = a ** xsigma
    yasigma = a ** ysigma

    # We need a simple bound A1 < asigma  (see II Section 3.1 and 3.3)
    xA1=ctx.power(2, ctx.mag(xasigma)-1)
    yA1=ctx.power(2, ctx.mag(yasigma)-1)

    # We compute various epsilon's  (see II end of Section 3.1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps/6.
    xeps2 = eps * xA1/3.
    yeps2 = eps * yA1/3.

    #  COMPUTING SOME COEFFICIENTS THAT DEPENDS
    #                ON  sigma
    #  constant b and c  (see I  Theorem 2 formula (26) )
    #  coefficients A and B1  (see I Section 6.1 equation (50))
    #
    # here we not need high precision
    ctx.prec = 15
    if xsigma > 0:
        xb = 2.
        xc = math.pow(9,xsigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        xA = math.pow(9,xsigma)
        xB1 = 1
    else:
        xb = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        xc = math.pow(2,-xsigma)/4.44288
        xA = math.pow(2,-xsigma)
        xB1 = 1.10789   #  = 2*sqrt(1-log(2))

    if(ysigma > 0):
        yb = 2.
        yc = math.pow(9,ysigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        yA = math.pow(9,ysigma)
        yB1 = 1
    else:
        yb = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        yc = math.pow(2,-ysigma)/4.44288
        yA = math.pow(2,-ysigma)
        yB1 = 1.10789   #  = 2*sqrt(1-log(2))

    #  COMPUTING L THE NUMBER OF TERMS NEEDED IN THE RIEMANN-SIEGEL
    #                         CORRECTION
    #  See II Section 3.2
    ctx.prec = 15
    xL = 1
    while 3*xc*ctx.gamma(xL*0.5) * ctx.power(xb*a,-xL) >= xeps2:
        xL = xL+1
    xL = max(2,xL)
    yL = 1
    while 3*yc*ctx.gamma(yL*0.5) * ctx.power(yb*a,-yL) >= yeps2:
        yL = yL+1
    yL = max(2,yL)

    #  The number L has to satify some conditions.
    #  If not RS can not compute Rzeta(s) with the prescribed precision
    #  (see II, Section 3.2 condition (20)  ) and
    #  (II, Section 3.3 condition (22) ). Also we have added
    #  an additional technical  condition in Section 3.17 Proposition 17
    if ((3*xL >= 2*a*a/25.) or (3*xL+2+xsigma<0) or (abs(xsigma) > a/2.) or \
        (3*yL >= 2*a*a/25.) or (3*yL+2+ysigma<0) or (abs(ysigma) > a/2.)):
        ctx.prec = wpinitial
        raise NotImplementedError("Riemann-Siegel can not compute with such precision")

    #  We take the maximum of the two values
    L = max(xL, yL)

    #  INITIALIZATION (CONTINUATION)
    #
    # eps3 is the constant defined on (II, Section 3.5 equation (27) )
    # each term of the RS correction must be computed with error <= eps3
    xeps3 =  xeps2/(4*xL)
    yeps3 =  yeps2/(4*yL)

    # eps4 is defined on (II Section 3.6  equation (30) )
    # each component of the formula (II Section 3.6 equation (29) )
    # must be computed with error <= eps4
    xeps4 = xeps3/(3*xL)
    yeps4 = yeps3/(3*yL)

    # COMPUTING M NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    xM = aux_M_Fp(ctx, xA, xeps4, a, xB1, xL)
    yM = aux_M_Fp(ctx, yA, yeps4, a, yB1, yL)
    M = max(xM, yM)

    # COMPUTING NUMBER OF TERMS J NEEDED
    h3 = aux_J_needed(ctx, xA, xeps4, a, xB1, xM)
    h4 = aux_J_needed(ctx, yA, yeps4, a, yB1, yM)
    h3 = min(h3,h4)
    J = 12
    jvalue = (2*ctx.pi)**J / ctx.gamma(J+1)
    while jvalue > h3:
        J = J+1
        jvalue = (2*ctx.pi)*jvalue/J

    # COMPUTING eps5[m] for 1 <= m <= 21
    #  See II Section 10 equation (43)
    #  We choose the minimum of the two possibilities
    eps5={}
    xforeps5 = math.pi*math.pi*xB1*a
    yforeps5 = math.pi*math.pi*yB1*a
    for m in range(0,22):
        xaux1 = math.pow(xforeps5, m/3)/(316.*xA)
        yaux1 = math.pow(yforeps5, m/3)/(316.*yA)
        aux1 = min(xaux1, yaux1)
        aux2 = ctx.gamma(m+1)/ctx.gamma(m/3.0+0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = (aux1*aux2*min(xeps4,yeps4))

    # COMPUTING wpfp
    #  See II Section 3.13 equation (59)
    twenty = min(3*L-3, 21)+1
    aux = 6812*J
    wpfp = ctx.mag(44*J)
    for m in range(0,twenty):
        wpfp = max(wpfp, ctx.mag(aux*ctx.gamma(m+1)/eps5[m]))

    # COMPUTING N AND p
    #  See II Section
    ctx.prec = wpfp + ctx.mag(t)+20
    a = ctx.sqrt(t/(2*ctx.pi))
    N = ctx.floor(a)
    p = 1-2*(a-N)

    # now we get a rounded version of p
    # to the precision wpfp
    # this possibly is not necessary
    num=ctx.floor(p*(ctx.mpf('2')**wpfp))
    difference = p * (ctx.mpf('2')**wpfp)-num
    if (difference < 0.5):
        num = num
    else:
        num = num+1
    p = ctx.convert(num * (ctx.mpf('2')**(-wpfp)))

    # COMPUTING THE COEFFICIENTS c[n] = cc[n]
    # We shall use the notation cc[n], since there is
    # a constant that is called c
    # See II Section 3.14
    # We compute the coefficients and also save then in a
    # cache.  The bulk of the computation is passed to
    # the function  coef()
    #
    #  eps6 is defined in II Section 3.13  equation (58)
    eps6 = ctx.power(ctx.convert(2*ctx.pi), J)/(ctx.gamma(J+1)*3*J)

    #  Now we compute the coefficients
    cc = {}
    cont = {}
    cont, pipowers = coef(ctx, J, eps6)
    cc=cont.copy()   # we need a copy since we have to change his values.
    Fp={}            # this is the adequate locus of this
    for n in range(M, 3*L-2):
        Fp[n] = 0
    Fp={}
    ctx.prec = wpfp
    for m in range(0,M+1):
        sumP = 0
        for k in range(2*J-m-1,-1,-1):
            sumP = (sumP * p)+ cc[k]
        Fp[m] = sumP
        # preparation of the new coefficients
        for k in range(0,2*J-m-1):
            cc[k] = (k+1)* cc[k+1]

    # COMPUTING THE NUMBERS  xd[u,n,k], yd[u,n,k]
    #  See II Section 3.17
    #
    #  First we compute the working precisions xwpd[k]
    #   Se II equation (92)
    xwpd={}
    d1 = max(6,ctx.mag(40*L*L))
    xd2 = 13+ctx.mag((1+abs(xsigma))*xA)-ctx.mag(xeps4)-1
    xconst = ctx.ln(8/(ctx.pi*ctx.pi*a*a*xB1*xB1)) /2
    for n in range(0,L):
        xd3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*xconst)+xd2
        xwpd[n]=max(xd3,d1)

    # procedure of II Section 3.17
    ctx.prec = xwpd[1]+10
    xpsigma = 1-(2*xsigma)
    xd = {}
    xd[0,0,-2]=0; xd[0,0,-1]=0; xd[0,0,0]=1; xd[0,0,1]=0
    xd[0,-1,-2]=0; xd[0,-1,-1]=0; xd[0,-1,0]=1; xd[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = xwpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if(m!=0):
                m1 = ctx.one/m
                c1= m1/4
                c2=(xpsigma*m1)/2
                c3=-(m+1)
                xd[0,n,k]=c3*xd[0,n-1,k-2]+c1*xd[0,n-1,k]+c2*xd[0,n-1,k-1]
            else:
                xd[0,n,k]=0
                for r in range(0,k):
                    add=xd[0,n,r]*(ctx.mpf('1.0')*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    xd[0,n,k] -= ((-1)**(k-r))*add
        xd[0,n,-2]=0; xd[0,n,-1]=0; xd[0,n,3*n//2+1]=0
    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if( (mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    xd[mu,n,k] = 0
    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = xwpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*xd[mu-2,n-2,k-3]+2*(xsigma+n-2)*xd[mu-1,n-2,k-3]
                xd[mu,n,k] = aux - xd[mu-1,n-1,k-1]

    #  Now we compute the working precisions ywpd[k]
    #   Se II equation (92)
    ywpd={}
    d1 = max(6,ctx.mag(40*L*L))
    yd2 = 13+ctx.mag((1+abs(ysigma))*yA)-ctx.mag(yeps4)-1
    yconst = ctx.ln(8/(ctx.pi*ctx.pi*a*a*yB1*yB1)) /2
    for n in range(0,L):
        yd3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*yconst)+yd2
        ywpd[n]=max(yd3,d1)

    # procedure of II Section 3.17
    ctx.prec = ywpd[1]+10
    ypsigma = 1-(2*ysigma)
    yd = {}
    yd[0,0,-2]=0; yd[0,0,-1]=0; yd[0,0,0]=1; yd[0,0,1]=0
    yd[0,-1,-2]=0; yd[0,-1,-1]=0; yd[0,-1,0]=1; yd[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = ywpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if(m!=0):
                m1 = ctx.one/m
                c1= m1/4
                c2=(ypsigma*m1)/2
                c3=-(m+1)
                yd[0,n,k]=c3*yd[0,n-1,k-2]+c1*yd[0,n-1,k]+c2*yd[0,n-1,k-1]
            else:
                yd[0,n,k]=0
                for r in range(0,k):
                    add=yd[0,n,r]*(ctx.mpf('1.0')*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    yd[0,n,k] -= ((-1)**(k-r))*add
        yd[0,n,-2]=0; yd[0,n,-1]=0; yd[0,n,3*n//2+1]=0

    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if( (mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    yd[mu,n,k] = 0
    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = ywpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*yd[mu-2,n-2,k-3]+2*(ysigma+n-2)*yd[mu-1,n-2,k-3]
                yd[mu,n,k] = aux - yd[mu-1,n-1,k-1]

    # COMPUTING THE COEFFICIENTS xtcoef[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    xwptcoef={}
    xwpterm={}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    xc2 = ctx.mag(68*(L+2)*xA)
    xc4 = ctx.mag(xB1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        xc3 = xc2 - k*xc4+ctx.mag(ctx.fac(k+0.5))/2.
        xwptcoef[k] = (max(c1,xc3-ctx.mag(xeps4)+1)+1 +20)*1.5
        xwpterm[k] = (max(c1,ctx.mag(L+2)+xc3-ctx.mag(xeps3)+1)+1 +20)
    ywptcoef={}
    ywpterm={}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
