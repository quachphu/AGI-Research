        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for an Elliptic (Cauer) filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `ellip` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `ellip`.

    See Also
    --------
    ellip : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design an analog highpass filter such that the passband is within 3 dB
    above 30 rad/s, while rejecting -60 dB at 10 rad/s. Plot its
    frequency response, showing the passband and stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> N, Wn = signal.ellipord(30, 10, 3, 60, True)
    >>> b, a = signal.ellip(N, 3, 60, Wn, 'high', True)
    >>> w, h = signal.freqs(b, a, np.logspace(0, 3, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Elliptical highpass filter fit to constraints')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.1, 10,  10,  .1], [1e4, 1e4, -60, -60], '0.9', lw=0) # stop
    >>> plt.fill([30, 30, 1e9, 1e9], [-99,  -3,  -3, -99], '0.9', lw=0) # pass
    >>> plt.axis([1, 300, -80, 3])
    >>> plt.show()

    """
    fs = _validate_fs(fs, allow_none=True)
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'ellip')

    arg1_sq = _pow10m1(0.1 * gpass) / _pow10m1(0.1 * gstop)
    arg0 = 1.0 / nat
    d0 = special.ellipk(arg0 ** 2), special.ellipkm1(arg0 ** 2)
    d1 = special.ellipk(arg1_sq), special.ellipkm1(arg1_sq)
    ord = int(ceil(d0[0] * d1[1] / (d0[1] * d1[0])))

    wn = _postprocess_wn(passb, analog, fs)

    return ord, wn


def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.

    See Also
    --------
    butter : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * pi * m / (2 * N))
    k = 1
    return z, p, k


def cheb1ap(N, rp):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rp` decibels of ripple in the passband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    cheby1 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero error
        # Even order filters have DC gain of -rp dB
        return np.array([]), np.array([]), 10**(-rp/20)
    z = np.array([])

    # Ripple factor (epsilon)
    eps = np.sqrt(10 ** (0.1 * rp) - 1.0)
    mu = 1.0 / N * arcsinh(1 / eps)

    # Arrange poles in an ellipse on the left half of the S-plane
    m = np.arange(-N+1, N, 2)
    theta = pi * m / (2*N)
    p = -sinh(mu + 1j*theta)

    k = np.prod(-p, axis=0).real
    if N % 2 == 0:
        k = k / sqrt(1 + eps * eps)

    return z, p, k


def cheb2ap(N, rs):
    """
    Return (z,p,k) for Nth-order Chebyshev type II analog lowpass filter.

    The returned filter prototype has attenuation of at least ``rs`` decibels
    in the stopband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the attenuation first reaches ``rs``.

    See Also
    --------
    cheby2 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        return np.array([]), np.array([]), 1

    # Ripple factor (epsilon)
    de = 1.0 / sqrt(10 ** (0.1 * rs) - 1)
    mu = arcsinh(1.0 / de) / N

    if N % 2:
        m = np.concatenate((np.arange(-N+1, 0, 2), np.arange(2, N, 2)))
    else:
        m = np.arange(-N+1, N, 2)

    z = -conjugate(1j / sin(m * pi / (2.0 * N)))

    # Poles around the unit circle like Butterworth
    p = -exp(1j * pi * np.arange(-N+1, N, 2) / (2 * N))
    # Warp into Chebyshev II
    p = sinh(mu) * p.real + 1j * cosh(mu) * p.imag
    p = 1.0 / p

    k = (np.prod(-p, axis=0) / np.prod(-z, axis=0)).real
    return z, p, k


EPSILON = 2e-16

# number of terms in solving degree equation
_ELLIPDEG_MMAX = 7


def _ellipdeg(n, m1):
    """Solve degree equation using nomes

    Given n, m1, solve
       n * K(m) / K'(m) = K1(m1) / K1'(m1)
    for m

    See [1], Eq. (49)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """
    K1 = special.ellipk(m1)
    K1p = special.ellipkm1(m1)

    q1 = np.exp(-np.pi * K1p / K1)
    q = q1 ** (1/n)

    mnum = np.arange(_ELLIPDEG_MMAX + 1)
    mden = np.arange(1, _ELLIPDEG_MMAX + 2)

    num = np.sum(q ** (mnum * (mnum+1)))
    den = 1 + 2 * np.sum(q ** (mden**2))

    return 16 * q * (num / den) ** 4


# Maximum number of iterations in Landen transformation recursion
# sequence.  10 is conservative; unit tests pass with 4, Orfanidis
# (see _arc_jac_cn [1]) suggests 5.
_ARC_JAC_SN_MAXITER = 10


def _arc_jac_sn(w, m):
    """Inverse Jacobian elliptic sn

    Solve for z in w = sn(z, m)

    Parameters
    ----------
    w : complex scalar
        argument

    m : scalar
        modulus; in interval [0, 1]


    See [1], Eq. (56)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

    """

    def _complement(kx):
        # (1-k**2) ** 0.5; the expression below
        # works for small kx
        return ((1 - kx) * (1 + kx)) ** 0.5

    k = m ** 0.5

    if k > 1:
        return np.nan
    elif k == 1:
        return np.arctanh(w)

    ks = [k]
    niter = 0
    while ks[-1] != 0:
        k_ = ks[-1]
        k_p = _complement(k_)
        ks.append((1 - k_p) / (1 + k_p))
        niter += 1
        if niter > _ARC_JAC_SN_MAXITER:
            raise ValueError('Landen transformation not converging')

    K = np.prod(1 + np.array(ks[1:])) * np.pi/2

    wns = [w]

    for kn, knext in zip(ks[:-1], ks[1:]):
        wn = wns[-1]
        wnext = (2 * wn /
                 ((1 + knext) * (1 + _complement(kn * wn))))
        wns.append(wnext)

    u = 2 / np.pi * np.arcsin(wns[-1])

    z = K * u
    return z


def _arc_jac_sc1(w, m):
    """Real inverse Jacobian sc, with complementary modulus

    Solve for z in w = sc(z, 1-m)

    w - real scalar

    m - modulus

    From [1], sc(z, m) = -i * sn(i * z, 1 - m)

    References
    ----------
    # noqa: E501
    .. [1] https://functions.wolfram.com/EllipticFunctions/JacobiSC/introductions/JacobiPQs/ShowAll.html,
       "Representations through other Jacobi functions"

    """

    zcomplex = _arc_jac_sn(1j * w, m)
    if abs(zcomplex.real) > 1e-14:
        raise ValueError

    return zcomplex.imag


def ellipap(N, rp, rs):
    """Return (z,p,k) of Nth-order elliptic analog lowpass filter.

    The filter is a normalized prototype that has `rp` decibels of ripple
    in the passband and a stopband `rs` decibels down.

    The filter's angular (e.g., rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    ellip : Filter design function using this prototype

    References
    ----------
    .. [1] Lutovac, Tosic, and Evans, "Filter Design for Signal Processing",
           Chapters 5 and 12.

    .. [2] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        # Even order filters have DC gain of -rp dB
        return np.array([]), np.array([]), 10**(-rp/20)
    elif N == 1:
        p = -sqrt(1.0 / _pow10m1(0.1 * rp))
        k = -p
        z = []
        return asarray(z), asarray(p), k

    eps_sq = _pow10m1(0.1 * rp)

    eps = np.sqrt(eps_sq)
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0:
        raise ValueError("Cannot design a filter with given rp and rs"
                         " specifications.")

    val = special.ellipk(ck1_sq), special.ellipkm1(ck1_sq)

    m = _ellipdeg(N, ck1_sq)

    capk = special.ellipk(m)

    j = np.arange(1 - N % 2, N, 2)
    jj = len(j)

    [s, c, d, phi] = special.ellipj(j * capk / N, m * np.ones(jj))
    snew = np.compress(abs(s) > EPSILON, s, axis=-1)
    z = 1.0 / (sqrt(m) * snew)
    z = 1j * z
    z = np.concatenate((z, conjugate(z)))

    r = _arc_jac_sc1(1. / eps, ck1_sq)
    v0 = capk * r / (N * val[0])

    [sv, cv, dv, phi] = special.ellipj(v0, 1 - m)
    p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)

    if N % 2:
        newp = np.compress(
            abs(p.imag) > EPSILON * np.sqrt(np.sum(p * np.conjugate(p), axis=0).real),
            p, axis=-1
        )
        p = np.concatenate((p, conjugate(newp)))
    else:
        p = np.concatenate((p, conjugate(p)))

    k = (np.prod(-p, axis=0) / np.prod(-z, axis=0)).real
    if N % 2 == 0:
        k = k / np.sqrt(1 + eps_sq)

    return z, p, k


# TODO: Make this a real public function scipy.misc.ff
def _falling_factorial(x, n):
    r"""
    Return the factorial of `x` to the `n` falling.

    This is defined as:

    .. math::   x^\underline n = (x)_n = x (x-1) \cdots (x-n+1)

    This can more efficiently calculate ratios of factorials, since:

    n!/m! == falling_factorial(n, n-m)

    where n >= m

    skipping the factors that cancel out

    the usual factorial n! == ff(n, n)
    """
    val = 1
    for k in range(x - n + 1, x + 1):
        val *= k
    return val


def _bessel_poly(n, reverse=False):
    """
    Return the coefficients of Bessel polynomial of degree `n`

    If `reverse` is true, a reverse Bessel polynomial is output.

    Output is a list of coefficients:
    [1]                   = 1
    [1,  1]               = 1*s   +  1
    [1,  3,  3]           = 1*s^2 +  3*s   +  3
    [1,  6, 15, 15]       = 1*s^3 +  6*s^2 + 15*s   +  15
    [1, 10, 45, 105, 105] = 1*s^4 + 10*s^3 + 45*s^2 + 105*s + 105
    etc.

    Output is a Python list of arbitrary precision long ints, so n is only
    limited by your hardware's memory.

    Sequence is http://oeis.org/A001498, and output can be confirmed to
    match http://oeis.org/A001498/b001498.txt :

    >>> from scipy.signal._filter_design import _bessel_poly
    >>> i = 0
    >>> for n in range(51):
    ...     for x in _bessel_poly(n, reverse=True):
    ...         print(i, x)
    ...         i += 1

    """
    if abs(int(n)) != n:
        raise ValueError("Polynomial order must be a nonnegative integer")
    else:
        n = int(n)  # np.int32 doesn't work, for instance

    out = []
    for k in range(n + 1):
        num = _falling_factorial(2*n - k, n)
        den = 2**(n - k) * math.factorial(k)
        out.append(num // den)

    if reverse:
        return out[::-1]
    else:
        return out


def _campos_zeros(n):
    """
    Return approximate zero locations of Bessel polynomials y_n(x) for order
    `n` using polynomial fit (Campos-Calderon 2011)
    """
    if n == 1:
        return asarray([-1+0j])

    s = npp_polyval(n, [0, 0, 2, 0, -3, 1])
    b3 = npp_polyval(n, [16, -8]) / s
    b2 = npp_polyval(n, [-24, -12, 12]) / s
    b1 = npp_polyval(n, [8, 24, -12, -2]) / s
    b0 = npp_polyval(n, [0, -6, 0, 5, -1]) / s

    r = npp_polyval(n, [0, 0, 2, 1])
    a1 = npp_polyval(n, [-6, -6]) / r
    a2 = 6 / r

    k = np.arange(1, n+1)
    x = npp_polyval(k, [0, a1, a2])
    y = npp_polyval(k, [b0, b1, b2, b3])

    return x + 1j*y


def _aberth(f, fp, x0, tol=1e-15, maxiter=50):
    """
    Given a function `f`, its first derivative `fp`, and a set of initial
    guesses `x0`, simultaneously find the roots of the polynomial using the
    Aberth-Ehrlich method.

    ``len(x0)`` should equal the number of roots of `f`.

    (This is not a complete implementation of Bini's algorithm.)
    """

    N = len(x0)

    x = array(x0, complex)
    beta = np.empty_like(x0)

    for iteration in range(maxiter):
        alpha = -f(x) / fp(x)  # Newton's method

        # Model "repulsion" between zeros
        for k in range(N):
            beta[k] = np.sum(1/(x[k] - x[k+1:]))
            beta[k] += np.sum(1/(x[k] - x[:k]))

        x += alpha / (1 + alpha * beta)

        if not all(np.isfinite(x)):
            raise RuntimeError('Root-finding calculation failed')

        # Mekwi: The iterative process can be stopped when |hn| has become
        # less than the largest error one is willing to permit in the root.
        if all(abs(alpha) <= tol):
            break
    else:
        raise Exception('Zeros failed to converge')

    return x


def _bessel_zeros(N):
    """
    Find zeros of ordinary Bessel polynomial of order `N`, by root-finding of
    modified Bessel function of the second kind
    """
    if N == 0:
        return asarray([])

    # Generate starting points
    x0 = _campos_zeros(N)

    # Zeros are the same for exp(1/x)*K_{N+0.5}(1/x) and Nth-order ordinary
    # Bessel polynomial y_N(x)
    def f(x):
        return special.kve(N+0.5, 1/x)

    # First derivative of above
    def fp(x):
        return (special.kve(N-0.5, 1/x)/(2*x**2) -
                special.kve(N+0.5, 1/x)/(x**2) +
                special.kve(N+1.5, 1/x)/(2*x**2))

    # Starting points converge to true zeros
    x = _aberth(f, fp, x0)

    # Improve precision using Newton's method on each
    for i in range(len(x)):
        x[i] = optimize.newton(f, x[i], fp, tol=1e-15)

    # Average complex conjugates to make them exactly symmetrical
    x = np.mean((x, x[::-1].conj()), 0)

    # Zeros should sum to -1
    if abs(np.sum(x) + 1) > 1e-15:
        raise RuntimeError('Generated zeros are inaccurate')

    return x


def _norm_factor(p, k):
    """
    Numerically find frequency shift to apply to delay-normalized filter such
    that -3 dB point is at 1 rad/sec.

    `p` is an array_like of polynomial poles
    `k` is a float gain

    First 10 values are listed in "Bessel Scale Factors" table,
    "Bessel Filters Polynomials, Poles and Circuit Elements 2003, C. Bond."
    """
    p = asarray(p, dtype=complex)

    def G(w):
        """
        Gain of filter
        """
        return abs(k / prod(1j*w - p))

    def cutoff(w):
        """
        When gain = -3 dB, return 0
        """
        return G(w) - 1/np.sqrt(2)

    return optimize.newton(cutoff, 1.5)


def besselap(N, norm='phase'):
    """
    Return (z,p,k) for analog prototype of an Nth-order Bessel filter.

    Parameters
    ----------
    N : int
        The order of the filter.
    norm : {'phase', 'delay', 'mag'}, optional
        Frequency normalization:

        ``phase``
            The filter is normalized such that the phase response reaches its
            midpoint at an angular (e.g., rad/s) cutoff frequency of 1. This
            happens for both low-pass and high-pass filters, so this is the
            "phase-matched" case. [6]_

            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.

            This is the default, and matches MATLAB's implementation.

        ``delay``
            The filter is normalized such that the group delay in the passband
            is 1 (e.g., 1 second). This is the "natural" type obtained by
            solving Bessel polynomials

        ``mag``
            The filter is normalized such that the gain magnitude is -3 dB at
            angular frequency 1. This is called "frequency normalization" by
            Bond. [1]_

        .. versionadded:: 0.18.0

    Returns
    -------
    z : ndarray
        Zeros of the transfer function. Is always an empty array.
    p : ndarray
        Poles of the transfer function.
    k : scalar
        Gain of the transfer function. For phase-normalized, this is always 1.

    See Also
    --------
    bessel : Filter design function using this prototype

    Notes
    -----
    To find the pole locations, approximate starting points are generated [2]_
    for the zeros of the ordinary Bessel polynomial [3]_, then the
    Aberth-Ehrlich method [4]_ [5]_ is used on the Kv(x) Bessel function to
    calculate more accurate zeros, and these locations are then inverted about
    the unit circle.

    References
    ----------
    .. [1] C.R. Bond, "Bessel Filter Constants",
           http://www.crbond.com/papers/bsf.pdf
    .. [2] Campos and Calderon, "Approximate closed-form formulas for the
           zeros of the Bessel Polynomials", :arXiv:`1105.0957`.
    .. [3] Thomson, W.E., "Delay Networks having Maximally Flat Frequency
           Characteristics", Proceedings of the Institution of Electrical
           Engineers, Part III, November 1949, Vol. 96, No. 44, pp. 487-490.
    .. [4] Aberth, "Iteration Methods for Finding all Zeros of a Polynomial
           Simultaneously", Mathematics of Computation, Vol. 27, No. 122,
           April 1973
    .. [5] Ehrlich, "A modified Newton method for polynomials", Communications
           of the ACM, Vol. 10, Issue 2, pp. 107-108, Feb. 1967,
           :DOI:`10.1145/363067.363115`
    .. [6] Miller and Bohn, "A Bessel Filter Crossover, and Its Relation to
           Others", RaneNote 147, 1998,
           https://www.ranecommercial.com/legacy/note147.html

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")

    N = int(N)  # calculation below doesn't always fit in np.int64
    if N == 0:
        p = []
        k = 1
    else:
        # Find roots of reverse Bessel polynomial
        p = 1/_bessel_zeros(N)

        a_last = _falling_factorial(2*N, N) // 2**N

        # Shift them to a different normalization if required
        if norm in ('delay', 'mag'):
            # Normalized for group delay of 1
            k = a_last
            if norm == 'mag':
                # -3 dB magnitude point is at 1 rad/sec
                norm_factor = _norm_factor(p, k)
                p /= norm_factor
                k = norm_factor**-N * a_last
        elif norm == 'phase':
            # Phase-matched (1/2 max phase shift at 1 rad/sec)
            # Asymptotes are same as Butterworth filter
            p *= 10**(-math.log10(a_last)/N)
            k = 1
        else:
            raise ValueError('normalization not understood')

    return asarray([]), asarray(p, dtype=complex), float(k)


def iirnotch(w0, Q, fs=2.0):
    """
    Design second-order IIR notch digital filter.

    A notch filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    Parameters
    ----------
    w0 : float
        Frequency to remove from a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirpeak

    Notes
    -----
    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the 60 Hz component from a
    signal sampled at 200 Hz, using a quality factor Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fs = 200.0  # Sample frequency (Hz)
    >>> f0 = 60.0  # Frequency to be removed from signal (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design notch filter
    >>> b, a = signal.iirnotch(f0, Q, fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude [dB]", color='blue')
    >>> ax[0].set_xlim([0, 100])
    >>> ax[0].set_ylim([-25, 10])
    >>> ax[0].grid(True)
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Phase [deg]", color='green')
    >>> ax[1].set_xlabel("Frequency [Hz]")
    >>> ax[1].set_xlim([0, 100])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid(True)
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "notch", fs)


def iirpeak(w0, Q, fs=2.0):
    """
    Design second-order IIR peak (resonant) digital filter.

    A peak filter is a band-pass filter with a narrow bandwidth
    (high quality factor). It rejects components outside a narrow
    frequency band.

    Parameters
    ----------
    w0 : float
        Frequency to be retained in a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        peak filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirnotch

    Notes
    -----
    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the frequencies other than the 300 Hz
    component from a signal sampled at 1000 Hz, using a quality factor Q = 30

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 1000.0  # Sample frequency (Hz)
    >>> f0 = 300.0  # Frequency to be retained (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design peak filter
    >>> b, a = signal.iirpeak(f0, Q, fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude [dB]", color='blue')
    >>> ax[0].set_xlim([0, 500])
    >>> ax[0].set_ylim([-50, 10])
    >>> ax[0].grid(True)
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Phase [deg]", color='green')
    >>> ax[1].set_xlabel("Frequency [Hz]")
    >>> ax[1].set_xlim([0, 500])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid(True)
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "peak", fs)


def _design_notch_peak_filter(w0, Q, ftype, fs=2.0):
    """
    Design notch or peak digital filter.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. If `fs` is specified,
        this is in the same units as `fs`. By default, it is a normalized
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:

            - notch filter : ``notch``
            - peak filter  : ``peak``
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0:

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """
    fs = _validate_fs(fs, allow_none=False)

    # Guarantee that the inputs are floats
    w0 = float(w0)
    Q = float(Q)
    w0 = 2*w0/fs

    # Checks if w0 is within the range
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError("w0 should be such that 0 < w0 < 1")

    # Get bandwidth
    bw = w0/Q

    # Normalize inputs
    bw = bw*np.pi
    w0 = w0*np.pi

    if ftype not in ("notch", "peak"):
        raise ValueError("Unknown ftype.")

    # Compute beta according to Eqs. 11.3.4 (p.575) and 11.3.19 (p.579) from
    # reference [1]. Due to assuming a -3 dB attenuation value, i.e, assuming
    # gb = 1 / np.sqrt(2), the following terms simplify to:
    #   (np.sqrt(1.0 - gb**2.0) / gb) = 1
    #   (gb / np.sqrt(1.0 - gb**2.0)) = 1
    beta = np.tan(bw/2.0)

    # Compute gain: formula 11.3.6 (p.575) from reference [1]
    gain = 1.0/(1.0+beta)

    # Compute numerator b and denominator a
    # formulas 11.3.7 (p.575) and 11.3.21 (p.579)
    # from reference [1]
    if ftype == "notch":
        b = gain*np.array([1.0, -2.0*np.cos(w0), 1.0])
    else:
        b = (1.0-gain)*np.array([1.0, 0.0, -1.0])
    a = np.array([1.0, -2.0*gain*np.cos(w0), (2.0*gain-1.0)])

    return b, a


def iircomb(w0, Q, ftype='notch', fs=2.0, *, pass_zero=False):
    """
    Design IIR notching or peaking digital comb filter.

    A notching comb filter consists of regularly-spaced band-stop filters with
    a narrow bandwidth (high quality factor). Each rejects a narrow frequency
    band and leaves the rest of the spectrum little changed.

    A peaking comb filter consists of regularly-spaced band-pass filters with
    a narrow bandwidth (high quality factor). Each rejects components outside
    a narrow frequency band.

    Parameters
    ----------
    w0 : float
        The fundamental frequency of the comb filter (the spacing between its
        peaks). This must evenly divide the sampling frequency. If `fs` is
        specified, this is in the same units as `fs`. By default, it is
        a normalized scalar that must satisfy  ``0 < w0 < 1``, with
        ``w0 = 1`` corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : {'notch', 'peak'}
        The type of comb filter generated by the function. If 'notch', then
        the Q factor applies to the notches. If 'peak', then the Q factor
        applies to the peaks.  Default is 'notch'.
    fs : float, optional
        The sampling frequency of the signal. Default is 2.0.
    pass_zero : bool, optional
        If False (default), the notches (nulls) of the filter are centered on
        frequencies [0, w0, 2*w0, ...], and the peaks are centered on the
        midpoints [w0/2, 3*w0/2, 5*w0/2, ...].  If True, the peaks are centered
        on [0, w0, 2*w0, ...] (passing zero frequency) and vice versa.

        .. versionadded:: 1.9.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    Raises
    ------
    ValueError
        If `w0` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `fs` is not divisible by `w0`, if `ftype`
        is not 'notch' or 'peak'

    See Also
    --------
    iirnotch
    iirpeak

    Notes
    -----
    For implementation details, see [1]_. The TF implementation of the
    comb filter is numerically stable even at higher orders due to the
    use of a single repeated pole, which won't suffer from precision loss.

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996, ch. 11, "Digital Filter Design"

    Examples
    --------
    Design and plot notching comb filter at 20 Hz for a
    signal sampled at 200 Hz, using quality factor Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fs = 200.0  # Sample frequency (Hz)
    >>> f0 = 20.0  # Frequency to be removed from signal (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design notching comb filter
    >>> b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> response = abs(h)
    >>> # To avoid divide by zero when graphing
    >>> response[response == 0] = 1e-20
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    >>> ax[0].plot(freq, 20*np.log10(abs(response)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude [dB]", color='blue')
    >>> ax[0].set_xlim([0, 100])
    >>> ax[0].set_ylim([-30, 10])
    >>> ax[0].grid(True)
    >>> ax[1].plot(freq, (np.angle(h)*180/np.pi+180)%360 - 180, color='green')
    >>> ax[1].set_ylabel("Phase [deg]", color='green')
    >>> ax[1].set_xlabel("Frequency [Hz]")
    >>> ax[1].set_xlim([0, 100])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid(True)
    >>> plt.show()

    Design and plot peaking comb filter at 250 Hz for a
    signal sampled at 1000 Hz, using quality factor Q = 30

    >>> fs = 1000.0  # Sample frequency (Hz)
    >>> f0 = 250.0  # Frequency to be retained (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design peaking filter
    >>> b, a = signal.iircomb(f0, Q, ftype='peak', fs=fs, pass_zero=True)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> response = abs(h)
    >>> # To avoid divide by zero when graphing
    >>> response[response == 0] = 1e-20
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    >>> ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude [dB]", color='blue')
    >>> ax[0].set_xlim([0, 500])
    >>> ax[0].set_ylim([-80, 10])
    >>> ax[0].grid(True)
    >>> ax[1].plot(freq, (np.angle(h)*180/np.pi+180)%360 - 180, color='green')
    >>> ax[1].set_ylabel("Phase [deg]", color='green')
    >>> ax[1].set_xlabel("Frequency [Hz]")
    >>> ax[1].set_xlim([0, 500])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid(True)
    >>> plt.show()
    """

    # Convert w0, Q, and fs to float
    w0 = float(w0)
    Q = float(Q)
    fs = _validate_fs(fs, allow_none=False)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    if not 0 < w0 < fs / 2:
        raise ValueError(f"w0 must be between 0 and {fs / 2}"
                         f" (Nyquist), but given {w0}.")
    if ftype not in ('notch', 'peak'):
        raise ValueError('ftype must be either notch or peak.')

    # Compute the order of the filter
    N = round(fs / w0)

    # Check for cutoff frequency divisibility
    if abs(w0 - fs/N)/fs > 1e-14:
        raise ValueError('fs must be divisible by w0.')

    # Compute frequency in radians and filter bandwidth
    # Eq. 11.3.1 (p. 574) from reference [1]
    w0 = (2 * np.pi * w0) / fs
    w_delta = w0 / Q

    # Define base gain values depending on notch or peak filter
    # Compute -3dB attenuation
    # Eqs. 11.4.1 and 11.4.2 (p. 582) from reference [1]
    if ftype == 'notch':
        G0, G = 1, 0
    elif ftype == 'peak':
        G0, G = 0, 1

    # Compute beta according to Eq. 11.5.3 (p. 591) from reference [1]. Due to
    # assuming a -3 dB attenuation value, i.e, assuming GB = 1 / np.sqrt(2),
    # the following term simplifies to:
    #   np.sqrt((GB**2 - G0**2) / (G**2 - GB**2)) = 1
    beta = np.tan(N * w_delta / 4)

    # Compute filter coefficients
    # Eq 11.5.1 (p. 590) variables a, b, c from reference [1]
    ax = (1 - beta) / (1 + beta)
    bx = (G0 + G * beta) / (1 + beta)
    cx = (G0 - G * beta) / (1 + beta)

    # Last coefficients are negative to get peaking comb that passes zero or
    # notching comb that doesn't.
    negative_coef = ((ftype == 'peak' and pass_zero) or
                     (ftype == 'notch' and not pass_zero))

    # Compute numerator coefficients
    # Eq 11.5.1 (p. 590) or Eq 11.5.4 (p. 591) from reference [1]
    # b - cz^-N or b + cz^-N
    b = np.zeros(N + 1)
    b[0] = bx
    if negative_coef:
        b[-1] = -cx
    else:
        b[-1] = +cx

    # Compute denominator coefficients
    # Eq 11.5.1 (p. 590) or Eq 11.5.4 (p. 591) from reference [1]
    # 1 - az^-N or 1 + az^-N
    a = np.zeros(N + 1)
    a[0] = 1
    if negative_coef:
        a[-1] = -ax
    else:
        a[-1] = +ax

    return b, a


def _hz_to_erb(hz):
    """
    Utility for converting from frequency (Hz) to the
    Equivalent Rectangular Bandwidth (ERB) scale
    ERB = frequency / EarQ + minBW
    """
    EarQ = 9.26449
    minBW = 24.7
    return hz / EarQ + minBW


def gammatone(freq, ftype, order=None, numtaps=None, fs=None):
    """
    Gammatone filter design.

    This function computes the coefficients of an FIR or IIR gammatone
    digital filter [1]_.

    Parameters
    ----------
    freq : float
        Center frequency of the filter (expressed in the same units
        as `fs`).
    ftype : {'fir', 'iir'}
        The type of filter the function generates. If 'fir', the function
        will generate an Nth order FIR gammatone filter. If 'iir', the
        function will generate an 8th order digital IIR filter, modeled as
        as 4th order gammatone filter.
    order : int, optional
        The order of the filter. Only used when ``ftype='fir'``.
        Default is 4 to model the human auditory system. Must be between
        0 and 24.
    numtaps : int, optional
        Length of the filter. Only used when ``ftype='fir'``.
        Default is ``fs*0.015`` if `fs` is greater than 1000,
        15 if `fs` is less than or equal to 1000.
    fs : float, optional
        The sampling frequency of the signal. `freq` must be between
        0 and ``fs/2``. Default is 2.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials of the filter.

    Raises
    ------
    ValueError
        If `freq` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `ftype` is not 'fir' or 'iir', if `order` is less than
        or equal to 0 or greater than 24 when ``ftype='fir'``

    See Also
    --------
    firwin
    iirfilter

    References
    ----------
    .. [1] Slaney, Malcolm, "An Efficient Implementation of the
        Patterson-Holdsworth Auditory Filter Bank", Apple Computer
        Technical Report 35, 1993, pp.3-8, 34-39.

    Examples
    --------
    16-sample 4th order FIR Gammatone filter centered at 440 Hz

    >>> from scipy import signal
    >>> signal.gammatone(440, 'fir', numtaps=16, fs=16000)
    (array([ 0.00000000e+00,  2.22196719e-07,  1.64942101e-06,  4.99298227e-06,
        1.01993969e-05,  1.63125770e-05,  2.14648940e-05,  2.29947263e-05,
        1.76776931e-05,  2.04980537e-06, -2.72062858e-05, -7.28455299e-05,
       -1.36651076e-04, -2.19066855e-04, -3.18905076e-04, -4.33156712e-04]),
       [1.0])

    IIR Gammatone filter centered at 440 Hz

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fc, fs = 440, 16000
    >>> b, a = signal.gammatone(fc, 'iir', fs=fs)
    >>> w, h = signal.freqz(b, a)
    >>> plt.plot(w * fs / (2 * np.pi), 20 * np.log10(abs(h)))
    >>> plt.xscale('log')
    >>> plt.title('Gammatone filter frequency response')
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(fc, color='green') # cutoff frequency
    >>> plt.show()
    """
    # Converts freq to float
    freq = float(freq)

    # Set sampling rate if not passed
    if fs is None:
        fs = 2
    fs = _validate_fs(fs, allow_none=False)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    filter_types = ['fir', 'iir']
    if not 0 < freq < fs / 2:
        raise ValueError(f"The frequency must be between 0 and {fs / 2}"
                         f" (Nyquist), but given {freq}.")
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')

    # Calculate FIR gammatone filter
    if ftype == 'fir':
        # Set order and numtaps if not passed
        if order is None:
            order = 4
        order = operator.index(order)

        if numtaps is None:
            numtaps = max(int(fs * 0.015), 15)
        numtaps = operator.index(numtaps)

        # Check for invalid order
        if not 0 < order <= 24:
            raise ValueError("Invalid order: order must be > 0 and <= 24.")

        # Gammatone impulse response settings
        t = np.arange(numtaps) / fs
        bw = 1.019 * _hz_to_erb(freq)

        # Calculate the FIR gammatone filter
        b = (t ** (order - 1)) * np.exp(-2 * np.pi * bw * t)
        b *= np.cos(2 * np.pi * freq * t)

        # Scale the FIR filter so the frequency response is 1 at cutoff
        scale_factor = 2 * (2 * np.pi * bw) ** (order)
        scale_factor /= float_factorial(order - 1)
        scale_factor /= fs
        b *= scale_factor
        a = [1.0]

    # Calculate IIR gammatone filter
    elif ftype == 'iir':
        # Raise warning if order and/or numtaps is passed
        if order is not None:
            warnings.warn('order is not used for IIR gammatone filter.', stacklevel=2)
        if numtaps is not None:
            warnings.warn('numtaps is not used for IIR gammatone filter.', stacklevel=2)

        # Gammatone impulse response settings
        T = 1./fs
        bw = 2 * np.pi * 1.019 * _hz_to_erb(freq)
        fr = 2 * freq * np.pi * T
        bwT = bw * T

        # Calculate the gain to normalize the volume at the center frequency
        g1 = -2 * np.exp(2j * fr) * T
        g2 = 2 * np.exp(-(bwT) + 1j * fr) * T
        g3 = np.sqrt(3 + 2 ** (3 / 2)) * np.sin(fr)
        g4 = np.sqrt(3 - 2 ** (3 / 2)) * np.sin(fr)
        g5 = np.exp(2j * fr)

        g = g1 + g2 * (np.cos(fr) - g4)
        g *= (g1 + g2 * (np.cos(fr) + g4))
        g *= (g1 + g2 * (np.cos(fr) - g3))
        g *= (g1 + g2 * (np.cos(fr) + g3))
        g /= ((-2 / np.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / np.exp(bwT)) ** 4)
        g = np.abs(g)

        # Create empty filter coefficient lists
        b = np.empty(5)
        a = np.empty(9)

        # Calculate the numerator coefficients
        b[0] = (T ** 4) / g
        b[1] = -4 * T ** 4 * np.cos(fr) / np.exp(bw * T) / g
        b[2] = 6 * T ** 4 * np.cos(2 * fr) / np.exp(2 * bw * T) / g
        b[3] = -4 * T ** 4 * np.cos(3 * fr) / np.exp(3 * bw * T) / g
        b[4] = T ** 4 * np.cos(4 * fr) / np.exp(4 * bw * T) / g

        # Calculate the denominator coefficients
        a[0] = 1
        a[1] = -8 * np.cos(fr) / np.exp(bw * T)
        a[2] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(2 * bw * T)
        a[3] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[3] /= np.exp(3 * bw * T)
        a[4] = 2 * (18 + 16 * np.cos(2 * fr) + np.cos(4 * fr))
        a[4] /= np.exp(4 * bw * T)
        a[5] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[5] /= np.exp(5 * bw * T)
        a[6] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(6 * bw * T)
        a[7] = -8 * np.cos(fr) / np.exp(7 * bw * T)
        a[8] = np.exp(-8 * bw * T)

    return b, a


filter_dict = {'butter': [buttap, buttord],
               'butterworth': [buttap, buttord],

               'cauer': [ellipap, ellipord],
               'elliptic': [ellipap, ellipord],
               'ellip': [ellipap, ellipord],

               'bessel': [besselap],
               'bessel_phase': [besselap],
               'bessel_delay': [besselap],
               'bessel_mag': [besselap],

               'cheby1': [cheb1ap, cheb1ord],
               'chebyshev1': [cheb1ap, cheb1ord],
               'chebyshevi': [cheb1ap, cheb1ord],

               'cheby2': [cheb2ap, cheb2ord],
               'chebyshev2': [cheb2ap, cheb2ord],
               'chebyshevii': [cheb2ap, cheb2ord],
               }

band_dict = {'band': 'bandpass',
             'bandpass': 'bandpass',
             'pass': 'bandpass',
             'bp': 'bandpass',

             'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',

             'l': 'lowpass',
             'low': 'lowpass',
             'lowpass': 'lowpass',
             'lp': 'lowpass',

             'high': 'highpass',
             'highpass': 'highpass',
             'h': 'highpass',
             'hp': 'highpass',
             }

bessel_norms = {'bessel': 'phase',
                'bessel_phase': 'phase',
                'bessel_delay': 'delay',
                'bessel_mag': 'mag'}
<|endoftext|># Author: Eric Larson
# 2014

"""Tools for MLS generation"""

import numpy as np

from ._max_len_seq_inner import _max_len_seq_inner

__all__ = ['max_len_seq']


# These are definitions of linear shift register taps for use in max_len_seq()
_mls_taps = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
             9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
             14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
             18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
             23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
             27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
             31: [28], 32: [31, 30, 10]}

def max_len_seq(nbits, state=None, length=None, taps=None):
    """
    Maximum length sequence (MLS) generator.

    Parameters
    ----------
    nbits : int
        Number of bits to use. Length of the resulting sequence will
        be ``(2**nbits) - 1``. Note that generating long sequences
        (e.g., greater than ``nbits == 16``) can take a long time.
    state : array_like, optional
        If array, must be of length ``nbits``, and will be cast to binary
        (bool) representation. If None, a seed of ones will be used,
        producing a repeatable representation. If ``state`` is all
        zeros, an error is raised as this is invalid. Default: None.
    length : int, optional
        Number of samples to compute. If None, the entire length
        ``(2**nbits) - 1`` is computed.
    taps : array_like, optional
        Polynomial taps to use (e.g., ``[7, 6, 1]`` for an 8-bit sequence).
        If None, taps will be automatically selected (for up to
        ``nbits == 32``).

    Returns
    -------
    seq : array
        Resulting MLS sequence of 0's and 1's.
    state : array
        The final state of the shift register.

    Notes
    -----
    The algorithm for MLS generation is generically described in:

        https://en.wikipedia.org/wiki/Maximum_length_sequence

    The default values for taps are specifically taken from the first
    option listed for each value of ``nbits`` in:

        https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm

    .. versionadded:: 0.15.0

    Examples
    --------
    MLS uses binary convention:

    >>> from scipy.signal import max_len_seq
    >>> max_len_seq(4)[0]
    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=int8)

    MLS has a white spectrum (except for DC):

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from numpy.fft import fft, ifft, fftshift, fftfreq
    >>> seq = max_len_seq(6)[0]*2-1  # +1 and -1
    >>> spec = fft(seq)
    >>> N = len(seq)
    >>> plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    Circular autocorrelation of MLS is an impulse:

    >>> acorrcirc = ifft(spec * np.conj(spec)).real
    >>> plt.figure()
    >>> plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    Linear autocorrelation of MLS is approximately an impulse:

    >>> acorr = np.correlate(seq, seq, 'full')
    >>> plt.figure()
    >>> plt.plot(np.arange(-N+1, N), acorr, '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    """
    taps_dtype = np.int32 if np.intp().itemsize == 4 else np.int64
    if taps is None:
        if nbits not in _mls_taps:
            known_taps = np.array(list(_mls_taps.keys()))
            raise ValueError(f'nbits must be between {known_taps.min()} and '
                             f'{known_taps.max()} if taps is None')
        taps = np.array(_mls_taps[nbits], taps_dtype)
    else:
        taps = np.unique(np.array(taps, taps_dtype))[::-1]
        if np.any(taps < 0) or np.any(taps > nbits) or taps.size < 1:
            raise ValueError('taps must be non-empty with values between '
                             'zero and nbits (inclusive)')
        taps = np.array(taps)  # needed for Cython and Pythran
    n_max = (2**nbits) - 1
    if length is None:
        length = n_max
    else:
        length = int(length)
        if length < 0:
            raise ValueError('length must be greater than or equal to 0')
    # We use int8 instead of bool here because NumPy arrays of bools
    # don't seem to work nicely with Cython
    if state is None:
        state = np.ones(nbits, dtype=np.int8, order='c')
    else:
        # makes a copy if need be, ensuring it's 0's and 1's
        state = np.array(state, dtype=bool, order='c').astype(np.int8)
    if state.ndim != 1 or state.size != nbits:
        raise ValueError('state must be a 1-D array of size nbits')
    if np.all(state == 0):
        raise ValueError('state must not be all zeros')

    seq = np.empty(length, dtype=np.int8, order='c')
    state = _max_len_seq_inner(taps, state, nbits, length, seq)
    return seq, state
<|endoftext|># Author: Travis Oliphant
# 1999 -- 2002

from __future__ import annotations  # Provides typing union operator `|` in Python 3.9
import operator
import math
from math import prod as _prod
import timeit
import warnings
from typing import Literal

from numpy._typing import ArrayLike

from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt


__all__ = ['correlate', 'correlation_lags', 'correlate2d',
           'convolve', 'convolve2d', 'fftconvolve', 'oaconvolve',
           'order_filter', 'medfilt', 'medfilt2d', 'wiener', 'lfilter',
           'lfiltic', 'sosfilt', 'deconvolve', 'hilbert', 'hilbert2', 'envelope',
           'unique_roots', 'invres', 'invresz', 'residue',
           'residuez', 'resample', 'resample_poly', 'detrend',
           'lfilter_zi', 'sosfilt_zi', 'sosfiltfilt', 'choose_conv_method',
           'filtfilt', 'decimate', 'vectorstrength']


_modedict = {'valid': 0, 'same': 1, 'full': 2}

_boundarydict = {'fill': 0, 'pad': 0, 'wrap': 2, 'circular': 2, 'symm': 1,
                 'symmetric': 1, 'reflect': 4}


def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.") from e


def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError as e:
        raise ValueError("Acceptable boundary flags are 'fill', 'circular' "
                         "(or 'wrap'), and 'symmetric' (or 'symm').") from e


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """Determine if inputs arrays need to be swapped in `"valid"` mode.

    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.

    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.

    Note that if the mode provided is not 'valid', False is immediately
    returned.

    """
    if mode != 'valid':
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError("For 'valid' mode, one must be at least "
                         "as large as the other in every dimension")

    return not ok1


def _reject_objects(arr, name):
    """Warn if arr.dtype is object or longdouble.
    """
    dt = np.asarray(arr).dtype
    if not (np.issubdtype(dt, np.integer)
            or dt in [np.bool_, np.float16, np.float32, np.float64,
                      np.complex64, np.complex128]
    ):
        msg = (
            f"dtype={dt} is not supported by {name} and will raise an error in "
            f"SciPy 1.17.0. Supported dtypes are: boolean, integer, `np.float16`,"
            f"`np.float32`, `np.float64`, `np.complex64`, `np.complex128`."
        )
        warnings.warn(msg, category=DeprecationWarning, stacklevel=3)


def correlate(in1, in2, mode='full', method='auto'):
    r"""
    Cross-correlate two N-dimensional arrays.

    Cross-correlate `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the correlation.

        ``direct``
           The correlation is determined directly from sums, the definition of
           correlation.
        ``fft``
           The Fast Fourier Transform is used to perform the correlation more
           quickly (only available for numerical arrays.)
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See `convolve` Notes for more detail.

           .. versionadded:: 0.19.0

    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : contains more documentation on `method`.
    correlation_lags : calculates the lag / displacement indices array for 1D
        cross-correlation.

    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as::

        z[...,k,...] = sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])

    This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')``
    then

    .. math::

          z[k] = (x * y)(k - N + 1)
               = \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}

    for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`

    where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
    and :math:`y_m` is 0 when m is outside the range of y.

    ``method='fft'`` only works for numerical arrays as it relies on
    `fftconvolve`. In certain cases (i.e., arrays of objects or when
    rounding integers can lose precision), ``method='direct'`` is always used.

    When using "same" mode with even-length inputs, the outputs of `correlate`
    and `correlate2d` differ: There is a 1-index offset between them.

    Examples
    --------
    Implement a matched filter using cross-correlation, to recover a signal
    that has passed through a noisy channel.

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    >>> sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    >>> sig_noise = sig + rng.standard_normal(len(sig))
    >>> corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128

    >>> clock = np.arange(64, len(sig), 128)
    >>> fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.plot(clock, sig[clock], 'ro')
    >>> ax_orig.set_title('Original signal')
    >>> ax_noise.plot(sig_noise)
    >>> ax_noise.set_title('Signal with noise')
    >>> ax_corr.plot(corr)
    >>> ax_corr.plot(clock, corr[clock], 'ro')
    >>> ax_corr.axhline(0.5, ls=':')
    >>> ax_corr.set_title('Cross-correlated with rectangular pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> plt.show()

    Compute the cross-correlation of a noisy signal with the original signal.

    >>> x = np.arange(128) / 128
    >>> sig = np.sin(2 * np.pi * x)
    >>> sig_noise = sig + rng.standard_normal(len(sig))
    >>> corr = signal.correlate(sig_noise, sig)
    >>> lags = signal.correlation_lags(len(sig), len(sig_noise))
    >>> corr /= np.max(corr)

    >>> fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('Original signal')
    >>> ax_orig.set_xlabel('Sample Number')
    >>> ax_noise.plot(sig_noise)
    >>> ax_noise.set_title('Signal with noise')
    >>> ax_noise.set_xlabel('Sample Number')
    >>> ax_corr.plot(lags, corr)
    >>> ax_corr.set_title('Cross-correlated signal')
    >>> ax_corr.set_xlabel('Lag')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_noise.margins(0, 0.1)
    >>> ax_corr.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> plt.show()

    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)
    _reject_objects(in1, 'correlate')
    _reject_objects(in2, 'correlate')

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        val = _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.") from e

    # this either calls fftconvolve or this function with method=='direct'
    if method in ('fft', 'auto'):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    elif method == 'direct':
        # fastpath to faster numpy.correlate for 1d inputs when possible
        if _np_conv_ok(in1, in2, mode):
            return np.correlate(in1, in2, mode)

        # _correlateND is far slower when in2.size > in1.size, so swap them
        # and then undo the effect afterward if mode == 'full'.  Also, it fails
        # with 'valid' mode if in2 is larger than in1, so swap those, too.
        # Don't swap inputs for 'same' mode, since shape of in1 matters.
        swapped_inputs = ((mode == 'full') and (in2.size > in1.size) or
                          _inputs_swap_needed(mode, in1.shape, in2.shape))

        if swapped_inputs:
            in1, in2 = in2, in1

        if mode == 'valid':
            ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
            out = np.empty(ps, in1.dtype)

            z = _sigtools._correlateND(in1, in2, out, val)

        else:
            ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]

            # zero pad input
            in1zpadded = np.zeros(ps, in1.dtype)
            sc = tuple(slice(0, i) for i in in1.shape)
            in1zpadded[sc] = in1.copy()

            if mode == 'full':
                out = np.empty(ps, in1.dtype)
            elif mode == 'same':
                out = np.empty(in1.shape, in1.dtype)

            z = _sigtools._correlateND(in1zpadded, in2, out, val)

        if swapped_inputs:
            # Reverse and conjugate to undo the effect of swapping inputs
            z = _reverse_and_conj(z)

        return z

    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")


def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.

    Parameters
    ----------
    in1_len : int
        First input size.
    in2_len : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.

    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.

    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.

    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:

    .. math::

        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt

    Where :math:`\tau` is defined as the displacement, also known as the lag.

    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:

    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]

    Where :math:`n` is the lag.

    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    else:
        raise ValueError(f"Mode {mode} is invalid")
    return lags


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    """Handle the axes argument for frequency-domain convolution.

    Returns the inputs and axes in a standard form, eliminating redundant axes,
    swapping the inputs if necessary, and checking for various potential
    errors.

    Parameters
    ----------
    in1 : array
        First input.
    in2 : array
        Second input.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the FFTs.
    sorted_axes : bool, optional
        If `True`, sort the axes.
        Default is `False`, do not sort.

    Returns
    -------
    in1 : array
        The first input, possible swapped with the second input.
    in2 : array
        The second input, possible swapped with the first input.
    axes : list of ints
        Axes over which to compute the FFTs.

    """
    s1 = in1.shape
    s2 = in2.shape
    noaxes = axes is None

    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)

    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")

    # Axes of length 1 can rely on broadcasting rules for multiply,
    # no fft needed.
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

    if sorted_axes:
        axes.sort()

    if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
               for a in range(in1.ndim) if a not in axes):
        raise ValueError("incompatible shapes for in1 and in2:"
                         f" {s1} and {s2}")

    # Check that input sizes are compatible with 'valid' mode.
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # Convolution is commutative; order doesn't have any effect on output.
        in1, in2 = in2, in1

    return in1, in2, axes


def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    if not len(axes):
        return in1 * in2

    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    ret = ifft(sp1 * sp2, fshape, axes=axes)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
    if mode == "full":
        return ret.copy()
    elif mode == "same":
        return _centered(ret, s1).copy()
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid).copy()
    else:
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.

    Examples
    --------
    Autocorrelation of white noise is an impulse.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> sig = rng.standard_normal(1000)
    >>> autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(np.arange(-len(sig)+1,len(sig)), autocorr)
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()

    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.

    >>> from scipy import datasets
    >>> face = datasets.face(gray=True)
    >>> kernel = np.outer(signal.windows.gaussian(70, 8),
    ...                   signal.windows.gaussian(70, 8))
    >>> blurred = signal.fftconvolve(face, kernel, mode='same')

    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(3, 1,
    ...                                                      figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(kernel, cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(blurred, cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()

    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _calc_oa_lens(s1, s2):
    """Calculate the optimal FFT lengths for overlap-add convolution.

    The calculation is done for a single dimension.

    Parameters
    ----------
    s1 : int
        Size of the dimension for the first array.
    s2 : int
        Size of the dimension for the second array.

    Returns
    -------
    block_size : int
        The size of the FFT blocks.
    overlap : int
        The amount of overlap between two blocks.
    in1_step : int
        The size of each step for the first array.
    in2_step : int
        The size of each step for the first array.

    """
    # Set up the arguments for the conventional FFT approach.
    fallback = (s1+s2-1, None, s1, s2)

    # Use conventional FFT convolve if sizes are same.
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback

    if s2 > s1:
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # There cannot be a useful block size if s2 is more than half of s1.
    if s2 >= s1/2:
        return fallback

    # Derivation of optimal block length
    # For original formula see:
    # https://en.wikipedia.org/wiki/Overlap-add_method
    #
    # Formula:
    # K = overlap = s2-1
    # N = block_size
    # C = complexity
    # e = exponential, exp(1)
    #
    # C = (N*(log2(N)+1))/(N-K)
    # C = (N*log2(2N))/(N-K)
    # C = N/(N-K) * log2(2N)
    # C1 = N/(N-K)
    # C2 = log2(2N) = ln(2N)/ln(2)
    #
    # dC1/dN = (1*(N-K)-N)/(N-K)^2 = -K/(N-K)^2
    # dC2/dN = 2/(2*N*ln(2)) = 1/(N*ln(2))
    #
    # dC/dN = dC1/dN*C2 + dC2/dN*C1
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + N/(N*ln(2)*(N-K))
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + 1/(ln(2)*(N-K))
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + (N-K)/(ln(2)*(N-K)^2)
    # dC/dN = (-K*ln(2N) + (N-K)/(ln(2)*(N-K)^2)
    # dC/dN = (N - K*ln(2N) - K)/(ln(2)*(N-K)^2)
    #
    # Solve for minimum, where dC/dN = 0
    # 0 = (N - K*ln(2N) - K)/(ln(2)*(N-K)^2)
    # 0 * ln(2)*(N-K)^2 = N - K*ln(2N) - K
    # 0 = N - K*ln(2N) - K
    # 0 = N - K*(ln(2N) + 1)
    # 0 = N - K*ln(2Ne)
    # N = K*ln(2Ne)
    # N/K = ln(2Ne)
    #
    # e^(N/K) = e^ln(2Ne)
    # e^(N/K) = 2Ne
    # 1/e^(N/K) = 1/(2*N*e)
    # e^(N/-K) = 1/(2*N*e)
    # e^(N/-K) = K/N*1/(2*K*e)
    # N/K*e^(N/-K) = 1/(2*e*K)
    # N/-K*e^(N/-K) = -1/(2*e*K)
    #
    # Using Lambert W function
    # https://en.wikipedia.org/wiki/Lambert_W_function
    # x = W(y) It is the solution to y = x*e^x
    # x = N/-K
    # y = -1/(2*e*K)
    #
    # N/-K = W(-1/(2*e*K))
    #
    # N = -K*W(-1/(2*e*K))
    overlap = s2-1
    opt_size = -overlap*lambertw(-1/(2*math.e*overlap), k=-1).real
    block_size = sp_fft.next_fast_len(math.ceil(opt_size))

    # Use conventional FFT convolve if there is only going to be one block.
    if block_size >= s1:
        return fallback

    if not swapped:
        in1_step = block_size-s2+1
        in2_step = s2
    else:
        in1_step = s2
        in2_step = block_size-s2+1

    return block_size, overlap, in1_step, in2_step


def oaconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using the overlap-add method.

    Convolve `in1` and `in2` using the overlap-add method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    and generally much faster than `fftconvolve` when one array is much
    larger than the other, but can be slower when only a few output values are
    needed or when the arrays are very similar in shape, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    fftconvolve : An implementation of convolution using FFT.

    Notes
    -----
    .. versionadded:: 1.4.0

    References
    ----------
    .. [1] Wikipedia, "Overlap-add_method".
           https://en.wikipedia.org/wiki/Overlap-add_method
    .. [2] Richard G. Lyons. Understanding Digital Signal Processing,
           Third Edition, 2011. Chapter 13.10.
           ISBN 13: 978-0137-02741-5

    Examples
    --------
    Convolve a 100,000 sample signal with a 512-sample filter.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> sig = rng.standard_normal(100000)
    >>> filt = signal.firwin(512, 0.01)
    >>> fsig = signal.oaconvolve(sig, filt)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(fsig)
    >>> ax_mag.set_title('Filtered noise')
    >>> fig.tight_layout()
    >>> fig.show()

    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])
    elif in1.shape == in2.shape:  # Equivalent to fftconvolve
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=True)

    s1 = in1.shape
    s2 = in2.shape

    if not axes:
        ret = in1 * in2
        return _apply_conv_mode(ret, s1, s2, mode, axes)

    # Calculate this now since in1 is changed later
    shape_final = [None if i not in axes else
                   s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    # Calculate the block sizes for the output, steps, first and second inputs.
    # It is simpler to calculate them all together than doing them in separate
    # loops due to all the special cases that need to be handled.
    optimal_sizes = ((-1, -1, s1[i], s2[i]) if i not in axes else
                     _calc_oa_lens(s1[i], s2[i]) for i in range(in1.ndim))
    block_size, overlaps, \
        in1_step, in2_step = zip(*optimal_sizes)

    # Fall back to fftconvolve if there is only one block in every dimension.
    if in1_step == s1 and in2_step == s2:
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    # Figure out the number of steps and padding.
    # This would get too complicated in a list comprehension.
    nsteps1 = []
    nsteps2 = []
    pad_size1 = []
    pad_size2 = []
    for i in range(in1.ndim):
        if i not in axes:
            pad_size1 += [(0, 0)]
            pad_size2 += [(0, 0)]
            continue

        if s1[i] > in1_step[i]:
            curnstep1 = math.ceil((s1[i]+1)/in1_step[i])
            if (block_size[i] - overlaps[i])*curnstep1 < shape_final[i]:
                curnstep1 += 1

            curpad1 = curnstep1*in1_step[i] - s1[i]
        else:
            curnstep1 = 1
            curpad1 = 0

        if s2[i] > in2_step[i]:
            curnstep2 = math.ceil((s2[i]+1)/in2_step[i])
            if (block_size[i] - overlaps[i])*curnstep2 < shape_final[i]:
                curnstep2 += 1

            curpad2 = curnstep2*in2_step[i] - s2[i]
        else:
            curnstep2 = 1
            curpad2 = 0

        nsteps1 += [curnstep1]
        nsteps2 += [curnstep2]
        pad_size1 += [(0, curpad1)]
        pad_size2 += [(0, curpad2)]

    # Pad the array to a size that can be reshaped to the desired shape
    # if necessary.
    if not all(curpad == (0, 0) for curpad in pad_size1):
        in1 = np.pad(in1, pad_size1, mode='constant', constant_values=0)

    if not all(curpad == (0, 0) for curpad in pad_size2):
        in2 = np.pad(in2, pad_size2, mode='constant', constant_values=0)

    # Reshape the overlap-add parts to input block sizes.
    split_axes = [iax+i for i, iax in enumerate(axes)]
    fft_axes = [iax+1 for iax in split_axes]

    # We need to put each new dimension before the corresponding dimension
    # being reshaped in order to get the data in the right layout at the end.
    reshape_size1 = list(in1_step)
    reshape_size2 = list(in2_step)
    for i, iax in enumerate(split_axes):
        reshape_size1.insert(iax, nsteps1[i])
        reshape_size2.insert(iax, nsteps2[i])

    in1 = in1.reshape(*reshape_size1)
    in2 = in2.reshape(*reshape_size2)

    # Do the convolution.
    fft_shape = [block_size[i] for i in axes]
    ret = _freq_domain_conv(in1, in2, fft_axes, fft_shape, calc_fast_len=False)

    # Do the overlap-add.
    for ax, ax_fft, ax_split in zip(axes, fft_axes, split_axes):
        overlap = overlaps[ax]
        if overlap is None:
            continue

        ret, overpart = np.split(ret, [-overlap], ax_fft)
        overpart = np.split(overpart, [-1], ax_split)[0]

        ret_overpart = np.split(ret, [overlap], ax_fft)[0]
        ret_overpart = np.split(ret_overpart, [1], ax_split)[1]
        ret_overpart += overpart

    # Reshape back to the correct dimensionality.
    shape_ret = [ret.shape[i] if i not in fft_axes else
                 ret.shape[i]*ret.shape[i-1]
                 for i in range(ret.ndim) if i not in split_axes]
    ret = ret.reshape(*shape_ret)

    # Slice to the correct size.
    slice_final = tuple([slice(islice) for islice in shape_final])
    ret = ret[slice_final]

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _numeric_arrays(arrays, kinds='buifc'):
    """
    See if a list of arrays are all numeric.

    Parameters
    ----------
    arrays : array or list of arrays
        arrays to check if numeric.
    kinds : string-like
        The dtypes of the arrays to be checked. If the dtype.kind of
        the ndarrays are not in this string the function returns False and
        otherwise returns True.
    """
    if isinstance(arrays, np.ndarray):
        return arrays.dtype.kind in kinds
    for array_ in arrays:
        if array_.dtype.kind not in kinds:
            return False
    return True


def _conv_ops(x_shape, h_shape, mode):
    """
    Find the number of operations required for direct/fft methods of
    convolution. The direct operations were recorded by making a dummy class to
    record the number of operations by overriding ``__mul__`` and ``__add__``.
    The FFT operations rely on the (well-known) computational complexity of the
    FFT (and the implementation of ``_freq_domain_conv``).

    """
    if mode == "full":
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "valid":
        out_shape = [abs(n - k) + 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "same":
        out_shape = x_shape
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         f" 'same', or 'full', not mode={mode}")

    s1, s2 = x_shape, h_shape
    if len(x_shape) == 1:
        s1, s2 = s1[0], s2[0]
        if mode == "full":
            direct_ops = s1 * s2
        elif mode == "valid":
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        elif mode == "same":
            direct_ops = (s1 * s2 if s1 < s2 else
                          s1 * s2 - (s2 // 2) * ((s2 + 1) // 2))
    else:
        if mode == "full":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "valid":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "same":
            direct_ops = _prod(s1) * _prod(s2)

    full_out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    N = _prod(full_out_shape)
    fft_ops = 3 * N * np.log(N)  # 3 separate FFTs of size full_out_shape
    return fft_ops, direct_ops


def _fftconv_faster(x, h, mode):
    """
    See if using fftconvolve or convolve is faster.

    Parameters
    ----------
    x : np.ndarray
        Signal
    h : np.ndarray
        Kernel
    mode : str
        Mode passed to convolve

    Returns
    -------
    fft_faster : bool

    Notes
    -----
    See docstring of `choose_conv_method` for details on tuning hardware.

    See pull request 11031 for more detail:
    https://github.com/scipy/scipy/pull/11031.

    """
    fft_ops, direct_ops = _conv_ops(x.shape, h.shape, mode)
    offset = -1e-3 if x.ndim == 1 else -1e-4
    constants = {
            "valid": (1.89095737e-9, 2.1364985e-10, offset),
            "full": (1.7649070e-9, 2.1414831e-10, offset),
            "same": (3.2646654e-9, 2.8478277e-10, offset)
            if h.size <= x.size
            else (3.21635404e-9, 1.1773253e-8, -1e-5),
    } if x.ndim == 1 else {
            "valid": (1.85927e-9, 2.11242e-8, offset),
            "full": (1.99817e-9, 1.66174e-8, offset),
            "same": (2.04735e-9, 1.55367e-8, offset),
    }
    O_fft, O_direct, O_offset = constants[mode]
    return O_fft * fft_ops < O_direct * direct_ops + O_offset


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()


def _np_conv_ok(volume, kernel, mode):
    """
    See if numpy supports convolution of `volume` and `kernel` (i.e. both are
    1D ndarrays and of the appropriate shape).  NumPy's 'same' mode uses the
    size of the larger input, while SciPy's uses the size of the first input.

    Invalid mode strings will return False and be caught by the calling func.
    """
    if volume.ndim == kernel.ndim == 1:
        if mode in ('full', 'valid'):
            return True
        elif mode == 'same':
            return volume.size >= kernel.size
    else:
        return False


def _timeit_fast(stmt="pass", setup="pass", repeat=3):
    """
    Returns the time the statement/function took, in seconds.

    Faster, less precise version of IPython's timeit. `stmt` can be a statement
    written as a string or a callable.

    Will do only 1 loop (like IPython's timeit) with no repetitions
    (unlike IPython) for very slow functions.  For fast functions, only does
    enough loops to take 5 ms, which seems to produce similar results (on
    Windows at least), and avoids doing an extraneous cycle that isn't
    measured.

    """
    timer = timeit.Timer(stmt, setup)

    # determine number of calls per rep so total time for 1 rep >= 5 ms
    x = 0
    for p in range(0, 10):
        number = 10**p
        x = timer.timeit(number)  # seconds
        if x >= 5e-3 / 10:  # 5 ms for final test, 1/10th that for this one
            break
    if x > 1:  # second
        # If it's macroscopic, don't bother with repetitions
        best = x
    else:
        number *= 10
        r = timer.repeat(repeat, number)
        best = min(r)

    sec = best / number
    return sec


def choose_conv_method(in1, in2, mode='full', measure=False):
    """
    Find the fastest convolution/correlation method.

    This primarily exists to be called during the ``method='auto'`` option in
    `convolve` and `correlate`. It can also be used to determine the value of
    ``method`` for many different convolutions of the same dtype/shape.
    In addition, it supports timing the convolution to adapt the value of
    ``method`` to a particular set of inputs and/or hardware.

    Parameters
    ----------
    in1 : array_like
        The first argument passed into the convolution function.
    in2 : array_like
        The second argument passed into the convolution function.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    measure : bool, optional
        If True, run and time the convolution of `in1` and `in2` with both
        methods and return the fastest. If False (default), predict the fastest
        method using precomputed values.

    Returns
    -------
    method : str
        A string indicating which convolution method is fastest, either
        'direct' or 'fft'
    times : dict, optional
        A dictionary containing the times (in seconds) needed for each method.
        This value is only returned if ``measure=True``.

    See Also
    --------
    convolve
    correlate

    Notes
    -----
    Generally, this method is 99% accurate for 2D signals and 85% accurate
    for 1D signals for randomly chosen input sizes. For precision, use
    ``measure=True`` to find the fastest method by timing the convolution.
    This can be used to avoid the minimal overhead of finding the fastest
    ``method`` later, or to adapt the value of ``method`` to a particular set
    of inputs.

    Experiments were run on an Amazon EC2 r5a.2xlarge machine to test this
    function. These experiments measured the ratio between the time required
    when using ``method='auto'`` and the time required for the fastest method
    (i.e., ``ratio = time_auto / min(time_fft, time_direct)``). In these
    experiments, we found:

    * There is a 95% chance of this ratio being less than 1.5 for 1D signals
      and a 99% chance of being less than 2.5 for 2D signals.
    * The ratio was always less than 2.5/5 for 1D/2D signals respectively.
    * This function is most inaccurate for 1D convolutions that take between 1
      and 10 milliseconds with ``method='direct'``. A good proxy for this
      (at least in our experiments) is ``1e6 <= in1.size * in2.size <= 1e7``.

    The 2D results almost certainly generalize to 3D/4D/etc because the
    implementation is the same (the 1D implementation is different).

    All the numbers above are specific to the EC2 machine. However, we did find
    that this function generalizes fairly decently across hardware. The speed
    tests were of similar quality (and even slightly better) than the same
    tests performed on the machine to tune this function's numbers (a mid-2014
    15-inch MacBook Pro with 16GB RAM and a 2.5GHz Intel i7 processor).

    There are cases when `fftconvolve` supports the inputs but this function
    returns `direct` (e.g., to protect against floating point integer
    precision).

    .. versionadded:: 0.19

    Examples
    --------
    Estimate the fastest method for a given input:

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> img = rng.random((32, 32))
    >>> filter = rng.random((8, 8))
    >>> method = signal.choose_conv_method(img, filter, mode='same')
    >>> method
    'fft'

    This can then be applied to other arrays of the same dtype and shape:

    >>> img2 = rng.random((32, 32))
    >>> filter2 = rng.random((8, 8))
    >>> corr2 = signal.correlate(img2, filter2, mode='same', method=method)
    >>> conv2 = signal.convolve(img2, filter2, mode='same', method=method)

    The output of this function (``method``) works with `correlate` and
    `convolve`.

    """
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    _reject_objects(volume, 'choose_conv_method')
    _reject_objects(kernel, 'choose_conv_method')

    if measure:
        times = {}
        for method in ['fft', 'direct']:
            times[method] = _timeit_fast(lambda: convolve(volume, kernel,
                                         mode=mode, method=method))

        chosen_method = 'fft' if times['fft'] < times['direct'] else 'direct'
        return chosen_method, times

    # for integer input,
    # catch when more precision required than float provides (representing an
    # integer as float can lose precision in fftconvolve if larger than 2**52)
    if any([_numeric_arrays([x], kinds='ui') for x in [volume, kernel]]):
        max_value = int(np.abs(volume).max()) * int(np.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        if max_value > 2**np.finfo('float').nmant - 1:
            return 'direct'

    if _numeric_arrays([volume, kernel], kinds='b'):
        return 'direct'

    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return 'fft'

    return 'direct'


def convolve(in1, in2, mode='full', method='auto'):
    """
    Convolve two N-dimensional arrays.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See Notes for more detail.

           .. versionadded:: 0.19.0

    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Warns
    -----
    RuntimeWarning
        Use of the FFT convolution on input containing NAN or INF will lead
        to the entire output being NAN or INF. Use method='direct' when your
        input contains NAN or INF values.

    See Also
    --------
    numpy.polymul : performs polynomial multiplication (same operation, but
                    also accepts poly1d objects)
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve : Always uses the FFT method.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.

    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force ``method='direct'`` (more detail
    in `choose_conv_method` docstring).

    Examples
    --------
    Smooth a square pulse using a Hann window:

    >>> import numpy as np
    >>> from scipy import signal
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> win = signal.windows.hann(50)
    >>> filtered = signal.convolve(sig, win, mode='same') / sum(win)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(win)
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(filtered)
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    _reject_objects(volume, 'correlate')
    _reject_objects(kernel, 'correlate')

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("volume and kernel should have the same "
                         "dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == 'auto':
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == 'fft':
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = np.result_type(volume, kernel)
        if result_type.kind in {'u', 'i'}:
            out = np.around(out)

        if np.isnan(out.flat[0]) or np.isinf(out.flat[0]):
            warnings.warn("Use of fft convolution on input with NAN or inf"
                          " results in NAN or inf output. Consider using"
                          " method='direct' instead.",
                          category=RuntimeWarning, stacklevel=2)

        return out.astype(result_type)
    elif method == 'direct':
        # fastpath to faster numpy.convolve for 1d inputs when possible
        if _np_conv_ok(volume, kernel, mode):
            return np.convolve(volume, kernel, mode)

        return correlate(volume, _reverse_and_conj(kernel), mode, 'direct')
    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")


def order_filter(a, domain, rank):
    """
    Perform an order filter on an N-D array.

    Perform an order filter on the array in. The domain argument acts as a
    mask centered over each pixel. The non-zero elements of domain are
    used to select elements surrounding each input pixel which are placed
    in a list. The list is sorted, and the output for that pixel is the
    element corresponding to rank in the sorted list.

    Parameters
    ----------
    a : ndarray
        The N-dimensional input array.
    domain : array_like
        A mask array with the same number of dimensions as `a`.
        Each dimension should have an odd number of elements.
    rank : int
        A non-negative integer which selects the element from the
        sorted list (0 corresponds to the smallest element, 1 is the
        next smallest element, etc.).

    Returns
    -------
    out : ndarray
        The results of the order filter in an array with the same
        shape as `a`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> x = np.arange(25).reshape(5, 5)
    >>> domain = np.identity(3)
    >>> x
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])
    >>> signal.order_filter(x, domain, 0)
    array([[  0,   0,   0,   0,   0],
           [  0,   0,   1,   2,   0],
           [  0,   5,   6,   7,   0],
           [  0,  10,  11,  12,   0],
           [  0,   0,   0,   0,   0]])
    >>> signal.order_filter(x, domain, 2)
    array([[  6,   7,   8,   9,   4],
           [ 11,  12,  13,  14,   9],
           [ 16,  17,  18,  19,  14],
           [ 21,  22,  23,  24,  19],
           [ 20,  21,  22,  23,  24]])

    """
    domain = np.asarray(domain)
    for dimsize in domain.shape:
        if (dimsize % 2) != 1:
            raise ValueError("Each dimension of domain argument "
                             "should have an odd number of elements.")

    a = np.asarray(a)
    if not (np.issubdtype(a.dtype, np.integer)
            or a.dtype in [np.float32, np.float64]):
        raise ValueError(f"dtype={a.dtype} is not supported by order_filter")

    result = ndimage.rank_filter(a, rank, footprint=domain, mode='constant')
    return result


def medfilt(volume, kernel_size=None):
    """
    Perform a median filter on an N-dimensional array.

    Apply a median filter to the input array using a local window-size
    given by `kernel_size`. The array will automatically be zero-padded.

    Parameters
    ----------
    volume : array_like
        An N-dimensional input array.
    kernel_size : array_like, optional
        A scalar or an N-length list giving the size of the median filter
        window in each dimension.  Elements of `kernel_size` should be odd.
        If `kernel_size` is a scalar, then this scalar is used as the size in
        each dimension. Default size is 3 for each dimension.

    Returns
    -------
    out : ndarray
        An array the same size as input containing the median filtered
        result.

    Warns
    -----
    UserWarning
        If array size is smaller than kernel size along any dimension

    See Also
    --------
    scipy.ndimage.median_filter
    scipy.signal.medfilt2d

    Notes
    -----
    The more general function `scipy.ndimage.median_filter` has a more
    efficient implementation of a median filter and therefore runs much faster.

    For 2-dimensional images with ``uint8``, ``float32`` or ``float64`` dtypes,
    the specialised function `scipy.signal.medfilt2d` may be faster.

    """
    volume = np.atleast_1d(volume)
    if not (np.issubdtype(volume.dtype, np.integer)
            or volume.dtype in [np.float32, np.float64]):
        raise ValueError(f"dtype={volume.dtype} is not supported by medfilt")

    if kernel_size is None:
        kernel_size = [3] * volume.ndim
    kernel_size = np.asarray(kernel_size)
    if kernel_size.shape == ():
        kernel_size = np.repeat(kernel_size.item(), volume.ndim)

    for k in range(volume.ndim):
        if (kernel_size[k] % 2) != 1:
            raise ValueError("Each element of kernel_size should be odd.")
    if any(k > s for k, s in zip(kernel_size, volume.shape)):
        warnings.warn('kernel_size exceeds volume extent: the volume will be '
                      'zero-padded.',
                      stacklevel=2)

    size = math.prod(kernel_size)
    result = ndimage.rank_filter(volume, size // 2, size=kernel_size,
                                 mode='constant')

    return result


def wiener(im, mysize=None, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or array_like, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.

    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.

    Notes
    -----
    This implementation is similar to wiener2 in Matlab/Octave.
    For more details see [1]_

    References
    ----------
    .. [1] Lim, Jae S., Two-Dimensional Signal and Image Processing,
           Englewood Cliffs, NJ, Prentice Hall, 1990, p. 548.

    Examples
    --------
    >>> from scipy.datasets import face
    >>> from scipy.signal import wiener
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> img = rng.random((40, 40))    #Create a random image
    >>> filtered_img = wiener(img, (5, 5))  #Filter the image
    >>> f, (plot1, plot2) = plt.subplots(1, 2)
    >>> plot1.imshow(img)
    >>> plot2.imshow(filtered_img)
    >>> plt.show()

    """
    im = np.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), im.ndim)

    # Estimate the local mean
    size = math.prod(mysize)
    lMean = correlate(im, np.ones(mysize), 'same') / size

    # Estimate the local variance
    lVar = (correlate(im ** 2, np.ones(mysize), 'same') / size - lMean ** 2)

    # Estimate the noise power if needed.
    if noise is None:
        noise = np.mean(np.ravel(lVar), axis=0)

    res = (im - lMean)
    res *= (1 - noise / lVar)
    res += lMean
    out = np.where(lVar < noise, lMean, res)

    return out


def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """
    Convolve two 2-dimensional arrays.

    Convolve `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    out : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Examples
    --------
    Compute the gradient of an image by 2D convolution with a complex Scharr
    operator.  (Horizontal operator is real, vertical is imaginary.)  Use
    symmetric boundary condition to avoid creating edges at the image
    boundaries.

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy import datasets
    >>> ascent = datasets.ascent()
    >>> scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
    ...                    [-10+0j, 0+ 0j, +10 +0j],
    ...                    [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    >>> grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(ascent, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_mag.imshow(np.absolute(grad), cmap='gray')
    >>> ax_mag.set_title('Gradient magnitude')
    >>> ax_mag.set_axis_off()
    >>> ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
    >>> ax_ang.set_title('Gradient orientation')
    >>> ax_ang.set_axis_off()
    >>> fig.show()

    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError('convolve2d inputs must both be 2-D arrays')

    if _inputs_swap_needed(mode, in1.shape, in2.shape):
        in1, in2 = in2, in1

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)
    out = _sigtools._convolve2d(in1, in2, 1, val, bval, fillvalue)
    return out


def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """
    Cross-correlate two 2-dimensional arrays.

    Cross correlate `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    Notes
    -----
    When using "same" mode with even-length inputs, the outputs of `correlate`
    and `correlate2d` differ: There is a 1-index offset between them.

    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:

    >>> import numpy as np
    >>> from scipy import signal, datasets, ndimage
    >>> rng = np.random.default_rng()
    >>> face = datasets.face(gray=True) - datasets.face(gray=True).mean()
    >>> face = ndimage.zoom(face[30:500, 400:950], 0.5)  # extract the face
    >>> template = np.copy(face[135:165, 140:175])  # right eye
    >>> template -= template.mean()
    >>> face = face + rng.standard_normal(face.shape) * 50  # add noise
    >>> corr = signal.correlate2d(face, template, boundary='symm', mode='same')
    >>> y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
    ...                                                     figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(template, cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(corr, cmap='gray')
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    >>> ax_orig.plot(x, y, 'ro')
    >>> fig.show()

    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError('correlate2d inputs must both be 2-D arrays')

    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
    if swapped_inputs:
        in1, in2 = in2, in1

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)
    out = _sigtools._convolve2d(in1, in2.conj(), 0, val, bval, fillvalue)

    if swapped_inputs:
        out = out[::-1, ::-1]

    return out


def medfilt2d(input, kernel_size=3):
    """
    Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size
    given by `kernel_size` (must be odd). The array is zero-padded
    automatically.

    Parameters
    ----------
    input : array_like
        A 2-dimensional input array.
    kernel_size : array_like, optional
        A scalar or a list of length 2, giving the size of the
        median filter window in each dimension.  Elements of
        `kernel_size` should be odd.  If `kernel_size` is a scalar,
        then this scalar is used as the size in each dimension.
        Default is a kernel of size (3, 3).

    Returns
    -------
    out : ndarray
        An array the same size as input containing the median filtered
        result.

    See Also
    --------
    scipy.ndimage.median_filter

    Notes
    -----
    This is faster than `medfilt` when the input dtype is ``uint8``,
    ``float32``, or ``float64``; for other types, this falls back to
    `medfilt`. In some situations, `scipy.ndimage.median_filter` may be
    faster than this function.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> x = np.arange(25).reshape(5, 5)
    >>> x
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    # Replaces i,j with the median out of 5*5 window

    >>> signal.medfilt2d(x, kernel_size=5)
    array([[ 0,  0,  2,  0,  0],
           [ 0,  3,  7,  4,  0],
           [ 2,  8, 12,  9,  4],
           [ 0,  8, 12,  9,  0],
           [ 0,  0, 12,  0,  0]])

    # Replaces i,j with the median out of default 3*3 window

    >>> signal.medfilt2d(x)
    array([[ 0,  1,  2,  3,  0],
           [ 1,  6,  7,  8,  4],
           [ 6, 11, 12, 13,  9],
           [11, 16, 17, 18, 14],
           [ 0, 16, 17, 18,  0]])

    # Replaces i,j with the median out of default 5*3 window

    >>> signal.medfilt2d(x, kernel_size=[5,3])
    array([[ 0,  1,  2,  3,  0],
           [ 0,  6,  7,  8,  3],
           [ 5, 11, 12, 13,  8],
           [ 5, 11, 12, 13,  8],
           [ 0, 11, 12, 13,  0]])

    # Replaces i,j with the median out of default 3*5 window

    >>> signal.medfilt2d(x, kernel_size=[3,5])
    array([[ 0,  0,  2,  1,  0],
           [ 1,  5,  7,  6,  3],
           [ 6, 10, 12, 11,  8],
           [11, 15, 17, 16, 13],
           [ 0, 15, 17, 16,  0]])

    # As seen in the examples,
    # kernel numbers must be odd and not exceed original array dim

    """
    image = np.asarray(input)

    # checking dtype.type, rather than just dtype, is necessary for
    # excluding np.longdouble with MS Visual C.
    if image.dtype.type not in (np.ubyte, np.float32, np.float64):
        return medfilt(image, kernel_size)

    if kernel_size is None:
        kernel_size = [3] * 2
    kernel_size = np.asarray(kernel_size)
    if kernel_size.shape == ():
        kernel_size = np.repeat(kernel_size.item(), 2)

    for size in kernel_size:
        if (size % 2) != 1:
            raise ValueError("Each element of kernel_size should be odd.")

    return _sigtools._medfilt2d(image, kernel_size)


def lfilter(b, a, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence, `x`, using a digital filter.  This works for many
    fundamental data types (including Object type).  The filter is a direct
    form II transposed implementation of the standard difference equation
    (see Notes).

    The function `sosfilt` (and filter design using ``output='sos'``) should be
    preferred over `lfilter` for most filtering tasks, as second-order sections
    have fewer numerical problems.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    See Also
    --------
    lfiltic : Construct initial conditions for `lfilter`.
    lfilter_zi : Compute initial state (steady state of step response) for
                 `lfilter`.
    filtfilt : A forward-backward filter, to obtain a filter with zero phase.
    savgol_filter : A Savitzky-Golay filter.
    sosfilt: Filter data using cascaded second-order sections.
    sosfiltfilt: A forward-backward filter using second-order sections.

    Notes
    -----
    The filter function is implemented as a direct II transposed structure.
    This means that the filter implements::

       a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                             - a[1]*y[n-1] - ... - a[N]*y[n-N]

    where `M` is the degree of the numerator, `N` is the degree of the
    denominator, and `n` is the sample number.  It is implemented using
    the following difference equations (assuming M = N)::

         a[0]*y[n] = b[0] * x[n]               + d[0][n-1]
           d[0][n] = b[1] * x[n] - a[1] * y[n] + d[1][n-1]
           d[1][n] = b[2] * x[n] - a[2] * y[n] + d[2][n-1]
         ...
         d[N-2][n] = b[N-1]*x[n] - a[N-1]*y[n] + d[N-1][n-1]
         d[N-1][n] = b[N] * x[n] - a[N] * y[n]

    where `d` are the state variables.

    The rational transfer function describing this filter in the
    z-transform domain is::

                             -1              -M
                 b[0] + b[1]z  + ... + b[M] z
         Y(z) = -------------------------------- X(z)
                             -1              -N
                 a[0] + a[1]z  + ... + a[N] z

    Examples
    --------
    Generate a noisy signal to be filtered:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> t = np.linspace(-1, 1, 201)
    >>> x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
    ...      0.1*np.sin(2*np.pi*1.25*t + 1) +
    ...      0.18*np.cos(2*np.pi*3.85*t))
    >>> xn = x + rng.standard_normal(len(t)) * 0.08

    Create an order 3 lowpass butterworth filter:

    >>> b, a = signal.butter(3, 0.05)

    Apply the filter to xn.  Use lfilter_zi to choose the initial condition of
    the filter:

    >>> zi = signal.lfilter_zi(b, a)
    >>> z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

    Apply the filter again, to have a result filtered at an order the same as
    filtfilt:

    >>> z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

    Use filtfilt to apply the filter:

    >>> y = signal.filtfilt(b, a, xn)

    Plot the original signal and the various filtered versions:

    >>> plt.figure
    >>> plt.plot(t, xn, 'b', alpha=0.75)
    >>> plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
    >>> plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
    ...             'filtfilt'), loc='best')
    >>> plt.grid(True)
    >>> plt.show()

    """
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    _reject_objects(x, 'lfilter')
    _reject_objects(a, 'lfilter')
    _reject_objects(b, 'lfilter')

    if len(a) == 1:
        # This path only supports types fdgFDGO to mirror _linear_filter below.
        # Any of b, a, x, or zi can set the dtype, but there is no default
        # casting of other types; instead a NotImplementedError is raised.
        b = np.asarray(b)
        a = np.asarray(a)
        if b.ndim != 1 and a.ndim != 1:
            raise ValueError('object of too small depth for desired array')
        x = _validate_x(x)
        inputs = [b, a, x]
        if zi is not None:
            # _linear_filter does not broadcast zi, but does do expansion of
            # singleton dims.
            zi = np.asarray(zi)
            if zi.ndim != x.ndim:
                raise ValueError('object of too small depth for desired array')
            expected_shape = list(x.shape)
            expected_shape[axis] = b.shape[0] - 1
            expected_shape = tuple(expected_shape)
            # check the trivial case where zi is the right shape first
            if zi.shape != expected_shape:
                strides = zi.ndim * [None]
                if axis < 0:
                    axis += zi.ndim
                for k in range(zi.ndim):
                    if k == axis and zi.shape[k] == expected_shape[k]:
                        strides[k] = zi.strides[k]
                    elif k != axis and zi.shape[k] == expected_shape[k]:
                        strides[k] = zi.strides[k]
                    elif k != axis and zi.shape[k] == 1:
                        strides[k] = 0
                    else:
                        raise ValueError('Unexpected shape for zi: expected '
                                         f'{expected_shape}, found {zi.shape}.')
                zi = np.lib.stride_tricks.as_strided(zi, expected_shape,
                                                     strides)
            inputs.append(zi)
        dtype = np.result_type(*inputs)

        if dtype.char not in 'fdgFDGO':
            raise NotImplementedError(f"input type '{dtype}' not supported")

        b = np.array(b, dtype=dtype)
        a = np.asarray(a, dtype=dtype)
        b /= a[0]
        x = np.asarray(x, dtype=dtype)

        out_full = np.apply_along_axis(lambda y: np.convolve(b, y), axis, x)
        ind = out_full.ndim * [slice(None)]
        if zi is not None:
            ind[axis] = slice(zi.shape[axis])
            out_full[tuple(ind)] += zi

        ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
        out = out_full[tuple(ind)]

        if zi is None:
            return out
        else:
            ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
            zf = out_full[tuple(ind)]
            return out, zf
    else:
        if zi is None:
            return _sigtools._linear_filter(b, a, x, axis)
        else:
            return _sigtools._linear_filter(b, a, x, axis, zi)


def lfiltic(b, a, y, x=None):
    """
    Construct initial conditions for lfilter given input and output vectors.

    Given a linear filter (b, a) and initial conditions on the output `y`
    and the input `x`, return the initial conditions on the state vector zi
    which is used by `lfilter` to generate the output given the input.

    Parameters
    ----------
    b : array_like
        Linear filter term.
    a : array_like
        Linear filter term.
    y : array_like
        Initial conditions.

        If ``N = len(a) - 1``, then ``y = {y[-1], y[-2], ..., y[-N]}``.

        If `y` is too short, it is padded with zeros.
    x : array_like, optional
        Initial conditions.

        If ``M = len(b) - 1``, then ``x = {x[-1], x[-2], ..., x[-M]}``.

        If `x` is not given, its initial conditions are assumed zero.

        If `x` is too short, it is padded with zeros.

    Returns
    -------
    zi : ndarray
        The state vector ``zi = {z_0[-1], z_1[-1], ..., z_K-1[-1]}``,
        where ``K = max(M, N)``.

    See Also
    --------
    lfilter, lfilter_zi

    """
    N = np.size(a) - 1
    M = np.size(b) - 1
    K = max(M, N)
    y = np.asarray(y)

    if x is None:
        result_type = np.result_type(np.asarray(b), np.asarray(a), y)
        if result_type.kind in 'bui':
            result_type = np.float64
        x = np.zeros(M, dtype=result_type)
    else:
        x = np.asarray(x)

        result_type = np.result_type(np.asarray(b), np.asarray(a), y, x)
        if result_type.kind in 'bui':
            result_type = np.float64
        x = x.astype(result_type)

        L = np.size(x)
        if L < M:
            x = np.r_[x, np.zeros(M - L)]

    y = y.astype(result_type)
    zi = np.zeros(K, result_type)

    L = np.size(y)
    if L < N:
        y = np.r_[y, np.zeros(N - L)]

    for m in range(M):
        zi[m] = np.sum(b[m + 1:] * x[:M - m], axis=0)

    for m in range(N):
        zi[m] -= np.sum(a[m + 1:] * y[:N - m], axis=0)

    return zi


def deconvolve(signal, divisor):
    """Deconvolves ``divisor`` out of ``signal`` using inverse filtering.

    Returns the quotient and remainder such that
    ``signal = convolve(divisor, quotient) + remainder``

    Parameters
    ----------
    signal : (N,) array_like
        Signal data, typically a recorded signal
    divisor : (N,) array_like
        Divisor data, typically an impulse response or filter that was
        applied to the original signal

    Returns
    -------
    quotient : ndarray
        Quotient, typically the recovered original signal
    remainder : ndarray
        Remainder

    See Also
    --------
    numpy.polydiv : performs polynomial division (same operation, but
                    also accepts poly1d objects)

    Examples
    --------
    Deconvolve a signal that's been filtered:

    >>> from scipy import signal
    >>> original = [0, 1, 0, 0, 1, 1, 0, 0]
    >>> impulse_response = [2, 1]
    >>> recorded = signal.convolve(impulse_response, original)
    >>> recorded
    array([0, 2, 1, 0, 2, 3, 1, 0, 0])
    >>> recovered, remainder = signal.deconvolve(recorded, impulse_response)
    >>> recovered
    array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.])

    """
    num = np.atleast_1d(signal)
    den = np.atleast_1d(divisor)
    if num.ndim > 1:
        raise ValueError("signal must be 1-D.")
    if den.ndim > 1:
        raise ValueError("divisor must be 1-D.")
    N = len(num)
    D = len(den)
    if D > N:
        quot = []
        rem = num
    else:
        input = np.zeros(N - D + 1, float)
        input[0] = 1
        quot = lfilter(num, den, input)
        rem = num - convolve(den, quot, mode='full')
    return quot, rem


def hilbert(x, N=None, axis=-1):
    r"""FFT-based computation of the analytic signal.

    The analytic signal is calculated by filtering out the negative frequencies and
    doubling the amplitudes of the positive frequencies in the FFT domain.
    The imaginary part of the result is the hilbert transform of the real-valued input
    signal.
    
    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of a real-valued signal ``x(t)``
    can be expressed as [1]_

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y\ ,

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [2]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex-valued signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Wikipedia, "Hilbert Transform".
           https://en.wikipedia.org/wiki/Hilbert_transform
    .. [3] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [4] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    See Also
    --------
    envelope: Compute envelope of a real- or complex-valued signal.

    Examples
    --------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.

    Let's create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import hilbert, chirp
    ...
    >>> duration, fs = 1, 400  # 1 s signal with sampling frequency of 400 Hz
    >>> t = np.arange(int(fs*duration)) / fs  # timestamps of samples
    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

    The amplitude envelope is given by the magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.

    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = np.abs(analytic_signal)
    >>> instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    >>> instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
    ...
    >>> fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all', tight_layout=True)
    >>> ax0.set_title("Amplitude-modulated Chirp Signal")
    >>> ax0.set_ylabel("Amplitude")
    >>> ax0.plot(t, signal, label='Signal')
    >>> ax0.plot(t, amplitude_envelope, label='Envelope')
    >>> ax0.legend()
    >>> ax1.set(xlabel="Time in seconds", ylabel="Phase in rad", ylim=(0, 120))
    >>> ax1.plot(t[1:], instantaneous_frequency, 'C2-', label='Instantaneous Phase')
    >>> ax1.legend()
    >>> plt.show()

    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = sp_fft.fft(x, N, axis=axis)
    h = np.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = sp_fft.ifft(Xf * h, axis=axis)
    return x


def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or np.any(np.asarray(N) <= 0):
        raise ValueError("When given as a tuple, N must hold exactly "
                         "two positive integers")

    Xf = sp_fft.fft2(x, N, axes=(0, 1))
    h1 = np.zeros(N[0], dtype=Xf.dtype)
    h2 = np.zeros(N[1], dtype=Xf.dtype)
    for h in (h1, h2):
        N1 = h.shape[0]
        if N1 % 2 == 0:
            h[0] = h[N1 // 2] = 1
            h[1:N1 // 2] = 2
        else:
            h[0] = 1
            h[1:(N1 + 1) // 2] = 2

    h = h1[:, np.newaxis] * h2[np.newaxis, :]
    k = x.ndim
    while k > 2:
        h = h[:, np.newaxis]
        k -= 1
    x = sp_fft.ifft2(Xf * h, axes=(0, 1))
    return x


def envelope(z: np.ndarray, bp_in: tuple[int | None, int | None] = (1, None), *,
             n_out: int | None = None, squared: bool = False,
             residual: Literal['lowpass', 'all', None] = 'lowpass',
             axis: int = -1) -> np.ndarray:
    r"""Compute the envelope of a real- or complex-valued signal.

    Parameters
    ----------
    z : ndarray
        Real- or complex-valued input signal, which is assumed to be made up of ``n``
        samples and having sampling interval ``T``. `z` may also be a multidimensional
        array with the time axis being defined by `axis`.
    bp_in : tuple[int | None, int | None], optional
        2-tuple defining the frequency band ``bp_in[0]:bp_in[1]`` of the input filter.
        The corner frequencies are specified as integer multiples of ``1/(n*T)`` with
        ``-n//2 <= bp_in[0] < bp_in[1] <= (n+1)//2`` being the allowed frequency range.
        ``None`` entries are replaced with ``-n//2`` or ``(n+1)//2`` respectively. The
        default of ``(1, None)`` removes the mean value as well as the negative
        frequency components.
    n_out : int | None, optional
        If not ``None`` the output will be resampled to `n_out` samples. The default
        of ``None`` sets the output to the same length as the input `z`.
    squared : bool, optional
        If set, the square of the envelope is returned. The bandwidth of the squared
        envelope is often smaller than the non-squared envelope bandwidth due to the
        nonlinear nature of the utilized absolute value function. I.e., the embedded
        square root function typically produces addiational harmonics.
        The default is ``False``.
    residual : Literal['lowpass', 'all', None], optional
        This option determines what kind of residual, i.e., the signal part which the
        input bandpass filter removes, is returned. ``'all'`` returns everything except
        the contents of the frequency band ``bp_in[0]:bp_in[1]``, ``'lowpass'``
        returns the contents of the frequency band ``< bp_in[0]``. If ``None`` then
        only the envelope is returned. Default: ``'lowpass'``.
    axis : int, optional
       Axis of `z` over which to compute the envelope. Default is last the axis.

    Returns
    -------
    ndarray
        If parameter `residual` is ``None`` then an array ``z_env`` with the same shape
        as the input `z` is returned, containing its envelope. Otherwise, an array with
        shape ``(2, *z.shape)``, containing the arrays ``z_env`` and ``z_res``, stacked
        along the first axis, is returned.
        It allows unpacking, i.e., ``z_env, z_res = envelope(z, residual='all')``.
        The residual ``z_res`` contains the signal part which the input bandpass filter
        removed, depending on the parameter `residual`. Note that for real-valued
        signals, a real-valued residual is returned. Hence, the negative frequency
        components of `bp_in` are ignored.

    Notes
    -----
    Any complex-valued signal :math:`z(t)` can be described by a real-valued
    instantaneous amplitude :math:`a(t)` and a real-valued instantaneous phase
    :math:`\phi(t)`, i.e., :math:`z(t) = a(t) \exp\!\big(j \phi(t)\big)`. The
    envelope is defined as the absolute value of the amplitude :math:`|a(t)| = |z(t)|`,
    which is at the same time the absolute value of the signal. Hence, :math:`|a(t)|`
    "envelopes" the class of all signals with amplitude :math:`a(t)` and arbitrary
    phase :math:`\phi(t)`.
    For real-valued signals, :math:`x(t) = a(t) \cos\!\big(\phi(t)\big)` is the
    analogous formulation. Hence, :math:`|a(t)|` can be determined by converting
    :math:`x(t)` into an analytic signal :math:`z_a(t)` by means of a Hilbert
    transform, i.e.,
    :math:`z_a(t) = a(t) \cos\!\big(\phi(t)\big) + j a(t) \sin\!\big(\phi(t) \big)`,
    which produces a complex-valued signal with the same envelope :math:`|a(t)|`.

    The implementation is based on computing the FFT of the input signal and then
    performing the necessary operations in Fourier space. Hence, the typical FFT
    caveats need to be taken into account:

    * The signal is assumed to be periodic. Discontinuities between signal start and
      end can lead to unwanted results due to Gibbs phenomenon.
    * The FFT is slow if the signal length is prime or very long. Also, the memory
      demands are typically higher than a comparable FIR/IIR filter based
      implementation.
    * The frequency spacing ``1 / (n*T)`` for corner frequencies of the bandpass filter
      corresponds to the frequencies produced by ``scipy.fft.fftfreq(len(z), T)``.

    If the envelope of a complex-valued signal `z` with no bandpass filtering is
    desired, i.e., ``bp_in=(None, None)``, then the envelope corresponds to the
    absolute value. Hence, it is more efficient to use ``np.abs(z)`` instead of this
    function.

    Although computing the envelope based on the analytic signal [1]_ is the natural
    method for real-valued signals, other methods are also frequently used. The most
    popular alternative is probably the so-called "square-law" envelope detector and
    its relatives [2]_. They do not always compute the correct result for all kinds of
    signals, but are usually correct and typically computationally more efficient for
    most kinds of narrowband signals. The definition for an envelope presented here is
    common where instantaneous amplitude and phase are of interest (e.g., as described
    in [3]_). There exist also other concepts, which rely on the general mathematical
    idea of an envelope [4]_: A pragmatic approach is to determine all upper and lower
    signal peaks and use a spline interpolation to determine the curves [5]_.


    References
    ----------
    .. [1] "Analytic Signal", Wikipedia,
       https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Lyons, Richard, "Digital envelope detection: The good, the bad, and the
       ugly", IEEE Signal Processing Magazine 34.4 (2017): 183-187.
       `PDF <https://community.infineon.com/gfawx74859/attachments/gfawx74859/psoc135/46469/1/R.%20Lyons_envelope_detection_v3.pdf>`__
    .. [3] T.G. Kincaid, "The complex representation of signals.",
       TIS R67# MH5, General Electric Co. (1966).
       `PDF <https://apps.dtic.mil/sti/tr/pdf/ADA953296.pdf>`__
    .. [4] "Envelope (mathematics)", Wikipedia,
       https://en.wikipedia.org/wiki/Envelope_(mathematics)
    .. [5] Yang, Yanli. "A signal theoretic approach for envelope analysis of
       real-valued signals." IEEE Access 5 (2017): 5623-5630.
       `PDF <https://ieeexplore.ieee.org/iel7/6287639/6514899/07891054.pdf>`__


    See Also
    --------
    hilbert: Compute analytic signal by means of Hilbert transform.


    Examples
    --------
    The following plot illustrates the envelope of a signal with variable frequency and
    a low-frequency drift. To separate the drift from the envelope, a 4 Hz highpass
    filter is used. The low-pass residuum of the input bandpass filter is utilized to
    determine an asymmetric upper and lower bound to enclose the signal. Due to the
    smoothness of the resulting envelope, it is down-sampled from 500 to 40 samples.
    Note that the instantaneous amplitude ``a_x`` and the computed envelope ``x_env``
    are not perfectly identical. This is due to the signal not being perfectly periodic
    as well as the existence of some spectral overlapping of ``x_carrier`` and
    ``x_drift``. Hence, they cannot be completely separated by a bandpass filter.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.signal.windows import gaussian
    >>> from scipy.signal import envelope
    ...
    >>> n, n_out = 500, 40  # number of signal samples and envelope samples
    >>> T = 2 / n  # sampling interval for 2 s duration
    >>> t = np.arange(n) * T  # time stamps
    >>> a_x = gaussian(len(t), 0.4/T)  # instantaneous amplitude
    >>> phi_x = 30*np.pi*t + 35*np.cos(2*np.pi*0.25*t)  # instantaneous phase
    >>> x_carrier = a_x * np.cos(phi_x)
    >>> x_drift = 0.3 * gaussian(len(t), 0.4/T)  # drift
    >>> x = x_carrier + x_drift
    ...
    >>> bp_in = (int(4 * (n*T)), None)  # 4 Hz highpass input filter
    >>> x_env, x_res = envelope(x, bp_in, n_out=n_out)
    >>> t_out = np.arange(n_out) * (n / n_out) * T
    ...
    >>> fg0, ax0 = plt.subplots(1, 1, tight_layout=True)
    >>> ax0.set_title(r"$4\,$Hz Highpass Envelope of Drifting Signal")
    >>> ax0.set(xlabel="Time in seconds", xlim=(0, n*T), ylabel="Amplitude")
    >>> ax0.plot(t, x, 'C0-', alpha=0.5, label="Signal")
    >>> ax0.plot(t, x_drift, 'C2--', alpha=0.25, label="Drift")
    >>> ax0.plot(t_out, x_res+x_env, 'C1.-', alpha=0.5, label="Envelope")
    >>> ax0.plot(t_out, x_res-x_env, 'C1.-', alpha=0.5, label=None)
    >>> ax0.grid(True)
    >>> ax0.legend()
    >>> plt.show()

    The second example provides a geometric envelope interpretation of complex-valued
    signals: The following two plots show the complex-valued signal as a blue
    3d-trajectory and the envelope as an orange round tube with varying diameter, i.e.,
    as :math:`|a(t)| \exp(j\rho(t))`, with :math:`\rho(t)\in[-\pi,\pi]`. Also, the
    projection into the 2d real and imaginary coordinate planes of trajectory and tube
    is depicted. Every point of the complex-valued signal touches the tube's surface.

    The left plot shows an analytic signal, i.e, the phase difference between
    imaginary and real part is always 90 degrees, resulting in a spiraling trajectory.
    It can be seen that in this case the real part has also the expected envelope,
    i.e., representing the absolute value of the instantaneous amplitude.

    The right plot shows the real part of that analytic signal being interpreted
    as a complex-vauled signal, i.e., having zero imaginary part. There the resulting
    envelope is not as smooth as in the analytic case and the instantaneous amplitude
    in the real plane is not recovered. If ``z_re`` had been passed as a real-valued
    signal, i.e., as ``z_re = z.real`` instead of ``z_re = z.real + 0j``, the result
    would have been identical to the left plot. The reason for this is that real-valued
    signals are interpreted as being the real part of a complex-valued analytic signal.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.signal.windows import gaussian
    >>> from scipy.signal import envelope
    ...
    >>> n, T = 1000, 1/1000  # number of samples and sampling interval
    >>> t = np.arange(n) * T  # time stamps for 1 s duration
    >>> f_c = 3  # Carrier frequency for signal
    >>> z = gaussian(len(t), 0.3/T) * np.exp(2j*np.pi*f_c*t)  # analytic signal
    >>> z_re = z.real + 0j  # complex signal with zero imaginary part
    ...
    >>> e_a, e_r = (envelope(z_, (None, None), residual=None) for z_ in (z, z_re))
    ...
    >>> # Generate grids to visualize envelopes as 2d and 3d surfaces:
    >>> E2d_t, E2_amp = np.meshgrid(t, [-1, 1])
    >>> E2d_1 = np.ones_like(E2_amp)
    >>> E3d_t, E3d_phi = np.meshgrid(t, np.linspace(-np.pi, np.pi, 300))
    >>> ma = 1.8  # maximum axis values in real and imaginary direction
    ...
    >>> fg0 = plt.figure(figsize=(6.2, 4.))
    >>> ax00 = fg0.add_subplot(1, 2, 1, projection='3d')
    >>> ax01 = fg0.add_subplot(1, 2, 2, projection='3d', sharex=ax00,
    ...                        sharey=ax00, sharez=ax00)
    >>> ax00.set_title("Analytic Signal")
    >>> ax00.set(xlim=(0, 1), ylim=(-ma, ma), zlim=(-ma, ma))
    >>> ax01.set_title("Real-valued Signal")
    >>> for z_, e_, ax_ in zip((z, z.real), (e_a, e_r), (ax00, ax01)):
    ...     ax_.set(xlabel="Time $t$", ylabel="Real Amp. $x(t)$",
    ...             zlabel="Imag. Amp. $y(t)$")
    ...     ax_.plot(t, z_.real, 'C0-', zs=-ma, zdir='z', alpha=0.5, label="Real")
    ...     ax_.plot_surface(E2d_t, e_*E2_amp, -ma*E2d_1, color='C1', alpha=0.25)
    ...     ax_.plot(t, z_.imag, 'C0-', zs=+ma, zdir='y', alpha=0.5, label="Imag.")
    ...     ax_.plot_surface(E2d_t, ma*E2d_1, e_*E2_amp, color='C1', alpha=0.25)
    ...     ax_.plot(t, z_.real, z_.imag, 'C0-', label="Signal")
    ...     ax_.plot_surface(E3d_t, e_*np.cos(E3d_phi), e_*np.sin(E3d_phi),
    ...                      color='C1', alpha=0.5, shade=True, label="Envelope")
    ...     ax_.view_init(elev=22.7, azim=-114.3)
    >>> fg0.subplots_adjust(left=0.08, right=0.97, wspace=0.15)
    >>> plt.show()
    """
    if not (-z.ndim <= axis < z.ndim):
        raise ValueError(f"Invalid parameter {axis=} for {z.shape=}!")
    if not (z.shape[axis] > 0):
        raise ValueError(f"z.shape[axis] not > 0 for {z.shape=}, {axis=}!")
    if len(bp_in) != 2 or not all((isinstance(b_, int) or b_ is None) for b_ in bp_in):
        raise ValueError(f"{bp_in=} isn't a 2-tuple of type (int | None, int | None)!")
    if not ((isinstance(n_out, int) and 0 < n_out) or n_out is None):
        raise ValueError(f"{n_out=} is not a positive integer or None!")
    if residual not in ('lowpass', 'all', None):
        raise ValueError(f"{residual=} not in ['lowpass', 'all', None]!")

    n = z.shape[axis]  # number of time samples of input
    n_out = n if n_out is None else n_out
    fak = n_out / n  # scaling factor for resampling

    bp = slice(bp_in[0] if bp_in[0] is not None else -(n//2),
               bp_in[1] if bp_in[1] is not None else (n+1)//2)
    if not (-n//2 <= bp.start < bp.stop <= (n+1)//2):
        raise ValueError("`-n//2 <= bp_in[0] < bp_in[1] <= (n+1)//2` does not hold " +
                         f"for n={z.shape[axis]=} and {bp_in=}!")

    # moving active axis to end allows to use `...` for indexing:
    z = np.moveaxis(z, axis, -1)

    if np.iscomplexobj(z):
        Z = sp_fft.fft(z)
    else:  # avoid calculating negative frequency bins for real signals:
        Z = np.zeros_like(z, dtype=sp_fft.rfft(z.flat[:1]).dtype)
        Z[..., :n//2 + 1] = sp_fft.rfft(z)
        if bp.start > 0:  # make signal analytic within bp_in band:
            Z[..., bp] *= 2
        elif bp.stop > 0:
            Z[..., 1:bp.stop] *= 2
    if not (bp.start <= 0 < bp.stop):  # envelope is invariant to freq. shifts.
        z_bb = sp_fft.ifft(Z[..., bp], n=n_out) * fak  # baseband signal
    else:
        bp_shift = slice(bp.start + n//2, bp.stop + n//2)
        z_bb = sp_fft.ifft(sp_fft.fftshift(Z, axes=-1)[..., bp_shift], n=n_out) * fak

    z_env = np.abs(z_bb) if not squared else z_bb.real ** 2 + z_bb.imag ** 2
    z_env = np.moveaxis(z_env, -1, axis)

    # Calculate the residual from the input bandpass filter:
    if residual is None:
        return z_env
    if not (bp.start <= 0 < bp.stop):
        Z[..., bp] = 0
    else:
        Z[..., :bp.stop], Z[..., bp.start:] = 0, 0
    if residual == 'lowpass':
        if bp.stop > 0:
            Z[..., bp.stop:(n+1) // 2] = 0
        else:
            Z[..., bp.start:], Z[..., 0:(n + 1) // 2] = 0, 0

    z_res = fak * (sp_fft.ifft(Z, n=n_out) if np.iscomplexobj(z) else
                   sp_fft.irfft(Z, n=n_out))
    return np.stack((z_env, np.moveaxis(z_res, -1, axis)), axis=0)

def _cmplx_sort(p):
    """Sort roots based on magnitude.

    Parameters
    ----------
    p : array_like
        The roots to sort, as a 1-D array.

    Returns
    -------
    p_sorted : ndarray
        Sorted roots.
    indx : ndarray
        Array of indices needed to sort the input `p`.

    Examples
    --------
    >>> from scipy import signal
    >>> vals = [1, 4, 1+1.j, 3]
    >>> p_sorted, indx = signal.cmplx_sort(vals)
    >>> p_sorted
    array([1.+0.j, 1.+1.j, 3.+0.j, 4.+0.j])
    >>> indx
    array([0, 2, 3, 1])
    """
    p = np.asarray(p)
    indx = np.argsort(abs(p))
    return np.take(p, indx, 0), indx


def unique_roots(p, tol=1e-3, rtype='min'):
    """Determine unique roots and their multiplicities from a list of roots.

    Parameters
    ----------
    p : array_like
        The list of roots.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. Refer to Notes about
        the details on roots grouping.
    rtype : {'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}, optional
        How to determine the returned root if multiple roots are within
        `tol` of each other.

          - 'max', 'maximum': pick the maximum of those roots
          - 'min', 'minimum': pick the minimum of those roots
          - 'avg', 'mean': take the average of those roots

        When finding minimum or maximum among complex roots they are compared
        first by the real part and then by the imaginary part.

    Returns
    -------
    unique : ndarray
        The list of unique roots.
    multiplicity : ndarray
        The multiplicity of each root.

    Notes
    -----
    If we have 3 roots ``a``, ``b`` and ``c``, such that ``a`` is close to
    ``b`` and ``b`` is close to ``c`` (distance is less than `tol`), then it
    doesn't necessarily mean that ``a`` is close to ``c``. It means that roots
    grouping is not unique. In this function we use "greedy" grouping going
    through the roots in the order they are given in the input `p`.

    This utility function is not specific to roots but can be used for any
    sequence of values for which uniqueness and multiplicity has to be
    determined. For a more general routine, see `numpy.unique`.

    Examples
    --------
    >>> from scipy import signal
    >>> vals = [0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3]
    >>> uniq, mult = signal.unique_roots(vals, tol=2e-2, rtype='avg')

    Check which roots have multiplicity larger than 1:

    >>> uniq[mult > 1]
    array([ 1.305])
    """
    if rtype in ['max', 'maximum']:
        reduce = np.max
    elif rtype in ['min', 'minimum']:
        reduce = np.min
    elif rtype in ['avg', 'mean']:
        reduce = np.mean
    else:
        raise ValueError("`rtype` must be one of "
                         "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")

    p = np.asarray(p)

    points = np.empty((len(p), 2))
    points[:, 0] = np.real(p)
    points[:, 1] = np.imag(p)
    tree = cKDTree(points)

    p_unique = []
    p_multiplicity = []
    used = np.zeros(len(p), dtype=bool)
    for i in range(len(p)):
        if used[i]:
            continue

        group = tree.query_ball_point(points[i], tol)
        group = [x for x in group if not used[x]]

        p_unique.append(reduce(p[group]))
        p_multiplicity.append(len(group))

        used[group] = True

    return np.asarray(p_unique), np.asarray(p_multiplicity)


def invres(r, p, k, tol=1e-3, rtype='avg'):
    """Compute b(s) and a(s) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `invresz`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    residue, invresz, unique_roots

    """
    r = np.atleast_1d(r)
    p = np.atleast_1d(p)
    k = np.trim_zeros(np.atleast_1d(k), 'f')

    unique_poles, multiplicity = _group_poles(p, tol, rtype)
    factors, denominator = _compute_factors(unique_poles, multiplicity,
                                            include_powers=True)

    if len(k) == 0:
        numerator = 0
    else:
        numerator = np.polymul(k, denominator)

    for residue, factor in zip(r, factors):
        numerator = np.polyadd(numerator, residue * factor)

    return numerator, denominator


def _compute_factors(roots, multiplicity, include_powers=False):
    """Compute the total polynomial divided by factors for each root."""
    current = np.array([1])
    suffixes = [current]
    for pole, mult in zip(roots[-1:0:-1], multiplicity[-1:0:-1]):
        monomial = np.array([1, -pole])
        for _ in range(mult):
            current = np.polymul(current, monomial)
        suffixes.append(current)
    suffixes = suffixes[::-1]

    factors = []
    current = np.array([1])
    for pole, mult, suffix in zip(roots, multiplicity, suffixes):
        monomial = np.array([1, -pole])
        block = []
        for i in range(mult):
            if i == 0 or include_powers:
                block.append(np.polymul(current, suffix))
            current = np.polymul(current, monomial)
        factors.extend(reversed(block))

    return factors, current


def _compute_residues(poles, multiplicity, numerator):
    denominator_factors, _ = _compute_factors(poles, multiplicity)
    numerator = numerator.astype(poles.dtype)

    residues = []
    for pole, mult, factor in zip(poles, multiplicity,
                                  denominator_factors):
        if mult == 1:
            residues.append(np.polyval(numerator, pole) /
                            np.polyval(factor, pole))
        else:
            numer = numerator.copy()
            monomial = np.array([1, -pole])
            factor, d = np.polydiv(factor, monomial)

            block = []
            for _ in range(mult):
                numer, n = np.polydiv(numer, monomial)
                r = n[0] / d[0]
                numer = np.polysub(numer, r * factor)
                block.append(r)

            residues.extend(reversed(block))

    return np.asarray(residues)


def residue(b, a, tol=1e-3, rtype='avg'):
    """Compute partial-fraction expansion of b(s) / a(s).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `residuez`.

    See Notes for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    See Also
    --------
    invres, residuez, numpy.poly, unique_roots

    Notes
    -----
    The "deflation through subtraction" algorithm is used for
    computations --- method 6 in [1]_.

    The form of partial fraction expansion depends on poles multiplicity in
    the exact mathematical sense. However there is no way to exactly
    determine multiplicity of roots of a polynomial in numerical computing.
    Thus you should think of the result of `residue` with given `tol` as
    partial fraction expansion computed for the denominator composed of the
    computed poles with empirically determined multiplicity. The choice of
    `tol` can drastically change the result if there are close poles.

    References
    ----------
    .. [1] J. F. Mahoney, B. D. Sivazlian, "Partial fractions expansion: a
           review of computational methodology and efficiency", Journal of
           Computational and Applied Mathematics, Vol. 9, 1983.
    """
    b = np.asarray(b)
    a = np.asarray(a)
    if (np.issubdtype(b.dtype, np.complexfloating)
            or np.issubdtype(a.dtype, np.complexfloating)):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)

    b = np.trim_zeros(np.atleast_1d(b), 'f')
    a = np.trim_zeros(np.atleast_1d(a), 'f')

    if a.size == 0:
        raise ValueError("Denominator `a` is zero.")

    poles = np.roots(a)
    if b.size == 0:
        return np.zeros(poles.shape), _cmplx_sort(poles)[0], np.array([])

    if len(b) < len(a):
        k = np.empty(0)
    else:
        k, b = np.polydiv(b, a)

    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]

    residues = _compute_residues(unique_poles, multiplicity, b)

    index = 0
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        index += mult

    return residues / a[0], poles, k


def residuez(b, a, tol=1e-3, rtype='avg'):
    """Compute partial-fraction expansion of b(z) / a(z).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `residue`.

    See Notes of `residue` for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    See Also
    --------
    invresz, residue, unique_roots
    """
    b = np.asarray(b)
    a = np.asarray(a)
    if (np.issubdtype(b.dtype, np.complexfloating)
            or np.issubdtype(a.dtype, np.complexfloating)):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)

    b = np.trim_zeros(np.atleast_1d(b), 'b')
    a = np.trim_zeros(np.atleast_1d(a), 'b')

    if a.size == 0:
        raise ValueError("Denominator `a` is zero.")
    elif a[0] == 0:
        raise ValueError("First coefficient of determinant `a` must be "
                         "non-zero.")

    poles = np.roots(a)
    if b.size == 0:
        return np.zeros(poles.shape), _cmplx_sort(poles)[0], np.array([])

    b_rev = b[::-1]
    a_rev = a[::-1]

    if len(b_rev) < len(a_rev):
        k_rev = np.empty(0)
    else:
        k_rev, b_rev = np.polydiv(b_rev, a_rev)

    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]

    residues = _compute_residues(1 / unique_poles, multiplicity, b_rev)

    index = 0
    powers = np.empty(len(residues), dtype=int)
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        powers[index:index + mult] = 1 + np.arange(mult)
        index += mult

    residues *= (-poles) ** powers / a_rev[0]

    return residues, poles, k_rev[::-1]


def _group_poles(poles, tol, rtype):
    if rtype in ['max', 'maximum']:
        reduce = np.max
    elif rtype in ['min', 'minimum']:
        reduce = np.min
    elif rtype in ['avg', 'mean']:
        reduce = np.mean
    else:
        raise ValueError("`rtype` must be one of "
                         "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")

    unique = []
    multiplicity = []

    pole = poles[0]
    block = [pole]
    for i in range(1, len(poles)):
        if abs(poles[i] - pole) <= tol:
            block.append(pole)
        else:
            unique.append(reduce(block))
            multiplicity.append(len(block))
            pole = poles[i]
            block = [pole]

    unique.append(reduce(block))
    multiplicity.append(len(block))

    return np.asarray(unique), np.asarray(multiplicity)


def invresz(r, p, k, tol=1e-3, rtype='avg'):
    """Compute b(z) and a(z) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `invres`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    residuez, unique_roots, invres

    """
    r = np.atleast_1d(r)
    p = np.atleast_1d(p)
    k = np.trim_zeros(np.atleast_1d(k), 'b')

    unique_poles, multiplicity = _group_poles(p, tol, rtype)
    factors, denominator = _compute_factors(unique_poles, multiplicity,
                                            include_powers=True)

    if len(k) == 0:
        numerator = 0
    else:
        numerator = np.polymul(k[::-1], denominator[::-1])

    for residue, factor in zip(r, factors):
        numerator = np.polyadd(numerator, residue * factor[::-1])

    return numerator[::-1], denominator


def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the equally spaced sample
        positions associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:
        ``time`` Consider the input `x` as time-domain (Default),
        ``freq`` Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `scipy.signal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it is used solely to calculate the resampled
    positions `resampled_t`

    As noted, `resample` uses FFT transformations, which can be very 
    slow if the number of input or output samples is large and prime; 
    see :func:`~scipy.fft.fft`. In such cases, it can be faster to first downsample 
    a signal of length ``n`` with :func:`~scipy.signal.resample_poly` by a factor of 
    ``n//num`` before using `resample`. Note that this approach changes the 
    characteristics of the antialiasing filter.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> import numpy as np
    >>> from scipy import signal

    >>> x = np.linspace(0, 10, 20, endpoint=False)
    >>> y = np.cos(-x**2/6.0)
    >>> f = signal.resample(y, 100)
    >>> xnew = np.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()

    Consider the following signal  ``y`` where ``len(y)`` is a large  prime number:

    >>> N = 55949
    >>> freq = 100
    >>> x = np.linspace(0, 1, N)
    >>> y = np.cos(2 * np.pi * freq * x)

    Due to ``N`` being prime,

    >>> num = 5000
    >>> f = signal.resample(signal.resample_poly(y, 1, N // num), num)

    runs significantly faster than

    >>> f = signal.resample(y, num)
    """

    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or"
                         f" 'freq', not domain={domain}")

    x = np.asarray(x)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = np.isrealobj(x)

    if domain == 'time':
        # Forward transform
        if real_input:
            X = sp_fft.rfft(x, axis=axis)
        else:  # Full complex FFT
            X = sp_fft.fft(x, axis=axis)
    else:  # domain == 'freq'
        X = x

    # Apply window to spectrum
    if window is not None:
        if callable(window):
            W = window(sp_fft.fftfreq(Nx))
        elif isinstance(window, np.ndarray):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            W = sp_fft.ifftshift(get_window(window, Nx))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # Fold the window back on itself to mimic complex behavior
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequencies (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = np.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # Copy negative frequency components
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            if real_input:
                sl[axis] = slice(N//2, N//2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                # select the component of Y at frequency +N/2,
                # add the component of X at -N/2
                sl[axis] = slice(-N//2, -N//2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                # set the component at -N/2 equal to the component at +N/2
                sl[axis] = slice(num-N//2, num-N//2 + 1)
                Y[tuple(sl)] = temp

    # Inverse transform
    if real_input:
        y = sp_fft.irfft(Y, num, axis=axis)
    else:
        y = sp_fft.ifft(Y, axis=axis, overwrite_x=True)

    y *= (float(num) / float(Nx))

    if t is None:
        return y
    else:
        new_t = np.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t


def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0),
                  padtype='constant', cval=None):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. By default, values beyond the boundary of the signal are assumed
    to be zero during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by `scipy.signal.upfirdn`.
        Changes assumptions on values beyond the boundary. If `constant`,
        assumed to be `cval` (default zero). If `line` assumed to continue a
        linear trend defined by the first and last points. `mean`, `median`,
        `maximum` and `minimum` work as in `np.pad` and assume that the values
        beyond the boundary are the mean, median, maximum or minimum
        respectively of the array along the axis.

        .. versionadded:: 1.4.0
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

        .. versionadded:: 1.4.0

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `scipy.signal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `scipy.signal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `scipy.signal.get_window`
    and `scipy.signal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    By default, the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(0, 10, 20, endpoint=False)
    >>> y = np.cos(-x**2/6.0)
    >>> f_fft = signal.resample(y, 100)
    >>> f_poly = signal.resample_poly(y, 100, 20)
    >>> xnew = np.linspace(0, 10, 100, endpoint=False)

    >>> plt.plot(xnew, f_fft, 'b.-', xnew, f_poly, 'r.-')
    >>> plt.plot(x, y, 'ko-')
    >>> plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()

    This default behaviour can be changed by using the padtype option:

    >>> N = 5
    >>> x = np.linspace(0, 1, N, endpoint=False)
    >>> y = 2 + x**2 - 1.7*np.sin(x) + .2*np.cos(11*x)
    >>> y2 = 1 + x**3 + 0.1*np.sin(x) + .1*np.cos(11*x)
    >>> Y = np.stack([y, y2], axis=-1)
    >>> up = 4
    >>> xr = np.linspace(0, 1, N*up, endpoint=False)

    >>> y2 = signal.resample_poly(Y, up, 1, padtype='constant')
    >>> y3 = signal.resample_poly(Y, up, 1, padtype='mean')
    >>> y4 = signal.resample_poly(Y, up, 1, padtype='line')

    >>> for i in [0,1]:
    ...     plt.figure()
    ...     plt.plot(xr, y4[:,i], 'g.', label='line')
    ...     plt.plot(xr, y3[:,i], 'y.', label='mean')
    ...     plt.plot(xr, y2[:,i], 'r.', label='constant')
    ...     plt.plot(x, Y[:,i], 'k-')
    ...     plt.legend()
    >>> plt.show()

    """
    x = np.asarray(x)
    if up != int(up):
        raise ValueError("up must be an integer")
    if down != int(down):
        raise ValueError("down must be an integer")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    if cval is not None and padtype != 'constant':
        raise ValueError('cval has no effect when padtype is ', padtype)

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_in = x.shape[axis]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list | np.ndarray)):
        window = np.array(window)  # use array to force a copy (we modify it)
        if window.ndim > 1:
            raise ValueError('window must be 1-D')
        half_len = (window.size - 1) // 2
        h = window
    else:
        # Design a linear-phase low-pass FIR filter
        max_rate = max(up, down)
        f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
        half_len = 10 * max_rate  # reasonable cutoff for sinc-like function
        if np.issubdtype(x.dtype, np.complexfloating):
            h = firwin(2 * half_len + 1, f_c,
                       window=window).astype(x.dtype)  # match dtype of x
        elif np.issubdtype(x.dtype, np.floating):
            h = firwin(2 * half_len + 1, f_c,
                       window=window).astype(x.dtype)  # match dtype of x
        else:
            h = firwin(2 * half_len + 1, f_c,
                       window=window)
    h *= up

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = (down - half_len % down)
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in,
                      up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate((np.zeros(n_pre_pad, dtype=h.dtype), h,
                        np.zeros(n_post_pad, dtype=h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out

    # Remove background depending on the padtype option
    funcs = {'mean': np.mean, 'median': np.median,
             'minimum': np.amin, 'maximum': np.amax}
    upfirdn_kwargs = {'mode': 'constant', 'cval': 0}
    if padtype in funcs:
        background_values = funcs[padtype](x, axis=axis, keepdims=True)
    elif padtype in _upfirdn_modes:
        upfirdn_kwargs = {'mode': padtype}
        if padtype == 'constant':
            if cval is None:
                cval = 0
            upfirdn_kwargs['cval'] = cval
    else:
        raise ValueError(
            'padtype must be one of: maximum, mean, median, minimum, ' +
            ', '.join(_upfirdn_modes))

    if padtype in funcs:
        x = x - background_values

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis=axis, **upfirdn_kwargs)
    keep = [slice(None), ]*x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    y_keep = y[tuple(keep)]

    # Add background back
    if padtype in funcs:
        y_keep += background_values

    return y_keep


def vectorstrength(events, period):
    '''
    Determine the vector strength of the events corresponding to the given
    period.

    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.

    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".

    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.

    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.

    References
    ----------
    van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating vector
        strength: Auditory system, electric fish, and noise.
        Chaos 21, 047508 (2011);
        :doi:`10.1063/1.3670512`.
    van Hemmen, JL.  Vector strength after Goldberg, Brown, and von Mises:
        biological and mathematical perspectives.  Biol Cybern.
        2013 Aug;107(4):385-96. :doi:`10.1007/s00422-013-0561-7`.
    van Hemmen, JL and Vollmayr, AN.  Resonating vector strength: what happens
        when we vary the "probing" frequency while keeping the spike times
        fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
        :doi:`10.1007/s00422-013-0560-8`.
    '''
    events = np.asarray(events)
    period = np.asarray(period)
    if events.ndim > 1:
        raise ValueError('events cannot have dimensions more than 1')
    if period.ndim > 1:
        raise ValueError('period cannot have dimensions more than 1')

    # we need to know later if period was originally a scalar
    scalarperiod = not period.ndim

    events = np.atleast_2d(events)
    period = np.atleast_2d(period)
    if (period <= 0).any():
        raise ValueError('periods must be positive')

    # this converts the times to vectors
    vectors = np.exp(np.dot(2j*np.pi/period.T, events))

    # the vector strength is just the magnitude of the mean of the vectors
    # the vector phase is the angle of the mean of the vectors
    vectormean = np.mean(vectors, axis=1)
    strength = abs(vectormean)
    phase = np.angle(vectormean)

    # if the original period was a scalar, return scalars
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
    return strength, phase


def detrend(data: np.ndarray, axis: int = -1,
            type: Literal['linear', 'constant'] = 'linear',
            bp: ArrayLike | int = 0, overwrite_data: bool = False) -> np.ndarray:
    r"""Remove linear or constant trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Notes
    -----
    Detrending can be interpreted as subtracting a least squares fit polynomial:
    Setting the parameter `type` to 'constant' corresponds to fitting a zeroth degree
    polynomial, 'linear' to a first degree polynomial. Consult the example below.

    See Also
    --------
    numpy.polynomial.polynomial.Polynomial.fit: Create least squares fit polynomial.


    Examples
    --------
    The following example detrends the function :math:`x(t) = \sin(\pi t) + 1/4`:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.signal import detrend
    ...
    >>> t = np.linspace(-0.5, 0.5, 21)
    >>> x = np.sin(np.pi*t) + 1/4
    ...
    >>> x_d_const = detrend(x, type='constant')
    >>> x_d_linear = detrend(x, type='linear')
    ...
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.set_title(r"Detrending $x(t)=\sin(\pi t) + 1/4$")
    >>> ax1.set(xlabel="t", ylabel="$x(t)$", xlim=(t[0], t[-1]))
    >>> ax1.axhline(y=0, color='black', linewidth=.5)
    >>> ax1.axvline(x=0, color='black', linewidth=.5)
    >>> ax1.plot(t, x, 'C0.-',  label="No detrending")
    >>> ax1.plot(t, x_d_const, 'C1x-', label="type='constant'")
    >>> ax1.plot(t, x_d_linear, 'C2+-', label="type='linear'")
    >>> ax1.legend()
    >>> plt.show()

    Alternatively, NumPy's `~numpy.polynomial.polynomial.Polynomial` can be used for
    detrending as well:

    >>> pp0 = np.polynomial.Polynomial.fit(t, x, deg=0)  # fit degree 0 polynomial
    >>> np.allclose(x_d_const, x - pp0(t))  # compare with constant detrend
    True
    >>> pp1 = np.polynomial.Polynomial.fit(t, x, deg=1)  # fit degree 1 polynomial
    >>> np.allclose(x_d_linear, x - pp1(t))  # compare with linear detrend
    True

    Note that `~numpy.polynomial.polynomial.Polynomial` also allows fitting higher
    degree polynomials. Consult its documentation on how to extract the polynomial
    coefficients.
    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.concatenate(np.atleast_1d(0, bp, N))))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")

        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = np.moveaxis(data, axis, 0)
        newdata_shape = newdata.shape
        newdata = newdata.reshape(N, -1)

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)

#        Nreg = len(bp) - 1
        # Find leastsq fit and remove it for each piece
        for m in range(len(bp) - 1):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.arange(1, Npts + 1, dtype=dtype) / Npts
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - A @ coef

        # Put data back in original shape.
        newdata = newdata.reshape(newdata_shape)
        ret = np.moveaxis(newdata, 0, axis)
        return ret


def lfilter_zi(b, a):
    """
    Construct initial conditions for lfilter for step response steady-state.

    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    b, a : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.

    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.

    See Also
    --------
    lfilter, lfiltic, filtfilt

    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::

        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)

    where z(n) is a vector of length m, A has shape (m, m), B has shape
    (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
    a scalar).  lfilter_zi solves::

        zi = A*zi + B

    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.

    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter,
    which is the implementation used by scipy.signal.lfilter, are::

        A = scipy.linalg.companion(a).T
        B = b[1:] - a[1:]*b[0]

    assuming ``a[0]`` is 1.0; if ``a[0]`` is not 1, `a` and `b` are first
    divided by a[0].

    Examples
    --------
    The following code creates a lowpass Butterworth filter. Then it
    applies that filter to an array whose values are all 1.0; the
    output is also all 1.0, as expected for a lowpass filter.  If the
    `zi` argument of `lfilter` had not been given, the output would have
    shown the transient signal.

    >>> from numpy import array, ones
    >>> from scipy.signal import lfilter, lfilter_zi, butter
    >>> b, a = butter(5, 0.25)
    >>> zi = lfilter_zi(b, a)
    >>> y, zo = lfilter(b, a, ones(10), zi=zi)
    >>> y
    array([1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

    Another example:

    >>> x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    >>> y, zf = lfilter(b, a, x, zi=zi*x[0])
    >>> y
    array([ 0.5       ,  0.5       ,  0.5       ,  0.49836039,  0.48610528,
        0.44399389,  0.35505241])

    Note that the `zi` argument to `lfilter` was computed using
    `lfilter_zi` and scaled by ``x[0]``.  Then the output `y` has no
    transient until the input drops from 0.5 to 0.0.

    """

    # FIXME: Can this function be replaced with an appropriate
    # use of lfiltic?  For example, when b,a = butter(N,Wn),
    #    lfiltic(b, a, y=numpy.ones_like(a), x=numpy.ones_like(b)).
    #

    # We could use scipy.signal.normalize, but it uses warnings in
    # cases where a ValueError is more appropriate, and it allows
    # b to be 2D.
    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be 1-D.")
    a = np.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be 1-D.")

    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    if a[0] != 1.0:
        # Normalize the coefficients so a[0] == 1.
        b = b / a[0]
        a = a / a[0]

    n = max(len(a), len(b))

    # Pad a or b with zeros so they are the same length.
    if len(a) < n:
        a = np.r_[a, np.zeros(n - len(a), dtype=a.dtype)]
    elif len(b) < n:
        b = np.r_[b, np.zeros(n - len(b), dtype=b.dtype)]

    IminusA = np.eye(n - 1, dtype=np.result_type(a, b)) - linalg.companion(a).T
    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    zi = np.linalg.solve(IminusA, B)

    # For future reference: we could also use the following
    # explicit formulas to solve the linear system:
    #
    # zi = np.zeros(n - 1)
    # zi[0] = B.sum() / IminusA[:,0].sum()
    # asum = 1.0
    # csum = 0.0
    # for k in range(1,n-1):
    #     asum += a[k]
    #     csum += b[k] - a[k]*b[0]
    #     zi[k] = asum*zi[0] - csum

    return zi


def sosfilt_zi(sos):
    """
    Construct initial conditions for sosfilt for step response steady-state.

    Compute an initial state `zi` for the `sosfilt` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    zi : ndarray
        Initial conditions suitable for use with ``sosfilt``, shape
        ``(n_sections, 2)``.

    See Also
    --------
    sosfilt, zpk2sos

    Notes
    -----
    .. versionadded:: 0.16.0

    Examples
    --------
    Filter a rectangular pulse that begins at time 0, with and without
    the use of the `zi` argument of `scipy.signal.sosfilt`.

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> sos = signal.butter(9, 0.125, output='sos')
    >>> zi = signal.sosfilt_zi(sos)
    >>> x = (np.arange(250) < 100).astype(int)
    >>> f1 = signal.sosfilt(sos, x)
    >>> f2, zo = signal.sosfilt(sos, x, zi=zi)

    >>> plt.plot(x, 'k--', label='x')
    >>> plt.plot(f1, 'b', alpha=0.5, linewidth=2, label='filtered')
    >>> plt.plot(f2, 'g', alpha=0.25, linewidth=4, label='filtered with zi')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    sos = np.asarray(sos)
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')

    if sos.dtype.kind in 'bui':
        sos = sos.astype(np.float64)

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2), dtype=sos.dtype)
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        zi[section] = scale * lfilter_zi(b, a)
        # If H(z) = B(z)/A(z) is this section's transfer function, then
        # b.sum()/a.sum() is H(1), the gain at omega=0.  That's the steady
        # state value of this section's step response.
        scale *= b.sum() / a.sum()

    return zi


def _filtfilt_gust(b, a, x, axis=-1, irlen=None):
    """Forward-backward IIR filter that uses Gustafsson's method.

    Apply the IIR filter defined by ``(b,a)`` to `x` twice, first forward
    then backward, using Gustafsson's initial conditions [1]_.

    Let ``y_fb`` be the result of filtering first forward and then backward,
    and let ``y_bf`` be the result of filtering first backward then forward.
    Gustafsson's method is to compute initial conditions for the forward
    pass and the backward pass such that ``y_fb == y_bf``.

    Parameters
    ----------
    b : scalar or 1-D ndarray
        Numerator coefficients of the filter.
    a : scalar or 1-D ndarray
        Denominator coefficients of the filter.
    x : ndarray
        Data to be filtered.
    axis : int, optional
        Axis of `x` to be filtered.  Default is -1.
    irlen : int or None, optional
        The length of the nonnegligible part of the impulse response.
        If `irlen` is None, or if the length of the signal is less than
        ``2 * irlen``, then no part of the impulse response is ignored.

    Returns
    -------
    y : ndarray
        The filtered data.
    x0 : ndarray
        Initial condition for the forward filter.
    x1 : ndarray
        Initial condition for the backward filter.

    Notes
    -----
    Typically the return values `x0` and `x1` are not needed by the
    caller.  The intended use of these return values is in unit tests.

    References
    ----------
    .. [1] F. Gustaffson. Determining the initial states in forward-backward
           filtering. Transactions on Signal Processing, 46(4):988-992, 1996.

    """
    # In the comments, "Gustafsson's paper" and [1] refer to the
    # paper referenced in the docstring.

    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    order = max(len(b), len(a)) - 1
    if order == 0:
        # The filter is just scalar multiplication, with no state.
        scale = (b[0] / a[0])**2
        y = scale * x
        return y, np.array([]), np.array([])

    if axis != -1 or axis != x.ndim - 1:
        # Move the axis containing the data to the end.
        x = np.swapaxes(x, axis, x.ndim - 1)

    # n is the number of samples in the data to be filtered.
    n = x.shape[-1]

    if irlen is None or n <= 2*irlen:
        m = n
    else:
        m = irlen

    # Create Obs, the observability matrix (called O in the paper).
    # This matrix can be interpreted as the operator that propagates
    # an arbitrary initial state to the output, assuming the input is
    # zero.
    # In Gustafsson's paper, the forward and backward filters are not
    # necessarily the same, so he has both O_f and O_b.  We use the same
    # filter in both directions, so we only need O. The same comment
    # applies to S below.
    Obs = np.zeros((m, order))
    zi = np.zeros(order)
    zi[0] = 1
    Obs[:, 0] = lfilter(b, a, np.zeros(m), zi=zi)[0]
    for k in range(1, order):
        Obs[k:, k] = Obs[:-k, 0]

    # Obsr is O^R (Gustafsson's notation for row-reversed O)
    Obsr = Obs[::-1]

    # Create S.  S is the matrix that applies the filter to the reversed
    # propagated initial conditions.  That is,
    #     out = S.dot(zi)
    # is the same as
    #     tmp, _ = lfilter(b, a, zeros(), zi=zi)  # Propagate ICs.
    #     out = lfilter(b, a, tmp[::-1])          # Reverse and filter.

    # Equations (5) & (6) of [1]
