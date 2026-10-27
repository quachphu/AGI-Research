    return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]

def gmpy_mpf_mul_int(s, n, prec, rnd=round_fast):
    """Multiply by a Python integer."""
    sign, man, exp, bc = s
    if not man:
        return mpf_mul(s, from_int(n), prec, rnd)
    if not n:
        return fzero
    if n < 0:
        sign ^= 1
        n = -n
    man *= n
    return normalize(sign, man, exp, bitcount(man), prec, rnd)

def python_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman*tman
    if man:
        bc = sbc + tbc - 1
        bc += int(man>>bc)
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
    return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]

def python_mpf_mul_int(s, n, prec, rnd=round_fast):
    """Multiply by a Python integer."""
    sign, man, exp, bc = s
    if not man:
        return mpf_mul(s, from_int(n), prec, rnd)
    if not n:
        return fzero
    if n < 0:
        sign ^= 1
        n = -n
    man *= n
    # Generally n will be small
    if n < 1024:
        bc += bctable[int(n)] - 1
    else:
        bc += bitcount(n) - 1
    bc += int(man>>bc)
    return normalize(sign, man, exp, bc, prec, rnd)


if BACKEND == 'gmpy':
    mpf_mul = gmpy_mpf_mul
    mpf_mul_int = gmpy_mpf_mul_int
else:
    mpf_mul = python_mpf_mul
    mpf_mul_int = python_mpf_mul_int

def mpf_shift(s, n):
    """Quickly multiply the raw mpf s by 2**n without rounding."""
    sign, man, exp, bc = s
    if not man:
        return s
    return sign, man, exp+n, bc

def mpf_frexp(x):
    """Convert x = y*2**n to (y, n) with abs(y) in [0.5, 1) if nonzero"""
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return (fzero, 0)
        else:
            raise ValueError
    return mpf_shift(x, -bc-exp), bc+exp

def mpf_div(s, t, prec, rnd=round_fast):
    """Floating-point division"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if not sman or not tman:
        if s == fzero:
            if t == fzero: raise ZeroDivisionError
            if t == fnan: return fnan
            return fzero
        if t == fzero:
            raise ZeroDivisionError
        s_special = (not sman) and sexp
        t_special = (not tman) and texp
        if s_special and t_special:
            return fnan
        if s == fnan or t == fnan:
            return fnan
        if not t_special:
            if t == fzero:
                return fnan
            return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]
        return fzero
    sign = ssign ^ tsign
    if tman == 1:
        return normalize1(sign, sman, sexp-texp, sbc, prec, rnd)
    # Same strategy as for addition: if there is a remainder, perturb
    # the result a few bits outside the precision range before rounding
    extra = prec - sbc + tbc + 5
    if extra < 5:
        extra = 5
    quot, rem = divmod(sman<<extra, tman)
    if rem:
        quot = (quot<<1) + 1
        extra += 1
        return normalize1(sign, quot, sexp-texp-extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, sexp-texp-extra, bitcount(quot), prec, rnd)

def mpf_rdiv_int(n, t, prec, rnd=round_fast):
    """Floating-point division n/t with a Python integer as numerator"""
    sign, man, exp, bc = t
    if not n or not man:
        return mpf_div(from_int(n), t, prec, rnd)
    if n < 0:
        sign ^= 1
        n = -n
    extra = prec + bc + 5
    quot, rem = divmod(n<<extra, man)
    if rem:
        quot = (quot<<1) + 1
        extra += 1
        return normalize1(sign, quot, -exp-extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, -exp-extra, bitcount(quot), prec, rnd)

def mpf_mod(s, t, prec, rnd=round_fast):
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if ((not sman) and sexp) or ((not tman) and texp):
        return fnan
    # Important special case: do nothing if t is larger
    if ssign == tsign and texp > sexp+sbc:
        return s
    # Another important special case: this allows us to do e.g. x % 1.0
    # to find the fractional part of x, and it will work when x is huge.
    if tman == 1 and sexp > texp+tbc:
        return fzero
    base = min(sexp, texp)
    sman = (-1)**ssign * sman
    tman = (-1)**tsign * tman
    man = (sman << (sexp-base)) % (tman << (texp-base))
    if man >= 0:
        sign = 0
    else:
        man = -man
        sign = 1
    return normalize(sign, man, base, bitcount(man), prec, rnd)

reciprocal_rnd = {
  round_down : round_up,
  round_up : round_down,
  round_floor : round_ceiling,
  round_ceiling : round_floor,
  round_nearest : round_nearest
}

negative_rnd = {
  round_down : round_down,
  round_up : round_up,
  round_floor : round_ceiling,
  round_ceiling : round_floor,
  round_nearest : round_nearest
}

def mpf_pow_int(s, n, prec, rnd=round_fast):
    """Compute s**n, where s is a raw mpf and n is a Python integer."""
    sign, man, exp, bc = s

    if (not man) and exp:
        if s == finf:
            if n > 0: return s
            if n == 0: return fnan
            return fzero
        if s == fninf:
            if n > 0: return [finf, fninf][n & 1]
            if n == 0: return fnan
            return fzero
        return fnan

    n = int(n)
    if n == 0: return fone
    if n == 1: return mpf_pos(s, prec, rnd)
    if n == 2:
        _, man, exp, bc = s
        if not man:
            return fzero
        man = man*man
        if man == 1:
            return (0, MPZ_ONE, exp+exp, 1)
        bc = bc + bc - 2
        bc += bctable[int(man>>bc)]
        return normalize1(0, man, exp+exp, bc, prec, rnd)
    if n == -1: return mpf_div(fone, s, prec, rnd)
    if n < 0:
        inverse = mpf_pow_int(s, -n, prec+5, reciprocal_rnd[rnd])
        return mpf_div(fone, inverse, prec, rnd)

    result_sign = sign & n

    # Use exact integer power when the exact mantissa is small
    if man == 1:
        return (result_sign, MPZ_ONE, exp*n, 1)
    if bc*n < 1000:
        man **= n
        return normalize1(result_sign, man, exp*n, bitcount(man), prec, rnd)

    # Use directed rounding all the way through to maintain rigorous
    # bounds for interval arithmetic
    rounds_down = (rnd == round_nearest) or \
        shifts_down[rnd][result_sign]

    # Now we perform binary exponentiation. Need to estimate precision
    # to avoid rounding errors from temporary operations. Roughly log_2(n)
    # operations are performed.
    workprec = prec + 4*bitcount(n) + 4
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm*man
            pe = pe+exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                if rounds_down:
                    pm = pm >> (pbc-workprec)
                else:
                    pm = -((-pm) >> (pbc-workprec))
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        man = man*man
        exp = exp+exp
        bc = bc + bc - 2
        bc = bc + bctable[int(man >> bc)]
        if bc > workprec:
            if rounds_down:
                man = man >> (bc-workprec)
            else:
                man = -((-man) >> (bc-workprec))
            exp += bc - workprec
            bc = workprec
        n = n // 2

    return normalize(result_sign, pm, pe, pbc, prec, rnd)


def mpf_perturb(x, eps_sign, prec, rnd):
    """
    For nonzero x, calculate x + eps with directed rounding, where
    eps < prec relatively and eps has the given sign (0 for
    positive, 1 for negative).

    With rounding to nearest, this is taken to simply normalize
    x to the given precision.
    """
    if rnd == round_nearest:
        return mpf_pos(x, prec, rnd)
    sign, man, exp, bc = x
    eps = (eps_sign, MPZ_ONE, exp+bc-prec-1, 1)
    if sign:
        away = (rnd in (round_down, round_ceiling)) ^ eps_sign
    else:
        away = (rnd in (round_up, round_ceiling)) ^ eps_sign
    if away:
        return mpf_add(x, eps, prec, rnd)
    else:
        return mpf_pos(x, prec, rnd)


#----------------------------------------------------------------------------#
#                              Radix conversion                              #
#----------------------------------------------------------------------------#

def to_digits_exp(s, dps):
    """Helper function for representing the floating-point number s as
    a decimal with dps digits. Returns (sign, string, exponent) where
    sign is '' or '-', string is the digit string, and exponent is
    the decimal exponent as an int.

    If inexact, the decimal representation is rounded toward zero."""

    # Extract sign first so it doesn't mess up the string digit count
    if s[0]:
        sign = '-'
        s = mpf_neg(s)
    else:
        sign = ''
    _sign, man, exp, bc = s

    if not man:
        return '', '0', 0

    bitprec = int(dps * math.log(10,2)) + 10

    # Cut down to size
    # TODO: account for precision when doing this
    exp_from_1 = exp + bc
    if abs(exp_from_1) > 3500:
        from .libelefun import mpf_ln2, mpf_ln10
        # Set b = int(exp * log(2)/log(10))
        # If exp is huge, we must use high-precision arithmetic to
        # find the nearest power of ten
        expprec = bitcount(abs(exp)) + 5
        tmp = from_int(exp)
        tmp = mpf_mul(tmp, mpf_ln2(expprec))
        tmp = mpf_div(tmp, mpf_ln10(expprec), expprec)
        b = to_int(tmp)
        s = mpf_div(s, mpf_pow_int(ften, b, bitprec), bitprec)
        _sign, man, exp, bc = s
        exponent = b
    else:
        exponent = 0

    # First, calculate mantissa digits by converting to a binary
    # fixed-point number and then converting that number to
    # a decimal fixed-point number.
    fixprec = max(bitprec - exp - bc, 0)
    fixdps = int(fixprec / math.log(10,2) + 0.5)
    sf = to_fixed(s, fixprec)
    sd = bin_to_radix(sf, fixprec, 10, fixdps)
    digits = numeral(sd, base=10, size=dps)

    exponent += len(digits) - fixdps - 1
    return sign, digits, exponent

def to_str(s, dps, strip_zeros=True, min_fixed=None, max_fixed=None,
    show_zero_exponent=False):
    """
    Convert a raw mpf to a decimal floating-point literal with at
    most `dps` decimal digits in the mantissa (not counting extra zeros
    that may be inserted for visual purposes).

    The number will be printed in fixed-point format if the position
    of the leading digit is strictly between min_fixed
    (default = min(-dps/3,-5)) and max_fixed (default = dps).

    To force fixed-point format always, set min_fixed = -inf,
    max_fixed = +inf. To force floating-point format, set
    min_fixed >= max_fixed.

    The literal is formatted so that it can be parsed back to a number
    by to_str, float() or Decimal().
    """

    # Special numbers
    if not s[1]:
        if s == fzero:
            if dps: t = '0.0'
            else:   t = '.0'
            if show_zero_exponent:
                t += 'e+0'
            return t
        if s == finf: return '+inf'
        if s == fninf: return '-inf'
        if s == fnan: return 'nan'
        raise ValueError

    if min_fixed is None: min_fixed = min(-(dps//3), -5)
    if max_fixed is None: max_fixed = dps

    # to_digits_exp rounds to floor.
    # This sometimes kills some instances of "...00001"
    sign, digits, exponent = to_digits_exp(s, dps+3)

    # No digits: show only .0; round exponent to nearest
    if not dps:
        if digits[0] in '56789':
            exponent += 1
        digits = ".0"

    else:
        # Rounding up kills some instances of "...99999"
        if len(digits) > dps and digits[dps] in '56789':
            digits = digits[:dps]
            i = dps - 1
            while i >= 0 and digits[i] == '9':
                i -= 1
            if i >= 0:
                digits = digits[:i] + str(int(digits[i]) + 1) + '0' * (dps - i - 1)
            else:
                digits = '1' + '0' * (dps - 1)
                exponent += 1
        else:
            digits = digits[:dps]

        # Prettify numbers close to unit magnitude
        if min_fixed < exponent < max_fixed:
            if exponent < 0:
                digits = ("0"*int(-exponent)) + digits
                split = 1
            else:
                split = exponent + 1
                if split > dps:
                    digits += "0"*(split-dps)
            exponent = 0
        else:
            split = 1

        digits = (digits[:split] + "." + digits[split:])

        if strip_zeros:
            # Clean up trailing zeros
            digits = digits.rstrip('0')
            if digits[-1] == ".":
                digits += "0"

    if exponent == 0 and dps and not show_zero_exponent: return sign + digits
    if exponent >= 0: return sign + digits + "e+" + str(exponent)
    if exponent < 0: return sign + digits + "e" + str(exponent)

def str_to_man_exp(x, base=10):
    """Helper function for from_str."""
    x = x.lower().rstrip('l')
    # Verify that the input is a valid float literal
    float(x)
    # Split into mantissa, exponent
    parts = x.split('e')
    if len(parts) == 1:
        exp = 0
    else: # == 2
        x = parts[0]
        exp = int(parts[1])
    # Look for radix point in mantissa
    parts = x.split('.')
    if len(parts) == 2:
        a, b = parts[0], parts[1].rstrip('0')
        exp -= len(b)
        x = a + b
    x = MPZ(int(x, base))
    return x, exp

special_str = {'inf':finf, '+inf':finf, '-inf':fninf, 'nan':fnan}

def from_str(x, prec, rnd=round_fast):
    """Create a raw mpf from a decimal literal, rounding in the
    specified direction if the input number cannot be represented
    exactly as a binary floating-point number with the given number of
    bits. The literal syntax accepted is the same as for Python
    floats.

    TODO: the rounding does not work properly for large exponents.
    """
    x = x.lower().strip()
    if x in special_str:
        return special_str[x]

    if '/' in x:
        p, q = x.split('/')
        p, q = p.rstrip('l'), q.rstrip('l')
        return from_rational(int(p), int(q), prec, rnd)

    man, exp = str_to_man_exp(x, base=10)

    # XXX: appropriate cutoffs & track direction
    # note no factors of 5
    if abs(exp) > 400:
        s = from_int(man, prec+10)
        s = mpf_mul(s, mpf_pow_int(ften, exp, prec+10), prec, rnd)
    else:
        if exp >= 0:
            s = from_int(man * 10**exp, prec, rnd)
        else:
            s = from_rational(man, 10**-exp, prec, rnd)
    return s

# Binary string conversion. These are currently mainly used for debugging
# and could use some improvement in the future

def from_bstr(x):
    man, exp = str_to_man_exp(x, base=2)
    man = MPZ(man)
    sign = 0
    if man < 0:
        man = -man
        sign = 1
    bc = bitcount(man)
    return normalize(sign, man, exp, bc, bc, round_floor)

def to_bstr(x):
    sign, man, exp, bc = x
    return ['','-'][sign] + numeral(man, size=bitcount(man), base=2) + ("e%i" % exp)


#----------------------------------------------------------------------------#
#                                Square roots                                #
#----------------------------------------------------------------------------#


def mpf_sqrt(s, prec, rnd=round_fast):
    """
    Compute the square root of a nonnegative mpf value. The
    result is correctly rounded.
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult("square root of a negative number")
    if not man:
        return s
    if exp & 1:
        exp -= 1
        man <<= 1
        bc += 1
    elif man == 1:
        return normalize1(sign, man, exp//2, bc, prec, rnd)
    shift = max(4, 2*prec-bc+4)
    shift += shift & 1
    if rnd in 'fd':
        man = isqrt(man<<shift)
    else:
        man, rem = sqrtrem(man<<shift)
        # Perturb up
        if rem:
            man = (man<<1)+1
            shift += 2
    return from_man_exp(man, (exp-shift)//2, prec, rnd)

def mpf_hypot(x, y, prec, rnd=round_fast):
    """Compute the Euclidean norm sqrt(x**2 + y**2) of two raw mpfs
    x and y."""
    if y == fzero: return mpf_abs(x, prec, rnd)
    if x == fzero: return mpf_abs(y, prec, rnd)
    hypot2 = mpf_add(mpf_mul(x,x), mpf_mul(y,y), prec+4)
    return mpf_sqrt(hypot2, prec, rnd)


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as ext_lib
        mpf_add = ext_lib.mpf_add
        mpf_sub = ext_lib.mpf_sub
        mpf_mul = ext_lib.mpf_mul
        mpf_div = ext_lib.mpf_div
        mpf_sqrt = ext_lib.mpf_sqrt
    except ImportError:
        pass
<|endoftext|>"""
Computational functions for interval arithmetic.

"""

from .backend import xrange

from .libmpf import (
    ComplexResult,
    round_down, round_up, round_floor, round_ceiling, round_nearest,
    prec_to_dps, repr_dps, dps_to_prec,
    bitcount,
    from_float,
    fnan, finf, fninf, fzero, fhalf, fone, fnone,
    mpf_sign, mpf_lt, mpf_le, mpf_gt, mpf_ge, mpf_eq, mpf_cmp,
    mpf_min_max,
    mpf_floor, from_int, to_int, to_str, from_str,
    mpf_abs, mpf_neg, mpf_pos, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_shift, mpf_pow_int,
    from_man_exp, MPZ_ONE)

from .libelefun import (
    mpf_log, mpf_exp, mpf_sqrt, mpf_atan, mpf_atan2,
    mpf_pi, mod_pi2, mpf_cos_sin
)

from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma

def mpi_str(s, prec):
    sa, sb = s
    dps = prec_to_dps(prec) + 5
    return "[%s, %s]" % (to_str(sa, dps), to_str(sb, dps))
    #dps = prec_to_dps(prec)
    #m = mpi_mid(s, prec)
    #d = mpf_shift(mpi_delta(s, 20), -1)
    #return "%s +/- %s" % (to_str(m, dps), to_str(d, 3))

mpi_zero = (fzero, fzero)
mpi_one = (fone, fone)

def mpi_eq(s, t):
    return s == t

def mpi_ne(s, t):
    return s != t

def mpi_lt(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_lt(sb, ta): return True
    if mpf_ge(sa, tb): return False
    return None

def mpi_le(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_le(sb, ta): return True
    if mpf_gt(sa, tb): return False
    return None

def mpi_gt(s, t): return mpi_lt(t, s)
def mpi_ge(s, t): return mpi_le(t, s)

def mpi_add(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_add(sa, ta, prec, round_floor)
    b = mpf_add(sb, tb, prec, round_ceiling)
    if a == fnan: a = fninf
    if b == fnan: b = finf
    return a, b

def mpi_sub(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_sub(sa, tb, prec, round_floor)
    b = mpf_sub(sb, ta, prec, round_ceiling)
    if a == fnan: a = fninf
    if b == fnan: b = finf
    return a, b

def mpi_delta(s, prec):
    sa, sb = s
    return mpf_sub(sb, sa, prec, round_up)

def mpi_mid(s, prec):
    sa, sb = s
    return mpf_shift(mpf_add(sa, sb, prec, round_nearest), -1)

def mpi_pos(s, prec):
    sa, sb = s
    a = mpf_pos(sa, prec, round_floor)
    b = mpf_pos(sb, prec, round_ceiling)
    return a, b

def mpi_neg(s, prec=0):
    sa, sb = s
    a = mpf_neg(sb, prec, round_floor)
    b = mpf_neg(sa, prec, round_ceiling)
    return a, b

def mpi_abs(s, prec=0):
    sa, sb = s
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    # Both points nonnegative?
    if sas >= 0:
        a = mpf_pos(sa, prec, round_floor)
        b = mpf_pos(sb, prec, round_ceiling)
    # Upper point nonnegative?
    elif sbs >= 0:
        a = fzero
        negsa = mpf_neg(sa)
        if mpf_lt(negsa, sb):
            b = mpf_pos(sb, prec, round_ceiling)
        else:
            b = mpf_pos(negsa, prec, round_ceiling)
    # Both negative?
    else:
        a = mpf_neg(sb, prec, round_floor)
        b = mpf_neg(sa, prec, round_ceiling)
    return a, b

# TODO: optimize
def mpi_mul_mpf(s, t, prec):
    return mpi_mul(s, (t, t), prec)

def mpi_div_mpf(s, t, prec):
    return mpi_div(s, (t, t), prec)

def mpi_mul(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    if sas == sbs == 0:
        # Should maybe be undefined
        if ta == fninf or tb == finf:
            return fninf, finf
        return fzero, fzero
    if tas == tbs == 0:
        # Should maybe be undefined
        if sa == fninf or sb == finf:
            return fninf, finf
        return fzero, fzero
    if sas >= 0:
        # positive * positive
        if tas >= 0:
            a = mpf_mul(sa, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # positive * negative
        elif tbs <= 0:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sa, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # positive * both signs
        else:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    elif sbs <= 0:
        # negative * positive
        if tas >= 0:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sb, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # negative * negative
        elif tbs <= 0:
            a = mpf_mul(sb, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # negative * both signs
        else:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    else:
        # General case: perform all cross-multiplications and compare
        # Since the multiplications can be done exactly, we need only
        # do 4 (instead of 8: two for each rounding mode)
        cases = [mpf_mul(sa, ta), mpf_mul(sa, tb), mpf_mul(sb, ta), mpf_mul(sb, tb)]
        if fnan in cases:
            a, b = (fninf, finf)
        else:
            a, b = mpf_min_max(cases)
            a = mpf_pos(a, prec, round_floor)
            b = mpf_pos(b, prec, round_ceiling)
    return a, b

def mpi_square(s, prec=0):
    sa, sb = s
    if mpf_ge(sa, fzero):
        a = mpf_mul(sa, sa, prec, round_floor)
        b = mpf_mul(sb, sb, prec, round_ceiling)
    elif mpf_le(sb, fzero):
        a = mpf_mul(sb, sb, prec, round_floor)
        b = mpf_mul(sa, sa, prec, round_ceiling)
    else:
        sa = mpf_neg(sa)
        sa, sb = mpf_min_max([sa, sb])
        a = fzero
        b = mpf_mul(sb, sb, prec, round_ceiling)
    return a, b

def mpi_div(s, t, prec):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    # 0 / X
    if sas == sbs == 0:
        # 0 / <interval containing 0>
        if (tas < 0 and tbs > 0) or (tas == 0 or tbs == 0):
            return fninf, finf
        return fzero, fzero
    # Denominator contains both negative and positive numbers;
    # this should properly be a multi-interval, but the closest
    # match is the entire (extended) real line
    if tas < 0 and tbs > 0:
        return fninf, finf
    # Assume denominator to be nonnegative
    if tas < 0:
        return mpi_div(mpi_neg(s), mpi_neg(t), prec)
    # Division by zero
    # XXX: make sure all results make sense
    if tas == 0:
        # Numerator contains both signs?
        if sas < 0 and sbs > 0:
            return fninf, finf
        if tas == tbs:
            return fninf, finf
        # Numerator positive?
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = finf
        if sbs <= 0:
            a = fninf
            b = mpf_div(sb, tb, prec, round_ceiling)
    # Division with positive denominator
    # We still have to handle nans resulting from inf/0 or inf/inf
    else:
        # Nonnegative numerator
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = mpf_div(sb, ta, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # Nonpositive numerator
        elif sbs <= 0:
            a = mpf_div(sa, ta, prec, round_floor)
            b = mpf_div(sb, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # Numerator contains both signs?
        else:
            a = mpf_div(sa, ta, prec, round_floor)
            b = mpf_div(sb, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    return a, b

def mpi_pi(prec):
    a = mpf_pi(prec, round_floor)
    b = mpf_pi(prec, round_ceiling)
    return a, b

def mpi_exp(s, prec):
    sa, sb = s
    # exp is monotonic
    a = mpf_exp(sa, prec, round_floor)
    b = mpf_exp(sb, prec, round_ceiling)
    return a, b

def mpi_log(s, prec):
    sa, sb = s
    # log is monotonic
    a = mpf_log(sa, prec, round_floor)
    b = mpf_log(sb, prec, round_ceiling)
    return a, b

def mpi_sqrt(s, prec):
    sa, sb = s
    # sqrt is monotonic
    a = mpf_sqrt(sa, prec, round_floor)
    b = mpf_sqrt(sb, prec, round_ceiling)
    return a, b

def mpi_atan(s, prec):
    sa, sb = s
    a = mpf_atan(sa, prec, round_floor)
    b = mpf_atan(sb, prec, round_ceiling)
    return a, b

def mpi_pow_int(s, n, prec):
    sa, sb = s
    if n < 0:
        return mpi_div((fone, fone), mpi_pow_int(s, -n, prec+20), prec)
    if n == 0:
        return (fone, fone)
    if n == 1:
        return s
    if n == 2:
        return mpi_square(s, prec)
    # Odd -- signs are preserved
    if n & 1:
        a = mpf_pow_int(sa, n, prec, round_floor)
        b = mpf_pow_int(sb, n, prec, round_ceiling)
    # Even -- important to ensure positivity
    else:
        sas = mpf_sign(sa)
        sbs = mpf_sign(sb)
        # Nonnegative?
        if sas >= 0:
            a = mpf_pow_int(sa, n, prec, round_floor)
            b = mpf_pow_int(sb, n, prec, round_ceiling)
        # Nonpositive?
        elif sbs <= 0:
            a = mpf_pow_int(sb, n, prec, round_floor)
            b = mpf_pow_int(sa, n, prec, round_ceiling)
        # Mixed signs?
        else:
            a = fzero
            # max(-a,b)**n
            sa = mpf_neg(sa)
            if mpf_ge(sa, sb):
                b = mpf_pow_int(sa, n, prec, round_ceiling)
            else:
                b = mpf_pow_int(sb, n, prec, round_ceiling)
    return a, b

def mpi_pow(s, t, prec):
    ta, tb = t
    if ta == tb and ta not in (finf, fninf):
        if ta == from_int(to_int(ta)):
            return mpi_pow_int(s, to_int(ta), prec)
        if ta == fhalf:
            return mpi_sqrt(s, prec)
    u = mpi_log(s, prec + 20)
    v = mpi_mul(u, t, prec + 20)
    return mpi_exp(v, prec)

def MIN(x, y):
    if mpf_le(x, y):
        return x
    return y

def MAX(x, y):
    if mpf_ge(x, y):
        return x
    return y

def cos_sin_quadrant(x, wp):
    sign, man, exp, bc = x
    if x == fzero:
        return fone, fzero, 0
    # TODO: combine evaluation code to avoid duplicate modulo
    c, s = mpf_cos_sin(x, wp)
    t, n, wp_ = mod_pi2(man, exp, exp+bc, 15)
    if sign:
        n = -1-n
    return c, s, n

def mpi_cos_sin(x, prec):
    a, b = x
    if a == b == fzero:
        return (fone, fone), (fzero, fzero)
    # Guaranteed to contain both -1 and 1
    if (finf in x) or (fninf in x):
        return (fnone, fone), (fnone, fone)
    wp = prec + 20
    ca, sa, na = cos_sin_quadrant(a, wp)
    cb, sb, nb = cos_sin_quadrant(b, wp)
    ca, cb = mpf_min_max([ca, cb])
    sa, sb = mpf_min_max([sa, sb])
    # Both functions are monotonic within one quadrant
    if na == nb:
        pass
    # Guaranteed to contain both -1 and 1
    elif nb - na >= 4:
        return (fnone, fone), (fnone, fone)
    else:
        # cos has maximum between a and b
        if na//4 != nb//4:
            cb = fone
        # cos has minimum
        if (na-2)//4 != (nb-2)//4:
            ca = fnone
        # sin has maximum
        if (na-1)//4 != (nb-1)//4:
            sb = fone
        # sin has minimum
        if (na-3)//4 != (nb-3)//4:
            sa = fnone
    # Perturb to force interval rounding
    more = from_man_exp((MPZ_ONE<<wp) + (MPZ_ONE<<10), -wp)
    less = from_man_exp((MPZ_ONE<<wp) - (MPZ_ONE<<10), -wp)
    def finalize(v, rounding):
        if bool(v[0]) == (rounding == round_floor):
            p = more
        else:
            p = less
        v = mpf_mul(v, p, prec, rounding)
        sign, man, exp, bc = v
        if exp+bc >= 1:
            if sign:
                return fnone
            return fone
        return v
    ca = finalize(ca, round_floor)
    cb = finalize(cb, round_ceiling)
    sa = finalize(sa, round_floor)
    sb = finalize(sb, round_ceiling)
    return (ca,cb), (sa,sb)

def mpi_cos(x, prec):
    return mpi_cos_sin(x, prec)[0]

def mpi_sin(x, prec):
    return mpi_cos_sin(x, prec)[1]

def mpi_tan(x, prec):
    cos, sin = mpi_cos_sin(x, prec+20)
    return mpi_div(sin, cos, prec)

def mpi_cot(x, prec):
    cos, sin = mpi_cos_sin(x, prec+20)
    return mpi_div(cos, sin, prec)

def mpi_from_str_a_b(x, y, percent, prec):
    wp = prec + 20
    xa = from_str(x, wp, round_floor)
    xb = from_str(x, wp, round_ceiling)
    #ya = from_str(y, wp, round_floor)
    y = from_str(y, wp, round_ceiling)
    assert mpf_ge(y, fzero)
    if percent:
        y = mpf_mul(MAX(mpf_abs(xa), mpf_abs(xb)), y, wp, round_ceiling)
        y = mpf_div(y, from_int(100), wp, round_ceiling)
    a = mpf_sub(xa, y, prec, round_floor)
    b = mpf_add(xb, y, prec, round_ceiling)
    return a, b

def mpi_from_str(s, prec):
    """
    Parse an interval number given as a string.

    Allowed forms are

    "-1.23e-27"
        Any single decimal floating-point literal.
    "a +- b"  or  "a (b)"
        a is the midpoint of the interval and b is the half-width
    "a +- b%"  or  "a (b%)"
        a is the midpoint of the interval and the half-width
        is b percent of a (`a \times b / 100`).
    "[a, b]"
        The interval indicated directly.
    "x[y,z]e"
        x are shared digits, y and z are unequal digits, e is the exponent.

    """
    e = ValueError("Improperly formed interval number '%s'" % s)
    s = s.replace(" ", "")
    wp = prec + 20
    if "+-" in s:
        x, y = s.split("+-")
        return mpi_from_str_a_b(x, y, False, prec)
    # case 2
    elif "(" in s:
        # Don't confuse with a complex number (x,y)
        if s[0] == "(" or ")" not in s:
            raise e
        s = s.replace(")", "")
        percent = False
        if "%" in s:
            if s[-1] != "%":
                raise e
            percent = True
            s = s.replace("%", "")
        x, y = s.split("(")
        return mpi_from_str_a_b(x, y, percent, prec)
    elif "," in s:
        if ('[' not in s) or (']' not in s):
            raise e
        if s[0] == '[':
            # case 3
            s = s.replace("[", "")
            s = s.replace("]", "")
            a, b = s.split(",")
            a = from_str(a, prec, round_floor)
            b = from_str(b, prec, round_ceiling)
            return a, b
        else:
            # case 4
            x, y = s.split('[')
            y, z = y.split(',')
            if 'e' in s:
                z, e = z.split(']')
            else:
                z, e = z.rstrip(']'), ''
            a = from_str(x+y+e, prec, round_floor)
            b = from_str(x+z+e, prec, round_ceiling)
            return a, b
    else:
        a = from_str(s, prec, round_floor)
        b = from_str(s, prec, round_ceiling)
        return a, b

def mpi_to_str(x, dps, use_spaces=True, brackets='[]', mode='brackets', error_dps=4, **kwargs):
    """
    Convert a mpi interval to a string.

    **Arguments**

    *dps*
        decimal places to use for printing
    *use_spaces*
        use spaces for more readable output, defaults to true
    *brackets*
        pair of strings (or two-character string) giving left and right brackets
    *mode*
        mode of display: 'plusminus', 'percent', 'brackets' (default) or 'diff'
    *error_dps*
        limit the error to *error_dps* digits (mode 'plusminus and 'percent')

    Additional keyword arguments are forwarded to the mpf-to-string conversion
    for the components of the output.

    **Examples**

        >>> from mpmath import mpi, mp
        >>> mp.dps = 30
        >>> x = mpi(1, 2)._mpi_
        >>> mpi_to_str(x, 2, mode='plusminus')
        '1.5 +- 0.5'
        >>> mpi_to_str(x, 2, mode='percent')
        '1.5 (33.33%)'
        >>> mpi_to_str(x, 2, mode='brackets')
        '[1.0, 2.0]'
        >>> mpi_to_str(x, 2, mode='brackets' , brackets=('<', '>'))
        '<1.0, 2.0>'
        >>> x = mpi('5.2582327113062393041', '5.2582327113062749951')._mpi_
        >>> mpi_to_str(x, 15, mode='diff')
        '5.2582327113062[4, 7]'
        >>> mpi_to_str(mpi(0)._mpi_, 2, mode='percent')
        '0.0 (0.0%)'

    """
    prec = dps_to_prec(dps)
    wp = prec + 20
    a, b = x
    mid = mpi_mid(x, prec)
    delta = mpi_delta(x, prec)
    a_str = to_str(a, dps, **kwargs)
    b_str = to_str(b, dps, **kwargs)
    mid_str = to_str(mid, dps, **kwargs)
    sp = ""
    if use_spaces:
        sp = " "
    br1, br2 = brackets
    if mode == 'plusminus':
        delta_str = to_str(mpf_shift(delta,-1), dps, **kwargs)
        s = mid_str + sp + "+-" + sp + delta_str
    elif mode == 'percent':
        if mid == fzero:
            p = fzero
        else:
            # p = 100 * delta(x) / (2*mid(x))
            p = mpf_mul(delta, from_int(100))
            p = mpf_div(p, mpf_mul(mid, from_int(2)), wp)
        s = mid_str + sp + "(" + to_str(p, error_dps) + "%)"
    elif mode == 'brackets':
        s = br1 + a_str + "," + sp + b_str + br2
    elif mode == 'diff':
        # use more digits if str(x.a) and str(x.b) are equal
        if a_str == b_str:
            a_str = to_str(a, dps+3, **kwargs)
            b_str = to_str(b, dps+3, **kwargs)
        # separate mantissa and exponent
        a = a_str.split('e')
        if len(a) == 1:
            a.append('')
        b = b_str.split('e')
        if len(b) == 1:
            b.append('')
        if a[1] == b[1]:
            if a[0] != b[0]:
                for i in xrange(len(a[0]) + 1):
                    if a[0][i] != b[0][i]:
                        break
                s = (a[0][:i] + br1 + a[0][i:] + ',' + sp + b[0][i:] + br2
                     + 'e'*min(len(a[1]), 1) + a[1])
            else: # no difference
                s = a[0] + br1 + br2 + 'e'*min(len(a[1]), 1) + a[1]
        else:
            s = br1 + 'e'.join(a) + ',' + sp + 'e'.join(b) + br2
    else:
        raise ValueError("'%s' is unknown mode for printing mpi" % mode)
    return s

def mpci_add(x, y, prec):
    a, b = x
    c, d = y
    return mpi_add(a, c, prec), mpi_add(b, d, prec)

def mpci_sub(x, y, prec):
    a, b = x
    c, d = y
    return mpi_sub(a, c, prec), mpi_sub(b, d, prec)

def mpci_neg(x, prec=0):
    a, b = x
    return mpi_neg(a, prec), mpi_neg(b, prec)

def mpci_pos(x, prec):
    a, b = x
    return mpi_pos(a, prec), mpi_pos(b, prec)

def mpci_mul(x, y, prec):
    # TODO: optimize for real/imag cases
    a, b = x
    c, d = y
    r1 = mpi_mul(a,c)
    r2 = mpi_mul(b,d)
    re = mpi_sub(r1,r2,prec)
    i1 = mpi_mul(a,d)
    i2 = mpi_mul(b,c)
    im = mpi_add(i1,i2,prec)
    return re, im

def mpci_div(x, y, prec):
    # TODO: optimize for real/imag cases
    a, b = x
    c, d = y
    wp = prec+20
    m1 = mpi_square(c)
    m2 = mpi_square(d)
    m = mpi_add(m1,m2,wp)
    re = mpi_add(mpi_mul(a,c), mpi_mul(b,d), wp)
    im = mpi_sub(mpi_mul(b,c), mpi_mul(a,d), wp)
    re = mpi_div(re, m, prec)
    im = mpi_div(im, m, prec)
    return re, im

def mpci_exp(x, prec):
    a, b = x
    wp = prec+20
    r = mpi_exp(a, wp)
    c, s = mpi_cos_sin(b, wp)
    a = mpi_mul(r, c, prec)
    b = mpi_mul(r, s, prec)
    return a, b

def mpi_shift(x, n):
    a, b = x
    return mpf_shift(a,n), mpf_shift(b,n)

def mpi_cosh_sinh(x, prec):
    # TODO: accuracy for small x
    wp = prec+20
    e1 = mpi_exp(x, wp)
    e2 = mpi_div(mpi_one, e1, wp)
    c = mpi_add(e1, e2, prec)
    s = mpi_sub(e1, e2, prec)
    c = mpi_shift(c, -1)
    s = mpi_shift(s, -1)
    return c, s

def mpci_cos(x, prec):
    a, b = x
    wp = prec+10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(c, ch, prec)
    im = mpi_mul(s, sh, prec)
    return re, mpi_neg(im)

def mpci_sin(x, prec):
    a, b = x
    wp = prec+10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(s, ch, prec)
    im = mpi_mul(c, sh, prec)
    return re, im

def mpci_abs(x, prec):
    a, b = x
    if a == mpi_zero:
        return mpi_abs(b)
    if b == mpi_zero:
        return mpi_abs(a)
    # Important: nonnegative
    a = mpi_square(a)
    b = mpi_square(b)
    t = mpi_add(a, b, prec+20)
    return mpi_sqrt(t, prec)

def mpi_atan2(y, x, prec):
    ya, yb = y
    xa, xb = x
    # Constrained to the real line
    if ya == yb == fzero:
        if mpf_ge(xa, fzero):
            return mpi_zero
        return mpi_pi(prec)
    # Right half-plane
    if mpf_ge(xa, fzero):
        if mpf_ge(ya, fzero):
            a = mpf_atan2(ya, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xa, prec, round_floor)
        if mpf_ge(yb, fzero):
            b = mpf_atan2(yb, xa, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    # Upper half-plane
    elif mpf_ge(ya, fzero):
        b = mpf_atan2(ya, xa, prec, round_ceiling)
        if mpf_le(xb, fzero):
            a = mpf_atan2(yb, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xb, prec, round_floor)
    # Lower half-plane
    elif mpf_le(yb, fzero):
        a = mpf_atan2(yb, xa, prec, round_floor)
        if mpf_le(xb, fzero):
            b = mpf_atan2(ya, xb, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    # Covering the origin
    else:
        b = mpf_pi(prec, round_ceiling)
        a = mpf_neg(b)
    return a, b

def mpci_arg(z, prec):
    x, y = z
    return mpi_atan2(y, x, prec)

def mpci_log(z, prec):
    x, y = z
    re = mpi_log(mpci_abs(z, prec+20), prec)
    im = mpci_arg(z, prec)
    return re, im

def mpci_pow(x, y, prec):
    # TODO: recognize/speed up real cases, integer y
    yre, yim = y
    if yim == mpi_zero:
        ya, yb = yre
        if ya == yb:
            sign, man, exp, bc = yb
            if man and exp >= 0:
                return mpci_pow_int(x, (-1)**sign * int(man<<exp), prec)
            # x^0
            if yb == fzero:
                return mpci_pow_int(x, 0, prec)
    wp = prec+20
    return mpci_exp(mpci_mul(y, mpci_log(x, wp), wp), prec)

def mpci_square(x, prec):
    a, b = x
    # (a+bi)^2 = (a^2-b^2) + 2abi
    re = mpi_sub(mpi_square(a), mpi_square(b), prec)
    im = mpi_mul(a, b, prec)
    im = mpi_shift(im, 1)
    return re, im

def mpci_pow_int(x, n, prec):
    if n < 0:
        return mpci_div((mpi_one,mpi_zero), mpci_pow_int(x, -n, prec+20), prec)
    if n == 0:
        return mpi_one, mpi_zero
    if n == 1:
        return mpci_pos(x, prec)
    if n == 2:
        return mpci_square(x, prec)
    wp = prec + 20
    result = (mpi_one, mpi_zero)
    while n:
        if n & 1:
            result = mpci_mul(result, x, wp)
            n -= 1
        x = mpci_square(x, wp)
        n >>= 1
    return mpci_pos(result, prec)

gamma_min_a = from_float(1.46163214496)
gamma_min_b = from_float(1.46163214497)
gamma_min = (gamma_min_a, gamma_min_b)
gamma_mono_imag_a = from_float(-1.1)
gamma_mono_imag_b = from_float(1.1)

def mpi_overlap(x, y):
    a, b = x
    c, d = y
    if mpf_lt(d, a): return False
    if mpf_gt(c, b): return False
    return True

# type = 0 -- gamma
# type = 1 -- factorial
# type = 2 -- 1/gamma
# type = 3 -- log-gamma

def mpi_gamma(z, prec, type=0):
    a, b = z
    wp = prec+20

    if type == 1:
        return mpi_gamma(mpi_add(z, mpi_one, wp), prec, 0)

    # increasing
    if mpf_gt(a, gamma_min_b):
        if type == 0:
            c = mpf_gamma(a, prec, round_floor)
            d = mpf_gamma(b, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(b, prec, round_floor)
            d = mpf_rgamma(a, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(a, prec, round_floor)
            d = mpf_loggamma(b, prec, round_ceiling)
    # decreasing
    elif mpf_gt(a, fzero) and mpf_lt(b, gamma_min_a):
        if type == 0:
            c = mpf_gamma(b, prec, round_floor)
            d = mpf_gamma(a, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(a, prec, round_floor)
            d = mpf_rgamma(b, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(b, prec, round_floor)
            d = mpf_loggamma(a, prec, round_ceiling)
    else:
        # TODO: reflection formula
        znew = mpi_add(z, mpi_one, wp)
        if type == 0: return mpi_div(mpi_gamma(znew, prec+2, 0), z, prec)
        if type == 2: return mpi_mul(mpi_gamma(znew, prec+2, 2), z, prec)
        if type == 3: return mpi_sub(mpi_gamma(znew, prec+2, 3), mpi_log(z, prec+2), prec)
    return c, d

def mpci_gamma(z, prec, type=0):
    (a1,a2), (b1,b2) = z

    # Real case
    if b1 == b2 == fzero and (type != 3 or mpf_gt(a1,fzero)):
        return mpi_gamma(z, prec, type), mpi_zero

    # Estimate precision
    wp = prec+20
    if type != 3:
        amag = a2[2]+a2[3]
        bmag = b2[2]+b2[3]
        if a2 != fzero:
            mag = max(amag, bmag)
        else:
            mag = bmag
        an = abs(to_int(a2))
        bn = abs(to_int(b2))
        absn = max(an, bn)
        gamma_size = max(0,absn*mag)
        wp += bitcount(gamma_size)

    # Assume type != 1
    if type == 1:
        (a1,a2) = mpi_add((a1,a2), mpi_one, wp); z = (a1,a2), (b1,b2)
        type = 0

    # Avoid non-monotonic region near the negative real axis
    if mpf_lt(a1, gamma_min_b):
        if mpi_overlap((b1,b2), (gamma_mono_imag_a, gamma_mono_imag_b)):
            # TODO: reflection formula
            #if mpf_lt(a2, mpf_shift(fone,-1)):
            #    znew = mpci_sub((mpi_one,mpi_zero),z,wp)
            #    ...
            # Recurrence:
            # gamma(z) = gamma(z+1)/z
            znew = mpi_add((a1,a2), mpi_one, wp), (b1,b2)
            if type == 0: return mpci_div(mpci_gamma(znew, prec+2, 0), z, prec)
            if type == 2: return mpci_mul(mpci_gamma(znew, prec+2, 2), z, prec)
            if type == 3: return mpci_sub(mpci_gamma(znew, prec+2, 3), mpci_log(z,prec+2), prec)

    # Use monotonicity (except for a small region close to the
    # origin and near poles)
    # upper half-plane
    if mpf_ge(b1, fzero):
        minre = mpc_loggamma((a1,b2), wp, round_floor)
        maxre = mpc_loggamma((a2,b1), wp, round_ceiling)
        minim = mpc_loggamma((a1,b1), wp, round_floor)
        maxim = mpc_loggamma((a2,b2), wp, round_ceiling)
    # lower half-plane
    elif mpf_le(b2, fzero):
        minre = mpc_loggamma((a1,b1), wp, round_floor)
        maxre = mpc_loggamma((a2,b2), wp, round_ceiling)
        minim = mpc_loggamma((a2,b1), wp, round_floor)
        maxim = mpc_loggamma((a1,b2), wp, round_ceiling)
    # crosses real axis
    else:
        maxre = mpc_loggamma((a2,fzero), wp, round_ceiling)
        # stretches more into the lower half-plane
        if mpf_gt(mpf_neg(b1), b2):
            minre = mpc_loggamma((a1,b1), wp, round_ceiling)
        else:
            minre = mpc_loggamma((a1,b2), wp, round_ceiling)
        minim = mpc_loggamma((a2,b1), wp, round_floor)
        maxim = mpc_loggamma((a2,b2), wp, round_floor)

    w = (minre[0], maxre[0]), (minim[1], maxim[1])
    if type == 3:
        return mpi_pos(w[0], prec), mpi_pos(w[1], prec)
    if type == 2:
        w = mpci_neg(w)
    return mpci_exp(w, prec)

def mpi_loggamma(z, prec): return mpi_gamma(z, prec, type=3)
def mpci_loggamma(z, prec): return mpci_gamma(z, prec, type=3)

def mpi_rgamma(z, prec): return mpi_gamma(z, prec, type=2)
def mpci_rgamma(z, prec): return mpci_gamma(z, prec, type=2)

def mpi_factorial(z, prec): return mpi_gamma(z, prec, type=1)
def mpci_factorial(z, prec): return mpci_gamma(z, prec, type=1)
<|endoftext|>"""
This module implements computation of hypergeometric and related
functions. In particular, it provides code for generic summation
of hypergeometric series. Optimized versions for various special
cases are also provided.
"""

import operator
import math

from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_

from .libintmath import gcd

from .libmpf import (\
    ComplexResult, round_fast, round_nearest,
    negative_rnd, bitcount, to_fixed, from_man_exp, from_int, to_int,
    from_rational,
    fzero, fone, fnone, ftwo, finf, fninf, fnan,
    mpf_sign, mpf_add, mpf_abs, mpf_pos,
    mpf_cmp, mpf_lt, mpf_le, mpf_gt, mpf_min_max,
    mpf_perturb, mpf_neg, mpf_shift, mpf_sub, mpf_mul, mpf_div,
    sqrt_fixed, mpf_sqrt, mpf_rdiv_int, mpf_pow_int,
    to_rational,
)

from .libelefun import (\
    mpf_pi, mpf_exp, mpf_log, pi_fixed, mpf_cos_sin, mpf_cos, mpf_sin,
    mpf_sqrt, agm_fixed,
)

from .libmpc import (\
    mpc_one, mpc_sub, mpc_mul_mpf, mpc_mul, mpc_neg, complex_int_pow,
    mpc_div, mpc_add_mpf, mpc_sub_mpf,
    mpc_log, mpc_add, mpc_pos, mpc_shift,
    mpc_is_infnan, mpc_zero, mpc_sqrt, mpc_abs,
    mpc_mpf_div, mpc_square, mpc_exp
)

from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed

class NoConvergence(Exception):
    pass


#-----------------------------------------------------------------------#
#                                                                       #
#                     Generic hypergeometric series                     #
#                                                                       #
#-----------------------------------------------------------------------#

"""
TODO:

1. proper mpq parsing
2. imaginary z special-cased (also: rational, integer?)
3. more clever handling of series that don't converge because of stupid
   upwards rounding
4. checking for cancellation

"""

def make_hyp_summator(key):
    """
    Returns a function that sums a generalized hypergeometric series,
    for given parameter types (integer, rational, real, complex).

    """
    p, q, param_types, ztype = key

    pstring = "".join(param_types)
    fname = "hypsum_%i_%i_%s_%s_%s" % (p, q, pstring[:p], pstring[p:], ztype)
    #print "generating hypsum", fname

    have_complex_param = 'C' in param_types
    have_complex_arg = ztype == 'C'
    have_complex = have_complex_param or have_complex_arg

    source = []
    add = source.append

    aint = []
    arat = []
    bint = []
    brat = []
    areal = []
    breal = []
    acomplex = []
    bcomplex = []

    #add("wp = prec + 40")
    add("MAX = kwargs.get('maxterms', wp*100)")
    add("HIGH = MPZ_ONE<<epsshift")
    add("LOW = -HIGH")

    # Setup code
    add("SRE = PRE = one = (MPZ_ONE << wp)")
    if have_complex:
        add("SIM = PIM = MPZ_ZERO")

    if have_complex_arg:
        add("xsign, xm, xe, xbc = z[0]")
        add("if xsign: xm = -xm")
        add("ysign, ym, ye, ybc = z[1]")
        add("if ysign: ym = -ym")
    else:
        add("xsign, xm, xe, xbc = z")
        add("if xsign: xm = -xm")

    add("offset = xe + wp")
    add("if offset >= 0:")
    add("    ZRE = xm << offset")
    add("else:")
    add("    ZRE = xm >> (-offset)")
    if have_complex_arg:
        add("offset = ye + wp")
        add("if offset >= 0:")
        add("    ZIM = ym << offset")
        add("else:")
        add("    ZIM = ym >> (-offset)")

    for i, flag in enumerate(param_types):
        W = ["A", "B"][i >= p]
        if flag == 'Z':
            ([aint,bint][i >= p]).append(i)
            add("%sINT_%i = coeffs[%i]" % (W, i, i))
        elif flag == 'Q':
            ([arat,brat][i >= p]).append(i)
            add("%sP_%i, %sQ_%i = coeffs[%i]._mpq_" % (W, i, W, i, i))
        elif flag == 'R':
            ([areal,breal][i >= p]).append(i)
            add("xsign, xm, xe, xbc = coeffs[%i]._mpf_" % i)
            add("if xsign: xm = -xm")
            add("offset = xe + wp")
            add("if offset >= 0:")
            add("    %sREAL_%i = xm << offset" % (W, i))
            add("else:")
            add("    %sREAL_%i = xm >> (-offset)" % (W, i))
        elif flag == 'C':
            ([acomplex,bcomplex][i >= p]).append(i)
            add("__re, __im = coeffs[%i]._mpc_" % i)
            add("xsign, xm, xe, xbc = __re")
            add("if xsign: xm = -xm")
            add("ysign, ym, ye, ybc = __im")
            add("if ysign: ym = -ym")

            add("offset = xe + wp")
            add("if offset >= 0:")
            add("    %sCRE_%i = xm << offset" % (W, i))
            add("else:")
            add("    %sCRE_%i = xm >> (-offset)" % (W, i))
            add("offset = ye + wp")
            add("if offset >= 0:")
            add("    %sCIM_%i = ym << offset" % (W, i))
            add("else:")
            add("    %sCIM_%i = ym >> (-offset)" % (W, i))
        else:
            raise ValueError

    l_areal = len(areal)
    l_breal = len(breal)
    cancellable_real = min(l_areal, l_breal)
    noncancellable_real_num = areal[cancellable_real:]
    noncancellable_real_den = breal[cancellable_real:]

    # LOOP
    add("for n in xrange(1,10**8):")

    add("    if n in magnitude_check:")
    add("        p_mag = bitcount(abs(PRE))")
    if have_complex:
        add("        p_mag = max(p_mag, bitcount(abs(PIM)))")
    add("        magnitude_check[n] = wp-p_mag")

    # Real factors
    multiplier = " * ".join(["AINT_#".replace("#", str(i)) for i in aint] + \
                            ["AP_#".replace("#", str(i)) for i in arat] + \
                            ["BQ_#".replace("#", str(i)) for i in brat])

    divisor    = " * ".join(["BINT_#".replace("#", str(i)) for i in bint] + \
                            ["BP_#".replace("#", str(i)) for i in brat] + \
                            ["AQ_#".replace("#", str(i)) for i in arat] + ["n"])

    if multiplier:
        add("    mul = " + multiplier)
    add("    div = " + divisor)

    # Check for singular terms
    add("    if not div:")
    if multiplier:
        add("        if not mul:")
        add("            break")
    add("        raise ZeroDivisionError")

    # Update product
    if have_complex:

        # TODO: when there are several real parameters and just a few complex
        # (maybe just the complex argument), we only need to do about
        # half as many ops if we accumulate the real factor in a single real variable
        for k in range(cancellable_real): add("    PRE = PRE * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PRE = (PRE * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PRE = (PRE << wp) // BREAL_#".replace("#", str(i)))
        for k in range(cancellable_real): add("    PIM = PIM * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PIM = (PIM * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PIM = (PIM << wp) // BREAL_#".replace("#", str(i)))

        if multiplier:
            if have_complex_arg:
                add("    PRE, PIM = (mul*(PRE*ZRE-PIM*ZIM))//div, (mul*(PIM*ZRE+PRE*ZIM))//div")
                add("    PRE >>= wp")
                add("    PIM >>= wp")
            else:
                add("    PRE = ((mul * PRE * ZRE) >> wp) // div")
                add("    PIM = ((mul * PIM * ZRE) >> wp) // div")
        else:
            if have_complex_arg:
                add("    PRE, PIM = (PRE*ZRE-PIM*ZIM)//div, (PIM*ZRE+PRE*ZIM)//div")
                add("    PRE >>= wp")
                add("    PIM >>= wp")
            else:
                add("    PRE = ((PRE * ZRE) >> wp) // div")
                add("    PIM = ((PIM * ZRE) >> wp) // div")

        for i in acomplex:
            add("    PRE, PIM = PRE*ACRE_#-PIM*ACIM_#, PIM*ACRE_#+PRE*ACIM_#".replace("#", str(i)))
            add("    PRE >>= wp")
            add("    PIM >>= wp")

        for i in bcomplex:
            add("    mag = BCRE_#*BCRE_#+BCIM_#*BCIM_#".replace("#", str(i)))
            add("    re = PRE*BCRE_# + PIM*BCIM_#".replace("#", str(i)))
            add("    im = PIM*BCRE_# - PRE*BCIM_#".replace("#", str(i)))
            add("    PRE = (re << wp) // mag".replace("#", str(i)))
            add("    PIM = (im << wp) // mag".replace("#", str(i)))

    else:
        for k in range(cancellable_real): add("    PRE = PRE * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PRE = (PRE * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PRE = (PRE << wp) // BREAL_#".replace("#", str(i)))
        if multiplier:
            add("    PRE = ((PRE * mul * ZRE) >> wp) // div")
        else:
            add("    PRE = ((PRE * ZRE) >> wp) // div")

    # Add product to sum
    if have_complex:
        add("    SRE += PRE")
        add("    SIM += PIM")
        add("    if (HIGH > PRE > LOW) and (HIGH > PIM > LOW):")
        add("        break")
    else:
        add("    SRE += PRE")
        add("    if HIGH > PRE > LOW:")
        add("        break")

    #add("    from mpmath import nprint, log, ldexp")
    #add("    nprint([n, log(abs(PRE),2), ldexp(PRE,-wp)])")

    add("    if n > MAX:")
    add("        raise NoConvergence('Hypergeometric series converges too slowly. Try increasing maxterms.')")

    # +1 all parameters for next loop
    for i in aint:     add("    AINT_# += 1".replace("#", str(i)))
    for i in bint:     add("    BINT_# += 1".replace("#", str(i)))
    for i in arat:     add("    AP_# += AQ_#".replace("#", str(i)))
    for i in brat:     add("    BP_# += BQ_#".replace("#", str(i)))
    for i in areal:    add("    AREAL_# += one".replace("#", str(i)))
    for i in breal:    add("    BREAL_# += one".replace("#", str(i)))
    for i in acomplex: add("    ACRE_# += one".replace("#", str(i)))
    for i in bcomplex: add("    BCRE_# += one".replace("#", str(i)))

    if have_complex:
        add("a = from_man_exp(SRE, -wp, prec, 'n')")
        add("b = from_man_exp(SIM, -wp, prec, 'n')")

        add("if SRE:")
        add("    if SIM:")
        add("        magn = max(a[2]+a[3], b[2]+b[3])")
        add("    else:")
        add("        magn = a[2]+a[3]")
        add("elif SIM:")
        add("    magn = b[2]+b[3]")
        add("else:")
        add("    magn = -wp+1")

        add("return (a, b), True, magn")
    else:
        add("a = from_man_exp(SRE, -wp, prec, 'n')")

        add("if SRE:")
        add("    magn = a[2]+a[3]")
        add("else:")
        add("    magn = -wp+1")

        add("return a, False, magn")

    source = "\n".join(("    " + line) for line in source)
    source = ("def %s(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):\n" % fname) + source

    namespace = {}

    exec_(source, globals(), namespace)

    #print source
    return source, namespace[fname]


if BACKEND == 'sage':

    def make_hyp_summator(key):
        """
        Returns a function that sums a generalized hypergeometric series,
        for given parameter types (integer, rational, real, complex).
        """
        from sage.libs.mpmath.ext_main import hypsum_internal
        p, q, param_types, ztype = key
        def _hypsum(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):
            return hypsum_internal(p, q, param_types, ztype, coeffs, z,
                prec, wp, epsshift, magnitude_check, kwargs)

        return "(none)", _hypsum


#-----------------------------------------------------------------------#
#                                                                       #
#                              Error functions                          #
#                                                                       #
#-----------------------------------------------------------------------#

# TODO: mpf_erf should call mpf_erfc when appropriate (currently
#    only the converse delegation is implemented)

def mpf_erf(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fzero
        if x == finf: return fone
        if x== fninf: return fnone
        return fnan
    size = exp + bc
    lg = math.log
    # The approximation erf(x) = 1 is accurate to > x^2 * log(e,2) bits
    if size > 3 and 2*(size-1) + 0.528766 > lg(prec,2):
        if sign:
            return mpf_perturb(fnone, 0, prec, rnd)
        else:
            return mpf_perturb(fone, 1, prec, rnd)
    # erf(x) ~ 2*x/sqrt(pi) close to 0
    if size < -prec:
        # 2*x
        x = mpf_shift(x,1)
        c = mpf_sqrt(mpf_pi(prec+20), prec+20)
        # TODO: interval rounding
        return mpf_div(x, c, prec, rnd)
    wp = prec + abs(size) + 25
    # Taylor series for erf, fixed-point summation
    t = abs(to_fixed(x, wp))
    t2 = (t*t) >> wp
    s, term, k = t, 12345, 1
    while term:
        t = ((t * t2) >> wp) // k
        term = t // (2*k+1)
        if k & 1:
            s -= term
        else:
            s += term
        k += 1
    s = (s << (wp+1)) // sqrt_fixed(pi_fixed(wp), wp)
    if sign:
        s = -s
    return from_man_exp(s, -wp, prec, rnd)

# If possible, we use the asymptotic series for erfc.
# This is an alternating divergent asymptotic series, so
# the error is at most equal to the first omitted term.
# Here we check if the smallest term is small enough
# for a given x and precision
def erfc_check_series(x, prec):
    n = to_int(x)
    if n**2 * 1.44 > prec:
        return True
    return False

def mpf_erfc(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fone
        if x == finf: return fzero
        if x == fninf: return ftwo
        return fnan
    wp = prec + 20
    mag = bc+exp
    # Preserve full accuracy when exponent grows huge
    wp += max(0, 2*mag)
    regular_erf = sign or mag < 2
    if regular_erf or not erfc_check_series(x, wp):
        if regular_erf:
            return mpf_sub(fone, mpf_erf(x, prec+10, negative_rnd[rnd]), prec, rnd)
        # 1-erf(x) ~ exp(-x^2), increase prec to deal with cancellation
        n = to_int(x)+1
        return mpf_sub(fone, mpf_erf(x, prec + int(n**2*1.44) + 10), prec, rnd)
    s = term = MPZ_ONE << wp
    term_prev = 0
    t = (2 * to_fixed(x, wp) ** 2) >> wp
    k = 1
    while 1:
        term = ((term * (2*k - 1)) << wp) // t
        if k > 4 and term > term_prev or not term:
            break
        if k & 1:
            s -= term
        else:
            s += term
        term_prev = term
        #print k, to_str(from_man_exp(term, -wp, 50), 10)
        k += 1
    s = (s << wp) // sqrt_fixed(pi_fixed(wp), wp)
    s = from_man_exp(s, -wp, wp)
    z = mpf_exp(mpf_neg(mpf_mul(x,x,wp),wp),wp)
    y = mpf_div(mpf_mul(z, s, wp), x, prec, rnd)
    return y


#-----------------------------------------------------------------------#
#                                                                       #
#                         Exponential integrals                         #
#                                                                       #
#-----------------------------------------------------------------------#

def ei_taylor(x, prec):
    s = t = x
    k = 2
    while t:
        t = ((t*x) >> prec) // k
        s += t // k
        k += 1
    return s

def complex_ei_taylor(zre, zim, prec):
    _abs = abs
    sre = tre = zre
    sim = tim = zim
    k = 2
    while _abs(tre) + _abs(tim) > 5:
        tre, tim = ((tre*zre-tim*zim)//k)>>prec, ((tre*zim+tim*zre)//k)>>prec
        sre += tre // k
        sim += tim // k
        k += 1
    return sre, sim

def ei_asymptotic(x, prec):
    one = MPZ_ONE << prec
    x = t = ((one << prec) // x)
    s = one + x
    k = 2
    while t:
        t = (k*t*x) >> prec
        s += t
        k += 1
    return s

def complex_ei_asymptotic(zre, zim, prec):
    _abs = abs
    one = MPZ_ONE << prec
    M = (zim*zim + zre*zre) >> prec
    # 1 / z
    xre = tre = (zre << prec) // M
    xim = tim = ((-zim) << prec) // M
    sre = one + xre
    sim = xim
    k = 2
    while _abs(tre) + _abs(tim) > 1000:
        #print tre, tim
        tre, tim = ((tre*xre-tim*xim)*k)>>prec, ((tre*xim+tim*xre)*k)>>prec
        sre += tre
        sim += tim
        k += 1
        if k > prec:
            raise NoConvergence
    return sre, sim

def mpf_ei(x, prec, rnd=round_fast, e1=False):
    if e1:
        x = mpf_neg(x)
    sign, man, exp, bc = x
    if e1 and not sign:
        if x == fzero:
            return finf
        raise ComplexResult("E1(x) for x < 0")
    if man:
        xabs = 0, man, exp, bc
        xmag = exp+bc
        wp = prec + 20
        can_use_asymp = xmag > wp
        if not can_use_asymp:
            if exp >= 0:
                xabsint = man << exp
            else:
                xabsint = man >> (-exp)
            can_use_asymp = xabsint > int(wp*0.693) + 10
        if can_use_asymp:
            if xmag > wp:
                v = fone
            else:
                v = from_man_exp(ei_asymptotic(to_fixed(x, wp), wp), -wp)
            v = mpf_mul(v, mpf_exp(x, wp), wp)
            v = mpf_div(v, x, prec, rnd)
        else:
            wp += 2*int(to_int(xabs))
            u = to_fixed(x, wp)
            v = ei_taylor(u, wp) + euler_fixed(wp)
            t1 = from_man_exp(v,-wp)
            t2 = mpf_log(xabs,wp)
            v = mpf_add(t1, t2, prec, rnd)
    else:
        if x == fzero: v = fninf
        elif x == finf: v = finf
        elif x == fninf: v = fzero
        else: v = fnan
    if e1:
        v = mpf_neg(v)
    return v

def mpc_ei(z, prec, rnd=round_fast, e1=False):
    if e1:
        z = mpc_neg(z)
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero:
        if e1:
            x = mpf_neg(mpf_ei(a, prec, rnd))
            if not asign:
                y = mpf_neg(mpf_pi(prec, rnd))
            else:
                y = fzero
            return x, y
        else:
            return mpf_ei(a, prec, rnd), fzero
    if a != fzero:
        if not aman or not bman:
            return (fnan, fnan)
    wp = prec + 40
    amag = aexp+abc
    bmag = bexp+bbc
    zmag = max(amag, bmag)
    can_use_asymp = zmag > wp
    if not can_use_asymp:
        zabsint = abs(to_int(a)) + abs(to_int(b))
        can_use_asymp = zabsint > int(wp*0.693) + 20
    try:
        if can_use_asymp:
            if zmag > wp:
                v = fone, fzero
            else:
                zre = to_fixed(a, wp)
                zim = to_fixed(b, wp)
                vre, vim = complex_ei_asymptotic(zre, zim, wp)
                v = from_man_exp(vre, -wp), from_man_exp(vim, -wp)
            v = mpc_mul(v, mpc_exp(z, wp), wp)
            v = mpc_div(v, z, wp)
            if e1:
                v = mpc_neg(v, prec, rnd)
            else:
                x, y = v
                if bsign:
                    v = mpf_pos(x, prec, rnd), mpf_sub(y, mpf_pi(wp), prec, rnd)
                else:
                    v = mpf_pos(x, prec, rnd), mpf_add(y, mpf_pi(wp), prec, rnd)
            return v
    except NoConvergence:
        pass
    #wp += 2*max(0,zmag)
    wp += 2*int(to_int(mpc_abs(z, 5)))
    zre = to_fixed(a, wp)
    zim = to_fixed(b, wp)
    vre, vim = complex_ei_taylor(zre, zim, wp)
    vre += euler_fixed(wp)
    v = from_man_exp(vre,-wp), from_man_exp(vim,-wp)
    if e1:
        u = mpc_log(mpc_neg(z),wp)
    else:
        u = mpc_log(z,wp)
    v = mpc_add(v, u, prec, rnd)
    if e1:
        v = mpc_neg(v)
    return v

def mpf_e1(x, prec, rnd=round_fast):
    return mpf_ei(x, prec, rnd, True)

def mpc_e1(x, prec, rnd=round_fast):
    return mpc_ei(x, prec, rnd, True)

def mpf_expint(n, x, prec, rnd=round_fast, gamma=False):
    """
    E_n(x), n an integer, x real

    With gamma=True, computes Gamma(n,x)   (upper incomplete gamma function)

    Returns (real, None) if real, otherwise (real, imag)
    The imaginary part is an optional branch cut term

    """
    sign, man, exp, bc = x
    if not man:
        if gamma:
            if x == fzero:
                # Actually gamma function pole
                if n <= 0:
                    return finf, None
                return mpf_gamma_int(n, prec, rnd), None
            if x == finf:
                return fzero, None
            # TODO: could return finite imaginary value at -inf
            return fnan, fnan
        else:
            if x == fzero:
                if n > 1:
                    return from_rational(1, n-1, prec, rnd), None
                else:
                    return finf, None
            if x == finf:
                return fzero, None
            return fnan, fnan
    n_orig = n
    if gamma:
        n = 1-n
    wp = prec + 20
    xmag = exp + bc
    # Beware of near-poles
    if xmag < -10:
        raise NotImplementedError
    nmag = bitcount(abs(n))
    have_imag = n > 0 and sign
    negx = mpf_neg(x)
    # Skip series if direct convergence
    if n == 0 or 2*nmag - xmag < -wp:
        if gamma:
            v = mpf_exp(negx, wp)
            re = mpf_mul(v, mpf_pow_int(x, n_orig-1, wp), prec, rnd)
        else:
            v = mpf_exp(negx, wp)
            re = mpf_div(v, x, prec, rnd)
    else:
        # Finite number of terms, or...
        can_use_asymptotic_series = -3*wp < n <= 0
        # ...large enough?
        if not can_use_asymptotic_series:
            xi = abs(to_int(x))
            m = min(max(1, xi-n), 2*wp)
            siz = -n*nmag + (m+n)*bitcount(abs(m+n)) - m*xmag - (144*m//100)
            tol = -wp-10
            can_use_asymptotic_series = siz < tol
        if can_use_asymptotic_series:
            r = ((-MPZ_ONE) << (wp+wp)) // to_fixed(x, wp)
            m = n
            t = r*m
            s = MPZ_ONE << wp
            while m and t:
                s += t
                m += 1
                t = (m*r*t) >> wp
            v = mpf_exp(negx, wp)
            if gamma:
                # ~ exp(-x) * x^(n-1) * (1 + ...)
                v = mpf_mul(v, mpf_pow_int(x, n_orig-1, wp), wp)
            else:
                # ~ exp(-x)/x * (1 + ...)
                v = mpf_div(v, x, wp)
            re = mpf_mul(v, from_man_exp(s, -wp), prec, rnd)
        elif n == 1:
            re = mpf_neg(mpf_ei(negx, prec, rnd))
        elif n > 0 and n < 3*wp:
            T1 = mpf_neg(mpf_ei(negx, wp))
            if gamma:
                if n_orig & 1:
                    T1 = mpf_neg(T1)
            else:
                T1 = mpf_mul(T1, mpf_pow_int(negx, n-1, wp), wp)
            r = t = to_fixed(x, wp)
            facs = [1] * (n-1)
            for k in range(1,n-1):
                facs[k] = facs[k-1] * k
            facs = facs[::-1]
            s = facs[0] << wp
            for k in range(1, n-1):
                if k & 1:
                    s -= facs[k] * t
                else:
                    s += facs[k] * t
                t = (t*r) >> wp
            T2 = from_man_exp(s, -wp, wp)
            T2 = mpf_mul(T2, mpf_exp(negx, wp))
            if gamma:
                T2 = mpf_mul(T2, mpf_pow_int(x, n_orig, wp), wp)
            R = mpf_add(T1, T2)
            re = mpf_div(R, from_int(ifac(n-1)), prec, rnd)
        else:
            raise NotImplementedError
    if have_imag:
        M = from_int(-ifac(n-1))
        if gamma:
            im = mpf_div(mpf_pi(wp), M, prec, rnd)
            if n_orig & 1:
                im = mpf_neg(im)
        else:
            im = mpf_div(mpf_mul(mpf_pi(wp), mpf_pow_int(negx, n_orig-1, wp), wp), M, prec, rnd)
        return re, im
    else:
        return re, None

def mpf_ci_si_taylor(x, wp, which=0):
    """
    0 - Ci(x) - (euler+log(x))
    1 - Si(x)
    """
    x = to_fixed(x, wp)
    x2 = -(x*x) >> wp
    if which == 0:
        s, t, k = 0, (MPZ_ONE<<wp), 2
    else:
        s, t, k = x, x, 3
    while t:
        t = (t*x2//(k*(k-1)))>>wp
        s += t//k
        k += 2
    return from_man_exp(s, -wp)

def mpc_ci_si_taylor(re, im, wp, which=0):
    # The following code is only designed for small arguments,
    # and not too small arguments (for relative accuracy)
    if re[1]:
        mag = re[2]+re[3]
    elif im[1]:
        mag = im[2]+im[3]
    if im[1]:
        mag = max(mag, im[2]+im[3])
    if mag > 2 or mag < -wp:
        raise NotImplementedError
    wp += (2-mag)
    zre = to_fixed(re, wp)
    zim = to_fixed(im, wp)
    z2re = (zim*zim-zre*zre)>>wp
    z2im = (-2*zre*zim)>>wp
    tre = zre
    tim = zim
    one = MPZ_ONE<<wp
    if which == 0:
        sre, sim, tre, tim, k = 0, 0, (MPZ_ONE<<wp), 0, 2
    else:
        sre, sim, tre, tim, k = zre, zim, zre, zim, 3
    while max(abs(tre), abs(tim)) > 2:
        f = k*(k-1)
        tre, tim = ((tre*z2re-tim*z2im)//f)>>wp, ((tre*z2im+tim*z2re)//f)>>wp
        sre += tre//k
        sim += tim//k
        k += 2
    return from_man_exp(sre, -wp), from_man_exp(sim, -wp)

def mpf_ci_si(x, prec, rnd=round_fast, which=2):
    """
    Calculation of Ci(x), Si(x) for real x.

    which = 0 -- returns (Ci(x), -)
    which = 1 -- returns (Si(x), -)
    which = 2 -- returns (Ci(x), Si(x))

    Note: if x < 0, Ci(x) needs an additional imaginary term, pi*i.
    """
    wp = prec + 20
    sign, man, exp, bc = x
    ci, si = None, None
    if not man:
        if x == fzero:
            return (fninf, fzero)
        if x == fnan:
            return (x, x)
        ci = fzero
        if which != 0:
            if x == finf:
                si = mpf_shift(mpf_pi(prec, rnd), -1)
            if x == fninf:
                si = mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return (ci, si)
    # For small x: Ci(x) ~ euler + log(x), Si(x) ~ x
    mag = exp+bc
    if mag < -wp:
        if which != 0:
            si = mpf_perturb(x, 1-sign, prec, rnd)
        if which != 1:
            y = mpf_euler(wp)
            xabs = mpf_abs(x)
            ci = mpf_add(y, mpf_log(xabs, wp), prec, rnd)
        return ci, si
    # For huge x: Ci(x) ~ sin(x)/x, Si(x) ~ pi/2
    elif mag > wp:
        if which != 0:
            if sign:
                si = mpf_neg(mpf_pi(prec, negative_rnd[rnd]))
            else:
                si = mpf_pi(prec, rnd)
            si = mpf_shift(si, -1)
        if which != 1:
            ci = mpf_div(mpf_sin(x, wp), x, prec, rnd)
        return ci, si
    else:
        wp += abs(mag)
    # Use an asymptotic series? The smallest value of n!/x^n
    # occurs for n ~ x, where the magnitude is ~ exp(-x).
    asymptotic = mag-1 > math.log(wp, 2)
    # Case 1: convergent series near 0
    if not asymptotic:
        if which != 0:
            si = mpf_pos(mpf_ci_si_taylor(x, wp, 1), prec, rnd)
        if which != 1:
            ci = mpf_ci_si_taylor(x, wp, 0)
            ci = mpf_add(ci, mpf_euler(wp), wp)
            ci = mpf_add(ci, mpf_log(mpf_abs(x), wp), prec, rnd)
        return ci, si
    x = mpf_abs(x)
    # Case 2: asymptotic series for x >> 1
    xf = to_fixed(x, wp)
    xr = (MPZ_ONE<<(2*wp)) // xf   # 1/x
    s1 = (MPZ_ONE << wp)
    s2 = xr
    t = xr
    k = 2
    while t:
        t = -t
        t = (t*xr*k)>>wp
        k += 1
        s1 += t
        t = (t*xr*k)>>wp
        k += 1
        s2 += t
    s1 = from_man_exp(s1, -wp)
    s2 = from_man_exp(s2, -wp)
    s1 = mpf_div(s1, x, wp)
    s2 = mpf_div(s2, x, wp)
    cos, sin = mpf_cos_sin(x, wp)
    # Ci(x) = sin(x)*s1-cos(x)*s2
    # Si(x) = pi/2-cos(x)*s1-sin(x)*s2
    if which != 0:
        si = mpf_add(mpf_mul(cos, s1), mpf_mul(sin, s2), wp)
        si = mpf_sub(mpf_shift(mpf_pi(wp), -1), si, wp)
        if sign:
            si = mpf_neg(si)
        si = mpf_pos(si, prec, rnd)
    if which != 1:
        ci = mpf_sub(mpf_mul(sin, s1), mpf_mul(cos, s2), prec, rnd)
    return ci, si

def mpf_ci(x, prec, rnd=round_fast):
    if mpf_sign(x) < 0:
        raise ComplexResult
    return mpf_ci_si(x, prec, rnd, 0)[0]

def mpf_si(x, prec, rnd=round_fast):
    return mpf_ci_si(x, prec, rnd, 1)[1]

def mpc_ci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        ci = mpf_ci_si(re, prec, rnd, 0)[0]
        if mpf_sign(re) < 0:
            return (ci, mpf_pi(prec, rnd))
        return (ci, fzero)
    wp = prec + 20
    cre, cim = mpc_ci_si_taylor(re, im, wp, 0)
    cre = mpf_add(cre, mpf_euler(wp), wp)
    ci = mpc_add((cre, cim), mpc_log(z, wp), prec, rnd)
    return ci

def mpc_si(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        return (mpf_ci_si(re, prec, rnd, 1)[1], fzero)
    wp = prec + 20
    z = mpc_ci_si_taylor(re, im, wp, 1)
    return mpc_pos(z, prec, rnd)


#-----------------------------------------------------------------------#
#                                                                       #
#                             Bessel functions                          #
#                                                                       #
#-----------------------------------------------------------------------#

# A Bessel function of the first kind of integer order, J_n(x), is
# given by the power series

#             oo
#             ___         k         2 k + n
#            \        (-1)     / x \
#    J_n(x) = )    ----------- | - |
#            /___  k! (k + n)! \ 2 /
#            k = 0

# Simplifying the quotient between two successive terms gives the
# ratio x^2 / (-4*k*(k+n)). Hence, we only need one full-precision
# multiplication and one division by a small integer per term.
# The complex version is very similar, the only difference being
# that the multiplication is actually 4 multiplies.

# In the general case, we have
# J_v(x) = (x/2)**v / v! * 0F1(v+1, (-1/4)*z**2)

# TODO: for extremely large x, we could use an asymptotic
# trigonometric approximation.

# TODO: recompute at higher precision if the fixed-point mantissa
# is very small

def mpf_besseljn(n, x, prec, rounding=round_fast):
    prec += 50
    negate = n < 0 and n & 1
    mag = x[2]+x[3]
    n = abs(n)
    wp = prec + 20 + n*bitcount(n)
    if mag < 0:
        wp -= n * mag
    x = to_fixed(x, wp)
    x2 = (x**2) >> wp
    if not n:
        s = t = MPZ_ONE << wp
    else:
        s = t = (x**n // ifac(n)) >> ((n-1)*wp + n)
    k = 1
    while t:
        t = ((t * x2) // (-4*k*(k+n))) >> wp
        s += t
        k += 1
    if negate:
        s = -s
    return from_man_exp(s, -wp, prec, rounding)

def mpc_besseljn(n, z, prec, rounding=round_fast):
    negate = n < 0 and n & 1
    n = abs(n)
    origprec = prec
    zre, zim = z
    mag = max(zre[2]+zre[3], zim[2]+zim[3])
    prec += 20 + n*bitcount(n) + abs(mag)
    if mag < 0:
        prec -= n * mag
    zre = to_fixed(zre, prec)
    zim = to_fixed(zim, prec)
    z2re = (zre**2 - zim**2) >> prec
    z2im = (zre*zim) >> (prec-1)
    if not n:
        sre = tre = MPZ_ONE << prec
        sim = tim = MPZ_ZERO
    else:
        re, im = complex_int_pow(zre, zim, n)
        sre = tre = (re // ifac(n)) >> ((n-1)*prec + n)
        sim = tim = (im // ifac(n)) >> ((n-1)*prec + n)
    k = 1
    while abs(tre) + abs(tim) > 3:
        p = -4*k*(k+n)
        tre, tim = tre*z2re - tim*z2im, tim*z2re + tre*z2im
        tre = (tre // p) >> prec
        tim = (tim // p) >> prec
        sre += tre
        sim += tim
        k += 1
    if negate:
        sre = -sre
        sim = -sim
    re = from_man_exp(sre, -prec, origprec, rounding)
    im = from_man_exp(sim, -prec, origprec, rounding)
    return (re, im)

def mpf_agm(a, b, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(a,b) for
    nonnegative mpf values a, b.
    """
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign or bsign:
        raise ComplexResult("agm of a negative number")
    # Handle inf, nan or zero in either operand
    if not (aman and bman):
        if a == fnan or b == fnan:
            return fnan
        if a == finf:
            if b == fzero:
                return fnan
            return finf
        if b == finf:
            if a == fzero:
                return fnan
            return finf
        # agm(0,x) = agm(x,0) = 0
        return fzero
    wp = prec + 20
    amag = aexp+abc
    bmag = bexp+bbc
    mag_delta = amag - bmag
    # Reduce to roughly the same magnitude using floating-point AGM
    abs_mag_delta = abs(mag_delta)
    if abs_mag_delta > 10:
        while abs_mag_delta > 10:
            a, b = mpf_shift(mpf_add(a,b,wp),-1), \
                mpf_sqrt(mpf_mul(a,b,wp),wp)
            abs_mag_delta //= 2
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        amag = aexp+abc
        bmag = bexp+bbc
        mag_delta = amag - bmag
    #print to_float(a), to_float(b)
    # Use agm(a,b) = agm(x*a,x*b)/x to obtain a, b ~= 1
    min_mag = min(amag,bmag)
    max_mag = max(amag,bmag)
    n = 0
    # If too small, we lose precision when going to fixed-point
    if min_mag < -8:
        n = -min_mag
    # If too large, we waste time using fixed-point with large numbers
    elif max_mag > 20:
        n = -max_mag
    if n:
        a = mpf_shift(a, n)
        b = mpf_shift(b, n)
    #print to_float(a), to_float(b)
    af = to_fixed(a, wp)
    bf = to_fixed(b, wp)
    g = agm_fixed(af, bf, wp)
    return from_man_exp(g, -wp-n, prec, rnd)

def mpf_agm1(a, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(1,a) for a nonnegative
    mpf value a.
    """
    return mpf_agm(fone, a, prec, rnd)

def mpc_agm(a, b, prec, rnd=round_fast):
    """
    Complex AGM.

    TODO:
    * check that convergence works as intended
    * optimize
    * select a nonarbitrary branch
    """
    if mpc_is_infnan(a) or mpc_is_infnan(b):
        return fnan, fnan
    if mpc_zero in (a, b):
        return fzero, fzero
    if mpc_neg(a) == b:
        return fzero, fzero
    wp = prec+20
    eps = mpf_shift(fone, -wp+10)
    while 1:
        a1 = mpc_shift(mpc_add(a, b, wp), -1)
        b1 = mpc_sqrt(mpc_mul(a, b, wp), wp)
        a, b = a1, b1
        size = mpf_min_max([mpc_abs(a,10), mpc_abs(b,10)])[1]
        err = mpc_abs(mpc_sub(a, b, 10), 10)
        if size == fzero or mpf_lt(err, mpf_mul(eps, size)):
            return a

def mpc_agm1(a, prec, rnd=round_fast):
    return mpc_agm(mpc_one, a, prec, rnd)

def mpf_ellipk(x, prec, rnd=round_fast):
    if not x[1]:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return fzero
        if x == fnan:
            return x
    if x == fone:
        return finf
    # TODO: for |x| << 1/2, one could use fall back to
    # pi/2 * hyp2f1_rat((1,2),(1,2),(1,1), x)
    wp = prec + 15
    # Use K(x) = pi/2/agm(1,a) where a = sqrt(1-x)
    # The sqrt raises ComplexResult if x > 0
    a = mpf_sqrt(mpf_sub(fone, x, wp), wp)
    v = mpf_agm1(a, wp)
    r = mpf_div(mpf_pi(wp), v, prec, rnd)
    return mpf_shift(r, -1)

def mpc_ellipk(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return mpc_zero
        if mpf_le(re, fone):
            return mpf_ellipk(re, prec, rnd), fzero
    wp = prec + 15
    a = mpc_sqrt(mpc_sub(mpc_one, z, wp), wp)
    v = mpc_agm1(a, wp)
    r = mpc_mpf_div(mpf_pi(wp), v, prec, rnd)
    return mpc_shift(r, -1)

def mpf_ellipe(x, prec, rnd=round_fast):
    # http://functions.wolfram.com/EllipticIntegrals/
    # EllipticK/20/01/0001/
    # E = (1-m)*(K'(m)*2*m + K(m))
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return finf
        if x == fnan:
            return x
        if x == finf:
            raise ComplexResult
    if x == fone:
        return fone
    wp = prec+20
    mag = exp+bc
    if mag < -wp:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    # Compute a finite difference for K'
    p = max(mag, 0) - wp
    h = mpf_shift(fone, p)
    K = mpf_ellipk(x, 2*wp)
    Kh = mpf_ellipk(mpf_sub(x, h), 2*wp)
    Kdiff = mpf_shift(mpf_sub(K, Kh), -p)
    t = mpf_sub(fone, x)
    b = mpf_mul(Kdiff, mpf_shift(x,1), wp)
    return mpf_mul(t, mpf_add(K, b), prec, rnd)

def mpc_ellipe(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return (fzero, finf)
        if mpf_le(re, fone):
            return mpf_ellipe(re, prec, rnd), fzero
    wp = prec + 15
    mag = mpc_abs(z, 1)
    p = max(mag[2]+mag[3], 0) - wp
    h = mpf_shift(fone, p)
    K = mpc_ellipk(z, 2*wp)
    Kh = mpc_ellipk(mpc_add_mpf(z, h, 2*wp), 2*wp)
    Kdiff = mpc_shift(mpc_sub(Kh, K, wp), -p)
    t = mpc_sub(mpc_one, z, wp)
    b = mpc_mul(Kdiff, mpc_shift(z,1), wp)
    return mpc_mul(t, mpc_add(K, b, wp), prec, rnd)
<|endoftext|>"""
This module implements computation of elementary transcendental
functions (powers, logarithms, trigonometric and hyperbolic
functions, inverse trigonometric and hyperbolic) for real
floating-point numbers.

For complex and interval implementations of the same functions,
see libmpc and libmpi.

"""

import math
from bisect import bisect

from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND

from .libmpf import (
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast,
    ComplexResult,
    bitcount, bctable, lshift, rshift, giant_steps, sqrt_fixed,
    from_int, to_int, from_man_exp, to_fixed, to_float, from_float,
    from_rational, normalize,
    fzero, fone, fnone, fhalf, finf, fninf, fnan,
    mpf_cmp, mpf_sign, mpf_abs,
    mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_div, mpf_shift,
    mpf_rdiv_int, mpf_pow_int, mpf_sqrt,
    reciprocal_rnd, negative_rnd, mpf_perturb,
    isqrt_fast
)

from .libintmath import ifib


#-------------------------------------------------------------------------------
# Tuning parameters
#-------------------------------------------------------------------------------

# Cutoff for computing exp from cosh+sinh. This reduces the
# number of terms by half, but also requires a square root which
# is expensive with the pure-Python square root code.
if BACKEND == 'python':
    EXP_COSH_CUTOFF = 600
else:
    EXP_COSH_CUTOFF = 400
# Cutoff for using more than 2 series
EXP_SERIES_U_CUTOFF = 1500

# Also basically determined by sqrt
if BACKEND == 'python':
    COS_SIN_CACHE_PREC = 400
else:
    COS_SIN_CACHE_PREC = 200
COS_SIN_CACHE_STEP = 8
cos_sin_cache = {}

# Number of integer logarithms to cache (for zeta sums)
MAX_LOG_INT_CACHE = 2000
log_int_cache = {}

LOG_TAYLOR_PREC = 2500  # Use Taylor series with caching up to this prec
LOG_TAYLOR_SHIFT = 9    # Cache log values in steps of size 2^-N
log_taylor_cache = {}
# prec/size ratio of x for fastest convergence in AGM formula
LOG_AGM_MAG_PREC_RATIO = 20

ATAN_TAYLOR_PREC = 3000  # Same as for log
ATAN_TAYLOR_SHIFT = 7   # steps of size 2^-N
atan_taylor_cache = {}


# ~= next power of two + 20
cache_prec_steps = [22,22]
for k in xrange(1, bitcount(LOG_TAYLOR_PREC)+1):
    cache_prec_steps += [min(2**k,LOG_TAYLOR_PREC)+20] * 2**(k-1)


#----------------------------------------------------------------------------#
#                                                                            #
#                   Elementary mathematical constants                        #
#                                                                            #
#----------------------------------------------------------------------------#

def constant_memo(f):
    """
    Decorator for caching computed values of mathematical
    constants. This decorator should be applied to a
    function taking a single argument prec as input and
    returning a fixed-point value with the given precision.
    """
    f.memo_prec = -1
    f.memo_val = None
    def g(prec, **kwargs):
        memo_prec = f.memo_prec
        if prec <= memo_prec:
            return f.memo_val >> (memo_prec-prec)
        newprec = int(prec*1.05+10)
        f.memo_val = f(newprec, **kwargs)
        f.memo_prec = newprec
        return f.memo_val >> (newprec-prec)
    g.__name__ = f.__name__
    g.__doc__ = f.__doc__
    return g

def def_mpf_constant(fixed):
    """
    Create a function that computes the mpf value for a mathematical
    constant, given a function that computes the fixed-point value.

    Assumptions: the constant is positive and has magnitude ~= 1;
    the fixed-point function rounds to floor.
    """
    def f(prec, rnd=round_fast):
        wp = prec + 20
        v = fixed(wp)
        if rnd in (round_up, round_ceiling):
            v += 1
        return normalize(0, v, -wp, bitcount(v), prec, rnd)
    f.__doc__ = fixed.__doc__
    return f

def bsp_acot(q, a, b, hyperbolic):
    if b - a == 1:
        a1 = MPZ(2*a + 3)
        if hyperbolic or a&1:
            return MPZ_ONE, a1 * q**2, a1
        else:
            return -MPZ_ONE, a1 * q**2, a1
    m = (a+b)//2
    p1, q1, r1 = bsp_acot(q, a, m, hyperbolic)
    p2, q2, r2 = bsp_acot(q, m, b, hyperbolic)
    return q2*p1 + r1*p2, q1*q2, r1*r2

# the acoth(x) series converges like the geometric series for x^2
# N = ceil(p*log(2)/(2*log(x)))
def acot_fixed(a, prec, hyperbolic):
    """
    Compute acot(a) or acoth(a) for an integer a with binary splitting; see
    http://numbers.computation.free.fr/Constants/Algorithms/splitting.html
    """
    N = int(0.35 * prec/math.log(a) + 20)
    p, q, r = bsp_acot(a, 0,N, hyperbolic)
    return ((p+q)<<prec)//(q*a)

def machin(coefs, prec, hyperbolic=False):
    """
    Evaluate a Machin-like formula, i.e., a linear combination of
    acot(n) or acoth(n) for specific integer values of n, using fixed-
    point arithmetic. The input should be a list [(c, n), ...], giving
    c*acot[h](n) + ...
    """
    extraprec = 10
    s = MPZ_ZERO
    for a, b in coefs:
        s += MPZ(a) * acot_fixed(MPZ(b), prec+extraprec, hyperbolic)
    return (s >> extraprec)

# Logarithms of integers are needed for various computations involving
# logarithms, powers, radix conversion, etc

@constant_memo
def ln2_fixed(prec):
    """
    Computes ln(2). This is done with a hyperbolic Machin-type formula,
    with binary splitting at high precision.
    """
    return machin([(18, 26), (-2, 4801), (8, 8749)], prec, True)

@constant_memo
def ln10_fixed(prec):
    """
    Computes ln(10). This is done with a hyperbolic Machin-type formula.
    """
    return machin([(46, 31), (34, 49), (20, 161)], prec, True)


r"""
For computation of pi, we use the Chudnovsky series:

             oo
             ___        k
      1     \       (-1)  (6 k)! (A + B k)
    ----- =  )     -----------------------
    12 pi   /___               3  3k+3/2
                    (3 k)! (k!)  C
            k = 0

where A, B, and C are certain integer constants. This series adds roughly
14 digits per term. Note that C^(3/2) can be extracted so that the
series contains only rational terms. This makes binary splitting very
efficient.

The recurrence formulas for the binary splitting were taken from
ftp://ftp.gmplib.org/pub/src/gmp-chudnovsky.c

Previously, Machin's formula was used at low precision and the AGM iteration
was used at high precision. However, the Chudnovsky series is essentially as
fast as the Machin formula at low precision and in practice about 3x faster
than the AGM at high precision (despite theoretically having a worse
asymptotic complexity), so there is no reason not to use it in all cases.

"""

# Constants in Chudnovsky's series
CHUD_A = MPZ(13591409)
CHUD_B = MPZ(545140134)
CHUD_C = MPZ(640320)
CHUD_D = MPZ(12)

def bs_chudnovsky(a, b, level, verbose):
    """
    Computes the sum from a to b of the series in the Chudnovsky
    formula. Returns g, p, q where p/q is the sum as an exact
    fraction and g is a temporary value used to save work
    for recursive calls.
    """
    if b-a == 1:
        g = MPZ((6*b-5)*(2*b-1)*(6*b-1))
        p = b**3 * CHUD_C**3 // 24
        q = (-1)**b * g * (CHUD_A+CHUD_B*b)
    else:
        if verbose and level < 4:
            print("  binary splitting", a, b)
        mid = (a+b)//2
        g1, p1, q1 = bs_chudnovsky(a, mid, level+1, verbose)
        g2, p2, q2 = bs_chudnovsky(mid, b, level+1, verbose)
        p = p1*p2
        g = g1*g2
        q = q1*p2 + q2*g1
    return g, p, q

@constant_memo
def pi_fixed(prec, verbose=False, verbose_base=None):
    """
    Compute floor(pi * 2**prec) as a big integer.

    This is done using Chudnovsky's series (see comments in
    libelefun.py for details).
    """
    # The Chudnovsky series gives 14.18 digits per term
    N = int(prec/3.3219280948/14.181647462 + 2)
    if verbose:
        print("binary splitting with N =", N)
    g, p, q = bs_chudnovsky(0, N, 0, verbose)
    sqrtC = isqrt_fast(CHUD_C<<(2*prec))
    v = p*CHUD_C*sqrtC//((q+CHUD_A*p)*CHUD_D)
    return v

def degree_fixed(prec):
    return pi_fixed(prec)//180

def bspe(a, b):
    """
    Sum series for exp(1)-1 between a, b, returning the result
    as an exact fraction (p, q).
    """
    if b-a == 1:
        return MPZ_ONE, MPZ(b)
    m = (a+b)//2
    p1, q1 = bspe(a, m)
    p2, q2 = bspe(m, b)
    return p1*q2+p2, q1*q2

@constant_memo
def e_fixed(prec):
    """
    Computes exp(1). This is done using the ordinary Taylor series for
    exp, with binary splitting. For a description of the algorithm,
    see:

        http://numbers.computation.free.fr/Constants/
            Algorithms/splitting.html
    """
    # Slight overestimate of N needed for 1/N! < 2**(-prec)
    # This could be tightened for large N.
    N = int(1.1*prec/math.log(prec) + 20)
    p, q = bspe(0,N)
    return ((p+q)<<prec)//q

@constant_memo
def phi_fixed(prec):
    """
    Computes the golden ratio, (1+sqrt(5))/2
    """
    prec += 10
    a = isqrt_fast(MPZ_FIVE<<(2*prec)) + (MPZ_ONE << prec)
    return a >> 11

mpf_phi    = def_mpf_constant(phi_fixed)
mpf_pi     = def_mpf_constant(pi_fixed)
mpf_e      = def_mpf_constant(e_fixed)
mpf_degree = def_mpf_constant(degree_fixed)
mpf_ln2    = def_mpf_constant(ln2_fixed)
mpf_ln10   = def_mpf_constant(ln10_fixed)


@constant_memo
def ln_sqrt2pi_fixed(prec):
    wp = prec + 10
    # ln(sqrt(2*pi)) = ln(2*pi)/2
    return to_fixed(mpf_log(mpf_shift(mpf_pi(wp), 1), wp), prec-1)

@constant_memo
def sqrtpi_fixed(prec):
    return sqrt_fixed(pi_fixed(prec), prec)

mpf_sqrtpi   = def_mpf_constant(sqrtpi_fixed)
mpf_ln_sqrt2pi   = def_mpf_constant(ln_sqrt2pi_fixed)


#----------------------------------------------------------------------------#
#                                                                            #
#                                    Powers                                  #
#                                                                            #
#----------------------------------------------------------------------------#

def mpf_pow(s, t, prec, rnd=round_fast):
    """
    Compute s**t. Raises ComplexResult if s is negative and t is
    fractional.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if ssign and texp < 0:
        raise ComplexResult("negative number raised to a fractional power")
    if texp >= 0:
        return mpf_pow_int(s, (-1)**tsign * (tman<<texp), prec, rnd)
    # s**(n/2) = sqrt(s)**n
    if texp == -1:
        if tman == 1:
            if tsign:
                return mpf_div(fone, mpf_sqrt(s, prec+10,
                    reciprocal_rnd[rnd]), prec, rnd)
            return mpf_sqrt(s, prec, rnd)
        else:
            if tsign:
                return mpf_pow_int(mpf_sqrt(s, prec+10,
                    reciprocal_rnd[rnd]), -tman, prec, rnd)
            return mpf_pow_int(mpf_sqrt(s, prec+10, rnd), tman, prec, rnd)
    # General formula: s**t = exp(t*log(s))
    # TODO: handle rnd direction of the logarithm carefully
    c = mpf_log(s, prec+10, rnd)
    return mpf_exp(mpf_mul(t, c), prec, rnd)

def int_pow_fixed(y, n, prec):
    """n-th power of a fixed point number with precision prec

       Returns the power in the form man, exp,
       man * 2**exp ~= y**n
    """
    if n == 2:
        return (y*y), 0
    bc = bitcount(y)
    exp = 0
    workprec = 2 * (prec + 4*bitcount(n) + 4)
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm*y
            pe = pe+exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                pm = pm >> (pbc-workprec)
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        y = y*y
        exp = exp+exp
        bc = bc + bc - 2
        bc = bc + bctable[int(y >> bc)]
        if bc > workprec:
            y = y >> (bc-workprec)
            exp += bc - workprec
            bc = workprec
        n = n // 2
    return pm, pe

# froot(s, n, prec, rnd) computes the real n-th root of a
# positive mpf tuple s.
# To compute the root we start from a 50-bit estimate for r
# generated with ordinary floating-point arithmetic, and then refine
# the value to full accuracy using the iteration

#            1  /                     y       \
#   r     = --- | (n-1)  * r   +  ----------  |
#    n+1     n  \           n     r_n**(n-1)  /

# which is simply Newton's method applied to the equation r**n = y.
# With giant_steps(start, prec+extra) = [p0,...,pm, prec+extra]
# and y = man * 2**-shift  one has
# (man * 2**exp)**(1/n) =
# y**(1/n) * 2**(start-prec/n) * 2**(p0-start) * ... * 2**(prec+extra-pm) *
# 2**((exp+shift-(n-1)*prec)/n -extra))
# The last factor is accounted for in the last line of froot.

def nthroot_fixed(y, n, prec, exp1):
    start = 50
    try:
        y1 = rshift(y, prec - n*start)
        r = MPZ(int(y1**(1.0/n)))
    except OverflowError:
        y1 = from_int(y1, start)
        fn = from_int(n)
        fn = mpf_rdiv_int(1, fn, start)
        r = mpf_pow(y1, fn, start)
        r = to_int(r)
    extra = 10
    extra1 = n
    prevp = start
    for p in giant_steps(start, prec+extra):
        pm, pe = int_pow_fixed(r, n-1, prevp)
        r2 = rshift(pm, (n-1)*prevp - p - pe - extra1)
        B = lshift(y, 2*p-prec+extra1)//r2
        r = (B + (n-1) * lshift(r, p-prevp))//n
        prevp = p
    return r

def mpf_nthroot(s, n, prec, rnd=round_fast):
    """nth-root of a positive number

    Use the Newton method when faster, otherwise use x**(1/n)
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult("nth root of a negative number")
    if not man:
        if s == fnan:
            return fnan
        if s == fzero:
            if n > 0:
                return fzero
            if n == 0:
                return fone
            return finf
        # Infinity
        if not n:
            return fnan
        if n < 0:
            return fzero
        return finf
    flag_inverse = False
    if n < 2:
        if n == 0:
            return fone
        if n == 1:
            return mpf_pos(s, prec, rnd)
        if n == -1:
            return mpf_div(fone, s, prec, rnd)
        # n < 0
        rnd = reciprocal_rnd[rnd]
        flag_inverse = True
        extra_inverse = 5
        prec += extra_inverse
        n = -n
    if n > 20 and (n >= 20000 or prec < int(233 + 28.3 * n**0.62)):
        prec2 = prec + 10
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, prec2)
        r = mpf_pow(s, nth, prec2, rnd)
        s = normalize(r[0], r[1], r[2], r[3], prec, rnd)
        if flag_inverse:
            return mpf_div(fone, s, prec-extra_inverse, rnd)
        else:
            return s
    # Convert to a fixed-point number with prec2 bits.
    prec2 = prec + 2*n - (prec%n)
    # a few tests indicate that
    # for 10 < n < 10**4 a bit more precision is needed
    if n > 10:
        prec2 += prec2//10
        prec2 = prec2 - prec2%n
    # Mantissa may have more bits than we need. Trim it down.
    shift = bc - prec2
    # Adjust exponents to make prec2 and exp+shift multiples of n.
    sign1 = 0
    es = exp+shift
    if es < 0:
        sign1 = 1
        es = -es
    if sign1:
        shift += es%n
    else:
        shift -= es%n
    man = rshift(man, shift)
    extra = 10
    exp1 = ((exp+shift-(n-1)*prec2)//n) - extra
    rnd_shift = 0
    if flag_inverse:
        if rnd == 'u' or rnd == 'c':
            rnd_shift = 1
    else:
        if rnd == 'd' or rnd == 'f':
            rnd_shift = 1
    man = nthroot_fixed(man+rnd_shift, n, prec2, exp1)
    s = from_man_exp(man, exp1, prec, rnd)
    if flag_inverse:
        return mpf_div(fone, s, prec-extra_inverse, rnd)
    else:
        return s

def mpf_cbrt(s, prec, rnd=round_fast):
    """cubic root of a positive number"""
    return mpf_nthroot(s, 3, prec, rnd)

#----------------------------------------------------------------------------#
#                                                                            #
#                                Logarithms                                  #
#                                                                            #
#----------------------------------------------------------------------------#


def log_int_fixed(n, prec, ln2=None):
    """
    Fast computation of log(n), caching the value for small n,
    intended for zeta sums.
    """
    if n in log_int_cache:
        value, vprec = log_int_cache[n]
        if vprec >= prec:
            return value >> (vprec - prec)
    wp = prec + 10
    if wp <= LOG_TAYLOR_SHIFT:
        if ln2 is None:
            ln2 = ln2_fixed(wp)
        r = bitcount(n)
        x = n << (wp-r)
        v = log_taylor_cached(x, wp) + r*ln2
    else:
        v = to_fixed(mpf_log(from_int(n), wp+5), wp)
    if n < MAX_LOG_INT_CACHE:
        log_int_cache[n] = (v, wp)
    return v >> (wp-prec)

def agm_fixed(a, b, prec):
    """
    Fixed-point computation of agm(a,b), assuming
    a, b both close to unit magnitude.
    """
    i = 0
    while 1:
        anew = (a+b)>>1
        if i > 4 and abs(a-anew) < 8:
            return a
        b = isqrt_fast(a*b)
        a = anew
        i += 1
    return a

def log_agm(x, prec):
    """
    Fixed-point computation of -log(x) = log(1/x), suitable
    for large precision. It is required that 0 < x < 1. The
    algorithm used is the Sasaki-Kanada formula

        -log(x) = pi/agm(theta2(x)^2,theta3(x)^2). [1]

    For faster convergence in the theta functions, x should
    be chosen closer to 0.

    Guard bits must be added by the caller.

    HYPOTHESIS: if x = 2^(-n), n bits need to be added to
    account for the truncation to a fixed-point number,
    and this is the only significant cancellation error.

    The number of bits lost to roundoff is small and can be
    considered constant.

    [1] Richard P. Brent, "Fast Algorithms for High-Precision
        Computation of Elementary Functions (extended abstract)",
        http://wwwmaths.anu.edu.au/~brent/pd/RNC7-Brent.pdf

    """
    x2 = (x*x) >> prec
    # Compute jtheta2(x)**2
    s = a = b = x2
    while a:
        b = (b*x2) >> prec
        a = (a*b) >> prec
        s += a
    s += (MPZ_ONE<<prec)
    s = (s*s)>>(prec-2)
    s = (s*isqrt_fast(x<<prec))>>prec
    # Compute jtheta3(x)**2
    t = a = b = x
    while a:
        b = (b*x2) >> prec
        a = (a*b) >> prec
        t += a
    t = (MPZ_ONE<<prec) + (t<<1)
    t = (t*t)>>prec
    # Final formula
    p = agm_fixed(s, t, prec)
    return (pi_fixed(prec) << prec) // p

def log_taylor(x, prec, r=0):
    """
    Fixed-point calculation of log(x). It is assumed that x is close
    enough to 1 for the Taylor series to converge quickly. Convergence
    can be improved by specifying r > 0 to compute
    log(x^(1/2^r))*2^r, at the cost of performing r square roots.

    The caller must provide sufficient guard bits.
    """
    for i in xrange(r):
        x = isqrt_fast(x<<prec)
    one = MPZ_ONE << prec
    v = ((x-one)<<prec)//(x+one)
    sign = v < 0
    if sign:
        v = -v
    v2 = (v*v) >> prec
    v4 = (v2*v2) >> prec
    s0 = v
    s1 = v//3
    v = (v*v4) >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = (v*v4) >> prec
        k += 2
    s1 = (s1*v2) >> prec
    s = (s0+s1) << (1+r)
    if sign:
        return -s
    return s

def log_taylor_cached(x, prec):
    """
    Fixed-point computation of log(x), assuming x in (0.5, 2)
    and prec <= LOG_TAYLOR_PREC.
    """
    n = x >> (prec-LOG_TAYLOR_SHIFT)
    cached_prec = cache_prec_steps[prec]
    dprec = cached_prec - prec
    if (n, cached_prec) in log_taylor_cache:
        a, log_a = log_taylor_cache[n, cached_prec]
    else:
        a = n << (cached_prec - LOG_TAYLOR_SHIFT)
        log_a = log_taylor(a, cached_prec, 8)
        log_taylor_cache[n, cached_prec] = (a, log_a)
    a >>= dprec
    log_a >>= dprec
    u = ((x - a) << prec) // a
    v = (u << prec) // ((MPZ_TWO << prec) + u)
    v2 = (v*v) >> prec
    v4 = (v2*v2) >> prec
    s0 = v
    s1 = v//3
    v = (v*v4) >> prec
    k = 5
    while v:
        s0 += v//k
        k += 2
        s1 += v//k
        v = (v*v4) >> prec
        k += 2
    s1 = (s1*v2) >> prec
    s = (s0+s1) << 1
    return log_a + s

def mpf_log(x, prec, rnd=round_fast):
    """
    Compute the natural logarithm of the mpf value x. If x is negative,
    ComplexResult is raised.
    """
    sign, man, exp, bc = x
    #------------------------------------------------------------------
    # Handle special values
    if not man:
        if x == fzero: return fninf
        if x == finf: return finf
        if x == fnan: return fnan
    if sign:
        raise ComplexResult("logarithm of a negative number")
    wp = prec + 20
    #------------------------------------------------------------------
    # Handle log(2^n) = log(n)*2.
    # Here we catch the only possible exact value, log(1) = 0
    if man == 1:
        if not exp:
            return fzero
        return from_man_exp(exp*ln2_fixed(wp), -wp, prec, rnd)
    mag = exp+bc
    abs_mag = abs(mag)
    #------------------------------------------------------------------
    # Handle x = 1+eps, where log(x) ~ x. We need to check for
    # cancellation when moving to fixed-point math and compensate
    # by increasing the precision. Note that abs_mag in (0, 1) <=>
    # 0.5 < x < 2 and x != 1
    if abs_mag <= 1:
        # Calculate t = x-1 to measure distance from 1 in bits
        tsign = 1-abs_mag
        if tsign:
            tman = (MPZ_ONE<<bc) - man
        else:
            tman = man - (MPZ_ONE<<(bc-1))
        tbc = bitcount(tman)
        cancellation = bc - tbc
        if cancellation > wp:
            t = normalize(tsign, tman, abs_mag-bc, tbc, tbc, 'n')
            return mpf_perturb(t, tsign, prec, rnd)
        else:
            wp += cancellation
        # TODO: if close enough to 1, we could use Taylor series
        # even in the AGM precision range, since the Taylor series
        # converges rapidly
    #------------------------------------------------------------------
    # Another special case:
    # n*log(2) is a good enough approximation
    if abs_mag > 10000:
        if bitcount(abs_mag) > wp:
            return from_man_exp(exp*ln2_fixed(wp), -wp, prec, rnd)
    #------------------------------------------------------------------
    # General case.
    # Perform argument reduction using log(x) = log(x*2^n) - n*log(2):
    # If we are in the Taylor precision range, choose magnitude 0 or 1.
    # If we are in the AGM precision range, choose magnitude -m for
    # some large m; benchmarking on one machine showed m = prec/20 to be
    # optimal between 1000 and 100,000 digits.
    if wp <= LOG_TAYLOR_PREC:
        m = log_taylor_cached(lshift(man, wp-bc), wp)
        if mag:
            m += mag*ln2_fixed(wp)
    else:
        optimal_mag = -wp//LOG_AGM_MAG_PREC_RATIO
        n = optimal_mag - mag
        x = mpf_shift(x, n)
        wp += (-optimal_mag)
        m = -log_agm(to_fixed(x, wp), wp)
        m -= n*ln2_fixed(wp)
    return from_man_exp(m, -wp, prec, rnd)

def mpf_log_hypot(a, b, prec, rnd):
    """
    Computes log(sqrt(a^2+b^2)) accurately.
    """
    # If either a or b is inf/nan/0, assume it to be a
    if not b[1]:
        a, b = b, a
    # a is inf/nan/0
    if not a[1]:
        # both are inf/nan/0
        if not b[1]:
            if a == b == fzero:
                return fninf
            if fnan in (a, b):
                return fnan
            # at least one term is (+/- inf)^2
            return finf
        # only a is inf/nan/0
        if a == fzero:
            # log(sqrt(0+b^2)) = log(|b|)
            return mpf_log(mpf_abs(b), prec, rnd)
        if a == fnan:
            return fnan
        return finf
    # Exact
    a2 = mpf_mul(a,a)
    b2 = mpf_mul(b,b)
    extra = 20
    # Not exact
    h2 = mpf_add(a2, b2, prec+extra)
    cancelled = mpf_add(h2, fnone, 10)
    mag_cancelled = cancelled[2]+cancelled[3]
    # Just redo the sum exactly if necessary (could be smarter
    # and avoid memory allocation when a or b is precisely 1
    # and the other is tiny...)
    if cancelled == fzero or mag_cancelled < -extra//2:
        h2 = mpf_add(a2, b2, prec+extra-min(a2[2],b2[2]))
    return mpf_shift(mpf_log(h2, prec, rnd), -1)


#----------------------------------------------------------------------
# Inverse tangent
#

def atan_newton(x, prec):
    if prec >= 100:
        r = math.atan(int((x>>(prec-53)))/2.0**53)
    else:
        r = math.atan(int(x)/2.0**prec)
    prevp = 50
    r = MPZ(int(r * 2.0**53) >> (53-prevp))
    extra_p = 50
    for wp in giant_steps(prevp, prec):
        wp += extra_p
        r = r << (wp-prevp)
        cos, sin = cos_sin_fixed(r, wp)
        tan = (sin << wp) // cos
        a = ((tan-rshift(x, prec-wp)) << wp) // ((MPZ_ONE<<wp) + ((tan**2)>>wp))
        r = r - a
        prevp = wp
    return rshift(r, prevp-prec)

def atan_taylor_get_cached(n, prec):
    # Taylor series with caching wins up to huge precisions
    # To avoid unnecessary precomputation at low precision, we
    # do it in steps
    # Round to next power of 2
    prec2 = (1<<(bitcount(prec-1))) + 20
    dprec = prec2 - prec
    if (n, prec2) in atan_taylor_cache:
        a, atan_a = atan_taylor_cache[n, prec2]
    else:
        a = n << (prec2 - ATAN_TAYLOR_SHIFT)
        atan_a = atan_newton(a, prec2)
        atan_taylor_cache[n, prec2] = (a, atan_a)
    return (a >> dprec), (atan_a >> dprec)

def atan_taylor(x, prec):
    n = (x >> (prec-ATAN_TAYLOR_SHIFT))
    a, atan_a = atan_taylor_get_cached(n, prec)
    d = x - a
    s0 = v = (d << prec) // ((a**2 >> prec) + (a*d >> prec) + (MPZ_ONE << prec))
    v2 = (v**2 >> prec)
    v4 = (v2 * v2) >> prec
    s1 = v//3
    v = (v * v4) >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = (v * v4) >> prec
        k += 2
    s1 = (s1 * v2) >> prec
    s = s0 - s1
    return atan_a + s

def atan_inf(sign, prec, rnd):
    if not sign:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))

def mpf_atan(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fzero
        if x == finf: return atan_inf(0, prec, rnd)
        if x == fninf: return atan_inf(1, prec, rnd)
        return fnan
    mag = exp + bc
    # Essentially infinity
    if mag > prec+20:
        return atan_inf(sign, prec, rnd)
    # Essentially ~ x
    if -mag > prec+20:
        return mpf_perturb(x, 1-sign, prec, rnd)
    wp = prec + 30 + abs(mag)
    # For large x, use atan(x) = pi/2 - atan(1/x)
    if mag >= 2:
        x = mpf_rdiv_int(1, x, wp)
        reciprocal = True
    else:
        reciprocal = False
    t = to_fixed(x, wp)
    if sign:
        t = -t
    if wp < ATAN_TAYLOR_PREC:
        a = atan_taylor(t, wp)
    else:
        a = atan_newton(t, wp)
    if reciprocal:
        a = ((pi_fixed(wp)>>1)+1) - a
    if sign:
        a = -a
    return from_man_exp(a, -wp, prec, rnd)

# TODO: cleanup the special cases
def mpf_atan2(y, x, prec, rnd=round_fast):
    xsign, xman, xexp, xbc = x
    ysign, yman, yexp, ybc = y
    if not yman:
        if y == fzero and x != fnan:
            if mpf_sign(x) >= 0:
                return fzero
            return mpf_pi(prec, rnd)
        if y in (finf, fninf):
            if x in (finf, fninf):
                return fnan
            # pi/2
            if y == finf:
                return mpf_shift(mpf_pi(prec, rnd), -1)
            # -pi/2
            return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return fnan
    if ysign:
        return mpf_neg(mpf_atan2(mpf_neg(y), x, prec, negative_rnd[rnd]))
    if not xman:
        if x == fnan:
            return fnan
        if x == finf:
            return fzero
        if x == fninf:
            return mpf_pi(prec, rnd)
        if y == fzero:
            return fzero
        return mpf_shift(mpf_pi(prec, rnd), -1)
    tquo = mpf_atan(mpf_div(y, x, prec+4), prec+4)
    if xsign:
        return mpf_add(mpf_pi(prec+4), tquo, prec, rnd)
    else:
        return mpf_pos(tquo, prec, rnd)

def mpf_asin(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if bc+exp > 0 and x not in (fone, fnone):
        raise ComplexResult("asin(x) is real only for -1 <= x <= 1")
    # asin(x) = 2*atan(x/(1+sqrt(1-x**2)))
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_add(fone, mpf_sqrt(mpf_sub(fone, a, wp), wp), wp)
    c = mpf_div(x, b, wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)

def mpf_acos(x, prec, rnd=round_fast):
    # acos(x) = 2*atan(sqrt(1-x**2)/(1+x))
    sign, man, exp, bc = x
    if bc + exp > 0:
        if x not in (fone, fnone):
            raise ComplexResult("acos(x) is real only for -1 <= x <= 1")
        if x == fnone:
            return mpf_pi(prec, rnd)
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_sqrt(mpf_sub(fone, a, wp), wp)
    c = mpf_div(b, mpf_add(fone, x, wp), wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)

def mpf_asinh(x, prec, rnd=round_fast):
    wp = prec + 20
    sign, man, exp, bc = x
    mag = exp+bc
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, 1-sign, prec, rnd)
        wp += (-mag)
    # asinh(x) = log(x+sqrt(x**2+1))
    # use reflection symmetry to avoid cancellation
    q = mpf_sqrt(mpf_add(mpf_mul(x, x), fone, wp), wp)
    q = mpf_add(mpf_abs(x), q, wp)
    if sign:
        return mpf_neg(mpf_log(q, prec, negative_rnd[rnd]))
    else:
        return mpf_log(q, prec, rnd)

def mpf_acosh(x, prec, rnd=round_fast):
    # acosh(x) = log(x+sqrt(x**2-1))
    wp = prec + 15
    if mpf_cmp(x, fone) == -1:
        raise ComplexResult("acosh(x) is real only for x >= 1")
    q = mpf_sqrt(mpf_add(mpf_mul(x,x), fnone, wp), wp)
    return mpf_log(mpf_add(x, q, wp), prec, rnd)

def mpf_atanh(x, prec, rnd=round_fast):
    # atanh(x) = log((1+x)/(1-x))/2
    sign, man, exp, bc = x
    if (not man) and exp:
        if x in (fzero, fnan):
            return x
        raise ComplexResult("atanh(x) is real only for -1 <= x <= 1")
    mag = bc + exp
    if mag > 0:
        if mag == 1 and man == 1:
            return [finf, fninf][sign]
        raise ComplexResult("atanh(x) is real only for -1 <= x <= 1")
    wp = prec + 15
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, sign, prec, rnd)
        wp += (-mag)
    a = mpf_add(x, fone, wp)
    b = mpf_sub(fone, x, wp)
    return mpf_shift(mpf_log(mpf_div(a, b, wp), prec, rnd), -1)

def mpf_fibonacci(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fninf:
            return fnan
        return x
    # F(2^n) ~= 2^(2^n)
    size = abs(exp+bc)
    if exp >= 0:
        # Exact
        if size < 10 or size <= bitcount(prec):
            return from_int(ifib(to_int(x)), prec, rnd)
    # Use the modified Binet formula
    wp = prec + size + 20
    a = mpf_phi(wp)
    b = mpf_add(mpf_shift(a, 1), fnone, wp)
    u = mpf_pow(a, x, wp)
    v = mpf_cos_pi(x, wp)
    v = mpf_div(v, u, wp)
    u = mpf_sub(u, v, wp)
    u = mpf_div(u, b, prec, rnd)
    return u


#-------------------------------------------------------------------------------
# Exponential-type functions
#-------------------------------------------------------------------------------

def exponential_series(x, prec, type=0):
    """
    Taylor series for cosh/sinh or cos/sin.

    type = 0 -- returns exp(x)  (slightly faster than cosh+sinh)
    type = 1 -- returns (cosh(x), sinh(x))
    type = 2 -- returns (cos(x), sin(x))
    """
    if x < 0:
        x = -x
        sign = 1
    else:
        sign = 0
    r = int(0.5*prec**0.5)
    xmag = bitcount(x) - prec
    r = max(0, xmag + r)
    extra = 10 + 2*max(r,-xmag)
    wp = prec + extra
    x <<= (extra - r)
    one = MPZ_ONE << wp
    alt = (type == 2)
    if prec < EXP_SERIES_U_CUTOFF:
        x2 = a = (x*x) >> wp
        x4 = (x2*x2) >> wp
        s0 = s1 = MPZ_ZERO
        k = 2
        while a:
            a //= (k-1)*k; s0 += a; k += 2
            a //= (k-1)*k; s1 += a; k += 2
            a = (a*x4) >> wp
        s1 = (x2*s1) >> wp
        if alt:
            c = s1 - s0 + one
        else:
            c = s1 + s0 + one
    else:
        u = int(0.3*prec**0.35)
        x2 = a = (x*x) >> wp
        xpowers = [one, x2]
        for i in xrange(1, u):
            xpowers.append((xpowers[-1]*x2)>>wp)
        sums = [MPZ_ZERO] * u
        k = 2
        while a:
            for i in xrange(u):
                a //= (k-1)*k
                if alt and k & 2: sums[i] -= a
                else:             sums[i] += a
                k += 2
            a = (a*xpowers[-1]) >> wp
        for i in xrange(1, u):
            sums[i] = (sums[i]*xpowers[i]) >> wp
        c = sum(sums) + one
    if type == 0:
        s = isqrt_fast(c*c - (one<<wp))
        if sign:
            v = c - s
        else:
            v = c + s
        for i in xrange(r):
            v = (v*v) >> wp
        return v >> extra
    else:
        # Repeatedly apply the double-angle formula
        # cosh(2*x) = 2*cosh(x)^2 - 1
        # cos(2*x) = 2*cos(x)^2 - 1
        pshift = wp-1
        for i in xrange(r):
            c = ((c*c) >> pshift) - one
        # With the abs, this is the same for sinh and sin
        s = isqrt_fast(abs((one<<wp) - c*c))
        if sign:
            s = -s
        return (c>>extra), (s>>extra)

def exp_basecase(x, prec):
    """
    Compute exp(x) as a fixed-point number. Works for any x,
    but for speed should have |x| < 1. For an arbitrary number,
    use exp(x) = exp(x-m*log(2)) * 2^m where m = floor(x/log(2)).
    """
    if prec > EXP_COSH_CUTOFF:
        return exponential_series(x, prec, 0)
    r = int(prec**0.5)
    prec += r
    s0 = s1 = (MPZ_ONE << prec)
    k = 2
    a = x2 = (x*x) >> prec
    while a:
        a //= k; s0 += a; k += 1
        a //= k; s1 += a; k += 1
        a = (a*x2) >> prec
    s1 = (s1*x) >> prec
    s = s0 + s1
    u = r
    while r:
        s = (s*s) >> prec
        r -= 1
    return s >> u

def exp_expneg_basecase(x, prec):
    """
    Computation of exp(x), exp(-x)
    """
    if prec > EXP_COSH_CUTOFF:
        cosh, sinh = exponential_series(x, prec, 1)
        return cosh+sinh, cosh-sinh
    a = exp_basecase(x, prec)
    b = (MPZ_ONE << (prec+prec)) // a
    return a, b

def cos_sin_basecase(x, prec):
    """
    Compute cos(x), sin(x) as fixed-point numbers, assuming x
    in [0, pi/2). For an arbitrary number, use x' = x - m*(pi/2)
    where m = floor(x/(pi/2)) along with quarter-period symmetries.
    """
    if prec > COS_SIN_CACHE_PREC:
        return exponential_series(x, prec, 2)
    precs = prec - COS_SIN_CACHE_STEP
    t = x >> precs
    n = int(t)
    if n not in cos_sin_cache:
        w = t<<(10+COS_SIN_CACHE_PREC-COS_SIN_CACHE_STEP)
        cos_t, sin_t = exponential_series(w, 10+COS_SIN_CACHE_PREC, 2)
        cos_sin_cache[n] = (cos_t>>10), (sin_t>>10)
    cos_t, sin_t = cos_sin_cache[n]
    offset = COS_SIN_CACHE_PREC - prec
    cos_t >>= offset
    sin_t >>= offset
    x -= t << precs
    cos = MPZ_ONE << prec
    sin = x
    k = 2
    a = -((x*x) >> prec)
    while a:
        a //= k; cos += a; k += 1; a = (a*x) >> prec
        a //= k; sin += a; k += 1; a = -((a*x) >> prec)
    return ((cos*cos_t-sin*sin_t) >> prec), ((sin*cos_t+cos*sin_t) >> prec)

def mpf_exp(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if man:
        mag = bc + exp
        wp = prec + 14
        if sign:
            man = -man
        # TODO: the best cutoff depends on both x and the precision.
        if prec > 600 and exp >= 0:
            # Need about log2(exp(n)) ~= 1.45*mag extra precision
            e = mpf_e(wp+int(1.45*mag))
            return mpf_pow_int(e, man<<exp, prec, rnd)
        if mag < -wp:
            return mpf_perturb(fone, sign, prec, rnd)
        # |x| >= 2
        if mag > 1:
            # For large arguments: exp(2^mag*(1+eps)) =
            # exp(2^mag)*exp(2^mag*eps) = exp(2^mag)*(1 + 2^mag*eps + ...)
            # so about mag extra bits is required.
            wpmod = wp + mag
            offset = exp + wpmod
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            lg2 = ln2_fixed(wpmod)
            n, t = divmod(t, lg2)
            n = int(n)
            t >>= mag
        else:
            offset = exp + wp
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            n = 0
        man = exp_basecase(t, wp)
        return from_man_exp(man, n-wp, prec, rnd)
    if not exp:
        return fone
    if x == fninf:
        return fzero
    return x


def mpf_cosh_sinh(x, prec, rnd=round_fast, tanh=0):
    """Simultaneously compute (cosh(x), sinh(x)) for real x"""
    sign, man, exp, bc = x
    if (not man) and exp:
        if tanh:
            if x == finf: return fone
            if x == fninf: return fnone
            return fnan
        if x == finf: return (finf, finf)
        if x == fninf: return (finf, fninf)
        return fnan, fnan
    mag = exp+bc
    wp = prec+14
    if mag < -4:
        # Extremely close to 0, sinh(x) ~= x and cosh(x) ~= 1
        if mag < -wp:
            if tanh:
                return mpf_perturb(x, 1-sign, prec, rnd)
            cosh = mpf_perturb(fone, 0, prec, rnd)
            sinh = mpf_perturb(x, sign, prec, rnd)
            return cosh, sinh
        # Fix for cancellation when computing sinh
        wp += (-mag)
    # Does exp(-2*x) vanish?
    if mag > 10:
        if 3*(1<<(mag-1)) > wp:
            # XXX: rounding
            if tanh:
                return mpf_perturb([fone,fnone][sign], 1-sign, prec, rnd)
            c = s = mpf_shift(mpf_exp(mpf_abs(x), prec, rnd), -1)
            if sign:
                s = mpf_neg(s)
            return c, s
    # |x| > 1
    if mag > 1:
        wpmod = wp + mag
        offset = exp + wpmod
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        lg2 = ln2_fixed(wpmod)
        n, t = divmod(t, lg2)
        n = int(n)
        t >>= mag
    else:
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        n = 0
    a, b = exp_expneg_basecase(t, wp)
    # TODO: optimize division precision
    cosh = a + (b>>(2*n))
    sinh = a - (b>>(2*n))
    if sign:
        sinh = -sinh
    if tanh:
        man = (sinh << wp) // cosh
        return from_man_exp(man, -wp, prec, rnd)
    else:
        cosh = from_man_exp(cosh, n-wp-1, prec, rnd)
        sinh = from_man_exp(sinh, n-wp-1, prec, rnd)
        return cosh, sinh


def mod_pi2(man, exp, mag, wp):
    # Reduce to standard interval
    if mag > 0:
        i = 0
        while 1:
            cancellation_prec = 20 << i
            wpmod = wp + mag + cancellation_prec
            pi2 = pi_fixed(wpmod-1)
            pi4 = pi2 >> 1
            offset = wpmod + exp
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            n, y = divmod(t, pi2)
            if y > pi4:
                small = pi2 - y
            else:
                small = y
            if small >> (wp+mag-10):
                n = int(n)
                t = y >> mag
                wp = wpmod - mag
                break
            i += 1
    else:
        wp += (-mag)
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        n = 0
    return t, n, wp


def mpf_cos_sin(x, prec, rnd=round_fast, which=0, pi=False):
    """
    which:
    0 -- return cos(x), sin(x)
    1 -- return cos(x)
    2 -- return sin(x)
    3 -- return tan(x)

    if pi=True, compute for pi*x
    """
    sign, man, exp, bc = x
    if not man:
        if exp:
            c, s = fnan, fnan
        else:
            c, s = fone, fzero
        if which == 0: return c, s
        if which == 1: return c
        if which == 2: return s
        if which == 3: return s

    mag = bc + exp
    wp = prec + 10

    # Extremely small?
    if mag < 0:
        if mag < -wp:
            if pi:
                x = mpf_mul(x, mpf_pi(wp))
            c = mpf_perturb(fone, 1, prec, rnd)
            s = mpf_perturb(x, 1-sign, prec, rnd)
            if which == 0: return c, s
            if which == 1: return c
            if which == 2: return s
            if which == 3: return mpf_perturb(x, sign, prec, rnd)
    if pi:
        if exp >= -1:
            if exp == -1:
                c = fzero
                s = (fone, fnone)[bool(man & 2) ^ sign]
            elif exp == 0:
                c, s = (fnone, fzero)
            else:
                c, s = (fone, fzero)
            if which == 0: return c, s
            if which == 1: return c
            if which == 2: return s
            if which == 3: return mpf_div(s, c, prec, rnd)
        # Subtract nearest half-integer (= mod by pi/2)
        n = ((man >> (-exp-2)) + 1) >> 1
        man = man - (n << (-exp-1))
        mag2 = bitcount(man) + exp
        wp = prec + 10 - mag2
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        t = (t*pi_fixed(wp)) >> wp
    else:
        t, n, wp = mod_pi2(man, exp, mag, wp)
    c, s = cos_sin_basecase(t, wp)
    m = n & 3
    if   m == 1: c, s = -s, c
    elif m == 2: c, s = -c, -s
    elif m == 3: c, s = s, -c
    if sign:
        s = -s
    if which == 0:
        c = from_man_exp(c, -wp, prec, rnd)
        s = from_man_exp(s, -wp, prec, rnd)
        return c, s
    if which == 1:
        return from_man_exp(c, -wp, prec, rnd)
    if which == 2:
        return from_man_exp(s, -wp, prec, rnd)
    if which == 3:
        return from_rational(s, c, prec, rnd)

def mpf_cos(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 1)
def mpf_sin(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 2)
def mpf_tan(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 3)
def mpf_cos_sin_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 0, 1)
def mpf_cos_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 1, 1)
def mpf_sin_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 2, 1)
def mpf_cosh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd)[0]
def mpf_sinh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd)[1]
def mpf_tanh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd, tanh=1)


# Low-overhead fixed-point versions

def cos_sin_fixed(x, prec, pi2=None):
    if pi2 is None:
        pi2 = pi_fixed(prec-1)
    n, t = divmod(x, pi2)
    n = int(n)
    c, s = cos_sin_basecase(t, prec)
    m = n & 3
    if m == 0: return c, s
    if m == 1: return -s, c
    if m == 2: return -c, -s
    if m == 3: return s, -c

def exp_fixed(x, prec, ln2=None):
    if ln2 is None:
        ln2 = ln2_fixed(prec)
    n, t = divmod(x, ln2)
    n = int(n)
    v = exp_basecase(t, prec)
    if n >= 0:
        return v << n
    else:
        return v >> (-n)


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as _lbmp
        mpf_sqrt = _lbmp.mpf_sqrt
        mpf_exp = _lbmp.mpf_exp
        mpf_log = _lbmp.mpf_log
        mpf_cos = _lbmp.mpf_cos
        mpf_sin = _lbmp.mpf_sin
        mpf_pow = _lbmp.mpf_pow
        exp_fixed = _lbmp.exp_fixed
        cos_sin_fixed = _lbmp.cos_sin_fixed
        log_int_fixed = _lbmp.log_int_fixed
    except (ImportError, AttributeError):
        print("Warning: Sage imports in libelefun failed")
<|endoftext|>"""
Linear algebra
--------------

Linear equations
................

Basic linear algebra is implemented; you can for example solve the linear
equation system::

      x + 2*y = -10
    3*x + 4*y =  10

using ``lu_solve``::

    >>> from mpmath import *
    >>> mp.pretty = False
    >>> A = matrix([[1, 2], [3, 4]])
    >>> b = matrix([-10, 10])
    >>> x = lu_solve(A, b)
    >>> x
    matrix(
    [['30.0'],
     ['-20.0']])

If you don't trust the result, use ``residual`` to calculate the residual ||A*x-b||::

    >>> residual(A, x, b)
    matrix(
    [['3.46944695195361e-18'],
     ['3.46944695195361e-18']])
    >>> str(eps)
    '2.22044604925031e-16'

As you can see, the solution is quite accurate. The error is caused by the
inaccuracy of the internal floating point arithmetic. Though, it's even smaller
than the current machine epsilon, which basically means you can trust the
result.

If you need more speed, use NumPy, or ``fp.lu_solve`` for a floating-point computation.

    >>> fp.lu_solve(A, b)   # doctest: +ELLIPSIS
    matrix(...)

``lu_solve`` accepts overdetermined systems. It is usually not possible to solve
such systems, so the residual is minimized instead. Internally this is done
using Cholesky decomposition to compute a least squares approximation. This means
that that ``lu_solve`` will square the errors. If you can't afford this, use
``qr_solve`` instead. It is twice as slow but more accurate, and it calculates
the residual automatically.


Matrix factorization
....................

The function ``lu`` computes an explicit LU factorization of a matrix::

    >>> P, L, U = lu(matrix([[0,2,3],[4,5,6],[7,8,9]]))
    >>> print(P)
    [0.0  0.0  1.0]
    [1.0  0.0  0.0]
    [0.0  1.0  0.0]
    >>> print(L)
    [              1.0                0.0  0.0]
    [              0.0                1.0  0.0]
    [0.571428571428571  0.214285714285714  1.0]
    >>> print(U)
    [7.0  8.0                9.0]
    [0.0  2.0                3.0]
    [0.0  0.0  0.214285714285714]
    >>> print(P.T*L*U)
    [0.0  2.0  3.0]
    [4.0  5.0  6.0]
    [7.0  8.0  9.0]

Interval matrices
-----------------

Matrices may contain interval elements. This allows one to perform
basic linear algebra operations such as matrix multiplication
and equation solving with rigorous error bounds::

    >>> a = iv.matrix([['0.1','0.3','1.0'],
    ...             ['7.1','5.5','4.8'],
    ...             ['3.2','4.4','5.6']])
    >>>
    >>> b = iv.matrix(['4','0.6','0.5'])
    >>> c = iv.lu_solve(a, b)
    >>> print(c)
    [   [5.2582327113062568605927528666, 5.25823271130625686059275702219]]
    [[-13.1550493962678375411635581388, -13.1550493962678375411635540152]]
    [  [7.42069154774972557628979076189, 7.42069154774972557628979190734]]
    >>> print(a*c)
    [  [3.99999999999999999999999844904, 4.00000000000000000000000155096]]
    [[0.599999999999999999999968898009, 0.600000000000000000000031763736]]
    [[0.499999999999999999999979320485, 0.500000000000000000000020679515]]
"""

# TODO:
# *implement high-level qr()
# *test unitvector
# *iterative solving

from copy import copy

from ..libmp.backend import xrange

class LinearAlgebraMethods(object):

    def LU_decomp(ctx, A, overwrite=False, use_cache=True):
        """
        LU-factorization of a n*n matrix using the Gauss algorithm.
        Returns L and U in one matrix and the pivot indices.

        Use overwrite to specify whether A will be overwritten with L and U.
        """
        if not A.rows == A.cols:
            raise ValueError('need n*n matrix')
        # get from cache if possible
        if use_cache and isinstance(A, ctx.matrix) and A._LU:
            return A._LU
        if not overwrite:
            orig = A
            A = A.copy()
        tol = ctx.absmin(ctx.mnorm(A,1) * ctx.eps) # each pivot element has to be bigger
        n = A.rows
        p = [None]*(n - 1)
        for j in xrange(n - 1):
            # pivoting, choose max(abs(reciprocal row sum)*abs(pivot element))
            biggest = 0
            for k in xrange(j, n):
                s = ctx.fsum([ctx.absmin(A[k,l]) for l in xrange(j, n)])
                if ctx.absmin(s) <= tol:
                    raise ZeroDivisionError('matrix is numerically singular')
                current = 1/s * ctx.absmin(A[k,j])
                if current > biggest: # TODO: what if equal?
                    biggest = current
                    p[j] = k
            # swap rows according to p
            ctx.swap_row(A, j, p[j])
            if ctx.absmin(A[j,j]) <= tol:
                raise ZeroDivisionError('matrix is numerically singular')
            # calculate elimination factors and add rows
            for i in xrange(j + 1, n):
                A[i,j] /= A[j,j]
                for k in xrange(j + 1, n):
                    A[i,k] -= A[i,j]*A[j,k]
        if ctx.absmin(A[n - 1,n - 1]) <= tol:
            raise ZeroDivisionError('matrix is numerically singular')
        # cache decomposition
        if not overwrite and isinstance(orig, ctx.matrix):
            orig._LU = (A, p)
        return A, p

    def L_solve(ctx, L, b, p=None):
        """
        Solve the lower part of a LU factorized matrix for y.
        """
        if L.rows != L.cols:
            raise RuntimeError("need n*n matrix")
        n = L.rows
        if len(b) != n:
            raise ValueError("Value should be equal to n")
        b = copy(b)
        if p: # swap b according to p
            for k in xrange(0, len(p)):
                ctx.swap_row(b, k, p[k])
        # solve
        for i in xrange(1, n):
            for j in xrange(i):
                b[i] -= L[i,j] * b[j]
        return b

    def U_solve(ctx, U, y):
        """
        Solve the upper part of a LU factorized matrix for x.
        """
        if U.rows != U.cols:
            raise RuntimeError("need n*n matrix")
        n = U.rows
        if len(y) != n:
            raise ValueError("Value should be equal to n")
        x = copy(y)
        for i in xrange(n - 1, -1, -1):
            for j in xrange(i + 1, n):
                x[i] -= U[i,j] * x[j]
            x[i] /= U[i,i]
        return x

    def lu_solve(ctx, A, b, **kwargs):
        """
        Ax = b => x

        Solve a determined or overdetermined linear equations system.
        Fast LU decomposition is used, which is less accurate than QR decomposition
        (especially for overdetermined systems), but it's twice as efficient.
        Use qr_solve if you want more precision or have to solve a very ill-
        conditioned system.

        If you specify real=True, it does not check for overdeterminded complex
        systems.
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows < A.cols:
                raise ValueError('cannot solve underdetermined system')
            if A.rows > A.cols:
                # use least-squares method if overdetermined
                # (this increases errors)
                AH = A.H
                A = AH * A
                b = AH * b
                if (kwargs.get('real', False) or
                    not sum(type(i) is ctx.mpc for i in A)):
                    # TODO: necessary to check also b?
                    x = ctx.cholesky_solve(A, b)
                else:
                    x = ctx.lu_solve(A, b)
            else:
                # LU factorization
                A, p = ctx.LU_decomp(A)
                b = ctx.L_solve(A, b, p)
                x = ctx.U_solve(A, b)
        finally:
            ctx.prec = prec
        return x

    def improve_solution(ctx, A, x, b, maxsteps=1):
        """
        Improve a solution to a linear equation system iteratively.

        This re-uses the LU decomposition and is thus cheap.
        Usually 3 up to 4 iterations are giving the maximal improvement.
        """
        if A.rows != A.cols:
            raise RuntimeError("need n*n matrix") # TODO: really?
        for _ in xrange(maxsteps):
            r = ctx.residual(A, x, b)
            if ctx.norm(r, 2) < 10*ctx.eps:
                break
            # this uses cached LU decomposition and is thus cheap
            dx = ctx.lu_solve(A, -r)
            x += dx
        return x

    def lu(ctx, A):
        """
        A -> P, L, U

        LU factorisation of a square matrix A. L is the lower, U the upper part.
        P is the permutation matrix indicating the row swaps.

        P*A = L*U

        If you need efficiency, use the low-level method LU_decomp instead, it's
        much more memory efficient.
        """
        # get factorization
        A, p = ctx.LU_decomp(A)
        n = A.rows
        L = ctx.matrix(n)
        U = ctx.matrix(n)
        for i in xrange(n):
            for j in xrange(n):
                if i > j:
                    L[i,j] = A[i,j]
                elif i == j:
                    L[i,j] = 1
                    U[i,j] = A[i,j]
                else:
                    U[i,j] = A[i,j]
        # calculate permutation matrix
        P = ctx.eye(n)
        for k in xrange(len(p)):
            ctx.swap_row(P, k, p[k])
        return P, L, U

    def unitvector(ctx, n, i):
        """
        Return the i-th n-dimensional unit vector.
        """
        assert 0 < i <= n, 'this unit vector does not exist'
        return [ctx.zero]*(i-1) + [ctx.one] + [ctx.zero]*(n-i)

    def inverse(ctx, A, **kwargs):
        """
        Calculate the inverse of a matrix.

        If you want to solve an equation system Ax = b, it's recommended to use
        solve(A, b) instead, it's about 3 times more efficient.
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A
            A = ctx.matrix(A, **kwargs).copy()
            n = A.rows
            # get LU factorisation
            A, p = ctx.LU_decomp(A)
            cols = []
            # calculate unit vectors and solve corresponding system to get columns
            for i in xrange(1, n + 1):
                e = ctx.unitvector(n, i)
                y = ctx.L_solve(A, e, p)
                cols.append(ctx.U_solve(A, y))
            # convert columns to matrix
            inv = []
            for i in xrange(n):
                row = []
                for j in xrange(n):
                    row.append(cols[j][i])
                inv.append(row)
            result = ctx.matrix(inv, **kwargs)
        finally:
            ctx.prec = prec
        return result

    def householder(ctx, A):
        """
        (A|b) -> H, p, x, res

        (A|b) is the coefficient matrix with left hand side of an optionally
        overdetermined linear equation system.
        H and p contain all information about the transformation matrices.
        x is the solution, res the residual.
        """
        if not isinstance(A, ctx.matrix):
            raise TypeError("A should be a type of ctx.matrix")
        m = A.rows
        n = A.cols
        if m < n - 1:
            raise RuntimeError("Columns should not be less than rows")
        # calculate Householder matrix
        p = []
        for j in xrange(0, n - 1):
            s = ctx.fsum(abs(A[i,j])**2 for i in xrange(j, m))
            if not abs(s) > ctx.eps:
                raise ValueError('matrix is numerically singular')
            p.append(-ctx.sign(ctx.re(A[j,j])) * ctx.sqrt(s))
            kappa = ctx.one / (s - p[j] * A[j,j])
            A[j,j] -= p[j]
            for k in xrange(j+1, n):
                y = ctx.fsum(ctx.conj(A[i,j]) * A[i,k] for i in xrange(j, m)) * kappa
                for i in xrange(j, m):
                    A[i,k] -= A[i,j] * y
        # solve Rx = c1
        x = [A[i,n - 1] for i in xrange(n - 1)]
        for i in xrange(n - 2, -1, -1):
            x[i] -= ctx.fsum(A[i,j] * x[j] for j in xrange(i + 1, n - 1))
            x[i] /= p[i]
        # calculate residual
        if not m == n - 1:
            r = [A[m-1-i, n-1] for i in xrange(m - n + 1)]
        else:
            # determined system, residual should be 0
            r = [0]*m # maybe a bad idea, changing r[i] will change all elements
        return A, p, x, r

    #def qr(ctx, A):
    #    """
    #    A -> Q, R
    #
    #    QR factorisation of a square matrix A using Householder decomposition.
    #    Q is orthogonal, this leads to very few numerical errors.
    #
    #    A = Q*R
    #    """
    #    H, p, x, res = householder(A)
    # TODO: implement this

    def residual(ctx, A, x, b, **kwargs):
        """
        Calculate the residual of a solution to a linear equation system.

        r = A*x - b for A*x = b
        """
        oldprec = ctx.prec
        try:
            ctx.prec *= 2
            A, x, b = ctx.matrix(A, **kwargs), ctx.matrix(x, **kwargs), ctx.matrix(b, **kwargs)
            return A*x - b
        finally:
            ctx.prec = oldprec

    def qr_solve(ctx, A, b, norm=None, **kwargs):
        """
        Ax = b => x, ||Ax - b||

        Solve a determined or overdetermined linear equations system and
        calculate the norm of the residual (error).
        QR decomposition using Householder factorization is applied, which gives very
        accurate results even for ill-conditioned matrices. qr_solve is twice as
        efficient.
        """
        if norm is None:
            norm = ctx.norm
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows < A.cols:
                raise ValueError('cannot solve underdetermined system')
            H, p, x, r = ctx.householder(ctx.extend(A, b))
            res = ctx.norm(r)
            # calculate residual "manually" for determined systems
            if res == 0:
                res = ctx.norm(ctx.residual(A, x, b))
            return ctx.matrix(x, **kwargs), res
        finally:
            ctx.prec = prec

    def cholesky(ctx, A, tol=None):
        r"""
        Cholesky decomposition of a symmetric positive-definite matrix `A`.
        Returns a lower triangular matrix `L` such that `A = L \times L^T`.
        More generally, for a complex Hermitian positive-definite matrix,
        a Cholesky decomposition satisfying `A = L \times L^H` is returned.

        The Cholesky decomposition can be used to solve linear equation
        systems twice as efficiently as LU decomposition, or to
        test whether `A` is positive-definite.

        The optional parameter ``tol`` determines the tolerance for
        verifying positive-definiteness.

        **Examples**

        Cholesky decomposition of a positive-definite symmetric matrix::

            >>> from mpmath import *
            >>> mp.dps = 25; mp.pretty = True
            >>> A = eye(3) + hilbert(3)
            >>> nprint(A)
            [     2.0      0.5  0.333333]
            [     0.5  1.33333      0.25]
            [0.333333     0.25       1.2]
            >>> L = cholesky(A)
            >>> nprint(L)
            [ 1.41421      0.0      0.0]
            [0.353553  1.09924      0.0]
            [0.235702  0.15162  1.05899]
            >>> chop(A - L*L.T)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]

        Cholesky decomposition of a Hermitian matrix::

            >>> A = eye(3) + matrix([[0,0.25j,-0.5j],[-0.25j,0,0],[0.5j,0,0]])
            >>> L = cholesky(A)
            >>> nprint(L)
            [          1.0                0.0                0.0]
            [(0.0 - 0.25j)  (0.968246 + 0.0j)                0.0]
            [ (0.0 + 0.5j)  (0.129099 + 0.0j)  (0.856349 + 0.0j)]
            >>> chop(A - L*L.H)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]

        Attempted Cholesky decomposition of a matrix that is not positive
        definite::

            >>> A = -eye(3) + hilbert(3)
            >>> L = cholesky(A)
            Traceback (most recent call last):
              ...
            ValueError: matrix is not positive-definite

        **References**

        1. [Wikipedia]_ http://en.wikipedia.org/wiki/Cholesky_decomposition

        """
        if not isinstance(A, ctx.matrix):
            raise RuntimeError("A should be a type of ctx.matrix")
        if not A.rows == A.cols:
            raise ValueError('need n*n matrix')
        if tol is None:
            tol = +ctx.eps
        n = A.rows
        L = ctx.matrix(n)
        for j in xrange(n):
            c = ctx.re(A[j,j])
            if abs(c-A[j,j]) > tol:
                raise ValueError('matrix is not Hermitian')
            s = c - ctx.fsum((L[j,k] for k in xrange(j)),
                absolute=True, squared=True)
            if s < tol:
                raise ValueError('matrix is not positive-definite')
            L[j,j] = ctx.sqrt(s)
            for i in xrange(j, n):
                it1 = (L[i,k] for k in xrange(j))
                it2 = (L[j,k] for k in xrange(j))
                t = ctx.fdot(it1, it2, conjugate=True)
                L[i,j] = (A[i,j] - t) / L[j,j]
        return L

    def cholesky_solve(ctx, A, b, **kwargs):
        """
        Ax = b => x

        Solve a symmetric positive-definite linear equation system.
        This is twice as efficient as lu_solve.

        Typical use cases:
        * A.T*A
        * Hessian matrix
        * differential equations
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows !=  A.cols:
                raise ValueError('can only solve determined system')
            # Cholesky factorization
            L = ctx.cholesky(A)
            # solve
            n = L.rows
            if len(b) != n:
                raise ValueError("Value should be equal to n")
            for i in xrange(n):
                b[i] -= ctx.fsum(L[i,j] * b[j] for j in xrange(i))
                b[i] /= L[i,i]
            x = ctx.U_solve(L.T, b)
            return x
        finally:
            ctx.prec = prec

    def det(ctx, A):
        """
        Calculate the determinant of a matrix.
        """
        prec = ctx.prec
        try:
            # do not overwrite A
            A = ctx.matrix(A).copy()
            # use LU factorization to calculate determinant
            try:
                R, p = ctx.LU_decomp(A)
            except ZeroDivisionError:
                return 0
            z = 1
            for i, e in enumerate(p):
                if i != e:
                    z *= -1
            for i in xrange(A.rows):
                z *= R[i,i]
            return z
        finally:
            ctx.prec = prec

    def cond(ctx, A, norm=None):
        """
        Calculate the condition number of a matrix using a specified matrix norm.

        The condition number estimates the sensitivity of a matrix to errors.
        Example: small input errors for ill-conditioned coefficient matrices
        alter the solution of the system dramatically.

        For ill-conditioned matrices it's recommended to use qr_solve() instead
        of lu_solve(). This does not help with input errors however, it just avoids
        to add additional errors.

        Definition:    cond(A) = ||A|| * ||A**-1||
        """
        if norm is None:
            norm = lambda x: ctx.mnorm(x,1)
        return norm(A) * norm(ctx.inverse(A))

    def lu_solve_mat(ctx, a, b):
        """Solve a * x = b  where a and b are matrices."""
        r = ctx.matrix(a.rows, b.cols)
        for i in range(b.cols):
            c = ctx.lu_solve(a, b.column(i))
            for j in range(len(c)):
                r[j, i] = c[j]
        return r

    def qr(ctx, A, mode = 'full', edps = 10):
        """
        Compute a QR factorization $A = QR$ where
        A is an m x n matrix of real or complex numbers where m >= n

        mode has following meanings:
        (1) mode = 'raw' returns two matrixes (A, tau) in the
            internal format used by LAPACK
        (2) mode = 'skinny' returns the leading n columns of Q
            and n rows of R
        (3) Any other value returns the leading m columns of Q
            and m rows of R

        edps is the increase in mp precision used for calculations

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> mp.pretty = True
            >>> A = matrix([[1, 2], [3, 4], [1, 1]])
            >>> Q, R = qr(A)
            >>> Q
            [-0.301511344577764   0.861640436855329   0.408248290463863]
            [-0.904534033733291  -0.123091490979333  -0.408248290463863]
            [-0.301511344577764  -0.492365963917331   0.816496580927726]
            >>> R
            [-3.3166247903554  -4.52267016866645]
            [             0.0  0.738548945875996]
            [             0.0                0.0]
            >>> Q * R
            [1.0  2.0]
            [3.0  4.0]
            [1.0  1.0]
            >>> chop(Q.T * Q)
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]
            >>> B = matrix([[1+0j, 2-3j], [3+j, 4+5j]])
            >>> Q, R = qr(B)
            >>> nprint(Q)
            [     (-0.301511 + 0.0j)   (0.0695795 - 0.95092j)]
            [(-0.904534 - 0.301511j)  (-0.115966 + 0.278318j)]
            >>> nprint(R)
            [(-3.31662 + 0.0j)  (-5.72872 - 2.41209j)]
            [              0.0       (3.91965 + 0.0j)]
            >>> Q * R
            [(1.0 + 0.0j)  (2.0 - 3.0j)]
            [(3.0 + 1.0j)  (4.0 + 5.0j)]
            >>> chop(Q.T * Q.conjugate())
            [1.0  0.0]
            [0.0  1.0]

        """

        # check values before continuing
        assert isinstance(A, ctx.matrix)
        m = A.rows
        n = A.cols
        assert n >= 0
        assert m >= n
        assert edps >= 0

        # check for complex data type
        cmplx = any(type(x) is ctx.mpc for x in A)

        # temporarily increase the precision and initialize
        with ctx.extradps(edps):
            tau = ctx.matrix(n,1)
            A = A.copy()

            # ---------------
            # FACTOR MATRIX A
            # ---------------
            if cmplx:
                one = ctx.mpc('1.0', '0.0')
                zero = ctx.mpc('0.0', '0.0')
                rzero = ctx.mpf('0.0')

                # main loop to factor A (complex)
                for j in xrange(0, n):
                    alpha = A[j,j]
                    alphr = ctx.re(alpha)
                    alphi = ctx.im(alpha)

                    if (m-j) >= 2:
                        xnorm = ctx.fsum( A[i,j]*ctx.conj(A[i,j]) for i in xrange(j+1, m) )
                        xnorm = ctx.re( ctx.sqrt(xnorm) )
                    else:
                        xnorm = rzero

                    if (xnorm == rzero) and (alphi == rzero):
                        tau[j] = zero
                        continue

                    if alphr < rzero:
                        beta = ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)
                    else:
                        beta = -ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)

                    tau[j] = ctx.mpc( (beta - alphr) / beta, -alphi / beta )
                    t = -ctx.conj(tau[j])
                    za = one / (alpha - beta)

                    for i in xrange(j+1, m):
                        A[i,j] *= za

                    A[j,j] = one
                    for k in xrange(j+1, n):
                        y = ctx.fsum(A[i,j] * ctx.conj(A[i,k]) for i in xrange(j, m))
                        temp = t * ctx.conj(y)
                        for i in xrange(j, m):
                            A[i,k] += A[i,j] * temp

                    A[j,j] = ctx.mpc(beta, '0.0')
            else:
                one = ctx.mpf('1.0')
                zero = ctx.mpf('0.0')

                # main loop to factor A (real)
                for j in xrange(0, n):
                    alpha = A[j,j]

                    if (m-j) > 2:
                        xnorm = ctx.fsum( (A[i,j])**2 for i in xrange(j+1, m) )
                        xnorm = ctx.sqrt(xnorm)
                    elif (m-j) == 2:
                        xnorm = abs( A[m-1,j] )
                    else:
                        xnorm = zero

                    if xnorm == zero:
                        tau[j] = zero
                        continue

                    if alpha < zero:
                        beta = ctx.sqrt(alpha**2 + xnorm**2)
                    else:
                        beta = -ctx.sqrt(alpha**2 + xnorm**2)

                    tau[j] = (beta - alpha) / beta
                    t = -tau[j]
                    da = one / (alpha - beta)

                    for i in xrange(j+1, m):
                        A[i,j] *= da

                    A[j,j] = one
                    for k in xrange(j+1, n):
                        y = ctx.fsum( A[i,j] * A[i,k] for i in xrange(j, m) )
                        temp = t * y
                        for i in xrange(j,m):
                            A[i,k] += A[i,j] * temp

                    A[j,j] = beta

            # return factorization in same internal format as LAPACK
            if (mode == 'raw') or (mode == 'RAW'):
                return A, tau

            # ----------------------------------
            # FORM Q USING BACKWARD ACCUMULATION
            # ----------------------------------

            # form R before the values are overwritten
            R = A.copy()
            for j in xrange(0, n):
                for i in xrange(j+1, m):
                    R[i,j] = zero

            # set the value of p (number of columns of Q to return)
            p = m
            if (mode == 'skinny') or (mode == 'SKINNY'):
                p = n

            # add columns to A if needed and initialize
            A.cols += (p-n)
            for j in xrange(0, p):
                A[j,j] = one
                for i in xrange(0, j):
                    A[i,j] = zero

            # main loop to form Q
            for j in xrange(n-1, -1, -1):
                t = -tau[j]
                A[j,j] += t

                for k in xrange(j+1, p):
                    if cmplx:
                        y = ctx.fsum(A[i,j] * ctx.conj(A[i,k]) for i in xrange(j+1, m))
                        temp = t * ctx.conj(y)
                    else:
                        y = ctx.fsum(A[i,j] * A[i,k] for i in xrange(j+1, m))
                        temp = t * y
                    A[j,k] = temp
                    for i in xrange(j+1, m):
                        A[i,k] += A[i,j] * temp

                for i in xrange(j+1, m):
                    A[i, j] *= t

            return A, R[0:p,0:n]

        # ------------------
        # END OF FUNCTION QR
        # ------------------
<|endoftext|>from ..libmp.backend import xrange

# TODO: should use diagonalization-based algorithms

class MatrixCalculusMethods(object):

    def _exp_pade(ctx, a):
        """
        Exponential of a matrix using Pade approximants.

        See G. H. Golub, C. F. van Loan 'Matrix Computations',
        third Ed., page 572

        TODO:
         - find a good estimate for q
         - reduce the number of matrix multiplications to improve
           performance
        """
        def eps_pade(p):
            return ctx.mpf(2)**(3-2*p) * \
                ctx.factorial(p)**2/(ctx.factorial(2*p)**2 * (2*p + 1))
        q = 4
        extraq = 8
        while 1:
            if eps_pade(q) < ctx.eps:
                break
            q += 1
        q += extraq
        j = int(max(1, ctx.mag(ctx.mnorm(a,'inf'))))
        extra = q
        prec = ctx.prec
        ctx.dps += extra + 3
        try:
            a = a/2**j
            na = a.rows
            den = ctx.eye(na)
            num = ctx.eye(na)
            x = ctx.eye(na)
            c = ctx.mpf(1)
            for k in range(1, q+1):
                c *= ctx.mpf(q - k + 1)/((2*q - k + 1) * k)
                x = a*x
                cx = c*x
                num += cx
                den += (-1)**k * cx
            f = ctx.lu_solve_mat(den, num)
            for k in range(j):
                f = f*f
        finally:
            ctx.prec = prec
        return f*1

    def expm(ctx, A, method='taylor'):
        r"""
        Computes the matrix exponential of a square matrix `A`, which is defined
        by the power series

        .. math ::

            \exp(A) = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \ldots

        With method='taylor', the matrix exponential is computed
        using the Taylor series. With method='pade', Pade approximants
        are used instead.

        **Examples**

        Basic examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> expm(zeros(3))
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]
            >>> expm(eye(3))
            [2.71828182845905               0.0               0.0]
            [             0.0  2.71828182845905               0.0]
            [             0.0               0.0  2.71828182845905]
            >>> expm([[1,1,0],[1,0,1],[0,1,0]])
            [ 3.86814500615414  2.26812870852145  0.841130841230196]
            [ 2.26812870852145  2.44114713886289   1.42699786729125]
            [0.841130841230196  1.42699786729125    1.6000162976327]
            >>> expm([[1,1,0],[1,0,1],[0,1,0]], method='pade')
            [ 3.86814500615414  2.26812870852145  0.841130841230196]
            [ 2.26812870852145  2.44114713886289   1.42699786729125]
            [0.841130841230196  1.42699786729125    1.6000162976327]
            >>> expm([[1+j, 0], [1+j,1]])
            [(1.46869393991589 + 2.28735528717884j)                        0.0]
            [  (1.03776739863568 + 3.536943175722j)  (2.71828182845905 + 0.0j)]

        Matrices with large entries are allowed::

            >>> expm(matrix([[1,2],[2,3]])**25)
            [5.65024064048415e+2050488462815550  9.14228140091932e+2050488462815550]
            [9.14228140091932e+2050488462815550  1.47925220414035e+2050488462815551]

        The identity `\exp(A+B) = \exp(A) \exp(B)` does not hold for
        noncommuting matrices::

            >>> A = hilbert(3)
            >>> B = A + eye(3)
            >>> chop(mnorm(A*B - B*A))
            0.0
            >>> chop(mnorm(expm(A+B) - expm(A)*expm(B)))
            0.0
            >>> B = A + ones(3)
            >>> mnorm(A*B - B*A)
            1.8
            >>> mnorm(expm(A+B) - expm(A)*expm(B))
            42.0927851137247

        """
        if method == 'pade':
            prec = ctx.prec
            try:
                A = ctx.matrix(A)
                ctx.prec += 2*A.rows
                res = ctx._exp_pade(A)
            finally:
                ctx.prec = prec
            return res
        A = ctx.matrix(A)
        prec = ctx.prec
        j = int(max(1, ctx.mag(ctx.mnorm(A,'inf'))))
        j += int(0.5*prec**0.5)
        try:
            ctx.prec += 10 + 2*j
            tol = +ctx.eps
            A = A/2**j
            T = A
            Y = A**0 + A
            k = 2
            while 1:
                T *= A * (1/ctx.mpf(k))
                if ctx.mnorm(T, 'inf') < tol:
                    break
                Y += T
                k += 1
            for k in xrange(j):
                Y = Y*Y
        finally:
            ctx.prec = prec
        Y *= 1
        return Y

    def cosm(ctx, A):
        r"""
        Gives the cosine of a square matrix `A`, defined in analogy
        with the matrix exponential.

        Examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> cosm(X)
            [0.54030230586814               0.0               0.0]
            [             0.0  0.54030230586814               0.0]
            [             0.0               0.0  0.54030230586814]
            >>> X = hilbert(3)
            >>> cosm(X)
            [ 0.424403834569555  -0.316643413047167  -0.221474945949293]
            [-0.316643413047167   0.820646708837824  -0.127183694770039]
            [-0.221474945949293  -0.127183694770039   0.909236687217541]
            >>> X = matrix([[1+j,-2],[0,-j]])
            >>> cosm(X)
            [(0.833730025131149 - 0.988897705762865j)  (1.07485840848393 - 0.17192140544213j)]
            [                                     0.0               (1.54308063481524 + 0.0j)]
        """
        B = 0.5 * (ctx.expm(A*ctx.j) + ctx.expm(A*(-ctx.j)))
        if not sum(A.apply(ctx.im).apply(abs)):
            B = B.apply(ctx.re)
        return B

    def sinm(ctx, A):
        r"""
        Gives the sine of a square matrix `A`, defined in analogy
        with the matrix exponential.

        Examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> sinm(X)
            [0.841470984807897                0.0                0.0]
            [              0.0  0.841470984807897                0.0]
            [              0.0                0.0  0.841470984807897]
            >>> X = hilbert(3)
            >>> sinm(X)
            [0.711608512150994  0.339783913247439  0.220742837314741]
            [0.339783913247439  0.244113865695532  0.187231271174372]
            [0.220742837314741  0.187231271174372  0.155816730769635]
            >>> X = matrix([[1+j,-2],[0,-j]])
            >>> sinm(X)
            [(1.29845758141598 + 0.634963914784736j)  (-1.96751511930922 + 0.314700021761367j)]
            [                                    0.0                  (0.0 - 1.1752011936438j)]
        """
        B = (-0.5j) * (ctx.expm(A*ctx.j) - ctx.expm(A*(-ctx.j)))
        if not sum(A.apply(ctx.im).apply(abs)):
            B = B.apply(ctx.re)
        return B

    def _sqrtm_rot(ctx, A, _may_rotate):
        # If the iteration fails to converge, cheat by performing
        # a rotation by a complex number
        u = ctx.j**0.3
        return ctx.sqrtm(u*A, _may_rotate) / ctx.sqrt(u)

    def sqrtm(ctx, A, _may_rotate=2):
        r"""
        Computes a square root of the square matrix `A`, i.e. returns
        a matrix `B = A^{1/2}` such that `B^2 = A`. The square root
        of a matrix, if it exists, is not unique.

        **Examples**

        Square roots of some simple matrices::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> sqrtm([[1,0], [0,1]])
            [1.0  0.0]
            [0.0  1.0]
            >>> sqrtm([[0,0], [0,0]])
            [0.0  0.0]
            [0.0  0.0]
            >>> sqrtm([[2,0],[0,1]])
            [1.4142135623731  0.0]
            [            0.0  1.0]
            >>> sqrtm([[1,1],[1,0]])
            [ (0.920442065259926 - 0.21728689675164j)  (0.568864481005783 + 0.351577584254143j)]
            [(0.568864481005783 + 0.351577584254143j)  (0.351577584254143 - 0.568864481005783j)]
            >>> sqrtm([[1,0],[0,1]])
            [1.0  0.0]
            [0.0  1.0]
            >>> sqrtm([[-1,0],[0,1]])
            [(0.0 - 1.0j)           0.0]
            [         0.0  (1.0 + 0.0j)]
            >>> sqrtm([[j,0],[0,j]])
            [(0.707106781186547 + 0.707106781186547j)                                       0.0]
            [                                     0.0  (0.707106781186547 + 0.707106781186547j)]

        A square root of a rotation matrix, giving the corresponding
        half-angle rotation matrix::

            >>> t1 = 0.75
            >>> t2 = t1 * 0.5
            >>> A1 = matrix([[cos(t1), -sin(t1)], [sin(t1), cos(t1)]])
            >>> A2 = matrix([[cos(t2), -sin(t2)], [sin(t2), cos(t2)]])
            >>> sqrtm(A1)
            [0.930507621912314  -0.366272529086048]
            [0.366272529086048   0.930507621912314]
            >>> A2
            [0.930507621912314  -0.366272529086048]
            [0.366272529086048   0.930507621912314]

        The identity `(A^2)^{1/2} = A` does not necessarily hold::

            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> sqrtm(A**2)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> sqrtm(A)**2
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> A = matrix([[-4,1,4],[7,-8,9],[10,2,11]])
            >>> sqrtm(A**2)
            [  7.43715112194995  -0.324127569985474   1.8481718827526]
            [-0.251549715716942    9.32699765900402  2.48221180985147]
            [  4.11609388833616   0.775751877098258   13.017955697342]
            >>> chop(sqrtm(A)**2)
            [-4.0   1.0   4.0]
            [ 7.0  -8.0   9.0]
            [10.0   2.0  11.0]

        For some matrices, a square root does not exist::

            >>> sqrtm([[0,1], [0,0]])
            Traceback (most recent call last):
              ...
            ZeroDivisionError: matrix is numerically singular

        Two examples from the documentation for Matlab's ``sqrtm``::

            >>> mp.dps = 15; mp.pretty = True
            >>> sqrtm([[7,10],[15,22]])
            [1.56669890360128  1.74077655955698]
            [2.61116483933547  4.17786374293675]
            >>>
            >>> X = matrix(\
            ...   [[5,-4,1,0,0],
            ...   [-4,6,-4,1,0],
            ...   [1,-4,6,-4,1],
            ...   [0,1,-4,6,-4],
            ...   [0,0,1,-4,5]])
            >>> Y = matrix(\
            ...   [[2,-1,-0,-0,-0],
            ...   [-1,2,-1,0,-0],
            ...   [0,-1,2,-1,0],
            ...   [-0,0,-1,2,-1],
            ...   [-0,-0,-0,-1,2]])
            >>> mnorm(sqrtm(X) - Y)
            4.53155328326114e-19

        """
        A = ctx.matrix(A)
        # Trivial
        if A*0 == A:
            return A
        prec = ctx.prec
        if _may_rotate:
            d = ctx.det(A)
            if abs(ctx.im(d)) < 16*ctx.eps and ctx.re(d) < 0:
                return ctx._sqrtm_rot(A, _may_rotate-1)
        try:
            ctx.prec += 10
            tol = ctx.eps * 128
            Y = A
            Z = I = A**0
            k = 0
            # Denman-Beavers iteration
            while 1:
                Yprev = Y
                try:
                    Y, Z = 0.5*(Y+ctx.inverse(Z)), 0.5*(Z+ctx.inverse(Y))
                except ZeroDivisionError:
                    if _may_rotate:
                        Y = ctx._sqrtm_rot(A, _may_rotate-1)
                        break
                    else:
                        raise
                mag1 = ctx.mnorm(Y-Yprev, 'inf')
                mag2 = ctx.mnorm(Y, 'inf')
                if mag1 <= mag2*tol:
                    break
                if _may_rotate and k > 6 and not mag1 < mag2 * 0.001:
                    return ctx._sqrtm_rot(A, _may_rotate-1)
                k += 1
                if k > ctx.prec:
                    raise ctx.NoConvergence
        finally:
            ctx.prec = prec
        Y *= 1
        return Y

    def logm(ctx, A):
        r"""
        Computes a logarithm of the square matrix `A`, i.e. returns
        a matrix `B = \log(A)` such that `\exp(B) = A`. The logarithm
        of a matrix, if it exists, is not unique.

        **Examples**

        Logarithms of some simple matrices::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> logm(X)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            >>> logm(2*X)
            [0.693147180559945                0.0                0.0]
            [              0.0  0.693147180559945                0.0]
            [              0.0                0.0  0.693147180559945]
            >>> logm(expm(X))
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]

        A logarithm of a complex matrix::

            >>> X = matrix([[2+j, 1, 3], [1-j, 1-2*j, 1], [-4, -5, j]])
            >>> B = logm(X)
            >>> nprint(B)
            [ (0.808757 + 0.107759j)    (2.20752 + 0.202762j)   (1.07376 - 0.773874j)]
            [ (0.905709 - 0.107795j)  (0.0287395 - 0.824993j)  (0.111619 + 0.514272j)]
            [(-0.930151 + 0.399512j)   (-2.06266 - 0.674397j)  (0.791552 + 0.519839j)]
            >>> chop(expm(B))
            [(2.0 + 1.0j)           1.0           3.0]
            [(1.0 - 1.0j)  (1.0 - 2.0j)           1.0]
            [        -4.0          -5.0  (0.0 + 1.0j)]

        A matrix `X` close to the identity matrix, for which
        `\log(\exp(X)) = \exp(\log(X)) = X` holds::

            >>> X = eye(3) + hilbert(3)/4
            >>> X
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]
            >>> logm(expm(X))
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]
            >>> expm(logm(X))
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]

        A logarithm of a rotation matrix, giving back the angle of
        the rotation::

            >>> t = 3.7
            >>> A = matrix([[cos(t),sin(t)],[-sin(t),cos(t)]])
            >>> chop(logm(A))
            [             0.0  -2.58318530717959]
            [2.58318530717959                0.0]
            >>> (2*pi-t)
            2.58318530717959

        For some matrices, a logarithm does not exist::

            >>> logm([[1,0], [0,0]])
            Traceback (most recent call last):
              ...
            ZeroDivisionError: matrix is numerically singular

        Logarithm of a matrix with large entries::

            >>> logm(hilbert(3) * 10**20).apply(re)
            [ 45.5597513593433  1.27721006042799  0.317662687717978]
            [ 1.27721006042799  42.5222778973542   2.24003708791604]
            [0.317662687717978  2.24003708791604    42.395212822267]

        """
        A = ctx.matrix(A)
        prec = ctx.prec
        try:
            ctx.prec += 10
            tol = ctx.eps * 128
            I = A**0
            B = A
            n = 0
            while 1:
                B = ctx.sqrtm(B)
                n += 1
                if ctx.mnorm(B-I, 'inf') < 0.125:
                    break
            T = X = B-I
            L = X*0
            k = 1
            while 1:
                if k & 1:
                    L += T / k
                else:
                    L -= T / k
                T *= X
                if ctx.mnorm(T, 'inf') < tol:
                    break
                k += 1
                if k > ctx.prec:
                    raise ctx.NoConvergence
        finally:
            ctx.prec = prec
        L *= 2**n
        return L

    def powm(ctx, A, r):
        r"""
        Computes `A^r = \exp(A \log r)` for a matrix `A` and complex
        number `r`.

        **Examples**

        Powers and inverse powers of a matrix::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> powm(A, 2)
            [ 63.0  20.0   69.0]
            [174.0  89.0  199.0]
            [164.0  48.0  179.0]
            >>> chop(powm(powm(A, 4), 1/4.))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(20)(powm)(A, -4), -1/4.)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> chop(powm(powm(A, 1+0.5j), 1/(1+0.5j)))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(5)(powm)(A, -1.5), -1/(1.5))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]

        A Fibonacci-generating matrix::

            >>> powm([[1,1],[1,0]], 10)
            [89.0  55.0]
            [55.0  34.0]
            >>> fib(10)
            55.0
            >>> powm([[1,1],[1,0]], 6.5)
            [(16.5166626964253 - 0.0121089837381789j)  (10.2078589271083 + 0.0195927472575932j)]
            [(10.2078589271083 + 0.0195927472575932j)  (6.30880376931698 - 0.0317017309957721j)]
            >>> (phi**6.5 - (1-phi)**6.5)/sqrt(5)
            (10.2078589271083 - 0.0195927472575932j)
            >>> powm([[1,1],[1,0]], 6.2)
            [ (14.3076953002666 - 0.008222855781077j)  (8.81733464837593 + 0.0133048601383712j)]
            [(8.81733464837593 + 0.0133048601383712j)  (5.49036065189071 - 0.0215277159194482j)]
            >>> (phi**6.2 - (1-phi)**6.2)/sqrt(5)
            (8.81733464837593 - 0.0133048601383712j)

        """
        A = ctx.matrix(A)
        r = ctx.convert(r)
        prec = ctx.prec
        try:
            ctx.prec += 10
            if ctx.isint(r):
                v = A ** int(r)
            elif ctx.isint(r*2):
                y = int(r*2)
                v = ctx.sqrtm(A) ** y
            else:
                v = ctx.expm(r*ctx.logm(A))
        finally:
            ctx.prec = prec
        v *= 1
        return v
<|endoftext|>#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################
#     module for the symmetric eigenvalue problem
#       Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
#
# todo:
#  - implement balancing
#
##################################################################################################

"""
The symmetric eigenvalue problem.
---------------------------------

This file contains routines for the symmetric eigenvalue problem.

high level routines:

  eigsy : real symmetric (ordinary) eigenvalue problem
  eighe : complex hermitian (ordinary) eigenvalue problem
  eigh  : unified interface for eigsy and eighe
  svd_r : singular value decomposition for real matrices
  svd_c : singular value decomposition for complex matrices
  svd   : unified interface for svd_r and svd_c


low level routines:

  r_sy_tridiag : reduction of real symmetric matrix to real symmetric tridiagonal matrix
  c_he_tridiag_0 : reduction of complex hermitian matrix to real symmetric tridiagonal matrix
  c_he_tridiag_1 : auxiliary routine to c_he_tridiag_0
  c_he_tridiag_2 : auxiliary routine to c_he_tridiag_0
  tridiag_eigen : solves the real symmetric tridiagonal matrix eigenvalue problem
  svd_r_raw : raw singular value decomposition for real matrices
  svd_c_raw : raw singular value decomposition for complex matrices
"""

from ..libmp.backend import xrange
from .eigen import defun


def r_sy_tridiag(ctx, A, D, E, calc_ev = True):
    """
    This routine transforms a real symmetric matrix A to a real symmetric
    tridiagonal matrix T using an orthogonal similarity transformation:
          Q' * A * Q = T     (here ' denotes the matrix transpose).
    The orthogonal matrix Q is build up from Householder reflectors.

    parameters:
      A         (input/output) On input, A contains the real symmetric matrix of
                dimension (n,n). On output, if calc_ev is true, A contains the
                orthogonal matrix Q, otherwise A is destroyed.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      calc_ev   (input) If calc_ev is true, this routine explicitly calculates the
                orthogonal matrix Q which is then returned in A. If calc_ev is
                false, Q is not explicitly calculated resulting in a shorter run time.

    This routine is a python translation of the fortran routine tred2.f in the
    software library EISPACK (see netlib.org) which itself is based on the algol
    procedure tred2 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """

    # note : the vector v of the i-th houshoulder reflector is stored in a[(i+1):,i]
    #        whereas v/<v,v> is stored in a[i,(i+1):]

    n = A.rows
    for i in xrange(n - 1, 0, -1):
        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(A[k,i])

        scale_inv = 0
        if scale != 0:
            scale_inv = 1/scale

        # sadly there are floating point numbers not equal to zero whose reciprocal is infinity

        if i == 1 or scale == 0 or ctx.isinf(scale_inv):
            E[i] = A[i-1,i]        # nothing to do
            D[i] = 0
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[k,i] *= scale_inv
            H += A[k,i] * A[k,i]

        F = A[i-1,i]
        G = ctx.sqrt(H)
        if F > 0:
            G = -G
        E[i] = scale * G
        H -= F * G
        A[i-1,i] = F - G
        F = 0

        # apply housholder transformation

        for j in xrange(0, i):
            if calc_ev:
                A[i,j] = A[j,i] / H

            G = 0                  # calculate A*U
            for k in xrange(0, j + 1):
                G += A[k,j] * A[k,i]
            for k in xrange(j + 1, i):
                G += A[j,k] * A[k,i]

            E[j] = G / H           # calculate P
            F += E[j] * A[j,i]

        HH = F / (2 * H)

        for j in xrange(0, i):     # calculate reduced A
            F = A[j,i]
            G = E[j] - HH * F      # calculate Q
            E[j] = G

            for k in xrange(0, j + 1):
                A[k,j] -= F * E[k] + G * A[k,i]

        D[i] = H

    for i in xrange(1, n):         # better for compatibility
        E[i-1] = E[i]
    E[n-1] = 0

    if calc_ev:
        D[0] = 0
        for i in xrange(0, n):
            if D[i] != 0:
                for j in xrange(0, i):     # accumulate transformation matrices
                    G = 0
                    for k in xrange(0, i):
                        G += A[i,k] * A[k,j]
                    for k in xrange(0, i):
                        A[k,j] -= G * A[k,i]

            D[i] = A[i,i]
            A[i,i] = 1

            for j in xrange(0, i):
                A[j,i] = A[i,j] = 0
    else:
        for i in xrange(0, n):
            D[i] = A[i,i]





def c_he_tridiag_0(ctx, A, D, E, T):
    """
    This routine transforms a complex hermitian matrix A to a real symmetric
    tridiagonal matrix T using an unitary similarity transformation:
          Q' * A * Q = T     (here ' denotes the hermitian matrix transpose,
                              i.e. transposition und conjugation).
    The unitary matrix Q is build up from Householder reflectors and
    an unitary diagonal matrix.

    parameters:
      A         (input/output) On input, A contains the complex hermitian matrix
                of dimension (n,n). On output, A contains the unitary matrix Q
                in compressed form.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix.

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      T         (output) complex array of length n, contains a unitary diagonal
                matrix.

    This routine is a python translation (in slightly modified form) of the fortran
    routine htridi.f in the software library EISPACK (see netlib.org) which itself
    is a complex version of the algol procedure tred1 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """

    n = A.rows
    T[n-1] = 1
    for i in xrange(n - 1, 0, -1):

        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[k,i])) + abs(ctx.im(A[k,i]))

        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale

        # sadly there are floating point numbers not equal to zero whose reciprocal is infinity

        if scale == 0 or ctx.isinf(scale_inv):
            E[i] = 0
            D[i] = 0
            T[i-1] = 1
            continue

        if i == 1:
            F = A[i-1,i]
            f = abs(F)
            E[i] = f
            D[i] = 0
            if f != 0:
                T[i-1] = T[i] * F / f
            else:
                T[i-1] = T[i]
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[k,i] *= scale_inv
            rr = ctx.re(A[k,i])
            ii = ctx.im(A[k,i])
            H += rr * rr + ii * ii

        F = A[i-1,i]
        f = abs(F)
        G = ctx.sqrt(H)
        H += G * f
        E[i] = scale * G
        if f != 0:
            F = F / f
            TZ = - T[i] * F              # T[i-1]=-T[i]*F, but we need T[i-1] as temporary storage
            G *= F
        else:
            TZ = -T[i]                   # T[i-1]=-T[i]
        A[i-1,i] += G
        F = 0

        # apply housholder transformation

        for j in xrange(0, i):
            A[i,j] = A[j,i] / H

            G = 0                        # calculate A*U
            for k in xrange(0, j + 1):
                G += ctx.conj(A[k,j]) * A[k,i]
            for k in xrange(j + 1, i):
                G += A[j,k] * A[k,i]

            T[j] = G / H                 # calculate P
            F += ctx.conj(T[j]) * A[j,i]

        HH = F / (2 * H)

        for j in xrange(0, i):           # calculate reduced A
            F = A[j,i]
            G = T[j] - HH * F            # calculate Q
            T[j] = G

            for k in xrange(0, j + 1):
                A[k,j] -= ctx.conj(F) * T[k] + ctx.conj(G) * A[k,i]
                # as we use the lower left part for storage
                # we have to use the transpose of the normal formula

        T[i-1] = TZ
        D[i] = H

    for i in xrange(1, n):                # better for compatibility
        E[i-1] = E[i]
    E[n-1] = 0

    D[0] = 0
    for i in xrange(0, n):
        zw = D[i]
        D[i] = ctx.re(A[i,i])
        A[i,i] = zw







def c_he_tridiag_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in c_he_tridiag_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           c_he_tridiag_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

    """

    n = A.rows

    for i in xrange(0, n):
        if A[i,i] != 0:
            for j in xrange(0, i):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i,k]) * A[k,j]
                for k in xrange(0, i):
                    A[k,j] -= G * A[k,i]

        A[i,i] = 1

        for j in xrange(0, i):
            A[j,i] = A[i,j] = 0

    for i in xrange(0, n):
        for k in xrange(0, n):
            A[i,k] *= T[k]




def c_he_tridiag_2(ctx, A, T, B):
    """
    This routine applied the unitary matrix Q described in c_he_tridiag_0
    onto the the matrix B, i.e. it forms Q*B.

    parameters:
      A    (input) On input, A is the same matrix as delivered by c_he_tridiag_0.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

      B    (input/output) On input, B is a complex matrix. On output B is replaced
           by Q*B.

    This routine is a python translation of the fortran routine htribk.f in the
    software library EISPACK (see netlib.org). See c_he_tridiag_0 for more
    references.
    """

    n = A.rows

    for i in xrange(0, n):
        for k in xrange(0, n):
            B[k,i] *= T[k]

    for i in xrange(0, n):
        if A[i,i] != 0:
            for j in xrange(0, n):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i,k]) * B[k,j]
                for k in xrange(0, i):
                    B[k,j] -= G * A[k,i]





def tridiag_eigen(ctx, d, e, z = False):
    """
    This subroutine find the eigenvalues and the first components of the
    eigenvectors of a real symmetric tridiagonal matrix using the implicit
    QL method.

    parameters:

      d (input/output) real array of length n. on input, d contains the diagonal
        elements of the input matrix. on output, d contains the eigenvalues in
        ascending order.

      e (input) real array of length n. on input, e contains the offdiagonal
        elements of the input matrix in e[0:(n-1)]. On output, e has been
        destroyed.

      z (input/output) If z is equal to False, no eigenvectors will be computed.
        Otherwise on input z should have the format z[0:m,0:n] (i.e. a real or
        complex matrix of dimension (m,n) ). On output this matrix will be
        multiplied by the matrix of the eigenvectors (i.e. the columns of this
        matrix are the eigenvectors): z --> z*EV
        That means if z[i,j]={1 if j==j; 0 otherwise} on input, then on output
        z will contain the first m components of the eigenvectors. That means
        if m is equal to n, the i-th eigenvector will be z[:,i].

    This routine is a python translation (in slightly modified form) of the
    fortran routine imtql2.f in the software library EISPACK (see netlib.org)
    which itself is based on the algol procudure imtql2 desribed in:
     - num. math. 12, p. 377-383(1968) by matrin and wilkinson
     - modified in num. math. 15, p. 450(1970) by dubrulle
     - handbook for auto. comp., vol. II-linear algebra, p. 241-248 (1971)
    See also the routine gaussq.f in netlog.org or acm algorithm 726.
    """

    n = len(d)
    e[n-1] = 0
    iterlim = 2 * ctx.dps

    for l in xrange(n):
        j = 0
        while 1:
            m = l
            while 1:
                # look for a small subdiagonal element
                if m + 1 == n:
                    break
                if abs(e[m]) <= ctx.eps * (abs(d[m]) + abs(d[m + 1])):
                    break
                m = m + 1
            if m == l:
                break

            if j >= iterlim:
                raise RuntimeError("tridiag_eigen: no convergence to an eigenvalue after %d iterations" % iterlim)

            j += 1

            # form shift

            p = d[l]
            g = (d[l + 1] - p) / (2 * e[l])
            r = ctx.hypot(g, 1)

            if g < 0:
                s = g - r
            else:
                s = g + r

            g = d[m] - p + e[l] / s

            s, c, p = 1, 1, 0

            for i in xrange(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) > abs(g):             # this here is a slight improvement also used in gaussq.f or acm algorithm 726.
                    c = g / f
                    r = ctx.hypot(c, 1)
                    e[i + 1] = f * r
                    s = 1 / r
                    c = c * s
                else:
                    s = f / g
                    r = ctx.hypot(s, 1)
                    e[i + 1] = g * r
                    c = 1 / r
                    s = s * c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                if not isinstance(z, bool):
                    # calculate eigenvectors
                    for w in xrange(z.rows):
                        f = z[w,i+1]
                        z[w,i+1] = s * z[w,i] + c * f
                        z[w,i  ] = c * z[w,i] - s * f

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0

    for ii in xrange(1, n):
        # sort eigenvalues and eigenvectors (bubble-sort)
        i = ii - 1
        k = i
        p = d[i]
        for j in xrange(ii, n):
            if d[j] >= p:
                continue
            k = j
            p = d[k]
        if k == i:
            continue
        d[k] = d[i]
        d[i] = p

        if not isinstance(z, bool):
            for w in xrange(z.rows):
                p = z[w,i]
                z[w,i] = z[w,k]
                z[w,k] = p

########################################################################################

@defun
def eigsy(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    square matrix A. Given A, an orthogonal matrix Q is calculated which
    diagonalizes A:

          Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) is a diagonal matrix whose diagonal is E.
    ' denotes the transpose.

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

          A Q[:,i] = E[i] Q[:,i]


    input:

      A: real matrix of format (n,n) which is symmetric
         (i.e. A=A' or A[i,j]=A[j,i])

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: orthogonal matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

          E          if eigvals_only is true
         (E, Q)      if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigsy(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigsy(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eighe, eigh, eig
    """

    if not overwrite_a:
        A = A.copy()

    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)

    if eigvals_only:
        r_sy_tridiag(ctx, A, d, e, calc_ev = False)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        r_sy_tridiag(ctx, A, d, e, calc_ev = True)
        tridiag_eigen(ctx, d, e, A)
        return (d, A)


@defun
def eighe(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    This routine solves the (ordinary) eigenvalue problem for a complex
    hermitian square matrix A. Given A, an unitary matrix Q is calculated which
    diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]


    input:

      A: complex matrix of format (n,n) which is hermitian
         (i.e. A=A' or A[i,j]=conj(A[j,i]))

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: unitary matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

           E         if eigvals_only is true
          (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[1, -3 - 1j], [-3 + 1j, -2]])
      >>> E = mp.eighe(A, eigvals_only = True)
      >>> print(E)
      [-4.0]
      [ 3.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eighe(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eigh, eig
    """

    if not overwrite_a:
        A = A.copy()

    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)
    t = ctx.zeros(A.rows, 1)

    if eigvals_only:
        c_he_tridiag_0(ctx, A, d, e, t)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        c_he_tridiag_0(ctx, A, d, e, t)
        B = ctx.eye(A.rows)
        tridiag_eigen(ctx, d, e, B)
        c_he_tridiag_2(ctx, A, t, B)
        return (d, B)

@defun
def eigh(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    "eigh" is a unified interface for "eigsy" and "eighe". Depending on
    whether A is real or complex the appropriate function is called.

    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    or complex hermitian square matrix A. Given A, an orthogonal (A real) or
    unitary (A complex) matrix Q is calculated which diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]

    input:

      A: a real or complex square matrix of format (n,n) which is symmetric
         (i.e. A[i,j]=A[j,i]) or hermitian (i.e. A[i,j]=conj(A[j,i])).

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: an orthogonal or unitary matrix of format (n,n). contains the
         eigenvectors of A as columns.

    return value:

          E         if eigvals_only is true
         (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigh(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigh(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eigh(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eighe, eig
    """

    iscomplex = any(type(x) is ctx.mpc for x in A)

    if iscomplex:
        return ctx.eighe(A, eigvals_only = eigvals_only, overwrite_a = overwrite_a)
    else:
        return ctx.eigsy(A, eigvals_only = eigvals_only, overwrite_a = overwrite_a)


@defun
def gauss_quadrature(ctx, n, qtype = "legendre", alpha = 0, beta = 0):
    """
    This routine calulates gaussian quadrature rules for different
    families of orthogonal polynomials. Let (a, b) be an interval,
    W(x) a positive weight function and n a positive integer.
    Then the purpose of this routine is to calculate pairs (x_k, w_k)
    for k=0, 1, 2, ... (n-1) which give

      int(W(x) * F(x), x = a..b) = sum(w_k * F(x_k),k = 0..(n-1))

    exact for all polynomials F(x) of degree (strictly) less than 2*n. For all
    integrable functions F(x) the sum is a (more or less) good approximation to
    the integral. The x_k are called nodes (which are the zeros of the
    related orthogonal polynomials) and the w_k are called the weights.

    parameters
       n        (input) The degree of the quadrature rule, i.e. its number of
                nodes.

       qtype    (input) The family of orthogonal polynmomials for which to
                compute the quadrature rule. See the list below.

       alpha    (input) real number, used as parameter for some orthogonal
                polynomials

       beta     (input) real number, used as parameter for some orthogonal
                polynomials.

    return value

      (X, W)    a pair of two real arrays where x_k = X[k] and w_k = W[k].


    orthogonal polynomials:

      qtype           polynomial
      -----           ----------

      "legendre"      Legendre polynomials, W(x)=1 on the interval (-1, +1)
      "legendre01"    shifted Legendre polynomials, W(x)=1 on the interval (0, +1)
      "hermite"       Hermite polynomials, W(x)=exp(-x*x) on (-infinity,+infinity)
      "laguerre"      Laguerre polynomials, W(x)=exp(-x) on (0,+infinity)
      "glaguerre"     generalized Laguerre polynomials, W(x)=exp(-x)*x**alpha
                      on (0, +infinity)
      "chebyshev1"    Chebyshev polynomials of the first kind, W(x)=1/sqrt(1-x*x)
                      on (-1, +1)
      "chebyshev2"    Chebyshev polynomials of the second kind, W(x)=sqrt(1-x*x)
                      on (-1, +1)
      "jacobi"        Jacobi polynomials, W(x)=(1-x)**alpha * (1+x)**beta on (-1, +1)
                      with alpha>-1 and beta>-1

    examples:
      >>> from mpmath import mp
      >>> f = lambda x: x**8 + 2 * x**6 - 3 * x**4 + 5 * x**2 - 7
      >>> X, W = mp.gauss_quadrature(5, "hermite")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = mp.sqrt(mp.pi) * 57 / 16
      >>> C = mp.quad(lambda x: mp.exp(- x * x) * f(x), [-mp.inf, +mp.inf])
      >>> mp.nprint((mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10)))
      (0.0, 0.0)

      >>> f = lambda x: x**5 - 2 * x**4 + 3 * x**3 - 5 * x**2 + 7 * x - 11
      >>> X, W = mp.gauss_quadrature(3, "laguerre")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = 76
      >>> C = mp.quad(lambda x: mp.exp(-x) * f(x), [0, +mp.inf])
      >>> mp.nprint(mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10))
      .0

      # orthogonality of the chebyshev polynomials:
      >>> f = lambda x: mp.chebyt(3, x) * mp.chebyt(2, x)
      >>> X, W = mp.gauss_quadrature(3, "chebyshev1")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> print(mp.chop(A, tol = 1e-10))
      0.0

    references:
      - golub and welsch, "calculations of gaussian quadrature rules", mathematics of
        computation 23, p. 221-230 (1969)
      - golub, "some modified matrix eigenvalue problems", siam review 15, p. 318-334 (1973)
      - stroud and secrest, "gaussian quadrature formulas", prentice-hall (1966)

    See also the routine gaussq.f in netlog.org or ACM Transactions on
    Mathematical Software algorithm 726.
    """

    d = ctx.zeros(n, 1)
    e = ctx.zeros(n, 1)
    z = ctx.zeros(1, n)

    z[0,0] = 1

    if qtype == "legendre":
        # legendre on the range -1 +1 , abramowitz, table 25.4, p.916
        w = 2
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j * j / (4 * j * j - ctx.mpf(1)))
    elif qtype == "legendre01":
        # legendre shifted to 0 1        , abramowitz, table 25.8, p.921
        w = 1
        for i in xrange(n):
            d[i] = 1 / ctx.mpf(2)
            j = i + 1
            e[i] = ctx.sqrt(j * j / (16 * j * j - ctx.mpf(4)))
    elif qtype == "hermite":
        # hermite on the range -inf +inf , abramowitz, table 25.10,p.924
        w = ctx.sqrt(ctx.pi)
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j / ctx.mpf(2))
    elif qtype == "laguerre":
        # laguerre on the range 0 +inf , abramowitz, table 25.9, p. 923
        w = 1
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1
            e[i] = j
    elif qtype=="chebyshev1":
        # chebyshev polynimials of the first kind
        w = ctx.pi
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
        e[0] = ctx.sqrt(1 / ctx.mpf(2))
    elif qtype == "chebyshev2":
        # chebyshev polynimials of the second kind
        w = ctx.pi / 2
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
    elif qtype == "glaguerre":
        # generalized laguerre on the range 0 +inf
        w = ctx.gamma(1 + alpha)
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1 + alpha
            e[i] = ctx.sqrt(j * (j + alpha))
    elif qtype == "jacobi":
        # jacobi polynomials
        alpha = ctx.mpf(alpha)
        beta = ctx.mpf(beta)
        ab = alpha + beta
        abi = ab + 2
        w = (2**(ab+1)) * ctx.gamma(alpha + 1) * ctx.gamma(beta + 1) / ctx.gamma(abi)
        d[0] = (beta - alpha) / abi
        e[0] = ctx.sqrt(4 * (1 + alpha) * (1 + beta) / ((abi + 1) * (abi * abi)))
        a2b2 = beta * beta - alpha * alpha
        for i in xrange(1, n):
            j = i + 1
            abi = 2 * j + ab
            d[i] = a2b2 / ((abi - 2) * abi)
            e[i] = ctx.sqrt(4 * j * (j + alpha) * (j + beta) * (j + ab) / ((abi * abi - 1) * abi * abi))
    elif isinstance(qtype, str):
        raise ValueError("unknown quadrature rule \"%s\"" % qtype)
    elif not isinstance(qtype, str):
        w = qtype(d, e)
    else:
        assert 0

    tridiag_eigen(ctx, d, e, z)

    for i in xrange(len(z)):
        z[i] *= z[i]

    z = z.transpose()
    return (d, w * z)

##################################################################################################
##################################################################################################
##################################################################################################

def svd_r_raw(ctx, A, V = False, calc_u = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the transpose.
    Householder bidiagonalization and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
