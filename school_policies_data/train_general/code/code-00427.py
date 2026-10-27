                    return  # complex terms might cancel
                t_not_re_im = t
            else:
                return

        if t_not_re_im:
            if t_not_re_im.is_extended_real is False:
                if real:  # like 3
                    return zero  # 3*(smthng like 2 + I or i) is not real
            if t_not_re_im.is_imaginary is False:  # symbolic 2 or 2 + I
                if not real:  # like I
                    return zero  # I*(smthng like 2 or 2 + I) is not real
        elif zero is False:
            return real  # can't be trumped by 0
        elif real:
            return real  # doesn't matter what zero is

    def _eval_is_imaginary(self):
        if all(a.is_zero is False and a.is_finite for a in self.args):
            return self._eval_real_imag(False)

    def _eval_is_hermitian(self):
        return self._eval_herm_antiherm(True)

    def _eval_is_antihermitian(self):
        return self._eval_herm_antiherm(False)

    def _eval_herm_antiherm(self, herm):
        for t in self.args:
            if t.is_hermitian is None or t.is_antihermitian is None:
                return
            if t.is_hermitian:
                continue
            elif t.is_antihermitian:
                herm = not herm
            else:
                return

        if herm is not False:
            return herm

        is_zero = self._eval_is_zero()
        if is_zero:
            return True
        elif is_zero is False:
            return herm

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all((x.is_rational and fuzzy_not(x.is_zero)) is True for x in others):
                    return True
                return
            if a is None:
                return
        if all(x.is_real for x in self.args):
            return False

    def _eval_is_extended_positive(self):
        """Return True if self is positive, False if not, and None if it
        cannot be determined.

        Explanation
        ===========

        This algorithm is non-recursive and works by keeping track of the
        sign which changes when a negative or nonpositive is encountered.
        Whether a nonpositive or nonnegative is seen is also tracked since
        the presence of these makes it impossible to return True, but
        possible to return False if the end result is nonpositive. e.g.

            pos * neg * nonpositive -> pos or zero -> None is returned
            pos * neg * nonnegative -> neg or zero -> False is returned
        """
        return self._eval_pos_neg(1)

    def _eval_pos_neg(self, sign):
        saw_NON = saw_NOT = False
        for t in self.args:
            if t.is_extended_positive:
                continue
            elif t.is_extended_negative:
                sign = -sign
            elif t.is_zero:
                if all(a.is_finite for a in self.args):
                    return False
                return
            elif t.is_extended_nonpositive:
                sign = -sign
                saw_NON = True
            elif t.is_extended_nonnegative:
                saw_NON = True
            # FIXME: is_positive/is_negative is False doesn't take account of
            # Symbol('x', infinite=True, extended_real=True) which has
            # e.g. is_positive is False but has uncertain sign.
            elif t.is_positive is False:
                sign = -sign
                if saw_NOT:
                    return
                saw_NOT = True
            elif t.is_negative is False:
                if saw_NOT:
                    return
                saw_NOT = True
            else:
                return
        if sign == 1 and saw_NON is False and saw_NOT is False:
            return True
        if sign < 0:
            return False

    def _eval_is_extended_negative(self):
        return self._eval_pos_neg(-1)

    def _eval_is_odd(self):
        is_integer = self._eval_is_integer()
        if is_integer is not True:
            return is_integer

        from sympy.simplify.radsimp import fraction
        n, d = fraction(self)
        if d.is_Integer and d.is_even:
            # if minimal power of 2 in num vs den is
            # positive then we have an even number
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(n) if i.is_even]) - trailing(d.p)
                    ).is_positive:
                return False
            return
        r, acc = True, 1
        for t in self.args:
            if abs(t) is S.One:
                continue
            if t.is_even:
                return False
            if r is False:
                pass
            elif acc != 1 and (acc + t).is_odd:
                r = False
            elif t.is_even is None:
                r = None
            acc = t
        return r

    def _eval_is_even(self):
        from sympy.simplify.radsimp import fraction
        n, d = fraction(self)
        if n.is_Integer and n.is_even:
            # if minimal power of 2 in den vs num is not
            # negative then this is not an integer and
            # can't be even
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(d) if i.is_even]) - trailing(n.p)
                    ).is_nonnegative:
                return False

    def _eval_is_composite(self):
        """
        Here we count the number of arguments that have a minimum value
        greater than two.
        If there are more than one of such a symbol then the result is composite.
        Else, the result cannot be determined.
        """
        number_of_args = 0 # count of symbols with minimum value greater than one
        for arg in self.args:
            if not (arg.is_integer and arg.is_positive):
                return None
            if (arg-1).is_positive:
                number_of_args += 1

        if number_of_args > 1:
            return True

    def _eval_subs(self, old, new):
        from sympy.functions.elementary.complexes import sign
        from sympy.ntheory.factor_ import multiplicity
        from sympy.simplify.powsimp import powdenest
        from sympy.simplify.radsimp import fraction

        if not old.is_Mul:
            return None

        # try keep replacement literal so -2*x doesn't replace 4*x
        if old.args[0].is_Number and old.args[0] < 0:
            if self.args[0].is_Number:
                if self.args[0] < 0:
                    return self._subs(-old, -new)
                return None

        def base_exp(a):
            # if I and -1 are in a Mul, they get both end up with
            # a -1 base (see issue 6421); all we want here are the
            # true Pow or exp separated into base and exponent
            from sympy.functions.elementary.exponential import exp
            if a.is_Pow or isinstance(a, exp):
                return a.as_base_exp()
            return a, S.One

        def breakup(eq):
            """break up powers of eq when treated as a Mul:
                   b**(Rational*e) -> b**e, Rational
                commutatives come back as a dictionary {b**e: Rational}
                noncommutatives come back as a list [(b**e, Rational)]
            """

            (c, nc) = (defaultdict(int), [])
            for a in Mul.make_args(eq):
                a = powdenest(a)
                (b, e) = base_exp(a)
                if e is not S.One:
                    (co, _) = e.as_coeff_mul()
                    b = Pow(b, e/co)
                    e = co
                if a.is_commutative:
                    c[b] += e
                else:
                    nc.append([b, e])
            return (c, nc)

        def rejoin(b, co):
            """
            Put rational back with exponent; in general this is not ok, but
            since we took it from the exponent for analysis, it's ok to put
            it back.
            """

            (b, e) = base_exp(b)
            return Pow(b, e*co)

        def ndiv(a, b):
            """if b divides a in an extractive way (like 1/4 divides 1/2
            but not vice versa, and 2/5 does not divide 1/3) then return
            the integer number of times it divides, else return 0.
            """
            if not b.q % a.q or not a.q % b.q:
                return int(a/b)
            return 0

        # give Muls in the denominator a chance to be changed (see issue 5651)
        # rv will be the default return value
        rv = None
        n, d = fraction(self)
        self2 = self
        if d is not S.One:
            self2 = n._subs(old, new)/d._subs(old, new)
            if not self2.is_Mul:
                return self2._subs(old, new)
            if self2 != self:
                rv = self2

        # Now continue with regular substitution.

        # handle the leading coefficient and use it to decide if anything
        # should even be started; we always know where to find the Rational
        # so it's a quick test

        co_self = self2.args[0]
        co_old = old.args[0]
        co_xmul = None
        if co_old.is_Rational and co_self.is_Rational:
            # if coeffs are the same there will be no updating to do
            # below after breakup() step; so skip (and keep co_xmul=None)
            if co_old != co_self:
                co_xmul = co_self.extract_multiplicatively(co_old)
        elif co_old.is_Rational:
            return rv

        # break self and old into factors

        (c, nc) = breakup(self2)
        (old_c, old_nc) = breakup(old)

        # update the coefficients if we had an extraction
        # e.g. if co_self were 2*(3/35*x)**2 and co_old = 3/5
        # then co_self in c is replaced by (3/5)**2 and co_residual
        # is 2*(1/7)**2

        if co_xmul and co_xmul.is_Rational and abs(co_old) != 1:
            mult = S(multiplicity(abs(co_old), co_self))
            c.pop(co_self)
            if co_old in c:
                c[co_old] += mult
            else:
                c[co_old] = mult
            co_residual = co_self/co_old**mult
        else:
            co_residual = 1

        # do quick tests to see if we can't succeed

        ok = True
        if len(old_nc) > len(nc):
            # more non-commutative terms
            ok = False
        elif len(old_c) > len(c):
            # more commutative terms
            ok = False
        elif {i[0] for i in old_nc}.difference({i[0] for i in nc}):
            # unmatched non-commutative bases
            ok = False
        elif set(old_c).difference(set(c)):
            # unmatched commutative terms
            ok = False
        elif any(sign(c[b]) != sign(old_c[b]) for b in old_c):
            # differences in sign
            ok = False
        if not ok:
            return rv

        if not old_c:
            cdid = None
        else:
            rat = []
            for (b, old_e) in old_c.items():
                c_e = c[b]
                rat.append(ndiv(c_e, old_e))
                if not rat[-1]:
                    return rv
            cdid = min(rat)

        if not old_nc:
            ncdid = None
            for i in range(len(nc)):
                nc[i] = rejoin(*nc[i])
        else:
            ncdid = 0  # number of nc replacements we did
            take = len(old_nc)  # how much to look at each time
            limit = cdid or S.Infinity  # max number that we can take
            failed = []  # failed terms will need subs if other terms pass
            i = 0
            while limit and i + take <= len(nc):
                hit = False

                # the bases must be equivalent in succession, and
                # the powers must be extractively compatible on the
                # first and last factor but equal in between.

                rat = []
                for j in range(take):
                    if nc[i + j][0] != old_nc[j][0]:
                        break
                    elif j == 0:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif j == take - 1:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif nc[i + j][1] != old_nc[j][1]:
                        break
                    else:
                        rat.append(1)
                    j += 1
                else:
                    ndo = min(rat)
                    if ndo:
                        if take == 1:
                            if cdid:
                                ndo = min(cdid, ndo)
                            nc[i] = Pow(new, ndo)*rejoin(nc[i][0],
                                    nc[i][1] - ndo*old_nc[0][1])
                        else:
                            ndo = 1

                            # the left residual

                            l = rejoin(nc[i][0], nc[i][1] - ndo*
                                    old_nc[0][1])

                            # eliminate all middle terms

                            mid = new

                            # the right residual (which may be the same as the middle if take == 2)

                            ir = i + take - 1
                            r = (nc[ir][0], nc[ir][1] - ndo*
                                 old_nc[-1][1])
                            if r[1]:
                                if i + take < len(nc):
                                    nc[i:i + take] = [l*mid, r]
                                else:
                                    r = rejoin(*r)
                                    nc[i:i + take] = [l*mid*r]
                            else:

                                # there was nothing left on the right

                                nc[i:i + take] = [l*mid]

                        limit -= ndo
                        ncdid += ndo
                        hit = True
                if not hit:

                    # do the subs on this failing factor

                    failed.append(i)
                i += 1
            else:

                if not ncdid:
                    return rv

                # although we didn't fail, certain nc terms may have
                # failed so we rebuild them after attempting a partial
                # subs on them

                failed.extend(range(i, len(nc)))
                for i in failed:
                    nc[i] = rejoin(*nc[i]).subs(old, new)

        # rebuild the expression

        if cdid is None:
            do = ncdid
        elif ncdid is None:
            do = cdid
        else:
            do = min(ncdid, cdid)

        margs = []
        for b in c:
            if b in old_c:

                # calculate the new exponent

                e = c[b] - old_c[b]*do
                margs.append(rejoin(b, e))
            else:
                margs.append(rejoin(b.subs(old, new), c[b]))
        if cdid and not ncdid:

            # in case we are replacing commutative with non-commutative,
            # we want the new term to come at the front just like the
            # rest of this routine

            margs = [Pow(new, cdid)] + margs
        return co_residual*self2.func(*margs)*self2.func(*nc)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from .function import PoleError
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.order import Order

        def coeff_exp(term, x):
            lt = term.as_coeff_exponent(x)
            if lt[0].has(x):
                try:
                    lt = term.leadterm(x)
                except ValueError:
                    return term, S.Zero
            return lt

        ords = []

        try:
            for t in self.args:
                coeff, exp = t.leadterm(x)
                if not coeff.has(x):
                    ords.append((t, exp))
                else:
                    raise ValueError

            n0 = sum(t[1] for t in ords if t[1].is_number)
            facs = []
            for t, m in ords:
                n1 = ceiling(n - n0 + (m if m.is_number else 0))
                s = t.nseries(x, n=n1, logx=logx, cdir=cdir)
                ns = s.getn()
                if ns is not None:
                    if ns < n1:  # less than expected
                        n -= n1 - ns    # reduce n
                facs.append(s)

        except (ValueError, NotImplementedError, TypeError, PoleError):
            # XXX: Catching so many generic exceptions around a large block of
            # code will mask bugs. Whatever purpose catching these exceptions
            # serves should be handled in a different way.
            n0 = sympify(sum(t[1] for t in ords if t[1].is_number))
            if n0.is_nonnegative:
                n0 = S.Zero
            facs = [t.nseries(x, n=ceiling(n-n0), logx=logx, cdir=cdir) for t in self.args]
            from sympy.simplify.powsimp import powsimp
            res = powsimp(self.func(*facs).expand(), combine='exp', deep=True)
            if res.has(Order):
                res += Order(x**n, x)
            return res

        res = S.Zero
        ords2 = [Add.make_args(factor) for factor in facs]

        for fac in product(*ords2):
            ords3 = [coeff_exp(term, x) for term in fac]
            coeffs, powers = zip(*ords3)
            power = sum(powers)
            if (power - n).is_negative:
                res += Mul(*coeffs)*(x**power)

        def max_degree(e, x):
            if e is x:
                return S.One
            if e.is_Atom:
                return S.Zero
            if e.is_Add:
                return max(max_degree(a, x) for a in e.args)
            if e.is_Mul:
                return Add(*[max_degree(a, x) for a in e.args])
            if e.is_Pow:
                return max_degree(e.base, x)*e.exp
            return S.Zero

        if self.is_polynomial(x):
            from sympy.polys.polyerrors import PolynomialError
            from sympy.polys.polytools import degree
            try:
                if max_degree(self, x) >= n or degree(self, x) != degree(res, x):
                    res += Order(x**n, x)
            except PolynomialError:
                pass
            else:
                return res

        if res != self:
            if (self - res).subs(x, 0) == S.Zero and n > 0:
                lt = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
                if lt == S.Zero:
                    return res
            res += Order(x**n, x)
        return res

    def _eval_as_leading_term(self, x, logx, cdir):
        return self.func(*[t.as_leading_term(x, logx=logx, cdir=cdir) for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args[::-1]])

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args[::-1]])

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (-3*sqrt(2)*(2 - 2*sqrt(2))).as_content_primitive()
        (6, -sqrt(2)*(1 - sqrt(2)))

        See docstring of Expr.as_content_primitive for more examples.
        """

        coef = S.One
        args = []
        for a in self.args:
            c, p = a.as_content_primitive(radical=radical, clear=clear)
            coef *= c
            if p is not S.One:
                args.append(p)
        # don't use self._from_args here to reconstruct args
        # since there may be identical args now that should be combined
        # e.g. (2+2*x)*(3+3*x) should be (6, (1 + x)**2) not (6, (1+x)*(1+x))
        return coef, self.func(*args)

    def as_ordered_factors(self, order=None):
        """Transform an expression into an ordered list of factors.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x, y

        >>> (2*x*y*sin(x)*cos(x)).as_ordered_factors()
        [2, x, y, sin(x), cos(x)]

        """
        cpart, ncpart = self.args_cnc()
        cpart.sort(key=lambda expr: expr.sort_key(order=order))
        return cpart + ncpart

    @property
    def _sorted_args(self):
        return tuple(self.as_ordered_factors())

mul = AssocOpDispatcher('mul')


def prod(a, start=1):
    """Return product of elements of a. Start with int 1 so if only
       ints are included then an int result is returned.

    Examples
    ========

    >>> from sympy import prod, S
    >>> prod(range(3))
    0
    >>> type(_) is int
    True
    >>> prod([S(2), 3])
    6
    >>> _.is_Integer
    True

    You can start the product at something other than 1:

    >>> prod([1, 2], 3)
    6

    """
    return reduce(operator.mul, a, start)


def _keep_coeff(coeff, factors, clear=True, sign=False):
    """Return ``coeff*factors`` unevaluated if necessary.

    If ``clear`` is False, do not keep the coefficient as a factor
    if it can be distributed on a single factor such that one or
    more terms will still have integer coefficients.

    If ``sign`` is True, allow a coefficient of -1 to remain factored out.

    Examples
    ========

    >>> from sympy.core.mul import _keep_coeff
    >>> from sympy.abc import x, y
    >>> from sympy import S

    >>> _keep_coeff(S.Half, x + 2)
    (x + 2)/2
    >>> _keep_coeff(S.Half, x + 2, clear=False)
    x/2 + 1
    >>> _keep_coeff(S.Half, (x + 2)*y, clear=False)
    y*(x + 2)/2
    >>> _keep_coeff(S(-1), x + y)
    -x - y
    >>> _keep_coeff(S(-1), x + y, sign=True)
    -(x + y)
    """
    if not coeff.is_Number:
        if factors.is_Number:
            factors, coeff = coeff, factors
        else:
            return coeff*factors
    if factors is S.One:
        return coeff
    if coeff is S.One:
        return factors
    elif coeff is S.NegativeOne and not sign:
        return -factors
    elif factors.is_Add:
        if not clear and coeff.is_Rational and coeff.q != 1:
            args = [i.as_coeff_Mul() for i in factors.args]
            args = [(_keep_coeff(c, coeff), m) for c, m in args]
            if any(c.is_Integer for c, _ in args):
                return Add._from_args([Mul._from_args(
                    i[1:] if i[0] == 1 else i) for i in args])
        return Mul(coeff, factors, evaluate=False)
    elif factors.is_Mul:
        margs = list(factors.args)
        if margs[0].is_Number:
            margs[0] *= coeff
            if margs[0] == 1:
                margs.pop(0)
        else:
            margs.insert(0, coeff)
        return Mul._from_args(margs)
    else:
        m = coeff*factors
        if m.is_Number and not factors.is_Number:
            m = Mul._from_args((coeff, factors))
        return m

def expand_2arg(e):
    def do(e):
        if e.is_Mul:
            c, r = e.as_coeff_Mul()
            if c.is_Number and r.is_Add:
                return _unevaluated_Add(*[c*ri for ri in r.args])
        return e
    return bottom_up(e, do)


from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
<|endoftext|>from types import FunctionType

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains import ZZ, QQ

from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot


def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc,
                normalize_last=True, normalize=True, zero_above=True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = isimp(a*mat[p] - b*mat[p + q])

    isimp = _get_intermediate_simp(_dotprodsimp)
    piv_row, piv_col = 0, 0
    pivot_cols = []
    swaps = []

    # use a fraction free method to zero above and below each pivot
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val, \
        assumed_nonzero, newly_determined = _find_reasonable_pivot(
                get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        # _find_reasonable_pivot may have simplified some things
        # in the process.  Let's not let them go to waste
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset*cols + piv_col] = val

        if pivot_offset is None:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))

        # if we aren't normalizing last, we normalize
        # before we zero the other rows
        if normalize_last is False:
            i, j = piv_row, piv_col
            mat[i*cols + j] = one
            for p in range(i*cols + j + 1, (i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)
            # after normalizing, the pivot value is 1
            pivot_val = one

        # zero above and below the pivot
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # don't zero above the pivot unless we're told.
            if zero_above is False and row < piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row*cols + piv_col]
            if iszerofunc(val):
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i*cols + piv_j]
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)

    return mat, tuple(pivot_cols), tuple(swaps)


# This functions is a candidate for caching if it gets implemented for matrices.
def _row_reduce(M, iszerofunc, simpfunc, normalize_last=True,
                normalize=True, zero_above=True):

    mat, pivot_cols, swaps = _row_reduce_list(list(M), M.rows, M.cols, M.one,
            iszerofunc, simpfunc, normalize_last=normalize_last,
            normalize=normalize, zero_above=zero_above)

    return M._new(M.rows, M.cols, mat), pivot_cols, swaps


def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""

    if M.rows <= 0 or M.cols <= 0:
        return True

    zeros_below = all(iszerofunc(t) for t in M[1:, 0])

    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)

    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)


def _echelon_form(M, iszerofunc=_iszero, simplify=False, with_pivots=False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form. Note
    that echelon form of a matrix is *not* unique, however, properties like the
    row space and the null space are preserved.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.echelon_form()
    Matrix([
    [1,  2],
    [0, -2]])
    """

    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify

    mat, pivots, _ = _row_reduce(M, iszerofunc, simpfunc,
            normalize_last=True, normalize=False, zero_above=False)

    if with_pivots:
        return mat, pivots

    return mat


# This functions is a candidate for caching if it gets implemented for matrices.
def _rank(M, iszerofunc=_iszero, simplify=False):
    """Returns the rank of a matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rank()
    2
    >>> n = Matrix(3, 3, range(1, 10))
    >>> n.rank()
    2
    """

    def _permute_complexity_right(M, iszerofunc):
        """Permute columns with complicated elements as
        far right as they can go.  Since the ``sympy`` row reduction
        algorithms start on the left, having complexity right-shifted
        speeds things up.

        Returns a tuple (mat, perm) where perm is a permutation
        of the columns to perform to shift the complex columns right, and mat
        is the permuted matrix."""

        def complexity(i):
            # the complexity of a column will be judged by how many
            # element's zero-ness cannot be determined
            return sum(1 if iszerofunc(e) is None else 0 for e in M[:, i])

        complex = [(complexity(i), i) for i in range(M.cols)]
        perm    = [j for (i, j) in sorted(complex)]

        return (M.permute(perm, orientation='cols'), perm)

    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify

    # for small matrices, we compute the rank explicitly
    # if is_zero on elements doesn't answer the question
    # for small matrices, we fall back to the full routine.
    if M.rows <= 0 or M.cols <= 0:
        return 0

    if M.rows <= 1 or M.cols <= 1:
        zeros = [iszerofunc(x) for x in M]

        if False in zeros:
            return 1

    if M.rows == 2 and M.cols == 2:
        zeros = [iszerofunc(x) for x in M]

        if False not in zeros and None not in zeros:
            return 0

        d = M.det()

        if iszerofunc(d) and False in zeros:
            return 1
        if iszerofunc(d) is False:
            return 2

    mat, _       = _permute_complexity_right(M, iszerofunc=iszerofunc)
    _, pivots, _ = _row_reduce(mat, iszerofunc, simpfunc, normalize_last=True,
            normalize=False, zero_above=False)

    return len(pivots)


def _to_DM_ZZ_QQ(M):
    # We have to test for _rep here because there are tests that otherwise fail
    # with e.g. "AttributeError: 'SubspaceOnlyMatrix' object has no attribute
    # '_rep'." There is almost certainly no value in such tests. The
    # presumption seems to be that someone could create a new class by
    # inheriting some of the Matrix classes and not the full set that is used
    # by the standard Matrix class but if anyone tried that it would fail in
    # many ways.
    if not hasattr(M, '_rep'):
        return None

    rep = M._rep
    K = rep.domain

    if K.is_ZZ:
        return rep
    elif K.is_QQ:
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep
    else:
        if not all(e.is_Rational for e in M):
            return None
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep.convert_to(QQ)


def _rref_dm(dM):
    """Compute the reduced row echelon form of a DomainMatrix."""
    K = dM.domain

    if K.is_ZZ:
        dM_rref, den, pivots = dM.rref_den(keep_domain=False)
        dM_rref = dM_rref.to_field() / den
    elif K.is_QQ:
        dM_rref, pivots = dM.rref()
    else:
        assert False  # pragma: no cover

    M_rref = dM_rref.to_Matrix()

    return M_rref, pivots


def _rref(M, iszerofunc=_iszero, simplify=False, pivots=True,
        normalize_last=True):
    """Return reduced row-echelon form of matrix and indices
    of pivot vars.

    Parameters
    ==========

    iszerofunc : Function
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Function
        A function used to simplify elements when looking for a pivot.
        By default SymPy's ``simplify`` is used.

    pivots : True or False
        If ``True``, a tuple containing the row-reduced matrix and a tuple
        of pivot columns is returned.  If ``False`` just the row-reduced
        matrix is returned.

    normalize_last : True or False
        If ``True``, no pivots are normalized to `1` until after all
        entries above and below each pivot are zeroed.  This means the row
        reduction algorithm is fraction free until the very last step.
        If ``False``, the naive row reduction procedure is used where
        each pivot is normalized to be `1` before row operations are
        used to zero above and below the pivot.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rref()
    (Matrix([
    [1, 0],
    [0, 1]]), (0, 1))
    >>> rref_matrix, rref_pivots = m.rref()
    >>> rref_matrix
    Matrix([
    [1, 0],
    [0, 1]])
    >>> rref_pivots
    (0, 1)

    ``iszerofunc`` can correct rounding errors in matrices with float
    values. In the following example, calling ``rref()`` leads to
    floating point errors, incorrectly row reducing the matrix.
    ``iszerofunc= lambda x: abs(x) < 1e-9`` sets sufficiently small numbers
    to zero, avoiding this error.

    >>> m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])
    >>> m.rref()
    (Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), (0, 1, 2))
    >>> m.rref(iszerofunc=lambda x:abs(x)<1e-9)
    (Matrix([
    [1, 0, -0.301369863013699, 0],
    [0, 1, -0.712328767123288, 0],
    [0, 0,         0,          0]]), (0, 1))

    Notes
    =====

    The default value of ``normalize_last=True`` can provide significant
    speedup to row reduction, especially on matrices with symbols.  However,
    if you depend on the form row reduction algorithm leaves entries
    of the matrix, set ``normalize_last=False``
    """
    # Try to use DomainMatrix for ZZ or QQ
    dM = _to_DM_ZZ_QQ(M)

    if dM is not None:
        # Use DomainMatrix for ZZ or QQ
        mat, pivot_cols = _rref_dm(dM)
    else:
        # Use the generic Matrix routine.
        if isinstance(simplify, FunctionType):
            simpfunc = simplify
        else:
            simpfunc = _simplify

        mat, pivot_cols, _ = _row_reduce(M, iszerofunc, simpfunc,
                normalize_last, normalize=True, zero_above=True)

    if pivots:
        return mat, pivot_cols
    else:
        return mat
<|endoftext|># sympy.matrices.kind

from sympy.core.kind import Kind, _NumberKind, NumberKind
from sympy.core.mul import Mul


class MatrixKind(Kind):
    """
    Kind for all matrices in SymPy.

    Basic class for this kind is ``MatrixBase`` and ``MatrixExpr``,
    but any expression representing the matrix can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is
        :class:`sympy.core.kind.NumberKind`,
        which means that the matrix contains only numbers.

    Examples
    ========

    Any instance of matrix class has kind ``MatrixKind``:

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 2)
    >>> A.kind
    MatrixKind(NumberKind)

    An expression representing a matrix may not be an instance of
    the Matrix class, but it will have kind ``MatrixKind``:

    >>> from sympy import MatrixExpr, Integral
    >>> from sympy.abc import x
    >>> intM = Integral(A, x)
    >>> isinstance(intM, MatrixExpr)
    False
    >>> intM.kind
    MatrixKind(NumberKind)

    Use ``isinstance()`` to check for ``MatrixKind`` without specifying the
    element kind. Use ``is`` to check the kind including the element kind:

    >>> from sympy import Matrix
    >>> from sympy.core import NumberKind
    >>> from sympy.matrices import MatrixKind
    >>> M = Matrix([1, 2])
    >>> isinstance(M.kind, MatrixKind)
    True
    >>> M.kind is MatrixKind(NumberKind)
    True

    See Also
    ========

    sympy.core.kind.NumberKind
    sympy.core.kind.UndefinedKind
    sympy.core.containers.TupleKind
    sympy.sets.sets.SetKind

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "MatrixKind(%s)" % self.element_kind


@Mul._kind_dispatcher.register(_NumberKind, MatrixKind)
def num_mat_mul(k1, k2):
    """
    Return MatrixKind. The element kind is selected by recursive dispatching.
    Do not need to dispatch in reversed order because KindDispatcher
    searches for this automatically.
    """
    # Deal with Mul._kind_dispatcher's commutativity
    # XXX: this function is called with either k1 or k2 as MatrixKind because
    # the Mul kind dispatcher is commutative. Maybe it shouldn't be. Need to
    # swap the args here because NumberKind does not have an element_kind
    # attribute.
    if not isinstance(k2, MatrixKind):
        k1, k2 = k2, k1
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    return MatrixKind(elemk)


@Mul._kind_dispatcher.register(MatrixKind, MatrixKind)
def mat_mat_mul(k1, k2):
    """
    Return MatrixKind. The element kind is selected by recursive dispatching.
    """
    elemk = Mul._kind_dispatcher(k1.element_kind, k2.element_kind)
    return MatrixKind(elemk)
<|endoftext|>from sympy.utilities.iterables import \
    flatten, connected_components, strongly_connected_components
from .exceptions import NonSquareMatrixError


def _connected_components(M):
    """Returns the list of connected vertices of the graph when
    a square matrix is viewed as a weighted graph.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])
    >>> A.connected_components()
    [[0, 3, 8], [1, 6, 7], [2, 4, 5]]

    Notes
    =====

    Even if any symbolic elements of the matrix can be indeterminate
    to be zero mathematically, this only takes the account of the
    structural aspect of the matrix, so they will considered to be
    nonzero.
    """
    if not M.is_square:
        raise NonSquareMatrixError

    V = range(M.rows)
    E = sorted(M.todok().keys())
    return connected_components((V, E))


def _strongly_connected_components(M):
    """Returns the list of strongly connected vertices of the graph when
    a square matrix is viewed as a weighted graph.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])
    >>> A.strongly_connected_components()
    [[0, 4, 6], [2, 3, 7], [1, 5, 8]]
    """
    if not M.is_square:
        raise NonSquareMatrixError

    # RepMatrix uses the more efficient DomainMatrix.scc() method
    rep = getattr(M, '_rep', None)
    if rep is not None:
        return rep.scc()

    V = range(M.rows)
    E = sorted(M.todok().keys())
    return strongly_connected_components((V, E))


def _connected_components_decomposition(M):
    """Decomposes a square matrix into block diagonal form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Returns
    =======

    P, B : PermutationMatrix, BlockDiagMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block diagonal matrix of
        the result of the permutation.

        If you would like to get the diagonal blocks from the
        BlockDiagMatrix, see
        :meth:`~sympy.matrices.expressions.blockmatrix.BlockDiagMatrix.get_diag_blocks`.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])

    >>> P, B = A.connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((1 3)(2 8 5 7 4 6))
    >>> pprint(B)
    [[66  68  67]                            ]
    [[          ]                            ]
    [[86  88  87]       0             0      ]
    [[          ]                            ]
    [[76  78  77]                            ]
    [                                        ]
    [              [55  54  53]              ]
    [              [          ]              ]
    [     0        [45  44  43]       0      ]
    [              [          ]              ]
    [              [35  34  33]              ]
    [                                        ]
    [                            [0   1   2 ]]
    [                            [          ]]
    [     0             0        [10  11  12]]
    [                            [          ]]
    [                            [20  21  22]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T*B*P == A
    True

    Notes
    =====

    This problem corresponds to the finding of the connected components
    of a graph, when a matrix is viewed as a weighted graph.
    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
    from sympy.matrices.expressions.permutation import PermutationMatrix

    iblocks = M.connected_components()

    p = Permutation(flatten(iblocks))
    P = PermutationMatrix(p)

    blocks = []
    for b in iblocks:
        blocks.append(M[b, b])
    B = BlockDiagMatrix(*blocks)
    return P, B


def _strongly_connected_components_decomposition(M, lower=True):
    """Decomposes a square matrix into block triangular form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Parameters
    ==========

    lower : bool
        Makes $B$ lower block triangular when ``True``.
        Otherwise, makes $B$ upper block triangular.

    Returns
    =======

    P, B : PermutationMatrix, BlockMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block triangular matrix of
        the result of the permutation.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])

    A lower block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((8)(1 4 3 2 6)(5 7))
    >>> pprint(B)
    [[44  43  45]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[34  33  35]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[54  53  55]   [0  0  0]     [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [62  61  60]  [66  68  67]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [82  81  80]  [86  88  87]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [72  71  70]  [76  78  77]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True

    An upper block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition(lower=False)
    >>> pprint(P)
    PermutationMatrix((0 1 5 7 4 3 2 8 6))
    >>> pprint(B)
    [[66  68  67]  [62  61  60]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[86  88  87]  [82  81  80]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[76  78  77]  [72  71  70]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]     [0  0  0]    [44  43  45]]
    [ [       ]     [       ]    [          ]]
    [ [0  0  0]     [0  0  0]    [34  33  35]]
    [ [       ]     [       ]    [          ]]
    [ [0  0  0]     [0  0  0]    [54  53  55]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True
    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    from sympy.matrices.expressions.permutation import PermutationMatrix

    iblocks = M.strongly_connected_components()
    if not lower:
        iblocks = list(reversed(iblocks))

    p = Permutation(flatten(iblocks))
    P = PermutationMatrix(p)

    rows = []
    for a in iblocks:
        cols = []
        for b in iblocks:
            cols.append(M[a, b])
        rows.append(cols)
    B = BlockMatrix(rows)
    return P, B
<|endoftext|>"""
A module containing deprecated matrix mixin classes.

The classes in this module are deprecated and will be removed in a future
release. They are kept here for backwards compatibility in case downstream
code was subclassing them.

Importing anything else from this module is deprecated so anything here
should either not be used or should be imported from somewhere else.
"""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce

from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.numbers import Integer
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.utilities.exceptions import sympy_deprecation_warning
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray

from .utilities import _get_intermediate_simp_bool


# These exception types were previously defined in this module but were moved
# to exceptions.py. We reimport them here for backwards compatibility in case
# downstream code was importing them from here.
from .exceptions import ( # noqa: F401
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
    NonPositiveDefiniteMatrixError
)


_DEPRECATED_MIXINS = (
    'MatrixShaping',
    'MatrixSpecial',
    'MatrixProperties',
    'MatrixOperations',
    'MatrixArithmetic',
    'MatrixCommon',
    'MatrixDeterminant',
    'MatrixReductions',
    'MatrixSubspaces',
    'MatrixEigen',
    'MatrixCalculus',
    'MatrixDeprecated',
)


class _MatrixDeprecatedMeta(type):

    #
    # Override the default __instancecheck__ implementation to ensure that
    # e.g. isinstance(M, MatrixCommon) still works when M is one of the
    # matrix classes. Matrix no longer inherits from MatrixCommon so
    # isinstance(M, MatrixCommon) would now return False by default.
    #
    # There were lots of places in the codebase where this was being done
    # so it seems likely that downstream code may be doing it too. All use
    # of these mixins is deprecated though so we give a deprecation warning
    # unconditionally if they are being used with isinstance.
    #
    # Any code seeing this deprecation warning should be changed to use
    # isinstance(M, MatrixBase) instead which also works in previous versions
    # of SymPy.
    #

    def __instancecheck__(cls, instance):

        sympy_deprecation_warning(
            f"""
            Checking whether an object is an instance of {cls.__name__} is
            deprecated.

            Use `isinstance(obj, Matrix)` instead of `isinstance(obj, {cls.__name__})`.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-matrix-mixins",
            stacklevel=3,
        )

        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.matrices import (
            MatrixDeterminant,
            MatrixReductions,
            MatrixSubspaces,
            MatrixEigen,
            MatrixCalculus,
            MatrixDeprecated
        )

        all_mixins = (
            MatrixRequired,
            MatrixShaping,
            MatrixSpecial,
            MatrixProperties,
            MatrixOperations,
            MatrixArithmetic,
            MatrixCommon,
            MatrixDeterminant,
            MatrixReductions,
            MatrixSubspaces,
            MatrixEigen,
            MatrixCalculus,
            MatrixDeprecated
        )

        if cls in all_mixins and isinstance(instance, MatrixBase):
            return True
        else:
            return super().__instancecheck__(instance)


class MatrixRequired(metaclass=_MatrixDeprecatedMeta):
    """Deprecated mixin class for making matrix classes."""

    rows: int
    cols: int
    _simplify = None

    def __init_subclass__(cls, **kwargs):

        # Warn if any downstream code is subclassing this class or any of the
        # deprecated mixin classes that are all ultimately subclasses of this
        # class.
        #
        # We don't want to warn about the deprecated mixins themselves being
        # created, but only about them being used as mixins by downstream code.
        # Otherwise just importing this module would trigger a warning.
        # Ultimately the whole module should be deprecated and removed but for
        # SymPy 1.13 it is premature to do that given that this module was the
        # main way to import matrix exception types in all previous versions.

        if cls.__name__ not in _DEPRECATED_MIXINS:
            sympy_deprecation_warning(
                f"""
                Inheriting from the Matrix mixin classes is deprecated.

                The class {cls.__name__} is subclassing a deprecated mixin.
                """,
                deprecated_since_version="1.13",
                active_deprecations_target="deprecated-matrix-mixins",
                stacklevel=3,
            )

        super().__init_subclass__(**kwargs)

    @classmethod
    def _new(cls, *args, **kwargs):
        """`_new` must, at minimum, be callable as
        `_new(rows, cols, mat) where mat is a flat list of the
        elements of the matrix."""
        raise NotImplementedError("Subclasses must implement this.")

    def __eq__(self, other):
        raise NotImplementedError("Subclasses must implement this.")

    def __getitem__(self, key):
        """Implementations of __getitem__ should accept ints, in which
        case the matrix is indexed as a flat list, tuples (i,j) in which
        case the (i,j) entry is returned, slices, or mixed tuples (a,b)
        where a and b are any combination of slices and integers."""
        raise NotImplementedError("Subclasses must implement this.")

    def __len__(self):
        """The total number of entries in the matrix."""
        raise NotImplementedError("Subclasses must implement this.")

    @property
    def shape(self):
        raise NotImplementedError("Subclasses must implement this.")


class MatrixShaping(MatrixRequired):
    """Provides basic matrix shaping and extracting of submatrices"""

    def _eval_col_del(self, col):
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        return self._new(self.rows, self.cols - 1, entry)

    def _eval_col_insert(self, pos, other):

        def entry(i, j):
            if j < pos:
                return self[i, j]
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]
            return self[i, j - other.cols]

        return self._new(self.rows, self.cols + other.cols, entry)

    def _eval_col_join(self, other):
        rows = self.rows

        def entry(i, j):
            if i < rows:
                return self[i, j]
            return other[i - rows, j]

        return classof(self, other)._new(self.rows + other.rows, self.cols,
                                         entry)

    def _eval_extract(self, rowsList, colsList):
        mat = list(self)
        cols = self.cols
        indices = (i * cols + j for i in rowsList for j in colsList)
        return self._new(len(rowsList), len(colsList),
                         [mat[i] for i in indices])

    def _eval_get_diag_blocks(self):
        sub_blocks = []

        def recurse_sub_blocks(M):
            for i in range(1, M.shape[0] + 1):
                if i == 1:
                    to_the_right = M[0, i:]
                    to_the_bottom = M[i:, 0]
                else:
                    to_the_right = M[:i, i:]
                    to_the_bottom = M[i:, :i]
                if any(to_the_right) or any(to_the_bottom):
                    continue
                sub_blocks.append(M[:i, :i])
                if M.shape != M[:i, :i].shape:
                    recurse_sub_blocks(M[i:, i:])
                return

        recurse_sub_blocks(self)
        return sub_blocks

    def _eval_row_del(self, row):
        def entry(i, j):
            return self[i, j] if i < row else self[i + 1, j]
        return self._new(self.rows - 1, self.cols, entry)

    def _eval_row_insert(self, pos, other):
        entries = list(self)
        insert_pos = pos * self.cols
        entries[insert_pos:insert_pos] = list(other)
        return self._new(self.rows + other.rows, self.cols, entries)

    def _eval_row_join(self, other):
        cols = self.cols

        def entry(i, j):
            if j < cols:
                return self[i, j]
            return other[i, j - cols]

        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         entry)

    def _eval_tolist(self):
        return [list(self[i,:]) for i in range(self.rows)]

    def _eval_todok(self):
        dok = {}
        rows, cols = self.shape
        for i in range(rows):
            for j in range(cols):
                val = self[i, j]
                if val != self.zero:
                    dok[i, j] = val
        return dok

    def _eval_vec(self):
        rows = self.rows

        def entry(n, _):
            # we want to read off the columns first
            j = n // rows
            i = n - j * rows
            return self[i, j]

        return self._new(len(self), 1, entry)

    def _eval_vech(self, diagonal):
        c = self.cols
        v = []
        if diagonal:
            for j in range(c):
                for i in range(j, c):
                    v.append(self[i, j])
        else:
            for j in range(c):
                for i in range(j + 1, c):
                    v.append(self[i, j])
        return self._new(len(v), 1, v)

    def col_del(self, col):
        """Delete the specified column."""
        if col < 0:
            col += self.cols
        if not 0 <= col < self.cols:
            raise IndexError("Column {} is out of range.".format(col))
        return self._eval_col_del(col)

    def col_insert(self, pos, other):
        """Insert one or more columns at the given column position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.col_insert(1, V)
        Matrix([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        See Also
        ========

        col
        row_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return type(self)(other)

        pos = as_int(pos)

        if pos < 0:
            pos = self.cols + pos
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols

        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))

        return self._eval_col_insert(pos, other)

    def col_join(self, other):
        """Concatenates two matrices along self's last and other's first row.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.col_join(V)
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

        See Also
        ========

        col
        row_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))
        return self._eval_col_join(other)

    def col(self, j):
        """Elementary column selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).col(0)
        Matrix([
        [1],
        [0]])

        See Also
        ========

        row
        col_del
        col_join
        col_insert
        """
        return self[:, j]

    def extract(self, rowsList, colsList):
        r"""Return a submatrix by specifying a list of rows and columns.
        Negative indices can be given. All indices must be in the range
        $-n \le i < n$ where $n$ is the number of rows or columns.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(4, 3, range(12))
        >>> m
        Matrix([
        [0,  1,  2],
        [3,  4,  5],
        [6,  7,  8],
        [9, 10, 11]])
        >>> m.extract([0, 1, 3], [0, 1])
        Matrix([
        [0,  1],
        [3,  4],
        [9, 10]])

        Rows or columns can be repeated:

        >>> m.extract([0, 0, 1], [-1])
        Matrix([
        [2],
        [2],
        [5]])

        Every other row can be taken by using range to provide the indices:

        >>> m.extract(range(0, m.rows, 2), [-1])
        Matrix([
        [2],
        [8]])

        RowsList or colsList can also be a list of booleans, in which case
        the rows or columns corresponding to the True values will be selected:

        >>> m.extract([0, 1, 2, 3], [True, False, True])
        Matrix([
        [0,  2],
        [3,  5],
        [6,  8],
        [9, 11]])
        """

        if not is_sequence(rowsList) or not is_sequence(colsList):
            raise TypeError("rowsList and colsList must be iterable")
        # ensure rowsList and colsList are lists of integers
        if rowsList and all(isinstance(i, bool) for i in rowsList):
            rowsList = [index for index, item in enumerate(rowsList) if item]
        if colsList and all(isinstance(i, bool) for i in colsList):
            colsList = [index for index, item in enumerate(colsList) if item]

        # ensure everything is in range
        rowsList = [a2idx(k, self.rows) for k in rowsList]
        colsList = [a2idx(k, self.cols) for k in colsList]

        return self._eval_extract(rowsList, colsList)

    def get_diag_blocks(self):
        """Obtains the square sub-matrices on the main diagonal of a square matrix.

        Useful for inverting symbolic matrices or solving systems of
        linear equations which may be decoupled by having a block diagonal
        structure.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y, z
        >>> A = Matrix([[1, 3, 0, 0], [y, z*z, 0, 0], [0, 0, x, 0], [0, 0, 0, 0]])
        >>> a1, a2, a3 = A.get_diag_blocks()
        >>> a1
        Matrix([
        [1,    3],
        [y, z**2]])
        >>> a2
        Matrix([[x]])
        >>> a3
        Matrix([[0]])

        """
        return self._eval_get_diag_blocks()

    @classmethod
    def hstack(cls, *args):
        """Return a matrix formed by joining args horizontally (i.e.
        by repeated application of row_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.hstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.row_join, args)

    def reshape(self, rows, cols):
        """Reshape the matrix. Total number of elements must remain the same.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 3, lambda i, j: 1)
        >>> m
        Matrix([
        [1, 1, 1],
        [1, 1, 1]])
        >>> m.reshape(1, 6)
        Matrix([[1, 1, 1, 1, 1, 1]])
        >>> m.reshape(3, 2)
        Matrix([
        [1, 1],
        [1, 1],
        [1, 1]])

        """
        if self.rows * self.cols != rows * cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        return self._new(rows, cols, lambda i, j: self[i * cols + j])

    def row_del(self, row):
        """Delete the specified row."""
        if row < 0:
            row += self.rows
        if not 0 <= row < self.rows:
            raise IndexError("Row {} is out of range.".format(row))

        return self._eval_row_del(row)

    def row_insert(self, pos, other):
        """Insert one or more rows at the given row position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.row_insert(1, V)
        Matrix([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]])

        See Also
        ========

        row
        col_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return self._new(other)

        pos = as_int(pos)

        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows

        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))

        return self._eval_row_insert(pos, other)

    def row_join(self, other):
        """Concatenates two matrices along self's last and rhs's first column

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.row_join(V)
        Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

        See Also
        ========

        row
        col_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))
        return self._eval_row_join(other)

    def diagonal(self, k=0):
        """Returns the kth diagonal of self. The main diagonal
        corresponds to `k=0`; diagonals above and below correspond to
        `k > 0` and `k < 0`, respectively. The values of `self[i, j]`
        for which `j - i = k`, are returned in order of increasing
        `i + j`, starting with `i + j = |k|`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(3, 3, lambda i, j: j - i); m
        Matrix([
        [ 0,  1, 2],
        [-1,  0, 1],
        [-2, -1, 0]])
        >>> _.diagonal()
        Matrix([[0, 0, 0]])
        >>> m.diagonal(1)
        Matrix([[1, 1]])
        >>> m.diagonal(-2)
        Matrix([[-2]])

        Even though the diagonal is returned as a Matrix, the element
        retrieval can be done with a single index:

        >>> Matrix.diag(1, 2, 3).diagonal()[1]  # instead of [0, 1]
        2

        See Also
        ========

        diag
        """
        rv = []
        k = as_int(k)
        r = 0 if k > 0 else -k
        c = 0 if r else k
        while True:
            if r == self.rows or c == self.cols:
                break
            rv.append(self[r, c])
            r += 1
            c += 1
        if not rv:
            raise ValueError(filldedent('''
            The %s diagonal is out of range [%s, %s]''' % (
            k, 1 - self.rows, self.cols - 1)))
        return self._new(1, len(rv), rv)

    def row(self, i):
        """Elementary row selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).row(0)
        Matrix([[1, 0]])

        See Also
        ========

        col
        row_del
        row_join
        row_insert
        """
        return self[i, :]

    @property
    def shape(self):
        """The shape (dimensions) of the matrix as the 2-tuple (rows, cols).

        Examples
        ========

        >>> from sympy import zeros
        >>> M = zeros(2, 3)
        >>> M.shape
        (2, 3)
        >>> M.rows
        2
        >>> M.cols
        3
        """
        return (self.rows, self.cols)

    def todok(self):
        """Return the matrix as dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix.eye(3)
        >>> M.todok()
        {(0, 0): 1, (1, 1): 1, (2, 2): 1}
        """
        return self._eval_todok()

    def tolist(self):
        """Return the Matrix as a nested Python list.

        Examples
        ========

        >>> from sympy import Matrix, ones
        >>> m = Matrix(3, 3, range(9))
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
        >>> m.tolist()
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> ones(3, 0).tolist()
        [[], [], []]

        When there are no rows then it will not be possible to tell how
        many columns were in the original matrix:

        >>> ones(0, 3).tolist()
        []

        """
        if not self.rows:
            return []
        if not self.cols:
            return [[] for i in range(self.rows)]
        return self._eval_tolist()

    def todod(M):
        """Returns matrix as dict of dicts containing non-zero elements of the Matrix

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1],[0, 3]])
        >>> A
        Matrix([
        [0, 1],
        [0, 3]])
        >>> A.todod()
        {0: {1: 1}, 1: {1: 3}}


        """
        rowsdict = {}
        Mlol = M.tolist()
        for i, Mi in enumerate(Mlol):
            row = {j: Mij for j, Mij in enumerate(Mi) if Mij}
            if row:
                rowsdict[i] = row
        return rowsdict

    def vec(self):
        """Return the Matrix converted into a one column matrix by stacking columns

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 3], [2, 4]])
        >>> m
        Matrix([
        [1, 3],
        [2, 4]])
        >>> m.vec()
        Matrix([
        [1],
        [2],
        [3],
        [4]])

        See Also
        ========

        vech
        """
        return self._eval_vec()

    def vech(self, diagonal=True, check_symmetry=True):
        """Reshapes the matrix into a column vector by stacking the
        elements in the lower triangle.

        Parameters
        ==========

        diagonal : bool, optional
            If ``True``, it includes the diagonal elements.

        check_symmetry : bool, optional
            If ``True``, it checks whether the matrix is symmetric.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 2], [2, 3]])
        >>> m
        Matrix([
        [1, 2],
        [2, 3]])
        >>> m.vech()
        Matrix([
        [1],
        [2],
        [3]])
        >>> m.vech(diagonal=False)
        Matrix([[2]])

        Notes
        =====

        This should work for symmetric matrices and ``vech`` can
        represent symmetric matrices in vector form with less size than
        ``vec``.

        See Also
        ========

        vec
        """
        if not self.is_square:
            raise NonSquareMatrixError

        if check_symmetry and not self.is_symmetric():
            raise ValueError("The matrix is not symmetric.")

        return self._eval_vech(diagonal)

    @classmethod
    def vstack(cls, *args):
        """Return a matrix formed by joining args vertically (i.e.
        by repeated application of col_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.vstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.col_join, args)


class MatrixSpecial(MatrixRequired):
    """Construction of special matrices"""

    @classmethod
    def _eval_diag(cls, rows, cols, diag_dict):
        """diag_dict is a defaultdict containing
        all the entries of the diagonal matrix."""
        def entry(i, j):
            return diag_dict[(i, j)]
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_eye(cls, rows, cols):
        vals = [cls.zero]*(rows*cols)
        vals[::cols+1] = [cls.one]*min(rows, cols)
        return cls._new(rows, cols, vals, copy=False)

    @classmethod
    def _eval_jordan_block(cls, size: int, eigenvalue, band='upper'):
        if band == 'lower':
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif j + 1 == i:
                    return cls.one
                return cls.zero
        else:
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif i + 1 == j:
                    return cls.one
                return cls.zero
        return cls._new(size, size, entry)

    @classmethod
    def _eval_ones(cls, rows, cols):
        def entry(i, j):
            return cls.one
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        return cls._new(rows, cols, [cls.zero]*(rows*cols), copy=False)

    @classmethod
    def _eval_wilkinson(cls, n):
        def entry(i, j):
            return cls.one if i + 1 == j else cls.zero

        D = cls._new(2*n + 1, 2*n + 1, entry)

        wminus = cls.diag(list(range(-n, n + 1)), unpack=True) + D + D.T
        wplus = abs(cls.diag(list(range(-n, n + 1)), unpack=True)) + D + D.T

        return wminus, wplus

    @classmethod
    def diag(kls, *args, strict=False, unpack=True, rows=None, cols=None, **kwargs):
        """Returns a matrix with the specified diagonal.
        If matrices are passed, a block-diagonal matrix
        is created (i.e. the "direct sum" of the matrices).

        kwargs
        ======

        rows : rows of the resulting matrix; computed if
               not given.

        cols : columns of the resulting matrix; computed if
               not given.

        cls : class for the resulting matrix

        unpack : bool which, when True (default), unpacks a single
        sequence rather than interpreting it as a Matrix.

        strict : bool which, when False (default), allows Matrices to
        have variable-length rows.

        Examples
        ========

        >>> from sympy import Matrix
        >>> Matrix.diag(1, 2, 3)
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])

        The current default is to unpack a single sequence. If this is
        not desired, set `unpack=False` and it will be interpreted as
        a matrix.

        >>> Matrix.diag([1, 2, 3]) == Matrix.diag(1, 2, 3)
        True

        When more than one element is passed, each is interpreted as
        something to put on the diagonal. Lists are converted to
        matrices. Filling of the diagonal always continues from
        the bottom right hand corner of the previous item: this
        will create a block-diagonal matrix whether the matrices
        are square or not.

        >>> col = [1, 2, 3]
        >>> row = [[4, 5]]
        >>> Matrix.diag(col, row)
        Matrix([
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [0, 4, 5]])

        When `unpack` is False, elements within a list need not all be
        of the same length. Setting `strict` to True would raise a
        ValueError for the following:

        >>> Matrix.diag([[1, 2, 3], [4, 5], [6]], unpack=False)
        Matrix([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0]])

        The type of the returned matrix can be set with the ``cls``
        keyword.

        >>> from sympy import ImmutableMatrix
        >>> from sympy.utilities.misc import func_name
        >>> func_name(Matrix.diag(1, cls=ImmutableMatrix))
        'ImmutableDenseMatrix'

        A zero dimension matrix can be used to position the start of
        the filling at the start of an arbitrary row or column:

        >>> from sympy import ones
        >>> r2 = ones(0, 2)
        >>> Matrix.diag(r2, 1, 2)
        Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 2]])

        See Also
        ========
        eye
        diagonal
        .dense.diag
        .expressions.blockmatrix.BlockMatrix
        .sparsetools.banded
       """
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.dense import Matrix
        from sympy.matrices import SparseMatrix
        klass = kwargs.get('cls', kls)
        if unpack and len(args) == 1 and is_sequence(args[0]) and \
                not isinstance(args[0], MatrixBase):
            args = args[0]

        # fill a default dict with the diagonal entries
        diag_entries = defaultdict(int)
        rmax = cmax = 0  # keep track of the biggest index seen
        for m in args:
            if isinstance(m, list):
                if strict:
                    # if malformed, Matrix will raise an error
                    _ = Matrix(m)
                    r, c = _.shape
                    m = _.tolist()
                else:
                    r, c, smat = SparseMatrix._handle_creation_inputs(m)
                    for (i, j), _ in smat.items():
                        diag_entries[(i + rmax, j + cmax)] = _
                    m = []  # to skip process below
            elif hasattr(m, 'shape'):  # a Matrix
                # convert to list of lists
                r, c = m.shape
                m = m.tolist()
            else:  # in this case, we're a single value
                diag_entries[(rmax, cmax)] = m
                rmax += 1
                cmax += 1
                continue
            # process list of lists
            for i, mi in enumerate(m):
                for j, _ in enumerate(mi):
                    diag_entries[(i + rmax, j + cmax)] = _
            rmax += r
            cmax += c
        if rows is None:
            rows, cols = cols, rows
        if rows is None:
            rows, cols = rmax, cmax
        else:
            cols = rows if cols is None else cols
        if rows < rmax or cols < cmax:
            raise ValueError(filldedent('''
                The constructed matrix is {} x {} but a size of {} x {}
                was specified.'''.format(rmax, cmax, rows, cols)))
        return klass._eval_diag(rows, cols, diag_entries)

    @classmethod
    def eye(kls, rows, cols=None, **kwargs):
        """Returns an identity matrix.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_eye(rows, cols)

    @classmethod
    def jordan_block(kls, size=None, eigenvalue=None, *, band='upper', **kwargs):
        """Returns a Jordan block

        Parameters
        ==========

        size : Integer, optional
            Specifies the shape of the Jordan block matrix.

        eigenvalue : Number or Symbol
            Specifies the value for the main diagonal of the matrix.

            .. note::
                The keyword ``eigenval`` is also specified as an alias
                of this keyword, but it is not recommended to use.

                We may deprecate the alias in later release.

        band : 'upper' or 'lower', optional
            Specifies the position of the off-diagonal to put `1` s on.

        cls : Matrix, optional
            Specifies the matrix class of the output form.

            If it is not specified, the class type where the method is
            being executed on will be returned.

        Returns
        =======

        Matrix
            A Jordan block matrix.

        Raises
        ======

        ValueError
            If insufficient arguments are given for matrix size
            specification, or no eigenvalue is given.

        Examples
        ========

        Creating a default Jordan block:

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> Matrix.jordan_block(4, x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        Creating an alternative Jordan block matrix where `1` is on
        lower off-diagonal:

        >>> Matrix.jordan_block(4, x, band='lower')
        Matrix([
        [x, 0, 0, 0],
        [1, x, 0, 0],
        [0, 1, x, 0],
        [0, 0, 1, x]])

        Creating a Jordan block with keyword arguments

        >>> Matrix.jordan_block(size=4, eigenvalue=x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Jordan_matrix
        """
        klass = kwargs.pop('cls', kls)

        eigenval = kwargs.get('eigenval', None)
        if eigenvalue is None and eigenval is None:
            raise ValueError("Must supply an eigenvalue")
        elif eigenvalue != eigenval and None not in (eigenval, eigenvalue):
            raise ValueError(
                "Inconsistent values are given: 'eigenval'={}, "
                "'eigenvalue'={}".format(eigenval, eigenvalue))
        else:
            if eigenval is not None:
                eigenvalue = eigenval

        if size is None:
            raise ValueError("Must supply a matrix size")

        size = as_int(size)
        return klass._eval_jordan_block(size, eigenvalue, band)

    @classmethod
    def ones(kls, rows, cols=None, **kwargs):
        """Returns a matrix of ones.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_ones(rows, cols)

    @classmethod
    def zeros(kls, rows, cols=None, **kwargs):
        """Returns a matrix of zeros.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_zeros(rows, cols)

    @classmethod
    def companion(kls, poly):
        """Returns a companion matrix of a polynomial.

        Examples
        ========

        >>> from sympy import Matrix, Poly, Symbol, symbols
        >>> x = Symbol('x')
        >>> c0, c1, c2, c3, c4 = symbols('c0:5')
        >>> p = Poly(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + x**5, x)
        >>> Matrix.companion(p)
        Matrix([
        [0, 0, 0, 0, -c0],
        [1, 0, 0, 0, -c1],
        [0, 1, 0, 0, -c2],
        [0, 0, 1, 0, -c3],
        [0, 0, 0, 1, -c4]])
        """
        poly = kls._sympify(poly)
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))

        size = poly.degree()
        if not size >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        coeffs = poly.all_coeffs()
        def entry(i, j):
            if j == size - 1:
                return -coeffs[-1 - i]
            elif i == j + 1:
                return kls.one
            return kls.zero
        return kls._new(size, size, entry)


    @classmethod
    def wilkinson(kls, n, **kwargs):
        """Returns two square Wilkinson Matrix of size 2*n + 1
        $W_{2n + 1}^-, W_{2n + 1}^+ =$ Wilkinson(n)

        Examples
        ========

        >>> from sympy import Matrix
        >>> wminus, wplus = Matrix.wilkinson(3)
        >>> wminus
        Matrix([
        [-3,  1,  0, 0, 0, 0, 0],
        [ 1, -2,  1, 0, 0, 0, 0],
        [ 0,  1, -1, 1, 0, 0, 0],
        [ 0,  0,  1, 0, 1, 0, 0],
        [ 0,  0,  0, 1, 1, 1, 0],
        [ 0,  0,  0, 0, 1, 2, 1],
        [ 0,  0,  0, 0, 0, 1, 3]])
        >>> wplus
        Matrix([
        [3, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 3]])

        References
        ==========

        .. [1] https://blogs.mathworks.com/cleve/2013/04/15/wilkinsons-matrices-2/
        .. [2] J. H. Wilkinson, The Algebraic Eigenvalue Problem, Claredon Press, Oxford, 1965, 662 pp.

        """
        klass = kwargs.get('cls', kls)
        n = as_int(n)
        return klass._eval_wilkinson(n)

class MatrixProperties(MatrixRequired):
    """Provides basic properties of a matrix."""

    def _eval_atoms(self, *types):
        result = set()
        for i in self:
            result.update(i.atoms(*types))
        return result

    def _eval_free_symbols(self):
        return set().union(*(i.free_symbols for i in self if i))

    def _eval_has(self, *patterns):
        return any(a.has(*patterns) for a in self)

    def _eval_is_anti_symmetric(self, simpfunc):
        if not all(simpfunc(self[i, j] + self[j, i]).is_zero for i in range(self.rows) for j in range(self.cols)):
            return False
        return True

    def _eval_is_diagonal(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self[i, j]:
                    return False
        return True

    # _eval_is_hermitian is called by some general SymPy
    # routines and has a different *args signature.  Make
    # sure the names don't clash by adding `_matrix_` in name.
    def _eval_is_matrix_hermitian(self, simpfunc):
        mat = self._new(self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i].conjugate()))
        return mat.is_zero_matrix

    def _eval_is_Identity(self) -> FuzzyBool:
        def dirac(i, j):
            if i == j:
                return 1
            return 0

        return all(self[i, j] == dirac(i, j)
                for i in range(self.rows)
                for j in range(self.cols))

    def _eval_is_lower_hessenberg(self):
        return all(self[i, j].is_zero
                   for i in range(self.rows)
                   for j in range(i + 2, self.cols))

    def _eval_is_lower(self):
        return all(self[i, j].is_zero
                   for i in range(self.rows)
                   for j in range(i + 1, self.cols))

    def _eval_is_symbolic(self):
        return self.has(Symbol)

    def _eval_is_symmetric(self, simpfunc):
        mat = self._new(self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i]))
        return mat.is_zero_matrix

    def _eval_is_zero_matrix(self):
        if any(i.is_zero == False for i in self):
            return False
        if any(i.is_zero is None for i in self):
            return None
        return True

    def _eval_is_upper_hessenberg(self):
        return all(self[i, j].is_zero
                   for i in range(2, self.rows)
                   for j in range(min(self.cols, (i - 1))))

    def _eval_values(self):
        return [i for i in self if not i.is_zero]

    def _has_positive_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_positive for x in diagonal_entries)

    def _has_nonnegative_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_nonnegative for x in diagonal_entries)

    def atoms(self, *types):
        """Returns the atoms that form the current object.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import Matrix
        >>> Matrix([[x]])
        Matrix([[x]])
        >>> _.atoms()
        {x}
        >>> Matrix([[x, y], [y, x]])
        Matrix([
        [x, y],
        [y, x]])
        >>> _.atoms()
        {x, y}
        """

        types = tuple(t if isinstance(t, type) else type(t) for t in types)
        if not types:
            types = (Atom,)
        return self._eval_atoms(*types)

    @property
    def free_symbols(self):
        """Returns the free symbols within the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix([[x], [1]]).free_symbols
        {x}
        """
        return self._eval_free_symbols()

    def has(self, *patterns):
        """Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import Matrix, SparseMatrix, Float
        >>> from sympy.abc import x, y
        >>> A = Matrix(((1, x), (0.2, 3)))
        >>> B = SparseMatrix(((1, x), (0.2, 3)))
        >>> A.has(x)
        True
        >>> A.has(y)
        False
        >>> A.has(Float)
        True
        >>> B.has(x)
        True
        >>> B.has(y)
        False
        >>> B.has(Float)
        True
        """
        return self._eval_has(*patterns)

    def is_anti_symmetric(self, simplify=True):
        """Check if matrix M is an antisymmetric matrix,
        that is, M is a square matrix with all M[i, j] == -M[j, i].

        When ``simplify=True`` (default), the sum M[i, j] + M[j, i] is
        simplified before testing to see if it is zero. By default,
        the SymPy simplify function is used. To use a custom function
        set simplify to a function that accepts a single argument which
        returns a simplified expression. To skip simplification, set
        simplify to False but note that although this will be faster,
        it may induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> m = Matrix(2, 2, [0, 1, -1, 0])
        >>> m
        Matrix([
        [ 0, 1],
        [-1, 0]])
        >>> m.is_anti_symmetric()
        True
        >>> x, y = symbols('x y')
        >>> m = Matrix(2, 3, [0, 0, x, -y, 0, 0])
        >>> m
        Matrix([
        [ 0, 0, x],
        [-y, 0, 0]])
        >>> m.is_anti_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [0, x**2 + 2*x + 1, y,
        ...                   -(x + 1)**2, 0, x*y,
        ...                   -y, -x*y, 0])

        Simplification of matrix elements is done by default so even
        though two elements which should be equal and opposite would not
        pass an equality test, the matrix is still reported as
        anti-symmetric:

        >>> m[0, 1] == -m[1, 0]
        False
        >>> m.is_anti_symmetric()
        True

        If ``simplify=False`` is used for the case when a Matrix is already
        simplified, this will speed things up. Here, we see that without
        simplification the matrix does not appear anti-symmetric:

        >>> print(m.is_anti_symmetric(simplify=False))
        None

        But if the matrix were already expanded, then it would appear
        anti-symmetric and simplification in the is_anti_symmetric routine
        is not needed:

        >>> m = m.expand()
        >>> m.is_anti_symmetric(simplify=False)
        True
        """
        # accept custom simplification
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _simplify if simplify else lambda x: x

        if not self.is_square:
            return False
        return self._eval_is_anti_symmetric(simpfunc)

    def is_diagonal(self):
        """Check if matrix is diagonal,
        that is matrix in which the entries outside the main diagonal are all zero.

        Examples
        ========

        >>> from sympy import Matrix, diag
        >>> m = Matrix(2, 2, [1, 0, 0, 2])
        >>> m
        Matrix([
        [1, 0],
        [0, 2]])
        >>> m.is_diagonal()
        True

        >>> m = Matrix(2, 2, [1, 1, 0, 2])
        >>> m
        Matrix([
        [1, 1],
        [0, 2]])
        >>> m.is_diagonal()
        False

        >>> m = diag(1, 2, 3)
        >>> m
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])
        >>> m.is_diagonal()
        True

        See Also
        ========

        is_lower
        is_upper
        sympy.matrices.matrixbase.MatrixCommon.is_diagonalizable
        diagonalize
        """
        return self._eval_is_diagonal()

    @property
    def is_weakly_diagonally_dominant(self):
        r"""Tests if the matrix is row weakly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row weakly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| \ge \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_weakly_diagonally_dominant
        True

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_weakly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_weakly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_nonnegative

        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_strongly_diagonally_dominant(self):
        r"""Tests if the matrix is row strongly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row strongly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| > \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_strongly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_positive

        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_hermitian(self):
        """Checks if the matrix is Hermitian.

        In a Hermitian matrix element i,j is the complex conjugate of
        element j,i.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy import I
        >>> from sympy.abc import x
        >>> a = Matrix([[1, I], [-I, 1]])
        >>> a
        Matrix([
        [ 1, I],
        [-I, 1]])
        >>> a.is_hermitian
        True
        >>> a[0, 0] = 2*I
        >>> a.is_hermitian
        False
        >>> a[0, 0] = x
        >>> a.is_hermitian
        >>> a[0, 1] = a[1, 0]*I
        >>> a.is_hermitian
        False
        """
        if not self.is_square:
            return False

        return self._eval_is_matrix_hermitian(_simplify)

    @property
    def is_Identity(self) -> FuzzyBool:
        if not self.is_square:
            return False
        return self._eval_is_Identity()

    @property
    def is_lower_hessenberg(self):
        r"""Checks if the matrix is in the lower-Hessenberg form.

        The lower hessenberg matrix has zero entries
        above the first superdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 0, 0], [5, 2, 3, 0], [3, 4, 3, 7], [5, 6, 1, 1]])
        >>> a
        Matrix([
        [1, 2, 0, 0],
        [5, 2, 3, 0],
        [3, 4, 3, 7],
        [5, 6, 1, 1]])
        >>> a.is_lower_hessenberg
        True

        See Also
        ========

        is_upper_hessenberg
        is_lower
        """
        return self._eval_is_lower_hessenberg()

    @property
    def is_lower(self):
        """Check if matrix is a lower triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_lower
        True

        >>> m = Matrix(4, 3, [0, 0, 0, 2, 0, 0, 1, 4, 0, 6, 6, 5])
        >>> m
        Matrix([
        [0, 0, 0],
        [2, 0, 0],
        [1, 4, 0],
        [6, 6, 5]])
        >>> m.is_lower
        True

        >>> from sympy.abc import x, y
        >>> m = Matrix(2, 2, [x**2 + y, y**2 + x, 0, x + y])
        >>> m
        Matrix([
        [x**2 + y, x + y**2],
        [       0,    x + y]])
        >>> m.is_lower
        False

        See Also
        ========

        is_upper
        is_diagonal
        is_lower_hessenberg
        """
        return self._eval_is_lower()

    @property
    def is_square(self):
        """Checks if a matrix is square.

        A matrix is square if the number of rows equals the number of columns.
        The empty matrix is square by definition, since the number of rows and
        the number of columns are both zero.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> b = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> c = Matrix([])
        >>> a.is_square
        False
        >>> b.is_square
        True
        >>> c.is_square
        True
        """
        return self.rows == self.cols

    def is_symbolic(self):
        """Checks if any elements contain Symbols.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.is_symbolic()
        True

        """
        return self._eval_is_symbolic()

    def is_symmetric(self, simplify=True):
        """Check if matrix is symmetric matrix,
        that is square matrix and is equal to its transpose.

        By default, simplifications occur before testing symmetry.
        They can be skipped using 'simplify=False'; while speeding things a bit,
        this may however induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [0, 1, 1, 2])
        >>> m
        Matrix([
        [0, 1],
        [1, 2]])
        >>> m.is_symmetric()
        True

        >>> m = Matrix(2, 2, [0, 1, 2, 0])
        >>> m
        Matrix([
        [0, 1],
        [2, 0]])
        >>> m.is_symmetric()
        False

        >>> m = Matrix(2, 3, [0, 0, 0, 0, 0, 0])
        >>> m
        Matrix([
        [0, 0, 0],
        [0, 0, 0]])
        >>> m.is_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
        >>> m
        Matrix([
        [         1, x**2 + 2*x + 1, y],
        [(x + 1)**2,              2, 0],
        [         y,              0, 3]])
        >>> m.is_symmetric()
        True

        If the matrix is already simplified, you may speed-up is_symmetric()
        test by using 'simplify=False'.

        >>> bool(m.is_symmetric(simplify=False))
        False
        >>> m1 = m.expand()
        >>> m1.is_symmetric(simplify=False)
        True
        """
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _simplify if simplify else lambda x: x

        if not self.is_square:
            return False

        return self._eval_is_symmetric(simpfunc)

    @property
    def is_upper_hessenberg(self):
        """Checks if the matrix is the upper-Hessenberg form.

        The upper hessenberg matrix has zero entries
        below the first subdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 4, 2, 3], [3, 4, 1, 7], [0, 2, 3, 4], [0, 0, 1, 3]])
        >>> a
        Matrix([
        [1, 4, 2, 3],
        [3, 4, 1, 7],
        [0, 2, 3, 4],
        [0, 0, 1, 3]])
        >>> a.is_upper_hessenberg
        True

        See Also
        ========

        is_lower_hessenberg
        is_upper
        """
        return self._eval_is_upper_hessenberg()

    @property
    def is_upper(self):
        """Check if matrix is an upper triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_upper
        True

        >>> m = Matrix(4, 3, [5, 1, 9, 0, 4, 6, 0, 0, 5, 0, 0, 0])
        >>> m
        Matrix([
        [5, 1, 9],
        [0, 4, 6],
        [0, 0, 5],
        [0, 0, 0]])
        >>> m.is_upper
        True

        >>> m = Matrix(2, 3, [4, 2, 5, 6, 1, 1])
        >>> m
        Matrix([
        [4, 2, 5],
        [6, 1, 1]])
        >>> m.is_upper
        False

        See Also
        ========

        is_lower
        is_diagonal
        is_upper_hessenberg
        """
        return all(self[i, j].is_zero
                   for i in range(1, self.rows)
                   for j in range(min(i, self.cols)))

    @property
    def is_zero_matrix(self):
        """Checks if a matrix is a zero matrix.

        A matrix is zero if every element is zero.  A matrix need not be square
        to be considered zero.  The empty matrix is zero by the principle of
        vacuous truth.  For a matrix that may or may not be zero (e.g.
        contains a symbol), this will be None

        Examples
        ========

        >>> from sympy import Matrix, zeros
        >>> from sympy.abc import x
        >>> a = Matrix([[0, 0], [0, 0]])
        >>> b = zeros(3, 4)
        >>> c = Matrix([[0, 1], [0, 0]])
        >>> d = Matrix([])
        >>> e = Matrix([[x, 0], [0, 0]])
        >>> a.is_zero_matrix
        True
        >>> b.is_zero_matrix
        True
        >>> c.is_zero_matrix
        False
        >>> d.is_zero_matrix
        True
        >>> e.is_zero_matrix
        """
        return self._eval_is_zero_matrix()

    def values(self):
        """Return non-zero values of self."""
        return self._eval_values()


class MatrixOperations(MatrixRequired):
    """Provides basic matrix shape and elementwise
    operations.  Should not be instantiated directly."""

    def _eval_adjoint(self):
        return self.transpose().conjugate()

    def _eval_applyfunc(self, f):
        out = self._new(self.rows, self.cols, [f(x) for x in self])
        return out

    def _eval_as_real_imag(self):  # type: ignore
        return (self.applyfunc(re), self.applyfunc(im))

    def _eval_conjugate(self):
        return self.applyfunc(lambda x: x.conjugate())

    def _eval_permute_cols(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[i, mapping[j]]

        return self._new(self.rows, self.cols, entry)

    def _eval_permute_rows(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[mapping[i], j]

        return self._new(self.rows, self.cols, entry)

    def _eval_trace(self):
        return sum(self[i, i] for i in range(self.rows))

    def _eval_transpose(self):
        return self._new(self.cols, self.rows, lambda i, j: self[j, i])

    def adjoint(self):
        """Conjugate transpose or Hermitian conjugation."""
        return self._eval_adjoint()

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        return self._eval_applyfunc(f)

    def as_real_imag(self, deep=True, **hints):
        """Returns a tuple containing the (real, imaginary) part of matrix."""
        # XXX: Ignoring deep and hints...
        return self._eval_as_real_imag()

    def conjugate(self):
        """Return the by-element conjugation.

        Examples
        ========

        >>> from sympy import SparseMatrix, I
        >>> a = SparseMatrix(((1, 2 + I), (3, 4), (I, -I)))
        >>> a
        Matrix([
        [1, 2 + I],
        [3,     4],
        [I,    -I]])
        >>> a.C
        Matrix([
        [ 1, 2 - I],
        [ 3,     4],
        [-I,     I]])

        See Also
        ========

        transpose: Matrix transposition
        H: Hermite conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        return self._eval_conjugate()

    def doit(self, **hints):
        return self.applyfunc(lambda x: x.doit(**hints))

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """Apply evalf() to each element of self."""
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        return self.applyfunc(lambda i: i.evalf(n, **options))

    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
               mul=True, log=True, multinomial=True, basic=True, **hints):
        """Apply core.function.expand to each entry of the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix(1, 1, [x*(x+1)])
        Matrix([[x*(x + 1)]])
        >>> _.expand()
        Matrix([[x**2 + x]])

        """
        return self.applyfunc(lambda x: x.expand(
            deep, modulus, power_base, power_exp, mul, log, multinomial, basic,
            **hints))

    @property
    def H(self):
        """Return Hermite conjugate.

        Examples
        ========

        >>> from sympy import Matrix, I
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m
        Matrix([
        [    0],
        [1 + I],
        [    2],
        [    3]])
        >>> m.H
        Matrix([[0, 1 - I, 2, 3]])

        See Also
        ========

        conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        return self.T.C

    def permute(self, perm, orientation='rows', direction='forward'):
        r"""Permute the rows or columns of a matrix by the given list of
        swaps.

        Parameters
        ==========

        perm : Permutation, list, or list of lists
            A representation for the permutation.

            If it is ``Permutation``, it is used directly with some
            resizing with respect to the matrix size.

            If it is specified as list of lists,
            (e.g., ``[[0, 1], [0, 2]]``), then the permutation is formed
            from applying the product of cycles. The direction how the
            cyclic product is applied is described in below.

            If it is specified as a list, the list should represent
            an array form of a permutation. (e.g., ``[1, 2, 0]``) which
            would would form the swapping function
            `0 \mapsto 1, 1 \mapsto 2, 2\mapsto 0`.

        orientation : 'rows', 'cols'
            A flag to control whether to permute the rows or the columns

        direction : 'forward', 'backward'
            A flag to control whether to apply the permutations from
            the start of the list first, or from the back of the list
            first.

            For example, if the permutation specification is
            ``[[0, 1], [0, 2]]``,

            If the flag is set to ``'forward'``, the cycle would be
            formed as `0 \mapsto 2, 2 \mapsto 1, 1 \mapsto 0`.

            If the flag is set to ``'backward'``, the cycle would be
            formed as `0 \mapsto 1, 1 \mapsto 2, 2 \mapsto 0`.

            If the argument ``perm`` is not in a form of list of lists,
            this flag takes no effect.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.permute([[0, 1], [0, 2]], orientation='rows', direction='forward')
        Matrix([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.permute([[0, 1], [0, 2]], orientation='rows', direction='backward')
        Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]])

        Notes
        =====

        If a bijective function
        `\sigma : \mathbb{N}_0 \rightarrow \mathbb{N}_0` denotes the
        permutation.

        If the matrix `A` is the matrix to permute, represented as
        a horizontal or a vertical stack of vectors:

        .. math::
            A =
            \begin{bmatrix}
            a_0 \\ a_1 \\ \vdots \\ a_{n-1}
            \end{bmatrix} =
            \begin{bmatrix}
            \alpha_0 & \alpha_1 & \cdots & \alpha_{n-1}
            \end{bmatrix}

        If the matrix `B` is the result, the permutation of matrix rows
        is defined as:

        .. math::
            B := \begin{bmatrix}
            a_{\sigma(0)} \\ a_{\sigma(1)} \\ \vdots \\ a_{\sigma(n-1)}
            \end{bmatrix}

        And the permutation of matrix columns is defined as:

        .. math::
            B := \begin{bmatrix}
            \alpha_{\sigma(0)} & \alpha_{\sigma(1)} &
            \cdots & \alpha_{\sigma(n-1)}
            \end{bmatrix}
        """
        from sympy.combinatorics import Permutation

        # allow british variants and `columns`
        if direction == 'forwards':
            direction = 'forward'
        if direction == 'backwards':
            direction = 'backward'
        if orientation == 'columns':
            orientation = 'cols'

        if direction not in ('forward', 'backward'):
            raise TypeError("direction='{}' is an invalid kwarg. "
                            "Try 'forward' or 'backward'".format(direction))
        if orientation not in ('rows', 'cols'):
            raise TypeError("orientation='{}' is an invalid kwarg. "
                            "Try 'rows' or 'cols'".format(orientation))

        if not isinstance(perm, (Permutation, Iterable)):
            raise ValueError(
                "{} must be a list, a list of lists, "
                "or a SymPy permutation object.".format(perm))

        # ensure all swaps are in range
        max_index = self.rows if orientation == 'rows' else self.cols
        if not all(0 <= t <= max_index for t in flatten(list(perm))):
            raise IndexError("`swap` indices out of range.")

        if perm and not isinstance(perm, Permutation) and \
            isinstance(perm[0], Iterable):
            if direction == 'forward':
                perm = list(reversed(perm))
            perm = Permutation(perm, size=max_index+1)
        else:
            perm = Permutation(perm, size=max_index+1)

        if orientation == 'rows':
            return self._eval_permute_rows(perm)
        if orientation == 'cols':
            return self._eval_permute_cols(perm)

    def permute_cols(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='cols', direction=direction)``

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='cols', direction=direction)

    def permute_rows(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='rows', direction=direction)``

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='rows', direction=direction)

    def refine(self, assumptions=True):
        """Apply refine to each element of the matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, Abs, sqrt, Q
        >>> x = Symbol('x')
        >>> Matrix([[Abs(x)**2, sqrt(x**2)],[sqrt(x**2), Abs(x)**2]])
        Matrix([
        [ Abs(x)**2, sqrt(x**2)],
        [sqrt(x**2),  Abs(x)**2]])
        >>> _.refine(Q.real(x))
        Matrix([
        [  x**2, Abs(x)],
        [Abs(x),   x**2]])

        """
        return self.applyfunc(lambda x: refine(x, assumptions))

    def replace(self, F, G, map=False, simultaneous=True, exact=None):
        """Replaces Function F in Matrix entries with Function G.

        Examples
        ========

        >>> from sympy import symbols, Function, Matrix
        >>> F, G = symbols('F, G', cls=Function)
        >>> M = Matrix(2, 2, lambda i, j: F(i+j)) ; M
        Matrix([
        [F(0), F(1)],
        [F(1), F(2)]])
        >>> N = M.replace(F,G)
        >>> N
        Matrix([
        [G(0), G(1)],
        [G(1), G(2)]])
        """
        return self.applyfunc(
            lambda x: x.replace(F, G, map=map, simultaneous=simultaneous, exact=exact))

    def rot90(self, k=1):
        """Rotates Matrix by 90 degrees

        Parameters
        ==========

        k : int
            Specifies how many times the matrix is rotated by 90 degrees
            (clockwise when positive, counter-clockwise when negative).

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> A = Matrix(2, 2, symbols('a:d'))
        >>> A
        Matrix([
        [a, b],
        [c, d]])

        Rotating the matrix clockwise one time:

        >>> A.rot90(1)
        Matrix([
        [c, a],
        [d, b]])

        Rotating the matrix anticlockwise two times:

        >>> A.rot90(-2)
        Matrix([
        [d, c],
        [b, a]])
        """

        mod = k%4
        if mod == 0:
            return self
        if mod == 1:
            return self[::-1, ::].T
        if mod == 2:
            return self[::-1, ::-1]
        if mod == 3:
            return self[::, ::-1].T

    def simplify(self, **kwargs):
        """Apply simplify to each element of the matrix.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, sin, cos
        >>> SparseMatrix(1, 1, [x*sin(y)**2 + x*cos(y)**2])
        Matrix([[x*sin(y)**2 + x*cos(y)**2]])
        >>> _.simplify()
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.simplify(**kwargs))

    def subs(self, *args, **kwargs):  # should mirror core.basic.subs
        """Return a new matrix with subs applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.subs(x, y)
        Matrix([[y]])
        >>> Matrix(_).subs(y, x)
        Matrix([[x]])
        """

        if len(args) == 1 and  not isinstance(args[0], (dict, set)) and iter(args[0]) and not is_sequence(args[0]):
            args = (list(args[0]),)

        return self.applyfunc(lambda x: x.subs(*args, **kwargs))

    def trace(self):
        """
        Returns the trace of a square matrix i.e. the sum of the
        diagonal elements.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.trace()
        5

        """
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        return self._eval_trace()

    def transpose(self):
        """
        Returns the transpose of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.transpose()
        Matrix([
        [1, 3],
        [2, 4]])

        >>> from sympy import Matrix, I
        >>> m=Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m.transpose()
        Matrix([
        [    1, 3],
        [2 + I, 4]])
        >>> m.T == m.transpose()
        True

        See Also
        ========

        conjugate: By-element conjugation

        """
        return self._eval_transpose()

    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    @property
    def C(self):
        '''By-element conjugation'''
        return self.conjugate()

    def n(self, *args, **kwargs):
        """Apply evalf() to each element of self."""
        return self.evalf(*args, **kwargs)

    def xreplace(self, rule):  # should mirror core.basic.xreplace
        """Return a new matrix with xreplace applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.xreplace({x: y})
        Matrix([[y]])
        >>> Matrix(_).xreplace({y: x})
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.xreplace(rule))

    def _eval_simplify(self, **kwargs):
        # XXX: We can't use self.simplify here as mutable subclasses will
        # override simplify and have it return None
        return MatrixOperations.simplify(self, **kwargs)

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        return self.applyfunc(lambda x: trigsimp(x, **opts))

    def upper_triangular(self, k=0):
        """Return the elements on and above the kth diagonal of a matrix.
        If k is not specified then simply returns upper-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.upper_triangular()
        Matrix([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]])

        >>> A.upper_triangular(2)
        Matrix([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

        >>> A.upper_triangular(-1)
        Matrix([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1]])

        """

        def entry(i, j):
            return self[i, j] if i + k <= j else self.zero

        return self._new(self.rows, self.cols, entry)


    def lower_triangular(self, k=0):
        """Return the elements on and below the kth diagonal of a matrix.
        If k is not specified then simply returns lower-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.lower_triangular()
        Matrix([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]])

        >>> A.lower_triangular(-2)
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]])

        >>> A.lower_triangular(1)
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]])

        """

        def entry(i, j):
            return self[i, j] if i + k >= j else self.zero

        return self._new(self.rows, self.cols, entry)



class MatrixArithmetic(MatrixRequired):
    """Provides basic matrix arithmetic operations.
    Should not be instantiated directly."""

    _op_priority = 10.01

    def _eval_Abs(self):
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        return self._new(self.rows, self.cols,
                         lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):
        def entry(i, j):
            vec = [self[i,k]*other[k,j] for k in range(self.cols)]
            try:
                return Add(*vec)
            except (TypeError, SympifyError):
                # Some matrices don't work with `sum` or `Add`
                # They don't work with `sum` because `sum` tries to add `0`
                # Fall back to a safe way to multiply if the `Add` fails.
                return reduce(lambda a, b: a + b, vec)

        return self._new(self.rows, other.cols, entry)

    def _eval_matrix_mul_elementwise(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other[i,j])

    def _eval_matrix_rmul(self, other):
        def entry(i, j):
            return sum(other[i,k]*self[k,j] for k in range(other.cols))
        return self._new(other.rows, self.cols, entry)

    def _eval_pow_by_recursion(self, num):
        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion(num - 1)
        else:
            a = b = self._eval_pow_by_recursion(num // 2)

        return a.multiply(b)

    def _eval_pow_by_cayley(self, exp):
        from sympy.discrete.recurrences import linrec_coeffs
        row = self.shape[0]
        p = self.charpoly()

        coeffs = (-p).all_coeffs()[1:]
        coeffs = linrec_coeffs(coeffs, exp)
        new_mat = self.eye(row)
        ans = self.zeros(row)

        for i in range(row):
            ans += coeffs[i]*new_mat
            new_mat *= self

        return ans

    def _eval_pow_by_recursion_dotprodsimp(self, num, prevsimp=None):
        if prevsimp is None:
            prevsimp = [True]*len(self)

        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion_dotprodsimp(num - 1,
                    prevsimp=prevsimp)
        else:
            a = b = self._eval_pow_by_recursion_dotprodsimp(num // 2,
                    prevsimp=prevsimp)

        m     = a.multiply(b, dotprodsimp=False)
        lenm  = len(m)
        elems = [None]*lenm

        for i in range(lenm):
            if prevsimp[i]:
                elems[i], prevsimp[i] = _dotprodsimp(m[i], withsimp=True)
            else:
                elems[i] = m[i]

        return m._new(m.rows, m.cols, elems)

    def _eval_scalar_mul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other)

    def _eval_scalar_rmul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: other*self[i,j])

    def _eval_Mod(self, other):
        return self._new(self.rows, self.cols, lambda i, j: Mod(self[i, j], other))

    # Python arithmetic functions
    def __abs__(self):
        """Returns a new matrix with entry-wise absolute values."""
        return self._eval_Abs()

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """Return self + other, raising ShapeError if shapes do not match."""
        if isinstance(other, NDimArray): # Matrix and array addition is currently not implemented
            return NotImplemented
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check.
        if hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ShapeError("Matrix size mismatch: %s + %s" % (
                    self.shape, other.shape))

        # honest SymPy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            # call the highest-priority class's _eval_add
            a, b = self, other
            if a.__class__ != classof(a, b):
                b, a = a, b
            return a._eval_add(b)
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_add(self, other)

        raise TypeError('cannot add %s and %s' % (type(self), type(other)))

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            return NotImplemented

        return self.__mul__(other)

    def __mod__(self, other):
        return self.applyfunc(lambda x: x % other)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """Return self*other where other is either a scalar or a matrix
        of compatible dimensions.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> 2*A == A*2 == Matrix([[2, 4, 6], [8, 10, 12]])
        True
        >>> B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> A*B
        Matrix([
        [30, 36, 42],
        [66, 81, 96]])
        >>> B*A
        Traceback (most recent call last):
        ...
        ShapeError: Matrices size mismatch.
        >>>

        See Also
        ========

        matrix_multiply_elementwise
        """

        return self.multiply(other)

    def multiply(self, other, dotprodsimp=None):
        """Same as __mul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """

        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check. Double check other is not explicitly not a Matrix.
        if (hasattr(other, 'shape') and len(other.shape) == 2 and
            (getattr(other, 'is_Matrix', True) or
             getattr(other, 'is_MatrixLike', True))):
            if self.shape[1] != other.shape[0]:
                raise ShapeError("Matrix size mismatch: %s * %s." % (
                    self.shape, other.shape))

        # honest SymPy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_mul(other)
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m

        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_mul(self, other)

        # if 'other' is not iterable then scalar multiplication.
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                pass

        return NotImplemented

    def multiply_elementwise(self, other):
        """Return the Hadamard product (elementwise product) of A and B

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
        >>> A.multiply_elementwise(B)
        Matrix([
        [  0, 10, 200],
        [300, 40,   5]])

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.cross
        sympy.matrices.matrixbase.MatrixBase.dot
        multiply
        """
        if self.shape != other.shape:
            raise ShapeError("Matrix shapes must agree {} != {}".format(self.shape, other.shape))

        return self._eval_matrix_mul_elementwise(other)

    def __neg__(self):
        return self._eval_scalar_mul(-1)

    @call_highest_priority('__rpow__')
    def __pow__(self, exp):
        """Return self**exp a scalar or symbol."""

        return self.pow(exp)


    def pow(self, exp, method=None):
        r"""Return self**exp a scalar or symbol.

        Parameters
        ==========

        method : multiply, mulsimp, jordan, cayley
            If multiply then it returns exponentiation using recursion.
            If jordan then Jordan form exponentiation will be used.
            If cayley then the exponentiation is done using Cayley-Hamilton
            theorem.
            If mulsimp then the exponentiation is done using recursion
            with dotprodsimp. This specifies whether intermediate term
            algebraic simplification is used during naive matrix power to
            control expression blowup and thus speed up calculation.
            If None, then it heuristically decides which method to use.

        """

        if method is not None and method not in ['multiply', 'mulsimp', 'jordan', 'cayley']:
            raise TypeError('No such method')
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        a = self
        jordan_pow = getattr(a, '_matrix_pow_by_jordan_blocks', None)
        exp = sympify(exp)

        if exp.is_zero:
            return a._new(a.rows, a.cols, lambda i, j: int(i == j))
        if exp == 1:
            return a

        diagonal = getattr(a, 'is_diagonal', None)
        if diagonal is not None and diagonal():
            return a._new(a.rows, a.cols, lambda i, j: a[i,j]**exp if i == j else 0)

        if exp.is_Number and exp % 1 == 0:
            if a.rows == 1:
                return a._new([[a[0]**exp]])
            if exp < 0:
                exp = -exp
                a = a.inv()
        # When certain conditions are met,
        # Jordan block algorithm is faster than
        # computation by recursion.
        if method == 'jordan':
            try:
                return jordan_pow(exp)
            except MatrixError:
                if method == 'jordan':
                    raise

        elif method == 'cayley':
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("cayley method is only valid for integer powers")
            return a._eval_pow_by_cayley(exp)

        elif method == "mulsimp":
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("mulsimp method is only valid for integer powers")
            return a._eval_pow_by_recursion_dotprodsimp(exp)

        elif method == "multiply":
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("multiply method is only valid for integer powers")
            return a._eval_pow_by_recursion(exp)

        elif method is None and exp.is_Number and exp % 1 == 0:
            if exp.is_Float:
                exp = Integer(exp)
            # Decide heuristically which method to apply
            if a.rows == 2 and exp > 100000:
                return jordan_pow(exp)
            elif _get_intermediate_simp_bool(True, None):
                return a._eval_pow_by_recursion_dotprodsimp(exp)
            elif exp > 10000:
                return a._eval_pow_by_cayley(exp)
            else:
                return a._eval_pow_by_recursion(exp)

        if jordan_pow:
            try:
                return jordan_pow(exp)
            except NonInvertibleMatrixError:
                # Raised by jordan_pow on zero determinant matrix unless exp is
                # definitely known to be a non-negative integer.
                # Here we raise if n is definitely not a non-negative integer
                # but otherwise we can leave this as an unevaluated MatPow.
                if exp.is_integer is False or exp.is_nonnegative is False:
                    raise

        from sympy.matrices.expressions import MatPow
        return MatPow(a, exp)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    @call_highest_priority('__matmul__')
    def __rmatmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            return NotImplemented

        return self.__rmul__(other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self.rmultiply(other)

    def rmultiply(self, other, dotprodsimp=None):
        """Same as __rmul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check. Double check other is not explicitly not a Matrix.
        if (hasattr(other, 'shape') and len(other.shape) == 2 and
            (getattr(other, 'is_Matrix', True) or
             getattr(other, 'is_MatrixLike', True))):
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

        # honest SymPy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_rmul(other)
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_rmul(self, other)

        # if 'other' is not iterable then scalar multiplication.
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                pass

        return NotImplemented

    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        return (-self) + a

    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        return self + (-a)


class MatrixCommon(MatrixArithmetic, MatrixOperations, MatrixProperties,
                  MatrixSpecial, MatrixShaping):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""
    _diff_wrt: bool = True


class _MinimalMatrix:
    """Class providing the minimum functionality
    for a matrix-like object and implementing every method
    required for a `MatrixRequired`.  This class does not have everything
    needed to become a full-fledged SymPy object, but it will satisfy the
    requirements of anything inheriting from `MatrixRequired`.  If you wish
    to make a specialized matrix type, make sure to implement these
    methods and properties with the exception of `__init__` and `__repr__`
    which are included for convenience."""

    is_MatrixLike = True
    _sympify = staticmethod(sympify)
    _class_priority = 3
    zero = S.Zero
    one = S.One

    is_Matrix = True
    is_MatrixExpr = False

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, rows, cols=None, mat=None, copy=False):
        if isfunction(mat):
            # if we passed in a function, use that to populate the indices
            mat = [mat(i, j) for i in range(rows) for j in range(cols)]
        if cols is None and mat is None:
            mat = rows
        rows, cols = getattr(mat, 'shape', (rows, cols))
        try:
            # if we passed in a list of lists, flatten it and set the size
            if cols is None and mat is None:
                mat = rows
            cols = len(mat[0])
            rows = len(mat)
            mat = [x for l in mat for x in l]
        except (IndexError, TypeError):
            pass
        self.mat = tuple(self._sympify(x) for x in mat)
        self.rows, self.cols = rows, cols
        if self.rows is None or self.cols is None:
            raise NotImplementedError("Cannot initialize matrix with given parameters")

    def __getitem__(self, key):
        def _normalize_slices(row_slice, col_slice):
            """Ensure that row_slice and col_slice do not have
            `None` in their arguments.  Any integers are converted
            to slices of length 1"""
            if not isinstance(row_slice, slice):
                row_slice = slice(row_slice, row_slice + 1, None)
            row_slice = slice(*row_slice.indices(self.rows))

            if not isinstance(col_slice, slice):
                col_slice = slice(col_slice, col_slice + 1, None)
            col_slice = slice(*col_slice.indices(self.cols))

            return (row_slice, col_slice)

        def _coord_to_index(i, j):
            """Return the index in _mat corresponding
            to the (i,j) position in the matrix. """
            return i * self.cols + j

        if isinstance(key, tuple):
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                # if the coordinates are not slices, make them so
                # and expand the slices so they don't contain `None`
                i, j = _normalize_slices(i, j)

                rowsList, colsList = list(range(self.rows))[i], \
                                     list(range(self.cols))[j]
                indices = (i * self.cols + j for i in rowsList for j in
                           colsList)
                return self._new(len(rowsList), len(colsList),
                                 [self.mat[i] for i in indices])

            # if the key is a tuple of ints, change
            # it to an array index
            key = _coord_to_index(i, j)
        return self.mat[key]

    def __eq__(self, other):
        try:
            classof(self, other)
        except TypeError:
            return False
        return (
            self.shape == other.shape and list(self) == list(other))

    def __len__(self):
        return self.rows*self.cols

    def __repr__(self):
        return "_MinimalMatrix({}, {}, {})".format(self.rows, self.cols,
                                                   self.mat)

    @property
    def shape(self):
        return (self.rows, self.cols)


class _CastableMatrix: # this is needed here ONLY FOR TESTS.
    def as_mutable(self):
        return self

    def as_immutable(self):
        return self


class _MatrixWrapper:
    """Wrapper class providing the minimum functionality for a matrix-like
    object: .rows, .cols, .shape, indexability, and iterability. CommonMatrix
    math operations should work on matrix-like objects. This one is intended for
    matrix-like objects which use the same indexing format as SymPy with respect
    to returning matrix elements instead of rows for non-tuple indexes.
    """

    is_Matrix     = False # needs to be here because of __getattr__
    is_MatrixLike = True

    def __init__(self, mat, shape):
        self.mat = mat
        self.shape = shape
        self.rows, self.cols = shape

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return sympify(self.mat.__getitem__(key))

        return sympify(self.mat.__getitem__((key // self.rows, key % self.cols)))

    def __iter__(self): # supports numpy.matrix and numpy.array
        mat = self.mat
        cols = self.cols

        return iter(sympify(mat[r, c]) for r in range(self.rows) for c in range(cols))


def _matrixify(mat):
    """If `mat` is a Matrix or is matrix-like,
    return a Matrix or MatrixWrapper object.  Otherwise
    `mat` is passed through without modification."""

    if getattr(mat, 'is_Matrix', False) or getattr(mat, 'is_MatrixLike', False):
        return mat

    if not(getattr(mat, 'is_Matrix', True) or getattr(mat, 'is_MatrixLike', True)):
        return mat

    shape = None

    if hasattr(mat, 'shape'): # numpy, scipy.sparse
        if len(mat.shape) == 2:
            shape = mat.shape
    elif hasattr(mat, 'rows') and hasattr(mat, 'cols'): # mpmath
        shape = (mat.rows, mat.cols)

    if shape:
        return _MatrixWrapper(mat, shape)

    return mat


def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    if not isinstance(j, int):
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError("Invalid index a[%r]" % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    return int(j)


def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.

    Currently the strategy is that immutability is contagious.

    Examples
    ========

    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.matrixbase import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    priority_A = getattr(A, '_class_priority', None)
    priority_B = getattr(B, '_class_priority', None)
    if None not in (priority_A, priority_B):
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__

    try:
        import numpy
    except ImportError:
        pass
    else:
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__

    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))
<|endoftext|>from __future__ import annotations
import random

from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence

from .exceptions import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrixbase import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve


__doctest_requires__ = {('symarray',): ['numpy']}


def _iszero(x):
    """Returns True if x is zero."""
    return x.is_zero


class DenseMatrix(RepMatrix):
    """Matrix implementation based on DomainMatrix as the internal representation"""

    #
    # DenseMatrix is a superclass for both MutableDenseMatrix and
    # ImmutableDenseMatrix. Methods shared by both classes but not for the
    # Sparse classes should be implemented here.
    #

    is_MatrixExpr: bool = False

    _op_priority = 10.01
    _class_priority = 4

    @property
    def _mat(self):
        sympy_deprecation_warning(
            """
            The private _mat attribute of Matrix is deprecated. Use the
            .flat() method instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-private-matrix-attributes"
        )

        return self.flat()

    def _eval_inverse(self, **kwargs):
        return self.inv(method=kwargs.get('method', 'GE'),
                        iszerofunc=kwargs.get('iszerofunc', _iszero),
                        try_block_diag=kwargs.get('try_block_diag', False))

    def as_immutable(self):
        """Returns an Immutable version of this Matrix
        """
        from .immutable import ImmutableDenseMatrix as cls
        return cls._fromrep(self._rep.copy())

    def as_mutable(self):
        """Returns a mutable version of this matrix

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        return Matrix(self)

    def cholesky(self, hermitian=True):
        return _cholesky(self, hermitian=hermitian)

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition(self, hermitian=hermitian)

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve(self, rhs)

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve(self, rhs)

    cholesky.__doc__               = _cholesky.__doc__
    LDLdecomposition.__doc__       = _LDLdecomposition.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__


def _force_mutable(x):
    """Return a matrix as a Matrix, otherwise return x."""
    if getattr(x, 'is_Matrix', False):
        return x.as_mutable()
    elif isinstance(x, Basic):
        return x
    elif hasattr(x, '__array__'):
        a = x.__array__()
        if len(a.shape) == 0:
            return sympify(a)
        return Matrix(x)
    return x


class MutableDenseMatrix(DenseMatrix, MutableRepMatrix):

    def simplify(self, **kwargs):
        """Applies simplify to the elements of a matrix in place.

        This is a shortcut for M.applyfunc(lambda x: simplify(x, ratio, measure))

        See Also
        ========

        sympy.simplify.simplify.simplify
        """
        from sympy.simplify.simplify import simplify as _simplify
        for (i, j), element in self.todok().items():
            self[i, j] = _simplify(element, **kwargs)


MutableMatrix = Matrix = MutableDenseMatrix

###########
# Numpy Utility Functions:
# list2numpy, matrix2numpy, symmarray
###########


def list2numpy(l, dtype=object):  # pragma: no cover
    """Converts Python list of SymPy expressions to a NumPy array.

    See Also
    ========

    matrix2numpy
    """
    from numpy import empty
    a = empty(len(l), dtype)
    for i, s in enumerate(l):
        a[i] = s
    return a


def matrix2numpy(m, dtype=object):  # pragma: no cover
    """Converts SymPy's matrix to a NumPy array.

    See Also
    ========

    list2numpy
    """
    from numpy import empty
    a = empty(m.shape, dtype)
    for i in range(m.rows):
        for j in range(m.cols):
            a[i, j] = m[i, j]
    return a


###########
# Rotation matrices:
# rot_givens, rot_axis[123], rot_ccw_axis[123]
###########


def rot_givens(i, j, theta, dim=3):
    r"""Returns a a Givens rotation matrix, a a rotation in the
    plane spanned by two coordinates axes.

    Explanation
    ===========

    The Givens rotation corresponds to a generalization of rotation
    matrices to any number of dimensions, given by:

    .. math::
        G(i, j, \theta) =
            \begin{bmatrix}
                1   & \cdots &    0   & \cdots &    0   & \cdots &    0   \\
                \vdots & \ddots & \vdots &        & \vdots &        & \vdots \\
                0   & \cdots &    c   & \cdots &   -s   & \cdots &    0   \\
                \vdots &        & \vdots & \ddots & \vdots &        & \vdots \\
                0   & \cdots &    s   & \cdots &    c   & \cdots &    0   \\
                \vdots &        & \vdots &        & \vdots & \ddots & \vdots \\
                0   & \cdots &    0   & \cdots &    0   & \cdots &    1
            \end{bmatrix}

    Where $c = \cos(\theta)$ and $s = \sin(\theta)$ appear at the intersections
    ``i``\th and ``j``\th rows and columns.

    For fixed ``i > j``\, the non-zero elements of a Givens matrix are
    given by:

    - $g_{kk} = 1$ for $k \ne i,\,j$
    - $g_{kk} = c$ for $k = i,\,j$
    - $g_{ji} = -g_{ij} = -s$

    Parameters
    ==========

    i : int between ``0`` and ``dim - 1``
        Represents first axis
    j : int between ``0`` and ``dim - 1``
        Represents second axis
    dim : int bigger than 1
        Number of dimensions. Defaults to 3.

    Examples
    ========

    >>> from sympy import pi, rot_givens

    A counterclockwise rotation of pi/3 (60 degrees) around
    the third axis (z-axis):

    >>> rot_givens(1, 0, pi/3)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_givens(1, 0, pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    This can be generalized to any number
    of dimensions:

    >>> rot_givens(1, 0, pi/2, dim=4)
    Matrix([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Givens_rotation

    See Also
    ========

    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    if not isinstance(dim, int) or dim < 2:
        raise ValueError('dim must be an integer biggen than one, '
                         'got {}.'.format(dim))

    if i == j:
        raise ValueError('i and j must be different, '
                         'got ({}, {})'.format(i, j))

    for ij in [i, j]:
        if not isinstance(ij, int) or ij < 0 or ij > dim - 1:
            raise ValueError('i and j must be integers between 0 and '
                             '{}, got i={} and j={}.'.format(dim-1, i, j))

    theta = sympify(theta)
    c = cos(theta)
    s = sin(theta)
    M = eye(dim)
    M[i, i] = c
    M[j, j] = c
    M[i, j] = s
    M[j, i] = -s
    return M


def rot_axis3(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 3-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `z`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                 \cos(\theta) & \sin(\theta) & 0 \\
                -\sin(\theta) & \cos(\theta) & 0 \\
                            0 &            0 & 1
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis3

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis3(theta)
    Matrix([
    [       1/2, sqrt(3)/2, 0],
    [-sqrt(3)/2,       1/2, 0],
    [         0,         0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis3(pi/2)
    Matrix([
    [ 0, 1, 0],
    [-1, 0, 0],
    [ 0, 0, 1]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    """
    return rot_givens(0, 1, theta, dim=3)


def rot_axis2(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 2-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `y`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                \cos(\theta) & 0 & -\sin(\theta) \\
                           0 & 1 &             0 \\
                \sin(\theta) & 0 &  \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis2

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis2(theta)
    Matrix([
    [      1/2, 0, -sqrt(3)/2],
    [        0, 1,          0],
    [sqrt(3)/2, 0,        1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis2(pi/2)
    Matrix([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(2, 0, theta, dim=3)


def rot_axis1(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &             0 &            0 \\
                0 &  \cos(\theta) & \sin(\theta) \\
                0 & -\sin(\theta) & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis1(theta)
    Matrix([
    [1,          0,         0],
    [0,        1/2, sqrt(3)/2],
    [0, -sqrt(3)/2,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis1(pi/2)
    Matrix([
    [1,  0, 0],
    [0,  0, 1],
    [0, -1, 0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    """
    return rot_givens(1, 2, theta, dim=3)


def rot_ccw_axis3(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 3-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `z`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                \cos(\theta) & -\sin(\theta) & 0 \\
                \sin(\theta) &  \cos(\theta) & 0 \\
                           0 &             0 & 1
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis3

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis3(theta)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis3(pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    """
    return rot_givens(1, 0, theta, dim=3)


def rot_ccw_axis2(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 2-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `y`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                 \cos(\theta) & 0 & \sin(\theta) \\
                            0 & 1 &            0 \\
                -\sin(\theta) & 0 & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis2

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis2(theta)
    Matrix([
    [       1/2, 0, sqrt(3)/2],
    [         0, 1,         0],
    [-sqrt(3)/2, 0,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis2(pi/2)
    Matrix([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(0, 2, theta, dim=3)


def rot_ccw_axis1(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &            0 &             0 \\
                0 & \cos(\theta) & -\sin(\theta) \\
                0 & \sin(\theta) &  \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis1(theta)
    Matrix([
    [1,         0,          0],
    [0,       1/2, -sqrt(3)/2],
    [0, sqrt(3)/2,        1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis1(pi/2)
    Matrix([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(2, 1, theta, dim=3)


@doctest_depends_on(modules=('numpy',))
def symarray(prefix, shape, **kwargs):  # pragma: no cover
    r"""Create a numpy ndarray of symbols (as an object array).

    The created symbols are named ``prefix_i1_i2_``...  You should thus provide a
    non-empty prefix if you want your symbols to be unique for different output
    arrays, as SymPy symbols with identical names are the same object.

    Parameters
    ----------

    prefix : string
      A prefix prepended to the name of every symbol.

    shape : int or tuple
      Shape of the created array.  If an int, the array is one-dimensional; for
      more than one dimension the shape must be a tuple.

    \*\*kwargs : dict
      keyword arguments passed on to Symbol

    Examples
    ========
    These doctests require numpy.

    >>> from sympy import symarray
    >>> symarray('', 3)
    [_0 _1 _2]

    If you want multiple symarrays to contain distinct symbols, you *must*
    provide unique prefixes:

    >>> a = symarray('', 3)
    >>> b = symarray('', 3)
    >>> a[0] == b[0]
    True
    >>> a = symarray('a', 3)
    >>> b = symarray('b', 3)
    >>> a[0] == b[0]
    False

    Creating symarrays with a prefix:

    >>> symarray('a', 3)
    [a_0 a_1 a_2]

    For more than one dimension, the shape must be given as a tuple:

    >>> symarray('a', (2, 3))
    [[a_0_0 a_0_1 a_0_2]
     [a_1_0 a_1_1 a_1_2]]
    >>> symarray('a', (2, 3, 2))
    [[[a_0_0_0 a_0_0_1]
      [a_0_1_0 a_0_1_1]
      [a_0_2_0 a_0_2_1]]
    <BLANKLINE>
     [[a_1_0_0 a_1_0_1]
      [a_1_1_0 a_1_1_1]
      [a_1_2_0 a_1_2_1]]]

    For setting assumptions of the underlying Symbols:

    >>> [s.is_real for s in symarray('a', 2, real=True)]
    [True, True]
    """
    from numpy import empty, ndindex
    arr = empty(shape, dtype=object)
    for index in ndindex(shape):
        arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))),
                            **kwargs)
    return arr


###############
# Functions
###############

def casoratian(seqs, n, zero=True):
    """Given linear difference operator L of order 'k' and homogeneous
       equation Ly = 0 we want to compute kernel of L, which is a set
       of 'k' sequences: a(n), b(n), ... z(n).

       Solutions of L are linearly independent iff their Casoratian,
       denoted as C(a, b, ..., z), do not vanish for n = 0.

       Casoratian is defined by k x k determinant::

                  +  a(n)     b(n)     . . . z(n)     +
                  |  a(n+1)   b(n+1)   . . . z(n+1)   |
                  |    .         .     .        .     |
                  |    .         .       .      .     |
                  |    .         .         .    .     |
                  +  a(n+k-1) b(n+k-1) . . . z(n+k-1) +

       It proves very useful in rsolve_hyper() where it is applied
       to a generating set of a recurrence to factor out linearly
       dependent solutions and return a basis:

       >>> from sympy import Symbol, casoratian, factorial
       >>> n = Symbol('n', integer=True)

       Exponential and factorial are linearly independent:

       >>> casoratian([2**n, factorial(n)], n) != 0
       True

    """

    seqs = list(map(sympify, seqs))

    if not zero:
        f = lambda i, j: seqs[j].subs(n, n + i)
    else:
        f = lambda i, j: seqs[j].subs(n, i)

    k = len(seqs)

    return Matrix(k, k, f).det()


def eye(*args, **kwargs):
    """Create square identity matrix n x n

    See Also
    ========

    diag
    zeros
    ones
    """

    return Matrix.eye(*args, **kwargs)


def diag(*values, strict=True, unpack=False, **kwargs):
    """Returns a matrix with the provided values placed on the
    diagonal. If non-square matrices are included, they will
    produce a block-diagonal matrix.

    Examples
    ========

    This version of diag is a thin wrapper to Matrix.diag that differs
    in that it treats all lists like matrices -- even when a single list
    is given. If this is not desired, either put a `*` before the list or
    set `unpack=True`.

    >>> from sympy import diag

    >>> diag([1, 2, 3], unpack=True)  # = diag(1,2,3) or diag(*[1,2,3])
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    >>> diag([1, 2, 3])  # a column vector
    Matrix([
    [1],
    [2],
    [3]])

    See Also
    ========
    .matrixbase.MatrixBase.eye
    .matrixbase.MatrixBase.diagonal
    .matrixbase.MatrixBase.diag
    .expressions.blockmatrix.BlockMatrix
    """
    return Matrix.diag(*values, strict=strict, unpack=unpack, **kwargs)


def GramSchmidt(vlist, orthonormal=False):
    """Apply the Gram-Schmidt process to a set of vectors.

    Parameters
    ==========

    vlist : List of Matrix
        Vectors to be orthogonalized for.

    orthonormal : Bool, optional
        If true, return an orthonormal basis.

    Returns
    =======

    vlist : List of Matrix
        Orthogonalized vectors

    Notes
    =====

    This routine is mostly duplicate from ``Matrix.orthogonalize``,
    except for some difference that this always raises error when
    linearly dependent vectors are found, and the keyword ``normalize``
    has been named as ``orthonormal`` in this function.

    See Also
    ========

    .matrixbase.MatrixBase.orthogonalize

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    return MutableDenseMatrix.orthogonalize(
        *vlist, normalize=orthonormal, rankcheck=True
    )


def hessian(f, varlist, constraints=()):
    """Compute Hessian matrix for a function f wrt parameters in varlist
    which may be given as a sequence or a row/column vector. A list of
    constraints may optionally be given.

    Examples
    ========

    >>> from sympy import Function, hessian, pprint
    >>> from sympy.abc import x, y
    >>> f = Function('f')(x, y)
    >>> g1 = Function('g')(x, y)
    >>> g2 = x**2 + 3*y
    >>> pprint(hessian(f, (x, y), [g1, g2]))
    [                   d               d            ]
    [     0        0    --(g(x, y))     --(g(x, y))  ]
    [                   dx              dy           ]
    [                                                ]
    [     0        0        2*x              3       ]
    [                                                ]
    [                     2               2          ]
    [d                   d               d           ]
    [--(g(x, y))  2*x   ---(f(x, y))   -----(f(x, y))]
    [dx                   2            dy dx         ]
    [                   dx                           ]
    [                                                ]
    [                     2               2          ]
    [d                   d               d           ]
    [--(g(x, y))   3   -----(f(x, y))   ---(f(x, y)) ]
    [dy                dy dx              2          ]
    [                                   dy           ]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hessian_matrix

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.jacobian
    wronskian
    """
    # f is the expression representing a function f, return regular matrix
    if isinstance(varlist, MatrixBase):
        if 1 not in varlist.shape:
            raise ShapeError("`varlist` must be a column or row vector.")
        if varlist.cols == 1:
            varlist = varlist.T
        varlist = varlist.tolist()[0]
    if is_sequence(varlist):
        n = len(varlist)
        if not n:
            raise ShapeError("`len(varlist)` must not be zero.")
    else:
        raise ValueError("Improper variable list in hessian function")
    if not getattr(f, 'diff'):
        # check differentiability
        raise ValueError("Function `f` (%s) is not differentiable" % f)
    m = len(constraints)
    N = m + n
    out = zeros(N)
    for k, g in enumerate(constraints):
        if not getattr(g, 'diff'):
            # check differentiability
            raise ValueError("Function `f` (%s) is not differentiable" % f)
        for i in range(n):
            out[k, i + m] = g.diff(varlist[i])
    for i in range(n):
        for j in range(i, n):
            out[i + m, j + m] = f.diff(varlist[i]).diff(varlist[j])
    for i in range(N):
        for j in range(i + 1, N):
            out[j, i] = out[i, j]
    return out


def jordan_cell(eigenval, n):
    """
    Create a Jordan block:

    Examples
    ========

    >>> from sympy import jordan_cell
    >>> from sympy.abc import x
    >>> jordan_cell(x, 4)
    Matrix([
    [x, 1, 0, 0],
    [0, x, 1, 0],
    [0, 0, x, 1],
    [0, 0, 0, x]])
    """

    return Matrix.jordan_block(size=n, eigenvalue=eigenval)


def matrix_multiply_elementwise(A, B):
    """Return the Hadamard product (elementwise product) of A and B

    >>> from sympy import Matrix, matrix_multiply_elementwise
    >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
    >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
    >>> matrix_multiply_elementwise(A, B)
    Matrix([
    [  0, 10, 200],
    [300, 40,   5]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.__mul__
    """
    return A.multiply_elementwise(B)


def ones(*args, **kwargs):
    """Returns a matrix of ones with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    See Also
    ========

    zeros
    eye
    diag
    """

    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')

    return Matrix.ones(*args, **kwargs)


def randMatrix(r, c=None, min=0, max=99, seed=None, symmetric=False,
               percent=100, prng=None):
    """Create random matrix with dimensions ``r`` x ``c``. If ``c`` is omitted
    the matrix will be square. If ``symmetric`` is True the matrix must be
    square. If ``percent`` is less than 100 then only approximately the given
    percentage of elements will be non-zero.

    The pseudo-random number generator used to generate matrix is chosen in the
    following way.

    * If ``prng`` is supplied, it will be used as random number generator.
      It should be an instance of ``random.Random``, or at least have
      ``randint`` and ``shuffle`` methods with same signatures.
    * if ``prng`` is not supplied but ``seed`` is supplied, then new
      ``random.Random`` with given ``seed`` will be created;
    * otherwise, a new ``random.Random`` with default seed will be used.

    Examples
    ========

    >>> from sympy import randMatrix
    >>> randMatrix(3) # doctest:+SKIP
    [25, 45, 27]
    [44, 54,  9]
    [23, 96, 46]
    >>> randMatrix(3, 2) # doctest:+SKIP
    [87, 29]
    [23, 37]
    [90, 26]
    >>> randMatrix(3, 3, 0, 2) # doctest:+SKIP
    [0, 2, 0]
    [2, 0, 1]
    [0, 0, 1]
    >>> randMatrix(3, symmetric=True) # doctest:+SKIP
    [85, 26, 29]
    [26, 71, 43]
    [29, 43, 57]
    >>> A = randMatrix(3, seed=1)
    >>> B = randMatrix(3, seed=2)
    >>> A == B
    False
    >>> A == randMatrix(3, seed=1)
    True
    >>> randMatrix(3, symmetric=True, percent=50) # doctest:+SKIP
    [77, 70,  0],
    [70,  0,  0],
    [ 0,  0, 88]
    """
    # Note that ``Random()`` is equivalent to ``Random(None)``
    prng = prng or random.Random(seed)

    if c is None:
        c = r

    if symmetric and r != c:
        raise ValueError('For symmetric matrices, r must equal c, but %i != %i' % (r, c))

    ij = range(r * c)
    if percent != 100:
        ij = prng.sample(ij, int(len(ij)*percent // 100))

    m = zeros(r, c)

    if not symmetric:
        for ijk in ij:
            i, j = divmod(ijk, c)
            m[i, j] = prng.randint(min, max)
    else:
        for ijk in ij:
            i, j = divmod(ijk, c)
            if i <= j:
                m[i, j] = m[j, i] = prng.randint(min, max)

    return m


def wronskian(functions, var, method='bareiss'):
    """
    Compute Wronskian for [] of functions

    ::

                         | f1       f2        ...   fn      |
                         | f1'      f2'       ...   fn'     |
                         |  .        .        .      .      |
        W(f1, ..., fn) = |  .        .         .     .      |
                         |  .        .          .    .      |
                         |  (n)      (n)            (n)     |
                         | D   (f1) D   (f2)  ...  D   (fn) |

    see: https://en.wikipedia.org/wiki/Wronskian

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.jacobian
    hessian
    """

    functions = [sympify(f) for f in functions]
    n = len(functions)
    if n == 0:
        return S.One
    W = Matrix(n, n, lambda i, j: functions[i].diff(var, j))
    return W.det(method)


def zeros(*args, **kwargs):
    """Returns a matrix of zeros with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    See Also
    ========

    ones
    eye
    diag
    """

    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')

    return Matrix.zeros(*args, **kwargs)
<|endoftext|>from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce

from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom, Basic
from sympy.core.kind import UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.mod import Mod
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.core.function import diff
from sympy.polys import cancel
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.printing import sstr
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.core.singleton import S
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.combinatorial.factorials import binomial, factorial

import mpmath as mp
from collections.abc import Callable
from sympy.utilities.iterables import reshape
from sympy.core.expr import Expr
from sympy.core.power import Pow
from sympy.core.symbol import uniquely_named_symbol

from .utilities import _dotprodsimp, _simplify as _utilities_simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.core.decorators import call_highest_priority
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import NotIterable

from .utilities import _get_intermediate_simp_bool

from .kind import MatrixKind

from .exceptions import (
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
)

from .utilities import _iszero, _is_zero_after_expand_mul

from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per,
    _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU,
    _minor, _minor_submatrix)

from .reductions import _is_echelon, _echelon_form, _rank, _rref

from .solvers import (
    _diagonal_solve, _lower_triangular_solve, _upper_triangular_solve,
    _cholesky_solve, _LDLsolve, _LUsolve, _QRsolve, _gauss_jordan_solve,
    _pinv_solve, _cramer_solve, _solve, _solve_least_squares)

from .inverse import (
    _pinv, _inv_ADJ, _inv_GE, _inv_LU, _inv_CH, _inv_LDL, _inv_QR,
    _inv, _inv_block)

from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize

from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)

from .decompositions import (
    _rank_decomposition, _cholesky, _LDLdecomposition,
    _LUdecomposition, _LUdecomposition_Simple, _LUdecompositionFF,
    _singular_value_decomposition, _QRdecomposition, _upper_hessenberg_decomposition)

from .graph import (
    _connected_components, _connected_components_decomposition,
    _strongly_connected_components, _strongly_connected_components_decomposition)


__doctest_requires__ = {
    ('MatrixBase.is_indefinite',
     'MatrixBase.is_positive_definite',
     'MatrixBase.is_positive_semidefinite',
     'MatrixBase.is_negative_definite',
     'MatrixBase.is_negative_semidefinite'): ['matplotlib'],
}


class MatrixBase(Printable):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""

    _op_priority = 10.01

    # Added just for numpy compatibility
    __array_priority__ = 11

    is_Matrix = True
    _class_priority = 3
    _sympify = staticmethod(sympify)
    zero = S.Zero
    one = S.One

    _diff_wrt: bool = True
    rows: int
    cols: int
    _simplify = None

    @classmethod
    def _new(cls, *args, **kwargs):
        """`_new` must, at minimum, be callable as
        `_new(rows, cols, mat) where mat is a flat list of the
        elements of the matrix."""
        raise NotImplementedError("Subclasses must implement this.")

    def __eq__(self, other):
        raise NotImplementedError("Subclasses must implement this.")

    def __getitem__(self, key):
        """Implementations of __getitem__ should accept ints, in which
        case the matrix is indexed as a flat list, tuples (i,j) in which
        case the (i,j) entry is returned, slices, or mixed tuples (a,b)
        where a and b are any combination of slices and integers."""
        raise NotImplementedError("Subclasses must implement this.")

    @property
    def shape(self):
        """The shape (dimensions) of the matrix as the 2-tuple (rows, cols).

        Examples
        ========

        >>> from sympy import zeros
        >>> M = zeros(2, 3)
        >>> M.shape
        (2, 3)
        >>> M.rows
        2
        >>> M.cols
        3
        """
        return (self.rows, self.cols)

    def _eval_col_del(self, col):
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        return self._new(self.rows, self.cols - 1, entry)

    def _eval_col_insert(self, pos, other):

        def entry(i, j):
            if j < pos:
                return self[i, j]
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]
            return self[i, j - other.cols]

        return self._new(self.rows, self.cols + other.cols, entry)

    def _eval_col_join(self, other):
        rows = self.rows

        def entry(i, j):
            if i < rows:
                return self[i, j]
            return other[i - rows, j]

        return classof(self, other)._new(self.rows + other.rows, self.cols,
                                         entry)

    def _eval_extract(self, rowsList, colsList):
        mat = list(self)
        cols = self.cols
        indices = (i * cols + j for i in rowsList for j in colsList)
        return self._new(len(rowsList), len(colsList),
                         [mat[i] for i in indices])

    def _eval_get_diag_blocks(self):
        sub_blocks = []

        def recurse_sub_blocks(M):
            for i in range(1, M.shape[0] + 1):
                if i == 1:
                    to_the_right = M[0, i:]
                    to_the_bottom = M[i:, 0]
                else:
                    to_the_right = M[:i, i:]
                    to_the_bottom = M[i:, :i]
                if any(to_the_right) or any(to_the_bottom):
                    continue
                sub_blocks.append(M[:i, :i])
                if M.shape != M[:i, :i].shape:
                    recurse_sub_blocks(M[i:, i:])
                return

        recurse_sub_blocks(self)
        return sub_blocks

    def _eval_row_del(self, row):
        def entry(i, j):
            return self[i, j] if i < row else self[i + 1, j]
        return self._new(self.rows - 1, self.cols, entry)

    def _eval_row_insert(self, pos, other):
        entries = list(self)
        insert_pos = pos * self.cols
        entries[insert_pos:insert_pos] = list(other)
        return self._new(self.rows + other.rows, self.cols, entries)

    def _eval_row_join(self, other):
        cols = self.cols

        def entry(i, j):
            if j < cols:
                return self[i, j]
            return other[i, j - cols]

        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         entry)

    def _eval_tolist(self):
        return [list(self[i,:]) for i in range(self.rows)]

    def _eval_todok(self):
        dok = {}
        rows, cols = self.shape
        for i in range(rows):
            for j in range(cols):
                val = self[i, j]
                if val != self.zero:
                    dok[i, j] = val
        return dok

    @classmethod
    def _eval_from_dok(cls, rows, cols, dok):
        out_flat = [cls.zero] * (rows * cols)
        for (i, j), val in dok.items():
            out_flat[i * cols + j] = val
        return cls._new(rows, cols, out_flat)

    def _eval_vec(self):
        rows = self.rows

        def entry(n, _):
            # we want to read off the columns first
            j = n // rows
            i = n - j * rows
            return self[i, j]

        return self._new(len(self), 1, entry)

    def _eval_vech(self, diagonal):
        c = self.cols
        v = []
        if diagonal:
            for j in range(c):
                for i in range(j, c):
                    v.append(self[i, j])
        else:
            for j in range(c):
                for i in range(j + 1, c):
                    v.append(self[i, j])
        return self._new(len(v), 1, v)

    def col_del(self, col):
        """Delete the specified column."""
        if col < 0:
            col += self.cols
        if not 0 <= col < self.cols:
            raise IndexError("Column {} is out of range.".format(col))
        return self._eval_col_del(col)

    def col_insert(self, pos, other):
        """Insert one or more columns at the given column position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.col_insert(1, V)
        Matrix([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        See Also
        ========

        col
        row_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return type(self)(other)

        pos = as_int(pos)

        if pos < 0:
            pos = self.cols + pos
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols

        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))

        return self._eval_col_insert(pos, other)

    def col_join(self, other):
        """Concatenates two matrices along self's last and other's first row.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.col_join(V)
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

        See Also
        ========

        col
        row_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))
        return self._eval_col_join(other)

    def col(self, j):
        """Elementary column selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).col(0)
        Matrix([
        [1],
        [0]])

        See Also
        ========

        row
        col_del
        col_join
        col_insert
        """
        return self[:, j]

    def extract(self, rowsList, colsList):
        r"""Return a submatrix by specifying a list of rows and columns.
        Negative indices can be given. All indices must be in the range
        $-n \le i < n$ where $n$ is the number of rows or columns.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(4, 3, range(12))
        >>> m
        Matrix([
        [0,  1,  2],
        [3,  4,  5],
        [6,  7,  8],
        [9, 10, 11]])
        >>> m.extract([0, 1, 3], [0, 1])
        Matrix([
        [0,  1],
        [3,  4],
        [9, 10]])

        Rows or columns can be repeated:

        >>> m.extract([0, 0, 1], [-1])
        Matrix([
        [2],
        [2],
        [5]])

        Every other row can be taken by using range to provide the indices:

        >>> m.extract(range(0, m.rows, 2), [-1])
        Matrix([
        [2],
        [8]])

        RowsList or colsList can also be a list of booleans, in which case
        the rows or columns corresponding to the True values will be selected:

        >>> m.extract([0, 1, 2, 3], [True, False, True])
        Matrix([
        [0,  2],
        [3,  5],
        [6,  8],
        [9, 11]])
        """

        if not is_sequence(rowsList) or not is_sequence(colsList):
            raise TypeError("rowsList and colsList must be iterable")
        # ensure rowsList and colsList are lists of integers
        if rowsList and all(isinstance(i, bool) for i in rowsList):
            rowsList = [index for index, item in enumerate(rowsList) if item]
        if colsList and all(isinstance(i, bool) for i in colsList):
            colsList = [index for index, item in enumerate(colsList) if item]

        # ensure everything is in range
        rowsList = [a2idx(k, self.rows) for k in rowsList]
        colsList = [a2idx(k, self.cols) for k in colsList]

        return self._eval_extract(rowsList, colsList)

    def get_diag_blocks(self):
        """Obtains the square sub-matrices on the main diagonal of a square matrix.

        Useful for inverting symbolic matrices or solving systems of
        linear equations which may be decoupled by having a block diagonal
        structure.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y, z
        >>> A = Matrix([[1, 3, 0, 0], [y, z*z, 0, 0], [0, 0, x, 0], [0, 0, 0, 0]])
        >>> a1, a2, a3 = A.get_diag_blocks()
        >>> a1
        Matrix([
        [1,    3],
        [y, z**2]])
        >>> a2
        Matrix([[x]])
        >>> a3
        Matrix([[0]])

        """
        return self._eval_get_diag_blocks()

    @classmethod
    def hstack(cls, *args):
        """Return a matrix formed by joining args horizontally (i.e.
        by repeated application of row_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.hstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.row_join, args)

    def reshape(self, rows, cols):
        """Reshape the matrix. Total number of elements must remain the same.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 3, lambda i, j: 1)
        >>> m
        Matrix([
        [1, 1, 1],
        [1, 1, 1]])
        >>> m.reshape(1, 6)
        Matrix([[1, 1, 1, 1, 1, 1]])
        >>> m.reshape(3, 2)
        Matrix([
        [1, 1],
        [1, 1],
        [1, 1]])

        """
        if self.rows * self.cols != rows * cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        dok = {divmod(i*self.cols + j, cols):
            v for (i, j), v in self.todok().items()}
        return self._eval_from_dok(rows, cols, dok)

    def row_del(self, row):
        """Delete the specified row."""
        if row < 0:
            row += self.rows
        if not 0 <= row < self.rows:
            raise IndexError("Row {} is out of range.".format(row))

        return self._eval_row_del(row)

    def row_insert(self, pos, other):
        """Insert one or more rows at the given row position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.row_insert(1, V)
        Matrix([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]])

        See Also
        ========

        row
        col_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return self._new(other)

        pos = as_int(pos)

        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows

        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))

        return self._eval_row_insert(pos, other)

    def row_join(self, other):
        """Concatenates two matrices along self's last and rhs's first column

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.row_join(V)
        Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

        See Also
        ========

        row
        col_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))
        return self._eval_row_join(other)

    def diagonal(self, k=0):
        """Returns the kth diagonal of self. The main diagonal
        corresponds to `k=0`; diagonals above and below correspond to
        `k > 0` and `k < 0`, respectively. The values of `self[i, j]`
        for which `j - i = k`, are returned in order of increasing
        `i + j`, starting with `i + j = |k|`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(3, 3, lambda i, j: j - i); m
        Matrix([
        [ 0,  1, 2],
        [-1,  0, 1],
        [-2, -1, 0]])
        >>> _.diagonal()
        Matrix([[0, 0, 0]])
        >>> m.diagonal(1)
        Matrix([[1, 1]])
        >>> m.diagonal(-2)
        Matrix([[-2]])

        Even though the diagonal is returned as a Matrix, the element
        retrieval can be done with a single index:

        >>> Matrix.diag(1, 2, 3).diagonal()[1]  # instead of [0, 1]
        2

        See Also
        ========

        diag
        """
        rv = []
        k = as_int(k)
        r = 0 if k > 0 else -k
        c = 0 if r else k
        while True:
            if r == self.rows or c == self.cols:
                break
            rv.append(self[r, c])
            r += 1
            c += 1
        if not rv:
            raise ValueError(filldedent('''
            The %s diagonal is out of range [%s, %s]''' % (
            k, 1 - self.rows, self.cols - 1)))
        return self._new(1, len(rv), rv)

    def row(self, i):
        """Elementary row selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).row(0)
        Matrix([[1, 0]])

        See Also
        ========

        col
        row_del
        row_join
        row_insert
        """
        return self[i, :]

    def todok(self):
        """Return the matrix as dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix.eye(3)
        >>> M.todok()
        {(0, 0): 1, (1, 1): 1, (2, 2): 1}
        """
        return self._eval_todok()

    @classmethod
    def from_dok(cls, rows, cols, dok):
        """Create a matrix from a dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> d = {(0, 0): 1, (1, 2): 3, (2, 1): 4}
        >>> Matrix.from_dok(3, 3, d)
        Matrix([
        [1, 0, 0],
        [0, 0, 3],
        [0, 4, 0]])
        """
        dok = {ij: cls._sympify(val) for ij, val in dok.items()}
        return cls._eval_from_dok(rows, cols, dok)

    def tolist(self):
        """Return the Matrix as a nested Python list.

        Examples
        ========

        >>> from sympy import Matrix, ones
        >>> m = Matrix(3, 3, range(9))
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
        >>> m.tolist()
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> ones(3, 0).tolist()
        [[], [], []]

        When there are no rows then it will not be possible to tell how
        many columns were in the original matrix:

        >>> ones(0, 3).tolist()
        []

        """
        if not self.rows:
            return []
        if not self.cols:
            return [[] for i in range(self.rows)]
        return self._eval_tolist()

    def todod(M):
        """Returns matrix as dict of dicts containing non-zero elements of the Matrix

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1],[0, 3]])
        >>> A
        Matrix([
        [0, 1],
        [0, 3]])
        >>> A.todod()
        {0: {1: 1}, 1: {1: 3}}


        """
        rowsdict = {}
        Mlol = M.tolist()
        for i, Mi in enumerate(Mlol):
            row = {j: Mij for j, Mij in enumerate(Mi) if Mij}
            if row:
                rowsdict[i] = row
        return rowsdict

    def vec(self):
        """Return the Matrix converted into a one column matrix by stacking columns

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 3], [2, 4]])
        >>> m
        Matrix([
        [1, 3],
        [2, 4]])
        >>> m.vec()
        Matrix([
        [1],
        [2],
        [3],
        [4]])

        See Also
        ========

        vech
        """
        return self._eval_vec()

    def vech(self, diagonal=True, check_symmetry=True):
        """Reshapes the matrix into a column vector by stacking the
        elements in the lower triangle.

        Parameters
        ==========

        diagonal : bool, optional
            If ``True``, it includes the diagonal elements.

        check_symmetry : bool, optional
            If ``True``, it checks whether the matrix is symmetric.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 2], [2, 3]])
        >>> m
        Matrix([
        [1, 2],
        [2, 3]])
        >>> m.vech()
        Matrix([
        [1],
        [2],
        [3]])
        >>> m.vech(diagonal=False)
        Matrix([[2]])

        Notes
        =====

        This should work for symmetric matrices and ``vech`` can
        represent symmetric matrices in vector form with less size than
        ``vec``.

        See Also
        ========

        vec
        """
        if not self.is_square:
            raise NonSquareMatrixError

        if check_symmetry and not self.is_symmetric():
            raise ValueError("The matrix is not symmetric.")

        return self._eval_vech(diagonal)

    @classmethod
    def vstack(cls, *args):
        """Return a matrix formed by joining args vertically (i.e.
        by repeated application of col_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.vstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.col_join, args)

    @classmethod
    def _eval_diag(cls, rows, cols, diag_dict):
        """diag_dict is a defaultdict containing
        all the entries of the diagonal matrix."""
        def entry(i, j):
            return diag_dict[(i, j)]
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_eye(cls, rows, cols):
        vals = [cls.zero]*(rows*cols)
        vals[::cols+1] = [cls.one]*min(rows, cols)
        return cls._new(rows, cols, vals, copy=False)

    @classmethod
    def _eval_jordan_block(cls, size: int, eigenvalue, band='upper'):
        if band == 'lower':
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif j + 1 == i:
                    return cls.one
                return cls.zero
        else:
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif i + 1 == j:
                    return cls.one
                return cls.zero
        return cls._new(size, size, entry)

    @classmethod
    def _eval_ones(cls, rows, cols):
        def entry(i, j):
            return cls.one
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        return cls._new(rows, cols, [cls.zero]*(rows*cols), copy=False)

    @classmethod
    def _eval_wilkinson(cls, n):
        def entry(i, j):
            return cls.one if i + 1 == j else cls.zero

        D = cls._new(2*n + 1, 2*n + 1, entry)

        wminus = cls.diag(list(range(-n, n + 1)), unpack=True) + D + D.T
        wplus = abs(cls.diag(list(range(-n, n + 1)), unpack=True)) + D + D.T

        return wminus, wplus

    @classmethod
    def diag(kls, *args, strict=False, unpack=True, rows=None, cols=None, **kwargs):
        """Returns a matrix with the specified diagonal.
        If matrices are passed, a block-diagonal matrix
        is created (i.e. the "direct sum" of the matrices).

        kwargs
        ======

        rows : rows of the resulting matrix; computed if
               not given.

        cols : columns of the resulting matrix; computed if
               not given.

        cls : class for the resulting matrix

        unpack : bool which, when True (default), unpacks a single
        sequence rather than interpreting it as a Matrix.

        strict : bool which, when False (default), allows Matrices to
        have variable-length rows.

        Examples
        ========

        >>> from sympy import Matrix
        >>> Matrix.diag(1, 2, 3)
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])

        The current default is to unpack a single sequence. If this is
        not desired, set `unpack=False` and it will be interpreted as
        a matrix.

        >>> Matrix.diag([1, 2, 3]) == Matrix.diag(1, 2, 3)
        True

        When more than one element is passed, each is interpreted as
        something to put on the diagonal. Lists are converted to
        matrices. Filling of the diagonal always continues from
