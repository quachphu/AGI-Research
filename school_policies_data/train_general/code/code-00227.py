            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.false

    def __le__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.false

    @_sympifyit("other", NotImplemented)
    def __mod__(self, other: sympy.Expr) -> sympy.Expr:
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self) -> "IntInfinity":
        return self

    def ceiling(self) -> "IntInfinity":
        return self


int_oo = S.IntInfinity


def is_infinite(expr: sympy.Basic) -> bool:
    """Check if an expression is any type of infinity (positive or negative).

    This handles both sympy's built-in infinities (oo, -oo) and PyTorch's
    integer infinities (int_oo, -int_oo).

    Note: We cannot rely on sympy's is_finite property because IntInfinity
    and NegativeIntInfinity have is_integer=True, which implies is_finite=True
    in sympy's assumption system.
    """
    return expr in (
        S.Infinity,
        S.NegativeInfinity,
        S.IntInfinity,
        S.NegativeIntInfinity,
    )


# pyrefly: ignore [invalid-inheritance]
class NegativeIntInfinity(Number, metaclass=Singleton):
    """Negative integer infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    IntInfinity
    """

    # Ensure we get dispatched to before plain numbers
    _op_priority = 100.0

    is_integer = True
    is_extended_real = True
    is_commutative = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__: tuple[str, ...] = ()

    def __new__(cls) -> "NegativeIntInfinity":
        return AtomicExpr.__new__(cls)

    def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> sympy.Basic | None:
        if self == old:
            return new
        return None

    def _sympystr(self, printer: Any) -> str:
        return "-int_oo"

    """
    def _eval_evalf(self, prec=None):
        return Float('-inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    @_sympifyit("other", NotImplemented)
    def __add__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    __radd__ = __add__

    @_sympifyit("other", NotImplemented)
    def __sub__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other: sympy.Expr) -> sympy.Expr:
        return (-self).__add__(other)

    @_sympifyit("other", NotImplemented)
    def __mul__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.IntInfinity
        return Number.__mul__(self, other)

    __rmul__ = __mul__

    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity  # truediv returns float
        return Number.__truediv__(self, other)

    def __abs__(self) -> sympy.Expr:
        return S.IntInfinity

    def __neg__(self) -> sympy.Expr:
        return S.IntInfinity

    def _eval_power(self, expt: sympy.Expr) -> sympy.Expr | None:
        if expt.is_number:
            if expt in (
                S.NaN,
                S.Infinity,
                S.NegativeInfinity,
                S.IntInfinity,
                S.NegativeIntInfinity,
            ):
                return S.NaN

            if isinstance(expt, sympy.Integer) and expt.is_extended_positive:
                if expt.is_odd:
                    return S.NegativeIntInfinity
                else:
                    return S.IntInfinity

            inf_part = S.IntInfinity**expt
            s_part = S.NegativeOne**expt
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            if (
                inf_part is S.ComplexInfinity
                and s_part.is_finite
                and not s_part.is_zero
            ):
                return S.ComplexInfinity
            return s_part * inf_part
        return None

    def _as_mpf_val(self, prec: int) -> Any:
        return mlib.fninf

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        return other is S.NegativeIntInfinity

    def __ne__(self, other: object) -> bool:
        return other is not S.NegativeIntInfinity

    def __gt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.false

    def __ge__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.false

    def __lt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.true

    def __le__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.true

    @_sympifyit("other", NotImplemented)
    def __mod__(self, other: sympy.Expr) -> sympy.Expr:
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self) -> "NegativeIntInfinity":
        return self

    def ceiling(self) -> "NegativeIntInfinity":
        return self

    def as_powers_dict(self) -> dict[sympy.Expr, int]:
        return {S.NegativeOne: 1, S.IntInfinity: 1}
<|endoftext|>import functools
import math
import operator
import sys
from collections.abc import Callable
from typing import SupportsFloat, TYPE_CHECKING, TypeVar
from typing_extensions import TypeVarTuple, Unpack

import sympy
from sympy import S
from sympy.core import sympify
from sympy.core.expr import Expr
from sympy.core.function import Application
from sympy.core.logic import _torf, fuzzy_and, fuzzy_or
from sympy.core.numbers import equal_valued
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.sorting import ordered
from sympy.core.traversal import walk
from sympy.printing.precedence import PRECEDENCE
from sympy.utilities.iterables import sift

from torch.torch_version import TorchVersion

from .numbers import int_oo, IntInfinity, is_infinite


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


_T = TypeVar("_T", bound=SupportsFloat)
_Ts = TypeVarTuple("_Ts")

# sympy.gcd on wide Add expressions triggers expensive polynomial-GCD
# (dmp_sqr / dmp_rr_div) that scales poorly in the number of symbols.
# safe_gcd() falls back to simple_floordiv_gcd past this threshold.
_MAX_ADD_TERMS_FOR_POLY_GCD = 20


def _is_wide_add(expr: sympy.Basic) -> bool:
    return isinstance(expr, sympy.Add) and len(expr.args) > _MAX_ADD_TERMS_FOR_POLY_GCD


def safe_gcd(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    """Like sympy.gcd but avoids polynomial-GCD on wide Add expressions,
    falling back to simple_floordiv_gcd instead."""
    if _is_wide_add(a) or _is_wide_add(b):
        return simple_floordiv_gcd(a, b)
    return sympy.gcd(a, b)


# Portions of this file are adapted from the Sympy codebase, which was
# licensed as follows:
#
#   Copyright (c) 2006-2023 SymPy Development Team
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#     a. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#     b. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     c. Neither the name of SymPy nor the names of its contributors
#        may be used to endorse or promote products derived from this software
#        without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#   DAMAGE.

__all__ = [
    "FloorDiv",
    "ModularIndexing",
    "Where",
    "PythonMod",
    "Mod",
    "CleanDiv",
    "CeilToInt",
    "FloorToInt",
    "CeilDiv",
    "IntTrueDiv",
    "FloatTrueDiv",
    "LShift",
    "RShift",
    "IsNonOverlappingAndDenseIndicator",
    "TruncToFloat",
    "TruncToInt",
    "RoundToInt",
    "RoundDecimal",
    "ToFloat",
    "FloatPow",
    "PowByNatural",
    "Identity",
]


def _is_symbols_binary_summation(expr: sympy.Expr) -> bool:
    # No need to check that two args are not the same, since expr is pre-optimized but we do it anyway.
    return (
        isinstance(expr, sympy.Expr)
        and expr.is_Add
        and len(expr._args) == 2
        and expr._args[0].is_symbol
        and expr._args[1].is_symbol
        and expr._args[0] is not expr._args[1]
    )


def _keep_float(
    f: Callable[[Unpack[_Ts]], _T],
) -> Callable[[Unpack[_Ts]], _T | sympy.Float]:
    @functools.wraps(f)
    def inner(*args: Unpack[_Ts]) -> _T | sympy.Float:
        r: _T | sympy.Float = f(*args)
        if any(isinstance(a, sympy.Float) for a in args) and not isinstance(
            r, sympy.Float
        ):
            r = sympy.Float(float(r))
        return r

    # pyrefly: ignore [bad-return]
    return inner


def fuzzy_eq(x: bool | None, y: bool | None) -> bool | None:
    if None in (x, y):
        return None
    return x == y


def simple_floordiv_gcd(p: sympy.Basic, q: sympy.Basic) -> sympy.Basic:
    """
    Fast path for sympy.gcd, using a simple factoring strategy.

    We try to rewrite p and q in the form n*e*p1 + n*e*p2 and n*e*q0,
    where n is the greatest common integer factor and e is the largest
    syntactic common factor (i.e., common sub-expression) in p and q.
    Then the gcd returned is n*e, cancelling which we would be left with
    p1 + p2 and q0.

    Note that further factoring of p1 + p2 and q0 might be possible with
    sympy.factor (which uses domain-specific theories). E.g., we are unable
    to find that x*y + x + y + 1 is divisible by x + 1. More generally,
    when q is of the form q1 + q2 (instead of being already factored) it
    might be necessary to fall back on sympy.gcd.
    """

    def integer_coefficient(x: sympy.Basic) -> int:
        integer_coefficients: list[int] = [
            abs(int(arg))
            for arg in sympy.Mul.make_args(x)
            if isinstance(arg, (int, sympy.Integer))
        ]
        return math.prod(integer_coefficients)

    def integer_factor(expr: sympy.Basic) -> int:
        integer_factors: Iterable[int] = map(
            integer_coefficient, sympy.Add.make_args(expr)
        )
        return functools.reduce(math.gcd, integer_factors)

    gcd: int = math.gcd(integer_factor(p), integer_factor(q))
    p, q = p / gcd, q / gcd  # type: ignore[operator, assignment]  # remove in py3.12

    base_splits: list[tuple[sympy.Basic, ...]] = list(
        map(sympy.Mul.make_args, sympy.Add.make_args(p))
    )
    divisor_split: tuple[sympy.Basic, ...] = sympy.Mul.make_args(q)
    for x in divisor_split:
        if all(x in base_split for base_split in base_splits):
            gcd = gcd * x  # type: ignore[operator]  # remove in py3.12
    return gcd  # type: ignore[return-value]  # remove in py3.12


# It would be nice to have assertions on whether or not inputs is_integer
# However, with bugs like https://github.com/sympy/sympy/issues/26620 sympy
# sometimes inconsistently reports floats as integers.
#
# What we can assume from sympy is that if something is an int, it
# definitely is is_integer, but if it is a float it may or may not
# be is_integer.  So we are unable to do strong asserts that things
# are NOT integers.


# TODO: In Triton, // rounds to zero, but in Python, it is floor division.
# When we can prove both arguments are non-negative, we should just have a
# GenericFloorDiv (name pending) which can codegen efficiently in Python/C,
# and then PythonFloorDiv and CIntDiv which have the appropriate rounding
# semantics.
#
# Right now, FloorDiv de facto changes behavior if arguments are negative or
# not, this can potentially cause correctness issues.
class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)

    NB: This is Python-style floor division, round to -Inf
    """

    nargs: tuple[int, ...] = (2,)
    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @property
    def base(self) -> sympy.Basic:
        return self.args[0]

    @property
    def divisor(self) -> sympy.Basic:
        return self.args[1]

    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        base = printer.parenthesize(self.base, PRECEDENCE["Atom"] - 0.5)
        divisor = printer.parenthesize(self.divisor, PRECEDENCE["Atom"] - 0.5)
        return f"({base}//{divisor})"

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base: sympy.Integer, divisor: sympy.Integer) -> sympy.Basic | None:
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # Assert triggered by inequality solver
        # assert base.is_integer, base
        # assert divisor.is_integer, divisor

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        if is_infinite(base) and is_infinite(divisor):
            return sympy.nan
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and equal_valued(divisor, 1):
            return base
        if base.is_integer and equal_valued(divisor, -1):
            return sympy.Mul(base, -1)
        if base == divisor:
            return sympy.S.One

        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (is_infinite(base) or is_infinite(divisor))
        ):
            r = float(base) / float(divisor)
            if r == math.inf:
                return int_oo
            elif r == -math.inf:
                return -int_oo
            elif math.isnan(r):
                return sympy.nan
            else:
                return sympy.Integer(math.floor(r))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Integer(int(base) // int(divisor))
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        # Expands (x + y) // b into x // b + y // b.
        # This only works if floor is an identity, i.e. x / b is an integer.
        if isinstance(divisor, sympy.Integer):
            quotients = 0
            terms: list[sympy.Expr] = []
            for term in sympy.Add.make_args(base):
                quotient = term / divisor

                # This is a sympy bug fixed in https://github.com/sympy/sympy/pull/28442
                # sympy can generate a quotient with (1/22)*.... such that quotient.is_integer is True
                # FloorDiv should not allow that as output. see
                quotient_is_integer = None
                if isinstance(quotient, sympy.Mul) and TorchVersion(
                    sympy.__version__
                ) < TorchVersion("1.15.0"):
                    rationals = quotient.atoms(sympy.Rational)
                    all_rationals_ints = all(r.q == 1 for r in rationals)
                    quotient_is_integer = quotient.is_integer and all_rationals_ints
                else:
                    quotient_is_integer = quotient.is_integer

                if quotient_is_integer:
                    terms.append(term)
                    quotients += quotient

            if len(terms) != 0:
                # Passing evaluate = False since expression will be optimized during the subtraction post its construction.
                return (
                    FloorDiv(base - sympy.Add(*terms, evaluate=False), divisor)
                    + quotients
                )

        try:
            gcd = simple_floordiv_gcd(base, divisor)
            if equal_valued(gcd, 1) and isinstance(divisor, sympy.Add):
                gcd = safe_gcd(base, divisor)
            if not equal_valued(gcd, 1):
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276

        return None

    def _eval_is_nonnegative(self) -> bool | None:
        p, q = self.args[:2]
        if all([p.is_integer, q.is_integer, p.is_nonnegative, q.is_nonnegative]):
            return True
        return None


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c where % is the C modulus
    """

    nargs: tuple[int, ...] = (3,)
    is_integer: bool = True
    precedence: int = 35  # lower precedence than add

    @classmethod
    def eval(
        cls, base: sympy.Integer, divisor: sympy.Integer, modulus: sympy.Integer
    ) -> sympy.Basic | None:
        if base == 0 or modulus == 1:
            return sympy.S.Zero
        if (
            isinstance(base, sympy.Integer)
            and isinstance(divisor, sympy.Integer)
            and isinstance(modulus, sympy.Integer)
        ):
            return (base // divisor) % modulus

        try:
            if divisor != 1:
                gcd = safe_gcd(base, divisor)
                if gcd != 1:
                    return ModularIndexing(
                        sympy.simplify(base / gcd),
                        sympy.simplify(divisor / gcd),
                        modulus,
                    )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276

        # Guard on width: the per-term gcd calls are cheap (via safe_gcd),
        # but the result feeds into downstream sympy.expand() which blows up
        # on wide Add expressions (e.g. in sizevars.simplify).
        if isinstance(base, sympy.Add) and not _is_wide_add(base):
            new_terms: list[sympy.Integer] = []
            all_nonnegative: bool = True
            for term in base.args:
                if safe_gcd(term, modulus * divisor) != modulus * divisor:
                    if term.is_nonnegative is not True:
                        # workaround for https://github.com/triton-lang/triton/issues/619,
                        # if terms are not provably nonnegative, // can produce
                        # the wrong result
                        # TODO if https://github.com/triton-lang/triton/issues/619 is fixed
                        # this optimization would become valid
                        all_nonnegative = False
                        break
                    else:
                        new_terms.append(term)

            if len(new_terms) != len(base.args) and all_nonnegative:
                return ModularIndexing(sum(new_terms), divisor, modulus)

        if isinstance(base, FloorDiv):
            return ModularIndexing(base.args[0], base.args[1] * divisor, modulus)

        return None

    def _eval_is_nonnegative(self) -> bool | None:
        p, q = self.args[:2]
        return fuzzy_eq(p.is_nonnegative, q.is_nonnegative)  # type: ignore[attr-defined]


class Where(sympy.Function):
    """
    Good ol' ternary operator
    """

    nargs: tuple[int, ...] = (3,)
    precedence: int = 35  # lower precedence than add

    def _eval_is_integer(self) -> bool | None:
        return True if self.args[1].is_integer and self.args[2].is_integer else None  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self) -> bool | None:
        return (
            True
            if self.args[1].is_nonnegative and self.args[2].is_nonnegative  # type: ignore[attr-defined]
            else None
        )

    def _eval_is_positive(self) -> bool | None:
        return True if self.args[1].is_positive and self.args[2].is_positive else None  # type: ignore[attr-defined]

    @classmethod
    def eval(cls, c: sympy.Basic, p: sympy.Basic, q: sympy.Basic) -> sympy.Basic | None:
        if c == sympy.true:
            return p
        elif c == sympy.false:
            return q
        return None


# Python-style modulus: take sign from RHS
class PythonMod(sympy.Function):
    nargs: tuple[int, ...] = (2,)

    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @classmethod
    def eval(cls, p: sympy.Expr, q: sympy.Expr) -> sympy.Expr | None:
        # python test/dynamo/test_export.py -k ExportTests.test_trivial_constraint
        # Triggered by sympy.solvers.inequalities.reduce_inequalities
        # assert p.is_integer, p
        # assert q.is_integer, q

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")

        # Three cases:
        #   1. p == 0
        #   2. p is either q or -q
        #   3. p is integer and q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero

        # Evaluate if they are both literals.
        if q.is_Number and p.is_Number:
            return p % q

        # If q == 2, it's a matter of whether p is odd or even.
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One

        # If p is a multiple of q.
        r = p / q
        if r.is_integer:
            return S.Zero

        # If p < q and its ratio is positive, then:
        #   - floor(p / q) = 0
        #   - p % q = p - floor(p / q) * q = p
        less = p < q
        # pyrefly: ignore [missing-attribute]
        if less.is_Boolean and bool(less) and r.is_positive:
            return p

        if sympy.Mod(p, q) == 0:
            return S.Zero

        return None

    # NB: args[1] for PythonMod
    def _eval_is_nonnegative(self) -> bool | None:
        return True if self.args[1].is_positive else None  # type: ignore[attr-defined]

    def _eval_is_nonpositive(self) -> bool | None:
        return True if self.args[1].is_negative else None  # type: ignore[attr-defined]

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        p = printer.parenthesize(self.args[0], PRECEDENCE["Atom"] - 0.5)
        q = printer.parenthesize(self.args[1], PRECEDENCE["Atom"] - 0.5)
        abs_q = str(q) if self.args[1].is_positive else f"abs({q})"
        return f"({p} % {q}) < 0 ? {p} % {q} + {abs_q} : {p} % {q}"


# Generic modulus: only defined on non-negative arguments
class Mod(sympy.Function):
    nargs = (2,)
    precedence: int = 35  # lower precedence than add

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, p: sympy.Expr, q: sympy.Expr) -> sympy.Expr | None:
        # This was adapted from: sympy/core/mod.py

        # Triggered by
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # assert p.is_integer, p
        # assert q.is_integer, q

        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")

        # Three cases:
        #   1. p == 0
        #   2. p is either q or -q
        #   3. p is integer and q == 1
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero

        # Evaluate if they are both literals.
        if q.is_Number and p.is_Number:
            if p < 0:
                raise AssertionError(p)
            if q < 1:
                raise AssertionError(q)
            return p % q

        # If q == 2, it's a matter of whether p is odd or even.
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One

        # If p is a multiple of q.
        r = p / q
        if r.is_integer:
            return S.Zero

        # If p < q and its ratio is positive, then:
        #   - floor(p / q) = 0
        #   - p % q = p - floor(p / q) * q = p
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p


class CleanDiv(FloorDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """


# Don't use sympy ceiling/floor as they will attempt simplifications involving
# frac
class CeilToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.ceil(float(number)))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(int64_t)(ceil({number}))"


class FloorToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Integer):
            return number
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.floor(float(number)))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(int64_t)(floor({number}))"


class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """

    is_integer = True

    def __new__(cls, base: sympy.Expr, divisor: sympy.Expr) -> sympy.Basic:
        base = sympy.sympify(base)
        divisor = sympy.sympify(divisor)
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        else:
            return FloorDiv(base + (divisor - 1), divisor)


class LShift(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base: sympy.Expr, shift: sympy.Expr) -> sympy.Basic:
        if shift.is_negative:
            raise ValueError("negative shift count")
        return base * PowByNatural(sympy.Integer(2), shift)


class RShift(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, base: sympy.Expr, shift: sympy.Expr) -> sympy.Basic:
        if shift.is_negative:
            raise ValueError("negative shift count")
        return FloorDiv(base, PowByNatural(sympy.Integer(2), shift))


class MinMaxBase(Expr, LatticeOp):  # type: ignore[misc]
    def __new__(cls, *original_args: sympy.Expr, **assumptions: bool) -> sympy.Basic:
        from sympy.core.parameters import global_parameters

        evaluate = assumptions.pop("evaluate", global_parameters.evaluate)
        args = (sympify(arg) for arg in original_args)

        # See the comment in _satisfy_unique_summations_symbols.
        unique_summations_symbols = (
            None
            if not evaluate
            else cls._satisfy_unique_summations_symbols(original_args)
        )

        if evaluate:
            try:
                # first standard filter, for cls.zero and cls.identity
                # also reshape Max(a, Max(b, c)) to Max(a, b, c)
                args = frozenset(cls._new_args_filter(args))  # type: ignore[assignment]
            except ShortCircuit:
                return cls.zero  # type: ignore[attr-defined]

            # No need to run _collapse_arguments and _find_localzeros, see the comment
            # in _satisfy_unique_summations_symbols.
            if unique_summations_symbols is None:
                # remove redundant args that are easily identified
                args = cls._collapse_arguments(args, **assumptions)

                # find local zeros
                args = cls._find_localzeros(args, **assumptions)

        args = frozenset(args)

        if not args:
            return cls.identity  # type: ignore[attr-defined]

        if len(args) == 1:
            return list(args).pop()

        # base creation
        obj = Expr.__new__(cls, *ordered(args), **assumptions)
        obj._argset = args

        obj.unique_summations_symbols = unique_summations_symbols
        return obj

    @classmethod
    def _collapse_known_multiplicative_terms(
        cls, values: set[sympy.Expr]
    ) -> sympy.Expr | None:
        if len(values) != 2:
            return None

        a, b = values
        a_coeff, a_term = a.as_coeff_Mul()
        b_coeff, b_term = b.as_coeff_Mul()
        if a_term != b_term:
            return None

        if not (a_coeff.is_comparable and b_coeff.is_comparable):
            return None

        if a_coeff == b_coeff:
            return a

        if a_term.is_nonnegative:
            a_is_smaller = a_coeff < b_coeff
        elif a_term.is_nonpositive:
            a_is_smaller = a_coeff > b_coeff
        else:
            return None

        if cls is Min:
            return a if a_is_smaller else b
        if cls is Max:
            return b if a_is_smaller else a

        raise AssertionError(f"impossible {cls}")

    @classmethod
    def _satisfy_unique_summations_symbols(
        cls, args: "Sequence[sympy.Expr]"
    ) -> set[sympy.core.symbol.Symbol] | None:
        """
        One common case in some models is building expressions of the form
        max(max(max(a+b...), c+d), e+f) which is simplified to max(a+b, c+d, e+f, ...).
        For such expressions, we call the Max constructor X times (once for each nested
        max) and the expression gets flattened.

        An expensive cost in constructing those expressions is running _collapse_arguments
        and _find_localzeros. However, those two optimizations are unnecessary when the args
        to max are all of the form a+b, c+d, ..etc where each term uses a unique set of symbols.

        This function is used to detect such properties of the expressions we are building
        and if so inform that we do not need to run those optimizations. To detect those,
        we store a property in the expression that tells that this expression is a min/max
        operation over terms that use unique symbols "unique_summations_symbols". This property
        also memoize the set of symbols used in all the terms to make it faster to detect this
        property inductively.

        When we apply max to add a new term, all we need to do is check if the new term uses
        unique symbols (with respect to existing terms and itself).
        Example:
        t = Max(a+b, c+d) ==> satisfies the property
        Max(t, h+j)       ==> h,j not in [a,b,c,d] => satisfy the property.

        The function returns None if the new expression does not satisfy the unique_summations_symbols
        property. Otherwise, it returns a new set of unique symbols.
        """
        if len(args) != 2:
            return None

        (lhs, rhs) = (
            (args[1], args[0])
            if isinstance(args[1], MinMaxBase)
            else (args[0], args[1])
        )

        if not _is_symbols_binary_summation(rhs):
            return None

        # base case max(a+b, c+d) ==> satisfies the property if a+b and c+d use unique symbols.
        if _is_symbols_binary_summation(lhs):
            return cls._unique_symbols(args)

        # inductive case max(t, h+j) ==> satisfies the property if h, j not in t.unique_summations_symbols
        if isinstance(lhs, MinMaxBase):
            lhs_unique_summations_symbols = getattr(
                lhs, "unique_summations_symbols", None
            )
            if lhs_unique_summations_symbols is not None:
                return cls._unique_symbols([rhs], lhs_unique_summations_symbols)

        return None

    @classmethod
    def _unique_symbols(
        cls,
        args: "Iterable[sympy.Expr]",
        initial_set: set[sympy.core.symbol.Symbol] | None = None,
    ) -> set[sympy.core.symbol.Symbol] | None:
        """
        Return seen_symbols if all atoms in all args are all unique symbols,
        else returns None. initial_set can be used to represent initial value for seen_symbols
        """
        seen_symbols = set() if initial_set is None else initial_set.copy()
        for arg in args:
            for element in arg.atoms():
                if not isinstance(element, sympy.core.symbol.Symbol):
                    return None
                elif element in seen_symbols:
                    return None
                else:
                    seen_symbols.add(element)
        return seen_symbols

    @classmethod
    def _collapse_arguments(
        cls, args: "Iterable[sympy.Expr]", **assumptions: bool
    ) -> "Iterable[sympy.Expr]":
        """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
        if not args:
            return args
        args = list(ordered(args))
        if cls is Min:
            other = Max
        else:
            other = Min  # type: ignore[assignment]

        # find global comparable max of Max and min of Min if a new
        # value is being introduced in these args at position 0 of
        # the ordered args
        if args[0].is_number:
            sifted = mins, maxs = [], []  # type: ignore[var-annotated]
            for i in args:
                for v in walk(i, Min, Max):
                    if v.args[0].is_comparable:
                        sifted[isinstance(v, Max)].append(v)
            small = Min.identity
            for i in mins:
                v = i.args[0]
                if v.is_number and (v < small) == True:  # noqa: E712
                    small = v
            big = Max.identity
            for i in maxs:
                v = i.args[0]
                if v.is_number and (v > big) == True:  # noqa: E712
                    big = v
            # at the point when this function is called from __new__,
            # there may be more than one numeric arg present since
            # local zeros have not been handled yet, so look through
            # more than the first arg
            if cls is Min:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg < small) == True:  # noqa: E712
                        small = arg
            elif cls == Max:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg > big) == True:  # noqa: E712
                        big = arg
            T = None
            if cls is Min:
                if small != Min.identity:
                    other = Max
                    T = small
            elif big != Max.identity:
                other = Min  # type: ignore[assignment]
                T = big
            if T is not None:
                # remove numerical redundancy
                for i in range(len(args)):
                    a = args[i]
                    if isinstance(a, other):
                        a0 = a.args[0]
                        if (  # noqa: E712
                            (a0 > T) if other == Max else (a0 < T)
                        ) == True:
                            args[i] = cls.identity  # type: ignore[attr-defined]

        # remove redundant symbolic args
        def do(ai: sympy.Expr, a: sympy.Expr) -> sympy.Expr:
            if not isinstance(ai, (Min, Max)):
                return ai
            cond = a in ai.args
            if not cond:
                return ai.func(*[do(i, a) for i in ai.args], evaluate=False)
            if isinstance(ai, cls):
                return ai.func(*[do(i, a) for i in ai.args if i != a], evaluate=False)
            return a

        for i, a in enumerate(args):
            args[i + 1 :] = [do(ai, a) for ai in args[i + 1 :]]

        # factor out common elements as for
        # Min(Max(x, y), Max(x, z)) -> Max(x, Min(y, z))
        # and vice versa when swapping Min/Max -- do this only for the
        # easy case where all functions contain something in common;
        # trying to find some optimal subset of args to modify takes
        # too long

        def factor_minmax(args: list[sympy.Expr]) -> list[sympy.Expr]:
            is_other = lambda arg: isinstance(arg, other)  # noqa: E731
            other_args, remaining_args = sift(args, is_other, binary=True)
            if not other_args:
                return args

            # Min(Max(x, y, z), Max(x, y, u, v)) -> {x,y}, ({z}, {u,v})
            arg_sets = [set(arg.args) for arg in other_args]
            common = set.intersection(*arg_sets)
            if not common:
                return args

            new_other_args = list(common)
            arg_sets_diff = [arg_set - common for arg_set in arg_sets]

            # If any set is empty after removing common then all can be
            # discarded e.g. Min(Max(a, b, c), Max(a, b)) -> Max(a, b)
            if all(arg_sets_diff):
                other_args_diff = [other(*s, evaluate=False) for s in arg_sets_diff]
                new_other_args.append(cls(*other_args_diff, evaluate=False))

            other_args_factored = other(*new_other_args, evaluate=False)
            return remaining_args + [other_args_factored]

        if len(args) > 1:
            args = factor_minmax(args)

        return args

    @classmethod
    def _new_args_filter(
        cls, arg_sequence: "Iterable[sympy.Expr]"
    ) -> "Iterator[sympy.Expr]":
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
        for arg in arg_sequence:
            # pre-filter, checking comparability of arguments
            if (
                not isinstance(arg, Expr)
                or arg.is_extended_real is False
                or (arg.is_number and not arg.is_comparable)
            ):
                raise ValueError(f"The argument '{arg}' is not comparable.")

            if arg == cls.zero:  # type: ignore[attr-defined]
                raise ShortCircuit(arg)
            elif arg == cls.identity:  # type: ignore[attr-defined]
                continue
            elif arg.func == cls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def _find_localzeros(
        cls, values: "Iterable[sympy.Expr]", **options: bool
    ) -> set[sympy.Expr]:
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.

        Unlike the sympy implementation, we only look for zero and one, we don't
        do generic is connected test pairwise which is slow
        """

        # First, collapse all numeric arguments
        other_values = set()
        num_value = None
        for arg in values:
            if arg.is_Number:
                if num_value is None:
                    num_value = arg
                else:
                    if cls is Max:
                        num_value = max(num_value, arg)
                    elif cls is Min:
                        num_value = min(num_value, arg)
                    else:
                        raise AssertionError(f"impossible {cls}")
            else:
                other_values.add(arg)

        # Special cases when there is only one symbolic value
        if num_value is None:
            collapsed_value = cls._collapse_known_multiplicative_terms(other_values)
            if collapsed_value is not None:
                return {collapsed_value}
            return other_values

        if len(other_values) == 0:
            return {num_value}

        if len(other_values) == 1:
            other_value = next(iter(other_values))
            if num_value in (0.0, 0) and other_value.is_nonnegative:
                return other_values if cls is Max else {num_value}
            if num_value == 1 and other_value.is_positive:
                return other_values if cls is Max else {num_value}

        other_values.add(num_value)
        return other_values

    _eval_is_algebraic = lambda s: _torf(i.is_algebraic for i in s.args)  # noqa: E731
    _eval_is_antihermitian = lambda s: _torf(  # noqa: E731
        i.is_antihermitian for i in s.args
    )
    _eval_is_commutative = lambda s: _torf(  # noqa: E731
        i.is_commutative for i in s.args
    )
    _eval_is_complex = lambda s: _torf(i.is_complex for i in s.args)  # noqa: E731
    _eval_is_composite = lambda s: _torf(i.is_composite for i in s.args)  # noqa: E731
    _eval_is_even = lambda s: _torf(i.is_even for i in s.args)  # noqa: E731
    _eval_is_finite = lambda s: _torf(i.is_finite for i in s.args)  # noqa: E731
    _eval_is_hermitian = lambda s: _torf(i.is_hermitian for i in s.args)  # noqa: E731
    _eval_is_imaginary = lambda s: _torf(i.is_imaginary for i in s.args)  # noqa: E731
    _eval_is_infinite = lambda s: _torf(i.is_infinite for i in s.args)  # noqa: E731
    _eval_is_integer = lambda s: _torf(i.is_integer for i in s.args)  # noqa: E731
    _eval_is_irrational = lambda s: _torf(i.is_irrational for i in s.args)  # noqa: E731
    _eval_is_negative = lambda s: _torf(i.is_negative for i in s.args)  # noqa: E731
    _eval_is_noninteger = lambda s: _torf(i.is_noninteger for i in s.args)  # noqa: E731
    _eval_is_nonnegative = lambda s: _torf(  # noqa: E731
        i.is_nonnegative for i in s.args
    )
    _eval_is_nonpositive = lambda s: _torf(  # noqa: E731
        i.is_nonpositive for i in s.args
    )
    _eval_is_nonzero = lambda s: _torf(i.is_nonzero for i in s.args)  # noqa: E731
    _eval_is_odd = lambda s: _torf(i.is_odd for i in s.args)  # noqa: E731
    _eval_is_polar = lambda s: _torf(i.is_polar for i in s.args)  # noqa: E731
    _eval_is_positive = lambda s: _torf(i.is_positive for i in s.args)  # noqa: E731
    _eval_is_prime = lambda s: _torf(i.is_prime for i in s.args)  # noqa: E731
    _eval_is_rational = lambda s: _torf(i.is_rational for i in s.args)  # noqa: E731
    _eval_is_real = lambda s: _torf(i.is_real for i in s.args)  # noqa: E731
    _eval_is_extended_real = lambda s: _torf(  # noqa: E731
        i.is_extended_real for i in s.args
    )
    _eval_is_transcendental = lambda s: _torf(  # noqa: E731
        i.is_transcendental for i in s.args
    )
    _eval_is_zero = lambda s: _torf(i.is_zero for i in s.args)  # noqa: E731


class Max(MinMaxBase, Application):  # type: ignore[misc]
    r"""
    Return, if possible, the maximum value of the list.
    """

    zero = S.Infinity
    identity = S.NegativeInfinity

    def _eval_is_positive(self):  # type:ignore[override]
        return fuzzy_or(a.is_positive for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):  # type:ignore[override]
        return fuzzy_or(a.is_nonnegative for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_negative(self):  # type:ignore[override]
        return fuzzy_and(a.is_negative for a in self.args)


class Min(MinMaxBase, Application):  # type: ignore[misc]
    """
    Return, if possible, the minimum value of the list.
    """

    zero = S.NegativeInfinity
    identity = S.Infinity

    def _eval_is_positive(self):  # type:ignore[override]
        return fuzzy_and(a.is_positive for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_nonnegative(self):  # type:ignore[override]
        return fuzzy_and(a.is_nonnegative for a in self.args)  # type: ignore[attr-defined]

    def _eval_is_negative(self):  # type:ignore[override]
        return fuzzy_or(a.is_negative for a in self.args)


def safe_pow(base: sympy.Integer, exp: sympy.Integer) -> int | IntInfinity:
    sign = 1
    if base < 0:
        base = -base
        sign = 1 if exp % 2 == 0 else -1
    return sign * _safe_pow(base, exp)


# Prevent people from overflowing pow
def _safe_pow(base: sympy.Integer, exponent: sympy.Integer) -> int | IntInfinity:
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")

    if exponent == 0:
        return 1

    half_exp = safe_pow(base, exponent // 2)
    if half_exp is int_oo:
        return int_oo

    # TODO: microoptimization is to avoid overflowing into arbitrary precision
    # and detect overflow prior to doing operations

    result = half_exp * half_exp
    if result > sys.maxsize:
        return int_oo

    if exponent % 2 == 1:
        result *= base
        if result > sys.maxsize:
            return int_oo

    return result


class PowByNatural(sympy.Function):
    is_integer = True

    precedence: int = 50  # precedence of mul

    @classmethod
    def eval(cls, base: sympy.Expr, exp: sympy.Expr) -> sympy.Basic | None:
        if isinstance(base, sympy.Integer) and isinstance(exp, sympy.Integer):
            r = safe_pow(base, exp)
            if r in (-int_oo, int_oo):
                return r
            return sympy.Integer(r)
        if isinstance(exp, sympy.Integer):
            # Rely on regular sympy Pow for this (note that iterated
            # multiplication turns into a Pow anyway, you can't escape!!)
            return sympy.Pow(base, exp)
        if exp in (int_oo, sympy.oo):
            if base.is_nonnegative:
                return int_oo
            elif base.is_negative:
                return sympy.zoo  # this is apparently what (-2)**sympy.oo does
        # NB: do NOT translate into sympy.Pow, we will lose knowledge that exp
        # is a natural number if we do
        return None


# base is assumed to be nonnegative, thereby prevent complex numbers from
# occurring
class FloatPow(sympy.Function):
    is_real = True

    precedence: int = 60  # precedence of pow

    @classmethod
    def eval(cls, base: sympy.Expr, exp: sympy.Expr) -> sympy.Basic | None:
        # NB: These test sympy.Number, not sympy.Float, because:
        #   - Sometimes we may have sympy.oo or int_oo, and that's not a Float
        #     (but coerces to math.Inf)
        #   - Sometimes Float(0.0) will unpredictably decay to Integer(0),
        #     but we should still accept it in floatey contexts
        if isinstance(base, sympy.Number) and isinstance(exp, sympy.Number):
            return sympy.Float(float(base) ** float(exp))
        # NB: do not do any nontrivial reasoning
        return None


# Overloaded to be compatible with regular Python.
# https://github.com/pytorch/pytorch/issues/90900
#
# In particular, sympy division is willing to simplify x/x == 1
# where 1 is an integer, but this must be a float if x was float.
class FloatTrueDiv(sympy.Function):
    is_real = True

    precedence: int = 35  # lower precedence than add

    @classmethod
    def eval(cls, base: sympy.Expr, divisor: sympy.Expr) -> sympy.Basic | None:
        # assert base.is_integer is not True, base
        # assert divisor.is_integer is not True, divisor

        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if isinstance(base, sympy.Number) and isinstance(divisor, sympy.Number):
            return sympy.Float(float(base) / float(divisor))
        return None


# Overloaded to be compatible with regular Python.  We distinguish this from
# FloatTrueDiv, because the code generation has to be different for this case:
# Python has a fancy algorithm for integer true division that isn't just
# "promote both arguments to float and use float division", so you need to
# codegen it differently.  While technically you can work it out from the
# types of the input, this is often inconvenient to do in Inductor codegen,
# so just have a different operator
# NB: Right now, Inductor codegen doesn't implement this correctly lol
class IntTrueDiv(sympy.Function):
    is_real = True

    precedence: int = 35  # lower precedence than add

    @classmethod
    def eval(cls, base: sympy.Expr, divisor: sympy.Expr) -> sympy.Basic | None:
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (is_infinite(base) or is_infinite(divisor))
        ):
            # Don't have to worry about precision here, you're getting zero or
            # inf from the division
            return sympy.Float(float(base) / float(divisor))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Float(int(base) / int(divisor))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        base = printer.parenthesize(self.args[0], PRECEDENCE["Atom"] - 0.5)
        divisor = printer.parenthesize(self.args[1], PRECEDENCE["Atom"] - 0.5)
        return f"((int){base}/(int){divisor})"


# TODO: As an indicator, this != 0 implies == 1 (and vice versa).
# Because we do not have the ability to guard on the stride permutation
# at the moment, it is hard to make further inferences when this is true,
# as although we know the tensor is contiguous in *some* layout, we don't
# know which one (however, you could, for example, make the inference that
# reshaping this to a 1D tensor can be guard-free.)
class IsNonOverlappingAndDenseIndicator(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, *args: sympy.Expr) -> int | None:
        if len(args) % 2 != 0:
            raise AssertionError(
                f"expected an even number of arguments, got {len(args)}"
            )
        dim = len(args) // 2
        sizes = args[0:dim]
        strides = args[dim:]

        # sym_node imported in torch.__init__. Local import to avoid an import cycle
        from torch.fx.experimental.symbolic_shapes import (
            eval_is_non_overlapping_and_dense,
        )

        if all(isinstance(a, sympy.Integer) for a in args):
            return eval_is_non_overlapping_and_dense(
                [int(a) for a in sizes], [int(a) for a in strides]
            )

        if dim == 1:
            # Manually implement the rank one short circuit
            if strides[0].is_Number and strides[0] == 1:
                return 1

            if sizes[0].is_Number and sizes[0] < 2:
                return 1

            # return 0 case covered by case above

            # TODO: Inability to access size-obliviousness sucks: if we have a
            # size oblivious test on a size-like unbacked SymInt, we could
            # confidently return zero when we have a size-like u0 stride
            # and a size-like u1 size.  Maybe a fancy ValueRanges analysis for
            # this function could help figure this out.

        if all(isinstance(a, sympy.Integer) for a in strides):
            if dim == 0:
                raise AssertionError("dim must not be zero")
            # When all strides are integral, we can sort, and the size for the
            # largest stride doesn't matter and can be arbitrarily symbolic
            s_sizes, s_strides = zip(
                *sorted(zip(sizes, strides, strict=True), key=operator.itemgetter(1)),
                strict=True,
            )
            # Put something arbitrary in the max size spot, it'll be ignored
            if all(isinstance(a, sympy.Integer) for a in s_sizes[:-1]):
                s_sizes = s_sizes[:-1] + (42,)
                # We can reuse the regular eval, because it is invariant to
                # permutation of dimensions
                return eval_is_non_overlapping_and_dense(
                    [int(a) for a in s_sizes], [int(a) for a in s_strides]
                )

        return None


# NB: this is inconsistent with math.trunc in Python
class TruncToFloat(sympy.Function):
    is_real = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        if number in (sympy.oo, -sympy.oo):
            return number
        # assert number.is_integer is not True, number
        if isinstance(number, sympy.Number):
            # NB: It is safe to use truncation to integer, which is what
            # math.trunc does, as Python integers are arbitrary precision and
            # so we are guaranteed not to lose precision when we do this
            return sympy.Float(math.trunc(float(number)))
        return None


class TruncToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number
        if number in (sympy.oo, int_oo):
            return int_oo
        if number in (-sympy.oo, -int_oo):
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(math.trunc(float(number)))
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(int64_t)(trunc({number}))"


# This is float -> int
class RoundToInt(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number

        if number is sympy.oo:
            return int_oo
        if number is -sympy.oo:
            return -int_oo
        if isinstance(number, sympy.Number):
            return sympy.Integer(round(float(number), 0))
        return None


# To get float -> int, Python style round semantics.
#
#   x = PyFloat_AsDouble(self);
#   if (o_ndigits == Py_None) {
#       /* single-argument round or with None ndigits:
#        * round to nearest integer */
#       rounded = round(x);
#       if (fabs(x-rounded) == 0.5)
#           /* halfway case: round to even */
#           rounded = 2.0*round(x/2.0);
#       return PyLong_FromDouble(rounded);
#   }


# NB: Like Round, this only ever returns floats.  ndigits cannot be None
class RoundDecimal(sympy.Function):
    is_real = True

    @classmethod
    def eval(cls, number: sympy.Expr, ndigits: sympy.Expr) -> sympy.Basic | None:
        # assert number.is_integer is not True, number

        if isinstance(number, sympy.Number) and isinstance(ndigits, sympy.Integer):
            return sympy.Float(round(float(number), int(ndigits)))
        return None


class ToFloat(sympy.Function):
    is_real = True

    @classmethod
    def eval(cls, number: sympy.Expr) -> sympy.Basic | None:
        if number in [sympy.oo, -sympy.oo]:
            return number

        if isinstance(number, sympy.Integer):
            return sympy.Float(int(number))
        if number is int_oo:
            return sympy.oo
        if number is -int_oo:
            return -sympy.oo
        return None

    def _ccode(self, printer: sympy.printing.codeprinter.CodePrinter) -> str:
        number = printer._print(self.args[0])
        return f"(double)({number})"


class Identity(sympy.Function):
    """
    Prevents expansion and other optimizations
    """

    precedence = 10

    def __repr__(self) -> str:  # type: ignore[override]
        return f"Identity({self.args[0]})"

    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        """Controls how sympy's StrPrinter prints this"""
        return f"({printer.doprint(self.args[0])})"

    def _eval_is_real(self) -> bool | None:
        return self.args[0].is_real

    def _eval_is_integer(self) -> bool | None:
        return self.args[0].is_integer  # type: ignore[attr-defined]

    @property
    def is_number(self) -> bool:
        # Treat Identity as numeric only when the argument is comparable.
        # This avoids creating numeric non-comparable Identity(I) terms.
        return bool(self.args[0].is_number and self.args[0].is_comparable)

    @property
    def is_comparable(self) -> bool:
        # Delegate comparability to the wrapped argument.
        return bool(self.args[0].is_comparable)

    def _eval_expand_identity(self, **hints: bool) -> sympy.Expr:
        # Removes the identity op.
        return self.args[0]

    def __int__(self) -> int:
        return int(self.args[0])

    def _identity_atom_compare(
        self, other: sympy.Expr | int, op: Callable[[sympy.Expr, sympy.Expr], bool]
    ) -> sympy.logic.boolalg.BooleanAtom | None:
        """
        Fast path for comparing wrapped numeric atomics against other numeric atomics.
        Keep compound expressions on SymPy's default symbolic path.
        """
        arg = self.args[0]
        if isinstance(other, int):
            other = sympy.Integer(other)
        if not isinstance(other, sympy.Expr):
            return None
        if not (arg.is_Atom and arg.is_number and arg.is_comparable):
            return None
        if not (other.is_Atom and other.is_number and other.is_comparable):
            return None
        return sympy.S.true if op(arg, other) else sympy.S.false

    def __ge__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a >= b)
        return out if out is not None else super().__ge__(other)

    def __gt__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a > b)
        return out if out is not None else super().__gt__(other)

    def __le__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a <= b)
        return out if out is not None else super().__le__(other)

    def __lt__(self, other: sympy.Expr) -> sympy.Basic:
        out = self._identity_atom_compare(other, lambda a, b: a < b)
        return out if out is not None else super().__lt__(other)

    def __float__(self) -> float:
        return float(self.args[0])


def make_opaque_unary_fn(name: str) -> type[sympy.Function]:
    class OpaqueUnaryFn(sympy.Function):
        """
        Unlike the builtin sympy functions on real numbers like sympy.sqrt,
        these equivalents do not do any nontrivial reasoning besides
        constant propagation.  This helps avoid performing transformations
        that are valid for real numbers but are invalid for floating point;
        in particular, while we are willing to make optimizations that change
        numerics for Tensor compute, we are NOT willing to make optimizations
        that change numerics for size compute.
        """

        _torch_handler_name = name
        _torch_unpickler = make_opaque_unary_fn

        @classmethod
        def eval(cls, a: sympy.Expr) -> sympy.Basic | None:
            if isinstance(a, (sympy.Integer, sympy.Float)):
                # Python converts to float64 before computing, c.f.
                # >>> math.sin(2**53+1)
                # -0.848925964814655
                # >>> math.sin(float(2**53+1))
                # -0.848925964814655
                try:
                    return sympy.Float(getattr(math, name)(float(a)))
                # Just use sympy semantics for infinity/overflow, you might get some
                # weird objects but ask silly questions, get silly answers
                except OverflowError:
                    return getattr(sympy, name)(a)
            elif a in [sympy.oo, -sympy.oo, sympy.zoo, -sympy.zoo, int_oo, -int_oo]:
                if a is int_oo:
                    a = sympy.oo
                if a is -int_oo:
                    a = -sympy.oo
                if name == "log2":
                    return sympy.log(a, 2)
                return getattr(sympy, name)(a)
            return None

    nm = "OpaqueUnaryFn_" + name
    OpaqueUnaryFn.__name__ = nm
    OpaqueUnaryFn.__qualname__ = nm

    return OpaqueUnaryFn


# Keep in sync with math_op_names in torch/fx/experimental/sym_node.py
OpaqueUnaryFn_sqrt = make_opaque_unary_fn("sqrt")
OpaqueUnaryFn_cos = make_opaque_unary_fn("cos")
OpaqueUnaryFn_cosh = make_opaque_unary_fn("cosh")
OpaqueUnaryFn_sin = make_opaque_unary_fn("sin")
OpaqueUnaryFn_sinh = make_opaque_unary_fn("sinh")
OpaqueUnaryFn_tan = make_opaque_unary_fn("tan")
OpaqueUnaryFn_tanh = make_opaque_unary_fn("tanh")
OpaqueUnaryFn_asin = make_opaque_unary_fn("asin")
OpaqueUnaryFn_acos = make_opaque_unary_fn("acos")
OpaqueUnaryFn_atan = make_opaque_unary_fn("atan")
OpaqueUnaryFn_exp = make_opaque_unary_fn("exp")
OpaqueUnaryFn_log = make_opaque_unary_fn("log")
OpaqueUnaryFn_asinh = make_opaque_unary_fn("asinh")
OpaqueUnaryFn_log2 = make_opaque_unary_fn("log2")


def make_opaque_bitwise_fn(name: str, real_op_name: str) -> type[sympy.Function]:
    if name == "bitwise_and":
        prec = PRECEDENCE["BitwiseAnd"]
    elif name == "bitwise_xor":
        prec = PRECEDENCE["BitwiseXor"]
    elif name == "bitwise_or":
        prec = PRECEDENCE["BitwiseOr"]
    else:
        raise AssertionError(f"unrecognized {name}")

    class BitwiseFn(sympy.Function):
        _torch_handler_name = name
        precedence: int = prec
        _torch_unpickler = functools.partial(
            make_opaque_bitwise_fn, real_op_name=real_op_name
        )

        @classmethod
        def eval(cls, a: sympy.Expr, b: sympy.Expr) -> sympy.Basic | None:
            if a.is_Boolean and b.is_Boolean:
                return getattr(operator, real_op_name)(a, b)
            if a.is_Boolean:
                a = sympy.Integer(1 if a else 0)
            if b.is_Boolean:
                b = sympy.Integer(1 if b else 0)
            if isinstance(a, (sympy.Integer, int)) and isinstance(
                b, (sympy.Integer, int)
            ):
                return sympy.Integer(getattr(operator, real_op_name)(int(a), int(b)))
            return None

    nm = "BitwiseFn_" + name
    BitwiseFn.__name__ = nm
    BitwiseFn.__qualname__ = nm

    return BitwiseFn


BitwiseFn_bitwise_and = make_opaque_bitwise_fn("bitwise_and", "and_")
BitwiseFn_bitwise_or = make_opaque_bitwise_fn("bitwise_or", "or_")
BitwiseFn_bitwise_xor = make_opaque_bitwise_fn("bitwise_xor", "xor")
<|endoftext|># mypy: allow-untyped-defs
import torch


def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    """
    This is PyTorch implementation of main part of reorder_meta()
    function, from tools/util/include/cutlass/util/host_reorder.h file
    of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
    GEMM decides upon layout of this matrix, and at the moment for the
    sparse GEMM executed on tensor cores, this is layout described by
    ColumnMajorInterleaved<2> data structure, in
    include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    reordering of meta matrix into meta_reordered matrix calculated
    according to these segments of CUTLASS code is re-implemented here.
    Note that this calculation produces offsets for scattering metadata
    matrix elements into reordered metadata matrix elements (or,
    equivalently, for gathering reordered metadata matrix element back
    into metadata matrix elements).
    """
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group = 32 if meta_dtype.itemsize == 2 else 16
    interweave = 4 if meta_dtype.itemsize == 2 else 2
    dst_rows = (
        dst_rows // group * group
        + (dst_rows % 8) * interweave
        + (dst_rows % group) // 8
    )

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)


def sparse_semi_structured_from_dense_cutlass(dense):
    """
    This function converts dense matrix into sparse semi-structured
    representation, producing "compressed" matrix, in the layout used by
    CUTLASS backend, and corresponding metadata matrix.
    """
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, _m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, _m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, then False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    # Reorder meta tensor elements.
    meta_reordered = meta.new_empty((m * meta_ncols,))  # type: ignore[possibly-undefined]
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    # pyrefly: ignore [unbound-name]
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return (sparse, meta_reordered.view(m, meta_ncols))


def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    """
    This function performs reverse of the function above - it
    reconstructs dense matrix from a pair of "compressed" matrix, given
    in the layout used by CUTLASS backend, and accompanying metadata
    matrix.
    """
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    if sparse.dtype != torch.float:
        ksparse = 4
    else:
        ksparse = 2

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of sparse matrix {m}"
        )
    if meta_ncols * ksparse * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * ksparse * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    # Undo meta tensor elements reordering.
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets).view(m, meta_ncols)

    # Unpack sparse tensor back to original dense tensor, using
    # information provided by meta tensor.  Note that torch.float
    # datatype is handled pretty much the same as
    # torch.half/torch.bfloat16, as metadata for a pair of torch.float
    # value is encoded as if underlying 8 bytes contain four
    # torch.half/torch.bfloat16 values, where either first two or last
    # two are zeros.
    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem),
        dtype=meta_dtype,
        device=device,
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
        torch.arange(0, 2 * m * k // ksparse, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    if sparse.dtype != torch.float:
        dense.scatter_(0, dense_offsets, sparse.view(-1))
    else:
        dense.view(torch.half).scatter_(
            0, dense_offsets, sparse.view(torch.half).view(-1)
        )

    return dense.view(m, 2 * k)


def _sparse_semi_structured_tile(dense):
    """
    This function computes a 2:4 sparse tile by greedily taking the largest values.

    Since we take the largest values greedily, how the sorting algorithm handles duplicates affects
    the ultimate sparsity pattern.

    Note that this function does not have the same sorting semantics as our CUDA backend,
    which is exposed via `torch._sparse_semi_structured_tile` and thus returns a different pattern.
    """

    def greedy_prune_tile(tile):
        num_kept_row = [0, 0, 0, 0]
        num_kept_col = [0, 0, 0, 0]

        for x in tile.flatten().sort(descending=True, stable=True).indices:
            r, c = x // 4, x % 4
            if num_kept_row[r] < 2 and num_kept_col[c] < 2:
                num_kept_row[r] += 1
                num_kept_col[c] += 1
            else:
                tile[r, c] = 0

    for batch in dense.unfold(0, 4, 4).unfold(1, 4, 4):
        for tile in batch:
            greedy_prune_tile(tile)

    return dense


def _compute_compressed_swizzled_bitmask(dense):
    """
    Calculates the compressed swizzled bitmask from a dense tensor
    """

    # first we need to convert the dense tensor to a bitmask
    int_bitmask = dense.bool().to(torch.uint8)

    # Each thread is responsible for an 8x8 tile, which contains 4 4x4 tiles:
    # A, B, C and D, as displayed in the following schema:
    # +---+---+
    # | A | B |
    # +---+---+
    # | C | D |
    # +---+---+

    # we first need to split into the 8x8 tiles
    bitmask_8x8_chunks = int_bitmask.unfold(0, 8, 8).unfold(1, 8, 8)

    # then we unfold again to get our individual 4x4 tiles
    bitmask_4x4_chunks = bitmask_8x8_chunks.unfold(2, 4, 4).unfold(3, 4, 4)

    # Each 4x4 bitmask defines two 8-bit integers, which encode the sparsity pattern
    # of that tile. Note that the least significant bit is stored first.
    # [1 1 0 0]
    # [1 1 0 0]  ->  0011 0011 ->   51
    # [0 0 1 1]      1100 1100      204
    # [0 0 1 1]

    # reshape tensor to expand tiles into 8-bit vectors
    bitmask_binary_representation = bitmask_4x4_chunks.reshape(
        *bitmask_4x4_chunks.shape[:2], 4, 2, 8
    )

    # to convert from binary representation, we can do a matmul with powers of two
    powers_of_two = 2 ** torch.arange(8, dtype=torch.float, device="cuda")
    # To run on GPU: cast to float to do matmul and then cast back
    compressed_swizzled_bitmask = (
        bitmask_binary_representation.to(torch.float) @ powers_of_two
    ).to(torch.uint8)

    return compressed_swizzled_bitmask
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math
import os
import weakref
from functools import lru_cache

import torch
from torch._dynamo.utils import warn_once
from torch.utils._triton import has_triton

from ._triton_ops_meta import _get_device_name, get_meta


TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE = int(
    os.getenv("TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE", 2)
)


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def check_bsr_layout(f_name, t):
    check(
        t.layout == torch.sparse_bsr,
        f"{f_name}(): only BSR sparse format is supported for the sparse argument.",
    )


def check_device(f_name, t, device):
    check(
        t.device == device and t.device.type in ("cuda", "xpu"),
        f"{f_name}(): all inputs are expected to be on the same GPU device.",
    )


def check_mm_compatible_shapes(f_name, lhs, rhs):
    check(
        lhs.dim() >= 2 and rhs.dim() >= 2,
        f"{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, "
        f"but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}.",
    )

    _m, kl = lhs.shape[-2:]
    kr, _n = rhs.shape[-2:]

    check(
        kl == kr,
        f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, "
        f"got lhs.shape[-1] == {kl} which is not equal to rhs.shape[-2] == {kr}.",
    )


def check_dtype(f_name, t, dtype, *additional_dtypes):
    check(
        t.dtype == dtype
        and t.dtype
        in ((torch.half, torch.bfloat16, torch.float) + tuple(*additional_dtypes)),
        f"{f_name}(): all inputs are expected to be of the same dtype "
        f"and one of (half, bfloat16, float32) or {additional_dtypes}, "
        f"but got dtype == {t.dtype}.",
    )


def check_blocksize(f_name, blocksize):
    if len(blocksize) != 2:
        raise AssertionError(f"blocksize must have length 2, got {len(blocksize)}")

    def is_power_of_two(v):
        return not (v & (v - 1))

    def is_compatible_blocksize(b):
        res = True
        for blocksize in b:
            # Triton loads only blocks which are at least 16 and powers of 2.
            res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
        return res

    check(
        is_compatible_blocksize(blocksize),
        f"{f_name}(): sparse inputs' blocksize ({blocksize[0]}, {blocksize[1]}) "
        "should be at least 16 and a power of 2 in each dimension.",
    )


def make_triton_contiguous(t):
    """Return input as a triton-contiguous tensor.

    A triton-contiguous tensor is defined as a tensor that has strides
    with minimal value smaller than or equal to 1.

    While triton kernels support triton-non-contiguous tensors (all
    strides being greater than 1) arguments, a considerable slow-down
    occurs because tensor data is copied element-wise rather than
    chunk-wise. Zero strides is assumed to not have this defect.
    """
    if min(t.stride()) > 1:
        # TODO: investigate if contiguity along other axes than the
        # last one can be beneficial for performance
        return t.contiguous()
    else:
        return t


def broadcast_batch_dims(f_name, *tensors):
    try:
        return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    except Exception:
        check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")


def slicer(dim, slice_range, *tensors):
    for t in tensors:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice_range
        yield t[slices]


def multidim_slicer(dims, slices, *tensors):
    for t in tensors:
        s = [slice(None)] * t.dim()
        for d, d_slice in zip(dims, slices, strict=False):
            if d is not None:
                s[d] = d_slice
        yield t[tuple(s)]


def ptr_stride_extractor(*tensors):
    for t in tensors:
        yield t
        yield from t.stride()


def grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
    if len(full_grid) < 0 or len(full_grid) > 3:
        raise AssertionError(f"full_grid length must be 0-3, got {len(full_grid)}")
    if len(grid_blocks) < 0 or len(grid_blocks) > 3:
        raise AssertionError(f"grid_blocks length must be 0-3, got {len(grid_blocks)}")

    import itertools

    def generate_grid_points():
        for fg, mg in zip(full_grid, grid_blocks, strict=False):
            yield range(0, fg, mg)

    def generate_sliced_tensors(slices):
        for t, t_dims in tensor_dims_map.items():
            yield next(multidim_slicer(t_dims, slices, t))

    for grid_point in itertools.product(*generate_grid_points()):
        grid = [
            min(fg - gp, mg)
            for fg, gp, mg in zip(full_grid, grid_point, grid_blocks, strict=False)
        ]
        slices = [slice(gp, gp + g) for gp, g in zip(grid_point, grid, strict=False)]
        # grid_points are iterated in a "contiguous" order, i.e.
        # left dimensions traversed slower than right dimensions.
        # This order is reversed for CUDA grids.
        yield grid[::-1], *generate_sliced_tensors(slices)


def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=None):
    # cuda_max_grid = (2 ** 31 - 1, 2 ** 16 - 1, 2 ** 16 - 1)
    cuda_max_grid = (2147483647, 65535, 65535)[::-1]
    if grid_blocks is None:
        grid_blocks = cuda_max_grid
    else:

        def valid_grid_dim(g, mg):
            if g is None:
                return mg
            else:
                # grid must be at least 1 and no greater than mg
                return max(1, min(g, mg))

        grid_blocks = tuple(
            valid_grid_dim(g, mg)
            for g, mg in zip(grid_blocks, cuda_max_grid, strict=False)
        )  # type: ignore[assignment]

    for grid, *sliced_tensors in grid_partitioner(
        full_grid, grid_blocks, tensor_dims_map
    ):
        kernel(grid, *sliced_tensors)


def prepare_inputs(bsr, *dense_tensors):
    # Introduce fake batch dimension if not present for convenience.
    crow_indices = bsr.crow_indices().unsqueeze(0)
    col_indices = bsr.col_indices().unsqueeze(0)
    values = make_triton_contiguous(bsr.values().unsqueeze(0))
    tensors = [make_triton_contiguous(t.unsqueeze(0)) for t in dense_tensors]

    # Compute broadcasted batch dimension
    batch_dims_broadcasted = torch.broadcast_shapes(
        values.shape[:-3], *(t.shape[:-2] for t in tensors)
    )

    # Broadcast batch dimensions and squash.
    # The result can be either a view or a copy.
    def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
        return t.broadcast_to(batch_dims + invariant_dims).flatten(
            0, len(batch_dims) - 1
        )

    crow_indices = batch_broadcast_and_squash(
        crow_indices, batch_dims_broadcasted, (-1,)
    )

    col_indices = batch_broadcast_and_squash(col_indices, batch_dims_broadcasted, (-1,))
    values = batch_broadcast_and_squash(
        values, batch_dims_broadcasted, values.shape[-3:]
    )
    tensors = [
        batch_broadcast_and_squash(t, batch_dims_broadcasted, t.shape[-2:])
        for t in tensors
    ]

    return crow_indices, col_indices, values, *tensors


def broadcast_batch_dims_bsr(f_name, bsr, *tensors):
    batch_shape = broadcast_batch_dims(f_name, bsr, *tensors)

    crow_indices = bsr.crow_indices().broadcast_to(batch_shape + (-1,))
    col_indices = bsr.col_indices().broadcast_to(batch_shape + (-1,))
    values = bsr.values().broadcast_to(batch_shape + bsr.values().shape[-3:])
    size = batch_shape + bsr.shape[-2:]
    return torch.sparse_compressed_tensor(
        crow_indices, col_indices, values, size=size, layout=bsr.layout
    )


# NOTE: this function will ALWAYS create a view
def tile_to_blocksize(t, blocksize):
    *rest, m, n = t.shape
    new_shape = rest + [
        m // blocksize[0],
        blocksize[0],
        n // blocksize[1],
        blocksize[1],
    ]
    # using .view instead of .reshape to ensure that the result is
    # indeed a view:
    return t.view(new_shape).transpose(-3, -2)


def as1Dbatch(tensor):
    """Return tensor as 3D tensor by either prepending new dimensions to
    the tensor shape (when ``tensor.ndim < 3``), or by collapsing
    starting dimensions into the first dimension (when ``tensor.ndim >
    3``).
    """
    while tensor.ndim < 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 3:
        tensor = tensor.flatten(0, tensor.ndim - 3)
    if tensor.ndim != 3:
        raise AssertionError(
            f"tensor should have 3 dimensions after reshape, got {tensor.shape}"
        )
    return tensor


def scatter_mm(blocks, others, indices_data, *, accumulators=None):
    """Scattered matrix multiplication of tensors.

    A scattered matrix multiplication is defined as a series of matrix
    multiplications applied to input tensors according to the input
    and output mappings specified by indices data.

    The following indices data formats are supported for defining a
    scattered matrix multiplication operation (:attr:`indices_data[0]`
    holds the name of the indices data format as specified below):

    - ``"scatter_mm"`` - matrix multiplications scattered in batches
      of tensors.

      If :attr:`blocks` is a :math:`(* \times M \times K) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor,
      and :attr:`indices = indices_data['indices']` is a :math:`(*
      \times 3)` tensor, then the operation is equivalent to the
      following code::

        c_offsets, pq = indices_data[1:]
        for r in range(len(c_offsets) - 1):
            for g in range(c_offsets[r], c_offsets[r + 1]):
                p, q = pq[g]
                accumulators[r] += blocks[p] @ others[q]

    - ``"bsr_strided_mm"`` - matrix multiplications scattered in
      batches of tensors and a tensor.

      If :attr:`blocks` is a :math:`(Ms \times Ks) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor, then
      the operation is equivalent to the following code::

        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
        for b in range(nbatches):
            for i, r in enumerate(r_offsets):
                r0, r1 = divmod(r, N)
                acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                for g in range(c_indices[i], c_indices[i + 1]):
                    p = p_offsets[g]
                    q0, q1 = divmod(q_offsets[g], N)
                    acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]

      where ``Ns = N // meta['SPLIT_N']``, and ``M`` and ``K`` are
      integer multiples of ``Ms`` and ``Ks``, respectively.

    - ``"bsr_strided_mm_compressed"`` - matrix multiplications
      scattered in batches of tensors and a tensor. A memory and
      processor efficient version of ``"bsr_strided_mm"`` format.  If
      :attr:`blocks` is a :math:`(Ms \times Ks) tensor, :attr:`others`
      is a :math:`(* \times K \times N)` tensor, :attr:`accumulators`
      is a :math:`(* \times M \times N)` tensor, then the operation is
      equivalent to the following code::

        c_indices, r_offsets, q_offsets, meta = indices_data[1:]
        for b in range(nbatches):
            for r in r_offsets:
                m = (r // N) // Ms
                n = (r % N) // Ns
                r0, r1 = divmod(r, N)
                c0, c1 = c_indices[m], c_indices[m + 1]
                acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                for i, p in enumerate(range(c0, c1)):
                    q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i]
                    q0, q1 = divmod(q, N)
                    acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]

      where ``Ns = N // meta['SPLIT_N']``, and ``M`` and ``K`` are
      integer multiples of ``Ms`` and ``Ks``, respectively.

      Notice that the order of ``r_offsets`` items can be arbitrary;
      this property enables defining swizzle operators via
      rearrangements of ``r_offsets`` items..

    Auxiliary functions are provided for pre-computing
    :attr:`indices_data`. For example,
    :func:`bsr_scatter_mm_indices_data` is used to define indices data
    for matrix multiplication of BSR and strided tensors.

    Parameters
    ----------
    blocks (Tensor): a 3-D tensor of first matrices to be multiplied

    others (Tensor): a tensor of second matrices to be multiplied. If
      ``indices_data[0]=="scatter_mm"``, the tensor is a 1-D batch
      tensor of second input matrices to be multiplied. Otherwise, the
      second input matrices are slices of the :attr:`others` tensor.
    indices_data (tuple): a format data that defines the inputs and
      outputs of scattered matrix multiplications.

    Keyword arguments
    -----------------

    accumulators (Tensor, optional): a tensor of matrix product
      accumulators. If ``indices_data[0]=="scatter_mm"``, the tensor
      is a 1-D batch tensor of output matrices. Otherwise, output
      matrices are slices of the :attr:`accumulators` tensor.
    """
    indices_format = indices_data[0]

    if blocks.ndim != 3:
        raise AssertionError(f"blocks must be 3D, got {blocks.ndim}D")
    _P, Ms, Ks = blocks.shape

    if indices_format == "scatter_mm":
        c_offsets, pq = indices_data[1:]

        if others.ndim != 3:
            raise AssertionError(f"others must be 3D, got {others.ndim}D")
        _Q, Ks_, Ns = others.shape
        if Ks != Ks_:
            raise AssertionError(f"blocks K ({Ks}) != others K ({Ks_})")

        if accumulators is None:
            R = c_offsets.shape[0] - 1
            accumulators = torch.zeros(
                (R, Ms, Ns), dtype=blocks.dtype, device=blocks.device
            )
        else:
            R, Ms_, Ns_ = accumulators.shape
            if Ms_ != Ms:
                raise AssertionError(f"accumulators Ms ({Ms_}) != blocks Ms ({Ms})")
            if Ns_ != Ns:
                raise AssertionError(f"accumulators Ns ({Ns_}) != others Ns ({Ns})")

        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm2 is None:
            for r in range(c_offsets.shape[0] - 1):
                g0 = c_offsets[r]
                g1 = c_offsets[r + 1]
                for g in range(g0, g1):
                    p, q = pq[g]

                    accumulators[r] += blocks[p] @ others[q]
        else:
            _scatter_mm2(blocks, others, c_offsets, pq, accumulators)
        return accumulators

    elif indices_format == "bsr_strided_mm":
        others_shape = others.shape
        others = as1Dbatch(others)

        B, K, N = others.shape
        if K % Ks != 0:
            raise AssertionError(f"K ({K}) must be divisible by Ks ({Ks})")

        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
        SPLIT_N = meta["SPLIT_N"]

        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros(
                (*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device
            )
        else:
            M, N_ = accumulators.shape[-2:]
            if N_ != N:
                raise AssertionError(f"accumulators N ({N_}) != others N ({N})")

        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)

        Ns = N // SPLIT_N

        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm6 is None:
            accumulators.zero_()
            for b in range(B):
                for r in range(r_offsets.shape[0]):
                    r_ = r_offsets[r].item()
                    g0 = c_indices[r].item()
                    g1 = c_indices[r + 1].item()
                    r0, r1 = divmod(r_, N)
                    acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                    for g in range(g0, g1):
                        p, q = p_offsets[g], q_offsets[g]
                        q0, q1 = divmod(q.item(), N)
                        acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]
        else:
            _scatter_mm6(
                blocks,
                others,
                c_indices,
                r_offsets,
                p_offsets,
                q_offsets,
                meta,
                accumulators,
            )
        return accumulators.view(accumulators_shape)

    elif indices_format == "bsr_strided_mm_compressed":
        others_shape = others.shape
        others = as1Dbatch(others)

        B, K, N = others.shape
        if K % Ks != 0:
            raise AssertionError(f"K ({K}) must be divisible by Ks ({Ks})")

        c_indices, r_offsets, q_offsets, meta = indices_data[1:]
        SPLIT_N = meta["SPLIT_N"]

        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros(
                (*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device
            )
        else:
            M, N_ = accumulators.shape[-2:]
            if N_ != N:
                raise AssertionError(f"accumulators N ({N_}) != others N ({N})")

        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)

        Ns = N // SPLIT_N

        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm6 is None:
            for b in range(B):
                for j in range(len(r_offsets)):
                    r0, r1 = divmod(r_offsets[j].item(), N)
                    m = r0 // Ms
                    n = r1 // Ns
                    c0 = c_indices[m].item()
                    c1 = c_indices[m + 1].item()
                    acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                    for i, p in enumerate(range(c0, c1)):
                        q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i].item()
                        q0, q1 = divmod(q, N)
                        acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]
        else:
            p_offsets = torch.empty(
                (0,), dtype=q_offsets.dtype, device=q_offsets.device
            )
            _scatter_mm6(
                blocks,
                others,
                c_indices,
                r_offsets,
                p_offsets,
                q_offsets,
                meta,
                accumulators,
            )
        return accumulators.view(accumulators_shape)

    else:
        raise NotImplementedError(indices_format)


def scatter_mm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    GROUP_SIZE=None,
    TILE_M=None,
    TILE_N=None,
    SPLIT_N=None,
    num_warps=None,
    num_stages=None,
    **extra,
):
    if {TILE_M, TILE_N, SPLIT_N, num_warps, num_stages, GROUP_SIZE} == {None}:
        device_name = _get_device_name()
        meta = get_meta(
            "scatter_mm",
            (M, K, N, Ms, Ks),
            device_name,
            version=(0, torch.float16, 0.5),
        )
        if meta is not None:
            meta.update(**extra)
            return meta
        # The following parameters are optimized for the performance
        # equilibrium points of bsr-dense and dense-dense matrix
        # multiplications when using GPU card NVIDIA GeForce RTX 2060
        # SUPER. For points far from the performance equilibrium
        # points as well as for other GPU cards, the optimal
        # parameters are likely different from what specified below.
        if (M, K, N) == (256,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 1
                TILE_M = 16
                TILE_N = 16
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 2
                TILE_M = 32
                TILE_N = 16
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 1
                TILE_M = 32
                TILE_N = 32
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 1
                TILE_M = 32
                TILE_N = 32
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
        elif (M, K, N) == (512,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 8
                TILE_M = 16
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 2  # noqa: E225, E231
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 8
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 2  # noqa: E225, E231
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 4
                TILE_M = 32
                TILE_N = 128
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 8
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
        elif (M, K, N) == (1024,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 4
                TILE_M = 16
                TILE_N = 128
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1  # noqa: E225, E231
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 8
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1  # noqa: E225, E231
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 2  # noqa: E225, E231
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (256, 256):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
        elif (M, K, N) == (2048,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 4
                TILE_M = 16
                TILE_N = 128
                GROUP_SIZE = 8
                num_stages = 1
                num_warps = 1  # noqa: E225, E231
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 4
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 1  # noqa: E225, E231
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 4
                TILE_M = 64
                TILE_N = 128
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 8
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
            elif (Ms, Ks) == (256, 256):
                SPLIT_N = 4
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225, E231
        elif (M, K, N) == (4096,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 2
                TILE_M = 16
                TILE_N = 256
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 2  # noqa: E225, E231
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 2
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1  # noqa: E225, E231
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 2
                TILE_M = 64
                TILE_N = 128
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225, E231

    if SPLIT_N is None:
        # Assume NVIDIA GeForce RTX 2060 SUPER:
        # With the probability of 92% (99.9% when N > 512), the
        # performance will not be worse more than 2% from the
        # performance when using an optimal value.  Otherwise, when N
        # <= 512, using the following heuristics may give up to 15%
        # lower performance.
        SPLIT_N = {
            16: 1,
            32: 2,
            64: 4,
            128: 8,
            256: 16,
            512: 8,
            1024: 16,
            4096: 32,
            8192: 64,
        }.get(N, 16)
        if Ms >= 512 and N >= 2048:
            SPLIT_N = 1
    Ns = N // SPLIT_N
    if TILE_M is None:
        TILE_M = min(64 if Ns < 512 else 32, Ms)
    if TILE_N is None:
        TILE_N = min(64 if Ns < 512 else 32, Ns)
    num_stages = num_stages or 1
    if num_warps is None:
        if min(M, N) > 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 256:
            num_warps = {16: 1, 32: 4}.get(Ms, 4)
        else:
            num_warps = {16: 1, 32: 2}.get(Ms, 4)
    GROUP_SIZE = GROUP_SIZE or 4

    if TILE_M > Ms:
        raise AssertionError(f"TILE_M ({TILE_M}) must be <= Ms ({Ms})")
    if TILE_N > Ns:
        raise AssertionError(f"TILE_N ({TILE_N}) must be <= Ns ({Ns})")
    if Ms > M:
        raise AssertionError(f"Ms ({Ms}) must be <= M ({M})")
    if Ns > N:
        raise AssertionError(f"Ns ({Ns}) must be <= N ({N})")
    if Ks > K:
        raise AssertionError(f"Ks ({Ks}) must be <= K ({K})")

    return dict(
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        GROUP_SIZE=GROUP_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        SPLIT_N=SPLIT_N,
        **extra,
    )


def bsr_dense_addmm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    beta,
    alpha,
    SPLIT_N=None,
    GROUP_SIZE_ROW=None,
    num_warps=None,
    num_stages=None,
    sparsity=None,
    dtype=None,
    out_dtype=None,
    _version=0,
    **extra,
):
    # Specifying _version is useful for situations when one wants to
    # discard existing triton kernel tuning results, say, in testing
    # bsr_dense_addmm_meta functionality.
    if dtype is None:
        dtype = torch.float16
    if out_dtype is None:
        out_dtype = dtype
    if sparsity is None:
        sparsity = 0.5
    if {SPLIT_N, num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = _get_device_name()
        key = (M, K, N, Ms, Ks, beta == 0, beta == 1, alpha == 1)
        if dtype is out_dtype:
            version_dtype = dtype
        else:
            version_dtype = dtype, out_dtype
        meta = get_meta(
            "bsr_dense_addmm",
            key,
            device_name,
            version=(_version, version_dtype, sparsity),
        )
        if meta is None and sparsity != 0.5:
            meta = get_meta(
                "bsr_dense_addmm",
                key,
                device_name,
                version=(_version, version_dtype, 0.5),
            )
        if meta is None and dtype is not out_dtype:
            meta = get_meta(
                "bsr_dense_addmm", key, device_name, version=(_version, dtype, 0.5)
            )
        if meta is None:
            # find approximate meta such that N % SPLIT_N == 0.
            matching_meta = get_meta(
                "bsr_dense_addmm",
                (*key[:2], "*", *key[3:]),
                device_name,
                version=(_version, version_dtype, 0.5),
            )
            if matching_meta is None and dtype is not out_dtype:
                matching_meta = get_meta(
                    "bsr_dense_addmm",
                    (*key[:2], "*", *key[3:]),
                    device_name,
                    version=(_version, dtype, 0.5),
                )
            for mkey in sorted(matching_meta or {}):
                meta_ = matching_meta[mkey]
                n = mkey[2]
                split_n = meta_["SPLIT_N"]
                c = n // split_n
                if N % c == 0 and n <= N:
                    meta = dict(meta_)
                    meta["SPLIT_N"] = N // c
        if meta is not None:
            meta.update(**extra)
            return meta
        else:
            # see [Computing optimal kernel parameters] in
            # _triton_ops_meta.py for ways to avoid this warning
            # message
            warn_once(
                "bsr_dense_addmm uses non-optimal triton kernel parameters"
                f" for {M=} {K=} {N=} {Ms=}, {Ks=} {beta=} {alpha=} {dtype=} {out_dtype=}"
            )

    SPLIT_N = SPLIT_N or max(N // Ms, 1)
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    num_stages = num_stages or 1
    num_warps = num_warps or 4
    return dict(
        SPLIT_N=SPLIT_N,
        GROUP_SIZE_ROW=GROUP_SIZE_ROW,
        num_stages=num_stages,
        num_warps=num_warps,
        **extra,
    )


class TensorAsKey:
    """A light-weight wrapper of a tensor that enables storing tensors as
    keys with efficient memory reference based comparison as an
    approximation to data equality based keys.

    Motivation: the hash value of a torch tensor is tensor instance
    based that does not use data equality and makes the usage of
    tensors as keys less useful. For instance, the result of
    ``len({a.crow_indices(), a.crow_indices()})`` is `2`, although,
    the tensor results from `crow_indices` method call are equal, in
    fact, these share the same data storage.
    On the other hand, for efficient caching of tensors we want to
    avoid calling torch.equal that compares tensors item-wise.

    TensorAsKey offers a compromise in that it guarantees key equality
    of tensors that references data in the same storage in the same
    manner and without accessing underlying data. However, this
    approach does not always guarantee correctness. For instance, for
    a complex tensor ``x``, we have ``TensorAsKey(x) ==
    TensorAsKey(x.conj())`` while ``torch.equal(x, x.conj())`` would
    return False.
    """

    def __init__(self, obj):
        def get_tensor_key(obj):
            # Warning: TensorAsKey does not track negative nor
            # conjugate bits of its input object because in the use
            # case of wrapping compressed/plain indices of compressed
            # sparse tensors (that are always integer tensors with
            # non-negative items) these bits are never set. However,
            # when extending the use of TensorAsKey to float or
            # complex tensors, the values of these bits (see is_neg
            # and is_conj methods) must be included in the key as
            # well.
            if obj.dtype.is_floating_point or obj.dtype.is_complex:
                raise AssertionError(
                    f"TensorAsKey does not support floating point or complex dtype: {obj.dtype}"
                )
            return (
                obj.data_ptr(),
                obj.storage_offset(),
                obj.shape,
                obj.stride(),
                obj.dtype,
            )

        self._obj_ref = weakref.ref(obj)
        if obj.layout is torch.strided:
            self.key = get_tensor_key(obj)
        elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.key = (
                get_tensor_key(obj.crow_indices()),
                get_tensor_key(obj.col_indices()),
            )
        elif obj.layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.key = (
                get_tensor_key(obj.ccol_indices()),
                get_tensor_key(obj.row_indices()),
            )
        else:
            raise NotImplementedError(obj.layout)
        self._hash = hash(self.key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, TensorAsKey):
            return False
        if self.obj is None or other.obj is None:
            # dead objects always compare unequal unless these are
            # same objects
            return self is other
        return self.key == other.key

    @property
    def obj(self):
        """Return object if alive, otherwise None."""
        return self._obj_ref()


@lru_cache(maxsize=TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE)
def _bsr_scatter_mm_indices_data(
    indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, compressed_sparse_tensor_as_key
):
    bsr = compressed_sparse_tensor_as_key.obj
    if bsr is None:
        raise AssertionError("compressed_sparse_tensor_as_key.obj is None")
    crow_indices, col_indices = bsr.crow_indices(), bsr.col_indices()
    device = crow_indices.device
    indices_dtype = torch.int32

    if indices_format == "bsr_strided_mm_compressed":
        Ns = N // SPLIT_N
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            q_offsets_lst.append(
                (col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N)
                + b.repeat_interleave(r1 - r0)
            )
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = crow_indices
        # swizzle operation: mm elements with longer sums are computed first:
        nnz_per_row = crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N)
        nnz_per_row, indices = nnz_per_row.sort(descending=True, stable=True)
        r_offsets = r_offsets[indices]
        return (indices_format, c_indices, r_offsets, q_offsets)

    elif indices_format == "bsr_strided_mm":
        Ns = N // SPLIT_N
        p_offsets_lst = []
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            p_offsets_lst.append(
                torch.arange(r0, r1, dtype=indices_dtype, device=device).repeat(SPLIT_N)
            )
            q_offsets_lst.append(
                (col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N)
                + b.repeat_interleave(r1 - r0)
            )
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = torch.cat(
            (
                crow_indices[:1],
                torch.cumsum(
                    crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N),
                    0,
                ),
            )
        )
        p_offsets = torch.cat(p_offsets_lst)
        return (indices_format, c_indices, r_offsets, p_offsets, q_offsets)

    elif indices_format == "scatter_mm":
        Ns = Ms
        c_indices = [0]
        pq_offsets = []
        # todo: eliminate inner for-loops for efficiency
        for b in range(nbatches):
            for m in range(M // Ms):
                r0 = crow_indices[m].item()
                r1 = crow_indices[m + 1].item()
                for n in range(N // Ns):
                    c_indices.append(c_indices[-1] + r1 - r0)
                    for t in range(r1 - r0):
                        p = r0 + t
                        q = (col_indices[p].item() + b * (K // Ks)) * (N // Ns) + n
                        pq_offsets.append([p, q])

        return (
            indices_format,
            torch.tensor(c_indices, dtype=indices_dtype, device=device),
            torch.tensor(pq_offsets, dtype=indices_dtype, device=device),
        )

    else:
        raise ValueError(
            f"Invalid {indices_format=}. Expected bsr_strided_mm_compressed|bsr_strided_mm|scatter_mm"
        )


def bsr_scatter_mm_indices_data(
    bsr, other, indices_format="bsr_strided_mm_compressed", **meta_input
):
    """Computes indices data for :func:`scatter_mm` used in BSR and
    strided tensor matrix multiplication.
    """
    if bsr.dense_dim() != 0:
        raise AssertionError(f"bsr.dense_dim() must be 0, got {bsr.dense_dim()}")
    if bsr.ndim != 2:
        raise AssertionError(f"bsr must be 2D (no batch dims), got {bsr.ndim}D")
    blocksize = bsr.values().shape[-2:]
    M, K = bsr.shape
    Ms, Ks = blocksize
    K_, N = other.shape[-2:]
    if K_ != K:
        raise AssertionError(f"other K ({K_}) != bsr K ({K})")
    nbatches = other.shape[:-2].numel()

    meta = scatter_mm_meta(M, K, N, Ms, Ks, **meta_input)
    if "allow_tf32" not in meta_input:
        meta.update(allow_tf32=bsr.dtype in {torch.float16, torch.bfloat16})
    SPLIT_N = meta["SPLIT_N"]
    indices_data = _bsr_scatter_mm_indices_data(
        indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, TensorAsKey(bsr)
    )

    if indices_format == "bsr_strided_mm_compressed":
        meta.update(is_compressed=True)
        return indices_data + (meta,)
    elif indices_format == "bsr_strided_mm":
        meta.update(is_compressed=False)
        return indices_data + (meta,)
    else:
        return indices_data


def bsr_scatter_mm(bsr, other, indices_data=None, out=None):
    """BSR @ strided -> strided"""

    if bsr.ndim != 2:
        raise AssertionError(f"bsr must be 2D, got {bsr.ndim}D")
    if other.ndim < 2:
        raise AssertionError(
            f"other must have at least 2 dimensions, got {other.ndim}D"
        )

    Ms, Ks, Ns = bsr.shape[-2], bsr.shape[-1], other.shape[-1]
    blocksize = bsr.values().shape[-2:]

    if indices_data is None:
        indices_data = bsr_scatter_mm_indices_data(
            bsr, other, indices_format="bsr_strided_mm_compressed"
        )

    indices_format = indices_data[0]

    if out is None:
        out = torch.empty(
            (*other.shape[:-2], Ms, Ns), dtype=bsr.dtype, device=bsr.device
        )
    out_shape = out.shape
    out = as1Dbatch(out)

    if bsr._nnz() == 0:
        out.zero_()
    elif indices_format in {"bsr_strided_mm_compressed", "bsr_strided_mm"}:
        out.zero_()
        scatter_mm(bsr.values(), other, indices_data, accumulators=out)
    elif indices_format == "scatter_mm":
        nbatches = other.shape[:-2].numel()
        accumulators = torch.zeros(
            (
                nbatches * Ms // blocksize[0] * Ns // blocksize[0],
                blocksize[0],
                blocksize[0],
            ),
            dtype=bsr.dtype,
            device=bsr.device,
        )
        others = (
            as1Dbatch(other)
            .transpose(-2, -1)
            .view(
                nbatches,
                Ns // blocksize[0],
                blocksize[0],
                Ks // blocksize[1],
                blocksize[1],
            )
            .movedim(
                (3, 1, 4, 2), (1, 2, 3, 4)
            )  # equivalent to .transpose(-3, -2).transpose(-2, -1).transpose(-4, -3)
            .flatten(0, 2)
        )
        scatter_mm(bsr.values(), others, indices_data, accumulators=accumulators)
        out.copy_(
            accumulators.unflatten(
                0, (nbatches, Ms // blocksize[0], Ns // blocksize[0])
            )
            .movedim(
                (1, 2, 3, 4), (3, 1, 4, 2)
            )  # equivalent to .transpose(-4, -3).transpose(-2, -1).transpose(-3, -2)
            .reshape(nbatches, Ns, Ms)
            .transpose(-2, -1)
        )
    else:
        raise NotImplementedError(indices_format)

    return out.view(out_shape)


def _int_bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=1,
    alpha=1,
    left_alpha: torch.Tensor | None = None,
    right_alpha: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    skip_checks: bool = False,
    max_grid: tuple[int | None, int | None, int | None] | None = None,
    meta: dict | None = None,
):
    if out is None and dense.dtype is torch.int8:
        f_name = "_int_bsr_dense_addmm"
        crow_indices = bsr.crow_indices()
        batch_ndim = crow_indices.dim() - 1
        M = bsr.shape[batch_ndim]
        N = dense.shape[-1]
        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
        out = torch.empty(
            original_batch_dims_broadcasted + (M, N),
            dtype=torch.int32,
            device=dense.device,
        )
    return bsr_dense_addmm(
        input,
        bsr,
        dense,
        beta=beta,
        alpha=alpha,
        left_alpha=left_alpha,
        right_alpha=right_alpha,
        out=out,
        skip_checks=skip_checks,
        max_grid=max_grid,
        meta=meta,
    )


def bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=1,
    alpha=1,
    left_alpha: torch.Tensor | None = None,
    right_alpha: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    skip_checks: bool = False,
    max_grid: tuple[int | None, int | None, int | None] | None = None,
    meta: dict | None = None,
):
    """Compute

      out = beta * input + left_alpha.reshape(-1, 1) * (alpha * (bsr @ dense)) * right_alpha.reshape(1, -1)

    where left_alpha, right_alpha are (* + 1)-D tensors when
    specified, otherwise, these are treated as tensors filled with
    ones.
    """
    f_name = "bsr_dense_addmm"
    values = bsr.values()
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    batch_ndim = crow_indices.dim() - 1
    M, K = bsr.shape[batch_ndim : batch_ndim + 2]
    blocksize = values.shape[batch_ndim + 1 : batch_ndim + 3]
    N = dense.shape[-1]

    # todo: implement checks

    original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
    if out is None:
        out = dense.new_empty(original_batch_dims_broadcasted + (M, N))

    if bsr._nnz() == 0 or alpha == 0 or N == 0 or M == 0 or K == 0:
        if beta == 0:
            out.zero_()
        else:
            out.copy_(input)
            if beta != 1:
                out.mul_(beta)
        return out

    left_alpha_is_one = False
    right_alpha_is_one = False
    if left_alpha is None:
        left_alpha_is_one = True
        left_alpha = dense.new_empty(()).expand(
            *original_batch_dims_broadcasted, M, N
        )  # not referenced
    else:
        left_alpha = left_alpha.view(*original_batch_dims_broadcasted, M, 1).expand(
            *original_batch_dims_broadcasted, M, N
        )

    if right_alpha is None:
        right_alpha_is_one = True
        right_alpha = dense.new_empty(()).expand(
            *original_batch_dims_broadcasted, M, N
        )  # not referenced
    else:
        right_alpha = right_alpha.view(*original_batch_dims_broadcasted, 1, N).expand(
            *original_batch_dims_broadcasted, M, N
        )
    if left_alpha.stride()[-1] != 0:
        raise AssertionError(
            f"left_alpha.stride()[-1] must be 0, got {left_alpha.stride()[-1]}"
        )
    if right_alpha.stride()[-2] != 0:
        raise AssertionError(
            f"right_alpha.stride()[-2] must be 0, got {right_alpha.stride()[-2]}"
        )

    if meta is None:
        sparsity = round(1 - bsr._nnz() * blocksize[0] * blocksize[1] / (M * K), 2)
        meta = bsr_dense_addmm_meta(
            M,
            K,
            N,
            blocksize[0],
            blocksize[1],
            beta,
            alpha,
            sparsity=sparsity,
            dtype=dense.dtype,
            out_dtype=out.dtype,
        )
    out_backup = out

    (
        crow_indices,
        col_indices,
        values,
        input,
        dense,
        left_alpha,
        right_alpha,
        out,
    ) = prepare_inputs(bsr, input, dense, left_alpha, right_alpha, out)

    BM, BK = blocksize
    SPLIT_N = meta.get("SPLIT_N", N // BM)
    BN = N // SPLIT_N

    out_untiled = out
    out = tile_to_blocksize(out, (BM, BN))
    dense = tile_to_blocksize(dense, (BK, BN))
    input = tile_to_blocksize(input, (BM, BN))
    left_alpha = tile_to_blocksize(left_alpha, (BM, BN))
    right_alpha = tile_to_blocksize(right_alpha, (BM, BN))

    # tl.dot supports float16, float32, int32 as accumulator types.
    dot_out_dtype = {
        torch.float16: tl.float32,
        torch.bfloat16: tl.float32,
        torch.float32: tl.float64,
        torch.float64: tl.float64,
        torch.int8: tl.int32,
        torch.int32: tl.int32,
    }[out.dtype]

    n_batches = dense.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    n_block_cols = dense.size(-3)

    full_grid = (n_batches, n_block_cols, n_block_rows)
    if max_grid is not None:
        grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
    else:
        grid_blocks = None

    tensor_dims_map = {
        values: (0, None, None),
        crow_indices: (0, None, -1),
        col_indices: (0, None, None),
        input: (0, -3, -4),
        dense: (0, -3, None),
        left_alpha: (0, -3, -4),
        right_alpha: (0, -3, -4),
        out: (0, -3, -4),
    }

    if alpha == 0:
        raise AssertionError("alpha must not be 0")

    def kernel(grid, *sliced_tensors):
        # pyrefly: ignore [unsupported-operation]
        _bsr_strided_addmm_kernel[grid](
            *ptr_stride_extractor(*sliced_tensors),
            beta,
            alpha,
            beta_is_one=beta == 1,
            beta_is_nonzero=beta != 0,
            alpha_is_one=alpha == 1,
            left_alpha_is_one=left_alpha_is_one,
            right_alpha_is_one=right_alpha_is_one,
            BLOCKSIZE_ROW=BM,
            BLOCKSIZE_INNER=BK,
            BLOCKSIZE_COL=BN,
            allow_tf32=dot_out_dtype == tl.float32,
            acc_dtype=dot_out_dtype,
            **meta,
        )

    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    if out.data_ptr() != out_backup.data_ptr():
        # prepare_inputs has made a copy of out, copy its content back
        # to out_backup:
        out_backup.copy_(out_untiled.view(out_backup.shape))

    return out_backup


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _sampled_addmm_kernel(
        alpha,
        beta,
        IS_BETA_ZERO: tl.constexpr,
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        k,
        TILE_K: tl.constexpr,
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        mat1_ptr,
        mat1_batch_stride,
        mat1_tiled_row_stride,
        mat1_tiled_col_stride,
        mat1_row_block_stride,
        mat1_col_block_stride,
        mat2_ptr,
        mat2_batch_stride,
        mat2_tiled_row_stride,
        mat2_tiled_col_stride,
        mat2_row_block_stride,
        mat2_col_block_stride,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * col_block_arange[None, :]
        )

        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        # Advance mat1 to the current tiled row, ignore columns.
        mat1_block_ptrs = (
            mat1_ptr
            + mat1_batch_stride * batch_pid
            + mat1_tiled_row_stride * row_block_pid
            + mat1_row_block_stride * row_block_arange[:, None]
        )

        # Advance mat2 in batch and block col dimension.
        mat2_block_ptrs = (
            mat2_ptr
            + mat2_batch_stride * batch_pid
            + mat2_col_block_stride * col_block_arange[None, :]
        )

        k_tile_arange = tl.arange(0, TILE_K)
        for _ in range(row_nnz):
            acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

            # find column block index
            col_block = tl.load(col_index_nnz_ptr)

            for k_tile in range(0, k, TILE_K):
                k_offsets = k_tile + k_tile_arange
                mask_k = k_offsets < k

                mat1_block = tl.load(
                    mat1_block_ptrs + mat1_col_block_stride * k_offsets[None, :],
                    # pyrefly: ignore [bad-index]
                    mask=mask_k[None, :],
                    other=0.0,
                )

                mat2_block = tl.load(
                    mat2_block_ptrs
                    + mat2_tiled_col_stride * col_block
                    + mat2_row_block_stride * k_offsets[:, None],
                    # pyrefly: ignore [bad-index]
                    mask=mask_k[:, None],
                    other=0.0,
                )

                acc_block += tl.dot(
                    mat1_block, mat2_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
                )

            if IS_BETA_ZERO:
                acc_block *= alpha
            else:
                acc_block = alpha * acc_block + beta * tl.load(values_block_ptrs)

            # write result
            tl.store(values_block_ptrs, acc_block.to(values_ptr.dtype.element_ty))

            # advance val/col_index ptrs to the next block in the row.
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

    @triton.jit
    def _bsr_strided_dense_rowspace_kernel(
        # values prologue
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        # col_indices epilogue
        # dense prologue
        dense_ptr,
        dense_batch_stride,
        dense_tiled_row_stride,
        dense_tiled_col_stride,
        dense_row_block_stride,
        dense_col_block_stride,
        # dense epilogue
        # output prologue
        output_ptr,
        output_batch_stride,
        output_tiled_row_stride,
        output_tiled_col_stride,
        output_row_block_stride,
        output_col_block_stride,
        # output epilogue
        #
        # gh-113754: Always keep all constexpr arguments at the end of
        # triton kernel arguments list because with triton 2.1 or
        # earlier non-contiguous outputs will corrupt CUDA state due
        # to a triton bug (fixed in openai/triton#2262).
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
        GROUP_SIZE_ROW: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)

        row_block_pid, col_block_pid = tl.swizzle2d(
            row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
        )

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * col_block_arange[None, :]
        )

        # NOTE: dense is advanced into all dimensions but the tiled row one.
        # That will be advanced in the loop according to values in col_indices.
        dense_block_ptrs = (
            dense_ptr
            + dense_batch_stride * batch_pid
            + dense_tiled_col_stride * col_block_pid
            + dense_row_block_stride * col_block_arange[:, None]
            + dense_col_block_stride * row_block_arange[None, :]
        )

        # Pointers are set to exact write-to locations
        output_ptrs = (
            output_ptr
            + output_batch_stride * batch_pid
            + output_tiled_row_stride * row_block_pid
            + output_tiled_col_stride * col_block_pid
            + output_row_block_stride * row_block_arange[:, None]
            + output_col_block_stride * row_block_arange[None, :]
        )

        # Set pointer to the first nonzero element in the current row
        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs)

            # find which row of dense needs to get loaded
            # for multiplication with values_block.
            dense_row_idx = tl.load(col_index_nnz_ptr)
            dense_block = tl.load(
                dense_block_ptrs + dense_tiled_row_stride * dense_row_idx
            )

            # do block mm
            output_acc_block += tl.dot(
                values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
            )

            # move val/col_index ptrs to the next block in the row
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

        # write back the result
        tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))

    def _run_sampled_addmm_kernel(
        alpha,
        beta,
        is_beta_zero,
        blocksize,
        k,
        tile_k,
        values,
        crow_indices,
        col_indices,
        mat1,
        mat2,
        max_grid,
    ):
        n_batches = values.size(0)
        n_block_rows = crow_indices.size(-1) - 1

        full_grid = (n_batches, n_block_rows)
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:2][::-1]) + (None,) * (2 - len(max_grid[:2]))
        else:
            grid_blocks = None
        tensor_dims_map = {
            values: (0, None),
            crow_indices: (0, -1),
            col_indices: (0, None),
            mat1: (0, -4),
            mat2: (0, None),
        }
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        def kernel(grid, *sliced_tensors):
            _sampled_addmm_kernel[grid](
                alpha,
                beta,
                is_beta_zero,
                *blocksize,
                k,
                tile_k,
                *ptr_stride_extractor(*sliced_tensors),
                acc_dtype=acc_dtype,
                allow_tf32=allow_tf32,
                num_stages=1,
                num_warps=4,
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    def sampled_addmm(
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta=1.0,
        alpha=1.0,
        out: torch.Tensor | None = None,
        skip_checks: bool = False,
        max_grid: tuple[int | None, int | None, int | None] | None = None,
    ):
        f_name = "sampled_addmm"

        check_bsr_layout(f_name, input)
        input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)

        if not skip_checks:
            check_device(f_name, mat1, input.device)
            check_device(f_name, mat2, input.device)
            if beta != 0.0 and input.dtype is torch.bool:
                check(
                    False,
                    f"{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed.",
                )
            if input.dtype is not torch.bool:
                check_dtype(f_name, mat1, input.dtype)
                check_dtype(f_name, mat2, input.dtype)
            else:
                check_dtype(f_name, mat1, mat2.dtype)
            check_mm_compatible_shapes(f_name, mat1, mat2)
            if out is not None:
                check_bsr_layout(f_name, out)
                check_device(f_name, out, mat1.device)
                check_dtype(f_name, out, input.dtype)
                check(
                    out.shape == input_broadcasted.shape and out._nnz() == input._nnz(),
                    f"{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} "
                    f"and with nnz equal to {input_broadcasted._nnz()} "
                    f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}",
                )

        if out is None:
            out = input_broadcasted.to(mat1.dtype, copy=True)
        else:
            out.copy_(input_broadcasted)

        if out.numel() == 0 or out._nnz() == 0:
            return out

        blocksize = out.values().shape[-2:]
        k = mat1.size(-1)

        # NOTE: (m, 0) @ (0, n) == zeros(m, n)
        if alpha == 0.0 or k == 0:
            out.values().mul_(beta)
            return out

        # prepare inputs by reshaping them to be kernel-compatible
        out_backup = out
        crow_indices, col_indices, values, mat1, mat2 = prepare_inputs(out, mat1, mat2)

        mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
        mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
        tile_k = max(*blocksize)

        _run_sampled_addmm_kernel(
            alpha,
            beta,
            beta == 0.0,
            blocksize,
            k,
            tile_k,
            values,
            crow_indices,
            col_indices,
            mat1,
            mat2,
            max_grid,
        )

        # If nnz x block strides are not the same in out_backup.values and values,
        # it means that out_backup.values and values are not the views of each other,
        # so we have to copy.
        if out_backup.values().stride()[-3:] != values.stride()[-3:]:
            out_backup.values().copy_(values.reshape(out_backup.values().shape))
        return out_backup

    def bsr_dense_mm(
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        skip_checks: bool = False,
        max_grid: tuple[int | None, int | None, int | None] | None = None,
        meta: dict | None = None,
    ):
        f_name = "bsr_dense_mm"
        m, _kl = bsr.shape[-2:]
        if not skip_checks:
            check_bsr_layout(f_name, bsr)
            check_device(f_name, bsr, dense.device)
            check_dtype(f_name, bsr, dense.dtype, (torch.int8,))
            check_mm_compatible_shapes(f_name, bsr, dense)

            n = dense.size(-1)
            row_block, col_block = bsr.values().shape[-2:]
            check_blocksize(f_name, (row_block, col_block))
            check(
                not n % 16,
                f"{f_name}(): dense.size(-1) == {n} should be divisible by 16",
            )
        else:
            _kr, n = dense.shape[-2:]

        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)

        if out is not None and not skip_checks:
            expected_out_shape = original_batch_dims_broadcasted + (m, n)
            check(
                out.shape == expected_out_shape,
                "bsr_dense_mm(): `out` argument has wrong shape, "
                f"expected {expected_out_shape}, but got {out.shape}.",
            )
            check(
                out.is_contiguous() or out.transpose(-2, -1).is_contiguous(),
                "bsr_dense_mm(): only row-major/col-major `out` arguments are supported, "
                "i.e. (out.is_contiguous() or out.transpose(-2, -1).is_contiguous()) "
                "should be True.",
            )

        # Allocate out
        if out is None:
            out = dense.new_empty(original_batch_dims_broadcasted + (m, n))

        # Short circuit if lhs is zero
        if bsr._nnz() == 0:
            return out.zero_()

        # with beta==0, addmm ignores input content, so we can use out
        # as a placeholder for input because their shapes match:
        return bsr_dense_addmm(out, bsr, dense, alpha=1, beta=0, out=out)

    @triton.jit
    def _bsr_softmax_kernel(
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        values_ptr,
        values_batch_stride,
        values_row_block_stride,
        values_nnz_col_block_stride,
        row_block,
        col_block,
        MAX_ROW_NNZ: tl.constexpr,
        TILE: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=2)
        row_block_offset_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_arange = tl.arange(0, TILE)
        mask = row_arange < row_nnz * col_block

        curr_row_values_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_row_block_stride * row_block_offset_pid
            + nnz_offset * col_block
        )

        # find max in the row
        row_tile = tl.load(
            curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
        ).to(tl.float32)
        max_row_value = tl.max(row_tile, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(
                curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
            ).to(tl.float32)
            curr_max_row_value = tl.max(row_tile, axis=0)
            max_row_value = tl.where(
                max_row_value > curr_max_row_value, max_row_value, curr_max_row_value
            )

        # find denominator for stable softmax
        num = tl.exp(row_tile - max_row_value)
        denom = tl.sum(num, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange -= TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(
                curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
            ).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            denom += tl.sum(num, axis=0)

        # populate output
        tl.store(
            curr_row_values_ptrs + row_arange,
            (num / denom).to(values_ptr.dtype.element_ty),
            mask=mask,
        )
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(
                curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
            ).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            tl.store(
                curr_row_values_ptrs + row_arange,
                (num / denom).to(values_ptr.dtype.element_ty),
                mask=mask,
            )

    def bsr_softmax(input, max_row_nnz=None):
        f_name = "bsr_softmax"

        check_bsr_layout(f_name, input)
        check_dtype(f_name, input, input.dtype)

        if input._nnz() == 0 or input.numel() == 0:
            return input.clone()

        m, n = input.shape[-2:]
        nnz = input._nnz()
        row_block, col_block = input.values().shape[-2:]

        if max_row_nnz is None:
            max_row_nnz = triton.next_power_of_2(n)
        else:
            max_row_nnz = triton.next_power_of_2(max_row_nnz)

        crow_indices = input.crow_indices().unsqueeze(0).flatten(0, -2)
        # reshape values from
        # (b1, ..., bn, nnz, row_block, col_block) to
        # (b1 * ... * bn, row_block, nnz * col_block).
        # This simplifies batch dim manipulation and unlocks
        # the possibility to access all nnzs in any given row.
        if input.values().transpose(-3, -2).is_contiguous():
            # Need to clone to avoid `contiguous` returning a view.
            values = input.values().clone()
        else:
            values = input.values()
        values = (
            values.transpose(-3, -2)
            .contiguous()
            .unsqueeze(0)
            .flatten(0, -4)
            .reshape(-1, row_block, nnz * col_block)
        )
        full_grid = (values.shape[0], row_block, m // row_block)
        grid_blocks = None
        tensor_dims_map = {
            # We span nnz number of blocks, not nnz + 1,
            # hence crow_indices[..., :-1]
            crow_indices[..., :-1]: (0, None, -1),
            values: (0, None, None),
        }

        def kernel(grid, *sliced_tensors):
            _bsr_softmax_kernel[grid](
                *ptr_stride_extractor(*sliced_tensors),
                row_block,
                col_block,
                max_row_nnz,
                # Triton's max numel is bounded by 2 ** 17.
                min(2**17, max_row_nnz),
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

        values = (
            values.reshape(-1, row_block, nnz, col_block)
            .transpose(-3, -2)
            .reshape(*input.values().shape)
        )

        return torch.sparse_compressed_tensor(
            input.crow_indices().clone(),
            input.col_indices().clone(),
            values,
            size=input.shape,
            layout=input.layout,
        )

    def _scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ):
        f_name = "_scaled_dot_product_attention"
        check(not is_causal, f"{f_name}(): is_causal == True is not supported.")
        check(attn_mask is not None, f"{f_name}(): attn_mask == None is not supported.")
        if attn_mask is None:
            raise AssertionError("attn_mask must not be None")

        check(
            attn_mask.layout == torch.sparse_bsr,
            f"{f_name}(): "
            f"attn_mask.layout must be {torch.sparse_bsr}, but got "
            f"attn_mask.layout == {attn_mask.layout}.",
        )

        check_device(f_name, key, query.device)
        check_device(f_name, value, query.device)
        check_device(f_name, attn_mask, query.device)

        check_dtype(f_name, key, query.dtype)
        check_dtype(f_name, value, query.dtype)
        if attn_mask.dtype is not torch.bool:
            check_dtype(f_name, attn_mask, query.dtype)

        # pyrefly: ignore [not-callable]
        sdpa = sampled_addmm(
            attn_mask, query, key.transpose(-2, -1), beta=0.0, skip_checks=False
        )
        if scale is None and query.size(-1) == 0 or scale == 0.0:
            check(
                False,
                f"{f_name}(): current value of scale == {scale} "
                "results in division by zero.",
            )
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        sdpa.values().mul_(scale_factor)
        # pyrefly: ignore [not-callable]
        sdpa = bsr_softmax(sdpa)
        torch.nn.functional.dropout(sdpa.values(), p=dropout_p, inplace=True)
        # pyrefly: ignore [not-callable]
        sdpa = bsr_dense_mm(sdpa, value)
        return sdpa

    @triton.jit
    def _scatter_mm2_kernel(
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        blocks_ptr,
        blocks_stride_P,
        blocks_stride_M,
        blocks_stride_K,
        others_ptr,
        others_stride_Q,
        others_stride_K,
        others_stride_N,
        accumulators_ptr,
        accumulators_stride_R,
        accumulators_stride_M,
        accumulators_stride_N,
        pq_offsets_ptr,
        pq_offsets_stride,
        pq_ptr,
        pq_stride_T,
        pq_stride_1,
        dot_out_dtype: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        Ms = M // TILE_M

        pid_t = tl.program_id(axis=0)

        pid = tl.program_id(axis=1)
        pid_m = pid // Ms
        pid_n = pid % Ms

        rm = pid_m * TILE_M + tl.arange(0, TILE_M)
        rn = pid_n * TILE_N + tl.arange(0, TILE_N)
        rk = tl.arange(0, K)

        A_ptr = blocks_ptr + (
            rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K
        )
        B_ptr = others_ptr + (
            rk[:, None] * others_stride_K + rn[None, :] * others_stride_N
        )

        g0 = tl.load(pq_offsets_ptr + pid_t * pq_offsets_stride)
        g1 = tl.load(pq_offsets_ptr + (pid_t + 1) * pq_offsets_stride)

        if g0 == g1:
            return

        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)

        for i in range(g0, g1):
            p = tl.load(pq_ptr + i * pq_stride_T)
            q = tl.load(pq_ptr + i * pq_stride_T + pq_stride_1)
            A = tl.load(A_ptr + p * blocks_stride_P)
            B = tl.load(B_ptr + q * others_stride_Q)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        C_ptr = (
            accumulators_ptr
            + pid_t * accumulators_stride_R
            + (
                rm[:, None] * accumulators_stride_M
                + rn[None, :] * accumulators_stride_N
            )
        )
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm2(
        blocks: torch.Tensor,
        others: torch.Tensor,
        pq_offsets: torch.Tensor,
        pq_indices: torch.Tensor,
        accumulators: torch.Tensor,
    ):
        _P, M, K = blocks.shape
        _Q, _, N = others.shape

        meta = dict(
            TILE_M=max(16, M // 4), TILE_N=max(16, N // 4), num_stages=1, num_warps=2
        )

        def grid(META):
            return (
                pq_offsets.shape[0] - 1,
                triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
                1,
            )

        dot_out_dtype = {
            torch.float16: tl.float32,
            torch.bfloat16: tl.float32,
            torch.float32: tl.float64,
            torch.float64: tl.float64,
        }[accumulators.dtype]
        if "allow_tf32" not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)
        _scatter_mm2_kernel[grid](
            M,
            K,
            N,
            blocks,
            blocks.stride(0),
            blocks.stride(1),
            blocks.stride(2),
            others,
            others.stride(0),
            others.stride(1),
            others.stride(2),
            accumulators,
            accumulators.stride(0),
            accumulators.stride(1),
            accumulators.stride(2),
            pq_offsets,
            pq_offsets.stride(0),
            pq_indices,
            pq_indices.stride(0),
            pq_indices.stride(1),
            dot_out_dtype=dot_out_dtype,
            **meta,
        )

    @triton.jit
    def _scatter_mm6_kernel(
        nbatches,
        Ms,
        Ks: tl.constexpr,
        N,
        blocks_ptr,
        blocks_stride_P,
        blocks_stride_M,
        blocks_stride_K,
        others_ptr,
        others_stride_B,
        others_stride_K,
        others_stride_N,
        accumulators_ptr,
        accumulators_stride_B,
        accumulators_stride_M,
        accumulators_stride_N,
        c_indices_ptr,
        r_offsets_ptr,
        p_offsets_ptr,
        q_offsets_ptr,
        is_compressed: tl.constexpr,
        dot_out_dtype: tl.constexpr,
        SPLIT_N: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        Ns = N // SPLIT_N
        BLOCKS_M = Ms // TILE_M
        BLOCKS_N = Ns // TILE_N

        pid_t_ = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        pid_b = pid_t_ % nbatches
        pid_t = pid_t_ // nbatches

        num_pid_in_group = GROUP_SIZE * BLOCKS_N
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = min(BLOCKS_M - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * TILE_M + tl.arange(0, TILE_M)
        rn = pid_n * TILE_N + tl.arange(0, TILE_N)
        rk = tl.arange(0, Ks)
        A_ptr = blocks_ptr + (
            rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K
        )
        B_ptr = (
            others_ptr
            + pid_b * others_stride_B
            + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
        )

        # When is_compressed is True, r is the only variable that
        # depends on pid_t. This property allows sorting r values
        # before calling the kernel. The sorting of r is equivalent to
        # defining swizzle operator outside of the kernel.
        r = tl.load(r_offsets_ptr + pid_t)

        if is_compressed:
            m = (r // N) // Ms
            n = (r % N) // Ns
            r0 = tl.load(c_indices_ptr + m)
            r1 = tl.load(c_indices_ptr + m + 1)
            g0 = n * r1 + (SPLIT_N - n) * r0
            nnz = r1 - r0
        else:
            g0 = tl.load(c_indices_ptr + pid_t)
            g1 = tl.load(c_indices_ptr + pid_t + 1)
            nnz = g1 - g0

        q_ptr = q_offsets_ptr + g0
        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)

        if is_compressed:
            A_ptr += r0 * blocks_stride_P  # type: ignore[possibly-undefined]
            for _ in range(nnz):
                q = tl.load(q_ptr)
                B = tl.load(B_ptr + q)
                A = tl.load(A_ptr)
                acc_block += tl.dot(
                    A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32
                )
                A_ptr += blocks_stride_P
                q_ptr += 1
        else:
            p_ptr = p_offsets_ptr + g0
            for _ in range(nnz):
                q = tl.load(q_ptr)
                B = tl.load(B_ptr + q)
                p = tl.load(p_ptr)
                A = tl.load(A_ptr + p * blocks_stride_P)
                p_ptr += 1
                q_ptr += 1
                acc_block += tl.dot(
                    A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32
                )

        C_ptr = (
            accumulators_ptr
            + r
            + pid_b * accumulators_stride_B
            + (
                rm[:, None] * accumulators_stride_M
                + rn[None, :] * accumulators_stride_N
            )
        )
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm6(
        blocks: torch.Tensor,
        others: torch.Tensor,
        c_indices: torch.Tensor,
        r_offsets: torch.Tensor,
        p_offsets: torch.Tensor,
        q_offsets: torch.Tensor,
        meta: dict,
        accumulators: torch.Tensor,
        force_contiguous: bool = True,
    ):
        SPLIT_N = meta["SPLIT_N"]
        _P, Ms, Ks = blocks.shape
        B, _K, N = others.shape
        B_, _M, N_ = accumulators.shape
        if N_ != N:
            raise AssertionError(f"accumulators N ({N_}) != others N ({N})")
        Ns = N // SPLIT_N
        if B_ != B:
            raise AssertionError(f"accumulators B ({B_}) != others B ({B})")

        def grid(META):
            return (
                r_offsets.shape[0] * B,
                triton.cdiv(Ms, META["TILE_M"]) * triton.cdiv(Ns, META["TILE_N"]),
            )

        dot_out_dtype = {
            torch.float16: tl.float32,
            torch.bfloat16: tl.float32,
            torch.float32: tl.float64,
            torch.float64: tl.float64,
        }[accumulators.dtype]
        if "allow_tf32" not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)

        if c_indices.stride(0) != 1:
            raise AssertionError(
                f"c_indices.stride(0) must be 1, got {c_indices.stride(0)}"
            )
        if r_offsets.stride(0) != 1:
            raise AssertionError(
                f"r_offsets.stride(0) must be 1, got {r_offsets.stride(0)}"
            )
        if p_offsets.stride(0) != 1:
            raise AssertionError(
                f"p_offsets.stride(0) must be 1, got {p_offsets.stride(0)}"
            )
        if q_offsets.stride(0) != 1:
            raise AssertionError(
                f"q_offsets.stride(0) must be 1, got {q_offsets.stride(0)}"
            )

        # Re non-contiguous tensor arguments. Sometimes triton kernel
        # launches may fail with
        #
        #   RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
        #
        # that appears to be case when the size of a non-contiguous
        # tensor argument is larger than a certain threshold. Could
        # this be related to shared memory or L1 cache size of a GPU
        # card? In anycase, ensuring that tensor arguments are
        # contiguous seems to avoid the above exception. So, in the
        # following we'll always convert tensor arguments to
        # C-contiguous tensors.

        if force_contiguous:
            blocks = blocks.contiguous()
            others = others.contiguous()
            if not accumulators.is_contiguous():
                accumulators_ = accumulators.contiguous()
            else:
                accumulators_ = accumulators
        else:
            accumulators_ = accumulators

        _scatter_mm6_kernel[grid](
            B,
            Ms,
            Ks,
            N,
            blocks,
            blocks.stride(0),
            blocks.stride(1),
            blocks.stride(2),
            others,
            others.stride(0),
            others.stride(1),
            others.stride(2),
            accumulators_,
            accumulators_.stride(0),
            accumulators_.stride(1),
            accumulators_.stride(2),
            c_indices,
            r_offsets,
            p_offsets,
            q_offsets,
            dot_out_dtype=dot_out_dtype,
            **meta,
        )

        if force_contiguous and not accumulators.is_contiguous():
            accumulators.copy_(accumulators_)

    @triton.jit
    def _bsr_strided_addmm_kernel(
        # values prologue
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        # col_indices epilogue
        # input prologue
        input_ptr,
        input_batch_stride,
        input_tiled_row_stride,
        input_tiled_col_stride,
        input_row_block_stride,
        input_col_block_stride,
        # input epilogue
        # dense prologue
        dense_ptr,
        dense_batch_stride,
        dense_tiled_row_stride,
        dense_tiled_col_stride,
        dense_row_block_stride,
        dense_col_block_stride,
        # dense epilogue
        # left_alpha prologue
        left_alpha_ptr,
        left_alpha_batch_stride,
        left_alpha_tiled_row_stride,
        left_alpha_tiled_col_stride: tl.constexpr,
        left_alpha_row_block_stride,
        left_alpha_col_block_stride: tl.constexpr,
        # left_alpha epilogue
        # right_alpha prologue
        right_alpha_ptr,
        right_alpha_batch_stride,
        right_alpha_tiled_row_stride: tl.constexpr,
        right_alpha_tiled_col_stride,
        right_alpha_row_block_stride: tl.constexpr,
        right_alpha_col_block_stride,
        # right_alpha epilogue
        # output prologue
        output_ptr,
        output_batch_stride,
        output_tiled_row_stride,
        output_tiled_col_stride,
        output_row_block_stride,
        output_col_block_stride,
        # output epilogue
        beta,
        alpha,
        beta_is_one: tl.constexpr,
        beta_is_nonzero: tl.constexpr,
        alpha_is_one: tl.constexpr,
        left_alpha_is_one: tl.constexpr,
        right_alpha_is_one: tl.constexpr,
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        BLOCKSIZE_INNER: tl.constexpr,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
        GROUP_SIZE_ROW: tl.constexpr,
        SPLIT_N: tl.constexpr,
    ):
        # left/right_alpha tensors are originally (* + 1)-dimensional
        if left_alpha_tiled_col_stride != 0:
            raise AssertionError(
                f"left_alpha_tiled_col_stride must be 0, got {left_alpha_tiled_col_stride}"
            )
        if left_alpha_col_block_stride != 0:
            raise AssertionError(
                f"left_alpha_col_block_stride must be 0, got {left_alpha_col_block_stride}"
            )
        if right_alpha_tiled_row_stride != 0:
            raise AssertionError(
                f"right_alpha_tiled_row_stride must be 0, got {right_alpha_tiled_row_stride}"
            )
        if right_alpha_row_block_stride != 0:
            raise AssertionError(
                f"right_alpha_row_block_stride must be 0, got {right_alpha_row_block_stride}"
            )

        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)

        row_block_pid, col_block_pid = tl.swizzle2d(
            row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
        )

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        row_nnz = nnz_offset_next - nnz_offset

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        inner_block_arange = tl.arange(0, BLOCKSIZE_INNER)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * inner_block_arange[None, :]
        )

        # NOTE: dense is advanced into all dimensions but the tiled row one.
        # That will be advanced in the loop according to values in col_indices.
        dense_block_ptrs = (
            dense_ptr
            + dense_batch_stride * batch_pid
            + dense_tiled_col_stride * col_block_pid
            + dense_row_block_stride * inner_block_arange[:, None]
            + dense_col_block_stride * col_block_arange[None, :]
        )

        # Pointers are set to exact write-to locations
        output_ptrs = (
            output_ptr
            + output_batch_stride * batch_pid
            + output_tiled_row_stride * row_block_pid
            + output_tiled_col_stride * col_block_pid
            + output_row_block_stride * row_block_arange[:, None]
            + output_col_block_stride * col_block_arange[None, :]
        )

        # Set pointer to the first nonzero element in the current row
        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs)

            # find which row of dense needs to get loaded
            # for multiplication with values_block.
            dense_row_idx = tl.load(col_index_nnz_ptr)
            dense_block = tl.load(
                dense_block_ptrs + dense_tiled_row_stride * dense_row_idx
            )

            # do block mm
            output_acc_block += tl.dot(
                values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
            )

            # move val/col_index ptrs to the next block in the row
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

        if not alpha_is_one:
            output_acc_block *= alpha

        if not left_alpha_is_one:
            left_alpha_ptrs = (
                left_alpha_ptr
                + left_alpha_batch_stride * batch_pid
                + left_alpha_tiled_row_stride * row_block_pid
                + left_alpha_tiled_col_stride * col_block_pid
                + left_alpha_row_block_stride * row_block_arange[:, None]
                + left_alpha_col_block_stride * col_block_arange[None, :]
            )
            output_acc_block *= tl.load(left_alpha_ptrs)

        if not right_alpha_is_one:
            right_alpha_ptrs = (
                right_alpha_ptr
                + right_alpha_batch_stride * batch_pid
                + right_alpha_tiled_row_stride * row_block_pid
                + right_alpha_tiled_col_stride * col_block_pid
                + right_alpha_row_block_stride * row_block_arange[:, None]
                + right_alpha_col_block_stride * col_block_arange[None, :]
            )
            output_acc_block *= tl.load(right_alpha_ptrs)

        if beta_is_nonzero:
            input_ptrs = (
                input_ptr
                + input_batch_stride * batch_pid
                + input_tiled_row_stride * row_block_pid
                + input_tiled_col_stride * col_block_pid
                + input_row_block_stride * row_block_arange[:, None]
                + input_col_block_stride * col_block_arange[None, :]
            )
            if beta_is_one:
                output_acc_block += tl.load(input_ptrs)
            else:
                output_acc_block += beta * tl.load(input_ptrs)

        # write back the result
        tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))

else:
    bsr_softmax = None  # type: ignore[assignment]
    bsr_dense_mm = None  # type: ignore[assignment]
    sampled_addmm = None  # type: ignore[assignment]
    _scaled_dot_product_attention = None  # type: ignore[assignment]
    _scatter_mm2 = None  # type: ignore[assignment]
    _scatter_mm6 = None  # type: ignore[assignment]
    _bsr_strided_addmm_kernel = None  # type: ignore[assignment]
<|endoftext|># mypy: allow-untyped-defs
# The Tensor classes are added to this module by python_tensor.cpp
# A workaround to support both TorchScript and MyPy:
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch._C import _add_docstr, _sparse  # type: ignore[attr-defined]

# Semi structured sparsity support
from .semi_structured import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured,
)


if TYPE_CHECKING:
    from torch.types import _dtype as DType

    DimOrDims = int | tuple[int, ...] | list[int] | None
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
    DimOrDims = tuple[int] | None


__all__ = [
    "addmm",
    "check_sparse_tensor_invariants",
    "mm",
    "sum",
    "softmax",
    # pyrefly: ignore [bad-dunder-all]
    "solve",
    "log_softmax",
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
    "as_sparse_gradcheck",
]

addmm = _add_docstr(
    _sparse._sparse_addmm,
    r"""
sparse.addmm(mat, mat1, mat2, *, beta=1., alpha=1.) -> Tensor

This function does exact same thing as :func:`torch.addmm` in the forward,
except that it supports backward for sparse COO and CSR matrix :attr:`mat1`.
When :attr:`mat1` is a COO tensor it must have `sparse_dim = 2`.

Supports both CSR and COO storage formats.

.. note::
    **Gradient support:**

    - **COO @ Dense**: Backward is supported for both inputs. The gradient for the
      sparse input is returned as a sparse COO tensor.
    - **CSR @ Dense**: Backward is supported for both inputs. The gradient for the
      sparse input is returned as a sparse CSR tensor.
    - **CSC/BSR/BSC @ Dense**: Not supported.
    - **Sparse @ Sparse** (COO @ COO, CSR @ CSR): Forward works, but backward is
      not supported.

Args:
    mat (Tensor): a dense matrix to be added
    mat1 (Tensor): a sparse matrix to be multiplied
    mat2 (Tensor): a dense matrix to be multiplied
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
""",
)


mm = _add_docstr(
    _sparse._sparse_mm,
    r"""
    Performs a matrix multiplication of the sparse matrix :attr:`mat1`
    and the (sparse or strided) matrix :attr:`mat2`. Similar to :func:`torch.mm`, if :attr:`mat1` is a
    :math:`(n \times m)` tensor, :attr:`mat2` is a :math:`(m \times p)` tensor, out will be a
    :math:`(n \times p)` tensor.
    When :attr:`mat1` is a COO tensor it must have `sparse_dim = 2`.

    Supports both CSR and COO storage formats.

.. note::
    **Gradient support:**

    - **COO @ Dense**: Backward is supported for both inputs. The gradient for the
      sparse input is returned as a sparse COO tensor.
    - **CSR @ Dense**: Backward is supported for both inputs. The gradient for the
      sparse input is returned as a sparse CSR tensor.
    - **CSC/BSR/BSC @ Dense**: Not supported.
    - **Sparse @ Sparse** (COO @ COO, CSR @ CSR): Forward works, but backward is
      not supported.
    - **Mixed formats** (COO @ CSR, CSR @ COO): Not supported.

    This function also additionally accepts an optional :attr:`reduce` argument that allows
    specification of an optional reduction operation, mathematically performs the following operation:

.. math::

    z_{ij} = \bigoplus_{k = 0}^{K - 1} x_{ik} y_{kj}

where :math:`\bigoplus` defines the reduce operator. :attr:`reduce` is implemented only for
CSR storage format on CPU device.

Args:
    mat1 (Tensor): the first sparse matrix to be multiplied
    mat2 (Tensor): the second matrix to be multiplied, which could be sparse or dense
    reduce (str, optional): the reduction operation to apply for non-unique indices
        (:obj:`"sum"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`). Default :obj:`"sum"`.

Shape:
    The format of the output tensor of this function follows:
    - sparse x sparse -> sparse
    - sparse x dense -> dense

Example::

    >>> a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
    >>> a
    tensor(indices=tensor([[0, 0, 1],
                           [0, 2, 1]]),
           values=tensor([1., 2., 3.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo, requires_grad=True)
    >>> b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
    >>> b
    tensor([[0., 1.],
            [2., 0.],
            [0., 0.]], requires_grad=True)
    >>> y = torch.sparse.mm(a, b)
    >>> y
    tensor([[0., 1.],
            [6., 0.]], grad_fn=<SparseAddmmBackward0>)
    >>> y.sum().backward()
    >>> a.grad
    tensor(indices=tensor([[0, 0, 1],
                           [0, 2, 1]]),
           values=tensor([1., 0., 2.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> c = a.detach().to_sparse_csr()
    >>> c
    tensor(crow_indices=tensor([0, 2, 3]),
           col_indices=tensor([0, 2, 1]),
           values=tensor([1., 2., 3.]), size=(2, 3), nnz=3,
           layout=torch.sparse_csr)
    >>> y1 = torch.sparse.mm(c, b, 'sum')
    >>> y1
    tensor([[0., 1.],
            [6., 0.]], grad_fn=<SparseMmReduceImplBackward0>)
    >>> y2 = torch.sparse.mm(c, b, 'max')
    >>> y2
    tensor([[0., 1.],
            [6., 0.]], grad_fn=<SparseMmReduceImplBackward0>)
""",
)


sampled_addmm = _add_docstr(
    _sparse.sparse_sampled_addmm,
    r"""
sparse.sampled_addmm(input, mat1, mat2, *, beta=1., alpha=1., out=None) -> Tensor

Performs a matrix multiplication of the dense matrices :attr:`mat1` and :attr:`mat2` at the locations
specified by the sparsity pattern of :attr:`input`. The matrix :attr:`input` is added to the final result.

Mathematically this performs the following operation:

.. math::

    \text{out} = \alpha\ (\text{mat1} \mathbin{@} \text{mat2})*\text{spy}(\text{input}) + \beta\ \text{input}

where :math:`\text{spy}(\text{input})` is the sparsity pattern matrix of :attr:`input`, :attr:`alpha`
and :attr:`beta` are the scaling factors.
:math:`\text{spy}(\text{input})` has value 1 at the positions where :attr:`input` has non-zero values, and 0 elsewhere.

.. note::
    :attr:`input` must be a sparse CSR tensor. :attr:`mat1` and :attr:`mat2` must be dense tensors.

Args:
    input (Tensor): a sparse CSR matrix of shape `(m, n)` to be added and used to compute
        the sampled matrix multiplication
    mat1 (Tensor): a dense matrix of shape `(m, k)` to be multiplied
    mat2 (Tensor): a dense matrix of shape `(k, n)` to be multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> input = torch.eye(3, device='cuda').to_sparse_csr()
    >>> mat1 = torch.randn(3, 5, device='cuda')
    >>> mat2 = torch.randn(5, 3, device='cuda')
    >>> torch.sparse.sampled_addmm(input, mat1, mat2)
    tensor(crow_indices=tensor([0, 1, 2, 3]),
        col_indices=tensor([0, 1, 2]),
        values=tensor([ 0.2847, -0.7805, -0.1900]), device='cuda:0',
        size=(3, 3), nnz=3, layout=torch.sparse_csr)
    >>> torch.sparse.sampled_addmm(input, mat1, mat2).to_dense()
    tensor([[ 0.2847,  0.0000,  0.0000],
        [ 0.0000, -0.7805,  0.0000],
        [ 0.0000,  0.0000, -0.1900]], device='cuda:0')
    >>> torch.sparse.sampled_addmm(input, mat1, mat2, beta=0.5, alpha=0.5)
    tensor(crow_indices=tensor([0, 1, 2, 3]),
        col_indices=tensor([0, 1, 2]),
        values=tensor([ 0.1423, -0.3903, -0.0950]), device='cuda:0',
        size=(3, 3), nnz=3, layout=torch.sparse_csr)
""",
)


def sum(input: Tensor, dim: DimOrDims = None, dtype: DType | None = None) -> Tensor:
    r"""Return the sum of each row of the given sparse tensor.

    Returns the sum of each row of the sparse tensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a dense tensor instead of a sparse tensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr:`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input sparse tensor
        dim (int or tuple of ints): a dimension or a list of dimensions to reduce. Default: reduce
            over all dims.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: dtype of :attr:`input`.

    Example::

        >>> nnz = 3
        >>> dims = [5, 5, 2, 3]
        >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                           torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
        >>> V = torch.randn(nnz, dims[2], dims[3])
        >>> size = torch.Size(dims)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> with torch.sparse.check_sparse_tensor_invariants():
        ...     S = torch.sparse_coo_tensor(I, V, size)
        >>> S
        tensor(indices=tensor([[2, 0, 3],
                               [2, 4, 1]]),
               values=tensor([[[-0.6438, -1.6467,  1.4004],
                               [ 0.3411,  0.0918, -0.2312]],

                              [[ 0.5348,  0.0634, -2.0494],
                               [-0.7125, -1.0646,  2.1844]],

                              [[ 0.1276,  0.1874, -0.6334],
                               [-1.9682, -0.5340,  0.7483]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        # when sum over only part of sparse_dims, return a sparse tensor
        >>> torch.sparse.sum(S, [1, 3])
        tensor(indices=tensor([[0, 2, 3]]),
               values=tensor([[-1.4512,  0.4073],
                              [-0.8901,  0.2017],
                              [-0.3183, -1.7539]]),
               size=(5, 2), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim, return a dense tensor
        # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([-2.6596, -1.1450])
    """
    if dtype is None:
        if dim is not None:
            return torch._sparse_sum(input, dim)
        else:
            return torch._sparse_sum(input)
    else:
        if dim is not None:
            return torch._sparse_sum(input, dim, dtype=dtype)
        else:
            return torch._sparse_sum(input, dtype=dtype)


softmax = _add_docstr(
    _sparse._sparse_softmax,
    r"""
sparse.softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

where :math:`i, j` run over sparse tensor indices and unspecified
entries are ignores. This is equivalent to defining unspecified
entries as negative infinity so that :math:`exp(x_k) = 0` when the
entry with index :math:`k` has not specified.

It is applied to all slices along `dim`, and will re-scale them so
that the elements lie in the range `[0, 1]` and sum to 1.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type
        of returned tensor.  If specified, the input tensor is
        casted to :attr:`dtype` before the operation is
        performed. This is useful for preventing data type
        overflows. Default: None
""",
)


spsolve = _add_docstr(
    _sparse._spsolve,
    r"""
sparse.spsolve(input, other, *, left=True) -> Tensor

Computes the solution of a square system of linear equations with
a unique solution. Its purpose is similar to :func:`torch.linalg.solve`,
except that the system is defined by a sparse CSR matrix with layout
`sparse_csr`.

Args:
    input (Tensor): a sparse CSR matrix of shape `(n, n)` representing the
        coefficients of the linear system.
    other (Tensor): a dense matrix of shape `(n, )` representing the right-hand
        side of the linear system.
    left (bool, optional): whether to solve the system for `input @ out = other`
        (default) or `out @ input = other`. Only `left=True` is supported.
""",
)

log_softmax = _add_docstr(
    _sparse._sparse_log_softmax,
    r"""
sparse.log_softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function followed by logarithm.

See :class:`~torch.sparse.softmax` for more details.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type
        of returned tensor.  If specified, the input tensor is
        casted to :attr:`dtype` before the operation is
        performed. This is useful for preventing data type
        overflows. Default: None
""",
)


spdiags = _add_docstr(
    _sparse._spdiags,
    r"""
sparse.spdiags(diagonals, offsets, shape, layout=None) -> Tensor

Creates a sparse 2D tensor by placing the values from rows of
:attr:`diagonals` along specified diagonals of the output

The :attr:`offsets` tensor controls which diagonals are set.

- If :attr:`offsets[i]` = 0, it is the main diagonal
- If :attr:`offsets[i]` < 0, it is below the main diagonal
- If :attr:`offsets[i]` > 0, it is above the main diagonal

The number of rows in :attr:`diagonals` must match the length of :attr:`offsets`,
and an offset may not be repeated.

Args:
    diagonals (Tensor): Matrix storing diagonals row-wise
    offsets (Tensor): The diagonals to be set, stored as a vector
    shape (2-tuple of ints): The desired shape of the result
Keyword args:
    layout (:class:`torch.layout`, optional): The desired layout of the
        returned tensor. ``torch.sparse_coo``, ``torch.sparse_csc`` and ``torch.sparse_csr``
        are supported. Default: ``torch.sparse_coo``

Examples:

Set the main and first two lower diagonals of a matrix::

    >>> diags = torch.arange(9).reshape(3, 3)
    >>> diags
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    >>> s = torch.sparse.spdiags(diags, torch.tensor([0, -1, -2]), (3, 3))
    >>> s
    tensor(indices=tensor([[0, 1, 2, 1, 2, 2],
                           [0, 1, 2, 0, 1, 0]]),
           values=tensor([0, 1, 2, 3, 4, 6]),
           size=(3, 3), nnz=6, layout=torch.sparse_coo)
    >>> s.to_dense()
    tensor([[0, 0, 0],
            [3, 1, 0],
            [6, 4, 2]])


Change the output layout::

    >>> diags = torch.arange(9).reshape(3, 3)
    >>> diags
    tensor([[0, 1, 2],[3, 4, 5], [6, 7, 8])
    >>> s = torch.sparse.spdiags(diags, torch.tensor([0, -1, -2]), (3, 3), layout=torch.sparse_csr)
    >>> s
    tensor(crow_indices=tensor([0, 1, 3, 6]),
           col_indices=tensor([0, 0, 1, 0, 1, 2]),
           values=tensor([0, 3, 1, 6, 4, 2]), size=(3, 3), nnz=6,
           layout=torch.sparse_csr)
    >>> s.to_dense()
    tensor([[0, 0, 0],
            [3, 1, 0],
            [6, 4, 2]])

Set partial diagonals of a large output::

    >>> diags = torch.tensor([[1, 2], [3, 4]])
    >>> offsets = torch.tensor([0, -1])
    >>> torch.sparse.spdiags(diags, offsets, (5, 5)).to_dense()
    tensor([[1, 0, 0, 0, 0],
            [3, 2, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])

.. note::

    When setting the values along a given diagonal the index into the diagonal
    and the index into the row of :attr:`diagonals` is taken as the
    column index in the output. This has the effect that when setting a diagonal
    with a positive offset `k` the first value along that diagonal will be
    the value in position `k` of the row of :attr:`diagonals`

Specifying a positive offset::

    >>> diags = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    >>> torch.sparse.spdiags(diags, torch.tensor([0, 1, 2]), (5, 5)).to_dense()
    tensor([[1, 2, 3, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])
""",
)


class check_sparse_tensor_invariants:
    """A tool to control checking sparse tensor invariants.

    The following options exists to manage sparse tensor invariants
    checking in sparse tensor construction:

    1. Using a context manager:

       .. code:: python

           with torch.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. Using a procedural approach:

       .. code:: python

           prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
           torch.sparse.check_sparse_tensor_invariants.enable()

           run_my_model()

           if not prev_checks_enabled:
               torch.sparse.check_sparse_tensor_invariants.disable()

    3. Using function decoration:

       .. code:: python

           @torch.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

           run_my_model()

    4. Using ``check_invariants`` keyword argument in sparse tensor constructor call.
       For example:

       >>> torch.sparse_csr_tensor([0, 1, 3], [0, 1], [1, 2], check_invariants=True)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
       RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
    """

    @staticmethod
    def is_enabled():
        r"""Return True if the sparse tensor invariants checking is enabled.

        .. note::

            Use :func:`torch.sparse.check_sparse_tensor_invariants.enable` or
            :func:`torch.sparse.check_sparse_tensor_invariants.disable` to
            manage the state of the sparse tensor invariants checks.
        """
        return torch._C._check_sparse_tensor_invariants()

    @staticmethod
    def enable():
        r"""Enable sparse tensor invariants checking in sparse tensor constructors.

        .. note::

            By default, the sparse tensor invariants checks are disabled. Use
            :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled` to
            retrieve the current state of sparse tensor invariants checking.

        .. note::

            The sparse tensor invariants check flag is effective to all sparse
            tensor constructors, both in Python and ATen.

        The flag can be locally overridden by the ``check_invariants``
        optional argument of the sparse tensor constructor functions.
        """
        torch._C._set_check_sparse_tensor_invariants(True)

    @staticmethod
    def disable():
        r"""Disable sparse tensor invariants checking in sparse tensor constructors.

        See :func:`torch.sparse.check_sparse_tensor_invariants.enable` for more information.
        """
        torch._C._set_check_sparse_tensor_invariants(False)

    # context manager support
    def __init__(self, enable=True):
        self.state = enable
        self.saved_state: bool | None = None

    def __enter__(self):
        if self.saved_state is not None:
            raise RuntimeError(
                "This context manager instance is already activated."
                " Use a different context manager instance for context nesting."
            )
        self.saved_state = self.is_enabled()
        torch._C._set_check_sparse_tensor_invariants(self.state)

    def __exit__(self, type, value, traceback):
        if self.saved_state is None:
            raise AssertionError("saved_state should not be None on exit")
        torch._C._set_check_sparse_tensor_invariants(self.saved_state)
        self.saved_state = None

    # decorator support
    def __call__(self, mth):
        def test_mth(*args, **kwargs):
            with type(self)(self.state):
                return mth(*args, **kwargs)

        return test_mth


def as_sparse_gradcheck(gradcheck):
    """Decorate function, to extend gradcheck for sparse tensors.

    Decorator for torch.autograd.gradcheck or its functools.partial
    variants that extends the gradcheck function with support to input
    functions that operate on or/and return sparse tensors.

    The specified gradcheck function itself is guaranteed to operate
    on strided tensors only.

    For example:

    >>> warnings.filterwarnings(
    ...     "ignore", message=".*Sparse CSR tensor support is in beta state"
    ... )  # docs: hide
    >>> gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
    >>> x = (
    ...     torch.tensor([[0, 1], [2, 3]], dtype=torch.float64)
    ...     .to_sparse_coo()
    ...     .requires_grad_(True)
    ... )
    >>> gradcheck(lambda x: x.to_sparse_csr(), x)
    True
    """

    def gradcheck_with_sparse_support(func, inputs, **kwargs):
        """
        Create gradcheck with support for sparse tensors.

        Same as :func:`torch.autograd.gradcheck` but with sparse tensors inputs and outputs support.
        """
        masked = kwargs.pop("masked", False)
        sparse_layouts = {
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        sparse_compressed_layouts = {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        sparse_block_layouts = {torch.sparse_bsr, torch.sparse_bsc}
        STRIDED_REPRESENTATION = "__STRIDED_REPRESENTATION__"

        def convert_to_strided_representation(args):
            """Convert differentiable non-strided tensors to a representation containing differentiable strided tensors."""
            if not isinstance(args, (list, tuple)):
                args = (args,)
            new_args: list[Any] = []
            for obj in args:
                if (
                    isinstance(obj, torch.Tensor)
                    and obj.requires_grad
                    and obj.layout in sparse_layouts
                ):
                    d = {
                        "layout": obj.layout,
                        "shape": obj.shape,
                    }
                    if not masked:
                        # Materialize unspecified elements with zero values
                        batch_dim = obj.ndim - obj.dense_dim() - obj.sparse_dim()
                        blocksize = (
                            obj.values().shape[batch_dim + 1 : batch_dim + 3]
                            if obj.layout in sparse_block_layouts
                            else None
                        )
                        full_mask = torch.ones(
                            obj.shape, device=obj.device, dtype=torch.bool
                        ).to_sparse(
                            layout=obj.layout,
                            blocksize=blocksize,
                            dense_dim=obj.dense_dim(),
                        )
                        obj = obj.to_dense().sparse_mask(full_mask)
                    if obj.layout is torch.sparse_coo:
                        # pyrefly: ignore [no-matching-overload]
                        d.update(
                            # pyrefly: ignore [bad-argument-type]
                            indices=obj._indices(),
                            # pyrefly: ignore [bad-argument-type]
                            is_coalesced=obj.is_coalesced(),
                        )
                        values = obj._values()
                    elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        # pyrefly: ignore [no-matching-overload]
                        d.update(
                            # pyrefly: ignore [bad-argument-type]
                            compressed_indices=obj.crow_indices(),
                            # pyrefly: ignore [bad-argument-type]
                            plain_indices=obj.col_indices(),
                        )
                        values = obj.values()
                    else:
                        # pyrefly: ignore [no-matching-overload]
                        d.update(
                            # pyrefly: ignore [bad-argument-type]
                            compressed_indices=obj.ccol_indices(),
                            # pyrefly: ignore [bad-argument-type]
                            plain_indices=obj.row_indices(),
                        )
                        values = obj.values()
                    new_args.extend(
                        (STRIDED_REPRESENTATION, d, values.requires_grad_(True))
                    )
                else:
                    new_args.append(obj)
            return tuple(new_args)

        def restore_from_strided_representation(args):
            """Restore non-strided differentiable tensors from their strided representations."""
            new_args = []
            args = list(args)
            while args:
                a = args.pop(0)
                if a == STRIDED_REPRESENTATION:
                    d, values = args.pop(0), args.pop(0)
                    if d["layout"] is torch.sparse_coo:
                        a = torch.sparse_coo_tensor(
                            d["indices"],
                            values,
                            size=d["shape"],
                            is_coalesced=d["is_coalesced"],
                        )
                    elif d["layout"] in sparse_compressed_layouts:
                        a = torch.sparse_compressed_tensor(
                            d["compressed_indices"],
                            d["plain_indices"],
                            values,
                            size=d["shape"],
                            layout=d["layout"],
                        )
                    else:
                        raise NotImplementedError(
                            f"conversion of {d['layout']} strided representation to tensor"
                        )
                new_args.append(a)
            return tuple(new_args)

        def func_wrapper(*args, **kwargs):
            restored_args = restore_from_strided_representation(args)

            # convert differentiable output sparse tensors to strided
            # tensors:
            outputs = func(*restored_args, **kwargs)

            strided_outputs = (
                tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
            )
            strided_outputs = tuple(
                (
                    o.to_dense(masked_grad=masked)
                    if isinstance(o, torch.Tensor)
                    and o.requires_grad
                    and o.layout in sparse_layouts
                    else o
                )
                for o in strided_outputs
            )

            return (
                strided_outputs
                if isinstance(outputs, (list, tuple))
                else strided_outputs[0]
            )

        args = (func_wrapper, convert_to_strided_representation(inputs))

        return gradcheck(*args, **kwargs)

    return gradcheck_with_sparse_support
<|endoftext|># mypy: allow-untyped-defs
import contextlib

import torch


__all__ = [
    "fallback_dispatcher",
    "semi_sparse_values",
    "semi_sparse_indices",
    "semi_sparse_t",
    "semi_sparse_view",
    "semi_sparse_detach",
    "semi_sparse_mm",
    "semi_sparse_addmm",
    "semi_sparse_linear",
    "semi_sparse_scaled_mm",
    "semi_sparse_clone",
    "semi_sparse_to",
    "semi_sparse_to_copy",
]


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def fallback_dispatcher(func, types, args, kwargs):
    with no_dispatch():
        return func(*args)


def semi_sparse_values(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    A = args[0]
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        return A.packed.ravel()[:num_kept_elements:].view(m, -1)
    else:
        return A.packed.detach()


def semi_sparse_indices(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    A = args[0]
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        metadata = A.packed.ravel()[num_kept_elements:].view(m, -1)
        return metadata.view(torch.int32 if A.dtype == torch.int32 else torch.int16)
    else:
        return A.meta


def semi_sparse_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    self = args[0]
    if not isinstance(self, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(self).__name__}"
        )
    if len(self.shape) != 2:
        raise AssertionError(f"expected 2D tensor, got {len(self.shape)}D")
    # Because we cannot go from the compressed representation back to the dense representation currently,
    # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
    # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
    # pyrefly: ignore [no-matching-overload]
    return self.__class__(
        torch.Size([self.shape[-1], self.shape[0]]),
        packed=self.packed_t,
        meta=self.meta_t,
        packed_t=self.packed,
        meta_t=self.meta,
        compressed_swizzled_bitmask=(
            self.compressed_swizzled_bitmask.transpose(0, 1)
