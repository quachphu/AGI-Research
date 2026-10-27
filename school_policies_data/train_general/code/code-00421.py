        # If the other is not an Expression, and neither a Tuple
        if not isinstance(other, (Expr, Tuple)):
            return S.false

        # self in rectangular form
        if not self.polar:
            re, im = other if isTuple else other.as_real_imag()
            return tfn[fuzzy_or(fuzzy_and([
                pset.args[0]._contains(re),
                pset.args[1]._contains(im)])
                for pset in self.psets)]

        # self in polar form
        elif self.polar:
            if other.is_zero:
                # ignore undefined complex argument
                return tfn[fuzzy_or(pset.args[0]._contains(S.Zero)
                    for pset in self.psets)]
            if isTuple:
                r, theta = other
            else:
                r, theta = Abs(other), arg(other)
            if theta.is_real and theta.is_number:
                # angles in psets are normalized to [0, 2pi)
                theta %= 2*S.Pi
                return tfn[fuzzy_or(fuzzy_and([
                    pset.args[0]._contains(r),
                    pset.args[1]._contains(theta)])
                    for pset in self.psets)]


class CartesianComplexRegion(ComplexRegion):
    r"""
    Set representing a square region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, I, Interval
    >>> region = ComplexRegion(Interval(1, 3) * Interval(4, 6))
    >>> 2 + 5*I in region
    True
    >>> 5*I in region
    False

    See also
    ========

    ComplexRegion
    PolarComplexRegion
    Complexes
    """

    polar = False
    variables = symbols('x, y', cls=Dummy)

    def __new__(cls, sets):

        if sets == S.Reals*S.Reals:
            return S.Complexes

        if all(_a.is_FiniteSet for _a in sets.args) and (len(sets.args) == 2):

            # ** ProductSet of FiniteSets in the Complex Plane. **
            # For Cases like ComplexRegion({2, 4}*{3}), It
            # would return {2 + 3*I, 4 + 3*I}

            # FIXME: This should probably be handled with something like:
            # return ImageSet(Lambda((x, y), x+I*y), sets).rewrite(FiniteSet)
            complex_num = []
            for x in sets.args[0]:
                for y in sets.args[1]:
                    complex_num.append(x + S.ImaginaryUnit*y)
            return FiniteSet(*complex_num)
        else:
            return Set.__new__(cls, sets)

    @property
    def expr(self):
        x, y = self.variables
        return x + S.ImaginaryUnit*y


class PolarComplexRegion(ComplexRegion):
    r"""
    Set representing a polar region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, oo, pi, I
    >>> rset = Interval(0, oo)
    >>> thetaset = Interval(0, pi)
    >>> upper_half_plane = ComplexRegion(rset * thetaset, polar=True)
    >>> 1 + I in upper_half_plane
    True
    >>> 1 - I in upper_half_plane
    False

    See also
    ========

    ComplexRegion
    CartesianComplexRegion
    Complexes

    """

    polar = True
    variables = symbols('r, theta', cls=Dummy)

    def __new__(cls, sets):

        new_sets = []
        # sets is Union of ProductSets
        if not sets.is_ProductSet:
            for k in sets.args:
                new_sets.append(k)
        # sets is ProductSets
        else:
            new_sets.append(sets)
        # Normalize input theta
        for k, v in enumerate(new_sets):
            new_sets[k] = ProductSet(v.args[0],
                                     normalize_theta_set(v.args[1]))
        sets = Union(*new_sets)
        return Set.__new__(cls, sets)

    @property
    def expr(self):
        r, theta = self.variables
        return r*(cos(theta) + S.ImaginaryUnit*sin(theta))


class Complexes(CartesianComplexRegion, metaclass=Singleton):
    """
    The :class:`Set` of all complex numbers

    Examples
    ========

    >>> from sympy import S, I
    >>> S.Complexes
    Complexes
    >>> 1 + I in S.Complexes
    True

    See also
    ========

    Reals
    ComplexRegion

    """

    is_empty = False
    is_finite_set = False

    # Override property from superclass since Complexes has no args
    @property
    def sets(self):
        return ProductSet(S.Reals, S.Reals)

    def __new__(cls):
        return Set.__new__(cls)
<|endoftext|>from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING, overload
from functools import reduce
from collections import defaultdict
from collections.abc import Mapping, Iterable
import inspect

from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
    fuzzy_not)
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
                                       subsets)
from sympy.utilities.misc import func_name, filldedent

from mpmath import mpi, mpf

from mpmath.libmp.libmpf import prec_to_dps


tfn = defaultdict(lambda: None, {
    True: S.true,
    S.true: S.true,
    False: S.false,
    S.false: S.false})


@sympify_method_args
class Set(Basic, EvalfMixin):
    """
    The base class for any kind of set.

    Explanation
    ===========

    This is not meant to be used directly as a container of items. It does not
    behave like the builtin ``set``; see :class:`FiniteSet` for that.

    Real intervals are represented by the :class:`Interval` class and unions of
    sets by the :class:`Union` class. The empty set is represented by the
    :class:`EmptySet` class and available as a singleton as ``S.EmptySet``.
    """

    __slots__: tuple[()] = ()

    is_number = False
    is_iterable = False
    is_interval = False

    is_FiniteSet = False
    is_Interval = False
    is_ProductSet = False
    is_Union = False
    is_Intersection: FuzzyBool = None
    is_UniversalSet: FuzzyBool = None
    is_Complement: FuzzyBool = None
    is_ComplexRegion = False

    is_empty: FuzzyBool = None
    is_finite_set: FuzzyBool = None

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return None

    if TYPE_CHECKING:

        def __new__(cls, *args: Basic | complex) -> Set:
            ...

        @overload # type: ignore
        def subs(self, arg1: Mapping[Basic | complex, Set | complex], arg2: None=None) -> Set: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Set | complex]], arg2: None=None, **kwargs: Any) -> Set: ...
        @overload
        def subs(self, arg1: Set | complex, arg2: Set | complex) -> Set: ...
        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, # type: ignore
                 arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Set:
            assert False

        def evalf(self, n: int = 15, subs: dict[Basic, Basic | float] | None = None,
                  maxn: int = 100, chop: bool = False, strict: bool  = False,
                  quad: str | None = None, verbose: bool = False) -> Set:
            ...

        n = evalf

    @staticmethod
    def _infimum_key(expr):
        """
        Return infimum (if possible) else S.Infinity.
        """
        try:
            infimum = expr.inf
            assert infimum.is_comparable
            infimum = infimum.evalf()  # issue #18505
        except (NotImplementedError,
                AttributeError, AssertionError, ValueError):
            infimum = S.Infinity
        return infimum

    def union(self, other):
        """
        Returns the union of ``self`` and ``other``.

        Examples
        ========

        As a shortcut it is possible to use the ``+`` operator:

        >>> from sympy import Interval, FiniteSet
        >>> Interval(0, 1).union(Interval(2, 3))
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(0, 1) + Interval(2, 3)
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
        Union({3}, Interval.Lopen(1, 2))

        Similarly it is possible to use the ``-`` operator for set differences:

        >>> Interval(0, 2) - Interval(0, 1)
        Interval.Lopen(1, 2)
        >>> Interval(1, 3) - FiniteSet(2)
        Union(Interval.Ropen(1, 2), Interval.Lopen(2, 3))

        """
        return Union(self, other)

    def intersect(self, other):
        """
        Returns the intersection of 'self' and 'other'.

        Examples
        ========

        >>> from sympy import Interval

        >>> Interval(1, 3).intersect(Interval(1, 2))
        Interval(1, 2)

        >>> from sympy import imageset, Lambda, symbols, S
        >>> n, m = symbols('n m')
        >>> a = imageset(Lambda(n, 2*n), S.Integers)
        >>> a.intersect(imageset(Lambda(m, 2*m + 1), S.Integers))
        EmptySet

        """
        return Intersection(self, other)

    def intersection(self, other):
        """
        Alias for :meth:`intersect()`
        """
        return self.intersect(other)

    def is_disjoint(self, other):
        """
        Returns True if ``self`` and ``other`` are disjoint.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 2).is_disjoint(Interval(1, 2))
        False
        >>> Interval(0, 2).is_disjoint(Interval(3, 4))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Disjoint_sets
        """
        return self.intersect(other) == S.EmptySet

    def isdisjoint(self, other):
        """
        Alias for :meth:`is_disjoint()`
        """
        return self.is_disjoint(other)

    def complement(self, universe):
        r"""
        The complement of 'self' w.r.t the given universe.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(0, 1).complement(S.Reals)
        Union(Interval.open(-oo, 0), Interval.open(1, oo))

        >>> Interval(0, 1).complement(S.UniversalSet)
        Complement(UniversalSet, Interval(0, 1))

        """
        return Complement(universe, self)

    def _complement(self, other):
        # this behaves as other - self
        if isinstance(self, ProductSet) and isinstance(other, ProductSet):
            # If self and other are disjoint then other - self == self
            if len(self.sets) != len(other.sets):
                return other

            # There can be other ways to represent this but this gives:
            # (A x B) - (C x D) = ((A - C) x B) U (A x (B - D))
            overlaps = []
            pairs = list(zip(self.sets, other.sets))
            for n in range(len(pairs)):
                sets = (o if i != n else o-s for i, (s, o) in enumerate(pairs))
                overlaps.append(ProductSet(*sets))
            return Union(*overlaps)

        elif isinstance(other, Interval):
            if isinstance(self, (Interval, FiniteSet)):
                return Intersection(other, self.complement(S.Reals))

        elif isinstance(other, Union):
            return Union(*(o - self for o in other.args))

        elif isinstance(other, Complement):
            return Complement(other.args[0], Union(other.args[1], self), evaluate=False)

        elif other is S.EmptySet:
            return S.EmptySet

        elif isinstance(other, FiniteSet):
            sifted = sift(other, lambda x: fuzzy_bool(self.contains(x)))
            # ignore those that are contained in self
            return Union(FiniteSet(*(sifted[False])),
                Complement(FiniteSet(*(sifted[None])), self, evaluate=False)
                if sifted[None] else S.EmptySet)

    def symmetric_difference(self, other):
        """
        Returns symmetric difference of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(1, 3).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(3, oo))
        >>> Interval(1, 10).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(10, oo))

        >>> from sympy import S, EmptySet
        >>> S.Reals.symmetric_difference(EmptySet)
        Reals

        References
        ==========
        .. [1] https://en.wikipedia.org/wiki/Symmetric_difference

        """
        return SymmetricDifference(self, other)

    def _symmetric_difference(self, other):
        return Union(Complement(self, other), Complement(other, self))

    @property
    def inf(self):
        """
        The infimum of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).inf
        0
        >>> Union(Interval(0, 1), Interval(2, 3)).inf
        0

        """
        return self._inf

    @property
    def _inf(self):
        raise NotImplementedError("(%s)._inf" % self)

    @property
    def sup(self):
        """
        The supremum of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).sup
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).sup
        3

        """
        return self._sup

    @property
    def _sup(self):
        raise NotImplementedError("(%s)._sup" % self)

    def contains(self, other):
        """
        Returns a SymPy value indicating whether ``other`` is contained
        in ``self``: ``true`` if it is, ``false`` if it is not, else
        an unevaluated ``Contains`` expression (or, as in the case of
        ConditionSet and a union of FiniteSet/Intervals, an expression
        indicating the conditions for containment).

        Examples
        ========

        >>> from sympy import Interval, S
        >>> from sympy.abc import x

        >>> Interval(0, 1).contains(0.5)
        True

        As a shortcut it is possible to use the ``in`` operator, but that
        will raise an error unless an affirmative true or false is not
        obtained.

        >>> Interval(0, 1).contains(x)
        (0 <= x) & (x <= 1)
        >>> x in Interval(0, 1)
        Traceback (most recent call last):
        ...
        TypeError: did not evaluate to a bool: None

        The result of 'in' is a bool, not a SymPy value

        >>> 1 in Interval(0, 2)
        True
        >>> _ is S.true
        False
        """
        from .contains import Contains
        other = sympify(other, strict=True)

        c = self._contains(other)
        if isinstance(c, Contains):
            return c
        if c is None:
            return Contains(other, self, evaluate=False)
        b = tfn[c]
        if b is None:
            return c
        return b

    def _contains(self, other):
        """Test if ``other`` is an element of the set ``self``.

        This is an internal method that is expected to be overridden by
        subclasses of ``Set`` and will be called by the public
        :func:`Set.contains` method or the :class:`Contains` expression.

        Parameters
        ==========

        other: Sympified :class:`Basic` instance
            The object whose membership in ``self`` is to be tested.

        Returns
        =======

        Symbolic :class:`Boolean` or ``None``.

        A return value of ``None`` indicates that it is unknown whether
        ``other`` is contained in ``self``. Returning ``None`` from here
        ensures that ``self.contains(other)`` or ``Contains(self, other)`` will
        return an unevaluated :class:`Contains` expression.

        If not ``None`` then the returned value is a :class:`Boolean` that is
        logically equivalent to the statement that ``other`` is an element of
        ``self``. Usually this would be either ``S.true`` or ``S.false`` but
        not always.
        """
        raise NotImplementedError(f"{type(self).__name__}._contains")

    def is_subset(self, other):
        """
        Returns True if ``self`` is a subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_subset(Interval(0, 1, left_open=True))
        False

        """
        if not isinstance(other, Set):
            raise ValueError("Unknown argument '%s'" % other)

        # Handle the trivial cases
        if self == other:
            return True
        is_empty = self.is_empty
        if is_empty is True:
            return True
        elif fuzzy_not(is_empty) and other.is_empty:
            return False
        if self.is_finite_set is False and other.is_finite_set:
            return False

        # Dispatch on subclass rules
        ret = self._eval_is_subset(other)
        if ret is not None:
            return ret
        ret = other._eval_is_superset(self)
        if ret is not None:
            return ret

        # Use pairwise rules from multiple dispatch
        from sympy.sets.handlers.issubset import is_subset_sets
        ret = is_subset_sets(self, other)
        if ret is not None:
            return ret

        # Fall back on computing the intersection
        # XXX: We shouldn't do this. A query like this should be handled
        # without evaluating new Set objects. It should be the other way round
        # so that the intersect method uses is_subset for evaluation.
        if self.intersect(other) == self:
            return True

    def _eval_is_subset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    def _eval_is_superset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    # This should be deprecated:
    def issubset(self, other):
        """
        Alias for :meth:`is_subset()`
        """
        return self.is_subset(other)

    def is_proper_subset(self, other):
        """
        Returns True if ``self`` is a proper subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_proper_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_proper_subset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_subset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    def is_superset(self, other):
        """
        Returns True if ``self`` is a superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_superset(Interval(0, 1))
        False
        >>> Interval(0, 1).is_superset(Interval(0, 1, left_open=True))
        True

        """
        if isinstance(other, Set):
            return other.is_subset(self)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    # This should be deprecated:
    def issuperset(self, other):
        """
        Alias for :meth:`is_superset()`
        """
        return self.is_superset(other)

    def is_proper_superset(self, other):
        """
        Returns True if ``self`` is a proper superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).is_proper_superset(Interval(0, 0.5))
        True
        >>> Interval(0, 1).is_proper_superset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_superset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    def _eval_powerset(self):
        from .powerset import PowerSet
        return PowerSet(self)

    def powerset(self):
        """
        Find the Power set of ``self``.

        Examples
        ========

        >>> from sympy import EmptySet, FiniteSet, Interval

        A power set of an empty set:

        >>> A = EmptySet
        >>> A.powerset()
        {EmptySet}

        A power set of a finite set:

        >>> A = FiniteSet(1, 2)
        >>> a, b, c = FiniteSet(1), FiniteSet(2), FiniteSet(1, 2)
        >>> A.powerset() == FiniteSet(a, b, c, EmptySet)
        True

        A power set of an interval:

        >>> Interval(1, 2).powerset()
        PowerSet(Interval(1, 2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Power_set

        """
        return self._eval_powerset()

    @property
    def measure(self):
        """
        The (Lebesgue) measure of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).measure
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).measure
        2

        """
        return self._measure

    @property
    def kind(self):
        """
        The kind of a Set

        Explanation
        ===========

        Any :class:`Set` will have kind :class:`SetKind` which is
        parametrised by the kind of the elements of the set. For example
        most sets are sets of numbers and will have kind
        ``SetKind(NumberKind)``. If elements of sets are different in kind than
        their kind will ``SetKind(UndefinedKind)``. See
        :class:`sympy.core.kind.Kind` for an explanation of the kind system.

        Examples
        ========

        >>> from sympy import Interval, Matrix, FiniteSet, EmptySet, ProductSet, PowerSet

        >>> FiniteSet(Matrix([1, 2])).kind
        SetKind(MatrixKind(NumberKind))

        >>> Interval(1, 2).kind
        SetKind(NumberKind)

        >>> EmptySet.kind
        SetKind()

        A :class:`sympy.sets.powerset.PowerSet` is a set of sets:

        >>> PowerSet({1, 2, 3}).kind
        SetKind(SetKind(NumberKind))

        A :class:`ProductSet` represents the set of tuples of elements of
        other sets. Its kind is :class:`sympy.core.containers.TupleKind`
        parametrised by the kinds of the elements of those sets:

        >>> p = ProductSet(FiniteSet(1, 2), FiniteSet(3, 4))
        >>> list(p)
        [(1, 3), (2, 3), (1, 4), (2, 4)]
        >>> p.kind
        SetKind(TupleKind(NumberKind, NumberKind))

        When all elements of the set do not have same kind, the kind
        will be returned as ``SetKind(UndefinedKind)``:

        >>> FiniteSet(0, Matrix([1, 2])).kind
        SetKind(UndefinedKind)

        The kind of the elements of a set are given by the ``element_kind``
        attribute of ``SetKind``:

        >>> Interval(1, 2).kind.element_kind
        NumberKind

        See Also
        ========

        NumberKind
        sympy.core.kind.UndefinedKind
        sympy.core.containers.TupleKind
        MatrixKind
        sympy.matrices.expressions.sets.MatrixSet
        sympy.sets.conditionset.ConditionSet
        Rationals
        Naturals
        Integers
        sympy.sets.fancysets.ImageSet
        sympy.sets.fancysets.Range
        sympy.sets.fancysets.ComplexRegion
        sympy.sets.powerset.PowerSet
        sympy.sets.sets.ProductSet
        sympy.sets.sets.Interval
        sympy.sets.sets.Union
        sympy.sets.sets.Intersection
        sympy.sets.sets.Complement
        sympy.sets.sets.EmptySet
        sympy.sets.sets.UniversalSet
        sympy.sets.sets.FiniteSet
        sympy.sets.sets.SymmetricDifference
        sympy.sets.sets.DisjointUnion
        """
        return self._kind()

    @property
    def boundary(self):
        """
        The boundary or frontier of a set.

        Explanation
        ===========

        A point x is on the boundary of a set S if

        1.  x is in the closure of S.
            I.e. Every neighborhood of x contains a point in S.
        2.  x is not in the interior of S.
            I.e. There does not exist an open set centered on x contained
            entirely within S.

        There are the points on the outer rim of S.  If S is open then these
        points need not actually be contained within S.

        For example, the boundary of an interval is its start and end points.
        This is true regardless of whether or not the interval is open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).boundary
        {0, 1}
        >>> Interval(0, 1, True, False).boundary
        {0, 1}
        """
        return self._boundary

    @property
    def is_open(self):
        """
        Property method to check whether a set is open.

        Explanation
        ===========

        A set is open if and only if it has an empty intersection with its
        boundary. In particular, a subset A of the reals is open if and only
        if each one of its points is contained in an open interval that is a
        subset of A.

        Examples
        ========
        >>> from sympy import S
        >>> S.Reals.is_open
        True
        >>> S.Rationals.is_open
        False
        """
        return Intersection(self, self.boundary).is_empty

    @property
    def is_closed(self):
        """
        A property method to check whether a set is closed.

        Explanation
        ===========

        A set is closed if its complement is an open set. The closedness of a
        subset of the reals is determined with respect to R and its standard
        topology.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).is_closed
        True
        """
        return self.boundary.is_subset(self)

    @property
    def closure(self):
        """
        Property method which returns the closure of a set.
        The closure is defined as the union of the set itself and its
        boundary.

        Examples
        ========
        >>> from sympy import S, Interval
        >>> S.Reals.closure
        Reals
        >>> Interval(0, 1).closure
        Interval(0, 1)
        """
        return self + self.boundary

    @property
    def interior(self):
        """
        Property method which returns the interior of a set.
        The interior of a set S consists all points of S that do not
        belong to the boundary of S.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).interior
        Interval.open(0, 1)
        >>> Interval(0, 1).boundary.interior
        EmptySet
        """
        return self - self.boundary

    @property
    def _boundary(self):
        raise NotImplementedError()

    @property
    def _measure(self):
        raise NotImplementedError("(%s)._measure" % self)

    def _kind(self):
        return SetKind(UndefinedKind)

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return self.func(*[arg.evalf(n=dps) for arg in self.args])

    @sympify_return([('other', 'Set')], NotImplemented)
    def __add__(self, other):
        return self.union(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __or__(self, other):
        return self.union(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __and__(self, other):
        return self.intersect(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __mul__(self, other):
        return ProductSet(self, other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __xor__(self, other):
        return SymmetricDifference(self, other)

    @sympify_return([('exp', Expr)], NotImplemented)
    def __pow__(self, exp):
        if not (exp.is_Integer and exp >= 0):
            raise ValueError("%s: Exponent must be a positive Integer" % exp)
        return ProductSet(*[self]*exp)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __sub__(self, other):
        return Complement(self, other)

    def __contains__(self, other):
        other = _sympify(other)
        c = self._contains(other)
        b = tfn[c]
        if b is None:
            # x in y must evaluate to T or F; to entertain a None
            # result with Set use y.contains(x)
            raise TypeError('did not evaluate to a bool: %r' % c)
        return b


class ProductSet(Set):
    """
    Represents a Cartesian Product of Sets.

    Explanation
    ===========

    Returns a Cartesian product given several sets as either an iterable
    or individual arguments.

    Can use ``*`` operator on any sets for convenient shorthand.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet, ProductSet
    >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
    >>> ProductSet(I, S)
    ProductSet(Interval(0, 5), {1, 2, 3})

    >>> (2, 2) in ProductSet(I, S)
    True

    >>> Interval(0, 1) * Interval(0, 1) # The unit square
    ProductSet(Interval(0, 1), Interval(0, 1))

    >>> coin = FiniteSet('H', 'T')
    >>> set(coin**2)
    {(H, H), (H, T), (T, H), (T, T)}

    The Cartesian product is not commutative or associative e.g.:

    >>> I*S == S*I
    False
    >>> (I*I)*I == I*(I*I)
    False

    Notes
    =====

    - Passes most operations down to the argument sets

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cartesian_product
    """
    is_ProductSet = True

    def __new__(cls, *sets, **assumptions):
        if len(sets) == 1 and iterable(sets[0]) and not isinstance(sets[0], (Set, set)):
            sympy_deprecation_warning(
                """
ProductSet(iterable) is deprecated. Use ProductSet(*iterable) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-productset-iterable",
            )
            sets = tuple(sets[0])

        sets = [sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Arguments to ProductSet should be of type Set")

        # Nullary product of sets is *not* the empty set
        if len(sets) == 0:
            return FiniteSet(())

        if S.EmptySet in sets:
            return S.EmptySet

        return Basic.__new__(cls, *sets, **assumptions)

    @property
    def sets(self):
        return self.args

    def flatten(self):
        def _flatten(sets):
            for s in sets:
                if s.is_ProductSet:
                    yield from _flatten(s.sets)
                else:
                    yield s
        return ProductSet(*_flatten(self.sets))



    def _contains(self, element):
        """
        ``in`` operator for ProductSets.

        Examples
        ========

        >>> from sympy import Interval
        >>> (2, 3) in Interval(0, 5) * Interval(0, 5)
        True

        >>> (10, 10) in Interval(0, 5) * Interval(0, 5)
        False

        Passes operation on to constituent sets
        """
        if element.is_Symbol:
            return None

        if not isinstance(element, Tuple) or len(element) != len(self.sets):
            return S.false

        return And(*[s.contains(e) for s, e in zip(self.sets, element)])

    def as_relational(self, *symbols):
        symbols = [_sympify(s) for s in symbols]
        if len(symbols) != len(self.sets) or not all(
                i.is_Symbol for i in symbols):
            raise ValueError(
                'number of symbols must match the number of sets')
        return And(*[s.as_relational(i) for s, i in zip(self.sets, symbols)])

    @property
    def _boundary(self):
        return Union(*(ProductSet(*(b + b.boundary if i != j else b.boundary
                                for j, b in enumerate(self.sets)))
                                for i, a in enumerate(self.sets)))

    @property
    def is_iterable(self):
        """
        A property method which tests whether a set is iterable or not.
        Returns True if set is iterable, otherwise returns False.

        Examples
        ========

        >>> from sympy import FiniteSet, Interval
        >>> I = Interval(0, 1)
        >>> A = FiniteSet(1, 2, 3, 4, 5)
        >>> I.is_iterable
        False
        >>> A.is_iterable
        True

        """
        return all(set.is_iterable for set in self.sets)

    def __iter__(self):
        """
        A method which implements is_iterable property method.
        If self.is_iterable returns True (both constituent sets are iterable),
        then return the Cartesian Product. Otherwise, raise TypeError.
        """
        return iproduct(*self.sets)

    @property
    def is_empty(self):
        return fuzzy_or(s.is_empty for s in self.sets)

    @property
    def is_finite_set(self):
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def _measure(self):
        measure = 1
        for s in self.sets:
            measure *= s.measure
        return measure

    def _kind(self):
        return SetKind(TupleKind(*(i.kind.element_kind for i in self.args)))

    def __len__(self):
        return reduce(lambda a, b: a*b, (len(s) for s in self.args))

    def __bool__(self):
        return all(self.sets)


class Interval(Set):
    """
    Represents a real interval as a Set.

    Usage:
        Returns an interval with end points ``start`` and ``end``.

        For ``left_open=True`` (default ``left_open`` is ``False``) the interval
        will be open on the left. Similarly, for ``right_open=True`` the interval
        will be open on the right.

    Examples
    ========

    >>> from sympy import Symbol, Interval
    >>> Interval(0, 1)
    Interval(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Lopen(0, 1)
    Interval.Lopen(0, 1)
    >>> Interval.open(0, 1)
    Interval.open(0, 1)

    >>> a = Symbol('a', real=True)
    >>> Interval(0, a)
    Interval(0, a)

    Notes
    =====
    - Only real end points are supported
    - ``Interval(a, b)`` with $a > b$ will return the empty set
    - Use the ``evalf()`` method to turn an Interval into an mpmath
      ``mpi`` interval instance

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_%28mathematics%29
    """
    is_Interval = True

    def __new__(cls, start, end, left_open=False, right_open=False):

        start = _sympify(start)
        end = _sympify(end)
        left_open = _sympify(left_open)
        right_open = _sympify(right_open)

        if not all(isinstance(a, (type(true), type(false)))
            for a in [left_open, right_open]):
            raise NotImplementedError(
                "left_open and right_open can have only true/false values, "
                "got %s and %s" % (left_open, right_open))

        # Only allow real intervals
        if fuzzy_not(fuzzy_and(i.is_extended_real for i in (start, end, end-start))):
            raise ValueError("Non-real intervals are not supported")

        # evaluate if possible
        if is_lt(end, start):
            return S.EmptySet
        elif (end - start).is_negative:
            return S.EmptySet

        if end == start and (left_open or right_open):
            return S.EmptySet
        if end == start and not (left_open or right_open):
            if start is S.Infinity or start is S.NegativeInfinity:
                return S.EmptySet
            return FiniteSet(end)

        # Make sure infinite interval end points are open.
        if start is S.NegativeInfinity:
            left_open = true
        if end is S.Infinity:
            right_open = true
        if start == S.Infinity or end == S.NegativeInfinity:
            return S.EmptySet

        return Basic.__new__(cls, start, end, left_open, right_open)

    @property
    def start(self):
        """
        The left end point of the interval.

        This property takes the same value as the ``inf`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).start
        0

        """
        return self._args[0]

    @property
    def end(self):
        """
        The right end point of the interval.

        This property takes the same value as the ``sup`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).end
        1

        """
        return self._args[1]

    @property
    def left_open(self):
        """
        True if interval is left-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, left_open=True).left_open
        True
        >>> Interval(0, 1, left_open=False).left_open
        False

        """
        return self._args[2]

    @property
    def right_open(self):
        """
        True if interval is right-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, right_open=True).right_open
        True
        >>> Interval(0, 1, right_open=False).right_open
        False

        """
        return self._args[3]

    @classmethod
    def open(cls, a, b):
        """Return an interval including neither boundary."""
        return cls(a, b, True, True)

    @classmethod
    def Lopen(cls, a, b):
        """Return an interval not including the left boundary."""
        return cls(a, b, True, False)

    @classmethod
    def Ropen(cls, a, b):
        """Return an interval not including the right boundary."""
        return cls(a, b, False, True)

    @property
    def _inf(self):
        return self.start

    @property
    def _sup(self):
        return self.end

    @property
    def left(self):
        return self.start

    @property
    def right(self):
        return self.end

    @property
    def is_empty(self):
        if self.left_open or self.right_open:
            cond = self.start >= self.end  # One/both bounds open
        else:
            cond = self.start > self.end  # Both bounds closed
        return fuzzy_bool(cond)

    @property
    def is_finite_set(self):
        return self.measure.is_zero

    def _complement(self, other):
        if other == S.Reals:
            a = Interval(S.NegativeInfinity, self.start,
                         True, not self.left_open)
            b = Interval(self.end, S.Infinity, not self.right_open, True)
            return Union(a, b)

        if isinstance(other, FiniteSet):
            nums = [m for m in other.args if m.is_number]
            if nums == []:
                return None

        return Set._complement(self, other)

    @property
    def _boundary(self):
        finite_points = [p for p in (self.start, self.end)
                         if abs(p) != S.Infinity]
        return FiniteSet(*finite_points)

    def _contains(self, other):
        if (not isinstance(other, Expr) or other is S.NaN
            or other.is_real is False or other.has(S.ComplexInfinity)):
                # if an expression has zoo it will be zoo or nan
                # and neither of those is real
                return false

        if self.start is S.NegativeInfinity and self.end is S.Infinity:
            if other.is_real is not None:
                return tfn[other.is_real]

        d = Dummy()
        return self.as_relational(d).subs(d, other)

    def as_relational(self, x):
        """Rewrite an interval in terms of inequalities and logic operators."""
        x = sympify(x)
        if self.right_open:
            right = x < self.end
        else:
            right = x <= self.end
        if self.left_open:
            left = self.start < x
        else:
            left = self.start <= x
        return And(left, right)

    @property
    def _measure(self):
        return self.end - self.start

    def _kind(self):
        return SetKind(NumberKind)

    def to_mpi(self, prec=53):
        return mpi(mpf(self.start._eval_evalf(prec)),
            mpf(self.end._eval_evalf(prec)))

    def _eval_evalf(self, prec):
        return Interval(self.left._evalf(prec), self.right._evalf(prec),
            left_open=self.left_open, right_open=self.right_open)

    def _is_comparable(self, other):
        is_comparable = self.start.is_comparable
        is_comparable &= self.end.is_comparable
        is_comparable &= other.start.is_comparable
        is_comparable &= other.end.is_comparable

        return is_comparable

    @property
    def is_left_unbounded(self):
        """Return ``True`` if the left endpoint is negative infinity. """
        return self.left is S.NegativeInfinity or self.left == Float("-inf")

    @property
    def is_right_unbounded(self):
        """Return ``True`` if the right endpoint is positive infinity. """
        return self.right is S.Infinity or self.right == Float("+inf")

    def _eval_Eq(self, other):
        if not isinstance(other, Interval):
            if isinstance(other, FiniteSet):
                return false
            elif isinstance(other, Set):
                return None
            return false


class Union(Set, LatticeOp):
    """
    Represents a union of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Union, Interval
    >>> Union(Interval(1, 2), Interval(3, 4))
    Union(Interval(1, 2), Interval(3, 4))

    The Union constructor will always try to merge overlapping intervals,
    if possible. For example:

    >>> Union(Interval(1, 2), Interval(2, 3))
    Interval(1, 3)

    See Also
    ========

    Intersection

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Union_%28set_theory%29
    """
    is_Union = True

    @property
    def identity(self):
        return S.EmptySet

    @property
    def zero(self):
        return S.UniversalSet

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs to merge intersections and iterables
        args = _sympify(args)

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_union(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    def args(self):
        return self._args

    def _complement(self, universe):
        # DeMorgan's Law
        return Intersection(s.complement(universe) for s in self.args)

    @property
    def _inf(self):
        # We use Min so that sup is meaningful in combination with symbolic
        # interval end points.
        return Min(*[set.inf for set in self.args])

    @property
    def _sup(self):
        # We use Max so that sup is meaningful in combination with symbolic
        # end points.
        return Max(*[set.sup for set in self.args])

    @property
    def is_empty(self):
        return fuzzy_and(set.is_empty for set in self.args)

    @property
    def is_finite_set(self):
        return fuzzy_and(set.is_finite_set for set in self.args)

    @property
    def _measure(self):
        # Measure of a union is the sum of the measures of the sets minus
        # the sum of their pairwise intersections plus the sum of their
        # triple-wise intersections minus ... etc...

        # Sets is a collection of intersections and a set of elementary
        # sets which made up those intersections (called "sos" for set of sets)
        # An example element might of this list might be:
        #    ( {A,B,C}, A.intersect(B).intersect(C) )

        # Start with just elementary sets (  ({A}, A), ({B}, B), ... )
        # Then get and subtract (  ({A,B}, (A int B), ... ) while non-zero
        sets = [(FiniteSet(s), s) for s in self.args]
        measure = 0
        parity = 1
        while sets:
            # Add up the measure of these sets and add or subtract it to total
            measure += parity * sum(inter.measure for sos, inter in sets)

            # For each intersection in sets, compute the intersection with every
            # other set not already part of the intersection.
            sets = ((sos + FiniteSet(newset), newset.intersect(intersection))
                    for sos, intersection in sets for newset in self.args
                    if newset not in sos)

            # Clear out sets with no measure
            sets = [(sos, inter) for sos, inter in sets if inter.measure != 0]

            # Clear out duplicates
            sos_list = []
            sets_list = []
            for _set in sets:
                if _set[0] in sos_list:
                    continue
                else:
                    sos_list.append(_set[0])
                    sets_list.append(_set)
            sets = sets_list

            # Flip Parity - next time subtract/add if we added/subtracted here
            parity *= -1
        return measure

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.EmptySet)
        if not kinds:
            return SetKind()
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind(UndefinedKind)

    @property
    def _boundary(self):
        def boundary_of_set(i):
            """ The boundary of set i minus interior of all other sets """
            b = self.args[i].boundary
            for j, a in enumerate(self.args):
                if j != i:
                    b = b - a.interior
            return b
        return Union(*map(boundary_of_set, range(len(self.args))))

    def _contains(self, other):
        return Or(*[s.contains(other) for s in self.args])

    def is_subset(self, other):
        return fuzzy_and(s.is_subset(other) for s in self.args)

    def as_relational(self, symbol):
        """Rewrite a Union in terms of equalities and logic operators. """
        if (len(self.args) == 2 and
                all(isinstance(i, Interval) for i in self.args)):
            # optimization to give 3 args as (x > 1) & (x < 5) & Ne(x, 3)
            # instead of as 4, ((1 <= x) & (x < 3)) | ((x <= 5) & (3 < x))
            # XXX: This should be ideally be improved to handle any number of
            # intervals and also not to assume that the intervals are in any
            # particular sorted order.
            a, b = self.args
            if a.sup == b.inf and a.right_open and b.left_open:
                mincond = symbol > a.inf if a.left_open else symbol >= a.inf
                maxcond = symbol < b.sup if b.right_open else symbol <= b.sup
                necond = Ne(symbol, a.sup)
                return And(necond, mincond, maxcond)
        return Or(*[i.as_relational(symbol) for i in self.args])

    @property
    def is_iterable(self):
        return all(arg.is_iterable for arg in self.args)

    def __iter__(self):
        return roundrobin(*(iter(arg) for arg in self.args))


class Intersection(Set, LatticeOp):
    """
    Represents an intersection of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Intersection, Interval
    >>> Intersection(Interval(1, 3), Interval(2, 4))
    Interval(2, 3)

    We often use the .intersect method

    >>> Interval(1,3).intersect(Interval(2,4))
    Interval(2, 3)

    See Also
    ========

    Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Intersection_%28set_theory%29
    """
    is_Intersection = True

    @property
    def identity(self):
        return S.UniversalSet

    @property
    def zero(self):
        return S.EmptySet

    def __new__(cls, *args , evaluate=None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        # flatten inputs to merge intersections and iterables
        args = list(ordered(set(_sympify(args))))

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_intersection(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    def args(self):
        return self._args

    @property
    def is_iterable(self):
        return any(arg.is_iterable for arg in self.args)

    @property
    def is_finite_set(self):
        if fuzzy_or(arg.is_finite_set for arg in self.args):
            return True

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.UniversalSet)
        if not kinds:
            return SetKind(UndefinedKind)
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind()

    @property
    def _inf(self):
        raise NotImplementedError()

    @property
    def _sup(self):
        raise NotImplementedError()

    def _contains(self, other):
        return And(*[set.contains(other) for set in self.args])

    def __iter__(self):
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        completed = False
        candidates = sets_sift[True] + sets_sift[None]

        finite_candidates, others = [], []
        for candidate in candidates:
            length = None
            try:
                length = len(candidate)
            except TypeError:
                others.append(candidate)

            if length is not None:
                finite_candidates.append(candidate)
        finite_candidates.sort(key=len)

        for s in finite_candidates + others:
            other_sets = set(self.args) - {s}
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            for x in s:
                try:
                    if x in other:
                        yield x
                except TypeError:
                    completed = False
            if completed:
                return

        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")

    @staticmethod
    def _handle_finite_sets(args):
        '''Simplify intersection of one or more FiniteSets and other sets'''

        # First separate the FiniteSets from the others
        fs_args, others = sift(args, lambda x: x.is_FiniteSet, binary=True)

        # Let the caller handle intersection of non-FiniteSets
        if not fs_args:
            return

        # Convert to Python sets and build the set of all elements
        fs_sets = [set(fs) for fs in fs_args]
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Extract elements that are definitely in or definitely not in the
        # intersection. Here we check contains for all of args.
        definite = set()
        for e in all_elements:
            inall = fuzzy_and(s.contains(e) for s in args)
            if inall is True:
                definite.add(e)
            if inall is not None:
                for s in fs_sets:
                    s.discard(e)

        # At this point all elements in all of fs_sets are possibly in the
        # intersection. In some cases this is because they are definitely in
        # the intersection of the finite sets but it's not clear if they are
        # members of others. We might have {m, n}, {m}, and Reals where we
        # don't know if m or n is real. We want to remove n here but it is
        # possibly in because it might be equal to m. So what we do now is
        # extract the elements that are definitely in the remaining finite
        # sets iteratively until we end up with {n}, {}. At that point if we
        # get any empty set all remaining elements are discarded.

        fs_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Need fuzzy containment testing
        fs_symsets = [FiniteSet(*s) for s in fs_sets]

        while fs_elements:
            for e in fs_elements:
                infs = fuzzy_and(s.contains(e) for s in fs_symsets)
                if infs is True:
                    definite.add(e)
                if infs is not None:
                    for n, s in enumerate(fs_sets):
                        # Update Python set and FiniteSet
                        if e in s:
                            s.remove(e)
                            fs_symsets[n] = FiniteSet(*s)
                    fs_elements.remove(e)
                    break
            # If we completed the for loop without removing anything we are
            # done so quit the outer while loop
            else:
                break

        # If any of the sets of remainder elements is empty then we discard
        # all of them for the intersection.
        if not all(fs_sets):
            fs_sets = [set()]

        # Here we fold back the definitely included elements into each fs.
        # Since they are definitely included they must have been members of
        # each FiniteSet to begin with. We could instead fold these in with a
        # Union at the end to get e.g. {3}|({x}&{y}) rather than {3,x}&{3,y}.
        if definite:
            fs_sets = [fs | definite for fs in fs_sets]

        if fs_sets == [set()]:
            return S.EmptySet

        sets = [FiniteSet(*s) for s in fs_sets]

        # Any set in others is redundant if it contains all the elements that
        # are in the finite sets so we don't need it in the Intersection
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())
        is_redundant = lambda o: all(fuzzy_bool(o.contains(e)) for e in all_elements)
        others = [o for o in others if not is_redundant(o)]

        if others:
            rest = Intersection(*others)
            # XXX: Maybe this shortcut should be at the beginning. For large
            # FiniteSets it could much more efficient to process the other
            # sets first...
            if rest is S.EmptySet:
                return S.EmptySet
            # Flatten the Intersection
            if rest.is_Intersection:
                sets.extend(rest.args)
            else:
                sets.append(rest)

        if len(sets) == 1:
            return sets[0]
        else:
            return Intersection(*sets, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite an Intersection in terms of equalities and logic operators"""
        return And(*[set.as_relational(symbol) for set in self.args])


class Complement(Set):
    r"""Represents the set difference or relative complement of a set with
    another set.

    $$A - B = \{x \in A \mid x \notin B\}$$


    Examples
    ========

    >>> from sympy import Complement, FiniteSet
    >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
    {0, 2}

    See Also
    =========

    Intersection, Union

    References
    ==========

    .. [1] https://mathworld.wolfram.com/ComplementSet.html
    """

    is_Complement = True

    def __new__(cls, a, b, evaluate=True):
        a, b = map(_sympify, (a, b))
        if evaluate:
            return Complement.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        """
        Simplify a :class:`Complement`.

        """
        if B == S.UniversalSet or A.is_subset(B):
            return S.EmptySet

        if isinstance(B, Union):
            return Intersection(*(s.complement(A) for s in B.args))

        result = B._complement(A)
        if result is not None:
            return result
        else:
            return Complement(A, B, evaluate=False)

    def _contains(self, other):
        A = self.args[0]
        B = self.args[1]
        return And(A.contains(other), Not(B.contains(other)))

    def as_relational(self, symbol):
        """Rewrite a complement in terms of equalities and logic
        operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = Not(B.as_relational(symbol))

        return And(A_rel, B_rel)

    def _kind(self):
        return self.args[0].kind

    @property
    def is_iterable(self):
        if self.args[0].is_iterable:
            return True

    @property
    def is_finite_set(self):
        A, B = self.args
        a_finite = A.is_finite_set
        if a_finite is True:
            return True
        elif a_finite is False and B.is_finite_set:
            return False

    def __iter__(self):
        A, B = self.args
        for a in A:
            if a not in B:
                    yield a
            else:
                continue


class EmptySet(Set, metaclass=Singleton):
    """
    Represents the empty set. The empty set is available as a singleton
    as ``S.EmptySet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.EmptySet
    EmptySet

    >>> Interval(1, 2).intersect(S.EmptySet)
    EmptySet

    See Also
    ========

    UniversalSet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Empty_set
    """
    is_empty = True
    is_finite_set = True
    is_FiniteSet = True

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return True

    @property
    def _measure(self):
        return 0

    def _contains(self, other):
        return false

    def as_relational(self, symbol):
        return false

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])

    def _eval_powerset(self):
        return FiniteSet(self)

    @property
    def _boundary(self):
        return self

    def _complement(self, other):
        return other

    def _kind(self):
        return SetKind()

    def _symmetric_difference(self, other):
        return other


class UniversalSet(Set, metaclass=Singleton):
    """
    Represents the set of all things.
    The universal set is available as a singleton as ``S.UniversalSet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.UniversalSet
    UniversalSet

    >>> Interval(1, 2).intersect(S.UniversalSet)
    Interval(1, 2)

    See Also
    ========

    EmptySet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Universal_set
    """

    is_UniversalSet = True
    is_empty = False
    is_finite_set = False

    def _complement(self, other):
        return S.EmptySet

    def _symmetric_difference(self, other):
        return other

    @property
    def _measure(self):
        return S.Infinity

    def _kind(self):
        return SetKind(UndefinedKind)

    def _contains(self, other):
        return true

    def as_relational(self, symbol):
        return true

    @property
    def _boundary(self):
        return S.EmptySet


class FiniteSet(Set):
    """
    Represents a finite set of Sympy expressions.

    Examples
    ========

    >>> from sympy import FiniteSet, Symbol, Interval, Naturals0
    >>> FiniteSet(1, 2, 3, 4)
    {1, 2, 3, 4}
    >>> 3 in FiniteSet(1, 2, 3, 4)
    True
    >>> FiniteSet(1, (1, 2), Symbol('x'))
    {1, x, (1, 2)}
    >>> FiniteSet(Interval(1, 2), Naturals0, {1, 2})
    FiniteSet({1, 2}, Interval(1, 2), Naturals0)
    >>> members = [1, 2, 3, 4]
    >>> f = FiniteSet(*members)
    >>> f
    {1, 2, 3, 4}
    >>> f - FiniteSet(2)
    {1, 3, 4}
    >>> f + FiniteSet(2, 5)
    {1, 2, 3, 4, 5}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite_set
    """
    is_FiniteSet = True
    is_iterable = True
    is_empty = False
    is_finite_set = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if evaluate:
            args = list(map(sympify, args))

            if len(args) == 0:
                return S.EmptySet
        else:
            args = list(map(sympify, args))

        # keep the form of the first canonical arg
        dargs = {}
        for i in reversed(list(ordered(args))):
            if i.is_Symbol:
                dargs[i] = i
            else:
                try:
                    dargs[i.as_dummy()] = i
                except TypeError:
                    # e.g. i = class without args like `Interval`
                    dargs[i] = i
        _args_set = set(dargs.values())
        args = list(ordered(_args_set, Set._infimum_key))
        obj = Basic.__new__(cls, *args)
        obj._args_set = _args_set
        return obj


    def __iter__(self):
        return iter(self.args)

    def _complement(self, other):
        if isinstance(other, Interval):
            # Splitting in sub-intervals is only done for S.Reals;
            # other cases that need splitting will first pass through
            # Set._complement().
            nums, syms = [], []
            for m in self.args:
                if m.is_number and m.is_real:
                    nums.append(m)
                elif m.is_real == False:
                    pass  # drop non-reals
                else:
                    syms.append(m)  # various symbolic expressions
            if other == S.Reals and nums != []:
                nums.sort()
                intervals = []  # Build up a list of intervals between the elements
                intervals += [Interval(S.NegativeInfinity, nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(Interval(a, b, True, True))  # both open
                intervals.append(Interval(nums[-1], S.Infinity, True, True))
                if syms != []:
                    return Complement(Union(*intervals, evaluate=False),
                            FiniteSet(*syms), evaluate=False)
                else:
                    return Union(*intervals, evaluate=False)
            elif nums == []:  # no splitting necessary or possible:
                if syms:
                    return Complement(other, FiniteSet(*syms), evaluate=False)
                else:
                    return other

        elif isinstance(other, FiniteSet):
            unk = []
            for i in self:
                c = sympify(other.contains(i))
                if c is not S.true and c is not S.false:
                    unk.append(i)
            unk = FiniteSet(*unk)
            if unk == self:
                return
            not_true = []
            for i in other:
                c = sympify(self.contains(i))
                if c is not S.true:
                    not_true.append(i)
            return Complement(FiniteSet(*not_true), unk)

        return Set._complement(self, other)

    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        Explanation
        ===========

        The actual test is for mathematical equality (as opposed to
        syntactical equality). In the worst case all elements of the
        set must be checked.

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        if other in self._args_set:
            return S.true
        else:
            # evaluate=True is needed to override evaluate=False context;
            # we need Eq to do the evaluation
            return Or(*[Eq(e, other, evaluate=True) for e in self.args])

    def _eval_is_subset(self, other):
        return fuzzy_and(other._contains(e) for e in self.args)

    @property
    def _boundary(self):
        return self

    @property
    def _inf(self):
        return Min(*self)

    @property
    def _sup(self):
        return Max(*self)

    @property
    def measure(self):
        return 0

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all(i.kind == self.args[0].kind for i in self.args):
            return SetKind(self.args[0].kind)
        else:
            return SetKind(UndefinedKind)

    def __len__(self):
        return len(self.args)

    def as_relational(self, symbol):
        """Rewrite a FiniteSet in terms of equalities and logic operators. """
        return Or(*[Eq(symbol, elem) for elem in self])

    def compare(self, other):
        return (hash(self) - hash(other))

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return FiniteSet(*[elem.evalf(n=dps) for elem in self])

    def _eval_simplify(self, **kwargs):
        from sympy.simplify import simplify
        return FiniteSet(*[simplify(elem, **kwargs) for elem in self])

    @property
    def _sorted_args(self):
        return self.args

    def _eval_powerset(self):
        return self.func(*[self.func(*s) for s in subsets(self.args)])

    def _eval_rewrite_as_PowerSet(self, *args, **kwargs):
        """Rewriting method for a finite set to a power set."""
        from .powerset import PowerSet

        is2pow = lambda n: bool(n and not n & (n - 1))
        if not is2pow(len(self)):
            return None

        fs_test = lambda arg: isinstance(arg, Set) and arg.is_FiniteSet
        if not all(fs_test(arg) for arg in args):
            return None

        biggest = max(args, key=len)
        for arg in subsets(biggest.args):
            arg_set = FiniteSet(*arg)
            if arg_set not in args:
                return None
        return PowerSet(biggest)

    def __ge__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return other.is_subset(self)

    def __gt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_superset(other)

    def __le__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_subset(other)

    def __lt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_subset(other)

    def __eq__(self, other):
        if isinstance(other, (set, frozenset)):
            return self._args_set == other
        return super().__eq__(other)

    __hash__ : Callable[[Basic], Any] = Basic.__hash__

_sympy_converter[set] = lambda x: FiniteSet(*x)
_sympy_converter[frozenset] = lambda x: FiniteSet(*x)


class SymmetricDifference(Set):
    """Represents the set of elements which are in either of the
    sets and not in their intersection.

    Examples
    ========

    >>> from sympy import SymmetricDifference, FiniteSet
    >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
    {1, 2, 4, 5}

    See Also
    ========

    Complement, Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_difference
    """

    is_SymmetricDifference = True

    def __new__(cls, a, b, evaluate=True):
        if evaluate:
            return SymmetricDifference.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        result = B._symmetric_difference(A)
        if result is not None:
            return result
        else:
            return SymmetricDifference(A, B, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite a symmetric_difference in terms of equalities and
        logic operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = B.as_relational(symbol)

        return Xor(A_rel, B_rel)

    @property
    def is_iterable(self):
        if all(arg.is_iterable for arg in self.args):
            return True

    def __iter__(self):

        args = self.args
        union = roundrobin(*(iter(arg) for arg in args))

        for item in union:
            count = 0
            for s in args:
                if item in s:
                    count += 1

            if count % 2 == 1:
                yield item



class DisjointUnion(Set):
    """ Represents the disjoint union (also known as the external disjoint union)
    of a finite number of sets.

    Examples
    ========

    >>> from sympy import DisjointUnion, FiniteSet, Interval, Union, Symbol
    >>> A = FiniteSet(1, 2, 3)
    >>> B = Interval(0, 5)
    >>> DisjointUnion(A, B)
    DisjointUnion({1, 2, 3}, Interval(0, 5))
    >>> DisjointUnion(A, B).rewrite(Union)
    Union(ProductSet({1, 2, 3}, {0}), ProductSet(Interval(0, 5), {1}))
    >>> C = FiniteSet(Symbol('x'), Symbol('y'), Symbol('z'))
    >>> DisjointUnion(C, C)
    DisjointUnion({x, y, z}, {x, y, z})
    >>> DisjointUnion(C, C).rewrite(Union)
    ProductSet({x, y, z}, {0, 1})

    References
    ==========

    https://en.wikipedia.org/wiki/Disjoint_union
    """

    def __new__(cls, *sets):
        dj_collection = []
        for set_i in sets:
            if isinstance(set_i, Set):
                dj_collection.append(set_i)
            else:
                raise TypeError("Invalid input: '%s', input args \
                    to DisjointUnion must be Sets" % set_i)
        obj = Basic.__new__(cls, *dj_collection)
        return obj

    @property
    def sets(self):
        return self.args

    @property
    def is_empty(self):
        return fuzzy_and(s.is_empty for s in self.sets)

    @property
    def is_finite_set(self):
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def is_iterable(self):
        if self.is_empty:
            return False
        iter_flag = True
        for set_i in self.sets:
            if not set_i.is_empty:
                iter_flag = iter_flag and set_i.is_iterable
        return iter_flag

    def _eval_rewrite_as_Union(self, *sets, **kwargs):
        """
        Rewrites the disjoint union as the union of (``set`` x {``i``})
        where ``set`` is the element in ``sets`` at index = ``i``
        """

        dj_union = S.EmptySet
        index = 0
        for set_i in sets:
            if isinstance(set_i, Set):
                cross = ProductSet(set_i, FiniteSet(index))
                dj_union = Union(dj_union, cross)
                index = index + 1
        return dj_union

    def _contains(self, element):
        """
        ``in`` operator for DisjointUnion

        Examples
        ========

        >>> from sympy import Interval, DisjointUnion
        >>> D = DisjointUnion(Interval(0, 1), Interval(0, 2))
        >>> (0.5, 0) in D
        True
        >>> (0.5, 1) in D
        True
        >>> (1.5, 0) in D
        False
        >>> (1.5, 1) in D
        True

        Passes operation on to constituent sets
        """
        if not isinstance(element, Tuple) or len(element) != 2:
            return S.false

        if not element[1].is_Integer:
            return S.false

        if element[1] >= len(self.sets) or element[1] < 0:
            return S.false

        return self.sets[element[1]]._contains(element[0])

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all(i.kind == self.args[0].kind for i in self.args):
            return self.args[0].kind
        else:
            return SetKind(UndefinedKind)

    def __iter__(self):
        if self.is_iterable:

            iters = []
            for i, s in enumerate(self.sets):
                iters.append(iproduct(s, {Integer(i)}))

            return iter(roundrobin(*iters))
        else:
            raise ValueError("'%s' is not iterable." % self)

    def __len__(self):
        """
        Returns the length of the disjoint union, i.e., the number of elements in the set.

        Examples
        ========

        >>> from sympy import FiniteSet, DisjointUnion, EmptySet
        >>> D1 = DisjointUnion(FiniteSet(1, 2, 3, 4), EmptySet, FiniteSet(3, 4, 5))
        >>> len(D1)
        7
        >>> D2 = DisjointUnion(FiniteSet(3, 5, 7), EmptySet, FiniteSet(3, 5, 7))
        >>> len(D2)
        6
        >>> D3 = DisjointUnion(EmptySet, EmptySet)
        >>> len(D3)
        0

        Adds up the lengths of the constituent sets.
        """

        if self.is_finite_set:
            size = 0
            for set in self.sets:
                size += len(set)
            return size
        else:
            raise ValueError("'%s' is not a finite set." % self)


def imageset(*args):
    r"""
    Return an image of the set under transformation ``f``.

    Explanation
    ===========

    If this function cannot compute the image, it returns an
    unevaluated ImageSet object.

    .. math::
        \{ f(x) \mid x \in \mathrm{self} \}

    Examples
    ========

    >>> from sympy import S, Interval, imageset, sin, Lambda
    >>> from sympy.abc import x

    >>> imageset(x, 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(lambda x: 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(Lambda(x, sin(x)), Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))

    >>> imageset(sin, Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))
    >>> imageset(lambda y: x + y, Interval(-2, 1))
    ImageSet(Lambda(y, x + y), Interval(-2, 1))

    Expressions applied to the set of Integers are simplified
    to show as few negatives as possible and linear expressions
    are converted to a canonical form. If this is not desirable
    then the unevaluated ImageSet should be used.

    >>> imageset(x, -2*x + 5, S.Integers)
    ImageSet(Lambda(x, 2*x + 1), Integers)

    See Also
    ========

    sympy.sets.fancysets.ImageSet

    """
    from .fancysets import ImageSet
    from .setexpr import set_function

    if len(args) < 2:
        raise ValueError('imageset expects at least 2 args, got: %s' % len(args))

    if isinstance(args[0], (Symbol, tuple)) and len(args) > 2:
        f = Lambda(args[0], args[1])
        set_list = args[2:]
    else:
        f = args[0]
        set_list = args[1:]

    if isinstance(f, Lambda):
        pass
    elif callable(f):
        nargs = getattr(f, 'nargs', {})
        if nargs:
            if len(nargs) != 1:
                raise NotImplementedError(filldedent('''
                    This function can take more than 1 arg
                    but the potentially complicated set input
                    has not been analyzed at this point to
                    know its dimensions. TODO
                    '''))
            N = nargs.args[0]
            if N == 1:
                s = 'x'
            else:
                s = [Symbol('x%i' % i) for i in range(1, N + 1)]
        else:
            s = inspect.signature(f).parameters

        dexpr = _sympify(f(*[Dummy() for i in s]))
        var = tuple(uniquely_named_symbol(
            Symbol(i), dexpr) for i in s)
        f = Lambda(var, f(*var))
    else:
        raise TypeError(filldedent('''
            expecting lambda, Lambda, or FunctionClass,
            not \'%s\'.''' % func_name(f)))

    if any(not isinstance(s, Set) for s in set_list):
        name = [func_name(s) for s in set_list]
        raise ValueError(
            'arguments after mapping should be sets, not %s' % name)

    if len(set_list) == 1:
        set = set_list[0]
        try:
            # TypeError if arg count != set dimensions
            r = set_function(f, set)
            if r is None:
                raise TypeError
            if not r:
                return r
        except TypeError:
            r = ImageSet(f, set)
        if isinstance(r, ImageSet):
            f, set = r.args

        if f.variables[0] == f.expr:
            return set

        if isinstance(set, ImageSet):
            # XXX: Maybe this should just be:
            # f2 = set.lambda
            # fun = Lambda(f2.signature, f(*f2.expr))
            # return imageset(fun, *set.base_sets)
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                x = set.lamda.variables[0]
                y = f.variables[0]
                return imageset(
                    Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)

        if r is not None:
            return r

    return ImageSet(f, *set_list)


def is_function_invertible_in_set(func, setv):
    """
    Checks whether function ``func`` is invertible when the domain is
    restricted to set ``setv``.
    """
    # Functions known to always be invertible:
    if func in (exp, log):
        return True
    u = Dummy("u")
    fdiff = func(u).diff(u)
    # monotonous functions:
    # TODO: check subsets (`func` in `setv`)
    if (fdiff > 0) == True or (fdiff < 0) == True:
        return True
    # TODO: support more
    return None


def simplify_union(args):
    """
    Simplify a :class:`Union` using known rules.

    Explanation
    ===========

    We first start with global rules like 'Merge all FiniteSets'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent.  This process depends
    on ``union_sets(a, b)`` functions.
    """
    from sympy.sets.handlers.union import union_sets

    # ===== Global Rules =====
    if not args:
        return S.EmptySet

    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")

    # Merge all finite sets
    finite_sets = [x for x in args if x.is_FiniteSet]
    if len(finite_sets) > 1:
        a = (x for set in finite_sets for x in set)
        finite_set = FiniteSet(*a)
        args = [finite_set] + [x for x in args if not x.is_FiniteSet]

    # ===== Pair-wise Rules =====
    # Here we depend on rules built into the constituent sets
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = union_sets(s, t)
                # This returns None if s does not know how to intersect
                # with t. Returns the newly intersected set otherwise
                if new_set is not None:
                    if not isinstance(new_set, set):
                        new_set = {new_set}
                    new_args = (args - {s, t}).union(new_set)
                    break
            if new_args:
                args = new_args
                break

    if len(args) == 1:
        return args.pop()
    else:
        return Union(*args, evaluate=False)


def simplify_intersection(args):
    """
    Simplify an intersection using known rules.

    Explanation
    ===========

    We first start with global rules like
    'if any empty sets return empty set' and 'distribute any unions'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent
    """

    # ===== Global Rules =====
    if not args:
        return S.UniversalSet

    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")

    # If any EmptySets return EmptySet
    if S.EmptySet in args:
        return S.EmptySet

    # Handle Finite sets
    rv = Intersection._handle_finite_sets(args)

    if rv is not None:
        return rv

    # If any of the sets are unions, return a Union of Intersections
    for s in args:
        if s.is_Union:
            other_sets = set(args) - {s}
            if len(other_sets) > 0:
                other = Intersection(*other_sets)
                return Union(*(Intersection(arg, other) for arg in s.args))
            else:
                return Union(*s.args)

    for s in args:
        if s.is_Complement:
            args.remove(s)
            other_sets = args + [s.args[0]]
            return Complement(Intersection(*other_sets), s.args[1])

    from sympy.sets.handlers.intersection import intersection_sets

    # At this stage we are guaranteed not to have any
    # EmptySets, FiniteSets, or Unions in the intersection

    # ===== Pair-wise Rules =====
    # Here we depend on rules built into the constituent sets
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = intersection_sets(s, t)
                # This returns None if s does not know how to intersect
                # with t. Returns the newly intersected set otherwise

                if new_set is not None:
                    new_args = (args - {s, t}).union({new_set})
                    break
            if new_args:
                args = new_args
                break

    if len(args) == 1:
        return args.pop()
    else:
        return Intersection(*args, evaluate=False)


def _handle_finite_sets(op, x, y, commutative):
    # Handle finite sets:
    fs_args, other = sift([x, y], lambda x: isinstance(x, FiniteSet), binary=True)
    if len(fs_args) == 2:
        return FiniteSet(*[op(i, j) for i in fs_args[0] for j in fs_args[1]])
    elif len(fs_args) == 1:
        sets = [_apply_operation(op, other[0], i, commutative) for i in fs_args[0]]
        return Union(*sets)
    else:
        return None


def _apply_operation(op, x, y, commutative):
    from .fancysets import ImageSet
    d = Dummy('d')

    out = _handle_finite_sets(op, x, y, commutative)
    if out is None:
        out = op(x, y)

    if out is None and commutative:
        out = op(y, x)
    if out is None:
        _x, _y = symbols("x y")
        if isinstance(x, Set) and not isinstance(y, Set):
            out = ImageSet(Lambda(d, op(d, y)), x).doit()
        elif not isinstance(x, Set) and isinstance(y, Set):
            out = ImageSet(Lambda(d, op(x, d)), y).doit()
        else:
            out = ImageSet(Lambda((_x, _y), op(_x, _y)), x, y)
    return out


def set_add(x, y):
    from sympy.sets.handlers.add import _set_add
    return _apply_operation(_set_add, x, y, commutative=True)


def set_sub(x, y):
    from sympy.sets.handlers.add import _set_sub
    return _apply_operation(_set_sub, x, y, commutative=False)


def set_mul(x, y):
    from sympy.sets.handlers.mul import _set_mul
    return _apply_operation(_set_mul, x, y, commutative=True)


def set_div(x, y):
    from sympy.sets.handlers.mul import _set_div
    return _apply_operation(_set_div, x, y, commutative=False)


def set_pow(x, y):
    from sympy.sets.handlers.power import _set_pow
    return _apply_operation(_set_pow, x, y, commutative=False)


def set_function(f, x):
    from sympy.sets.handlers.functions import _set_function
    return _set_function(f, x)


class SetKind(Kind):
    """
    SetKind is kind for all Sets

    Every instance of Set will have kind ``SetKind`` parametrised by the kind
    of the elements of the ``Set``. The kind of the elements might be
    ``NumberKind``, or ``TupleKind`` or something else. When not all elements
    have the same kind then the kind of the elements will be given as
    ``UndefinedKind``.

    Parameters
    ==========

    element_kind: Kind (optional)
        The kind of the elements of the set. In a well defined set all elements
        will have the same kind. Otherwise the kind should
        :class:`sympy.core.kind.UndefinedKind`. The ``element_kind`` argument is optional but
        should only be omitted in the case of ``EmptySet`` whose kind is simply
        ``SetKind()``

    Examples
    ========

    >>> from sympy import Interval
    >>> Interval(1, 2).kind
    SetKind(NumberKind)
    >>> Interval(1,2).kind.element_kind
    NumberKind

    See Also
    ========

    sympy.core.kind.NumberKind
    sympy.matrices.kind.MatrixKind
    sympy.core.containers.TupleKind
    """
    def __new__(cls, element_kind=None):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        if not self.element_kind:
            return "SetKind()"
        else:
            return "SetKind(%s)" % self.element_kind
<|endoftext|>from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from .fancysets import ImageSet
from .sets import set_add, set_sub, set_mul, set_div, set_pow, set_function


class SetExpr(Expr):
    """An expression that can take on values of a set.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet
    >>> from sympy.sets.setexpr import SetExpr

    >>> a = SetExpr(Interval(0, 5))
    >>> b = SetExpr(FiniteSet(1, 10))
    >>> (a + b).set
    Union(Interval(1, 6), Interval(10, 15))
    >>> (2*a + b).set
    Interval(1, 20)
    """
    _op_priority = 11.0

    def __new__(cls, setarg):
        return Expr.__new__(cls, setarg)

    set = property(lambda self: self.args[0])

    def _latex(self, printer):
        return r"SetExpr\left({}\right)".format(printer._print(self.set))

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return _setexpr_apply_operation(set_add, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return _setexpr_apply_operation(set_add, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return _setexpr_apply_operation(set_mul, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return _setexpr_apply_operation(set_mul, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return _setexpr_apply_operation(set_sub, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return _setexpr_apply_operation(set_sub, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return _setexpr_apply_operation(set_pow, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return _setexpr_apply_operation(set_pow, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return _setexpr_apply_operation(set_div, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return _setexpr_apply_operation(set_div, other, self)

    def _eval_func(self, func):
        # TODO: this could be implemented straight into `imageset`:
        res = set_function(func, self.set)
        if res is None:
            return SetExpr(ImageSet(func, self.set))
        return SetExpr(res)


def _setexpr_apply_operation(op, x, y):
    if isinstance(x, SetExpr):
        x = x.set
    if isinstance(y, SetExpr):
        y = y.set
    out = op(x, y)
    return SetExpr(out)
<|endoftext|>from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda, BadSignatureError
from sympy.core.logic import fuzzy_bool
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import And, as_Boolean
from sympy.utilities.iterables import sift, flatten, has_dups
from sympy.utilities.exceptions import sympy_deprecation_warning
from .contains import Contains
from .sets import Set, Union, FiniteSet, SetKind


adummy = Dummy('conditionset')


class ConditionSet(Set):
    r"""
    Set of elements which satisfies a given condition.

    .. math:: \{x \mid \textrm{condition}(x) = \texttt{True}, x \in S\}

    Examples
    ========

    >>> from sympy import Symbol, S, ConditionSet, pi, Eq, sin, Interval
    >>> from sympy.abc import x, y, z

    >>> sin_sols = ConditionSet(x, Eq(sin(x), 0), Interval(0, 2*pi))
    >>> 2*pi in sin_sols
    True
    >>> pi/2 in sin_sols
    False
    >>> 3*pi in sin_sols
    False
    >>> 5 in ConditionSet(x, x**2 > 4, S.Reals)
    True

    If the value is not in the base set, the result is false:

    >>> 5 in ConditionSet(x, x**2 > 4, Interval(2, 4))
    False

    Notes
    =====

    Symbols with assumptions should be avoided or else the
    condition may evaluate without consideration of the set:

    >>> n = Symbol('n', negative=True)
    >>> cond = (n > 0); cond
    False
    >>> ConditionSet(n, cond, S.Integers)
    EmptySet

    Only free symbols can be changed by using `subs`:

    >>> c = ConditionSet(x, x < 1, {x, z})
    >>> c.subs(x, y)
    ConditionSet(x, x < 1, {y, z})

    To check if ``pi`` is in ``c`` use:

    >>> pi in c
    False

    If no base set is specified, the universal set is implied:

    >>> ConditionSet(x, x < 1).base_set
    UniversalSet

    Only symbols or symbol-like expressions can be used:

    >>> ConditionSet(x + 1, x + 1 < 1, S.Integers)
    Traceback (most recent call last):
    ...
    ValueError: non-symbol dummy not recognized in condition

    When the base set is a ConditionSet, the symbols will be
    unified if possible with preference for the outermost symbols:

    >>> ConditionSet(x, x < y, ConditionSet(z, z + y < 2, S.Integers))
    ConditionSet(x, (x < y) & (x + y < 2), Integers)

    """
    def __new__(cls, sym, condition, base_set=S.UniversalSet):
        sym = _sympify(sym)
        flat = flatten([sym])
        if has_dups(flat):
            raise BadSignatureError("Duplicate symbols detected")
        base_set = _sympify(base_set)
        if not isinstance(base_set, Set):
            raise TypeError(
                'base set should be a Set object, not %s' % base_set)
        condition = _sympify(condition)

        if isinstance(condition, FiniteSet):
            condition_orig = condition
            temp = (Eq(lhs, 0) for lhs in condition)
            condition = And(*temp)
            sympy_deprecation_warning(
                f"""
Using a set for the condition in ConditionSet is deprecated. Use a boolean
instead.

In this case, replace

    {condition_orig}

with

    {condition}
""",
                deprecated_since_version='1.5',
                active_deprecations_target="deprecated-conditionset-set",
                )

        condition = as_Boolean(condition)

        if condition is S.true:
            return base_set

        if condition is S.false:
            return S.EmptySet

        if base_set is S.EmptySet:
            return S.EmptySet

        # no simple answers, so now check syms
        for i in flat:
            if not getattr(i, '_diff_wrt', False):
                raise ValueError('`%s` is not symbol-like' % i)

        if base_set.contains(sym) is S.false:
            raise TypeError('sym `%s` is not in base_set `%s`' % (sym, base_set))

        know = None
        if isinstance(base_set, FiniteSet):
            sifted = sift(
                base_set, lambda _: fuzzy_bool(condition.subs(sym, _)))
            if sifted[None]:
                know = FiniteSet(*sifted[True])
                base_set = FiniteSet(*sifted[None])
            else:
                return FiniteSet(*sifted[True])

        if isinstance(base_set, cls):
            s, c, b = base_set.args
            def sig(s):
                return cls(s, Eq(adummy, 0)).as_dummy().sym
            sa, sb = map(sig, (sym, s))
            if sa != sb:
                raise BadSignatureError('sym does not match sym of base set')
            reps = dict(zip(flatten([sym]), flatten([s])))
            if s == sym:
                condition = And(condition, c)
                base_set = b
            elif not c.free_symbols & sym.free_symbols:
                reps = {v: k for k, v in reps.items()}
                condition = And(condition, c.xreplace(reps))
                base_set = b
            elif not condition.free_symbols & s.free_symbols:
                sym = sym.xreplace(reps)
                condition = And(condition.xreplace(reps), c)
                base_set = b

        # flatten ConditionSet(Contains(ConditionSet())) expressions
        if isinstance(condition, Contains) and (sym == condition.args[0]):
            if isinstance(condition.args[1], Set):
                return condition.args[1].intersect(base_set)

        rv = Basic.__new__(cls, sym, condition, base_set)
        return rv if know is None else Union(know, rv)

    sym = property(lambda self: self.args[0])
    condition = property(lambda self: self.args[1])
    base_set = property(lambda self: self.args[2])

    @property
    def free_symbols(self):
        cond_syms = self.condition.free_symbols - self.sym.free_symbols
        return cond_syms | self.base_set.free_symbols

    @property
    def bound_symbols(self):
        return flatten([self.sym])

    def _contains(self, other):
        def ok_sig(a, b):
            tuples = [isinstance(i, Tuple) for i in (a, b)]
            c = tuples.count(True)
            if c == 1:
                return False
            if c == 0:
                return True
            return len(a) == len(b) and all(
                ok_sig(i, j) for i, j in zip(a, b))
        if not ok_sig(self.sym, other):
            return S.false

        # try doing base_cond first and return
        # False immediately if it is False
        base_cond = Contains(other, self.base_set)
        if base_cond is S.false:
            return S.false

        # Substitute other into condition. This could raise e.g. for
        # ConditionSet(x, 1/x >= 0, Reals).contains(0)
        lamda = Lambda((self.sym,), self.condition)
        try:
            lambda_cond = lamda(other)
        except TypeError:
            return None
        else:
            return And(base_cond, lambda_cond)

    def as_relational(self, other):
        f = Lambda(self.sym, self.condition)
        if isinstance(self.sym, Tuple):
            f = f(*other)
        else:
            f = f(other)
        return And(f, self.base_set.contains(other))

    def _eval_subs(self, old, new):
        sym, cond, base = self.args
        dsym = sym.subs(old, adummy)
        insym = dsym.has(adummy)
        # prioritize changing a symbol in the base
        newbase = base.subs(old, new)
        if newbase != base:
            if not insym:
                cond = cond.subs(old, new)
            return self.func(sym, cond, newbase)
        if insym:
            pass  # no change of bound symbols via subs
        elif getattr(new, '_diff_wrt', False):
            cond = cond.subs(old, new)
        else:
            pass  # let error about the symbol raise from __new__
        return self.func(sym, cond, base)

    def _kind(self):
        return SetKind(self.sym.kind)
<|endoftext|>from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.logic import fuzzy_and, fuzzy_bool, fuzzy_not, fuzzy_or
from sympy.core.relational import Eq
from sympy.sets.sets import FiniteSet, Interval, Set, Union, ProductSet
from sympy.sets.fancysets import Complexes, Reals, Range, Rationals
from sympy.multipledispatch import Dispatcher


_inf_sets = [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals, S.Complexes]


is_subset_sets = Dispatcher('is_subset_sets')


@is_subset_sets.register(Set, Set)
def _(a, b):
    return None

@is_subset_sets.register(Interval, Interval)
def _(a, b):
    # This is correct but can be made more comprehensive...
    if fuzzy_bool(a.start < b.start):
        return False
    if fuzzy_bool(a.end > b.end):
        return False
    if (b.left_open and not a.left_open and fuzzy_bool(Eq(a.start, b.start))):
        return False
    if (b.right_open and not a.right_open and fuzzy_bool(Eq(a.end, b.end))):
        return False

@is_subset_sets.register(Interval, FiniteSet)
def _(a_interval, b_fs):
    # An Interval can only be a subset of a finite set if it is finite
    # which can only happen if it has zero measure.
    if fuzzy_not(a_interval.measure.is_zero):
        return False

@is_subset_sets.register(Interval, Union)
def _(a_interval, b_u):
    if all(isinstance(s, (Interval, FiniteSet)) for s in b_u.args):
        intervals = [s for s in b_u.args if isinstance(s, Interval)]
        if all(fuzzy_bool(a_interval.start < s.start) for s in intervals):
            return False
        if all(fuzzy_bool(a_interval.end > s.end) for s in intervals):
            return False
        if a_interval.measure.is_nonzero:
            no_overlap = lambda s1, s2: fuzzy_or([
                    fuzzy_bool(s1.end <= s2.start),
                    fuzzy_bool(s1.start >= s2.end),
                    ])
            if all(no_overlap(s, a_interval) for s in intervals):
                return False

@is_subset_sets.register(Range, Range)
def _(a, b):
    if a.step == b.step == 1:
        return fuzzy_and([fuzzy_bool(a.start >= b.start),
                          fuzzy_bool(a.stop <= b.stop)])

@is_subset_sets.register(Range, Interval)
def _(a_range, b_interval):
    if a_range.step.is_positive:
        if b_interval.left_open and a_range.inf.is_finite:
            cond_left = a_range.inf > b_interval.left
        else:
            cond_left = a_range.inf >= b_interval.left
        if b_interval.right_open and a_range.sup.is_finite:
            cond_right = a_range.sup < b_interval.right
        else:
            cond_right = a_range.sup <= b_interval.right
        return fuzzy_and([cond_left, cond_right])

@is_subset_sets.register(Range, FiniteSet)
def _(a_range, b_finiteset):
    try:
        a_size = a_range.size
    except ValueError:
        # symbolic Range of unknown size
        return None
    if a_size > len(b_finiteset):
        return False
    elif any(arg.has(Symbol) for arg in a_range.args):
        return fuzzy_and(b_finiteset.contains(x) for x in a_range)
    else:
        # Checking A \ B == EmptySet is more efficient than repeated naive
        # membership checks on an arbitrary FiniteSet.
        a_set = set(a_range)
        b_remaining = len(b_finiteset)
        # Symbolic expressions and numbers of unknown type (integer or not) are
        # all counted as "candidates", i.e. *potentially* matching some a in
        # a_range.
        cnt_candidate = 0
        for b in b_finiteset:
            if b.is_Integer:
                a_set.discard(b)
            elif fuzzy_not(b.is_integer):
                pass
            else:
                cnt_candidate += 1
            b_remaining -= 1
            if len(a_set) > b_remaining + cnt_candidate:
                return False
            if len(a_set) == 0:
                return True
        return None

@is_subset_sets.register(Interval, Range)
def _(a_interval, b_range):
    if a_interval.measure.is_extended_nonzero:
        return False

@is_subset_sets.register(Interval, Rationals)
def _(a_interval, b_rationals):
    if a_interval.measure.is_extended_nonzero:
        return False

@is_subset_sets.register(Range, Complexes)
def _(a, b):
    return True

@is_subset_sets.register(Complexes, Interval)
def _(a, b):
    return False

@is_subset_sets.register(Complexes, Range)
def _(a, b):
    return False

@is_subset_sets.register(Complexes, Rationals)
def _(a, b):
    return False

@is_subset_sets.register(Rationals, Reals)
def _(a, b):
    return True

@is_subset_sets.register(Rationals, Range)
def _(a, b):
    return False

@is_subset_sets.register(ProductSet, FiniteSet)
def _(a_ps, b_fs):
    return fuzzy_and(b_fs.contains(x) for x in a_ps)
<|endoftext|>from sympy.core.basic import _aresame
from sympy.core.function import Lambda, expand_complex
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor, ceiling
from sympy.sets.fancysets import ComplexRegion
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.multipledispatch import Dispatcher
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import (Integers, Naturals, Reals, Range,
    ImageSet, Rationals)
from sympy.sets.sets import EmptySet, UniversalSet, imageset, ProductSet
from sympy.simplify.radsimp import numer


intersection_sets = Dispatcher('intersection_sets')


@intersection_sets.register(ConditionSet, ConditionSet)
def _(a, b):
    return None

@intersection_sets.register(ConditionSet, Set)
def _(a, b):
    return ConditionSet(a.sym, a.condition, Intersection(a.base_set, b))

@intersection_sets.register(Naturals, Integers)
def _(a, b):
    return a

@intersection_sets.register(Naturals, Naturals)
def _(a, b):
    return a if a is S.Naturals else b

@intersection_sets.register(Interval, Naturals)
def _(a, b):
    return intersection_sets(b, a)

@intersection_sets.register(ComplexRegion, Set)
def _(self, other):
    if other.is_ComplexRegion:
        # self in rectangular form
        if (not self.polar) and (not other.polar):
            return ComplexRegion(Intersection(self.sets, other.sets))

        # self in polar form
        elif self.polar and other.polar:
            r1, theta1 = self.a_interval, self.b_interval
            r2, theta2 = other.a_interval, other.b_interval
            new_r_interval = Intersection(r1, r2)
            new_theta_interval = Intersection(theta1, theta2)

            # 0 and 2*Pi means the same
            if ((2*S.Pi in theta1 and S.Zero in theta2) or
               (2*S.Pi in theta2 and S.Zero in theta1)):
                new_theta_interval = Union(new_theta_interval,
                                           FiniteSet(0))
            return ComplexRegion(new_r_interval*new_theta_interval,
                                polar=True)


    if other.is_subset(S.Reals):
        new_interval = []
        x = symbols("x", cls=Dummy, real=True)

        # self in rectangular form
        if not self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

        # self in polar form
        elif self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
                if S.Pi in element.args[1]:
                    new_interval.append(ImageSet(Lambda(x, -x), element.args[0]))
                if S.Zero in element.args[0]:
                    new_interval.append(FiniteSet(0))
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

@intersection_sets.register(Integers, Reals)
def _(a, b):
    return a

@intersection_sets.register(Range, Interval)
def _(a, b):
    # Check that there are no symbolic arguments
    if not all(i.is_number for i in a.args + b.args[:2]):
        return

    # In case of null Range, return an EmptySet.
    if a.size == 0:
        return S.EmptySet

    # trim down to self's size, and represent
    # as a Range with step 1.
    start = ceiling(max(b.inf, a.inf))
    if start not in b:
        start += 1
    end = floor(min(b.sup, a.sup))
    if end not in b:
        end -= 1
    return intersection_sets(a, Range(start, end + 1))

@intersection_sets.register(Range, Naturals)
def _(a, b):
    return intersection_sets(a, Interval(b.inf, S.Infinity))

@intersection_sets.register(Range, Range)
def _(a, b):
    # Check that there are no symbolic range arguments
    if not all(all(v.is_number for v in r.args) for r in [a, b]):
        return None

    # non-overlap quick exits
    if not b:
        return S.EmptySet
    if not a:
        return S.EmptySet
    if b.sup < a.inf:
        return S.EmptySet
    if b.inf > a.sup:
        return S.EmptySet

    # work with finite end at the start
    r1 = a
    if r1.start.is_infinite:
        r1 = r1.reversed
    r2 = b
    if r2.start.is_infinite:
        r2 = r2.reversed

    # If both ends are infinite then it means that one Range is just the set
    # of all integers (the step must be 1).
    if r1.start.is_infinite:
        return b
    if r2.start.is_infinite:
        return a

    from sympy.solvers.diophantine.diophantine import diop_linear

    # this equation represents the values of the Range;
    # it's a linear equation
    eq = lambda r, i: r.start + i*r.step

    # we want to know when the two equations might
    # have integer solutions so we use the diophantine
    # solver
    va, vb = diop_linear(eq(r1, Dummy('a')) - eq(r2, Dummy('b')))

    # check for no solution
    no_solution = va is None and vb is None
    if no_solution:
        return S.EmptySet

    # there is a solution
    # -------------------

    # find the coincident point, c
    a0 = va.as_coeff_Add()[0]
    c = eq(r1, a0)

    # find the first point, if possible, in each range
    # since c may not be that point
    def _first_finite_point(r1, c):
        if c == r1.start:
            return c
        # st is the signed step we need to take to
        # get from c to r1.start
        st = sign(r1.start - c)*step
        # use Range to calculate the first point:
        # we want to get as close as possible to
        # r1.start; the Range will not be null since
        # it will at least contain c
        s1 = Range(c, r1.start + st, st)[-1]
        if s1 == r1.start:
            pass
        else:
            # if we didn't hit r1.start then, if the
            # sign of st didn't match the sign of r1.step
            # we are off by one and s1 is not in r1
            if sign(r1.step) != sign(st):
                s1 -= st
        if s1 not in r1:
            return
        return s1

    # calculate the step size of the new Range
    step = abs(ilcm(r1.step, r2.step))
    s1 = _first_finite_point(r1, c)
    if s1 is None:
        return S.EmptySet
    s2 = _first_finite_point(r2, c)
    if s2 is None:
        return S.EmptySet

    # replace the corresponding start or stop in
    # the original Ranges with these points; the
    # result must have at least one point since
    # we know that s1 and s2 are in the Ranges
    def _updated_range(r, first):
        st = sign(r.step)*step
        if r.start.is_finite:
            rv = Range(first, r.stop, st)
        else:
            rv = Range(r.start, first + st, st)
        return rv
    r1 = _updated_range(a, s1)
    r2 = _updated_range(b, s2)

    # work with them both in the increasing direction
    if sign(r1.step) < 0:
        r1 = r1.reversed
    if sign(r2.step) < 0:
        r2 = r2.reversed

    # return clipped Range with positive step; it
    # can't be empty at this point
    start = max(r1.start, r2.start)
    stop = min(r1.stop, r2.stop)
    return Range(start, stop, step)


@intersection_sets.register(Range, Integers)
def _(a, b):
    return a


@intersection_sets.register(Range, Rationals)
def _(a, b):
    return a


@intersection_sets.register(ImageSet, Set)
def _(self, other):
    from sympy.solvers.diophantine import diophantine

    # Only handle the straight-forward univariate case
    if (len(self.lamda.variables) > 1
            or self.lamda.signature != self.lamda.variables):
        return None
    base_set = self.base_sets[0]

    # Intersection between ImageSets with Integers as base set
    # For {f(n) : n in Integers} & {g(m) : m in Integers} we solve the
    # diophantine equations f(n)=g(m).
    # If the solutions for n are {h(t) : t in Integers} then we return
    # {f(h(t)) : t in integers}.
    # If the solutions for n are {n_1, n_2, ..., n_k} then we return
    # {f(n_i) : 1 <= i <= k}.
    if base_set is S.Integers:
        gm = None
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            var = other.lamda.variables[0]
            # Symbol of second ImageSet lambda must be distinct from first
            m = Dummy('m')
            gm = gm.subs(var, m)
        elif other is S.Integers:
            m = gm = Dummy('m')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            try:
                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
            except (TypeError, NotImplementedError):
                # TypeError if equation not polynomial with rational coeff.
                # NotImplementedError if correct format but no solver.
                return
            # 3 cases are possible for solns:
            # - empty set,
            # - one or more parametric (infinite) solutions,
            # - a finite number of (non-parametric) solution couples.
            # Among those, there is one type of solution set that is
            # not helpful here: multiple parametric solutions.
            if len(solns) == 0:
                return S.EmptySet
            elif any(s.free_symbols for tupl in solns for s in tupl):
                if len(solns) == 1:
                    soln, solm = solns[0]
                    (t,) = soln.free_symbols
                    expr = fn.subs(n, soln.subs(t, n)).expand()
                    return imageset(Lambda(n, expr), S.Integers)
                else:
                    return
            else:
                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))

    if other == S.Reals:
        from sympy.solvers.solvers import denoms, solve_linear

        def _solution_union(exprs, sym):
            # return a union of linear solutions to i in expr;
            # if i cannot be solved, use a ConditionSet for solution
            sols = []
            for i in exprs:
                x, xis = solve_linear(i, 0, [sym])
                if x == sym:
                    sols.append(FiniteSet(xis))
                else:
                    sols.append(ConditionSet(sym, Eq(i, 0)))
            return Union(*sols)

        f = self.lamda.expr
        n = self.lamda.variables[0]

        n_ = Dummy(n.name, real=True)
        f_ = f.subs(n, n_)

        re, im = f_.as_real_imag()
        im = expand_complex(im)

        re = re.subs(n_, n)
        im = im.subs(n_, n)
        ifree = im.free_symbols
        lam = Lambda(n, re)
        if im.is_zero:
            # allow re-evaluation
            # of self in this case to make
            # the result canonical
            pass
        elif im.is_zero is False:
            return S.EmptySet
        elif ifree != {n}:
            return None
        else:
            # univarite imaginary part in same variable;
            # use numer instead of as_numer_denom to keep
            # this as fast as possible while still handling
            # simple cases
            base_set &= _solution_union(
                Mul.make_args(numer(im)), n)
        # exclude values that make denominators 0
        base_set -= _solution_union(denoms(f), n)
        return imageset(lam, base_set)

    elif isinstance(other, Interval):
        from sympy.solvers.solveset import (invert_real, invert_complex,
                                            solveset)

        f = self.lamda.expr
        n = self.lamda.variables[0]
        new_inf, new_sup = None, None
        new_lopen, new_ropen = other.left_open, other.right_open

        if f.is_real:
            inverter = invert_real
        else:
            inverter = invert_complex

        g1, h1 = inverter(f, other.inf, n)
        g2, h2 = inverter(f, other.sup, n)

        if all(isinstance(i, FiniteSet) for i in (h1, h2)):
            if g1 == n:
                if len(h1) == 1:
                    new_inf = h1.args[0]
            if g2 == n:
                if len(h2) == 1:
                    new_sup = h2.args[0]
            # TODO: Design a technique to handle multiple-inverse
            # functions

            # Any of the new boundary values cannot be determined
            if any(i is None for i in (new_sup, new_inf)):
                return


            range_set = S.EmptySet

            if all(i.is_real for i in (new_sup, new_inf)):
                # this assumes continuity of underlying function
                # however fixes the case when it is decreasing
                if new_inf > new_sup:
                    new_inf, new_sup = new_sup, new_inf
                new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                range_set = base_set.intersect(new_interval)
            else:
                if other.is_subset(S.Reals):
                    solutions = solveset(f, n, S.Reals)
                    if not isinstance(range_set, (ImageSet, ConditionSet)):
                        range_set = solutions.intersect(other)
                    else:
                        return

            if range_set is S.EmptySet:
                return S.EmptySet
            elif isinstance(range_set, Range) and range_set.size is not S.Infinity:
                range_set = FiniteSet(*list(range_set))

            if range_set is not None:
                return imageset(Lambda(n, f), range_set)
            return
        else:
            return


@intersection_sets.register(ProductSet, ProductSet)
def _(a, b):
    if len(b.args) != len(a.args):
        return S.EmptySet
    return ProductSet(*(i.intersect(j) for i, j in zip(a.sets, b.sets)))


@intersection_sets.register(Interval, Interval)
def _(a, b):
    # handle (-oo, oo)
    infty = S.NegativeInfinity, S.Infinity
    if a == Interval(*infty):
        l, r = a.left, a.right
        if l.is_real or l in infty or r.is_real or r in infty:
            return b

    # We can't intersect [0,3] with [x,6] -- we don't know if x>0 or x<0
    if not a._is_comparable(b):
        return None

    empty = False

    if a.start <= b.end and b.start <= a.end:
        # Get topology right.
        if a.start < b.start:
            start = b.start
            left_open = b.left_open
        elif a.start > b.start:
            start = a.start
            left_open = a.left_open
        else:
            start = a.start
            if not _aresame(a.start, b.start):
                # For example Integer(2) != Float(2)
                # Prefer the Float boundary because Floats should be
                # contagious in calculations.
                if b.start.has(Float) and not a.start.has(Float):
                    start = b.start
                elif a.start.has(Float) and not b.start.has(Float):
                    start = a.start
                else:
                    #this is to ensure that if Eq(a.start, b.start) but
                    #type(a.start) != type(b.start) the order of a and b
                    #does not matter for the result
                    start = list(ordered([a,b]))[0].start
            left_open = a.left_open or b.left_open

        if a.end < b.end:
            end = a.end
            right_open = a.right_open
        elif a.end > b.end:
            end = b.end
            right_open = b.right_open
        else:
            # see above for logic with start
            end = a.end
            if not _aresame(a.end, b.end):
                if b.end.has(Float) and not a.end.has(Float):
                    end = b.end
                elif a.end.has(Float) and not b.end.has(Float):
                    end = a.end
                else:
                    end = list(ordered([a,b]))[0].end
            right_open = a.right_open or b.right_open

        if end - start == 0 and (left_open or right_open):
            empty = True
    else:
        empty = True

    if empty:
        return S.EmptySet

    return Interval(start, end, left_open, right_open)

@intersection_sets.register(EmptySet, Set)
def _(a, b):
    return S.EmptySet

@intersection_sets.register(UniversalSet, Set)
def _(a, b):
    return b

@intersection_sets.register(FiniteSet, FiniteSet)
def _(a, b):
    return FiniteSet(*(a._elements & b._elements))

@intersection_sets.register(FiniteSet, Set)
def _(a, b):
    try:
        return FiniteSet(*[el for el in a if el in b])
    except TypeError:
        return None  # could not evaluate `el in b` due to symbolic ranges.

@intersection_sets.register(Set, Set)
def _(a, b):
    return None

@intersection_sets.register(Integers, Rationals)
def _(a, b):
    return a

@intersection_sets.register(Naturals, Rationals)
def _(a, b):
    return a

@intersection_sets.register(Rationals, Reals)
def _(a, b):
    return a

def _intlike_interval(a, b):
    try:
        if b._inf is S.NegativeInfinity and b._sup is S.Infinity:
            return a
        s = Range(max(a.inf, ceiling(b.left)), floor(b.right) + 1)
        return intersection_sets(s, b)  # take out endpoints if open interval
    except ValueError:
        return None

@intersection_sets.register(Integers, Interval)
def _(a, b):
    return _intlike_interval(a, b)

@intersection_sets.register(Naturals, Interval)
def _(a, b):
    return _intlike_interval(a, b)
<|endoftext|>from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.sets.sets import (EmptySet, FiniteSet, Intersection,
    Interval, ProductSet, Set, Union, UniversalSet)
from sympy.sets.fancysets import (ComplexRegion, Naturals, Naturals0,
    Integers, Rationals, Reals)
from sympy.multipledispatch import Dispatcher


union_sets = Dispatcher('union_sets')


@union_sets.register(Naturals0, Naturals)
def _(a, b):
    return a

@union_sets.register(Rationals, Naturals)
def _(a, b):
    return a

@union_sets.register(Rationals, Naturals0)
def _(a, b):
    return a

@union_sets.register(Reals, Naturals)
def _(a, b):
    return a

@union_sets.register(Reals, Naturals0)
def _(a, b):
    return a

@union_sets.register(Reals, Rationals)
def _(a, b):
    return a

@union_sets.register(Integers, Set)
def _(a, b):
    intersect = Intersection(a, b)
    if intersect == a:
        return b
    elif intersect == b:
        return a

@union_sets.register(ComplexRegion, Set)
def _(a, b):
    if b.is_subset(S.Reals):
        # treat a subset of reals as a complex region
        b = ComplexRegion.from_real(b)

    if b.is_ComplexRegion:
        # a in rectangular form
        if (not a.polar) and (not b.polar):
            return ComplexRegion(Union(a.sets, b.sets))
        # a in polar form
        elif a.polar and b.polar:
            return ComplexRegion(Union(a.sets, b.sets), polar=True)
    return None

@union_sets.register(EmptySet, Set)
def _(a, b):
    return b


@union_sets.register(UniversalSet, Set)
def _(a, b):
    return a

@union_sets.register(ProductSet, ProductSet)
def _(a, b):
    if b.is_subset(a):
        return a
    if len(b.sets) != len(a.sets):
        return None
    if len(a.sets) == 2:
        a1, a2 = a.sets
        b1, b2 = b.sets
        if a1 == b1:
            return a1 * Union(a2, b2)
        if a2 == b2:
            return Union(a1, b1) * a2
    return None

@union_sets.register(ProductSet, Set)
def _(a, b):
    if b.is_subset(a):
        return a
    return None

@union_sets.register(Interval, Interval)
def _(a, b):
    if a._is_comparable(b):
        # Non-overlapping intervals
        end = Min(a.end, b.end)
        start = Max(a.start, b.start)
        if (end < start or
           (end == start and (end not in a and end not in b))):
            return None
        else:
            start = Min(a.start, b.start)
            end = Max(a.end, b.end)

            left_open = ((a.start != start or a.left_open) and
                         (b.start != start or b.left_open))
            right_open = ((a.end != end or a.right_open) and
                          (b.end != end or b.right_open))
            return Interval(start, end, left_open, right_open)

@union_sets.register(Interval, UniversalSet)
def _(a, b):
    return S.UniversalSet

@union_sets.register(Interval, Set)
def _(a, b):
    # If I have open end points and these endpoints are contained in b
    # But only in case, when endpoints are finite. Because
    # interval does not contain oo or -oo.
    open_left_in_b_and_finite = (a.left_open and
                                     sympify(b.contains(a.start)) is S.true and
                                     a.start.is_finite)
    open_right_in_b_and_finite = (a.right_open and
                                      sympify(b.contains(a.end)) is S.true and
                                      a.end.is_finite)
    if open_left_in_b_and_finite or open_right_in_b_and_finite:
        # Fill in my end points and return
        open_left = a.left_open and a.start not in b
        open_right = a.right_open and a.end not in b
        new_a = Interval(a.start, a.end, open_left, open_right)
        return {new_a, b}
    return None

@union_sets.register(FiniteSet, FiniteSet)
def _(a, b):
    return FiniteSet(*(a._elements | b._elements))

@union_sets.register(FiniteSet, Set)
def _(a, b):
    # If `b` set contains one of my elements, remove it from `a`
    if any(b.contains(x) == True for x in a):
        return {
            FiniteSet(*[x for x in a if b.contains(x) != True]), b}
    return None

@union_sets.register(Set, Set)
def _(a, b):
    return None
<|endoftext|>from sympy.core.relational import Eq, is_eq
from sympy.core.basic import Basic
from sympy.core.logic import fuzzy_and, fuzzy_bool
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.sets.sets import tfn, ProductSet, Interval, FiniteSet, Set


@dispatch(Interval, FiniteSet) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return False


@dispatch(FiniteSet, Interval) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return False


@dispatch(Interval, Interval) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return And(Eq(lhs.left, rhs.left),
               Eq(lhs.right, rhs.right),
               lhs.left_open == rhs.left_open,
               lhs.right_open == rhs.right_open)

@dispatch(FiniteSet, FiniteSet) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    def all_in_both():
        s_set = set(lhs.args)
        o_set = set(rhs.args)
        yield fuzzy_and(lhs._contains(e) for e in o_set - s_set)
        yield fuzzy_and(rhs._contains(e) for e in s_set - o_set)

    return tfn[fuzzy_and(all_in_both())]


@dispatch(ProductSet, ProductSet) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    if len(lhs.sets) != len(rhs.sets):
        return False

    eqs = (is_eq(x, y) for x, y in zip(lhs.sets, rhs.sets))
    return tfn[fuzzy_and(map(fuzzy_bool, eqs))]


@dispatch(Set, Basic) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return False


@dispatch(Set, Set) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return tfn[fuzzy_and(a.is_subset(b) for a, b in [(lhs, rhs), (rhs, lhs)])]
<|endoftext|>from sympy.core import Basic, Expr
from sympy.core.function import Lambda
from sympy.core.numbers import oo, Infinity, NegativeInfinity, Zero, Integer
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.sets.fancysets import ImageSet
from sympy.sets.setexpr import set_div
from sympy.sets.sets import Set, Interval, FiniteSet, Union
from sympy.multipledispatch import Dispatcher


_x, _y = symbols("x y")


_set_pow = Dispatcher('_set_pow')


@_set_pow.register(Basic, Basic)
def _(x, y):
    return None

@_set_pow.register(Set, Set)
def _(x, y):
    return ImageSet(Lambda((_x, _y), (_x ** _y)), x, y)

@_set_pow.register(Expr, Expr)
def _(x, y):
    return x**y

@_set_pow.register(Interval, Zero)
def _(x, z):
    return FiniteSet(S.One)

@_set_pow.register(Interval, Integer)
def _(x, exponent):
    """
    Powers in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    s1 = x.start**exponent
    s2 = x.end**exponent
    if ((s2 > s1) if exponent > 0 else (x.end > -x.start)) == True:
        left_open = x.left_open
        right_open = x.right_open
        # TODO: handle unevaluated condition.
        sleft = s2
    else:
        # TODO: `s2 > s1` could be unevaluated.
        left_open = x.right_open
        right_open = x.left_open
        sleft = s1

    if x.start.is_positive:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)
    elif x.end.is_negative:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)

    # Case where x.start < 0 and x.end > 0:
    if exponent.is_odd:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(-oo, s1, True, x.left_open)
            return Union(Interval(-oo, s1, True, x.left_open), Interval(s2, oo, x.right_open))
        else:
            return Interval(s1, s2, x.left_open, x.right_open)
    elif exponent.is_even:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(s1, oo, x.left_open)
            return Interval(0, oo)
        else:
            return Interval(S.Zero, sleft, S.Zero not in x, left_open)

@_set_pow.register(Interval, Infinity)
def _(b, e):
    # TODO: add logic for open intervals?
    if b.start.is_nonnegative:
        if b.end < 1:
            return FiniteSet(S.Zero)
        if b.start > 1:
            return FiniteSet(S.Infinity)
        return Interval(0, oo)
    elif b.end.is_negative:
        if b.start > -1:
            return FiniteSet(S.Zero)
        if b.end < -1:
            return FiniteSet(-oo, oo)
        return Interval(-oo, oo)
    else:
        if b.start > -1:
            if b.end < 1:
                return FiniteSet(S.Zero)
            return Interval(0, oo)
        return Interval(-oo, oo)

@_set_pow.register(Interval, NegativeInfinity)
def _(b, e):
    return _set_pow(set_div(S.One, b), oo)
<|endoftext|>from sympy.core.numbers import oo, Infinity, NegativeInfinity
from sympy.core.singleton import S
from sympy.core import Basic, Expr
from sympy.multipledispatch import Dispatcher
from sympy.sets import Interval, FiniteSet



# XXX: The functions in this module are clearly not tested and are broken in a
# number of ways.

_set_add = Dispatcher('_set_add')
_set_sub = Dispatcher('_set_sub')


@_set_add.register(Basic, Basic)
def _(x, y):
    return None


@_set_add.register(Expr, Expr)
def _(x, y):
    return x+y


@_set_add.register(Interval, Interval)
def _(x, y):
    """
    Additions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    return Interval(x.start + y.start, x.end + y.end,
                    x.left_open or y.left_open, x.right_open or y.right_open)


@_set_add.register(Interval, Infinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet({S.Infinity})

@_set_add.register(Interval, NegativeInfinity)
def _(x, y):
    if x.end is S.Infinity:
        return Interval(-oo, oo)
    return FiniteSet({S.NegativeInfinity})


@_set_sub.register(Basic, Basic)
def _(x, y):
    return None


@_set_sub.register(Expr, Expr)
def _(x, y):
    return x-y


@_set_sub.register(Interval, Interval)
def _(x, y):
    """
    Subtractions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    return Interval(x.start - y.end, x.end - y.start,
                    x.left_open or y.right_open, x.right_open or y.left_open)


@_set_sub.register(Interval, Infinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet(-oo)

@_set_sub.register(Interval, NegativeInfinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet(-oo)
<|endoftext|>from sympy.core.singleton import S
from sympy.sets.sets import Set
from sympy.calculus.singularities import singularities
from sympy.core import Expr, Add
from sympy.core.function import Lambda, FunctionClass, diff, expand_mul
from sympy.core.numbers import Float, oo
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import true
from sympy.multipledispatch import Dispatcher
from sympy.sets import (imageset, Interval, FiniteSet, Union, ImageSet,
    Intersection, Range, Complement)
from sympy.sets.sets import EmptySet, is_function_invertible_in_set
from sympy.sets.fancysets import Integers, Naturals, Reals
from sympy.functions.elementary.exponential import match_real_imag


_x, _y = symbols("x y")

FunctionUnion = (FunctionClass, Lambda)

_set_function = Dispatcher('_set_function')


@_set_function.register(FunctionClass, Set)
def _(f, x):
    return None

@_set_function.register(FunctionUnion, FiniteSet)
def _(f, x):
    return FiniteSet(*map(f, x))

@_set_function.register(Lambda, Interval)
def _(f, x):
    from sympy.solvers.solveset import solveset
    from sympy.series import limit
    # TODO: handle functions with infinitely many solutions (eg, sin, tan)
    # TODO: handle multivariate functions

    expr = f.expr
    if len(expr.free_symbols) > 1 or len(f.variables) != 1:
        return
    var = f.variables[0]
    if not var.is_real:
        if expr.subs(var, Dummy(real=True)).is_real is False:
            return

    if expr.is_Piecewise:
        result = S.EmptySet
        domain_set = x
        for (p_expr, p_cond) in expr.args:
            if p_cond is true:
                intrvl = domain_set
            else:
                intrvl = p_cond.as_set()
                intrvl = Intersection(domain_set, intrvl)

            if p_expr.is_Number:
                image = FiniteSet(p_expr)
            else:
                image = imageset(Lambda(var, p_expr), intrvl)
            result = Union(result, image)

            # remove the part which has been `imaged`
            domain_set = Complement(domain_set, intrvl)
            if domain_set is S.EmptySet:
                break
        return result

    if not x.start.is_comparable or not x.end.is_comparable:
        return

    try:
        from sympy.polys.polyutils import _nsort
        sing = list(singularities(expr, var, x))
        if len(sing) > 1:
            sing = _nsort(sing)
    except NotImplementedError:
        return

    if x.left_open:
        _start = limit(expr, var, x.start, dir="+")
    elif x.start not in sing:
        _start = f(x.start)
    if x.right_open:
        _end = limit(expr, var, x.end, dir="-")
    elif x.end not in sing:
        _end = f(x.end)

    if len(sing) == 0:
        soln_expr = solveset(diff(expr, var), var)
        if not (isinstance(soln_expr, FiniteSet)
                or soln_expr is S.EmptySet):
            return
        solns = list(soln_expr)

        extr = [_start, _end] + [f(i) for i in solns
                                 if i.is_real and i in x]
        start, end = Min(*extr), Max(*extr)

        left_open, right_open = False, False
        if _start <= _end:
            # the minimum or maximum value can occur simultaneously
            # on both the edge of the interval and in some interior
            # point
            if start == _start and start not in solns:
                left_open = x.left_open
            if end == _end and end not in solns:
                right_open = x.right_open
        else:
            if start == _end and start not in solns:
                left_open = x.right_open
            if end == _start and end not in solns:
                right_open = x.left_open

        return Interval(start, end, left_open, right_open)
    else:
        return imageset(f, Interval(x.start, sing[0],
                                    x.left_open, True)) + \
            Union(*[imageset(f, Interval(sing[i], sing[i + 1], True, True))
                    for i in range(0, len(sing) - 1)]) + \
            imageset(f, Interval(sing[-1], x.end, True, x.right_open))

@_set_function.register(FunctionClass, Interval)
def _(f, x):
    if f == exp:
        return Interval(exp(x.start), exp(x.end), x.left_open, x.right_open)
    elif f == log:
        return Interval(log(x.start), log(x.end), x.left_open, x.right_open)
    return ImageSet(Lambda(_x, f(_x)), x)

@_set_function.register(FunctionUnion, Union)
def _(f, x):
    return Union(*(imageset(f, arg) for arg in x.args))

@_set_function.register(FunctionUnion, Intersection)
def _(f, x):
    # If the function is invertible, intersect the maps of the sets.
    if is_function_invertible_in_set(f, x):
        return Intersection(*(imageset(f, arg) for arg in x.args))
    else:
        return ImageSet(Lambda(_x, f(_x)), x)

@_set_function.register(FunctionUnion, EmptySet)
def _(f, x):
    return x

@_set_function.register(FunctionUnion, Set)
def _(f, x):
    return ImageSet(Lambda(_x, f(_x)), x)

@_set_function.register(FunctionUnion, Range)
def _(f, self):
    if not self:
        return S.EmptySet
    if not isinstance(f.expr, Expr):
        return
    if self.size == 1:
        return FiniteSet(f(self[0]))
    if f is S.IdentityFunction:
        return self

    x = f.variables[0]
    expr = f.expr
    # handle f that is linear in f's variable
    if x not in expr.free_symbols or x in expr.diff(x).free_symbols:
        return
    if self.start.is_finite:
        F = f(self.step*x + self.start)  # for i in range(len(self))
    else:
        F = f(-self.step*x + self[-1])
    F = expand_mul(F)
    if F != expr:
        return imageset(x, F, Range(self.size))

@_set_function.register(FunctionUnion, Integers)
def _(f, self):
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    n = f.variables[0]
    if expr == abs(n):
        return S.Naturals0

    # f(x) + c and f(-x) + c cover the same integers
    # so choose the form that has the fewest negatives
    c = f(0)
    fx = f(n) - c
    f_x = f(-n) - c
    neg_count = lambda e: sum(_.could_extract_minus_sign()
        for _ in Add.make_args(e))
    if neg_count(f_x) < neg_count(fx):
        expr = f_x + c

    a = Wild('a', exclude=[n])
    b = Wild('b', exclude=[n])
    match = expr.match(a*n + b)
    if match and match[a] and (
            not match[a].atoms(Float) and
            not match[b].atoms(Float)):
        # canonical shift
        a, b = match[a], match[b]
        if a in [1, -1]:
            # drop integer addends in b
            nonint = []
            for bi in Add.make_args(b):
                if not bi.is_integer:
                    nonint.append(bi)
            b = Add(*nonint)
        if b.is_number and a.is_real:
            # avoid Mod for complex numbers, #11391
            br, bi = match_real_imag(b)
            if br and br.is_comparable and a.is_comparable:
                br %= a
                b = br + S.ImaginaryUnit*bi
        elif b.is_number and a.is_imaginary:
            br, bi = match_real_imag(b)
            ai = a/S.ImaginaryUnit
            if bi and bi.is_comparable and ai.is_comparable:
                bi %= ai
                b = br + S.ImaginaryUnit*bi
        expr = a*n + b

    if expr != f.expr:
        return ImageSet(Lambda(n, expr), S.Integers)


@_set_function.register(FunctionUnion, Naturals)
def _(f, self):
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    x = f.variables[0]
    if not expr.free_symbols - {x}:
        if expr == abs(x):
            if self is S.Naturals:
                return self
            return S.Naturals0
        step = expr.coeff(x)
        c = expr.subs(x, 0)
        if c.is_Integer and step.is_Integer and expr == step*x + c:
            if self is S.Naturals:
                c += step
            if step > 0:
                if step == 1:
                    if c == 0:
                        return S.Naturals0
                    elif c == 1:
                        return S.Naturals
                return Range(c, oo, step)
            return Range(c, -oo, step)


@_set_function.register(FunctionUnion, Reals)
def _(f, self):
    expr = f.expr
    if not isinstance(expr, Expr):
        return
    return _set_function(f, Interval(-oo, oo))
<|endoftext|>from sympy.core import Basic, Expr
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.multipledispatch import Dispatcher
from sympy.sets.setexpr import set_mul
from sympy.sets.sets import Interval, Set


_x, _y = symbols("x y")


_set_mul = Dispatcher('_set_mul')
_set_div = Dispatcher('_set_div')


@_set_mul.register(Basic, Basic)
def _(x, y):
    return None

@_set_mul.register(Set, Set)
def _(x, y):
    return None

@_set_mul.register(Expr, Expr)
def _(x, y):
    return x*y

@_set_mul.register(Interval, Interval)
def _(x, y):
    """
    Multiplications in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    # TODO: some intervals containing 0 and oo will fail as 0*oo returns nan.
    comvals = (
        (x.start * y.start, bool(x.left_open or y.left_open)),
        (x.start * y.end, bool(x.left_open or y.right_open)),
        (x.end * y.start, bool(x.right_open or y.left_open)),
        (x.end * y.end, bool(x.right_open or y.right_open)),
    )
    # TODO: handle symbolic intervals
    minval, minopen = min(comvals)
    maxval, maxopen = max(comvals)
    return Interval(
        minval,
        maxval,
        minopen,
        maxopen
    )

@_set_div.register(Basic, Basic)
def _(x, y):
    return None

@_set_div.register(Expr, Expr)
def _(x, y):
    return x/y

@_set_div.register(Set, Set)
def _(x, y):
    return None

@_set_div.register(Interval, Interval)
def _(x, y):
    """
    Divisions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    if (y.start*y.end).is_negative:
        return Interval(-oo, oo)
    if y.start == 0:
        s2 = oo
    else:
        s2 = 1/y.start
    if y.end == 0:
        s1 = -oo
    else:
        s1 = 1/y.end
    return set_mul(x, Interval(s1, s2, y.right_open, y.left_open))
<|endoftext|>"""
.. deprecated:: 1.10

   ``sympy.core.compatibility`` is deprecated. See
   :ref:`sympy-core-compatibility`.

Reimplementations of constructs introduced in later versions of Python than
we support. Also some functions that are needed SymPy-wide and are located
here for easy import.

"""


from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("""
The sympy.core.compatibility submodule is deprecated.

This module was only ever intended for internal use. Some of the functions
that were in this module are available from the top-level SymPy namespace,
i.e.,

    from sympy import ordered, default_sort_key

The remaining were only intended for internal SymPy use and should not be used
by user code.
""",
                          deprecated_since_version="1.10",
                          active_deprecations_target="deprecated-sympy-core-compatibility",
                          )


from .sorting import ordered, _nodes, default_sort_key # noqa:F401
from sympy.utilities.misc import as_int as _as_int # noqa:F401
from sympy.utilities.iterables import iterable, is_sequence, NotIterable # noqa:F401
<|endoftext|>from __future__ import annotations

from .basic import Atom, Basic
from .coreerrors import LazyExceptionMessage
from .sorting import ordered
from .evalf import EvalfMixin
from .function import AppliedUndef
from .numbers import int_valued
from .singleton import S
from .sympify import _sympify, SympifyError
from .parameters import global_parameters
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from sympy.utilities.misc import filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning


__all__ = (
    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge',
    'Relational', 'Equality', 'Unequality', 'StrictLessThan', 'LessThan',
    'StrictGreaterThan', 'GreaterThan',
)

from .expr import Expr
from sympy.multipledispatch import dispatch
from .containers import Tuple
from .symbol import Symbol


def _nontrivBool(side):
    return isinstance(side, Boolean) and \
           not isinstance(side, Atom)


# Note, see issue 4986.  Ideally, we wouldn't want to subclass both Boolean
# and Expr.
# from .. import Expr


def _canonical(cond):
    # return a condition in which all relationals are canonical
    reps = {r: r.canonical for r in cond.atoms(Relational)}
    return cond.xreplace(reps)
    # XXX: AttributeError was being caught here but it wasn't triggered by any of
    # the tests so I've removed it...


def _canonical_coeff(rel):
    # return -2*x + 1 < 0 as x > 1/2
    # XXX make this part of Relational.canonical?
    rel = rel.canonical
    if not rel.is_Relational or rel.rhs.is_Boolean:
        return rel  # Eq(x, True)
    if not isinstance(rel.lhs, Expr):
        return rel.reversed  # e.g.: Eq(True, x) -> Eq(x, True)
    b, l = rel.lhs.as_coeff_Add(rational=True)
    m, lhs = l.as_coeff_Mul(rational=True)
    rhs = (rel.rhs - b)/m
    if m < 0:
        return rel.reversed.func(lhs, rhs)
    return rel.func(lhs, rhs)


class Relational(Boolean, EvalfMixin):
    """Base class for all relation types.

    Explanation
    ===========

    Subclasses of Relational should generally be instantiated directly, but
    Relational can be instantiated with a valid ``rop`` value to dispatch to
    the appropriate subclass.

    Parameters
    ==========

    rop : str or None
        Indicates what subclass to instantiate.  Valid values can be found
        in the keys of Relational.ValidRelationOperator.

    Examples
    ========

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
    Eq(y, x**2 + x)

    A relation's type can be defined upon creation using ``rop``.
    The relation type of an existing expression can be obtained
    using its ``rel_op`` property.
    Here is a table of all the relation types, along with their
    ``rop`` and ``rel_op`` values:

    +---------------------+----------------------------+------------+
    |Relation             |``rop``                     |``rel_op``  |
    +=====================+============================+============+
    |``Equality``         |``==`` or ``eq`` or ``None``|``==``      |
    +---------------------+----------------------------+------------+
    |``Unequality``       |``!=`` or ``ne``            |``!=``      |
    +---------------------+----------------------------+------------+
    |``GreaterThan``      |``>=`` or ``ge``            |``>=``      |
    +---------------------+----------------------------+------------+
    |``LessThan``         |``<=`` or ``le``            |``<=``      |
    +---------------------+----------------------------+------------+
    |``StrictGreaterThan``|``>`` or ``gt``             |``>``       |
    +---------------------+----------------------------+------------+
    |``StrictLessThan``   |``<`` or ``lt``             |``<``       |
    +---------------------+----------------------------+------------+

    For example, setting ``rop`` to ``==`` produces an
    ``Equality`` relation, ``Eq()``.
    So does setting ``rop`` to ``eq``, or leaving ``rop`` unspecified.
    That is, the first three ``Rel()`` below all produce the same result.
    Using a ``rop`` from a different row in the table produces a
    different relation type.
    For example, the fourth ``Rel()`` below using ``lt`` for ``rop``
    produces a ``StrictLessThan`` inequality:

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2, 'eq')
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2)
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2, 'lt')
        y < x**2 + x

    To obtain the relation type of an existing expression,
    get its ``rel_op`` property.
    For example, ``rel_op`` is ``==`` for the ``Equality`` relation above,
    and ``<`` for the strict less than inequality above:

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> my_equality = Rel(y, x + x**2, '==')
    >>> my_equality.rel_op
        '=='
    >>> my_inequality = Rel(y, x + x**2, 'lt')
    >>> my_inequality.rel_op
        '<'

    """
    __slots__ = ()

    ValidRelationOperator: dict[str | None, type[Relational]] = {}

    is_Relational = True

    # ValidRelationOperator - Defined below, because the necessary classes
    #   have not yet been defined

    def __new__(cls, lhs, rhs, rop=None, **assumptions):
        # If called by a subclass, do nothing special and pass on to Basic.
        if cls is not Relational:
            return Basic.__new__(cls, lhs, rhs, **assumptions)

        # XXX: Why do this? There should be a separate function to make a
        # particular subclass of Relational from a string.
        #
        # If called directly with an operator, look up the subclass
        # corresponding to that operator and delegate to it
        cls = cls.ValidRelationOperator.get(rop, None)
        if cls is None:
            raise ValueError("Invalid relational operator symbol: %r" % rop)

        if not issubclass(cls, (Eq, Ne)):
            # validate that Booleans are not being used in a relational
            # other than Eq/Ne;
            # Note: Symbol is a subclass of Boolean but is considered
            # acceptable here.
            if any(map(_nontrivBool, (lhs, rhs))):
                raise TypeError(filldedent('''
                    A Boolean argument can only be used in
                    Eq and Ne; all other relationals expect
                    real expressions.
                '''))

        return cls(lhs, rhs, **assumptions)

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self._args[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self._args[1]

    @property
    def reversed(self):
        """Return the relationship with sides reversed.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversed
        Eq(1, x)
        >>> x < 1
        x < 1
        >>> _.reversed
        1 > x
        """
        ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
        a, b = self.args
        return Relational.__new__(ops.get(self.func, self.func), b, a)

    @property
    def reversedsign(self):
        """Return the relationship with signs reversed.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversedsign
        Eq(-x, -1)
        >>> x < 1
        x < 1
        >>> _.reversedsign
        -x > -1
        """
        a, b = self.args
        if not (isinstance(a, BooleanAtom) or isinstance(b, BooleanAtom)):
            ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
            return Relational.__new__(ops.get(self.func, self.func), -a, -b)
        else:
            return self

    @property
    def negated(self):
        """Return the negated relationship.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.negated
        Ne(x, 1)
        >>> x < 1
        x < 1
        >>> _.negated
        x >= 1

        Notes
        =====

        This works more or less identical to ``~``/``Not``. The difference is
        that ``negated`` returns the relationship even if ``evaluate=False``.
        Hence, this is useful in code when checking for e.g. negated relations
        to existing ones as it will not be affected by the `evaluate` flag.

        """
        ops = {Eq: Ne, Ge: Lt, Gt: Le, Le: Gt, Lt: Ge, Ne: Eq}
        # If there ever will be new Relational subclasses, the following line
        # will work until it is properly sorted out
        # return ops.get(self.func, lambda a, b, evaluate=False: ~(self.func(a,
        #      b, evaluate=evaluate)))(*self.args, evaluate=False)
        return Relational.__new__(ops.get(self.func), *self.args)

    @property
    def weak(self):
        """return the non-strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x < 1).weak
        x <= 1
        >>> _.weak
        x <= 1
        """
        return self

    @property
    def strict(self):
        """return the strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x <= 1).strict
        x < 1
        >>> _.strict
        x < 1
        """
        return self

    def _eval_evalf(self, prec):
        return self.func(*[s._evalf(prec) for s in self.args])

    @property
    def canonical(self):
        """Return a canonical form of the relational by putting a
        number on the rhs, canonically removing a sign or else
        ordering the args canonically. No other simplification is
        attempted.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x < 2
        x < 2
        >>> _.reversed.canonical
        x < 2
        >>> (-y < x).canonical
        x > -y
        >>> (-y > x).canonical
        x < -y
        >>> (-y < -x).canonical
        x < y

        The canonicalization is recursively applied:

        >>> from sympy import Eq
        >>> Eq(x < y, y > x).canonical
        True
        """
        args = tuple([i.canonical if isinstance(i, Relational) else i for i in self.args])
        if args != self.args:
            r = self.func(*args)
            if not isinstance(r, Relational):
                return r
        else:
            r = self
        if r.rhs.is_number:
            if r.rhs.is_Number and r.lhs.is_Number and r.lhs > r.rhs:
                r = r.reversed
        elif r.lhs.is_number:
            r = r.reversed
        elif tuple(ordered(args)) != args:
            r = r.reversed

        LHS_CEMS = getattr(r.lhs, 'could_extract_minus_sign', None)
        RHS_CEMS = getattr(r.rhs, 'could_extract_minus_sign', None)

        if isinstance(r.lhs, BooleanAtom) or isinstance(r.rhs, BooleanAtom):
            return r

        # Check if first value has negative sign
        if LHS_CEMS and LHS_CEMS():
            return r.reversedsign
        elif not r.rhs.is_number and RHS_CEMS and RHS_CEMS():
            # Right hand side has a minus, but not lhs.
            # How does the expression with reversed signs behave?
            # This is so that expressions of the type
            # Eq(x, -y) and Eq(-x, y)
            # have the same canonical representation
            expr1, _ = ordered([r.lhs, -r.rhs])
            if expr1 != r.lhs:
                return r.reversed.reversedsign

        return r

    def equals(self, other, failing_expression=False):
        """Return True if the sides of the relationship are mathematically
        identical and the type of relationship is the same.
        If failing_expression is True, return the expression whose truth value
        was unknown."""
        if isinstance(other, Relational):
            if other in (self, self.reversed):
                return True
            a, b = self, other
            if a.func in (Eq, Ne) or b.func in (Eq, Ne):
                if a.func != b.func:
                    return False
                left, right = [i.equals(j,
                                        failing_expression=failing_expression)
                               for i, j in zip(a.args, b.args)]
                if left is True:
                    return right
                if right is True:
                    return left
                lr, rl = [i.equals(j, failing_expression=failing_expression)
                          for i, j in zip(a.args, b.reversed.args)]
                if lr is True:
                    return rl
                if rl is True:
                    return lr
                e = (left, right, lr, rl)
                if all(i is False for i in e):
                    return False
                for i in e:
                    if i not in (True, False):
                        return i
            else:
                if b.func != a.func:
                    b = b.reversed
                if a.func != b.func:
                    return False
                left = a.lhs.equals(b.lhs,
                                    failing_expression=failing_expression)
                if left is False:
                    return False
                right = a.rhs.equals(b.rhs,
                                     failing_expression=failing_expression)
                if right is False:
                    return False
                if left is True:
                    return right
                return left

    def _eval_simplify(self, **kwargs):
        from .add import Add
        from .expr import Expr
        r = self
        r = r.func(*[i.simplify(**kwargs) for i in r.args])
        if r.is_Relational:
            if not isinstance(r.lhs, Expr) or not isinstance(r.rhs, Expr):
                return r
            dif = r.lhs - r.rhs
            # replace dif with a valid Number that will
            # allow a definitive comparison with 0
            v = None
            if dif.is_comparable:
                v = dif.n(2)
                if any(i._prec == 1 for i in v.as_real_imag()):
                    rv, iv = [i.n(2) for i in dif.as_real_imag()]
                    v = rv + S.ImaginaryUnit*iv
            elif dif.equals(0):  # XXX this is expensive
                v = S.Zero
            if v is not None:
                r = r.func._eval_relation(v, S.Zero)
            r = r.canonical
            # If there is only one symbol in the expression,
            # try to write it on a simplified form
            free = list(filter(lambda x: x.is_real is not False, r.free_symbols))
            if len(free) == 1:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    x = free.pop()
                    dif = r.lhs - r.rhs
                    m, b = linear_coeffs(dif, x)
                    if m.is_zero is False:
                        if m.is_negative:
                            # Dividing with a negative number, so change order of arguments
                            # canonical will put the symbol back on the lhs later
                            r = r.func(-b / m, x)
                        else:
                            r = r.func(x, -b / m)
                    else:
                        r = r.func(b, S.Zero)
                except ValueError:
                    # maybe not a linear function, try polynomial
                    from sympy.polys.polyerrors import PolynomialError
                    from sympy.polys.polytools import gcd, Poly, poly
                    try:
                        p = poly(dif, x)
                        c = p.all_coeffs()
                        constant = c[-1]
                        c[-1] = 0
                        scale = gcd(c)
                        c = [ctmp / scale for ctmp in c]
                        r = r.func(Poly.from_list(c, x).as_expr(), -constant / scale)
                    except PolynomialError:
                        pass
            elif len(free) >= 2:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    from sympy.polys.polytools import gcd
                    free = list(ordered(free))
                    dif = r.lhs - r.rhs
                    m = linear_coeffs(dif, *free)
                    constant = m[-1]
                    del m[-1]
                    scale = gcd(m)
                    m = [mtmp / scale for mtmp in m]
                    nzm = list(filter(lambda f: f[0] != 0, list(zip(m, free))))
                    if scale.is_zero is False:
                        if constant != 0:
                            # lhs: expression, rhs: constant
                            newexpr = Add(*[i * j for i, j in nzm])
                            r = r.func(newexpr, -constant / scale)
                        else:
                            # keep first term on lhs
                            lhsterm = nzm[0][0] * nzm[0][1]
                            del nzm[0]
                            newexpr = Add(*[i * j for i, j in nzm])
                            r = r.func(lhsterm, -newexpr)

                    else:
                        r = r.func(constant, S.Zero)
                except ValueError:
                    pass
        # Did we get a simplified result?
        r = r.canonical
        measure = kwargs['measure']
        if measure(r) < kwargs['ratio'] * measure(self):
            return r
        else:
            return self

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        return self.func(trigsimp(self.lhs, **opts), trigsimp(self.rhs, **opts))

    def expand(self, **kwargs):
        args = (arg.expand(**kwargs) for arg in self.args)
        return self.func(*args)

    def __bool__(self) -> bool:
        raise TypeError(
            LazyExceptionMessage(
                lambda: f"cannot determine truth value of Relational: {self}"
            )
        )

    def _eval_as_set(self):
        # self is univariate and periodicity(self, x) in (0, None)
        from sympy.solvers.inequalities import solve_univariate_inequality
        from sympy.sets.conditionset import ConditionSet
        syms = self.free_symbols
        assert len(syms) == 1
        x = syms.pop()
        try:
            xset = solve_univariate_inequality(self, x, relational=False)
        except NotImplementedError:
            # solve_univariate_inequality raises NotImplementedError for
            # unsolvable equations/inequalities.
            xset = ConditionSet(x, self, S.Reals)
        return xset

    @property
    def binary_symbols(self):
        # override where necessary
        return set()


Rel = Relational


class Equality(Relational):
    """
    An equal relation between two objects.

    Explanation
    ===========

    Represents that two objects are equal.  If they can be easily shown
    to be definitively equal (or unequal), this will reduce to True (or
    False).  Otherwise, the relation is maintained as an unevaluated
    Equality object.  Use the ``simplify`` function on this object for
    more nontrivial evaluation of the equality relation.

    As usual, the keyword argument ``evaluate=False`` can be used to
    prevent any evaluation.

    Examples
    ========

    >>> from sympy import Eq, simplify, exp, cos
    >>> from sympy.abc import x, y
    >>> Eq(y, x + x**2)
    Eq(y, x**2 + x)
    >>> Eq(2, 5)
    False
    >>> Eq(2, 5, evaluate=False)
    Eq(2, 5)
    >>> _.doit()
    False
    >>> Eq(exp(x), exp(x).rewrite(cos))
    Eq(exp(x), sinh(x) + cosh(x))
    >>> simplify(_)
    True

    See Also
    ========

    sympy.logic.boolalg.Equivalent : for representing equality between two
        boolean expressions

    Notes
    =====

    Python treats 1 and True (and 0 and False) as being equal; SymPy
    does not. And integer will always compare as unequal to a Boolean:

    >>> Eq(True, 1), True == 1
    (False, True)

    This class is not the same as the == operator.  The == operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    If either object defines an ``_eval_Eq`` method, it can be used in place of
    the default algorithm.  If ``lhs._eval_Eq(rhs)`` or ``rhs._eval_Eq(lhs)``
    returns anything other than None, that return value will be substituted for
    the Equality.  If None is returned by ``_eval_Eq``, an Equality object will
    be created as usual.

    Since this object is already an expression, it does not respond to
    the method ``as_expr`` if one tries to create `x - y` from ``Eq(x, y)``.
    If ``eq = Eq(x, y)`` then write `eq.lhs - eq.rhs` to get ``x - y``.

    .. deprecated:: 1.5

       ``Eq(expr)`` with a single argument is a shorthand for ``Eq(expr, 0)``,
       but this behavior is deprecated and will be removed in a future version
       of SymPy.

    """
    rel_op = '=='

    __slots__ = ()

    is_Equality = True

    def __new__(cls, lhs, rhs, **options):
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        if evaluate:
            val = is_eq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)

        return Relational.__new__(cls, lhs, rhs)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs == rhs)

    def _eval_rewrite_as_Add(self, L, R, evaluate=True, **kwargs):
        """
        return Eq(L, R) as L - R. To control the evaluation of
        the result set pass `evaluate=True` to give L - R;
        if `evaluate=None` then terms in L and R will not cancel
        but they will be listed in canonical order; otherwise
        non-canonical args will be returned. If one side is 0, the
        non-zero side will be returned.

        .. deprecated:: 1.13

           The method ``Eq.rewrite(Add)`` is deprecated.
           See :ref:`eq-rewrite-Add` for details.

        Examples
        ========

        >>> from sympy import Eq, Add
        >>> from sympy.abc import b, x
        >>> eq = Eq(x + b, x - b)
        >>> eq.rewrite(Add)  #doctest: +SKIP
        2*b
        >>> eq.rewrite(Add, evaluate=None).args  #doctest: +SKIP
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).args  #doctest: +SKIP
        (b, x, b, -x)
        """
        sympy_deprecation_warning("""
        Eq.rewrite(Add) is deprecated.

        For ``eq = Eq(a, b)`` use ``eq.lhs - eq.rhs`` to obtain
        ``a - b``.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="eq-rewrite-Add",
            stacklevel=5,
        )
        from .add import _unevaluated_Add, Add
        if L == 0:
            return R
        if R == 0:
            return L
        if evaluate:
            # allow cancellation of args
            return L - R
        args = Add.make_args(L) + Add.make_args(-R)
        if evaluate is None:
            # no cancellation, but canonical
            return _unevaluated_Add(*args)
        # no cancellation, not canonical
        return Add._from_args(args)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()

    def _eval_simplify(self, **kwargs):
        # standard simplify
        e = super()._eval_simplify(**kwargs)
        if not isinstance(e, Equality):
            return e
        from .expr import Expr
        if not isinstance(e.lhs, Expr) or not isinstance(e.rhs, Expr):
            return e
        free = self.free_symbols
        if len(free) == 1:
            try:
                from .add import Add
                from sympy.solvers.solveset import linear_coeffs
                x = free.pop()
                m, b = linear_coeffs(
                    Add(e.lhs, -e.rhs, evaluate=False), x)
                if m.is_zero is False:
                    enew = e.func(x, -b / m)
                else:
                    enew = e.func(m * x, -b)
                measure = kwargs['measure']
                if measure(enew) <= kwargs['ratio'] * measure(e):
                    e = enew
            except ValueError:
                pass
        return e.canonical

    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        from sympy.integrals.integrals import integrate
        return integrate(self, *args, **kwargs)

    def as_poly(self, *gens, **kwargs):
        '''Returns lhs-rhs as a Poly

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x**2, 1).as_poly(x)
        Poly(x**2 - 1, x, domain='ZZ')
        '''
        return (self.lhs - self.rhs).as_poly(*gens, **kwargs)


Eq = Equality


class Unequality(Relational):
    """An unequal relation between two objects.

    Explanation
    ===========

    Represents that two objects are not equal.  If they can be shown to be
    definitively equal, this will reduce to False; if definitively unequal,
    this will reduce to True.  Otherwise, the relation is maintained as an
    Unequality object.

    Examples
    ========

    >>> from sympy import Ne
    >>> from sympy.abc import x, y
    >>> Ne(y, x+x**2)
    Ne(y, x**2 + x)

    See Also
    ========
    Equality

    Notes
    =====
    This class is not the same as the != operator.  The != operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    This class is effectively the inverse of Equality.  As such, it uses the
    same algorithms, including any available `_eval_Eq` methods.

    """
    rel_op = '!='

    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        if evaluate:
            val = is_neq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)

        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs != rhs)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()

    def _eval_simplify(self, **kwargs):
        # simplify as an equality
        eq = Equality(*self.args)._eval_simplify(**kwargs)
        if isinstance(eq, Equality):
            # send back Ne with the new args
            return self.func(*eq.args)
        return eq.negated  # result of Ne is the negated Eq


Ne = Unequality


class _Inequality(Relational):
    """Internal base class for all *Than types.

    Each subclass must implement _eval_relation to provide the method for
    comparing two real numbers.

    """
    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):

        try:
            lhs = _sympify(lhs)
            rhs = _sympify(rhs)
        except SympifyError:
            return NotImplemented

        evaluate = options.pop('evaluate', global_parameters.evaluate)
        if evaluate:
            for me in (lhs, rhs):
                if me.is_extended_real is False:
                    raise TypeError("Invalid comparison of non-real %s" % me)
                if me is S.NaN:
                    raise TypeError("Invalid NaN comparison")
            # First we invoke the appropriate inequality method of `lhs`
            # (e.g., `lhs.__lt__`).  That method will try to reduce to
            # boolean or raise an exception.  It may keep calling
            # superclasses until it reaches `Expr` (e.g., `Expr.__lt__`).
            # In some cases, `Expr` will just invoke us again (if neither it
            # nor a subclass was able to reduce to boolean or raise an
            # exception).  In that case, it must call us with
            # `evaluate=False` to prevent infinite recursion.
            return cls._eval_relation(lhs, rhs, **options)

        # make a "non-evaluated" Expr for the inequality
        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs, **options):
        val = cls._eval_fuzzy_relation(lhs, rhs)
        if val is None:
            return cls(lhs, rhs, evaluate=False)
        else:
            return _sympify(val)


class _Greater(_Inequality):
    """Not intended for general use

    _Greater is only used so that GreaterThan and StrictGreaterThan may
    subclass it for the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        return self._args[0]

    @property
    def lts(self):
        return self._args[1]


class _Less(_Inequality):
    """Not intended for general use.

    _Less is only used so that LessThan and StrictLessThan may subclass it for
    the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        return self._args[1]

    @property
    def lts(self):
        return self._args[0]


class GreaterThan(_Greater):
    r"""Class representations of inequalities.

    Explanation
    ===========

    The ``*Than`` classes represent inequal relationships, where the left-hand
    side is generally bigger or smaller than the right-hand side.  For example,
    the GreaterThan class represents an inequal relationship where the
    left-hand side is at least as big as the right side, if not bigger.  In
    mathematical notation:

    lhs $\ge$ rhs

    In total, there are four ``*Than`` classes, to represent the four
    inequalities:

    +-----------------+--------+
    |Class Name       | Symbol |
    +=================+========+
    |GreaterThan      | ``>=`` |
    +-----------------+--------+
    |LessThan         | ``<=`` |
    +-----------------+--------+
    |StrictGreaterThan| ``>``  |
    +-----------------+--------+
    |StrictLessThan   | ``<``  |
    +-----------------+--------+

    All classes take two arguments, lhs and rhs.

    +----------------------------+-----------------+
    |Signature Example           | Math Equivalent |
    +============================+=================+
    |GreaterThan(lhs, rhs)       |   lhs $\ge$ rhs |
    +----------------------------+-----------------+
    |LessThan(lhs, rhs)          |   lhs $\le$ rhs |
    +----------------------------+-----------------+
    |StrictGreaterThan(lhs, rhs) |   lhs $>$ rhs   |
    +----------------------------+-----------------+
    |StrictLessThan(lhs, rhs)    |   lhs $<$ rhs   |
    +----------------------------+-----------------+

    In addition to the normal .lhs and .rhs of Relations, ``*Than`` inequality
    objects also have the .lts and .gts properties, which represent the "less
    than side" and "greater than side" of the operator.  Use of .lts and .gts
    in an algorithm rather than .lhs and .rhs as an assumption of inequality
    direction will make more explicit the intent of a certain section of code,
    and will make it similarly more robust to client code changes:

    >>> from sympy import GreaterThan, StrictGreaterThan
    >>> from sympy import LessThan, StrictLessThan
    >>> from sympy import And, Ge, Gt, Le, Lt, Rel, S
    >>> from sympy.abc import x, y, z
    >>> from sympy.core.relational import Relational

    >>> e = GreaterThan(x, 1)
    >>> e
    x >= 1
    >>> '%s >= %s is the same as %s <= %s' % (e.gts, e.lts, e.lts, e.gts)
    'x >= 1 is the same as 1 <= x'

    Examples
    ========

    One generally does not instantiate these classes directly, but uses various
    convenience methods:

    >>> for f in [Ge, Gt, Le, Lt]:  # convenience wrappers
    ...     print(f(x, 2))
    x >= 2
    x > 2
    x <= 2
    x < 2

    Another option is to use the Python inequality operators (``>=``, ``>``,
    ``<=``, ``<``) directly.  Their main advantage over the ``Ge``, ``Gt``,
    ``Le``, and ``Lt`` counterparts, is that one can write a more
    "mathematical looking" statement rather than littering the math with
    oddball function calls.  However there are certain (minor) caveats of
    which to be aware (search for 'gotcha', below).

    >>> x >= 2
    x >= 2
    >>> _ == Ge(x, 2)
    True

    However, it is also perfectly valid to instantiate a ``*Than`` class less
    succinctly and less conveniently:

    >>> Rel(x, 1, ">")
    x > 1
    >>> Relational(x, 1, ">")
    x > 1

    >>> StrictGreaterThan(x, 1)
    x > 1
    >>> GreaterThan(x, 1)
    x >= 1
    >>> LessThan(x, 1)
    x <= 1
    >>> StrictLessThan(x, 1)
    x < 1

    Notes
    =====

    There are a couple of "gotchas" to be aware of when using Python's
    operators.

    The first is that what your write is not always what you get:

        >>> 1 < x
        x > 1

        Due to the order that Python parses a statement, it may
        not immediately find two objects comparable.  When ``1 < x``
        is evaluated, Python recognizes that the number 1 is a native
        number and that x is *not*.  Because a native Python number does
        not know how to compare itself with a SymPy object
        Python will try the reflective operation, ``x > 1`` and that is the
        form that gets evaluated, hence returned.

        If the order of the statement is important (for visual output to
        the console, perhaps), one can work around this annoyance in a
        couple ways:

        (1) "sympify" the literal before comparison

        >>> S(1) < x
        1 < x

        (2) use one of the wrappers or less succinct methods described
        above

        >>> Lt(1, x)
        1 < x
        >>> Relational(1, x, "<")
        1 < x

    The second gotcha involves writing equality tests between relationals
    when one or both sides of the test involve a literal relational:

        >>> e = x < 1; e
        x < 1
        >>> e == e  # neither side is a literal
        True
        >>> e == x < 1  # expecting True, too
        False
        >>> e != x < 1  # expecting False
        x < 1
        >>> x < 1 != x < 1  # expecting False or the same thing as before
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational

        The solution for this case is to wrap literal relationals in
        parentheses:

        >>> e == (x < 1)
        True
        >>> e != (x < 1)
        False
        >>> (x < 1) != (x < 1)
        False

    The third gotcha involves chained inequalities not involving
    ``==`` or ``!=``. Occasionally, one may be tempted to write:

        >>> e = x < y < z
        Traceback (most recent call last):
        ...
        TypeError: symbolic boolean expression has no truth value.

        Due to an implementation detail or decision of Python [1]_,
        there is no way for SymPy to create a chained inequality with
        that syntax so one must use And:

        >>> e = And(x < y, y < z)
        >>> type( e )
        And
        >>> e
        (x < y) & (y < z)

        Although this can also be done with the '&' operator, it cannot
        be done with the 'and' operarator:

        >>> (x < y) & (y < z)
        (x < y) & (y < z)
        >>> (x < y) and (y < z)
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational

    .. [1] This implementation detail is that Python provides no reliable
       method to determine that a chained inequality is being built.
       Chained comparison operators are evaluated pairwise, using "and"
       logic (see
       https://docs.python.org/3/reference/expressions.html#not-in). This
       is done in an efficient way, so that each object being compared
       is only evaluated once and the comparison can short-circuit. For
       example, ``1 > 2 > 3`` is evaluated by Python as ``(1 > 2) and (2
       > 3)``. The ``and`` operator coerces each side into a bool,
       returning the object itself when it short-circuits. The bool of
       the --Than operators will raise TypeError on purpose, because
       SymPy cannot determine the mathematical ordering of symbolic
       expressions. Thus, if we were to compute ``x > y > z``, with
       ``x``, ``y``, and ``z`` being Symbols, Python converts the
       statement (roughly) into these steps:

        (1) x > y > z
        (2) (x > y) and (y > z)
        (3) (GreaterThanObject) and (y > z)
        (4) (GreaterThanObject.__bool__()) and (y > z)
        (5) TypeError

       Because of the ``and`` added at step 2, the statement gets turned into a
       weak ternary statement, and the first object's ``__bool__`` method will
       raise TypeError.  Thus, creating a chained inequality is not possible.

           In Python, there is no way to override the ``and`` operator, or to
           control how it short circuits, so it is impossible to make something
           like ``x > y > z`` work.  There was a PEP to change this,
           :pep:`335`, but it was officially closed in March, 2012.

    """
    __slots__ = ()

    rel_op = '>='

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_ge(lhs, rhs)

    @property
    def strict(self):
        return Gt(*self.args)

Ge = GreaterThan


class LessThan(_Less):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '<='

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_le(lhs, rhs)

    @property
    def strict(self):
        return Lt(*self.args)

Le = LessThan


class StrictGreaterThan(_Greater):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '>'

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_gt(lhs, rhs)

    @property
    def weak(self):
        return Ge(*self.args)


Gt = StrictGreaterThan


class StrictLessThan(_Less):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '<'

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_lt(lhs, rhs)

    @property
    def weak(self):
        return Le(*self.args)

Lt = StrictLessThan

# A class-specific (not object-specific) data item used for a minor speedup.
# It is defined here, rather than directly in the class, because the classes
# that it references have not been defined until now (e.g. StrictLessThan).
Relational.ValidRelationOperator = {
    None: Equality,
    '==': Equality,
    'eq': Equality,
    '!=': Unequality,
    '<>': Unequality,
    'ne': Unequality,
    '>=': GreaterThan,
    'ge': GreaterThan,
    '<=': LessThan,
    'le': LessThan,
    '>': StrictGreaterThan,
    'gt': StrictGreaterThan,
    '<': StrictLessThan,
    'lt': StrictLessThan,
}


def _n2(a, b):
    """Return (a - b).evalf(2) if a and b are comparable, else None.
    This should only be used when a and b are already sympified.
    """
    # /!\ it is very important (see issue 8245) not to
    # use a re-evaluated number in the calculation of dif
    if a.is_comparable and b.is_comparable:
        dif = (a - b).evalf(2)
        if dif.is_comparable:
            return dif


@dispatch(Expr, Expr)
def _eval_is_ge(lhs, rhs):
    return None


@dispatch(Basic, Basic)
def _eval_is_eq(lhs, rhs):
    return None


@dispatch(Tuple, Expr) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return False


@dispatch(Tuple, AppliedUndef) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None


@dispatch(Tuple, Symbol) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None


@dispatch(Tuple, Tuple) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    if len(lhs) != len(rhs):
        return False

    return fuzzy_and(fuzzy_bool(is_eq(s, o)) for s, o in zip(lhs, rhs))


def is_lt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly less than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_ge(lhs, rhs, assumptions))


def is_gt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly greater than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_le(lhs, rhs, assumptions))


def is_le(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is less than or equal to rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return is_ge(rhs, lhs, assumptions)


def is_ge(lhs, rhs, assumptions=None):
    """
    Fuzzy bool for *lhs* is greater than or equal to *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified,
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    rhs : Expr
        The right-hand side of the expression, must be sympified
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the inequality.

    Returns
    =======

    ``True`` if *lhs* is greater than or equal to *rhs*, ``False`` if *lhs*
    is less than *rhs*, and ``None`` if the comparison between *lhs* and
    *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    The four comparison functions ``is_le``, ``is_lt``, ``is_ge``, and ``is_gt`` are
    each implemented in terms of ``is_ge`` in the following way:

    is_ge(x, y) := is_ge(x, y)
    is_le(x, y) := is_ge(y, x)
    is_lt(x, y) := fuzzy_not(is_ge(x, y))
    is_gt(x, y) := fuzzy_not(is_ge(y, x))

    Therefore, supporting new type with this function will ensure behavior for
    other three functions as well.

    To maintain these equivalences in fuzzy logic it is important that in cases where
    either x or y is non-real all comparisons will give None.

    Examples
    ========

    >>> from sympy import S, Q
    >>> from sympy.core.relational import is_ge, is_le, is_gt, is_lt
    >>> from sympy.abc import x
    >>> is_ge(S(2), S(0))
    True
    >>> is_ge(S(0), S(2))
    False
    >>> is_le(S(0), S(2))
    True
    >>> is_gt(S(0), S(2))
    False
    >>> is_lt(S(2), S(0))
    False

    Assumptions can be passed to evaluate the quality which is otherwise
    indeterminate.

    >>> print(is_ge(x, S(0)))
    None
    >>> is_ge(x, S(0), assumptions=Q.positive(x))
    True

    New types can be supported by dispatching to ``_eval_is_ge``.

    >>> from sympy import Expr, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyExpr(Expr):
    ...     def __new__(cls, arg):
    ...         return super().__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    >>> @dispatch(MyExpr, MyExpr)
    ... def _eval_is_ge(a, b):
    ...     return is_ge(a.value, b.value)
    >>> a = MyExpr(1)
    >>> b = MyExpr(2)
    >>> is_ge(b, a)
    True
    >>> is_le(a, b)
    True
    """
    from sympy.assumptions.wrapper import AssumptionsWrapper, is_extended_nonnegative

    if not (isinstance(lhs, Expr) and isinstance(rhs, Expr)):
        raise TypeError("Can only compare inequalities with Expr")

    retval = _eval_is_ge(lhs, rhs)

    if retval is not None:
        return retval
    else:
        n2 = _n2(lhs, rhs)
        if n2 is not None:
            # use float comparison for infinity.
            # otherwise get stuck in infinite recursion
            if n2 in (S.Infinity, S.NegativeInfinity):
                n2 = float(n2)
            return n2 >= 0

        _lhs = AssumptionsWrapper(lhs, assumptions)
        _rhs = AssumptionsWrapper(rhs, assumptions)
        if _lhs.is_extended_real and _rhs.is_extended_real:
            if (_lhs.is_infinite and _lhs.is_extended_positive) or (_rhs.is_infinite and _rhs.is_extended_negative):
                return True
            diff = lhs - rhs
            if diff is not S.NaN:
                rv = is_extended_nonnegative(diff, assumptions)
                if rv is not None:
                    return rv


def is_neq(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs does not equal rhs.

    See the docstring for :func:`~.is_eq` for more.
    """
    return fuzzy_not(is_eq(lhs, rhs, assumptions))


def is_eq(lhs, rhs, assumptions=None):
    """
    Fuzzy bool representing mathematical equality between *lhs* and *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified.

    rhs : Expr
        The right-hand side of the expression, must be sympified.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the equality.

    Returns
    =======

    ``True`` if *lhs* is equal to *rhs*, ``False`` is *lhs* is not equal to *rhs*,
    and ``None`` if the comparison between *lhs* and *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    :func:`~.is_neq` calls this function to return its value, so supporting
    new type with this function will ensure correct behavior for ``is_neq``
    as well.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.core.relational import is_eq, is_neq
    >>> from sympy.abc import x
    >>> is_eq(S(0), S(0))
    True
    >>> is_neq(S(0), S(0))
    False
    >>> is_eq(S(0), S(2))
    False
    >>> is_neq(S(0), S(2))
    True

    Assumptions can be passed to evaluate the equality which is otherwise
    indeterminate.

    >>> print(is_eq(x, S(0)))
    None
    >>> is_eq(x, S(0), assumptions=Q.zero(x))
    True

    New types can be supported by dispatching to ``_eval_is_eq``.

    >>> from sympy import Basic, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyBasic(Basic):
    ...     def __new__(cls, arg):
    ...         return Basic.__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    ...
    >>> @dispatch(MyBasic, MyBasic)
    ... def _eval_is_eq(a, b):
    ...     return is_eq(a.value, b.value)
    ...
    >>> a = MyBasic(1)
    >>> b = MyBasic(1)
    >>> is_eq(a, b)
    True
    >>> is_neq(a, b)
    False

    """
    # here, _eval_Eq is only called for backwards compatibility
    # new code should use is_eq with multiple dispatch as
    # outlined in the docstring
    for side1, side2 in (lhs, rhs), (rhs, lhs):
        eval_func = getattr(side1, '_eval_Eq', None)
        if eval_func is not None:
            retval = eval_func(side2)
            if retval is not None:
                return retval

    retval = _eval_is_eq(lhs, rhs)
    if retval is not None:
        return retval

    if dispatch(type(lhs), type(rhs)) != dispatch(type(rhs), type(lhs)):
        retval = _eval_is_eq(rhs, lhs)
        if retval is not None:
            return retval

    # retval is still None, so go through the equality logic
    # If expressions have the same structure, they must be equal.
    if lhs == rhs:
        return True  # e.g. True == True
    elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
        return False  # True != False
    elif not (lhs.is_Symbol or rhs.is_Symbol) and (
        isinstance(lhs, Boolean) !=
        isinstance(rhs, Boolean)):
        return False  # only Booleans can equal Booleans

    from sympy.assumptions.wrapper import (AssumptionsWrapper,
        is_infinite, is_extended_real)
    from .add import Add

    _lhs = AssumptionsWrapper(lhs, assumptions)
    _rhs = AssumptionsWrapper(rhs, assumptions)

    if _lhs.is_infinite or _rhs.is_infinite:
        if fuzzy_xor([_lhs.is_infinite, _rhs.is_infinite]):
            return False
        if fuzzy_xor([_lhs.is_extended_real, _rhs.is_extended_real]):
            return False
        if fuzzy_and([_lhs.is_extended_real, _rhs.is_extended_real]):
            return fuzzy_xor([_lhs.is_extended_positive, fuzzy_not(_rhs.is_extended_positive)])

        # Try to split real/imaginary parts and equate them
        I = S.ImaginaryUnit

        def split_real_imag(expr):
            real_imag = lambda t: (
                'real' if is_extended_real(t, assumptions) else
                'imag' if is_extended_real(I*t, assumptions) else None)
            return sift(Add.make_args(expr), real_imag)

        lhs_ri = split_real_imag(lhs)
        if not lhs_ri[None]:
            rhs_ri = split_real_imag(rhs)
            if not rhs_ri[None]:
                eq_real = is_eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']), assumptions)
                eq_imag = is_eq(I * Add(*lhs_ri['imag']), I * Add(*rhs_ri['imag']), assumptions)
                return fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))

        from sympy.functions.elementary.complexes import arg
        # Compare e.g. zoo with 1+I*oo by comparing args
        arglhs = arg(lhs)
        argrhs = arg(rhs)
        # Guard against Eq(nan, nan) -> False
        if not (arglhs == S.NaN and argrhs == S.NaN):
            return fuzzy_bool(is_eq(arglhs, argrhs, assumptions))

    if all(isinstance(i, Expr) for i in (lhs, rhs)):
        # see if the difference evaluates
        dif = lhs - rhs
        _dif = AssumptionsWrapper(dif, assumptions)
        z = _dif.is_zero
        if z is not None:
            if z is False and _dif.is_commutative:  # issue 10728
                return False
            if z:
                return True

        # is_zero cannot help decide integer/rational with Float
        c, t = dif.as_coeff_Add()
        if c.is_Float:
            if int_valued(c):
                if t.is_integer is False:
                    return False
            elif t.is_rational is False:
                return False

        n2 = _n2(lhs, rhs)
        if n2 is not None:
            return _sympify(n2 == 0)

        # see if the ratio evaluates
        n, d = dif.as_numer_denom()
        rv = None
        _n = AssumptionsWrapper(n, assumptions)
        _d = AssumptionsWrapper(d, assumptions)
        if _n.is_zero:
            rv = _d.is_nonzero
        elif _n.is_finite:
            if _d.is_infinite:
                rv = True
            elif _n.is_zero is False:
                rv = _d.is_infinite
                if rv is None:
                    # if the condition that makes the denominator
                    # infinite does not make the original expression
                    # True then False can be returned
                    from sympy.simplify.simplify import clear_coefficients
                    l, r = clear_coefficients(d, S.Infinity)
                    args = [_.subs(l, r) for _ in (lhs, rhs)]
                    if args != [lhs, rhs]:
                        rv = fuzzy_bool(is_eq(*args, assumptions))
                        if rv is True:
                            rv = None
        elif any(is_infinite(a, assumptions) for a in Add.make_args(n)):
            # (inf or nan)/x != 0
            rv = False
        if rv is not None:
            return rv
<|endoftext|>import os
USE_SYMENGINE = os.getenv('USE_SYMENGINE', '0')
USE_SYMENGINE = USE_SYMENGINE.lower() in ('1', 't', 'true')  # type: ignore

if USE_SYMENGINE:
    from symengine import (Symbol, Integer, sympify as sympify_symengine, S,
        SympifyError, exp, log, gamma, sqrt, I, E, pi, Matrix,
        sin, cos, tan, cot, csc, sec, asin, acos, atan, acot, acsc, asec,
        sinh, cosh, tanh, coth, asinh, acosh, atanh, acoth,
        lambdify, symarray, diff, zeros, eye, diag, ones,
        expand, Function, symbols, var, Add, Mul, Derivative,
        ImmutableMatrix, MatrixBase, Rational, Basic)
    from symengine.lib.symengine_wrapper import gcd as igcd
    from symengine import AppliedUndef

    def sympify(a, *, strict=False):
        """
        Notes
        =====

        SymEngine's ``sympify`` does not accept keyword arguments and is
        therefore not compatible with SymPy's ``sympify`` with ``strict=True``
        (which ensures that only the types for which an explicit conversion has
        been defined are converted). This wrapper adds an additional parameter
        ``strict`` (with default ``False``) that will raise a ``SympifyError``
        if ``strict=True`` and the argument passed to the parameter ``a`` is a
        string.

        See Also
        ========

        sympify: Converts an arbitrary expression to a type that can be used
            inside SymPy.

        """
        # The parameter ``a`` is used for this function to keep compatibility
        # with the SymEngine docstring.
        if strict and isinstance(a, str):
            raise SympifyError(a)
        return sympify_symengine(a)

    # Keep the SymEngine docstring and append the additional "Notes" and "See
    # Also" sections. Replacement of spaces is required to correctly format the
    # indentation of the combined docstring.
    sympify.__doc__ = (
        sympify_symengine.__doc__
        + sympify.__doc__.replace('        ', '    ')  # type: ignore
    )
else:
    from sympy.core.add import Add
    from sympy.core.basic import Basic
    from sympy.core.function import (diff, Function, AppliedUndef,
        expand, Derivative)
    from sympy.core.mul import Mul
    from sympy.core.intfunc import igcd
    from sympy.core.numbers import pi, I, Integer, Rational, E
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol, var, symbols
    from sympy.core.sympify import SympifyError, sympify
    from sympy.functions.elementary.exponential import log, exp
    from sympy.functions.elementary.hyperbolic import (coth, sinh,
        acosh, acoth, tanh, asinh, atanh, cosh)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (csc,
        asec, cos, atan, sec, acot, asin, tan, sin, cot, acsc, acos)
    from sympy.functions.special.gamma_functions import gamma
    from sympy.matrices.dense import (eye, zeros, diag, Matrix,
        ones, symarray)
    from sympy.matrices.immutable import ImmutableMatrix
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.utilities.lambdify import lambdify


#
# XXX: Handling of immutable and mutable matrices in SymEngine is inconsistent
# with SymPy's matrix classes in at least SymEngine version 0.7.0. Until that
# is fixed the function below is needed for consistent behaviour when
# attempting to simplify a matrix.
#
# Expected behaviour of a SymPy mutable/immutable matrix .simplify() method:
#
#   Matrix.simplify() : works in place, returns None
#   ImmutableMatrix.simplify() : returns a simplified copy
#
# In SymEngine both mutable and immutable matrices simplify in place and return
# None. This is inconsistent with the matrix being "immutable" and also the
# returned None leads to problems in the mechanics module.
#
# The simplify function should not be used because simplify(M) sympifies the
# matrix M and the SymEngine matrices all sympify to SymPy matrices. If we want
# to work with SymEngine matrices then we need to use their .simplify() method
# but that method does not work correctly with immutable matrices.
#
# The _simplify_matrix function can be removed when the SymEngine bug is fixed.
# Since this should be a temporary problem we do not make this function part of
# the public API.
#
#   SymEngine issue: https://github.com/symengine/symengine.py/issues/363
#

def _simplify_matrix(M):
    """Return a simplified copy of the matrix M"""
    if not isinstance(M, (Matrix, ImmutableMatrix)):
        raise TypeError("The matrix M must be an instance of Matrix or ImmutableMatrix")
    Mnew = M.as_mutable() # makes a copy if mutable
    Mnew.simplify()
    if isinstance(M, ImmutableMatrix):
        Mnew = Mnew.as_immutable()
    return Mnew


__all__ = [
    'Symbol', 'Integer', 'sympify', 'S', 'SympifyError', 'exp', 'log',
    'gamma', 'sqrt', 'I', 'E', 'pi', 'Matrix', 'sin', 'cos', 'tan', 'cot',
    'csc', 'sec', 'asin', 'acos', 'atan', 'acot', 'acsc', 'asec', 'sinh',
    'cosh', 'tanh', 'coth', 'asinh', 'acosh', 'atanh', 'acoth', 'lambdify',
    'symarray', 'diff', 'zeros', 'eye', 'diag', 'ones', 'expand', 'Function',
    'symbols', 'var', 'Add', 'Mul', 'Derivative', 'ImmutableMatrix',
    'MatrixBase', 'Rational', 'Basic', 'igcd', 'AppliedUndef',
]
<|endoftext|>"""sympify -- convert objects SymPy internal format"""

from __future__ import annotations

from typing import Any, Callable, overload, TYPE_CHECKING, TypeVar

import mpmath.libmp as mlib

from inspect import getmro
import string
from sympy.core.random import choice

from .parameters import global_parameters

from sympy.utilities.iterables import iterable


if TYPE_CHECKING:

    from sympy.core.basic import Basic
    from sympy.core.expr import Expr
    from sympy.core.numbers import Integer, Float

    Tbasic = TypeVar('Tbasic', bound=Basic)


class SympifyError(ValueError):
    def __init__(self, expr, base_exc=None):
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if self.base_exc is None:
            return "SympifyError: %r" % (self.expr,)

        return ("Sympify of expression '%s' failed, because of exception being "
            "raised:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__,
            str(self.base_exc)))


converter: dict[type[Any], Callable[[Any], Basic]] = {}

#holds the conversions defined in SymPy itself, i.e. non-user defined conversions
_sympy_converter: dict[type[Any], Callable[[Any], Basic]] = {}

#alias for clearer use in the library
_external_converter = converter

class CantSympify:
    """
    Mix in this trait to a class to disallow sympification of its instances.

    Examples
    ========

    >>> from sympy import sympify
    >>> from sympy.core.sympify import CantSympify

    >>> class Something(dict):
    ...     pass
    ...
    >>> sympify(Something())
