modules, whose generators are algebraic numbers.

There is an abstract base class called :py:class:`~.Module`, which has two
concrete subclasses, :py:class:`~.PowerBasis` and :py:class:`~.Submodule`.

Every module is defined by its basis, or set of generators:

* For a :py:class:`~.PowerBasis`, the generators are the first $n$ powers
  (starting with the zeroth) of an algebraic integer $\theta$ of degree $n$.
  The :py:class:`~.PowerBasis` is constructed by passing either the minimal
  polynomial of $\theta$, or an :py:class:`~.AlgebraicField` having $\theta$
  as its primitive element.

* For a :py:class:`~.Submodule`, the generators are a set of
  $\mathbb{Q}$-linear combinations of the generators of another module. That
  other module is then the "parent" of the :py:class:`~.Submodule`. The
  coefficients of the $\mathbb{Q}$-linear combinations may be given by an
  integer matrix, and a positive integer denominator. Each column of the matrix
  defines a generator.

>>> from sympy.polys import Poly, cyclotomic_poly, ZZ
>>> from sympy.abc import x
>>> from sympy.polys.matrices import DomainMatrix, DM
>>> from sympy.polys.numberfields.modules import PowerBasis
>>> T = Poly(cyclotomic_poly(5, x))
>>> A = PowerBasis(T)
>>> print(A)
PowerBasis(x**4 + x**3 + x**2 + x + 1)
>>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ), denom=3)
>>> print(B)
Submodule[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]/3
>>> print(B.parent)
PowerBasis(x**4 + x**3 + x**2 + x + 1)

Thus, every module is either a :py:class:`~.PowerBasis`,
or a :py:class:`~.Submodule`, some ancestor of which is a
:py:class:`~.PowerBasis`. (If ``S`` is a :py:class:`~.Submodule`, then its
ancestors are ``S.parent``, ``S.parent.parent``, and so on).

The :py:class:`~.ModuleElement` class represents a linear combination of the
generators of any module. Critically, the coefficients of this linear
combination are not restricted to be integers, but may be any rational
numbers. This is necessary so that any and all algebraic integers be
representable, starting from the power basis in a primitive element $\theta$
for the number field in question. For example, in a quadratic field
$\mathbb{Q}(\sqrt{d})$ where $d \equiv 1 \mod{4}$, a denominator of $2$ is
needed.

A :py:class:`~.ModuleElement` can be constructed from an integer column vector
and a denominator:

>>> U = Poly(x**2 - 5)
>>> M = PowerBasis(U)
>>> e = M(DM([[1], [1]], ZZ), denom=2)
>>> print(e)
[1, 1]/2
>>> print(e.module)
PowerBasis(x**2 - 5)

The :py:class:`~.PowerBasisElement` class is a subclass of
:py:class:`~.ModuleElement` that represents elements of a
:py:class:`~.PowerBasis`, and adds functionality pertinent to elements
represented directly over powers of the primitive element $\theta$.


Arithmetic with module elements
===============================

While a :py:class:`~.ModuleElement` represents a linear combination over the
generators of a particular module, recall that every module is either a
:py:class:`~.PowerBasis` or a descendant (along a chain of
:py:class:`~.Submodule` objects) thereof, so that in fact every
:py:class:`~.ModuleElement` represents an algebraic number in some field
$\mathbb{Q}(\theta)$, where $\theta$ is the defining element of some
:py:class:`~.PowerBasis`. It thus makes sense to talk about the number field
to which a given :py:class:`~.ModuleElement` belongs.

This means that any two :py:class:`~.ModuleElement` instances can be added,
subtracted, multiplied, or divided, provided they belong to the same number
field. Similarly, since $\mathbb{Q}$ is a subfield of every number field,
any :py:class:`~.ModuleElement` may be added, multiplied, etc. by any
rational number.

>>> from sympy import QQ
>>> from sympy.polys.numberfields.modules import to_col
>>> T = Poly(cyclotomic_poly(5))
>>> A = PowerBasis(T)
>>> C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
>>> e = A(to_col([0, 2, 0, 0]), denom=3)
>>> f = A(to_col([0, 0, 0, 7]), denom=5)
>>> g = C(to_col([1, 1, 1, 1]))
>>> e + f
[0, 10, 0, 21]/15
>>> e - f
[0, 10, 0, -21]/15
>>> e - g
[-9, -7, -9, -9]/3
>>> e + QQ(7, 10)
[21, 20, 0, 0]/30
>>> e * f
[-14, -14, -14, -14]/15
>>> e ** 2
[0, 0, 4, 0]/9
>>> f // g
[7, 7, 7, 7]/15
>>> f * QQ(2, 3)
[0, 0, 0, 14]/15

However, care must be taken with arithmetic operations on
:py:class:`~.ModuleElement`, because the module $C$ to which the result will
belong will be the nearest common ancestor (NCA) of the modules $A$, $B$ to
which the two operands belong, and $C$ may be different from either or both
of $A$ and $B$.

>>> A = PowerBasis(T)
>>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
>>> C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
>>> print((B(0) * C(0)).module == A)
True

Before the arithmetic operation is performed, copies of the two operands are
automatically converted into elements of the NCA (the operands themselves are
not modified). This upward conversion along an ancestor chain is easy: it just
requires the successive multiplication by the defining matrix of each
:py:class:`~.Submodule`.

Conversely, downward conversion, i.e. representing a given
:py:class:`~.ModuleElement` in a submodule, is also supported -- namely by
the :py:meth:`~sympy.polys.numberfields.modules.Submodule.represent` method
-- but is not guaranteed to succeed in general, since the given element may
not belong to the submodule. The main circumstance in which this issue tends
to arise is with multiplication, since modules, while closed under addition,
need not be closed under multiplication.


Multiplication
--------------

Generally speaking, a module need not be closed under multiplication, i.e. need
not form a ring. However, many of the modules we work with in the context of
number fields are in fact rings, and our classes do support multiplication.

Specifically, any :py:class:`~.Module` can attempt to compute its own
multiplication table, but this does not happen unless an attempt is made to
multiply two :py:class:`~.ModuleElement` instances belonging to it.

>>> A = PowerBasis(T)
>>> print(A._mult_tab is None)
True
>>> a = A(0)*A(1)
>>> print(A._mult_tab is None)
False

Every :py:class:`~.PowerBasis` is, by its nature, closed under multiplication,
so instances of :py:class:`~.PowerBasis` can always successfully compute their
multiplication table.

When a :py:class:`~.Submodule` attempts to compute its multiplication table,
it converts each of its own generators into elements of its parent module,
multiplies them there, in every possible pairing, and then tries to
represent the results in itself, i.e. as $\mathbb{Z}$-linear combinations
over its own generators. This will succeed if and only if the submodule is
in fact closed under multiplication.


Module Homomorphisms
====================

Many important number theoretic algorithms require the calculation of the
kernel of one or more module homomorphisms. Accordingly we have several
lightweight classes, :py:class:`~.ModuleHomomorphism`,
:py:class:`~.ModuleEndomorphism`, :py:class:`~.InnerEndomorphism`, and
:py:class:`~.EndomorphismRing`, which provide the minimal necessary machinery
to support this.

"""

from sympy.core.intfunc import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom


def to_col(coeffs):
    r"""Transform a list of integer coefficients into a column vector."""
    return DomainMatrix([[ZZ(c) for c in coeffs]], (1, len(coeffs)), ZZ).transpose()


class Module:
    """
    Generic finitely-generated module.

    This is an abstract base class, and should not be instantiated directly.
    The two concrete subclasses are :py:class:`~.PowerBasis` and
    :py:class:`~.Submodule`.

    Every :py:class:`~.Submodule` is derived from another module, referenced
    by its ``parent`` attribute. If ``S`` is a submodule, then we refer to
    ``S.parent``, ``S.parent.parent``, and so on, as the "ancestors" of
    ``S``. Thus, every :py:class:`~.Module` is either a
    :py:class:`~.PowerBasis` or a :py:class:`~.Submodule`, some ancestor of
    which is a :py:class:`~.PowerBasis`.
    """

    @property
    def n(self):
        """The number of generators of this module."""
        raise NotImplementedError

    def mult_tab(self):
        """
        Get the multiplication table for this module (if closed under mult).

        Explanation
        ===========

        Computes a dictionary ``M`` of dictionaries of lists, representing the
        upper triangular half of the multiplication table.

        In other words, if ``0 <= i <= j < self.n``, then ``M[i][j]`` is the
        list ``c`` of coefficients such that
        ``g[i] * g[j] == sum(c[k]*g[k], k in range(self.n))``,
        where ``g`` is the list of generators of this module.

        If ``j < i`` then ``M[i][j]`` is undefined.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> print(A.mult_tab())  # doctest: +SKIP
        {0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0],     3: [0, 0, 0, 1]},
                          1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1],     3: [-1, -1, -1, -1]},
                                           2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]},
                                                                3: {3: [0, 1, 0, 0]}}

        Returns
        =======

        dict of dict of lists

        Raises
        ======

        ClosureFailure
            If the module is not closed under multiplication.

        """
        raise NotImplementedError

    @property
    def parent(self):
        """
        The parent module, if any, for this module.

        Explanation
        ===========

        For a :py:class:`~.Submodule` this is its ``parent`` attribute; for a
        :py:class:`~.PowerBasis` this is ``None``.

        Returns
        =======

        :py:class:`~.Module`, ``None``

        See Also
        ========

        Module

        """
        return None

    def represent(self, elt):
        r"""
        Represent a module element as an integer-linear combination over the
        generators of this module.

        Explanation
        ===========

        In our system, to "represent" always means to write a
        :py:class:`~.ModuleElement` as a :ref:`ZZ`-linear combination over the
        generators of the present :py:class:`~.Module`. Furthermore, the
        incoming :py:class:`~.ModuleElement` must belong to an ancestor of
        the present :py:class:`~.Module` (or to the present
        :py:class:`~.Module` itself).

        The most common application is to represent a
        :py:class:`~.ModuleElement` in a :py:class:`~.Submodule`. For example,
        this is involved in computing multiplication tables.

        On the other hand, representing in a :py:class:`~.PowerBasis` is an
        odd case, and one which tends not to arise in practice, except for
        example when using a :py:class:`~.ModuleEndomorphism` on a
        :py:class:`~.PowerBasis`.

        In such a case, (1) the incoming :py:class:`~.ModuleElement` must
        belong to the :py:class:`~.PowerBasis` itself (since the latter has no
        proper ancestors) and (2) it is "representable" iff it belongs to
        $\mathbb{Z}[\theta]$ (although generally a
        :py:class:`~.PowerBasisElement` may represent any element of
        $\mathbb{Q}(\theta)$, i.e. any algebraic number).

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
        >>> from sympy.abc import zeta
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> a = A(to_col([2, 4, 6, 8]))

        The :py:class:`~.ModuleElement` ``a`` has all even coefficients.
        If we represent ``a`` in the submodule ``B = 2*A``, the coefficients in
        the column vector will be halved:

        >>> B = A.submodule_from_gens([2*A(i) for i in range(4)])
        >>> b = B.represent(a)
        >>> print(b.transpose())  # doctest: +SKIP
        DomainMatrix([[1, 2, 3, 4]], (1, 4), ZZ)

        However, the element of ``B`` so defined still represents the same
        algebraic number:

        >>> print(a.poly(zeta).as_expr())
        8*zeta**3 + 6*zeta**2 + 4*zeta + 2
        >>> print(B(b).over_power_basis().poly(zeta).as_expr())
        8*zeta**3 + 6*zeta**2 + 4*zeta + 2

        Parameters
        ==========

        elt : :py:class:`~.ModuleElement`
            The module element to be represented. Must belong to some ancestor
            module of this module (including this module itself).

        Returns
        =======

        :py:class:`~.DomainMatrix` over :ref:`ZZ`
            This will be a column vector, representing the coefficients of a
            linear combination of this module's generators, which equals the
            given element.

        Raises
        ======

        ClosureFailure
            If the given element cannot be represented as a :ref:`ZZ`-linear
            combination over this module.

        See Also
        ========

        .Submodule.represent
        .PowerBasis.represent

        """
        raise NotImplementedError

    def ancestors(self, include_self=False):
        """
        Return the list of ancestor modules of this module, from the
        foundational :py:class:`~.PowerBasis` downward, optionally including
        ``self``.

        See Also
        ========

        Module

        """
        c = self.parent
        a = [] if c is None else c.ancestors(include_self=True)
        if include_self:
            a.append(self)
        return a

    def power_basis_ancestor(self):
        """
        Return the :py:class:`~.PowerBasis` that is an ancestor of this module.

        See Also
        ========

        Module

        """
        if isinstance(self, PowerBasis):
            return self
        c = self.parent
        if c is not None:
            return c.power_basis_ancestor()
        return None

    def nearest_common_ancestor(self, other):
        """
        Locate the nearest common ancestor of this module and another.

        Returns
        =======

        :py:class:`~.Module`, ``None``

        See Also
        ========

        Module

        """
        sA = self.ancestors(include_self=True)
        oA = other.ancestors(include_self=True)
        nca = None
        for sa, oa in zip(sA, oA):
            if sa == oa:
                nca = sa
            else:
                break
        return nca

    @property
    def number_field(self):
        r"""
        Return the associated :py:class:`~.AlgebraicField`, if any.

        Explanation
        ===========

        A :py:class:`~.PowerBasis` can be constructed on a :py:class:`~.Poly`
        $f$ or on an :py:class:`~.AlgebraicField` $K$. In the latter case, the
        :py:class:`~.PowerBasis` and all its descendant modules will return $K$
        as their ``.number_field`` property, while in the former case they will
        all return ``None``.

        Returns
        =======

        :py:class:`~.AlgebraicField`, ``None``

        """
        return self.power_basis_ancestor().number_field

    def is_compat_col(self, col):
        """Say whether *col* is a suitable column vector for this module."""
        return isinstance(col, DomainMatrix) and col.shape == (self.n, 1) and col.domain.is_ZZ

    def __call__(self, spec, denom=1):
        r"""
        Generate a :py:class:`~.ModuleElement` belonging to this module.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> e = A(to_col([1, 2, 3, 4]), denom=3)
        >>> print(e)  # doctest: +SKIP
        [1, 2, 3, 4]/3
        >>> f = A(2)
        >>> print(f)  # doctest: +SKIP
        [0, 0, 1, 0]

        Parameters
        ==========

        spec : :py:class:`~.DomainMatrix`, int
            Specifies the numerators of the coefficients of the
            :py:class:`~.ModuleElement`. Can be either a column vector over
            :ref:`ZZ`, whose length must equal the number $n$ of generators of
            this module, or else an integer ``j``, $0 \leq j < n$, which is a
            shorthand for column $j$ of $I_n$, the $n \times n$ identity
            matrix.
        denom : int, optional (default=1)
            Denominator for the coefficients of the
            :py:class:`~.ModuleElement`.

        Returns
        =======

        :py:class:`~.ModuleElement`
            The coefficients are the entries of the *spec* vector, divided by
            *denom*.

        """
        if isinstance(spec, int) and 0 <= spec < self.n:
            spec = DomainMatrix.eye(self.n, ZZ)[:, spec].to_dense()
        if not self.is_compat_col(spec):
            raise ValueError('Compatible column vector required.')
        return make_mod_elt(self, spec, denom=denom)

    def starts_with_unity(self):
        """Say whether the module's first generator equals unity."""
        raise NotImplementedError

    def basis_elements(self):
        """
        Get list of :py:class:`~.ModuleElement` being the generators of this
        module.
        """
        return [self(j) for j in range(self.n)]

    def zero(self):
        """Return a :py:class:`~.ModuleElement` representing zero."""
        return self(0) * 0

    def one(self):
        """
        Return a :py:class:`~.ModuleElement` representing unity,
        and belonging to the first ancestor of this module (including
        itself) that starts with unity.
        """
        return self.element_from_rational(1)

    def element_from_rational(self, a):
        """
        Return a :py:class:`~.ModuleElement` representing a rational number.

        Explanation
        ===========

        The returned :py:class:`~.ModuleElement` will belong to the first
        module on this module's ancestor chain (including this module
        itself) that starts with unity.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, QQ
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> a = A.element_from_rational(QQ(2, 3))
        >>> print(a)  # doctest: +SKIP
        [2, 0, 0, 0]/3

        Parameters
        ==========

        a : int, :ref:`ZZ`, :ref:`QQ`

        Returns
        =======

        :py:class:`~.ModuleElement`

        """
        raise NotImplementedError

    def submodule_from_gens(self, gens, hnf=True, hnf_modulus=None):
        """
        Form the submodule generated by a list of :py:class:`~.ModuleElement`
        belonging to this module.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> gens = [A(0), 2*A(1), 3*A(2), 4*A(3)//5]
        >>> B = A.submodule_from_gens(gens)
        >>> print(B)  # doctest: +SKIP
        Submodule[[5, 0, 0, 0], [0, 10, 0, 0], [0, 0, 15, 0], [0, 0, 0, 4]]/5

        Parameters
        ==========

        gens : list of :py:class:`~.ModuleElement` belonging to this module.
        hnf : boolean, optional (default=True)
            If True, we will reduce the matrix into Hermite Normal Form before
            forming the :py:class:`~.Submodule`.
        hnf_modulus : int, None, optional (default=None)
            Modulus for use in the HNF reduction algorithm. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        See Also
        ========

        submodule_from_matrix

        """
        if not all(g.module == self for g in gens):
            raise ValueError('Generators must belong to this module.')
        n = len(gens)
        if n == 0:
            raise ValueError('Need at least one generator.')
        m = gens[0].n
        d = gens[0].denom if n == 1 else ilcm(*[g.denom for g in gens])
        B = DomainMatrix.zeros((m, 0), ZZ).hstack(*[(d // g.denom) * g.col for g in gens])
        if hnf:
            B = hermite_normal_form(B, D=hnf_modulus)
        return self.submodule_from_matrix(B, denom=d)

    def submodule_from_matrix(self, B, denom=1):
        """
        Form the submodule generated by the elements of this module indicated
        by the columns of a matrix, with an optional denominator.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.polys.matrices import DM
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(DM([
        ...     [0, 10, 0, 0],
        ...     [0,  0, 7, 0],
        ... ], ZZ).transpose(), denom=15)
        >>> print(B)  # doctest: +SKIP
        Submodule[[0, 10, 0, 0], [0, 0, 7, 0]]/15

        Parameters
        ==========

        B : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Each column gives the numerators of the coefficients of one
            generator of the submodule. Thus, the number of rows of *B* must
            equal the number of generators of the present module.
        denom : int, optional (default=1)
            Common denominator for all generators of the submodule.

        Returns
        =======

        :py:class:`~.Submodule`

        Raises
        ======

        ValueError
            If the given matrix *B* is not over :ref:`ZZ` or its number of rows
            does not equal the number of generators of the present module.

        See Also
        ========

        submodule_from_gens

        """
        m, n = B.shape
        if not B.domain.is_ZZ:
            raise ValueError('Matrix must be over ZZ.')
        if not m == self.n:
            raise ValueError('Matrix row count must match base module.')
        return Submodule(self, B, denom=denom)

    def whole_submodule(self):
        """
        Return a submodule equal to this entire module.

        Explanation
        ===========

        This is useful when you have a :py:class:`~.PowerBasis` and want to
        turn it into a :py:class:`~.Submodule` (in order to use methods
        belonging to the latter).

        """
        B = DomainMatrix.eye(self.n, ZZ)
        return self.submodule_from_matrix(B)

    def endomorphism_ring(self):
        """Form the :py:class:`~.EndomorphismRing` for this module."""
        return EndomorphismRing(self)


class PowerBasis(Module):
    """The module generated by the powers of an algebraic integer."""

    def __init__(self, T):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`
            Either (1) the monic, irreducible, univariate polynomial over
            :ref:`ZZ`, a root of which is the generator of the power basis,
            or (2) an :py:class:`~.AlgebraicField` whose primitive element
            is the generator of the power basis.

        """
        K = None
        if isinstance(T, AlgebraicField):
            K, T = T, T.ext.minpoly_of_element()
        # Sometimes incoming Polys are formally over QQ, although all their
        # coeffs are integral. We want them to be formally over ZZ.
        T = T.set_domain(ZZ)
        self.K = K
        self.T = T
        self._n = T.degree()
        self._mult_tab = None

    @property
    def number_field(self):
        return self.K

    def __repr__(self):
        return f'PowerBasis({self.T.as_expr()})'

    def __eq__(self, other):
        if isinstance(other, PowerBasis):
            return self.T == other.T
        return NotImplemented

    @property
    def n(self):
        return self._n

    def mult_tab(self):
        if self._mult_tab is None:
            self.compute_mult_tab()
        return self._mult_tab

    def compute_mult_tab(self):
        theta_pow = AlgIntPowers(self.T)
        M = {}
        n = self.n
        for u in range(n):
            M[u] = {}
            for v in range(u, n):
                M[u][v] = theta_pow[u + v]
        self._mult_tab = M

    def represent(self, elt):
        r"""
        Represent a module element as an integer-linear combination over the
        generators of this module.

        See Also
        ========

        .Module.represent
        .Submodule.represent

        """
        if elt.module == self and elt.denom == 1:
            return elt.column()
        else:
            raise ClosureFailure('Element not representable in ZZ[theta].')

    def starts_with_unity(self):
        return True

    def element_from_rational(self, a):
        return self(0) * a

    def element_from_poly(self, f):
        """
        Produce an element of this module, representing *f* after reduction mod
        our defining minimal polynomial.

        Parameters
        ==========

        f : :py:class:`~.Poly` over :ref:`ZZ` in same var as our defining poly.

        Returns
        =======

        :py:class:`~.PowerBasisElement`

        """
        n, k = self.n, f.degree()
        if k >= n:
            f = f % self.T
        if f == 0:
            return self.zero()
        d, c = dup_clear_denoms(f.rep.to_list(), QQ, convert=True)
        c = list(reversed(c))
        ell = len(c)
        z = [ZZ(0)] * (n - ell)
        col = to_col(c + z)
        return self(col, denom=d)

    def _element_from_rep_and_mod(self, rep, mod):
        """
        Produce a PowerBasisElement representing a given algebraic number.

        Parameters
        ==========

        rep : list of coeffs
            Represents the number as polynomial in the primitive element of the
            field.

        mod : list of coeffs
            Represents the minimal polynomial of the primitive element of the
            field.

        Returns
        =======

        :py:class:`~.PowerBasisElement`

        """
        if mod != self.T.rep.to_list():
            raise UnificationFailed('Element does not appear to be in the same field.')
        return self.element_from_poly(Poly(rep, self.T.gen))

    def element_from_ANP(self, a):
        """Convert an ANP into a PowerBasisElement. """
        return self._element_from_rep_and_mod(a.to_list(), a.mod_to_list())

    def element_from_alg_num(self, a):
        """Convert an AlgebraicNumber into a PowerBasisElement. """
        return self._element_from_rep_and_mod(a.rep.to_list(), a.minpoly.rep.to_list())


class Submodule(Module, IntegerPowerable):
    """A submodule of another module."""

    def __init__(self, parent, matrix, denom=1, mult_tab=None):
        """
        Parameters
        ==========

        parent : :py:class:`~.Module`
            The module from which this one is derived.
        matrix : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            The matrix whose columns define this submodule's generators as
            linear combinations over the parent's generators.
        denom : int, optional (default=1)
            Denominator for the coefficients given by the matrix.
        mult_tab : dict, ``None``, optional
            If already known, the multiplication table for this module may be
            supplied.

        """
        self._parent = parent
        self._matrix = matrix
        self._denom = denom
        self._mult_tab = mult_tab
        self._n = matrix.shape[1]
        self._QQ_matrix = None
        self._starts_with_unity = None
        self._is_sq_maxrank_HNF = None

    def __repr__(self):
        r = 'Submodule' + repr(self.matrix.transpose().to_Matrix().tolist())
        if self.denom > 1:
            r += f'/{self.denom}'
        return r

    def reduced(self):
        """
        Produce a reduced version of this submodule.

        Explanation
        ===========

        In the reduced version, it is guaranteed that 1 is the only positive
        integer dividing both the submodule's denominator, and every entry in
        the submodule's matrix.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        if self.denom == 1:
            return self
        g = igcd(self.denom, *self.coeffs)
        if g == 1:
            return self
        return type(self)(self.parent, (self.matrix / g).convert_to(ZZ), denom=self.denom // g, mult_tab=self._mult_tab)

    def discard_before(self, r):
        """
        Produce a new module by discarding all generators before a given
        index *r*.
        """
        W = self.matrix[:, r:]
        s = self.n - r
        M = None
        mt = self._mult_tab
        if mt is not None:
            M = {}
            for u in range(s):
                M[u] = {}
                for v in range(u, s):
                    M[u][v] = mt[r + u][r + v][r:]
        return Submodule(self.parent, W, denom=self.denom, mult_tab=M)

    @property
    def n(self):
        return self._n

    def mult_tab(self):
        if self._mult_tab is None:
            self.compute_mult_tab()
        return self._mult_tab

    def compute_mult_tab(self):
        gens = self.basis_element_pullbacks()
        M = {}
        n = self.n
        for u in range(n):
            M[u] = {}
            for v in range(u, n):
                M[u][v] = self.represent(gens[u] * gens[v]).flat()
        self._mult_tab = M

    @property
    def parent(self):
        return self._parent

    @property
    def matrix(self):
        return self._matrix

    @property
    def coeffs(self):
        return self.matrix.flat()

    @property
    def denom(self):
        return self._denom

    @property
    def QQ_matrix(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.matrix / self.denom``, and guaranteed to be dense.

        Explanation
        ===========

        Depending on how it is formed, a :py:class:`~.DomainMatrix` may have
        an internal representation that is sparse or dense. We guarantee a
        dense representation here, so that tests for equivalence of submodules
        always come out as expected.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.abc import x
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5, x))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(3*DomainMatrix.eye(4, ZZ), denom=6)
        >>> C = A.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
        >>> print(B.QQ_matrix == C.QQ_matrix)
        True

        Returns
        =======

        :py:class:`~.DomainMatrix` over :ref:`QQ`

        """
        if self._QQ_matrix is None:
            self._QQ_matrix = (self.matrix / self.denom).to_dense()
        return self._QQ_matrix

    def starts_with_unity(self):
        if self._starts_with_unity is None:
            self._starts_with_unity = self(0).equiv(1)
        return self._starts_with_unity

    def is_sq_maxrank_HNF(self):
        if self._is_sq_maxrank_HNF is None:
            self._is_sq_maxrank_HNF = is_sq_maxrank_HNF(self._matrix)
        return self._is_sq_maxrank_HNF

    def is_power_basis_submodule(self):
        return isinstance(self.parent, PowerBasis)

    def element_from_rational(self, a):
        if self.starts_with_unity():
            return self(0) * a
        else:
            return self.parent.element_from_rational(a)

    def basis_element_pullbacks(self):
        """
        Return list of this submodule's basis elements as elements of the
        submodule's parent module.
        """
        return [e.to_parent() for e in self.basis_elements()]

    def represent(self, elt):
        """
        Represent a module element as an integer-linear combination over the
        generators of this module.

        See Also
        ========

        .Module.represent
        .PowerBasis.represent

        """
        if elt.module == self:
            return elt.column()
        elif elt.module == self.parent:
            try:
                # The given element should be a ZZ-linear combination over our
                # basis vectors; however, due to the presence of denominators,
                # we need to solve over QQ.
                A = self.QQ_matrix
                b = elt.QQ_col
                x = A._solve(b)[0].transpose()
                x = x.convert_to(ZZ)
            except DMBadInputError:
                raise ClosureFailure('Element outside QQ-span of this basis.')
            except CoercionFailed:
                raise ClosureFailure('Element in QQ-span but not ZZ-span of this basis.')
            return x
        elif isinstance(self.parent, Submodule):
            coeffs_in_parent = self.parent.represent(elt)
            parent_element = self.parent(coeffs_in_parent)
            return self.represent(parent_element)
        else:
            raise ClosureFailure('Element outside ancestor chain of this module.')

    def is_compat_submodule(self, other):
        return isinstance(other, Submodule) and other.parent == self.parent

    def __eq__(self, other):
        if self.is_compat_submodule(other):
            return other.QQ_matrix == self.QQ_matrix
        return NotImplemented

    def add(self, other, hnf=True, hnf_modulus=None):
        """
        Add this :py:class:`~.Submodule` to another.

        Explanation
        ===========

        This represents the module generated by the union of the two modules'
        sets of generators.

        Parameters
        ==========

        other : :py:class:`~.Submodule`
        hnf : boolean, optional (default=True)
            If ``True``, reduce the matrix of the combined module to its
            Hermite Normal Form.
        hnf_modulus : :ref:`ZZ`, None, optional
            If a positive integer is provided, use this as modulus in the
            HNF reduction. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        d, e = self.denom, other.denom
        m = ilcm(d, e)
        a, b = m // d, m // e
        B = (a * self.matrix).hstack(b * other.matrix)
        if hnf:
            B = hermite_normal_form(B, D=hnf_modulus)
        return self.parent.submodule_from_matrix(B, denom=m)

    def __add__(self, other):
        if self.is_compat_submodule(other):
            return self.add(other)
        return NotImplemented

    __radd__ = __add__

    def mul(self, other, hnf=True, hnf_modulus=None):
        """
        Multiply this :py:class:`~.Submodule` by a rational number, a
        :py:class:`~.ModuleElement`, or another :py:class:`~.Submodule`.

        Explanation
        ===========

        To multiply by a rational number or :py:class:`~.ModuleElement` means
        to form the submodule whose generators are the products of this
        quantity with all the generators of the present submodule.

        To multiply by another :py:class:`~.Submodule` means to form the
        submodule whose generators are all the products of one generator from
        the one submodule, and one generator from the other.

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`, :py:class:`~.Submodule`
        hnf : boolean, optional (default=True)
            If ``True``, reduce the matrix of the product module to its
            Hermite Normal Form.
        hnf_modulus : :ref:`ZZ`, None, optional
            If a positive integer is provided, use this as modulus in the
            HNF reduction. See
            :py:func:`~sympy.polys.matrices.normalforms.hermite_normal_form`.

        Returns
        =======

        :py:class:`~.Submodule`

        """
        if is_rat(other):
            a, b = get_num_denom(other)
            if a == b == 1:
                return self
            else:
                return Submodule(self.parent,
                             self.matrix * a, denom=self.denom * b,
                             mult_tab=None).reduced()
        elif isinstance(other, ModuleElement) and other.module == self.parent:
            # The submodule is multiplied by an element of the parent module.
            # We presume this means we want a new submodule of the parent module.
            gens = [other * e for e in self.basis_element_pullbacks()]
            return self.parent.submodule_from_gens(gens, hnf=hnf, hnf_modulus=hnf_modulus)
        elif self.is_compat_submodule(other):
            # This case usually means you're multiplying ideals, and want another
            # ideal, i.e. another submodule of the same parent module.
            alphas, betas = self.basis_element_pullbacks(), other.basis_element_pullbacks()
            gens = [a * b for a in alphas for b in betas]
            return self.parent.submodule_from_gens(gens, hnf=hnf, hnf_modulus=hnf_modulus)
        return NotImplemented

    def __mul__(self, other):
        return self.mul(other)

    __rmul__ = __mul__

    def _first_power(self):
        return self

    def reduce_element(self, elt):
        r"""
        If this submodule $B$ has defining matrix $W$ in square, maximal-rank
        Hermite normal form, then, given an element $x$ of the parent module
        $A$, we produce an element $y \in A$ such that $x - y \in B$, and the
        $i$th coordinate of $y$ satisfies $0 \leq y_i < w_{i,i}$. This
        representative $y$ is unique, in the sense that every element of
        the coset $x + B$ reduces to it under this procedure.

        Explanation
        ===========

        In the special case where $A$ is a power basis for a number field $K$,
        and $B$ is a submodule representing an ideal $I$, this operation
        represents one of a few important ways of reducing an element of $K$
        modulo $I$ to obtain a "small" representative. See [Cohen00]_ Section
        1.4.3.

        Examples
        ========

        >>> from sympy import QQ, Poly, symbols
        >>> t = symbols('t')
        >>> k = QQ.alg_field_from_poly(Poly(t**3 + t**2 - 2*t + 8))
        >>> Zk = k.maximal_order()
        >>> A = Zk.parent
        >>> B = (A(2) - 3*A(0))*Zk
        >>> B.reduce_element(A(2))
        [3, 0, 0]

        Parameters
        ==========

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Returns
        =======

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Raises
        ======

        NotImplementedError
            If the given :py:class:`~.ModuleElement` does not belong to this
            submodule's parent module.
        StructureError
            If this submodule's defining matrix is not in square, maximal-rank
            Hermite normal form.

        References
        ==========

        .. [Cohen00] Cohen, H. *Advanced Topics in Computational Number
           Theory.*

        """
        if not elt.module == self.parent:
            raise NotImplementedError
        if not self.is_sq_maxrank_HNF():
            msg = "Reduction not implemented unless matrix square max-rank HNF"
            raise StructureError(msg)
        B = self.basis_element_pullbacks()
        a = elt
        for i in range(self.n - 1, -1, -1):
            b = B[i]
            q = a.coeffs[i]*b.denom // (b.coeffs[i]*a.denom)
            a -= q*b
        return a


def is_sq_maxrank_HNF(dm):
    r"""
    Say whether a :py:class:`~.DomainMatrix` is in that special case of Hermite
    Normal Form, in which the matrix is also square and of maximal rank.

    Explanation
    ===========

    We commonly work with :py:class:`~.Submodule` instances whose matrix is in
    this form, and it can be useful to be able to check that this condition is
    satisfied.

    For example this is the case with the :py:class:`~.Submodule` ``ZK``
    returned by :py:func:`~sympy.polys.numberfields.basis.round_two`, which
    represents the maximal order in a number field, and with ideals formed
    therefrom, such as ``2 * ZK``.

    """
    if dm.domain.is_ZZ and dm.is_square and dm.is_upper:
        n = dm.shape[0]
        for i in range(n):
            d = dm[i, i].element
            if d <= 0:
                return False
            for j in range(i + 1, n):
                if not (0 <= dm[i, j].element < d):
                    return False
        return True
    return False


def make_mod_elt(module, col, denom=1):
    r"""
    Factory function which builds a :py:class:`~.ModuleElement`, but ensures
    that it is a :py:class:`~.PowerBasisElement` if the module is a
    :py:class:`~.PowerBasis`.
    """
    if isinstance(module, PowerBasis):
        return PowerBasisElement(module, col, denom=denom)
    else:
        return ModuleElement(module, col, denom=denom)


class ModuleElement(IntegerPowerable):
    r"""
    Represents an element of a :py:class:`~.Module`.

    NOTE: Should not be constructed directly. Use the
    :py:meth:`~.Module.__call__` method or the :py:func:`make_mod_elt()`
    factory function instead.
    """

    def __init__(self, module, col, denom=1):
        """
        Parameters
        ==========

        module : :py:class:`~.Module`
            The module to which this element belongs.
        col : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Column vector giving the numerators of the coefficients of this
            element.
        denom : int, optional (default=1)
            Denominator for the coefficients of this element.

        """
        self.module = module
        self.col = col
        self.denom = denom
        self._QQ_col = None

    def __repr__(self):
        r = str([int(c) for c in self.col.flat()])
        if self.denom > 1:
            r += f'/{self.denom}'
        return r

    def reduced(self):
        """
        Produce a reduced version of this ModuleElement, i.e. one in which the
        gcd of the denominator together with all numerator coefficients is 1.
        """
        if self.denom == 1:
            return self
        g = igcd(self.denom, *self.coeffs)
        if g == 1:
            return self
        return type(self)(self.module,
                            (self.col / g).convert_to(ZZ),
                            denom=self.denom // g)

    def reduced_mod_p(self, p):
        """
        Produce a version of this :py:class:`~.ModuleElement` in which all
        numerator coefficients have been reduced mod *p*.
        """
        return make_mod_elt(self.module,
                            self.col.convert_to(FF(p)).convert_to(ZZ),
                            denom=self.denom)

    @classmethod
    def from_int_list(cls, module, coeffs, denom=1):
        """
        Make a :py:class:`~.ModuleElement` from a list of ints (instead of a
        column vector).
        """
        col = to_col(coeffs)
        return cls(module, col, denom=denom)

    @property
    def n(self):
        """The length of this element's column."""
        return self.module.n

    def __len__(self):
        return self.n

    def column(self, domain=None):
        """
        Get a copy of this element's column, optionally converting to a domain.
        """
        if domain is None:
            return self.col.copy()
        else:
            return self.col.convert_to(domain)

    @property
    def coeffs(self):
        return self.col.flat()

    @property
    def QQ_col(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.col / self.denom``, and guaranteed to be dense.

        See Also
        ========

        .Submodule.QQ_matrix

        """
        if self._QQ_col is None:
            self._QQ_col = (self.col / self.denom).to_dense()
        return self._QQ_col

    def to_parent(self):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to the parent of
        this element's module.
        """
        if not isinstance(self.module, Submodule):
            raise ValueError('Not an element of a Submodule.')
        return make_mod_elt(
            self.module.parent, self.module.matrix * self.col,
            denom=self.module.denom * self.denom)

    def to_ancestor(self, anc):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to a given
        ancestor of this element's module.

        Parameters
        ==========

        anc : :py:class:`~.Module`

        """
        if anc == self.module:
            return self
        else:
            return self.to_parent().to_ancestor(anc)

    def over_power_basis(self):
        """
        Transform into a :py:class:`~.PowerBasisElement` over our
        :py:class:`~.PowerBasis` ancestor.
        """
        e = self
        while not isinstance(e.module, PowerBasis):
            e = e.to_parent()
        return e

    def is_compat(self, other):
        """
        Test whether other is another :py:class:`~.ModuleElement` with same
        module.
        """
        return isinstance(other, ModuleElement) and other.module == self.module

    def unify(self, other):
        """
        Try to make a compatible pair of :py:class:`~.ModuleElement`, one
        equivalent to this one, and one equivalent to the other.

        Explanation
        ===========

        We search for the nearest common ancestor module for the pair of
        elements, and represent each one there.

        Returns
        =======

        Pair ``(e1, e2)``
            Each ``ei`` is a :py:class:`~.ModuleElement`, they belong to the
            same :py:class:`~.Module`, ``e1`` is equivalent to ``self``, and
            ``e2`` is equivalent to ``other``.

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` have no common ancestor module.

        """
        if self.module == other.module:
            return self, other
        nca = self.module.nearest_common_ancestor(other.module)
        if nca is not None:
            return self.to_ancestor(nca), other.to_ancestor(nca)
        raise UnificationFailed(f"Cannot unify {self} with {other}")

    def __eq__(self, other):
        if self.is_compat(other):
            return self.QQ_col == other.QQ_col
        return NotImplemented

    def equiv(self, other):
        """
        A :py:class:`~.ModuleElement` may test as equivalent to a rational
        number or another :py:class:`~.ModuleElement`, if they represent the
        same algebraic number.

        Explanation
        ===========

        This method is intended to check equivalence only in those cases in
        which it is easy to test; namely, when *other* is either a
        :py:class:`~.ModuleElement` that can be unified with this one (i.e. one
        which shares a common :py:class:`~.PowerBasis` ancestor), or else a
        rational number (which is easy because every :py:class:`~.PowerBasis`
        represents every rational number).

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`

        Returns
        =======

        bool

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` do not share a common
            :py:class:`~.PowerBasis` ancestor.

        """
        if self == other:
            return True
        elif isinstance(other, ModuleElement):
            a, b = self.unify(other)
            return a == b
        elif is_rat(other):
            if isinstance(self, PowerBasisElement):
                return self == self.module(0) * other
            else:
                return self.over_power_basis().equiv(other)
        return False

    def __add__(self, other):
        """
        A :py:class:`~.ModuleElement` can be added to a rational number, or to
        another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the other summand is a rational number, it will be converted into
        a :py:class:`~.ModuleElement` (belonging to the first ancestor of this
        module that starts with unity).

        In all cases, the sum belongs to the nearest common ancestor (NCA) of
        the modules of the two summands. If the NCA does not exist, we return
        ``NotImplemented``.
        """
        if self.is_compat(other):
            d, e = self.denom, other.denom
            m = ilcm(d, e)
            u, v = m // d, m // e
            col = to_col([u * a + v * b for a, b in zip(self.coeffs, other.coeffs)])
            return type(self)(self.module, col, denom=m).reduced()
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                return NotImplemented
            return a + b
        elif is_rat(other):
            return self + self.module.element_from_rational(other)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        """
        A :py:class:`~.ModuleElement` can be multiplied by a rational number,
        or by another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the multiplier is a rational number, the product is computed by
        operating directly on the coefficients of this
        :py:class:`~.ModuleElement`.

        When the multiplier is another :py:class:`~.ModuleElement`, the product
        will belong to the nearest common ancestor (NCA) of the modules of the
        two operands, and that NCA must have a multiplication table. If the NCA
        does not exist, we return ``NotImplemented``. If the NCA does not have
        a mult. table, ``ClosureFailure`` will be raised.
        """
        if self.is_compat(other):
            M = self.module.mult_tab()
            A, B = self.col.flat(), other.col.flat()
            n = self.n
            C = [0] * n
            for u in range(n):
                for v in range(u, n):
                    c = A[u] * B[v]
                    if v > u:
                        c += A[v] * B[u]
                    if c != 0:
                        R = M[u][v]
                        for k in range(n):
                            C[k] += c * R[k]
            d = self.denom * other.denom
            return self.from_int_list(self.module, C, denom=d)
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                return NotImplemented
            return a * b
        elif is_rat(other):
            a, b = get_num_denom(other)
            if a == b == 1:
                return self
            else:
                return make_mod_elt(self.module,
                                 self.col * a, denom=self.denom * b).reduced()
        return NotImplemented

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.module.one()

    def _first_power(self):
        return self

    def __floordiv__(self, a):
        if is_rat(a):
            a = QQ(a)
            return self * (1/a)
        elif isinstance(a, ModuleElement):
            return self * (1//a)
        return NotImplemented

    def __rfloordiv__(self, a):
        return a // self.over_power_basis()

    def __mod__(self, m):
        r"""
        Reduce this :py:class:`~.ModuleElement` mod a :py:class:`~.Submodule`.

        Parameters
        ==========

        m : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.Submodule`
            If a :py:class:`~.Submodule`, reduce ``self`` relative to this.
            If an integer or rational, reduce relative to the
            :py:class:`~.Submodule` that is our own module times this constant.

        See Also
        ========

        .Submodule.reduce_element

        """
        if is_rat(m):
            m = m * self.module.whole_submodule()
        if isinstance(m, Submodule) and m.parent == self.module:
            return m.reduce_element(self)
        return NotImplemented


class PowerBasisElement(ModuleElement):
    r"""
    Subclass for :py:class:`~.ModuleElement` instances whose module is a
    :py:class:`~.PowerBasis`.
    """

    @property
    def T(self):
        """Access the defining polynomial of the :py:class:`~.PowerBasis`."""
        return self.module.T

    def numerator(self, x=None):
        """Obtain the numerator as a polynomial over :ref:`ZZ`."""
        x = x or self.T.gen
        return Poly(reversed(self.coeffs), x, domain=ZZ)

    def poly(self, x=None):
        """Obtain the number as a polynomial over :ref:`QQ`."""
        return self.numerator(x=x) // self.denom

    @property
    def is_rational(self):
        """Say whether this element represents a rational number."""
        return self.col[1:, :].is_zero_matrix

    @property
    def generator(self):
        """
        Return a :py:class:`~.Symbol` to be used when expressing this element
        as a polynomial.

        If we have an associated :py:class:`~.AlgebraicField` whose primitive
        element has an alias symbol, we use that. Otherwise we use the variable
        of the minimal polynomial defining the power basis to which we belong.
        """
        K = self.module.number_field
        return K.ext.alias if K and K.ext.is_aliased else self.T.gen

    def as_expr(self, x=None):
        """Create a Basic expression from ``self``. """
        return self.poly(x or self.generator).as_expr()

    def norm(self, T=None):
        """Compute the norm of this number."""
        T = T or self.T
        x = T.gen
        A = self.numerator(x=x)
        return T.resultant(A) // self.denom ** self.n

    def inverse(self):
        f = self.poly()
        f_inv = f.invert(self.T)
        return self.module.element_from_poly(f_inv)

    def __rfloordiv__(self, a):
        return self.inverse() * a

    def _negative_power(self, e, modulo=None):
        return self.inverse() ** abs(e)

    def to_ANP(self):
        """Convert to an equivalent :py:class:`~.ANP`. """
        return ANP(list(reversed(self.QQ_col.flat())), QQ.map(self.T.rep.to_list()), QQ)

    def to_alg_num(self):
        """
        Try to convert to an equivalent :py:class:`~.AlgebraicNumber`.

        Explanation
        ===========

        In general, the conversion from an :py:class:`~.AlgebraicNumber` to a
        :py:class:`~.PowerBasisElement` throws away information, because an
        :py:class:`~.AlgebraicNumber` specifies a complex embedding, while a
        :py:class:`~.PowerBasisElement` does not. However, in some cases it is
        possible to convert a :py:class:`~.PowerBasisElement` back into an
        :py:class:`~.AlgebraicNumber`, namely when the associated
        :py:class:`~.PowerBasis` has a reference to an
        :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicNumber`

        Raises
        ======

        StructureError
            If the :py:class:`~.PowerBasis` to which this element belongs does
            not have an associated :py:class:`~.AlgebraicField`.

        """
        K = self.module.number_field
        if K:
            return K.to_alg_num(self.to_ANP())
        raise StructureError("No associated AlgebraicField")


class ModuleHomomorphism:
    r"""A homomorphism from one module to another."""

    def __init__(self, domain, codomain, mapping):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain of the mapping.

        codomain : :py:class:`~.Module`
            The codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module homomorphism. In particular, should
            accept elements of *domain* and return elements of *codomain*.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_gens([2*A(j) for j in range(4)])
        >>> phi = ModuleHomomorphism(A, B, lambda x: 6*x)
        >>> print(phi.matrix())  # doctest: +SKIP
        DomainMatrix([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]], (4, 4), ZZ)

        """
        self.domain = domain
        self.codomain = codomain
        self.mapping = mapping

    def matrix(self, modulus=None):
        r"""
        Compute the matrix of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the matrix should be reduced mod
            $p$.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            The matrix is over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        basis = self.domain.basis_elements()
        cols = [self.codomain.represent(self.mapping(elt)) for elt in basis]
        if not cols:
            return DomainMatrix.zeros((self.codomain.n, 0), ZZ).to_dense()
        M = cols[0].hstack(*cols[1:])
        if modulus:
            M = M.convert_to(FF(modulus))
        return M

    def kernel(self, modulus=None):
        r"""
        Compute a Submodule representing the kernel of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the kernel should be computed mod
            $p$.

        Returns
        =======

        :py:class:`~.Submodule`
            This submodule's generators span the kernel of this
            homomorphism over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        M = self.matrix(modulus=modulus)
        if modulus is None:
            M = M.convert_to(QQ)
        # Note: Even when working over a finite field, what we want here is
        # the pullback into the integers, so in this case the conversion to ZZ
        # below is appropriate. When working over ZZ, the kernel should be a
        # ZZ-submodule, so, while the conversion to QQ above was required in
        # order for the nullspace calculation to work, conversion back to ZZ
        # afterward should always work.
        # TODO:
        #  Watch <https://github.com/sympy/sympy/issues/21834>, which calls
        #  for fraction-free algorithms. If this is implemented, we can skip
        #  the conversion to `QQ` above.
        K = M.nullspace().convert_to(ZZ).transpose()
        return self.domain.submodule_from_matrix(K)


class ModuleEndomorphism(ModuleHomomorphism):
    r"""A homomorphism from one module to itself."""

    def __init__(self, domain, mapping):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The common domain and codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module endomorphism. In particular, should
            accept and return elements of *domain*.

        """
        super().__init__(domain, domain, mapping)


class InnerEndomorphism(ModuleEndomorphism):
    r"""
    An inner endomorphism on a module, i.e. the endomorphism corresponding to
    multiplication by a fixed element.
    """

    def __init__(self, domain, multiplier):
        r"""
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphism.

        multiplier : :py:class:`~.ModuleElement`
            The element $a$ defining the mapping as $x \mapsto a x$.

        """
        super().__init__(domain, lambda x: multiplier * x)
        self.multiplier = multiplier


class EndomorphismRing:
    r"""The ring of endomorphisms on a module."""

    def __init__(self, domain):
        """
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphisms.

        """
        self.domain = domain

    def inner_endomorphism(self, multiplier):
        r"""
        Form an inner endomorphism belonging to this endomorphism ring.

        Parameters
        ==========

        multiplier : :py:class:`~.ModuleElement`
            Element $a$ defining the inner endomorphism $x \mapsto a x$.

        Returns
        =======

        :py:class:`~.InnerEndomorphism`

        """
        return InnerEndomorphism(self.domain, multiplier)

    def represent(self, element):
        r"""
        Represent an element of this endomorphism ring, as a single column
        vector.

        Explanation
        ===========

        Let $M$ be a module, and $E$ its ring of endomorphisms. Let $N$ be
        another module, and consider a homomorphism $\varphi: N \rightarrow E$.
        In the event that $\varphi$ is to be represented by a matrix $A$, each
        column of $A$ must represent an element of $E$. This is possible when
        the elements of $E$ are themselves representable as matrices, by
        stacking the columns of such a matrix into a single column.

        This method supports calculating such matrices $A$, by representing
        an element of this endomorphism ring first as a matrix, and then
        stacking that matrix's columns into a single column.

        Examples
        ========

        Note that in these examples we print matrix transposes, to make their
        columns easier to inspect.

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> from sympy.polys.numberfields.modules import ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> M = PowerBasis(T)
        >>> E = M.endomorphism_ring()

        Let $\zeta$ be a primitive 5th root of unity, a generator of our field,
        and consider the inner endomorphism $\tau$ on the ring of integers,
        induced by $\zeta$:

        >>> zeta = M(1)
        >>> tau = E.inner_endomorphism(zeta)
        >>> tau.matrix().transpose()  # doctest: +SKIP
        DomainMatrix(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
            (4, 4), ZZ)

        The matrix representation of $\tau$ is as expected. The first column
        shows that multiplying by $\zeta$ carries $1$ to $\zeta$, the second
        column that it carries $\zeta$ to $\zeta^2$, and so forth.

        The ``represent`` method of the endomorphism ring ``E`` stacks these
        into a single column:

        >>> E.represent(tau).transpose()  # doctest: +SKIP
        DomainMatrix(
            [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1]],
            (1, 16), ZZ)

        This is useful when we want to consider a homomorphism $\varphi$ having
        ``E`` as codomain:

        >>> phi = ModuleHomomorphism(M, E, lambda x: E.inner_endomorphism(x))

        and we want to compute the matrix of such a homomorphism:

        >>> phi.matrix().transpose()  # doctest: +SKIP
        DomainMatrix(
            [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1],
            [0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, -1, -1, -1, -1, 1, 0, 0, 0, 0, 1, 0, 0]],
            (4, 16), ZZ)

        Note that the stacked matrix of $\tau$ occurs as the second column in
        this example. This is because $\zeta$ is the second basis element of
        ``M``, and $\varphi(\zeta) = \tau$.

        Parameters
        ==========

        element : :py:class:`~.ModuleEndomorphism` belonging to this ring.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            Column vector equalling the vertical stacking of all the columns
            of the matrix that represents the given *element* as a mapping.

        """
        if isinstance(element, ModuleEndomorphism) and element.domain == self.domain:
            M = element.matrix()
            # Transform the matrix into a single column, which should reproduce
            # the original columns, one after another.
            m, n = M.shape
            if n == 0:
                return M
            return M[:, 0].vstack(*[M[:, j] for j in range(1, n)])
        raise NotImplementedError


def find_min_poly(alpha, domain, x=None, powers=None):
    r"""
    Find a polynomial of least degree (not necessarily irreducible) satisfied
    by an element of a finitely-generated ring with unity.

    Examples
    ========

    For the $n$th cyclotomic field, $n$ an odd prime, consider the quadratic
    equation whose roots are the two periods of length $(n-1)/2$. Article 356
    of Gauss tells us that we should get $x^2 + x - (n-1)/4$ or
    $x^2 + x + (n+1)/4$ according to whether $n$ is 1 or 3 mod 4, respectively.

    >>> from sympy import Poly, cyclotomic_poly, primitive_root, QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.modules import PowerBasis, find_min_poly
    >>> n = 13
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+1) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x - 3
    >>> n = 19
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+2) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x + 5

    Parameters
    ==========

    alpha : :py:class:`~.ModuleElement`
        The element whose min poly is to be found, and whose module has
        multiplication and starts with unity.

    domain : :py:class:`~.Domain`
        The desired domain of the polynomial.

    x : :py:class:`~.Symbol`, optional
        The desired variable for the polynomial.

    powers : list, optional
        If desired, pass an empty list. The powers of *alpha* (as
        :py:class:`~.ModuleElement` instances) from the zeroth up to the degree
        of the min poly will be recorded here, as we compute them.

    Returns
    =======

    :py:class:`~.Poly`, ``None``
        The minimal polynomial for alpha, or ``None`` if no polynomial could be
        found over the desired domain.

    Raises
    ======

    MissingUnityError
        If the module to which alpha belongs does not start with unity.
    ClosureFailure
        If the module to which alpha belongs is not closed under
        multiplication.

    """
    R = alpha.module
    if not R.starts_with_unity():
        raise MissingUnityError("alpha must belong to finitely generated ring with unity.")
    if powers is None:
        powers = []
    one = R(0)
    powers.append(one)
    powers_matrix = one.column(domain=domain)
    ak = alpha
    m = None
    for k in range(1, R.n + 1):
        powers.append(ak)
        ak_col = ak.column(domain=domain)
        try:
            X = powers_matrix._solve(ak_col)[0]
        except DMBadInputError:
            # This means alpha^k still isn't in the domain-span of the lower powers.
            powers_matrix = powers_matrix.hstack(ak_col)
            ak *= alpha
        else:
            # alpha^k is in the domain-span of the lower powers, so we have found a
            # minimal-degree poly for alpha.
            coeffs = [1] + [-c for c in reversed(X.to_list_flat())]
            x = x or Dummy('x')
            if domain.is_FF:
                m = Poly(coeffs, x, modulus=domain.mod)
            else:
                m = Poly(coeffs, x, domain=domain)
            break
    return m
<|endoftext|>"""Special exception classes for numberfields. """


class ClosureFailure(Exception):
    r"""
    Signals that a :py:class:`ModuleElement` which we tried to represent in a
    certain :py:class:`Module` cannot in fact be represented there.

    Examples
    ========

    >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
    >>> T = Poly(cyclotomic_poly(5))
    >>> A = PowerBasis(T)
    >>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))

    Because we are in a cyclotomic field, the power basis ``A`` is an integral
    basis, and the submodule ``B`` is just the ideal $(2)$. Therefore ``B`` can
    represent an element having all even coefficients over the power basis:

    >>> a1 = A(to_col([2, 4, 6, 8]))
    >>> print(B.represent(a1))
    DomainMatrix([[1], [2], [3], [4]], (4, 1), ZZ)

    but ``B`` cannot represent an element with an odd coefficient:

    >>> a2 = A(to_col([1, 2, 2, 2]))
    >>> B.represent(a2)
    Traceback (most recent call last):
    ...
    ClosureFailure: Element in QQ-span but not ZZ-span of this basis.

    """
    pass


class StructureError(Exception):
    r"""
    Represents cases in which an algebraic structure was expected to have a
    certain property, or be of a certain type, but was not.
    """
    pass


class MissingUnityError(StructureError):
    r"""Structure should contain a unity element but does not."""
    pass


__all__ = [
    'ClosureFailure', 'StructureError', 'MissingUnityError',
]
<|endoftext|>from __future__ import annotations

from sympy.calculus.singularities import is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits
from .expr_with_limits import AddWithLimits
from .gosper import gosper_sum
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Float, _illegal
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Not
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor
from sympy.polys.rationaltools import together
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.contains import Contains
from sympy.sets.sets import FiniteSet, Interval
from sympy.utilities.iterables import sift
import itertools


class Sum(AddWithLimits, ExprWithIntLimits):
    r"""
    Represents unevaluated summation.

    Explanation
    ===========

    ``Sum`` represents a finite or infinite series, with the first argument
    being the general form of terms in the series, and the second argument
    being ``(dummy_variable, start, end)``, with ``dummy_variable`` taking
    all integer values from ``start`` through ``end``. In accordance with
    long-standing mathematical convention, the end term is included in the
    summation.

    Finite sums
    ===========

    For finite sums (and sums with symbolic limits assumed to be finite) we
    follow the summation convention described by Karr [1], especially
    definition 3 of section 1.4. The sum:

    .. math::

        \sum_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \sum_{m \leq i < n} f(i) = f(m) + f(m+1) + \ldots + f(n-2) + f(n-1)

    with the upper limit value `f(n)` excluded. The sum over an empty set is
    zero if and only if `m = n`:

    .. math::

        \sum_{m \leq i < n} f(i) = 0  \quad \mathrm{for} \quad  m = n

    Finally, for all other sums over empty sets we assume the following
    definition:

    .. math::

        \sum_{m \leq i < n} f(i) = - \sum_{n \leq i < m} f(i)  \quad \mathrm{for} \quad  m > n

    It is important to note that Karr defines all sums with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the summation convention. Indeed we have:

    .. math::

        \sum_{m \leq i < n} f(i) = \sum_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import i, k, m, n, x
    >>> from sympy import Sum, factorial, oo, IndexedBase, Function
    >>> Sum(k, (k, 1, m))
    Sum(k, (k, 1, m))
    >>> Sum(k, (k, 1, m)).doit()
    m**2/2 + m/2
    >>> Sum(k**2, (k, 1, m))
    Sum(k**2, (k, 1, m))
    >>> Sum(k**2, (k, 1, m)).doit()
    m**3/3 + m**2/2 + m/6
    >>> Sum(x**k, (k, 0, oo))
    Sum(x**k, (k, 0, oo))
    >>> Sum(x**k, (k, 0, oo)).doit()
    Piecewise((1/(1 - x), Abs(x) < 1), (Sum(x**k, (k, 0, oo)), True))
    >>> Sum(x**k/factorial(k), (k, 0, oo)).doit()
    exp(x)

    Here are examples to do summation with symbolic indices.  You
    can use either Function of IndexedBase classes:

    >>> f = Function('f')
    >>> Sum(f(n), (n, 0, 3)).doit()
    f(0) + f(1) + f(2) + f(3)
    >>> Sum(f(n), (n, 0, oo)).doit()
    Sum(f(n), (n, 0, oo))
    >>> f = IndexedBase('f')
    >>> Sum(f[n]**2, (n, 0, 3)).doit()
    f[0]**2 + f[1]**2 + f[2]**2 + f[3]**2

    An example showing that the symbolic result of a summation is still
    valid for seemingly nonsensical values of the limits. Then the Karr
    convention allows us to give a perfectly valid interpretation to
    those sums by interchanging the limits according to the above rules:

    >>> S = Sum(i, (i, 1, n)).doit()
    >>> S
    n**2/2 + n/2
    >>> S.subs(n, -4)
    6
    >>> Sum(i, (i, 1, -4)).doit()
    6
    >>> Sum(-i, (i, -3, 0)).doit()
    6

    An explicit example of the Karr summation convention:

    >>> S1 = Sum(i**2, (i, m, m+n-1)).doit()
    >>> S1
    m**2*n + m*n**2 - m*n + n**3/3 - n**2/2 + n/6
    >>> S2 = Sum(i**2, (i, m+n, m-1)).doit()
    >>> S2
    -m**2*n - m*n**2 + m*n - n**3/3 + n**2/2 - n/6
    >>> S1 + S2
    0
    >>> S3 = Sum(i, (i, m, m-1)).doit()
    >>> S3
    0

    See Also
    ========

    summation
    Product, sympy.concrete.products.product

    References
    ==========

    .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
           Volume 28 Issue 2, April 1981, Pages 305-350
           https://dl.acm.org/doi/10.1145/322248.322255
    .. [2] https://en.wikipedia.org/wiki/Summation#Capital-sigma_notation
    .. [3] https://en.wikipedia.org/wiki/Empty_sum
    """

    __slots__ = ()

    limits: tuple[tuple[Symbol, Expr, Expr]]

    def __new__(cls, function, *symbols, **assumptions):
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        if not hasattr(obj, 'limits'):
            return obj
        if any(len(l) != 3 or None in l for l in obj.limits):
            raise ValueError('Sum requires values for lower and upper bounds.')

        return obj

    def _eval_is_zero(self):
        # a Sum is only zero if its function is zero or if all terms
        # cancel out. This only answers whether the summand is zero; if
        # not then None is returned since we don't analyze whether all
        # terms cancel out.
        if self.function.is_zero or self.has_empty_sequence:
            return True

    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True
        return self.function.is_extended_real

    def _eval_is_positive(self):
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_positive

    def _eval_is_negative(self):
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_negative

    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True

    def doit(self, **hints):
        if hints.get('deep', True):
            f = self.function.doit(**hints)
        else:
            f = self.function

        # first make sure any definite limits have summation
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            elif did is not None:
                did = did.xreplace(undo)
            else:
                did = self
            return did


        if self.function.is_Matrix:
            expanded = self.expand()
            if self != expanded:
                return expanded.doit()
            return _eval_matrix_sum(self)

        for n, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            if dif == -1:
                # Any summation over an empty set is zero
                return S.Zero
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = -f

            newf = eval_sum(f, (i, a, b))
            if newf is None:
                if f == self.function:
                    zeta_function = self.eval_zeta_function(f, (i, a, b))
                    if zeta_function is not None:
                        return zeta_function
                    return self
                else:
                    return self.func(f, *self.limits[n:])
            f = newf

        if hints.get('deep', True):
            # eval_sum could return partially unevaluated
            # result with Piecewise.  In this case we won't
            # doit() recursively.
            if not isinstance(f, Piecewise):
                return f.doit(**hints)

        return f

    def eval_zeta_function(self, f, limits):
        """
        Check whether the function matches with the zeta function.

        If it matches, then return a `Piecewise` expression because
        zeta function does not converge unless `s > 1` and `q > 0`
        """
        i, a, b = limits
        if a.is_comparable and b.is_comparable and a > b:
            return self.eval_zeta_function(f, (i, b + S.One, a - S.One))
        if b is not S.Infinity:
            return
        w, y, z = Wild('w', exclude=[i]), Wild('y', exclude=[i]), Wild('z', exclude=[i])
        if result := f.match((w * i + y) ** (-z)):
            coeff = 1 / result[w] ** result[z]
            s = result[z]
            q = result[y] / result[w] + a
            return Piecewise((coeff * zeta(s, q),
                              And(Not(Contains(-q, S.Naturals0)), re(s) > S.One)),
                             (self, True))

    def _eval_derivative(self, x):
        """
        Differentiate wrt x as long as x is not in the free symbols of any of
        the upper or lower limits.

        Explanation
        ===========

        Sum(a*b*x, (x, 1, a)) can be differentiated wrt x or b but not `a`
        since the value of the sum is discontinuous in `a`. In a case
        involving a limit variable, the unevaluated derivative is returned.
        """

        # diff already confirmed that x is in the free symbols of self, but we
        # don't want to differentiate wrt any free symbol in the upper or lower
        # limits
        # XXX remove this test for free_symbols when the default _eval_derivative is in
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero

        # get limits and the function
        f, limits = self.function, list(self.limits)

        limit = limits.pop(-1)

        if limits:  # f is the argument to a Sum
            f = self.func(f, *limits)

        _, a, b = limit
        if x in a.free_symbols or x in b.free_symbols:
            return None
        df = Derivative(f, x, evaluate=True)
        rv = self.func(df, limit)
        return rv

    def _eval_difference_delta(self, n, step):
        k, _, upper = self.args[-1]
        new_upper = upper.subs(n, n + step)

        if len(self.args) == 2:
            f = self.args[0]
        else:
            f = self.func(*self.args[:-1])

        return Sum(f, (k, upper + 1, new_upper)).doit()

    def _eval_simplify(self, **kwargs):

        function = self.function

        if kwargs.get('deep', True):
            function = function.simplify(**kwargs)

        # split the function into adds
        terms = Add.make_args(expand(function))
        s_t = [] # Sum Terms
        o_t = [] # Other Terms

        for term in terms:
            if term.has(Sum):
                # if there is an embedded sum here
                # it is of the form x * (Sum(whatever))
                # hence we make a Mul out of it, and simplify all interior sum terms
                subterms = Mul.make_args(expand(term))
                out_terms = []
                for subterm in subterms:
                    # go through each term
                    if isinstance(subterm, Sum):
                        # if it's a sum, simplify it
                        out_terms.append(subterm._eval_simplify(**kwargs))
                    else:
                        # otherwise, add it as is
                        out_terms.append(subterm)

                # turn it back into a Mul
                s_t.append(Mul(*out_terms))
            else:
                o_t.append(term)

        # next try to combine any interior sums for further simplification
        from sympy.simplify.simplify import factor_sum, sum_combine
        result = Add(sum_combine(s_t), *o_t)

        return factor_sum(result, limits=self.limits)

    def is_convergent(self):
        r"""
        Checks for the convergence of a Sum.

        Explanation
        ===========

        We divide the study of convergence of infinite sums and products in
        two parts.

        First Part:
        One part is the question whether all the terms are well defined, i.e.,
        they are finite in a sum and also non-zero in a product. Zero
        is the analogy of (minus) infinity in products as
        :math:`e^{-\infty} = 0`.

        Second Part:
        The second part is the question of convergence after infinities,
        and zeros in products, have been omitted assuming that their number
        is finite. This means that we only consider the tail of the sum or
        product, starting from some point after which all terms are well
        defined.

        For example, in a sum of the form:

        .. math::

            \sum_{1 \leq i < \infty} \frac{1}{n^2 + an + b}

        where a and b are numbers. The routine will return true, even if there
        are infinities in the term sequence (at most two). An analogous
        product would be:

        .. math::

            \prod_{1 \leq i < \infty} e^{\frac{1}{n^2 + an + b}}

        This is how convergence is interpreted. It is concerned with what
        happens at the limit. Finding the bad terms is another independent
        matter.

        Note: It is responsibility of user to see that the sum or product
        is well defined.

        There are various tests employed to check the convergence like
        divergence test, root test, integral test, alternating series test,
        comparison tests, Dirichlet tests. It returns true if Sum is convergent
        and false if divergent and NotImplementedError if it cannot be checked.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Convergence_tests

        Examples
        ========

        >>> from sympy import factorial, S, Sum, Symbol, oo
        >>> n = Symbol('n', integer=True)
        >>> Sum(n/(n - 1), (n, 4, 7)).is_convergent()
        True
        >>> Sum(n/(2*n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Sum(factorial(n)/5**n, (n, 1, oo)).is_convergent()
        False
        >>> Sum(1/n**(S(6)/5), (n, 1, oo)).is_convergent()
        True

        See Also
        ========

        Sum.is_absolutely_convergent
        sympy.concrete.products.Product.is_convergent
        """
        p, q, r = symbols('p q r', cls=Wild)

        sym = self.limits[0][0]
        lower_limit = self.limits[0][1]
        upper_limit = self.limits[0][2]
        sequence_term = self.function.simplify()

        if len(sequence_term.free_symbols) > 1:
            raise NotImplementedError("convergence checking for more than one symbol "
                                      "containing series is not handled")

        if lower_limit.is_finite and upper_limit.is_finite:
            return S.true

        # transform sym -> -sym and swap the upper_limit = S.Infinity
        # and lower_limit = - upper_limit
        if lower_limit is S.NegativeInfinity:
            if upper_limit is S.Infinity:
                return Sum(sequence_term, (sym, 0, S.Infinity)).is_convergent() and \
                        Sum(sequence_term, (sym, S.NegativeInfinity, 0)).is_convergent()
            from sympy.simplify.simplify import simplify
            sequence_term = simplify(sequence_term.xreplace({sym: -sym}))
            lower_limit = -upper_limit
            upper_limit = S.Infinity

        sym_ = Dummy(sym.name, integer=True, positive=True)
        sequence_term = sequence_term.xreplace({sym: sym_})
        sym = sym_

        interval = Interval(lower_limit, upper_limit)

        # Piecewise function handle
        if sequence_term.is_Piecewise:
            for func, cond in sequence_term.args:
                # see if it represents something going to oo
                if cond == True or cond.as_set().sup is S.Infinity:
                    s = Sum(func, (sym, lower_limit, upper_limit))
                    return s.is_convergent()
            return S.true

        ###  -------- Divergence test ----------- ###
        try:
            lim_val = limit_seq(sequence_term, sym)
            if lim_val is not None and lim_val.is_zero is False:
                return S.false
        except NotImplementedError:
            pass

        try:
            lim_val_abs = limit_seq(abs(sequence_term), sym)
            if lim_val_abs is not None and lim_val_abs.is_zero is False:
                return S.false
        except NotImplementedError:
            pass

        order = O(sequence_term, (sym, S.Infinity))

        ### --------- p-series test (1/n**p) ---------- ###
        p_series_test = order.expr.match(sym**p)
        if p_series_test is not None:
            if p_series_test[p] < -1:
                return S.true
            if p_series_test[p] >= -1:
                return S.false

        ### ------------- comparison test ------------- ###
        # 1/(n**p*log(n)**q*log(log(n))**r) comparison
        n_log_test = (order.expr.match(1/(sym**p*log(1/sym)**q*log(-log(1/sym))**r)) or
                      order.expr.match(1/(sym**p*(-log(1/sym))**q*log(-log(1/sym))**r)))
        if n_log_test is not None:
            if (n_log_test[p] > 1 or
                (n_log_test[p] == 1 and n_log_test[q] > 1) or
                (n_log_test[p] == n_log_test[q] == 1 and n_log_test[r] > 1)):
                    return S.true
            return S.false

        ### ------------- Limit comparison test -----------###
        # (1/n) comparison
        try:
            lim_comp = limit_seq(sym*sequence_term, sym)
            if lim_comp is not None and lim_comp.is_number and lim_comp > 0:
                return S.false
        except NotImplementedError:
            pass

        ### ----------- ratio test ---------------- ###
        next_sequence_term = sequence_term.xreplace({sym: sym + 1})
        from sympy.simplify.combsimp import combsimp
        from sympy.simplify.powsimp import powsimp
        ratio = combsimp(powsimp(next_sequence_term/sequence_term))
        try:
            lim_ratio = limit_seq(ratio, sym)
            if lim_ratio is not None and lim_ratio.is_number and lim_ratio is not S.NaN:
                if abs(lim_ratio) > 1:
                    return S.false
                if abs(lim_ratio) < 1:
                    return S.true
        except NotImplementedError:
            lim_ratio = None

        ### ---------- Raabe's test -------------- ###
        if lim_ratio == 1:  # ratio test inconclusive
            test_val = sym*(sequence_term/
                         sequence_term.subs(sym, sym + 1) - 1)
            test_val = test_val.gammasimp()
            try:
                lim_val = limit_seq(test_val, sym)
                if lim_val is not None and lim_val.is_number:
                    if lim_val > 1:
                        return S.true
                    if lim_val < 1:
                        return S.false
            except NotImplementedError:
                pass

        ### ----------- root test ---------------- ###
        # lim = Limit(abs(sequence_term)**(1/sym), sym, S.Infinity)
        try:
            lim_evaluated = limit_seq(abs(sequence_term)**(1/sym), sym)
            if lim_evaluated is not None and lim_evaluated.is_number:
                if lim_evaluated < 1:
                    return S.true
                if lim_evaluated > 1:
                    return S.false
        except NotImplementedError:
            pass

        ### ------------- alternating series test ----------- ###
        dict_val = sequence_term.match(S.NegativeOne**(sym + p)*q)
        if not dict_val[p].has(sym) and is_decreasing(dict_val[q], interval):
            return S.true

        ### ------------- integral test -------------- ###
        check_interval = None
        from sympy.solvers.solveset import solveset
        maxima = solveset(sequence_term.diff(sym), sym, interval)
        if not maxima:
            check_interval = interval
        elif isinstance(maxima, FiniteSet) and maxima.sup.is_number:
            check_interval = Interval(maxima.sup, interval.sup)
        if (check_interval is not None and
            (is_decreasing(sequence_term, check_interval) or
            is_decreasing(-sequence_term, check_interval))):
                integral_val = Integral(
                    sequence_term, (sym, lower_limit, upper_limit))
                try:
                    integral_val_evaluated = integral_val.doit()
                    if integral_val_evaluated.is_number:
                        return S(integral_val_evaluated.is_finite)
                except NotImplementedError:
                    pass

        ### ----- Dirichlet and bounded times convergent tests ----- ###
        # TODO
        #
        # Dirichlet_test
        # https://en.wikipedia.org/wiki/Dirichlet%27s_test
        #
        # Bounded times convergent test
        # It is based on comparison theorems for series.
        # In particular, if the general term of a series can
        # be written as a product of two terms a_n and b_n
        # and if a_n is bounded and if Sum(b_n) is absolutely
        # convergent, then the original series Sum(a_n * b_n)
        # is absolutely convergent and so convergent.
        #
        # The following code can grows like 2**n where n is the
        # number of args in order.expr
        # Possibly combined with the potentially slow checks
        # inside the loop, could make this test extremely slow
        # for larger summation expressions.

        if order.expr.is_Mul:
            args = order.expr.args
            argset = set(args)

            ### -------------- Dirichlet tests -------------- ###
            m = Dummy('m', integer=True)
            def _dirichlet_test(g_n):
                try:
                    ing_val = limit_seq(Sum(g_n, (sym, interval.inf, m)).doit(), m)
                    if ing_val is not None and ing_val.is_finite:
                        return S.true
                except NotImplementedError:
                    pass

            ### -------- bounded times convergent test ---------###
            def _bounded_convergent_test(g1_n, g2_n):
                try:
                    lim_val = limit_seq(g1_n, sym)
                    if lim_val is not None and (lim_val.is_finite or (
                        isinstance(lim_val, AccumulationBounds)
                        and (lim_val.max - lim_val.min).is_finite)):
                            if Sum(g2_n, (sym, lower_limit, upper_limit)).is_absolutely_convergent():
                                return S.true
                except NotImplementedError:
                    pass

            for n in range(1, len(argset)):
                for a_tuple in itertools.combinations(args, n):
                    b_set = argset - set(a_tuple)
                    a_n = Mul(*a_tuple)
                    b_n = Mul(*b_set)

                    if is_decreasing(a_n, interval):
                        dirich = _dirichlet_test(b_n)
                        if dirich is not None:
                            return dirich

                    bc_test = _bounded_convergent_test(a_n, b_n)
                    if bc_test is not None:
                        return bc_test

        _sym = self.limits[0][0]
        sequence_term = sequence_term.xreplace({sym: _sym})
        raise NotImplementedError("The algorithm to find the Sum convergence of %s "
                                  "is not yet implemented" % (sequence_term))

    def is_absolutely_convergent(self):
        """
        Checks for the absolute convergence of an infinite series.

        Same as checking convergence of absolute value of sequence_term of
        an infinite series.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absolute_convergence

        Examples
        ========

        >>> from sympy import Sum, Symbol, oo
        >>> n = Symbol('n', integer=True)
        >>> Sum((-1)**n, (n, 1, oo)).is_absolutely_convergent()
        False
        >>> Sum((-1)**n/n**2, (n, 1, oo)).is_absolutely_convergent()
        True

        See Also
        ========

        Sum.is_convergent
        """
        return Sum(abs(self.function), self.limits).is_convergent()

    def euler_maclaurin(self, m=0, n=0, eps=0, eval_integral=True):
        """
        Return an Euler-Maclaurin approximation of self, where m is the
        number of leading terms to sum directly and n is the number of
        terms in the tail.

        With m = n = 0, this is simply the corresponding integral
        plus a first-order endpoint correction.

        Returns (s, e) where s is the Euler-Maclaurin approximation
        and e is the estimated error (taken to be the magnitude of
        the first omitted term in the tail):

            >>> from sympy.abc import k, a, b
            >>> from sympy import Sum
            >>> Sum(1/k, (k, 2, 5)).doit().evalf()
            1.28333333333333
            >>> s, e = Sum(1/k, (k, 2, 5)).euler_maclaurin()
            >>> s
            -log(2) + 7/20 + log(5)
            >>> from sympy import sstr
            >>> print(sstr((s.evalf(), e.evalf()), full_prec=True))
            (1.26629073187415, 0.0175000000000000)

        The endpoints may be symbolic:

            >>> s, e = Sum(1/k, (k, a, b)).euler_maclaurin()
            >>> s
            -log(a) + log(b) + 1/(2*b) + 1/(2*a)
            >>> e
            Abs(1/(12*b**2) - 1/(12*a**2))

        If the function is a polynomial of degree at most 2n+1, the
        Euler-Maclaurin formula becomes exact (and e = 0 is returned):

            >>> Sum(k, (k, 2, b)).euler_maclaurin()
            (b**2/2 + b/2 - 1, 0)
            >>> Sum(k, (k, 2, b)).doit()
            b**2/2 + b/2 - 1

        With a nonzero eps specified, the summation is ended
        as soon as the remainder term is less than the epsilon.
        """
        m = int(m)
        n = int(n)
        f = self.function
        if len(self.limits) != 1:
            raise ValueError("More than 1 limit")
        i, a, b = self.limits[0]
        if (a > b) == True:
            if a - b == 1:
                return S.Zero, S.Zero
            a, b = b + 1, a - 1
            f = -f
        s = S.Zero
        if m:
            if b.is_Integer and a.is_Integer:
                m = min(m, b - a + 1)
            if not eps or f.is_polynomial(i):
                s = Add(*[f.subs(i, a + k) for k in range(m)])
            else:
                term = f.subs(i, a)
                if term:
                    test = abs(term.evalf(3)) < eps
                    if test == True:
                        return s, abs(term)
                    elif not (test == False):
                        # a symbolic Relational class, can't go further
                        return term, S.Zero
                s = term
                for k in range(1, m):
                    term = f.subs(i, a + k)
                    if abs(term.evalf(3)) < eps and term != 0:
                        return s, abs(term)
                    s += term
            if b - a + 1 == m:
                return s, S.Zero
            a += m
        x = Dummy('x')
        I = Integral(f.subs(i, x), (x, a, b))
        if eval_integral:
            I = I.doit()
        s += I

        def fpoint(expr):
            if b is S.Infinity:
                return expr.subs(i, a), 0
            return expr.subs(i, a), expr.subs(i, b)
        fa, fb = fpoint(f)
        iterm = (fa + fb)/2
        g = f.diff(i)
        for k in range(1, n + 2):
            ga, gb = fpoint(g)
            term = bernoulli(2*k)/factorial(2*k)*(gb - ga)
            if k > n:
                break
            if eps and term:
                term_evalf = term.evalf(3)
                if term_evalf is S.NaN:
                    return S.NaN, S.NaN
                if abs(term_evalf) < eps:
                    break
            s += term
            g = g.diff(i, 2, simplify=False)
        return s + iterm, abs(term)


    def reverse_order(self, *indices):
        """
        Reverse the order of a limit in a Sum.

        Explanation
        ===========

        ``reverse_order(self, *indices)`` reverses some limits in the expression
        ``self`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y, a, b, c, d

        >>> Sum(x, (x, 0, 3)).reverse_order(x)
        Sum(-x, (x, 4, -1))
        >>> Sum(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(x, y)
        Sum(x*y, (x, 6, 0), (y, 7, -1))
        >>> Sum(x, (x, a, b)).reverse_order(x)
        Sum(-x, (x, b + 1, a - 1))
        >>> Sum(x, (x, a, b)).reverse_order(0)
        Sum(-x, (x, b + 1, a - 1))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x**2, (x, a, b), (x, c, d))
        >>> S
        Sum(x**2, (x, a, b), (x, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x**2, (x, b + 1, a - 1), (x, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x**2, (x, b + 1, a - 1), (x, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index, reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255
        """
        l_indices = list(indices)

        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = self.index(indx)

        e = 1
        limits = []
        for i, limit in enumerate(self.limits):
            l = limit
            if i in l_indices:
                e = -e
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)

        return Sum(e * self.function, *limits)

    def _eval_rewrite_as_Product(self, *args, **kwargs):
        from sympy.concrete.products import Product
        if self.function.is_extended_real:
            return log(Product(exp(self.function), *self.limits))


def summation(f, *symbols, **kwargs):
    r"""
    Compute the summation of f with respect to symbols.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Integral.
    summation(f, (i, a, b)) computes the sum of f with respect to i from a to b,
    i.e.,

    ::

                                    b
                                  ____
                                  \   `
        summation(f, (i, a, b)) =  )    f
                                  /___,
                                  i = a

    If it cannot compute the sum, it returns an unevaluated Sum object.
    Repeated sums can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import summation, oo, symbols, log
    >>> i, n, m = symbols('i n m', integer=True)

    >>> summation(2*i - 1, (i, 1, n))
    n**2
    >>> summation(1/2**i, (i, 0, oo))
    2
    >>> summation(1/log(n)**n, (n, 2, oo))
    Sum(log(n)**(-n), (n, 2, oo))
    >>> summation(i, (i, 0, n), (n, 0, m))
    m**3/6 + m**2/2 + m/3

    >>> from sympy.abc import x
    >>> from sympy import factorial
    >>> summation(x**n/factorial(n), (n, 0, oo))
    exp(x)

    See Also
    ========

    Sum
    Product, sympy.concrete.products.product

    """
    return Sum(f, *symbols, **kwargs).doit(deep=False)


def telescopic_direct(L, R, n, limits):
    """
    Returns the direct summation of the terms of a telescopic sum

    Explanation
    ===========

    L is the term with lower index
    R is the term with higher index
    n difference between the indexes of L and R

    Examples
    ========

    >>> from sympy.concrete.summations import telescopic_direct
    >>> from sympy.abc import k, a, b
    >>> telescopic_direct(1/k, -1/(k+2), 2, (k, a, b))
    -1/(b + 2) - 1/(b + 1) + 1/(a + 1) + 1/a

    """
    (i, a, b) = limits
    return Add(*[L.subs(i, a + m) + R.subs(i, b - m) for m in range(n)])


def telescopic(L, R, limits):
    '''
    Tries to perform the summation using the telescopic property.

    Return None if not possible.
    '''
    (i, a, b) = limits
    if L.is_Add or R.is_Add:
        return None

    # We want to solve(L.subs(i, i + m) + R, m)
    # First we try a simple match since this does things that
    # solve doesn't do, e.g. solve(cos(k+m)-cos(k), m) gives
    # a more complicated solution than m == 0.

    k = Wild("k")
    sol = (-R).match(L.subs(i, i + k))
    s = None
    if sol and k in sol:
        s = sol[k]
        if not (s.is_Integer and L.subs(i, i + s) + R == 0):
            # invalid match or match didn't work
            s = None

    # But there are things that match doesn't do that solve
    # can do, e.g. determine that 1/(x + m) = 1/(1 - x) when m = 1

    if s is None:
        m = Dummy('m')
        try:
            from sympy.solvers.solvers import solve
            sol = solve(L.subs(i, i + m) + R, m) or []
        except NotImplementedError:
            return None
        sol = [si for si in sol if si.is_Integer and
               (L.subs(i, i + si) + R).expand().is_zero]
        if len(sol) != 1:
            return None
        s = sol[0]

    if s < 0:
        return telescopic_direct(R, L, abs(s), (i, a, b))
    elif s > 0:
        return telescopic_direct(L, R, s, (i, a, b))


def eval_sum(f, limits):
    (i, a, b) = limits
    if f.is_zero:
        return S.Zero
    if i not in f.free_symbols:
        return f*(b - a + 1)
    if a == b:
        return f.subs(i, a)
    if a.is_comparable and b.is_comparable and a > b:
        return eval_sum(f, (i, b + S.One, a - S.One))
    if isinstance(f, Piecewise):
        if not any(i in arg.args[1].free_symbols for arg in f.args):
            # Piecewise conditions do not depend on the dummy summation variable,
            # therefore we can fold:     Sum(Piecewise((e, c), ...), limits)
            #                        --> Piecewise((Sum(e, limits), c), ...)
            newargs = []
            for arg in f.args:
                newexpr = eval_sum(arg.expr, limits)
                if newexpr is None:
                    return None
                newargs.append((newexpr, arg.cond))
            return f.func(*newargs)

    if f.has(KroneckerDelta):
        from .delta import deltasummation, _has_simple_delta
        f = f.replace(
            lambda x: isinstance(x, Sum),
            lambda x: x.factor()
        )
        if _has_simple_delta(f, limits[0]):
            return deltasummation(f, limits)

    dif = b - a
    definite = dif.is_Integer
    # Doing it directly may be faster if there are very few terms.
    if definite and (dif < 100):
        return eval_sum_direct(f, (i, a, b))
    if isinstance(f, Piecewise):
        return None
    # Try to do it symbolically. Even when the number of terms is
    # known, this can save time when b-a is big.
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    # Do it directly
    if definite:
        return eval_sum_direct(f, (i, a, b))


def eval_sum_direct(expr, limits):
    """
    Evaluate expression directly, but perform some simple checks first
    to possibly result in a smaller expression and faster execution.
    """
    (i, a, b) = limits

    dif = b - a
    # Linearity
    if expr.is_Mul:
        # Try factor out everything not including i
        without_i, with_i = expr.as_independent(i)
        if without_i != 1:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i*s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            L, R = expr.as_two_terms()

            if not L.has(i):
                sR = eval_sum_direct(R, (i, a, b))
                if sR:
                    return L*sR

            if not R.has(i):
                sL = eval_sum_direct(L, (i, a, b))
                if sL:
                    return sL*R

    # do this whether its an Add or Mul
    # e.g. apart(1/(25*i**2 + 45*i + 14)) and
    # apart(1/((5*i + 2)*(5*i + 7))) ->
    # -1/(5*(5*i + 7)) + 1/(5*(5*i + 2))
    try:
        expr = apart(expr, i)  # see if it becomes an Add
    except PolynomialError:
        pass

    if expr.is_Add:
        # Try factor out everything not including i
        without_i, with_i = expr.as_independent(i)
        if without_i != 0:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i*(dif + 1) + s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            L, R = expr.as_two_terms()
            lsum = eval_sum_direct(L, (i, a, b))
            rsum = eval_sum_direct(R, (i, a, b))

            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r

    return Add(*[expr.subs(i, a + j) for j in range(dif + 1)])


def eval_sum_symbolic(f, limits):
    f_orig = f
    (i, a, b) = limits
    if not f.has(i):
        return f*(b - a + 1)

    # Linearity
    if f.is_Mul:
        # Try factor out everything not including i
        without_i, with_i = f.as_independent(i)
        if without_i != 1:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i*s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            L, R = f.as_two_terms()

            if not L.has(i):
                sR = eval_sum_symbolic(R, (i, a, b))
                if sR:
                    return L*sR

            if not R.has(i):
                sL = eval_sum_symbolic(L, (i, a, b))
                if sL:
                    return sL*R

    # do this whether its an Add or Mul
    # e.g. apart(1/(25*i**2 + 45*i + 14)) and
    # apart(1/((5*i + 2)*(5*i + 7))) ->
    # -1/(5*(5*i + 7)) + 1/(5*(5*i + 2))
    try:
        f = apart(f, i)
    except PolynomialError:
        pass

    if f.is_Add:
        L, R = f.as_two_terms()
        lrsum = telescopic(L, R, (i, a, b))

        if lrsum:
            return lrsum

        # Try factor out everything not including i
        without_i, with_i = f.as_independent(i)
        if without_i != 0:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i*(b - a + 1) + s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            lsum = eval_sum_symbolic(L, (i, a, b))
            rsum = eval_sum_symbolic(R, (i, a, b))

            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r


    # Polynomial terms with Faulhaber's formula
    n = Wild('n')
    result = f.match(i**n)

    if result is not None:
        n = result[n]

        if n.is_Integer:
            if n >= 0:
                if (b is S.Infinity and a is not S.NegativeInfinity) or \
                   (a is S.NegativeInfinity and b is not S.Infinity):
                    return S.Infinity
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a))/(n + 1)).expand()
            elif a.is_Integer and a >= 1:
                if n == -1:
                    return harmonic(b) - harmonic(a - 1)
                else:
                    return harmonic(b, abs(n)) - harmonic(a - 1, abs(n))

    if not (a.has(S.Infinity, S.NegativeInfinity) or
            b.has(S.Infinity, S.NegativeInfinity)):
        # Geometric terms
        c1 = Wild('c1', exclude=[i])
        c2 = Wild('c2', exclude=[i])
        c3 = Wild('c3', exclude=[i])
        wexp = Wild('wexp')

        # Here we first attempt powsimp on f for easier matching with the
        # exponential pattern, and attempt expansion on the exponent for easier
        # matching with the linear pattern.
        e = f.powsimp().match(c1 ** wexp)
        if e is not None:
            e_exp = e.pop(wexp).expand().match(c2*i + c3)
            if e_exp is not None:
                e.update(e_exp)

                p = (c1**c3).subs(e)
                q = (c1**c2).subs(e)
                r = p*(q**a - q**(b + 1))/(1 - q)
                l = p*(b - a + 1)
                return Piecewise((l, Eq(q, S.One)), (r, True))

        r = gosper_sum(f, (i, a, b))

        if isinstance(r, (Mul,Add)):
            from sympy.simplify.radsimp import denom
            from sympy.solvers.solvers import solve
            non_limit = r.free_symbols - Tuple(*limits[1:]).free_symbols
            den = denom(together(r))
            den_sym = non_limit & den.free_symbols
            args = []
            for v in ordered(den_sym):
                try:
                    s = solve(den, v)
                    m = Eq(v, s[0]) if s else S.false
                    if m != False:
                        args.append((Sum(f_orig.subs(*m.args), limits).doit(), m))
                    break
                except NotImplementedError:
                    continue

            args.append((r, True))
            return Piecewise(*args)

        if r not in (None, S.NaN):
            return r

    h = eval_sum_hyper(f_orig, (i, a, b))
    if h is not None:
        return h

    r = eval_sum_residue(f_orig, (i, a, b))
    if r is not None:
        return r

    factored = f_orig.factor()
    if factored != f_orig:
        return eval_sum_symbolic(factored, (i, a, b))


def _eval_sum_hyper(f, i, a):
    """ Returns (res, cond). Sums from a to oo. """
    if a != 0:
        return _eval_sum_hyper(f.subs(i, i + a), i, 0)

    if f.subs(i, 0) == 0:
        from sympy.simplify.simplify import simplify
        if simplify(f.subs(i, Dummy('i', integer=True, positive=True))) == 0:
            return S.Zero, True
        return _eval_sum_hyper(f.subs(i, i + 1), i, 0)

    from sympy.simplify.simplify import hypersimp
    hs = hypersimp(f, i)
    if hs is None:
        return None

    if isinstance(hs, Float):
        from sympy.simplify.simplify import nsimplify
        hs = nsimplify(hs)

    from sympy.simplify.combsimp import combsimp
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.simplify.radsimp import fraction
    numer, denom = fraction(factor(hs))
    top, topl = numer.as_coeff_mul(i)
    bot, botl = denom.as_coeff_mul(i)
    ab = [top, bot]
    factors = [topl, botl]
    params = [[], []]
    for k in range(2):
        for fac in factors[k]:
            mul = 1
            if fac.is_Pow:
                mul = fac.exp
                fac = fac.base
                if not mul.is_Integer:
                    return None
            p = Poly(fac, i)
            if p.degree() != 1:
                return None
            m, n = p.all_coeffs()
            ab[k] *= m**mul
            params[k] += [n/m]*mul

    # Add "1" to numerator parameters, to account for implicit n! in
    # hypergeometric series.
    ap = params[0] + [1]
    bq = params[1]
    x = ab[0]/ab[1]
    h = hyper(ap, bq, x)
    f = combsimp(f)
    return f.subs(i, 0)*hyperexpand(h), h.convergence_statement


def eval_sum_hyper(f, i_a_b):
    i, a, b = i_a_b

    if f.is_hypergeometric(i) is False:
        return

    if (b - a).is_Integer:
        # We are never going to do better than doing the sum in the obvious way
        return None

    old_sum = Sum(f, (i, a, b))

    if b != S.Infinity:
        if a is S.NegativeInfinity:
            res = _eval_sum_hyper(f.subs(i, -i), i, -b)
            if res is not None:
                return Piecewise(res, (old_sum, True))
        else:
            n_illegal = lambda x: sum(x.count(_) for _ in _illegal)
            had = n_illegal(f)
            # check that no extra illegals are introduced
            res1 = _eval_sum_hyper(f, i, a)
            if res1 is None or n_illegal(res1) > had:
                return
            res2 = _eval_sum_hyper(f, i, b + 1)
            if res2 is None or n_illegal(res2) > had:
                return
            (res1, cond1), (res2, cond2) = res1, res2
            cond = And(cond1, cond2)
            if cond == False:
                return None
            return Piecewise((res1 - res2, cond), (old_sum, True))

    if a is S.NegativeInfinity:
        res1 = _eval_sum_hyper(f.subs(i, -i), i, 1)
        res2 = _eval_sum_hyper(f, i, 0)
        if res1 is None or res2 is None:
            return None
        res1, cond1 = res1
        res2, cond2 = res2
        cond = And(cond1, cond2)
        if cond == False or cond.as_set() == S.EmptySet:
            return None
        return Piecewise((res1 + res2, cond), (old_sum, True))

    # Now b == oo, a != -oo
    res = _eval_sum_hyper(f, i, a)
    if res is not None:
        r, c = res
        if c == False:
            if r.is_number:
                f = f.subs(i, Dummy('i', integer=True, positive=True) + a)
                if f.is_positive or f.is_zero:
                    return S.Infinity
                elif f.is_negative:
                    return S.NegativeInfinity
            return None
        return Piecewise(res, (old_sum, True))


def eval_sum_residue(f, i_a_b):
    r"""Compute the infinite summation with residues

    Notes
    =====

    If $f(n), g(n)$ are polynomials with $\deg(g(n)) - \deg(f(n)) \ge 2$,
    some infinite summations can be computed by the following residue
    evaluations.

    .. math::
        \sum_{n=-\infty, g(n) \ne 0}^{\infty} \frac{f(n)}{g(n)} =
        -\pi \sum_{\alpha|g(\alpha)=0}
        \text{Res}(\cot(\pi x) \frac{f(x)}{g(x)}, \alpha)

    .. math::
        \sum_{n=-\infty, g(n) \ne 0}^{\infty} (-1)^n \frac{f(n)}{g(n)} =
        -\pi \sum_{\alpha|g(\alpha)=0}
        \text{Res}(\csc(\pi x) \frac{f(x)}{g(x)}, \alpha)

    Examples
    ========

    >>> from sympy import Sum, oo, Symbol
    >>> x = Symbol('x')

    Doubly infinite series of rational functions.

    >>> Sum(1 / (x**2 + 1), (x, -oo, oo)).doit()
    pi/tanh(pi)

    Doubly infinite alternating series of rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, -oo, oo)).doit()
    pi/sinh(pi)

    Infinite series of even rational functions.

    >>> Sum(1 / (x**2 + 1), (x, 0, oo)).doit()
    1/2 + pi/(2*tanh(pi))

    Infinite series of alternating even rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, 0, oo)).doit()
    pi/(2*sinh(pi)) + 1/2

    This also have heuristics to transform arbitrarily shifted summand or
    arbitrarily shifted summation range to the canonical problem the
    formula can handle.

    >>> Sum(1 / (x**2 + 2*x + 2), (x, -1, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 4*x + 5), (x, -2, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 1, oo)).doit()
    -1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 2, oo)).doit()
    -1 + pi/(2*tanh(pi))

    References
    ==========

    .. [#] http://www.supermath.info/InfiniteSeriesandtheResidueTheorem.pdf

    .. [#] Asmar N.H., Grafakos L. (2018) Residue Theory.
           In: Complex Analysis with Applications.
           Undergraduate Texts in Mathematics. Springer, Cham.
           https://doi.org/10.1007/978-3-319-94063-2_5
    """
    i, a, b = i_a_b

    # If lower limit > upper limit: Karr Summation Convention
    if a.is_comparable and b.is_comparable and a > b:
        return eval_sum_residue(f, (i, b + S.One, a - S.One))

    def is_even_function(numer, denom):
        """Test if the rational function is an even function"""
        numer_even = all(i % 2 == 0 for (i,) in numer.monoms())
        denom_even = all(i % 2 == 0 for (i,) in denom.monoms())
        numer_odd = all(i % 2 == 1 for (i,) in numer.monoms())
        denom_odd = all(i % 2 == 1 for (i,) in denom.monoms())
        return (numer_even and denom_even) or (numer_odd and denom_odd)

    def match_rational(f, i):
        numer, denom = f.as_numer_denom()
        try:
            (numer, denom), opt = parallel_poly_from_expr((numer, denom), i)
        except (PolificationFailed, PolynomialError):
            return None
        return numer, denom

    def get_poles(denom):
        roots = denom.sqf_part().all_roots()
        roots = sift(roots, lambda x: x.is_integer)
        if None in roots:
            return None
        int_roots, nonint_roots = roots[True], roots[False]
        return int_roots, nonint_roots

    def get_shift(denom):
        n = denom.degree(i)
        a = denom.coeff_monomial(i**n)
        b = denom.coeff_monomial(i**(n-1))
        shift = - b / a / n
        return shift

    #Need a dummy symbol with no assumptions set for get_residue_factor
    z = Dummy('z')

    def get_residue_factor(numer, denom, alternating):
        residue_factor = (numer.as_expr() / denom.as_expr()).subs(i, z)
        if not alternating:
            residue_factor *= cot(S.Pi * z)
        else:
            residue_factor *= csc(S.Pi * z)
        return residue_factor

    # We don't know how to deal with symbolic constants in summand
    if f.free_symbols - {i}:
        return None

    if not (a.is_Integer or a in (S.Infinity, S.NegativeInfinity)):
        return None
    if not (b.is_Integer or b in (S.Infinity, S.NegativeInfinity)):
        return None

    # Quick exit heuristic for the sums which doesn't have infinite range
    if a != S.NegativeInfinity and b != S.Infinity:
        return None

    match = match_rational(f, i)
    if match:
        alternating = False
        numer, denom = match
    else:
        match = match_rational(f / S.NegativeOne**i, i)
        if match:
            alternating = True
            numer, denom = match
        else:
            return None

    if denom.degree(i) - numer.degree(i) < 2:
        return None

    if (a, b) == (S.NegativeInfinity, S.Infinity):
        poles = get_poles(denom)
        if poles is None:
            return None
        int_roots, nonint_roots = poles

        if int_roots:
            return None

        residue_factor = get_residue_factor(numer, denom, alternating)
        residues = [residue(residue_factor, z, root) for root in nonint_roots]
        return -S.Pi * sum(residues)

    if not (a.is_finite and b is S.Infinity):
        return None

    if not is_even_function(numer, denom):
        # Try shifting summation and check if the summand can be made
        # and even function from the origin.
        # Sum(f(n), (n, a, b)) => Sum(f(n + s), (n, a - s, b - s))
        shift = get_shift(denom)

        if not shift.is_Integer:
            return None
        if shift == 0:
            return None

        numer = numer.shift(shift)
        denom = denom.shift(shift)

        if not is_even_function(numer, denom):
            return None

        if alternating:
            f = S.NegativeOne**i * (S.NegativeOne**shift * numer.as_expr() / denom.as_expr())
        else:
            f = numer.as_expr() / denom.as_expr()
        return eval_sum_residue(f, (i, a-shift, b-shift))

    poles = get_poles(denom)
    if poles is None:
        return None
    int_roots, nonint_roots = poles

    if int_roots:
        int_roots = [int(root) for root in int_roots]
        int_roots_max = max(int_roots)
        int_roots_min = min(int_roots)
        # Integer valued poles must be next to each other
        # and also symmetric from origin (Because the function is even)
        if not len(int_roots) == int_roots_max - int_roots_min + 1:
            return None

        # Check whether the summation indices contain poles
        if a <= max(int_roots):
            return None

    residue_factor = get_residue_factor(numer, denom, alternating)
    residues = [residue(residue_factor, z, root) for root in int_roots + nonint_roots]
    full_sum = -S.Pi * sum(residues)

    if not int_roots:
        # Compute Sum(f, (i, 0, oo)) by adding a extraneous evaluation
        # at the origin.
        half_sum = (full_sum + f.xreplace({i: 0})) / 2

        # Add and subtract extraneous evaluations
        extraneous_neg = [f.xreplace({i: i0}) for i0 in range(int(a), 0)]
        extraneous_pos = [f.xreplace({i: i0}) for i0 in range(0, int(a))]
        result = half_sum + sum(extraneous_neg) - sum(extraneous_pos)

        return result

    # Compute Sum(f, (i, min(poles) + 1, oo))
    half_sum = full_sum / 2

    # Subtract extraneous evaluations
    extraneous = [f.xreplace({i: i0}) for i0 in range(max(int_roots) + 1, int(a))]
    result = half_sum - sum(extraneous)

    return result


def _eval_matrix_sum(expression):
    f = expression.function
    for limit in expression.limits:
        i, a, b = limit
        dif = b - a
        if dif.is_Integer:
            if (dif < 0) == True:
                a, b = b + 1, a - 1
                f = -f

            newf = eval_sum_direct(f, (i, a, b))
            if newf is not None:
                return newf.doit()


def _dummy_with_inherited_properties_concrete(limits):
    """
    Return a Dummy symbol that inherits as many assumptions as possible
    from the provided symbol and limits.

    If the symbol already has all True assumption shared by the limits
    then return None.
    """
    x, a, b = limits
    l = [a, b]

    assumptions_to_consider = ['extended_nonnegative', 'nonnegative',
                               'extended_nonpositive', 'nonpositive',
                               'extended_positive', 'positive',
                               'extended_negative', 'negative',
                               'integer', 'rational', 'finite',
                               'zero', 'real', 'extended_real']

    assumptions_to_keep = {}
    assumptions_to_add = {}
    for assum in assumptions_to_consider:
        assum_true = x._assumptions.get(assum, None)
        if assum_true:
            assumptions_to_keep[assum] = True
        elif all(getattr(i, 'is_' + assum) for i in l):
            assumptions_to_add[assum] = True
    if assumptions_to_add:
        assumptions_to_keep.update(assumptions_to_add)
        return Dummy('d', **assumptions_to_keep)
<|endoftext|>"""Various algorithms for helping identifying numbers and sequences."""


from sympy.concrete.products import (Product, product)
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public


@public
def find_simple_recurrence_vector(l):
    """
    This function is used internally by other functions from the
    sympy.concrete.guess module. While most users may want to rather use the
    function find_simple_recurrence when looking for recurrence relations
    among rational numbers, the current function may still be useful when
    some post-processing has to be done.

    Explanation
    ===========

    The function returns a vector of length n when a recurrence relation of
    order n is detected in the sequence of rational numbers v.

    If the returned vector has a length 1, then the returned value is always
    the list [0], which means that no relation has been found.

    While the functions is intended to be used with rational numbers, it should
    work for other kinds of real numbers except for some cases involving
    quadratic numbers; for that reason it should be used with some caution when
    the argument is not a list of rational numbers.

    Examples
    ========

    >>> from sympy.concrete.guess import find_simple_recurrence_vector
    >>> from sympy import fibonacci
    >>> find_simple_recurrence_vector([fibonacci(k) for k in range(12)])
    [1, -1, -1]

    See Also
    ========

    See the function sympy.concrete.guess.find_simple_recurrence which is more
    user-friendly.

    """
    q1 = [0]
    q2 = [1]
    b, z = 0, len(l) >> 1
    while len(q2) <= z:
        while l[b]==0:
            b += 1
            if b == len(l):
                c = 1
                for x in q2:
                    c = lcm(c, denom(x))
                if q2[0]*c < 0: c = -c
                for k in range(len(q2)):
                    q2[k] = int(q2[k]*c)
                return q2
        a = S.One/l[b]
        m = [a]
        for k in range(b+1, len(l)):
            m.append(-sum(l[j+1]*m[b-j-1] for j in range(b, k))*a)
        l, m = m, [0] * max(len(q2), b+len(q1))
        for k, q in enumerate(q2):
            m[k] = a*q
        for k, q in enumerate(q1):
            m[k+b] += q
        while m[-1]==0: m.pop() # because trailing zeros can occur
        q1, q2, b = q2, m, 1
    return [0]

@public
def find_simple_recurrence(v, A=Function('a'), N=Symbol('n')):
    """
    Detects and returns a recurrence relation from a sequence of several integer
    (or rational) terms. The name of the function in the returned expression is
    'a' by default; the main variable is 'n' by default. The smallest index in
    the returned expression is always n (and never n-1, n-2, etc.).

    Examples
    ========

    >>> from sympy.concrete.guess import find_simple_recurrence
    >>> from sympy import fibonacci
    >>> find_simple_recurrence([fibonacci(k) for k in range(12)])
    -a(n) - a(n + 1) + a(n + 2)

    >>> from sympy import Function, Symbol
    >>> a = [1, 1, 1]
    >>> for k in range(15): a.append(5*a[-1]-3*a[-2]+8*a[-3])
    >>> find_simple_recurrence(a, A=Function('f'), N=Symbol('i'))
    -8*f(i) + 3*f(i + 1) - 5*f(i + 2) + f(i + 3)

    """
    p = find_simple_recurrence_vector(v)
    n = len(p)
    if n <= 1: return S.Zero

    return Add(*[A(N+n-1-k)*p[k] for k in range(n)])


@public
def rationalize(x, maxcoeff=10000):
    """
    Helps identifying a rational number from a float (or mpmath.mpf) value by
    using a continued fraction. The algorithm stops as soon as a large partial
    quotient is detected (greater than 10000 by default).

    Examples
    ========

    >>> from sympy.concrete.guess import rationalize
    >>> from mpmath import cos, pi
    >>> rationalize(cos(pi/3))
    1/2

    >>> from mpmath import mpf
    >>> rationalize(mpf("0.333333333333333"))
    1/3

    While the function is rather intended to help 'identifying' rational
    values, it may be used in some cases for approximating real numbers.
    (Though other functions may be more relevant in that case.)

    >>> rationalize(pi, maxcoeff = 250)
    355/113

    See Also
    ========

    Several other methods can approximate a real number as a rational, like:

      * fractions.Fraction.from_decimal
      * fractions.Fraction.from_float
      * mpmath.identify
      * mpmath.pslq by using the following syntax: mpmath.pslq([x, 1])
      * mpmath.findpoly by using the following syntax: mpmath.findpoly(x, 1)
      * sympy.simplify.nsimplify (which is a more general function)

    The main difference between the current function and all these variants is
    that control focuses on magnitude of partial quotients here rather than on
    global precision of the approximation. If the real is "known to be" a
    rational number, the current function should be able to detect it correctly
    with the default settings even when denominator is great (unless its
    expansion contains unusually big partial quotients) which may occur
    when studying sequences of increasing numbers. If the user cares more
    on getting simple fractions, other methods may be more convenient.

    """
    p0, p1 = 0, 1
    q0, q1 = 1, 0
    a = floor(x)
    while a < maxcoeff or q1==0:
        p = a*p1 + p0
        q = a*q1 + q0
        p0, p1 = p1, p
        q0, q1 = q1, q
        if x==a: break
        x = 1/(x-a)
        a = floor(x)
    return sympify(p) / q


@public
def guess_generating_function_rational(v, X=Symbol('x')):
    """
    Tries to "guess" a rational generating function for a sequence of rational
    numbers v.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function_rational
    >>> from sympy import fibonacci
    >>> l = [fibonacci(k) for k in range(5,15)]
    >>> guess_generating_function_rational(l)
    (3*x + 5)/(-x**2 - x + 1)

    See Also
    ========

    sympy.series.approximants
    mpmath.pade

    """
    #   a) compute the denominator as q
    q = find_simple_recurrence_vector(v)
    n = len(q)
    if n <= 1: return None
    #   b) compute the numerator as p
    p = [sum(v[i-k]*q[k] for k in range(min(i+1, n)))
            for i in range(len(v)>>1)]
    return (sum(p[k]*X**k for k in range(len(p)))
            / sum(q[k]*X**k for k in range(n)))


@public
def guess_generating_function(v, X=Symbol('x'), types=['all'], maxsqrtn=2):
    """
    Tries to "guess" a generating function for a sequence of rational numbers v.
    Only a few patterns are implemented yet.

    Explanation
    ===========

    The function returns a dictionary where keys are the name of a given type of
    generating function. Six types are currently implemented:

         type  |  formal definition
        -------+----------------------------------------------------------------
        ogf    | f(x) = Sum(            a_k * x^k       ,  k: 0..infinity )
        egf    | f(x) = Sum(            a_k * x^k / k!  ,  k: 0..infinity )
        lgf    | f(x) = Sum( (-1)^(k+1) a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        hlgf   | f(x) = Sum(            a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        lgdogf | f(x) = derivate( log(Sum( a_k * x^k, k: 0..infinity )), x)
        lgdegf | f(x) = derivate( log(Sum( a_k * x^k / k!, k: 0..infinity )), x)

    In order to spare time, the user can select only some types of generating
    functions (default being ['all']). While forgetting to use a list in the
    case of a single type may seem to work most of the time as in: types='ogf'
    this (convenient) syntax may lead to unexpected extra results in some cases.

    Discarding a type when calling the function does not mean that the type will
    not be present in the returned dictionary; it only means that no extra
    computation will be performed for that type, but the function may still add
    it in the result when it can be easily converted from another type.

    Two generating functions (lgdogf and lgdegf) are not even computed if the
    initial term of the sequence is 0; it may be useful in that case to try
    again after having removed the leading zeros.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function as ggf
    >>> ggf([k+1 for k in range(12)], types=['ogf', 'lgf', 'hlgf'])
    {'hlgf': 1/(1 - x), 'lgf': 1/(x + 1), 'ogf': 1/(x**2 - 2*x + 1)}

    >>> from sympy import sympify
    >>> l = sympify("[3/2, 11/2, 0, -121/2, -363/2, 121]")
    >>> ggf(l)
    {'ogf': (x + 3/2)/(11*x**2 - 3*x + 1)}

    >>> from sympy import fibonacci
    >>> ggf([fibonacci(k) for k in range(5, 15)], types=['ogf'])
    {'ogf': (3*x + 5)/(-x**2 - x + 1)}

    >>> from sympy import factorial
    >>> ggf([factorial(k) for k in range(12)], types=['ogf', 'egf', 'lgf'])
    {'egf': 1/(1 - x)}

    >>> ggf([k+1 for k in range(12)], types=['egf'])
    {'egf': (x + 1)*exp(x), 'lgdegf': (x + 2)/(x + 1)}

    N-th root of a rational function can also be detected (below is an example
    coming from the sequence A108626 from https://oeis.org).
    The greatest n-th root to be tested is specified as maxsqrtn (default 2).

    >>> ggf([1, 2, 5, 14, 41, 124, 383, 1200, 3799, 12122, 38919])['ogf']
    sqrt(1/(x**4 + 2*x**2 - 4*x + 1))

    References
    ==========

    .. [1] "Concrete Mathematics", R.L. Graham, D.E. Knuth, O. Patashnik
    .. [2] https://oeis.org/wiki/Generating_functions

    """
    # List of all types of all g.f. known by the algorithm
    if 'all' in types:
        types = ('ogf', 'egf', 'lgf', 'hlgf', 'lgdogf', 'lgdegf')

    result = {}

    # Ordinary Generating Function (ogf)
    if 'ogf' in types:
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(v) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*v[i] for i in range(n+1)) for n in range(len(v))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['ogf'] = g**Rational(1, d+1)
                break

    # Exponential Generating Function (egf)
    if 'egf' in types:
        # Transform sequence (division by factorial)
        w, f = [], S.One
        for i, k in enumerate(v):
            f *= i if i else 1
            w.append(k/f)
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['egf'] = g**Rational(1, d+1)
                break

    # Logarithmic Generating Function (lgf)
    if 'lgf' in types:
        # Transform sequence (multiplication by (-1)^(n+1) / n)
        w, f = [], S.NegativeOne
        for i, k in enumerate(v):
            f = -f
            w.append(f*k/Integer(i+1))
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgf'] = g**Rational(1, d+1)
                break

    # Hyperbolic logarithmic Generating Function (hlgf)
    if 'hlgf' in types:
        # Transform sequence (division by n+1)
        w = []
        for i, k in enumerate(v):
            w.append(k/Integer(i+1))
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['hlgf'] = g**Rational(1, d+1)
                break

    # Logarithmic derivative of ordinary generating Function (lgdogf)
    if v[0] != 0 and ('lgdogf' in types
                       or ('ogf' in types and 'ogf' not in result)):
        # Transform sequence by computing f'(x)/f(x)
        # because log(f(x)) = integrate( f'(x)/f(x) )
        a, w = sympify(v[0]), []
        for n in range(len(v)-1):
            w.append(
               (v[n+1]*(n+1) - sum(w[-i-1]*v[i+1] for i in range(n)))/a)
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdogf'] = g**Rational(1, d+1)
                if 'ogf' not in result:
                    result['ogf'] = exp(integrate(result['lgdogf'], X))
                break

    # Logarithmic derivative of exponential generating Function (lgdegf)
    if v[0] != 0 and ('lgdegf' in types
                       or ('egf' in types and 'egf' not in result)):
        # Transform sequence / step 1 (division by factorial)
        z, f = [], S.One
        for i, k in enumerate(v):
            f *= i if i else 1
            z.append(k/f)
        # Transform sequence / step 2 by computing f'(x)/f(x)
        # because log(f(x)) = integrate( f'(x)/f(x) )
        a, w = z[0], []
        for n in range(len(z)-1):
            w.append(
               (z[n+1]*(n+1) - sum(w[-i-1]*z[i+1] for i in range(n)))/a)
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdegf'] = g**Rational(1, d+1)
                if 'egf' not in result:
                    result['egf'] = exp(integrate(result['lgdegf'], X))
                break

    return result


@public
def guess(l, all=False, evaluate=True, niter=2, variables=None):
    """
    This function is adapted from the Rate.m package for Mathematica
    written by Christian Krattenthaler.
    It tries to guess a formula from a given sequence of rational numbers.

    Explanation
    ===========

    In order to speed up the process, the 'all' variable is set to False by
    default, stopping the computation as some results are returned during an
    iteration; the variable can be set to True if more iterations are needed
    (other formulas may be found; however they may be equivalent to the first
    ones).

    Another option is the 'evaluate' variable (default is True); setting it
    to False will leave the involved products unevaluated.

    By default, the number of iterations is set to 2 but a greater value (up
    to len(l)-1) can be specified with the optional 'niter' variable.
    More and more convoluted results are found when the order of the
    iteration gets higher:

      * first iteration returns polynomial or rational functions;
      * second iteration returns products of rising factorials and their
        inverses;
      * third iteration returns products of products of rising factorials
        and their inverses;
      * etc.

    The returned formulas contain symbols i0, i1, i2, ... where the main
    variables is i0 (and auxiliary variables are i1, i2, ...). A list of
    other symbols can be provided in the 'variables' option; the length of
    the least should be the value of 'niter' (more is acceptable but only
    the first symbols will be used); in this case, the main variable will be
    the first symbol in the list.

    Examples
    ========

    >>> from sympy.concrete.guess import guess
    >>> guess([1,2,6,24,120], evaluate=False)
    [Product(i1 + 1, (i1, 1, i0 - 1))]

    >>> from sympy import symbols
    >>> r = guess([1,2,7,42,429,7436,218348,10850216], niter=4)
    >>> i0 = symbols("i0")
    >>> [r[0].subs(i0,n).doit() for n in range(1,10)]
    [1, 2, 7, 42, 429, 7436, 218348, 10850216, 911835460]
    """
    if any(a==0 for a in l[:-1]):
        return []
    N = len(l)
    niter = min(N-1, niter)
    myprod = product if evaluate else Product
    g = []
    res = []
    if variables is None:
        symb = symbols('i:'+str(niter))
    else:
        symb = variables
    for k, s in enumerate(symb):
        g.append(l)
        n, r = len(l), []
        for i in range(n-2-1, -1, -1):
            ri = rinterp(enumerate(g[k][:-1], start=1), i, X=s)
            if ((denom(ri).subs({s:n}) != 0)
                    and (ri.subs({s:n}) - g[k][-1] == 0)
                    and ri not in r):
                r.append(ri)
        if r:
            for i in range(k-1, -1, -1):
                r = [g[i][0]
                      * myprod(v, (symb[i+1], 1, symb[i]-1)) for v in r]
            if not all: return r
            res += r
        l = [Rational(l[i+1], l[i]) for i in range(N-k-1)]
    return res
<|endoftext|>from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
    Piecewise)
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrixbase import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning


def _common_new(cls, function, *symbols, discrete, **assumptions):
    """Return either a special return value or the tuple,
    (function, limits, orientation). This code is common to
    both ExprWithLimits and AddWithLimits."""
    function = sympify(function)

    if isinstance(function, Equality):
        # This transforms e.g. Integral(Eq(x, y)) to Eq(Integral(x), Integral(y))
        # but that is only valid for definite integrals.
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        if not (limits and all(len(limit) == 3 for limit in limits)):
            sympy_deprecation_warning(
                """
                Creating a indefinite integral with an Eq() argument is
                deprecated.

                This is because indefinite integrals do not preserve equality
                due to the arbitrary constants. If you want an equality of
                indefinite integrals, use Eq(Integral(a, x), Integral(b, x))
                explicitly.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-indefinite-integral-eq",
                stacklevel=5,
            )

        lhs = function.lhs
        rhs = function.rhs
        return Equality(cls(lhs, *symbols, **assumptions), \
                        cls(rhs, *symbols, **assumptions))

    if function is S.NaN:
        return S.NaN

    if symbols:
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        for i, li in enumerate(limits):
            if len(li) == 4:
                function = function.subs(li[0], li[-1])
                limits[i] = Tuple(*li[:-1])
    else:
        # symbol not provided -- we can still try to compute a general form
        free = function.free_symbols
        if len(free) != 1:
            raise ValueError(
                "specify dummy variables for %s" % function)
        limits, orientation = [Tuple(s) for s in free], 1

    # denest any nested calls
    while cls == type(function):
        limits = list(function.limits) + limits
        function = function.function

    # Any embedded piecewise functions need to be brought out to the
    # top level. We only fold Piecewise that contain the integration
    # variable.
    reps = {}
    symbols_of_integration = {i[0] for i in limits}
    for p in function.atoms(Piecewise):
        if not p.has(*symbols_of_integration):
            reps[p] = Dummy()
    # mask off those that don't
    function = function.xreplace(reps)
    # do the fold
    function = piecewise_fold(function)
    # remove the masking
    function = function.xreplace({v: k for k, v in reps.items()})

    return function, limits, orientation


def _process_limits(*symbols, discrete=None):
    """Process the list of symbols and convert them to canonical limits,
    storing them as Tuple(symbol, lower, upper). The orientation of
    the function is also returned when the upper limit is missing
    so (x, 1, None) becomes (x, None, 1) and the orientation is changed.
    In the case that a limit is specified as (symbol, Range), a list of
    length 4 may be returned if a change of variables is needed; the
    expression that should replace the symbol in the expression is
    the fourth element in the list.
    """
    limits = []
    orientation = 1
    if discrete is None:
        err_msg = 'discrete must be True or False'
    elif discrete:
        err_msg = 'use Range, not Interval or Relational'
    else:
        err_msg = 'use Interval or Relational, not Range'
    for V in symbols:
        if isinstance(V, (Relational, BooleanFunction)):
            if discrete:
                raise TypeError(err_msg)
            variable = V.atoms(Symbol).pop()
            V = (variable, V.as_set())
        elif isinstance(V, Symbol) or getattr(V, '_diff_wrt', False):
            if isinstance(V, Idx):
                if V.lower is None or V.upper is None:
                    limits.append(Tuple(V))
                else:
                    limits.append(Tuple(V, V.lower, V.upper))
            else:
                limits.append(Tuple(V))
            continue
        if is_sequence(V) and not isinstance(V, Set):
            if len(V) == 2 and isinstance(V[1], Set):
                V = list(V)
                if isinstance(V[1], Interval):  # includes Reals
                    if discrete:
                        raise TypeError(err_msg)
                    V[1:] = V[1].inf, V[1].sup
                elif isinstance(V[1], Range):
                    if not discrete:
                        raise TypeError(err_msg)
                    lo = V[1].inf
                    hi = V[1].sup
                    dx = abs(V[1].step)  # direction doesn't matter
                    if dx == 1:
                        V[1:] = [lo, hi]
                    else:
                        if lo is not S.NegativeInfinity:
                            V = [V[0]] + [0, (hi - lo)//dx, dx*V[0] + lo]
                        else:
                            V = [V[0]] + [0, S.Infinity, -dx*V[0] + hi]
                else:
                    # more complicated sets would require splitting, e.g.
                    # Union(Interval(1, 3), interval(6,10))
                    raise NotImplementedError(
                        'expecting Range' if discrete else
                        'Relational or single Interval' )
            V = sympify(flatten(V))  # list of sympified elements/None
            if isinstance(V[0], (Symbol, Idx)) or getattr(V[0], '_diff_wrt', False):
                newsymbol = V[0]
                if len(V) == 3:
                    # general case
                    if V[2] is None and V[1] is not None:
                        orientation *= -1
                    V = [newsymbol] + [i for i in V[1:] if i is not None]

                lenV = len(V)
                if not isinstance(newsymbol, Idx) or lenV == 3:
                    if lenV == 4:
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 3:
                        if isinstance(newsymbol, Idx):
                            # Idx represents an integer which may have
                            # specified values it can take on; if it is
                            # given such a value, an error is raised here
                            # if the summation would try to give it a larger
                            # or smaller value than permitted. None and Symbolic
                            # values will not raise an error.
                            lo, hi = newsymbol.lower, newsymbol.upper
                            try:
                                if lo is not None and not bool(V[1] >= lo):
                                    raise ValueError("Summation will set Idx value too low.")
                            except TypeError:
                                pass
                            try:
                                if hi is not None and not bool(V[2] <= hi):
                                    raise ValueError("Summation will set Idx value too high.")
                            except TypeError:
                                pass
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 1 or (lenV == 2 and V[1] is None):
                        limits.append(Tuple(newsymbol))
                        continue
                    elif lenV == 2:
                        limits.append(Tuple(newsymbol, V[1]))
                        continue

        raise ValueError('Invalid limits given: %s' % str(symbols))

    return limits, orientation


class ExprWithLimits(Expr):
    __slots__ = ('is_commutative',)

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.products import Product
        pre = _common_new(cls, function, *symbols,
            discrete=issubclass(cls, Product), **assumptions)
        if isinstance(pre, tuple):
            function, limits, _ = pre
        else:
            return pre

        # limits must have upper and lower bounds; the indefinite form
        # is not supported. This restriction does not apply to AddWithLimits
        if any(len(l) != 3 or None in l for l in limits):
            raise ValueError('ExprWithLimits requires values for lower and upper bounds.')

        obj = Expr.__new__(cls, **assumptions)
        arglist = [function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative  # limits already checked

        return obj

    @property
    def function(self):
        """Return the function applied across limits.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x
        >>> Integral(x**2, (x,)).function
        x**2

        See Also
        ========

        limits, variables, free_symbols
        """
        return self._args[0]

    @property
    def kind(self):
        return self.function.kind

    @property
    def limits(self):
        """Return the limits of expression.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i
        >>> Integral(x**i, (i, 1, 3)).limits
        ((i, 1, 3),)

        See Also
        ========

        function, variables, free_symbols
        """
        return self._args[1:]

    @property
    def variables(self):
        """Return a list of the limit variables.

        >>> from sympy import Sum
        >>> from sympy.abc import x, i
        >>> Sum(x**i, (i, 1, 3)).variables
        [i]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        return [l[0] for l in self.limits]

    @property
    def bound_symbols(self):
        """Return only variables that are dummy variables.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i, j, k
        >>> Integral(x**i, (i, 1, 3), (j, 2), k).bound_symbols
        [i, j]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        return [l[0] for l in self.limits if len(l) != 1]

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y
        >>> Sum(x, (x, y, 1)).free_symbols
        {y}
        """
        # don't test for any special values -- nominal free symbols
        # should be returned, e.g. don't return set() if the
        # function is zero -- treat it like an unevaluated expression.
        function, limits = self.function, self.limits
        # mask off non-symbol integration variables that have
        # more than themself as a free symbol
        reps = {i[0]: i[0] if i[0].free_symbols == {i[0]} else Dummy()
            for i in self.limits}
        function = function.xreplace(reps)
        isyms = function.free_symbols
        for xab in limits:
            v = reps[xab[0]]
            if len(xab) == 1:
                isyms.add(v)
                continue
            # take out the target symbol
            if v in isyms:
                isyms.remove(v)
            # add in the new symbols
            for i in xab[1:]:
                isyms.update(i.free_symbols)
        reps = {v: k for k, v in reps.items()}
        return {reps.get(_, _) for _ in isyms}

    @property
    def is_number(self):
        """Return True if the Sum has no free symbols, else False."""
        return not self.free_symbols

    def _eval_interval(self, x, a, b):
        limits = [(i if i[0] != x else (x, a, b)) for i in self.limits]
        integrand = self.function
        return self.func(integrand, *limits)

    def _eval_subs(self, old, new):
        """
        Perform substitutions over non-dummy variables
        of an expression with limits.  Also, can be used
        to specify point-evaluation of an abstract antiderivative.

        Examples
        ========

        >>> from sympy import Sum, oo
        >>> from sympy.abc import s, n
        >>> Sum(1/n**s, (n, 1, oo)).subs(s, 2)
        Sum(n**(-2), (n, 1, oo))

        >>> from sympy import Integral
        >>> from sympy.abc import x, a
        >>> Integral(a*x**2, x).subs(x, 4)
        Integral(a*x**2, (x, 4))

        See Also
        ========

        variables : Lists the integration variables
        transform : Perform mapping on the dummy variable for integrals
        change_index : Perform mapping on the sum and product dummy variables

        """
        func, limits = self.function, list(self.limits)

        # If one of the expressions we are replacing is used as a func index
        # one of two things happens.
        #   - the old variable first appears as a free variable
        #     so we perform all free substitutions before it becomes
        #     a func index.
        #   - the old variable first appears as a func index, in
        #     which case we ignore.  See change_index.

        # Reorder limits to match standard mathematical practice for scoping
        limits.reverse()

        if not isinstance(old, Symbol) or \
                old.free_symbols.intersection(self.free_symbols):
            sub_into_func = True
            for i, xab in enumerate(limits):
                if 1 == len(xab) and old == xab[0]:
                    if new._diff_wrt:
                        xab = (new,)
                    else:
                        xab = (old, old)
                limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                if len(xab[0].free_symbols.intersection(old.free_symbols)) != 0:
                    sub_into_func = False
                    break
            if isinstance(old, (AppliedUndef, UndefinedFunction)):
                sy2 = set(self.variables).intersection(set(new.atoms(Symbol)))
                sy1 = set(self.variables).intersection(set(old.args))
                if not sy2.issubset(sy1):
                    raise ValueError(
                        "substitution cannot create dummy dependencies")
                sub_into_func = True
            if sub_into_func:
                func = func.subs(old, new)
        else:
            # old is a Symbol and a dummy variable of some limit
            for i, xab in enumerate(limits):
                if len(xab) == 3:
                    limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                    if old == xab[0]:
                        break
        # simplify redundant limits (x, x)  to (x, )
        for i, xab in enumerate(limits):
            if len(xab) == 2 and (xab[0] - xab[1]).is_zero:
                limits[i] = Tuple(xab[0], )

        # Reorder limits back to representation-form
        limits.reverse()

        return self.func(func, *limits)

    @property
    def has_finite_limits(self):
        """
        Returns True if the limits are known to be finite, either by the
        explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be infinite, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 1, 8)).has_finite_limits
        True

        >>> Integral(x, (x, 1, oo)).has_finite_limits
        False

        >>> M = Symbol('M')
        >>> Sum(x, (x, 1, M)).has_finite_limits

        >>> N = Symbol('N', integer=True)
        >>> Product(x, (x, 1, N)).has_finite_limits
        True

        See Also
        ========

        has_reversed_limits

        """

        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                if any(l.is_infinite for l in lim[1:]):
                    # Any of the bounds are +/-oo
                    return False
                elif any(l.is_infinite is None for l in lim[1:]):
                    # Maybe there are assumptions on the variable?
                    if lim[0].is_infinite is None:
                        ret_None = True
            else:
                if lim[0].is_infinite is None:
                    ret_None = True

        if ret_None:
            return None
        return True

    @property
    def has_reversed_limits(self):
        """
        Returns True if the limits are known to be in reversed order, either
        by the explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be in normal order, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 8, 1)).has_reversed_limits
        True

        >>> Sum(x, (x, 1, oo)).has_reversed_limits
        False

        >>> M = Symbol('M')
        >>> Integral(x, (x, 1, M)).has_reversed_limits

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(x, (x, 1, N)).has_reversed_limits
        False

        >>> Product(x, (x, 2, N)).has_reversed_limits

        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits
        False

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence

        """
        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                var, a, b = lim
                dif = b - a
                if dif.is_extended_negative:
                    return True
                elif dif.is_extended_nonnegative:
                    continue
                else:
                    ret_None = True
            else:
                return None
        if ret_None:
            return None
        return False


class AddWithLimits(ExprWithLimits):
    r"""Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """

    __slots__ = ()

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.summations import Sum
        pre = _common_new(cls, function, *symbols,
            discrete=issubclass(cls, Sum), **assumptions)
        if isinstance(pre, tuple):
            function, limits, orientation = pre
        else:
            return pre

        obj = Expr.__new__(cls, **assumptions)
        arglist = [orientation*function]  # orientation not used in ExprWithLimits
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative  # limits already checked

        return obj

    def _eval_adjoint(self):
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.adjoint(), *self.limits)
        return None

    def _eval_conjugate(self):
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.conjugate(), *self.limits)
        return None

    def _eval_transpose(self):
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_factor(self, **hints):
        if 1 == len(self.limits):
            summand = self.function.factor(**hints)
            if summand.is_Mul:
                out = sift(summand.args, lambda w: w.is_commutative \
                    and not set(self.variables) & w.free_symbols)
                return Mul(*out[True])*self.func(Mul(*out[False]), \
                    *self.limits)
        else:
            summand = self.func(self.function, *self.limits[0:-1]).factor()
            if not summand.has(self.variables[-1]):
                return self.func(1, [self.limits[-1]]).doit()*summand
            elif isinstance(summand, Mul):
                return self.func(summand, self.limits[-1]).factor()
        return self

    def _eval_expand_basic(self, **hints):
        summand = self.function.expand(**hints)
        force = hints.get('force', False)
        if (summand.is_Add and (force or summand.is_commutative and
                 self.has_finite_limits is not False)):
            return Add(*[self.func(i, *self.limits) for i in summand.args])
        elif isinstance(summand, MatrixBase):
            return summand.applyfunc(lambda x: self.func(x, *self.limits))
        elif summand != self.function:
            return self.func(summand, *self.limits)
        return self
<|endoftext|>from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.core.singleton import S
from sympy.core.relational import Eq

class ReorderError(NotImplementedError):
    """
    Exception raised when trying to reorder dependent limits.
    """
    def __init__(self, expr, msg):
        super().__init__(
            "%s could not be reordered: %s." % (expr, msg))

class ExprWithIntLimits(ExprWithLimits):
    """
    Superclass for Product and Sum.

    See Also
    ========

    sympy.concrete.expr_with_limits.ExprWithLimits
    sympy.concrete.products.Product
    sympy.concrete.summations.Sum
    """
    __slots__ = ()

    def change_index(self, var, trafo, newvar=None):
        r"""
        Change index of a Sum or Product.

        Perform a linear transformation `x \mapsto a x + b` on the index variable
        `x`. For `a` the only values allowed are `\pm 1`. A new variable to be used
        after the change of index can also be specified.

        Explanation
        ===========

        ``change_index(expr, var, trafo, newvar=None)`` where ``var`` specifies the
        index variable `x` to transform. The transformation ``trafo`` must be linear
        and given in terms of ``var``. If the optional argument ``newvar`` is
        provided then ``var`` gets replaced by ``newvar`` in the final expression.

        Examples
        ========

        >>> from sympy import Sum, Product, simplify
        >>> from sympy.abc import x, y, a, b, c, d, u, v, i, j, k, l

        >>> S = Sum(x, (x, a, b))
        >>> S.doit()
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, x + 1, y)
        >>> Sn
        Sum(y - 1, (y, a + 1, b + 1))
        >>> Sn.doit()
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, -x, y)
        >>> Sn
        Sum(-y, (y, -b, -a))
        >>> Sn.doit()
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, x+u)
        >>> Sn
        Sum(-u + x, (x, a + u, b + u))
        >>> Sn.doit()
        -a**2/2 - a*u + a/2 + b**2/2 + b*u + b/2 - u*(-a + b + 1) + u
        >>> simplify(Sn.doit())
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, -x - u, y)
        >>> Sn
        Sum(-u - y, (y, -b - u, -a - u))
        >>> Sn.doit()
        -a**2/2 - a*u + a/2 + b**2/2 + b*u + b/2 - u*(-a + b + 1) + u
        >>> simplify(Sn.doit())
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> P = Product(i*j**2, (i, a, b), (j, c, d))
        >>> P
        Product(i*j**2, (i, a, b), (j, c, d))
        >>> P2 = P.change_index(i, i+3, k)
        >>> P2
        Product(j**2*(k - 3), (k, a + 3, b + 3), (j, c, d))
        >>> P3 = P2.change_index(j, -j, l)
        >>> P3
        Product(l**2*(k - 3), (k, a + 3, b + 3), (l, -d, -c))

        When dealing with symbols only, we can make a
        general linear transformation:

        >>> Sn = S.change_index(x, u*x+v, y)
        >>> Sn
        Sum((-v + y)/u, (y, b*u + v, a*u + v))
        >>> Sn.doit()
        -v*(a*u - b*u + 1)/u + (a**2*u**2/2 + a*u*v + a*u/2 - b**2*u**2/2 - b*u*v + b*u/2 + v)/u
        >>> simplify(Sn.doit())
        a**2*u/2 + a/2 - b**2*u/2 + b/2

        However, the last result can be inconsistent with usual
        summation where the index increment is always 1. This is
        obvious as we get back the original value only for ``u``
        equal +1 or -1.

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder,
        sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
        if newvar is None:
            newvar = var

        limits = []
        for limit in self.limits:
            if limit[0] == var:
                p = trafo.as_poly(var)
                if p.degree() != 1:
                    raise ValueError("Index transformation is not linear")
                alpha = p.coeff_monomial(var)
                beta = p.coeff_monomial(S.One)
                if alpha.is_number:
                    if alpha == S.One:
                        limits.append((newvar, alpha*limit[1] + beta, alpha*limit[2] + beta))
                    elif alpha == S.NegativeOne:
                        limits.append((newvar, alpha*limit[2] + beta, alpha*limit[1] + beta))
                    else:
                        raise ValueError("Linear transformation results in non-linear summation stepsize")
                else:
                    # Note that the case of alpha being symbolic can give issues if alpha < 0.
                    limits.append((newvar, alpha*limit[2] + beta, alpha*limit[1] + beta))
            else:
                limits.append(limit)

        function = self.function.subs(var, (var - beta)/alpha)
        function = function.subs(var, newvar)

        return self.func(function, *limits)


    def index(expr, x):
        """
        Return the index of a dummy variable in the list of limits.

        Explanation
        ===========

        ``index(expr, x)``  returns the index of the dummy variable ``x`` in the
        limits of ``expr``. Note that we start counting with 0 at the inner-most
        limits tuple.

        Examples
        ========

        >>> from sympy.abc import x, y, a, b, c, d
        >>> from sympy import Sum, Product
        >>> Sum(x*y, (x, a, b), (y, c, d)).index(x)
        0
        >>> Sum(x*y, (x, a, b), (y, c, d)).index(y)
        1
        >>> Product(x*y, (x, a, b), (y, c, d)).index(x)
        0
        >>> Product(x*y, (x, a, b), (y, c, d)).index(y)
        1

        See Also
        ========

        reorder_limit, reorder, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
        variables = [limit[0] for limit in expr.limits]

        if variables.count(x) != 1:
            raise ValueError(expr, "Number of instances of variable not equal to one")
        else:
            return variables.index(x)

    def reorder(expr, *arg):
        """
        Reorder limits in a expression containing a Sum or a Product.

        Explanation
        ===========

        ``expr.reorder(*arg)`` reorders the limits in the expression ``expr``
        according to the list of tuples given by ``arg``. These tuples can
        contain numerical indices or index variable names or involve both.

        Examples
        ========

        >>> from sympy import Sum, Product
        >>> from sympy.abc import x, y, z, a, b, c, d, e, f

        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((x, y))
        Sum(x*y, (y, c, d), (x, a, b))

        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder((x, y), (x, z), (y, z))
        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        >>> P = Product(x*y*z, (x, a, b), (y, c, d), (z, e, f))
        >>> P.reorder((x, y), (x, z), (y, z))
        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        We can also select the index variables by counting them, starting
        with the inner-most one:

        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder((0, 1))
        Sum(x**2, (x, c, d), (x, a, b))

        And of course we can mix both schemes:

        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((y, x))
        Sum(x*y, (y, c, d), (x, a, b))
        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((y, 0))
        Sum(x*y, (y, c, d), (x, a, b))

        See Also
        ========

        reorder_limit, index, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
        new_expr = expr

        for r in arg:
            if len(r) != 2:
                raise ValueError(r, "Invalid number of arguments")

            index1 = r[0]
            index2 = r[1]

            if not isinstance(r[0], int):
                index1 = expr.index(r[0])
            if not isinstance(r[1], int):
                index2 = expr.index(r[1])

            new_expr = new_expr.reorder_limit(index1, index2)

        return new_expr


    def reorder_limit(expr, x, y):
        """
        Interchange two limit tuples of a Sum or Product expression.

        Explanation
        ===========

        ``expr.reorder_limit(x, y)`` interchanges two limit tuples. The
        arguments ``x`` and ``y`` are integers corresponding to the index
        variables of the two limits which are to be interchanged. The
        expression ``expr`` has to be either a Sum or a Product.

        Examples
        ========

        >>> from sympy.abc import x, y, z, a, b, c, d, e, f
        >>> from sympy import Sum, Product

        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)
        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))
        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder_limit(1, 0)
        Sum(x**2, (x, c, d), (x, a, b))

        >>> Product(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)
        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        See Also
        ========

        index, reorder, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
        var = {limit[0] for limit in expr.limits}
        limit_x = expr.limits[x]
        limit_y = expr.limits[y]

        if (len(set(limit_x[1].free_symbols).intersection(var)) == 0 and
            len(set(limit_x[2].free_symbols).intersection(var)) == 0 and
            len(set(limit_y[1].free_symbols).intersection(var)) == 0 and
            len(set(limit_y[2].free_symbols).intersection(var)) == 0):

            limits = []
            for i, limit in enumerate(expr.limits):
                if i == x:
                    limits.append(limit_y)
                elif i == y:
                    limits.append(limit_x)
                else:
                    limits.append(limit)

            return type(expr)(expr.function, *limits)
        else:
            raise ReorderError(expr, "could not interchange the two limits specified")

    @property
    def has_empty_sequence(self):
        """
        Returns True if the Sum or Product is computed for an empty sequence.

        Examples
        ========

        >>> from sympy import Sum, Product, Symbol
        >>> m = Symbol('m')
        >>> Sum(m, (m, 1, 0)).has_empty_sequence
        True

        >>> Sum(m, (m, 1, 1)).has_empty_sequence
        False

        >>> M = Symbol('M', integer=True, positive=True)
        >>> Product(m, (m, 1, M)).has_empty_sequence
        False

        >>> Product(m, (m, 2, M)).has_empty_sequence

        >>> Product(m, (m, M + 1, M)).has_empty_sequence
        True

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(m, (m, N, M)).has_empty_sequence

        >>> N = Symbol('N', integer=True, negative=True)
        >>> Sum(m, (m, N, M)).has_empty_sequence
        False

        See Also
        ========

        has_reversed_limits
        has_finite_limits

        """
        ret_None = False
        for lim in self.limits:
            dif = lim[1] - lim[2]
            eq = Eq(dif, 1)
            if eq == True:
                return True
            elif eq == False:
                continue
            else:
                ret_None = True

        if ret_None:
            return None
        return False
<|endoftext|>"""
This module implements sums and products containing the Kronecker Delta function.

References
==========

.. [1] https://mathworld.wolfram.com/KroneckerDelta.html

"""
from .products import product
from .summations import Sum, summation
from sympy.core import Add, Mul, S, Dummy
from sympy.core.cache import cacheit
from sympy.core.sorting import default_sort_key
from sympy.functions import KroneckerDelta, Piecewise, piecewise_fold
from sympy.polys.polytools import factor
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve


@cacheit
def _expand_delta(expr, index):
    """
    Expand the first Add containing a simple KroneckerDelta.
    """
    if not expr.is_Mul:
        return expr
    delta = None
    func = Add
    terms = [S.One]
    for h in expr.args:
        if delta is None and h.is_Add and _has_simple_delta(h, index):
            delta = True
            func = h.func
            terms = [terms[0]*t for t in h.args]
        else:
            terms = [t*h for t in terms]
    return func(*terms)


@cacheit
def _extract_delta(expr, index):
    """
    Extract a simple KroneckerDelta from the expression.

    Explanation
    ===========

    Returns the tuple ``(delta, newexpr)`` where:

      - ``delta`` is a simple KroneckerDelta expression if one was found,
        or ``None`` if no simple KroneckerDelta expression was found.

      - ``newexpr`` is a Mul containing the remaining terms; ``expr`` is
        returned unchanged if no simple KroneckerDelta expression was found.

    Examples
    ========

    >>> from sympy import KroneckerDelta
    >>> from sympy.concrete.delta import _extract_delta
    >>> from sympy.abc import x, y, i, j, k
    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), i)
    (KroneckerDelta(i, j), 4*x*y)
    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), k)
    (None, 4*x*y*KroneckerDelta(i, j))

    See Also
    ========

    sympy.functions.special.tensor_functions.KroneckerDelta
    deltaproduct
    deltasummation
    """
    if not _has_simple_delta(expr, index):
        return (None, expr)
    if isinstance(expr, KroneckerDelta):
        return (expr, S.One)
    if not expr.is_Mul:
        raise ValueError("Incorrect expr")
    delta = None
    terms = []

    for arg in expr.args:
        if delta is None and _is_simple_delta(arg, index):
            delta = arg
        else:
            terms.append(arg)
    return (delta, expr.func(*terms))


@cacheit
def _has_simple_delta(expr, index):
    """
    Returns True if ``expr`` is an expression that contains a KroneckerDelta
    that is simple in the index ``index``, meaning that this KroneckerDelta
    is nonzero for a single value of the index ``index``.
    """
    if expr.has(KroneckerDelta):
        if _is_simple_delta(expr, index):
            return True
        if expr.is_Add or expr.is_Mul:
            return any(_has_simple_delta(arg, index) for arg in expr.args)
    return False


@cacheit
def _is_simple_delta(delta, index):
    """
    Returns True if ``delta`` is a KroneckerDelta and is nonzero for a single
    value of the index ``index``.
    """
    if isinstance(delta, KroneckerDelta) and delta.has(index):
        p = (delta.args[0] - delta.args[1]).as_poly(index)
        if p:
            return p.degree() == 1
    return False


@cacheit
def _remove_multiple_delta(expr):
    """
    Evaluate products of KroneckerDelta's.
    """
    if expr.is_Add:
        return expr.func(*list(map(_remove_multiple_delta, expr.args)))
    if not expr.is_Mul:
        return expr
    eqs = []
    newargs = []
    for arg in expr.args:
        if isinstance(arg, KroneckerDelta):
            eqs.append(arg.args[0] - arg.args[1])
        else:
            newargs.append(arg)
    if not eqs:
        return expr
    solns = solve(eqs, dict=True)
    if len(solns) == 0:
        return S.Zero
    elif len(solns) == 1:
        newargs += [KroneckerDelta(k, v) for k, v in solns[0].items()]
        expr2 = expr.func(*newargs)
        if expr != expr2:
            return _remove_multiple_delta(expr2)
    return expr


@cacheit
def _simplify_delta(expr):
    """
    Rewrite a KroneckerDelta's indices in its simplest form.
    """
    if isinstance(expr, KroneckerDelta):
        try:
            slns = solve(expr.args[0] - expr.args[1], dict=True)
            if slns and len(slns) == 1:
                return Mul(*[KroneckerDelta(*(key, value))
                            for key, value in slns[0].items()])
        except NotImplementedError:
            pass
    return expr


@cacheit
def deltaproduct(f, limit):
    """
    Handle products containing a KroneckerDelta.

    See Also
    ========

    deltasummation
    sympy.functions.special.tensor_functions.KroneckerDelta
    sympy.concrete.products.product
    """
    if ((limit[2] - limit[1]) < 0) == True:
        return S.One

    if not f.has(KroneckerDelta):
        return product(f, limit)

    if f.is_Add:
        # Identify the term in the Add that has a simple KroneckerDelta
        delta = None
        terms = []
        for arg in sorted(f.args, key=default_sort_key):
            if delta is None and _has_simple_delta(arg, limit[0]):
                delta = arg
            else:
                terms.append(arg)
        newexpr = f.func(*terms)
        k = Dummy("kprime", integer=True)
        if isinstance(limit[1], int) and isinstance(limit[2], int):
            result = deltaproduct(newexpr, limit) + sum(deltaproduct(newexpr, (limit[0], limit[1], ik - 1)) *
                delta.subs(limit[0], ik) *
                deltaproduct(newexpr, (limit[0], ik + 1, limit[2])) for ik in range(int(limit[1]), int(limit[2] + 1))
            )
        else:
            result = deltaproduct(newexpr, limit) + deltasummation(
                deltaproduct(newexpr, (limit[0], limit[1], k - 1)) *
                delta.subs(limit[0], k) *
                deltaproduct(newexpr, (limit[0], k + 1, limit[2])),
                (k, limit[1], limit[2]),
                no_piecewise=_has_simple_delta(newexpr, limit[0])
            )
        return _remove_multiple_delta(result)

    delta, _ = _extract_delta(f, limit[0])

    if not delta:
        g = _expand_delta(f, limit[0])
        if f != g:
            try:
                return factor(deltaproduct(g, limit))
            except AssertionError:
                return deltaproduct(g, limit)
        return product(f, limit)

    return _remove_multiple_delta(f.subs(limit[0], limit[1])*KroneckerDelta(limit[2], limit[1])) + \
        S.One*_simplify_delta(KroneckerDelta(limit[2], limit[1] - 1))


@cacheit
def deltasummation(f, limit, no_piecewise=False):
    """
    Handle summations containing a KroneckerDelta.

    Explanation
    ===========

    The idea for summation is the following:

    - If we are dealing with a KroneckerDelta expression, i.e. KroneckerDelta(g(x), j),
      we try to simplify it.

      If we could simplify it, then we sum the resulting expression.
      We already know we can sum a simplified expression, because only
      simple KroneckerDelta expressions are involved.

      If we could not simplify it, there are two cases:

      1) The expression is a simple expression: we return the summation,
         taking care if we are dealing with a Derivative or with a proper
         KroneckerDelta.

      2) The expression is not simple (i.e. KroneckerDelta(cos(x))): we can do
         nothing at all.

    - If the expr is a multiplication expr having a KroneckerDelta term:

      First we expand it.

      If the expansion did work, then we try to sum the expansion.

      If not, we try to extract a simple KroneckerDelta term, then we have two
      cases:

      1) We have a simple KroneckerDelta term, so we return the summation.

      2) We did not have a simple term, but we do have an expression with
         simplified KroneckerDelta terms, so we sum this expression.

    Examples
    ========

    >>> from sympy import oo, symbols
    >>> from sympy.abc import k
    >>> i, j = symbols('i, j', integer=True, finite=True)
    >>> from sympy.concrete.delta import deltasummation
    >>> from sympy import KroneckerDelta
    >>> deltasummation(KroneckerDelta(i, k), (k, -oo, oo))
    1
    >>> deltasummation(KroneckerDelta(i, k), (k, 0, oo))
    Piecewise((1, i >= 0), (0, True))
    >>> deltasummation(KroneckerDelta(i, k), (k, 1, 3))
    Piecewise((1, (i >= 1) & (i <= 3)), (0, True))
    >>> deltasummation(k*KroneckerDelta(i, j)*KroneckerDelta(j, k), (k, -oo, oo))
    j*KroneckerDelta(i, j)
    >>> deltasummation(j*KroneckerDelta(i, j), (j, -oo, oo))
    i
    >>> deltasummation(i*KroneckerDelta(i, j), (i, -oo, oo))
    j

    See Also
    ========

    deltaproduct
    sympy.functions.special.tensor_functions.KroneckerDelta
    sympy.concrete.sums.summation
    """
    if ((limit[2] - limit[1]) < 0) == True:
        return S.Zero

    if not f.has(KroneckerDelta):
        return summation(f, limit)

    x = limit[0]

    g = _expand_delta(f, x)
    if g.is_Add:
        return piecewise_fold(
            g.func(*[deltasummation(h, limit, no_piecewise) for h in g.args]))

    # try to extract a simple KroneckerDelta term
    delta, expr = _extract_delta(g, x)

    if (delta is not None) and (delta.delta_range is not None):
        dinf, dsup = delta.delta_range
        if (limit[1] - dinf <= 0) == True and (limit[2] - dsup >= 0) == True:
            no_piecewise = True

    if not delta:
        return summation(f, limit)

    solns = solve(delta.args[0] - delta.args[1], x)
    if len(solns) == 0:
        return S.Zero
    elif len(solns) != 1:
        return Sum(f, limit)
    value = solns[0]
    if no_piecewise:
        return expr.subs(x, value)
    return Piecewise(
        (expr.subs(x, value), Interval(*limit[1:3]).as_relational(value)),
        (S.Zero, True)
    )
<|endoftext|>from __future__ import annotations

from .expr_with_intlimits import ExprWithIntLimits
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import RisingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import quo, roots


class Product(ExprWithIntLimits):
    r"""
    Represents unevaluated products.

    Explanation
    ===========

    ``Product`` represents a finite or infinite product, with the first
    argument being the general form of terms in the series, and the second
    argument being ``(dummy_variable, start, end)``, with ``dummy_variable``
    taking all integer values from ``start`` through ``end``. In accordance
    with long-standing mathematical convention, the end term is included in
    the product.

    Finite products
    ===============

    For finite products (and products with symbolic limits assumed to be finite)
    we follow the analogue of the summation convention described by Karr [1],
    especially definition 3 of section 1.4. The product:

    .. math::

        \prod_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \prod_{m \leq i < n} f(i) = f(m) f(m+1) \cdot \ldots \cdot f(n-2) f(n-1)

    with the upper limit value `f(n)` excluded. The product over an empty set is
    one if and only if `m = n`:

    .. math::

        \prod_{m \leq i < n} f(i) = 1  \quad \mathrm{for} \quad  m = n

    Finally, for all other products over empty sets we assume the following
    definition:

    .. math::

        \prod_{m \leq i < n} f(i) = \frac{1}{\prod_{n \leq i < m} f(i)}  \quad \mathrm{for} \quad  m > n

    It is important to note that above we define all products with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the product convention. Indeed we have:

    .. math::

        \prod_{m \leq i < n} f(i) = \prod_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import a, b, i, k, m, n, x
    >>> from sympy import Product, oo
    >>> Product(k, (k, 1, m))
    Product(k, (k, 1, m))
    >>> Product(k, (k, 1, m)).doit()
    factorial(m)
    >>> Product(k**2,(k, 1, m))
    Product(k**2, (k, 1, m))
    >>> Product(k**2,(k, 1, m)).doit()
    factorial(m)**2

    Wallis' product for pi:

    >>> W = Product(2*i/(2*i-1) * 2*i/(2*i+1), (i, 1, oo))
    >>> W
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    Direct computation currently fails:

    >>> W.doit()
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    But we can approach the infinite product by a limit of finite products:

    >>> from sympy import limit
    >>> W2 = Product(2*i/(2*i-1)*2*i/(2*i+1), (i, 1, n))
    >>> W2
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, n))
    >>> W2e = W2.doit()
    >>> W2e
    4**n*factorial(n)**2/(2**(2*n)*RisingFactorial(1/2, n)*RisingFactorial(3/2, n))
    >>> limit(W2e, n, oo)
    pi/2

    By the same formula we can compute sin(pi/2):

    >>> from sympy import combsimp, pi, gamma, simplify
    >>> P = pi * x * Product(1 - x**2/k**2, (k, 1, n))
    >>> P = P.subs(x, pi/2)
    >>> P
    pi**2*Product(1 - pi**2/(4*k**2), (k, 1, n))/2
    >>> Pe = P.doit()
    >>> Pe
    pi**2*RisingFactorial(1 - pi/2, n)*RisingFactorial(1 + pi/2, n)/(2*factorial(n)**2)
    >>> limit(Pe, n, oo).gammasimp()
    sin(pi**2/2)
    >>> Pe.rewrite(gamma)
    (-1)**n*pi**2*gamma(pi/2)*gamma(n + 1 + pi/2)/(2*gamma(1 + pi/2)*gamma(-n + pi/2)*gamma(n + 1)**2)

    Products with the lower limit being larger than the upper one:

    >>> Product(1/i, (i, 6, 1)).doit()
    120
    >>> Product(i, (i, 2, 5)).doit()
    120

    The empty product:

    >>> Product(i, (i, n, n-1)).doit()
    1

    An example showing that the symbolic result of a product is still
    valid for seemingly nonsensical values of the limits. Then the Karr
    convention allows us to give a perfectly valid interpretation to
    those products by interchanging the limits according to the above rules:

    >>> P = Product(2, (i, 10, n)).doit()
    >>> P
    2**(n - 9)
    >>> P.subs(n, 5)
    1/16
    >>> Product(2, (i, 10, 5)).doit()
    1/16
    >>> 1/Product(2, (i, 6, 9)).doit()
    1/16

    An explicit example of the Karr summation convention applied to products:

    >>> P1 = Product(x, (i, a, b)).doit()
    >>> P1
    x**(-a + b + 1)
    >>> P2 = Product(x, (i, b+1, a-1)).doit()
    >>> P2
    x**(a - b - 1)
    >>> simplify(P1 * P2)
    1

    And another one:

    >>> P1 = Product(i, (i, b, a)).doit()
    >>> P1
    RisingFactorial(b, a - b + 1)
    >>> P2 = Product(i, (i, a+1, b-1)).doit()
    >>> P2
    RisingFactorial(a + 1, -a + b - 1)
    >>> P1 * P2
    RisingFactorial(b, a - b + 1)*RisingFactorial(a + 1, -a + b - 1)
    >>> combsimp(P1 * P2)
    1

    See Also
    ========

    Sum, summation
    product

    References
    ==========

    .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
           Volume 28 Issue 2, April 1981, Pages 305-350
           https://dl.acm.org/doi/10.1145/322248.322255
    .. [2] https://en.wikipedia.org/wiki/Multiplication#Capital_Pi_notation
    .. [3] https://en.wikipedia.org/wiki/Empty_product
    """

    __slots__ = ()

    limits: tuple[tuple[Symbol, Expr, Expr]]

    def __new__(cls, function, *symbols, **assumptions):
        obj = ExprWithIntLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def _eval_rewrite_as_Sum(self, *args, **kwargs):
        return exp(Sum(log(self.function), *self.limits))

    @property
    def term(self):
        return self._args[0]
    function = term

    def _eval_is_zero(self):
        if self.has_empty_sequence:
            return False

        z = self.term.is_zero
        if z is True:
            return True
        if self.has_finite_limits:
            # A Product is zero only if its term is zero assuming finite limits.
            return z

    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True

        return self.function.is_extended_real

    def _eval_is_positive(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_positive and self.has_finite_limits:
            return True

    def _eval_is_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_nonnegative and self.has_finite_limits:
            return True

    def _eval_is_extended_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_extended_nonnegative:
            return True

    def _eval_is_extended_nonpositive(self):
        if self.has_empty_sequence:
            return True

    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True

    def doit(self, **hints):
        # first make sure any definite limits have product
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        from sympy.simplify.powsimp import powsimp
        f = self.function
        for index, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = 1 / f

            g = self._eval_product(f, (i, a, b))
            if g in (None, S.NaN):
                return self.func(powsimp(f), *self.limits[index:])
            else:
                f = g

        if hints.get('deep', True):
            return f.doit(**hints)
        else:
            return powsimp(f)

    def _eval_conjugate(self):
        return self.func(self.function.conjugate(), *self.limits)

    def _eval_product(self, term, limits):

        (k, a, n) = limits

        if k not in term.free_symbols:
            if (term - 1).is_zero:
                return S.One
            return term**(n - a + 1)

        if a == n:
            return term.subs(k, a)

        from .delta import deltaproduct, _has_simple_delta
        if term.has(KroneckerDelta) and _has_simple_delta(term, limits[0]):
            return deltaproduct(term, limits)

        dif = n - a
        definite = dif.is_Integer
        if definite and (dif < 100):
            return self._eval_product_direct(term, limits)

        elif term.is_polynomial(k):
            poly = term.as_poly(k)

            A = B = Q = S.One

            all_roots = roots(poly)

            M = 0
            for r, m in all_roots.items():
                M += m
                A *= RisingFactorial(a - r, n - a + 1)**m
                Q *= (n - r)**m

            if M < poly.degree():
                arg = quo(poly, Q.as_poly(k))
                B = self.func(arg, (k, a, n)).doit()

            return poly.LC()**(n - a + 1) * A * B

        elif term.is_Add:
            factored = factor_terms(term, fraction=True)
            if factored.is_Mul:
                return self._eval_product(factored, (k, a, n))

        elif term.is_Mul:
            # Factor in part without the summation variable and part with
            without_k, with_k = term.as_coeff_mul(k)

            if len(with_k) >= 2:
                # More than one term including k, so still a multiplication
                exclude, include = [], []
                for t in with_k:
                    p = self._eval_product(t, (k, a, n))

                    if p is not None:
                        exclude.append(p)
                    else:
                        include.append(t)

                if not exclude:
                    return None
                else:
                    arg = term._new_rawargs(*include)
                    A = Mul(*exclude)
                    B = self.func(arg, (k, a, n)).doit()
                    return without_k**(n - a + 1)*A * B
            else:
                # Just a single term
                p = self._eval_product(with_k[0], (k, a, n))
                if p is None:
                    p = self.func(with_k[0], (k, a, n)).doit()
                return without_k**(n - a + 1)*p


        elif term.is_Pow:
            if not term.base.has(k):
                s = summation(term.exp, (k, a, n))

                return term.base**s
            elif not term.exp.has(k):
                p = self._eval_product(term.base, (k, a, n))

                if p is not None:
                    return p**term.exp

        elif isinstance(term, Product):
            evaluated = term.doit()
            f = self._eval_product(evaluated, limits)
            if f is None:
                return self.func(evaluated, limits)
            else:
                return f

        if definite:
            return self._eval_product_direct(term, limits)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import product_simplify
        rv = product_simplify(self, **kwargs)
        return rv.doit() if kwargs['doit'] else rv

    def _eval_transpose(self):
        if self.is_commutative:
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_product_direct(self, term, limits):
        (k, a, n) = limits
        return Mul(*[term.subs(k, a + i) for i in range(n - a + 1)])

    def _eval_derivative(self, x):
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero
        f, limits = self.function, list(self.limits)
        limit = limits.pop(-1)
        if limits:
            f = self.func(f, *limits)
        i, a, b = limit
        if x in a.free_symbols or x in b.free_symbols:
            return None
        h = Dummy()
