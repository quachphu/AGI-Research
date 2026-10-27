    will be automatically combined into a single tensor power object.

    Any Hilbert space, product, or sum may be raised to a tensor power. The
    ``TensorPowerHilbertSpace`` takes two arguments: the Hilbert space; and the
    tensor power (number).

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> n = symbols('n')
    >>> c = ComplexSpace(2)
    >>> hs = c**n
    >>> hs
    C(2)**n
    >>> hs.dimension
    2**n

    >>> c = ComplexSpace(2)
    >>> c*c
    C(2)**2
    >>> f = FockSpace()
    >>> c*f*f
    C(2)*F**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        return Basic.__new__(cls, *r)

    @classmethod
    def eval(cls, args):
        new_args = args[0], sympify(args[1])
        exp = new_args[1]
        #simplify hs**1 -> hs
        if exp is S.One:
            return args[0]
        #simplify hs**0 -> 1
        if exp is S.Zero:
            return S.One
        #check (and allow) for hs**(x+42+y...) case
        if len(exp.atoms()) == 1:
            if not (exp.is_Integer and exp >= 0 or exp.is_Symbol):
                raise ValueError('Hilbert spaces can only be raised to \
                positive integers or Symbols: %r' % exp)
        else:
            for power in exp.atoms():
                if not (power.is_Integer or power.is_Symbol):
                    raise ValueError('Tensor powers can only contain integers \
                    or Symbols: %r' % power)
        return new_args

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def dimension(self):
        if self.base.dimension is S.Infinity:
            return S.Infinity
        else:
            return self.base.dimension**self.exp

    def _sympyrepr(self, printer, *args):
        return "TensorPowerHilbertSpace(%s,%s)" % (printer._print(self.base,
        *args), printer._print(self.exp, *args))

    def _sympystr(self, printer, *args):
        return "%s**%s" % (printer._print(self.base, *args),
        printer._print(self.exp, *args))

    def _pretty(self, printer, *args):
        pform_exp = printer._print(self.exp, *args)
        if printer._use_unicode:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('\N{N-ARY CIRCLED TIMES OPERATOR}')))
        else:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('x')))
        pform_base = printer._print(self.base, *args)
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        base = printer._print(self.base, *args)
        exp = printer._print(self.exp, *args)
        return r'{%s}^{\otimes %s}' % (base, exp)
<|endoftext|>"""Functions for reordering operator expressions."""

import warnings

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

__all__ = [
    'normal_order',
    'normal_ordered_form'
]


def _expand_powers(factors):
    """
    Helper function for normal_ordered_form and normal_order: Expand a
    power expression to a multiplication expression so that that the
    expression can be handled by the normal ordering functions.
    """

    new_factors = []
    for factor in factors.args:
        if (isinstance(factor, Pow)
                and isinstance(factor.args[1], Integer)
                and factor.args[1] > 0):
            for n in range(factor.args[1]):
                new_factors.append(factor.args[0])
        else:
            new_factors.append(factor)

    return new_factors

def _normal_ordered_form_factor(product, independent=False, recursive_limit=10,
                                _recursive_depth=0):
    """
    Helper function for normal_ordered_form_factor: Write multiplication
    expression with bosonic or fermionic operators on normally ordered form,
    using the bosonic and fermionic commutation relations. The resulting
    operator expression is equivalent to the argument, but will in general be
    a sum of operator products instead of a simple product.
    """

    factors = _expand_powers(product)

    new_factors = []
    n = 0
    while n < len(factors) - 1:
        current, next = factors[n], factors[n + 1]
        if any(not isinstance(f, (FermionOp, BosonOp)) for f in (current, next)):
            new_factors.append(current)
            n += 1
            continue

        key_1 = (current.is_annihilation, str(current.name))
        key_2 = (next.is_annihilation, str(next.name))

        if key_1 <= key_2:
            new_factors.append(current)
            n += 1
            continue

        n += 2
        if current.is_annihilation and not next.is_annihilation:
            if isinstance(current, BosonOp) and isinstance(next, BosonOp):
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = Commutator(current, next)
                    new_factors.append(next * current + c)
                else:
                    new_factors.append(next * current + 1)
            elif isinstance(current, FermionOp) and isinstance(next, FermionOp):
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = AntiCommutator(current, next)
                    new_factors.append(-next * current + c)
                else:
                    new_factors.append(-next * current + 1)
        elif (current.is_annihilation == next.is_annihilation and
            isinstance(current, FermionOp) and isinstance(next, FermionOp)):
            new_factors.append(-next * current)
        else:
            new_factors.append(next * current)

    if n == len(factors) - 1:
        new_factors.append(factors[-1])

    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_ordered_form(expr,
                                   recursive_limit=recursive_limit,
                                   _recursive_depth=_recursive_depth + 1,
                                   independent=independent)


def _normal_ordered_form_terms(expr, independent=False, recursive_limit=10,
                               _recursive_depth=0):
    """
    Helper function for normal_ordered_form: loop through each term in an
    addition expression and call _normal_ordered_form_factor to perform the
    factor to an normally ordered expression.
    """

    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            new_term = _normal_ordered_form_factor(
                term, recursive_limit=recursive_limit,
                _recursive_depth=_recursive_depth, independent=independent)
            new_terms.append(new_term)
        else:
            new_terms.append(term)

    return Add(*new_terms)


def normal_ordered_form(expr, independent=False, recursive_limit=10,
                        _recursive_depth=0):
    """Write an expression with bosonic or fermionic operators on normal
    ordered form, where each term is normally ordered. Note that this
    normal ordered form is equivalent to the original expression.

    Parameters
    ==========

    expr : expression
        The expression write on normal ordered form.
    independent : bool (default False)
        Whether to consider operator with different names as operating in
        different Hilbert spaces. If False, the (anti-)commutation is left
        explicit.
    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_ordered_form
    >>> a = BosonOp("a")
    >>> normal_ordered_form(a * Dagger(a))
    1 + Dagger(a)*a
    """

    if _recursive_depth > recursive_limit:
        warnings.warn("Too many recursions, aborting")
        return expr

    if isinstance(expr, Add):
        return _normal_ordered_form_terms(expr,
                                          recursive_limit=recursive_limit,
                                          _recursive_depth=_recursive_depth,
                                          independent=independent)
    elif isinstance(expr, Mul):
        return _normal_ordered_form_factor(expr,
                                           recursive_limit=recursive_limit,
                                           _recursive_depth=_recursive_depth,
                                           independent=independent)
    else:
        return expr


def _normal_order_factor(product, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: Normal order a multiplication expression
    with bosonic or fermionic operators. In general the resulting operator
    expression will not be equivalent to original product.
    """

    factors = _expand_powers(product)

    n = 0
    new_factors = []
    while n < len(factors) - 1:

        if (isinstance(factors[n], BosonOp) and
                factors[n].is_annihilation):
            # boson
            if not isinstance(factors[n + 1], BosonOp):
                new_factors.append(factors[n])
            else:
                if factors[n + 1].is_annihilation:
                    new_factors.append(factors[n])
                else:
                    if factors[n].args[0] != factors[n + 1].args[0]:
                        new_factors.append(factors[n + 1] * factors[n])
                    else:
                        new_factors.append(factors[n + 1] * factors[n])
                    n += 1

        elif (isinstance(factors[n], FermionOp) and
              factors[n].is_annihilation):
            # fermion
            if not isinstance(factors[n + 1], FermionOp):
                new_factors.append(factors[n])
            else:
                if factors[n + 1].is_annihilation:
                    new_factors.append(factors[n])
                else:
                    if factors[n].args[0] != factors[n + 1].args[0]:
                        new_factors.append(-factors[n + 1] * factors[n])
                    else:
                        new_factors.append(-factors[n + 1] * factors[n])
                    n += 1

        else:
            new_factors.append(factors[n])

        n += 1

    if n == len(factors) - 1:
        new_factors.append(factors[-1])

    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_order(expr,
                            recursive_limit=recursive_limit,
                            _recursive_depth=_recursive_depth + 1)


def _normal_order_terms(expr, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: look through each term in an addition
    expression and call _normal_order_factor to perform the normal ordering
    on the factors.
    """

    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            new_term = _normal_order_factor(term,
                                            recursive_limit=recursive_limit,
                                            _recursive_depth=_recursive_depth)
            new_terms.append(new_term)
        else:
            new_terms.append(term)

    return Add(*new_terms)


def normal_order(expr, recursive_limit=10, _recursive_depth=0):
    """Normal order an expression with bosonic or fermionic operators. Note
    that this normal order is not equivalent to the original expression, but
    the creation and annihilation operators in each term in expr is reordered
    so that the expression becomes normal ordered.

    Parameters
    ==========

    expr : expression
        The expression to normal order.

    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_order
    >>> a = BosonOp("a")
    >>> normal_order(a * Dagger(a))
    Dagger(a)*a
    """
    if _recursive_depth > recursive_limit:
        warnings.warn("Too many recursions, aborting")
        return expr

    if isinstance(expr, Add):
        return _normal_order_terms(expr, recursive_limit=recursive_limit,
                                   _recursive_depth=_recursive_depth)
    elif isinstance(expr, Mul):
        return _normal_order_factor(expr, recursive_limit=recursive_limit,
                                    _recursive_depth=_recursive_depth)
    else:
        return expr
<|endoftext|>"""Fermionic quantum operators."""

from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta


__all__ = [
    'FermionOp',
    'FermionFockKet',
    'FermionFockBra'
]


class FermionOp(Operator):
    """A fermionic operator that satisfies {c, Dagger(c)} == 1.

    Parameters
    ==========

    name : str
        A string that labels the fermionic mode.

    annihilation : bool
        A bool that indicates if the fermionic operator is an annihilation
        (True, default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, AntiCommutator
    >>> from sympy.physics.quantum.fermion import FermionOp
    >>> c = FermionOp("c")
    >>> AntiCommutator(c, Dagger(c)).doit()
    1
    """
    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        return ("c", True)

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)

        if len(args) == 1:
            args = (args[0], S.One)

        if len(args) == 2:
            args = (args[0], Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_commutator_FermionOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # [c, d] = 0
            return S.Zero

        return None

    def _eval_anticommutator_FermionOp(self, other, **hints):
        if self.name == other.name:
            # {a^\dagger, a} = 1
            if not self.is_annihilation and other.is_annihilation:
                return S.One

        elif 'independent' in hints and hints['independent']:
            # {c, d} = 2 * c * d, because [c, d] = 0 for independent operators
            return 2 * self * other

        return None

    def _eval_anticommutator_BosonOp(self, other, **hints):
        # because fermions and bosons commute
        return 2 * self * other

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return FermionOp(str(self.name), not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)
        else:
            return r'{{%s}^\dagger}' % str(self.name)

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)
        else:
            return r'Dagger(%s)' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm('\N{DAGGER}')

    def _eval_power(self, exp):
        from sympy.core.singleton import S
        if exp == 0:
            return S.One
        elif exp == 1:
            return self
        elif (exp > 1) == True and exp.is_integer == True:
            return S.Zero
        elif (exp < 0) == True or exp.is_integer == False:
            raise ValueError("Fermionic operators can only be raised to a"
                " positive integer power")
        return Operator._eval_power(self, exp)

class FermionFockKet(Ket):
    """Fock state ket for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return FermionFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_FermionFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_FermionOp(self, op, **options):
        if op.is_annihilation:
            if self.n == 1:
                return FermionFockKet(0)
            else:
                return S.Zero
        else:
            if self.n == 0:
                return FermionFockKet(1)
            else:
                return S.Zero


class FermionFockBra(Bra):
    """Fock state bra for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return FermionFockKet
<|endoftext|>"""Symbolic inner product."""

from sympy.core.expr import Expr
from sympy.core.kind import NumberKind
from sympy.functions.elementary.complexes import conjugate
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger


__all__ = [
    'InnerProduct'
]


# InnerProduct is not an QExpr because it is really just a regular commutative
# number. We have gone back and forth about this, but we gain a lot by having
# it subclass Expr. The main challenges were getting Dagger to work
# (we use _eval_conjugate) and represent (we can use atoms and subs). Having
# it be an Expr, mean that there are no commutative QExpr subclasses,
# which simplifies the design of everything.

class InnerProduct(Expr):
    """An unevaluated inner product between a Bra and a Ket [1].

    Parameters
    ==========

    bra : BraBase or subclass
        The bra on the left side of the inner product.
    ket : KetBase or subclass
        The ket on the right side of the inner product.

    Examples
    ========

    Create an InnerProduct and check its properties:

        >>> from sympy.physics.quantum import Bra, Ket
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> ip = b*k
        >>> ip
        <b|k>
        >>> ip.bra
        <b|
        >>> ip.ket
        |k>

    In quantum expressions, inner products will be automatically
    identified and created::

        >>> b*k
        <b|k>

    In more complex expressions, where there is ambiguity in whether inner or
    outer products should be created, inner products have high priority::

        >>> k*b*k*b
        <b|k>*|k><b|

    Notice how the inner product <b|k> moved to the left of the expression
    because inner products are commutative complex numbers.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inner_product
    """

    kind = NumberKind

    is_complex = True

    def __new__(cls, bra, ket):
        # Keep the import of BraBase and KetBase here to avoid problems
        # with circular imports.
        from sympy.physics.quantum.state import KetBase, BraBase
        if not isinstance(ket, KetBase):
            raise TypeError('KetBase subclass expected, got: %r' % ket)
        if not isinstance(bra, BraBase):
            raise TypeError('BraBase subclass expected, got: %r' % ket)
        obj = Expr.__new__(cls, bra, ket)
        return obj

    @property
    def bra(self):
        return self.args[0]

    @property
    def ket(self):
        return self.args[1]

    def _eval_conjugate(self):
        return InnerProduct(Dagger(self.ket), Dagger(self.bra))

    def _sympyrepr(self, printer, *args):
        return '%s(%s,%s)' % (self.__class__.__name__,
            printer._print(self.bra, *args), printer._print(self.ket, *args))

    def _sympystr(self, printer, *args):
        sbra = printer._print(self.bra)
        sket = printer._print(self.ket)
        return '%s|%s' % (sbra[:-1], sket[1:])

    def _pretty(self, printer, *args):
        # Print state contents
        bra = self.bra._print_contents_pretty(printer, *args)
        ket = self.ket._print_contents_pretty(printer, *args)
        # Print brackets
        height = max(bra.height(), ket.height())
        use_unicode = printer._use_unicode
        lbracket, _ = self.bra._pretty_brackets(height, use_unicode)
        cbracket, rbracket = self.ket._pretty_brackets(height, use_unicode)
        # Build innerproduct
        pform = prettyForm(*bra.left(lbracket))
        pform = prettyForm(*pform.right(cbracket))
        pform = prettyForm(*pform.right(ket))
        pform = prettyForm(*pform.right(rbracket))
        return pform

    def _latex(self, printer, *args):
        bra_label = self.bra._print_contents_latex(printer, *args)
        ket = printer._print(self.ket, *args)
        return r'\left\langle %s \right. %s' % (bra_label, ket)

    def doit(self, **hints):
        try:
            r = self.ket._eval_innerproduct(self.bra, **hints)
        except NotImplementedError:
            try:
                r = conjugate(
                    self.bra.dual._eval_innerproduct(self.ket.dual, **hints)
                )
            except NotImplementedError:
                r = None
        if r is not None:
            return r
        return self
<|endoftext|>"""A cache for storing small matrices in multiple formats."""

from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.matrixutils import (
    to_sympy, to_numpy, to_scipy_sparse
)


class MatrixCache:
    """A cache for small matrices in different formats.

    This class takes small matrices in the standard ``sympy.Matrix`` format,
    and then converts these to both ``numpy.matrix`` and
    ``scipy.sparse.csr_matrix`` matrices. These matrices are then stored for
    future recovery.
    """

    def __init__(self, dtype='complex'):
        self._cache = {}
        self.dtype = dtype

    def cache_matrix(self, name, m):
        """Cache a matrix by its name.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        m : list of lists
            The raw matrix data as a SymPy Matrix.
        """
        try:
            self._sympy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._numpy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._scipy_sparse_matrix(name, m)
        except ImportError:
            pass

    def get_matrix(self, name, format):
        """Get a cached matrix by name and format.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        format : str
            The format desired ('sympy', 'numpy', 'scipy.sparse')
        """
        m = self._cache.get((name, format))
        if m is not None:
            return m
        raise NotImplementedError(
            'Matrix with name %s and format %s is not available.' %
            (name, format)
        )

    def _store_matrix(self, name, format, m):
        self._cache[(name, format)] = m

    def _sympy_matrix(self, name, m):
        self._store_matrix(name, 'sympy', to_sympy(m))

    def _numpy_matrix(self, name, m):
        m = to_numpy(m, dtype=self.dtype)
        self._store_matrix(name, 'numpy', m)

    def _scipy_sparse_matrix(self, name, m):
        # TODO: explore different sparse formats. But sparse.kron will use
        # coo in most cases, so we use that here.
        m = to_scipy_sparse(m, dtype=self.dtype)
        self._store_matrix(name, 'scipy.sparse', m)


sqrt2_inv = Pow(2, Rational(-1, 2), evaluate=False)

# Save the common matrices that we will need
matrix_cache = MatrixCache()
matrix_cache.cache_matrix('eye2', Matrix([[1, 0], [0, 1]]))
matrix_cache.cache_matrix('op11', Matrix([[0, 0], [0, 1]]))  # |1><1|
matrix_cache.cache_matrix('op00', Matrix([[1, 0], [0, 0]]))  # |0><0|
matrix_cache.cache_matrix('op10', Matrix([[0, 0], [1, 0]]))  # |1><0|
matrix_cache.cache_matrix('op01', Matrix([[0, 1], [0, 0]]))  # |0><1|
matrix_cache.cache_matrix('X', Matrix([[0, 1], [1, 0]]))
matrix_cache.cache_matrix('Y', Matrix([[0, -I], [I, 0]]))
matrix_cache.cache_matrix('Z', Matrix([[1, 0], [0, -1]]))
matrix_cache.cache_matrix('S', Matrix([[1, 0], [0, I]]))
matrix_cache.cache_matrix('T', Matrix([[1, 0], [0, exp(I*pi/4)]]))
matrix_cache.cache_matrix('H', sqrt2_inv*Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('Hsqrt2', Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix(
    'SWAP', Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
matrix_cache.cache_matrix('ZX', sqrt2_inv*Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('ZY', Matrix([[I, 0], [0, -I]]))
<|endoftext|>"""Abstract tensor product."""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.kind import KindDispatcher
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.matrices.dense import DenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix as ImmutableMatrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import (
    KetKind, _KetKind,
    BraKind, _BraKind,
    OperatorKind, _OperatorKind
)
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray,
    scipy_sparse_matrix,
    matrix_tensor_product
)
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.trace import Tr


__all__ = [
    'TensorProduct',
    'tensor_product_simp'
]

#-----------------------------------------------------------------------------
# Tensor product
#-----------------------------------------------------------------------------

_combined_printing = False


def combined_tensor_printing(combined):
    """Set flag controlling whether tensor products of states should be
    printed as a combined bra/ket or as an explicit tensor product of different
    bra/kets. This is a global setting for all TensorProduct class instances.

    Parameters
    ----------
    combine : bool
        When true, tensor product states are combined into one ket/bra, and
        when false explicit tensor product notation is used between each
        ket/bra.
    """
    global _combined_printing
    _combined_printing = combined


class TensorProduct(Expr):
    """The tensor product of two or more arguments.

    For matrices, this uses ``matrix_tensor_product`` to compute the Kronecker
    or tensor product matrix. For other objects a symbolic ``TensorProduct``
    instance is returned. The tensor product is a non-commutative
    multiplication that is used primarily with operators and states in quantum
    mechanics.

    Currently, the tensor product distinguishes between commutative and
    non-commutative arguments.  Commutative arguments are assumed to be scalars
    and are pulled out in front of the ``TensorProduct``. Non-commutative
    arguments remain in the resulting ``TensorProduct``.

    Parameters
    ==========

    args : tuple
        A sequence of the objects to take the tensor product of.

    Examples
    ========

    Start with a simple tensor product of SymPy matrices::

        >>> from sympy import Matrix
        >>> from sympy.physics.quantum import TensorProduct

        >>> m1 = Matrix([[1,2],[3,4]])
        >>> m2 = Matrix([[1,0],[0,1]])
        >>> TensorProduct(m1, m2)
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2],
        [3, 0, 4, 0],
        [0, 3, 0, 4]])
        >>> TensorProduct(m2, m1)
        Matrix([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4]])

    We can also construct tensor products of non-commutative symbols:

        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> tp = TensorProduct(A, B)
        >>> tp
        AxB

    We can take the dagger of a tensor product (note the order does NOT reverse
    like the dagger of a normal product):

        >>> from sympy.physics.quantum import Dagger
        >>> Dagger(tp)
        Dagger(A)xDagger(B)

    Expand can be used to distribute a tensor product across addition:

        >>> C = Symbol('C',commutative=False)
        >>> tp = TensorProduct(A+B,C)
        >>> tp
        (A + B)xC
        >>> tp.expand(tensorproduct=True)
        AxC + BxC
    """
    is_commutative = False

    _kind_dispatcher = KindDispatcher("TensorProduct_kind_dispatcher", commutative=True)

    @property
    def kind(self):
        """Calculate the kind of a tensor product by looking at its children."""
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    def __new__(cls, *args):
        if isinstance(args[0], (Matrix, ImmutableMatrix, numpy_ndarray,
                                                    scipy_sparse_matrix)):
            return matrix_tensor_product(*args)
        c_part, new_args = cls.flatten(sympify(args))
        c_part = Mul(*c_part)
        if len(new_args) == 0:
            return c_part
        elif len(new_args) == 1:
            return c_part * new_args[0]
        else:
            tp = Expr.__new__(cls, *new_args)
            return c_part * tp

    @classmethod
    def flatten(cls, args):
        # TODO: disallow nested TensorProducts.
        c_part = []
        nc_parts = []
        for arg in args:
            cp, ncp = arg.args_cnc()
            c_part.extend(list(cp))
            nc_parts.append(Mul._from_args(ncp))
        return c_part, nc_parts

    def _eval_adjoint(self):
        return TensorProduct(*[Dagger(i) for i in self.args])

    def _eval_rewrite(self, rule, args, **hints):
        return TensorProduct(*args).expand(tensorproduct=True)

    def _sympystr(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + '('
            s = s + printer._print(self.args[i])
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + ')'
            if i != length - 1:
                s = s + 'x'
        return s

    def _pretty(self, printer, *args):

        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            length = len(self.args)
            pform = printer._print('', *args)
            for i in range(length):
                next_pform = printer._print('', *args)
                length_i = len(self.args[i].args)
                for j in range(length_i):
                    part_pform = printer._print(self.args[i].args[j], *args)
                    next_pform = prettyForm(*next_pform.right(part_pform))
                    if j != length_i - 1:
                        next_pform = prettyForm(*next_pform.right(', '))

                if len(self.args[i].args) > 1:
                    next_pform = prettyForm(
                        *next_pform.parens(left='{', right='}'))
                pform = prettyForm(*pform.right(next_pform))
                if i != length - 1:
                    pform = prettyForm(*pform.right(',' + ' '))

            pform = prettyForm(*pform.left(self.args[0].lbracket))
            pform = prettyForm(*pform.right(self.args[0].rbracket))
            return pform

        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (Add, Mul)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))
        return pform

    def _latex(self, printer, *args):

        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            def _label_wrap(label, nlabels):
                return label if nlabels == 1 else r"\left\{%s\right\}" % label

            s = r", ".join([_label_wrap(arg._print_label_latex(printer, *args),
                                        len(arg.args)) for arg in self.args])

            return r"{%s%s%s}" % (self.args[0].lbracket_latex, s,
                                  self.args[0].rbracket_latex)

        length = len(self.args)
        s = ''
        for i in range(length):
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\left('
            # The extra {} brackets are needed to get matplotlib's latex
            # rendered to render this properly.
            s = s + '{' + printer._print(self.args[i], *args) + '}'
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\right)'
            if i != length - 1:
                s = s + '\\otimes '
        return s

    def doit(self, **hints):
        return TensorProduct(*[item.doit(**hints) for item in self.args])

    def _eval_expand_tensorproduct(self, **hints):
        """Distribute TensorProducts across addition."""
        args = self.args
        add_args = []
        for i in range(len(args)):
            if isinstance(args[i], Add):
                for aa in args[i].args:
                    tp = TensorProduct(*args[:i] + (aa,) + args[i + 1:])
                    c_part, nc_part = tp.args_cnc()
                    # Check for TensorProduct object: is the one object in nc_part, if any:
                    # (Note: any other object type to be expanded must be added here)
                    if len(nc_part) == 1 and isinstance(nc_part[0], TensorProduct):
                        nc_part = (nc_part[0]._eval_expand_tensorproduct(), )
                    add_args.append(Mul(*c_part)*Mul(*nc_part))
                break

        if add_args:
            return Add(*add_args)
        else:
            return self

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', None)
        exp = self

        if indices is None or len(indices) == 0:
            return Mul(*[Tr(arg).doit() for arg in exp.args])
        else:
            return Mul(*[Tr(value).doit() if idx in indices else value
                         for idx, value in enumerate(exp.args)])


def tensor_product_simp_Mul(e):
    """Simplify a Mul with tensor products.

    .. deprecated:: 1.14.
        The transformations applied by this function are not done automatically
        when tensor products are combined.

    Originally, the main use of this function is to simplify a ``Mul`` of
    ``TensorProduct``s to a ``TensorProduct`` of ``Muls``.
    """
    sympy_deprecation_warning(
        """
        tensor_product_simp_Mul has been deprecated. The transformations
        performed by this function are now done automatically when
        tensor products are multiplied.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-tensorproduct-simp'
    )
    return e

def tensor_product_simp_Pow(e):
    """Evaluates ``Pow`` expressions whose base is ``TensorProduct``

    .. deprecated:: 1.14.
        The transformations applied by this function are not done automatically
        when tensor products are combined.
    """
    sympy_deprecation_warning(
        """
        tensor_product_simp_Pow has been deprecated. The transformations
        performed by this function are now done automatically when
        tensor products are exponentiated.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-tensorproduct-simp'
    )
    return e


def tensor_product_simp(e, **hints):
    """Try to simplify and combine tensor products.

    .. deprecated:: 1.14.
        The transformations applied by this function are not done automatically
        when tensor products are combined.

    Originally, this function tried to pull expressions inside of ``TensorProducts``.
    It only worked for relatively simple cases where the products have
    only scalars, raw ``TensorProducts``, not ``Add``, ``Pow``, ``Commutators``
    of ``TensorProducts``.
    """
    sympy_deprecation_warning(
        """
        tensor_product_simp has been deprecated. The transformations
        performed by this function are now done automatically when
        tensor products are combined.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-tensorproduct-simp'
    )
    return e


@TensorProduct._kind_dispatcher.register(_OperatorKind, _OperatorKind)
def find_op_kind(e1, e2):
    return OperatorKind


@TensorProduct._kind_dispatcher.register(_KetKind, _KetKind)
def find_ket_kind(e1, e2):
    return KetKind


@TensorProduct._kind_dispatcher.register(_BraKind, _BraKind)
def find_bra_kind(e1, e2):
    return BraKind
<|endoftext|>"""Grover's algorithm and helper functions.

Todo:

* W gate construction (or perhaps -W gate based on Mermin's book)
* Generalize the algorithm for an unknown function that returns 1 on multiple
  qubit states, not just one.
* Implement _represent_ZGate in OracleGate
"""

from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit

__all__ = [
    'OracleGate',
    'WGate',
    'superposition_basis',
    'grover_iteration',
    'apply_grover'
]


def superposition_basis(nqubits):
    """Creates an equal superposition of the computational basis.

    Parameters
    ==========

    nqubits : int
        The number of qubits.

    Returns
    =======

    state : Qubit
        An equal superposition of the computational basis with nqubits.

    Examples
    ========

    Create an equal superposition of 2 qubits::

        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> superposition_basis(2)
        |0>/2 + |1>/2 + |2>/2 + |3>/2
    """

    amp = 1/sqrt(2**nqubits)
    return sum(amp*IntQubit(n, nqubits=nqubits) for n in range(2**nqubits))

class OracleGateFunction(Atom):
    """Wrapper for python functions used in `OracleGate`s"""

    def __new__(cls, function):
        if not callable(function):
            raise TypeError('Callable expected, got: %r' % function)
        obj = Atom.__new__(cls)
        obj.function = function
        return obj

    def _hashable_content(self):
        return type(self), self.function

    def __call__(self, *args):
        return self.function(*args)


class OracleGate(Gate):
    """A black box gate.

    The gate marks the desired qubits of an unknown function by flipping
    the sign of the qubits.  The unknown function returns true when it
    finds its desired qubits and false otherwise.

    Parameters
    ==========

    qubits : int
        Number of qubits.

    oracle : callable
        A callable function that returns a boolean on a computational basis.

    Examples
    ========

    Apply an Oracle gate that flips the sign of ``|2>`` on different qubits::

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(2, f)
        >>> qapply(v*IntQubit(2))
        -|2>
        >>> qapply(v*IntQubit(3))
        |3>
    """

    gate_name = 'V'
    gate_name_latex = 'V'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        if len(args) != 2:
            raise QuantumError(
                'Insufficient/excessive arguments to Oracle.  Please ' +
                'supply the number of qubits and an unknown function.'
            )
        sub_args = (args[0],)
        sub_args = UnitaryOperator._eval_args(sub_args)
        if not sub_args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % sub_args[0])

        function = args[1]
        if not isinstance(function, OracleGateFunction):
            function = OracleGateFunction(function)

        return (sub_args[0], function)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**args[0]

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def search_function(self):
        """The unknown function that helps find the sought after qubits."""
        return self.label[1]

    @property
    def targets(self):
        """A tuple of target qubits."""
        return sympify(tuple(range(self.args[0])))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this operator to a Qubit subclass.

        Parameters
        ==========

        qubits : Qubit
            The qubit subclass to apply this operator to.

        Returns
        =======

        state : Expr
            The resulting quantum state.
        """
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'OracleGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )
        # If function returns 1 on qubits
            # return the negative of the qubits (flip the sign)
        if self.search_function(qubits):
            return -qubits
        else:
            return qubits

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_ZGate(self, basis, **options):
        """
        Represent the OracleGate in the computational basis.
        """
        nbasis = 2**self.nqubits  # compute it only once
        matrixOracle = eye(nbasis)
        # Flip the sign given the output of the oracle function
        for i in range(nbasis):
            if self.search_function(IntQubit(i, nqubits=self.nqubits)):
                matrixOracle[i, i] = NegativeOne()
        return matrixOracle


class WGate(Gate):
    """General n qubit W Gate in Grover's algorithm.

    The gate performs the operation ``2|phi><phi| - 1`` on some qubits.
    ``|phi> = (tensor product of n Hadamards)*(|0> with n qubits)``

    Parameters
    ==========

    nqubits : int
        The number of qubits to operate on

    """

    gate_name = 'W'
    gate_name_latex = 'W'

    @classmethod
    def _eval_args(cls, args):
        if len(args) != 1:
            raise QuantumError(
                'Insufficient/excessive arguments to W gate.  Please ' +
                'supply the number of qubits to operate on.'
            )
        args = UnitaryOperator._eval_args(args)
        if not args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % args[0])
        return args

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        return sympify(tuple(reversed(range(self.args[0]))))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """
        qubits: a set of qubits (Qubit)
        Returns: quantum object (quantum expression - QExpr)
        """
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'WGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )

        # See 'Quantum Computer Science' by David Mermin p.92 -> W|a> result
        # Return (2/(sqrt(2^n)))|phi> - |a> where |a> is the current basis
        # state and phi is the superposition of basis states (see function
        # create_computational_basis above)
        basis_states = superposition_basis(self.nqubits)
        change_to_basis = (2/sqrt(2**self.nqubits))*basis_states
        return change_to_basis - qubits


def grover_iteration(qstate, oracle):
    """Applies one application of the Oracle and W Gate, WV.

    Parameters
    ==========

    qstate : Qubit
        A superposition of qubits.
    oracle : OracleGate
        The black box operator that flips the sign of the desired basis qubits.

    Returns
    =======

    Qubit : The qubits after applying the Oracle and W gate.

    Examples
    ========

    Perform one iteration of grover's algorithm to see a phase change::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> from sympy.physics.quantum.grover import grover_iteration
        >>> numqubits = 2
        >>> basis_states = superposition_basis(numqubits)
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(numqubits, f)
        >>> qapply(grover_iteration(basis_states, v))
        |2>

    """
    wgate = WGate(oracle.nqubits)
    return wgate*oracle*qstate


def apply_grover(oracle, nqubits, iterations=None):
    """Applies grover's algorithm.

    Parameters
    ==========

    oracle : callable
        The unknown callable function that returns true when applied to the
        desired qubits and false otherwise.

    Returns
    =======

    state : Expr
        The resulting state after Grover's algorithm has been iterated.

    Examples
    ========

    Apply grover's algorithm to an even superposition of 2 qubits::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import apply_grover
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> qapply(apply_grover(f, 2))
        |2>

    """
    if nqubits <= 0:
        raise QuantumError(
            'Grover\'s algorithm needs nqubits > 0, received %r qubits'
            % nqubits
        )
    if iterations is None:
        iterations = floor(sqrt(2**nqubits)*(pi/4))

    v = OracleGate(nqubits, oracle)
    iterated = superposition_basis(nqubits)
    for iter in range(iterations):
        iterated = grover_iteration(iterated, v)
        iterated = qapply(iterated)

    return iterated
<|endoftext|>"""Matplotlib based plotting of quantum circuits.

Todo:

* Optimize printing of large circuits.
* Get this to work with single gates.
* Do a better job checking the form of circuits to make sure it is a Mul of
  Gates.
* Get multi-target gates plotting.
* Get initial and final states to plot.
* Get measurements to plot. Might need to rethink measurement as a gate
  issue.
* Get scale and figsize to be handled in a better way.
* Write some tests/examples!
"""

from __future__ import annotations

from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS


__all__ = [
    'CircuitPlot',
    'circuit_plot',
    'labeller',
    'Mz',
    'Mx',
    'CreateOneQubitGate',
    'CreateCGate',
]

np = import_module('numpy')
matplotlib = import_module(
    'matplotlib', import_kwargs={'fromlist': ['pyplot']},
    catch=(RuntimeError,))  # This is raised in environments that have no display.

if np and matplotlib:
    pyplot = matplotlib.pyplot
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle

#from matplotlib import rc
#rc('text',usetex=True)

class CircuitPlot:
    """A class for managing a circuit plot."""

    scale = 1.0
    fontsize = 20.0
    linewidth = 1.0
    control_radius = 0.05
    not_radius = 0.15
    swap_delta = 0.05
    labels: list[str] = []
    inits: dict[str, str] = {}
    label_buffer = 0.5

    def __init__(self, c, nqubits, **kwargs):
        if not np or not matplotlib:
            raise ImportError('numpy or matplotlib not available.')
        self.circuit = c
        self.ngates = len(self.circuit.args)
        self.nqubits = nqubits
        self.update(kwargs)
        self._create_grid()
        self._create_figure()
        self._plot_wires()
        self._plot_gates()
        self._finish()

    def update(self, kwargs):
        """Load the kwargs into the instance dict."""
        self.__dict__.update(kwargs)

    def _create_grid(self):
        """Create the grid of wires."""
        scale = self.scale
        wire_grid = np.arange(0.0, self.nqubits*scale, scale, dtype=float)
        gate_grid = np.arange(0.0, self.ngates*scale, scale, dtype=float)
        self._wire_grid = wire_grid
        self._gate_grid = gate_grid

    def _create_figure(self):
        """Create the main matplotlib figure."""
        self._figure = pyplot.figure(
            figsize=(self.ngates*self.scale, self.nqubits*self.scale),
            facecolor='w',
            edgecolor='w'
        )
        ax = self._figure.add_subplot(
            1, 1, 1,
            frameon=True
        )
        ax.set_axis_off()
        offset = 0.5*self.scale
        ax.set_xlim(self._gate_grid[0] - offset, self._gate_grid[-1] + offset)
        ax.set_ylim(self._wire_grid[0] - offset, self._wire_grid[-1] + offset)
        ax.set_aspect('equal')
        self._axes = ax

    def _plot_wires(self):
        """Plot the wires of the circuit diagram."""
        xstart = self._gate_grid[0]
        xstop = self._gate_grid[-1]
        xdata = (xstart - self.scale, xstop + self.scale)
        for i in range(self.nqubits):
            ydata = (self._wire_grid[i], self._wire_grid[i])
            line = Line2D(
                xdata, ydata,
                color='k',
                lw=self.linewidth
            )
            self._axes.add_line(line)
            if self.labels:
                init_label_buffer = 0
                if self.inits.get(self.labels[i]): init_label_buffer = 0.25
                self._axes.text(
                    xdata[0]-self.label_buffer-init_label_buffer,ydata[0],
                    render_label(self.labels[i],self.inits),
                    size=self.fontsize,
                    color='k',ha='center',va='center')
        self._plot_measured_wires()

    def _plot_measured_wires(self):
        ismeasured = self._measurements()
        xstop = self._gate_grid[-1]
        dy = 0.04 # amount to shift wires when doubled
        # Plot doubled wires after they are measured
        for im in ismeasured:
            xdata = (self._gate_grid[ismeasured[im]],xstop+self.scale)
            ydata = (self._wire_grid[im]+dy,self._wire_grid[im]+dy)
            line = Line2D(
                xdata, ydata,
                color='k',
                lw=self.linewidth
            )
            self._axes.add_line(line)
        # Also double any controlled lines off these wires
        for i,g in enumerate(self._gates()):
            if isinstance(g, (CGate, CGateS)):
                wires = g.controls + g.targets
                for wire in wires:
                    if wire in ismeasured and \
                           self._gate_grid[i] > self._gate_grid[ismeasured[wire]]:
                        ydata = min(wires), max(wires)
                        xdata = self._gate_grid[i]-dy, self._gate_grid[i]-dy
                        line = Line2D(
                            xdata, ydata,
                            color='k',
                            lw=self.linewidth
                            )
                        self._axes.add_line(line)
    def _gates(self):
        """Create a list of all gates in the circuit plot."""
        gates = []
        if isinstance(self.circuit, Mul):
            for g in reversed(self.circuit.args):
                if isinstance(g, Gate):
                    gates.append(g)
        elif isinstance(self.circuit, Gate):
            gates.append(self.circuit)
        return gates

    def _plot_gates(self):
        """Iterate through the gates and plot each of them."""
        for i, gate in enumerate(self._gates()):
            gate.plot_gate(self, i)

    def _measurements(self):
        """Return a dict ``{i:j}`` where i is the index of the wire that has
        been measured, and j is the gate where the wire is measured.
        """
        ismeasured = {}
        for i,g in enumerate(self._gates()):
            if getattr(g,'measurement',False):
                for target in g.targets:
                    if target in ismeasured:
                        if ismeasured[target] > i:
                            ismeasured[target] = i
                    else:
                        ismeasured[target] = i
        return ismeasured

    def _finish(self):
        # Disable clipping to make panning work well for large circuits.
        for o in self._figure.findobj():
            o.set_clip_on(False)

    def one_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a single qubit gate."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        self._axes.text(
            x, y, t,
            color='k',
            ha='center',
            va='center',
            bbox={"ec": 'k', "fc": 'w', "fill": True, "lw": self.linewidth},
            size=self.fontsize
        )

    def two_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a two qubit gate. Does not work yet.
        """
        # x = self._gate_grid[gate_idx]
        # y = self._wire_grid[wire_idx]+0.5
        print(self._gate_grid)
        print(self._wire_grid)
        # unused:
        # obj = self._axes.text(
        #     x, y, t,
        #     color='k',
        #     ha='center',
        #     va='center',
        #     bbox=dict(ec='k', fc='w', fill=True, lw=self.linewidth),
        #     size=self.fontsize
        # )

    def control_line(self, gate_idx, min_wire, max_wire):
        """Draw a vertical control line."""
        xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
        ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
        line = Line2D(
            xdata, ydata,
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(line)

    def control_point(self, gate_idx, wire_idx):
        """Draw a control point."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.control_radius
        c = Circle(
            (x, y),
            radius*self.scale,
            ec='k',
            fc='k',
            fill=True,
            lw=self.linewidth
        )
        self._axes.add_patch(c)

    def not_point(self, gate_idx, wire_idx):
        """Draw a NOT gates as the circle with plus in the middle."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.not_radius
        c = Circle(
            (x, y),
            radius,
            ec='k',
            fc='w',
            fill=False,
            lw=self.linewidth
        )
        self._axes.add_patch(c)
        l = Line2D(
            (x, x), (y - radius, y + radius),
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(l)

    def swap_point(self, gate_idx, wire_idx):
        """Draw a swap point as a cross."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        d = self.swap_delta
        l1 = Line2D(
            (x - d, x + d),
            (y - d, y + d),
            color='k',
            lw=self.linewidth
        )
        l2 = Line2D(
            (x - d, x + d),
            (y + d, y - d),
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(l1)
        self._axes.add_line(l2)

def circuit_plot(c, nqubits, **kwargs):
    """Draw the circuit diagram for the circuit with nqubits.

    Parameters
    ==========

    c : circuit
        The circuit to plot. Should be a product of Gate instances.
    nqubits : int
        The number of qubits to include in the circuit. Must be at least
        as big as the largest ``min_qubits`` of the gates.
    """
    return CircuitPlot(c, nqubits, **kwargs)

def render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    >>> from sympy.physics.quantum.circuitplot import render_label
    >>> render_label('q0')
    '$\\\\left|q0\\\\right\\\\rangle$'
    >>> render_label('q0', {'q0':'0'})
    '$\\\\left|q0\\\\right\\\\rangle=\\\\left|0\\\\right\\\\rangle$'
    """
    init = inits.get(label)
    if init:
        return r'$\left|%s\right\rangle=\left|%s\right\rangle$' % (label, init)
    return r'$\left|%s\right\rangle$' % label

def labeller(n, symbol='q'):
    """Autogenerate labels for wires of quantum circuits.

    Parameters
    ==========

    n : int
        number of qubits in the circuit.
    symbol : string
        A character string to precede all gate labels. E.g. 'q_0', 'q_1', etc.

    >>> from sympy.physics.quantum.circuitplot import labeller
    >>> labeller(2)
    ['q_1', 'q_0']
    >>> labeller(3,'j')
    ['j_2', 'j_1', 'j_0']
    """
    return ['%s_%d' % (symbol,n-i-1) for i in range(n)]

class Mz(OneQubitGate):
    """Mock-up of a z measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mz'
    gate_name_latex='M_z'

class Mx(OneQubitGate):
    """Mock-up of an x measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mx'
    gate_name_latex='M_x'

class CreateOneQubitGate(type):
    def __new__(mcl, name, latexname=None):
        if not latexname:
            latexname = name
        return type(name + "Gate", (OneQubitGate,),
            {'gate_name': name, 'gate_name_latex': latexname})

def CreateCGate(name, latexname=None):
    """Use a lexical closure to make a controlled gate.
    """
    if not latexname:
        latexname = name
    onequbitgate = CreateOneQubitGate(name, latexname)
    def ControlledGate(ctrls,target):
        return CGate(tuple(ctrls),onequbitgate(target))
    return ControlledGate
<|endoftext|>"""Qubits for quantum computing.

Todo:
* Finish implementing measurement logic. This should include POVM.
* Update docstrings.
* Update tests.
"""


import math

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State

from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray, scipy_sparse_matrix
)
from mpmath.libmp.libintmath import bitcount

__all__ = [
    'Qubit',
    'QubitBra',
    'IntQubit',
    'IntQubitBra',
    'qubit_to_matrix',
    'matrix_to_qubit',
    'matrix_to_density',
    'measure_all',
    'measure_partial',
    'measure_partial_oneshot',
    'measure_all_oneshot'
]

#-----------------------------------------------------------------------------
# Qubit Classes
#-----------------------------------------------------------------------------


class QubitState(State):
    """Base class for Qubit and QubitBra."""

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # If we are passed a QubitState or subclass, we just take its qubit
        # values directly.
        if len(args) == 1 and isinstance(args[0], QubitState):
            return args[0].qubit_values

        # Turn strings into tuple of strings
        if len(args) == 1 and isinstance(args[0], str):
            args = tuple( S.Zero if qb == "0" else S.One for qb in args[0])
        else:
            args = tuple( S.Zero if qb == "0" else S.One if qb == "1" else qb for qb in args)
        args = tuple(_sympify(arg) for arg in args)

        # Validate input (must have 0 or 1 input)
        for element in args:
            if element not in (S.Zero, S.One):
                raise ValueError(
                    "Qubit values must be 0 or 1, got: %r" % element)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        return ComplexSpace(2)**len(args)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def dimension(self):
        """The number of Qubits in the state."""
        return len(self.qubit_values)

    @property
    def nqubits(self):
        return self.dimension

    @property
    def qubit_values(self):
        """Returns the values of the qubits as a tuple."""
        return self.label

    #-------------------------------------------------------------------------
    # Special methods
    #-------------------------------------------------------------------------

    def __len__(self):
        return self.dimension

    def __getitem__(self, bit):
        return self.qubit_values[int(self.dimension - bit - 1)]

    #-------------------------------------------------------------------------
    # Utility methods
    #-------------------------------------------------------------------------

    def flip(self, *bits):
        """Flip the bit(s) given."""
        newargs = list(self.qubit_values)
        for i in bits:
            bit = int(self.dimension - i - 1)
            if newargs[bit] == 1:
                newargs[bit] = 0
            else:
                newargs[bit] = 1
        return self.__class__(*tuple(newargs))


class Qubit(QubitState, Ket):
    """A multi-qubit ket in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    Examples
    ========

    Create a qubit in a couple of different ways and look at their attributes:

        >>> from sympy.physics.quantum.qubit import Qubit
        >>> Qubit(0,0,0)
        |000>
        >>> q = Qubit('0101')
        >>> q
        |0101>

        >>> q.nqubits
        4
        >>> len(q)
        4
        >>> q.dimension
        4
        >>> q.qubit_values
        (0, 1, 0, 1)

    We can flip the value of an individual qubit:

        >>> q.flip(1)
        |0111>

    We can take the dagger of a Qubit to get a bra:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> Dagger(q)
        <0101|
        >>> type(Dagger(q))
        <class 'sympy.physics.quantum.qubit.QubitBra'>

    Inner products work as expected:

        >>> ip = Dagger(q)*q
        >>> ip
        <0101|0101>
        >>> ip.doit()
        1
    """

    @classmethod
    def dual_class(self):
        return QubitBra

    def _eval_innerproduct_QubitBra(self, bra, **hints):
        if self.label == bra.label:
            return S.One
        else:
            return S.Zero

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """Represent this qubits in the computational basis (ZGate).
        """
        _format = options.get('format', 'sympy')
        n = 1
        definite_state = 0
        for it in reversed(self.qubit_values):
            definite_state += n*it
            n = n*2
        result = [0]*(2**self.dimension)
        result[int(definite_state)] = 1
        if _format == 'sympy':
            return Matrix(result)
        elif _format == 'numpy':
            import numpy as np
            return np.array(result, dtype='complex').transpose()
        elif _format == 'scipy.sparse':
            from scipy import sparse
            return sparse.csr_matrix(result, dtype='complex').transpose()

    def _eval_trace(self, bra, **kwargs):
        indices = kwargs.get('indices', [])

        #sort index list to begin trace from most-significant
        #qubit
        sorted_idx = list(indices)
        if len(sorted_idx) == 0:
            sorted_idx = list(range(0, self.nqubits))
        sorted_idx.sort()

        #trace out for each of index
        new_mat = self*bra
        for i in range(len(sorted_idx) - 1, -1, -1):
            # start from tracing out from leftmost qubit
            new_mat = self._reduced_density(new_mat, int(sorted_idx[i]))

        if (len(sorted_idx) == self.nqubits):
            #in case full trace was requested
            return new_mat[0]
        else:
            return matrix_to_density(new_mat)

    def _reduced_density(self, matrix, qubit, **options):
        """Compute the reduced density matrix by tracing out one qubit.
           The qubit argument should be of type Python int, since it is used
           in bit operations
        """
        def find_index_that_is_projected(j, k, qubit):
            bit_mask = 2**qubit - 1
            return ((j >> qubit) << (1 + qubit)) + (j & bit_mask) + (k << qubit)

        old_matrix = represent(matrix, **options)
        old_size = old_matrix.cols
        #we expect the old_size to be even
        new_size = old_size//2
        new_matrix = Matrix().zeros(new_size)

        for i in range(new_size):
            for j in range(new_size):
                for k in range(2):
                    col = find_index_that_is_projected(j, k, qubit)
                    row = find_index_that_is_projected(i, k, qubit)
                    new_matrix[i, j] += old_matrix[row, col]

        return new_matrix


class QubitBra(QubitState, Bra):
    """A multi-qubit bra in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    See also
    ========

    Qubit: Examples using qubits

    """
    @classmethod
    def dual_class(self):
        return Qubit


class IntQubitState(QubitState):
    """A base class for qubits that work with binary representations."""

    @classmethod
    def _eval_args(cls, args, nqubits=None):
        # The case of a QubitState instance
        if len(args) == 1 and isinstance(args[0], QubitState):
            return QubitState._eval_args(args)
        # otherwise, args should be integer
        elif not all(isinstance(a, (int, Integer)) for a in args):
            raise ValueError('values must be integers, got (%s)' % (tuple(type(a) for a in args),))
        # use nqubits if specified
        if nqubits is not None:
            if not isinstance(nqubits, (int, Integer)):
                raise ValueError('nqubits must be an integer, got (%s)' % type(nqubits))
            if len(args) != 1:
                raise ValueError(
                    'too many positional arguments (%s). should be (number, nqubits=n)' % (args,))
            return cls._eval_args_with_nqubits(args[0], nqubits)
        # For a single argument, we construct the binary representation of
        # that integer with the minimal number of bits.
        if len(args) == 1 and args[0] > 1:
            #rvalues is the minimum number of bits needed to express the number
            rvalues = reversed(range(bitcount(abs(args[0]))))
            qubit_values = [(args[0] >> i) & 1 for i in rvalues]
            return QubitState._eval_args(qubit_values)
        # For two numbers, the second number is the number of bits
        # on which it is expressed, so IntQubit(0,5) == |00000>.
        elif len(args) == 2 and args[1] > 1:
            return cls._eval_args_with_nqubits(args[0], args[1])
        else:
            return QubitState._eval_args(args)

    @classmethod
    def _eval_args_with_nqubits(cls, number, nqubits):
        need = bitcount(abs(number))
        if nqubits < need:
            raise ValueError(
                'cannot represent %s with %s bits' % (number, nqubits))
        qubit_values = [(number >> i) & 1 for i in reversed(range(nqubits))]
        return QubitState._eval_args(qubit_values)

    def as_int(self):
        """Return the numerical value of the qubit."""
        number = 0
        n = 1
        for i in reversed(self.qubit_values):
            number += n*i
            n = n << 1
        return number

    def _print_label(self, printer, *args):
        return str(self.as_int())

    def _print_label_pretty(self, printer, *args):
        label = self._print_label(printer, *args)
        return prettyForm(label)

    _print_label_repr = _print_label
    _print_label_latex = _print_label


class IntQubit(IntQubitState, Qubit):
    """A qubit ket that store integers as binary numbers in qubit values.

    The differences between this class and ``Qubit`` are:

    * The form of the constructor.
    * The qubit values are printed as their corresponding integer, rather
      than the raw qubit values. The internal storage format of the qubit
      values in the same as ``Qubit``.

    Parameters
    ==========

    values : int, tuple
        If a single argument, the integer we want to represent in the qubit
        values. This integer will be represented using the fewest possible
        number of qubits.
        If a pair of integers and the second value is more than one, the first
        integer gives the integer to represent in binary form and the second
        integer gives the number of qubits to use.
        List of zeros and ones is also accepted to generate qubit by bit pattern.

    nqubits : int
        The integer that represents the number of qubits.
        This number should be passed with keyword ``nqubits=N``.
        You can use this in order to avoid ambiguity of Qubit-style tuple of bits.
        Please see the example below for more details.

    Examples
    ========

    Create a qubit for the integer 5:

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qubit import Qubit
        >>> q = IntQubit(5)
        >>> q
        |5>

    We can also create an ``IntQubit`` by passing a ``Qubit`` instance.

        >>> q = IntQubit(Qubit('101'))
        >>> q
        |5>
        >>> q.as_int()
        5
        >>> q.nqubits
        3
        >>> q.qubit_values
        (1, 0, 1)

    We can go back to the regular qubit form.

        >>> Qubit(q)
        |101>

    Please note that ``IntQubit`` also accepts a ``Qubit``-style list of bits.
    So, the code below yields qubits 3, not a single bit ``1``.

        >>> IntQubit(1, 1)
        |3>

    To avoid ambiguity, use ``nqubits`` parameter.
    Use of this keyword is recommended especially when you provide the values by variables.

        >>> IntQubit(1, nqubits=1)
        |1>
        >>> a = 1
        >>> IntQubit(a, nqubits=1)
        |1>
    """
    @classmethod
    def dual_class(self):
        return IntQubitBra

    def _eval_innerproduct_IntQubitBra(self, bra, **hints):
        return Qubit._eval_innerproduct_QubitBra(self, bra)

class IntQubitBra(IntQubitState, QubitBra):
    """A qubit bra that store integers as binary numbers in qubit values."""

    @classmethod
    def dual_class(self):
        return IntQubit


#-----------------------------------------------------------------------------
# Qubit <---> Matrix conversion functions
#-----------------------------------------------------------------------------


def matrix_to_qubit(matrix):
    """Convert from the matrix repr. to a sum of Qubit objects.

    Parameters
    ----------
    matrix : Matrix, numpy.matrix, scipy.sparse
        The matrix to build the Qubit representation of. This works with
        SymPy matrices, numpy matrices and scipy.sparse sparse matrices.

    Examples
    ========

    Represent a state and then go back to its qubit form:

        >>> from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
        >>> from sympy.physics.quantum.represent import represent
        >>> q = Qubit('01')
        >>> matrix_to_qubit(represent(q))
        |01>
    """
    # Determine the format based on the type of the input matrix
    format = 'sympy'
    if isinstance(matrix, numpy_ndarray):
        format = 'numpy'
    if isinstance(matrix, scipy_sparse_matrix):
        format = 'scipy.sparse'

    # Make sure it is of correct dimensions for a Qubit-matrix representation.
    # This logic should work with sympy, numpy or scipy.sparse matrices.
    if matrix.shape[0] == 1:
        mlistlen = matrix.shape[1]
        nqubits = log(mlistlen, 2)
        ket = False
        cls = QubitBra
    elif matrix.shape[1] == 1:
        mlistlen = matrix.shape[0]
        nqubits = log(mlistlen, 2)
        ket = True
        cls = Qubit
    else:
        raise QuantumError(
            'Matrix must be a row/column vector, got %r' % matrix
        )
    if not isinstance(nqubits, Integer):
        raise QuantumError('Matrix must be a row/column vector of size '
                           '2**nqubits, got: %r' % matrix)
    # Go through each item in matrix, if element is non-zero, make it into a
    # Qubit item times the element.
    result = 0
    for i in range(mlistlen):
        if ket:
            element = matrix[i, 0]
        else:
            element = matrix[0, i]
        if format in ('numpy', 'scipy.sparse'):
            element = complex(element)
        if element:
            # Form Qubit array; 0 in bit-locations where i is 0, 1 in
            # bit-locations where i is 1
            qubit_array = [int(i & (1 << x) != 0) for x in range(nqubits)]
            qubit_array.reverse()
            result = result + element*cls(*qubit_array)

    # If SymPy simplified by pulling out a constant coefficient, undo that.
    if isinstance(result, (Mul, Add, Pow)):
        result = result.expand()

    return result


def matrix_to_density(mat):
    """
    Works by finding the eigenvectors and eigenvalues of the matrix.
    We know we can decompose rho by doing:
    sum(EigenVal*|Eigenvect><Eigenvect|)
    """
    from sympy.physics.quantum.density import Density
    eigen = mat.eigenvects()
    args = [[matrix_to_qubit(Matrix(
        [vector, ])), x[0]] for x in eigen for vector in x[2] if x[0] != 0]
    if (len(args) == 0):
        return S.Zero
    else:
        return Density(*args)


def qubit_to_matrix(qubit, format='sympy'):
    """Converts an Add/Mul of Qubit objects into it's matrix representation

    This function is the inverse of ``matrix_to_qubit`` and is a shorthand
    for ``represent(qubit)``.
    """
    return represent(qubit, format=format)


#-----------------------------------------------------------------------------
# Measurement
#-----------------------------------------------------------------------------


def measure_all(qubit, format='sympy', normalize=True):
    """Perform an ensemble measurement of all qubits.

    Parameters
    ==========

    qubit : Qubit, Add
        The qubit to measure. This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_all
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_all(q)
        [(|00>, 1/4), (|01>, 1/4), (|10>, 1/4), (|11>, 1/4)]
    """
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        results = []

        if normalize:
            m = m.normalized()

        size = max(m.shape)  # Max of shape to account for bra or ket
        nqubits = int(math.log(size)/math.log(2))
        for i in range(size):
            if m[i]:
                results.append(
                    (Qubit(IntQubit(i, nqubits=nqubits)), m[i]*conjugate(m[i]))
                )
        return results
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial(qubit, bits, format='sympy', normalize=True):
    """Perform a partial ensemble measure on the specified qubits.

    Parameters
    ==========

    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_partial
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_partial(q, (0,))
        [(sqrt(2)*|00>/2 + sqrt(2)*|10>/2, 1/2), (sqrt(2)*|01>/2 + sqrt(2)*|11>/2, 1/2)]
    """
    m = qubit_to_matrix(qubit, format)

    if isinstance(bits, (SYMPY_INTS, Integer)):
        bits = (int(bits),)

    if format == 'sympy':
        if normalize:
            m = m.normalized()

        possible_outcomes = _get_possible_outcomes(m, bits)

        # Form output from function.
        output = []
        for outcome in possible_outcomes:
            # Calculate probability of finding the specified bits with
            # given values.
            prob_of_outcome = 0
            prob_of_outcome += (outcome.H*outcome)[0]

            # If the output has a chance, append it to output with found
            # probability.
            if prob_of_outcome != 0:
                if normalize:
                    next_matrix = matrix_to_qubit(outcome.normalized())
                else:
                    next_matrix = matrix_to_qubit(outcome)

                output.append((
                    next_matrix,
                    prob_of_outcome
                ))

        return output
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial_oneshot(qubit, bits, format='sympy'):
    """Perform a partial oneshot measurement on the specified qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        m = m.normalized()
        possible_outcomes = _get_possible_outcomes(m, bits)

        # Form output from function
        random_number = random.random()
        total_prob = 0
        for outcome in possible_outcomes:
            # Calculate probability of finding the specified bits
            # with given values
            total_prob += (outcome.H*outcome)[0]
            if total_prob >= random_number:
                return matrix_to_qubit(outcome.normalized())
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def _get_possible_outcomes(m, bits):
    """Get the possible states that can be produced in a measurement.

    Parameters
    ----------
    m : Matrix
        The matrix representing the state of the system.
    bits : tuple, list
        Which bits will be measured.

    Returns
    -------
    result : list
        The list of possible states which can occur given this measurement.
        These are un-normalized so we can derive the probability of finding
        this state by taking the inner product with itself
    """

    # This is filled with loads of dirty binary tricks...You have been warned

    size = max(m.shape)  # Max of shape to account for bra or ket
    nqubits = int(math.log2(size) + .1)  # Number of qubits possible

    # Make the output states and put in output_matrices, nothing in them now.
    # Each state will represent a possible outcome of the measurement
    # Thus, output_matrices[0] is the matrix which we get when all measured
    # bits return 0. and output_matrices[1] is the matrix for only the 0th
    # bit being true
    output_matrices = []
    for i in range(1 << len(bits)):
        output_matrices.append(zeros(2**nqubits, 1))

    # Bitmasks will help sort how to determine possible outcomes.
    # When the bit mask is and-ed with a matrix-index,
    # it will determine which state that index belongs to
    bit_masks = []
    for bit in bits:
        bit_masks.append(1 << bit)

    # Make possible outcome states
    for i in range(2**nqubits):
        trueness = 0  # This tells us to which output_matrix this value belongs
        # Find trueness
        for j in range(len(bit_masks)):
            if i & bit_masks[j]:
                trueness += j + 1
        # Put the value in the correct output matrix
        output_matrices[trueness][i] = m[i]
    return output_matrices


def measure_all_oneshot(qubit, format='sympy'):
    """Perform a oneshot ensemble measurement on all qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    m = qubit_to_matrix(qubit)

    if format == 'sympy':
        m = m.normalized()
        random_number = random.random()
        total = 0
        result = 0
        for i in m:
            total += i*i.conjugate()
            if total > random_number:
                break
            result += 1
        return Qubit(IntQubit(result, nqubits=int(math.log2(max(m.shape)) + .1)))
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )
<|endoftext|>"""Quantum mechanical operators.

TODO:

* Fix early 0 in apply_operators.
* Debug and test apply_operators.
* Get cse working with classes in this file.
* Doctests and documentation of special methods for InnerProduct, Commutator,
  AntiCommutator, represent, apply_operators.
"""
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, expand)
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import OperatorKind
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
from sympy.utilities.exceptions import sympy_deprecation_warning



__all__ = [
    'Operator',
    'HermitianOperator',
    'UnitaryOperator',
    'IdentityOperator',
    'OuterProduct',
    'DifferentialOperator'
]

#-----------------------------------------------------------------------------
# Operators and outer products
#-----------------------------------------------------------------------------


class Operator(QExpr):
    """Base class for non-commuting quantum operators.

    An operator maps between quantum states [1]_. In quantum mechanics,
    observables (including, but not limited to, measured physical values) are
    represented as Hermitian operators [2]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    Create an operator and examine its attributes::

        >>> from sympy.physics.quantum import Operator
        >>> from sympy import I
        >>> A = Operator('A')
        >>> A
        A
        >>> A.hilbert_space
        H
        >>> A.label
        (A,)
        >>> A.is_commutative
        False

    Create another operator and do some arithmetic operations::

        >>> B = Operator('B')
        >>> C = 2*A*A + I*B
        >>> C
        2*A**2 + I*B

    Operators do not commute::

        >>> A.is_commutative
        False
        >>> B.is_commutative
        False
        >>> A*B == B*A
        False

    Polymonials of operators respect the commutation properties::

        >>> e = (A+B)**3
        >>> e.expand()
        A*B*A + A*B**2 + A**2*B + A**3 + B*A*B + B*A**2 + B**2*A + B**3

    Operator inverses are handle symbolically::

        >>> A.inv()
        A**(-1)
        >>> A*A.inv()
        1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Operator_%28physics%29
    .. [2] https://en.wikipedia.org/wiki/Observable
    """
    is_hermitian: Optional[bool] = None
    is_unitary: Optional[bool] = None
    @classmethod
    def default_args(self):
        return ("O",)

    kind = OperatorKind

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    _label_separator = ','

    def _print_operator_name(self, printer, *args):
        return self.__class__.__name__

    _print_operator_name_latex = _print_operator_name

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm(self.__class__.__name__)

    def _print_contents(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label(printer, *args)
        else:
            return '%s(%s)' % (
                self._print_operator_name(printer, *args),
                self._print_label(printer, *args)
            )

    def _print_contents_pretty(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label_pretty(printer, *args)
        else:
            pform = self._print_operator_name_pretty(printer, *args)
            label_pform = self._print_label_pretty(printer, *args)
            label_pform = prettyForm(
                *label_pform.parens(left='(', right=')')
            )
            pform = prettyForm(*pform.right(label_pform))
            return pform

    def _print_contents_latex(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label_latex(printer, *args)
        else:
            return r'%s\left(%s\right)' % (
                self._print_operator_name_latex(printer, *args),
                self._print_label_latex(printer, *args)
            )

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_commutator(self, other, **options):
        """Evaluate [self, other] if known, return None if not known."""
        return dispatch_method(self, '_eval_commutator', other, **options)

    def _eval_anticommutator(self, other, **options):
        """Evaluate [self, other] if known."""
        return dispatch_method(self, '_eval_anticommutator', other, **options)

    #-------------------------------------------------------------------------
    # Operator application
    #-------------------------------------------------------------------------

    def _apply_operator(self, ket, **options):
        return dispatch_method(self, '_apply_operator', ket, **options)

    def _apply_from_right_to(self, bra, **options):
        return None

    def matrix_element(self, *args):
        raise NotImplementedError('matrix_elements is not defined')

    def inverse(self):
        return self._eval_inverse()

    inv = inverse

    def _eval_inverse(self):
        return self**(-1)


class HermitianOperator(Operator):
    """A Hermitian operator that satisfies H == Dagger(H).

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, HermitianOperator
    >>> H = HermitianOperator('H')
    >>> Dagger(H)
    H
    """

    is_hermitian = True

    def _eval_inverse(self):
        if isinstance(self, UnitaryOperator):
            return self
        else:
            return Operator._eval_inverse(self)

    def _eval_power(self, exp):
        if isinstance(self, UnitaryOperator):
            # so all eigenvalues of self are 1 or -1
            if exp.is_even:
                from sympy.core.singleton import S
                return S.One # is identity, see Issue 24153.
            elif exp.is_odd:
                return self
        # No simplification in all other cases
        return Operator._eval_power(self, exp)


class UnitaryOperator(Operator):
    """A unitary operator that satisfies U*Dagger(U) == 1.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, UnitaryOperator
    >>> U = UnitaryOperator('U')
    >>> U*Dagger(U)
    1
    """
    is_unitary = True
    def _eval_adjoint(self):
        return self._eval_inverse()


class IdentityOperator(Operator):
    """An identity operator I that satisfies op * I == I * op == op for any
    operator op.

    .. deprecated:: 1.14.
        Use the scalar S.One instead as the multiplicative identity for
        operators and states.

    Parameters
    ==========

    N : Integer
        Optional parameter that specifies the dimension of the Hilbert space
        of operator. This is used when generating a matrix representation.

    Examples
    ========

    >>> from sympy.physics.quantum import IdentityOperator
    >>> IdentityOperator() # doctest: +SKIP
    I
    """
    is_hermitian = True
    is_unitary = True
    @property
    def dimension(self):
        return self.N

    @classmethod
    def default_args(self):
        return (oo,)

    def __init__(self, *args, **hints):
        sympy_deprecation_warning(
            """
            IdentityOperator has been deprecated. In the future, please use
            S.One as the identity for quantum operators and states.
            """,
            deprecated_since_version="1.14",
            active_deprecations_target='deprecated-operator-identity',
        )
        if not len(args) in (0, 1):
            raise ValueError('0 or 1 parameters expected, got %s' % args)

        self.N = args[0] if (len(args) == 1 and args[0]) else oo

    def _eval_commutator(self, other, **hints):
        return S.Zero

    def _eval_anticommutator(self, other, **hints):
        return 2 * other

    def _eval_inverse(self):
        return self

    def _eval_adjoint(self):
        return self

    def _apply_operator(self, ket, **options):
        return ket

    def _apply_from_right_to(self, bra, **options):
        return bra

    def _eval_power(self, exp):
        return self

    def _print_contents(self, printer, *args):
        return 'I'

    def _print_contents_pretty(self, printer, *args):
        return prettyForm('I')

    def _print_contents_latex(self, printer, *args):
        return r'{\mathcal{I}}'

    def _represent_default_basis(self, **options):
        if not self.N or self.N == oo:
            raise NotImplementedError('Cannot represent infinite dimensional' +
                                      ' identity operator as a matrix')

        format = options.get('format', 'sympy')
        if format != 'sympy':
            raise NotImplementedError('Representation in format ' +
                                      '%s not implemented.' % format)

        return eye(self.N)


class OuterProduct(Operator):
    """An unevaluated outer product between a ket and bra.

    This constructs an outer product between any subclass of ``KetBase`` and
    ``BraBase`` as ``|a><b|``. An ``OuterProduct`` inherits from Operator as they act as
    operators in quantum expressions.  For reference see [1]_.

    Parameters
    ==========

    ket : KetBase
        The ket on the left side of the outer product.
    bar : BraBase
        The bra on the right side of the outer product.

    Examples
    ========

    Create a simple outer product by hand and take its dagger::

        >>> from sympy.physics.quantum import Ket, Bra, OuterProduct, Dagger

        >>> k = Ket('k')
        >>> b = Bra('b')
        >>> op = OuterProduct(k, b)
        >>> op
        |k><b|
        >>> op.hilbert_space
        H
        >>> op.ket
        |k>
        >>> op.bra
        <b|
        >>> Dagger(op)
        |b><k|

    In quantum expressions, outer products will be automatically
    identified and created::

        >>> k*b
        |k><b|

    However, the creation of inner products always has higher priority than that of
    outer products:

        >>> b*k*b
        <b|k>*<b|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Outer_product
    """
    is_commutative = False

    def __new__(cls, *args, **old_assumptions):
        from sympy.physics.quantum.state import KetBase, BraBase

        if len(args) != 2:
            raise ValueError('2 parameters expected, got %d' % len(args))

        ket_expr = expand(args[0])
        bra_expr = expand(args[1])

        if (isinstance(ket_expr, (KetBase, Mul)) and
                isinstance(bra_expr, (BraBase, Mul))):
            ket_c, kets = ket_expr.args_cnc()
            bra_c, bras = bra_expr.args_cnc()

            if len(kets) != 1 or not isinstance(kets[0], KetBase):
                raise TypeError('KetBase subclass expected'
                                ', got: %r' % Mul(*kets))

            if len(bras) != 1 or not isinstance(bras[0], BraBase):
                raise TypeError('BraBase subclass expected'
                                ', got: %r' % Mul(*bras))

            if not kets[0].dual_class() == bras[0].__class__:
                raise TypeError(
                    'ket and bra are not dual classes: %r, %r' %
                    (kets[0].__class__, bras[0].__class__)
                    )

            # TODO: make sure the hilbert spaces of the bra and ket are
            # compatible
            obj = Expr.__new__(cls, *(kets[0], bras[0]), **old_assumptions)
            obj.hilbert_space = kets[0].hilbert_space
            return Mul(*(ket_c + bra_c)) * obj

        op_terms = []
        if isinstance(ket_expr, Add) and isinstance(bra_expr, Add):
            for ket_term in ket_expr.args:
                for bra_term in bra_expr.args:
                    op_terms.append(OuterProduct(ket_term, bra_term,
                                                 **old_assumptions))
        elif isinstance(ket_expr, Add):
            for ket_term in ket_expr.args:
                op_terms.append(OuterProduct(ket_term, bra_expr,
                                             **old_assumptions))
        elif isinstance(bra_expr, Add):
            for bra_term in bra_expr.args:
                op_terms.append(OuterProduct(ket_expr, bra_term,
                                             **old_assumptions))
        else:
            raise TypeError(
                'Expected ket and bra expression, got: %r, %r' %
                (ket_expr, bra_expr)
                )

        return Add(*op_terms)

    @property
    def ket(self):
        """Return the ket on the left side of the outer product."""
        return self.args[0]

    @property
    def bra(self):
        """Return the bra on the right side of the outer product."""
        return self.args[1]

    def _eval_adjoint(self):
        return OuterProduct(Dagger(self.bra), Dagger(self.ket))

    def _sympystr(self, printer, *args):
        return printer._print(self.ket) + printer._print(self.bra)

    def _sympyrepr(self, printer, *args):
        return '%s(%s,%s)' % (self.__class__.__name__,
            printer._print(self.ket, *args), printer._print(self.bra, *args))

    def _pretty(self, printer, *args):
        pform = self.ket._pretty(printer, *args)
        return prettyForm(*pform.right(self.bra._pretty(printer, *args)))

    def _latex(self, printer, *args):
        k = printer._print(self.ket, *args)
        b = printer._print(self.bra, *args)
        return k + b

    def _represent(self, **options):
        k = self.ket._represent(**options)
        b = self.bra._represent(**options)
        return k*b

    def _eval_trace(self, **kwargs):
        # TODO if operands are tensorproducts this may be will be handled
        # differently.

        return self.ket._eval_trace(self.bra, **kwargs)


class DifferentialOperator(Operator):
    """An operator for representing the differential operator, i.e. d/dx

    It is initialized by passing two arguments. The first is an arbitrary
    expression that involves a function, such as ``Derivative(f(x), x)``. The
    second is the function (e.g. ``f(x)``) which we are to replace with the
    ``Wavefunction`` that this ``DifferentialOperator`` is applied to.

    Parameters
    ==========

    expr : Expr
           The arbitrary expression which the appropriate Wavefunction is to be
           substituted into

    func : Expr
           A function (e.g. f(x)) which is to be replaced with the appropriate
           Wavefunction when this DifferentialOperator is applied

    Examples
    ========

    You can define a completely arbitrary expression and specify where the
    Wavefunction is to be substituted

    >>> from sympy import Derivative, Function, Symbol
    >>> from sympy.physics.quantum.operator import DifferentialOperator
    >>> from sympy.physics.quantum.state import Wavefunction
    >>> from sympy.physics.quantum.qapply import qapply
    >>> f = Function('f')
    >>> x = Symbol('x')
    >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
    >>> w = Wavefunction(x**2, x)
    >>> d.function
    f(x)
    >>> d.variables
    (x,)
    >>> qapply(d*w)
    Wavefunction(2, x)

    """

    @property
    def variables(self):
        """
        Returns the variables with which the function in the specified
        arbitrary expression is evaluated

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Symbol, Function, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
        >>> d.variables
        (x,)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.variables
        (x, y)
        """

        return self.args[-1].args

    @property
    def function(self):
        """
        Returns the function which is to be replaced with the Wavefunction

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))
        >>> d.function
        f(x)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.function
        f(x, y)
        """

        return self.args[-1]

    @property
    def expr(self):
        """
        Returns the arbitrary expression which is to have the Wavefunction
        substituted into it

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))
        >>> d.expr
        Derivative(f(x), x)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.expr
        Derivative(f(x, y), x) + Derivative(f(x, y), y)
        """

        return self.args[0]

    @property
    def free_symbols(self):
        """
        Return the free symbols of the expression.
        """

        return self.expr.free_symbols

    def _apply_operator_Wavefunction(self, func, **options):
        from sympy.physics.quantum.state import Wavefunction
        var = self.variables
        wf_vars = func.args[1:]

        f = self.function
        new_expr = self.expr.subs(f, func(*var))
        new_expr = new_expr.doit()

        return Wavefunction(new_expr, *wf_vars)

    def _eval_derivative(self, symbol):
        new_expr = Derivative(self.expr, symbol)
        return DifferentialOperator(new_expr, self.args[-1])

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print(self, printer, *args):
        return '%s(%s)' % (
            self._print_operator_name(printer, *args),
            self._print_label(printer, *args)
        )

    def _print_pretty(self, printer, *args):
        pform = self._print_operator_name_pretty(printer, *args)
        label_pform = self._print_label_pretty(printer, *args)
        label_pform = prettyForm(
            *label_pform.parens(left='(', right=')')
        )
        pform = prettyForm(*pform.right(label_pform))
        return pform
<|endoftext|>"""Hermitian conjugation."""

from sympy.core import Expr, sympify
from sympy.functions.elementary.complexes import adjoint

__all__ = [
    'Dagger'
]


class Dagger(adjoint):
    """General Hermitian conjugate operation.

    Explanation
    ===========

    Take the Hermetian conjugate of an argument [1]_. For matrices this
    operation is equivalent to transpose and complex conjugate [2]_.

    Parameters
    ==========

    arg : Expr
        The SymPy expression that we want to take the dagger of.
    evaluate : bool
        Whether the resulting expression should be directly evaluated.

    Examples
    ========

    Daggering various quantum objects:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> from sympy.physics.quantum.state import Ket, Bra
        >>> from sympy.physics.quantum.operator import Operator
        >>> Dagger(Ket('psi'))
        <psi|
        >>> Dagger(Bra('phi'))
        |phi>
        >>> Dagger(Operator('A'))
        Dagger(A)

    Inner and outer products::

        >>> from sympy.physics.quantum import InnerProduct, OuterProduct
        >>> Dagger(InnerProduct(Bra('a'), Ket('b')))
        <b|a>
        >>> Dagger(OuterProduct(Ket('a'), Bra('b')))
        |b><a|

    Powers, sums and products::

        >>> A = Operator('A')
        >>> B = Operator('B')
        >>> Dagger(A*B)
        Dagger(B)*Dagger(A)
        >>> Dagger(A+B)
        Dagger(A) + Dagger(B)
        >>> Dagger(A**2)
        Dagger(A)**2

    Dagger also seamlessly handles complex numbers and matrices::

        >>> from sympy import Matrix, I
        >>> m = Matrix([[1,I],[2,I]])
        >>> m
        Matrix([
        [1, I],
        [2, I]])
        >>> Dagger(m)
        Matrix([
        [ 1,  2],
        [-I, -I]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermitian_adjoint
    .. [2] https://en.wikipedia.org/wiki/Hermitian_transpose
    """

    @property
    def kind(self):
        """Find the kind of a dagger of something (just the kind of the something)."""
        return self.args[0].kind

    def __new__(cls, arg, evaluate=True):
        if hasattr(arg, 'adjoint') and evaluate:
            return arg.adjoint()
        elif hasattr(arg, 'conjugate') and hasattr(arg, 'transpose') and evaluate:
            return arg.conjugate().transpose()
        return Expr.__new__(cls, sympify(arg))

adjoint.__name__ = "Dagger"
adjoint._sympyrepr = lambda a, b: "Dagger(%s)" % b._print(a.args[0])
<|endoftext|>"""Bosonic quantum operators."""

from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta


__all__ = [
    'BosonOp',
    'BosonFockKet',
    'BosonFockBra',
    'BosonCoherentKet',
    'BosonCoherentBra'
]


class BosonOp(Operator):
    """A bosonic operator that satisfies [a, Dagger(a)] == 1.

    Parameters
    ==========

    name : str
        A string that labels the bosonic mode.

    annihilation : bool
        A bool that indicates if the bosonic operator is an annihilation (True,
        default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, Commutator
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> a = BosonOp("a")
    >>> Commutator(a, Dagger(a)).doit()
    1
    """

    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        return ("a", True)

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)

        if len(args) == 1:
            args = (args[0], S.One)

        if len(args) == 2:
            args = (args[0], Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_commutator_BosonOp(self, other, **hints):
        if self.name == other.name:
            # [a^\dagger, a] = -1
            if not self.is_annihilation and other.is_annihilation:
                return S.NegativeOne

        elif 'independent' in hints and hints['independent']:
            # [a, b] = 0
            return S.Zero

        return None

    def _eval_commutator_FermionOp(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_BosonOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # {a, b} = 2 * a * b, because [a, b] = 0
            return 2 * self * other

        return None

    def _eval_adjoint(self):
        return BosonOp(str(self.name), not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)
        else:
            return r'{{%s}^\dagger}' % str(self.name)

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)
        else:
            return r'Dagger(%s)' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm('\N{DAGGER}')


class BosonFockKet(Ket):
    """Fock state ket for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    def _eval_innerproduct_BosonFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_BosonOp(self, op, **options):
        if op.is_annihilation:
            return sqrt(self.n) * BosonFockKet(self.n - 1)
        else:
            return sqrt(self.n + 1) * BosonFockKet(self.n + 1)


class BosonFockBra(Bra):
    """Fock state bra for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonFockKet

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()


class BosonCoherentKet(Ket):
    """Coherent state ket for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Ket.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_BosonCoherentBra(self, bra, **hints):
        if self.alpha == bra.alpha:
            return S.One
        else:
            return exp(-(abs(self.alpha)**2 + abs(bra.alpha)**2 - 2 * conjugate(bra.alpha) * self.alpha)/2)

    def _apply_from_right_to_BosonOp(self, op, **options):
        if op.is_annihilation:
            return self.alpha * self
        else:
            return None


class BosonCoherentBra(Bra):
    """Coherent state bra for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Bra.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentKet

    def _apply_operator_BosonOp(self, op, **options):
        if not op.is_annihilation:
            return self.alpha * self
        else:
            return None
<|endoftext|>from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
from sympy.utilities.misc import filldedent

__all__ = ['Point']


class Point:
    """This object represents a point in a dynamic system.

    It stores the: position, velocity, and acceleration of a point.
    The position is a vector defined as the vector distance from a parent
    point to this point.

    Parameters
    ==========

    name : string
        The display name of the Point

    Examples
    ========

    >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> P = Point('P')
    >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    >>> O.set_vel(N, u1 * N.x + u2 * N.y + u3 * N.z)
    >>> O.acc(N)
    u1'*N.x + u2'*N.y + u3'*N.z

    ``symbols()`` can be used to create multiple Points in a single step, for
    example:

    >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> N = ReferenceFrame('N')
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> A, B = symbols('A B', cls=Point)
    >>> type(A)
    <class 'sympy.physics.vector.point.Point'>
    >>> A.set_vel(N, u1 * N.x + u2 * N.y)
    >>> B.set_vel(N, u2 * N.x + u1 * N.y)
    >>> A.acc(N) - B.acc(N)
    (u1' - u2')*N.x + (-u1' + u2')*N.y

    """

    def __init__(self, name):
        """Initialization of a Point object. """
        self.name = name
        self._pos_dict = {}
        self._vel_dict = {}
        self._acc_dict = {}
        self._pdlist = [self._pos_dict, self._vel_dict, self._acc_dict]

    def __str__(self):
        return self.name

    __repr__ = __str__

    def _check_point(self, other):
        if not isinstance(other, Point):
            raise TypeError('A Point must be supplied')

    def _pdict_list(self, other, num):
        """Returns a list of points that gives the shortest path with respect
        to position, velocity, or acceleration from this point to the provided
        point.

        Parameters
        ==========
        other : Point
            A point that may be related to this point by position, velocity, or
            acceleration.
        num : integer
            0 for searching the position tree, 1 for searching the velocity
            tree, and 2 for searching the acceleration tree.

        Returns
        =======
        list of Points
            A sequence of points from self to other.

        Notes
        =====

        It is not clear if num = 1 or num = 2 actually works because the keys
        to ``_vel_dict`` and ``_acc_dict`` are :class:`ReferenceFrame` objects
        which do not have the ``_pdlist`` attribute.

        """
        outlist = [[self]]
        oldlist = [[]]
        while outlist != oldlist:
            oldlist = outlist.copy()
            for v in outlist:
                templist = v[-1]._pdlist[num].keys()
                for v2 in templist:
                    if not v.__contains__(v2):
                        littletemplist = v + [v2]
                        if not outlist.__contains__(littletemplist):
                            outlist.append(littletemplist)
        for v in oldlist:
            if v[-1] != other:
                outlist.remove(v)
        outlist.sort(key=len)
        if len(outlist) != 0:
            return outlist[0]
        raise ValueError('No Connecting Path found between ' + other.name +
                         ' and ' + self.name)

    def a1pt_theory(self, otherpoint, outframe, interframe):
        """Sets the acceleration of this point with the 1-point theory.

        The 1-point theory for point acceleration looks like this:

        ^N a^P = ^B a^P + ^N a^O + ^N alpha^B x r^OP + ^N omega^B x (^N omega^B
        x r^OP) + 2 ^N omega^B x ^B v^P

        where O is a point fixed in B, P is a point moving in B, and B is
        rotating in frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 1-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's acceleration defined in (N)
        fixedframe : ReferenceFrame
            The intermediate frame in this calculation (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> q2 = dynamicsymbols('q2')
        >>> qd = dynamicsymbols('q', 1)
        >>> q2d = dynamicsymbols('q2', 1)
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.set_ang_vel(N, 5 * B.y)
        >>> O = Point('O')
        >>> P = O.locatenew('P', q * B.x + q2 * B.y)
        >>> P.set_vel(B, qd * B.x + q2d * B.y)
        >>> O.set_vel(N, 0)
        >>> P.a1pt_theory(O, N, B)
        (-25*q + q'')*B.x + q2''*B.y - 10*q'*B.z

        """

        _check_frame(outframe)
        _check_frame(interframe)
        self._check_point(otherpoint)
        dist = self.pos_from(otherpoint)
        v = self.vel(interframe)
        a1 = otherpoint.acc(outframe)
        a2 = self.acc(interframe)
        omega = interframe.ang_vel_in(outframe)
        alpha = interframe.ang_acc_in(outframe)
        self.set_acc(outframe, a2 + 2 * (omega.cross(v)) + a1 +
                     (alpha.cross(dist)) + (omega.cross(omega.cross(dist))))
        return self.acc(outframe)

    def a2pt_theory(self, otherpoint, outframe, fixedframe):
        """Sets the acceleration of this point with the 2-point theory.

        The 2-point theory for point acceleration looks like this:

        ^N a^P = ^N a^O + ^N alpha^B x r^OP + ^N omega^B x (^N omega^B x r^OP)

        where O and P are both points fixed in frame B, which is rotating in
        frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 2-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's acceleration defined in (N)
        fixedframe : ReferenceFrame
            The frame in which both points are fixed (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> N = ReferenceFrame('N')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> O = Point('O')
        >>> P = O.locatenew('P', 10 * B.x)
        >>> O.set_vel(N, 5 * N.x)
        >>> P.a2pt_theory(O, N, B)
        - 10*q'**2*B.x + 10*q''*B.y

        """

        _check_frame(outframe)
        _check_frame(fixedframe)
        self._check_point(otherpoint)
        dist = self.pos_from(otherpoint)
        a = otherpoint.acc(outframe)
        omega = fixedframe.ang_vel_in(outframe)
        alpha = fixedframe.ang_acc_in(outframe)
        self.set_acc(outframe, a + (alpha.cross(dist)) +
                     (omega.cross(omega.cross(dist))))
        return self.acc(outframe)

    def acc(self, frame):
        """The acceleration Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which the returned acceleration vector will be defined
            in.

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_acc(N, 10 * N.x)
        >>> p1.acc(N)
        10*N.x

        """

        _check_frame(frame)
        if not (frame in self._acc_dict):
            if self.vel(frame) != 0:
                return (self._vel_dict[frame]).dt(frame)
            else:
                return Vector(0)
        return self._acc_dict[frame]

    def locatenew(self, name, value):
        """Creates a new point with a position defined from this point.

        Parameters
        ==========

        name : str
            The name for the new point
        value : Vector
            The position of the new point relative to this point

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, Point
        >>> N = ReferenceFrame('N')
        >>> P1 = Point('P1')
        >>> P2 = P1.locatenew('P2', 10 * N.x)

        """

        if not isinstance(name, str):
            raise TypeError('Must supply a valid name')
        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        p = Point(name)
        p.set_pos(self, value)
        self.set_pos(p, -value)
        return p

    def pos_from(self, otherpoint):
        """Returns a Vector distance between this Point and the other Point.

        Parameters
        ==========

        otherpoint : Point
            The otherpoint we are locating this one relative to

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')
        >>> p1.set_pos(p2, 10 * N.x)
        >>> p1.pos_from(p2)
        10*N.x

        """

        outvec = Vector(0)
        plist = self._pdict_list(otherpoint, 0)
        for i in range(len(plist) - 1):
            outvec += plist[i]._pos_dict[plist[i + 1]]
        return outvec

    def set_acc(self, frame, value):
        """Used to set the acceleration of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which this point's acceleration is defined
        value : Vector
            The vector value of this point's acceleration in the frame

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_acc(N, 10 * N.x)
        >>> p1.acc(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(frame)
        self._acc_dict.update({frame: value})

    def set_pos(self, otherpoint, value):
        """Used to set the position of this point w.r.t. another point.

        Parameters
        ==========

        otherpoint : Point
            The other point which this point's location is defined relative to
        value : Vector
            The vector which defines the location of this point

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')
        >>> p1.set_pos(p2, 10 * N.x)
        >>> p1.pos_from(p2)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        self._check_point(otherpoint)
        self._pos_dict.update({otherpoint: value})
        otherpoint._pos_dict.update({self: -value})

    def set_vel(self, frame, value):
        """Sets the velocity Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which this point's velocity is defined
        value : Vector
            The vector value of this point's velocity in the frame

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(frame)
        self._vel_dict.update({frame: value})

    def v1pt_theory(self, otherpoint, outframe, interframe):
        """Sets the velocity of this point with the 1-point theory.

        The 1-point theory for point velocity looks like this:

        ^N v^P = ^B v^P + ^N v^O + ^N omega^B x r^OP

        where O is a point fixed in B, P is a point moving in B, and B is
        rotating in frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 1-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's velocity defined in (N)
        interframe : ReferenceFrame
            The intermediate frame in this calculation (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> q2 = dynamicsymbols('q2')
        >>> qd = dynamicsymbols('q', 1)
        >>> q2d = dynamicsymbols('q2', 1)
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.set_ang_vel(N, 5 * B.y)
        >>> O = Point('O')
        >>> P = O.locatenew('P', q * B.x + q2 * B.y)
        >>> P.set_vel(B, qd * B.x + q2d * B.y)
        >>> O.set_vel(N, 0)
        >>> P.v1pt_theory(O, N, B)
        q'*B.x + q2'*B.y - 5*q*B.z

        """

        _check_frame(outframe)
        _check_frame(interframe)
        self._check_point(otherpoint)
        dist = self.pos_from(otherpoint)
        v1 = self.vel(interframe)
        v2 = otherpoint.vel(outframe)
        omega = interframe.ang_vel_in(outframe)
        self.set_vel(outframe, v1 + v2 + (omega.cross(dist)))
        return self.vel(outframe)

    def v2pt_theory(self, otherpoint, outframe, fixedframe):
        """Sets the velocity of this point with the 2-point theory.

        The 2-point theory for point velocity looks like this:

        ^N v^P = ^N v^O + ^N omega^B x r^OP

        where O and P are both points fixed in frame B, which is rotating in
        frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 2-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's velocity defined in (N)
        fixedframe : ReferenceFrame
            The frame in which both points are fixed (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> N = ReferenceFrame('N')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> O = Point('O')
        >>> P = O.locatenew('P', 10 * B.x)
        >>> O.set_vel(N, 5 * N.x)
        >>> P.v2pt_theory(O, N, B)
        5*N.x + 10*q'*B.y

        """

        _check_frame(outframe)
        _check_frame(fixedframe)
        self._check_point(otherpoint)
        dist = self.pos_from(otherpoint)
        v = otherpoint.vel(outframe)
        omega = fixedframe.ang_vel_in(outframe)
        self.set_vel(outframe, v + (omega.cross(dist)))
        return self.vel(outframe)

    def vel(self, frame):
        """The velocity Vector of this Point in the ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which the returned velocity vector will be defined in

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        Velocities will be automatically calculated if possible, otherwise a
        ``ValueError`` will be returned. If it is possible to calculate
        multiple different velocities from the relative points, the points
        defined most directly relative to this point will be used. In the case
        of inconsistent relative positions of points, incorrect velocities may
        be returned. It is up to the user to define prior relative positions
        and velocities of points in a self-consistent way.

        >>> p = Point('p')
        >>> q = dynamicsymbols('q')
        >>> p.set_vel(N, 10 * N.x)
        >>> p2 = Point('p2')
        >>> p2.set_pos(p, q*N.x)
        >>> p2.vel(N)
        (Derivative(q(t), t) + 10)*N.x

        """

        _check_frame(frame)
        if not (frame in self._vel_dict):
            valid_neighbor_found = False
            is_cyclic = False
            visited = []
            queue = [self]
            candidate_neighbor = []
            while queue:  # BFS to find nearest point
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    for neighbor, neighbor_pos in node._pos_dict.items():
                        if neighbor in visited:
                            continue
                        try:
                            # Checks if pos vector is valid
                            neighbor_pos.express(frame)
                        except ValueError:
                            continue
                        if neighbor in queue:
                            is_cyclic = True
                        try:
                            # Checks if point has its vel defined in req frame
                            neighbor_velocity = neighbor._vel_dict[frame]
                        except KeyError:
                            queue.append(neighbor)
                            continue
                        candidate_neighbor.append(neighbor)
                        if not valid_neighbor_found:
                            self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
                            valid_neighbor_found = True
            if is_cyclic:
                warn(filldedent("""
                Kinematic loops are defined among the positions of points. This
                is likely not desired and may cause errors in your calculations.
                """))
            if len(candidate_neighbor) > 1:
                warn(filldedent(f"""
                Velocity of {self.name} automatically calculated based on point
                {candidate_neighbor[0].name} but it is also possible from
                points(s): {str(candidate_neighbor[1:])}. Velocities from these
                points are not necessarily the same. This may cause errors in
                your calculations."""))
            if valid_neighbor_found:
                return self._vel_dict[frame]
            else:
                raise ValueError(filldedent(f"""
                Velocity of point {self.name} has not been defined in
                ReferenceFrame {frame.name}."""))

        return self._vel_dict[frame]

    def partial_velocity(self, frame, *gen_speeds):
        """Returns the partial velocities of the linear velocity vector of this
        point in the given frame with respect to one or more provided
        generalized speeds.

        Parameters
        ==========
        frame : ReferenceFrame
            The frame with which the velocity is defined in.
        gen_speeds : functions of time
            The generalized speeds.

        Returns
        =======
        partial_velocities : tuple of Vector
            The partial velocity vectors corresponding to the provided
            generalized speeds.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, Point
        >>> from sympy.physics.vector import dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> p = Point('p')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> p.set_vel(N, u1 * N.x + u2 * A.y)
        >>> p.partial_velocity(N, u1)
        N.x
        >>> p.partial_velocity(N, u1, u2)
        (N.x, A.y)

        """

        from sympy.physics.vector.functions import partial_velocity

        vel = self.vel(frame)
        partials = partial_velocity([vel], gen_speeds, frame)[0]

        if len(partials) == 1:
            return partials[0]
        else:
            return tuple(partials)
<|endoftext|>__all__ = [
    'CoordinateSym', 'ReferenceFrame',

    'Dyadic',

    'Vector',

    'Point',

    'cross', 'dot', 'express', 'time_derivative', 'outer',
    'kinematic_equations', 'get_motion_params', 'partial_velocity',
    'dynamicsymbols',

    'vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting',

    'curl', 'divergence', 'gradient', 'is_conservative', 'is_solenoidal',
    'scalar_potential', 'scalar_potential_difference',

]
from .frame import CoordinateSym, ReferenceFrame

from .dyadic import Dyadic

from .vector import Vector

from .point import Point

from .functions import (cross, dot, express, time_derivative, outer,
        kinematic_equations, get_motion_params, partial_velocity,
        dynamicsymbols)

from .printing import (vprint, vsstrrepr, vsprint, vpprint, vlatex,
        init_vprinting)

from .fieldfunctions import (curl, divergence, gradient, is_conservative,
        is_solenoidal, scalar_potential, scalar_potential_difference)
<|endoftext|>from sympy import (diff, expand, sin, cos, sympify, eye, zeros,
                                ImmutableMatrix as Matrix, MatrixBase)
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate

from warnings import warn

__all__ = ['CoordinateSym', 'ReferenceFrame']


class CoordinateSym(Symbol):
    """
    A coordinate symbol/base scalar associated wrt a Reference Frame.

    Ideally, users should not instantiate this class. Instances of
    this class must only be accessed through the corresponding frame
    as 'frame[index]'.

    CoordinateSyms having the same frame and index parameters are equal
    (even though they may be instantiated separately).

    Parameters
    ==========

    name : string
        The display name of the CoordinateSym

    frame : ReferenceFrame
        The reference frame this base scalar belongs to

    index : 0, 1 or 2
        The index of the dimension denoted by this coordinate variable

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, CoordinateSym
    >>> A = ReferenceFrame('A')
    >>> A[1]
    A_y
    >>> type(A[0])
    <class 'sympy.physics.vector.frame.CoordinateSym'>
    >>> a_y = CoordinateSym('a_y', A, 1)
    >>> a_y == A[1]
    True

    """

    def __new__(cls, name, frame, index):
        # We can't use the cached Symbol.__new__ because this class depends on
        # frame and index, which are not passed to Symbol.__xnew__.
        assumptions = {}
        super()._sanitize(assumptions, cls)
        obj = super().__xnew__(cls, name, **assumptions)
        _check_frame(frame)
        if index not in range(0, 3):
            raise ValueError("Invalid index specified")
        obj._id = (frame, index)
        return obj

    def __getnewargs_ex__(self):
        return (self.name, *self._id), {}

    @property
    def frame(self):
        return self._id[0]

    def __eq__(self, other):
        # Check if the other object is a CoordinateSym of the same frame and
        # same index
        if isinstance(other, CoordinateSym):
            if other._id == self._id:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return (self._id[0].__hash__(), self._id[1]).__hash__()


class ReferenceFrame:
    """A reference frame in classical mechanics.

    ReferenceFrame is a class used to represent a reference frame in classical
    mechanics. It has a standard basis of three unit vectors in the frame's
    x, y, and z directions.

    It also can have a rotation relative to a parent frame; this rotation is
    defined by a direction cosine matrix relating this frame's basis vectors to
    the parent frame's basis vectors.  It can also have an angular velocity
    vector, defined in another frame.

    """
    _count = 0

    def __init__(self, name, indices=None, latexs=None, variables=None):
        """ReferenceFrame initialization method.

        A ReferenceFrame has a set of orthonormal basis vectors, along with
        orientations relative to other ReferenceFrames and angular velocities
        relative to other ReferenceFrames.

        Parameters
        ==========

        indices : tuple of str
            Enables the reference frame's basis unit vectors to be accessed by
            Python's square bracket indexing notation using the provided three
            indice strings and alters the printing of the unit vectors to
            reflect this choice.
        latexs : tuple of str
            Alters the LaTeX printing of the reference frame's basis unit
            vectors to the provided three valid LaTeX strings.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, vlatex
        >>> N = ReferenceFrame('N')
        >>> N.x
        N.x
        >>> O = ReferenceFrame('O', indices=('1', '2', '3'))
        >>> O.x
        O['1']
        >>> O['1']
        O['1']
        >>> P = ReferenceFrame('P', latexs=('A1', 'A2', 'A3'))
        >>> vlatex(P.x)
        'A1'

        ``symbols()`` can be used to create multiple Reference Frames in one
        step, for example:

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import symbols
        >>> A, B, C = symbols('A B C', cls=ReferenceFrame)
        >>> D, E = symbols('D E', cls=ReferenceFrame, indices=('1', '2', '3'))
        >>> A[0]
        A_x
        >>> D.x
        D['1']
        >>> E.y
        E['2']
        >>> type(A) == type(D)
        True

        Unit dyads for the ReferenceFrame can be accessed through the attributes ``xx``, ``xy``, etc. For example:

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> N.yz
        (N.y|N.z)
        >>> N.zx
        (N.z|N.x)
        >>> P = ReferenceFrame('P', indices=['1', '2', '3'])
        >>> P.xx
        (P['1']|P['1'])
        >>> P.zy
        (P['3']|P['2'])

        Unit dyadic is also accessible via the ``u`` attribute:

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> N.u
        (N.x|N.x) + (N.y|N.y) + (N.z|N.z)
        >>> P = ReferenceFrame('P', indices=['1', '2', '3'])
        >>> P.u
        (P['1']|P['1']) + (P['2']|P['2']) + (P['3']|P['3'])

        """

        if not isinstance(name, str):
            raise TypeError('Need to supply a valid name')
        # The if statements below are for custom printing of basis-vectors for
        # each frame.
        # First case, when custom indices are supplied
        if indices is not None:
            if not isinstance(indices, (tuple, list)):
                raise TypeError('Supply the indices as a list')
            if len(indices) != 3:
                raise ValueError('Supply 3 indices')
            for i in indices:
                if not isinstance(i, str):
                    raise TypeError('Indices must be strings')
            self.str_vecs = [(name + '[\'' + indices[0] + '\']'),
                             (name + '[\'' + indices[1] + '\']'),
                             (name + '[\'' + indices[2] + '\']')]
            self.pretty_vecs = [(name.lower() + "_" + indices[0]),
                                (name.lower() + "_" + indices[1]),
                                (name.lower() + "_" + indices[2])]
            self.latex_vecs = [(r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                                                             indices[0])),
                               (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                                                             indices[1])),
                               (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                                                             indices[2]))]
            self.indices = indices
        # Second case, when no custom indices are supplied
        else:
            self.str_vecs = [(name + '.x'), (name + '.y'), (name + '.z')]
            self.pretty_vecs = [name.lower() + "_x",
                                name.lower() + "_y",
                                name.lower() + "_z"]
            self.latex_vecs = [(r"\mathbf{\hat{%s}_x}" % name.lower()),
                               (r"\mathbf{\hat{%s}_y}" % name.lower()),
                               (r"\mathbf{\hat{%s}_z}" % name.lower())]
            self.indices = ['x', 'y', 'z']
        # Different step, for custom latex basis vectors
        if latexs is not None:
            if not isinstance(latexs, (tuple, list)):
                raise TypeError('Supply the indices as a list')
            if len(latexs) != 3:
                raise ValueError('Supply 3 indices')
            for i in latexs:
                if not isinstance(i, str):
                    raise TypeError('Latex entries must be strings')
            self.latex_vecs = latexs
        self.name = name
        self._var_dict = {}
        # The _dcm_dict dictionary will only store the dcms of adjacent
        # parent-child relationships. The _dcm_cache dictionary will store
        # calculated dcm along with all content of _dcm_dict for faster
        # retrieval of dcms.
        self._dcm_dict = {}
        self._dcm_cache = {}
        self._ang_vel_dict = {}
        self._ang_acc_dict = {}
        self._dlist = [self._dcm_dict, self._ang_vel_dict, self._ang_acc_dict]
        self._cur = 0
        self._x = Vector([(Matrix([1, 0, 0]), self)])
        self._y = Vector([(Matrix([0, 1, 0]), self)])
        self._z = Vector([(Matrix([0, 0, 1]), self)])
        # Associate coordinate symbols wrt this frame
        if variables is not None:
            if not isinstance(variables, (tuple, list)):
                raise TypeError('Supply the variable names as a list/tuple')
            if len(variables) != 3:
                raise ValueError('Supply 3 variable names')
            for i in variables:
                if not isinstance(i, str):
                    raise TypeError('Variable names must be strings')
        else:
            variables = [name + '_x', name + '_y', name + '_z']
        self.varlist = (CoordinateSym(variables[0], self, 0),
                        CoordinateSym(variables[1], self, 1),
                        CoordinateSym(variables[2], self, 2))
        ReferenceFrame._count += 1
        self.index = ReferenceFrame._count

    def __getitem__(self, ind):
        """
        Returns basis vector for the provided index, if the index is a string.

        If the index is a number, returns the coordinate variable correspon-
        -ding to that index.
        """
        if not isinstance(ind, str):
            if ind < 3:
                return self.varlist[ind]
            else:
                raise ValueError("Invalid index provided")
        if self.indices[0] == ind:
            return self.x
        if self.indices[1] == ind:
            return self.y
        if self.indices[2] == ind:
            return self.z
        else:
            raise ValueError('Not a defined index')

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __str__(self):
        """Returns the name of the frame. """
        return self.name

    __repr__ = __str__

    def _dict_list(self, other, num):
        """Returns an inclusive list of reference frames that connect this
        reference frame to the provided reference frame.

        Parameters
        ==========
        other : ReferenceFrame
            The other reference frame to look for a connecting relationship to.
        num : integer
            ``0``, ``1``, and ``2`` will look for orientation, angular
            velocity, and angular acceleration relationships between the two
            frames, respectively.

        Returns
        =======
        list
            Inclusive list of reference frames that connect this reference
            frame to the other reference frame.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> C = ReferenceFrame('C')
        >>> D = ReferenceFrame('D')
        >>> B.orient_axis(A, A.x, 1.0)
        >>> C.orient_axis(B, B.x, 1.0)
        >>> D.orient_axis(C, C.x, 1.0)
        >>> D._dict_list(A, 0)
        [D, C, B, A]

        Raises
        ======

        ValueError
            When no path is found between the two reference frames or ``num``
            is an incorrect value.

        """

        connect_type = {0: 'orientation',
                        1: 'angular velocity',
                        2: 'angular acceleration'}

        if num not in connect_type.keys():
            raise ValueError('Valid values for num are 0, 1, or 2.')

        possible_connecting_paths = [[self]]
        oldlist = [[]]
        while possible_connecting_paths != oldlist:
            oldlist = possible_connecting_paths.copy()
            for frame_list in possible_connecting_paths:
                frames_adjacent_to_last = frame_list[-1]._dlist[num].keys()
                for adjacent_frame in frames_adjacent_to_last:
                    if adjacent_frame not in frame_list:
                        connecting_path = frame_list + [adjacent_frame]
                        if connecting_path not in possible_connecting_paths:
                            possible_connecting_paths.append(connecting_path)

        for connecting_path in oldlist:
            if connecting_path[-1] != other:
                possible_connecting_paths.remove(connecting_path)
        possible_connecting_paths.sort(key=len)

        if len(possible_connecting_paths) != 0:
            return possible_connecting_paths[0]  # selects the shortest path

        msg = 'No connecting {} path found between {} and {}.'
        raise ValueError(msg.format(connect_type[num], self.name, other.name))

    def _w_diff_dcm(self, otherframe):
        """Angular velocity from time differentiating the DCM. """
        from sympy.physics.vector.functions import dynamicsymbols
        dcm2diff = otherframe.dcm(self)
        diffed = dcm2diff.diff(dynamicsymbols._t)
        angvelmat = diffed * dcm2diff.T
        w1 = trigsimp(expand(angvelmat[7]), recursive=True)
        w2 = trigsimp(expand(angvelmat[2]), recursive=True)
        w3 = trigsimp(expand(angvelmat[3]), recursive=True)
        return Vector([(Matrix([w1, w2, w3]), otherframe)])

    def variable_map(self, otherframe):
        """
        Returns a dictionary which expresses the coordinate variables
        of this frame in terms of the variables of otherframe.

        If Vector.simp is True, returns a simplified version of the mapped
        values. Else, returns them without simplification.

        Simplification of the expressions may take time.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The other frame to map the variables to

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> A = ReferenceFrame('A')
        >>> q = dynamicsymbols('q')
        >>> B = A.orientnew('B', 'Axis', [q, A.z])
        >>> A.variable_map(B)
        {A_x: B_x*cos(q(t)) - B_y*sin(q(t)), A_y: B_x*sin(q(t)) + B_y*cos(q(t)), A_z: B_z}

        """

        _check_frame(otherframe)
        if (otherframe, Vector.simp) in self._var_dict:
            return self._var_dict[(otherframe, Vector.simp)]
        else:
            vars_matrix = self.dcm(otherframe) * Matrix(otherframe.varlist)
            mapping = {}
            for i, x in enumerate(self):
                if Vector.simp:
                    mapping[self.varlist[i]] = trigsimp(vars_matrix[i],
                                                        method='fu')
                else:
                    mapping[self.varlist[i]] = vars_matrix[i]
            self._var_dict[(otherframe, Vector.simp)] = mapping
            return mapping

    def ang_acc_in(self, otherframe):
        """Returns the angular acceleration Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ``N_alpha_B``

        which represent the angular acceleration of B in N, where B is self,
        and N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame which the angular acceleration is returned in.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """

        _check_frame(otherframe)
        if otherframe in self._ang_acc_dict:
            return self._ang_acc_dict[otherframe]
        else:
            return self.ang_vel_in(otherframe).dt(otherframe)

    def ang_vel_in(self, otherframe):
        """Returns the angular velocity Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ^N omega ^B

        which represent the angular velocity of B in N, where B is self, and
        N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame which the angular velocity is returned in.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x

        """

        _check_frame(otherframe)
        flist = self._dict_list(otherframe, 1)
        outvec = Vector(0)
        for i in range(len(flist) - 1):
            outvec += flist[i]._ang_vel_dict[flist[i + 1]]
        return outvec

    def dcm(self, otherframe):
        r"""Returns the direction cosine matrix of this reference frame
        relative to the provided reference frame.

        The returned matrix can be used to express the orthogonal unit vectors
        of this frame in terms of the orthogonal unit vectors of
        ``otherframe``.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The reference frame which the direction cosine matrix of this frame
            is formed relative to.

        Examples
        ========

        The following example rotates the reference frame A relative to N by a
        simple rotation and then calculates the direction cosine matrix of N
        relative to A.

        >>> from sympy import symbols, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> N.dcm(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The second row of the above direction cosine matrix represents the
        ``N.y`` unit vector in N expressed in A. Like so:

        >>> Ny = 0*A.x + cos(q1)*A.y - sin(q1)*A.z

        Thus, expressing ``N.y`` in A should return the same result:

        >>> N.y.express(A)
        cos(q1)*A.y - sin(q1)*A.z

        Notes
        =====

        It is important to know what form of the direction cosine matrix is
        returned. If ``B.dcm(A)`` is called, it means the "direction cosine
        matrix of B rotated relative to A". This is the matrix
        :math:`{}^B\mathbf{C}^A` shown in the following relationship:

        .. math::

           \begin{bmatrix}
             \hat{\mathbf{b}}_1 \\
             \hat{\mathbf{b}}_2 \\
             \hat{\mathbf{b}}_3
           \end{bmatrix}
           =
           {}^B\mathbf{C}^A
           \begin{bmatrix}
             \hat{\mathbf{a}}_1 \\
             \hat{\mathbf{a}}_2 \\
             \hat{\mathbf{a}}_3
           \end{bmatrix}.

        :math:`{}^B\mathbf{C}^A` is the matrix that expresses the B unit
        vectors in terms of the A unit vectors.

        """

        _check_frame(otherframe)
        # Check if the dcm wrt that frame has already been calculated
        if otherframe in self._dcm_cache:
            return self._dcm_cache[otherframe]
        flist = self._dict_list(otherframe, 0)
        outdcm = eye(3)
        for i in range(len(flist) - 1):
            outdcm = outdcm * flist[i]._dcm_dict[flist[i + 1]]
        # After calculation, store the dcm in dcm cache for faster future
        # retrieval
        self._dcm_cache[otherframe] = outdcm
        otherframe._dcm_cache[self] = outdcm.T
        return outdcm

    def _dcm(self, parent, parent_orient):
        # If parent.oreint(self) is already defined,then
        # update the _dcm_dict of parent while over write
        # all content of self._dcm_dict and self._dcm_cache
        # with new dcm relation.
        # Else update _dcm_cache and _dcm_dict of both
        # self and parent.
        frames = self._dcm_cache.keys()
        dcm_dict_del = []
        dcm_cache_del = []
        if parent in frames:
            for frame in frames:
                if frame in self._dcm_dict:
                    dcm_dict_del += [frame]
                dcm_cache_del += [frame]
            # Reset the _dcm_cache of this frame, and remove it from the
            # _dcm_caches of the frames it is linked to. Also remove it from
            # the _dcm_dict of its parent
            for frame in dcm_dict_del:
                del frame._dcm_dict[self]
            for frame in dcm_cache_del:
                del frame._dcm_cache[self]
        # Reset the _dcm_dict
            self._dcm_dict = self._dlist[0] = {}
        # Reset the _dcm_cache
            self._dcm_cache = {}

        else:
            # Check for loops and raise warning accordingly.
            visited = []
            queue = list(frames)
            cont = True  # Flag to control queue loop.
            while queue and cont:
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    neighbors = node._dcm_dict.keys()
                    for neighbor in neighbors:
                        if neighbor == parent:
                            warn('Loops are defined among the orientation of '
                                 'frames. This is likely not desired and may '
                                 'cause errors in your calculations.')
                            cont = False
                            break
                        queue.append(neighbor)

        # Add the dcm relationship to _dcm_dict
        self._dcm_dict.update({parent: parent_orient.T})
        parent._dcm_dict.update({self: parent_orient})
        # Update the dcm cache
        self._dcm_cache.update({parent: parent_orient.T})
        parent._dcm_cache.update({self: parent_orient})

    def orient_axis(self, parent, axis, angle):
        """Sets the orientation of this reference frame with respect to a
        parent reference frame by rotating through an angle about an axis fixed
        in the parent reference frame.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        axis : Vector
            Vector fixed in the parent frame about about which this frame is
            rotated. It need not be a unit vector and the rotation follows the
            right hand rule.
        angle : sympifiable
            Angle in radians by which it the frame is to be rotated.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.orient_axis(N, N.x, q1)

        The ``orient_axis()`` method generates a direction cosine matrix and
        its transpose which defines the orientation of B relative to N and vice
        versa. Once orient is called, ``dcm()`` outputs the appropriate
        direction cosine matrix:

        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])
        >>> N.dcm(B)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The following two lines show that the sense of the rotation can be
        defined by negating the vector direction or the angle. Both lines
        produce the same result.

        >>> B.orient_axis(N, -N.x, q1)
        >>> B.orient_axis(N, N.x, -q1)

        """

        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)

        if not isinstance(axis, Vector) and isinstance(angle, Vector):
            axis, angle = angle, axis

        axis = _check_vector(axis)
        theta = sympify(angle)

        if not axis.dt(parent) == 0:
            raise ValueError('Axis cannot be time-varying.')
        unit_axis = axis.express(parent).normalize()
        unit_col = unit_axis.args[0][0]
        parent_orient_axis = (
            (eye(3) - unit_col * unit_col.T) * cos(theta) +
            Matrix([[0, -unit_col[2], unit_col[1]],
                    [unit_col[2], 0, -unit_col[0]],
                    [-unit_col[1], unit_col[0], 0]]) *
            sin(theta) + unit_col * unit_col.T)

        self._dcm(parent, parent_orient_axis)

        thetad = (theta).diff(dynamicsymbols._t)
        wvec = thetad*axis.express(parent).normalize()
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_explicit(self, parent, dcm):
        """Sets the orientation of this reference frame relative to another (parent) reference frame
        using a direction cosine matrix that describes the rotation from the parent to the child.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0, cos(q1), -sin(q1)],
        ...               [0, sin(q1), cos(q1)]])
        >>> A.orient_explicit(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        **Note carefully that** ``N.dcm(B)`` **(the transpose) would be passed
        into** ``orient_explicit()`` **for** ``A.dcm(N)`` **to match**
        ``B.dcm(N)``:

        >>> A.orient_explicit(N, N.dcm(B))
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """
        _check_frame(parent)
        # amounts must be a Matrix type object
        # (e.g. sympy.matrices.dense.MutableDenseMatrix).
        if not isinstance(dcm, MatrixBase):
            raise TypeError("Amounts must be a SymPy Matrix type object.")

        self.orient_dcm(parent, dcm.T)

    def orient_dcm(self, parent, dcm):
        """Sets the orientation of this reference frame relative to another (parent) reference frame
        using a direction cosine matrix that describes the rotation from the child to the parent.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0,  cos(q1), sin(q1)],
        ...               [0, -sin(q1), cos(q1)]])
        >>> A.orient_dcm(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """

        _check_frame(parent)
        # amounts must be a Matrix type object
        # (e.g. sympy.matrices.dense.MutableDenseMatrix).
        if not isinstance(dcm, MatrixBase):
            raise TypeError("Amounts must be a SymPy Matrix type object.")

        self._dcm(parent, dcm.T)

        wvec = self._w_diff_dcm(parent)
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def _rot(self, axis, angle):
        """DCM for simple axis 1,2,or 3 rotations."""
        if axis == 1:
            return Matrix([[1, 0, 0],
                           [0, cos(angle), -sin(angle)],
                           [0, sin(angle), cos(angle)]])
        elif axis == 2:
            return Matrix([[cos(angle), 0, sin(angle)],
                           [0, 1, 0],
                           [-sin(angle), 0, cos(angle)]])
        elif axis == 3:
            return Matrix([[cos(angle), -sin(angle), 0],
                           [sin(angle), cos(angle), 0],
                           [0, 0, 1]])

    def _parse_consecutive_rotations(self, angles, rotation_order):
        """Helper for orient_body_fixed and orient_space_fixed.

        Parameters
        ==========
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations. The order can be specified by the strings
            ``'XZX'``, ``'131'``, or the integer ``131``. There are 12 unique
            valid rotation orders.

        Returns
        =======

        amounts : list
            List of sympifiables corresponding to the rotation angles.
        rot_order : list
            List of integers corresponding to the axis of rotation.
        rot_matrices : list
            List of DCM around the given axis with corresponding magnitude.

        """
        amounts = list(angles)
        for i, v in enumerate(amounts):
            if not isinstance(v, Vector):
                amounts[i] = sympify(v)

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        # make sure XYZ => 123
        rot_order = translate(str(rotation_order), 'XYZxyz', '123123')
        if rot_order not in approved_orders:
            raise TypeError('The rotation order is not a valid order.')

        rot_order = [int(r) for r in rot_order]
        if not (len(amounts) == 3 & len(rot_order) == 3):
            raise TypeError('Body orientation takes 3 values & 3 orders')
        rot_matrices = [self._rot(order, amount)
                        for (order, amount) in zip(rot_order, amounts)]
        return amounts, rot_order, rot_matrices

    def orient_body_fixed(self, parent, angles, rotation_order):
        """Rotates this reference frame relative to the parent reference frame
        by right hand rotating through three successive body fixed simple axis
        rotations. Each subsequent axis of rotation is about the "body fixed"
        unit vectors of a new intermediate reference frame. This type of
        rotation is also referred to rotating through the `Euler and Tait-Bryan
        Angles`_.

        .. _Euler and Tait-Bryan Angles: https://en.wikipedia.org/wiki/Euler_angles

        The computed angular velocity in this method is by default expressed in
        the child's frame, so it is most preferable to use ``u1 * child.x + u2 *
        child.y + u3 * child.z`` as generalized speeds.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations about each intermediate reference frames'
            unit vectors. The Euler rotation about the X, Z', X'' axes can be
            specified by the strings ``'XZX'``, ``'131'``, or the integer
            ``131``. There are 12 unique valid rotation orders (6 Euler and 6
            Tait-Bryan): zxz, xyx, yzy, zyz, xzx, yxy, xyz, yzx, zxy, xzy, zyx,
            and yxz.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1, q2, q3 = symbols('q1, q2, q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B1 = ReferenceFrame('B1')
        >>> B2 = ReferenceFrame('B2')
        >>> B3 = ReferenceFrame('B3')

        For example, a classic Euler Angle rotation can be done by:

        >>> B.orient_body_fixed(N, (q1, q2, q3), 'XYX')
        >>> B.dcm(N)
        Matrix([
        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],
        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],
        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])

        This rotates reference frame B relative to reference frame N through
        ``q1`` about ``N.x``, then rotates B again through ``q2`` about
        ``B.y``, and finally through ``q3`` about ``B.x``. It is equivalent to
        three successive ``orient_axis()`` calls:

        >>> B1.orient_axis(N, N.x, q1)
        >>> B2.orient_axis(B1, B1.y, q2)
        >>> B3.orient_axis(B2, B2.x, q3)
        >>> B3.dcm(N)
        Matrix([
        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],
        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],
        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])

        Acceptable rotation orders are of length 3, expressed in as a string
        ``'XYZ'`` or ``'123'`` or integer ``123``. Rotations about an axis
        twice in a row are prohibited.

        >>> B.orient_body_fixed(N, (q1, q2, 0), 'ZXZ')
        >>> B.orient_body_fixed(N, (q1, q2, 0), '121')
        >>> B.orient_body_fixed(N, (q1, q2, q3), 123)

        """
        from sympy.physics.vector.functions import dynamicsymbols

        _check_frame(parent)

        amounts, rot_order, rot_matrices = self._parse_consecutive_rotations(
            angles, rotation_order)
        self._dcm(parent, rot_matrices[0] * rot_matrices[1] * rot_matrices[2])

        rot_vecs = [zeros(3, 1) for _ in range(3)]
        for i, order in enumerate(rot_order):
            rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
        u1, u2, u3 = rot_vecs[2] + rot_matrices[2].T * (
            rot_vecs[1] + rot_matrices[1].T * rot_vecs[0])
        wvec = u1 * self.x + u2 * self.y + u3 * self.z  # There is a double -
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_space_fixed(self, parent, angles, rotation_order):
        """Rotates this reference frame relative to the parent reference frame
        by right hand rotating through three successive space fixed simple axis
        rotations. Each subsequent axis of rotation is about the "space fixed"
        unit vectors of the parent reference frame.

        The computed angular velocity in this method is by default expressed in
        the child's frame, so it is most preferable to use ``u1 * child.x + u2 *
        child.y + u3 * child.z`` as generalized speeds.

        Parameters
        ==========
        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations about the parent reference frame's unit
            vectors. The order can be specified by the strings ``'XZX'``,
            ``'131'``, or the integer ``131``. There are 12 unique valid
            rotation orders.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1, q2, q3 = symbols('q1, q2, q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B1 = ReferenceFrame('B1')
        >>> B2 = ReferenceFrame('B2')
        >>> B3 = ReferenceFrame('B3')

        >>> B.orient_space_fixed(N, (q1, q2, q3), '312')
        >>> B.dcm(N)
        Matrix([
        [ sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1)],
        [-sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)],
        [                           sin(q3)*cos(q2),        -sin(q2),                           cos(q2)*cos(q3)]])

        is equivalent to:

        >>> B1.orient_axis(N, N.z, q1)
        >>> B2.orient_axis(B1, N.x, q2)
        >>> B3.orient_axis(B2, N.y, q3)
        >>> B3.dcm(N).simplify()
        Matrix([
        [ sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1)],
        [-sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)],
        [                           sin(q3)*cos(q2),        -sin(q2),                           cos(q2)*cos(q3)]])

        It is worth noting that space-fixed and body-fixed rotations are
        related by the order of the rotations, i.e. the reverse order of body
        fixed will give space fixed and vice versa.

        >>> B.orient_space_fixed(N, (q1, q2, q3), '231')
        >>> B.dcm(N)
        Matrix([
        [cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), -sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [       -sin(q2),                           cos(q2)*cos(q3),                            sin(q3)*cos(q2)],
        [sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1),  sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3)]])

        >>> B.orient_body_fixed(N, (q3, q2, q1), '132')
        >>> B.dcm(N)
        Matrix([
        [cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), -sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [       -sin(q2),                           cos(q2)*cos(q3),                            sin(q3)*cos(q2)],
        [sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1),  sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3)]])

        """
        from sympy.physics.vector.functions import dynamicsymbols

        _check_frame(parent)

        amounts, rot_order, rot_matrices = self._parse_consecutive_rotations(
            angles, rotation_order)
        self._dcm(parent, rot_matrices[2] * rot_matrices[1] * rot_matrices[0])

        rot_vecs = [zeros(3, 1) for _ in range(3)]
        for i, order in enumerate(rot_order):
            rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
        u1, u2, u3 = rot_vecs[0] + rot_matrices[0].T * (
            rot_vecs[1] + rot_matrices[1].T * rot_vecs[2])
        wvec = u1 * self.x + u2 * self.y + u3 * self.z  # There is a double -
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_quaternion(self, parent, numbers):
        """Sets the orientation of this reference frame relative to a parent
        reference frame via an orientation quaternion. An orientation
        quaternion is defined as a finite rotation a unit vector, ``(lambda_x,
        lambda_y, lambda_z)``, by an angle ``theta``. The orientation
        quaternion is described by four parameters:

        - ``q0 = cos(theta/2)``
        - ``q1 = lambda_x*sin(theta/2)``
        - ``q2 = lambda_y*sin(theta/2)``
        - ``q3 = lambda_z*sin(theta/2)``

        See `Quaternions and Spatial Rotation
        <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>`_ on
        Wikipedia for more information.

        Parameters
        ==========
        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        numbers : 4-tuple of sympifiable
            The four quaternion scalar numbers as defined above: ``q0``,
            ``q1``, ``q2``, ``q3``.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')

        Set the orientation:

        >>> B.orient_quaternion(N, (q0, q1, q2, q3))
        >>> B.dcm(N)
        Matrix([
        [q0**2 + q1**2 - q2**2 - q3**2,             2*q0*q3 + 2*q1*q2,            -2*q0*q2 + 2*q1*q3],
        [           -2*q0*q3 + 2*q1*q2, q0**2 - q1**2 + q2**2 - q3**2,             2*q0*q1 + 2*q2*q3],
        [            2*q0*q2 + 2*q1*q3,            -2*q0*q1 + 2*q2*q3, q0**2 - q1**2 - q2**2 + q3**2]])

        """

        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)

        numbers = list(numbers)
        for i, v in enumerate(numbers):
            if not isinstance(v, Vector):
                numbers[i] = sympify(v)

        if not (isinstance(numbers, (list, tuple)) & (len(numbers) == 4)):
            raise TypeError('Amounts are a list or tuple of length 4')
        q0, q1, q2, q3 = numbers
        parent_orient_quaternion = (
            Matrix([[q0**2 + q1**2 - q2**2 - q3**2,
                     2 * (q1 * q2 - q0 * q3),
                     2 * (q0 * q2 + q1 * q3)],
                    [2 * (q1 * q2 + q0 * q3),
                     q0**2 - q1**2 + q2**2 - q3**2,
                     2 * (q2 * q3 - q0 * q1)],
                    [2 * (q1 * q3 - q0 * q2),
                     2 * (q0 * q1 + q2 * q3),
                     q0**2 - q1**2 - q2**2 + q3**2]]))

        self._dcm(parent, parent_orient_quaternion)

        t = dynamicsymbols._t
        q0, q1, q2, q3 = numbers
        q0d = diff(q0, t)
        q1d = diff(q1, t)
        q2d = diff(q2, t)
        q3d = diff(q3, t)
        w1 = 2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1)
        w2 = 2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2)
        w3 = 2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3)
        wvec = Vector([(Matrix([w1, w2, w3]), self)])

        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient(self, parent, rot_type, amounts, rot_order=''):
        """Sets the orientation of this reference frame relative to another
        (parent) reference frame.

        .. note:: It is now recommended to use the ``.orient_axis,
           .orient_body_fixed, .orient_space_fixed, .orient_quaternion``
           methods for the different rotation types.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3,3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        """

        _check_frame(parent)

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()

        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')

        if rot_type == 'AXIS':
            self.orient_axis(parent, amounts[1], amounts[0])

        elif rot_type == 'DCM':
            self.orient_explicit(parent, amounts)

        elif rot_type == 'BODY':
            self.orient_body_fixed(parent, amounts, rot_order)

        elif rot_type == 'SPACE':
            self.orient_space_fixed(parent, amounts, rot_order)

        elif rot_type == 'QUATERNION':
            self.orient_quaternion(parent, amounts)

        else:
            raise NotImplementedError('That is not an implemented rotation')

    def orientnew(self, newname, rot_type, amounts, rot_order='',
                  variables=None, indices=None, latexs=None):
        r"""Returns a new reference frame oriented with respect to this
        reference frame.

        See ``ReferenceFrame.orient()`` for detailed examples of how to orient
        reference frames.

        Parameters
        ==========

        newname : str
            Name for the new reference frame.
        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3,3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.
        indices : tuple of str
            Enables the reference frame's basis unit vectors to be accessed by
            Python's square bracket indexing notation using the provided three
            indice strings and alters the printing of the unit vectors to
            reflect this choice.
        latexs : tuple of str
            Alters the LaTeX printing of the reference frame's basis unit
            vectors to the provided three valid LaTeX strings.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, vlatex
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = ReferenceFrame('N')

        Create a new reference frame A rotated relative to N through a simple
        rotation.

        >>> A = N.orientnew('A', 'Axis', (q0, N.x))

        Create a new reference frame B rotated relative to N through body-fixed
        rotations.

        >>> B = N.orientnew('B', 'Body', (q1, q2, q3), '123')

        Create a new reference frame C rotated relative to N through a simple
        rotation with unique indices and LaTeX printing.

        >>> C = N.orientnew('C', 'Axis', (q0, N.x), indices=('1', '2', '3'),
        ... latexs=(r'\hat{\mathbf{c}}_1',r'\hat{\mathbf{c}}_2',
        ... r'\hat{\mathbf{c}}_3'))
        >>> C['1']
        C['1']
        >>> print(vlatex(C['1']))
        \hat{\mathbf{c}}_1

        """

        newframe = self.__class__(newname, variables=variables,
                                  indices=indices, latexs=latexs)

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()

        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')

        if rot_type == 'AXIS':
            newframe.orient_axis(self, amounts[1], amounts[0])

        elif rot_type == 'DCM':
            newframe.orient_explicit(self, amounts)

        elif rot_type == 'BODY':
            newframe.orient_body_fixed(self, amounts, rot_order)

        elif rot_type == 'SPACE':
            newframe.orient_space_fixed(self, amounts, rot_order)

        elif rot_type == 'QUATERNION':
            newframe.orient_quaternion(self, amounts)

        else:
            raise NotImplementedError('That is not an implemented rotation')
        return newframe

    def set_ang_acc(self, otherframe, value):
        """Define the angular acceleration Vector in a ReferenceFrame.

        Defines the angular acceleration of this ReferenceFrame, in another.
        Angular acceleration can be defined with respect to multiple different
        ReferenceFrames. Care must be taken to not create loops which are
        inconsistent.

        Parameters
        ==========

        otherframe : ReferenceFrame
            A ReferenceFrame to define the angular acceleration in
        value : Vector
            The Vector representing angular acceleration

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(otherframe)
        self._ang_acc_dict.update({otherframe: value})
        otherframe._ang_acc_dict.update({self: -value})

    def set_ang_vel(self, otherframe, value):
        """Define the angular velocity vector in a ReferenceFrame.

        Defines the angular velocity of this ReferenceFrame, in another.
        Angular velocity can be defined with respect to multiple different
        ReferenceFrames. Care must be taken to not create loops which are
        inconsistent.

        Parameters
        ==========

        otherframe : ReferenceFrame
            A ReferenceFrame to define the angular velocity in
        value : Vector
            The Vector representing angular velocity

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(otherframe)
        self._ang_vel_dict.update({otherframe: value})
        otherframe._ang_vel_dict.update({self: -value})

    @property
    def x(self):
        """The basis Vector for the ReferenceFrame, in the x direction. """
        return self._x

    @property
    def y(self):
        """The basis Vector for the ReferenceFrame, in the y direction. """
        return self._y

    @property
    def z(self):
        """The basis Vector for the ReferenceFrame, in the z direction. """
        return self._z

    @property
    def xx(self):
        """Unit dyad of basis Vectors x and x for the ReferenceFrame."""
        return Vector.outer(self.x, self.x)

    @property
    def xy(self):
        """Unit dyad of basis Vectors x and y for the ReferenceFrame."""
        return Vector.outer(self.x, self.y)

    @property
    def xz(self):
        """Unit dyad of basis Vectors x and z for the ReferenceFrame."""
        return Vector.outer(self.x, self.z)

    @property
    def yx(self):
        """Unit dyad of basis Vectors y and x for the ReferenceFrame."""
        return Vector.outer(self.y, self.x)

    @property
    def yy(self):
        """Unit dyad of basis Vectors y and y for the ReferenceFrame."""
        return Vector.outer(self.y, self.y)

    @property
    def yz(self):
        """Unit dyad of basis Vectors y and z for the ReferenceFrame."""
        return Vector.outer(self.y, self.z)

    @property
    def zx(self):
        """Unit dyad of basis Vectors z and x for the ReferenceFrame."""
        return Vector.outer(self.z, self.x)

    @property
    def zy(self):
        """Unit dyad of basis Vectors z and y for the ReferenceFrame."""
        return Vector.outer(self.z, self.y)

    @property
    def zz(self):
        """Unit dyad of basis Vectors z and z for the ReferenceFrame."""
        return Vector.outer(self.z, self.z)

    @property
    def u(self):
        """Unit dyadic for the ReferenceFrame."""
        return self.xx + self.yy + self.zz

    def partial_velocity(self, frame, *gen_speeds):
        """Returns the partial angular velocities of this frame in the given
        frame with respect to one or more provided generalized speeds.

        Parameters
        ==========
        frame : ReferenceFrame
            The frame with which the angular velocity is defined in.
        gen_speeds : functions of time
            The generalized speeds.

        Returns
        =======
        partial_velocities : tuple of Vector
            The partial angular velocity vectors corresponding to the provided
            generalized speeds.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> A.set_ang_vel(N, u1 * A.x + u2 * N.y)
        >>> A.partial_velocity(N, u1)
        A.x
        >>> A.partial_velocity(N, u1, u2)
        (A.x, N.y)

        """

        from sympy.physics.vector.functions import partial_velocity

        vel = self.ang_vel_in(frame)
        partials = partial_velocity([vel], gen_speeds, frame)[0]

        if len(partials) == 1:
            return partials[0]
        else:
            return tuple(partials)


def _check_frame(other):
    from .vector import VectorTypeError
    if not isinstance(other, ReferenceFrame):
        raise VectorTypeError(other, ReferenceFrame('A'))
<|endoftext|>from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE

__all__ = ['vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


class VectorStrPrinter(StrPrinter):
    """String Printer for vector expressions. """

    def _print_Derivative(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if (bool(sum(i == t for i in e.variables)) &
                isinstance(type(e.args[0]), UndefinedFunction)):
            ol = str(e.args[0].func)
            for i, v in enumerate(e.variables):
                ol += dynamicsymbols._str
            return ol
        else:
            return StrPrinter().doprint(e)

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if isinstance(type(e), UndefinedFunction):
            return StrPrinter().doprint(e).replace("(%s)" % t, '')
        return e.func.__name__ + "(%s)" % self.stringify(e.args, ", ")


class VectorStrReprPrinter(VectorStrPrinter):
    """String repr printer for vector expressions."""
    def _print_str(self, s):
        return repr(s)


class VectorLatexPrinter(LatexPrinter):
    """Latex Printer for vector expressions. """

    def _print_Function(self, expr, exp=None):
        from sympy.physics.vector.functions import dynamicsymbols
        func = expr.func.__name__
        t = dynamicsymbols._t

        if (hasattr(self, '_print_' + func) and not
            isinstance(type(expr), UndefinedFunction)):
            return getattr(self, '_print_' + func)(expr, exp)
        elif isinstance(type(expr), UndefinedFunction) and (expr.args == (t,)):
            # treat this function like a symbol
            expr = Symbol(func)
            if exp is not None:
                # copied from LatexPrinter._helper_print_standard_power, which
                # we can't call because we only have exp as a string.
                base = self.parenthesize(expr, PRECEDENCE['Pow'])
                base = self.parenthesize_super(base)
                return r"%s^{%s}" % (base, exp)
            else:
                return super()._print(expr)
        else:
            return super()._print_Function(expr, exp)

    def _print_Derivative(self, der_expr):
        from sympy.physics.vector.functions import dynamicsymbols
        # make sure it is in the right form
        der_expr = der_expr.doit()
        if not isinstance(der_expr, Derivative):
            return r"\left(%s\right)" % self.doprint(der_expr)

        # check if expr is a dynamicsymbol
        t = dynamicsymbols._t
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        test1 = not all(True for i in red if i.free_symbols == {t})
        test2 = not all(t == i for i in syms)
        if test1 or test2:
            return super()._print_Derivative(der_expr)

        # done checking
        dots = len(syms)
        base = self._print_Function(expr)
        base_split = base.split('_', 1)
        base = base_split[0]
        if dots == 1:
            base = r"\dot{%s}" % base
        elif dots == 2:
            base = r"\ddot{%s}" % base
        elif dots == 3:
            base = r"\dddot{%s}" % base
        elif dots == 4:
            base = r"\ddddot{%s}" % base
        else:  # Fallback to standard printing
            return super()._print_Derivative(der_expr)
        if len(base_split) != 1:
            base += '_' + base_split[1]
        return base


class VectorPrettyPrinter(PrettyPrinter):
    """Pretty Printer for vectorialexpressions. """

    def _print_Derivative(self, deriv):
        from sympy.physics.vector.functions import dynamicsymbols
        # XXX use U('PARTIAL DIFFERENTIAL') here ?
        t = dynamicsymbols._t
        dot_i = 0
        syms = list(reversed(deriv.variables))

        while len(syms) > 0:
            if syms[-1] == t:
                syms.pop()
                dot_i += 1
            else:
                return super()._print_Derivative(deriv)

        if not (isinstance(type(deriv.expr), UndefinedFunction) and
                (deriv.expr.args == (t,))):
            return super()._print_Derivative(deriv)
        else:
            pform = self._print_Function(deriv.expr)

        # the following condition would happen with some sort of non-standard
        # dynamic symbol I guess, so we'll just print the SymPy way
        if len(pform.picture) > 1:
            return super()._print_Derivative(deriv)

        # There are only special symbols up to fourth-order derivatives
        if dot_i >= 5:
            return super()._print_Derivative(deriv)

        # Deal with special symbols
        dots = {0: "",
                1: "\N{COMBINING DOT ABOVE}",
                2: "\N{COMBINING DIAERESIS}",
                3: "\N{COMBINING THREE DOTS ABOVE}",
                4: "\N{COMBINING FOUR DOTS ABOVE}"}

        d = pform.__dict__
        # if unicode is false then calculate number of apostrophes needed and
        # add to output
        if not self._use_unicode:
            apostrophes = ""
            for i in range(0, dot_i):
                apostrophes += "'"
            d['picture'][0] += apostrophes + "(t)"
        else:
            d['picture'] = [center_accent(d['picture'][0], dots[dot_i])]
        return pform

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        # XXX works only for applied functions
        func = e.func
        args = e.args
        func_name = func.__name__
        pform = self._print_Symbol(Symbol(func_name))
