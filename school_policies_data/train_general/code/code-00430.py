        for pick in cycles_picks:
            new_cycles = []
            l = 0
            for cycle in pick:
                new_cycle = [i - p for i in cycle]
                new_cycles.append(new_cycle)
                l += len(cycle)
            p += l
            perm = Permutation(new_cycles)
            mat = PermutationMatrix(perm)
            args.append(mat)

        return BlockDiagMatrix(*args)


class MatrixPermute(MatrixExpr):
    r"""Symbolic representation for permuting matrix rows or columns.

    Parameters
    ==========

    perm : Permutation, PermutationMatrix
        The permutation to use for permuting the matrix.
        The permutation can be resized to the suitable one,

    axis : 0 or 1
        The axis to permute alongside.
        If `0`, it will permute the matrix rows.
        If `1`, it will permute the matrix columns.

    Notes
    =====

    This follows the same notation used in
    :meth:`sympy.matrices.matrixbase.MatrixBase.permute`.

    Examples
    ========

    >>> from sympy import Matrix, MatrixPermute
    >>> from sympy.combinatorics import Permutation

    Permuting the matrix rows:

    >>> p = Permutation(1, 2, 0)
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = MatrixPermute(A, p, axis=0)
    >>> B.as_explicit()
    Matrix([
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3]])

    Permuting the matrix columns:

    >>> B = MatrixPermute(A, p, axis=1)
    >>> B.as_explicit()
    Matrix([
    [2, 3, 1],
    [5, 6, 4],
    [8, 9, 7]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.permute
    """
    def __new__(cls, mat, perm, axis=S.Zero):
        from sympy.combinatorics.permutations import Permutation

        mat = _sympify(mat)
        if not mat.is_Matrix:
            raise ValueError(
                "{} must be a SymPy matrix instance.".format(perm))

        perm = _sympify(perm)
        if isinstance(perm, PermutationMatrix):
            perm = perm.args[0]

        if not isinstance(perm, Permutation):
            raise ValueError(
                "{} must be a SymPy Permutation or a PermutationMatrix " \
                "instance".format(perm))

        axis = _sympify(axis)
        if axis not in (0, 1):
            raise ValueError("The axis must be 0 or 1.")

        mat_size = mat.shape[axis]
        if mat_size != perm.size:
            try:
                perm = perm.resize(mat_size)
            except ValueError:
                raise ValueError(
                    "Size does not match between the permutation {} "
                    "and the matrix {} threaded over the axis {} "
                    "and cannot be converted."
                    .format(perm, mat, axis))

        return super().__new__(cls, mat, perm, axis)

    def doit(self, deep=True, **hints):
        mat, perm, axis = self.args

        if deep:
            mat = mat.doit(deep=deep, **hints)
            perm = perm.doit(deep=deep, **hints)

        if perm.is_Identity:
            return mat

        if mat.is_Identity:
            if axis is S.Zero:
                return PermutationMatrix(perm)
            elif axis is S.One:
                return PermutationMatrix(perm**-1)

        if isinstance(mat, (ZeroMatrix, OneMatrix)):
            return mat

        if isinstance(mat, MatrixPermute) and mat.args[2] == axis:
            return MatrixPermute(mat.args[0], perm * mat.args[1], axis)

        return self

    @property
    def shape(self):
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        mat, perm, axis = self.args

        if axis == 0:
            return mat[perm.apply(i), j]
        elif axis == 1:
            return mat[i, perm.apply(j)]

    def _eval_rewrite_as_MatMul(self, *args, **kwargs):
        from .matmul import MatMul

        mat, perm, axis = self.args

        deep = kwargs.get("deep", True)

        if deep:
            mat = mat.rewrite(MatMul)

        if axis == 0:
            return MatMul(PermutationMatrix(perm), mat)
        elif axis == 1:
            return MatMul(mat, PermutationMatrix(perm**-1))
<|endoftext|>from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.polys.polytools import Poly

from .matexpr import MatrixExpr


class CompanionMatrix(MatrixExpr):
    """A symbolic companion matrix of a polynomial.

    Examples
    ========

    >>> from sympy import Poly, Symbol, symbols
    >>> from sympy.matrices.expressions import CompanionMatrix
    >>> x = Symbol('x')
    >>> c0, c1, c2, c3, c4 = symbols('c0:5')
    >>> p = Poly(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + x**5, x)
    >>> CompanionMatrix(p)
    CompanionMatrix(Poly(x**5 + c4*x**4 + c3*x**3 + c2*x**2 + c1*x + c0,
    x, domain='ZZ[c0,c1,c2,c3,c4]'))
    """
    def __new__(cls, poly):
        poly = _sympify(poly)
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))
        if not poly.degree() >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        return super().__new__(cls, poly)


    @property
    def shape(self):
        poly = self.args[0]
        size = poly.degree()
        return size, size


    def _entry(self, i, j):
        if j == self.cols - 1:
            return -self.args[0].all_coeffs()[-1 - i]
        elif i == j + 1:
            return S.One
        return S.Zero


    def as_explicit(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix.companion(self.args[0])
<|endoftext|>from .matexpr import MatrixExpr
from sympy.core.function import FunctionClass, Lambda
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify, sympify
from sympy.matrices import Matrix
from sympy.functions.elementary.complexes import re, im


class FunctionMatrix(MatrixExpr):
    """Represents a matrix using a function (``Lambda``) which gives
    outputs according to the coordinates of each matrix entries.

    Parameters
    ==========

    rows : nonnegative integer. Can be symbolic.

    cols : nonnegative integer. Can be symbolic.

    lamda : Function, Lambda or str
        If it is a SymPy ``Function`` or ``Lambda`` instance,
        it should be able to accept two arguments which represents the
        matrix coordinates.

        If it is a pure string containing Python ``lambda`` semantics,
        it is interpreted by the SymPy parser and casted into a SymPy
        ``Lambda`` instance.

    Examples
    ========

    Creating a ``FunctionMatrix`` from ``Lambda``:

    >>> from sympy import FunctionMatrix, symbols, Lambda, MatPow
    >>> i, j, n, m = symbols('i,j,n,m')
    >>> FunctionMatrix(n, m, Lambda((i, j), i + j))
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Creating a ``FunctionMatrix`` from a SymPy function:

    >>> from sympy import KroneckerDelta
    >>> X = FunctionMatrix(3, 3, KroneckerDelta)
    >>> X.as_explicit()
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    Creating a ``FunctionMatrix`` from a SymPy undefined function:

    >>> from sympy import Function
    >>> f = Function('f')
    >>> X = FunctionMatrix(3, 3, f)
    >>> X.as_explicit()
    Matrix([
    [f(0, 0), f(0, 1), f(0, 2)],
    [f(1, 0), f(1, 1), f(1, 2)],
    [f(2, 0), f(2, 1), f(2, 2)]])

    Creating a ``FunctionMatrix`` from Python ``lambda``:

    >>> FunctionMatrix(n, m, 'lambda i, j: i + j')
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Example of lazy evaluation of matrix product:

    >>> Y = FunctionMatrix(1000, 1000, Lambda((i, j), i + j))
    >>> isinstance(Y*Y, MatPow) # this is an expression object
    True
    >>> (Y**2)[10,10] # So this is evaluated lazily
    342923500

    Notes
    =====

    This class provides an alternative way to represent an extremely
    dense matrix with entries in some form of a sequence, in a most
    sparse way.
    """
    def __new__(cls, rows, cols, lamda):
        rows, cols = _sympify(rows), _sympify(cols)
        cls._check_dim(rows)
        cls._check_dim(cols)

        lamda = sympify(lamda)
        if not isinstance(lamda, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(lamda))

        if 2 not in lamda.nargs:
            raise ValueError(
                '{} should be able to accept 2 arguments.'.format(lamda))

        if not isinstance(lamda, Lambda):
            i, j = Dummy('i'), Dummy('j')
            lamda = Lambda((i, j), lamda(i, j))

        return super().__new__(cls, rows, cols, lamda)

    @property
    def shape(self):
        return self.args[0:2]

    @property
    def lamda(self):
        return self.args[2]

    def _entry(self, i, j, **kwargs):
        return self.lamda(i, j)

    def _eval_trace(self):
        from sympy.matrices.expressions.trace import Trace
        from sympy.concrete.summations import Sum
        return Trace(self).rewrite(Sum).doit()

    def _eval_as_real_imag(self):
        return (re(Matrix(self)), im(Matrix(self)))
<|endoftext|>from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.functions.elementary.integers import floor

def normalize(i, parentsize):
    if isinstance(i, slice):
        i = (i.start, i.stop, i.step)
    if not isinstance(i, (tuple, list, Tuple)):
        if (i < 0) == True:
            i += parentsize
        i = (i, i+1, 1)
    i = list(i)
    if len(i) == 2:
        i.append(1)
    start, stop, step = i
    start = start or 0
    if stop is None:
        stop = parentsize
    if (start < 0) == True:
        start += parentsize
    if (stop < 0) == True:
        stop += parentsize
    step = step or 1

    if ((stop - start) * step < 1) == True:
        raise IndexError()

    return (start, stop, step)

class MatrixSlice(MatrixExpr):
    """ A MatrixSlice of a Matrix Expression

    Examples
    ========

    >>> from sympy import MatrixSlice, ImmutableMatrix
    >>> M = ImmutableMatrix(4, 4, range(16))
    >>> M
    Matrix([
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15]])

    >>> B = MatrixSlice(M, (0, 2), (2, 4))
    >>> ImmutableMatrix(B)
    Matrix([
    [2, 3],
    [6, 7]])
    """
    parent = property(lambda self: self.args[0])
    rowslice = property(lambda self: self.args[1])
    colslice = property(lambda self: self.args[2])

    def __new__(cls, parent, rowslice, colslice):
        rowslice = normalize(rowslice, parent.shape[0])
        colslice = normalize(colslice, parent.shape[1])
        if not (len(rowslice) == len(colslice) == 3):
            raise IndexError()
        if ((0 > rowslice[0]) == True or
            (parent.shape[0] < rowslice[1]) == True or
            (0 > colslice[0]) == True or
            (parent.shape[1] < colslice[1]) == True):
            raise IndexError()
        if isinstance(parent, MatrixSlice):
            return mat_slice_of_slice(parent, rowslice, colslice)
        return Basic.__new__(cls, parent, Tuple(*rowslice), Tuple(*colslice))

    @property
    def shape(self):
        rows = self.rowslice[1] - self.rowslice[0]
        rows = rows if self.rowslice[2] == 1 else floor(rows/self.rowslice[2])
        cols = self.colslice[1] - self.colslice[0]
        cols = cols if self.colslice[2] == 1 else floor(cols/self.colslice[2])
        return rows, cols

    def _entry(self, i, j, **kwargs):
        return self.parent._entry(i*self.rowslice[2] + self.rowslice[0],
                                  j*self.colslice[2] + self.colslice[0],
                                  **kwargs)

    @property
    def on_diag(self):
        return self.rowslice == self.colslice


def slice_of_slice(s, t):
    start1, stop1, step1 = s
    start2, stop2, step2 = t

    start = start1 + start2*step1
    step = step1 * step2
    stop = start1 + step1*stop2

    if stop > stop1:
        raise IndexError()

    return start, stop, step


def mat_slice_of_slice(parent, rowslice, colslice):
    """ Collapse nested matrix slices

    >>> from sympy import MatrixSymbol
    >>> X = MatrixSymbol('X', 10, 10)
    >>> X[:, 1:5][5:8, :]
    X[5:8, 1:5]
    >>> X[1:9:2, 2:6][1:3, 2]
    X[3:7:2, 4:5]
    """
    row = slice_of_slice(parent.rowslice, rowslice)
    col = slice_of_slice(parent.colslice, colslice)
    return MatrixSlice(parent.parent, row, col)
<|endoftext|>from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.exceptions import NonInvertibleMatrixError
from .matexpr import MatrixExpr


class ZeroMatrix(MatrixExpr):
    """The Matrix Zero 0 - additive identity

    Examples
    ========

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> A = MatrixSymbol('A', 3, 5)
    >>> Z = ZeroMatrix(3, 5)
    >>> A + Z
    A
    >>> Z*A.T
    0
    """
    is_ZeroMatrix = True

    def __new__(cls, m, n):
        m, n = _sympify(m), _sympify(n)
        cls._check_dim(m)
        cls._check_dim(n)

        return super().__new__(cls, m, n)

    @property
    def shape(self):
        return (self.args[0], self.args[1])

    def _eval_power(self, exp):
        # exp = -1, 0, 1 are already handled at this stage
        if (exp < 0) == True:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
        return self

    def _eval_transpose(self):
        return ZeroMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        return ZeroMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.Zero

    def _eval_determinant(self):
        return S.Zero

    def _eval_inverse(self):
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    def _eval_as_real_imag(self):
        return (self, self)

    def _eval_conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.Zero


class GenericZeroMatrix(ZeroMatrix):
    """
    A zero matrix without a specified shape

    This exists primarily so MatAdd() with no arguments can return something
    meaningful.
    """
    def __new__(cls):
        # super(ZeroMatrix, cls) instead of super(GenericZeroMatrix, cls)
        # because ZeroMatrix.__new__ doesn't have the same signature
        return super(ZeroMatrix, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")

    @property
    def cols(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")

    @property
    def shape(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")

    # Avoid Matrix.__eq__ which might call .shape
    def __eq__(self, other):
        return isinstance(other, GenericZeroMatrix)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return super().__hash__()



class Identity(MatrixExpr):
    """The Matrix Identity I - multiplicative identity

    Examples
    ========

    >>> from sympy import Identity, MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 5)
    >>> I = Identity(3)
    >>> I*A
    A
    """

    is_Identity = True

    def __new__(cls, n):
        n = _sympify(n)
        cls._check_dim(n)

        return super().__new__(cls, n)

    @property
    def rows(self):
        return self.args[0]

    @property
    def cols(self):
        return self.args[0]

    @property
    def shape(self):
        return (self.args[0], self.args[0])

    @property
    def is_square(self):
        return True

    def _eval_transpose(self):
        return self

    def _eval_trace(self):
        return self.rows

    def _eval_inverse(self):
        return self

    def _eval_as_real_imag(self):
        return (self, ZeroMatrix(*self.shape))

    def _eval_conjugate(self):
        return self

    def _eval_adjoint(self):
        return self

    def _entry(self, i, j, **kwargs):
        eq = Eq(i, j)
        if eq is S.true:
            return S.One
        elif eq is S.false:
            return S.Zero
        return KroneckerDelta(i, j, (0, self.cols-1))

    def _eval_determinant(self):
        return S.One

    def _eval_power(self, exp):
        return self


class GenericIdentity(Identity):
    """
    An identity matrix without a specified shape

    This exists primarily so MatMul() with no arguments can return something
    meaningful.
    """
    def __new__(cls):
        # super(Identity, cls) instead of super(GenericIdentity, cls) because
        # Identity.__new__ doesn't have the same signature
        return super(Identity, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def cols(self):
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def shape(self):
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def is_square(self):
        return True

    # Avoid Matrix.__eq__ which might call .shape
    def __eq__(self, other):
        return isinstance(other, GenericIdentity)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return super().__hash__()


class OneMatrix(MatrixExpr):
    """
    Matrix whose all entries are ones.
    """
    def __new__(cls, m, n, evaluate=False):
        m, n = _sympify(m), _sympify(n)
        cls._check_dim(m)
        cls._check_dim(n)

        if evaluate:
            condition = Eq(m, 1) & Eq(n, 1)
            if condition == True:
                return Identity(1)

        obj = super().__new__(cls, m, n)
        return obj

    @property
    def shape(self):
        return self._args

    @property
    def is_Identity(self):
        return self._is_1x1() == True

    def as_explicit(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix.ones(*self.shape)

    def doit(self, **hints):
        args = self.args
        if hints.get('deep', True):
            args = [a.doit(**hints) for a in args]
        return self.func(*args, evaluate=True)

    def _eval_power(self, exp):
        # exp = -1, 0, 1 are already handled at this stage
        if self._is_1x1() == True:
            return Identity(1)
        if (exp < 0) == True:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
        if ask(Q.integer(exp)):
            return self.shape[0] ** (exp - 1) * OneMatrix(*self.shape)
        return super()._eval_power(exp)

    def _eval_transpose(self):
        return OneMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        return OneMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.One*self.rows

    def _is_1x1(self):
        """Returns true if the matrix is known to be 1x1"""
        shape = self.shape
        return Eq(shape[0], 1) & Eq(shape[1], 1)

    def _eval_determinant(self):
        condition = self._is_1x1()
        if condition == True:
            return S.One
        elif condition == False:
            return S.Zero
        else:
            from sympy.matrices.expressions.determinant import Determinant
            return Determinant(self)

    def _eval_inverse(self):
        condition = self._is_1x1()
        if condition == True:
            return Identity(1)
        elif condition == False:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")
        else:
            from .inverse import Inverse
            return Inverse(self)

    def _eval_as_real_imag(self):
        return (self, ZeroMatrix(*self.shape))

    def _eval_conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.One
<|endoftext|>from collections import Counter

from sympy.core import Mul, sympify
from sympy.core.add import Add
from sympy.core.expr import ExprBuilder
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import log
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix
from sympy.strategies import (
    unpack, flatten, condition, exhaust, rm_id, sort
)
from sympy.utilities.exceptions import sympy_deprecation_warning


def hadamard_product(*matrices):
    """
    Return the elementwise (aka Hadamard) product of matrices.

    Examples
    ========

    >>> from sympy import hadamard_product, MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 3)
    >>> B = MatrixSymbol('B', 2, 3)
    >>> hadamard_product(A)
    A
    >>> hadamard_product(A, B)
    HadamardProduct(A, B)
    >>> hadamard_product(A, B)[0, 1]
    A[0, 1]*B[0, 1]
    """
    if not matrices:
        raise TypeError("Empty Hadamard product is undefined")
    if len(matrices) == 1:
        return matrices[0]
    return HadamardProduct(*matrices).doit()


class HadamardProduct(MatrixExpr):
    """
    Elementwise product of matrix expressions

    Examples
    ========

    Hadamard product for matrix symbols:

    >>> from sympy import hadamard_product, HadamardProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(hadamard_product(A, B), HadamardProduct)
    True

    Notes
    =====

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``hadamard_product()`` or ``HadamardProduct.doit``
    """
    is_HadamardProduct = True

    def __new__(cls, *args, evaluate=False, check=None):
        args = list(map(sympify, args))
        if len(args) == 0:
            # We currently don't have a way to support one-matrices of generic dimensions:
            raise ValueError("HadamardProduct needs at least one argument")

        if not all(isinstance(arg, MatrixExpr) for arg in args):
            raise TypeError("Mix of Matrix and Scalar symbols")

        if check is not None:
            sympy_deprecation_warning(
                "Passing check to HadamardProduct is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        if check is not False:
            validate(*args)

        obj = super().__new__(cls, *args)
        if evaluate:
            obj = obj.doit(deep=False)
        return obj

    @property
    def shape(self):
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        return Mul(*[arg._entry(i, j, **kwargs) for arg in self.args])

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardProduct(*list(map(transpose, self.args)))

    def doit(self, **hints):
        expr = self.func(*(i.doit(**hints) for i in self.args))
        # Check for explicit matrices:
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.immutable import ImmutableMatrix

        explicit = [i for i in expr.args if isinstance(i, MatrixBase)]
        if explicit:
            remainder = [i for i in expr.args if i not in explicit]
            expl_mat = ImmutableMatrix([
                Mul.fromiter(i) for i in zip(*explicit)
            ]).reshape(*self.shape)
            expr = HadamardProduct(*([expl_mat] + remainder))

        return canonicalize(expr)

    def _eval_derivative(self, x):
        terms = []
        args = list(self.args)
        for i in range(len(args)):
            factors = args[:i] + [args[i].diff(x)] + args[i+1:]
            terms.append(hadamard_product(*factors))
        return Add.fromiter(terms)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.matrices.expressions.matexpr import _make_matrix

        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind+1:]

            d = self.args[ind]._eval_derivative_matrix_lines(x)
            hadam = hadamard_product(*(right_args + left_args))
            diagonal = [(0, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.shape[j] != 1]
            for i in d:
                l1 = i._lines[i._first_line_index]
                l2 = i._lines[i._second_line_index]
                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ExprBuilder(_make_matrix, [l1]),
                                hadam,
                                ExprBuilder(_make_matrix, [l2]),
                            ]
                        ),
                    *diagonal],

                )
                i._first_pointer_parent = subexpr.args[0].args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args[2].args
                i._second_pointer_index = 0
                i._lines = [subexpr]
                lines.append(i)

        return lines


# TODO Implement algorithm for rewriting Hadamard product as diagonal matrix
# if matmul identy matrix is multiplied.
def canonicalize(x):
    """Canonicalize the Hadamard product ``x`` with mathematical properties.

    Examples
    ========

    >>> from sympy import MatrixSymbol, HadamardProduct
    >>> from sympy import OneMatrix, ZeroMatrix
    >>> from sympy.matrices.expressions.hadamard import canonicalize
    >>> from sympy import init_printing
    >>> init_printing(use_unicode=False)

    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = MatrixSymbol('B', 2, 2)
    >>> C = MatrixSymbol('C', 2, 2)

    Hadamard product associativity:

    >>> X = HadamardProduct(A, HadamardProduct(B, C))
    >>> X
    A.*(B.*C)
    >>> canonicalize(X)
    A.*B.*C

    Hadamard product commutativity:

    >>> X = HadamardProduct(A, B)
    >>> Y = HadamardProduct(B, A)
    >>> X
    A.*B
    >>> Y
    B.*A
    >>> canonicalize(X)
    A.*B
    >>> canonicalize(Y)
    A.*B

    Hadamard product identity:

    >>> X = HadamardProduct(A, OneMatrix(2, 2))
    >>> X
    A.*1
    >>> canonicalize(X)
    A

    Absorbing element of Hadamard product:

    >>> X = HadamardProduct(A, ZeroMatrix(2, 2))
    >>> X
    A.*0
    >>> canonicalize(X)
    0

    Rewriting to Hadamard Power

    >>> X = HadamardProduct(A, A, A)
    >>> X
    A.*A.*A
    >>> canonicalize(X)
     .3
    A

    Notes
    =====

    As the Hadamard product is associative, nested products can be flattened.

    The Hadamard product is commutative so that factors can be sorted for
    canonical form.

    A matrix of only ones is an identity for Hadamard product,
    so every matrices of only ones can be removed.

    Any zero matrix will make the whole product a zero matrix.

    Duplicate elements can be collected and rewritten as HadamardPower

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    """
    # Associativity
    rule = condition(
            lambda x: isinstance(x, HadamardProduct),
            flatten
        )
    fun = exhaust(rule)
    x = fun(x)

    # Identity
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            rm_id(lambda x: isinstance(x, OneMatrix))
        )
    x = fun(x)

    # Absorbing by Zero Matrix
    def absorb(x):
        if any(isinstance(c, ZeroMatrix) for c in x.args):
            return ZeroMatrix(*x.shape)
        else:
            return x
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            absorb
        )
    x = fun(x)

    # Rewriting with HadamardPower
    if isinstance(x, HadamardProduct):
        tally = Counter(x.args)

        new_arg = []
        for base, exp in tally.items():
            if exp == 1:
                new_arg.append(base)
            else:
                new_arg.append(HadamardPower(base, exp))

        x = HadamardProduct(*new_arg)

    # Commutativity
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            sort(default_sort_key)
        )
    x = fun(x)

    # Unpacking
    x = unpack(x)
    return x


def hadamard_power(base, exp):
    base = sympify(base)
    exp = sympify(exp)
    if exp == 1:
        return base
    if not base.is_Matrix:
        return base**exp
    if exp.is_Matrix:
        raise ValueError("cannot raise expression to a matrix")
    return HadamardPower(base, exp)


class HadamardPower(MatrixExpr):
    r"""
    Elementwise power of matrix expressions

    Parameters
    ==========

    base : scalar or matrix

    exp : scalar or matrix

    Notes
    =====

    There are four definitions for the hadamard power which can be used.
    Let's consider `A, B` as `(m, n)` matrices, and `a, b` as scalars.

    Matrix raised to a scalar exponent:

    .. math::
        A^{\circ b} = \begin{bmatrix}
        A_{0, 0}^b   & A_{0, 1}^b   & \cdots & A_{0, n-1}^b   \\
        A_{1, 0}^b   & A_{1, 1}^b   & \cdots & A_{1, n-1}^b   \\
        \vdots       & \vdots       & \ddots & \vdots         \\
        A_{m-1, 0}^b & A_{m-1, 1}^b & \cdots & A_{m-1, n-1}^b
        \end{bmatrix}

    Scalar raised to a matrix exponent:

    .. math::
        a^{\circ B} = \begin{bmatrix}
        a^{B_{0, 0}}   & a^{B_{0, 1}}   & \cdots & a^{B_{0, n-1}}   \\
        a^{B_{1, 0}}   & a^{B_{1, 1}}   & \cdots & a^{B_{1, n-1}}   \\
        \vdots         & \vdots         & \ddots & \vdots           \\
        a^{B_{m-1, 0}} & a^{B_{m-1, 1}} & \cdots & a^{B_{m-1, n-1}}
        \end{bmatrix}

    Matrix raised to a matrix exponent:

    .. math::
        A^{\circ B} = \begin{bmatrix}
        A_{0, 0}^{B_{0, 0}}     & A_{0, 1}^{B_{0, 1}}     &
        \cdots & A_{0, n-1}^{B_{0, n-1}}     \\
        A_{1, 0}^{B_{1, 0}}     & A_{1, 1}^{B_{1, 1}}     &
        \cdots & A_{1, n-1}^{B_{1, n-1}}     \\
        \vdots                  & \vdots                  &
        \ddots & \vdots                      \\
        A_{m-1, 0}^{B_{m-1, 0}} & A_{m-1, 1}^{B_{m-1, 1}} &
        \cdots & A_{m-1, n-1}^{B_{m-1, n-1}}
        \end{bmatrix}

    Scalar raised to a scalar exponent:

    .. math::
        a^{\circ b} = a^b
    """

    def __new__(cls, base, exp):
        base = sympify(base)
        exp = sympify(exp)

        if base.is_scalar and exp.is_scalar:
            return base ** exp

        if isinstance(base, MatrixExpr) and isinstance(exp, MatrixExpr):
            validate(base, exp)

        obj = super().__new__(cls, base, exp)
        return obj

    @property
    def base(self):
        return self._args[0]

    @property
    def exp(self):
        return self._args[1]

    @property
    def shape(self):
        if self.base.is_Matrix:
            return self.base.shape
        return self.exp.shape

    def _entry(self, i, j, **kwargs):
        base = self.base
        exp = self.exp

        if base.is_Matrix:
            a = base._entry(i, j, **kwargs)
        elif base.is_scalar:
            a = base
        else:
            raise ValueError(
                'The base {} must be a scalar or a matrix.'.format(base))

        if exp.is_Matrix:
            b = exp._entry(i, j, **kwargs)
        elif exp.is_scalar:
            b = exp
        else:
            raise ValueError(
                'The exponent {} must be a scalar or a matrix.'.format(exp))

        return a ** b

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardPower(transpose(self.base), self.exp)

    def _eval_derivative(self, x):
        dexp = self.exp.diff(x)
        logbase = self.base.applyfunc(log)
        dlbase = logbase.diff(x)
        return hadamard_product(
            dexp*logbase + self.exp*dlbase,
            self
        )

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.matrices.expressions.matexpr import _make_matrix

        lr = self.base._eval_derivative_matrix_lines(x)
        for i in lr:
            diagonal = [(1, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.base.shape[j] != 1]
            l1 = i._lines[i._first_line_index]
            l2 = i._lines[i._second_line_index]
            subexpr = ExprBuilder(
                ArrayDiagonal,
                [
                    ExprBuilder(
                        ArrayTensorProduct,
                        [
                            ExprBuilder(_make_matrix, [l1]),
                            self.exp*hadamard_power(self.base, self.exp-1),
                            ExprBuilder(_make_matrix, [l2]),
                        ]
                    ),
                *diagonal],
                validator=ArrayDiagonal._validate
            )
            i._first_pointer_parent = subexpr.args[0].args[0].args
            i._first_pointer_index = 0
            i._first_line_index = 0
            i._second_pointer_parent = subexpr.args[0].args[2].args
            i._second_pointer_index = 0
            i._second_line_index = 0
            i._lines = [subexpr]
        return lr
<|endoftext|>from sympy.core import Basic, Expr
from sympy.core.sympify import _sympify
from sympy.matrices.expressions.transpose import transpose


class DotProduct(Expr):
    """
    Dot product of vector matrices

    The input should be two 1 x n or n x 1 matrices. The output represents the
    scalar dotproduct.

    This is similar to using MatrixElement and MatMul, except DotProduct does
    not require that one vector to be a row vector and the other vector to be
    a column vector.

    >>> from sympy import MatrixSymbol, DotProduct
    >>> A = MatrixSymbol('A', 1, 3)
    >>> B = MatrixSymbol('B', 1, 3)
    >>> DotProduct(A, B)
    DotProduct(A, B)
    >>> DotProduct(A, B).doit()
    A[0, 0]*B[0, 0] + A[0, 1]*B[0, 1] + A[0, 2]*B[0, 2]
    """

    def __new__(cls, arg1, arg2):
        arg1, arg2 = _sympify((arg1, arg2))

        if not arg1.is_Matrix:
            raise TypeError("Argument 1 of DotProduct is not a matrix")
        if not arg2.is_Matrix:
            raise TypeError("Argument 2 of DotProduct is not a matrix")
        if not (1 in arg1.shape):
            raise TypeError("Argument 1 of DotProduct is not a vector")
        if not (1 in arg2.shape):
            raise TypeError("Argument 2 of DotProduct is not a vector")

        if set(arg1.shape) != set(arg2.shape):
            raise TypeError("DotProduct arguments are not the same length")

        return Basic.__new__(cls, arg1, arg2)

    def doit(self, expand=False, **hints):
        if self.args[0].shape == self.args[1].shape:
            if self.args[0].shape[0] == 1:
                mul = self.args[0]*transpose(self.args[1])
            else:
                mul = transpose(self.args[0])*self.args[1]
        else:
            if self.args[0].shape[0] == 1:
                mul = self.args[0]*self.args[1]
            else:
                mul = transpose(self.args[0])*transpose(self.args[1])

        return mul[0]
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError


class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)
    >>> Trace(eye(3))
    Trace(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> Trace(eye(3)).simplify()
    3
    """
    is_Trace = True
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)

        if not mat.is_Matrix:
            raise TypeError("input to Trace, %s, is not a matrix" % str(mat))

        if mat.is_square is False:
            raise NonSquareMatrixError("Trace of a non-square matrix")

        return Basic.__new__(cls, mat)

    def _eval_transpose(self):
        return self

    def _eval_derivative(self, v):
        from sympy.concrete.summations import Sum
        from .matexpr import MatrixElement
        if isinstance(v, MatrixElement):
            return self.rewrite(Sum).diff(v)
        expr = self.doit()
        if isinstance(expr, Trace):
            # Avoid looping infinitely:
            raise NotImplementedError
        return expr._eval_derivative(v)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayContraction
        r = self.args[0]._eval_derivative_matrix_lines(x)
        for lr in r:
            if lr.higher == 1:
                lr.higher = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                lr._lines[0],
                                lr._lines[1],
                            ]
                        ),
                        (1, 3),
                    ],
                    validator=ArrayContraction._validate
                )
            else:
                # This is not a matrix line:
                lr.higher = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                lr._lines[0],
                                lr._lines[1],
                                lr.higher,
                            ]
                        ),
                        (1, 3), (0, 2)
                    ]
                )
            lr._lines = [S.One, S.One]
            lr._first_pointer_parent = lr._lines
            lr._second_pointer_parent = lr._lines
            lr._first_pointer_index = 0
            lr._second_pointer_index = 1
        return r

    @property
    def arg(self):
        return self.args[0]

    def doit(self, **hints):
        if hints.get('deep', True):
            arg = self.arg.doit(**hints)
            result = arg._eval_trace()
            if result is not None:
                return result
            else:
                return Trace(arg)
        else:
            # _eval_trace would go too deep here
            if isinstance(self.arg, MatrixBase):
                return trace(self.arg)
            else:
                return Trace(self.arg)

    def as_explicit(self):
        return Trace(self.arg.as_explicit()).doit()

    def _normalize(self):
        # Normalization of trace of matrix products. Use transposition and
        # cyclic properties of traces to make sure the arguments of the matrix
        # product are sorted and the first argument is not a transposition.
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        trace_arg = self.arg
        if isinstance(trace_arg, MatMul):

            def get_arg_key(x):
                a = trace_arg.args[x]
                if isinstance(a, Transpose):
                    a = a.arg
                return default_sort_key(a)

            indmin = min(range(len(trace_arg.args)), key=get_arg_key)
            if isinstance(trace_arg.args[indmin], Transpose):
                trace_arg = Transpose(trace_arg).doit()
                indmin = min(range(len(trace_arg.args)), key=lambda x: default_sort_key(trace_arg.args[x]))
            trace_arg = MatMul.fromiter(trace_arg.args[indmin:] + trace_arg.args[:indmin])
            return Trace(trace_arg)
        return self

    def _eval_rewrite_as_Sum(self, expr, **kwargs):
        from sympy.concrete.summations import Sum
        i = uniquely_named_symbol('i', [expr])
        s = Sum(self.arg[i, i], (i, 0, self.arg.rows - 1))
        return s.doit()


def trace(expr):
    """Trace of a Matrix.  Sum of the diagonal elements.

    Examples
    ========

    >>> from sympy import trace, Symbol, MatrixSymbol, eye
    >>> n = Symbol('n')
    >>> X = MatrixSymbol('X', n, n)  # A square matrix
    >>> trace(2*X)
    2*Trace(X)
    >>> trace(eye(3))
    3
    """
    return Trace(expr).doit()
<|endoftext|>from sympy.core.basic import Basic
from sympy.matrices.expressions.matexpr import MatrixExpr


class Transpose(MatrixExpr):
    """
    The transpose of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the transpose, use the ``transpose()``
    function, or the ``.T`` attribute of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Transpose, transpose
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Transpose(A)
    A.T
    >>> A.T == transpose(A) == Transpose(A)
    True
    >>> Transpose(A*B)
    (A*B).T
    >>> transpose(A*B)
    B.T*A.T

    """
    is_Transpose = True

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            arg = arg.doit(**hints)
        _eval_transpose = getattr(arg, '_eval_transpose', None)
        if _eval_transpose is not None:
            result = _eval_transpose()
            return result if result is not None else Transpose(arg)
        else:
            return Transpose(arg)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape[::-1]

    def _entry(self, i, j, expand=False, **kwargs):
        return self.arg._entry(j, i, expand=expand, **kwargs)

    def _eval_adjoint(self):
        return self.arg.conjugate()

    def _eval_conjugate(self):
        return self.arg.adjoint()

    def _eval_transpose(self):
        return self.arg

    def _eval_trace(self):
        from .trace import Trace
        return Trace(self.arg)  # Trace(X.T) => Trace(X)

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import det
        return det(self.arg)

    def _eval_derivative(self, x):
        # x is a scalar:
        return self.arg._eval_derivative(x)

    def _eval_derivative_matrix_lines(self, x):
        lines = self.args[0]._eval_derivative_matrix_lines(x)
        return [i.transpose() for i in lines]


def transpose(expr):
    """Matrix transpose"""
    return Transpose(expr).doit(deep=False)


from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict


def refine_Transpose(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> X.T
    X.T
    >>> with assuming(Q.symmetric(X)):
    ...     print(refine(X.T))
    X
    """
    if ask(Q.symmetric(expr), assumptions):
        return expr.arg

    return expr

handlers_dict['Transpose'] = refine_Transpose
<|endoftext|>from __future__ import annotations
from functools import wraps

from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.kind import MatrixKind
from sympy.matrices.matrixbase import MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent


def _sympifyit(arg, retval=None):
    # This version of _sympifyit sympifies MutableMatrix objects
    def deco(func):
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                b = _sympify(b)
                return func(a, b)
            except SympifyError:
                return retval

        return __sympifyit_wrapper

    return deco


class MatrixExpr(Expr):
    """Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """
    __slots__: tuple[str, ...] = ()

    # Should not be considered iterable by the
    # sympy.utilities.iterables.iterable function. Subclass that actually are
    # iterable (i.e., explicit matrices) should set this to True.
    _iterable = False

    _op_priority = 11.0

    is_Matrix: bool = True
    is_MatrixExpr: bool = True
    is_Identity: FuzzyBool = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False

    is_commutative = False
    is_number = False
    is_symbol = False
    is_scalar = False

    kind: MatrixKind = MatrixKind()

    def __new__(cls, *args, **kwargs):
        args = map(_sympify, args)
        return Basic.__new__(cls, *args, **kwargs)

    # The following is adapted from the core Expr object

    @property
    def shape(self) -> tuple[Expr, Expr]:
        raise NotImplementedError

    @property
    def _add_handler(self):
        return MatAdd

    @property
    def _mul_handler(self):
        return MatMul

    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return MatAdd(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return MatAdd(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return MatAdd(other, -self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return MatPow(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError("Matrix Power not defined")

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * other**S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        raise NotImplementedError()
        #return MatMul(other, Pow(self, S.NegativeOne))

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def is_square(self) -> bool | None:
        rows, cols = self.shape
        if isinstance(rows, Integer) and isinstance(cols, Integer):
            return rows == cols
        if rows == cols:
            return True
        return None

    def _eval_conjugate(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(Transpose(self))

    def as_real_imag(self, deep=True, **hints):
        return self._eval_as_real_imag()

    def _eval_as_real_imag(self):
        real = S.Half * (self + self._eval_conjugate())
        im = (self - self._eval_conjugate())/(2*S.ImaginaryUnit)
        return (real, im)

    def _eval_inverse(self):
        return Inverse(self)

    def _eval_determinant(self):
        return Determinant(self)

    def _eval_transpose(self):
        return Transpose(self)

    def _eval_trace(self):
        return None

    def _eval_power(self, exp):
        """
        Override this in sub-classes to implement simplification of powers.  The cases where the exponent
        is -1, 0, 1 are already covered in MatPow.doit(), so implementations can exclude these cases.
        """
        return MatPow(self, exp)

    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            from sympy.simplify import simplify
            return self.func(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)

    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    def _eval_derivative(self, x):
        # `x` is a scalar:
        if self.has(x):
            # See if there are other methods using it:
            return super()._eval_derivative(x)
        else:
            return ZeroMatrix(*self.shape)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))


    def _entry(self, i, j, **kwargs):
        raise NotImplementedError(
            "Indexing not implemented for %s" % self.__class__.__name__)

    def adjoint(self):
        return adjoint(self)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return S.One, self

    def conjugate(self):
        return conjugate(self)

    def transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return transpose(self)

    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    def inverse(self):
        if self.is_square is False:
            raise NonSquareMatrixError('Inverse of non-square matrix')
        return self._eval_inverse()

    def inv(self):
        return self.inverse()

    def det(self):
        from sympy.matrices.expressions.determinant import det
        return det(self)

    @property
    def I(self):
        return self.inverse()

    def valid_index(self, i, j):
        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return (is_valid(i) and is_valid(j) and
                (self.rows is None or
                (i >= -self.rows) != False and (i < self.rows) != False) and
                (j >= -self.cols) != False and (j < self.cols) != False)

    def __getitem__(self, key):
        if not isinstance(key, tuple) and isinstance(key, slice):
            from sympy.matrices.expressions.slice import MatrixSlice
            return MatrixSlice(self, key, (0, None, 1))
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                from sympy.matrices.expressions.slice import MatrixSlice
                return MatrixSlice(self, i, j)
            i, j = _sympify(i), _sympify(j)
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid indices (%s, %s)" % (i, j))
        elif isinstance(key, (SYMPY_INTS, Integer)):
            # row-wise decomposition of matrix
            rows, cols = self.shape
            # allow single indexing if number of columns is known
            if not isinstance(cols, Integer):
                raise IndexError(filldedent('''
                    Single indexing is only supported when the number
                    of columns is known.'''))
            key = _sympify(key)
            i = key // cols
            j = key % cols
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid index %s" % key)
        elif isinstance(key, (Symbol, Expr)):
            raise IndexError(filldedent('''
                Only integers may be used when addressing the matrix
                with a single index.'''))
        raise IndexError("Invalid index, wanted %s[i,j]" % self)

    def _is_shape_symbolic(self) -> bool:
        return (not isinstance(self.rows, (SYMPY_INTS, Integer))
            or not isinstance(self.cols, (SYMPY_INTS, Integer)))

    def as_explicit(self):
        """
        Returns a dense Matrix with elements represented explicitly

        Returns an object of type ImmutableDenseMatrix.

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.as_explicit()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_mutable: returns mutable Matrix type

        """
        if self._is_shape_symbolic():
            raise ValueError(
                'Matrix with symbolic shape '
                'cannot be represented explicitly.')
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix([[self[i, j]
                            for j in range(self.cols)]
                            for i in range(self.rows)])

    def as_mutable(self):
        """
        Returns a dense, mutable matrix with elements represented explicitly

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.shape
        (3, 3)
        >>> I.as_mutable()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_explicit: returns ImmutableDenseMatrix
        """
        return self.as_explicit().as_mutable()

    def __array__(self, dtype=object, copy=None):
        if copy is not None and not copy:
            raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
        from numpy import empty
        a = empty(self.shape, dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                a[i, j] = self[i, j]
        return a

    def equals(self, other):
        """
        Test elementwise equality between matrices, potentially of different
        types

        >>> from sympy import Identity, eye
        >>> Identity(3).equals(eye(3))
        True
        """
        return self.as_explicit().equals(other)

    def canonicalize(self):
        return self

    def as_coeff_mmul(self):
        return S.One, MatMul(self)

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        r"""
        Parse expression of matrices with explicitly summed indices into a
        matrix expression without indices, if possible.

        This transformation expressed in mathematical notation:

        `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

        Optional parameter ``first_index``: specify which free index to use as
        the index starting the expression.

        Examples
        ========

        >>> from sympy import MatrixSymbol, MatrixExpr, Sum
        >>> from sympy.abc import i, j, k, l, N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B

        Transposition is detected:

        >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A.T*B

        Detect the trace:

        >>> expr = Sum(A[i, i], (i, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        Trace(A)

        More complicated expressions:

        >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B.T*A.T
        """
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
        first_indices = []
        if first_index is not None:
            first_indices.append(first_index)
        if last_index is not None:
            first_indices.append(last_index)
        arr = convert_indexed_to_array(expr, first_indices=first_indices)
        return convert_array_to_matrix(arr)

    def applyfunc(self, func):
        from .applyfunc import ElementwiseApplyFunction
        return ElementwiseApplyFunction(func, self)


@dispatch(MatrixExpr, Expr)
def _eval_is_eq(lhs, rhs): # noqa:F811
    return False

@dispatch(MatrixExpr, MatrixExpr)  # type: ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    if lhs.shape != rhs.shape:
        return False
    if (lhs - rhs).is_ZeroMatrix:
        return True

def get_postprocessor(cls):
    def _postprocessor(expr):
        # To avoid circular imports, we can't have MatMul/MatAdd on the top level
        mat_class = {Mul: MatMul, Add: MatAdd}[cls]
        nonmatrices = []
        matrices = []
        for term in expr.args:
            if isinstance(term, MatrixExpr):
                matrices.append(term)
            else:
                nonmatrices.append(term)

        if not matrices:
            return cls._from_args(nonmatrices)

        if nonmatrices:
            if cls == Mul:
                for i in range(len(matrices)):
                    if not matrices[i].is_MatrixExpr:
                        # If one of the matrices explicit, absorb the scalar into it
                        # (doit will combine all explicit matrices into one, so it
                        # doesn't matter which)
                        matrices[i] = matrices[i].__mul__(cls._from_args(nonmatrices))
                        nonmatrices = []
                        break

            else:
                # Maintain the ability to create Add(scalar, matrix) without
                # raising an exception. That way different algorithms can
                # replace matrix expressions with non-commutative symbols to
                # manipulate them like non-commutative scalars.
                return cls._from_args(nonmatrices + [mat_class(*matrices).doit(deep=False)])

        if mat_class == MatAdd:
            return mat_class(*matrices).doit(deep=False)
        return mat_class(cls._from_args(nonmatrices), *matrices).doit(deep=False)
    return _postprocessor


Basic._constructor_postprocessor_mapping[MatrixExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
}


def _matrix_derivative(expr, x, old_algorithm=False):

    if isinstance(expr, MatrixBase) or isinstance(x, MatrixBase):
        # Do not use array expressions for explicit matrices:
        old_algorithm = True

    if old_algorithm:
        return _matrix_derivative_old_algorithm(expr, x)

    from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix

    array_expr = convert_matrix_to_array(expr)
    diff_array_expr = array_derive(array_expr, x)
    diff_matrix_expr = convert_array_to_matrix(diff_array_expr)
    return diff_matrix_expr


def _matrix_derivative_old_algorithm(expr, x):
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    lines = expr._eval_derivative_matrix_lines(x)

    parts = [i.build() for i in lines]

    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix

    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]

    def _get_shape(elem):
        if isinstance(elem, MatrixExpr):
            return elem.shape
        return 1, 1

    def get_rank(parts):
        return sum(j not in (1, None) for i in parts for j in _get_shape(i))

    ranks = [get_rank(i) for i in parts]
    rank = ranks[0]

    def contract_one_dims(parts):
        if len(parts) == 1:
            return parts[0]
        else:
            p1, p2 = parts[:2]
            if p2.is_Matrix:
                p2 = p2.T
            if p1 == Identity(1):
                pbase = p2
            elif p2 == Identity(1):
                pbase = p1
            else:
                pbase = p1*p2
            if len(parts) == 2:
                return pbase
            else:  # len(parts) > 2
                if pbase.is_Matrix:
                    raise ValueError("")
                return pbase*Mul.fromiter(parts[2:])

    if rank <= 2:
        return Add.fromiter([contract_one_dims(i) for i in parts])

    return ArrayDerivative(expr, x)


class MatrixElement(Expr):
    parent = property(lambda self: self.args[0])
    i = property(lambda self: self.args[1])
    j = property(lambda self: self.args[2])
    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, n, m):
        n, m = map(_sympify, (n, m))
        if isinstance(name, str):
            name = Symbol(name)
        else:
            if isinstance(name, MatrixBase):
                if n.is_Integer and m.is_Integer:
                    return name[n, m]
                name = _sympify(name)  # change mutable into immutable
            else:
                name = _sympify(name)
                if not isinstance(name.kind, MatrixKind):
                    raise TypeError("First argument of MatrixElement should be a matrix")
            if not getattr(name, 'valid_index', lambda n, m: True)(n, m):
                raise IndexError('indices out of range')
        obj = Expr.__new__(cls, name, n, m)
        return obj

    @property
    def symbol(self):
        return self.args[0]

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return args[0][args[1], args[2]]

    @property
    def indices(self):
        return self.args[1:]

    def _eval_derivative(self, v):

        if not isinstance(v, MatrixElement):
            return self.parent.diff(v)[self.i, self.j]

        M = self.args[0]

        m, n = self.parent.shape

        if M == v.args[0]:
            return KroneckerDelta(self.args[1], v.args[1], (0, m-1)) * \
                   KroneckerDelta(self.args[2], v.args[2], (0, n-1))

        if isinstance(M, Inverse):
            from sympy.concrete.summations import Sum
            i, j = self.args[1:]
            i1, i2 = symbols("z1, z2", cls=Dummy)
            Y = M.args[0]
            r1, r2 = Y.shape
            return -Sum(M[i, i1]*Y[i1, i2].diff(v)*M[i2, j], (i1, 0, r1-1), (i2, 0, r2-1))

        if self.has(v.args[0]):
            return None

        return S.Zero


class MatrixSymbol(MatrixExpr):
    """Symbolic representation of a Matrix object

    Creates a SymPy Symbol to represent a Matrix. This matrix has a shape and
    can be included in Matrix Expressions

    Examples
    ========

    >>> from sympy import MatrixSymbol, Identity
    >>> A = MatrixSymbol('A', 3, 4) # A 3 by 4 Matrix
    >>> B = MatrixSymbol('B', 4, 3) # A 4 by 3 Matrix
    >>> A.shape
    (3, 4)
    >>> 2*A*B + Identity(3)
    I + 2*A*B
    """
    is_commutative = False
    is_symbol = True
    _diff_wrt = True

    def __new__(cls, name, n, m):
        n, m = _sympify(n), _sympify(m)

        cls._check_dim(m)
        cls._check_dim(n)

        if isinstance(name, str):
            name = Str(name)
        obj = Basic.__new__(cls, name, n, m)
        return obj

    @property
    def shape(self):
        return self.args[1], self.args[2]

    @property
    def name(self):
        return self.args[0].name

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    @property
    def free_symbols(self):
        return {self}

    def _eval_simplify(self, **kwargs):
        return self

    def _eval_derivative(self, x):
        # x is a scalar:
        return ZeroMatrix(self.shape[0], self.shape[1])

    def _eval_derivative_matrix_lines(self, x):
        if self != x:
            first = ZeroMatrix(x.shape[0], self.shape[0]) if self.shape[0] != 1 else S.Zero
            second = ZeroMatrix(x.shape[1], self.shape[1]) if self.shape[1] != 1 else S.Zero
            return [_LeftRightArgs(
                [first, second],
            )]
        else:
            first = Identity(self.shape[0]) if self.shape[0] != 1 else S.One
            second = Identity(self.shape[1]) if self.shape[1] != 1 else S.One
            return [_LeftRightArgs(
                [first, second],
            )]


def matrix_symbols(expr):
    return [sym for sym in expr.free_symbols if sym.is_Matrix]


class _LeftRightArgs:
    r"""
    Helper class to compute matrix derivatives.

    The logic: when an expression is derived by a matrix `X_{mn}`, two lines of
    matrix multiplications are created: the one contracted to `m` (first line),
    and the one contracted to `n` (second line).

    Transposition flips the side by which new matrices are connected to the
    lines.

    The trace connects the end of the two lines.
    """

    def __init__(self, lines, higher=S.One):
        self._lines = list(lines)
        self._first_pointer_parent = self._lines
        self._first_pointer_index = 0
        self._first_line_index = 0
        self._second_pointer_parent = self._lines
        self._second_pointer_index = 1
        self._second_line_index = 1
        self.higher = higher

    @property
    def first_pointer(self):
        return self._first_pointer_parent[self._first_pointer_index]

    @first_pointer.setter
    def first_pointer(self, value):
        self._first_pointer_parent[self._first_pointer_index] = value

    @property
    def second_pointer(self):
        return self._second_pointer_parent[self._second_pointer_index]

    @second_pointer.setter
    def second_pointer(self, value):
        self._second_pointer_parent[self._second_pointer_index] = value

    def __repr__(self):
        built = [self._build(i) for i in self._lines]
        return "_LeftRightArgs(lines=%s, higher=%s)" % (
            built,
            self.higher,
        )

    def transpose(self):
        self._first_pointer_parent, self._second_pointer_parent = self._second_pointer_parent, self._first_pointer_parent
        self._first_pointer_index, self._second_pointer_index = self._second_pointer_index, self._first_pointer_index
        self._first_line_index, self._second_line_index = self._second_line_index, self._first_line_index
        return self

    @staticmethod
    def _build(expr):
        if isinstance(expr, ExprBuilder):
            return expr.build()
        if isinstance(expr, list):
            if len(expr) == 1:
                return expr[0]
            else:
                return expr[0](*[_LeftRightArgs._build(i) for i in expr[1]])
        else:
            return expr

    def build(self):
        data = [self._build(i) for i in self._lines]
        if self.higher != 1:
            data += [self._build(self.higher)]
        data = list(data)
        return data

    def matrix_form(self):
        if self.first != 1 and self.higher != 1:
            raise ValueError("higher dimensional array cannot be represented")

        def _get_shape(elem):
            if isinstance(elem, MatrixExpr):
                return elem.shape
            return (None, None)

        if _get_shape(self.first)[1] != _get_shape(self.second)[1]:
            # Remove one-dimensional identity matrices:
            # (this is needed by `a.diff(a)` where `a` is a vector)
            if _get_shape(self.second) == (1, 1):
                return self.first*self.second[0, 0]
            if _get_shape(self.first) == (1, 1):
                return self.first[1, 1]*self.second.T
            raise ValueError("incompatible shapes")
        if self.first != 1:
            return self.first*self.second.T
        else:
            return self.higher

    def rank(self):
        """
        Number of dimensions different from trivial (warning: not related to
        matrix rank).
        """
        rank = 0
        if self.first != 1:
            rank += sum(i != 1 for i in self.first.shape)
        if self.second != 1:
            rank += sum(i != 1 for i in self.second.shape)
        if self.higher != 1:
            rank += 2
        return rank

    def _multiply_pointer(self, pointer, other):
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from ...tensor.array.expressions.array_expressions import ArrayContraction

        subexpr = ExprBuilder(
            ArrayContraction,
            [
                ExprBuilder(
                    ArrayTensorProduct,
                    [
                        pointer,
                        other
                    ]
                ),
                (1, 2)
            ],
            validator=ArrayContraction._validate
        )

        return subexpr

    def append_first(self, other):
        self.first_pointer *= other

    def append_second(self, other):
        self.second_pointer *= other


def _make_matrix(x):
    from sympy.matrices.immutable import ImmutableDenseMatrix
    if isinstance(x, MatrixExpr):
        return x
    return ImmutableDenseMatrix([[x]])


from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
<|endoftext|>from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
        do_one, new)
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices.matrixbase import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate

from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix


# XXX: MatMul should perhaps not subclass directly from Mul
class MatMul(MatrixExpr, Mul):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True

    identity = GenericIdentity()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity

        # This must be removed aggressively in the constructor to avoid
        # TypeErrors from GenericIdentity().shape
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        obj = Basic.__new__(cls, *args)
        factor, matrices = obj.as_coeff_matrices()

        if check is not None:
            sympy_deprecation_warning(
                "Passing check to MatMul is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        if check is not False:
            validate(*matrices)

        if not matrices:
            # Should it be
            #
            # return Basic.__neq__(cls, factor, GenericIdentity()) ?
            return factor

        if evaluate:
            return cls._evaluate(obj)

        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    @property
    def shape(self):
        matrices = [arg for arg in self.args if arg.is_Matrix]
        return (matrices[0].rows, matrices[-1].cols)

    def _entry(self, i, j, expand=True, **kwargs):
        # Avoid cyclic imports
        from sympy.concrete.summations import Sum
        from sympy.matrices.immutable import ImmutableMatrix

        coeff, matrices = self.as_coeff_matrices()

        if len(matrices) == 1:  # situation like 2*X, matmul is just X
            return coeff * matrices[0][i, j]

        indices = [None]*(len(matrices) + 1)
        ind_ranges = [None]*(len(matrices) - 1)
        indices[0] = i
        indices[-1] = j

        def f():
            counter = 1
            while True:
                yield Dummy("i_%i" % counter)
                counter += 1

        dummy_generator = kwargs.get("dummy_generator", f())

        for i in range(1, len(matrices)):
            indices[i] = next(dummy_generator)

        for i, arg in enumerate(matrices[:-1]):
            ind_ranges[i] = arg.shape[1] - 1
        matrices = [arg._entry(indices[i], indices[i+1], dummy_generator=dummy_generator) for i, arg in enumerate(matrices)]
        expr_in_sum = Mul.fromiter(matrices)
        if any(v.has(ImmutableMatrix) for v in matrices):
            expand = True
        result = coeff*Sum(
                expr_in_sum,
                *zip(indices[1:-1], [0]*len(ind_ranges), ind_ranges)
            )

        # Don't waste time in result.doit() if the sum bounds are symbolic
        if not any(isinstance(v, (Integer, int)) for v in ind_ranges):
            expand = False
        return result.doit() if expand else result

    def as_coeff_matrices(self):
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)
        if coeff.is_commutative is False:
            raise NotImplementedError("noncommutative scalars in MatMul are not supported.")

        return coeff, matrices

    def as_coeff_mmul(self):
        coeff, matrices = self.as_coeff_matrices()
        return coeff, MatMul(*matrices)

    def expand(self, **kwargs):
        expanded = super(MatMul, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _eval_transpose(self):
        """Transposition of matrix multiplication.

        Notes
        =====

        The following rules are applied.

        Transposition for matrix multiplied with another matrix:
        `\\left(A B\\right)^{T} = B^{T} A^{T}`

        Transposition for matrix multiplied with scalar:
        `\\left(c A\\right)^{T} = c A^{T}`

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transpose
        """
        coeff, matrices = self.as_coeff_matrices()
        return MatMul(
            coeff, *[transpose(arg) for arg in matrices[::-1]]).doit()

    def _eval_adjoint(self):
        return MatMul(*[adjoint(arg) for arg in self.args[::-1]]).doit()

    def _eval_trace(self):
        factor, mmul = self.as_coeff_mmul()
        if factor != 1:
            from .trace import trace
            return factor * trace(mmul.doit())

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import Determinant
        factor, matrices = self.as_coeff_matrices()
        square_matrices = only_squares(*matrices)
        return factor**self.rows * Mul(*list(map(Determinant, square_matrices)))

    def _eval_inverse(self):
        if all(arg.is_square for arg in self.args if isinstance(arg, MatrixExpr)):
            return MatMul(*(
                arg.inverse() if isinstance(arg, MatrixExpr) else arg**-1
                    for arg in self.args[::-1]
                )
            ).doit()
        return Inverse(self)

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = tuple(arg.doit(**hints) for arg in self.args)
        else:
            args = self.args

        # treat scalar*MatrixSymbol or scalar*MatPow separately
        expr = canonicalize(MatMul(*args))
        return expr

    # Needed for partial compatibility with Mul
    def args_cnc(self, cset=False, warn=True, **kwargs):
        coeff_c = [x for x in self.args if x.is_commutative]
        coeff_nc = [x for x in self.args if not x.is_commutative]
        if cset:
            clen = len(coeff_c)
            coeff_c = set(coeff_c)
            if clen and warn and len(coeff_c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in coeff_c if list(self.args).count(ci) > 1])
        return [coeff_c, coeff_nc]

    def _eval_derivative_matrix_lines(self, x):
        from .transpose import Transpose
        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind+1:]

            if right_args:
                right_mat = MatMul.fromiter(right_args)
            else:
                right_mat = Identity(self.shape[1])
            if left_args:
                left_rev = MatMul.fromiter([Transpose(i).doit() if i.is_Matrix else i for i in reversed(left_args)])
            else:
                left_rev = Identity(self.shape[0])

            d = self.args[ind]._eval_derivative_matrix_lines(x)
            for i in d:
                i.append_first(left_rev)
                i.append_second(right_mat)
                lines.append(i)

        return lines

mul.register_handlerclass((Mul, MatMul), MatMul)


# Rules
def newmul(*args):
    if args[0] == 1:
        args = args[1:]
    return new(MatMul, *args)

def any_zeros(mul):
    if any(arg.is_zero or (arg.is_Matrix and arg.is_ZeroMatrix)
                       for arg in mul.args):
        matrices = [arg for arg in mul.args if arg.is_Matrix]
        return ZeroMatrix(matrices[0].rows, matrices[-1].cols)
    return mul

def merge_explicit(matmul):
    """ Merge explicit MatrixBase arguments

    >>> from sympy import MatrixSymbol, Matrix, MatMul, pprint
    >>> from sympy.matrices.expressions.matmul import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = Matrix([[1, 1], [1, 1]])
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatMul(A, B, C)
    >>> pprint(X)
      [1  1] [1  2]
    A*[    ]*[    ]
      [1  1] [3  4]
    >>> pprint(merge_explicit(X))
      [4  6]
    A*[    ]
      [4  6]

    >>> X = MatMul(B, A, C)
    >>> pprint(X)
    [1  1]   [1  2]
    [    ]*A*[    ]
    [1  1]   [3  4]
    >>> pprint(merge_explicit(X))
    [1  1]   [1  2]
    [    ]*A*[    ]
    [1  1]   [3  4]
    """
    if not any(isinstance(arg, MatrixBase) for arg in matmul.args):
        return matmul
    newargs = []
    last = matmul.args[0]
    for arg in matmul.args[1:]:
        if isinstance(arg, (MatrixBase, Number)) and isinstance(last, (MatrixBase, Number)):
            last = last * arg
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)

    return MatMul(*newargs)

def remove_ids(mul):
    """ Remove Identities from a MatMul

    This is a modified version of sympy.strategies.rm_id.
    This is necessary because MatMul may contain both MatrixExprs and Exprs
    as args.

    See Also
    ========

    sympy.strategies.rm_id
    """
    # Separate Exprs from MatrixExprs in args
    factor, mmul = mul.as_coeff_mmul()
    # Apply standard rm_id for MatMuls
    result = rm_id(lambda x: x.is_Identity is True)(mmul)
    if result != mmul:
        return newmul(factor, *result.args)  # Recombine and return
    else:
        return mul

def factor_in_front(mul):
    factor, matrices = mul.as_coeff_matrices()
    if factor != 1:
        return newmul(factor, *matrices)
    return mul

def combine_powers(mul):
    r"""Combine consecutive powers with the same base into one, e.g.
    $$A \times A^2 \Rightarrow A^3$$

    This also cancels out the possible matrix inverses using the
    knowledgebase of :class:`~.Inverse`, e.g.,
    $$ Y \times X \times X^{-1} \Rightarrow Y $$
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]

    for i in range(1, len(args)):
        A = new_args[-1]
        B = args[i]

        if isinstance(B, Inverse) and isinstance(B.arg, MatMul):
            Bargs = B.arg.args
            l = len(Bargs)
            if list(Bargs) == new_args[-l:]:
                new_args = new_args[:-l] + [Identity(B.shape[0])]
                continue

        if isinstance(A, Inverse) and isinstance(A.arg, MatMul):
            Aargs = A.arg.args
            l = len(Aargs)
            if list(Aargs) == args[i:i+l]:
                identity = Identity(A.shape[0])
                new_args[-1] = identity
                for j in range(i, i+l):
                    args[j] = identity
                continue

        if A.is_square == False or B.is_square == False:
            new_args.append(B)
            continue

        if isinstance(A, MatPow):
            A_base, A_exp = A.args
        else:
            A_base, A_exp = A, S.One

        if isinstance(B, MatPow):
            B_base, B_exp = B.args
        else:
            B_base, B_exp = B, S.One

        if A_base == B_base:
            new_exp = A_exp + B_exp
            new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
            continue
        elif not isinstance(B_base, MatrixBase):
            try:
                B_base_inv = B_base.inverse()
            except NonInvertibleMatrixError:
                B_base_inv = None
            if B_base_inv is not None and A_base == B_base_inv:
                new_exp = A_exp - B_exp
                new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
                continue
        new_args.append(B)

    return newmul(factor, *new_args)

def combine_permutations(mul):
    """Refine products of permutation matrices as the products of cycles.
    """
    args = mul.args
    l = len(args)
    if l < 2:
        return mul

    result = [args[0]]
    for i in range(1, l):
        A = result[-1]
        B = args[i]
        if isinstance(A, PermutationMatrix) and \
            isinstance(B, PermutationMatrix):
            cycle_1 = A.args[0]
            cycle_2 = B.args[0]
            result[-1] = PermutationMatrix(cycle_1 * cycle_2)
        else:
            result.append(B)

    return MatMul(*result)

def combine_one_matrices(mul):
    """
    Combine products of OneMatrix

    e.g. OneMatrix(2, 3) * OneMatrix(3, 4) -> 3 * OneMatrix(2, 4)
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]

    for B in args[1:]:
        A = new_args[-1]
        if not isinstance(A, OneMatrix) or not isinstance(B, OneMatrix):
            new_args.append(B)
            continue
        new_args.pop()
        new_args.append(OneMatrix(A.shape[0], B.shape[1]))
        factor *= A.shape[1]

    return newmul(factor, *new_args)

def distribute_monom(mul):
    """
    Simplify MatMul expressions but distributing
    rational term to MatMul.

    e.g. 2*(A+B) -> 2*A + 2*B
    """
    args = mul.args
    if len(args) == 2:
        from .matadd import MatAdd
        if args[0].is_MatAdd and args[1].is_Rational:
            return MatAdd(*[MatMul(mat, args[1]).doit() for mat in args[0].args])
        if args[1].is_MatAdd and args[0].is_Rational:
            return MatAdd(*[MatMul(args[0], mat).doit() for mat in args[1].args])
    return mul

rules = (
    distribute_monom, any_zeros, remove_ids, combine_one_matrices, combine_powers, unpack, rm_id(lambda x: x == 1),
    merge_explicit, factor_in_front, flatten, combine_permutations)

canonicalize = exhaust(typed({MatMul: do_one(*rules)}))

def only_squares(*matrices):
    """factor matrices only if they are square"""
    if matrices[0].rows != matrices[-1].cols:
        raise RuntimeError("Invalid matrices being multiplied")
    out = []
    start = 0
    for i, M in enumerate(matrices):
        if M.cols == matrices[start].rows:
            out.append(MatMul(*matrices[start:i+1]).doit())
            start = i+1
    return out


def refine_MatMul(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> expr = X * X.T
    >>> print(expr)
    X*X.T
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(expr))
    I
    """
    newargs = []
    exprargs = []

    for args in expr.args:
        if args.is_Matrix:
            exprargs.append(args)
        else:
            newargs.append(args)

    last = exprargs[0]
    for arg in exprargs[1:]:
        if arg == last.T and ask(Q.orthogonal(arg), assumptions):
            last = Identity(arg.shape[0])
        elif arg == last.conjugate() and ask(Q.unitary(arg), assumptions):
            last = Identity(arg.shape[0])
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)

    return MatMul(*newargs)


handlers_dict['MatMul'] = refine_MatMul
<|endoftext|>from sympy.core.sympify import _sympify

from sympy.matrices.expressions import MatrixExpr
from sympy.core import S, Eq, Ge
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta


class DiagonalMatrix(MatrixExpr):
    """DiagonalMatrix(M) will create a matrix expression that
    behaves as though all off-diagonal elements,
    `M[i, j]` where `i != j`, are zero.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalMatrix, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> D = DiagonalMatrix(MatrixSymbol('x', 2, 3))
    >>> D[1, 2]
    0
    >>> D[1, 1]
    x[1, 1]

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> D.diagonal_length
    2
    >>> DiagonalMatrix(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When one of the dimensions is symbolic the other will be treated as
    though it is smaller:

    >>> tall = DiagonalMatrix(MatrixSymbol('x', n, 3))
    >>> tall.diagonal_length
    3
    >>> tall[10, 1]
    0

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalMatrix(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])

    shape = property(lambda self: self.arg.shape)  # type:ignore

    @property
    def diagonal_length(self):
        r, c = self.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m

    def _entry(self, i, j, **kwargs):
        if self.diagonal_length is not None:
            if Ge(i, self.diagonal_length) is S.true:
                return S.Zero
            elif Ge(j, self.diagonal_length) is S.true:
                return S.Zero
        eq = Eq(i, j)
        if eq is S.true:
            return self.arg[i, i]
        elif eq is S.false:
            return S.Zero
        return self.arg[i, j]*KroneckerDelta(i, j)


class DiagonalOf(MatrixExpr):
    """DiagonalOf(M) will create a matrix expression that
    is equivalent to the diagonal of `M`, represented as
    a single column matrix.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalOf, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> x = MatrixSymbol('x', 2, 3)
    >>> diag = DiagonalOf(x)
    >>> diag.shape
    (2, 1)

    The diagonal can be addressed like a matrix or vector and will
    return the corresponding element of the original matrix:

    >>> diag[1, 0] == diag[1] == x[1, 1]
    True

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> diag.diagonal_length
    2
    >>> DiagonalOf(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When only one of the dimensions is symbolic the other will be
    treated as though it is smaller:

    >>> dtall = DiagonalOf(MatrixSymbol('x', n, 3))
    >>> dtall.diagonal_length
    3

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalOf(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])
    @property
    def shape(self):
        r, c = self.arg.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m, S.One

    @property
    def diagonal_length(self):
        return self.shape[0]

    def _entry(self, i, j, **kwargs):
        return self.arg._entry(i, i, **kwargs)


class DiagMatrix(MatrixExpr):
    """
    Turn a vector into a diagonal matrix.
    """
    def __new__(cls, vector):
        vector = _sympify(vector)
        obj = MatrixExpr.__new__(cls, vector)
        shape = vector.shape
        dim = shape[1] if shape[0] == 1 else shape[0]
        if vector.shape[0] != 1:
            obj._iscolumn = True
        else:
            obj._iscolumn = False
        obj._shape = (dim, dim)
        obj._vector = vector
        return obj

    @property
    def shape(self):
        return self._shape

    def _entry(self, i, j, **kwargs):
        if self._iscolumn:
            result = self._vector._entry(i, 0, **kwargs)
        else:
            result = self._vector._entry(0, j, **kwargs)
        if i != j:
            result *= KroneckerDelta(i, j)
        return result

    def _eval_transpose(self):
        return self

    def as_explicit(self):
        from sympy.matrices.dense import diag
        return diag(*list(self._vector.as_explicit()))

    def doit(self, **hints):
        from sympy.assumptions import ask, Q
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        from sympy.matrices.dense import eye
        from sympy.matrices.matrixbase import MatrixBase
        vector = self._vector
        # This accounts for shape (1, 1) and identity matrices, among others:
        if ask(Q.diagonal(vector)):
            return vector
        if isinstance(vector, MatrixBase):
            ret = eye(max(vector.shape))
            for i in range(ret.shape[0]):
                ret[i, i] = vector[i]
            return type(vector)(ret)
        if vector.is_MatMul:
            matrices = [arg for arg in vector.args if arg.is_Matrix]
            scalars = [arg for arg in vector.args if arg not in matrices]
            if scalars:
                return Mul.fromiter(scalars)*DiagMatrix(MatMul.fromiter(matrices).doit()).doit()
        if isinstance(vector, Transpose):
            vector = vector.arg
        return DiagMatrix(vector)


def diagonalize_vector(vector):
    return DiagMatrix(vector).doit()
<|endoftext|>""" A module which handles Matrix Expressions """

from .slice import MatrixSlice
from .blockmatrix import BlockMatrix, BlockDiagMatrix, block_collapse, blockcut
from .companion import CompanionMatrix
from .funcmatrix import FunctionMatrix
from .inverse import Inverse
from .matadd import MatAdd
from .matexpr import MatrixExpr, MatrixSymbol, matrix_symbols
from .matmul import MatMul
from .matpow import MatPow
from .trace import Trace, trace
from .determinant import Determinant, det, Permanent, per
from .transpose import Transpose
from .adjoint import Adjoint
from .hadamard import hadamard_product, HadamardProduct, hadamard_power, HadamardPower
from .diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
from .dotproduct import DotProduct
from .kronecker import kronecker_product, KroneckerProduct, combine_kronecker
from .permutation import PermutationMatrix, MatrixPermute
from .sets import MatrixSet
from .special import ZeroMatrix, Identity, OneMatrix

__all__ = [
    'MatrixSlice',

    'BlockMatrix', 'BlockDiagMatrix', 'block_collapse', 'blockcut',
    'FunctionMatrix',

    'CompanionMatrix',

    'Inverse',

    'MatAdd',

    'Identity', 'MatrixExpr', 'MatrixSymbol', 'ZeroMatrix', 'OneMatrix',
    'matrix_symbols', 'MatrixSet',

    'MatMul',

    'MatPow',

    'Trace', 'trace',

    'Determinant', 'det',

    'Transpose',

    'Adjoint',

    'hadamard_product', 'HadamardProduct', 'hadamard_power', 'HadamardPower',

    'DiagonalMatrix', 'DiagonalOf', 'DiagMatrix', 'diagonalize_vector',

    'DotProduct',

    'kronecker_product', 'KroneckerProduct', 'combine_kronecker',

    'PermutationMatrix', 'MatrixPermute',

    'Permanent', 'per'
]
<|endoftext|>from sympy.core import Basic
from sympy.functions import adjoint, conjugate
from sympy.matrices.expressions.matexpr import MatrixExpr


class Adjoint(MatrixExpr):
    """
    The Hermitian adjoint of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the adjoint, use the ``adjoint()``
    function.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Adjoint, adjoint
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Adjoint(A*B)
    Adjoint(A*B)
    >>> adjoint(A*B)
    Adjoint(B)*Adjoint(A)
    >>> adjoint(A*B) == Adjoint(A*B)
    False
    >>> adjoint(A*B) == Adjoint(A*B).doit()
    True
    """
    is_Adjoint = True

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            return adjoint(arg.doit(**hints))
        else:
            return adjoint(self.arg)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape[::-1]

    def _entry(self, i, j, **kwargs):
        return conjugate(self.arg._entry(j, i, **kwargs))

    def _eval_adjoint(self):
        return self.arg

    def _eval_transpose(self):
        return self.arg.conjugate()

    def _eval_conjugate(self):
        return self.arg.transpose()

    def _eval_trace(self):
        from sympy.matrices.expressions.trace import Trace
        return conjugate(Trace(self.arg))
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.matrixbase import MatrixBase


class Determinant(Expr):
    """Matrix Determinant

    Represents the determinant of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Determinant, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Determinant(A)
    Determinant(A)
    >>> Determinant(eye(3)).doit()
    1
    """
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError("Input to Determinant, %s, not a matrix" % str(mat))

        if mat.is_square is False:
            raise NonSquareMatrixError("Det of a non-square matrix")

        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    @property
    def kind(self):
        return self.arg.kind.element_kind

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True):
            arg = arg.doit(**hints)

        result = arg._eval_determinant()
        if result is not None:
            return result

        return self


def det(matexpr):
    """ Matrix Determinant

    Examples
    ========

    >>> from sympy import MatrixSymbol, det, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> det(A)
    Determinant(A)
    >>> det(eye(3))
    1
    """

    return Determinant(matexpr).doit()

class Permanent(Expr):
    """Matrix Permanent

    Represents the permanent of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Permanent, ones
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Permanent(A)
    Permanent(A)
    >>> Permanent(ones(3, 3)).doit()
    6
    """

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError("Input to Permanent, %s, not a matrix" % str(mat))

        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    def doit(self, expand=False, **hints):
        if isinstance(self.arg, MatrixBase):
            return self.arg.per()
        else:
            return self

def per(matexpr):
    """ Matrix Permanent

    Examples
    ========

    >>> from sympy import MatrixSymbol, Matrix, per, ones
    >>> A = MatrixSymbol('A', 3, 3)
    >>> per(A)
    Permanent(A)
    >>> per(ones(5, 5))
    120
    >>> M = Matrix([1, 2, 5])
    >>> per(M)
    8
    """

    return Permanent(matexpr).doit()

from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict


def refine_Determinant(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine, det
    >>> X = MatrixSymbol('X', 2, 2)
    >>> det(X)
    Determinant(X)
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(det(X)))
    1
    """
    if ask(Q.orthogonal(expr.arg), assumptions):
        return S.One
    elif ask(Q.singular(expr.arg), assumptions):
        return S.Zero
    elif ask(Q.unit_triangular(expr.arg), assumptions):
        return S.One

    return expr


handlers_dict['Determinant'] = refine_Determinant
<|endoftext|>from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt


class DFT(MatrixExpr):
    r"""
    Returns a discrete Fourier transform matrix. The matrix is scaled
    with :math:`\frac{1}{\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy.matrices.expressions.fourier import DFT
    >>> DFT(3)
    DFT(3)
    >>> DFT(3).as_explicit()
    Matrix([
    [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3],
    [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3]])
    >>> DFT(n).shape
    (n, n)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/DFT_matrix

    """

    def __new__(cls, n):
        n = _sympify(n)
        cls._check_dim(n)

        obj = super().__new__(cls, n)
        return obj

    n = property(lambda self: self.args[0])  # type: ignore
    shape = property(lambda self: (self.n, self.n))  # type: ignore

    def _entry(self, i, j, **kwargs):
        w = exp(-2*S.Pi*I/self.n)
        return w**(i*j) / sqrt(self.n)

    def _eval_inverse(self):
        return IDFT(self.n)


class IDFT(DFT):
    r"""
    Returns an inverse discrete Fourier transform matrix. The matrix is scaled
    with :math:`\frac{1}{\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform

    Examples
    ========

    >>> from sympy.matrices.expressions.fourier import DFT, IDFT
    >>> IDFT(3)
    IDFT(3)
    >>> IDFT(4)*DFT(4)
    I

    See Also
    ========

    DFT

    """
    def _entry(self, i, j, **kwargs):
        w = exp(-2*S.Pi*I/self.n)
        return w**(-i*j) / sqrt(self.n)

    def _eval_inverse(self):
        return DFT(self.n)
<|endoftext|>from sympy.assumptions.ask import (Q, ask)
from sympy.core import Basic, Add, Mul, S
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import re, im
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent

from sympy.matrices import Matrix, ShapeError
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.matrices.expressions.trace import trace
from sympy.matrices.expressions.transpose import Transpose, transpose


class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrixbase.MatrixBase.irregular
    """
    def __new__(cls, *args, **kwargs):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        if len(args) != 1 or \
                not is_sequence(args[0]) or \
                len({isMat(r) for r in args[0]}) != 1:
            raise ValueError(filldedent('''
                expecting a sequence of 1 or more rows
                containing Matrices.'''))
        rows = args[0] if args else []
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]  # rows is not list of lists or []
            # regularity check
            # same number of matrices in each row
            blocky = ok = len({len(r) for r in rows}) == 1
            if ok:
                # same number of rows for each matrix in a row
                for r in rows:
                    ok = len({i.rows for i in r}) == 1
                    if not ok:
                        break
                blocky = ok
                if ok:
                    # same number of cols for each matrix in each col
                    for c in range(len(rows[0])):
                        ok = len({rows[i][c].cols
                            for i in range(len(rows))}) == 1
                        if not ok:
                            break
            if not ok:
                # same total cols in each row
                ok = len({
                    sum(i.cols for i in r) for r in rows}) == 1
                if blocky and ok:
                    raise ValueError(filldedent('''
                        Although this matrix is comprised of blocks,
                        the blocks do not fill the matrix in a
                        size-symmetric fashion. To create a full matrix
                        from these arguments, pass them directly to
                        Matrix.'''))
                raise ValueError(filldedent('''
                    When there are not the same number of rows in each
                    row's matrices or there are not the same number of
                    total columns in each row, the matrix is not a
                    block matrix. If this matrix is known to consist of
                    blocks fully filling a 2-D space then see
                    Matrix.irregular.'''))
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        obj = Basic.__new__(cls, mat)
        return obj

    @property
    def shape(self):
        numrows = numcols = 0
        M = self.blocks
        for i in range(M.shape[0]):
            numrows += M[i, 0].shape[0]
        for i in range(M.shape[1]):
            numcols += M[0, i].shape[1]
        return (numrows, numcols)

    @property
    def blockshape(self):
        return self.blocks.shape

    @property
    def blocks(self):
        return self.args[0]

    @property
    def rowblocksizes(self):
        return [self.blocks[i, 0].rows for i in range(self.blockshape[0])]

    @property
    def colblocksizes(self):
        return [self.blocks[0, i].cols for i in range(self.blockshape[1])]

    def structurally_equal(self, other):
        return (isinstance(other, BlockMatrix)
            and self.shape == other.shape
            and self.blockshape == other.blockshape
            and self.rowblocksizes == other.rowblocksizes
            and self.colblocksizes == other.colblocksizes)

    def _blockmul(self, other):
        if (isinstance(other, BlockMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockMatrix(self.blocks*other.blocks)

        return self * other

    def _blockadd(self, other):
        if (isinstance(other, BlockMatrix)
                and self.structurally_equal(other)):
            return BlockMatrix(self.blocks + other.blocks)

        return self + other

    def _eval_transpose(self):
        # Flip all the individual matrices
        matrices = [transpose(matrix) for matrix in self.blocks]
        # Make a copy
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        # Transpose the block structure
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_adjoint(self):
        return BlockMatrix(
            Matrix(self.blockshape[0], self.blockshape[1], self.blocks).adjoint()
        )

    def _eval_trace(self):
        if self.rowblocksizes == self.colblocksizes:
            blocks = [self.blocks[i, i] for i in range(self.blockshape[0])]
            return Add(*[trace(block) for block in blocks])

    def _eval_determinant(self):
        if self.blockshape == (1, 1):
            return det(self.blocks[0, 0])
        if self.blockshape == (2, 2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            if ask(Q.invertible(A)):
                return det(A)*det(D - C*A.I*B)
            elif ask(Q.invertible(D)):
                return det(D)*det(A - B*D.I*C)
        return Determinant(self)

    def _eval_as_real_imag(self):
        real_matrices = [re(matrix) for matrix in self.blocks]
        real_matrices = Matrix(self.blockshape[0], self.blockshape[1], real_matrices)

        im_matrices = [im(matrix) for matrix in self.blocks]
        im_matrices = Matrix(self.blockshape[0], self.blockshape[1], im_matrices)

        return (BlockMatrix(real_matrices), BlockMatrix(im_matrices))

    def _eval_derivative(self, x):
        return BlockMatrix(self.blocks.diff(x))

    def transpose(self):
        """Return transpose of matrix.

        Examples
        ========

        >>> from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix
        >>> from sympy.abc import m, n
        >>> X = MatrixSymbol('X', n, n)
        >>> Y = MatrixSymbol('Y', m, m)
        >>> Z = MatrixSymbol('Z', n, m)
        >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
        >>> B.transpose()
        Matrix([
        [X.T,  0],
        [Z.T, Y.T]])
        >>> _.transpose()
        Matrix([
        [X, Z],
        [0, Y]])
        """
        return self._eval_transpose()

    def schur(self, mat = 'A', generalized = False):
        """Return the Schur Complement of the 2x2 BlockMatrix

        Parameters
        ==========

        mat : String, optional
            The matrix with respect to which the
            Schur Complement is calculated. 'A' is
            used by default

        generalized : bool, optional
            If True, returns the generalized Schur
            Component which uses Moore-Penrose Inverse

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])

        The default Schur Complement is evaluated with "A"

        >>> X.schur()
        -C*A**(-1)*B + D
        >>> X.schur('D')
        A - B*D**(-1)*C

        Schur complement with non-invertible matrices is not
        defined. Instead, the generalized Schur complement can
        be calculated which uses the Moore-Penrose Inverse. To
        achieve this, `generalized` must be set to `True`

        >>> X.schur('B', generalized=True)
        C - D*(B.T*B)**(-1)*B.T*A
        >>> X.schur('C', generalized=True)
        -A*(C.T*C)**(-1)*C.T*D + B

        Returns
        =======

        M : Matrix
            The Schur Complement Matrix

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If given matrix is non-invertible

        References
        ==========

        .. [1] Wikipedia Article on Schur Component : https://en.wikipedia.org/wiki/Schur_complement

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.pinv
        """

        if self.blockshape == (2, 2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            d={'A' : A, 'B' : B, 'C' : C, 'D' : D}
            try:
                inv = (d[mat].T*d[mat]).inv()*d[mat].T if generalized else d[mat].inv()
                if mat == 'A':
                    return D - C * inv * B
                elif mat == 'B':
                    return C - D * inv * A
                elif mat == 'C':
                    return B - A * inv * D
                elif mat == 'D':
                    return A - B * inv * C
                #For matrices where no sub-matrix is square
                return self
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('The given matrix is not invertible. Please set generalized=True \
            to compute the generalized Schur Complement which uses Moore-Penrose Inverse')
        else:
            raise ShapeError('Schur Complement can only be calculated for 2x2 block matrices')

    def LDUdecomposition(self):
        """Returns the Block LDU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, D, U) : Matrices
            L : Lower Diagonal Matrix
            D : Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, D, U = X.LDUdecomposition()
        >>> block_collapse(L*D*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LDU decomposition cannot be calculated when\
                    "A" is singular')
            Ip = Identity(B.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            L = BlockMatrix([[Ip, Z], [C*AI, Iq]])
            D = BlockDiagMatrix(A, self.schur())
            U = BlockMatrix([[Ip, AI*B],[Z.T, Iq]])
            return L, D, U
        else:
            raise ShapeError("Block LDU decomposition is supported only for 2x2 block matrices")

    def UDLdecomposition(self):
        """Returns the Block UDL decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (U, D, L) : Matrices
            U : Upper Diagonal Matrix
            D : Diagonal Matrix
            L : Lower Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> U, D, L = X.UDLdecomposition()
        >>> block_collapse(U*D*L)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "D" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                DI = D.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block UDL decomposition cannot be calculated when\
                    "D" is singular')
            Ip = Identity(A.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            U = BlockMatrix([[Ip, B*DI], [Z.T, Iq]])
            D = BlockDiagMatrix(self.schur('D'), D)
            L = BlockMatrix([[Ip, Z],[DI*C, Iq]])
            return U, D, L
        else:
            raise ShapeError("Block UDL decomposition is supported only for 2x2 block matrices")

    def LUdecomposition(self):
        """Returns the Block LU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, U) : Matrices
            L : Lower Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, U = X.LUdecomposition()
        >>> block_collapse(L*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                A = A**S.Half
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LU decomposition cannot be calculated when\
                    "A" is singular')
            Z = ZeroMatrix(*B.shape)
            Q = self.schur()**S.Half
            L = BlockMatrix([[A, Z], [C*AI, Q]])
            U = BlockMatrix([[A, AI*B],[Z.T, Q]])
            return L, U
        else:
            raise ShapeError("Block LU decomposition is supported only for 2x2 block matrices")

    def _entry(self, i, j, **kwargs):
        # Find row entry
        orig_i, orig_j = i, j
        for row_block, numrows in enumerate(self.rowblocksizes):
            cmp = i < numrows
            if cmp == True:
                break
            elif cmp == False:
                i -= numrows
            elif row_block < self.blockshape[0] - 1:
                # Can't tell which block and it's not the last one, return unevaluated
                return MatrixElement(self, orig_i, orig_j)
        for col_block, numcols in enumerate(self.colblocksizes):
            cmp = j < numcols
            if cmp == True:
                break
            elif cmp == False:
                j -= numcols
            elif col_block < self.blockshape[1] - 1:
                return MatrixElement(self, orig_i, orig_j)
        return self.blocks[row_block, col_block][i, j]

    @property
    def is_Identity(self):
        if self.blockshape[0] != self.blockshape[1]:
            return False
        for i in range(self.blockshape[0]):
            for j in range(self.blockshape[1]):
                if i==j and not self.blocks[i, j].is_Identity:
                    return False
                if i!=j and not self.blocks[i, j].is_ZeroMatrix:
                    return False
        return True

    @property
    def is_structurally_symmetric(self):
        return self.rowblocksizes == self.colblocksizes

    def equals(self, other):
        if self == other:
            return True
        if (isinstance(other, BlockMatrix) and self.blocks == other.blocks):
            return True
        return super().equals(other)


class BlockDiagMatrix(BlockMatrix):
    """A sparse matrix with block matrices along its diagonals

    Examples
    ========

    >>> from sympy import MatrixSymbol, BlockDiagMatrix, symbols
    >>> n, m, l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> BlockDiagMatrix(X, Y)
    Matrix([
    [X, 0],
    [0, Y]])

    Notes
    =====

    If you want to get the individual diagonal blocks, use
    :meth:`get_diag_blocks`.

    See Also
    ========

    sympy.matrices.dense.diag
    """
    def __new__(cls, *mats):
        return Basic.__new__(BlockDiagMatrix, *[_sympify(m) for m in mats])

    @property
    def diag(self):
        return self.args

    @property
    def blocks(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        mats = self.args
        data = [[mats[i] if i == j else ZeroMatrix(mats[i].rows, mats[j].cols)
                        for j in range(len(mats))]
                        for i in range(len(mats))]
        return ImmutableDenseMatrix(data, evaluate=False)

    @property
    def shape(self):
        return (sum(block.rows for block in self.args),
                sum(block.cols for block in self.args))

    @property
    def blockshape(self):
        n = len(self.args)
        return (n, n)

    @property
    def rowblocksizes(self):
        return [block.rows for block in self.args]

    @property
    def colblocksizes(self):
        return [block.cols for block in self.args]

    def _all_square_blocks(self):
        """Returns true if all blocks are square"""
        return all(mat.is_square for mat in self.args)

    def _eval_determinant(self):
        if self._all_square_blocks():
            return Mul(*[det(mat) for mat in self.args])
        # At least one block is non-square.  Since the entire matrix must be square we know there must
        # be at least two blocks in this matrix, in which case the entire matrix is necessarily rank-deficient
        return S.Zero

    def _eval_inverse(self, expand='ignored'):
        if self._all_square_blocks():
            return BlockDiagMatrix(*[mat.inverse() for mat in self.args])
        # See comment in _eval_determinant()
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_transpose(self):
        return BlockDiagMatrix(*[mat.transpose() for mat in self.args])

    def _blockmul(self, other):
        if (isinstance(other, BlockDiagMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockDiagMatrix(*[a*b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockmul(self, other)

    def _blockadd(self, other):
        if (isinstance(other, BlockDiagMatrix) and
                self.blockshape == other.blockshape and
                self.rowblocksizes == other.rowblocksizes and
                self.colblocksizes == other.colblocksizes):
            return BlockDiagMatrix(*[a + b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockadd(self, other)

    def get_diag_blocks(self):
        """Return the list of diagonal blocks of the matrix.

        Examples
        ========

        >>> from sympy import BlockDiagMatrix, Matrix

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[5, 6], [7, 8]])
        >>> M = BlockDiagMatrix(A, B)

        How to get diagonal blocks from the block diagonal matrix:

        >>> diag_blocks = M.get_diag_blocks()
        >>> diag_blocks[0]
        Matrix([
        [1, 2],
        [3, 4]])
        >>> diag_blocks[1]
        Matrix([
        [5, 6],
        [7, 8]])
        """
        return self.args


def block_collapse(expr):
    """Evaluates a block matrix expression

    >>> from sympy import MatrixSymbol, BlockMatrix, symbols, Identity, ZeroMatrix, block_collapse
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m, n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])
    """
    from sympy.strategies.util import expr_fns

    hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)

    conditioned_rl = condition(
        hasbm,
        typed(
            {MatAdd: do_one(bc_matadd, bc_block_plus_ident),
             MatMul: do_one(bc_matmul, bc_dist),
             MatPow: bc_matmul,
             Transpose: bc_transpose,
             Inverse: bc_inverse,
             BlockMatrix: do_one(bc_unpack, deblock)}
        )
    )

    rule = exhaust(
        bottom_up(
            exhaust(conditioned_rl),
            fns=expr_fns
        )
    )

    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result

def bc_unpack(expr):
    if expr.blockshape == (1, 1):
        return expr.blocks[0, 0]
    return expr

def bc_matadd(expr):
    args = sift(expr.args, lambda M: isinstance(M, BlockMatrix))
    blocks = args[True]
    if not blocks:
        return expr

    nonblocks = args[False]
    block = blocks[0]
    for b in blocks[1:]:
        block = block._blockadd(b)
    if nonblocks:
        return MatAdd(*nonblocks) + block
    else:
        return block

def bc_block_plus_ident(expr):
    idents = [arg for arg in expr.args if arg.is_Identity]
    if not idents:
        return expr

    blocks = [arg for arg in expr.args if isinstance(arg, BlockMatrix)]
    if (blocks and all(b.structurally_equal(blocks[0]) for b in blocks)
               and blocks[0].is_structurally_symmetric):
        block_id = BlockDiagMatrix(*[Identity(k)
                                        for k in blocks[0].rowblocksizes])
        rest = [arg for arg in expr.args if not arg.is_Identity and not isinstance(arg, BlockMatrix)]
        return MatAdd(block_id * len(idents), *blocks, *rest).doit()

    return expr

def bc_dist(expr):
    """ Turn  a*[X, Y] into [a*X, a*Y] """
    factor, mat = expr.as_coeff_mmul()
    if factor == 1:
        return expr

    unpacked = unpack(mat)

    if isinstance(unpacked, BlockDiagMatrix):
        B = unpacked.diag
        new_B = [factor * mat for mat in B]
        return BlockDiagMatrix(*new_B)
    elif isinstance(unpacked, BlockMatrix):
        B = unpacked.blocks
        new_B = [
            [factor * B[i, j] for j in range(B.cols)] for i in range(B.rows)]
        return BlockMatrix(new_B)
    return expr


def bc_matmul(expr):
    if isinstance(expr, MatPow):
        if expr.args[1].is_Integer and expr.args[1] > 0:
            factor, matrices = 1, [expr.args[0]]*expr.args[1]
        else:
            return expr
    else:
        factor, matrices = expr.as_coeff_matrices()

    i = 0
    while (i+1 < len(matrices)):
        A, B = matrices[i:i+2]
        if isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix):
            matrices[i] = A._blockmul(B)
            matrices.pop(i+1)
        elif isinstance(A, BlockMatrix):
            matrices[i] = A._blockmul(BlockMatrix([[B]]))
            matrices.pop(i+1)
        elif isinstance(B, BlockMatrix):
            matrices[i] = BlockMatrix([[A]])._blockmul(B)
            matrices.pop(i+1)
        else:
            i+=1
    return MatMul(factor, *matrices).doit()

def bc_transpose(expr):
    collapse = block_collapse(expr.arg)
    return collapse._eval_transpose()


def bc_inverse(expr):
    if isinstance(expr.arg, BlockDiagMatrix):
        return expr.inverse()

    expr2 = blockinverse_1x1(expr)
    if expr != expr2:
        return expr2
    return blockinverse_2x2(Inverse(reblock_2x2(expr.arg)))

def blockinverse_1x1(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (1, 1):
        mat = Matrix([[expr.arg.blocks[0].inverse()]])
        return BlockMatrix(mat)
    return expr


def blockinverse_2x2(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (2, 2):
        # See: Inverses of 2x2 Block Matrices, Tzon-Tzer Lu and Sheng-Hua Shiou
        [[A, B],
         [C, D]] = expr.arg.blocks.tolist()

        formula = _choose_2x2_inversion_formula(A, B, C, D)
        if formula != None:
            MI = expr.arg.schur(formula).I
        if formula == 'A':
            AI = A.I
            return BlockMatrix([[AI + AI * B * MI * C * AI, -AI * B * MI], [-MI * C * AI, MI]])
        if formula == 'B':
            BI = B.I
            return BlockMatrix([[-MI * D * BI, MI], [BI + BI * A * MI * D * BI, -BI * A * MI]])
        if formula == 'C':
            CI = C.I
            return BlockMatrix([[-CI * D * MI, CI + CI * D * MI * A * CI], [MI, -MI * A * CI]])
        if formula == 'D':
            DI = D.I
            return BlockMatrix([[MI, -MI * B * DI], [-DI * C * MI, DI + DI * C * MI * B * DI]])

    return expr


def _choose_2x2_inversion_formula(A, B, C, D):
    """
    Assuming [[A, B], [C, D]] would form a valid square block matrix, find
    which of the classical 2x2 block matrix inversion formulas would be
    best suited.

    Returns 'A', 'B', 'C', 'D' to represent the algorithm involving inversion
    of the given argument or None if the matrix cannot be inverted using
    any of those formulas.
    """
    # Try to find a known invertible matrix.  Note that the Schur complement
    # is currently not being considered for this
    A_inv = ask(Q.invertible(A))
    if A_inv == True:
        return 'A'
    B_inv = ask(Q.invertible(B))
    if B_inv == True:
        return 'B'
    C_inv = ask(Q.invertible(C))
    if C_inv == True:
        return 'C'
    D_inv = ask(Q.invertible(D))
    if D_inv == True:
        return 'D'
    # Otherwise try to find a matrix that isn't known to be non-invertible
    if A_inv != False:
        return 'A'
    if B_inv != False:
        return 'B'
    if C_inv != False:
        return 'C'
    if D_inv != False:
        return 'D'
    return None


def deblock(B):
    """ Flatten a BlockMatrix of BlockMatrices """
    if not isinstance(B, BlockMatrix) or not B.blocks.has(BlockMatrix):
        return B
    wrap = lambda x: x if isinstance(x, BlockMatrix) else BlockMatrix([[x]])
    bb = B.blocks.applyfunc(wrap)  # everything is a block

    try:
        MM = Matrix(0, sum(bb[0, i].blocks.shape[1] for i in range(bb.shape[1])), [])
        for row in range(0, bb.shape[0]):
            M = Matrix(bb[row, 0].blocks)
            for col in range(1, bb.shape[1]):
                M = M.row_join(bb[row, col].blocks)
            MM = MM.col_join(M)

        return BlockMatrix(MM)
    except ShapeError:
        return B


def reblock_2x2(expr):
    """
    Reblock a BlockMatrix so that it has 2x2 blocks of block matrices.  If
    possible in such a way that the matrix continues to be invertible using the
    classical 2x2 block inversion formulas.
    """
    if not isinstance(expr, BlockMatrix) or not all(d > 2 for d in expr.blockshape):
        return expr

    BM = BlockMatrix  # for brevity's sake
    rowblocks, colblocks = expr.blockshape
    blocks = expr.blocks
    for i in range(1, rowblocks):
        for j in range(1, colblocks):
            # try to split rows at i and cols at j
            A = bc_unpack(BM(blocks[:i, :j]))
            B = bc_unpack(BM(blocks[:i, j:]))
            C = bc_unpack(BM(blocks[i:, :j]))
            D = bc_unpack(BM(blocks[i:, j:]))

            formula = _choose_2x2_inversion_formula(A, B, C, D)
            if formula is not None:
                return BlockMatrix([[A, B], [C, D]])

    # else: nothing worked, just split upper left corner
    return BM([[blocks[0, 0], BM(blocks[0, 1:])],
               [BM(blocks[1:, 0]), BM(blocks[1:, 1:])]])


def bounds(sizes):
    """ Convert sequence of numbers into pairs of low-high pairs

    >>> from sympy.matrices.expressions.blockmatrix import bounds
    >>> bounds((1, 10, 50))
    [(0, 1), (1, 11), (11, 61)]
    """
    low = 0
    rv = []
    for size in sizes:
        rv.append((low, low + size))
        low += size
    return rv

def blockcut(expr, rowsizes, colsizes):
    """ Cut a matrix expression into Blocks

    >>> from sympy import ImmutableMatrix, blockcut
    >>> M = ImmutableMatrix(4, 4, range(16))
    >>> B = blockcut(M, (1, 3), (1, 3))
    >>> type(B).__name__
    'BlockMatrix'
    >>> ImmutableMatrix(B.blocks[0, 1])
    Matrix([[1, 2, 3]])
    """

    rowbounds = bounds(rowsizes)
    colbounds = bounds(colsizes)
    return BlockMatrix([[MatrixSlice(expr, rowbound, colbound)
                         for colbound in colbounds]
                         for rowbound in rowbounds])
<|endoftext|>from functools import reduce
import operator

from sympy.core import Basic, sympify
from sympy.core.add import add, Add, _could_extract_minus_sign
from sympy.core.sorting import default_sort_key
from sympy.functions import adjoint
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import (rm_id, unpack, flatten, sort, condition,
    exhaust, do_one, glom)
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix, GenericZeroMatrix
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning

# XXX: MatAdd should perhaps not subclass directly from Add
class MatAdd(MatrixExpr, Add):
    """A Sum of Matrix Expressions

    MatAdd inherits from and operates like SymPy Add

    Examples
    ========

    >>> from sympy import MatAdd, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> C = MatrixSymbol('C', 5, 5)
    >>> MatAdd(A, B, C)
    A + B + C
    """
    is_MatAdd = True

    identity = GenericZeroMatrix()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity

        # This must be removed aggressively in the constructor to avoid
        # TypeErrors from GenericZeroMatrix().shape
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))

        if not all(isinstance(arg, MatrixExpr) for arg in args):
            raise TypeError("Mix of Matrix and Scalar symbols")

        obj = Basic.__new__(cls, *args)

        if check is not None:
            sympy_deprecation_warning(
                "Passing check to MatAdd is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        if check is not False:
            validate(*args)

        if evaluate:
            obj = cls._evaluate(obj)

        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    @property
    def shape(self):
        return self.args[0].shape

    def could_extract_minus_sign(self):
        return _could_extract_minus_sign(self)

    def expand(self, **kwargs):
        expanded = super(MatAdd, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _entry(self, i, j, **kwargs):
        return Add(*[arg._entry(i, j, **kwargs) for arg in self.args])

    def _eval_transpose(self):
        return MatAdd(*[transpose(arg) for arg in self.args]).doit()

    def _eval_adjoint(self):
        return MatAdd(*[adjoint(arg) for arg in self.args]).doit()

    def _eval_trace(self):
        from .trace import trace
        return Add(*[trace(arg) for arg in self.args]).doit()

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(MatAdd(*args))

    def _eval_derivative_matrix_lines(self, x):
        add_lines = [arg._eval_derivative_matrix_lines(x) for arg in self.args]
        return [j for i in add_lines for j in i]

add.register_handlerclass((Add, MatAdd), MatAdd)


factor_of = lambda arg: arg.as_coeff_mmul()[0]
matrix_of = lambda arg: unpack(arg.as_coeff_mmul()[1])
def combine(cnt, mat):
    if cnt == 1:
        return mat
    else:
        return cnt * mat


def merge_explicit(matadd):
    """ Merge explicit MatrixBase arguments

    Examples
    ========

    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint
    >>> from sympy.matrices.expressions.matadd import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = eye(2)
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatAdd(A, B, C)
    >>> pprint(X)
        [1  0]   [1  2]
    A + [    ] + [    ]
        [0  1]   [3  4]
    >>> pprint(merge_explicit(X))
        [2  2]
    A + [    ]
        [3  5]
    """
    groups = sift(matadd.args, lambda arg: isinstance(arg, MatrixBase))
    if len(groups[True]) > 1:
        return MatAdd(*(groups[False] + [reduce(operator.add, groups[True])]))
    else:
        return matadd


rules = (rm_id(lambda x: x == 0 or isinstance(x, ZeroMatrix)),
         unpack,
         flatten,
         glom(matrix_of, factor_of, combine),
         merge_explicit,
         sort(default_sort_key))

canonicalize = exhaust(condition(lambda x: isinstance(x, MatAdd),
                                 do_one(*rules)))
<|endoftext|>from sympy.core.assumptions import check_assumptions
from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.matrices.kind import MatrixKind
from sympy.sets.sets import Set, SetKind
from sympy.core.kind import NumberKind
from .matexpr import MatrixExpr


class MatrixSet(Set):
    """
    MatrixSet represents the set of matrices with ``shape = (n, m)`` over the
    given set.

    Examples
    ========

    >>> from sympy.matrices import MatrixSet
    >>> from sympy import S, I, Matrix
    >>> M = MatrixSet(2, 2, set=S.Reals)
    >>> X = Matrix([[1, 2], [3, 4]])
    >>> X in M
    True
    >>> X = Matrix([[1, 2], [I, 4]])
    >>> X in M
    False

    """
    is_empty = False

    def __new__(cls, n, m, set):
        n, m, set = _sympify(n), _sympify(m), _sympify(set)
        cls._check_dim(n)
        cls._check_dim(m)
        if not isinstance(set, Set):
            raise TypeError("{} should be an instance of Set.".format(set))
        return Set.__new__(cls, n, m, set)

    @property
    def shape(self):
        return self.args[:2]

    @property
    def set(self):
        return self.args[2]

    def _contains(self, other):
        if not isinstance(other, MatrixExpr):
            raise TypeError("{} should be an instance of MatrixExpr.".format(other))
        if other.shape != self.shape:
            are_symbolic = any(_sympify(x).is_Symbol for x in other.shape + self.shape)
            if are_symbolic:
                return None
            return False
        return fuzzy_and(self.set.contains(x) for x in other)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))

    def _kind(self):
        return SetKind(MatrixKind(NumberKind))
<|endoftext|>from sympy.core.sympify import _sympify
from sympy.core import S, Basic

from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions.matpow import MatPow


class Inverse(MatPow):
    """
    The multiplicative inverse of a matrix expression

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the inverse, use the ``.inverse()``
    method of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Inverse
    >>> A = MatrixSymbol('A', 3, 3)
    >>> B = MatrixSymbol('B', 3, 3)
    >>> Inverse(A)
    A**(-1)
    >>> A.inverse() == Inverse(A)
    True
    >>> (A*B).inverse()
    B**(-1)*A**(-1)
    >>> Inverse(A*B)
    (A*B)**(-1)

    """
    is_Inverse = True
    exp = S.NegativeOne

    def __new__(cls, mat, exp=S.NegativeOne):
        # exp is there to make it consistent with
        # inverse.func(*inverse.args) == inverse
        mat = _sympify(mat)
        exp = _sympify(exp)
        if not mat.is_Matrix:
            raise TypeError("mat should be a matrix")
        if mat.is_square is False:
            raise NonSquareMatrixError("Inverse of non-square matrix %s" % mat)
        return Basic.__new__(cls, mat, exp)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape

    def _eval_inverse(self):
        return self.arg

    def _eval_transpose(self):
        return Inverse(self.arg.transpose())

    def _eval_adjoint(self):
        return Inverse(self.arg.adjoint())

    def _eval_conjugate(self):
        return Inverse(self.arg.conjugate())

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import det
        return 1/det(self.arg)

    def doit(self, **hints):
        if 'inv_expand' in hints and hints['inv_expand'] == False:
            return self

        arg = self.arg
        if hints.get('deep', True):
            arg = arg.doit(**hints)

        return arg.inverse()

    def _eval_derivative_matrix_lines(self, x):
        arg = self.args[0]
        lines = arg._eval_derivative_matrix_lines(x)
        for line in lines:
            line.first_pointer *= -self.T
            line.second_pointer *= self
        return lines


from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict


def refine_Inverse(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> X.I
    X**(-1)
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(X.I))
    X.T
    """
    if ask(Q.orthogonal(expr), assumptions):
        return expr.arg.T
    elif ask(Q.unitary(expr), assumptions):
        return expr.arg.conjugate()
    elif ask(Q.singular(expr), assumptions):
        raise ValueError("Inverse of singular matrix %s" % expr.arg)

    return expr

handlers_dict['Inverse'] = refine_Inverse
<|endoftext|>"""Implementation of the Kronecker product"""
from functools import reduce
from math import prod

from sympy.core import Mul, sympify
from sympy.functions import adjoint
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.expressions.special import Identity
from sympy.matrices.matrixbase import MatrixBase
from sympy.strategies import (
    canon, condition, distribute, do_one, exhaust, flatten, typed, unpack)
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift

from .matadd import MatAdd
from .matmul import MatMul
from .matpow import MatPow


def kronecker_product(*matrices):
    """
    The Kronecker product of two or more arguments.

    This computes the explicit Kronecker product for subclasses of
    ``MatrixBase`` i.e. explicit matrices. Otherwise, a symbolic
    ``KroneckerProduct`` object is returned.


    Examples
    ========

    For ``MatrixSymbol`` arguments a ``KroneckerProduct`` object is returned.
    Elements of this matrix can be obtained by indexing, or for MatrixSymbols
    with known dimension the explicit matrix can be obtained with
    ``.as_explicit()``

    >>> from sympy import kronecker_product, MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = MatrixSymbol('B', 2, 2)
    >>> kronecker_product(A)
    A
    >>> kronecker_product(A, B)
    KroneckerProduct(A, B)
    >>> kronecker_product(A, B)[0, 1]
    A[0, 0]*B[0, 1]
    >>> kronecker_product(A, B).as_explicit()
    Matrix([
        [A[0, 0]*B[0, 0], A[0, 0]*B[0, 1], A[0, 1]*B[0, 0], A[0, 1]*B[0, 1]],
        [A[0, 0]*B[1, 0], A[0, 0]*B[1, 1], A[0, 1]*B[1, 0], A[0, 1]*B[1, 1]],
        [A[1, 0]*B[0, 0], A[1, 0]*B[0, 1], A[1, 1]*B[0, 0], A[1, 1]*B[0, 1]],
        [A[1, 0]*B[1, 0], A[1, 0]*B[1, 1], A[1, 1]*B[1, 0], A[1, 1]*B[1, 1]]])

    For explicit matrices the Kronecker product is returned as a Matrix

    >>> from sympy import Matrix, kronecker_product
    >>> sigma_x = Matrix([
    ... [0, 1],
    ... [1, 0]])
    ...
    >>> Isigma_y = Matrix([
    ... [0, 1],
    ... [-1, 0]])
    ...
    >>> kronecker_product(sigma_x, Isigma_y)
    Matrix([
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0],
    [ 0, 1,  0, 0],
    [-1, 0,  0, 0]])

    See Also
    ========
        KroneckerProduct

    """
    if not matrices:
        raise TypeError("Empty Kronecker product is undefined")
    if len(matrices) == 1:
        return matrices[0]
    else:
        return KroneckerProduct(*matrices).doit()


class KroneckerProduct(MatrixExpr):
    """
    The Kronecker product of two or more arguments.

    The Kronecker product is a non-commutative product of matrices.
    Given two matrices of dimension (m, n) and (s, t) it produces a matrix
    of dimension (m s, n t).

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``kronecker_product()`` or call the ``.doit()`` or  ``.as_explicit()``
    methods.

    >>> from sympy import KroneckerProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(KroneckerProduct(A, B), KroneckerProduct)
    True
    """
    is_KroneckerProduct = True

    def __new__(cls, *args, check=True):
        args = list(map(sympify, args))
        if all(a.is_Identity for a in args):
            ret = Identity(prod(a.rows for a in args))
            if all(isinstance(a, MatrixBase) for a in args):
                return ret.as_explicit()
            else:
                return ret

        if check:
            validate(*args)
        return super().__new__(cls, *args)

    @property
    def shape(self):
        rows, cols = self.args[0].shape
        for mat in self.args[1:]:
            rows *= mat.rows
            cols *= mat.cols
        return (rows, cols)

    def _entry(self, i, j, **kwargs):
        result = 1
        for mat in reversed(self.args):
            i, m = divmod(i, mat.rows)
            j, n = divmod(j, mat.cols)
            result *= mat[m, n]
        return result

    def _eval_adjoint(self):
        return KroneckerProduct(*list(map(adjoint, self.args))).doit()

    def _eval_conjugate(self):
        return KroneckerProduct(*[a.conjugate() for a in self.args]).doit()

    def _eval_transpose(self):
        return KroneckerProduct(*list(map(transpose, self.args))).doit()

    def _eval_trace(self):
        from .trace import trace
        return Mul(*[trace(a) for a in self.args])

    def _eval_determinant(self):
        from .determinant import det, Determinant
        if not all(a.is_square for a in self.args):
            return Determinant(self)

        m = self.rows
        return Mul(*[det(a)**(m/a.rows) for a in self.args])

    def _eval_inverse(self):
        try:
            return KroneckerProduct(*[a.inverse() for a in self.args])
        except ShapeError:
            from sympy.matrices.expressions.inverse import Inverse
            return Inverse(self)

    def structurally_equal(self, other):
        '''Determine whether two matrices have the same Kronecker product structure

        Examples
        ========

        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols
        >>> m, n = symbols(r'm, n', integer=True)
        >>> A = MatrixSymbol('A', m, m)
        >>> B = MatrixSymbol('B', n, n)
        >>> C = MatrixSymbol('C', m, m)
        >>> D = MatrixSymbol('D', n, n)
        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(C, D))
        True
        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(D, C))
        False
        >>> KroneckerProduct(A, B).structurally_equal(C)
        False
        '''
        # Inspired by BlockMatrix
        return (isinstance(other, KroneckerProduct)
                and self.shape == other.shape
                and len(self.args) == len(other.args)
                and all(a.shape == b.shape for (a, b) in zip(self.args, other.args)))

    def has_matching_shape(self, other):
        '''Determine whether two matrices have the appropriate structure to bring matrix
        multiplication inside the KroneckerProdut

        Examples
        ========
        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols
        >>> m, n = symbols(r'm, n', integer=True)
        >>> A = MatrixSymbol('A', m, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(B, A))
        True
        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(A, B))
        False
        >>> KroneckerProduct(A, B).has_matching_shape(A)
        False
        '''
        return (isinstance(other, KroneckerProduct)
                and self.cols == other.rows
                and len(self.args) == len(other.args)
                and all(a.cols == b.rows for (a, b) in zip(self.args, other.args)))

    def _eval_expand_kroneckerproduct(self, **hints):
        return flatten(canon(typed({KroneckerProduct: distribute(KroneckerProduct, MatAdd)}))(self))

    def _kronecker_add(self, other):
        if self.structurally_equal(other):
            return self.__class__(*[a + b for (a, b) in zip(self.args, other.args)])
        else:
            return self + other

    def _kronecker_mul(self, other):
        if self.has_matching_shape(other):
            return self.__class__(*[a*b for (a, b) in zip(self.args, other.args)])
        else:
            return self * other

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(KroneckerProduct(*args))


def validate(*args):
    if not all(arg.is_Matrix for arg in args):
        raise TypeError("Mix of Matrix and Scalar symbols")


# rules

def extract_commutative(kron):
    c_part = []
    nc_part = []
    for arg in kron.args:
        c, nc = arg.args_cnc()
        c_part.extend(c)
        nc_part.append(Mul._from_args(nc))

    c_part = Mul(*c_part)
    if c_part != 1:
        return c_part*KroneckerProduct(*nc_part)
    return kron


def matrix_kronecker_product(*matrices):
    """Compute the Kronecker product of a sequence of SymPy Matrices.

    This is the standard Kronecker product of matrices [1].

    Parameters
    ==========

    matrices : tuple of MatrixBase instances
        The matrices to take the Kronecker product of.

    Returns
    =======

    matrix : MatrixBase
        The Kronecker product matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.expressions.kronecker import (
    ... matrix_kronecker_product)

    >>> m1 = Matrix([[1,2],[3,4]])
    >>> m2 = Matrix([[1,0],[0,1]])
    >>> matrix_kronecker_product(m1, m2)
    Matrix([
    [1, 0, 2, 0],
    [0, 1, 0, 2],
    [3, 0, 4, 0],
    [0, 3, 0, 4]])
    >>> matrix_kronecker_product(m2, m1)
    Matrix([
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 3, 4]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_product
    """
    # Make sure we have a sequence of Matrices
    if not all(isinstance(m, MatrixBase) for m in matrices):
        raise TypeError(
            'Sequence of Matrices expected, got: %s' % repr(matrices)
        )

    # Pull out the first element in the product.
    matrix_expansion = matrices[-1]
    # Do the kronecker product working from right to left.
    for mat in reversed(matrices[:-1]):
        rows = mat.rows
        cols = mat.cols
        # Go through each row appending kronecker product to.
        # running matrix_expansion.
        for i in range(rows):
            start = matrix_expansion*mat[i*cols]
            # Go through each column joining each item
            for j in range(cols - 1):
                start = start.row_join(
                    matrix_expansion*mat[i*cols + j + 1]
                )
            # If this is the first element, make it the start of the
            # new row.
            if i == 0:
                next = start
            else:
                next = next.col_join(start)
        matrix_expansion = next

    MatrixClass = max(matrices, key=lambda M: M._class_priority).__class__
    if isinstance(matrix_expansion, MatrixClass):
        return matrix_expansion
    else:
        return MatrixClass(matrix_expansion)


def explicit_kronecker_product(kron):
    # Make sure we have a sequence of Matrices
    if not all(isinstance(m, MatrixBase) for m in kron.args):
        return kron

    return matrix_kronecker_product(*kron.args)


rules = (unpack,
         explicit_kronecker_product,
         flatten,
         extract_commutative)

canonicalize = exhaust(condition(lambda x: isinstance(x, KroneckerProduct),
                                 do_one(*rules)))


def _kronecker_dims_key(expr):
    if isinstance(expr, KroneckerProduct):
        return tuple(a.shape for a in expr.args)
    else:
        return (0,)


def kronecker_mat_add(expr):
    args = sift(expr.args, _kronecker_dims_key)
    nonkrons = args.pop((0,), None)
    if not args:
        return expr

    krons = [reduce(lambda x, y: x._kronecker_add(y), group)
             for group in args.values()]

    if not nonkrons:
        return MatAdd(*krons)
    else:
        return MatAdd(*krons) + nonkrons


def kronecker_mat_mul(expr):
    # modified from block matrix code
    factor, matrices = expr.as_coeff_matrices()

    i = 0
    while i < len(matrices) - 1:
        A, B = matrices[i:i+2]
        if isinstance(A, KroneckerProduct) and isinstance(B, KroneckerProduct):
            matrices[i] = A._kronecker_mul(B)
            matrices.pop(i+1)
        else:
            i += 1

    return factor*MatMul(*matrices)


def kronecker_mat_pow(expr):
    if isinstance(expr.base, KroneckerProduct) and all(a.is_square for a in expr.base.args):
        return KroneckerProduct(*[MatPow(a, expr.exp) for a in expr.base.args])
    else:
        return expr


def combine_kronecker(expr):
    """Combine KronekeckerProduct with expression.

    If possible write operations on KroneckerProducts of compatible shapes
    as a single KroneckerProduct.

    Examples
    ========

    >>> from sympy.matrices.expressions import combine_kronecker
    >>> from sympy import MatrixSymbol, KroneckerProduct, symbols
    >>> m, n = symbols(r'm, n', integer=True)
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', n, m)
    >>> combine_kronecker(KroneckerProduct(A, B)*KroneckerProduct(B, A))
    KroneckerProduct(A*B, B*A)
    >>> combine_kronecker(KroneckerProduct(A, B)+KroneckerProduct(B.T, A.T))
    KroneckerProduct(A + B.T, B + A.T)
    >>> C = MatrixSymbol('C', n, n)
    >>> D = MatrixSymbol('D', m, m)
    >>> combine_kronecker(KroneckerProduct(C, D)**m)
    KroneckerProduct(C**m, D**m)
    """
    def haskron(expr):
        return isinstance(expr, MatrixExpr) and expr.has(KroneckerProduct)

    rule = exhaust(
        bottom_up(exhaust(condition(haskron, typed(
            {MatAdd: kronecker_mat_add,
             MatMul: kronecker_mat_mul,
             MatPow: kronecker_mat_pow})))))
    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result
<|endoftext|>from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q

class Factorization(MatrixExpr):
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: self.arg.shape)  # type: ignore

class LofLU(Factorization):
    @property
    def predicates(self):
        return (Q.lower_triangular,)
class UofLU(Factorization):
    @property
    def predicates(self):
        return (Q.upper_triangular,)

class LofCholesky(LofLU): pass
class UofCholesky(UofLU): pass

class QofQR(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)
class RofQR(Factorization):
    @property
    def predicates(self):
        return (Q.upper_triangular,)

class EigenVectors(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)
class EigenValues(Factorization):
    @property
    def predicates(self):
        return (Q.diagonal,)

class UofSVD(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)
class SofSVD(Factorization):
    @property
    def predicates(self):
        return (Q.diagonal,)
class VofSVD(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)


def lu(expr):
    return LofLU(expr), UofLU(expr)

def qr(expr):
    return QofQR(expr), RofQR(expr)

def eig(expr):
    return EigenValues(expr), EigenVectors(expr)

def svd(expr):
    return UofSVD(expr), SofSVD(expr), VofSVD(expr)
<|endoftext|>from sympy.core.relational import Eq
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.logic.boolalg import Boolean, And
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.exceptions import ShapeError
from typing import Union


def is_matadd_valid(*args: MatrixExpr) -> Boolean:
    """Return the symbolic condition how ``MatAdd``, ``HadamardProduct``
    makes sense.

    Parameters
    ==========

    args
        The list of arguments of matrices to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matadd_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matadd_valid(A, B)
    Eq(m, p) & Eq(n, q)
    """
    rows, cols = zip(*(arg.shape for arg in args))
    return And(
        *(Eq(i, j) for i, j in zip(rows[:-1], rows[1:])),
        *(Eq(i, j) for i, j in zip(cols[:-1], cols[1:])),
    )


def is_matmul_valid(*args: Union[MatrixExpr, Expr]) -> Boolean:
    """Return the symbolic condition how ``MatMul`` makes sense

    Parameters
    ==========

    args
        The list of arguments of matrices and scalar expressions to be tested
        for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matmul_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matmul_valid(A, B)
    Eq(n, p)
    """
    rows, cols = zip(*(arg.shape for arg in args if isinstance(arg, MatrixExpr)))
    return And(*(Eq(i, j) for i, j in zip(cols[:-1], rows[1:])))


def is_square(arg: MatrixExpr, /) -> Boolean:
    """Return the symbolic condition how the matrix is assumed to be square

    Parameters
    ==========

    arg
        The matrix to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_square

    >>> m, n = symbols('m n')
    >>> A = MatrixSymbol('A', m, n)
    >>> is_square(A)
    Eq(m, n)
    """
    return Eq(arg.rows, arg.cols)


def validate_matadd_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for addition only for integer values"""
    rows, cols = zip(*(x.shape for x in args))
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), rows))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {rows}")
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), cols))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {cols}")


def validate_matmul_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for multiplication only for integer values"""
    for A, B in zip(args[:-1], args[1:]):
        i, j = A.cols, B.rows
        if isinstance(i, (int, Integer)) and isinstance(j, (int, Integer)) and i != j:
            raise ShapeError("Matrices are not aligned", i, j)
<|endoftext|>from sympy.core.expr import ExprBuilder
from sympy.core.function import (Function, FunctionClass, Lambda)
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase


class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    Lambda(_d, exp(_d)).(X)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    Lambda(_d, exp(_d)).(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """

    def __new__(cls, function, expr):
        expr = _sympify(expr)
        if not expr.is_Matrix:
            raise ValueError("{} must be a matrix instance.".format(expr))

        if expr.shape == (1, 1):
            # Check if the function returns a matrix, in that case, just apply
            # the function instead of creating an ElementwiseApplyFunc object:
            ret = function(expr)
            if isinstance(ret, MatrixExpr):
                return ret

        if not isinstance(function, (FunctionClass, Lambda)):
            d = Dummy('d')
            function = Lambda(d, function(d))

        function = sympify(function)
        if not isinstance(function, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(function))

        if 1 not in function.nargs:
            raise ValueError(
                '{} should be able to accept 1 arguments.'.format(function))

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        obj = MatrixExpr.__new__(cls, function, expr)
        return obj

    @property
    def function(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    @property
    def shape(self):
        return self.expr.shape

    def doit(self, **hints):
        deep = hints.get("deep", True)
        expr = self.expr
        if deep:
            expr = expr.doit(**hints)
        function = self.function
        if isinstance(function, Lambda) and function.is_identity:
            # This is a Lambda containing the identity function.
            return expr
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        elif isinstance(expr, ElementwiseApplyFunction):
            return ElementwiseApplyFunction(
                lambda x: self.function(expr.function(x)),
                expr.expr
            ).doit(**hints)
        else:
            return self

    def _entry(self, i, j, **kwargs):
        return self.function(self.expr._entry(i, j, **kwargs))

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def _eval_derivative(self, x):
        from sympy.matrices.expressions.hadamard import hadamard_product
        dexpr = self.expr.diff(x)
        fdiff = self._get_function_fdiff()
        return hadamard_product(
            dexpr,
            ElementwiseApplyFunction(fdiff, self.expr)
        )

    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.special import Identity
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct

        fdiff = self._get_function_fdiff()
        lr = self.expr._eval_derivative_matrix_lines(x)
        ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
        if 1 in x.shape:
            # Vector:
            iscolumn = self.shape[1] == 1
            for i in lr:
                if iscolumn:
                    ptr1 = i.first_pointer
                    ptr2 = Identity(self.shape[1])
                else:
                    ptr1 = Identity(self.shape[0])
                    ptr2 = i.second_pointer

                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ewdiff,
                                ptr1,
                                ptr2,
                            ]
                        ),
                        (0, 2) if iscolumn else (1, 4)
                    ],
                    validator=ArrayDiagonal._validate
                )
                i._lines = [subexpr]
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 2
        else:
            # Matrix case:
            for i in lr:
                ptr1 = i.first_pointer
                ptr2 = i.second_pointer
                newptr1 = Identity(ptr1.shape[1])
                newptr2 = Identity(ptr2.shape[1])
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [ptr1, newptr1, ewdiff, ptr2, newptr2]
                        ),
                        (1, 2, 4),
                        (5, 7, 8),
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
        return lr

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import Transpose
        return self.func(self.function, Transpose(self.expr).doit())
<|endoftext|>from .matexpr import MatrixExpr
from .special import Identity
from sympy.core import S
from sympy.core.expr import ExprBuilder
from sympy.core.cache import cacheit
from sympy.core.power import Pow
from sympy.core.sympify import _sympify
from sympy.matrices import MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError


class MatPow(MatrixExpr):
    def __new__(cls, base, exp, evaluate=False, **options):
        base = _sympify(base)
        if not base.is_Matrix:
            raise TypeError("MatPow base should be a matrix")

        if base.is_square is False:
            raise NonSquareMatrixError("Power of non-square matrix %s" % base)

        exp = _sympify(exp)
        obj = super().__new__(cls, base, exp)

        if evaluate:
            obj = obj.doit(deep=False)

        return obj

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def shape(self):
        return self.base.shape

    @cacheit
    def _get_explicit_matrix(self):
        return self.base.as_explicit()**self.exp

    def _entry(self, i, j, **kwargs):
        from sympy.matrices.expressions import MatMul
        A = self.doit()
        if isinstance(A, MatPow):
            # We still have a MatPow, make an explicit MatMul out of it.
            if A.exp.is_Integer and A.exp.is_positive:
                A = MatMul(*[A.base for k in range(A.exp)])
            elif not self._is_shape_symbolic():
                return A._get_explicit_matrix()[i, j]
            else:
                # Leave the expression unevaluated:
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
        return A[i, j]

    def doit(self, **hints):
        if hints.get('deep', True):
            base, exp = (arg.doit(**hints) for arg in self.args)
        else:
            base, exp = self.args

        # combine all powers, e.g. (A ** 2) ** 3 -> A ** 6
        while isinstance(base, MatPow):
            exp *= base.args[1]
            base = base.args[0]

        if isinstance(base, MatrixBase):
            # Delegate
            return base ** exp

        # Handle simple cases so that _eval_power() in MatrixExpr sub-classes can ignore them
        if exp == S.One:
            return base
        if exp == S.Zero:
            return Identity(base.rows)
        if exp == S.NegativeOne:
            from sympy.matrices.expressions import Inverse
            return Inverse(base).doit(**hints)

        eval_power = getattr(base, '_eval_power', None)
        if eval_power is not None:
            return eval_power(exp)

        return MatPow(base, exp)

    def _eval_transpose(self):
        base, exp = self.args
        return MatPow(base.transpose(), exp)

    def _eval_adjoint(self):
        base, exp = self.args
        return MatPow(base.adjoint(), exp)

    def _eval_conjugate(self):
        base, exp = self.args
        return MatPow(base.conjugate(), exp)

    def _eval_derivative(self, x):
        return Pow._eval_derivative(self, x)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from .matmul import MatMul
        from .inverse import Inverse
        exp = self.exp
        if self.base.shape == (1, 1) and not exp.has(x):
            lr = self.base._eval_derivative_matrix_lines(x)
            for i in lr:
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                Identity(1),
                                i._lines[0],
                                exp*self.base**(exp-1),
                                i._lines[1],
                                Identity(1),
                            ]
                        ),
                        (0, 3, 4), (5, 7, 8)
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
            return lr
        if (exp > 0) == True:
            newexpr = MatMul.fromiter([self.base for i in range(exp)])
        elif (exp == -1) == True:
            return Inverse(self.base)._eval_derivative_matrix_lines(x)
        elif (exp < 0) == True:
            newexpr = MatMul.fromiter([Inverse(self.base) for i in range(-exp)])
        elif (exp == 0) == True:
            return self.doit()._eval_derivative_matrix_lines(x)
        else:
            raise NotImplementedError("cannot evaluate %s derived by %s" % (self, x))
        return newexpr._eval_derivative_matrix_lines(x)

    def _eval_inverse(self):
        return MatPow(self.base, -self.exp)
<|endoftext|>from sympy.core import AtomicExpr, Symbol, S
from sympy.core.sympify import _sympify
from sympy.printing.pretty.stringpict import prettyForm
from sympy.printing.precedence import PRECEDENCE
from sympy.core.kind import NumberKind


class BaseScalar(AtomicExpr):
    """
    A coordinate symbol/base scalar.

    Ideally, users should not instantiate this class.

    """

    kind = NumberKind

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        from sympy.vector.coordsysrect import CoordSys3D
        if pretty_str is None:
            pretty_str = "x{}".format(index)
        elif isinstance(pretty_str, Symbol):
            pretty_str = pretty_str.name
        if latex_str is None:
            latex_str = "x_{}".format(index)
        elif isinstance(latex_str, Symbol):
            latex_str = latex_str.name

        index = _sympify(index)
        system = _sympify(system)
        obj = super().__new__(cls, index, system)
        if not isinstance(system, CoordSys3D):
            raise TypeError("system should be a CoordSys3D")
        if index not in range(0, 3):
            raise ValueError("Invalid index specified.")
        # The _id is used for equating purposes, and for hashing
        obj._id = (index, system)
        obj._name = obj.name = system._name + '.' + system._variable_names[index]
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system

        return obj

    is_commutative = True
    is_symbol = True

    @property
    def free_symbols(self):
        return {self}

    _diff_wrt = True

    def _eval_derivative(self, s):
        if self == s:
            return S.One
        return S.Zero

    def _latex(self, printer=None):
        return self._latex_form

    def _pretty(self, printer=None):
        return prettyForm(self._pretty_form)

    precedence = PRECEDENCE['Atom']

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        return self._name
<|endoftext|>from sympy.core import Basic
from sympy.vector.operators import gradient, divergence, curl


class Del(Basic):
    """
    Represents the vector differential operator, usually represented in
    mathematical expressions as the 'nabla' symbol.
    """

    def __new__(cls):
        obj = super().__new__(cls)
        obj._name = "delop"
        return obj

    def gradient(self, scalar_field, doit=False):
        """
        Returns the gradient of the given scalar field, as a
        Vector instance.

        Parameters
        ==========

        scalar_field : SymPy expression
            The scalar field to calculate the gradient of.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> delop.gradient(9)
        0
        >>> delop(C.x*C.y*C.z).doit()
        C.y*C.z*C.i + C.x*C.z*C.j + C.x*C.y*C.k

        """

        return gradient(scalar_field, doit=doit)

    __call__ = gradient
    __call__.__doc__ = gradient.__doc__

    def dot(self, vect, doit=False):
        """
        Represents the dot product between this operator and a given
        vector - equal to the divergence of the vector field.

        Parameters
        ==========

        vect : Vector
            The vector whose divergence is to be calculated.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> delop = Del()
        >>> C = CoordSys3D('C')
        >>> delop.dot(C.x*C.i)
        Derivative(C.x, C.x)
        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)
        >>> (delop & v).doit()
        C.x*C.y + C.x*C.z + C.y*C.z

        """
        return divergence(vect, doit=doit)

    __and__ = dot
    __and__.__doc__ = dot.__doc__

    def cross(self, vect, doit=False):
        """
        Represents the cross product between this operator and a given
        vector - equal to the curl of the vector field.

        Parameters
        ==========

        vect : Vector
            The vector whose curl is to be calculated.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)
        >>> delop.cross(v, doit = True)
        (-C.x*C.y + C.x*C.z)*C.i + (C.x*C.y - C.y*C.z)*C.j +
            (-C.x*C.z + C.y*C.z)*C.k
        >>> (delop ^ C.i).doit()
        0

        """

        return curl(vect, doit=doit)

    __xor__ = cross
    __xor__.__doc__ = cross.__doc__

    def _sympystr(self, printer):
        return self._name
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit


class Point(Basic):
    """
    Represents a point in 3-D space.
    """

    def __new__(cls, name, position=Vector.zero, parent_point=None):
        name = str(name)
        # Check the args first
        if not isinstance(position, Vector):
            raise TypeError(
                "position should be an instance of Vector, not %s" % type(
                    position))
        if (not isinstance(parent_point, Point) and
                parent_point is not None):
            raise TypeError(
                "parent_point should be an instance of Point, not %s" % type(
                    parent_point))
        # Super class construction
        if parent_point is None:
            obj = super().__new__(cls, Str(name), position)
        else:
            obj = super().__new__(cls, Str(name), position, parent_point)
        # Decide the object parameters
        obj._name = name
        obj._pos = position
        if parent_point is None:
            obj._parent = None
            obj._root = obj
        else:
            obj._parent = parent_point
            obj._root = parent_point._root
        # Return object
        return obj

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of this Point with respect to
        another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this Point wrt it is
            returned. If its an instance of CoordSyRect, the position
            wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> N.origin.position_wrt(p1)
        (-10)*N.i

        """

        if (not isinstance(other, Point) and
                not isinstance(other, CoordSys3D)):
            raise TypeError(str(other) +
                            "is not a Point or CoordSys3D")
        if isinstance(other, CoordSys3D):
            other = other.origin
        # Handle special cases
        if other == self:
            return Vector.zero
        elif other == self._parent:
            return self._pos
        elif other._parent == self:
            return -1 * other._pos
        # Else, use point tree to calculate position
        rootindex, path = _path(self, other)
        result = Vector.zero
        for i in range(rootindex):
            result += path[i]._pos
        for i in range(rootindex + 1, len(path)):
            result -= path[i]._pos
        return result

    def locate_new(self, name, position):
        """
        Returns a new Point located at the given position wrt this
        Point.
        Thus, the position vector of the new Point wrt this one will
        be equal to the given 'position' parameter.

        Parameters
        ==========

        name : str
            Name of the new point

        position : Vector
            The position vector of the new Point wrt this one

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p1.position_wrt(N.origin)
        10*N.i

        """
        return Point(name, position, self)

    def express_coordinates(self, coordinate_system):
        """
        Returns the Cartesian/rectangular coordinates of this point
        wrt the origin of the given CoordSys3D instance.

        Parameters
        ==========

        coordinate_system : CoordSys3D
            The coordinate system to express the coordinates of this
            Point in.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p2 = p1.locate_new('p2', 5 * N.j)
        >>> p2.express_coordinates(N)
        (10, 5, 0)

        """

        # Determine the position vector
        pos_vect = self.position_wrt(coordinate_system.origin)
        # Express it in the given coordinate system
        return tuple(pos_vect.to_matrix(coordinate_system))

    def _sympystr(self, printer):
        return self._name
<|endoftext|>#sympy.vector.kind

from sympy.core.kind import Kind, _NumberKind, NumberKind
from sympy.core.mul import Mul

class VectorKind(Kind):
    """
    Kind for all vector objects in SymPy.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is
        :class:`sympy.core.kind.NumberKind`,
        which means that the vector contains only numbers.

    Examples
    ========

    Any instance of Vector class has kind ``VectorKind``:

    >>> from sympy.vector.coordsysrect import CoordSys3D
    >>> Sys = CoordSys3D('Sys')
    >>> Sys.i.kind
    VectorKind(NumberKind)

    Operations between instances of Vector keep also have the kind ``VectorKind``:

    >>> from sympy.core.add import Add
    >>> v1 = Sys.i * 2 + Sys.j * 3 + Sys.k * 4
    >>> v2 = Sys.i * Sys.x + Sys.j * Sys.y + Sys.k * Sys.z
    >>> v1.kind
    VectorKind(NumberKind)
    >>> v2.kind
    VectorKind(NumberKind)
    >>> Add(v1, v2).kind
    VectorKind(NumberKind)

    Subclasses of Vector also have the kind ``VectorKind``, such as
    Cross, VectorAdd, VectorMul or VectorZero.

    See Also
    ========

    sympy.core.kind.Kind
    sympy.matrices.kind.MatrixKind

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "VectorKind(%s)" % self.element_kind

@Mul._kind_dispatcher.register(_NumberKind, VectorKind)
def num_vec_mul(k1, k2):
    """
    The result of a multiplication between a number and a Vector should be of VectorKind.
    The element kind is selected by recursive dispatching.
    """
    if not isinstance(k2, VectorKind):
        k1, k2 = k2, k1
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    return VectorKind(elemk)
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, rot_axis1, rot_axis2, rot_axis3)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.core.cache import cacheit
from sympy.core.symbol import Str
import sympy.vector


class Orienter(Basic):
    """
    Super-class for all orienter classes.
    """

    def rotation_matrix(self):
        """
        The rotation matrix corresponding to this orienter
        instance.
        """
        return self._parent_orient


class AxisOrienter(Orienter):
    """
    Class to denote an axis orienter.
    """

    def __new__(cls, angle, axis):
        if not isinstance(axis, sympy.vector.Vector):
            raise TypeError("axis should be a Vector")
        angle = sympify(angle)

        obj = super().__new__(cls, angle, axis)
        obj._angle = angle
        obj._axis = axis

        return obj

    def __init__(self, angle, axis):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import AxisOrienter
        >>> orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> B = N.orient_new('B', (orienter, ))

        """
        # Dummy initializer for docstrings
        pass

    @cacheit
    def rotation_matrix(self, system):
        """
        The rotation matrix corresponding to this orienter
        instance.

        Parameters
        ==========

        system : CoordSys3D
            The coordinate system wrt which the rotation matrix
            is to be computed
        """

        axis = sympy.vector.express(self.axis, system).normalize()
        axis = axis.to_matrix(system)
        theta = self.angle
        parent_orient = ((eye(3) - axis * axis.T) * cos(theta) +
                         Matrix([[0, -axis[2], axis[1]],
                                 [axis[2], 0, -axis[0]],
                                 [-axis[1], axis[0], 0]]) * sin(theta) +
                         axis * axis.T)
        parent_orient = parent_orient.T
        return parent_orient

    @property
    def angle(self):
        return self._angle

    @property
    def axis(self):
        return self._axis


class ThreeAngleOrienter(Orienter):
    """
    Super-class for Body and Space orienters.
    """

    def __new__(cls, angle1, angle2, angle3, rot_order):
        if isinstance(rot_order, Str):
            rot_order = rot_order.name

        approved_orders = ('123', '231', '312', '132', '213',
                           '321', '121', '131', '212', '232',
                           '313', '323', '')
        original_rot_order = rot_order
        rot_order = str(rot_order).upper()
        if not (len(rot_order) == 3):
            raise TypeError('rot_order should be a str of length 3')
        rot_order = [i.replace('X', '1') for i in rot_order]
        rot_order = [i.replace('Y', '2') for i in rot_order]
        rot_order = [i.replace('Z', '3') for i in rot_order]
        rot_order = ''.join(rot_order)
        if rot_order not in approved_orders:
            raise TypeError('Invalid rot_type parameter')
        a1 = int(rot_order[0])
        a2 = int(rot_order[1])
        a3 = int(rot_order[2])
        angle1 = sympify(angle1)
        angle2 = sympify(angle2)
        angle3 = sympify(angle3)
        if cls._in_order:
            parent_orient = (_rot(a1, angle1) *
                             _rot(a2, angle2) *
                             _rot(a3, angle3))
        else:
            parent_orient = (_rot(a3, angle3) *
                             _rot(a2, angle2) *
                             _rot(a1, angle1))
        parent_orient = parent_orient.T

        obj = super().__new__(
            cls, angle1, angle2, angle3, Str(rot_order))
        obj._angle1 = angle1
        obj._angle2 = angle2
        obj._angle3 = angle3
        obj._rot_order = original_rot_order
        obj._parent_orient = parent_orient

        return obj

    @property
    def angle1(self):
        return self._angle1

    @property
    def angle2(self):
        return self._angle2

    @property
    def angle3(self):
        return self._angle3

    @property
    def rot_order(self):
        return self._rot_order


class BodyOrienter(ThreeAngleOrienter):
    """
    Class to denote a body-orienter.
    """

    _in_order = True

    def __new__(cls, angle1, angle2, angle3, rot_order):
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3,
                                         rot_order)
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, BodyOrienter
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> D = N.orient_new('D', (body_orienter, ))

        is same as

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter1 = AxisOrienter(q1, N.i)
        >>> D = N.orient_new('D', (axis_orienter1, ))
        >>> axis_orienter2 = AxisOrienter(q2, D.j)
        >>> D = D.orient_new('D', (axis_orienter2, ))
        >>> axis_orienter3 = AxisOrienter(q3, D.k)
        >>> D = D.orient_new('D', (axis_orienter3, ))

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> body_orienter1 = BodyOrienter(q1, q2, q3, '123')
        >>> body_orienter2 = BodyOrienter(q1, q2, 0, 'ZXZ')
        >>> body_orienter3 = BodyOrienter(0, 0, 0, 'XYX')

        """
        # Dummy initializer for docstrings
        pass


class SpaceOrienter(ThreeAngleOrienter):
    """
    Class to denote a space-orienter.
    """

    _in_order = False

    def __new__(cls, angle1, angle2, angle3, rot_order):
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3,
                                         rot_order)
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        See Also
        ========

        BodyOrienter : Orienter to orient systems wrt Euler angles.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, SpaceOrienter
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> D = N.orient_new('D', (space_orienter, ))

        is same as

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter1 = AxisOrienter(q1, N.i)
        >>> B = N.orient_new('B', (axis_orienter1, ))
        >>> axis_orienter2 = AxisOrienter(q2, N.j)
        >>> C = B.orient_new('C', (axis_orienter2, ))
        >>> axis_orienter3 = AxisOrienter(q3, N.k)
        >>> D = C.orient_new('C', (axis_orienter3, ))

        """
        # Dummy initializer for docstrings
        pass


class QuaternionOrienter(Orienter):
    """
    Class to denote a quaternion-orienter.
    """

    def __new__(cls, q0, q1, q2, q3):
        q0 = sympify(q0)
        q1 = sympify(q1)
        q2 = sympify(q2)
        q3 = sympify(q3)
        parent_orient = (Matrix([[q0 ** 2 + q1 ** 2 - q2 ** 2 -
                                  q3 ** 2,
                                  2 * (q1 * q2 - q0 * q3),
                                  2 * (q0 * q2 + q1 * q3)],
                                 [2 * (q1 * q2 + q0 * q3),
                                  q0 ** 2 - q1 ** 2 +
                                  q2 ** 2 - q3 ** 2,
                                  2 * (q2 * q3 - q0 * q1)],
                                 [2 * (q1 * q3 - q0 * q2),
                                  2 * (q0 * q1 + q2 * q3),
                                  q0 ** 2 - q1 ** 2 -
                                  q2 ** 2 + q3 ** 2]]))
        parent_orient = parent_orient.T

        obj = super().__new__(cls, q0, q1, q2, q3)
        obj._q0 = q0
        obj._q1 = q1
        obj._q2 = q2
        obj._q3 = q3
        obj._parent_orient = parent_orient

        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> B = N.orient_new('B', (q_orienter, ))

        """
        # Dummy initializer for docstrings
        pass

    @property
    def q0(self):
        return self._q0

    @property
    def q1(self):
        return self._q1

    @property
    def q2(self):
        return self._q2

    @property
    def q3(self):
        return self._q3


def _rot(axis, angle):
    """DCM for simple axis 1, 2 or 3 rotations. """
    if axis == 1:
        return Matrix(rot_axis1(angle).T)
    elif axis == 2:
        return Matrix(rot_axis2(angle).T)
    elif axis == 3:
        return Matrix(rot_axis3(angle).T)
<|endoftext|>import collections
from sympy.core.expr import Expr
from sympy.core import sympify, S, preorder_traversal
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Vector, VectorMul, VectorAdd, Cross, Dot
from sympy.core.function import Derivative
from sympy.core.add import Add
from sympy.core.mul import Mul


def _get_coord_systems(expr):
    g = preorder_traversal(expr)
    ret = set()
    for i in g:
        if isinstance(i, CoordSys3D):
            ret.add(i)
            g.skip()
    return frozenset(ret)


def _split_mul_args_wrt_coordsys(expr):
    d = collections.defaultdict(lambda: S.One)
    for i in expr.args:
        d[_get_coord_systems(i)] *= i
    return list(d.values())


class Gradient(Expr):
    """
    Represents unevaluated Gradient.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Gradient
    >>> R = CoordSys3D('R')
    >>> s = R.x*R.y*R.z
    >>> Gradient(s)
    Gradient(R.x*R.y*R.z)

    """

    def __new__(cls, expr):
        expr = sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def doit(self, **hints):
        return gradient(self._expr, doit=True)


class Divergence(Expr):
    """
    Represents unevaluated Divergence.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Divergence
    >>> R = CoordSys3D('R')
    >>> v = R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> Divergence(v)
    Divergence(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)

    """

    def __new__(cls, expr):
        expr = sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def doit(self, **hints):
        return divergence(self._expr, doit=True)


class Curl(Expr):
    """
    Represents unevaluated Curl.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Curl
    >>> R = CoordSys3D('R')
    >>> v = R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> Curl(v)
    Curl(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)

    """

    def __new__(cls, expr):
        expr = sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def doit(self, **hints):
        return curl(self._expr, doit=True)


def curl(vect, doit=True):
    """
    Returns the curl of a vector field computed wrt the base scalars
    of the given coordinate system.

    Parameters
    ==========

    vect : Vector
        The vector operand

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, curl
    >>> R = CoordSys3D('R')
    >>> v1 = R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> curl(v1)
    0
    >>> v2 = R.x*R.y*R.z*R.i
    >>> curl(v2)
    R.x*R.y*R.j + (-R.x*R.z)*R.k

    """

    coord_sys = _get_coord_systems(vect)

    if len(coord_sys) == 0:
        return Vector.zero
    elif len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
