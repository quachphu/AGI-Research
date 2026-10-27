def _(expr: Expr, x: _ArrayExpr):
    return ZeroArray(*x.shape)


@array_derive.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct, x: Expr):
    args = expr.args
    addend_list = []
    for i, arg in enumerate(expr.args):
        darg = array_derive(arg, x)
        if darg == 0:
            continue
        args_prev = args[:i]
        args_succ = args[i+1:]
        shape_prev = reduce(operator.add, map(get_shape, args_prev), ())
        shape_succ = reduce(operator.add, map(get_shape, args_succ), ())
        addend = _array_tensor_product(*args_prev, darg, *args_succ)
        tot1 = len(get_shape(x))
        tot2 = tot1 + len(shape_prev)
        tot3 = tot2 + len(get_shape(arg))
        tot4 = tot3 + len(shape_succ)
        perm = list(range(tot1, tot2)) + \
               list(range(tot1)) + list(range(tot2, tot3)) + \
               list(range(tot3, tot4))
        addend = _permute_dims(addend, _af_invert(perm))
        addend_list.append(addend)
    if len(addend_list) == 1:
        return addend_list[0]
    elif len(addend_list) == 0:
        return S.Zero
    else:
        return _array_add(*addend_list)


@array_derive.register(ArraySymbol)
def _(expr: ArraySymbol, x: _ArrayExpr):
    if expr == x:
        return _permute_dims(
            ArrayTensorProduct.fromiter(Identity(i) for i in expr.shape),
            [2*i for i in range(len(expr.shape))] + [2*i+1 for i in range(len(expr.shape))]
        )
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(MatrixSymbol)
def _(expr: MatrixSymbol, x: _ArrayExpr):
    m, n = expr.shape
    if expr == x:
        return _permute_dims(
            _array_tensor_product(Identity(m), Identity(n)),
            [0, 2, 1, 3]
        )
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(Identity)
def _(expr: Identity, x: _ArrayExpr):
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(OneMatrix)
def _(expr: OneMatrix, x: _ArrayExpr):
    return ZeroArray(*(x.shape + expr.shape))


@array_derive.register(Transpose)
def _(expr: Transpose, x: Expr):
    # D(A.T, A) ==> (m,n,i,j) ==> D(A_ji, A_mn) = d_mj d_ni
    # D(B.T, A) ==> (m,n,i,j) ==> D(B_ji, A_mn)
    fd = array_derive(expr.arg, x)
    return _permute_dims(fd, [0, 1, 3, 2])


@array_derive.register(Inverse)
def _(expr: Inverse, x: Expr):
    mat = expr.I
    dexpr = array_derive(mat, x)
    tp = _array_tensor_product(-expr, dexpr, expr)
    mp = _array_contraction(tp, (1, 4), (5, 6))
    pp = _permute_dims(mp, [1, 2, 0, 3])
    return pp


@array_derive.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction, x: Expr):
    assert get_rank(expr) == 2
    assert get_rank(x) == 2
    fdiff = expr._get_function_fdiff()
    dexpr = array_derive(expr.expr, x)
    tp = _array_tensor_product(
        ElementwiseApplyFunction(fdiff, expr.expr),
        dexpr
    )
    td = _array_diagonal(
        tp, (0, 4), (1, 5)
    )
    return td


@array_derive.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc, x: Expr):
    fdiff = expr._get_function_fdiff()
    subexpr = expr.expr
    dsubexpr = array_derive(subexpr, x)
    tp = _array_tensor_product(
        dsubexpr,
        ArrayElementwiseApplyFunc(fdiff, subexpr)
    )
    b = get_rank(x)
    c = get_rank(expr)
    diag_indices = [(b + i, b + c + i) for i in range(c)]
    return _array_diagonal(tp, *diag_indices)


@array_derive.register(MatrixExpr)
def _(expr: MatrixExpr, x: Expr):
    cg = convert_matrix_to_array(expr)
    return array_derive(cg, x)


@array_derive.register(HadamardProduct)
def _(expr: HadamardProduct, x: Expr):
    raise NotImplementedError()


@array_derive.register(ArrayContraction)
def _(expr: ArrayContraction, x: Expr):
    fd = array_derive(expr.expr, x)
    rank_x = len(get_shape(x))
    contraction_indices = expr.contraction_indices
    new_contraction_indices = [tuple(j + rank_x for j in i) for i in contraction_indices]
    return _array_contraction(fd, *new_contraction_indices)


@array_derive.register(ArrayDiagonal)
def _(expr: ArrayDiagonal, x: Expr):
    dsubexpr = array_derive(expr.expr, x)
    rank_x = len(get_shape(x))
    diag_indices = [[j + rank_x for j in i] for i in expr.diagonal_indices]
    return _array_diagonal(dsubexpr, *diag_indices)


@array_derive.register(ArrayAdd)
def _(expr: ArrayAdd, x: Expr):
    return _array_add(*[array_derive(arg, x) for arg in expr.args])


@array_derive.register(PermuteDims)
def _(expr: PermuteDims, x: Expr):
    de = array_derive(expr.expr, x)
    perm = [0, 1] + [i + 2 for i in expr.permutation.array_form]
    return _permute_dims(de, perm)


@array_derive.register(Reshape)
def _(expr: Reshape, x: Expr):
    de = array_derive(expr.expr, x)
    return Reshape(de, get_shape(x) + expr.shape)


def matrix_derive(expr, x):
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    ce = convert_matrix_to_array(expr)
    dce = array_derive(ce, x)
    return convert_array_to_matrix(dce).doit()
<|endoftext|>from collections import defaultdict

from sympy import Function
from sympy.combinatorics.permutations import _af_invert
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, \
    get_shape, ArrayElement, _array_tensor_product, _array_diagonal, _array_contraction, _array_add, \
    _permute_dims, OneArray, ArrayAdd
from sympy.tensor.array.expressions.utils import _get_argindex, _get_diagonal_indices


def convert_indexed_to_array(expr, first_indices=None):
    r"""
    Parse indexed expression into a form useful for code generation.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy import MatrixSymbol, Sum, symbols

    >>> i, j, k, d = symbols("i j k d")
    >>> M = MatrixSymbol("M", d, d)
    >>> N = MatrixSymbol("N", d, d)

    Recognize the trace in summation form:

    >>> expr = Sum(M[i, i], (i, 0, d-1))
    >>> convert_indexed_to_array(expr)
    ArrayContraction(M, (0, 1))

    Recognize the extraction of the diagonal by using the same index `i` on
    both axes of the matrix:

    >>> expr = M[i, i]
    >>> convert_indexed_to_array(expr)
    ArrayDiagonal(M, (0, 1))

    This function can help perform the transformation expressed in two
    different mathematical notations as:

    `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

    Recognize the matrix multiplication in summation form:

    >>> expr = Sum(M[i, j]*N[j, k], (j, 0, d-1))
    >>> convert_indexed_to_array(expr)
    ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

    Specify that ``k`` has to be the starting index:

    >>> convert_indexed_to_array(expr, first_indices=[k])
    ArrayContraction(ArrayTensorProduct(N, M), (0, 3))
    """

    result, indices = _convert_indexed_to_array(expr)

    if any(isinstance(i, (int, Integer)) for i in indices):
        result = ArrayElement(result, indices)
        indices = []

    if not first_indices:
        return result

    def _check_is_in(elem, indices):
        if elem in indices:
            return True
        if any(elem in i for i in indices if isinstance(i, frozenset)):
            return True
        return False

    repl = {j: i for i in indices if isinstance(i, frozenset) for j in i}
    first_indices = [repl.get(i, i) for i in first_indices]
    for i in first_indices:
        if not _check_is_in(i, indices):
            first_indices.remove(i)
    first_indices.extend([i for i in indices if not _check_is_in(i, first_indices)])

    def _get_pos(elem, indices):
        if elem in indices:
            return indices.index(elem)
        for i, e in enumerate(indices):
            if not isinstance(e, frozenset):
                continue
            if elem in e:
                return i
        raise ValueError("not found")

    permutation = _af_invert([_get_pos(i, first_indices) for i in indices])
    if isinstance(result, ArrayAdd):
        return _array_add(*[_permute_dims(arg, permutation) for arg in result.args])
    else:
        return _permute_dims(result, permutation)


def _convert_indexed_to_array(expr):
    if isinstance(expr, Sum):
        function = expr.function
        summation_indices = expr.variables
        subexpr, subindices = _convert_indexed_to_array(function)
        subindicessets = {j: i for i in subindices if isinstance(i, frozenset) for j in i}
        summation_indices = sorted({subindicessets.get(i, i) for i in summation_indices}, key=default_sort_key)
        # TODO: check that Kronecker delta is only contracted to one other element:
        kronecker_indices = set()
        if isinstance(function, Mul):
            for arg in function.args:
                if not isinstance(arg, KroneckerDelta):
                    continue
                arg_indices = sorted(set(arg.indices), key=default_sort_key)
                if len(arg_indices) == 2:
                    kronecker_indices.update(arg_indices)
        kronecker_indices = sorted(kronecker_indices, key=default_sort_key)
        # Check dimensional consistency:
        shape = get_shape(subexpr)
        if shape:
            for ind, istart, iend in expr.limits:
                i = _get_argindex(subindices, ind)
                if istart != 0 or iend+1 != shape[i]:
                    raise ValueError("summation index and array dimension mismatch: %s" % ind)
        contraction_indices = []
        subindices = list(subindices)
        if isinstance(subexpr, ArrayDiagonal):
            diagonal_indices = list(subexpr.diagonal_indices)
            dindices = subindices[-len(diagonal_indices):]
            subindices = subindices[:-len(diagonal_indices)]
            for index in summation_indices:
                if index in dindices:
                    position = dindices.index(index)
                    contraction_indices.append(diagonal_indices[position])
                    diagonal_indices[position] = None
            diagonal_indices = [i for i in diagonal_indices if i is not None]
            for i, ind in enumerate(subindices):
                if ind in summation_indices:
                    pass
            if diagonal_indices:
                subexpr = _array_diagonal(subexpr.expr, *diagonal_indices)
            else:
                subexpr = subexpr.expr

        axes_contraction = defaultdict(list)
        for i, ind in enumerate(subindices):
            include = all(j not in kronecker_indices for j in ind) if isinstance(ind, frozenset) else ind not in kronecker_indices
            if ind in summation_indices and include:
                axes_contraction[ind].append(i)
                subindices[i] = None
        for k, v in axes_contraction.items():
            if any(i in kronecker_indices for i in k) if isinstance(k, frozenset) else k in kronecker_indices:
                continue
            contraction_indices.append(tuple(v))
        free_indices = [i for i in subindices if i is not None]
        indices_ret = list(free_indices)
        indices_ret.sort(key=lambda x: free_indices.index(x))
        return _array_contraction(
                subexpr,
                *contraction_indices,
                free_indices=free_indices
            ), tuple(indices_ret)
    if isinstance(expr, Mul):
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        # Check if there are KroneckerDelta objects:
        kronecker_delta_repl = {}
        for arg in args:
            if not isinstance(arg, KroneckerDelta):
                continue
            # Diagonalize two indices:
            i, j = arg.indices
            kindices = set(arg.indices)
            if i in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[i])
            if j in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[j])
            kindices = frozenset(kindices)
            for index in kindices:
                kronecker_delta_repl[index] = kindices
        # Remove KroneckerDelta objects, their relations should be handled by
        # ArrayDiagonal:
        newargs = []
        newindices = []
        for arg, loc_indices in zip(args, indices):
            if isinstance(arg, KroneckerDelta):
                continue
            newargs.append(arg)
            newindices.append(loc_indices)
        flattened_indices = [kronecker_delta_repl.get(j, j) for i in newindices for j in i]
        diagonal_indices, ret_indices = _get_diagonal_indices(flattened_indices)
        tp = _array_tensor_product(*newargs)
        if diagonal_indices:
            return _array_diagonal(tp, *diagonal_indices), ret_indices
        else:
            return tp, ret_indices
    if isinstance(expr, MatrixElement):
        indices = expr.args[1:]
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return _array_diagonal(expr.args[0], *diagonal_indices), ret_indices
        else:
            return expr.args[0], ret_indices
    if isinstance(expr, ArrayElement):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return _array_diagonal(expr.name, *diagonal_indices), ret_indices
        else:
            return expr.name, ret_indices
    if isinstance(expr, Indexed):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return _array_diagonal(expr.base, *diagonal_indices), ret_indices
        else:
            return expr.args[0], ret_indices
    if isinstance(expr, IndexedBase):
        raise NotImplementedError
    if isinstance(expr, KroneckerDelta):
        return expr, expr.indices
    if isinstance(expr, Add):
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        args = list(args)
        # Check if all indices are compatible. Otherwise expand the dimensions:
        index0 = []
        shape0 = []
        for arg, arg_indices in zip(args, indices):
            arg_indices_set = set(arg_indices)
            arg_indices_missing = arg_indices_set.difference(index0)
            index0.extend([i for i in arg_indices if i in arg_indices_missing])
            arg_shape = get_shape(arg)
            shape0.extend([arg_shape[i] for i, e in enumerate(arg_indices) if e in arg_indices_missing])
        for i, (arg, arg_indices) in enumerate(zip(args, indices)):
            if len(arg_indices) < len(index0):
                missing_indices_pos = [i for i, e in enumerate(index0) if e not in arg_indices]
                missing_shape = [shape0[i] for i in missing_indices_pos]
                arg_indices = tuple(index0[j] for j in missing_indices_pos) + arg_indices
                args[i] = _array_tensor_product(OneArray(*missing_shape), args[i])
            permutation = Permutation([arg_indices.index(j) for j in index0])
            # Perform index permutations:
            args[i] = _permute_dims(args[i], permutation)
        return _array_add(*args), tuple(index0)
    if isinstance(expr, Pow):
        subexpr, subindices = _convert_indexed_to_array(expr.base)
        if isinstance(expr.exp, (int, Integer)):
            diags = zip(*[(2*i, 2*i + 1) for i in range(expr.exp)])
            arr = _array_diagonal(_array_tensor_product(*[subexpr for i in range(expr.exp)]), *diags)
            return arr, subindices
    if isinstance(expr, Function):
        subexpr, subindices = _convert_indexed_to_array(expr.args[0])
        return ArrayElementwiseApplyFunc(type(expr), subexpr), subindices
    return expr, ()
<|endoftext|>import bisect
from collections import defaultdict

from sympy.combinatorics import Permutation
from sympy.core.containers import Tuple
from sympy.core.numbers import Integer


def _get_mapping_from_subranks(subranks):
    mapping = {}
    counter = 0
    for i, rank in enumerate(subranks):
        for j in range(rank):
            mapping[counter] = (i, j)
            counter += 1
    return mapping


def _get_contraction_links(args, subranks, *contraction_indices):
    mapping = _get_mapping_from_subranks(subranks)
    contraction_tuples = [[mapping[j] for j in i] for i in contraction_indices]
    dlinks = defaultdict(dict)
    for links in contraction_tuples:
        if len(links) == 2:
            (arg1, pos1), (arg2, pos2) = links
            dlinks[arg1][pos1] = (arg2, pos2)
            dlinks[arg2][pos2] = (arg1, pos1)
            continue

    return args, dict(dlinks)


def _sort_contraction_indices(pairing_indices):
    pairing_indices = [Tuple(*sorted(i)) for i in pairing_indices]
    pairing_indices.sort(key=lambda x: min(x))
    return pairing_indices


def _get_diagonal_indices(flattened_indices):
    axes_contraction = defaultdict(list)
    for i, ind in enumerate(flattened_indices):
        if isinstance(ind, (int, Integer)):
            # If the indices is a number, there can be no diagonal operation:
            continue
        axes_contraction[ind].append(i)
    axes_contraction = {k: v for k, v in axes_contraction.items() if len(v) > 1}
    # Put the diagonalized indices at the end:
    ret_indices = [i for i in flattened_indices if i not in axes_contraction]
    diag_indices = list(axes_contraction)
    diag_indices.sort(key=lambda x: flattened_indices.index(x))
    diagonal_indices = [tuple(axes_contraction[i]) for i in diag_indices]
    ret_indices += diag_indices
    ret_indices = tuple(ret_indices)
    return diagonal_indices, ret_indices


def _get_argindex(subindices, ind):
    for i, sind in enumerate(subindices):
        if ind == sind:
            return i
        if isinstance(sind, (set, frozenset)) and ind in sind:
            return i
    raise IndexError("%s not found in %s" % (ind, subindices))


def _apply_recursively_over_nested_lists(func, arr):
    if isinstance(arr, (tuple, list, Tuple)):
        return tuple(_apply_recursively_over_nested_lists(func, i) for i in arr)
    elif isinstance(arr, Tuple):
        return Tuple.fromiter(_apply_recursively_over_nested_lists(func, i) for i in arr)
    else:
        return func(arr)


def _build_push_indices_up_func_transformation(flattened_contraction_indices):
    shifts = {0: 0}
    i = 0
    cumulative = 0
    while i < len(flattened_contraction_indices):
        j = 1
        while i+j < len(flattened_contraction_indices):
            if flattened_contraction_indices[i] + j != flattened_contraction_indices[i+j]:
                break
            j += 1
        cumulative += j
        shifts[flattened_contraction_indices[i]] = cumulative
        i += j
    shift_keys = sorted(shifts.keys())

    def func(idx):
        return shifts[shift_keys[bisect.bisect_right(shift_keys, idx)-1]]

    def transform(j):
        if j in flattened_contraction_indices:
            return None
        else:
            return j - func(j)

    return transform


def _build_push_indices_down_func_transformation(flattened_contraction_indices):
    N = flattened_contraction_indices[-1]+2

    shifts = [i for i in range(N) if i not in flattened_contraction_indices]

    def transform(j):
        if j < len(shifts):
            return shifts[j]
        else:
            return j + shifts[-1] - len(shifts) + 1

    return transform


def _apply_permutation_to_list(perm: Permutation, target_list: list):
    """
    Permute a list according to the given permutation.
    """
    new_list = [None for i in range(perm.size)]
    for i, e in enumerate(target_list):
        new_list[perm(i)] = e
    return new_list
<|endoftext|>r"""
Array expressions are expressions representing N-dimensional arrays, without
evaluating them. These expressions represent in a certain way abstract syntax
trees of operations on N-dimensional arrays.

Every N-dimensional array operator has a corresponding array expression object.

Table of correspondences:

=============================== =============================
         Array operator           Array expression operator
=============================== =============================
        tensorproduct                 ArrayTensorProduct
        tensorcontraction             ArrayContraction
        tensordiagonal                ArrayDiagonal
        permutedims                   PermuteDims
=============================== =============================

Examples
========

``ArraySymbol`` objects are the N-dimensional equivalent of ``MatrixSymbol``
objects in the matrix module:

>>> from sympy.tensor.array.expressions import ArraySymbol
>>> from sympy.abc import i, j, k
>>> A = ArraySymbol("A", (3, 2, 4))
>>> A.shape
(3, 2, 4)
>>> A[i, j, k]
A[i, j, k]
>>> A.as_explicit()
[[[A[0, 0, 0], A[0, 0, 1], A[0, 0, 2], A[0, 0, 3]],
  [A[0, 1, 0], A[0, 1, 1], A[0, 1, 2], A[0, 1, 3]]],
 [[A[1, 0, 0], A[1, 0, 1], A[1, 0, 2], A[1, 0, 3]],
  [A[1, 1, 0], A[1, 1, 1], A[1, 1, 2], A[1, 1, 3]]],
 [[A[2, 0, 0], A[2, 0, 1], A[2, 0, 2], A[2, 0, 3]],
  [A[2, 1, 0], A[2, 1, 1], A[2, 1, 2], A[2, 1, 3]]]]

Component-explicit arrays can be added inside array expressions:

>>> from sympy import Array
>>> from sympy import tensorproduct
>>> from sympy.tensor.array.expressions import ArrayTensorProduct
>>> a = Array([1, 2, 3])
>>> b = Array([i, j, k])
>>> expr = ArrayTensorProduct(a, b, b)
>>> expr
ArrayTensorProduct([1, 2, 3], [i, j, k], [i, j, k])
>>> expr.as_explicit() == tensorproduct(a, b, b)
True

Constructing array expressions from index-explicit forms
--------------------------------------------------------

Array expressions are index-implicit. This means they do not use any indices to
represent array operations. The function ``convert_indexed_to_array( ... )``
may be used to convert index-explicit expressions to array expressions.
It takes as input two parameters: the index-explicit expression and the order
of the indices:

>>> from sympy.tensor.array.expressions import convert_indexed_to_array
>>> from sympy import Sum
>>> A = ArraySymbol("A", (3, 3))
>>> B = ArraySymbol("B", (3, 3))
>>> convert_indexed_to_array(A[i, j], [i, j])
A
>>> convert_indexed_to_array(A[i, j], [j, i])
PermuteDims(A, (0 1))
>>> convert_indexed_to_array(A[i, j] + B[j, i], [i, j])
ArrayAdd(A, PermuteDims(B, (0 1)))
>>> convert_indexed_to_array(Sum(A[i, j]*B[j, k], (j, 0, 2)), [i, k])
ArrayContraction(ArrayTensorProduct(A, B), (1, 2))

The diagonal of a matrix in the array expression form:

>>> convert_indexed_to_array(A[i, i], [i])
ArrayDiagonal(A, (0, 1))

The trace of a matrix in the array expression form:

>>> convert_indexed_to_array(Sum(A[i, i], (i, 0, 2)), [i])
ArrayContraction(A, (0, 1))

Compatibility with matrices
---------------------------

Array expressions can be mixed with objects from the matrix module:

>>> from sympy import MatrixSymbol
>>> from sympy.tensor.array.expressions import ArrayContraction
>>> M = MatrixSymbol("M", 3, 3)
>>> N = MatrixSymbol("N", 3, 3)

Express the matrix product in the array expression form:

>>> from sympy.tensor.array.expressions import convert_matrix_to_array
>>> expr = convert_matrix_to_array(M*N)
>>> expr
ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

The expression can be converted back to matrix form:

>>> from sympy.tensor.array.expressions import convert_array_to_matrix
>>> convert_array_to_matrix(expr)
M*N

Add a second contraction on the remaining axes in order to get the trace of `M \cdot N`:

>>> expr_tr = ArrayContraction(expr, (0, 1))
>>> expr_tr
ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N), (1, 2)), (0, 1))

Flatten the expression by calling ``.doit()`` and remove the nested array contraction operations:

>>> expr_tr.doit()
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

Get the explicit form of the array expression:

>>> expr.as_explicit()
[[M[0, 0]*N[0, 0] + M[0, 1]*N[1, 0] + M[0, 2]*N[2, 0], M[0, 0]*N[0, 1] + M[0, 1]*N[1, 1] + M[0, 2]*N[2, 1], M[0, 0]*N[0, 2] + M[0, 1]*N[1, 2] + M[0, 2]*N[2, 2]],
 [M[1, 0]*N[0, 0] + M[1, 1]*N[1, 0] + M[1, 2]*N[2, 0], M[1, 0]*N[0, 1] + M[1, 1]*N[1, 1] + M[1, 2]*N[2, 1], M[1, 0]*N[0, 2] + M[1, 1]*N[1, 2] + M[1, 2]*N[2, 2]],
 [M[2, 0]*N[0, 0] + M[2, 1]*N[1, 0] + M[2, 2]*N[2, 0], M[2, 0]*N[0, 1] + M[2, 1]*N[1, 1] + M[2, 2]*N[2, 1], M[2, 0]*N[0, 2] + M[2, 1]*N[1, 2] + M[2, 2]*N[2, 2]]]

Express the trace of a matrix:

>>> from sympy import Trace
>>> convert_matrix_to_array(Trace(M))
ArrayContraction(M, (0, 1))
>>> convert_matrix_to_array(Trace(M*N))
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

Express the transposition of a matrix (will be expressed as a permutation of the axes:

>>> convert_matrix_to_array(M.T)
PermuteDims(M, (0 1))

Compute the derivative array expressions:

>>> from sympy.tensor.array.expressions import array_derive
>>> d = array_derive(M, M)
>>> d
PermuteDims(ArrayTensorProduct(I, I), (3)(1 2))

Verify that the derivative corresponds to the form computed with explicit matrices:

>>> d.as_explicit()
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]
>>> Me = M.as_explicit()
>>> Me.diff(Me)
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]

"""

__all__ = [
    "ArraySymbol", "ArrayElement", "ZeroArray", "OneArray",
    "ArrayTensorProduct",
    "ArrayContraction",
    "ArrayDiagonal",
    "PermuteDims",
    "ArrayAdd",
    "ArrayElementwiseApplyFunc",
    "Reshape",
    "convert_array_to_matrix",
    "convert_matrix_to_array",
    "convert_array_to_indexed",
    "convert_indexed_to_array",
    "array_derive",
]

from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, PermuteDims, ArrayDiagonal, \
    ArrayContraction, Reshape, ArraySymbol, ArrayElement, ZeroArray, OneArray, ArrayElementwiseApplyFunc
from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
<|endoftext|>from sympy import KroneckerProduct
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct)
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.tensor.array.expressions.array_expressions import \
    ArrayElementwiseApplyFunc, _array_tensor_product, _array_contraction, \
    _array_diagonal, _array_add, _permute_dims, Reshape


def convert_matrix_to_array(expr: Basic) -> Basic:
    if isinstance(expr, MatMul):
        args_nonmat = []
        args = []
        for arg in expr.args:
            if isinstance(arg, MatrixExpr):
                args.append(arg)
            else:
                args_nonmat.append(convert_matrix_to_array(arg))
        contractions = [(2*i+1, 2*i+2) for i in range(len(args)-1)]
        scalar = _array_tensor_product(*args_nonmat) if args_nonmat else S.One
        if scalar == 1:
            tprod = _array_tensor_product(
                *[convert_matrix_to_array(arg) for arg in args])
        else:
            tprod = _array_tensor_product(
                scalar,
                *[convert_matrix_to_array(arg) for arg in args])
        return _array_contraction(
                tprod,
                *contractions
        )
    elif isinstance(expr, MatAdd):
        return _array_add(
                *[convert_matrix_to_array(arg) for arg in expr.args]
        )
    elif isinstance(expr, Transpose):
        return _permute_dims(
                convert_matrix_to_array(expr.args[0]), [1, 0]
        )
    elif isinstance(expr, Trace):
        inner_expr: MatrixExpr = convert_matrix_to_array(expr.arg) # type: ignore
        return _array_contraction(inner_expr, (0, len(inner_expr.shape) - 1))
    elif isinstance(expr, Mul):
        return _array_tensor_product(*[convert_matrix_to_array(i) for i in expr.args])
    elif isinstance(expr, Pow):
        base = convert_matrix_to_array(expr.base)
        if (expr.exp > 0) == True:
            return _array_tensor_product(*[base for i in range(expr.exp)])
        else:
            return expr
    elif isinstance(expr, MatPow):
        base = convert_matrix_to_array(expr.base)
        if expr.exp.is_Integer != True:
            b = symbols("b", cls=Dummy)
            return ArrayElementwiseApplyFunc(Lambda(b, b**expr.exp), convert_matrix_to_array(base))
        elif (expr.exp > 0) == True:
            return convert_matrix_to_array(MatMul.fromiter(base for i in range(expr.exp)))
        else:
            return expr
    elif isinstance(expr, HadamardProduct):
        tp = _array_tensor_product(*[convert_matrix_to_array(arg) for arg in expr.args])
        diag = [[2*i for i in range(len(expr.args))], [2*i+1 for i in range(len(expr.args))]]
        return _array_diagonal(tp, *diag)
    elif isinstance(expr, HadamardPower):
        base, exp = expr.args
        if isinstance(exp, Integer) and exp > 0:
            return convert_matrix_to_array(HadamardProduct.fromiter(base for i in range(exp)))
        else:
            d = Dummy("d")
            return ArrayElementwiseApplyFunc(Lambda(d, d**exp), base)
    elif isinstance(expr, KroneckerProduct):
        kp_args = [convert_matrix_to_array(arg) for arg in expr.args]
        permutation = [2*i for i in range(len(kp_args))] + [2*i + 1 for i in range(len(kp_args))]
        return Reshape(_permute_dims(_array_tensor_product(*kp_args), permutation), expr.shape)
    else:
        return expr
<|endoftext|>from __future__ import annotations
import itertools
from collections import defaultdict
from typing import FrozenSet
from functools import singledispatch
from itertools import accumulate

from sympy import MatMul, Basic, Wild, KroneckerProduct
from sympy.assumptions.ask import (Q, ask)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert, Permutation
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
    ArrayTensorProduct, OneArray, get_rank, _get_subrank, ZeroArray, ArrayContraction, \
    ArrayAdd, _CodegenArrayAbstract, get_shape, ArrayElementwiseApplyFunc, _ArrayExpr, _EditArrayContraction, _ArgE, \
    ArrayElement, _array_tensor_product, _array_contraction, _array_diagonal, _array_add, _permute_dims
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks


def _get_candidate_for_matmul_from_contraction(scan_indices: list[int | None], remaining_args: list[_ArgE]) -> tuple[_ArgE | None, bool, int]:

    scan_indices_int: list[int] = [i for i in scan_indices if i is not None]
    if len(scan_indices_int) == 0:
        return None, False, -1

    transpose: bool = False
    candidate: _ArgE | None = None
    candidate_index: int = -1
    for arg_with_ind2 in remaining_args:
        if not isinstance(arg_with_ind2.element, MatrixExpr):
            continue
        for index in scan_indices_int:
            if candidate_index != -1 and candidate_index != index:
                # A candidate index has already been selected, check
                # repetitions only for that index:
                continue
            if index in arg_with_ind2.indices:
                if set(arg_with_ind2.indices) == {index}:
                    # Index repeated twice in arg_with_ind2
                    candidate = None
                    break
                if candidate is None:
                    candidate = arg_with_ind2
                    candidate_index = index
                    transpose = (index == arg_with_ind2.indices[1])
                else:
                    # Index repeated more than twice, break
                    candidate = None
                    break
    return candidate, transpose, candidate_index


def _insert_candidate_into_editor(editor: _EditArrayContraction, arg_with_ind: _ArgE, candidate: _ArgE, transpose1: bool, transpose2: bool):
    other = candidate.element
    other_index: int | None
    if transpose2:
        other = Transpose(other)
        other_index = candidate.indices[0]
    else:
        other_index = candidate.indices[1]
    new_element = (Transpose(arg_with_ind.element) if transpose1 else arg_with_ind.element) * other
    editor.args_with_ind.remove(candidate)
    new_arge = _ArgE(new_element)
    return new_arge, other_index


def _support_function_tp1_recognize(contraction_indices, args):
    if len(contraction_indices) == 0:
        return _a2m_tensor_product(*args)

    ac = _array_contraction(_array_tensor_product(*args), *contraction_indices)
    editor = _EditArrayContraction(ac)
    editor.track_permutation_start()

    while True:
        flag_stop = True
        for i, arg_with_ind in enumerate(editor.args_with_ind):
            if not isinstance(arg_with_ind.element, MatrixExpr):
                continue

            first_index = arg_with_ind.indices[0]
            second_index = arg_with_ind.indices[1]

            first_frequency = editor.count_args_with_index(first_index)
            second_frequency = editor.count_args_with_index(second_index)

            if first_index is not None and first_frequency == 1 and first_index == second_index:
                flag_stop = False
                arg_with_ind.element = Trace(arg_with_ind.element)._normalize()
                arg_with_ind.indices = []
                break

            scan_indices = []
            if first_frequency == 2:
                scan_indices.append(first_index)
            if second_frequency == 2:
                scan_indices.append(second_index)

            candidate, transpose, found_index = _get_candidate_for_matmul_from_contraction(scan_indices, editor.args_with_ind[i+1:])
            if candidate is not None:
                flag_stop = False
                editor.track_permutation_merge(arg_with_ind, candidate)
                transpose1 = found_index == first_index
                new_arge, other_index = _insert_candidate_into_editor(editor, arg_with_ind, candidate, transpose1, transpose)
                if found_index == first_index:
                    new_arge.indices = [second_index, other_index]
                else:
                    new_arge.indices = [first_index, other_index]
                set_indices = set(new_arge.indices)
                if len(set_indices) == 1 and set_indices != {None}:
                    # This is a trace:
                    new_arge.element = Trace(new_arge.element)._normalize()
                    new_arge.indices = []
                editor.args_with_ind[i] = new_arge
                # TODO: is this break necessary?
                break

        if flag_stop:
            break

    editor.refresh_indices()
    return editor.to_array_contraction()


def _find_trivial_matrices_rewrite(expr: ArrayTensorProduct):
    # If there are matrices of trivial shape in the tensor product (i.e. shape
    # (1, 1)), try to check if there is a suitable non-trivial MatMul where the
    # expression can be inserted.

    # For example, if "a" has shape (1, 1) and "b" has shape (k, 1), the
    # expressions "_array_tensor_product(a, b*b.T)" can be rewritten as
    # "b*a*b.T"

    trivial_matrices = []
    pos: int | None = None  # must be initialized else causes UnboundLocalError
    first: MatrixExpr | None = None  # may cause UnboundLocalError if not initialized
    second: MatrixExpr | None = None  # may cause UnboundLocalError if not initialized
    removed: list[int] = []
    counter: int = 0
    args: list[Basic | None] = list(expr.args)
    for i, arg in enumerate(expr.args):
        if isinstance(arg, MatrixExpr):
            if arg.shape == (1, 1):
                trivial_matrices.append(arg)
                args[i] = None
                removed.extend([counter, counter+1])
            elif pos is None and isinstance(arg, MatMul):
                margs = arg.args
                for j, e in enumerate(margs):
                    if isinstance(e, MatrixExpr) and e.shape[1] == 1:
                        pos = i
                        first = MatMul.fromiter(margs[:j+1])
                        second = MatMul.fromiter(margs[j+1:])
                        break
        counter += get_rank(arg)
    if pos is None:
        return expr, []
    args[pos] = (first*MatMul.fromiter(i for i in trivial_matrices)*second).doit()
    return _array_tensor_product(*[i for i in args if i is not None]), removed


def _find_trivial_kronecker_products_broadcast(expr: ArrayTensorProduct):
    newargs: list[Basic] = []
    removed = []
    count_dims = 0
    for arg in expr.args:
        count_dims += get_rank(arg)
        shape = get_shape(arg)
        current_range = [count_dims-i for i in range(len(shape), 0, -1)]
        if (shape == (1, 1) and len(newargs) > 0 and 1 not in get_shape(newargs[-1]) and
            isinstance(newargs[-1], MatrixExpr) and isinstance(arg, MatrixExpr)):
            # KroneckerProduct object allows the trick of broadcasting:
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            removed.extend(current_range)
        elif 1 not in shape and len(newargs) > 0 and get_shape(newargs[-1]) == (1, 1):
            # Broadcast:
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            prev_range = [i for i in range(min(current_range)) if i not in removed]
            removed.extend(prev_range[-2:])
        else:
            newargs.append(arg)
    return _array_tensor_product(*newargs), removed


@singledispatch
def _array2matrix(expr):
    return expr


@_array2matrix.register(ZeroArray)
def _(expr: ZeroArray):
    if get_rank(expr) == 2:
        return ZeroMatrix(*expr.shape)
    else:
        return expr


@_array2matrix.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    return _a2m_tensor_product(*[_array2matrix(arg) for arg in expr.args])


@_array2matrix.register(ArrayContraction)
def _(expr: ArrayContraction):
    expr = expr.flatten_contraction_of_diagonal()
    expr = identify_removable_identity_matrices(expr)
    expr = expr.split_multiple_contractions()
    expr = identify_hadamard_products(expr)
    if not isinstance(expr, ArrayContraction):
        return _array2matrix(expr)
    subexpr = expr.expr
    contraction_indices: tuple[tuple[int]] = expr.contraction_indices
    if contraction_indices == ((0,), (1,)) or (
        contraction_indices == ((0,),) and subexpr.shape[1] == 1
    ) or (
        contraction_indices == ((1,),) and subexpr.shape[0] == 1
    ):
        shape = subexpr.shape
        subexpr = _array2matrix(subexpr)
        if isinstance(subexpr, MatrixExpr):
            return OneMatrix(1, shape[0])*subexpr*OneMatrix(shape[1], 1)
    if isinstance(subexpr, ArrayTensorProduct):
        newexpr = _array_contraction(_array2matrix(subexpr), *contraction_indices)
        contraction_indices = newexpr.contraction_indices
        if any(i > 2 for i in newexpr.subranks):
            addends = _array_add(*[_a2m_tensor_product(*j) for j in itertools.product(*[i.args if isinstance(i,
                                                                                                                             ArrayAdd) else [i] for i in expr.expr.args])])
            newexpr = _array_contraction(addends, *contraction_indices)
        if isinstance(newexpr, ArrayAdd):
            ret = _array2matrix(newexpr)
            return ret
        assert isinstance(newexpr, ArrayContraction)
        ret = _support_function_tp1_recognize(contraction_indices, list(newexpr.expr.args))
        return ret
    elif not isinstance(subexpr, _CodegenArrayAbstract):
        ret = _array2matrix(subexpr)
        if isinstance(ret, MatrixExpr):
            assert expr.contraction_indices == ((0, 1),)
            return _a2m_trace(ret)
        else:
            return _array_contraction(ret, *expr.contraction_indices)


@_array2matrix.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    pexpr = _array_diagonal(_array2matrix(expr.expr), *expr.diagonal_indices)
    pexpr = identify_hadamard_products(pexpr)
    if isinstance(pexpr, ArrayDiagonal):
        pexpr = _array_diag2contr_diagmatrix(pexpr)
    if expr == pexpr:
        return expr
    return _array2matrix(pexpr)


@_array2matrix.register(PermuteDims)
def _(expr: PermuteDims):
    if expr.permutation.array_form == [1, 0]:
        return _a2m_transpose(_array2matrix(expr.expr))
    elif isinstance(expr.expr, ArrayTensorProduct):
        ranks = expr.expr.subranks
        inv_permutation = expr.permutation**(-1)
        newrange = [inv_permutation(i) for i in range(sum(ranks))]
        newpos = []
        counter = 0
        for rank in ranks:
            newpos.append(newrange[counter:counter+rank])
            counter += rank
        newargs = []
        newperm = []
        scalars = []
        for pos, arg in zip(newpos, expr.expr.args):
            if len(pos) == 0:
                scalars.append(_array2matrix(arg))
            elif pos == sorted(pos):
                newargs.append((_array2matrix(arg), pos[0]))
                newperm.extend(pos)
            elif len(pos) == 2:
                newargs.append((_a2m_transpose(_array2matrix(arg)), pos[0]))
                newperm.extend(reversed(pos))
            else:
                raise NotImplementedError()
        newargs = [i[0] for i in newargs]
        return _permute_dims(_a2m_tensor_product(*scalars, *newargs), _af_invert(newperm))
    elif isinstance(expr.expr, ArrayContraction):
        mat_mul_lines = _array2matrix(expr.expr)
        if not isinstance(mat_mul_lines, ArrayTensorProduct):
            return _permute_dims(mat_mul_lines, expr.permutation)
        # TODO: this assumes that all arguments are matrices, it may not be the case:
        permutation = Permutation(2*len(mat_mul_lines.args)-1)*expr.permutation
        permuted = [permutation(i) for i in range(2*len(mat_mul_lines.args))]
        args_array = [None for i in mat_mul_lines.args]
        for i in range(len(mat_mul_lines.args)):
            p1 = permuted[2*i]
            p2 = permuted[2*i+1]
            if p1 // 2 != p2 // 2:
                return _permute_dims(mat_mul_lines, permutation)
            if p1 > p2:
                args_array[i] = _a2m_transpose(mat_mul_lines.args[p1 // 2])
            else:
                args_array[i] = mat_mul_lines.args[p1 // 2]
        return _a2m_tensor_product(*args_array)
    else:
        return expr


@_array2matrix.register(ArrayAdd)
def _(expr: ArrayAdd):
    addends = [_array2matrix(arg) for arg in expr.args]
    return _a2m_add(*addends)


@_array2matrix.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    subexpr = _array2matrix(expr.expr)
    if isinstance(subexpr, MatrixExpr):
        if subexpr.shape != (1, 1):
            d = expr.function.bound_symbols[0]
            w = Wild("w", exclude=[d])
            p = Wild("p", exclude=[d])
            m = expr.function.expr.match(w*d**p)
            if m is not None:
                return m[w]*HadamardPower(subexpr, m[p])
        return ElementwiseApplyFunction(expr.function, subexpr)
    else:
        return ArrayElementwiseApplyFunc(expr.function, subexpr)


@_array2matrix.register(ArrayElement)
def _(expr: ArrayElement):
    ret = _array2matrix(expr.name)
    if isinstance(ret, MatrixExpr):
        return MatrixElement(ret, *expr.indices)
    return ArrayElement(ret, expr.indices)


@singledispatch
def _remove_trivial_dims(expr):
    return expr, []


@_remove_trivial_dims.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    # Recognize expressions like [x, y] with shape (k, 1, k, 1) as `x*y.T`.
    # The matrix expression has to be equivalent to the tensor product of the
    # matrices, with trivial dimensions (i.e. dim=1) dropped.
    # That is, add contractions over trivial dimensions:

    removed = []
    newargs = []
    cumul = list(accumulate([0] + [get_rank(arg) for arg in expr.args]))
    pending = None
    prev_i = None
    for i, arg in enumerate(expr.args):
        current_range = list(range(cumul[i], cumul[i+1]))
        if isinstance(arg, OneArray):
            removed.extend(current_range)
            continue
        if not isinstance(arg, (MatrixExpr, MatrixBase)):
            rarg, rem = _remove_trivial_dims(arg)
            removed.extend(rem)
            newargs.append(rarg)
            continue
        elif getattr(arg, "is_Identity", False) and arg.shape == (1, 1):
            if arg.shape == (1, 1):
                # Ignore identity matrices of shape (1, 1) - they are equivalent to scalar 1.
                removed.extend(current_range)
            continue
        elif arg.shape == (1, 1):
            arg, _ = _remove_trivial_dims(arg)
            # Matrix is equivalent to scalar:
            if len(newargs) == 0:
                newargs.append(arg)
            elif 1 in get_shape(newargs[-1]):
                if newargs[-1].shape[1] == 1:
                    newargs[-1] = newargs[-1]*arg
                else:
                    newargs[-1] = arg*newargs[-1]
                removed.extend(current_range)
            else:
                newargs.append(arg)
        elif 1 in arg.shape:
            k = [i for i in arg.shape if i != 1][0]
            if pending is None:
                pending = k
                prev_i = i
                newargs.append(arg)
            elif pending == k:
                prev = newargs[-1]
                if prev.shape[0] == 1:
                    d1 = cumul[prev_i]  # type: ignore
                    prev = _a2m_transpose(prev)
                else:
                    d1 = cumul[prev_i] + 1  # type: ignore
                if arg.shape[1] == 1:
                    d2 = cumul[i] + 1
                    arg = _a2m_transpose(arg)
                else:
                    d2 = cumul[i]
                newargs[-1] = prev*arg
                pending = None
                removed.extend([d1, d2])
            else:
                newargs.append(arg)
                pending = k
                prev_i = i
        else:
            newargs.append(arg)
            pending = None
    newexpr, newremoved = _a2m_tensor_product(*newargs), sorted(removed)
    if isinstance(newexpr, ArrayTensorProduct):
        newexpr, newremoved2 = _find_trivial_matrices_rewrite(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    if isinstance(newexpr, ArrayTensorProduct):
        newexpr, newremoved2 = _find_trivial_kronecker_products_broadcast(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    return newexpr, newremoved


@_remove_trivial_dims.register(ArrayAdd)
def _(expr: ArrayAdd):
    rec = [_remove_trivial_dims(arg) for arg in expr.args]
    newargs, removed = zip(*rec)
    if len({get_shape(i) for i in newargs}) > 1:
        return expr, []
    if len(removed) == 0:
        return expr, removed
    removed1 = removed[0]
    return _a2m_add(*newargs), removed1


@_remove_trivial_dims.register(PermuteDims)
def _(expr: PermuteDims):
    subexpr, subremoved = _remove_trivial_dims(expr.expr)
    p = expr.permutation.array_form
    pinv = _af_invert(expr.permutation.array_form)
    shift = list(accumulate([1 if i in subremoved else 0 for i in range(len(p))]))
    premoved = [pinv[i] for i in subremoved]
    p2 = [e - shift[e] for e in p if e not in subremoved]
    # TODO: check if subremoved should be permuted as well...
    newexpr = _permute_dims(subexpr, p2)
    premoved = sorted(premoved)
    if newexpr != expr:
        newexpr, removed2 = _remove_trivial_dims(_array2matrix(newexpr))
        premoved = _combine_removed(-1, premoved, removed2)
    return newexpr, premoved


@_remove_trivial_dims.register(ArrayContraction)
def _(expr: ArrayContraction):
    new_expr, removed0 = _array_contraction_to_diagonal_multiple_identity(expr)
    if new_expr != expr:
        new_expr2, removed1 = _remove_trivial_dims(_array2matrix(new_expr))
        removed = _combine_removed(-1, removed0, removed1)
        return new_expr2, removed
    rank1 = get_rank(expr)
    expr, removed1 = remove_identity_matrices(expr)
    if not isinstance(expr, ArrayContraction):
        expr2, removed2 = _remove_trivial_dims(expr)
        return expr2, _combine_removed(rank1, removed1, removed2)
    newexpr, removed2 = _remove_trivial_dims(expr.expr)
    shifts = list(accumulate([1 if i in removed2 else 0 for i in range(get_rank(expr.expr))]))
    new_contraction_indices = [tuple(j for j in i if j not in removed2) for i in expr.contraction_indices]
    # Remove possible empty tuples "()":
    new_contraction_indices = [i for i in new_contraction_indices if len(i) > 0]
    contraction_indices_flat = [j for i in expr.contraction_indices for j in i]
    removed2 = [i for i in removed2 if i not in contraction_indices_flat]
    new_contraction_indices = [tuple(j - shifts[j] for j in i) for i in new_contraction_indices]
    # Shift removed2:
    removed2 = ArrayContraction._push_indices_up(expr.contraction_indices, removed2)
    removed = _combine_removed(rank1, removed1, removed2)
    return _array_contraction(newexpr, *new_contraction_indices), list(removed)


def _remove_diagonalized_identity_matrices(expr: ArrayDiagonal):
    assert isinstance(expr, ArrayDiagonal)
    editor = _EditArrayContraction(expr)
    mapping = {i: {j for j in editor.args_with_ind if i in j.indices} for i in range(-1, -1-editor.number_of_diagonal_indices, -1)}
    removed = []
    counter: int = 0
    for i, arg_with_ind in enumerate(editor.args_with_ind):
        counter += len(arg_with_ind.indices)
        if isinstance(arg_with_ind.element, Identity):
            if None in arg_with_ind.indices and any(i is not None and (i < 0) == True for i in arg_with_ind.indices):
                diag_ind = [j for j in arg_with_ind.indices if j is not None][0]
                other = [j for j in mapping[diag_ind] if j != arg_with_ind][0]
                if not isinstance(other.element, MatrixExpr):
                    continue
                if 1 not in other.element.shape:
                    continue
                if None not in other.indices:
                    continue
                editor.args_with_ind[i].element = None
                none_index = other.indices.index(None)
                other.element = DiagMatrix(other.element)
                other_range = editor.get_absolute_range(other)
                removed.extend([other_range[0] + none_index])
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, get_rank(expr.expr))
    return editor.to_array_contraction(), removed


@_remove_trivial_dims.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    newexpr, removed = _remove_trivial_dims(expr.expr)
    shifts = list(accumulate([0] + [1 if i in removed else 0 for i in range(get_rank(expr.expr))]))
    new_diag_indices_map = {i: tuple(j for j in i if j not in removed) for i in expr.diagonal_indices}
    for old_diag_tuple, new_diag_tuple in new_diag_indices_map.items():
        if len(new_diag_tuple) == 1:
            removed = [i for i in removed if i not in old_diag_tuple]
    new_diag_indices = [tuple(j - shifts[j] for j in i) for i in new_diag_indices_map.values()]
    rank = get_rank(expr.expr)
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, rank)
    removed = sorted(set(removed))
    # If there are single axes to diagonalize remaining, it means that their
    # corresponding dimension has been removed, they no longer need diagonalization:
    new_diag_indices = [i for i in new_diag_indices if len(i) > 0]
    if len(new_diag_indices) > 0:
        newexpr2 = _array_diagonal(newexpr, *new_diag_indices, allow_trivial_diags=True)
    else:
        newexpr2 = newexpr
    if isinstance(newexpr2, ArrayDiagonal):
        newexpr3, removed2 = _remove_diagonalized_identity_matrices(newexpr2)
        removed = _combine_removed(-1, removed, removed2)
        return newexpr3, removed
    else:
        return newexpr2, removed


@_remove_trivial_dims.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction):
    subexpr, removed = _remove_trivial_dims(expr.expr)
    if subexpr.shape == (1, 1):
        # TODO: move this to ElementwiseApplyFunction
        return expr.function(subexpr), removed + [0, 1]
    return ElementwiseApplyFunction(expr.function, subexpr), []


@_remove_trivial_dims.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    subexpr, removed = _remove_trivial_dims(expr.expr)
    return ArrayElementwiseApplyFunc(expr.function, subexpr), removed


def convert_array_to_matrix(expr):
    r"""
    Recognize matrix expressions in codegen objects.

    If more than one matrix multiplication line have been detected, return a
    list with the matrix expressions.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy.tensor.array import tensorcontraction, tensorproduct
    >>> from sympy import MatrixSymbol, Sum
    >>> from sympy.abc import i, j, k, l, N
    >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> A = MatrixSymbol("A", N, N)
    >>> B = MatrixSymbol("B", N, N)
    >>> C = MatrixSymbol("C", N, N)
    >>> D = MatrixSymbol("D", N, N)

    >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A.T

    Transposition is detected:

    >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A.T*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A

    Detect the trace:

    >>> expr = Sum(A[i, i], (i, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A)

    Recognize some more complex traces:

    >>> expr = Sum(A[i, j]*B[j, i], (i, 0, N-1), (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A*B)

    More complicated expressions:

    >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B.T*A.T

    Expressions constructed from matrix expressions do not contain literal
    indices, the positions of free indices are returned instead:

    >>> expr = A*B
    >>> cg = convert_matrix_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B

    If more than one line of matrix multiplications is detected, return
    separate matrix multiplication factors embedded in a tensor product object:

    >>> cg = tensorcontraction(tensorproduct(A, B, C, D), (1, 2), (5, 6))
    >>> convert_array_to_matrix(cg)
    ArrayTensorProduct(A*B, C*D)

    The two lines have free indices at axes 0, 3 and 4, 7, respectively.
    """
    rec = _array2matrix(expr)
    rec, removed = _remove_trivial_dims(rec)
    return rec


def _array_diag2contr_diagmatrix(expr: ArrayDiagonal):
    if isinstance(expr.expr, ArrayTensorProduct):
        args = list(expr.expr.args)
        diag_indices = list(expr.diagonal_indices)
        mapping = _get_mapping_from_subranks([_get_subrank(arg) for arg in args])
        tuple_links = [[mapping[j] for j in i] for i in diag_indices]
        contr_indices = []
        total_rank = get_rank(expr)
        replaced = [False for arg in args]
        for i, (abs_pos, rel_pos) in enumerate(zip(diag_indices, tuple_links)):
            if len(abs_pos) != 2:
                continue
            (pos1_outer, pos1_inner), (pos2_outer, pos2_inner) = rel_pos
            arg1 = args[pos1_outer]
            arg2 = args[pos2_outer]
            if get_rank(arg1) != 2 or get_rank(arg2) != 2:
                if replaced[pos1_outer]:
                    diag_indices[i] = None
                if replaced[pos2_outer]:
                    diag_indices[i] = None
                continue
            pos1_in2 = 1 - pos1_inner
            pos2_in2 = 1 - pos2_inner
            if arg1.shape[pos1_in2] == 1:
                if arg1.shape[pos1_inner] != 1:
                    darg1 = DiagMatrix(arg1)
                else:
                    darg1 = arg1
                args.append(darg1)
                contr_indices.append(((pos2_outer, pos2_inner), (len(args)-1, pos1_inner)))
                total_rank += 1
                diag_indices[i] = None
                args[pos1_outer] = OneArray(arg1.shape[pos1_in2])
                replaced[pos1_outer] = True
            elif arg2.shape[pos2_in2] == 1:
                if arg2.shape[pos2_inner] != 1:
                    darg2 = DiagMatrix(arg2)
                else:
                    darg2 = arg2
                args.append(darg2)
                contr_indices.append(((pos1_outer, pos1_inner), (len(args)-1, pos2_inner)))
                total_rank += 1
                diag_indices[i] = None
                args[pos2_outer] = OneArray(arg2.shape[pos2_in2])
                replaced[pos2_outer] = True
        diag_indices_new = [i for i in diag_indices if i is not None]
        cumul = list(accumulate([0] + [get_rank(arg) for arg in args]))
        contr_indices2 = [tuple(cumul[a] + b for a, b in i) for i in contr_indices]
        tc = _array_contraction(
            _array_tensor_product(*args), *contr_indices2
        )
        td = _array_diagonal(tc, *diag_indices_new)
        return td
    return expr


def _a2m_mul(*args):
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        from sympy.matrices.expressions.matmul import MatMul
        return MatMul(*args).doit()
    else:
        return _array_contraction(
            _array_tensor_product(*args),
            *[(2*i-1, 2*i) for i in range(1, len(args))]
        )


def _a2m_tensor_product(*args):
    scalars = []
    arrays = []
    for arg in args:
        if isinstance(arg, (MatrixExpr, _ArrayExpr, _CodegenArrayAbstract)):
            arrays.append(arg)
        else:
            scalars.append(arg)
    scalar = Mul.fromiter(scalars)
    if len(arrays) == 0:
        return scalar
    if scalar != 1:
        if isinstance(arrays[0], _CodegenArrayAbstract):
            arrays = [scalar] + arrays
        else:
            arrays[0] *= scalar
    return _array_tensor_product(*arrays)


def _a2m_add(*args):
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        from sympy.matrices.expressions.matadd import MatAdd
        return MatAdd(*args).doit()
    else:
        return _array_add(*args)


def _a2m_trace(arg):
    if isinstance(arg, _CodegenArrayAbstract):
        return _array_contraction(arg, (0, 1))
    else:
        from sympy.matrices.expressions.trace import Trace
        return Trace(arg)


def _a2m_transpose(arg):
    if isinstance(arg, _CodegenArrayAbstract):
        return _permute_dims(arg, [1, 0])
    else:
        from sympy.matrices.expressions.transpose import Transpose
        return Transpose(arg).doit()


def identify_hadamard_products(expr: ArrayContraction | ArrayDiagonal):

    editor: _EditArrayContraction = _EditArrayContraction(expr)

    map_contr_to_args: dict[FrozenSet, list[_ArgE]] = defaultdict(list)
    map_ind_to_inds: dict[int | None, int] = defaultdict(int)
    for arg_with_ind in editor.args_with_ind:
        for ind in arg_with_ind.indices:
            map_ind_to_inds[ind] += 1
        if None in arg_with_ind.indices:
            continue
        map_contr_to_args[frozenset(arg_with_ind.indices)].append(arg_with_ind)

    k: FrozenSet[int]
    v: list[_ArgE]
    for k, v in map_contr_to_args.items():
        make_trace: bool = False
        if len(k) == 1 and next(iter(k)) >= 0 and sum(next(iter(k)) in i for i in map_contr_to_args) == 1:
            # This is a trace: the arguments are fully contracted with only one
            # index, and the index isn't used anywhere else:
            make_trace = True
            first_element = S.One
        elif len(k) != 2:
            # Hadamard product only defined for matrices:
            continue
        if len(v) == 1:
            # Hadamard product with a single argument makes no sense:
            continue
        for ind in k:
            if map_ind_to_inds[ind] <= 2:
                # There is no other contraction, skip:
                continue

        def check_transpose(x):
            x = [i if i >= 0 else -1-i for i in x]
            return x == sorted(x)

        # Check if expression is a trace:
        if all(map_ind_to_inds[j] == len(v) and j >= 0 for j in k) and all(j >= 0 for j in k):
            # This is a trace
            make_trace = True
            first_element = v[0].element
            if not check_transpose(v[0].indices):
                first_element = first_element.T # type: ignore
            hadamard_factors = v[1:]
        else:
            hadamard_factors = v

        # This is a Hadamard product:

        hp = hadamard_product(*[i.element if check_transpose(i.indices) else Transpose(i.element) for i in hadamard_factors])
        hp_indices = v[0].indices
        if not check_transpose(hadamard_factors[0].indices):
            hp_indices = list(reversed(hp_indices))
        if make_trace:
            hp = Trace(first_element*hp.T)._normalize()
            hp_indices = []
        editor.insert_after(v[0], _ArgE(hp, hp_indices))
        for i in v:
            editor.args_with_ind.remove(i)

    return editor.to_array_contraction()


def identify_removable_identity_matrices(expr):
    editor = _EditArrayContraction(expr)

    flag = True
    while flag:
        flag = False
        for arg_with_ind in editor.args_with_ind:
            if isinstance(arg_with_ind.element, Identity):
                k = arg_with_ind.element.shape[0]
                # Candidate for removal:
                if arg_with_ind.indices == [None, None]:
                    # Free identity matrix, will be cleared by _remove_trivial_dims:
                    continue
                elif None in arg_with_ind.indices:
                    ind = [j for j in arg_with_ind.indices if j is not None][0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 1:
                        # Identity matrix contracted only on one index with itself,
                        # transform to a OneArray(k) element:
                        editor.insert_after(arg_with_ind, OneArray(k))
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    elif counted > 2:
                        # Case counted = 2 is a matrix multiplication by identity matrix, skip it.
                        # Case counted > 2 is a multiple contraction,
                        # this is a case where the contraction becomes a diagonalization if the
                        # identity matrix is dropped.
                        continue
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted > 1:
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    else:
                        # This is a trace, skip it as it will be recognized somewhere else:
                        pass
            elif ask(Q.diagonal(arg_with_ind.element)):
                if arg_with_ind.indices == [None, None]:
                    continue
                elif None in arg_with_ind.indices:
                    pass
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 3:
                        # A_ai B_bi D_ii ==> A_ai D_ij B_bj
                        ind_new = editor.get_new_contraction_index()
                        other_args = [j for j in editor.args_with_ind if j != arg_with_ind]
                        other_args[1].indices = [ind_new if j == ind else j for j in other_args[1].indices]
                        arg_with_ind.indices = [ind, ind_new]
                        flag = True
                        break

    return editor.to_array_contraction()


def remove_identity_matrices(expr: ArrayContraction):
    editor = _EditArrayContraction(expr)
    removed: list[int] = []

    permutation_map = {}

    free_indices = list(accumulate([0] + [sum(i is None for i in arg.indices) for arg in editor.args_with_ind]))
    free_map = dict(zip(editor.args_with_ind, free_indices[:-1]))

    update_pairs = {}

    for ind in range(editor.number_of_contraction_indices):
        args = editor.get_args_with_index(ind)
        identity_matrices = [i for i in args if isinstance(i.element, Identity)]
        number_identity_matrices = len(identity_matrices)
        # If the contraction involves a non-identity matrix and multiple identity matrices:
        if number_identity_matrices != len(args) - 1 or number_identity_matrices == 0:
            continue
        # Get the non-identity element:
        non_identity = [i for i in args if not isinstance(i.element, Identity)][0]
        # Check that all identity matrices have at least one free index
        # (otherwise they would be contractions to some other elements)
        if any(None not in i.indices for i in identity_matrices):
            continue
        # Mark the identity matrices for removal:
        for i in identity_matrices:
            i.element = None
            removed.extend(range(free_map[i], free_map[i] + len([j for j in i.indices if j is None])))
        last_removed = removed.pop(-1)
        update_pairs[last_removed, ind] = non_identity.indices[:]
        # Remove the indices from the non-identity matrix, as the contraction
        # no longer exists:
        non_identity.indices = [None if i == ind else i for i in non_identity.indices]

    removed.sort()

    shifts = list(accumulate([1 if i in removed else 0 for i in range(get_rank(expr))]))
    for (last_removed, ind), non_identity_indices in update_pairs.items():
        pos = [free_map[non_identity] + i for i, e in enumerate(non_identity_indices) if e == ind]
        assert len(pos) == 1
        for j in pos:
            permutation_map[j] = last_removed

    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    ret_expr = editor.to_array_contraction()
    permutation = []
    counter = 0
    counter2 = 0
    for j in range(get_rank(expr)):
        if j in removed:
            continue
        if counter2 in permutation_map:
            target = permutation_map[counter2]
            permutation.append(target - shifts[target])
            counter2 += 1
        else:
            while counter in permutation_map.values():
                counter += 1
            permutation.append(counter)
            counter += 1
            counter2 += 1
    ret_expr2 = _permute_dims(ret_expr, _af_invert(permutation))
    return ret_expr2, removed


def _combine_removed(dim: int, removed1: list[int], removed2: list[int]) -> list[int]:
    # Concatenate two axis removal operations as performed by
    # _remove_trivial_dims,
    removed1 = sorted(removed1)
    removed2 = sorted(removed2)
    i = 0
    j = 0
    removed = []
    while True:
        if j >= len(removed2):
            while i < len(removed1):
                removed.append(removed1[i])
                i += 1
            break
        elif i < len(removed1) and removed1[i] <= i + removed2[j]:
            removed.append(removed1[i])
            i += 1
        else:
            removed.append(i + removed2[j])
            j += 1
    return removed


def _array_contraction_to_diagonal_multiple_identity(expr: ArrayContraction):
    editor = _EditArrayContraction(expr)
    editor.track_permutation_start()
    removed: list[int] = []
    diag_index_counter: int = 0
    for i in range(editor.number_of_contraction_indices):
        identities = []
        args = []
        for j, arg in enumerate(editor.args_with_ind):
            if i not in arg.indices:
                continue
            if isinstance(arg.element, Identity):
                identities.append(arg)
            else:
                args.append(arg)
        if len(identities) == 0:
            continue
        if len(args) + len(identities) < 3:
            continue
        new_diag_ind = -1 - diag_index_counter
        diag_index_counter += 1
        # Variable "flag" to control whether to skip this contraction set:
        flag: bool = True
        for i1, id1 in enumerate(identities):
            if None not in id1.indices:
                flag = True
                break
            free_pos = list(range(*editor.get_absolute_free_range(id1)))[0]
            editor._track_permutation[-1].append(free_pos) # type: ignore
            id1.element = None
            flag = False
            break
        if flag:
            continue
        for arg in identities[:i1] + identities[i1+1:]:
            arg.element = None
            removed.extend(range(*editor.get_absolute_free_range(arg)))
        for arg in args:
            arg.indices = [new_diag_ind if j == i else j for j in arg.indices]
    for j, e in enumerate(editor.args_with_ind):
        if e.element is None:
            editor._track_permutation[j] = None # type: ignore
    editor._track_permutation = [i for i in editor._track_permutation if i is not None] # type: ignore
    # Renumber permutation array form in order to deal with deleted positions:
    remap = {e: i for i, e in enumerate(sorted({k for j in editor._track_permutation for k in j}))}
    editor._track_permutation = [[remap[j] for j in i] for i in editor._track_permutation]
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    new_expr = editor.to_array_contraction()
    return new_expr, removed
<|endoftext|>import collections.abc
import operator
from itertools import accumulate

from sympy import Mul, Sum, Dummy, Add
from sympy.tensor.array.expressions import PermuteDims, ArrayAdd, ArrayElementwiseApplyFunc, Reshape
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, get_rank, ArrayContraction, \
    ArrayDiagonal, get_shape, _get_array_element_or_slice, _ArrayExpr
from sympy.tensor.array.expressions.utils import _apply_permutation_to_list


def convert_array_to_indexed(expr, indices):
    return _ConvertArrayToIndexed().do_convert(expr, indices)


class _ConvertArrayToIndexed:

    def __init__(self):
        self.count_dummies = 0

    def do_convert(self, expr, indices):
        if isinstance(expr, ArrayTensorProduct):
            cumul = list(accumulate([0] + [get_rank(arg) for arg in expr.args]))
            indices_grp = [indices[cumul[i]:cumul[i+1]] for i in range(len(expr.args))]
            return Mul.fromiter(self.do_convert(arg, ind) for arg, ind in zip(expr.args, indices_grp))
        if isinstance(expr, ArrayContraction):
            new_indices = [None for i in range(get_rank(expr.expr))]
            limits = []
            bottom_shape = get_shape(expr.expr)
            for contraction_index_grp in expr.contraction_indices:
                d = Dummy(f"d{self.count_dummies}")
                self.count_dummies += 1
                dim = bottom_shape[contraction_index_grp[0]]
                limits.append((d, 0, dim-1))
                for i in contraction_index_grp:
                    new_indices[i] = d
            j = 0
            for i in range(len(new_indices)):
                if new_indices[i] is None:
                    new_indices[i] = indices[j]
                    j += 1
            newexpr = self.do_convert(expr.expr, new_indices)
            return Sum(newexpr, *limits)
        if isinstance(expr, ArrayDiagonal):
            new_indices = [None for i in range(get_rank(expr.expr))]
            ind_pos = expr._push_indices_down(expr.diagonal_indices, list(range(len(indices))), get_rank(expr))
            for i, index in zip(ind_pos, indices):
                if isinstance(i, collections.abc.Iterable):
                    for j in i:
                        new_indices[j] = index
                else:
                    new_indices[i] = index
            newexpr = self.do_convert(expr.expr, new_indices)
            return newexpr
        if isinstance(expr, PermuteDims):
            permuted_indices = _apply_permutation_to_list(expr.permutation, indices)
            return self.do_convert(expr.expr, permuted_indices)
        if isinstance(expr, ArrayAdd):
            return Add.fromiter(self.do_convert(arg, indices) for arg in expr.args)
        if isinstance(expr, _ArrayExpr):
            return expr.__getitem__(tuple(indices))
        if isinstance(expr, ArrayElementwiseApplyFunc):
            return expr.function(self.do_convert(expr.expr, indices))
        if isinstance(expr, Reshape):
            shape_up = expr.shape
            shape_down = get_shape(expr.expr)
            cumul = list(accumulate([1] + list(reversed(shape_up)), operator.mul))
            one_index = Add.fromiter(i*s for i, s in zip(reversed(indices), cumul))
            dest_indices = [None for _ in shape_down]
            c = 1
            for i, e in enumerate(reversed(shape_down)):
                if c == 1:
                    if i == len(shape_down) - 1:
                        dest_indices[i] = one_index
                    else:
                        dest_indices[i] = one_index % e
                elif i == len(shape_down) - 1:
                    dest_indices[i] = one_index // c
                else:
                    dest_indices[i] = one_index // c % e
                c *= e
            dest_indices.reverse()
            return self.do_convert(expr.expr, dest_indices)
        return _get_array_element_or_slice(expr, indices)
<|endoftext|>from __future__ import annotations
import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate

import typing

from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import (Dummy, Symbol)
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
    _get_mapping_from_subranks, _build_push_indices_up_func_transformation, _get_contraction_links, \
    _build_push_indices_down_func_transformation
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify


class _ArrayExpr(Expr):
    shape: tuple[Expr, ...]

    def __getitem__(self, item):
        if not isinstance(item, collections.abc.Iterable):
            item = (item,)
        ArrayElement._check_shape(self, item)
        return self._get(item)

    def _get(self, item):
        return _get_array_element_or_slice(self, item)


class ArraySymbol(_ArrayExpr):
    """
    Symbol representing an array expression
    """

    _iterable = False

    def __new__(cls, symbol, shape: typing.Iterable) -> "ArraySymbol":
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        # symbol = _sympify(symbol)
        shape = Tuple(*map(_sympify, shape))
        obj = Expr.__new__(cls, symbol, shape)
        return obj

    @property
    def name(self):
        return self._args[0]

    @property
    def shape(self):
        return self._args[1]

    def as_explicit(self):
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("cannot express explicit array with symbolic shape")
        data = [self[i] for i in itertools.product(*[range(j) for j in self.shape])]
        return ImmutableDenseNDimArray(data).reshape(*self.shape)


class ArrayElement(Expr):
    """
    An element of an array.
    """

    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, indices):
        if isinstance(name, str):
            name = Symbol(name)
        name = _sympify(name)
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        indices = _sympify(tuple(indices))
        cls._check_shape(name, indices)
        obj = Expr.__new__(cls, name, indices)
        return obj

    @classmethod
    def _check_shape(cls, name, indices):
        indices = tuple(indices)
        if hasattr(name, "shape"):
            index_error = IndexError("number of indices does not match shape of the array")
            if len(indices) != len(name.shape):
                raise index_error
            if any((i >= s) == True for i, s in zip(indices, name.shape)):
                raise ValueError("shape is out of bounds")
        if any((i < 0) == True for i in indices):
            raise ValueError("shape contains negative values")

    @property
    def name(self):
        return self._args[0]

    @property
    def indices(self):
        return self._args[1]

    def _eval_derivative(self, s):
        if not isinstance(s, ArrayElement):
            return S.Zero

        if s == self:
            return S.One

        if s.name != self.name:
            return S.Zero

        return Mul.fromiter(KroneckerDelta(i, j) for i, j in zip(self.indices, s.indices))


class ZeroArray(_ArrayExpr):
    """
    Symbolic array of zeros. Equivalent to ``ZeroMatrix`` for matrices.
    """

    def __new__(cls, *shape):
        if len(shape) == 0:
            return S.Zero
        shape = map(_sympify, shape)
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        return self._args

    def as_explicit(self):
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("Cannot return explicit form for symbolic shape.")
        return ImmutableDenseNDimArray.zeros(*self.shape)

    def _get(self, item):
        return S.Zero


class OneArray(_ArrayExpr):
    """
    Symbolic array of ones.
    """

    def __new__(cls, *shape):
        if len(shape) == 0:
            return S.One
        shape = map(_sympify, shape)
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        return self._args

    def as_explicit(self):
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("Cannot return explicit form for symbolic shape.")
        return ImmutableDenseNDimArray([S.One for i in range(reduce(operator.mul, self.shape))]).reshape(*self.shape)

    def _get(self, item):
        return S.One


class _CodegenArrayAbstract(Basic):

    @property
    def subranks(self):
        """
        Returns the ranks of the objects in the uppermost tensor product inside
        the current object.  In case no tensor products are contained, return
        the atomic ranks.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> P = MatrixSymbol("P", 3, 3)

        Important: do not confuse the rank of the matrix with the rank of an array.

        >>> tp = tensorproduct(M, N, P)
        >>> tp.subranks
        [2, 2, 2]

        >>> co = tensorcontraction(tp, (1, 2), (3, 4))
        >>> co.subranks
        [2, 2, 2]
        """
        return self._subranks[:]

    def subrank(self):
        """
        The sum of ``subranks``.
        """
        return sum(self.subranks)

    @property
    def shape(self):
        return self._shape

    def doit(self, **hints):
        deep = hints.get("deep", True)
        if deep:
            return self.func(*[arg.doit(**hints) for arg in self.args])._canonicalize()
        else:
            return self._canonicalize()

class ArrayTensorProduct(_CodegenArrayAbstract):
    r"""
    Class to represent the tensor product of array-like objects.
    """

    def __new__(cls, *args, **kwargs):
        args = [_sympify(arg) for arg in args]

        canonicalize = kwargs.pop("canonicalize", False)

        ranks = [get_rank(arg) for arg in args]

        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        shapes = [get_shape(i) for i in args]

        if any(i is None for i in shapes):
            obj._shape = None
        else:
            obj._shape = tuple(j for i in shapes for j in i)
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        args = self.args
        args = self._flatten(args)

        ranks = [get_rank(arg) for arg in args]

        # Check if there are nested permutation and lift them up:
        permutation_cycles = []
        for i, arg in enumerate(args):
            if not isinstance(arg, PermuteDims):
                continue
            permutation_cycles.extend([[k + sum(ranks[:i]) for k in j] for j in arg.permutation.cyclic_form])
            args[i] = arg.expr
        if permutation_cycles:
            return _permute_dims(_array_tensor_product(*args), Permutation(sum(ranks)-1)*Permutation(permutation_cycles))

        if len(args) == 1:
            return args[0]

        # If any object is a ZeroArray, return a ZeroArray:
        if any(isinstance(arg, (ZeroArray, ZeroMatrix)) for arg in args):
            shapes = reduce(operator.add, [get_shape(i) for i in args], ())
            return ZeroArray(*shapes)

        # If there are contraction objects inside, transform the whole
        # expression into `ArrayContraction`:
        contractions = {i: arg for i, arg in enumerate(args) if isinstance(arg, ArrayContraction)}
        if contractions:
            ranks = [_get_subrank(arg) if isinstance(arg, ArrayContraction) else get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayContraction) else arg for arg in args])
            contraction_indices = [tuple(cumulative_ranks[i] + k for k in j) for i, arg in contractions.items() for j in arg.contraction_indices]
            return _array_contraction(tp, *contraction_indices)

        diagonals = {i: arg for i, arg in enumerate(args) if isinstance(arg, ArrayDiagonal)}
        if diagonals:
            inverse_permutation = []
            last_perm = []
            ranks = [get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            for i, arg in enumerate(args):
                if isinstance(arg, ArrayDiagonal):
                    i1 = get_rank(arg) - len(arg.diagonal_indices)
                    i2 = len(arg.diagonal_indices)
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(i1)])
                    last_perm.extend([cumulative_ranks[i] + j for j in range(i1, i1 + i2)])
                else:
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(get_rank(arg))])
            inverse_permutation.extend(last_perm)
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayDiagonal) else arg for arg in args])
            ranks2 = [_get_subrank(arg) if isinstance(arg, ArrayDiagonal) else get_rank(arg) for arg in args]
            cumulative_ranks2 = list(accumulate([0] + ranks2))[:-1]
            diagonal_indices = [tuple(cumulative_ranks2[i] + k for k in j) for i, arg in diagonals.items() for j in arg.diagonal_indices]
            return _permute_dims(_array_diagonal(tp, *diagonal_indices), _af_invert(inverse_permutation))

        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten(cls, args):
        args = [i for arg in args for i in (arg.args if isinstance(arg, cls) else [arg])]
        return args

    def as_explicit(self):
        return tensorproduct(*[arg.as_explicit() if hasattr(arg, "as_explicit") else arg for arg in self.args])


class ArrayAdd(_CodegenArrayAbstract):
    r"""
    Class for elementwise array additions.
    """

    def __new__(cls, *args, **kwargs):
        args = [_sympify(arg) for arg in args]
        ranks = [get_rank(arg) for arg in args]
        ranks = list(set(ranks))
        if len(ranks) != 1:
            raise ValueError("summing arrays of different ranks")
        shapes = [arg.shape for arg in args]
        if len({i for i in shapes if i is not None}) > 1:
            raise ValueError("mismatching shapes in addition")

        canonicalize = kwargs.pop("canonicalize", False)

        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        if any(i is None for i in shapes):
            obj._shape = None
        else:
            obj._shape = shapes[0]
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        args = self.args

        # Flatten:
        args = self._flatten_args(args)

        shapes = [get_shape(arg) for arg in args]
        args = [arg for arg in args if not isinstance(arg, (ZeroArray, ZeroMatrix))]
        if len(args) == 0:
            if any(i for i in shapes if i is None):
                raise NotImplementedError("cannot handle addition of ZeroMatrix/ZeroArray and undefined shape object")
            return ZeroArray(*shapes[0])
        elif len(args) == 1:
            return args[0]
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten_args(cls, args):
        new_args = []
        for arg in args:
            if isinstance(arg, ArrayAdd):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return new_args

    def as_explicit(self):
        return reduce(
            operator.add,
            [arg.as_explicit() if hasattr(arg, "as_explicit") else arg for arg in self.args])


class PermuteDims(_CodegenArrayAbstract):
    r"""
    Class to represent permutation of axes of arrays.

    Examples
    ========

    >>> from sympy.tensor.array import permutedims
    >>> from sympy import MatrixSymbol
    >>> M = MatrixSymbol("M", 3, 3)
    >>> cg = permutedims(M, [1, 0])

    The object ``cg`` represents the transposition of ``M``, as the permutation
    ``[1, 0]`` will act on its indices by switching them:

    `M_{ij} \Rightarrow M_{ji}`

    This is evident when transforming back to matrix form:

    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> convert_array_to_matrix(cg)
    M.T

    >>> N = MatrixSymbol("N", 3, 2)
    >>> cg = permutedims(N, [1, 0])
    >>> cg.shape
    (2, 3)

    There are optional parameters that can be used as alternative to the permutation:

    >>> from sympy.tensor.array.expressions import ArraySymbol, PermuteDims
    >>> M = ArraySymbol("M", (1, 2, 3, 4, 5))
    >>> expr = PermuteDims(M, index_order_old="ijklm", index_order_new="kijml")
    >>> expr
    PermuteDims(M, (0 2 1)(3 4))
    >>> expr.shape
    (3, 1, 2, 5, 4)

    Permutations of tensor products are simplified in order to achieve a
    standard form:

    >>> from sympy.tensor.array import tensorproduct
    >>> M = MatrixSymbol("M", 4, 5)
    >>> tp = tensorproduct(M, N)
    >>> tp.shape
    (4, 5, 3, 2)
    >>> perm1 = permutedims(tp, [2, 3, 1, 0])

    The args ``(M, N)`` have been sorted and the permutation has been
    simplified, the expression is equivalent:

    >>> perm1.expr.args
    (N, M)
    >>> perm1.shape
    (3, 2, 5, 4)
    >>> perm1.permutation
    (2 3)

    The permutation in its array form has been simplified from
    ``[2, 3, 1, 0]`` to ``[0, 1, 3, 2]``, as the arguments of the tensor
    product `M` and `N` have been switched:

    >>> perm1.permutation.array_form
    [0, 1, 3, 2]

    We can nest a second permutation:

    >>> perm2 = permutedims(perm1, [1, 0, 2, 3])
    >>> perm2.shape
    (2, 3, 5, 4)
    >>> perm2.permutation.array_form
    [1, 0, 3, 2]
    """

    def __new__(cls, expr, permutation=None, index_order_old=None, index_order_new=None, **kwargs):
        from sympy.combinatorics import Permutation
        expr = _sympify(expr)
        expr_rank = get_rank(expr)
        permutation = cls._get_permutation_from_arguments(permutation, index_order_old, index_order_new, expr_rank)
        permutation = Permutation(permutation)
        permutation_size = permutation.size
        if permutation_size != expr_rank:
            raise ValueError("Permutation size must be the length of the shape of expr")

        canonicalize = kwargs.pop("canonicalize", False)

        obj = Basic.__new__(cls, expr, permutation)
        obj._subranks = [get_rank(expr)]
        shape = get_shape(expr)
        if shape is None:
            obj._shape = None
        else:
            obj._shape = tuple(shape[permutation(i)] for i in range(len(shape)))
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        permutation = self.permutation
        if isinstance(expr, PermuteDims):
            subexpr = expr.expr
            subperm = expr.permutation
            permutation = permutation * subperm
            expr = subexpr
        if isinstance(expr, ArrayContraction):
            expr, permutation = self._PermuteDims_denestarg_ArrayContraction(expr, permutation)
        if isinstance(expr, ArrayTensorProduct):
            expr, permutation = self._PermuteDims_denestarg_ArrayTensorProduct(expr, permutation)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return ZeroArray(*[expr.shape[i] for i in permutation.array_form])
        plist = permutation.array_form
        if plist == sorted(plist):
            return expr
        return self.func(expr, permutation, canonicalize=False)

    @property
    def expr(self):
        return self.args[0]

    @property
    def permutation(self):
        return self.args[1]

    @classmethod
    def _PermuteDims_denestarg_ArrayTensorProduct(cls, expr, permutation):
        # Get the permutation in its image-form:
        perm_image_form = _af_invert(permutation.array_form)
        args = list(expr.args)
        # Starting index global position for every arg:
        cumul = list(accumulate([0] + expr.subranks))
        # Split `perm_image_form` into a list of list corresponding to the indices
        # of every argument:
        perm_image_form_in_components = [perm_image_form[cumul[i]:cumul[i+1]] for i in range(len(args))]
        # Create an index, target-position-key array:
        ps = [(i, sorted(comp)) for i, comp in enumerate(perm_image_form_in_components)]
        # Sort the array according to the target-position-key:
        # In this way, we define a canonical way to sort the arguments according
        # to the permutation.
        ps.sort(key=lambda x: x[1])
        # Read the inverse-permutation (i.e. image-form) of the args:
        perm_args_image_form = [i[0] for i in ps]
        # Apply the args-permutation to the `args`:
        args_sorted = [args[i] for i in perm_args_image_form]
        # Apply the args-permutation to the array-form of the permutation of the axes (of `expr`):
        perm_image_form_sorted_args = [perm_image_form_in_components[i] for i in perm_args_image_form]
        new_permutation = Permutation(_af_invert([j for i in perm_image_form_sorted_args for j in i]))
        return _array_tensor_product(*args_sorted), new_permutation

    @classmethod
    def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation):
        if not isinstance(expr, ArrayContraction):
            return expr, permutation
        if not isinstance(expr.expr, ArrayTensorProduct):
            return expr, permutation
        args = expr.expr.args
        subranks = [get_rank(arg) for arg in expr.expr.args]

        contraction_indices = expr.contraction_indices
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        cumul = list(accumulate([0] + subranks))

        # Spread the permutation in its array form across the args in the corresponding
        # tensor-product arguments with free indices:
        permutation_array_blocks_up = []
        image_form = _af_invert(permutation.array_form)
        counter = 0
        for i in range(len(subranks)):
            current = []
            for j in range(cumul[i], cumul[i+1]):
                if j in contraction_indices_flat:
                    continue
                current.append(image_form[counter])
                counter += 1
            permutation_array_blocks_up.append(current)

        # Get the map of axis repositioning for every argument of tensor-product:
        index_blocks = [list(range(cumul[i], cumul[i+1])) for i, e in enumerate(expr.subranks)]
        index_blocks_up = expr._push_indices_up(expr.contraction_indices, index_blocks)
        inverse_permutation = permutation**(-1)
        index_blocks_up_permuted = [[inverse_permutation(j) for j in i if j is not None] for i in index_blocks_up]

        # Sorting key is a list of tuple, first element is the index of `args`, second element of
        # the tuple is the sorting key to sort `args` of the tensor product:
        sorting_keys = list(enumerate(index_blocks_up_permuted))
        sorting_keys.sort(key=lambda x: x[1])

        # Now we can get the permutation acting on the args in its image-form:
        new_perm_image_form = [i[0] for i in sorting_keys]
        # Apply the args-level permutation to various elements:
        new_index_blocks = [index_blocks[i] for i in new_perm_image_form]
        new_index_perm_array_form = _af_invert([j for i in new_index_blocks for j in i])
        new_args = [args[i] for i in new_perm_image_form]
        new_contraction_indices = [tuple(new_index_perm_array_form[j] for j in i) for i in contraction_indices]
        new_expr = _array_contraction(_array_tensor_product(*new_args), *new_contraction_indices)
        new_permutation = Permutation(_af_invert([j for i in [permutation_array_blocks_up[k] for k in new_perm_image_form] for j in i]))
        return new_expr, new_permutation

    @classmethod
    def _check_permutation_mapping(cls, expr, permutation):
        subranks = expr.subranks
        index2arg = [i for i, arg in enumerate(expr.args) for j in range(expr.subranks[i])]
        permuted_indices = [permutation(i) for i in range(expr.subrank())]
        new_args = list(expr.args)
        arg_candidate_index = index2arg[permuted_indices[0]]
        current_indices = []
        new_permutation = []
        inserted_arg_cand_indices = set()
        for i, idx in enumerate(permuted_indices):
            if index2arg[idx] != arg_candidate_index:
                new_permutation.extend(current_indices)
                current_indices = []
                arg_candidate_index = index2arg[idx]
            current_indices.append(idx)
            arg_candidate_rank = subranks[arg_candidate_index]
            if len(current_indices) == arg_candidate_rank:
                new_permutation.extend(sorted(current_indices))
                local_current_indices = [j - min(current_indices) for j in current_indices]
                i1 = index2arg[i]
                new_args[i1] = _permute_dims(new_args[i1], Permutation(local_current_indices))
                inserted_arg_cand_indices.add(arg_candidate_index)
                current_indices = []
        new_permutation.extend(current_indices)

        # TODO: swap args positions in order to simplify the expression:
        # TODO: this should be in a function
        args_positions = list(range(len(new_args)))
        # Get possible shifts:
        maps = {}
        cumulative_subranks = [0] + list(accumulate(subranks))
        for i in range(len(subranks)):
            s = {index2arg[new_permutation[j]] for j in range(cumulative_subranks[i], cumulative_subranks[i+1])}
            if len(s) != 1:
                continue
            elem = next(iter(s))
            if i != elem:
                maps[i] = elem

        # Find cycles in the map:
        lines = []
        current_line = []
        while maps:
            if len(current_line) == 0:
                k, v = maps.popitem()
                current_line.append(k)
            else:
                k = current_line[-1]
                if k not in maps:
                    current_line = []
                    continue
                v = maps.pop(k)
            if v in current_line:
                lines.append(current_line)
                current_line = []
                continue
            current_line.append(v)
        for line in lines:
            for i, e in enumerate(line):
                args_positions[line[(i + 1) % len(line)]] = e

        # TODO: function in order to permute the args:
        permutation_blocks = [[new_permutation[cumulative_subranks[i] + j] for j in range(e)] for i, e in enumerate(subranks)]
        new_args = [new_args[i] for i in args_positions]
        new_permutation_blocks = [permutation_blocks[i] for i in args_positions]
        new_permutation2 = [j for i in new_permutation_blocks for j in i]
        return _array_tensor_product(*new_args), Permutation(new_permutation2)  # **(-1)

    @classmethod
    def _check_if_there_are_closed_cycles(cls, expr, permutation):
        args = list(expr.args)
        subranks = expr.subranks
        cyclic_form = permutation.cyclic_form
        cumulative_subranks = [0] + list(accumulate(subranks))
        cyclic_min = [min(i) for i in cyclic_form]
        cyclic_max = [max(i) for i in cyclic_form]
        cyclic_keep = []
        for i, cycle in enumerate(cyclic_form):
            flag = True
            for j in range(len(cumulative_subranks) - 1):
                if cyclic_min[i] >= cumulative_subranks[j] and cyclic_max[i] < cumulative_subranks[j+1]:
                    # Found a sinkable cycle.
                    args[j] = _permute_dims(args[j], Permutation([[k - cumulative_subranks[j] for k in cycle]]))
                    flag = False
                    break
            if flag:
                cyclic_keep.append(cycle)
        return _array_tensor_product(*args), Permutation(cyclic_keep, size=permutation.size)

    def nest_permutation(self):
        r"""
        DEPRECATED.
        """
        ret = self._nest_permutation(self.expr, self.permutation)
        if ret is None:
            return self
        return ret

    @classmethod
    def _nest_permutation(cls, expr, permutation):
        if isinstance(expr, ArrayTensorProduct):
            return _permute_dims(*cls._check_if_there_are_closed_cycles(expr, permutation))
        elif isinstance(expr, ArrayContraction):
            # Invert tree hierarchy: put the contraction above.
            cycles = permutation.cyclic_form
            newcycles = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *cycles)
            newpermutation = Permutation(newcycles)
            new_contr_indices = [tuple(newpermutation(j) for j in i) for i in expr.contraction_indices]
            return _array_contraction(PermuteDims(expr.expr, newpermutation), *new_contr_indices)
        elif isinstance(expr, ArrayAdd):
            return _array_add(*[PermuteDims(arg, permutation) for arg in expr.args])
        return None

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return permutedims(expr, self.permutation)

    @classmethod
    def _get_permutation_from_arguments(cls, permutation, index_order_old, index_order_new, dim):
        if permutation is None:
            if index_order_new is None or index_order_old is None:
                raise ValueError("Permutation not defined")
            return PermuteDims._get_permutation_from_index_orders(index_order_old, index_order_new, dim)
        else:
            if index_order_new is not None:
                raise ValueError("index_order_new cannot be defined with permutation")
            if index_order_old is not None:
                raise ValueError("index_order_old cannot be defined with permutation")
            return permutation

    @classmethod
    def _get_permutation_from_index_orders(cls, index_order_old, index_order_new, dim):
        if len(set(index_order_new)) != dim:
            raise ValueError("wrong number of indices in index_order_new")
        if len(set(index_order_old)) != dim:
            raise ValueError("wrong number of indices in index_order_old")
        if len(set.symmetric_difference(set(index_order_new), set(index_order_old))) > 0:
            raise ValueError("index_order_new and index_order_old must have the same indices")
        permutation = [index_order_old.index(i) for i in index_order_new]
        return permutation


class ArrayDiagonal(_CodegenArrayAbstract):
    r"""
    Class to represent the diagonal operator.

    Explanation
    ===========

    In a 2-dimensional array it returns the diagonal, this looks like the
    operation:

    `A_{ij} \rightarrow A_{ii}`

    The diagonal over axes 1 and 2 (the second and third) of the tensor product
    of two 2-dimensional arrays `A \otimes B` is

    `\Big[ A_{ab} B_{cd} \Big]_{abcd} \rightarrow \Big[ A_{ai} B_{id} \Big]_{adi}`

    In this last example the array expression has been reduced from
    4-dimensional to 3-dimensional. Notice that no contraction has occurred,
    rather there is a new index `i` for the diagonal, contraction would have
    reduced the array to 2 dimensions.

    Notice that the diagonalized out dimensions are added as new dimensions at
    the end of the indices.
    """

    def __new__(cls, expr, *diagonal_indices, **kwargs):
        expr = _sympify(expr)
        diagonal_indices = [Tuple(*sorted(i)) for i in diagonal_indices]
        canonicalize = kwargs.get("canonicalize", False)

        shape = get_shape(expr)
        if shape is not None:
            cls._validate(expr, *diagonal_indices, **kwargs)
            # Get new shape:
            positions, shape = cls._get_positions_shape(shape, diagonal_indices)
        else:
            positions = None
        if len(diagonal_indices) == 0:
            return expr
        obj = Basic.__new__(cls, expr, *diagonal_indices)
        obj._positions = positions
        obj._subranks = _get_subranks(expr)
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        diagonal_indices = self.diagonal_indices
        trivial_diags = [i for i in diagonal_indices if len(i) == 1]
        if len(trivial_diags) > 0:
            trivial_pos = {e[0]: i for i, e in enumerate(diagonal_indices) if len(e) == 1}
            diag_pos = {e: i for i, e in enumerate(diagonal_indices) if len(e) > 1}
            diagonal_indices_short = [i for i in diagonal_indices if len(i) > 1]
            rank1 = get_rank(self)
            rank2 = len(diagonal_indices)
            rank3 = rank1 - rank2
            inv_permutation = []
            counter1 = 0
            indices_down = ArrayDiagonal._push_indices_down(diagonal_indices_short, list(range(rank1)), get_rank(expr))
            for i in indices_down:
                if i in trivial_pos:
                    inv_permutation.append(rank3 + trivial_pos[i])
                elif isinstance(i, (Integer, int)):
                    inv_permutation.append(counter1)
                    counter1 += 1
                else:
                    inv_permutation.append(rank3 + diag_pos[i])
            permutation = _af_invert(inv_permutation)
            if len(diagonal_indices_short) > 0:
                return _permute_dims(_array_diagonal(expr, *diagonal_indices_short), permutation)
            else:
                return _permute_dims(expr, permutation)
        if isinstance(expr, ArrayAdd):
            return self._ArrayDiagonal_denest_ArrayAdd(expr, *diagonal_indices)
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayDiagonal_denest_ArrayDiagonal(expr, *diagonal_indices)
        if isinstance(expr, PermuteDims):
            return self._ArrayDiagonal_denest_PermuteDims(expr, *diagonal_indices)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            positions, shape = self._get_positions_shape(expr.shape, diagonal_indices)
            return ZeroArray(*shape)
        return self.func(expr, *diagonal_indices, canonicalize=False)

    @staticmethod
    def _validate(expr, *diagonal_indices, **kwargs):
        # Check that no diagonalization happens on indices with mismatched
        # dimensions:
        shape = get_shape(expr)
        for i in diagonal_indices:
            if any(j >= len(shape) for j in i):
                raise ValueError("index is larger than expression shape")
            if len({shape[j] for j in i}) != 1:
                raise ValueError("diagonalizing indices of different dimensions")
            if not kwargs.get("allow_trivial_diags", False) and len(i) <= 1:
                raise ValueError("need at least two axes to diagonalize")
            if len(set(i)) != len(i):
                raise ValueError("axis index cannot be repeated")

    @staticmethod
    def _remove_trivial_dimensions(shape, *diagonal_indices):
        return [tuple(j for j in i) for i in diagonal_indices if shape[i[0]] != 1]

    @property
    def expr(self):
        return self.args[0]

    @property
    def diagonal_indices(self):
        return self.args[1:]

    @staticmethod
    def _flatten(expr, *outer_diagonal_indices):
        inner_diagonal_indices = expr.diagonal_indices
        all_inner = [j for i in inner_diagonal_indices for j in i]
        all_inner.sort()
        # TODO: add API for total rank and cumulative rank:
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        outer_diagonal_indices = tuple(tuple(shifts[j] + j for j in i) for i in outer_diagonal_indices)
        diagonal_indices = inner_diagonal_indices + outer_diagonal_indices
        return _array_diagonal(expr.expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_ArrayAdd(cls, expr, *diagonal_indices):
        return _array_add(*[_array_diagonal(arg, *diagonal_indices) for arg in expr.args])

    @classmethod
    def _ArrayDiagonal_denest_ArrayDiagonal(cls, expr, *diagonal_indices):
        return cls._flatten(expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_PermuteDims(cls, expr: PermuteDims, *diagonal_indices):
        back_diagonal_indices = [[expr.permutation(j) for j in i] for i in diagonal_indices]
        nondiag = [i for i in range(get_rank(expr)) if not any(i in j for j in diagonal_indices)]
        back_nondiag = [expr.permutation(i) for i in nondiag]
        remap = {e: i for i, e in enumerate(sorted(back_nondiag))}
        new_permutation1 = [remap[i] for i in back_nondiag]
        shift = len(new_permutation1)
        diag_block_perm = [i + shift for i in range(len(back_diagonal_indices))]
        new_permutation = new_permutation1 + diag_block_perm
        return _permute_dims(
            _array_diagonal(
                expr.expr,
                *back_diagonal_indices
            ),
            new_permutation
        )

    def _push_indices_down_nonstatic(self, indices):
        transform = lambda x: self._positions[x] if x < len(self._positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    def _push_indices_up_nonstatic(self, indices):

        def transform(x):
            for i, e in enumerate(self._positions):
                if (isinstance(e, int) and x == e) or (isinstance(e, tuple) and x in e):
                    return i

        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_down(cls, diagonal_indices, indices, rank):
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)
        transform = lambda x: positions[x] if x < len(positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, diagonal_indices, indices, rank):
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)

        def transform(x):
            for i, e in enumerate(positions):
                if (isinstance(e, int) and x == e) or (isinstance(e, (tuple, Tuple)) and (x in e)):
                    return i

        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _get_positions_shape(cls, shape, diagonal_indices):
        data1 = tuple((i, shp) for i, shp in enumerate(shape) if not any(i in j for j in diagonal_indices))
        pos1, shp1 = zip(*data1) if data1 else ((), ())
        data2 = tuple((i, shape[i[0]]) for i in diagonal_indices)
        pos2, shp2 = zip(*data2) if data2 else ((), ())
        positions = pos1 + pos2
        shape = shp1 + shp2
        return positions, shape

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return tensordiagonal(expr, *self.diagonal_indices)


class ArrayElementwiseApplyFunc(_CodegenArrayAbstract):

    def __new__(cls, function, element):

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        obj = _CodegenArrayAbstract.__new__(cls, function, element)
        obj._subranks = _get_subranks(element)
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

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return expr.applyfunc(self.function)


class ArrayContraction(_CodegenArrayAbstract):
    r"""
    This class is meant to represent contractions of arrays in a form easily
    processable by the code printers.
    """

    def __new__(cls, expr, *contraction_indices, **kwargs):
        contraction_indices = _sort_contraction_indices(contraction_indices)
        expr = _sympify(expr)

        canonicalize = kwargs.get("canonicalize", False)

        obj = Basic.__new__(cls, expr, *contraction_indices)
        obj._subranks = _get_subranks(expr)
        obj._mapping = _get_mapping_from_subranks(obj._subranks)

        free_indices_to_position = {i: i for i in range(sum(obj._subranks)) if all(i not in cind for cind in contraction_indices)}
        obj._free_indices_to_position = free_indices_to_position

        shape = get_shape(expr)
        cls._validate(expr, *contraction_indices)
        if shape:
            shape = tuple(shp for i, shp in enumerate(shape) if not any(i in j for j in contraction_indices))
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        contraction_indices = self.contraction_indices

        if len(contraction_indices) == 0:
            return expr

        if isinstance(expr, ArrayContraction):
            return self._ArrayContraction_denest_ArrayContraction(expr, *contraction_indices)

        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return self._ArrayContraction_denest_ZeroArray(expr, *contraction_indices)

        if isinstance(expr, PermuteDims):
            return self._ArrayContraction_denest_PermuteDims(expr, *contraction_indices)

        if isinstance(expr, ArrayTensorProduct):
            expr, contraction_indices = self._sort_fully_contracted_args(expr, contraction_indices)
            expr, contraction_indices = self._lower_contraction_to_addends(expr, contraction_indices)
            if len(contraction_indices) == 0:
                return expr

        if isinstance(expr, ArrayDiagonal):
            return self._ArrayContraction_denest_ArrayDiagonal(expr, *contraction_indices)

        if isinstance(expr, ArrayAdd):
            return self._ArrayContraction_denest_ArrayAdd(expr, *contraction_indices)

        # Check single index contractions on 1-dimensional axes:
        contraction_indices = [i for i in contraction_indices if len(i) > 1 or get_shape(expr)[i[0]] != 1]
        if len(contraction_indices) == 0:
            return expr

        return self.func(expr, *contraction_indices, canonicalize=False)

    def __mul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError("Product of N-dim arrays is not uniquely defined. Use another method.")

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError("Product of N-dim arrays is not uniquely defined. Use another method.")

    @staticmethod
    def _validate(expr, *contraction_indices):
        shape = get_shape(expr)
        if shape is None:
            return

        # Check that no contraction happens when the shape is mismatched:
        for i in contraction_indices:
            if len({shape[j] for j in i if shape[j] != -1}) != 1:
                raise ValueError("contracting indices of different dimensions")

    @classmethod
    def _push_indices_down(cls, contraction_indices, indices):
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_down_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, contraction_indices, indices):
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_up_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _lower_contraction_to_addends(cls, expr, contraction_indices):
        if isinstance(expr, ArrayAdd):
            raise NotImplementedError()
        if not isinstance(expr, ArrayTensorProduct):
            return expr, contraction_indices
        subranks = expr.subranks
        cumranks = list(accumulate([0] + subranks))
        contraction_indices_remaining = []
        contraction_indices_args = [[] for i in expr.args]
        backshift = set()
        for contraction_group in contraction_indices:
            for j in range(len(expr.args)):
                if not isinstance(expr.args[j], ArrayAdd):
                    continue
                if all(cumranks[j] <= k < cumranks[j+1] for k in contraction_group):
                    contraction_indices_args[j].append([k - cumranks[j] for k in contraction_group])
                    backshift.update(contraction_group)
                    break
            else:
                contraction_indices_remaining.append(contraction_group)
        if len(contraction_indices_remaining) == len(contraction_indices):
            return expr, contraction_indices
        total_rank = get_rank(expr)
        shifts = list(accumulate([1 if i in backshift else 0 for i in range(total_rank)]))
        contraction_indices_remaining = [Tuple.fromiter(j - shifts[j] for j in i) for i in contraction_indices_remaining]
        ret = _array_tensor_product(*[
            _array_contraction(arg, *contr) for arg, contr in zip(expr.args, contraction_indices_args)
        ])
        return ret, contraction_indices_remaining

    def split_multiple_contractions(self):
        """
        Recognize multiple contractions and attempt at rewriting them as paired-contractions.

        This allows some contractions involving more than two indices to be
        rewritten as multiple contractions involving two indices, thus allowing
        the expression to be rewritten as a matrix multiplication line.

        Examples:

        * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C`

        Care for:
        - matrix being diagonalized (i.e. `A_ii`)
        - vectors being diagonalized (i.e. `a_i0`)

        Multiple contractions can be split into matrix multiplications if
        not more than two arguments are non-diagonals or non-vectors.
        Vectors get diagonalized while diagonal matrices remain diagonal.
        The non-diagonal matrices can be at the beginning or at the end
        of the final matrix multiplication line.
        """

        editor = _EditArrayContraction(self)

        contraction_indices = self.contraction_indices

        onearray_insert = []

        for indl, links in enumerate(contraction_indices):
            if len(links) <= 2:
                continue

            # Check multiple contractions:
            #
            # Examples:
            #
            # * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C \otimes OneArray(1)` with permutation (1 2)
            #
            # Care for:
            # - matrix being diagonalized (i.e. `A_ii`)
            # - vectors being diagonalized (i.e. `a_i0`)

            # Multiple contractions can be split into matrix multiplications if
            # not more than three arguments are non-diagonals or non-vectors.
            #
            # Vectors get diagonalized while diagonal matrices remain diagonal.
            # The non-diagonal matrices can be at the beginning or at the end
            # of the final matrix multiplication line.

            positions = editor.get_mapping_for_index(indl)

            # Also consider the case of diagonal matrices being contracted:
            current_dimension = self.expr.shape[links[0]]

            not_vectors = []
            vectors = []
            for arg_ind, rel_ind in positions:
                arg = editor.args_with_ind[arg_ind]
                mat = arg.element
                abs_arg_start, abs_arg_end = editor.get_absolute_range(arg)
                other_arg_pos = 1-rel_ind
                other_arg_abs = abs_arg_start + other_arg_pos
                if ((1 not in mat.shape) or
                    ((current_dimension == 1) is True and mat.shape != (1, 1)) or
                    any(other_arg_abs in l for li, l in enumerate(contraction_indices) if li != indl)
                ):
                    not_vectors.append((arg, rel_ind))
                else:
                    vectors.append((arg, rel_ind))
            if len(not_vectors) > 2:
                # If more than two arguments in the multiple contraction are
                # non-vectors and non-diagonal matrices, we cannot find a way
                # to split this contraction into a matrix multiplication line:
                continue
            # Three cases to handle:
            # - zero non-vectors
            # - one non-vector
            # - two non-vectors
            for v, rel_ind in vectors:
                v.element = diagonalize_vector(v.element)
            vectors_to_loop = not_vectors[:1] + vectors + not_vectors[1:]
            first_not_vector, rel_ind = vectors_to_loop[0]
            new_index = first_not_vector.indices[rel_ind]

            for v, rel_ind in vectors_to_loop[1:-1]:
                v.indices[rel_ind] = new_index
                new_index = editor.get_new_contraction_index()
                assert v.indices.index(None) == 1 - rel_ind
                v.indices[v.indices.index(None)] = new_index
                onearray_insert.append(v)

            last_vec, rel_ind = vectors_to_loop[-1]
            last_vec.indices[rel_ind] = new_index

        for v in onearray_insert:
            editor.insert_after(v, _ArgE(OneArray(1), [None]))

        return editor.to_array_contraction()

    def flatten_contraction_of_diagonal(self):
        if not isinstance(self.expr, ArrayDiagonal):
            return self
        contraction_down = self.expr._push_indices_down(self.expr.diagonal_indices, self.contraction_indices)
        new_contraction_indices = []
        diagonal_indices = self.expr.diagonal_indices[:]
        for i in contraction_down:
            contraction_group = list(i)
            for j in i:
                diagonal_with = [k for k in diagonal_indices if j in k]
                contraction_group.extend([l for k in diagonal_with for l in k])
                diagonal_indices = [k for k in diagonal_indices if k not in diagonal_with]
            new_contraction_indices.append(sorted(set(contraction_group)))

        new_contraction_indices = ArrayDiagonal._push_indices_up(diagonal_indices, new_contraction_indices)
        return _array_contraction(
            _array_diagonal(
                self.expr.expr,
                *diagonal_indices
            ),
            *new_contraction_indices
        )

    @staticmethod
    def _get_free_indices_to_position_map(free_indices, contraction_indices):
        free_indices_to_position = {}
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        counter = 0
        for ind in free_indices:
            while counter in flattened_contraction_indices:
                counter += 1
            free_indices_to_position[ind] = counter
            counter += 1
        return free_indices_to_position

    @staticmethod
    def _get_index_shifts(expr):
        """
        Get the mapping of indices at the positions before the contraction
        occurs.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> cg = tensorcontraction(tensorproduct(M, N), [1, 2])
        >>> cg._get_index_shifts(cg)
        [0, 2]

        Indeed, ``cg`` after the contraction has two dimensions, 0 and 1. They
        need to be shifted by 0 and 2 to get the corresponding positions before
        the contraction (that is, 0 and 3).
        """
        inner_contraction_indices = expr.contraction_indices
        all_inner = [j for i in inner_contraction_indices for j in i]
        all_inner.sort()
        # TODO: add API for total rank and cumulative rank:
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        return shifts

    @staticmethod
    def _convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices):
        shifts = ArrayContraction._get_index_shifts(expr)
        outer_contraction_indices = tuple(tuple(shifts[j] + j for j in i) for i in outer_contraction_indices)
        return outer_contraction_indices

    @staticmethod
    def _flatten(expr, *outer_contraction_indices):
        inner_contraction_indices = expr.contraction_indices
        outer_contraction_indices = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices)
        contraction_indices = inner_contraction_indices + outer_contraction_indices
        return _array_contraction(expr.expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ArrayContraction(cls, expr, *contraction_indices):
        return cls._flatten(expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ZeroArray(cls, expr, *contraction_indices):
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        shape = [e for i, e in enumerate(expr.shape) if i not in contraction_indices_flat]
        return ZeroArray(*shape)

    @classmethod
    def _ArrayContraction_denest_ArrayAdd(cls, expr, *contraction_indices):
        return _array_add(*[_array_contraction(i, *contraction_indices) for i in expr.args])

    @classmethod
    def _ArrayContraction_denest_PermuteDims(cls, expr, *contraction_indices):
        permutation = expr.permutation
        plist = permutation.array_form
        new_contraction_indices = [tuple(permutation(j) for j in i) for i in contraction_indices]
        new_plist = [i for i in plist if not any(i in j for j in new_contraction_indices)]
        new_plist = cls._push_indices_up(new_contraction_indices, new_plist)
        return _permute_dims(
            _array_contraction(expr.expr, *new_contraction_indices),
            Permutation(new_plist)
        )

    @classmethod
    def _ArrayContraction_denest_ArrayDiagonal(cls, expr: 'ArrayDiagonal', *contraction_indices):
        diagonal_indices = list(expr.diagonal_indices)
        down_contraction_indices = expr._push_indices_down(expr.diagonal_indices, contraction_indices, get_rank(expr.expr))
        # Flatten diagonally contracted indices:
        down_contraction_indices = [[k for j in i for k in (j if isinstance(j, (tuple, Tuple)) else [j])] for i in down_contraction_indices]
        new_contraction_indices = []
        for contr_indgrp in down_contraction_indices:
            ind = contr_indgrp[:]
            for j, diag_indgrp in enumerate(diagonal_indices):
                if diag_indgrp is None:
                    continue
                if any(i in diag_indgrp for i in contr_indgrp):
                    ind.extend(diag_indgrp)
                    diagonal_indices[j] = None
            new_contraction_indices.append(sorted(set(ind)))

        new_diagonal_indices_down = [i for i in diagonal_indices if i is not None]
        new_diagonal_indices = ArrayContraction._push_indices_up(new_contraction_indices, new_diagonal_indices_down)
        return _array_diagonal(
            _array_contraction(expr.expr, *new_contraction_indices),
            *new_diagonal_indices
        )

    @classmethod
    def _sort_fully_contracted_args(cls, expr, contraction_indices):
        if expr.shape is None:
            return expr, contraction_indices
        cumul = list(accumulate([0] + expr.subranks))
        index_blocks = [list(range(cumul[i], cumul[i+1])) for i in range(len(expr.args))]
        contraction_indices_flat = {j for i in contraction_indices for j in i}
        fully_contracted = [all(j in contraction_indices_flat for j in range(cumul[i], cumul[i+1])) for i, arg in enumerate(expr.args)]
        new_pos = sorted(range(len(expr.args)), key=lambda x: (0, default_sort_key(expr.args[x])) if fully_contracted[x] else (1,))
        new_args = [expr.args[i] for i in new_pos]
        new_index_blocks_flat = [j for i in new_pos for j in index_blocks[i]]
        index_permutation_array_form = _af_invert(new_index_blocks_flat)
        new_contraction_indices = [tuple(index_permutation_array_form[j] for j in i) for i in contraction_indices]
        new_contraction_indices = _sort_contraction_indices(new_contraction_indices)
        return _array_tensor_product(*new_args), new_contraction_indices

    def _get_contraction_tuples(self):
        r"""
        Return tuples containing the argument index and position within the
        argument of the index position.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)

        >>> cg = tensorcontraction(tensorproduct(A, B), (1, 2))
        >>> cg._get_contraction_tuples()
        [[(0, 1), (1, 0)]]

        Notes
        =====

        Here the contraction pair `(1, 2)` meaning that the 2nd and 3rd indices
        of the tensor product `A\otimes B` are contracted, has been transformed
        into `(0, 1)` and `(1, 0)`, identifying the same indices in a different
        notation. `(0, 1)` is the second index (1) of the first argument (i.e.
                0 or `A`). `(1, 0)` is the first index (i.e. 0) of the second
        argument (i.e. 1 or `B`).
        """
        mapping = self._mapping
        return [[mapping[j] for j in i] for i in self.contraction_indices]

    @staticmethod
    def _contraction_tuples_to_contraction_indices(expr, contraction_tuples):
        # TODO: check that `expr` has `.subranks`:
        ranks = expr.subranks
        cumulative_ranks = [0] + list(accumulate(ranks))
        return [tuple(cumulative_ranks[j]+k for j, k in i) for i in contraction_tuples]

    @property
    def free_indices(self):
        return self._free_indices[:]

    @property
    def free_indices_to_position(self):
        return dict(self._free_indices_to_position)

    @property
    def expr(self):
        return self.args[0]

    @property
    def contraction_indices(self):
        return self.args[1:]

    def _contraction_indices_to_components(self):
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            raise NotImplementedError("only for contractions of tensor products")
        ranks = expr.subranks
        mapping = {}
        counter = 0
        for i, rank in enumerate(ranks):
            for j in range(rank):
                mapping[counter] = (i, j)
                counter += 1
        return mapping

    def sort_args_by_name(self):
        """
        Sort arguments in the tensor product so that their order is lexicographical.

        Examples
        ========

        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        >>> cg = convert_matrix_to_array(C*D*A*B)
        >>> cg
        ArrayContraction(ArrayTensorProduct(A, D, C, B), (0, 3), (1, 6), (2, 5))
        >>> cg.sort_args_by_name()
        ArrayContraction(ArrayTensorProduct(A, D, B, C), (0, 3), (1, 4), (2, 7))
        """
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            return self
        args = expr.args
        sorted_data = sorted(enumerate(args), key=lambda x: default_sort_key(x[1]))
        pos_sorted, args_sorted = zip(*sorted_data)
        reordering_map = {i: pos_sorted.index(i) for i, arg in enumerate(args)}
        contraction_tuples = self._get_contraction_tuples()
        contraction_tuples = [[(reordering_map[j], k) for j, k in i] for i in contraction_tuples]
        c_tp = _array_tensor_product(*args_sorted)
        new_contr_indices = self._contraction_tuples_to_contraction_indices(
                c_tp,
                contraction_tuples
        )
        return _array_contraction(c_tp, *new_contr_indices)

    def _get_contraction_links(self):
        r"""
        Returns a dictionary of links between arguments in the tensor product
        being contracted.

        See the example for an explanation of the values.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        Matrix multiplications are pairwise contractions between neighboring
        matrices:

        `A_{ij} B_{jk} C_{kl} D_{lm}`

        >>> cg = convert_matrix_to_array(A*B*C*D)
        >>> cg
        ArrayContraction(ArrayTensorProduct(B, C, A, D), (0, 5), (1, 2), (3, 6))

        >>> cg._get_contraction_links()
        {0: {0: (2, 1), 1: (1, 0)}, 1: {0: (0, 1), 1: (3, 0)}, 2: {1: (0, 0)}, 3: {0: (1, 1)}}

        This dictionary is interpreted as follows: argument in position 0 (i.e.
        matrix `A`) has its second index (i.e. 1) contracted to `(1, 0)`, that
        is argument in position 1 (matrix `B`) on the first index slot of `B`,
        this is the contraction provided by the index `j` from `A`.

        The argument in position 1 (that is, matrix `B`) has two contractions,
        the ones provided by the indices `j` and `k`, respectively the first
        and second indices (0 and 1 in the sub-dict).  The link `(0, 1)` and
        `(2, 0)` respectively. `(0, 1)` is the index slot 1 (the 2nd) of
        argument in position 0 (that is, `A_{\ldot j}`), and so on.
        """
        args, dlinks = _get_contraction_links([self], self.subranks, *self.contraction_indices)
        return dlinks

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return tensorcontraction(expr, *self.contraction_indices)


class Reshape(_CodegenArrayAbstract):
    """
    Reshape the dimensions of an array expression.

    Examples
    ========

    >>> from sympy.tensor.array.expressions import ArraySymbol, Reshape
    >>> A = ArraySymbol("A", (6,))
    >>> A.shape
    (6,)
    >>> Reshape(A, (3, 2)).shape
    (3, 2)

    Check the component-explicit forms:

    >>> A.as_explicit()
    [A[0], A[1], A[2], A[3], A[4], A[5]]
    >>> Reshape(A, (3, 2)).as_explicit()
    [[A[0], A[1]], [A[2], A[3]], [A[4], A[5]]]

    """

    def __new__(cls, expr, shape):
        expr = _sympify(expr)
        if not isinstance(shape, Tuple):
            shape = Tuple(*shape)
        if Equality(Mul.fromiter(expr.shape), Mul.fromiter(shape)) == False:
            raise ValueError("shape mismatch")
        obj = Expr.__new__(cls, expr, shape)
        obj._shape = tuple(shape)
        obj._expr = expr
        return obj

    @property
    def shape(self):
        return self._shape

    @property
    def expr(self):
        return self._expr

    def doit(self, *args, **kwargs):
        if kwargs.get("deep", True):
            expr = self.expr.doit(*args, **kwargs)
        else:
            expr = self.expr
        if isinstance(expr, (MatrixBase, NDimArray)):
            return expr.reshape(*self.shape)
        return Reshape(expr, self.shape)

    def as_explicit(self):
        ee = self.expr
        if hasattr(ee, "as_explicit"):
            ee = ee.as_explicit()
        if isinstance(ee, MatrixBase):
            from sympy import Array
            ee = Array(ee)
        elif isinstance(ee, MatrixExpr):
            return self
        return ee.reshape(*self.shape)


class _ArgE:
    """
    The ``_ArgE`` object contains references to the array expression
    (``.element``) and a list containing the information about index
    contractions (``.indices``).

    Index contractions are numbered and contracted indices show the number of
    the contraction. Uncontracted indices have ``None`` value.

    For example:
    ``_ArgE(M, [None, 3])``
    This object means that expression ``M`` is part of an array contraction
    and has two indices, the first is not contracted (value ``None``),
    the second index is contracted to the 4th (i.e. number ``3``) group of the
    array contraction object.
    """
    indices: list[int | None]

    def __init__(self, element, indices: list[int | None] | None = None):
        self.element = element
        if indices is None:
            self.indices = [None for i in range(get_rank(element))]
        else:
            self.indices = indices

    def __str__(self):
        return "_ArgE(%s, %s)" % (self.element, self.indices)

    __repr__ = __str__


class _IndPos:
    """
    Index position, requiring two integers in the constructor:

    - arg: the position of the argument in the tensor product,
    - rel: the relative position of the index inside the argument.
    """
    def __init__(self, arg: int, rel: int):
        self.arg = arg
        self.rel = rel

    def __str__(self):
        return "_IndPos(%i, %i)" % (self.arg, self.rel)

    __repr__ = __str__

    def __iter__(self):
        yield from [self.arg, self.rel]


class _EditArrayContraction:
    """
    Utility class to help manipulate array contraction objects.

    This class takes as input an ``ArrayContraction`` object and turns it into
    an editable object.

    The field ``args_with_ind`` of this class is a list of ``_ArgE`` objects
    which can be used to easily edit the contraction structure of the
    expression.

    Once editing is finished, the ``ArrayContraction`` object may be recreated
    by calling the ``.to_array_contraction()`` method.
    """

    def __init__(self, base_array: typing.Union[ArrayContraction, ArrayDiagonal, ArrayTensorProduct]):

        expr: Basic
        diagonalized: tuple[tuple[int, ...], ...]
        contraction_indices: list[tuple[int]]
        if isinstance(base_array, ArrayContraction):
            mapping = _get_mapping_from_subranks(base_array.subranks)
            expr = base_array.expr
            contraction_indices = base_array.contraction_indices
            diagonalized = ()
        elif isinstance(base_array, ArrayDiagonal):

            if isinstance(base_array.expr, ArrayContraction):
                mapping = _get_mapping_from_subranks(base_array.expr.subranks)
                expr = base_array.expr.expr
                diagonalized = ArrayContraction._push_indices_down(base_array.expr.contraction_indices, base_array.diagonal_indices)
                contraction_indices = base_array.expr.contraction_indices
            elif isinstance(base_array.expr, ArrayTensorProduct):
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []
            else:
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []

        elif isinstance(base_array, ArrayTensorProduct):
            expr = base_array
            contraction_indices = []
            diagonalized = ()
        else:
            raise NotImplementedError()

        if isinstance(expr, ArrayTensorProduct):
            args = list(expr.args)
        else:
            args = [expr]

        args_with_ind: list[_ArgE] = [_ArgE(arg) for arg in args]
        for i, contraction_tuple in enumerate(contraction_indices):
            for j in contraction_tuple:
                arg_pos, rel_pos = mapping[j]
                args_with_ind[arg_pos].indices[rel_pos] = i
        self.args_with_ind: list[_ArgE] = args_with_ind
        self.number_of_contraction_indices: int = len(contraction_indices)
        self._track_permutation: list[list[int]] | None = None

        mapping = _get_mapping_from_subranks(base_array.subranks)

        # Trick: add diagonalized indices as negative indices into the editor object:
        for i, e in enumerate(diagonalized):
            for j in e:
                arg_pos, rel_pos = mapping[j]
                self.args_with_ind[arg_pos].indices[rel_pos] = -1 - i

    def insert_after(self, arg: _ArgE, new_arg: _ArgE):
        pos = self.args_with_ind.index(arg)
        self.args_with_ind.insert(pos + 1, new_arg)

    def get_new_contraction_index(self):
        self.number_of_contraction_indices += 1
        return self.number_of_contraction_indices - 1

    def refresh_indices(self):
        updates = {}
        for arg_with_ind in self.args_with_ind:
            updates.update({i: -1 for i in arg_with_ind.indices if i is not None})
        for i, e in enumerate(sorted(updates)):
            updates[e] = i
        self.number_of_contraction_indices = len(updates)
        for arg_with_ind in self.args_with_ind:
            arg_with_ind.indices = [updates.get(i, None) for i in arg_with_ind.indices]

    def merge_scalars(self):
        scalars = []
        for arg_with_ind in self.args_with_ind:
            if len(arg_with_ind.indices) == 0:
                scalars.append(arg_with_ind)
        for i in scalars:
            self.args_with_ind.remove(i)
        scalar = Mul.fromiter([i.element for i in scalars])
        if len(self.args_with_ind) == 0:
            self.args_with_ind.append(_ArgE(scalar))
        else:
            from sympy.tensor.array.expressions.from_array_to_matrix import _a2m_tensor_product
            self.args_with_ind[0].element = _a2m_tensor_product(scalar, self.args_with_ind[0].element)

    def to_array_contraction(self):

        # Count the ranks of the arguments:
        counter = 0
        # Create a collector for the new diagonal indices:
        diag_indices = defaultdict(list)

        count_index_freq = Counter()
        for arg_with_ind in self.args_with_ind:
            count_index_freq.update(Counter(arg_with_ind.indices))

        free_index_count = count_index_freq[None]

        # Construct the inverse permutation:
        inv_perm1 = []
        inv_perm2 = []
        # Keep track of which diagonal indices have already been processed:
        done = set()

        # Counter for the diagonal indices:
        counter4 = 0

        for arg_with_ind in self.args_with_ind:
            # If some diagonalization axes have been removed, they should be
            # permuted in order to keep the permutation.
            # Add permutation here
            counter2 = 0  # counter for the indices
            for i in arg_with_ind.indices:
                if i is None:
                    inv_perm1.append(counter4)
                    counter2 += 1
                    counter4 += 1
                    continue
                if i >= 0:
                    continue
                # Reconstruct the diagonal indices:
                diag_indices[-1 - i].append(counter + counter2)
                if count_index_freq[i] == 1 and i not in done:
                    inv_perm1.append(free_index_count - 1 - i)
                    done.add(i)
                elif i not in done:
                    inv_perm2.append(free_index_count - 1 - i)
                    done.add(i)
                counter2 += 1
            # Remove negative indices to restore a proper editor object:
            arg_with_ind.indices = [i if i is not None and i >= 0 else None for i in arg_with_ind.indices]
            counter += len([i for i in arg_with_ind.indices if i is None or i < 0])

        inverse_permutation = inv_perm1 + inv_perm2
        permutation = _af_invert(inverse_permutation)

        # Get the diagonal indices after the detection of HadamardProduct in the expression:
        diag_indices_filtered = [tuple(v) for v in diag_indices.values() if len(v) > 1]

        self.merge_scalars()
        self.refresh_indices()
        args = [arg.element for arg in self.args_with_ind]
        contraction_indices = self.get_contraction_indices()
        expr = _array_contraction(_array_tensor_product(*args), *contraction_indices)
        expr2 = _array_diagonal(expr, *diag_indices_filtered)
        if self._track_permutation is not None:
            permutation2 = _af_invert([j for i in self._track_permutation for j in i])
            expr2 = _permute_dims(expr2, permutation2)

        expr3 = _permute_dims(expr2, permutation)
        return expr3

    def get_contraction_indices(self) -> list[list[int]]:
        contraction_indices: list[list[int]] = [[] for i in range(self.number_of_contraction_indices)]
        current_position: int = 0
        for arg_with_ind in self.args_with_ind:
            for j in arg_with_ind.indices:
                if j is not None:
                    contraction_indices[j].append(current_position)
                current_position += 1
        return contraction_indices

    def get_mapping_for_index(self, ind) -> list[_IndPos]:
        if ind >= self.number_of_contraction_indices:
            raise ValueError("index value exceeding the index range")
        positions: list[_IndPos] = []
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, arg_ind in enumerate(arg_with_ind.indices):
                if ind == arg_ind:
                    positions.append(_IndPos(i, j))
        return positions

    def get_contraction_indices_to_ind_rel_pos(self) -> list[list[_IndPos]]:
        contraction_indices: list[list[_IndPos]] = [[] for i in range(self.number_of_contraction_indices)]
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, ind in enumerate(arg_with_ind.indices):
                if ind is not None:
                    contraction_indices[ind].append(_IndPos(i, j))
        return contraction_indices

    def count_args_with_index(self, index: int) -> int:
        """
        Count the number of arguments that have the given index.
        """
        counter: int = 0
        for arg_with_ind in self.args_with_ind:
            if index in arg_with_ind.indices:
                counter += 1
        return counter

    def get_args_with_index(self, index: int) -> list[_ArgE]:
        """
        Get a list of arguments having the given index.
        """
        ret: list[_ArgE] = [i for i in self.args_with_ind if index in i.indices]
        return ret

    @property
    def number_of_diagonal_indices(self):
        data = set()
        for arg in self.args_with_ind:
            data.update({i for i in arg.indices if i is not None and i < 0})
        return len(data)

    def track_permutation_start(self):
        permutation = []
        perm_diag = []
        counter = 0
        counter2 = -1
        for arg_with_ind in self.args_with_ind:
            perm = []
            for i in arg_with_ind.indices:
                if i is not None:
                    if i < 0:
                        perm_diag.append(counter2)
                        counter2 -= 1
                    continue
                perm.append(counter)
                counter += 1
            permutation.append(perm)
        max_ind = max(max(i) if i else -1 for i in permutation) if permutation else -1
        perm_diag = [max_ind - i for i in perm_diag]
        self._track_permutation = permutation + [perm_diag]

    def track_permutation_merge(self, destination: _ArgE, from_element: _ArgE):
        index_destination = self.args_with_ind.index(destination)
        index_element = self.args_with_ind.index(from_element)
        self._track_permutation[index_destination].extend(self._track_permutation[index_element]) # type: ignore
        self._track_permutation.pop(index_element) # type: ignore

    def get_absolute_free_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        Return the range of the free indices of the arg as absolute positions
        among all free indices.
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_free_indices = len([i for i in arg_with_ind.indices if i is None])
            if arg_with_ind == arg:
                return counter, counter + number_free_indices
            counter += number_free_indices
        raise IndexError("argument not found")

    def get_absolute_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        Return the absolute range of indices for arg, disregarding dummy
        indices.
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_indices = len(arg_with_ind.indices)
            if arg_with_ind == arg:
                return counter, counter + number_indices
            counter += number_indices
        raise IndexError("argument not found")


def get_rank(expr):
    if isinstance(expr, (MatrixExpr, MatrixElement)):
        return 2
    if isinstance(expr, _CodegenArrayAbstract):
        return len(expr.shape)
    if isinstance(expr, NDimArray):
        return expr.rank()
    if isinstance(expr, Indexed):
        return expr.rank
    if isinstance(expr, IndexedBase):
        shape = expr.shape
        if shape is None:
            return -1
        else:
            return len(shape)
    if hasattr(expr, "shape"):
        return len(expr.shape)
    return 0


def _get_subrank(expr):
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subrank()
    return get_rank(expr)


def _get_subranks(expr):
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subranks
    else:
        return [get_rank(expr)]


def get_shape(expr):
    if hasattr(expr, "shape"):
        return expr.shape
    return ()


def nest_permutation(expr):
    if isinstance(expr, PermuteDims):
        return expr.nest_permutation()
    else:
        return expr


def _array_tensor_product(*args, **kwargs):
    return ArrayTensorProduct(*args, canonicalize=True, **kwargs)


def _array_contraction(expr, *contraction_indices, **kwargs):
    return ArrayContraction(expr, *contraction_indices, canonicalize=True, **kwargs)


def _array_diagonal(expr, *diagonal_indices, **kwargs):
    return ArrayDiagonal(expr, *diagonal_indices, canonicalize=True, **kwargs)


def _permute_dims(expr, permutation, **kwargs):
    return PermuteDims(expr, permutation, canonicalize=True, **kwargs)


def _array_add(*args, **kwargs):
    return ArrayAdd(*args, canonicalize=True, **kwargs)


def _get_array_element_or_slice(expr, indices):
    return ArrayElement(expr, indices)
<|endoftext|>from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar


def wavefunction(n, x):
    """
    Returns the wavefunction for particle on ring.

    Parameters
    ==========

    n : The quantum number.
        Here ``n`` can be positive as well as negative
        which can be used to describe the direction of motion of particle.
    x :
        The angle.

    Examples
    ========

    >>> from sympy.physics.pring import wavefunction
    >>> from sympy import Symbol, integrate, pi
    >>> x=Symbol("x")
    >>> wavefunction(1, x)
    sqrt(2)*exp(I*x)/(2*sqrt(pi))
    >>> wavefunction(2, x)
    sqrt(2)*exp(2*I*x)/(2*sqrt(pi))
    >>> wavefunction(3, x)
    sqrt(2)*exp(3*I*x)/(2*sqrt(pi))

    The normalization of the wavefunction is:

    >>> integrate(wavefunction(2, x)*wavefunction(-2, x), (x, 0, 2*pi))
    1
    >>> integrate(wavefunction(4, x)*wavefunction(-4, x), (x, 0, 2*pi))
    1

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    """
    # sympify arguments
    n, x = S(n), S(x)
    return exp(n * I * x) / sqrt(2 * pi)


def energy(n, m, r):
    """
    Returns the energy of the state corresponding to quantum number ``n``.

    E=(n**2 * (hcross)**2) / (2 * m * r**2)

    Parameters
    ==========

    n :
        The quantum number.
    m :
        Mass of the particle.
    r :
        Radius of circle.

    Examples
    ========

    >>> from sympy.physics.pring import energy
    >>> from sympy import Symbol
    >>> m=Symbol("m")
    >>> r=Symbol("r")
    >>> energy(1, m, r)
    hbar**2/(2*m*r**2)
    >>> energy(2, m, r)
    2*hbar**2/(m*r**2)
    >>> energy(-2, 2.0, 3.0)
    0.111111111111111*hbar**2

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    """
    n, m, r = S(n), S(m), S(r)
    if n.is_integer:
        return (n**2 * hbar**2) / (2 * m * r**2)
    else:
        raise ValueError("'n' must be integer")
<|endoftext|># -*- coding: utf-8 -*-
r"""
Wigner, Clebsch-Gordan, Racah, and Gaunt coefficients

Collection of functions for calculating Wigner 3j, 6j, 9j,
Clebsch-Gordan, Racah as well as Gaunt coefficients exactly, all
evaluating to a rational number times the square root of a rational
number [Rasch03]_.

Please see the description of the individual functions for further
details and examples.

References
==========

.. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
  T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)
.. [Regge59] 'Symmetry Properties of Racah Coefficients',
  T. Regge, Nuovo Cimento, Volume 11, pp. 116 (1959)
.. [Edmonds74] A. R. Edmonds. Angular momentum in quantum mechanics.
  Investigations in physics, 4.; Investigations in physics, no. 4.
  Princeton, N.J., Princeton University Press, 1957.
.. [Rasch03] J. Rasch and A. C. H. Yu, 'Efficient Storage Scheme for
  Pre-calculated Wigner 3j, 6j and Gaunt Coefficients', SIAM
  J. Sci. Comput. Volume 25, Issue 4, pp. 1416-1428 (2003)
.. [Liberatodebrito82] 'FORTRAN program for the integral of three
  spherical harmonics', A. Liberato de Brito,
  Comput. Phys. Commun., Volume 25, pp. 81-85 (1982)
.. [Homeier96] 'Some Properties of the Coupling Coefficients of Real
  Spherical Harmonics and Their Relation to Gaunt Coefficients',
  H. H. H. Homeier and E. O. Steinborn J. Mol. Struct., Volume 368,
  pp. 31-37 (1996)

Credits and Copyright
=====================

This code was taken from Sage with the permission of all authors:

https://groups.google.com/forum/#!topic/sage-devel/M4NZdu-7O38

Authors
=======

- Jens Rasch (2009-03-24): initial version for Sage

- Jens Rasch (2009-05-31): updated to sage-4.0

- Oscar Gerardo Lazo Arjona (2017-06-18): added Wigner D matrices

- Phil Adam LeMaitre (2022-09-19): added real Gaunt coefficient

Copyright (C) 2008 Jens Rasch <jyr2000@gmail.com>

"""
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import int_valued
from sympy.core.function import Function
from sympy.core.numbers import (Float, I, Integer, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int

# This list of precomputed factorials is needed to massively
# accelerate future calculations of the various coefficients
_Factlist = [1]


def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.

    Parameters
    ==========

    nn : integer
        Highest factorial to be computed.

    Returns
    =======

    list of integers :
        The list of precomputed factorials.

    Examples
    ========

    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]


def _int_or_halfint(value):
    """return Python int unless value is half-int (then return float)"""
    if isinstance(value, int):
        return value
    elif type(value) is float:
        if value.is_integer():
            return int(value)  # an int
        if (2*value).is_integer():
            return value  # a float
    elif isinstance(value, Rational):
        if value.q == 2:
            return value.p/value.q  # a float
        elif value.q == 1:
            return value.p  # an int
    elif isinstance(value, Float):
        return _int_or_halfint(float(value))
    raise ValueError("expecting integer or half-integer, got %s" % value)


def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    Notes
    =====

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \neq 0`

    - zero for violating any one of the conditions
         `m_1  \in \{-|j_1|, \ldots, |j_1|\}`,
         `m_2  \in \{-|j_2|, \ldots, |j_2|\}`,
         `m_3  \in \{-|j_3|, \ldots, |j_3|\}`

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """

    j_1, j_2, j_3, m_1, m_2, m_3 = \
        map(_int_or_halfint, map(sympify,
            [j_1, j_2, j_3, m_1, m_2, m_3]))

    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return S.Zero
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return S.Zero
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return S.Zero
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return S.Zero
    if not (int_valued(j_1 - m_1) and \
            int_valued(j_2 - m_2) and \
            int_valued(j_3 - m_3)):
        return S.Zero

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    _calc_factlist(int(maxfact))

    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]

    ressqrt = sqrt(argsqrt)
    if ressqrt.is_complex or ressqrt.is_infinite:
        ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den

    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    res = ressqrt * sumres * prefid
    return res


def clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculates the Clebsch-Gordan coefficient.
    `\left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle`.

    The reference for this function is [Edmonds74]_.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.physics.wigner import clebsch_gordan
    >>> clebsch_gordan(S(3)/2, S(1)/2, 2, S(3)/2, S(1)/2, 2)
    1
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, S(3)/2, -S(1)/2, 1)
    sqrt(3)/2
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, -S(1)/2, S(1)/2, 0)
    -sqrt(2)/2

    Notes
    =====

    The Clebsch-Gordan coefficient will be evaluated via its relation
    to Wigner 3j symbols:

    .. math::

        \left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle
        =(-1)^{j_1-j_2+m_3} \sqrt{2j_3+1}
        \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,-m_3)

    See also the documentation on Wigner 3j symbols which exhibit much
    higher symmetry relations than the Clebsch-Gordan coefficient.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    j_1 = sympify(j_1)
    j_2 = sympify(j_2)
    j_3 = sympify(j_3)
    m_1 = sympify(m_1)
    m_2 = sympify(m_2)
    m_3 = sympify(m_3)

    w = wigner_3j(j_1, j_2, j_3, m_1, m_2, -m_3)

    return (-1) ** (j_1 - j_2 + m_3) * sqrt(2 * j_3 + 1) * w


def _big_delta_coeff(aa, bb, cc, prec=None):
    r"""
    Calculates the Delta coefficient of the 3 angular momenta for
    Racah symbols. Also checks that the differences are of integer
    value.

    Parameters
    ==========

    aa :
        First angular momentum, integer or half integer.
    bb :
        Second angular momentum, integer or half integer.
    cc :
        Third angular momentum, integer or half integer.
    prec :
        Precision of the ``sqrt()`` calculation.

    Returns
    =======

    double : Value of the Delta coefficient.

    Examples
    ========

        sage: from sage.functions.wigner import _big_delta_coeff
        sage: _big_delta_coeff(1,1,1)
        1/2*sqrt(1/6)
    """

    # the triangle test will only pass if a) all 3 values are ints or
    # b) 1 is an int and the other two are half-ints
    if not int_valued(aa + bb - cc):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(aa + cc - bb):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(bb + cc - aa):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if (aa + bb - cc) < 0:
        return S.Zero
    if (aa + cc - bb) < 0:
        return S.Zero
    if (bb + cc - aa) < 0:
        return S.Zero

    maxfact = max(aa + bb - cc, aa + cc - bb, bb + cc - aa, aa + bb + cc + 1)
    _calc_factlist(maxfact)

    argsqrt = Integer(_Factlist[int(aa + bb - cc)] *
                     _Factlist[int(aa + cc - bb)] *
                     _Factlist[int(bb + cc - aa)]) / \
        Integer(_Factlist[int(aa + bb + cc + 1)])

    ressqrt = sqrt(argsqrt)
    if prec:
        ressqrt = ressqrt.evalf(prec).as_real_imag()[0]
    return ressqrt


def racah(aa, bb, cc, dd, ee, ff, prec=None):
    r"""
    Calculate the Racah symbol `W(a,b,c,d;e,f)`.

    Parameters
    ==========

    a, ..., f :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import racah
    >>> racah(3,3,3,3,3,3)
    -1/14

    Notes
    =====

    The Racah symbol is related to the Wigner 6j symbol:

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
       =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    Please see the 6j symbol for its much richer symmetries and for
    additional properties.

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    prefac = _big_delta_coeff(aa, bb, ee, prec) * \
        _big_delta_coeff(cc, dd, ee, prec) * \
        _big_delta_coeff(aa, cc, ff, prec) * \
        _big_delta_coeff(bb, dd, ff, prec)
    if prefac == 0:
        return S.Zero
    imin = max(aa + bb + ee, cc + dd + ee, aa + cc + ff, bb + dd + ff)
    imax = min(aa + bb + cc + dd, aa + dd + ee + ff, bb + cc + ee + ff)

    maxfact = max(imax + 1, aa + bb + cc + dd, aa + dd + ee + ff,
                 bb + cc + ee + ff)
    _calc_factlist(maxfact)

    sumres = 0
    for kk in range(int(imin), int(imax) + 1):
        den = _Factlist[int(kk - aa - bb - ee)] * \
            _Factlist[int(kk - cc - dd - ee)] * \
            _Factlist[int(kk - aa - cc - ff)] * \
            _Factlist[int(kk - bb - dd - ff)] * \
            _Factlist[int(aa + bb + cc + dd - kk)] * \
            _Factlist[int(aa + dd + ee + ff - kk)] * \
            _Factlist[int(bb + cc + ee + ff - kk)]
        sumres = sumres + Integer((-1) ** kk * _Factlist[kk + 1]) / den

    res = prefac * sumres * (-1) ** int(aa + bb + cc + dd)
    return res


def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    r"""
    Calculate the Wigner 6j symbol `\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)`.

    Parameters
    ==========

    j_1, ..., j_6 :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_6j
    >>> wigner_6j(3,3,3,3,3,3)
    -1/14
    >>> wigner_6j(5,5,5,5,5,5)
    1/52

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_6j(2.5,2.5,2.5,2.5,2.5,2.5)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_6j(0.5,0.5,1.1,0.5,0.5,1.1)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Notes
    =====

    The Wigner 6j symbol is related to the Racah symbol but exhibits
    more symmetries as detailed below.

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
        =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    The Wigner 6j symbol obeys the following symmetry rules:

    - Wigner 6j symbols are left invariant under any permutation of
      the columns:

      .. math::

         \begin{aligned}
         \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          &=\operatorname{Wigner6j}(j_3,j_1,j_2,j_6,j_4,j_5) \\
          &=\operatorname{Wigner6j}(j_2,j_3,j_1,j_5,j_6,j_4) \\
          &=\operatorname{Wigner6j}(j_3,j_2,j_1,j_6,j_5,j_4) \\
          &=\operatorname{Wigner6j}(j_1,j_3,j_2,j_4,j_6,j_5) \\
          &=\operatorname{Wigner6j}(j_2,j_1,j_3,j_5,j_4,j_6)
         \end{aligned}

    - They are invariant under the exchange of the upper and lower
      arguments in each of any two columns, i.e.

      .. math::

         \begin{aligned}
         \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          &=\operatorname{Wigner6j}(j_1,j_5,j_6,j_4,j_2,j_3)\\
          &=\operatorname{Wigner6j}(j_4,j_2,j_6,j_1,j_5,j_3)\\
          &=\operatorname{Wigner6j}(j_4,j_5,j_3,j_1,j_2,j_6)
         \end{aligned}

    - additional 6 symmetries [Regge59]_ giving rise to 144 symmetries
      in total

    - only non-zero if any triple of `j`'s fulfill a triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    """
    j_1, j_2, j_3, j_4, j_5, j_6 = map(sympify, \
                [j_1, j_2, j_3, j_4, j_5, j_6])
    res = (-1) ** int(j_1 + j_2 + j_4 + j_5) * \
        racah(j_1, j_2, j_5, j_4, j_3, j_6, prec)
    return res


def wigner_9j(j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, prec=None):
    r"""
    Calculate the Wigner 9j symbol
    `\operatorname{Wigner9j}(j_1,j_2,j_3,j_4,j_5,j_6,j_7,j_8,j_9)`.

    Parameters
    ==========

    j_1, ..., j_9 :
        Integer or half integer.
    prec : precision, default
        ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_9j
    >>> wigner_9j(1,1,1, 1,1,1, 1,1,0, prec=64)
    0.05555555555555555555555555555555555555555555555555555555555555555

    >>> wigner_9j(1/2,1/2,0, 1/2,3/2,1, 0,1,1, prec=64)
    0.1666666666666666666666666666666666666666666666666666666666666667

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_9j(0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_9j(1,1,1, 0.5,1,1.5, 0.5,1,2.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.
    """
    j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9 = map(sympify, \
                [j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9])
    imax = int(min(j_1 + j_9, j_2 + j_6, j_4 + j_8) * 2)
    imin = imax % 2
    sumres = 0
    for kk in range(imin, int(imax) + 1, 2):
        sumres = sumres + (kk + 1) * \
            racah(j_1, j_2, j_9, j_6, j_3, kk / 2, prec) * \
            racah(j_4, j_6, j_8, j_2, j_5, kk / 2, prec) * \
            racah(j_1, j_4, j_9, j_8, j_7, kk / 2, prec)
    return sumres


def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the Gaunt coefficient.

    Explanation
    ===========

    The Gaunt coefficient is defined as the integral over three
    spherical harmonics:

    .. math::

        \begin{aligned}
        \operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Y_{l_1,m_1}(\Omega)
         Y_{l_2,m_2}(\Omega) Y_{l_3,m_3}(\Omega) \,d\Omega \\
        &=\sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
         \operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)
         \operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integer.
    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import gaunt
    >>> gaunt(1,0,1,1,0,-1)
    -1/(2*sqrt(pi))
    >>> gaunt(1000,1000,1200,9,3,-12).n(64)
    0.006895004219221134484332976156744208248842039317638217822322799675

    It is an error to use non-integer values for `l` and `m`::

        sage: gaunt(1.2,0,1.2,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        sage: gaunt(1,0,1,1.1,0,-1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The Gaunt coefficient obeys the following symmetry rules:

    - invariant under any permutation of the columns

      .. math::
        \begin{aligned}
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          &=Y(l_3,l_1,l_2,m_3,m_1,m_2) \\
          &=Y(l_2,l_3,l_1,m_2,m_3,m_1) \\
          &=Y(l_3,l_2,l_1,m_3,m_2,m_1) \\
          &=Y(l_1,l_3,l_2,m_1,m_3,m_2) \\
          &=Y(l_2,l_1,l_3,m_2,m_1,m_3)
        \end{aligned}

    - invariant under space inflection, i.e.

      .. math::
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          =Y(l_1,l_2,l_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 Regge symmetries as inherited
      for the `3j` symbols [Regge58]_

    - zero for `l_1`, `l_2`, `l_3` not fulfilling triangle relation

    - zero for violating any one of the conditions: `l_1 \ge |m_1|`,
      `l_2 \ge |m_2|`, `l_3 \ge |m_3|`

    - non-zero only for an even sum of the `l_i`, i.e.
      `L = l_1 + l_2 + l_3 = 2n` for `n` in `\mathbb{N}`

    Algorithms
    ==========

    This function uses the algorithm of [Liberatodebrito82]_ to
    calculate the value of the Gaunt coefficient exactly. Note that
    the formula contains alternating sums over large factorials and is
    therefore unsuitable for finite precision arithmetic and only
    useful for a computer algebra system [Rasch03]_.

    Authors
    =======

    Jens Rasch (2009-03-24): initial version for Sage.
    """
    l_1, l_2, l_3, m_1, m_2, m_3 = [
        as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]

    if l_1 + l_2 - l_3 < 0:
        return S.Zero
    if l_1 - l_2 + l_3 < 0:
        return S.Zero
    if -l_1 + l_2 + l_3 < 0:
        return S.Zero
    if (m_1 + m_2 + m_3) != 0:
        return S.Zero
    if (abs(m_1) > l_1) or (abs(m_2) > l_2) or (abs(m_3) > l_3):
        return S.Zero
    bigL, remL = divmod(l_1 + l_2 + l_3, 2)
    if remL % 2:
        return S.Zero

    imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
    imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)

    _calc_factlist(max(l_1 + l_2 + l_3 + 1, imax + 1))

    ressqrt = sqrt((2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * \
        _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * \
        _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / \
        (4*pi))

    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] *
                     _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3])/ \
        _Factlist[2 * bigL + 1]/ \
        (_Factlist[bigL - l_1] *
         _Factlist[bigL - l_2] * _Factlist[bigL - l_3])

    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * \
            _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * \
            _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        sumres = sumres + Integer((-1) ** ii) / den

    res = ressqrt * prefac * sumres * Integer((-1) ** (bigL + l_3 + m_1 - m_2))
    if prec is not None:
        res = res.n(prec)
    return res


def real_gaunt(l_1, l_2, l_3, mu_1, mu_2, mu_3, prec=None):
    r"""
    Calculate the real Gaunt coefficient.

    Explanation
    ===========

    The real Gaunt coefficient is defined as the integral over three
    real spherical harmonics:

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,\mu_1,\mu_2,\mu_3)
        &=\int Z^{\mu_1}_{l_1}(\Omega)
         Z^{\mu_2}_{l_2}(\Omega) Z^{\mu_3}_{l_3}(\Omega) \,d\Omega \\
        \end{aligned}

    Alternatively, it can be defined in terms of the standard Gaunt
    coefficient by relating the real spherical harmonics to the standard
    spherical harmonics via a unitary transformation `U`, i.e.
    `Z^{\mu}_{l}(\Omega)=\sum_{m'}U^{\mu}_{m'}Y^{m'}_{l}(\Omega)` [Homeier96]_.
    The real Gaunt coefficient is then defined as

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,\mu_1,\mu_2,\mu_3)
        &=\int Z^{\mu_1}_{l_1}(\Omega)
         Z^{\mu_2}_{l_2}(\Omega) Z^{\mu_3}_{l_3}(\Omega) \,d\Omega \\
        &=\sum_{m'_1 m'_2 m'_3} U^{\mu_1}_{m'_1}U^{\mu_2}_{m'_2}U^{\mu_3}_{m'_3}
         \operatorname{Gaunt}(l_1,l_2,l_3,m'_1,m'_2,m'_3)
        \end{aligned}

    The unitary matrix `U` has components

    .. math::
        \begin{aligned}
        U^\mu_{m} = \delta_{|\mu||m|}*(\delta_{m0}\delta_{\mu 0} + \frac{1}{\sqrt{2}}\big[\Theta(\mu)\big(\delta_{m\mu}+(-1)^{m}\delta_{m-\mu}\big)
        +i \Theta(-\mu)\big((-1)^{m}\delta_{m\mu}-\delta_{m-\mu}\big)\big])
        \end{aligned}


    where `\delta_{ij}` is the Kronecker delta symbol and `\Theta` is a step
    function defined as

    .. math::
        \begin{aligned}
        \Theta(x) = \begin{cases} 1 \,\text{for}\, x > 0 \\ 0 \,\text{for}\, x \leq 0 \end{cases}
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, mu_1, mu_2, mu_3 :
        Integer degree and order

    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========
    >>> from sympy.physics.wigner import real_gaunt
    >>> real_gaunt(1,1,2,-1,1,-2)
    sqrt(15)/(10*sqrt(pi))
    >>> real_gaunt(10,10,20,-9,-9,0,prec=64)
    -0.00002480019791932209313156167176797577821140084216297395518482071448

    It is an error to use non-integer values for `l` and `\mu`::
        real_gaunt(2.8,0.5,1.3,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer

        real_gaunt(2,2,4,0.7,1,-3.4)
        Traceback (most recent call last):
        ...
        ValueError: mu values must be integer

    Notes
    =====

    The real Gaunt coefficient inherits from the standard Gaunt coefficient,
    the invariance under any permutation of the pairs `(l_i, \mu_i)` and the
    requirement that the sum of the `l_i` be even to yield a non-zero value.
    It also obeys the following symmetry rules:

    - zero for `l_1`, `l_2`, `l_3` not fulfilling the condition
      `l_1 \in \{l_{\text{max}}, l_{\text{max}}-2, \ldots, l_{\text{min}}\}`,
      where `l_{\text{max}} = l_2+l_3`,

      .. math::
          \begin{aligned}
          l_{\text{min}} = \begin{cases} \kappa(l_2, l_3, \mu_2, \mu_3) & \text{if}\,
          \kappa(l_2, l_3, \mu_2, \mu_3) + l_{\text{max}}\, \text{is even} \\
          \kappa(l_2, l_3, \mu_2, \mu_3)+1 & \text{if}\, \kappa(l_2, l_3, \mu_2, \mu_3) +
          l_{\text{max}}\, \text{is odd}\end{cases}
          \end{aligned}

      and `\kappa(l_2, l_3, \mu_2, \mu_3) = \max{\big(|l_2-l_3|, \min{\big(|\mu_2+\mu_3|,
      |\mu_2-\mu_3|\big)}\big)}`

    - zero for an odd number of negative `\mu_i`

    Algorithms
    ==========

    This function uses the algorithms of [Homeier96]_ and [Rasch03]_ to
    calculate the value of the real Gaunt coefficient exactly. Note that
    the formula used in [Rasch03]_ contains alternating sums over large
    factorials and is therefore unsuitable for finite precision arithmetic
    and only useful for a computer algebra system [Rasch03]_. However, this
    function can in principle use any algorithm that computes the Gaunt
    coefficient, so it is suitable for finite precision arithmetic in so far
    as the algorithm which computes the Gaunt coefficient is.
    """
    l_1, l_2, l_3, mu_1, mu_2, mu_3 = [
        as_int(i) for i in (l_1, l_2, l_3, mu_1, mu_2, mu_3)]

    # check for quick exits
    if sum(1 for i in (mu_1, mu_2, mu_3) if i < 0) % 2:
        return S.Zero  # odd number of negative m
    if (l_1 + l_2 + l_3) % 2:
        return S.Zero  # sum of l is odd
    lmax = l_2 + l_3
    lmin = max(abs(l_2 - l_3), min(abs(mu_2 + mu_3), abs(mu_2 - mu_3)))
    if (lmin + lmax) % 2:
        lmin += 1
    if lmin not in range(lmax, lmin - 2, -2):
        return S.Zero

    kron_del = lambda i, j: 1 if i == j else 0
    s = lambda e: -1 if e % 2 else 1  #  (-1)**e to give +/-1, avoiding float when e<0

    t = lambda x: 1 if x > 0 else 0
    A = lambda mu, m: t(-mu) * (s(m) * kron_del(m, mu) - kron_del(m, -mu))
    B = lambda mu, m: t(mu) * (kron_del(m, mu) + s(m) * kron_del(m, -mu))
    U = lambda mu, m: kron_del(abs(mu), abs(m)) * (kron_del(mu, 0) * kron_del(m, 0) + (B(mu, m) + I * A(mu, m))/sqrt(2))

    ugnt = 0
    for m1 in range(-l_1, l_1+1):
        U1 = U(mu_1, m1)
        for m2 in range(-l_2, l_2+1):
            U2 = U(mu_2, m2)
            U3 = U(mu_3,-m1-m2)
            ugnt = ugnt + re(U1*U2*U3)*gaunt(l_1, l_2, l_3, m1, m2, -m1 - m2, prec=prec)

    return ugnt


class Wigner3j(Function):

    def doit(self, **hints):
        if all(obj.is_number for obj in self.args):
            return wigner_3j(*self.args)
        else:
            return self

def dot_rot_grad_Ynm(j, p, l, m, theta, phi):
    r"""
    Returns dot product of rotational gradients of spherical harmonics.

    Explanation
    ===========

    This function returns the right hand side of the following expression:

    .. math ::
        \vec{R}Y{_j^{p}} \cdot \vec{R}Y{_l^{m}} = (-1)^{m+p}
        \sum\limits_{k=|l-j|}^{l+j}Y{_k^{m+p}}  * \alpha_{l,m,j,p,k} *
        \frac{1}{2} (k^2-j^2-l^2+k-j-l)


    Arguments
    =========

    j, p, l, m .... indices in spherical harmonics (expressions or integers)
    theta, phi .... angle arguments in spherical harmonics

    Example
    =======

    >>> from sympy import symbols
    >>> from sympy.physics.wigner import dot_rot_grad_Ynm
    >>> theta, phi = symbols("theta phi")
    >>> dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit()
    3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))

    """
    j = sympify(j)
    p = sympify(p)
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    phi = sympify(phi)
    k = Dummy("k")

    def alpha(l,m,j,p,k):
        return sqrt((2*l+1)*(2*j+1)*(2*k+1)/(4*pi)) * \
                Wigner3j(j, l, k, S.Zero, S.Zero, S.Zero) * \
                Wigner3j(j, l, k, p, m, -m-p)

    return (S.NegativeOne)**(m+p) * Sum(Ynm(k, m+p, theta, phi) * alpha(l,m,j,p,k) / 2 \
        *(k**2-j**2-l**2+k-j-l), (k, abs(l-j), l+j))


def wigner_d_small(J, beta):
    """Return the small Wigner d matrix for angular momentum J.

    Explanation
    ===========

    J : An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    beta : A real number representing the Euler angle of rotation about
        the so-called line of nodes. See [Edmonds74]_.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation( in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{d}_{\\beta} = \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)

    such that

    .. math ::
        d^{(J)}_{m',m}(\\beta) = \\mathtt{wigner\\_d\\_small(J,beta)[J-mprime,J-m]}

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.15.

    Examples
    ========

    >>> from sympy import Integer, symbols, pi, pprint
    >>> from sympy.physics.wigner import wigner_d_small
    >>> half = 1/Integer(2)
    >>> beta = symbols("beta", real=True)
    >>> pprint(wigner_d_small(half, beta), use_unicode=True)
    ⎡   ⎛β⎞      ⎛β⎞⎤
    ⎢cos⎜─⎟   sin⎜─⎟⎥
    ⎢   ⎝2⎠      ⎝2⎠⎥
    ⎢               ⎥
    ⎢    ⎛β⎞     ⎛β⎞⎥
    ⎢-sin⎜─⎟  cos⎜─⎟⎥
    ⎣    ⎝2⎠     ⎝2⎠⎦

    >>> pprint(wigner_d_small(2*half, beta), use_unicode=True)
    ⎡        2⎛β⎞              ⎛β⎞    ⎛β⎞           2⎛β⎞     ⎤
    ⎢     cos ⎜─⎟        √2⋅sin⎜─⎟⋅cos⎜─⎟        sin ⎜─⎟     ⎥
    ⎢         ⎝2⎠              ⎝2⎠    ⎝2⎠            ⎝2⎠     ⎥
    ⎢                                                        ⎥
    ⎢       ⎛β⎞    ⎛β⎞       2⎛β⎞      2⎛β⎞        ⎛β⎞    ⎛β⎞⎥
    ⎢-√2⋅sin⎜─⎟⋅cos⎜─⎟  - sin ⎜─⎟ + cos ⎜─⎟  √2⋅sin⎜─⎟⋅cos⎜─⎟⎥
    ⎢       ⎝2⎠    ⎝2⎠        ⎝2⎠       ⎝2⎠        ⎝2⎠    ⎝2⎠⎥
    ⎢                                                        ⎥
    ⎢        2⎛β⎞               ⎛β⎞    ⎛β⎞          2⎛β⎞     ⎥
    ⎢     sin ⎜─⎟        -√2⋅sin⎜─⎟⋅cos⎜─⎟       cos ⎜─⎟     ⎥
    ⎣         ⎝2⎠               ⎝2⎠    ⎝2⎠           ⎝2⎠     ⎦

    From table 4 in [Edmonds74]_

    >>> pprint(wigner_d_small(half, beta).subs({beta:pi/2}), use_unicode=True)
    ⎡ √2   √2⎤
    ⎢ ──   ──⎥
    ⎢ 2    2 ⎥
    ⎢        ⎥
    ⎢-√2   √2⎥
    ⎢────  ──⎥
    ⎣ 2    2 ⎦

    >>> pprint(wigner_d_small(2*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡       √2      ⎤
    ⎢1/2    ──   1/2⎥
    ⎢       2       ⎥
    ⎢               ⎥
    ⎢-√2         √2 ⎥
    ⎢────   0    ── ⎥
    ⎢ 2          2  ⎥
    ⎢               ⎥
    ⎢      -√2      ⎥
    ⎢1/2   ────  1/2⎥
    ⎣       2       ⎦

    >>> pprint(wigner_d_small(3*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡ √2    √6    √6   √2⎤
    ⎢ ──    ──    ──   ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢-√6   -√2    √2   √6⎥
    ⎢────  ────   ──   ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢ √6   -√2   -√2   √6⎥
    ⎢ ──   ────  ────  ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢-√2    √6   -√6   √2⎥
    ⎢────   ──   ────  ──⎥
    ⎣ 4     4     4    4 ⎦

    >>> pprint(wigner_d_small(4*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡             √6            ⎤
    ⎢1/4   1/2    ──   1/2   1/4⎥
    ⎢             4             ⎥
    ⎢                           ⎥
    ⎢-1/2  -1/2   0    1/2   1/2⎥
    ⎢                           ⎥
    ⎢ √6                     √6 ⎥
    ⎢ ──    0    -1/2   0    ── ⎥
    ⎢ 4                      4  ⎥
    ⎢                           ⎥
    ⎢-1/2  1/2    0    -1/2  1/2⎥
    ⎢                           ⎥
    ⎢             √6            ⎥
    ⎢1/4   -1/2   ──   -1/2  1/4⎥
    ⎣             4             ⎦

    """
    M = [J-i for i in range(2*J+1)]
    d = zeros(2*J+1)

    # Mi corresponds to Edmonds' $m'$, and Mj to $m$.
    for i, Mi in enumerate(M):
        for j, Mj in enumerate(M):

            # We get the maximum and minimum value of sigma.
            sigmamax = min([J-Mi, J-Mj])
            sigmamin = max([0, -Mi-Mj])

            dij = sqrt(factorial(J+Mi)*factorial(J-Mi) /
                       factorial(J+Mj)/factorial(J-Mj))
            terms = [(-1)**(J-Mi-s) *
                     binomial(J+Mj, J-Mi-s) *
                     binomial(J-Mj, s) *
                     cos(beta/2)**(2*s+Mi+Mj) *
                     sin(beta/2)**(2*J-2*s-Mj-Mi)
                     for s in range(sigmamin, sigmamax+1)]

            d[i, j] = dij*Add(*terms)

    return ImmutableMatrix(d)


def wigner_d(J, alpha, beta, gamma):
    """Return the Wigner D matrix for angular momentum J.

    Explanation
    ===========

    J :
        An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    alpha, beta, gamma - Real numbers representing the Euler.
        Angles of rotation about the so-called figure axis, line of nodes,
        and vertical. See [Edmonds74]_, however note that the symbols alpha
        and gamma are swapped in this implementation.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation (in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{D}_{\\alpha \\beta \\gamma} =
        \\exp\\big( \\frac{i\\alpha}{\\hbar} J_z\\big)
        \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)
        \\exp\\big( \\frac{i\\gamma}{\\hbar} J_z\\big)

    such that

    .. math ::
        \\mathcal{D}^{(J)}_{m',m}(\\alpha, \\beta, \\gamma) =
        \\mathtt{wigner_d(J, alpha, beta, gamma)[J-mprime,J-m]}

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.12, however note that the angles alpha and gamma are swapped
    in this implementation.

    Examples
    ========

    The simplest possible example:

    >>> from sympy.physics.wigner import wigner_d
    >>> from sympy import Integer, symbols, pprint
    >>> half = 1/Integer(2)
    >>> alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    >>> pprint(wigner_d(half, alpha, beta, gamma), use_unicode=True)
    ⎡  ⅈ⋅α  ⅈ⋅γ             ⅈ⋅α  -ⅈ⋅γ         ⎤
    ⎢  ───  ───             ───  ─────        ⎥
    ⎢   2    2     ⎛β⎞       2     2      ⎛β⎞ ⎥
    ⎢ ℯ   ⋅ℯ   ⋅cos⎜─⎟     ℯ   ⋅ℯ     ⋅sin⎜─⎟ ⎥
    ⎢              ⎝2⎠                    ⎝2⎠ ⎥
    ⎢                                         ⎥
    ⎢  -ⅈ⋅α   ⅈ⋅γ          -ⅈ⋅α   -ⅈ⋅γ        ⎥
    ⎢  ─────  ───          ─────  ─────       ⎥
    ⎢    2     2     ⎛β⎞     2      2      ⎛β⎞⎥
    ⎢-ℯ     ⋅ℯ   ⋅sin⎜─⎟  ℯ     ⋅ℯ     ⋅cos⎜─⎟⎥
    ⎣                ⎝2⎠                   ⎝2⎠⎦

    """
    d = wigner_d_small(J, beta)
    M = [J-i for i in range(2*J+1)]
    # Mi corresponds to Edmonds' $m'$, and Mj to $m$.
    D = [[exp(I*Mi*alpha)*d[i, j]*exp(I*Mj*gamma)
          for j, Mj in enumerate(M)] for i, Mi in enumerate(M)]
    return ImmutableMatrix(D)
<|endoftext|>from sympy.core import S, pi, Rational
from sympy.functions import assoc_laguerre, sqrt, exp, factorial, factorial2


def R_nl(n, l, nu, r):
    """
    Returns the radial wavefunction R_{nl} for a 3d isotropic harmonic
    oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.  Corresponds to the number of nodes in
        the wavefunction.  ``n >= 0``
    l :
        The quantum number for orbital angular momentum.
    nu :
        mass-scaled frequency: nu = m*omega/(2*hbar) where `m` is the mass
        and `omega` the frequency of the oscillator.
        (in atomic units ``nu == omega/2``)
    r :
        Radial coordinate.

    Examples
    ========

    >>> from sympy.physics.sho import R_nl
    >>> from sympy.abc import r, nu, l
    >>> R_nl(0, 0, 1, r)
    2*2**(3/4)*exp(-r**2)/pi**(1/4)
    >>> R_nl(1, 0, 1, r)
    4*2**(1/4)*sqrt(3)*(3/2 - 2*r**2)*exp(-r**2)/(3*pi**(1/4))

    l, nu and r may be symbolic:

    >>> R_nl(0, 0, nu, r)
    2*2**(3/4)*sqrt(nu**(3/2))*exp(-nu*r**2)/pi**(1/4)
    >>> R_nl(0, l, 1, r)
    r**l*sqrt(2**(l + 3/2)*2**(l + 2)/factorial2(2*l + 1))*exp(-r**2)/pi**(1/4)

    The normalization of the radial wavefunction is:

    >>> from sympy import Integral, oo
    >>> Integral(R_nl(0, 0, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(1, 0, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(1, 1, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000

    """
    n, l, nu, r = map(S, [n, l, nu, r])

    # formula uses n >= 1 (instead of nodal n >= 0)
    n = n + 1
    C = sqrt(
            ((2*nu)**(l + Rational(3, 2))*2**(n + l + 1)*factorial(n - 1))/
            (sqrt(pi)*(factorial2(2*n + 2*l - 1)))
    )
    return C*r**(l)*exp(-nu*r**2)*assoc_laguerre(n - 1, l + S.Half, 2*nu*r**2)


def E_nl(n, l, hw):
    """
    Returns the Energy of an isotropic harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    l :
        The orbital angular momentum.
    hw :
        The harmonic oscillator parameter.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_nl = (2*n + l + 3/2)*hw

    Examples
    ========

    >>> from sympy.physics.sho import E_nl
    >>> from sympy import symbols
    >>> x, y, z = symbols('x, y, z')
    >>> E_nl(x, y, z)
    z*(2*x + y + 3/2)
    """
    return (2*n + l + Rational(3, 2))*hw
<|endoftext|>"""
Second quantization operators and states for bosons.

This follow the formulation of Fetter and Welecka, "Quantum Theory
of Many-Particle Systems."
"""
from collections import defaultdict

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups

__all__ = [
    'Dagger',
    'KroneckerDelta',
    'BosonicOperator',
    'AnnihilateBoson',
    'CreateBoson',
    'AnnihilateFermion',
    'CreateFermion',
    'FockState',
    'FockStateBra',
    'FockStateKet',
    'FockStateBosonKet',
    'FockStateBosonBra',
    'FockStateFermionKet',
    'FockStateFermionBra',
    'BBra',
    'BKet',
    'FBra',
    'FKet',
    'F',
    'Fd',
    'B',
    'Bd',
    'apply_operators',
    'InnerProduct',
    'BosonicBasis',
    'VarBosonicBasis',
    'FixedBosonicBasis',
    'Commutator',
    'matrix_rep',
    'contraction',
    'wicks',
    'NO',
    'evaluate_deltas',
    'AntiSymmetricTensor',
    'substitute_dummies',
    'PermutationOperator',
    'simplify_index_permutations',
]


class SecondQuantizationError(Exception):
    pass


class AppliesOnlyToSymbolicIndex(SecondQuantizationError):
    pass


class ContractionAppliesOnlyToFermions(SecondQuantizationError):
    pass


class ViolationOfPauliPrinciple(SecondQuantizationError):
    pass


class SubstitutionOfAmbigousOperatorFailed(SecondQuantizationError):
    pass


