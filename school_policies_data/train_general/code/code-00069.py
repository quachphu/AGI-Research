    >>> import numpy as np
    >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
    array([ True, False, False, False, False,  True])

    """
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, _nx.complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return res[()]   # convert to scalar if needed


@array_function_dispatch(_is_type_dispatcher)
def isreal(x):
    """
    Returns a bool array, where True if input element is real.

    If element has complex type with zero imaginary part, the return value
    for that element is True.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray, bool
        Boolean array of same shape as `x`.

    Notes
    -----
    `isreal` may behave unexpectedly for string or object arrays (see examples)

    See Also
    --------
    iscomplex
    isrealobj : Return True if x is not a complex type.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j], dtype=complex)
    >>> np.isreal(a)
    array([False,  True,  True,  True,  True, False])

    The function does not work on string arrays.

    >>> a = np.array([2j, "a"], dtype="U")
    >>> np.isreal(a)  # Warns about non-elementwise comparison
    False

    Returns True for all elements in input array of ``dtype=object`` even if
    any of the elements is complex.

    >>> a = np.array([1, "2", 3+4j], dtype=object)
    >>> np.isreal(a)
    array([ True,  True,  True])

    isreal should not be used with object arrays

    >>> a = np.array([1+2j, 2+1j], dtype=object)
    >>> np.isreal(a)
    array([ True,  True])

    """
    return imag(x) == 0


@array_function_dispatch(_is_type_dispatcher)
def iscomplexobj(x):
    """
    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    isrealobj, iscomplex

    Examples
    --------
    >>> import numpy as np
    >>> np.iscomplexobj(1)
    False
    >>> np.iscomplexobj(1+0j)
    True
    >>> np.iscomplexobj([3, 1+0j, True])
    True

    """
    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = asarray(x).dtype.type
    return issubclass(type_, _nx.complexfloating)


@array_function_dispatch(_is_type_dispatcher)
def isrealobj(x):
    """
    Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, False if `x` is of a complex type.

    See Also
    --------
    iscomplexobj, isreal

    Notes
    -----
    The function is only meant for arrays with numerical values but it
    accepts all other objects. Since it assumes array input, the return
    value of other objects may be True.

    >>> np.isrealobj('A string')
    True
    >>> np.isrealobj(False)
    True
    >>> np.isrealobj(None)
    True

    Examples
    --------
    >>> import numpy as np
    >>> np.isrealobj(1)
    True
    >>> np.isrealobj(1+0j)
    False
    >>> np.isrealobj([3, 1+0j, True])
    False

    """
    return not iscomplexobj(x)

#-----------------------------------------------------------------------------

def _getmaxmin(t):
    from numpy._core import getlimits
    f = getlimits.finfo(t)
    return f.max, f.min


def _nan_to_num_dispatcher(x, copy=None, nan=None, posinf=None, neginf=None):
    return (x,)


@array_function_dispatch(_nan_to_num_dispatcher)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """
    Replace NaN with zero and infinity with large finite numbers (default
    behaviour) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    If `x` is inexact, NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    Parameters
    ----------
    x : scalar or array_like
        Input data.
    copy : bool, optional
        Whether to create a copy of `x` (True) or to replace values
        in-place (False). The in-place operation only occurs if
        casting to an array does not require a copy.
        Default is True.
    nan : int, float, optional
        Value to be used to fill NaN values. If no value is passed
        then NaN values will be replaced with 0.0.
    posinf : int, float, optional
        Value to be used to fill positive infinity values. If no value is
        passed then positive infinity values will be replaced with a very
        large number.
    neginf : int, float, optional
        Value to be used to fill negative infinity values. If no value is
        passed then negative infinity values will be replaced with a very
        small (or negative) number.

    Returns
    -------
    out : ndarray
        `x`, with the non-finite values replaced. If `copy` is False, this may
        be `x` itself.

    See Also
    --------
    isinf : Shows which elements are positive or negative infinity.
    isneginf : Shows which elements are negative infinity.
    isposinf : Shows which elements are positive infinity.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite : Shows which elements are finite (not NaN, not infinity)

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> import numpy as np
    >>> np.nan_to_num(np.inf)
    1.7976931348623157e+308
    >>> np.nan_to_num(-np.inf)
    -1.7976931348623157e+308
    >>> np.nan_to_num(np.nan)
    0.0
    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
    >>> np.nan_to_num(x)
    array([ 1.79769313e+308, -1.79769313e+308,  0.00000000e+000, # may vary
           -1.28000000e+002,  1.28000000e+002])
    >>> np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333)
    array([ 3.3333333e+07,  3.3333333e+07, -9.9990000e+03,
           -1.2800000e+02,  1.2800000e+02])
    >>> y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)])
    array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000, # may vary
         -1.28000000e+002,   1.28000000e+002])
    >>> np.nan_to_num(y)
    array([  1.79769313e+308 +0.00000000e+000j, # may vary
             0.00000000e+000 +0.00000000e+000j,
             0.00000000e+000 +1.79769313e+308j])
    >>> np.nan_to_num(y, nan=111111, posinf=222222)
    array([222222.+111111.j, 111111.     +0.j, 111111.+222222.j])
    """
    x = _nx.array(x, subok=True, copy=copy)
    xtype = x.dtype.type

    isscalar = (x.ndim == 0)

    if not issubclass(xtype, _nx.inexact):
        return x[()] if isscalar else x

    iscomplex = issubclass(xtype, _nx.complexfloating)

    dest = (x.real, x.imag) if iscomplex else (x,)
    maxf, minf = _getmaxmin(x.real.dtype)
    if posinf is not None:
        maxf = posinf
    if neginf is not None:
        minf = neginf
    for d in dest:
        idx_nan = isnan(d)
        idx_posinf = isposinf(d)
        idx_neginf = isneginf(d)
        _nx.copyto(d, nan, where=idx_nan)
        _nx.copyto(d, maxf, where=idx_posinf)
        _nx.copyto(d, minf, where=idx_neginf)
    return x[()] if isscalar else x

#-----------------------------------------------------------------------------

def _real_if_close_dispatcher(a, tol=None):
    return (a,)


@array_function_dispatch(_real_if_close_dispatcher)
def real_if_close(a, tol=100):
    """
    If input is complex with all imaginary parts close to zero, return
    real parts.

    "Close to zero" is defined as `tol` * (machine epsilon of the type for
    `a`).

    Parameters
    ----------
    a : array_like
        Input array.
    tol : float
        Tolerance in machine epsilons for the complex part of the elements
        in the array. If the tolerance is <=1, then the absolute tolerance
        is used.

    Returns
    -------
    out : ndarray
        If `a` is real, the type of `a` is used for the output.  If `a`
        has complex elements, the returned type is float.

    See Also
    --------
    real, imag, angle

    Notes
    -----
    Machine epsilon varies from machine to machine and between data types
    but Python floats on most platforms have a machine epsilon equal to
    2.2204460492503131e-16.  You can use 'np.finfo(float).eps' to print
    out the machine epsilon for floats.

    Examples
    --------
    >>> import numpy as np
    >>> np.finfo(float).eps
    2.2204460492503131e-16 # may vary

    >>> np.real_if_close([2.1 + 4e-14j, 5.2 + 3e-15j], tol=1000)
    array([2.1, 5.2])
    >>> np.real_if_close([2.1 + 4e-13j, 5.2 + 3e-15j], tol=1000)
    array([2.1+4.e-13j, 5.2 + 3e-15j])

    """
    a = asanyarray(a)
    type_ = a.dtype.type
    if not issubclass(type_, _nx.complexfloating):
        return a
    if tol > 1:
        f = getlimits.finfo(type_)
        tol = f.eps * tol
    if _nx.all(_nx.absolute(a.imag) < tol):
        a = a.real
    return a


#-----------------------------------------------------------------------------

_namefromtype = {'S1': 'character',
                 '?': 'bool',
                 'b': 'signed char',
                 'B': 'unsigned char',
                 'h': 'short',
                 'H': 'unsigned short',
                 'i': 'integer',
                 'I': 'unsigned integer',
                 'l': 'long integer',
                 'L': 'unsigned long integer',
                 'q': 'long long integer',
                 'Q': 'unsigned long long integer',
                 'f': 'single precision',
                 'd': 'double precision',
                 'g': 'long precision',
                 'F': 'complex single precision',
                 'D': 'complex double precision',
                 'G': 'complex long double precision',
                 'S': 'string',
                 'U': 'unicode',
                 'V': 'void',
                 'O': 'object'
                 }

@set_module('numpy')
def typename(char):
    """
    Return a description for the given data type code.

    Parameters
    ----------
    char : str
        Data type code.

    Returns
    -------
    out : str
        Description of the input data type code.

    See Also
    --------
    dtype

    Examples
    --------
    >>> import numpy as np
    >>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',
    ...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']
    >>> for typechar in typechars:
    ...     print(typechar, ' : ', np.typename(typechar))
    ...
    S1  :  character
    ?  :  bool
    B  :  unsigned char
    D  :  complex double precision
    G  :  complex long double precision
    F  :  complex single precision
    I  :  unsigned integer
    H  :  unsigned short
    L  :  unsigned long integer
    O  :  object
    Q  :  unsigned long long integer
    S  :  string
    U  :  unicode
    V  :  void
    b  :  signed char
    d  :  double precision
    g  :  long precision
    f  :  single precision
    i  :  integer
    h  :  short
    l  :  long integer
    q  :  long long integer

    """
    return _namefromtype[char]

#-----------------------------------------------------------------------------


#determine the "minimum common type" for a group of arrays.
array_type = [[_nx.float16, _nx.float32, _nx.float64, _nx.longdouble],
              [None, _nx.complex64, _nx.complex128, _nx.clongdouble]]
array_precision = {_nx.float16: 0,
                   _nx.float32: 1,
                   _nx.float64: 2,
                   _nx.longdouble: 3,
                   _nx.complex64: 1,
                   _nx.complex128: 2,
                   _nx.clongdouble: 3}


def _common_type_dispatcher(*arrays):
    return arrays


@array_function_dispatch(_common_type_dispatcher)
def common_type(*arrays):
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point) scalar
    type, even if all the arrays are integer arrays. If one of the inputs is
    an integer array, the minimum precision type that is returned is a
    64-bit floating point dtype.

    All input arrays except int64 and uint64 can be safely cast to the
    returned dtype without loss of information.

    Parameters
    ----------
    array1, array2, ... : ndarrays
        Input arrays.

    Returns
    -------
    out : data type code
        Data type code.

    See Also
    --------
    dtype, mintypecode

    Examples
    --------
    >>> np.common_type(np.arange(2, dtype=np.float32))
    <class 'numpy.float32'>
    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))
    <class 'numpy.float64'>
    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))
    <class 'numpy.complex128'>

    """
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]
<|endoftext|>import functools
import warnings

import numpy._core.numeric as _nx
from numpy._core.numeric import asarray, zeros, zeros_like, array, asanyarray
from numpy._core.fromnumeric import reshape, transpose
from numpy._core.multiarray import normalize_axis_index
from numpy._core._multiarray_umath import _array_converter
from numpy._core import overrides
from numpy._core import vstack, atleast_3d
from numpy._core.numeric import normalize_axis_tuple
from numpy._core.overrides import set_module
from numpy._core.shape_base import _arrays_for_stack_dispatcher
from numpy.lib._index_tricks_impl import ndindex
from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells


__all__ = [
    'column_stack', 'row_stack', 'dstack', 'array_split', 'split',
    'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims',
    'apply_along_axis', 'kron', 'tile', 'take_along_axis',
    'put_along_axis'
    ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis+1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def _take_along_axis_dispatcher(arr, indices, axis):
    return (arr, indices)


@array_function_dispatch(_take_along_axis_dispatcher)
def take_along_axis(arr, indices, axis):
    """
    Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Source array
    indices : ndarray (Ni..., J, Nk...)
        Indices to take along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions Ni and Nj only need to broadcast
        against `arr`.
    axis : int
        The axis to take 1d slices along. If axis is None, the input array is
        treated as if it had first been flattened to 1d, for consistency with
        `sort` and `argsort`.

    Returns
    -------
    out: ndarray (Ni..., J, Nk...)
        The indexed result.

    Notes
    -----
    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii`` and ``kk`` to a tuple of indices::

        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
        J = indices.shape[axis]  # Need not equal M
        out = np.empty(Ni + (J,) + Nk)

        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                a_1d       = a      [ii + s_[:,] + kk]
                indices_1d = indices[ii + s_[:,] + kk]
                out_1d     = out    [ii + s_[:,] + kk]
                for j in range(J):
                    out_1d[j] = a_1d[indices_1d[j]]

    Equivalently, eliminating the inner loop, the last two lines would be::

                out_1d[:] = a_1d[indices_1d]

    See Also
    --------
    take : Take along an axis, using the same indices for every 1d slice
    put_along_axis :
        Put values into the destination array by matching 1d index and data slices

    Examples
    --------
    >>> import numpy as np

    For this sample array

    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can sort either by using sort directly, or argsort and this function

    >>> np.sort(a, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> ai = np.argsort(a, axis=1)
    >>> ai
    array([[0, 2, 1],
           [1, 2, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])

    The same works for max and min, if you maintain the trivial dimension
    with ``keepdims``:

    >>> np.max(a, axis=1, keepdims=True)
    array([[30],
           [60]])
    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[30],
           [60]])

    If we want to get the max and min at the same time, we can stack the
    indices first

    >>> ai_min = np.argmin(a, axis=1, keepdims=True)
    >>> ai_max = np.argmax(a, axis=1, keepdims=True)
    >>> ai = np.concatenate([ai_min, ai_max], axis=1)
    >>> ai
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 30],
           [40, 60]])
    """
    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                'when axis=None, `indices` must have a single dimension.')
        arr = arr.flat
        arr_shape = (len(arr),)  # flatiter has no .shape
        axis = 0
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    return arr[_make_along_axis_idx(arr_shape, indices, axis)]


def _put_along_axis_dispatcher(arr, indices, values, axis):
    return (arr, indices, values)


@array_function_dispatch(_put_along_axis_dispatcher)
def put_along_axis(arr, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Destination array.
    indices : ndarray (Ni..., J, Nk...)
        Indices to change along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        against `arr`.
    values : array_like (Ni..., J, Nk...)
        values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is None, the destination
        array is treated as if a flattened 1d view had been created of it.

    Notes
    -----
    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii`` and ``kk`` to a tuple of indices::

        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
        J = indices.shape[axis]  # Need not equal M

        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                a_1d       = a      [ii + s_[:,] + kk]
                indices_1d = indices[ii + s_[:,] + kk]
                values_1d  = values [ii + s_[:,] + kk]
                for j in range(J):
                    a_1d[indices_1d[j]] = values_1d[j]

    Equivalently, eliminating the inner loop, the last two lines would be::

                a_1d[indices_1d] = values_1d

    See Also
    --------
    take_along_axis :
        Take values from the input array by matching 1d index and data slices

    Examples
    --------
    >>> import numpy as np

    For this sample array

    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can replace the maximum values with:

    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.put_along_axis(a, ai, 99, axis=1)
    >>> a
    array([[10, 99, 20],
           [99, 40, 50]])

    """
    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                'when axis=None, `indices` must have a single dimension.')
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter has no .shape
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    arr[_make_along_axis_idx(arr_shape, indices, axis)] = values


def _apply_along_axis_dispatcher(func1d, axis, arr, *args, **kwargs):
    return (arr,)


@array_function_dispatch(_apply_along_axis_dispatcher)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Apply a function to 1-D slices along the given axis.

    Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
    and `a` is a 1-D slice of `arr` along `axis`.

    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                f = func1d(arr[ii + s_[:,] + kk])
                Nj = f.shape
                for jj in ndindex(Nj):
                    out[ii + jj + kk] = f[jj]

    Equivalently, eliminating the inner loop, this can be expressed as::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])

    Parameters
    ----------
    func1d : function (M,) -> (Nj...)
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis.
    axis : integer
        Axis along which `arr` is sliced.
    arr : ndarray (Ni..., M, Nk...)
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`.

    Returns
    -------
    out : ndarray  (Ni..., Nj..., Nk...)
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value
        of `func1d`. So if `func1d` returns a scalar `out` will have one
        fewer dimensions than `arr`.

    See Also
    --------
    apply_over_axes : Apply a function repeatedly over multiple axes.

    Examples
    --------
    >>> import numpy as np
    >>> def my_func(a):
    ...     \"\"\"Average first and last element of a 1-D array\"\"\"
    ...     return (a[0] + a[-1]) * 0.5
    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> np.apply_along_axis(my_func, 0, b)
    array([4., 5., 6.])
    >>> np.apply_along_axis(my_func, 1, b)
    array([2.,  5.,  8.])

    For a function that returns a 1D array, the number of dimensions in
    `outarr` is the same as `arr`.

    >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])
    >>> np.apply_along_axis(sorted, 1, b)
    array([[1, 7, 8],
           [3, 4, 9],
           [2, 5, 6]])

    For a function that returns a higher dimensional array, those dimensions
    are inserted in place of the `axis` dimension.

    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> np.apply_along_axis(np.diag, -1, b)
    array([[[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]],
           [[4, 0, 0],
            [0, 5, 0],
            [0, 0, 6]],
           [[7, 0, 0],
            [0, 8, 0],
            [0, 0, 9]]])
    """
    # handle negative axes
    conv = _array_converter(arr)
    arr = conv[0]

    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_view = transpose(arr, in_dims[:axis] + in_dims[axis+1:] + [axis])

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            'Cannot apply_along_axis when any iteration dimensions are 0'
        ) from None
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    if not isinstance(res, matrix):
        buff = zeros_like(res, shape=inarr_view.shape[:-1] + res.shape)
    else:
        # Matrices are nasty with reshaping, so do not preserve them here.
        buff = zeros(inarr_view.shape[:-1] + res.shape, dtype=res.dtype)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0 : axis] +
        buff_dims[buff.ndim-res.ndim : buff.ndim] +
        buff_dims[axis : buff.ndim-res.ndim]
    )

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = asanyarray(func1d(inarr_view[ind], *args, **kwargs))

    res = transpose(buff, buff_permute)
    return conv.wrap(res)


def _apply_over_axes_dispatcher(func, a, axes):
    return (a,)


@array_function_dispatch(_apply_over_axes_dispatcher)
def apply_over_axes(func, a, axes):
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first
    element of `axes`.  The result `res` of the function call must have
    either the same dimensions as `a` or one less dimension.  If `res`
    has one less dimension than `a`, a dimension is inserted before
    `axis`.  The call to `func` is then repeated for each axis in `axes`,
    with `res` as the first argument.

    Parameters
    ----------
    func : function
        This function must take two arguments, `func(a, axis)`.
    a : array_like
        Input array.
    axes : array_like
        Axes over which `func` is applied; the elements must be integers.

    Returns
    -------
    apply_over_axis : ndarray
        The output array.  The number of dimensions is the same as `a`,
        but the shape can be different.  This depends on whether `func`
        changes the shape of its output with respect to its input.

    See Also
    --------
    apply_along_axis :
        Apply a function to 1-D slices of an array along the given axis.

    Notes
    -----
    This function is equivalent to tuple axis arguments to reorderable ufuncs
    with keepdims=True. Tuple axis arguments to ufuncs have been available since
    version 1.7.0.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(24).reshape(2,3,4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])

    Sum over axes 0 and 2. The result has same number of dimensions
    as the original array:

    >>> np.apply_over_axes(np.sum, a, [0,2])
    array([[[ 60],
            [ 92],
            [124]]])

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.sum(a, axis=(0,2), keepdims=True)
    array([[[ 60],
            [ 92],
            [124]]])

    """
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                                 "an array of the correct shape")
    return val


def _expand_dims_dispatcher(a, axis):
    return (a,)


@array_function_dispatch(_expand_dims_dispatcher)
def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or tuple of ints
        Position in the expanded axes where the new axis (or axes) is placed.

        .. deprecated:: 1.13.0
            Passing an axis where ``axis > a.ndim`` will be treated as
            ``axis == a.ndim``, and passing ``axis < -a.ndim - 1`` will
            be treated as ``axis == 0``. This behavior is deprecated.

    Returns
    -------
    result : ndarray
        View of `a` with the number of dimensions increased.

    See Also
    --------
    squeeze : The inverse operation, removing singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    atleast_1d, atleast_2d, atleast_3d

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[np.newaxis, :]`` or ``x[np.newaxis]``:

    >>> y = np.expand_dims(x, axis=0)
    >>> y
    array([[1, 2]])
    >>> y.shape
    (1, 2)

    The following is equivalent to ``x[:, np.newaxis]``:

    >>> y = np.expand_dims(x, axis=1)
    >>> y
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)

    ``axis`` may also be a tuple:

    >>> y = np.expand_dims(x, axis=(0, 1))
    >>> y
    array([[[1, 2]]])

    >>> y = np.expand_dims(x, axis=(2, 0))
    >>> y
    array([[[1],
            [2]]])

    Note that some examples may use ``None`` instead of ``np.newaxis``.  These
    are the same objects:

    >>> np.newaxis is None
    True

    """
    if isinstance(a, matrix):
        a = asarray(a)
    else:
        a = asanyarray(a)

    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return a.reshape(shape)


# NOTE: Remove once deprecation period passes
@set_module("numpy")
def row_stack(tup, *, dtype=None, casting="same_kind"):
    # Deprecated in NumPy 2.0, 2023-08-18
    warnings.warn(
        "`row_stack` alias is deprecated. "
        "Use `np.vstack` directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return vstack(tup, dtype=dtype, casting=casting)


row_stack.__doc__ = vstack.__doc__


def _column_stack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)


@array_function_dispatch(_column_stack_dispatcher)
def column_stack(tup):
    """
    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    stacked : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.column_stack((a,b))
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    arrays = []
    for v in tup:
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=None, subok=True, ndmin=2).T
        arrays.append(arr)
    return _nx.concatenate(arrays, 1)


def _dstack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)


@array_function_dispatch(_dstack_dispatcher)
def dstack(tup):
    """
    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of arrays
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    dsplit : Split array along third axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.dstack((a,b))
    array([[[1, 2],
            [2, 3],
            [3, 4]]])

    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[2],[3],[4]])
    >>> np.dstack((a,b))
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])

    """
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return _nx.concatenate(arrs, 2)


def _replace_zero_by_x_arrays(sub_arys):
    for i in range(len(sub_arys)):
        if _nx.ndim(sub_arys[i]) == 0:
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
        elif _nx.sometrue(_nx.equal(_nx.shape(sub_arys[i]), 0)):
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
    return sub_arys


def _array_split_dispatcher(ary, indices_or_sections, axis=None):
    return (ary, indices_or_sections)


@array_function_dispatch(_array_split_dispatcher)
def array_split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays.

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis. For an array of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.

    See Also
    --------
    split : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]

    >>> x = np.arange(9)
    >>> np.array_split(x, 4)
    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]

    """
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section+1] +
                         (Nsections-extras) * [Neach_section])
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    sary = _nx.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))

    return sub_arys


def _split_dispatcher(ary, indices_or_sections, axis=None):
    return (ary, indices_or_sections)


@array_function_dispatch(_split_dispatcher)
def split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays as views into `ary`.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays as views into `ary`.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    See Also
    --------
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.  Does not raise an exception if
                  an equal division cannot be made.
    hsplit : Split array into multiple sub-arrays horizontally (column-wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(9.0)
    >>> np.split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]

    >>> x = np.arange(8.0)
    >>> np.split(x, [3, 5, 6, 10])
    [array([0.,  1.,  2.]),
     array([3.,  4.]),
     array([5.]),
     array([6.,  7.]),
     array([], dtype=float64)]

    """
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = ary.shape[axis]
        if N % sections:
            raise ValueError(
                'array split does not result in an equal division') from None
    return array_split(ary, indices_or_sections, axis)


def _hvdsplit_dispatcher(ary, indices_or_sections):
    return (ary, indices_or_sections)


@array_function_dispatch(_hvdsplit_dispatcher)
def hsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays horizontally (column-wise).

    Please refer to the `split` documentation.  `hsplit` is equivalent
    to `split` with ``axis=1``, the array is always split along the second
    axis except for 1-D arrays, where it is split at ``axis=0``.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])
    >>> np.hsplit(x, 2)
    [array([[  0.,   1.],
           [  4.,   5.],
           [  8.,   9.],
           [12.,  13.]]),
     array([[  2.,   3.],
           [  6.,   7.],
           [10.,  11.],
           [14.,  15.]])]
    >>> np.hsplit(x, np.array([3, 6]))
    [array([[ 0.,   1.,   2.],
           [ 4.,   5.,   6.],
           [ 8.,   9.,  10.],
           [12.,  13.,  14.]]),
     array([[ 3.],
           [ 7.],
           [11.],
           [15.]]),
     array([], shape=(4, 0), dtype=float64)]

    With a higher dimensional array the split is still along the second axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0.,  1.],
            [2.,  3.]],
           [[4.,  5.],
            [6.,  7.]]])
    >>> np.hsplit(x, 2)
    [array([[[0.,  1.]],
           [[4.,  5.]]]),
     array([[[2.,  3.]],
           [[6.,  7.]]])]

    With a 1-D array, the split is along axis 0.

    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> np.hsplit(x, 2)
    [array([0, 1, 2]), array([3, 4, 5])]

    """
    if _nx.ndim(ary) == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        return split(ary, indices_or_sections, 0)


@array_function_dispatch(_hvdsplit_dispatcher)
def vsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays vertically (row-wise).

    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
    to ``split`` with `axis=0` (default), the array is always split along the
    first axis regardless of the array dimension.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])
    >>> np.vsplit(x, 2)
    [array([[0., 1., 2., 3.],
            [4., 5., 6., 7.]]),
     array([[ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])]
    >>> np.vsplit(x, np.array([3, 6]))
    [array([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([[12., 13., 14., 15.]]),
     array([], shape=(0, 4), dtype=float64)]

    With a higher dimensional array the split is still along the first axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0.,  1.],
            [2.,  3.]],
           [[4.,  5.],
            [6.,  7.]]])
    >>> np.vsplit(x, 2)
    [array([[[0., 1.],
             [2., 3.]]]),
     array([[[4., 5.],
             [6., 7.]]])]

    """
    if _nx.ndim(ary) < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)


@array_function_dispatch(_hvdsplit_dispatcher)
def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16.0).reshape(2, 2, 4)
    >>> x
    array([[[ 0.,   1.,   2.,   3.],
            [ 4.,   5.,   6.,   7.]],
           [[ 8.,   9.,  10.,  11.],
            [12.,  13.,  14.,  15.]]])
    >>> np.dsplit(x, 2)
    [array([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]), array([[[ 2.,  3.],
            [ 6.,  7.]],
           [[10., 11.],
            [14., 15.]]])]
    >>> np.dsplit(x, np.array([3, 6]))
    [array([[[ 0.,   1.,   2.],
            [ 4.,   5.,   6.]],
           [[ 8.,   9.,  10.],
            [12.,  13.,  14.]]]),
     array([[[ 3.],
            [ 7.]],
           [[11.],
            [15.]]]),
    array([], shape=(2, 2, 0), dtype=float64)]
    """
    if _nx.ndim(ary) < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    return split(ary, indices_or_sections, 2)


def get_array_wrap(*args):
    """Find the wrapper for the array with the highest priority.

    In case of ties, leftmost wins. If no wrapper is found, return None.

    .. deprecated:: 2.0
    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`get_array_wrap` is deprecated. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    wrappers = sorted((getattr(x, '__array_priority__', 0), -i,
                 x.__array_wrap__) for i, x in enumerate(args)
                                   if hasattr(x, '__array_wrap__'))
    if wrappers:
        return wrappers[-1][-1]
    return None


def _kron_dispatcher(a, b):
    return (a, b)


@array_function_dispatch(_kron_dispatcher)
def kron(a, b):
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Parameters
    ----------
    a, b : array_like

    Returns
    -------
    out : ndarray

    See Also
    --------
    outer : The outer product

    Notes
    -----
    The function assumes that the number of dimensions of `a` and `b`
    are the same, if necessary prepending the smallest with ones.
    If ``a.shape = (r0,r1,..,rN)`` and ``b.shape = (s0,s1,...,sN)``,
    the Kronecker product has shape ``(r0*s0, r1*s1, ..., rN*SN)``.
    The elements are products of elements from `a` and `b`, organized
    explicitly by::

        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

    where::

        kt = it * st + jt,  t = 0,...,N

    In the common 2-D case (N=1), the block structure can be visualized::

        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
         [  ...                              ...   ],
         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]


    Examples
    --------
    >>> import numpy as np
    >>> np.kron([1,10,100], [5,6,7])
    array([  5,   6,   7, ..., 500, 600, 700])
    >>> np.kron([5,6,7], [1,10,100])
    array([  5,  50, 500, ...,   7,  70, 700])

    >>> np.kron(np.eye(2), np.ones((2,2)))
    array([[1.,  1.,  0.,  0.],
           [1.,  1.,  0.,  0.],
           [0.,  0.,  1.,  1.],
           [0.,  0.,  1.,  1.]])

    >>> a = np.arange(100).reshape((2,5,2,5))
    >>> b = np.arange(24).reshape((2,3,4))
    >>> c = np.kron(a,b)
    >>> c.shape
    (2, 10, 6, 20)
    >>> I = (1,3,0,2)
    >>> J = (0,2,1)
    >>> J1 = (0,) + J             # extend to ndim=4
    >>> S1 = (1,) + b.shape
    >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
    >>> c[K] == a[I]*b[J]
    True

    """
    # Working:
    # 1. Equalise the shapes by prepending smaller array with 1s
    # 2. Expand shapes of both the arrays by adding new axes at
    #    odd positions for 1st array and even positions for 2nd
    # 3. Compute the product of the modified array
    # 4. The inner most array elements now contain the rows of
    #    the Kronecker product
    # 5. Reshape the result to kron's shape, which is same as
    #    product of shapes of the two arrays.
    b = asanyarray(b)
    a = array(a, copy=None, subok=True, ndmin=b.ndim)
    is_any_mat = isinstance(a, matrix) or isinstance(b, matrix)
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)

    if (nda == 0 or ndb == 0):
        return _nx.multiply(a, b)

    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = reshape(a, as_)
    if not b.flags.contiguous:
        b = reshape(b, bs)

    # Equalise the shapes by prepending smaller one with 1s
    as_ = (1,)*max(0, ndb-nda) + as_
    bs = (1,)*max(0, nda-ndb) + bs

    # Insert empty dimensions
    a_arr = expand_dims(a, axis=tuple(range(ndb-nda)))
    b_arr = expand_dims(b, axis=tuple(range(nda-ndb)))

    # Compute the product
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd*2, 2)))
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd*2, 2)))
    # In case of `mat`, convert result to `array`
    result = _nx.multiply(a_arr, b_arr, subok=(not is_any_mat))

    # Reshape back
    result = result.reshape(_nx.multiply(as_, bs))

    return result if not is_any_mat else matrix(result, copy=False)


def _tile_dispatcher(A, reps):
    return (A, reps)


@array_function_dispatch(_tile_dispatcher)
def tile(A, reps):
    """
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by prepending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    A : array_like
        The input array.
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    See Also
    --------
    repeat : Repeat elements of an array.
    broadcast_to : Broadcast an array to a new shape

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    """
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    if all(x == 1 for x in tup) and isinstance(A, _nx.ndarray):
        # Fixes the problem that the function does not make a copy if A is a
        # numpy array and the repetitions are 1 in all dimensions
        return _nx.array(A, copy=True, subok=True, ndmin=d)
    else:
        # Note that no copy of zero-sized arrays is made. However since they
        # have no data there is no risk of an inadvertent overwrite.
        c = _nx.array(A, copy=None, subok=True, ndmin=d)
    if (d < c.ndim):
        tup = (1,)*(c.ndim-d) + tup
    shape_out = tuple(s*t for s, t in zip(c.shape, tup))
    n = c.size
    if n > 0:
        for dim_in, nrep in zip(c.shape, tup):
            if nrep != 1:
                c = c.reshape(-1, n).repeat(nrep, 0)
            n //= dim_in
    return c.reshape(shape_out)
<|endoftext|>"""
IO related functions.
"""
import os
import re
import functools
import itertools
import warnings
import weakref
import contextlib
import operator
from operator import itemgetter
from collections.abc import Mapping
import pickle

import numpy as np
from . import format
from ._datasource import DataSource
from numpy._core import overrides
from numpy._core.multiarray import packbits, unpackbits
from numpy._core._multiarray_umath import _load_from_filelike
from numpy._core.overrides import finalize_array_function_like, set_module
from ._iotools import (
    LineSplitter, NameValidator, StringConverter, ConverterError,
    ConverterLockError, ConversionWarning, _is_string_like,
    has_nested_fields, flatten_dtype, easy_dtype, _decode_line
    )
from numpy._utils import asunicode, asbytes


__all__ = [
    'savetxt', 'loadtxt', 'genfromtxt', 'load', 'save', 'savez',
    'savez_compressed', 'packbits', 'unpackbits', 'fromregex'
    ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


class BagObj:
    """
    BagObj(obj)

    Convert attribute look-ups to getitems on the object passed in.

    Parameters
    ----------
    obj : class instance
        Object on which attribute look-up is performed.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib._npyio_impl import BagObj as BO
    >>> class BagDemo:
    ...     def __getitem__(self, key): # An instance of BagObj(BagDemo)
    ...                                 # will call this method when any
    ...                                 # attribute look-up is required
    ...         result = "Doesn't matter what you want, "
    ...         return result + "you're gonna get this"
    ...
    >>> demo_obj = BagDemo()
    >>> bagobj = BO(demo_obj)
    >>> bagobj.hello_there
    "Doesn't matter what you want, you're gonna get this"
    >>> bagobj.I_can_be_anything
    "Doesn't matter what you want, you're gonna get this"

    """

    def __init__(self, obj):
        # Use weakref to make NpzFile objects collectable by refcount
        self._obj = weakref.proxy(obj)

    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        """
        Enables dir(bagobj) to list the files in an NpzFile.

        This also enables tab-completion in an interpreter or IPython.
        """
        return list(object.__getattribute__(self, '_obj').keys())


def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    if not hasattr(file, 'read'):
        file = os.fspath(file)
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, *args, **kwargs)


@set_module('numpy.lib.npyio')
class NpzFile(Mapping):
    """
    NpzFile(fid)

    A dictionary-like object with lazy-loading of files in the zipped
    archive provided on construction.

    `NpzFile` is used to load files in the NumPy ``.npz`` data archive
    format. It assumes that files in the archive have a ``.npy`` extension,
    other files are ignored.

    The arrays and file strings are lazily loaded on either
    getitem access using ``obj['key']`` or attribute lookup using
    ``obj.f.key``. A list of all files (without ``.npy`` extensions) can
    be obtained with ``obj.files`` and the ZipFile object itself using
    ``obj.zip``.

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ``.npy`` extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    f : BagObj instance
        An object on which attribute can be performed as an alternative
        to getitem access on the `NpzFile` instance itself.
    allow_pickle : bool, optional
        Allow loading pickled data. Default: False
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass on to pickle.load.
        These are only useful when loading object arrays saved on
        Python 2 when using Python 3.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Parameters
    ----------
    fid : file, str, or pathlib.Path
        The zipped archive to open. This is either a file-like object
        or a string containing the path to the archive.
    own_fid : bool, optional
        Whether NpzFile should close the file handle.
        Requires that `fid` is a file-like object.

    Examples
    --------
    >>> import numpy as np
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)

    >>> npz = np.load(outfile)
    >>> isinstance(npz, np.lib.npyio.NpzFile)
    True
    >>> npz
    NpzFile 'object' with keys: x, y
    >>> sorted(npz.files)
    ['x', 'y']
    >>> npz['x']  # getitem access
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> npz.f.x  # attribute lookup
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    # Make __exit__ safe if zipfile_factory raises an exception
    zip = None
    fid = None
    _MAX_REPR_ARRAY_COUNT = 5

    def __init__(self, fid, own_fid=False, allow_pickle=False,
                 pickle_kwargs=None, *,
                 max_header_size=format._MAX_HEADER_SIZE):
        # Import is postponed to here since zipfile depends on gzip, an
        # optional component of the so-called standard library.
        _zip = zipfile_factory(fid)
        self._files = _zip.namelist()
        self.files = []
        self.allow_pickle = allow_pickle
        self.max_header_size = max_header_size
        self.pickle_kwargs = pickle_kwargs
        for x in self._files:
            if x.endswith('.npy'):
                self.files.append(x[:-4])
            else:
                self.files.append(x)
        self.zip = _zip
        self.f = BagObj(self)
        if own_fid:
            self.fid = fid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close the file.

        """
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        self.f = None  # break reference cycle

    def __del__(self):
        self.close()

    # Implement the Mapping ABC
    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        # FIXME: This seems like it will copy strings around
        #   more than is strictly necessary.  The zipfile
        #   will read the string and then
        #   the format.read_array will copy the string
        #   to another place in memory.
        #   It would be better if the zipfile could read
        #   (or at least uncompress) the data
        #   directly into the array memory.
        member = False
        if key in self._files:
            member = True
        elif key in self.files:
            member = True
            key += '.npy'
        if member:
            bytes = self.zip.open(key)
            magic = bytes.read(len(format.MAGIC_PREFIX))
            bytes.close()
            if magic == format.MAGIC_PREFIX:
                bytes = self.zip.open(key)
                return format.read_array(bytes,
                                         allow_pickle=self.allow_pickle,
                                         pickle_kwargs=self.pickle_kwargs,
                                         max_header_size=self.max_header_size)
            else:
                return self.zip.read(key)
        else:
            raise KeyError(f"{key} is not a file in the archive")

    def __contains__(self, key):
        return (key in self._files or key in self.files)

    def __repr__(self):
        # Get filename or default to `object`
        if isinstance(self.fid, str):
            filename = self.fid
        else:
            filename = getattr(self.fid, "name", "object")

        # Get the name of arrays
        array_names = ', '.join(self.files[:self._MAX_REPR_ARRAY_COUNT])
        if len(self.files) > self._MAX_REPR_ARRAY_COUNT:
            array_names += "..."
        return f"NpzFile {filename!r} with keys: {array_names}"

    # Work around problems with the docstrings in the Mapping methods
    # They contain a `->`, which confuses the type annotation interpretations
    # of sphinx-docs. See gh-25964

    def get(self, key, default=None, /):
        """
        D.get(k,[,d]) returns D[k] if k in D, else d.  d defaults to None.
        """
        return Mapping.get(self, key, default)

    def items(self):
        """
        D.items() returns a set-like object providing a view on the items
        """
        return Mapping.items(self)

    def keys(self):
        """
        D.keys() returns a set-like object providing a view on the keys
        """
        return Mapping.keys(self)

    def values(self):
        """
        D.values() returns a set-like object providing a view on the values
        """
        return Mapping.values(self)


@set_module('numpy')
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding='ASCII', *, max_header_size=format._MAX_HEADER_SIZE):
    """
    Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    .. warning:: Loading files that contain object arrays uses the ``pickle``
                 module, which is not secure against erroneous or maliciously
                 constructed data. Consider passing ``allow_pickle=False`` to
                 load data that is known not to contain object arrays for the
                 safer handling of untrusted sources.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        ``seek()`` and ``read()`` methods and must always
        be opened in binary mode.  Pickled files require that the
        file-like object support the ``readline()`` method as well.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode (see
        `numpy.memmap` for a detailed description of the modes).  A
        memory-mapped array is kept on disk. However, it can be accessed
        and sliced like any ndarray.  Memory mapping is especially useful
        for accessing small fragments of large files without reading the
        entire file into memory.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons for
        disallowing pickles include security, as loading pickled data can
        execute arbitrary code. If pickles are disallowed, loading object
        arrays will fail. Default: False
    fix_imports : bool, optional
        Only useful when loading Python 2 generated pickled files on Python 3,
        which includes npy/npz files containing object arrays. If `fix_imports`
        is True, pickle will try to map the old Python 2 names to the new names
        used in Python 3.
    encoding : str, optional
        What encoding to use when reading Python 2 strings. Only useful when
        loading Python 2 generated pickled files in Python 3, which includes
        npy/npz files containing object arrays. Values other than 'latin1',
        'ASCII', and 'bytes' are not allowed, as they can corrupt numerical
        data. Default: 'ASCII'
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance
        of NpzFile class must be closed to avoid leaking file descriptors.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.
    UnpicklingError
        If ``allow_pickle=True``, but the file cannot be loaded as a pickle.
    ValueError
        The file contains an object array, but ``allow_pickle=False`` given.
    EOFError
        When calling ``np.load`` multiple times on the same file handle,
        if all data has already been read

    See Also
    --------
    save, savez, savez_compressed, loadtxt
    memmap : Create a memory-map to an array stored in a file on disk.
    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

    Notes
    -----
    - If the file contains pickle data, then whatever object is stored
      in the pickle is returned.
    - If the file is a ``.npy`` file, then a single array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is
      returned, containing ``{filename: array}`` key-value pairs, one for
      each file in the archive.
    - If the file is a ``.npz`` file, the returned value supports the
      context manager protocol in a similar fashion to the open function::

        with load('foo.npz') as data:
            a = data['a']

      The underlying file descriptor is closed when exiting the 'with'
      block.

    Examples
    --------
    >>> import numpy as np

    Store data to disk, and load it again:

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> np.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Store compressed data to disk, and load it again:

    >>> a=np.array([[1, 2, 3], [4, 5, 6]])
    >>> b=np.array([1, 2])
    >>> np.savez('/tmp/123.npz', a=a, b=b)
    >>> data = np.load('/tmp/123.npz')
    >>> data['a']
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> data['b']
    array([1, 2])
    >>> data.close()

    Mem-map the stored array, and then access the second row
    directly from disk:

    >>> X = np.load('/tmp/123.npy', mmap_mode='r')
    >>> X[1, :]
    memmap([4, 5, 6])

    """
    if encoding not in ('ASCII', 'latin1', 'bytes'):
        # The 'encoding' value for pickle also affects what encoding
        # the serialized binary data of NumPy arrays is loaded
        # in. Pickle does not pass on the encoding information to
        # NumPy. The unpickling code in numpy._core.multiarray is
        # written to assume that unicode data appearing where binary
        # should be is in 'latin1'. 'bytes' is also safe, as is 'ASCII'.
        #
        # Other encoding values can corrupt binary data, and we
        # purposefully disallow them. For the same reason, the errors=
        # argument is not exposed, as values other than 'strict'
        # result can similarly silently corrupt numerical data.
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")

    pickle_kwargs = dict(encoding=encoding, fix_imports=fix_imports)

    with contextlib.ExitStack() as stack:
        if hasattr(file, 'read'):
            fid = file
            own_fid = False
        else:
            fid = stack.enter_context(open(os.fspath(file), "rb"))
            own_fid = True

        # Code to distinguish from NumPy binary files and pickles.
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06'  # empty zip files start with this
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        if not magic:
            raise EOFError("No data left in file")
        # If the file size is less than N, we need to make sure not
        # to seek past the beginning of the file
        fid.seek(-min(N, len(magic)), 1)  # back-up
        if magic.startswith((_ZIP_PREFIX, _ZIP_SUFFIX)):
            # zip-file (assume .npz)
            # Potentially transfer file ownership to NpzFile
            stack.pop_all()
            ret = NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle,
                          pickle_kwargs=pickle_kwargs,
                          max_header_size=max_header_size)
            return ret
        elif magic == format.MAGIC_PREFIX:
            # .npy file
            if mmap_mode:
                if allow_pickle:
                    max_header_size = 2**64
                return format.open_memmap(file, mode=mmap_mode,
                                          max_header_size=max_header_size)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle,
                                         pickle_kwargs=pickle_kwargs,
                                         max_header_size=max_header_size)
        else:
            # Try a pickle
            if not allow_pickle:
                raise ValueError(
                    "This file contains pickled (object) data. If you trust "
                    "the file you can load it unsafely using the "
                    "`allow_pickle=` keyword argument or `pickle.load()`.")
            try:
                return pickle.load(fid, **pickle_kwargs)
            except Exception as e:
                raise pickle.UnpicklingError(
                    f"Failed to interpret file {file!r} as a pickle") from e


def _save_dispatcher(file, arr, allow_pickle=None, fix_imports=None):
    return (arr,)


@array_function_dispatch(_save_dispatcher)
def save(file, arr, allow_pickle=True, fix_imports=np._NoValue):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        File or filename to which the data is saved. If file is a file-object,
        then the filename is unchanged.  If file is a string or Path,
        a ``.npy`` extension will be appended to the filename if it does not
        already have one.
    arr : array_like
        Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    fix_imports : bool, optional
        The `fix_imports` flag is deprecated and has no effect.

        .. deprecated:: 2.1
            This flag is ignored since NumPy 1.17 and was only needed to
            support loading some files in Python 2 written in Python 3.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` archive
    savetxt, load

    Notes
    -----
    For a description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    Any data saved to the file is appended to the end of the file.

    Examples
    --------
    >>> import numpy as np

    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


    >>> with open('test.npy', 'wb') as f:
    ...     np.save(f, np.array([1, 2]))
    ...     np.save(f, np.array([1, 3]))
    >>> with open('test.npy', 'rb') as f:
    ...     a = np.load(f)
    ...     b = np.load(f)
    >>> print(a, b)
    # [1 2] [1 3]
    """
    if fix_imports is not np._NoValue:
        # Deprecated 2024-05-16, NumPy 2.1
        warnings.warn(
            "The 'fix_imports' flag is deprecated and has no effect. "
            "(Deprecated in NumPy 2.1)",
            DeprecationWarning, stacklevel=2)
    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        file_ctx = open(file, "wb")

    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle,
                           pickle_kwargs=dict(fix_imports=fix_imports))


def _savez_dispatcher(file, *args, allow_pickle=True, **kwds):
    yield from args
    yield from kwds.values()


@array_function_dispatch(_savez_dispatcher)
def savez(file, *args, allow_pickle=True, **kwds):
    """Save several arrays into a single file in uncompressed ``.npz`` format.

    Provide arrays as keyword arguments to store them under the
    corresponding name in the output file: ``savez(fn, x=x, y=y)``.

    If arrays are specified as positional arguments, i.e., ``savez(fn,
    x, y)``, their names will be `arr_0`, `arr_1`, etc.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    args : Arguments, optional
        Arrays to save to the file. Please use keyword arguments (see
        `kwds` below) to assign names to arrays.  Arrays specified as
        args will be named "arr_0", "arr_1", and so on.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    kwds : Keyword arguments, optional
        Arrays to save to the file. Each array will be saved to the
        output file with its corresponding keyword name.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.
    savetxt : Save an array to a file as plain text.
    savez_compressed : Save several arrays into a compressed ``.npz`` archive

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is not compressed and each file
    in the archive contains one variable in ``.npy`` format. For a
    description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    When opening the saved ``.npz`` file with `load` a `~lib.npyio.NpzFile`
    object is returned. This is a dictionary-like object which can be queried
    for its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Keys passed in `kwds` are used as filenames inside the ZIP archive.
    Therefore, keys should be valid filenames; e.g., avoid keys that begin with
    ``/`` or contain ``.``.

    When naming variables with keyword arguments, it is not possible to name a
    variable ``file``, as this would cause the ``file`` argument to be defined
    twice in the call to ``savez``.

    Examples
    --------
    >>> import numpy as np
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)

    Using `savez` with \\*args, the arrays are saved with default names.

    >>> np.savez(outfile, x, y)
    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['arr_0', 'arr_1']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using `savez` with \\**kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)
    >>> npzfile = np.load(outfile)
    >>> sorted(npzfile.files)
    ['x', 'y']
    >>> npzfile['x']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    _savez(file, args, kwds, False, allow_pickle=allow_pickle)


def _savez_compressed_dispatcher(file, *args, allow_pickle=True, **kwds):
    yield from args
    yield from kwds.values()


@array_function_dispatch(_savez_compressed_dispatcher)
def savez_compressed(file, *args, allow_pickle=True, **kwds):
    """
    Save several arrays into a single file in compressed ``.npz`` format.

    Provide arrays as keyword arguments to store them under the
    corresponding name in the output file: ``savez_compressed(fn, x=x, y=y)``.

    If arrays are specified as positional arguments, i.e.,
    ``savez_compressed(fn, x, y)``, their names will be `arr_0`, `arr_1`, etc.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    args : Arguments, optional
        Arrays to save to the file. Please use keyword arguments (see
        `kwds` below) to assign names to arrays.  Arrays specified as
        args will be named "arr_0", "arr_1", and so on.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    kwds : Keyword arguments, optional
        Arrays to save to the file. Each array will be saved to the
        output file with its corresponding keyword name.

    Returns
    -------
    None

    See Also
    --------
    numpy.save : Save a single array to a binary file in NumPy format.
    numpy.savetxt : Save an array to a file as plain text.
    numpy.savez : Save several arrays into an uncompressed ``.npz`` file format
    numpy.load : Load the files created by savez_compressed.

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is compressed with
    ``zipfile.ZIP_DEFLATED`` and each file in the archive contains one variable
    in ``.npy`` format. For a description of the ``.npy`` format, see
    :py:mod:`numpy.lib.format`.


    When opening the saved ``.npz`` file with `load` a `~lib.npyio.NpzFile`
    object is returned. This is a dictionary-like object which can be queried
    for its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Examples
    --------
    >>> import numpy as np
    >>> test_array = np.random.rand(3, 2)
    >>> test_vector = np.random.rand(4)
    >>> np.savez_compressed('/tmp/123', a=test_array, b=test_vector)
    >>> loaded = np.load('/tmp/123.npz')
    >>> print(np.array_equal(test_array, loaded['a']))
    True
    >>> print(np.array_equal(test_vector, loaded['b']))
    True

    """
    _savez(file, args, kwds, True, allow_pickle=allow_pickle)


def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile

    if not hasattr(file, 'write'):
        file = os.fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError(
                "Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zipf = zipfile_factory(file, mode="w", compression=compression)
    try:
        for key, val in namedict.items():
            fname = key + '.npy'
            val = np.asanyarray(val)
            # always force zip64, gh-10776
            with zipf.open(fname, 'w', force_zip64=True) as fid:
                format.write_array(fid, val,
                                   allow_pickle=allow_pickle,
                                   pickle_kwargs=pickle_kwargs)
    finally:
        zipf.close()


def _ensure_ndmin_ndarray_check_param(ndmin):
    """Just checks if the param ndmin is supported on
        _ensure_ndmin_ndarray. It is intended to be used as
        verification before running anything expensive.
        e.g. loadtxt, genfromtxt
    """
    # Check correctness of the values of `ndmin`
    if ndmin not in [0, 1, 2]:
        raise ValueError(f"Illegal value of ndmin keyword: {ndmin}")

def _ensure_ndmin_ndarray(a, *, ndmin: int):
    """This is a helper function of loadtxt and genfromtxt to ensure
        proper minimum dimension as requested

        ndim : int. Supported values 1, 2, 3
                    ^^ whenever this changes, keep in sync with
                       _ensure_ndmin_ndarray_check_param
    """
    # Verify that the array has at least dimensions `ndmin`.
    # Tweak the size and shape of the arrays - remove extraneous dimensions
    if a.ndim > ndmin:
        a = np.squeeze(a)
    # and ensure we have the minimum number of dimensions asked for
    # - has to be in this order for the odd case ndmin=1, a.squeeze().ndim=0
    if a.ndim < ndmin:
        if ndmin == 1:
            a = np.atleast_1d(a)
        elif ndmin == 2:
            a = np.atleast_2d(a).T

    return a


# amount of lines loadtxt reads in one chunk, can be overridden for testing
_loadtxt_chunksize = 50000


def _check_nonneg_int(value, name="argument"):
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _preprocess_comments(iterable, comments, encoding):
    """
    Generator that consumes a line iterated iterable and strips out the
    multiple (or multi-character) comments from lines.
    This is a pre-processing step to achieve feature parity with loadtxt
    (we assume that this feature is a nieche feature).
    """
    for line in iterable:
        if isinstance(line, bytes):
            # Need to handle conversion here, or the splitting would fail
            line = line.decode(encoding)

        for c in comments:
            line = line.split(c, 1)[0]

        yield line


# The number of rows we read in one go if confronted with a parametric dtype
_loadtxt_chunksize = 50000


def _read(fname, *, delimiter=',', comment='#', quote='"',
          imaginary_unit='j', usecols=None, skiplines=0,
          max_rows=None, converters=None, ndmin=None, unpack=False,
          dtype=np.float64, encoding=None):
    r"""
    Read a NumPy array from a text file.
    This is a helper function for loadtxt.

    Parameters
    ----------
    fname : file, str, or pathlib.Path
        The filename or the file to be read.
    delimiter : str, optional
        Field delimiter of the fields in line of the file.
        Default is a comma, ','.  If None any sequence of whitespace is
        considered a delimiter.
    comment : str or sequence of str or None, optional
        Character that begins a comment.  All text from the comment
        character to the end of the line is ignored.
        Multiple comments or multiple-character comment strings are supported,
        but may be slower and `quote` must be empty if used.
        Use None to disable all use of comments.
    quote : str or None, optional
        Character that is used to quote string fields. Default is '"'
        (a double quote). Use None to disable quote support.
    imaginary_unit : str, optional
        Character that represent the imaginary unit `sqrt(-1)`.
        Default is 'j'.
    usecols : array_like, optional
        A one-dimensional array of integer column numbers.  These are the
        columns from the file to be included in the array.  If this value
        is not given, all the columns are used.
    skiplines : int, optional
        Number of lines to skip before interpreting the data in the file.
    max_rows : int, optional
        Maximum number of rows of data to read.  Default is to read the
        entire file.
    converters : dict or callable, optional
        A function to parse all columns strings into the desired value, or
        a dictionary mapping column number to a parser function.
        E.g. if column 0 is a date string: ``converters = {0: datestr2num}``.
        Converters can also be used to provide a default value for missing
        data, e.g. ``converters = lambda s: float(s.strip() or 0)`` will
        convert empty fields to 0.
        Default: None
    ndmin : int, optional
        Minimum dimension of the array returned.
        Allowed values are 0, 1 or 2.  Default is 0.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = read(...)``.  When used with a structured
        data-type, arrays are returned for each field.  Default is False.
    dtype : numpy data type
        A NumPy dtype instance, can be a structured dtype to map to the
        columns of the file.
    encoding : str, optional
        Encoding used to decode the inputfile. The special value 'bytes'
        (the default) enables backwards-compatible behavior for `converters`,
        ensuring that inputs to the converter functions are encoded
        bytes objects. The special value 'bytes' has no additional effect if
        ``converters=None``. If encoding is ``'bytes'`` or ``None``, the
        default system encoding is used.

    Returns
    -------
    ndarray
        NumPy array.
    """
    # Handle special 'bytes' keyword for encoding
    byte_converters = False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True

    if dtype is None:
        raise TypeError("a dtype must be provided.")
    dtype = np.dtype(dtype)

    read_dtype_via_object_chunks = None
    if dtype.kind in 'SUM' and (
            dtype == "S0" or dtype == "U0" or dtype == "M8" or dtype == 'm8'):
        # This is a legacy "flexible" dtype.  We do not truly support
        # parametric dtypes currently (no dtype discovery step in the core),
        # but have to support these for backward compatibility.
        read_dtype_via_object_chunks = dtype
        dtype = np.dtype(object)

    if usecols is not None:
        # Allow usecols to be a single int or a sequence of ints, the C-code
        # handles the rest
        try:
            usecols = list(usecols)
        except TypeError:
            usecols = [usecols]

    _ensure_ndmin_ndarray_check_param(ndmin)

    if comment is None:
        comments = None
    else:
        # assume comments are a sequence of strings
        if "" in comment:
            raise ValueError(
                "comments cannot be an empty string. Use comments=None to "
                "disable comments."
            )
        comments = tuple(comment)
        comment = None
        if len(comments) == 0:
            comments = None  # No comments at all
        elif len(comments) == 1:
            # If there is only one comment, and that comment has one character,
            # the normal parsing can deal with it just fine.
            if isinstance(comments[0], str) and len(comments[0]) == 1:
                comment = comments[0]
                comments = None
        else:
            # Input validation if there are multiple comment characters
            if delimiter in comments:
                raise TypeError(
                    f"Comment characters '{comments}' cannot include the "
                    f"delimiter '{delimiter}'"
                )

    # comment is now either a 1 or 0 character string or a tuple:
    if comments is not None:
        # Note: An earlier version support two character comments (and could
        #       have been extended to multiple characters, we assume this is
        #       rare enough to not optimize for.
        if quote is not None:
            raise ValueError(
                "when multiple comments or a multi-character comment is "
                "given, quotes are not supported.  In this case quotechar "
                "must be set to None.")

    if len(imaginary_unit) != 1:
        raise ValueError('len(imaginary_unit) must be 1.')

    _check_nonneg_int(skiplines)
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    else:
        # Passing -1 to the C code means "read the entire file".
        max_rows = -1

    fh_closing_ctx = contextlib.nullcontext()
    filelike = False
    try:
        if isinstance(fname, os.PathLike):
            fname = os.fspath(fname)
        if isinstance(fname, str):
            fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            if encoding is None:
                encoding = getattr(fh, 'encoding', 'latin1')

            fh_closing_ctx = contextlib.closing(fh)
            data = fh
            filelike = True
        else:
            if encoding is None:
                encoding = getattr(fname, 'encoding', 'latin1')
            data = iter(fname)
    except TypeError as e:
        raise ValueError(
            f"fname must be a string, filehandle, list of strings,\n"
            f"or generator. Got {type(fname)} instead.") from e

    with fh_closing_ctx:
        if comments is not None:
            if filelike:
                data = iter(data)
                filelike = False
            data = _preprocess_comments(data, comments, encoding)

        if read_dtype_via_object_chunks is None:
            arr = _load_from_filelike(
                data, delimiter=delimiter, comment=comment, quote=quote,
                imaginary_unit=imaginary_unit,
                usecols=usecols, skiplines=skiplines, max_rows=max_rows,
                converters=converters, dtype=dtype,
                encoding=encoding, filelike=filelike,
                byte_converters=byte_converters)

        else:
            # This branch reads the file into chunks of object arrays and then
            # casts them to the desired actual dtype.  This ensures correct
            # string-length and datetime-unit discovery (like `arr.astype()`).
            # Due to chunking, certain error reports are less clear, currently.
            if filelike:
                data = iter(data)  # cannot chunk when reading from file
                filelike = False

            c_byte_converters = False
            if read_dtype_via_object_chunks == "S":
                c_byte_converters = True  # Use latin1 rather than ascii

            chunks = []
            while max_rows != 0:
                if max_rows < 0:
                    chunk_size = _loadtxt_chunksize
                else:
                    chunk_size = min(_loadtxt_chunksize, max_rows)

                next_arr = _load_from_filelike(
                    data, delimiter=delimiter, comment=comment, quote=quote,
                    imaginary_unit=imaginary_unit,
                    usecols=usecols, skiplines=skiplines, max_rows=chunk_size,
                    converters=converters, dtype=dtype,
                    encoding=encoding, filelike=filelike,
                    byte_converters=byte_converters,
                    c_byte_converters=c_byte_converters)
                # Cast here already.  We hope that this is better even for
                # large files because the storage is more compact.  It could
                # be adapted (in principle the concatenate could cast).
                chunks.append(next_arr.astype(read_dtype_via_object_chunks))

                skiplines = 0  # Only have to skip for first chunk
                if max_rows >= 0:
                    max_rows -= chunk_size
                if len(next_arr) < chunk_size:
                    # There was less data than requested, so we are done.
                    break

            # Need at least one chunk, but if empty, the last one may have
            # the wrong shape.
            if len(chunks) > 1 and len(chunks[-1]) == 0:
                del chunks[-1]
            if len(chunks) == 1:
                arr = chunks[0]
            else:
                arr = np.concatenate(chunks, axis=0)

    # NOTE: ndmin works as advertised for structured dtypes, but normally
    #       these would return a 1D result plus the structured dimension,
    #       so ndmin=2 adds a third dimension even when no squeezing occurs.
    #       A `squeeze=False` could be a better solution (pandas uses squeeze).
    arr = _ensure_ndmin_ndarray(arr, ndmin=ndmin)

    if arr.shape:
        if arr.shape[0] == 0:
            warnings.warn(
                f'loadtxt: input contained no data: "{fname}"',
                category=UserWarning,
                stacklevel=3
            )

    if unpack:
        # Unpack structured dtypes if requested:
        dt = arr.dtype
        if dt.names is not None:
            # For structured arrays, return an array for each field.
            return [arr[field] for field in dt.names]
        else:
            return arr.T
    else:
        return arr


@finalize_array_function_like
@set_module('numpy')
def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding=None, max_rows=None, *, quotechar=None,
            like=None):
    r"""
    Load data from a text file.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read.  If the filename
        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
        that generators must return bytes or strings. The strings
        in a list or produced by a generator are treated as lines.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str or None, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The character used to separate the values. For backwards compatibility,
        byte strings will be decoded as 'latin1'. The default is whitespace.

        .. versionchanged:: 1.23.0
           Only single character delimiters are supported. Newline characters
           cannot be used as the delimiter.

    converters : dict or callable, optional
        Converter functions to customize value parsing. If `converters` is
        callable, the function is applied to all columns, else it must be a
        dict that maps column number to a parser function.
        See examples for further details.
        Default: None.

        .. versionchanged:: 1.23.0
           The ability to pass a single callable to be applied to all columns
           was added.

    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The default, None, results in all columns being read.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``.  When used with a
        structured data-type, arrays are returned for each field.
        Default is False.
    ndmin : int, optional
        The returned array will have at least `ndmin` dimensions.
        Otherwise mono-dimensional axes will be squeezed.
        Legal values: 0 (default), 1 or 2.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams.
        The special value 'bytes' enables backward compatibility workarounds
        that ensures you receive byte arrays as results if possible and passes
        'latin1' encoded strings to converters. Override this value to receive
        unicode arrays and pass strings as input to converters.  If set to None
        the system default is used. The default value is 'bytes'.

        .. versionchanged:: 2.0
            Before NumPy 2, the default was ``'bytes'`` for Python 2
            compatibility. The default is now ``None``.

    max_rows : int, optional
        Read `max_rows` rows of content after `skiprows` lines. The default is
        to read all the rows. Note that empty rows containing no data such as
        empty lines and comment lines are not counted towards `max_rows`,
        while such lines are counted in `skiprows`.

        .. versionchanged:: 1.23.0
            Lines containing no data, including comment lines (e.g., lines
            starting with '#' or as specified via `comments`) are not counted
            towards `max_rows`.
    quotechar : unicode character or None, optional
        The character used to denote the start and end of a quoted item.
        Occurrences of the delimiter or comment characters are ignored within
        a quoted item. The default value is ``quotechar=None``, which means
        quoting support is disabled.

        If two consecutive instances of `quotechar` are found within a quoted
        field, the first is treated as an escape character. See examples.

        .. versionadded:: 1.23.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    load, fromstring, fromregex
    genfromtxt : Load data with missing values handled as specified.
    scipy.io.loadmat : reads MATLAB data files

    Notes
    -----
    This function aims to be a fast reader for simply formatted files.  The
    `genfromtxt` function provides more sophisticated handling of, e.g.,
    lines with missing values.

    Each row in the input text file must have the same number of values to be
    able to read all values. If all rows do not have same number of values, a
    subset of up to n columns (where n is the least number of values present
    in all rows) can be read by specifying the columns via `usecols`.

    The strings produced by the Python float.hex method can be used as
    input for floats.

    Examples
    --------
    >>> import numpy as np
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\n2 3")
    >>> np.loadtxt(c)
    array([[0., 1.],
           [2., 3.]])

    >>> d = StringIO("M 21 72\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([(b'M', 21, 72.), (b'F', 35, 58.)],
          dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\n3,0,4")
    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([1., 3.])
    >>> y
    array([2., 4.])

    The `converters` argument is used to specify functions to preprocess the
    text prior to parsing. `converters` can be a dictionary that maps
    preprocessing functions to each column:

    >>> s = StringIO("1.618, 2.296\n3.141, 4.669\n")
    >>> conv = {
    ...     0: lambda x: np.floor(float(x)),  # conversion fn for column 0
    ...     1: lambda x: np.ceil(float(x)),  # conversion fn for column 1
    ... }
    >>> np.loadtxt(s, delimiter=",", converters=conv)
    array([[1., 3.],
           [3., 5.]])

    `converters` can be a callable instead of a dictionary, in which case it
    is applied to all columns:

    >>> s = StringIO("0xDE 0xAD\n0xC0 0xDE")
    >>> import functools
    >>> conv = functools.partial(int, base=16)
    >>> np.loadtxt(s, converters=conv)
    array([[222., 173.],
           [192., 222.]])

    This example shows how `converters` can be used to convert a field
    with a trailing minus sign into a negative number.

    >>> s = StringIO("10.01 31.25-\n19.22 64.31\n17.57- 63.94")
    >>> def conv(fld):
    ...     return -float(fld[:-1]) if fld.endswith("-") else float(fld)
    ...
    >>> np.loadtxt(s, converters=conv)
    array([[ 10.01, -31.25],
           [ 19.22,  64.31],
           [-17.57,  63.94]])

    Using a callable as the converter can be particularly useful for handling
    values with different formatting, e.g. floats with underscores:

    >>> s = StringIO("1 2.7 100_000")
    >>> np.loadtxt(s, converters=float)
    array([1.e+00, 2.7e+00, 1.e+05])

    This idea can be extended to automatically handle values specified in
    many different formats, such as hex values:

    >>> def conv(val):
    ...     try:
    ...         return float(val)
    ...     except ValueError:
    ...         return float.fromhex(val)
    >>> s = StringIO("1, 2.5, 3_000, 0b4, 0x1.4000000000000p+2")
    >>> np.loadtxt(s, delimiter=",", converters=conv)
    array([1.0e+00, 2.5e+00, 3.0e+03, 1.8e+02, 5.0e+00])

    Or a format where the ``-`` sign comes after the number:

    >>> s = StringIO("10.01 31.25-\n19.22 64.31\n17.57- 63.94")
    >>> conv = lambda x: -float(x[:-1]) if x.endswith("-") else float(x)
    >>> np.loadtxt(s, converters=conv)
    array([[ 10.01, -31.25],
           [ 19.22,  64.31],
           [-17.57,  63.94]])

    Support for quoted fields is enabled with the `quotechar` parameter.
    Comment and delimiter characters are ignored when they appear within a
    quoted item delineated by `quotechar`:

    >>> s = StringIO('"alpha, #42", 10.0\n"beta, #64", 2.0\n')
    >>> dtype = np.dtype([("label", "U12"), ("value", float)])
    >>> np.loadtxt(s, dtype=dtype, delimiter=",", quotechar='"')
    array([('alpha, #42', 10.), ('beta, #64',  2.)],
          dtype=[('label', '<U12'), ('value', '<f8')])

    Quoted fields can be separated by multiple whitespace characters:

    >>> s = StringIO('"alpha, #42"       10.0\n"beta, #64" 2.0\n')
    >>> dtype = np.dtype([("label", "U12"), ("value", float)])
    >>> np.loadtxt(s, dtype=dtype, delimiter=None, quotechar='"')
    array([('alpha, #42', 10.), ('beta, #64',  2.)],
          dtype=[('label', '<U12'), ('value', '<f8')])

    Two consecutive quote characters within a quoted field are treated as a
    single escaped character:

    >>> s = StringIO('"Hello, my name is ""Monty""!"')
    >>> np.loadtxt(s, dtype="U", delimiter=",", quotechar='"')
    array('Hello, my name is "Monty"!', dtype='<U26')

    Read subset of columns when all rows do not contain equal number of values:

    >>> d = StringIO("1 2\n2 4\n3 9 12\n4 16 20")
    >>> np.loadtxt(d, usecols=(0, 1))
    array([[ 1.,  2.],
           [ 2.,  4.],
           [ 3.,  9.],
           [ 4., 16.]])

    """

    if like is not None:
        return _loadtxt_with_like(
            like, fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows, usecols=usecols,
            unpack=unpack, ndmin=ndmin, encoding=encoding,
            max_rows=max_rows
        )

    if isinstance(delimiter, bytes):
        delimiter.decode("latin1")

    if dtype is None:
        dtype = np.float64

    comment = comments
    # Control character type conversions for Py3 convenience
    if comment is not None:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [
            x.decode('latin1') if isinstance(x, bytes) else x for x in comment]
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode('latin1')

    arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
                converters=converters, skiplines=skiprows, usecols=usecols,
                unpack=unpack, ndmin=ndmin, encoding=encoding,
                max_rows=max_rows, quote=quotechar)

    return arr


_loadtxt_with_like = array_function_dispatch()(loadtxt)


def _savetxt_dispatcher(fname, X, fmt=None, delimiter=None, newline=None,
                        header=None, footer=None, comments=None,
                        encoding=None):
    return (X,)


@array_function_dispatch(_savetxt_dispatcher)
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : filename, file handle or pathlib.Path
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : 1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:

        * a single specifier, ``fmt='%.4e'``, resulting in numbers formatted
          like ``' (%s+%sj)' % (fmt, fmt)``
        * a full string specifying every real and imaginary part, e.g.
          ``' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'`` for 3 columns
        * a list of specifiers, one per column - in this case, the real
          and imaginary part must have separate specifiers,
          e.g. ``['%.3e + %.3ej', '(%.15e%+.15ej)']`` for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
    header : str, optional
        String that will be written at the beginning of the file.
    footer : str, optional
        String that will be written at the end of the file.
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ',  as expected by e.g.
        ``numpy.loadtxt``.
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output
        streams. If the encoding is something other than 'bytes' or 'latin1'
        you will not be able to load the file in NumPy versions < 1.14. Default
        is 'latin1'.

    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into an uncompressed ``.npz`` archive
    savez_compressed : Save several arrays into a compressed ``.npz`` archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to precede result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <https://docs.python.org/library/string.html#format-specification-mini-language>`_,
           Python Documentation.

    Examples
    --------
    >>> import numpy as np
    >>> x = y = z = np.arange(0.0,5.0,1.0)
    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    """

    class WriteWrap:
        """Convert to bytes on bytestream inputs.

        """
        def __init__(self, fh, encoding):
            self.fh = fh
            self.encoding = encoding
            self.do_write = self.first_write

        def close(self):
            self.fh.close()

        def write(self, v):
            self.do_write(v)

        def write_bytes(self, v):
            if isinstance(v, bytes):
                self.fh.write(v)
            else:
                self.fh.write(v.encode(self.encoding))

        def write_normal(self, v):
            self.fh.write(asunicode(v))

        def first_write(self, v):
            try:
                self.write_normal(v)
                self.write = self.write_normal
            except TypeError:
                # input is probably a bytestream
                self.write_bytes(v)
                self.write = self.write_bytes

    own_fh = False
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if _is_string_like(fname):
        # datasource doesn't support creating a new file ...
        open(fname, 'wt').close()
        fh = np.lib._datasource.open(fname, 'wt', encoding=encoding)
        own_fh = True
    elif hasattr(fname, 'write'):
        # wrap to handle byte output streams
        fh = WriteWrap(fname, encoding or 'latin1')
    else:
        raise ValueError('fname must be a string or file handle')

    try:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        if X.ndim == 0 or X.ndim > 2:
            raise ValueError(
                "Expected 1D or 2D array, got %dD array instead" % X.ndim)
        elif X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(X.dtype.names)
        else:
            ncol = X.shape[1]

        iscomplex_X = np.iscomplexobj(X)
        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = delimiter.join(fmt)
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))

        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(comments + header + newline)
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                    row2.append(number.real)
                    row2.append(number.imag)
                s = format % tuple(row2) + newline
                fh.write(s.replace('+-', '-'))
        else:
            for row in X:
                try:
                    v = format % tuple(row) + newline
                except TypeError as e:
                    raise TypeError("Mismatch between array dtype ('%s') and "
                                    "format specifier ('%s')"
                                    % (str(X.dtype), format)) from e
                fh.write(v)

        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(comments + footer + newline)
    finally:
        if own_fh:
            fh.close()


@set_module('numpy')
def fromregex(file, regexp, dtype, encoding=None):
    r"""
    Construct an array from a text file, using regular expression parsing.

    The returned array is always a structured array, and is constructed from
    all matches of the regular expression in the file. Groups in the regular
    expression are converted to fields of the structured array.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Filename or file object to read.

        .. versionchanged:: 1.22.0
            Now accepts `os.PathLike` implementations.

    regexp : str or regexp
        Regular expression used to parse the file.
        Groups in the regular expression correspond to fields in the dtype.
    dtype : dtype or list of dtypes
        Dtype for the structured array; must be a structured datatype.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams.

    Returns
    -------
    output : ndarray
        The output array, containing the part of the content of `file` that
        was matched by `regexp`. `output` is always a structured array.

    Raises
    ------
    TypeError
        When `dtype` is not a valid dtype for a structured array.

    See Also
    --------
    fromstring, loadtxt

    Notes
    -----
    Dtypes for structured arrays can be specified in several forms, but all
    forms specify at least the data type and field name. For details see
    `basics.rec`.

    Examples
    --------
    >>> import numpy as np
    >>> from io import StringIO
    >>> text = StringIO("1312 foo\n1534  bar\n444   qux")

    >>> regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]
    >>> output = np.fromregex(text, regexp,
    ...                       [('num', np.int64), ('key', 'S3')])
    >>> output
    array([(1312, b'foo'), (1534, b'bar'), ( 444, b'qux')],
          dtype=[('num', '<i8'), ('key', 'S3')])
    >>> output['num']
    array([1312, 1534,  444])

    """
    own_fh = False
    if not hasattr(file, "read"):
        file = os.fspath(file)
        file = np.lib._datasource.open(file, 'rt', encoding=encoding)
        own_fh = True

    try:
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        if dtype.names is None:
            raise TypeError('dtype must be a structured datatype.')

        content = file.read()
        if isinstance(content, bytes) and isinstance(regexp, str):
            regexp = asbytes(regexp)

        if not hasattr(regexp, 'match'):
            regexp = re.compile(regexp)
        seq = regexp.findall(content)
        if seq and not isinstance(seq[0], tuple):
            # Only one group is in the regexp.
            # Create the new array as a single data-type and then
            #   re-interpret as a single-field structured array.
            newdtype = np.dtype(dtype[dtype.names[0]])
            output = np.array(seq, dtype=newdtype)
            output.dtype = dtype
        else:
            output = np.array(seq, dtype=dtype)

        return output
    finally:
        if own_fh:
            file.close()


#####--------------------------------------------------------------------------
#---- --- ASCII functions ---
#####--------------------------------------------------------------------------


@finalize_array_function_like
@set_module('numpy')
def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None,
               deletechars=''.join(sorted(NameValidator.defaultdeletechars)),
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt="f%i", unpack=None, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding=None,
               *, ndmin=0, like=None):
    """
    Load data from a text file, with missing values handled as specified.

    Each line past the first `skip_header` lines is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read.  If the filename
        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
        that generators must return bytes or strings. The strings
        in a list or produced by a generator are treated as lines.
    dtype : dtype, optional
        Data type of the resulting array.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : str, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded.
    delimiter : str, int, or sequence, optional
        The string used to separate values.  By default, any consecutive
        whitespaces act as delimiter.  An integer or sequence of integers
        can also be provided as width(s) of each field.
    skiprows : int, optional
        `skiprows` was removed in numpy 1.10. Please use `skip_header` instead.
    skip_header : int, optional
        The number of lines to skip at the beginning of the file.
    skip_footer : int, optional
        The number of lines to skip at the end of the file.
    converters : variable, optional
        The set of functions that convert the data of a column to a value.
        The converters can also be used to provide a default value
        for missing data: ``converters = {3: lambda s: float(s or 0)}``.
    missing : variable, optional
        `missing` was removed in numpy 1.10. Please use `missing_values`
        instead.
    missing_values : variable, optional
        The set of strings corresponding to missing data.
    filling_values : variable, optional
        The set of values to be used as default when the data are missing.
    usecols : sequence, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, str, sequence}, optional
        If `names` is True, the field names are read from the first line after
        the first `skip_header` lines. This line can optionally be preceded
        by a comment delimiter. Any content before the comment delimiter is
        discarded. If `names` is a sequence or a single-string of
        comma-separated names, the names will be used to define the field
        names in a structured dtype. If `names` is None, the names of the
        dtype fields will be used, if any.
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended with an
        underscore: for example, `file` would become `file_`.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    defaultfmt : str, optional
        A format used to define default field names, such as "f%i" or "f_%02i".
    autostrip : bool, optional
        Whether to automatically strip white spaces from the variables.
    replace_space : char, optional
        Character(s) used in replacement of white spaces in the variable
        names. By default, use a '_'.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        If True, field names are case sensitive.
        If False or 'upper', field names are converted to upper case.
        If 'lower', field names are converted to lower case.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = genfromtxt(...)``.  When used with a
        structured data-type, arrays are returned for each field.
        Default is False.
    usemask : bool, optional
        If True, return a masked array.
        If False, return a regular array.
    loose : bool, optional
        If True, do not raise errors for invalid values.
    invalid_raise : bool, optional
        If True, an exception is raised if an inconsistency is detected in the
        number of columns.
        If False, a warning is emitted and the offending lines are skipped.
    max_rows : int,  optional
        The maximum number of rows to read. Must not be used with skip_footer
        at the same time.  If given, the value must be at least 1. Default is
        to read the entire file.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply when `fname`
        is a file object. The special value 'bytes' enables backward
        compatibility workarounds that ensure that you receive byte arrays
        when possible and passes latin1 encoded strings to converters.
        Override this value to receive unicode arrays and pass strings
        as input to converters.  If set to None the system default is used.
        The default value is 'bytes'.

        .. versionchanged:: 2.0
            Before NumPy 2, the default was ``'bytes'`` for Python 2
            compatibility. The default is now ``None``.

    ndmin : int, optional
        Same parameter as `loadtxt`

        .. versionadded:: 1.23.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Data read from the text file. If `usemask` is True, this is a
        masked array.

    See Also
    --------
    numpy.loadtxt : equivalent function when no data is missing.

    Notes
    -----
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When variables are named (either by a flexible dtype or with a `names`
      sequence), there must not be any header in the file (else a ValueError
      exception is raised).
    * Individual values are not stripped of spaces by default.
      When using a custom converter, make sure the function does remove spaces.
    * Custom converters may receive unexpected values due to dtype
      discovery.

    References
    ----------
    .. [1] NumPy User Guide, section `I/O with NumPy
           <https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_.

    Examples
    --------
    >>> from io import StringIO
    >>> import numpy as np

    Comma delimited file with mixed dtype

    >>> s = StringIO("1,1.3,abcde")
    >>> data = np.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
    ... ('mystring','S5')], delimiter=",")
    >>> data
    array((1, 1.3, b'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

    Using dtype = None

    >>> _ = s.seek(0) # needed for StringIO example only
    >>> data = np.genfromtxt(s, dtype=None,
    ... names = ['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '<U5')])

    Specifying dtype and names

    >>> _ = s.seek(0)
    >>> data = np.genfromtxt(s, dtype="i8,f8,S5",
    ... names=['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, b'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

    An example with fixed-width columns

    >>> s = StringIO("11.3abcde")
    >>> data = np.genfromtxt(s, dtype=None, names=['intvar','fltvar','strvar'],
    ...     delimiter=[1,3,5])
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', '<U5')])

    An example to show comments

    >>> f = StringIO('''
    ... text,# of chars
    ... hello world,11
    ... numpy,5''')
    >>> np.genfromtxt(f, dtype='S12,S12', delimiter=',')
    array([(b'text', b''), (b'hello world', b'11'), (b'numpy', b'5')],
      dtype=[('f0', 'S12'), ('f1', 'S12')])

    """

    if like is not None:
        return _genfromtxt_with_like(
            like, fname, dtype=dtype, comments=comments, delimiter=delimiter,
            skip_header=skip_header, skip_footer=skip_footer,
            converters=converters, missing_values=missing_values,
            filling_values=filling_values, usecols=usecols, names=names,
            excludelist=excludelist, deletechars=deletechars,
            replace_space=replace_space, autostrip=autostrip,
            case_sensitive=case_sensitive, defaultfmt=defaultfmt,
            unpack=unpack, usemask=usemask, loose=loose,
            invalid_raise=invalid_raise, max_rows=max_rows, encoding=encoding,
            ndmin=ndmin,
        )

    _ensure_ndmin_ndarray_check_param(ndmin)

    if max_rows is not None:
        if skip_footer:
            raise ValueError(
                    "The keywords 'skip_footer' and 'max_rows' can not be "
                    "specified at the same time.")
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")

    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr
    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        raise TypeError(
            "The input argument 'converter' should be a valid dictionary "
            "(got '%s' instead)" % type(user_converters))

    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    else:
        byte_converters = False

    # Initialize the filehandle, the LineSplitter and the NameValidator
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if isinstance(fname, str):
        fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
        fid_ctx = contextlib.closing(fid)
    else:
        fid = fname
        fid_ctx = contextlib.nullcontext(fid)
    try:
        fhd = iter(fid)
    except TypeError as e:
        raise TypeError(
            "fname must be a string, a filehandle, a sequence of strings,\n"
            f"or an iterator of strings. Got {type(fname)} instead."
        ) from e
    with fid_ctx:
        split_line = LineSplitter(delimiter=delimiter, comments=comments,
                                  autostrip=autostrip, encoding=encoding)
        validate_names = NameValidator(excludelist=excludelist,
                                       deletechars=deletechars,
                                       case_sensitive=case_sensitive,
                                       replace_space=replace_space)

        # Skip the first `skip_header` rows
        try:
            for i in range(skip_header):
                next(fhd)

            # Keep on until we find the first valid values
            first_values = None

            while not first_values:
                first_line = _decode_line(next(fhd), encoding)
                if (names is True) and (comments is not None):
                    if comments in first_line:
                        first_line = (
                            ''.join(first_line.split(comments)[1:]))
                first_values = split_line(first_line)
        except StopIteration:
            # return an empty array if the datafile is empty
            first_line = ''
            first_values = []
            warnings.warn(
                'genfromtxt: Empty input file: "%s"' % fname, stacklevel=2
            )

        # Should we take the first values as names ?
        if names is True:
            fval = first_values[0].strip()
            if comments is not None:
                if fval in comments:
                    del first_values[0]

        # Check the columns to use: make sure `usecols` is a list
        if usecols is not None:
            try:
                usecols = [_.strip() for _ in usecols.split(",")]
            except AttributeError:
                try:
                    usecols = list(usecols)
                except TypeError:
                    usecols = [usecols, ]
        nbcols = len(usecols or first_values)

        # Check the names and overwrite the dtype.names if needed
        if names is True:
            names = validate_names([str(_.strip()) for _ in first_values])
            first_line = ''
        elif _is_string_like(names):
            names = validate_names([_.strip() for _ in names.split(',')])
        elif names:
            names = validate_names(names)
        # Get the dtype
        if dtype is not None:
            dtype = easy_dtype(dtype, defaultfmt=defaultfmt, names=names,
                               excludelist=excludelist,
                               deletechars=deletechars,
                               case_sensitive=case_sensitive,
                               replace_space=replace_space)
        # Make sure the names is a list (for 2.5)
        if names is not None:
            names = list(names)

        if usecols:
            for (i, current) in enumerate(usecols):
                # if usecols is a list of names, convert to a list of indices
                if _is_string_like(current):
                    usecols[i] = names.index(current)
                elif current < 0:
                    usecols[i] = current + len(first_values)
            # If the dtype is not None, make sure we update it
            if (dtype is not None) and (len(dtype) > nbcols):
                descr = dtype.descr
                dtype = np.dtype([descr[_] for _ in usecols])
                names = list(dtype.names)
            # If `names` is not None, update the names
            elif (names is not None) and (len(names) > nbcols):
                names = [names[_] for _ in usecols]
        elif (names is not None) and (dtype is not None):
            names = list(dtype.names)

        # Process the missing values ...............................
        # Rename missing_values for convenience
        user_missing_values = missing_values or ()
        if isinstance(user_missing_values, bytes):
            user_missing_values = user_missing_values.decode('latin1')

        # Define the list of missing_values (one column: one list)
        missing_values = [[''] for _ in range(nbcols)]

        # We have a dictionary: process it field by field
        if isinstance(user_missing_values, dict):
            # Loop on the items
            for (key, val) in user_missing_values.items():
                # Is the key a string ?
                if _is_string_like(key):
                    try:
                        # Transform it into an integer
                        key = names.index(key)
                    except ValueError:
                        # We couldn't find it: the name must have been dropped
                        continue
                # Redefine the key as needed if it's a column number
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                # Transform the value as a list of string
                if isinstance(val, (list, tuple)):
                    val = [str(_) for _ in val]
                else:
                    val = [str(val), ]
                # Add the value(s) to the current list of missing
                if key is None:
                    # None acts as default
                    for miss in missing_values:
                        miss.extend(val)
                else:
                    missing_values[key].extend(val)
        # We have a sequence : each item matches a column
        elif isinstance(user_missing_values, (list, tuple)):
            for (value, entry) in zip(user_missing_values, missing_values):
                value = str(value)
                if value not in entry:
                    entry.append(value)
        # We have a string : apply it to all entries
        elif isinstance(user_missing_values, str):
            user_value = user_missing_values.split(",")
            for entry in missing_values:
                entry.extend(user_value)
        # We have something else: apply it to all entries
        else:
            for entry in missing_values:
                entry.extend([str(user_missing_values)])

        # Process the filling_values ...............................
        # Rename the input for convenience
        user_filling_values = filling_values
        if user_filling_values is None:
            user_filling_values = []
        # Define the default
        filling_values = [None] * nbcols
        # We have a dictionary : update each entry individually
        if isinstance(user_filling_values, dict):
            for (key, val) in user_filling_values.items():
                if _is_string_like(key):
                    try:
                        # Transform it into an integer
                        key = names.index(key)
                    except ValueError:
                        # We couldn't find it: the name must have been dropped
                        continue
                # Redefine the key if it's a column number
                # and usecols is defined
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                # Add the value to the list
                filling_values[key] = val
        # We have a sequence : update on a one-to-one basis
        elif isinstance(user_filling_values, (list, tuple)):
            n = len(user_filling_values)
            if (n <= nbcols):
                filling_values[:n] = user_filling_values
            else:
                filling_values = user_filling_values[:nbcols]
        # We have something else : use it for all entries
        else:
            filling_values = [user_filling_values] * nbcols

        # Initialize the converters ................................
        if dtype is None:
            # Note: we can't use a [...]*nbcols, as we would have 3 times
            # the same converter, instead of 3 different converters.
            converters = [
                StringConverter(None, missing_values=miss, default=fill)
                for (miss, fill) in zip(missing_values, filling_values)
            ]
        else:
            dtype_flat = flatten_dtype(dtype, flatten_base=True)
            # Initialize the converters
            if len(dtype_flat) > 1:
                # Flexible type : get a converter from each dtype
                zipit = zip(dtype_flat, missing_values, filling_values)
                converters = [StringConverter(dt,
                                              locked=True,
                                              missing_values=miss,
                                              default=fill)
                              for (dt, miss, fill) in zipit]
            else:
                # Set to a default converter (but w/ different missing values)
                zipit = zip(missing_values, filling_values)
                converters = [StringConverter(dtype,
                                              locked=True,
                                              missing_values=miss,
                                              default=fill)
                              for (miss, fill) in zipit]
        # Update the converters to use the user-defined ones
        uc_update = []
        for (j, conv) in user_converters.items():
            # If the converter is specified by column names,
            # use the index instead
            if _is_string_like(j):
                try:
                    j = names.index(j)
                    i = j
                except ValueError:
                    continue
            elif usecols:
                try:
                    i = usecols.index(j)
                except ValueError:
                    # Unused converter specified
                    continue
            else:
                i = j
            # Find the value to test - first_line is not filtered by usecols:
            if len(first_line):
                testing_value = first_values[j]
            else:
                testing_value = None
            if conv is bytes:
                user_conv = asbytes
            elif byte_converters:
                # Converters may use decode to workaround numpy's old
                # behavior, so encode the string again before passing
                # to the user converter.
                def tobytes_first(x, conv):
                    if type(x) is bytes:
                        return conv(x)
                    return conv(x.encode("latin1"))
                user_conv = functools.partial(tobytes_first, conv=conv)
            else:
                user_conv = conv
            converters[i].update(user_conv, locked=True,
                                 testing_value=testing_value,
                                 default=filling_values[i],
                                 missing_values=missing_values[i],)
            uc_update.append((i, user_conv))
        # Make sure we have the corrected keys in user_converters...
        user_converters.update(uc_update)

        # Fixme: possible error as following variable never used.
        # miss_chars = [_.missing_values for _ in converters]

        # Initialize the output lists ...
        # ... rows
        rows = []
        append_to_rows = rows.append
        # ... masks
        if usemask:
            masks = []
            append_to_masks = masks.append
        # ... invalid
        invalid = []
        append_to_invalid = invalid.append

        # Parse each line
        for (i, line) in enumerate(itertools.chain([first_line, ], fhd)):
            values = split_line(line)
            nbvalues = len(values)
            # Skip an empty line
            if nbvalues == 0:
                continue
            if usecols:
                # Select only the columns we need
                try:
                    values = [values[_] for _ in usecols]
                except IndexError:
                    append_to_invalid((i + skip_header + 1, nbvalues))
                    continue
            elif nbvalues != nbcols:
                append_to_invalid((i + skip_header + 1, nbvalues))
                continue
            # Store the values
            append_to_rows(tuple(values))
            if usemask:
                append_to_masks(tuple([v.strip() in m
                                       for (v, m) in zip(values,
                                                         missing_values)]))
            if len(rows) == max_rows:
                break

    # Upgrade the converters (if needed)
    if dtype is None:
        for (i, converter) in enumerate(converters):
            current_column = [itemgetter(i)(_m) for _m in rows]
            try:
                converter.iterupgrade(current_column)
            except ConverterLockError:
                errmsg = "Converter #%i is locked and cannot be upgraded: " % i
                current_column = map(itemgetter(i), rows)
                for (j, value) in enumerate(current_column):
                    try:
                        converter.upgrade(value)
                    except (ConverterError, ValueError):
                        errmsg += "(occurred line #%i for value '%s')"
                        errmsg %= (j + 1 + skip_header, value)
                        raise ConverterError(errmsg)

    # Check that we don't have invalid values
    nbinvalid = len(invalid)
    if nbinvalid > 0:
        nbrows = len(rows) + nbinvalid - skip_footer
        # Construct the error message
        template = "    Line #%%i (got %%i columns instead of %i)" % nbcols
        if skip_footer > 0:
            nbinvalid_skipped = len([_ for _ in invalid
                                     if _[0] > nbrows + skip_header])
            invalid = invalid[:nbinvalid - nbinvalid_skipped]
            skip_footer -= nbinvalid_skipped
#
#            nbrows -= skip_footer
#            errmsg = [template % (i, nb)
#                      for (i, nb) in invalid if i < nbrows]
#        else:
        errmsg = [template % (i, nb)
                  for (i, nb) in invalid]
        if len(errmsg):
            errmsg.insert(0, "Some errors were detected !")
            errmsg = "\n".join(errmsg)
            # Raise an exception ?
            if invalid_raise:
                raise ValueError(errmsg)
            # Issue a warning ?
            else:
                warnings.warn(errmsg, ConversionWarning, stacklevel=2)

    # Strip the last skip_footer data
    if skip_footer > 0:
        rows = rows[:-skip_footer]
        if usemask:
            masks = masks[:-skip_footer]

    # Convert each value according to the converter:
    # We want to modify the list in place to avoid creating a new one...
    if loose:
        rows = list(
            zip(*[[conv._loose_call(_r) for _r in map(itemgetter(i), rows)]
                  for (i, conv) in enumerate(converters)]))
    else:
        rows = list(
            zip(*[[conv._strict_call(_r) for _r in map(itemgetter(i), rows)]
                  for (i, conv) in enumerate(converters)]))

    # Reset the dtype
    data = rows
    if dtype is None:
        # Get the dtypes from the types of the converters
        column_types = [conv.type for conv in converters]
        # Find the columns with strings...
        strcolidx = [i for (i, v) in enumerate(column_types)
                     if v == np.str_]

        if byte_converters and strcolidx:
            # convert strings back to bytes for backward compatibility
            warnings.warn(
                "Reading unicode strings without specifying the encoding "
                "argument is deprecated. Set the encoding, use None for the "
                "system default.",
                np.exceptions.VisibleDeprecationWarning, stacklevel=2)

            def encode_unicode_cols(row_tup):
                row = list(row_tup)
                for i in strcolidx:
                    row[i] = row[i].encode('latin1')
                return tuple(row)

            try:
                data = [encode_unicode_cols(r) for r in data]
            except UnicodeEncodeError:
                pass
            else:
                for i in strcolidx:
                    column_types[i] = np.bytes_

        # Update string types to be the right length
        sized_column_types = column_types[:]
        for i, col_type in enumerate(column_types):
            if np.issubdtype(col_type, np.character):
                n_chars = max(len(row[i]) for row in data)
                sized_column_types[i] = (col_type, n_chars)

        if names is None:
            # If the dtype is uniform (before sizing strings)
            base = {
                c_type
                for c, c_type in zip(converters, column_types)
                if c._checked}
            if len(base) == 1:
                uniform_type, = base
                (ddtype, mdtype) = (uniform_type, bool)
            else:
                ddtype = [(defaultfmt % i, dt)
                          for (i, dt) in enumerate(sized_column_types)]
                if usemask:
                    mdtype = [(defaultfmt % i, bool)
                              for (i, dt) in enumerate(sized_column_types)]
        else:
            ddtype = list(zip(names, sized_column_types))
            mdtype = list(zip(names, [bool] * len(sized_column_types)))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        # Overwrite the initial dtype names if needed
        if names and dtype.names is not None:
            dtype.names = names
        # Case 1. We have a structured type
        if len(dtype_flat) > 1:
            # Nested dtype, eg [('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # First, create the array using a flattened dtype:
            # [('a', int), ('b1', int), ('b2', float)]
            # Then, view the array using the specified dtype.
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    raise NotImplementedError(
                        "Nested fields involving objects are not supported...")
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            # Now, process the rowmasks the same way
            if usemask:
                rowmasks = np.array(
                    masks, dtype=np.dtype([('', bool) for t in dtype_flat]))
                # Construct the new dtype
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        # Case #2. We have a basic dtype
        else:
            # We used some user-defined converters
            if user_converters:
                ishomogeneous = True
                descr = []
                for i, ttype in enumerate([conv.type for conv in converters]):
                    # Keep the dtype of the current converter
                    if i in user_converters:
                        ishomogeneous &= (ttype == dtype.type)
                        if np.issubdtype(ttype, np.character):
                            ttype = (ttype, max(len(row[i]) for row in data))
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                # So we changed the dtype ?
                if not ishomogeneous:
                    # We have more than one field
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    # We have only one field: drop the name if not needed.
                    else:
                        dtype = np.dtype(ttype)
            #
            output = np.array(data, dtype)
            if usemask:
                if dtype.names is not None:
                    mdtype = [(_, bool) for _ in dtype.names]
                else:
                    mdtype = bool
                outputmask = np.array(masks, dtype=mdtype)
    # Try to take care of the missing data we missed
    names = output.dtype.names
    if usemask and names:
        for (name, conv) in zip(names, converters):
            missing_values = [conv(_) for _ in conv.missing_values
                              if _ != '']
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    # Construct the final array
    if usemask:
        output = output.view(MaskedArray)
        output._mask = outputmask

    output = _ensure_ndmin_ndarray(output, ndmin=ndmin)

    if unpack:
        if names is None:
            return output.T
        elif len(names) == 1:
            # squeeze single-name dtypes too
            return output[names[0]]
        else:
            # For structured arrays with multiple fields,
            # return an array for each field.
            return [output[field] for field in names]
    return output


_genfromtxt_with_like = array_function_dispatch()(genfromtxt)


def recfromtxt(fname, **kwargs):
    """
    Load ASCII data from a file and return it in a record array.

    If ``usemask=False`` a standard `recarray` is returned,
    if ``usemask=True`` a MaskedRecords array is returned.

    .. deprecated:: 2.0
        Use `numpy.genfromtxt` instead.

    Parameters
    ----------
    fname, kwargs : For a description of input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function

    Notes
    -----
    By default, `dtype` is None, which means that the data-type of the output
    array will be determined from the data.

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`recfromtxt` is deprecated, "
        "use `numpy.genfromtxt` instead."
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    kwargs.setdefault("dtype", None)
    usemask = kwargs.get('usemask', False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


def recfromcsv(fname, **kwargs):
    """
    Load ASCII data stored in a comma-separated file.

    The returned array is a record array (if ``usemask=False``, see
    `recarray`) or a masked record array (if ``usemask=True``,
    see `ma.mrecords.MaskedRecords`).

    .. deprecated:: 2.0
        Use `numpy.genfromtxt` with comma as `delimiter` instead.

    Parameters
    ----------
    fname, kwargs : For a description of input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function to load ASCII data.

    Notes
    -----
    By default, `dtype` is None, which means that the data-type of the output
    array will be determined from the data.

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`recfromcsv` is deprecated, "
        "use `numpy.genfromtxt` with comma as `delimiter` instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    # Set default kwargs for genfromtxt as relevant to csv import.
    kwargs.setdefault("case_sensitive", "lower")
    kwargs.setdefault("names", True)
    kwargs.setdefault("delimiter", ",")
    kwargs.setdefault("dtype", None)
    output = genfromtxt(fname, **kwargs)

    usemask = kwargs.get("usemask", False)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output
<|endoftext|>"""
Histogram-related functions
"""
import contextlib
import functools
import operator
import warnings

import numpy as np
from numpy._core import overrides

__all__ = ['histogram', 'histogramdd', 'histogram_bin_edges']

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def _ptp(x):
    """Peak-to-peak value of x.

    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    return _unsigned_subtract(x.max(), x.min())


def _hist_bin_sqrt(x, range):
    """
    Square root histogram bin estimator.

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return _ptp(x) / np.sqrt(x.size)


def _hist_bin_sturges(x, range):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return _ptp(x) / (np.log2(x.size) + 1.0)


def _hist_bin_rice(x, range):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x, range):
    """
    Scott histogram bin estimator.

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return (24.0 * np.pi**0.5 / x.size)**(1.0 / 3.0) * np.std(x)


def _hist_bin_stone(x, range):
    """
    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).

    The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.
    The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.
    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule

    This paper by Stone appears to be the origination of this rule.
    https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    range : (float, float)
        The lower and upper range of the bins.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """

    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    nbins_upper_bound = max(100, int(np.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn("The number of bins estimated may be suboptimal.",
                      RuntimeWarning, stacklevel=3)
    return ptp_x / nbins


def _hist_bin_doane(x, range):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            # These three operations add up to
            # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
            # but use only one temp array instead of three
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return _ptp(x) / (1.0 + np.log2(x.size) +
                                    np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0


def _hist_bin_fd(x, range):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 0 for the bin width.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(x, range):
    """
    Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators if the FD bin width is non-zero.
    If the bin width from the FD estimator is 0, the Sturges estimator is used.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x` and bad for data with limited
    variance. The Sturges estimator is quite good for small (<1000) datasets
    and is the default in the R language. This method gives good off-the-shelf
    behaviour.

    If there is limited variance the IQR can be 0, which results in the
    FD bin width being 0 too. This is not a valid bin width, so
    ``np.histogram_bin_edges`` chooses 1 bin instead, which may not be optimal.
    If the IQR is 0, it's unlikely any variance-based estimators will be of
    use, so we revert to the Sturges estimator, which only uses the size of the
    dataset in its calculation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    del range  # unused
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        return sturges_bw

# Private dict initialized at module load time
_hist_bin_selectors = {'stone': _hist_bin_stone,
                       'auto': _hist_bin_auto,
                       'doane': _hist_bin_doane,
                       'fd': _hist_bin_fd,
                       'rice': _hist_bin_rice,
                       'scott': _hist_bin_scott,
                       'sqrt': _hist_bin_sqrt,
                       'sturges': _hist_bin_sturges}


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """
    a = np.asarray(a)

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == np.bool:
        warnings.warn("Converting input from {} to {} for compatibility."
                      .format(a.dtype, np.uint8),
                      RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong
    }
    dt = np.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned.  The input may be negative python integers so
        # ensure we pass in arrays with the initial dtype (related to NEP 50).
        return np.subtract(np.asarray(a, dtype=dt), np.asarray(b, dtype=dt),
                           casting='unsafe', dtype=unsigned_dt)


def _get_bin_edges(a, bins, range, weights):
    """
    Computes the bins used internally by `histogram`.

    Parameters
    ==========
    a : ndarray
        Ravelled data array
    bins, range
        Forwarded arguments from `histogram`.
    weights : ndarray, optional
        Ravelled weights array, or None

    Returns
    =======
    bin_edges : ndarray
        Array of bin edges
    uniform_bins : (Number, Number, int):
        The upper bound, lowerbound, and number of bins, used in the optimized
        implementation of `histogram` that works on uniform bins.
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, str):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ValueError(
                "{!r} is not a valid estimator for `bins`".format(bin_name))
        if weights is not None:
            raise TypeError("Automated estimation of the number of "
                            "bins is not supported for weighted data")

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not np.logical_and.reduce(keep):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                if np.issubdtype(a.dtype, np.integer) and width < 1:
                    width = 1
                n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError(
                '`bins` must be an integer, a string, or an array') from e
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        first_edge, last_edge = _get_outer_edges(a, range)

    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError(
                '`bins` must increase monotonically, when an array')

    else:
        raise ValueError('`bins` must be 1d, when an array')

    if n_equal_bins is not None:
        # gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = np.linspace(
            first_edge, last_edge, n_equal_bins + 1,
            endpoint=True, dtype=bin_type)
        if np.any(bin_edges[:-1] >= bin_edges[1:]):
            raise ValueError(
                f'Too many bins for data range. Cannot create {n_equal_bins} '
                f'finite-sized bins.')
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None


def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.

    In the context of a histogram, this makes the last bin edge inclusive
    """
    return np.concatenate((
        a.searchsorted(v[:-1], 'left'),
        a.searchsorted(v[-1:], 'right')
    ))


def _histogram_bin_edges_dispatcher(a, bins=None, range=None, weights=None):
    return (a, bins, weights)


@array_function_dispatch(_histogram_bin_edges_dispatcher)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    r"""
    Function to calculate only the edges of the bins used by the `histogram`
    function.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

        If `bins` is a string from the list below, `histogram_bin_edges` will
        use the method chosen to calculate the optimal bin width and
        consequently the number of bins (see the Notes section for more detail
        on the estimators) from the data that falls within the requested range.
        While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Minimum bin width between the 'sturges' and 'fd' estimators.
            Provides good all-around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that takes into account data variability
            and data size.

        'stone'
            Estimator based on leave-one-out cross-validation estimate of
            the integrated squared error. Can be regarded as a generalization
            of Scott's rule.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). This is currently not used by any of the bin estimators,
        but may be in the future.

    Returns
    -------
    bin_edges : array of dtype float
        The edges to pass into `histogram`

    See Also
    --------
    histogram

    Notes
    -----
    The methods to estimate the optimal number of bins are well founded
    in literature, and are inspired by the choices R provides for
    histogram visualisation. Note that having the number of bins
    proportional to :math:`n^{1/3}` is asymptotically optimal, which is
    why it appears in most estimators. These are simply plug-in methods
    that give good starting points for number of bins. In the equations
    below, :math:`h` is the binwidth and :math:`n_h` is the number of
    bins. All estimators that compute bin counts are recast to bin width
    using the `ptp` of the data. The final bin count is obtained from
    ``np.round(np.ceil(range / h))``. The final bin width is often less
    than what is returned by the estimators below.

    'auto' (minimum bin width of the 'sturges' and 'fd' estimators)
        A compromise to get a good value. For small datasets the Sturges
        value will usually be chosen, while larger datasets will usually
        default to FD.  Avoids the overly conservative behaviour of FD
        and Sturges for small and large datasets respectively.
        Switchover point is usually :math:`a.size \approx 1000`.

    'fd' (Freedman Diaconis Estimator)
        .. math:: h = 2 \frac{IQR}{n^{1/3}}

        The binwidth is proportional to the interquartile range (IQR)
        and inversely proportional to cube root of a.size. Can be too
        conservative for small datasets, but is quite good for large
        datasets. The IQR is very robust to outliers.

    'scott'
        .. math:: h = \sigma \sqrt[3]{\frac{24 \sqrt{\pi}}{n}}

        The binwidth is proportional to the standard deviation of the
        data and inversely proportional to cube root of ``x.size``. Can
        be too conservative for small datasets, but is quite good for
        large datasets. The standard deviation is not very robust to
        outliers. Values are very similar to the Freedman-Diaconis
        estimator in the absence of outliers.

    'rice'
        .. math:: n_h = 2n^{1/3}

        The number of bins is only proportional to cube root of
        ``a.size``. It tends to overestimate the number of bins and it
        does not take into account data variability.

    'sturges'
        .. math:: n_h = \log _{2}(n) + 1

        The number of bins is the base 2 log of ``a.size``.  This
        estimator assumes normality of data and is too conservative for
        larger, non-normal datasets. This is the default method in R's
        ``hist`` method.

    'doane'
        .. math:: n_h = 1 + \log_{2}(n) +
                        \log_{2}\left(1 + \frac{|g_1|}{\sigma_{g_1}}\right)

            g_1 = mean\left[\left(\frac{x - \mu}{\sigma}\right)^3\right]

            \sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}

        An improved version of Sturges' formula that produces better
        estimates for non-normal datasets. This estimator attempts to
        account for the skew of the data.

    'sqrt'
        .. math:: n_h = \sqrt n

        The simplest and fastest estimator. Only takes into account the
        data size.

    Additionally, if the data is of integer dtype, then the binwidth will never
    be less than 1.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    >>> np.histogram_bin_edges(arr, bins='auto', range=(0, 1))
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> np.histogram_bin_edges(arr, bins=2)
    array([0. , 2.5, 5. ])

    For consistency with histogram, an array of pre-computed bins is
    passed through unmodified:

    >>> np.histogram_bin_edges(arr, [1, 2])
    array([1, 2])

    This function allows one set of bins to be computed, and reused across
    multiple histograms:

    >>> shared_bins = np.histogram_bin_edges(arr, bins='auto')
    >>> shared_bins
    array([0., 1., 2., 3., 4., 5.])

    >>> group_id = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> hist_0, _ = np.histogram(arr[group_id == 0], bins=shared_bins)
    >>> hist_1, _ = np.histogram(arr[group_id == 1], bins=shared_bins)

    >>> hist_0; hist_1
    array([1, 1, 0, 1, 0])
    array([2, 0, 1, 1, 2])

    Which gives more easily comparable results than using separate bins for
    each histogram:

    >>> hist_0, bins_0 = np.histogram(arr[group_id == 0], bins='auto')
    >>> hist_1, bins_1 = np.histogram(arr[group_id == 1], bins='auto')
    >>> hist_0; hist_1
    array([1, 1, 1])
    array([2, 1, 1, 2])
    >>> bins_0; bins_1
    array([0., 1., 2., 3.])
    array([0.  , 1.25, 2.5 , 3.75, 5.  ])

    """
    a, weights = _ravel_and_check_weights(a, weights)
    bin_edges, _ = _get_bin_edges(a, bins, range, weights)
    return bin_edges


def _histogram_dispatcher(
        a, bins=None, range=None, density=None, weights=None):
    return (a, bins, weights)


@array_function_dispatch(_histogram_dispatcher)
def histogram(a, bins=10, range=None, density=None, weights=None):
    r"""
    Compute the histogram of a dataset.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
        Please note that the ``dtype`` of `weights` will also become the
        ``dtype`` of the returned accumulator (`hist`), so it must be
        large enough to hold accumulated values as well.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.  If `weights` are given,
        ``hist.dtype`` will be taken from `weights`.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words,
    if `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
    the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
    *includes* 4.


    Examples
    --------
    >>> import numpy as np
    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> np.sum(hist * np.diff(bin_edges))
    1.0

    Automated Bin Selection Methods example, using 2 peak random data
    with 2000 points.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np

        rng = np.random.RandomState(10)  # deterministic random data
        a = np.hstack((rng.normal(size=1000),
                       rng.normal(loc=5, scale=2, size=1000)))
        plt.hist(a, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    """
    a, weights = _ravel_and_check_weights(a, weights)

    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = np.dtype(np.intp)
    else:
        ntype = weights.dtype

    # We set a block size, as this allows us to iterate over chunks when
    # computing histograms, to minimize memory usage.
    BLOCK = 65536

    # The fast path uses bincount, but that only works for certain types
    # of weight
    simple_weights = (
        weights is None or
        np.can_cast(weights.dtype, np.double) or
        np.can_cast(weights.dtype, complex)
    )

    if uniform_bins is not None and simple_weights:
        # Fast algorithm for equal bins
        # We now convert values of a to bin indices, under the assumption of
        # equal bin widths (which is valid here).
        first_edge, last_edge, n_equal_bins = uniform_bins

        # Initialize empty histogram
        n = np.zeros(n_equal_bins, ntype)

        # Pre-compute histogram scaling factor
        norm_numerator = n_equal_bins
        norm_denom = _unsigned_subtract(last_edge, first_edge)

        # We iterate over blocks here for two reasons: the first is that for
        # large arrays, it is actually faster (for example for a 10^8 array it
        # is 2x as fast) and it results in a memory footprint 3x lower in the
        # limit of large arrays.
        for i in _range(0, len(a), BLOCK):
            tmp_a = a[i:i+BLOCK]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i:i + BLOCK]

            # Only include values in the right range
            keep = (tmp_a >= first_edge)
            keep &= (tmp_a <= last_edge)
            if not np.logical_and.reduce(keep):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]

            # This cast ensures no type promotions occur below, which gh-10322
            # make unpredictable. Getting it wrong leads to precision errors
            # like gh-8123.
            tmp_a = tmp_a.astype(bin_edges.dtype, copy=False)

            # Compute the bin indices, and for values that lie exactly on
            # last_edge we need to subtract one
            f_indices = ((_unsigned_subtract(tmp_a, first_edge) / norm_denom)
                         * norm_numerator)
            indices = f_indices.astype(np.intp)
            indices[indices == n_equal_bins] -= 1

            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = ((tmp_a >= bin_edges[indices + 1])
                         & (indices != n_equal_bins - 1))
            indices[increment] += 1

            # We now compute the histogram using bincount
            if ntype.kind == 'c':
                n.real += np.bincount(indices, weights=tmp_w.real,
                                      minlength=n_equal_bins)
                n.imag += np.bincount(indices, weights=tmp_w.imag,
                                      minlength=n_equal_bins)
            else:
                n += np.bincount(indices, weights=tmp_w,
                                 minlength=n_equal_bins).astype(ntype)
    else:
        # Compute via cumulative histogram
        cum_n = np.zeros(bin_edges.shape, ntype)
        if weights is None:
            for i in _range(0, len(a), BLOCK):
                sa = np.sort(a[i:i+BLOCK])
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = np.zeros(1, dtype=ntype)
            for i in _range(0, len(a), BLOCK):
                tmp_a = a[i:i+BLOCK]
                tmp_w = weights[i:i+BLOCK]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = np.concatenate((zero, sw.cumsum()))
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]

        n = np.diff(cum_n)

    if density:
        db = np.array(np.diff(bin_edges), float)
        return n/db/n.sum(), bin_edges

    return n, bin_edges


def _histogramdd_dispatcher(sample, bins=None, range=None, density=None,
                            weights=None):
    if hasattr(sample, 'shape'):  # same condition as used in histogramdd
        yield sample
    else:
        yield from sample
    with contextlib.suppress(TypeError):
        yield from bins
    yield weights


@array_function_dispatch(_histogramdd_dispatcher)
def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : (N, D) array, or (N, D) array_like
        The data to be histogrammed.

        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z))``.

        The first form should be preferred.

    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if density is True. If density is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See density and weights
        for the different possible semantics.
    edges : tuple of ndarrays
        A tuple of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> r = rng.normal(size=(100,3))
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, np.intp)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                'sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]

    # normalize the range argument
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:,i], range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                	"`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                    .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = np.bincount(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D*(slice(1, -1),)
    hist = hist[core]

    if density:
        # calculate the probability density function
        s = hist.sum()
        for i in _range(D):
            shape = np.ones(D, int)
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist, edges
<|endoftext|>import functools
import sys
import math
import warnings

import numpy as np
from .._utils import set_module
import numpy._core.numeric as _nx
from numpy._core.numeric import ScalarType, array
from numpy._core.numerictypes import issubdtype

import numpy.matrixlib as matrixlib
from numpy._core.multiarray import ravel_multi_index, unravel_index
from numpy._core import overrides, linspace
from numpy.lib.stride_tricks import as_strided
from numpy.lib._function_base_impl import diff


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_',
    's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal',
    'diag_indices', 'diag_indices_from'
]


def _ix__dispatcher(*args):
    return args


@array_function_dispatch(_ix__dispatcher)
def ix_(*args):
    """
    Construct an open mesh from multiple sequences.

    This function takes N 1-D sequences and returns N outputs with N
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.

    Using `ix_` one can quickly construct index arrays that will index
    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Parameters
    ----------
    args : 1-D sequences
        Each sequence should be of integer or boolean type.
        Boolean sequences will be interpreted as boolean masks for the
        corresponding dimension (equivalent to passing in
        ``np.nonzero(boolean_sequence)``).

    Returns
    -------
    out : tuple of ndarrays
        N arrays with N dimensions each, with N the number of input
        sequences. Together these arrays form an open mesh.

    See Also
    --------
    ogrid, mgrid, meshgrid

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10).reshape(2, 5)
    >>> a
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> ixgrid = np.ix_([0, 1], [2, 4])
    >>> ixgrid
    (array([[0],
           [1]]), array([[2, 4]]))
    >>> ixgrid[0].shape, ixgrid[1].shape
    ((2, 1), (1, 2))
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    >>> ixgrid = np.ix_([True, True], [2, 4])
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])
    >>> ixgrid = np.ix_([True, True], [False, False, True, False, True])
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if not isinstance(new, _nx.ndarray):
            new = np.asarray(new)
            if new.size == 0:
                # Explicitly type empty arrays to avoid float default
                new = new.astype(_nx.intp)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if issubdtype(new.dtype, _nx.bool):
            new, = new.nonzero()
        new = new.reshape((1,)*k + (new.size,) + (1,)*(nd-k-1))
        out.append(new)
    return tuple(out)


class nd_grid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.

    Notes
    -----
    Two instances of `nd_grid` are made available in the NumPy namespace,
    `mgrid` and `ogrid`, approximately defined as::

        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)

    Users should use these pre-defined instances instead of using `nd_grid`
    directly.
    """
    __slots__ = ('sparse',)

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        try:
            size = []
            # Mimic the behavior of `np.arange` and use a data type
            # which is at least as large as `np.int_`
            num_list = [0]
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = abs(step)
                    size.append(int(step))
                else:
                    size.append(
                        int(math.ceil((stop - start) / (step*1.0))))
                num_list += [start, stop, step]
            typ = _nx.result_type(*num_list)
            if self.sparse:
                nn = [_nx.arange(_x, dtype=_t)
                      for _x, _t in zip(size, (typ,)*len(size))]
            else:
                nn = _nx.indices(size, typ)
            for k, kk in enumerate(key):
                step = kk.step
                start = kk.start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = int(abs(step))
                    if step != 1:
                        step = (kk.stop - start) / float(step - 1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [_nx.newaxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None, None)
                    nn[k] = nn[k][tuple(slobj)]
                    slobj[k] = _nx.newaxis
                return tuple(nn)  # ogrid -> tuple of arrays
            return nn  # mgrid -> ndarray
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, (_nx.complexfloating, complex)):
                # Prevent the (potential) creation of integer arrays
                step_float = abs(step)
                step = length = int(step_float)
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                typ = _nx.result_type(start, stop, step_float)
                return _nx.arange(0, length, 1, dtype=typ)*step + start
            else:
                return _nx.arange(start, stop, step)


class MGridClass(nd_grid):
    """
    An instance which returns a dense multi-dimensional "meshgrid".

    An instance which returns a dense (or fleshed out) mesh-grid
    when indexed, so that each returned argument has the same shape.
    The dimensions and number of the output arrays are equal to the
    number of indexing dimensions.  If the step length is not a complex
    number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid : ndarray
        A single array, containing a set of `ndarray`\\ s all of the same
        dimensions. stacked along the first axis.

    See Also
    --------
    ogrid : like `mgrid` but returns open (not fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
