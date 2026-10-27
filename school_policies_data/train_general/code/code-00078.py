        For each element in `self`, return a titlecased version of the
        string: words start with uppercase characters, all remaining cased
        characters are lowercase.

        See Also
        --------
        char.title

        """
        return asarray(title(self))

    def translate(self, table, deletechars=None):
        """
        For each element in `self`, return a copy of the string where
        all characters occurring in the optional argument
        `deletechars` are removed, and the remaining characters have
        been mapped through the given translation table.

        See Also
        --------
        char.translate

        """
        return asarray(translate(self, table, deletechars))

    def upper(self):
        """
        Return an array with the elements of `self` converted to
        uppercase.

        See Also
        --------
        char.upper

        """
        return asarray(upper(self))

    def zfill(self, width):
        """
        Return the numeric string left-filled with zeros in a string of
        length `width`.

        See Also
        --------
        char.zfill

        """
        return asarray(zfill(self, width))

    def isnumeric(self):
        """
        For each element in `self`, return True if there are only
        numeric characters in the element.

        See Also
        --------
        char.isnumeric

        """
        return isnumeric(self)

    def isdecimal(self):
        """
        For each element in `self`, return True if there are only
        decimal characters in the element.

        See Also
        --------
        char.isdecimal

        """
        return isdecimal(self)


@set_module("numpy.char")
def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """
    Create a `~numpy.char.chararray`.

    .. note::
       This class is provided for numarray backward-compatibility.
       New code (not concerned with numarray compatibility) should use
       arrays of type `bytes_` or `str_` and use the free functions
       in :mod:`numpy.char` for fast vectorized string operations instead.

    Versus a NumPy array of dtype `bytes_` or `str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
       and infix operators (e.g. ``+, *, %``)

    Parameters
    ----------
    obj : array of str or unicode-like

    itemsize : int, optional
        `itemsize` is the number of characters per scalar in the
        resulting array.  If `itemsize` is None, and `obj` is an
        object array or a Python list, the `itemsize` will be
        automatically determined.  If `itemsize` is provided and `obj`
        is of type str or unicode, then the `obj` string will be
        chunked into `itemsize` pieces.

    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if ``__array__`` returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`itemsize`, unicode, `order`, etc.).

    unicode : bool, optional
        When true, the resulting `~numpy.char.chararray` can contain Unicode
        characters, when false only 8-bit characters.  If unicode is
        None and `obj` is one of the following:

        - a `~numpy.char.chararray`,
        - an ndarray of type :class:`str_` or :class:`bytes_`
        - a Python :class:`str` or :class:`bytes` object,

        then the unicode setting of the output array will be
        automatically determined.

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    
    Examples
    --------

    >>> import numpy as np
    >>> char_array = np.char.array(['hello', 'world', 'numpy','array'])
    >>> char_array
    chararray(['hello', 'world', 'numpy', 'array'], dtype='<U5')

    """
    if isinstance(obj, (bytes, str)):
        if unicode is None:
            if isinstance(obj, str):
                unicode = True
            else:
                unicode = False

        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) // itemsize

        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj, order=order)

    if isinstance(obj, (list, tuple)):
        obj = asnarray(obj)

    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
        # If we just have a vanilla chararray, create a chararray
        # view around it.
        if not isinstance(obj, chararray):
            obj = obj.view(chararray)

        if itemsize is None:
            itemsize = obj.itemsize
            # itemsize is in 8-bit chars, so for Unicode, we need
            # to divide by the size of a single Unicode character,
            # which for NumPy is always 4
            if issubclass(obj.dtype.type, str_):
                itemsize //= 4

        if unicode is None:
            if issubclass(obj.dtype.type, str_):
                unicode = True
            else:
                unicode = False

        if unicode:
            dtype = str_
        else:
            dtype = bytes_

        if order is not None:
            obj = asnarray(obj, order=order)
        if (copy or
                (itemsize != obj.itemsize) or
                (not unicode and isinstance(obj, str_)) or
                (unicode and isinstance(obj, bytes_))):
            obj = obj.astype((dtype, int(itemsize)))
        return obj

    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
        if itemsize is None:
            # Since no itemsize was specified, convert the input array to
            # a list so the ndarray constructor will automatically
            # determine the itemsize for us.
            obj = obj.tolist()
            # Fall through to the default case

    if unicode:
        dtype = str_
    else:
        dtype = bytes_

    if itemsize is None:
        val = narray(obj, dtype=dtype, order=order, subok=True)
    else:
        val = narray(obj, dtype=(dtype, itemsize), order=order, subok=True)
    return val.view(chararray)


@set_module("numpy.char")
def asarray(obj, itemsize=None, unicode=None, order=None):
    """
    Convert the input to a `~numpy.char.chararray`, copying the data only if
    necessary.

    Versus a NumPy array of dtype `bytes_` or `str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
       and infix operators (e.g. ``+``, ``*``, ``%``)

    Parameters
    ----------
    obj : array of str or unicode-like

    itemsize : int, optional
        `itemsize` is the number of characters per scalar in the
        resulting array.  If `itemsize` is None, and `obj` is an
        object array or a Python list, the `itemsize` will be
        automatically determined.  If `itemsize` is provided and `obj`
        is of type str or unicode, then the `obj` string will be
        chunked into `itemsize` pieces.

    unicode : bool, optional
        When true, the resulting `~numpy.char.chararray` can contain Unicode
        characters, when false only 8-bit characters.  If unicode is
        None and `obj` is one of the following:

        - a `~numpy.char.chararray`,
        - an ndarray of type `str_` or `unicode_`
        - a Python str or unicode object,

        then the unicode setting of the output array will be
        automatically determined.

    order : {'C', 'F'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).

    Examples
    --------
    >>> import numpy as np
    >>> np.char.asarray(['hello', 'world'])
    chararray(['hello', 'world'], dtype='<U5')

    """
    return array(obj, itemsize, copy=False,
                 unicode=unicode, order=order)
<|endoftext|>import functools
import itertools
import operator
import sys
import warnings
import numbers
import builtins
import math

import numpy as np
from . import multiarray
from . import numerictypes as nt
from .multiarray import (
    ALLOW_THREADS, BUFSIZE, CLIP, MAXDIMS, MAY_SHARE_BOUNDS, MAY_SHARE_EXACT,
    RAISE, WRAP, arange, array, asarray, asanyarray, ascontiguousarray,
    asfortranarray, broadcast, can_cast, concatenate, copyto, dot, dtype,
    empty, empty_like, flatiter, frombuffer, from_dlpack, fromfile, fromiter,
    fromstring, inner, lexsort, matmul, may_share_memory, min_scalar_type,
    ndarray, nditer, nested_iters, promote_types, putmask, result_type,
    shares_memory, vdot, where, zeros, normalize_axis_index, vecdot
)

from . import overrides
from . import umath
from . import shape_base
from .overrides import finalize_array_function_like, set_module
from .umath import (multiply, invert, sin, PINF, NAN)
from . import numerictypes
from ..exceptions import AxisError
from ._ufunc_config import errstate

bitwise_not = invert
ufunc = type(sin)
newaxis = None

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'newaxis', 'ndarray', 'flatiter', 'nditer', 'nested_iters', 'ufunc',
    'arange', 'array', 'asarray', 'asanyarray', 'ascontiguousarray',
    'asfortranarray', 'zeros', 'count_nonzero', 'empty', 'broadcast', 'dtype',
    'fromstring', 'fromfile', 'frombuffer', 'from_dlpack', 'where',
    'argwhere', 'copyto', 'concatenate', 'lexsort', 'astype',
    'can_cast', 'promote_types', 'min_scalar_type',
    'result_type', 'isfortran', 'empty_like', 'zeros_like', 'ones_like',
    'correlate', 'convolve', 'inner', 'dot', 'outer', 'vdot', 'roll',
    'rollaxis', 'moveaxis', 'cross', 'tensordot', 'little_endian',
    'fromiter', 'array_equal', 'array_equiv', 'indices', 'fromfunction',
    'isclose', 'isscalar', 'binary_repr', 'base_repr', 'ones',
    'identity', 'allclose', 'putmask',
    'flatnonzero', 'inf', 'nan', 'False_', 'True_', 'bitwise_not',
    'full', 'full_like', 'matmul', 'vecdot', 'shares_memory',
    'may_share_memory']


def _zeros_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


@array_function_dispatch(_zeros_like_dispatcher)
def zeros_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.zeros_like(y)
    array([0.,  0.,  0.])

    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    # needed instead of a 0 to get same result as zeros for string dtypes
    z = zeros(1, dtype=res.dtype)
    multiarray.copyto(res, z, casting='unsafe')
    return res


@finalize_array_function_like
@set_module('numpy')
def ones(shape, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])

    >>> np.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[1.],
           [1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])

    """
    if like is not None:
        return _ones_with_like(
            like, shape, dtype=dtype, order=order, device=device
        )

    a = empty(shape, dtype, order, device=device)
    multiarray.copyto(a, 1, casting='unsafe')
    return a


_ones_with_like = array_function_dispatch()(ones)


def _ones_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


@array_function_dispatch(_ones_like_dispatcher)
def ones_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    ones : Return a new array setting values to one.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.ones_like(y)
    array([1.,  1.,  1.])

    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    multiarray.copyto(res, 1, casting='unsafe')
    return res


def _full_dispatcher(
    shape, fill_value, dtype=None, order=None, *, device=None, like=None
):
    return(like,)


@finalize_array_function_like
@set_module('numpy')
def full(shape, fill_value, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
         ``np.array(fill_value).dtype``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> import numpy as np
    >>> np.full((2, 2), np.inf)
    array([[inf, inf],
           [inf, inf]])
    >>> np.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])

    >>> np.full((2, 2), [1, 2])
    array([[1, 2],
           [1, 2]])

    """
    if like is not None:
        return _full_with_like(
            like, shape, fill_value, dtype=dtype, order=order, device=device
        )

    if dtype is None:
        fill_value = asarray(fill_value)
        dtype = fill_value.dtype
    a = empty(shape, dtype, order, device=device)
    multiarray.copyto(a, fill_value, casting='unsafe')
    return a


_full_with_like = array_function_dispatch()(full)


def _full_like_dispatcher(
    a, fill_value, dtype=None, order=None, subok=None, shape=None,
    *, device=None
):
    return (a,)


@array_function_dispatch(_full_like_dispatcher)
def full_like(
    a, fill_value, dtype=None, order='K', subok=True, shape=None,
    *, device=None
):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : array_like
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6, dtype=int)
    >>> np.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1])
    >>> np.full_like(x, 0.1)
    array([0, 0, 0, 0, 0, 0])
    >>> np.full_like(x, 0.1, dtype=np.double)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    >>> np.full_like(x, np.nan, dtype=np.double)
    array([nan, nan, nan, nan, nan, nan])

    >>> y = np.arange(6, dtype=np.double)
    >>> np.full_like(y, 0.1)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    >>> y = np.zeros([2, 2, 3], dtype=int)
    >>> np.full_like(y, [0, 0, 255])
    array([[[  0,   0, 255],
            [  0,   0, 255]],
           [[  0,   0, 255],
            [  0,   0, 255]]])
    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    multiarray.copyto(res, fill_value, casting='unsafe')
    return res


def _count_nonzero_dispatcher(a, axis=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_count_nonzero_dispatcher)
def count_nonzero(a, axis=None, *, keepdims=False):
    """
    Counts the number of non-zero values in the array ``a``.

    The word "non-zero" is in reference to the Python 2.x
    built-in method ``__nonzero__()`` (renamed ``__bool__()``
    in Python 3.x) of Python objects that tests an object's
    "truthfulness". For example, any number is considered
    truthful if it is nonzero, whereas any string is considered
    truthful if it is not the empty string. Thus, this function
    (recursively) counts how many elements in ``a`` (and in
    sub-arrays thereof) have their ``__nonzero__()`` or ``__bool__()``
    method evaluated to ``True``.

    Parameters
    ----------
    a : array_like
        The array for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of ``a``.
    keepdims : bool, optional
        If this is set to True, the axes that are counted are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    count : int or array of int
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    nonzero : Return the coordinates of all the non-zero values.

    Examples
    --------
    >>> import numpy as np
    >>> np.count_nonzero(np.eye(4))
    4
    >>> a = np.array([[0, 1, 7, 0],
    ...               [3, 0, 2, 19]])
    >>> np.count_nonzero(a)
    5
    >>> np.count_nonzero(a, axis=0)
    array([1, 1, 2, 1])
    >>> np.count_nonzero(a, axis=1)
    array([2, 3])
    >>> np.count_nonzero(a, axis=1, keepdims=True)
    array([[2],
           [3]])
    """
    if axis is None and not keepdims:
        return multiarray.count_nonzero(a)

    a = asanyarray(a)

    # TODO: this works around .astype(bool) not working properly (gh-9847)
    if np.issubdtype(a.dtype, np.character):
        a_bool = a != a.dtype.type()
    else:
        a_bool = a.astype(np.bool, copy=False)

    return a_bool.sum(axis=axis, dtype=np.intp, keepdims=keepdims)


@set_module('numpy')
def isfortran(a):
    """
    Check if the array is Fortran contiguous but *not* C contiguous.

    This function is obsolete. If you only want to check if an array is Fortran
    contiguous use ``a.flags.f_contiguous`` instead.

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------
    isfortran : bool
        Returns True if the array is Fortran contiguous but *not* C contiguous.


    Examples
    --------

    np.array allows to specify whether the array is written in C-contiguous
    order (last index varies the fastest), or FORTRAN-contiguous order in
    memory (first index varies the fastest).

    >>> import numpy as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False

    >>> b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(b)
    True


    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.isfortran(b)
    True

    C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

    >>> np.isfortran(np.array([1, 2], order='F'))
    False

    """
    return a.flags.fnc


def _argwhere_dispatcher(a):
    return (a,)


@array_function_dispatch(_argwhere_dispatcher)
def argwhere(a):
    """
    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : (N, a.ndim) ndarray
        Indices of elements that are non-zero. Indices are grouped by element.
        This array will have shape ``(N, a.ndim)`` where ``N`` is the number of
        non-zero items.

    See Also
    --------
    where, nonzero

    Notes
    -----
    ``np.argwhere(a)`` is almost the same as ``np.transpose(np.nonzero(a))``,
    but produces a result of the correct shape for a 0D array.

    The output of ``argwhere`` is not suitable for indexing arrays.
    For this purpose use ``nonzero(a)`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2,3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argwhere(x>1)
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    # nonzero does not behave well on 0d, so promote to 1d
    if np.ndim(a) == 0:
        a = shape_base.atleast_1d(a)
        # then remove the added dimension
        return argwhere(a)[:, :0]
    return transpose(nonzero(a))


def _flatnonzero_dispatcher(a):
    return (a,)


@array_function_dispatch(_flatnonzero_dispatcher)
def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to ``np.nonzero(np.ravel(a))[0]``.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of ``a.ravel()``
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    ravel : Return a 1-D array containing the elements of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(-2, 3)
    >>> x
    array([-2, -1,  0,  1,  2])
    >>> np.flatnonzero(x)
    array([0, 1, 3, 4])

    Use the indices of the non-zero elements as an index array to extract
    these elements:

    >>> x.ravel()[np.flatnonzero(x)]
    array([-2, -1,  1,  2])

    """
    return np.nonzero(np.ravel(a))[0]


def _correlate_dispatcher(a, v, mode=None):
    return (a, v)


@array_function_dispatch(_correlate_dispatcher)
def correlate(a, v, mode='valid'):
    r"""
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts [1]_:

    .. math:: c_k = \sum_n a_{n+k} \cdot \overline{v}_n

    with a and v sequences being zero-padded where necessary and
    :math:`\overline v` denoting complex conjugation.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `convolve` docstring.  Note that the default
        is 'valid', unlike `convolve`, which uses 'full'.

    Returns
    -------
    out : ndarray
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    convolve : Discrete, linear convolution of two one-dimensional sequences.
    scipy.signal.correlate : uses FFT which has superior performance
        on large arrays.

    Notes
    -----
    The definition of correlation above is not unique and sometimes
    correlation may be defined differently. Another common definition is [1]_:

    .. math:: c'_k = \sum_n a_{n} \cdot \overline{v_{n+k}}

    which is related to :math:`c_k` by :math:`c'_k = c_{-k}`.

    `numpy.correlate` may perform slowly in large arrays (i.e. n = 1e5)
    because it does not use the FFT to compute the convolution; in that case,
    `scipy.signal.correlate` might be preferable.

    References
    ----------
    .. [1] Wikipedia, "Cross-correlation",
           https://en.wikipedia.org/wiki/Cross-correlation

    Examples
    --------
    >>> import numpy as np
    >>> np.correlate([1, 2, 3], [0, 1, 0.5])
    array([3.5])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "same")
    array([2. ,  3.5,  3. ])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
    array([0.5,  2. ,  3.5,  3. ,  0. ])

    Using complex sequences:

    >>> np.correlate([1+1j, 2, 3-1j], [0, 1, 0.5j], 'full')
    array([ 0.5-0.5j,  1.0+0.j ,  1.5-1.5j,  3.0-1.j ,  0.0+0.j ])

    Note that you get the time reversed, complex conjugated result
    (:math:`\overline{c_{-k}}`) when the two input sequences a and v change
    places:

    >>> np.correlate([0, 1, 0.5j], [1+1j, 2, 3-1j], 'full')
    array([ 0.0+0.j ,  3.0+1.j ,  1.5+1.5j,  1.0+0.j ,  0.5+0.5j])

    """
    return multiarray.correlate2(a, v, mode)


def _convolve_dispatcher(a, v, mode=None):
    return (a, v)


@array_function_dispatch(_convolve_dispatcher)
def convolve(a, v, mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    The convolution operator is often seen in signal processing, where it
    models the effect of a linear time-invariant system on a signal [1]_.  In
    probability theory, the sum of two independent random variables is
    distributed according to the convolution of their individual
    distributions.

    If `v` is longer than `a`, the arrays are swapped before computation.

    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array.
    v : (M,) array_like
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    scipy.signal.fftconvolve : Convolve two arrays using the Fast Fourier
                               Transform.
    scipy.linalg.toeplitz : Used to construct the convolution operator.
    polymul : Polynomial multiplication. Same output as convolve, but also
              accepts poly1d objects as input.

    Notes
    -----
    The discrete convolution operation is defined as

    .. math:: (a * v)_n = \\sum_{m = -\\infty}^{\\infty} a_m v_{n - m}

    It can be shown that a convolution :math:`x(t) * y(t)` in time/space
    is equivalent to the multiplication :math:`X(f) Y(f)` in the Fourier
    domain, after appropriate padding (padding is necessary to prevent
    circular convolution).  Since multiplication is more efficient (faster)
    than convolution, the function `scipy.signal.fftconvolve` exploits the
    FFT to calculate the convolution of large data-sets.

    References
    ----------
    .. [1] Wikipedia, "Convolution",
        https://en.wikipedia.org/wiki/Convolution

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> import numpy as np
    >>> np.convolve([1, 2, 3], [0, 1, 0.5])
    array([0. , 1. , 2.5, 4. , 1.5])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:

    >>> np.convolve([1,2,3],[0,1,0.5], 'same')
    array([1. ,  2.5,  4. ])

    The two arrays are of the same length, so there
    is only one position where they completely overlap:

    >>> np.convolve([1,2,3],[0,1,0.5], 'valid')
    array([2.5])

    """
    a, v = array(a, copy=None, ndmin=1), array(v, copy=None, ndmin=1)
    if (len(v) > len(a)):
        a, v = v, a
    if len(a) == 0:
        raise ValueError('a cannot be empty')
    if len(v) == 0:
        raise ValueError('v cannot be empty')
    return multiarray.correlate(a, v[::-1], mode)


def _outer_dispatcher(a, b, out=None):
    return (a, b, out)


@array_function_dispatch(_outer_dispatcher)
def outer(a, b, out=None):
    """
    Compute the outer product of two vectors.

    Given two vectors `a` and `b` of length ``M`` and ``N``, respectively,
    the outer product [1]_ is::

      [[a_0*b_0  a_0*b_1 ... a_0*b_{N-1} ]
       [a_1*b_0    .
       [ ...          .
       [a_{M-1}*b_0            a_{M-1}*b_{N-1} ]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``

    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    linalg.outer : An Array API compatible variation of ``np.outer``,
                   which accepts 1-dimensional inputs only.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
           ed., Baltimore, MD, Johns Hopkins University Press, 1996,
           pg. 8.

    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> import numpy as np
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

    An example using a "vector" of letters:

    >>> x = np.array(['a', 'b', 'c'], dtype=object)
    >>> np.outer(x, [1, 2, 3])
    array([['a', 'aa', 'aaa'],
           ['b', 'bb', 'bbb'],
           ['c', 'cc', 'ccc']], dtype=object)

    """
    a = asarray(a)
    b = asarray(b)
    return multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)


def _tensordot_dispatcher(a, b, axes=None):
    return (a, b)


@array_function_dispatch(_tensordot_dispatcher)
def tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : ndarray
        The tensor dot product of the input.

    See Also
    --------
    dot, einsum

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer_like, the sequence of axes for evaluation
    will be: from the -Nth axis to the -1th axis in `a`,
    and from the 0th axis to (N-1)th axis in `b`.
    For example, ``axes = 2`` is the equal to
    ``axes = [[-2, -1], [0, 1]]``.
    When N-1 is smaller than 0, or when -N is larger than -1,
    the element of `a` and `b` are defined as the `axes`.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    The calculation can be referred to ``numpy.einsum``.

    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.

    Examples
    -------- 
    An example on integer_like:

    >>> a_0 = np.array([[1, 2], [3, 4]])
    >>> b_0 = np.array([[5, 6], [7, 8]])
    >>> c_0 = np.tensordot(a_0, b_0, axes=0)
    >>> c_0.shape
    (2, 2, 2, 2)
    >>> c_0
    array([[[[ 5,  6],
             [ 7,  8]],
            [[10, 12],
             [14, 16]]],
           [[[15, 18],
             [21, 24]],
            [[20, 24],
             [28, 32]]]])

    An example on array_like:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
           
    A slower but equivalent way of computing the same...
    
    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += a[k,n,i] * b[n,k,j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])

    An extended example taking advantage of the overloading of + and \\*:

    >>> a = np.array(range(1, 9))
    >>> a.shape = (2, 2, 2)
    >>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    >>> a; A
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([['a', 'b'],
           ['c', 'd']], dtype=object)

    >>> np.tensordot(a, A) # third argument default is 2 for double-contraction
    array(['abbcccdddd', 'aaaaabbbbbbcccccccdddddddd'], dtype=object)

    >>> np.tensordot(a, A, 1)
    array([[['acc', 'bdd'],
            ['aaacccc', 'bbbdddd']],
           [['aaaaacccccc', 'bbbbbdddddd'],
            ['aaaaaaacccccccc', 'bbbbbbbdddddddd']]], dtype=object)

    >>> np.tensordot(a, A, 0) # tensor product (result too long to incl.)
    array([[[[['a', 'b'],
              ['c', 'd']],
              ...

    >>> np.tensordot(a, A, (0, 1))
    array([[['abbbbb', 'cddddd'],
            ['aabbbbbb', 'ccdddddd']],
           [['aaabbbbbbb', 'cccddddddd'],
            ['aaaabbbbbbbb', 'ccccdddddddd']]], dtype=object)

    >>> np.tensordot(a, A, (2, 1))
    array([[['abb', 'cdd'],
            ['aaabbbb', 'cccdddd']],
           [['aaaaabbbbbb', 'cccccdddddd'],
            ['aaaaaaabbbbbbbb', 'cccccccdddddddd']]], dtype=object)

    >>> np.tensordot(a, A, ((0, 1), (0, 1)))
    array(['abbbcccccddddddd', 'aabbbbccccccdddddddd'], dtype=object)

    >>> np.tensordot(a, A, ((2, 1), (1, 0)))
    array(['acccbbdddd', 'aaaaacccccccbbbbbbdddddddd'], dtype=object)

    """
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = asarray(a), asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = math.prod(as_[axis] for axis in axes_a)
    newshape_a = (math.prod([as_[ax] for ax in notin]), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = math.prod(bs[axis] for axis in axes_b)
    newshape_b = (N2, math.prod([bs[ax] for ax in notin]))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dot(at, bt)
    return res.reshape(olda + oldb)


def _roll_dispatcher(a, shift, axis=None):
    return (a,)


@array_function_dispatch(_roll_dispatcher)
def roll(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Notes
    -----
    Supports rolling over multiple dimensions simultaneously.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> np.roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> np.roll(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x2 = np.reshape(x, (2, 5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> np.roll(x2, 1)
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> np.roll(x2, -1)
    array([[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 0]])
    >>> np.roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, 1, axis=1)
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    >>> np.roll(x2, -1, axis=1)
    array([[1, 2, 3, 4, 0],
           [6, 7, 8, 9, 5]])
    >>> np.roll(x2, (1, 1), axis=(1, 0))
    array([[9, 5, 6, 7, 8],
           [4, 0, 1, 2, 3]])
    >>> np.roll(x2, (2, 1), axis=(1, 0))
    array([[8, 9, 5, 6, 7],
           [3, 4, 0, 1, 2]])

    """
    a = asanyarray(a)
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)

    else:
        axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        broadcasted = broadcast(shift, axis)
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences")
        shifts = {ax: 0 for ax in range(a.ndim)}
        for sh, ax in broadcasted:
            shifts[ax] += int(sh)

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            if offset:
                # (original, result), (original, result)
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        result = empty_like(a)
        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result


def _rollaxis_dispatcher(a, axis, start=None):
    return (a,)


@array_function_dispatch(_rollaxis_dispatcher)
def rollaxis(a, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    This function continues to be supported for backward compatibility, but you
    should prefer `moveaxis`. The `moveaxis` function was added in NumPy
    1.11.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        The axis to be rolled. The positions of the other axes do not
        change relative to one another.
    start : int, optional
        When ``start <= axis``, the axis is rolled back until it lies in
        this position. When ``start > axis``, the axis is rolled until it
        lies before this position. The default, 0, results in a "complete"
        roll. The following table describes how negative values of ``start``
        are interpreted:

        .. table::
           :align: left

           +-------------------+----------------------+
           |     ``start``     | Normalized ``start`` |
           +===================+======================+
           | ``-(arr.ndim+1)`` | raise ``AxisError``  |
           +-------------------+----------------------+
           | ``-arr.ndim``     | 0                    |
           +-------------------+----------------------+
           | |vdots|           | |vdots|              |
           +-------------------+----------------------+
           | ``-1``            | ``arr.ndim-1``       |
           +-------------------+----------------------+
           | ``0``             | ``0``                |
           +-------------------+----------------------+
           | |vdots|           | |vdots|              |
           +-------------------+----------------------+
           | ``arr.ndim``      | ``arr.ndim``         |
           +-------------------+----------------------+
           | ``arr.ndim + 1``  | raise ``AxisError``  |
           +-------------------+----------------------+

        .. |vdots|   unicode:: U+22EE .. Vertical Ellipsis

    Returns
    -------
    res : ndarray
        For NumPy >= 1.10.0 a view of `a` is always returned. For earlier
        NumPy versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    See Also
    --------
    moveaxis : Move array axes to new positions.
    roll : Roll the elements of an array by a number of positions along a
        given axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ones((3,4,5,6))
    >>> np.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> np.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> np.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """
    n = a.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise AxisError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        return a[...]
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)


@set_module("numpy.lib.array_utils")
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.

    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated

    See also
    --------
    normalize_axis_index : normalizing a single scalar axis
    """
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError('repeated axis in `{}` argument'.format(argname))
        else:
            raise ValueError('repeated axis')
    return axis


def _moveaxis_dispatcher(a, source, destination):
    return (a,)


@array_function_dispatch(_moveaxis_dispatcher)
def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    Parameters
    ----------
    a : np.ndarray
        The array whose axes should be reordered.
    source : int or sequence of int
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : np.ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    transpose : Permute the dimensions of an array.
    swapaxes : Interchange two axes of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)

    These all achieve the same result:

    >>> np.transpose(x).shape
    (5, 4, 3)
    >>> np.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)

    """
    try:
        # allow duck-array types if they define transpose
        transpose = a.transpose
    except AttributeError:
        a = asarray(a)
        transpose = a.transpose

    source = normalize_axis_tuple(source, a.ndim, 'source')
    destination = normalize_axis_tuple(destination, a.ndim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = transpose(order)
    return result


def _cross_dispatcher(a, b, axisa=None, axisb=None, axisc=None, axis=None):
    return (a, b)


@array_function_dispatch(_cross_dispatcher)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Return the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
    to both `a` and `b`.  If `a` and `b` are arrays of vectors, the vectors
    are defined by the last axis of `a` and `b` by default, and these axes
    can have dimensions 2 or 3.  Where the dimension of either `a` or `b` is
    2, the third component of the input vector is assumed to be zero and the
    cross product calculated accordingly.  In cases where both input vectors
    have dimension 2, the z-component of the cross product is returned.

    Parameters
    ----------
    a : array_like
        Components of the first vector(s).
    b : array_like
        Components of the second vector(s).
    axisa : int, optional
        Axis of `a` that defines the vector(s).  By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s).  By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s).  Ignored if
        both input vectors have dimension 2, as the return is scalar.
        By default, the last axis.
    axis : int, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    c : ndarray
        Vector cross product(s).

    Raises
    ------
    ValueError
        When the dimension of the vector(s) in `a` and/or `b` does not
        equal 2 or 3.

    See Also
    --------
    inner : Inner product
    outer : Outer product.
    linalg.cross : An Array API compatible variation of ``np.cross``,
                   which accepts (arrays of) 3-element vectors only.
    ix_ : Construct index arrays.

    Notes
    -----
    Supports full broadcasting of the inputs.

    Dimension-2 input arrays were deprecated in 2.0.0. If you do need this
    functionality, you can use::

        def cross2d(x, y):
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    Examples
    --------
    Vector cross-product.

    >>> import numpy as np
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([-3,  6, -3])

    One vector with dimension 2.

    >>> x = [1, 2]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Equivalently:

    >>> x = [1, 2, 0]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Both vectors with dimension 2.

    >>> x = [1,2]
    >>> y = [4,5]
    >>> np.cross(x, y)
    array(-3)

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the *right-hand rule*.

    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> y = np.array([[4,5,6], [1,2,3]])
    >>> np.cross(x, y)
    array([[-3,  6, -3],
           [ 3, -6,  3]])

    The orientation of `c` can be changed using the `axisc` keyword.

    >>> np.cross(x, y, axisc=0)
    array([[-3,  3],
           [ 6, -6],
           [-3,  3]])

    Change the vector definition of `x` and `y` using `axisa` and `axisb`.

    >>> x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
    >>> y = np.array([[7, 8, 9], [4,5,6], [1,2,3]])
    >>> np.cross(x, y)
    array([[ -6,  12,  -6],
           [  0,   0,   0],
           [  6, -12,   6]])
    >>> np.cross(x, y, axisa=0, axisb=0)
    array([[-24,  48, -24],
           [-30,  60, -30],
           [-36,  72, -36]])

    """
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3
    a = asarray(a)
    b = asarray(b)

    if (a.ndim < 1) or (b.ndim < 1):
        raise ValueError("At least one array has zero dimension")

    # Check axisa and axisb are within bounds
    axisa = normalize_axis_index(axisa, a.ndim, msg_prefix='axisa')
    axisb = normalize_axis_index(axisb, b.ndim, msg_prefix='axisb')

    # Move working axis to the end of the shape
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    msg = ("incompatible dimensions for cross product\n"
           "(dimension must be 2 or 3)")
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)
    if a.shape[-1] == 2 or b.shape[-1] == 2:
        # Deprecated in NumPy 2.0, 2023-09-26
        warnings.warn(
            "Arrays of 2-dimensional vectors are deprecated. Use arrays of "
            "3-dimensional vectors instead. (deprecated in NumPy 2.0)",
            DeprecationWarning, stacklevel=2
        )

    # Create the output array
    shape = broadcast(a[..., 0], b[..., 0]).shape
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # Check axisc is within bounds
        axisc = normalize_axis_index(axisc, len(shape), msg_prefix='axisc')
    dtype = promote_types(a.dtype, b.dtype)
    cp = empty(shape, dtype)

    # recast arrays as dtype
    a = a.astype(dtype)
    b = b.astype(dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            multiply(a0, b1, out=cp)
            cp -= a1 * b0
            return cp
        else:
            assert b.shape[-1] == 3
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            multiply(a1, b2, out=cp0)
            multiply(a0, b2, out=cp1)
            negative(cp1, out=cp1)
            multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            multiply(a1, b2, out=cp0)
            tmp = array(a2 * b1)
            cp0 -= tmp
            multiply(a2, b0, out=cp1)
            multiply(a0, b2, out=tmp)
            cp1 -= tmp
            multiply(a0, b1, out=cp2)
            multiply(a1, b0, out=tmp)
            cp2 -= tmp
        else:
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (b2 = 0)
            # cp1 = a2 * b0 - 0  (b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            multiply(a2, b1, out=cp0)
            negative(cp0, out=cp0)
            multiply(a2, b0, out=cp1)
            multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0

    return moveaxis(cp, -1, axisc)


little_endian = (sys.byteorder == 'little')


@set_module('numpy')
def indices(dimensions, dtype=int, sparse=False):
    """
    Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0, 1, ...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : dtype, optional
        Data type of the result.
    sparse : boolean, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is False.

    Returns
    -------
    grid : one ndarray or tuple of ndarrays
        If sparse is False:
            Returns one array of grid indices,
            ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True:
            Returns a tuple of arrays, with
            ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
            dimensions[i] in the ith place

    See Also
    --------
    mgrid, ogrid, meshgrid

    Notes
    -----
    The output shape in the dense case is obtained by prepending the number
    of dimensions in front of the tuple of dimensions, i.e. if `dimensions`
    is a tuple ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N, r0, ..., rN-1)``.

    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k, i0, i1, ..., iN-1] = ik

    Examples
    --------
    >>> import numpy as np
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 1, 2],
           [0, 1, 2]])

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0, 1, 2],
           [4, 5, 6]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.

    If sparse is set to true, the grid will be returned in a sparse
    representation.

    >>> i, j = np.indices((2, 3), sparse=True)
    >>> i.shape
    (2, 1)
    >>> j.shape
    (1, 3)
    >>> i        # row indices
    array([[0],
           [1]])
    >>> j        # column indices
    array([[0, 1, 2]])

    """
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,)*N
    if sparse:
        res = tuple()
    else:
        res = empty((N,)+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i+1:]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


@finalize_array_function_like
@set_module('numpy')
def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    """
    Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    Parameters
    ----------
    function : callable
        The function is called with N parameters, where N is the rank of
        `shape`.  Each parameter represents the coordinates of the array
        varying along a specific axis.  For example, if `shape`
        were ``(2, 2)``, then the parameters would be
        ``array([[0, 0], [1, 1]])`` and ``array([[0, 1], [0, 1]])``
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `function`.
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `function`.
        By default, `dtype` is float.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    fromfunction : any
        The result of the call to `function` is passed back directly.
        Therefore the shape of `fromfunction` is completely determined by
        `function`.  If `function` returns a scalar value, the shape of
        `fromfunction` would not match the `shape` parameter.

    See Also
    --------
    indices, meshgrid

    Notes
    -----
    Keywords other than `dtype` and `like` are passed to `function`.

    Examples
    --------
    >>> import numpy as np
    >>> np.fromfunction(lambda i, j: i, (2, 2), dtype=float)
    array([[0., 0.],
           [1., 1.]])

    >>> np.fromfunction(lambda i, j: j, (2, 2), dtype=float)
    array([[0., 1.],
           [0., 1.]])

    >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]])

    >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

    """
    if like is not None:
        return _fromfunction_with_like(
                like, function, shape, dtype=dtype, **kwargs)

    args = indices(shape, dtype=dtype)
    return function(*args, **kwargs)


_fromfunction_with_like = array_function_dispatch()(fromfunction)


def _frombuffer(buf, dtype, shape, order):
    return frombuffer(buf, dtype=dtype).reshape(shape, order=order)


@set_module('numpy')
def isscalar(element):
    """
    Returns True if the type of `element` is a scalar type.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `element` is a scalar type, False if it is not.

    See Also
    --------
    ndim : Get the number of dimensions of an array

    Notes
    -----
    If you need a stricter way to identify a *numerical* scalar, use
    ``isinstance(x, numbers.Number)``, as that returns ``False`` for most
    non-numerical elements such as strings.

    In most cases ``np.ndim(x) == 0`` should be used instead of this function,
    as that will also return true for 0d arrays. This is how numpy overloads
    functions in the style of the ``dx`` arguments to `gradient` and
    the ``bins`` argument to `histogram`. Some key differences:

    +------------------------------------+---------------+-------------------+
    | x                                  |``isscalar(x)``|``np.ndim(x) == 0``|
    +====================================+===============+===================+
    | PEP 3141 numeric objects           | ``True``      | ``True``          |
    | (including builtins)               |               |                   |
    +------------------------------------+---------------+-------------------+
    | builtin string and buffer objects  | ``True``      | ``True``          |
    +------------------------------------+---------------+-------------------+
    | other builtin objects, like        | ``False``     | ``True``          |
    | `pathlib.Path`, `Exception`,       |               |                   |
    | the result of `re.compile`         |               |                   |
    +------------------------------------+---------------+-------------------+
    | third-party objects like           | ``False``     | ``True``          |
    | `matplotlib.figure.Figure`         |               |                   |
    +------------------------------------+---------------+-------------------+
    | zero-dimensional numpy arrays      | ``False``     | ``True``          |
    +------------------------------------+---------------+-------------------+
    | other numpy arrays                 | ``False``     | ``False``         |
    +------------------------------------+---------------+-------------------+
    | `list`, `tuple`, and other         | ``False``     | ``False``         |
    | sequence objects                   |               |                   |
    +------------------------------------+---------------+-------------------+

    Examples
    --------
    >>> import numpy as np

    >>> np.isscalar(3.1)
    True

    >>> np.isscalar(np.array(3.1))
    False

    >>> np.isscalar([3.1])
    False

    >>> np.isscalar(False)
    True

    >>> np.isscalar('numpy')
    True

    NumPy supports PEP 3141 numbers:

    >>> from fractions import Fraction
    >>> np.isscalar(Fraction(5, 17))
    True
    >>> from numbers import Number
    >>> np.isscalar(Number())
    True

    """
    return (isinstance(element, generic)
            or type(element) in ScalarType
            or isinstance(element, numbers.Number))


@set_module('numpy')
def binary_repr(num, width=None):
    """
    Return the binary representation of the input number as a string.

    For negative numbers, if width is not given, a minus sign is added to the
    front. If width is given, the two's complement of the number is
    returned, with respect to that width.

    In a two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit two's-complement
    system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    num : int
        Only an integer decimal number can be used.
    width : int, optional
        The length of the returned string if `num` is positive, or the length
        of the two's complement if `num` is negative, provided that `width` is
        at least a sufficient number of bits for `num` to be represented in
        the designated form. If the `width` value is insufficient, an error is
        raised.

    Returns
    -------
    bin : str
        Binary representation of `num` or two's complement of `num`.

    See Also
    --------
    base_repr: Return a string representation of a number in the given base
               system.
    bin: Python's built-in binary representation generator of an integer.

    Notes
    -----
    `binary_repr` is equivalent to using `base_repr` with base 2, but about 25x
    faster.

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        https://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    >>> import numpy as np
    >>> np.binary_repr(3)
    '11'
    >>> np.binary_repr(-3)
    '-11'
    >>> np.binary_repr(3, width=4)
    '0011'

    The two's complement is returned when the input number is negative and
    width is specified:

    >>> np.binary_repr(-3, width=3)
    '101'
    >>> np.binary_repr(-3, width=5)
    '11101'

    """
    def err_if_insufficient(width, binwidth):
        if width is not None and width < binwidth:
            raise ValueError(
                f"Insufficient bit {width=} provided for {binwidth=}"
            )

    # Ensure that num is a Python integer to avoid overflow or unwanted
    # casts to floating point.
    num = operator.index(num)

    if num == 0:
        return '0' * (width or 1)

    elif num > 0:
        binary = bin(num)[2:]
        binwidth = len(binary)
        outwidth = (binwidth if width is None
                    else builtins.max(binwidth, width))
        err_if_insufficient(width, binwidth)
        return binary.zfill(outwidth)

    else:
        if width is None:
            return '-' + bin(-num)[2:]

        else:
            poswidth = len(bin(-num)[2:])

            # See gh-8679: remove extra digit
            # for numbers at boundaries.
            if 2**(poswidth - 1) == -num:
                poswidth -= 1

            twocomp = 2**(poswidth + 1) + num
            binary = bin(twocomp)[2:]
            binwidth = len(binary)

            outwidth = builtins.max(binwidth, width)
            err_if_insufficient(width, binwidth)
            return '1' * (outwidth - binwidth) + binary


@set_module('numpy')
def base_repr(number, base=2, padding=0):
    """
    Return a string representation of a number in the given base system.

    Parameters
    ----------
    number : int
        The value to convert. Positive and negative values are handled.
    base : int, optional
        Convert `number` to the `base` number system. The valid range is 2-36,
        the default value is 2.
    padding : int, optional
        Number of zeros padded on the left. Default is 0 (no padding).

    Returns
    -------
    out : str
        String representation of `number` in `base` system.

    See Also
    --------
    binary_repr : Faster version of `base_repr` for base 2.

    Examples
    --------
    >>> import numpy as np
    >>> np.base_repr(5)
    '101'
    >>> np.base_repr(6, 5)
    '11'
    >>> np.base_repr(7, base=5, padding=3)
    '00012'

    >>> np.base_repr(10, base=16)
    'A'
    >>> np.base_repr(32, base=16)
    '20'

    """
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    elif base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")

    num = abs(int(number))
    res = []
    while num:
        res.append(digits[num % base])
        num //= base
    if padding:
        res.append('0' * padding)
    if number < 0:
        res.append('-')
    return ''.join(reversed(res or '0'))


# These are all essentially abbreviations
# These might wind up in a special abbreviations module


def _maketup(descr, val):
    dt = dtype(descr)
    # Place val in all scalar tuples:
    fields = dt.fields
    if fields is None:
        return val
    else:
        res = [_maketup(fields[name][0], val) for name in dt.names]
        return tuple(res)


@finalize_array_function_like
@set_module('numpy')
def identity(n, dtype=None, *, like=None):
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> import numpy as np
    >>> np.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    """
    if like is not None:
        return _identity_with_like(like, n, dtype=dtype)

    from numpy import eye
    return eye(n, dtype=dtype, like=like)


_identity_with_like = array_function_dispatch()(identity)


def _allclose_dispatcher(a, b, rtol=None, atol=None, equal_nan=None):
    return (a, b, rtol, atol)


@array_function_dispatch(_allclose_dispatcher)
def allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 with magnitudes much smaller than one (see Notes).

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : array_like
        The relative tolerance parameter (see Notes).
    atol : array_like
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    isclose, all, any, equal

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.::

     absolute(a - b) <= (atol + rtol * absolute(b))

    The above equation is not symmetric in `a` and `b`, so that
    ``allclose(a, b)`` might be different from ``allclose(b, a)`` in
    some rare cases.

    The default value of `atol` is not appropriate when the reference value
    `b` has magnitude smaller than one. For example, it is unlikely that
    ``a = 1e-9`` and ``b = 2e-9`` should be considered "close", yet
    ``allclose(1e-9, 2e-9)`` is ``True`` with default settings. Be sure
    to select `atol` for the use case at hand, especially for defining the
    threshold below which a non-zero value in `a` will be considered "close"
    to a very small or zero value in `b`.

    The comparison of `a` and `b` uses standard broadcasting, which
    means that `a` and `b` need not have the same shape in order for
    ``allclose(a, b)`` to evaluate to True.  The same is true for
    `equal` but not `array_equal`.

    `allclose` is not defined for non-numeric data types.
    `bool` is considered a numeric data-type for this purpose.

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    False

    >>> np.allclose([1e10,1e-8], [1.00001e10,1e-9])
    True

    >>> np.allclose([1e10,1e-8], [1.0001e10,1e-9])
    False

    >>> np.allclose([1.0, np.nan], [1.0, np.nan])
    False

    >>> np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    True


    """
    res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
    return builtins.bool(res)


def _isclose_dispatcher(a, b, rtol=None, atol=None, equal_nan=None):
    return (a, b, rtol, atol)


@array_function_dispatch(_isclose_dispatcher)
def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 with magnitudes much smaller than one (see Notes).

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : array_like
        The relative tolerance parameter (see Notes).
    atol : array_like
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose
    math.isclose

    Notes
    -----
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.::

     absolute(a - b) <= (atol + rtol * absolute(b))

    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`.

    The default value of `atol` is not appropriate when the reference value
    `b` has magnitude smaller than one. For example, it is unlikely that
    ``a = 1e-9`` and ``b = 2e-9`` should be considered "close", yet
    ``isclose(1e-9, 2e-9)`` is ``True`` with default settings. Be sure
    to select `atol` for the use case at hand, especially for defining the
    threshold below which a non-zero value in `a` will be considered "close"
    to a very small or zero value in `b`.

    `isclose` is not defined for non-numeric data types.
    :class:`bool` is considered a numeric data-type for this purpose.

    Examples
    --------
    >>> import numpy as np
    >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])
    array([ True, False])

    >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])
    array([ True, True])

    >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])
    array([False,  True])

    >>> np.isclose([1.0, np.nan], [1.0, np.nan])
    array([ True, False])

    >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    array([ True, True])

    >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
    array([ True, False])

    >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
    array([False, False])

    >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
    array([ True,  True])

    >>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
    array([False,  True])

    """
    # Turn all but python scalars into arrays.
    x, y, atol, rtol = (
        a if isinstance(a, (int, float, complex)) else asanyarray(a)
        for a in (a, b, atol, rtol))

    # Make sure y is an inexact type to avoid bad behavior on abs(MIN_INT).
    # This will cause casting of x later. Also, make sure to allow subclasses
    # (e.g., for numpy.ma).
    # NOTE: We explicitly allow timedelta, which used to work. This could
    #       possibly be deprecated. See also gh-18286.
    #       timedelta works if `atol` is an integer or also a timedelta.
    #       Although, the default tolerances are unlikely to be useful
    if (dtype := getattr(y, "dtype", None)) is not None and dtype.kind != "m":
        dt = multiarray.result_type(y, 1.)
        y = asanyarray(y, dtype=dt)
    elif isinstance(y, int):
        y = float(y)

    with errstate(invalid='ignore'):
        result = (less_equal(abs(x-y), atol + rtol * abs(y))
                  & isfinite(y)
                  | (x == y))
        if equal_nan:
            result |= isnan(x) & isnan(y)

    return result[()]  # Flatten 0d arrays to scalars


def _array_equal_dispatcher(a1, a2, equal_nan=None):
    return (a1, a2)


_no_nan_types = {
    # should use np.dtype.BoolDType, but as of writing
    # that fails the reloading test.
    type(dtype(nt.bool)),
    type(dtype(nt.int8)),
    type(dtype(nt.int16)),
    type(dtype(nt.int32)),
    type(dtype(nt.int64)),
}


def _dtype_cannot_hold_nan(dtype):
    return type(dtype) in _no_nan_types


@array_function_dispatch(_array_equal_dispatcher)
def array_equal(a1, a2, equal_nan=False):
    """
    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.
    equal_nan : bool
        Whether to compare NaN's as equal. If the dtype of a1 and a2 is
        complex, values will be considered equal if either the real or the
        imaginary component of a given value is ``nan``.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.

    See Also
    --------
    allclose: Returns True if two arrays are element-wise equal within a
              tolerance.
    array_equiv: Returns True if input arrays are shape consistent and all
                 elements equal.

    Examples
    --------
    >>> import numpy as np

    >>> np.array_equal([1, 2], [1, 2])
    True

    >>> np.array_equal(np.array([1, 2]), np.array([1, 2]))
    True

    >>> np.array_equal([1, 2], [1, 2, 3])
    False

    >>> np.array_equal([1, 2], [1, 4])
    False

    >>> a = np.array([1, np.nan])
    >>> np.array_equal(a, a)
    False

    >>> np.array_equal(a, a, equal_nan=True)
    True

    When ``equal_nan`` is True, complex values with nan components are
    considered equal if either the real *or* the imaginary components are nan.

    >>> a = np.array([1 + 1j])
    >>> b = a.copy()
    >>> a.real = np.nan
    >>> b.imag = np.nan
    >>> np.array_equal(a, b, equal_nan=True)
    True
    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    if a1.shape != a2.shape:
        return False
    if not equal_nan:
        return builtins.bool((asanyarray(a1 == a2)).all())

    if a1 is a2:
        # nan will compare equal so an array will compare equal to itself.
        return True

    cannot_have_nan = (_dtype_cannot_hold_nan(a1.dtype)
                       and _dtype_cannot_hold_nan(a2.dtype))
    if cannot_have_nan:
        return builtins.bool(asarray(a1 == a2).all())

    # Handling NaN values if equal_nan is True
    a1nan, a2nan = isnan(a1), isnan(a2)
    # NaN's occur at different locations
    if not (a1nan == a2nan).all():
        return False
    # Shapes of a1, a2 and masks are guaranteed to be consistent by this point
    return builtins.bool((a1[~a1nan] == a2[~a1nan]).all())


def _array_equiv_dispatcher(a1, a2):
    return (a1, a2)


@array_function_dispatch(_array_equiv_dispatcher)
def array_equiv(a1, a2):
    """
    Returns True if input arrays are shape consistent and all elements equal.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    out : bool
        True if equivalent, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> np.array_equiv([1, 2], [1, 2])
    True
    >>> np.array_equiv([1, 2], [1, 3])
    False

    Showing the shape equivalence:

    >>> np.array_equiv([1, 2], [[1, 2], [1, 2]])
    True
    >>> np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
    False

    >>> np.array_equiv([1, 2], [[1, 2], [1, 3]])
    False

    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    try:
        multiarray.broadcast(a1, a2)
    except Exception:
        return False

    return builtins.bool(asanyarray(a1 == a2).all())


def _astype_dispatcher(x, dtype, /, *, copy=None, device=None):
    return (x, dtype)


@array_function_dispatch(_astype_dispatcher)
def astype(x, dtype, /, *, copy=True, device=None):
    """
    Copies an array to a specified data type.

    This function is an Array API compatible alternative to
    `numpy.ndarray.astype`.

    Parameters
    ----------
    x : ndarray
        Input NumPy array to cast. ``array_likes`` are explicitly not
        supported here.
    dtype : dtype
        Data type of the result.
    copy : bool, optional
        Specifies whether to copy an array when the specified dtype matches
        the data type of the input array ``x``. If ``True``, a newly allocated
        array must always be returned. If ``False`` and the specified dtype
        matches the data type of the input array, the input array must be
        returned; otherwise, a newly allocated array must be returned.
        Defaults to ``True``.
    device : str, optional
        The device on which to place the returned array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.1.0

    Returns
    -------
    out : ndarray
        An array having the specified data type.

    See Also
    --------
    ndarray.astype

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3]); arr
    array([1, 2, 3])
    >>> np.astype(arr, np.float64)
    array([1., 2., 3.])

    Non-copy case:

    >>> arr = np.array([1, 2, 3])
    >>> arr_noncpy = np.astype(arr, arr.dtype, copy=False)
    >>> np.shares_memory(arr, arr_noncpy)
    True

    """
    if not (isinstance(x, np.ndarray) or isscalar(x)):
        raise TypeError(
            "Input should be a NumPy array or scalar. "
            f"It is a {type(x)} instead."
        )
    if device is not None and device != "cpu":
        raise ValueError(
            'Device not understood. Only "cpu" is allowed, but received:'
            f' {device}'
        )
    return x.astype(dtype, copy=copy)


inf = PINF
nan = NAN
False_ = nt.bool(False)
True_ = nt.bool(True)


def extend_all(module):
    existing = set(__all__)
    mall = module.__all__
    for a in mall:
        if a not in existing:
            __all__.append(a)


from .umath import *
from .numerictypes import *
from . import fromnumeric
from .fromnumeric import *
from . import arrayprint
from .arrayprint import *
from . import _asarray
from ._asarray import *
from . import _ufunc_config
from ._ufunc_config import *
extend_all(fromnumeric)
extend_all(umath)
extend_all(numerictypes)
extend_all(arrayprint)
extend_all(_asarray)
extend_all(_ufunc_config)
<|endoftext|>"""
numerictypes: Define the numeric type objects

This module is designed so "from numerictypes import \\*" is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    sctypeDict

  Type objects (not all will be available, depends on platform):
      see variable sctypes for which ones you have

    Bit-width names

    int8 int16 int32 int64 int128
    uint8 uint16 uint32 uint64 uint128
    float16 float32 float64 float96 float128 float256
    complex32 complex64 complex128 complex192 complex256 complex512
    datetime64 timedelta64

    c-based names

    bool

    object_

    void, str_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    double, cdouble,
    longdouble, clongdouble,

   As part of the type-hierarchy:    xx -- is bit-width

   generic
     +-> bool                                   (kind=b)
     +-> number
     |   +-> integer
     |   |   +-> signedinteger     (intxx)      (kind=i)
     |   |   |     byte
     |   |   |     short
     |   |   |     intc
     |   |   |     intp
     |   |   |     int_
     |   |   |     longlong
     |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
     |   |         ubyte
     |   |         ushort
     |   |         uintc
     |   |         uintp
     |   |         uint
     |   |         ulonglong
     |   +-> inexact
     |       +-> floating          (floatxx)    (kind=f)
     |       |     half
     |       |     single
     |       |     double
     |       |     longdouble
     |       \\-> complexfloating  (complexxx)  (kind=c)
     |             csingle
     |             cdouble
     |             clongdouble
     +-> flexible
     |   +-> character
     |   |     bytes_                           (kind=S)
     |   |     str_                             (kind=U)
     |   |
     |   \\-> void                              (kind=V)
     \\-> object_ (not used much)               (kind=O)

"""
import numbers
import warnings

from . import multiarray as ma
from .multiarray import (
        ndarray, dtype, datetime_data, datetime_as_string,
        busday_offset, busday_count, is_busday, busdaycalendar
        )
from .._utils import set_module

# we add more at the bottom
__all__ = [
    'ScalarType', 'typecodes', 'issubdtype', 'datetime_data',
    'datetime_as_string', 'busday_offset', 'busday_count',
    'is_busday', 'busdaycalendar', 'isdtype'
]

# we don't need all these imports, but we need to keep them for compatibility
# for users using np._core.numerictypes.UPPER_TABLE
from ._string_helpers import (  # noqa: F401
    english_lower, english_upper, english_capitalize, LOWER_TABLE, UPPER_TABLE
)

from ._type_aliases import (
    sctypeDict, allTypes, sctypes
)
from ._dtype import _kind_name

# we don't export these for import *, but we do want them accessible
# as numerictypes.bool, etc.
from builtins import bool, int, float, complex, object, str, bytes  # noqa: F401, UP029


# We use this later
generic = allTypes['generic']

genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16',
                   'int32', 'uint32', 'int64', 'uint64', 'int128',
                   'uint128', 'float16',
                   'float32', 'float64', 'float80', 'float96', 'float128',
                   'float256',
                   'complex32', 'complex64', 'complex128', 'complex160',
                   'complex192', 'complex256', 'complex512', 'object']

@set_module('numpy')
def maximum_sctype(t):
    """
    Return the scalar type of highest precision of the same kind as the input.

    .. deprecated:: 2.0
        Use an explicit dtype like int64 or float64 instead.

    Parameters
    ----------
    t : dtype or dtype specifier
        The input data type. This can be a `dtype` object or an object that
        is convertible to a `dtype`.

    Returns
    -------
    out : dtype
        The highest precision data type of the same kind (`dtype.kind`) as `t`.

    See Also
    --------
    obj2sctype, mintypecode, sctype2char
    dtype

    Examples
    --------
    >>> from numpy._core.numerictypes import maximum_sctype
    >>> maximum_sctype(int)
    <class 'numpy.int64'>
    >>> maximum_sctype(np.uint8)
    <class 'numpy.uint64'>
    >>> maximum_sctype(complex)
    <class 'numpy.complex256'> # may vary

    >>> maximum_sctype(str)
    <class 'numpy.str_'>

    >>> maximum_sctype('i2')
    <class 'numpy.int64'>
    >>> maximum_sctype('f4')
    <class 'numpy.float128'> # may vary

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`maximum_sctype` is deprecated. Use an explicit dtype like int64 "
        "or float64 instead. (deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    g = obj2sctype(t)
    if g is None:
        return t
    t = g
    base = _kind_name(dtype(t))
    if base in sctypes:
        return sctypes[base][-1]
    else:
        return t


@set_module('numpy')
def issctype(rep):
    """
    Determines whether the given object represents a scalar data-type.

    Parameters
    ----------
    rep : any
        If `rep` is an instance of a scalar dtype, True is returned. If not,
        False is returned.

    Returns
    -------
    out : bool
        Boolean result of check whether `rep` is a scalar dtype.

    See Also
    --------
    issubsctype, issubdtype, obj2sctype, sctype2char

    Examples
    --------
    >>> from numpy._core.numerictypes import issctype
    >>> issctype(np.int32)
    True
    >>> issctype(list)
    False
    >>> issctype(1.1)
    False

    Strings are also a scalar type:

    >>> issctype(np.dtype('str'))
    True

    """
    if not isinstance(rep, (type, dtype)):
        return False
    try:
        res = obj2sctype(rep)
        if res and res != object_:
            return True
        else:
            return False
    except Exception:
        return False


@set_module('numpy')
def obj2sctype(rep, default=None):
    """
    Return the scalar dtype or NumPy equivalent of Python type of an object.

    Parameters
    ----------
    rep : any
        The object of which the type is returned.
    default : any, optional
        If given, this is returned for objects whose types can not be
        determined. If not given, None is returned for those objects.

    Returns
    -------
    dtype : dtype or Python type
        The data type of `rep`.

    See Also
    --------
    sctype2char, issctype, issubsctype, issubdtype

    Examples
    --------
    >>> from numpy._core.numerictypes import obj2sctype
    >>> obj2sctype(np.int32)
    <class 'numpy.int32'>
    >>> obj2sctype(np.array([1., 2.]))
    <class 'numpy.float64'>
    >>> obj2sctype(np.array([1.j]))
    <class 'numpy.complex128'>

    >>> obj2sctype(dict)
    <class 'numpy.object_'>
    >>> obj2sctype('string')

    >>> obj2sctype(1, default=list)
    <class 'list'>

    """
    # prevent abstract classes being upcast
    if isinstance(rep, type) and issubclass(rep, generic):
        return rep
    # extract dtype from arrays
    if isinstance(rep, ndarray):
        return rep.dtype.type
    # fall back on dtype to convert
    try:
        res = dtype(rep)
    except Exception:
        return default
    else:
        return res.type


@set_module('numpy')
def issubclass_(arg1, arg2):
    """
    Determine if a class is a subclass of a second class.

    `issubclass_` is equivalent to the Python built-in ``issubclass``,
    except that it returns False instead of raising a TypeError if one
    of the arguments is not a class.

    Parameters
    ----------
    arg1 : class
        Input class. True is returned if `arg1` is a subclass of `arg2`.
    arg2 : class or tuple of classes.
        Input class. If a tuple of classes, True is returned if `arg1` is a
        subclass of any of the tuple elements.

    Returns
    -------
    out : bool
        Whether `arg1` is a subclass of `arg2` or not.

    See Also
    --------
    issubsctype, issubdtype, issctype

    Examples
    --------
    >>> np.issubclass_(np.int32, int)
    False
    >>> np.issubclass_(np.int32, float)
    False
    >>> np.issubclass_(np.float64, float)
    True

    """
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False


@set_module('numpy')
def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    Parameters
    ----------
    arg1, arg2 : dtype or dtype specifier
        Data-types.

    Returns
    -------
    out : bool
        The result.

    See Also
    --------
    issctype, issubdtype, obj2sctype

    Examples
    --------
    >>> from numpy._core import issubsctype
    >>> issubsctype('S8', str)
    False
    >>> issubsctype(np.array([1]), int)
    True
    >>> issubsctype(np.array([1]), float)
    False

    """
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))


class _PreprocessDTypeError(Exception):
    pass


def _preprocess_dtype(dtype):
    """
    Preprocess dtype argument by:
      1. fetching type from a data type
      2. verifying that types are built-in NumPy dtypes
    """
    if isinstance(dtype, ma.dtype):
        dtype = dtype.type
    if isinstance(dtype, ndarray) or dtype not in allTypes.values():
        raise _PreprocessDTypeError
    return dtype


@set_module('numpy')
def isdtype(dtype, kind):
    """
    Determine if a provided dtype is of a specified data type ``kind``.

    This function only supports built-in NumPy's data types.
    Third-party dtypes are not yet supported.

    Parameters
    ----------
    dtype : dtype
        The input dtype.
    kind : dtype or str or tuple of dtypes/strs.
        dtype or dtype kind. Allowed dtype kinds are:
        * ``'bool'`` : boolean kind
        * ``'signed integer'`` : signed integer data types
        * ``'unsigned integer'`` : unsigned integer data types
        * ``'integral'`` : integer data types
        * ``'real floating'`` : real-valued floating-point data types
        * ``'complex floating'`` : complex floating-point data types
        * ``'numeric'`` : numeric data types

    Returns
    -------
    out : bool

    See Also
    --------
    issubdtype

    Examples
    --------
    >>> import numpy as np
    >>> np.isdtype(np.float32, np.float64)
    False
    >>> np.isdtype(np.float32, "real floating")
    True
    >>> np.isdtype(np.complex128, ("real floating", "complex floating"))
    True

    """
    try:
        dtype = _preprocess_dtype(dtype)
    except _PreprocessDTypeError:
        raise TypeError(
            "dtype argument must be a NumPy dtype, "
            f"but it is a {type(dtype)}."
        ) from None

    input_kinds = kind if isinstance(kind, tuple) else (kind,)

    processed_kinds = set()

    for kind in input_kinds:
        if kind == "bool":
            processed_kinds.add(allTypes["bool"])
        elif kind == "signed integer":
            processed_kinds.update(sctypes["int"])
        elif kind == "unsigned integer":
            processed_kinds.update(sctypes["uint"])
        elif kind == "integral":
            processed_kinds.update(sctypes["int"] + sctypes["uint"])
        elif kind == "real floating":
            processed_kinds.update(sctypes["float"])
        elif kind == "complex floating":
            processed_kinds.update(sctypes["complex"])
        elif kind == "numeric":
            processed_kinds.update(
                sctypes["int"] + sctypes["uint"] +
                sctypes["float"] + sctypes["complex"]
            )
        elif isinstance(kind, str):
            raise ValueError(
                "kind argument is a string, but"
                f" {kind!r} is not a known kind name."
            )
        else:
            try:
                kind = _preprocess_dtype(kind)
            except _PreprocessDTypeError:
                raise TypeError(
                    "kind argument must be comprised of "
                    "NumPy dtypes or strings only, "
                    f"but is a {type(kind)}."
                ) from None
            processed_kinds.add(kind)

    return dtype in processed_kinds


@set_module('numpy')
def issubdtype(arg1, arg2):
    r"""
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    This is like the builtin :func:`issubclass`, but for `dtype`\ s.

    Parameters
    ----------
    arg1, arg2 : dtype_like
        `dtype` or object coercible to one

    Returns
    -------
    out : bool

    See Also
    --------
    :ref:`arrays.scalars` : Overview of the numpy type hierarchy.

    Examples
    --------
    `issubdtype` can be used to check the type of arrays:

    >>> ints = np.array([1, 2, 3], dtype=np.int32)
    >>> np.issubdtype(ints.dtype, np.integer)
    True
    >>> np.issubdtype(ints.dtype, np.floating)
    False

    >>> floats = np.array([1, 2, 3], dtype=np.float32)
    >>> np.issubdtype(floats.dtype, np.integer)
    False
    >>> np.issubdtype(floats.dtype, np.floating)
    True

    Similar types of different sizes are not subdtypes of each other:

    >>> np.issubdtype(np.float64, np.float32)
    False
    >>> np.issubdtype(np.float32, np.float64)
    False

    but both are subtypes of `floating`:

    >>> np.issubdtype(np.float64, np.floating)
    True
    >>> np.issubdtype(np.float32, np.floating)
    True

    For convenience, dtype-like objects are allowed too:

    >>> np.issubdtype('S1', np.bytes_)
    True
    >>> np.issubdtype('i4', np.signedinteger)
    True

    """
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type

    return issubclass(arg1, arg2)


@set_module('numpy')
def sctype2char(sctype):
    """
    Return the string representation of a scalar dtype.

    Parameters
    ----------
    sctype : scalar dtype or object
        If a scalar dtype, the corresponding string character is
        returned. If an object, `sctype2char` tries to infer its scalar type
        and then return the corresponding string character.

    Returns
    -------
    typechar : str
        The string character corresponding to the scalar type.

    Raises
    ------
    ValueError
        If `sctype` is an object for which the type can not be inferred.

    See Also
    --------
    obj2sctype, issctype, issubsctype, mintypecode

    Examples
    --------
    >>> from numpy._core.numerictypes import sctype2char
    >>> for sctype in [np.int32, np.double, np.cdouble, np.bytes_, np.ndarray]:
    ...     print(sctype2char(sctype))
    l # may vary
    d
    D
    S
    O

    >>> x = np.array([1., 2-1.j])
    >>> sctype2char(x)
    'D'
    >>> sctype2char(list)
    'O'

    """
    sctype = obj2sctype(sctype)
    if sctype is None:
        raise ValueError("unrecognized type")
    if sctype not in sctypeDict.values():
        # for compatibility
        raise KeyError(sctype)
    return dtype(sctype).char


def _scalar_type_key(typ):
    """A ``key`` function for `sorted`."""
    dt = dtype(typ)
    return (dt.kind.lower(), dt.itemsize)


ScalarType = [int, float, complex, bool, bytes, str, memoryview]
ScalarType += sorted(set(sctypeDict.values()), key=_scalar_type_key)
ScalarType = tuple(ScalarType)


# Now add the types we've determined to this module
for key in allTypes:
    globals()[key] = allTypes[key]
    __all__.append(key)

del key

typecodes = {'Character': 'c',
             'Integer': 'bhilqnp',
             'UnsignedInteger': 'BHILQNP',
             'Float': 'efdg',
             'Complex': 'FDG',
             'AllInteger': 'bBhHiIlLqQnNpP',
             'AllFloat': 'efdgFDG',
             'Datetime': 'Mm',
             'All': '?bhilqnpBHILQNPefdgFDGSUVOMm'}

# backwards compatibility --- deprecated name
# Formal deprecation: Numpy 1.20.0, 2020-10-19 (see numpy/__init__.py)
typeDict = sctypeDict

def _register_types():
    numbers.Integral.register(integer)
    numbers.Complex.register(inexact)
    numbers.Real.register(floating)
    numbers.Number.register(number)


_register_types()
<|endoftext|>"""Implementation of __array_function__ overrides from NEP-18."""
import collections
import functools

from .._utils import set_module
from .._utils._inspect import getargspec
from numpy._core._multiarray_umath import (
    add_docstring,  _get_implementing_args, _ArrayFunctionDispatcher)


ARRAY_FUNCTIONS = set()

array_function_like_doc = (
    """like : array_like, optional
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument."""
)

def get_array_function_like_doc(public_api, docstring_template=""):
    ARRAY_FUNCTIONS.add(public_api)
    docstring = public_api.__doc__ or docstring_template
    return docstring.replace("${ARRAY_FUNCTION_LIKE}", array_function_like_doc)

def finalize_array_function_like(public_api):
    public_api.__doc__ = get_array_function_like_doc(public_api)
    return public_api


add_docstring(
    _ArrayFunctionDispatcher,
    """
    Class to wrap functions with checks for __array_function__ overrides.

    All arguments are required, and can only be passed by position.

    Parameters
    ----------
    dispatcher : function or None
        The dispatcher function that returns a single sequence-like object
        of all arguments relevant.  It must have the same signature (except
        the default values) as the actual implementation.
        If ``None``, this is a ``like=`` dispatcher and the
        ``_ArrayFunctionDispatcher`` must be called with ``like`` as the
        first (additional and positional) argument.
    implementation : function
        Function that implements the operation on NumPy arrays without
        overrides.  Arguments passed calling the ``_ArrayFunctionDispatcher``
        will be forwarded to this (and the ``dispatcher``) as if using
        ``*args, **kwargs``.

    Attributes
    ----------
    _implementation : function
        The original implementation passed in.
    """)


# exposed for testing purposes; used internally by _ArrayFunctionDispatcher
add_docstring(
    _get_implementing_args,
    """
    Collect arguments on which to call __array_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of possibly array-like arguments to check for
        __array_function__ methods.

    Returns
    -------
    Sequence of arguments with __array_function__ methods, in the order in
    which they should be called.
    """)


ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')


def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    implementation_spec = ArgSpec(*getargspec(implementation))
    dispatcher_spec = ArgSpec(*getargspec(dispatcher))

    if (implementation_spec.args != dispatcher_spec.args or
            implementation_spec.varargs != dispatcher_spec.varargs or
            implementation_spec.keywords != dispatcher_spec.keywords or
            (bool(implementation_spec.defaults) !=
             bool(dispatcher_spec.defaults)) or
            (implementation_spec.defaults is not None and
             len(implementation_spec.defaults) !=
             len(dispatcher_spec.defaults))):
        raise RuntimeError('implementation and dispatcher for %s have '
                           'different function signatures' % implementation)

    if implementation_spec.defaults is not None:
        if dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults):
            raise RuntimeError('dispatcher functions can only use None for '
                               'default argument values')


def array_function_dispatch(dispatcher=None, module=None, verify=True,
                            docs_from_dispatcher=False):
    """Decorator for adding dispatch with the __array_function__ protocol.

    See NEP-18 for example usage.

    Parameters
    ----------
    dispatcher : callable or None
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__array_function__``.

        If `None`, the first argument is used as the single `like=` argument
        and not passed on.  A function implementing `like=` must call its
        dispatcher with `like` as the first non-keyword argument.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.

    """
    def decorator(implementation):
        if verify:
            if dispatcher is not None:
                verify_matching_signatures(implementation, dispatcher)
            else:
                # Using __code__ directly similar to verify_matching_signature
                co = implementation.__code__
                last_arg = co.co_argcount + co.co_kwonlyargcount - 1
                last_arg = co.co_varnames[last_arg]
                if last_arg != "like" or co.co_kwonlyargcount == 0:
                    raise RuntimeError(
                        "__array_function__ expects `like=` to be the last "
                        "argument and a keyword-only argument. "
                        f"{implementation} does not seem to comply.")

        if docs_from_dispatcher:
            add_docstring(implementation, dispatcher.__doc__)

        public_api = _ArrayFunctionDispatcher(dispatcher, implementation)
        public_api = functools.wraps(implementation)(public_api)

        if module is not None:
            public_api.__module__ = module

        ARRAY_FUNCTIONS.add(public_api)

        return public_api

    return decorator


def array_function_from_dispatcher(
        implementation, module=None, verify=True, docs_from_dispatcher=True):
    """Like array_function_dispatcher, but with function arguments flipped."""

    def decorator(dispatcher):
        return array_function_dispatch(
            dispatcher, module, verify=verify,
            docs_from_dispatcher=docs_from_dispatcher)(implementation)
    return decorator
<|endoftext|>"""
This module contains a set of functions for record arrays.
"""
import os
import warnings
from collections import Counter
from contextlib import nullcontext

from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from .arrayprint import _get_legacy_print_mode

# All of the functions allow formats to be a dtype
__all__ = [
    'record', 'recarray', 'format_parser', 'fromarrays', 'fromrecords',
    'fromstring', 'fromfile', 'array', 'find_duplicate',
]


ndarray = sb.ndarray

_byteorderconv = {'b': '>',
                  'l': '<',
                  'n': '=',
                  'B': '>',
                  'L': '<',
                  'N': '=',
                  'S': 's',
                  's': 's',
                  '>': '>',
                  '<': '<',
                  '=': '=',
                  '|': '|',
                  'I': '|',
                  'i': '|'}

# formats regular expression
# allows multidimensional spec with a tuple syntax in front
# of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  '
# are equally allowed

numfmt = nt.sctypeDict


@set_module('numpy.rec')
def find_duplicate(list):
    """Find duplication in a list, return a list of duplicated elements"""
    return [
        item
        for item, counts in Counter(list).items()
        if counts > 1
    ]


@set_module('numpy.rec')
class format_parser:
    """
    Class to convert formats, names, titles description to a dtype.

    After constructing the format_parser object, the dtype attribute is
    the converted data-type:
    ``dtype = format_parser(formats, names, titles).dtype``

    Attributes
    ----------
    dtype : dtype
        The converted data-type.

    Parameters
    ----------
    formats : str or list of str
        The format description, either specified as a string with
        comma-separated format descriptions in the form ``'f8, i4, S5'``, or
        a list of format description strings  in the form
        ``['f8', 'i4', 'S5']``.
    names : str or list/tuple of str
        The field names, either specified as a comma-separated string in the
        form ``'col1, col2, col3'``, or as a list or tuple of strings in the
        form ``['col1', 'col2', 'col3']``.
        An empty list can be used, in that case default field names
        ('f0', 'f1', ...) are used.
    titles : sequence
        Sequence of title strings. An empty list can be used to leave titles
        out.
    aligned : bool, optional
        If True, align the fields by padding as the C-compiler would.
        Default is False.
    byteorder : str, optional
        If specified, all the fields will be changed to the
        provided byte-order.  Otherwise, the default byte-order is
        used. For all available string specifiers, see `dtype.newbyteorder`.

    See Also
    --------
    numpy.dtype, numpy.typename

    Examples
    --------
    >>> import numpy as np
    >>> np.rec.format_parser(['<f8', '<i4'], ['col1', 'col2'],
    ...                      ['T1', 'T2']).dtype
    dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4')])

    `names` and/or `titles` can be empty lists. If `titles` is an empty list,
    titles will simply not appear. If `names` is empty, default field names
    will be used.

    >>> np.rec.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
    ...                      []).dtype
    dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '<S5')])
    >>> np.rec.format_parser(['<f8', '<i4', '<a5'], [], []).dtype
    dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', 'S5')])

    """

    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)

    def _parseFormats(self, formats, aligned=False):
        """ Parse the field formats """

        if formats is None:
            raise ValueError("Need formats argument")
        if isinstance(formats, list):
            dtype = sb.dtype(
                [
                    ('f{}'.format(i), format_)
                    for i, format_ in enumerate(formats)
                ],
                aligned,
            )
        else:
            dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        if fields is None:
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        keys = dtype.names
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)

    def _setfieldnames(self, names, titles):
        """convert input field names into a list and assign to the _names
        attribute """

        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                names = names.split(',')
            else:
                raise NameError("illegal input names %s" % repr(names))

            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []

        # if the names are not specified, they will be assigned as
        #  "f0, f1, f2,..."
        # if not enough names are specified, they will be assigned as "f[n],
        # f[n+1],..." etc. where n is the number of specified names..."
        self._names += ['f%d' % i for i in range(len(self._names),
                                                 self._nfields)]
        # check for redundant names
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError("Duplicate field names: %s" % _dup)

        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []

        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))

    def _createdtype(self, byteorder):
        dtype = sb.dtype({
            'names': self._names,
            'formats': self._f_formats,
            'offsets': self._offsets,
            'titles': self._titles,
        })
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]
            dtype = dtype.newbyteorder(byteorder)

        self.dtype = dtype


class record(nt.void):
    """A data-type scalar that allows field access as attribute lookup.
    """

    # manually set name and module so that this class's type shows up
    # as numpy.record when printed
    __name__ = 'record'
    __module__ = 'numpy'

    def __repr__(self):
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        return super().__repr__()

    def __str__(self):
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        return super().__str__()

    def __getattribute__(self, attr):
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            obj = self.getfield(*res[:2])
            # if it has fields return a record,
            # otherwise return the object
            try:
                dt = obj.dtype
            except AttributeError:
                #happens if field is Object type
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        else:
            raise AttributeError("'record' object has no "
                    "attribute '%s'" % attr)

    def __setattr__(self, attr, val):
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError("Cannot set '%s' attribute" % attr)
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            return self.setfield(val, *res[:2])
        else:
            if getattr(self, attr, None):
                return nt.void.__setattr__(self, attr, val)
            else:
                raise AttributeError("'record' object has no "
                        "attribute '%s'" % attr)

    def __getitem__(self, indx):
        obj = nt.void.__getitem__(self, indx)

        # copy behavior of record.__getattribute__,
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        else:
            # return a single element
            return obj

    def pprint(self):
        """Pretty-print all fields."""
        # pretty-print all fields
        names = self.dtype.names
        maxlen = max(len(name) for name in names)
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return "\n".join(rows)

# The recarray is almost identical to a standard array (which supports
#   named fields already)  The biggest difference is that it can use
#   attribute-lookup to find the fields and it is constructed using
#   a record.

# If byteorder is given it forces a particular byteorder on all
#  the fields (and any subfields)


@set_module("numpy.rec")
class recarray(ndarray):
    """Construct an ndarray that allows field access using attributes.

    Arrays may have a data-types containing fields, analogous
    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
    where each entry in the array is a pair of ``(int, float)``.  Normally,
    these attributes are accessed using dictionary lookups such as ``arr['x']``
    and ``arr['y']``.  Record arrays allow the fields to be accessed as members
    of the array, using ``arr.x`` and ``arr.y``.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    dtype : data-type, optional
        The desired data-type.  By default, the data-type is determined
        from `formats`, `names`, `titles`, `aligned` and `byteorder`.
    formats : list of data-types, optional
        A list containing the data-types for the different columns, e.g.
        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
        convention of using types directly, i.e. ``(int, float, int)``.
        Note that `formats` must be a list, not a tuple.
        Given that `formats` is somewhat limited, we recommend specifying
        `dtype` instead.
    names : tuple of str, optional
        The name of each column, e.g. ``('x', 'y', 'z')``.
    buf : buffer, optional
        By default, a new array is created of the given shape and data-type.
        If `buf` is specified and is an object exposing the buffer interface,
        the array will use the memory from the existing buffer.  In this case,
        the `offset` and `strides` keywords are available.

    Other Parameters
    ----------------
    titles : tuple of str, optional
        Aliases for column names.  For example, if `names` were
        ``('x', 'y', 'z')`` and `titles` is
        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
    byteorder : {'<', '>', '='}, optional
        Byte-order for all fields.
    aligned : bool, optional
        Align the fields in memory as the C-compiler would.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    offset : int, optional
        Start reading buffer (`buf`) from this offset onwards.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    rec : recarray
        Empty array of the given shape and type.

    See Also
    --------
    numpy.rec.fromrecords : Construct a record array from data.
    numpy.record : fundamental data-type for `recarray`.
    numpy.rec.format_parser : determine data-type from formats, names, titles.

    Notes
    -----
    This constructor can be compared to ``empty``: it creates a new record
    array but does not fill it with data.  To create a record array from data,
    use one of the following methods:

    1. Create a standard ndarray and convert it to a record array,
       using ``arr.view(np.recarray)``
    2. Use the `buf` keyword.
    3. Use `np.rec.fromrecords`.

    Examples
    --------
    Create an array with two fields, ``x`` and ``y``:

    >>> import numpy as np
    >>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', '<f8'), ('y', '<i8')])
    >>> x
    array([(1., 2), (3., 4)], dtype=[('x', '<f8'), ('y', '<i8')])

    >>> x['x']
    array([1., 3.])

    View the array as a record array:

    >>> x = x.view(np.recarray)

    >>> x.x
    array([1., 3.])

    >>> x.y
    array([2, 4])

    Create a new, empty record array:

    >>> np.recarray((2,),
    ... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP
    rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),
           (3471280, 1.2134086255804012e-316, 0)],
          dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])

    """

    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):

        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder
            ).dtype

        if buf is None:
            self = ndarray.__new__(
                subtype, shape, (record, descr), order=order
            )
        else:
            self = ndarray.__new__(
                subtype, shape, (record, descr), buffer=buf,
                offset=offset, strides=strides, order=order
            )
        return self

    def __array_finalize__(self, obj):
        if self.dtype.type is not record and self.dtype.names is not None:
            # if self.dtype is not np.record, invoke __setattr__ which will
            # convert it to a record if it is a void dtype.
            self.dtype = self.dtype

    def __getattribute__(self, attr):
        # See if ndarray has this attr, and return it if so. (note that this
        # means a field with the same name as an ndarray attr cannot be
        # accessed by attribute).
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass

        # look for a field with this name
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError("recarray has no attribute %s" % attr) from e
        obj = self.getfield(*res)

        # At this point obj will always be a recarray, since (see
        # PyArray_GetField) the type of obj is inherited. Next, if obj.dtype is
        # non-structured, convert it to an ndarray. Then if obj is structured
        # with void type convert it to the same dtype.type (eg to preserve
        # numpy.record type if present), since nested structured fields do not
        # inherit type. Don't do this for non-void structures though.
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)

    # Save the dictionary.
    # If the attr is a field name and not in the saved dictionary
    # Undo any "setting" of the attribute and do a setfield
    # Thus, you can't create attributes on-the-fly that are field names.
    def __setattr__(self, attr, val):

        # Automatically convert (void) structured types to records
        # (but not non-void structures, subarrays, or non-structured voids)
        if (
            attr == 'dtype' and
            issubclass(val.type, nt.void) and
            val.names is not None
        ):
            val = sb.dtype((record, val))

        newattr = attr not in self.__dict__
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                return ret
            if newattr:
                # We just added this one or this setattr worked on an
                # internal attribute.
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                "record array has no attribute %s" % attr
            ) from e
        return self.setfield(val, *res)

    def __getitem__(self, indx):
        obj = super().__getitem__(indx)

        # copy behavior of getattr, except that here
        # we might also be returning a single element
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                obj = obj.view(type(self))
                if issubclass(obj.dtype.type, nt.void):
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=ndarray)
        else:
            # return a single element
            return obj

    def __repr__(self):

        repr_dtype = self.dtype
        if (
            self.dtype.type is record or
            not issubclass(self.dtype.type, nt.void)
        ):
            # If this is a full record array (has numpy.record dtype),
            # or if it has a scalar (non-void) dtype with no records,
            # represent it using the rec.array function. Since rec.array
            # converts dtype to a numpy.record for us, convert back
            # to non-record before printing
            if repr_dtype.type is record:
                repr_dtype = sb.dtype((nt.void, repr_dtype))
            prefix = "rec.array("
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            # otherwise represent it using np.array plus a view
            # This should only happen if the user is playing
            # strange games with dtypes.
            prefix = "array("
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'

        # get data/shape string. logic taken from numeric.array_repr
        if self.size > 0 or self.shape == (0,):
            lst = sb.array2string(
                self, separator=', ', prefix=prefix, suffix=',')
        else:
            # show zero-length shape unless it is (0,)
            lst = "[], shape=%s" % (repr(self.shape),)

        lf = '\n'+' '*len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf  # trailing space
        return fmt % (lst, lf, repr_dtype)

    def field(self, attr, val=None):
        if isinstance(attr, int):
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]

        fielddict = ndarray.__getattribute__(self, 'dtype').fields

        res = fielddict[attr][:2]

        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj
            return obj.view(ndarray)
        else:
            return self.setfield(val, *res)


def _deprecate_shape_0_as_None(shape):
    if shape == 0:
        warnings.warn(
            "Passing `shape=0` to have the shape be inferred is deprecated, "
            "and in future will be equivalent to `shape=(0,)`. To infer "
            "the shape and suppress this warning, pass `shape=None` instead.",
            FutureWarning, stacklevel=3)
        return None
    else:
        return shape


@set_module("numpy.rec")
def fromarrays(arrayList, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    """Create a record array from a (flat) list of arrays

    Parameters
    ----------
    arrayList : list or tuple
        List of array-like objects (such as lists, tuples,
        and ndarrays).
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of the resulting array. If not provided, inferred from
        ``arrayList[0]``.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.rec.format_parser` to construct a dtype. See that function for
        detailed documentation.

    Returns
    -------
    np.recarray
        Record array consisting of given arrayList columns.

    Examples
    --------
    >>> x1=np.array([1,2,3,4])
    >>> x2=np.array(['a','dd','xyz','12'])
    >>> x3=np.array([1.1,2,3,4])
    >>> r = np.rec.fromarrays([x1,x2,x3],names='a,b,c')
    >>> print(r[1])
    (2, 'dd', 2.0) # may vary
    >>> x1[1]=34
    >>> r.a
    array([1, 2, 3, 4])

    >>> x1 = np.array([1, 2, 3, 4])
    >>> x2 = np.array(['a', 'dd', 'xyz', '12'])
    >>> x3 = np.array([1.1, 2, 3,4])
    >>> r = np.rec.fromarrays(
    ...     [x1, x2, x3],
    ...     dtype=np.dtype([('a', np.int32), ('b', 'S3'), ('c', np.float32)]))
    >>> r
    rec.array([(1, b'a', 1.1), (2, b'dd', 2. ), (3, b'xyz', 3. ),
               (4, b'12', 4. )],
              dtype=[('a', '<i4'), ('b', 'S3'), ('c', '<f4')])
    """

    arrayList = [sb.asarray(x) for x in arrayList]

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        shape = arrayList[0].shape
    elif isinstance(shape, int):
        shape = (shape,)

    if formats is None and dtype is None:
        # go through each object in the list to see if it is an ndarray
        # and determine the formats.
        formats = [obj.dtype for obj in arrayList]

    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    _names = descr.names

    # Determine shape from data-type.
    if len(descr) != len(arrayList):
        raise ValueError("mismatch between the number of fields "
                "and the number of arrays")

    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]

    _array = recarray(shape, descr)

    # populate the record array (makes a copy)
    for k, obj in enumerate(arrayList):
        nn = descr[k].ndim
        testshape = obj.shape[:obj.ndim - nn]
        name = _names[k]
        if testshape != shape:
            raise ValueError(f'array-shape mismatch in array {k} ("{name}")')

        _array[name] = obj

    return _array


@set_module("numpy.rec")
def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    """Create a recarray from a list of records in text form.

    Parameters
    ----------
    recList : sequence
        data in the same field may be heterogeneous - they will be promoted
        to the highest data type.
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        shape of each array.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.

        If both `formats` and `dtype` are None, then this will auto-detect
        formats. Use list of tuples rather than list of lists for faster
        processing.

    Returns
    -------
    np.recarray
        record array consisting of given recList rows.

    Examples
    --------
    >>> r=np.rec.fromrecords([(456,'dbe',1.2),(2,'de',1.3)],
    ... names='col1,col2,col3')
    >>> print(r[0])
    (456, 'dbe', 1.2)
    >>> r.col1
    array([456,   2])
    >>> r.col2
    array(['dbe', 'de'], dtype='<U3')
    >>> import pickle
    >>> pickle.loads(pickle.dumps(r))
    rec.array([(456, 'dbe', 1.2), (  2, 'de', 1.3)],
              dtype=[('col1', '<i8'), ('col2', '<U3'), ('col3', '<f8')])
    """

    if formats is None and dtype is None:  # slower
        obj = sb.array(recList, dtype=object)
        arrlist = [
            sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])
        ]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned, byteorder=byteorder)

    if dtype is not None:
        descr = sb.dtype((record, dtype))
    else:
        descr = format_parser(
            formats, names, titles, aligned, byteorder
        ).dtype

    try:
        retval = sb.array(recList, dtype=descr)
    except (TypeError, ValueError):
        # NumPy 1.19.0, 2020-01-01
        shape = _deprecate_shape_0_as_None(shape)
        if shape is None:
            shape = len(recList)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError("Can only deal with 1-d array.")
        _array = recarray(shape, descr)
        for k in range(_array.size):
            _array[k] = tuple(recList[k])
        # list of lists instead of list of tuples ?
        # 2018-02-07, 1.14.1
        warnings.warn(
            "fromrecords expected a list of tuples, may have received a list "
            "of lists instead. In the future that will raise an error",
            FutureWarning, stacklevel=2)
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape

    res = retval.view(recarray)

    return res


@set_module("numpy.rec")
def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    r"""Create a record array from binary data

    Note that despite the name of this function it does not accept `str`
    instances.

    Parameters
    ----------
    datastring : bytes-like
        Buffer of binary data
    dtype : data-type, optional
        Valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the buffer to start reading from.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.


    Returns
    -------
    np.recarray
        Record array view into the data in datastring. This will be readonly
        if `datastring` is readonly.

    See Also
    --------
    numpy.frombuffer

    Examples
    --------
    >>> a = b'\x01\x02\x03abc'
    >>> np.rec.fromstring(a, dtype='u1,u1,u1,S3')
    rec.array([(1, 2, 3, b'abc')],
            dtype=[('f0', 'u1'), ('f1', 'u1'), ('f2', 'u1'), ('f3', 'S3')])

    >>> grades_dtype = [('Name', (np.str_, 10)), ('Marks', np.float64),
    ...                 ('GradeLevel', np.int32)]
    >>> grades_array = np.array([('Sam', 33.3, 3), ('Mike', 44.4, 5),
    ...                         ('Aadi', 66.6, 6)], dtype=grades_dtype)
    >>> np.rec.fromstring(grades_array.tobytes(), dtype=grades_dtype)
    rec.array([('Sam', 33.3, 3), ('Mike', 44.4, 5), ('Aadi', 66.6, 6)],
            dtype=[('Name', '<U10'), ('Marks', '<f8'), ('GradeLevel', '<i4')])

    >>> s = '\x01\x02\x03abc'
    >>> np.rec.fromstring(s, dtype='u1,u1,u1,S3')
    Traceback (most recent call last):
       ...
    TypeError: a bytes-like object is required, not 'str'
    """

    if dtype is None and formats is None:
        raise TypeError("fromstring() needs a 'dtype' or 'formats' argument")

    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype

    itemsize = descr.itemsize

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)

    if shape in (None, -1):
        shape = (len(datastring) - offset) // itemsize

    _array = recarray(shape, descr, buf=datastring, offset=offset)
    return _array

def get_remaining_size(fd):
    pos = fd.tell()
    try:
        fd.seek(0, 2)
        return fd.tell() - pos
    finally:
        fd.seek(pos, 0)


@set_module("numpy.rec")
def fromfile(fd, dtype=None, shape=None, offset=0, formats=None,
             names=None, titles=None, aligned=False, byteorder=None):
    """Create an array from binary file data

    Parameters
    ----------
    fd : str or file type
        If file is a string or a path-like object then that file is opened,
        else it is assumed to be a file object. The file object must
        support random access (i.e. it must have tell and seek methods).
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        shape of each array.
    offset : int, optional
        Position in the file to start reading from.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation

    Returns
    -------
    np.recarray
        record array consisting of data enclosed in file.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> a = np.empty(10,dtype='f8,i4,a5')
    >>> a[5] = (0.5,10,'abcde')
    >>>
    >>> fd=TemporaryFile()
    >>> a = a.view(a.dtype.newbyteorder('<'))
    >>> a.tofile(fd)
    >>>
    >>> _ = fd.seek(0)
    >>> r=np.rec.fromfile(fd, formats='f8,i4,a5', shape=10,
    ... byteorder='<')
    >>> print(r[5])
    (0.5, 10, b'abcde')
    >>> r.shape
    (10,)
    """

    if dtype is None and formats is None:
        raise TypeError("fromfile() needs a 'dtype' or 'formats' argument")

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        shape = (-1,)
    elif isinstance(shape, int):
        shape = (shape,)

    if hasattr(fd, 'readinto'):
        # GH issue 2504. fd supports io.RawIOBase or io.BufferedIOBase
        # interface. Example of fd: gzip, BytesIO, BufferedReader
        # file already opened
        ctx = nullcontext(fd)
    else:
        # open file
        ctx = open(os.fspath(fd), 'rb')

    with ctx as fd:
        if offset > 0:
            fd.seek(offset, 1)
        size = get_remaining_size(fd)

        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder
            ).dtype

        itemsize = descr.itemsize

        shapeprod = sb.array(shape).prod(dtype=nt.intp)
        shapesize = shapeprod * itemsize
        if shapesize < 0:
            shape = list(shape)
            shape[shape.index(-1)] = size // -shapesize
            shape = tuple(shape)
            shapeprod = sb.array(shape).prod(dtype=nt.intp)

        nbytes = shapeprod * itemsize

        if nbytes > size:
            raise ValueError(
                    "Not enough bytes left in file for specified "
                    "shape and type."
                )

        # create the array
        _array = recarray(shape, descr)
        nbytesread = fd.readinto(_array.data)
        if nbytesread != nbytes:
            raise OSError("Didn't read as many bytes as expected")

    return _array


@set_module("numpy.rec")
def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
          names=None, titles=None, aligned=False, byteorder=None, copy=True):
    """
    Construct a record array from a wide-variety of objects.

    A general-purpose record array constructor that dispatches to the
    appropriate `recarray` creation function based on the inputs (see Notes).

    Parameters
    ----------
    obj : any
        Input object. See Notes for details on how various input types are
        treated.
    dtype : data-type, optional
        Valid dtype for array.
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the file or buffer to start reading from.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.
    copy : bool, optional
        Whether to copy the input object (True), or to use a reference instead.
        This option only applies when the input is an ndarray or recarray.
        Defaults to True.

    Returns
    -------
    np.recarray
        Record array created from the specified object.

    Notes
    -----
    If `obj` is ``None``, then call the `~numpy.recarray` constructor. If
    `obj` is a string, then call the `fromstring` constructor. If `obj` is a
    list or a tuple, then if the first object is an `~numpy.ndarray`, call
    `fromarrays`, otherwise call `fromrecords`. If `obj` is a
    `~numpy.recarray`, then make a copy of the data in the recarray
    (if ``copy=True``) and use the new formats, names, and titles. If `obj`
    is a file, then call `fromfile`. Finally, if obj is an `ndarray`, then
    return ``obj.view(recarray)``, making a copy of the data if ``copy=True``.

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.rec.array(a)
    rec.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]],
              dtype=int64)

    >>> b = [(1, 1), (2, 4), (3, 9)]
    >>> c = np.rec.array(b, formats = ['i2', 'f2'], names = ('x', 'y'))
    >>> c
    rec.array([(1, 1.), (2, 4.), (3, 9.)],
              dtype=[('x', '<i2'), ('y', '<f2')])

    >>> c.x
    array([1, 2, 3], dtype=int16)

    >>> c.y
    array([1.,  4.,  9.], dtype=float16)

    >>> r = np.rec.array(['abc','def'], names=['col1','col2'])
    >>> print(r.col1)
    abc

    >>> r.col1
    array('abc', dtype='<U3')

    >>> r.col2
    array('def', dtype='<U3')
    """

    if ((isinstance(obj, (type(None), str)) or hasattr(obj, 'readinto')) and
           formats is None and dtype is None):
        raise ValueError("Must define formats (or dtype) if object is "
                         "None, string, or an open file")

    kwds = {}
    if dtype is not None:
        dtype = sb.dtype(dtype)
    elif formats is not None:
        dtype = format_parser(formats, names, titles,
                              aligned, byteorder).dtype
    else:
        kwds = {'formats': formats,
                'names': names,
                'titles': titles,
                'aligned': aligned,
                'byteorder': byteorder
                }

    if obj is None:
        if shape is None:
            raise ValueError("Must define a shape if obj is None")
        return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)

    elif isinstance(obj, bytes):
        return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (tuple, list)):
            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
        else:
            return fromarrays(obj, dtype=dtype, shape=shape, **kwds)

    elif isinstance(obj, recarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new

    elif hasattr(obj, 'readinto'):
        return fromfile(obj, dtype=dtype, shape=shape, offset=offset)

    elif isinstance(obj, ndarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new.view(recarray)

    else:
        interface = getattr(obj, "__array_interface__", None)
        if interface is None or not isinstance(interface, dict):
            raise ValueError("Unknown input type")
        obj = sb.array(obj)
        if dtype is not None and (obj.dtype != dtype):
            obj = obj.view(dtype)
        return obj.view(recarray)
<|endoftext|>"""
A place for internal code

Some things are more easily handled Python.

"""
import ast
import math
import re
import sys
import warnings

from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types, StringDType
from numpy import _NoValue
try:
    import ctypes
except ImportError:
    ctypes = None

IS_PYPY = sys.implementation.name == 'pypy'

if sys.byteorder == 'little':
    _nbo = '<'
else:
    _nbo = '>'

def _makenames_list(adict, align):
    allfields = []

    for fname, obj in adict.items():
        n = len(obj)
        if not isinstance(obj, tuple) or n not in (2, 3):
            raise ValueError("entry not a 2- or 3- tuple")
        if n > 2 and obj[2] == fname:
            continue
        num = int(obj[1])
        if num < 0:
            raise ValueError("invalid offset.")
        format = dtype(obj[0], align=align)
        if n > 2:
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    # sort by offsets
    allfields.sort(key=lambda x: x[2])
    names = [x[0] for x in allfields]
    formats = [x[1] for x in allfields]
    offsets = [x[2] for x in allfields]
    titles = [x[3] for x in allfields]

    return names, formats, offsets, titles

# Called in PyArray_DescrConverter function when
#  a dictionary without "names" and "formats"
#  fields is used as a data-type descriptor.
def _usefields(adict, align):
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
        names, formats, offsets, titles = _makenames_list(adict, align)
    else:
        formats = []
        offsets = []
        titles = []
        for name in names:
            res = adict[name]
            formats.append(res[0])
            offsets.append(res[1])
            if len(res) > 2:
                titles.append(res[2])
            else:
                titles.append(None)

    return dtype({"names": names,
                  "formats": formats,
                  "offsets": offsets,
                  "titles": titles}, align)


# construct an array_protocol descriptor list
#  from the fields attribute of a descriptor
# This calls itself recursively but should eventually hit
#  a descriptor that has no fields and then return
#  a simple typestring

def _array_descr(descriptor):
    fields = descriptor.fields
    if fields is None:
        subdtype = descriptor.subdtype
        if subdtype is None:
            if descriptor.metadata is None:
                return descriptor.str
            else:
                new = descriptor.metadata.copy()
                if new:
                    return (descriptor.str, new)
                else:
                    return descriptor.str
        else:
            return (_array_descr(subdtype[0]), subdtype[1])

    names = descriptor.names
    ordered_fields = [fields[x] + (x,) for x in names]
    result = []
    offset = 0
    for field in ordered_fields:
        if field[1] > offset:
            num = field[1] - offset
            result.append(('', f'|V{num}'))
            offset += num
        elif field[1] < offset:
            raise ValueError(
                "dtype.descr is not defined for types with overlapping or "
                "out-of-order fields")
        if len(field) > 3:
            name = (field[2], field[3])
        else:
            name = field[2]
        if field[0].subdtype:
            tup = (name, _array_descr(field[0].subdtype[0]),
                   field[0].subdtype[1])
        else:
            tup = (name, _array_descr(field[0]))
        offset += field[0].itemsize
        result.append(tup)

    if descriptor.itemsize > offset:
        num = descriptor.itemsize - offset
        result.append(('', f'|V{num}'))

    return result


# format_re was originally from numarray by J. Todd Miller

format_re = re.compile(r'(?P<order1>[<>|=]?)'
                       r'(?P<repeats> *[(]?[ ,0-9]*[)]? *)'
                       r'(?P<order2>[<>|=]?)'
                       r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)')
sep_re = re.compile(r'\s*,\s*')
space_re = re.compile(r'\s+$')

# astr is a string (perhaps comma separated)

_convorder = {'=': _nbo}

def _commastring(astr):
    startindex = 0
    result = []
    islist = False
    while startindex < len(astr):
        mo = format_re.match(astr, pos=startindex)
        try:
            (order1, repeats, order2, dtype) = mo.groups()
        except (TypeError, AttributeError):
            raise ValueError(
                f'format number {len(result)+1} of "{astr}" is not recognized'
                ) from None
        startindex = mo.end()
        # Separator or ending padding
        if startindex < len(astr):
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    raise ValueError(
                        'format number %d of "%s" is not recognized' %
                        (len(result)+1, astr))
                startindex = mo.end()
                islist = True

        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if (order1 != order2):
                raise ValueError(
                    'inconsistent byte-order specification %s and %s' %
                    (order1, order2))
            order = order1

        if order in ('|', '=', _nbo):
            order = ''
        dtype = order + dtype
        if repeats == '':
            newitem = dtype
        else:
            if (repeats[0] == "(" and repeats[-1] == ")"
                    and repeats[1:-1].strip() != ""
                    and "," not in repeats):
                warnings.warn(
                    'Passing in a parenthesized single number for repeats '
                    'is deprecated; pass either a single number or indicate '
                    'a tuple with a comma, like "(2,)".', DeprecationWarning,
                    stacklevel=2)
            newitem = (dtype, ast.literal_eval(repeats))

        result.append(newitem)

    return result if islist else result[0]

class dummy_ctype:

    def __init__(self, cls):
        self._cls = cls

    def __mul__(self, other):
        return self

    def __call__(self, *other):
        return self._cls(other)

    def __eq__(self, other):
        return self._cls == other._cls

    def __ne__(self, other):
        return self._cls != other._cls

def _getintp_ctype():
    val = _getintp_ctype.cache
    if val is not None:
        return val
    if ctypes is None:
        import numpy as np
        val = dummy_ctype(np.intp)
    else:
        char = dtype('n').char
        if char == 'i':
            val = ctypes.c_int
        elif char == 'l':
            val = ctypes.c_long
        elif char == 'q':
            val = ctypes.c_longlong
        else:
            val = ctypes.c_long
    _getintp_ctype.cache = val
    return val


_getintp_ctype.cache = None

# Used for .ctypes attribute of ndarray

class _missing_ctypes:
    def cast(self, num, obj):
        return num.value

    class c_void_p:
        def __init__(self, ptr):
            self.value = ptr


class _ctypes:
    def __init__(self, array, ptr=None):
        self._arr = array

        if ctypes:
            self._ctypes = ctypes
            self._data = self._ctypes.c_void_p(ptr)
        else:
            # fake a pointer-like object that holds onto the reference
            self._ctypes = _missing_ctypes()
            self._data = self._ctypes.c_void_p(ptr)
            self._data._objects = array

        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False

    def data_as(self, obj):
        """
        Return the data pointer cast to a particular c-types object.
        For example, calling ``self._as_parameter_`` is equivalent to
        ``self.data_as(ctypes.c_void_p)``. Perhaps you want to use
        the data as a pointer to a ctypes array of floating-point data:
        ``self.data_as(ctypes.POINTER(ctypes.c_double))``.

        The returned pointer will keep a reference to the array.
        """
        # _ctypes.cast function causes a circular reference of self._data in
        # self._data._objects. Attributes of self._data cannot be released
        # until gc.collect is called. Make a copy of the pointer first then
        # let it hold the array reference. This is a workaround to circumvent
        # the CPython bug https://bugs.python.org/issue12836.
        ptr = self._ctypes.cast(self._data, obj)
        ptr._arr = self._arr
        return ptr

    def shape_as(self, obj):
        """
        Return the shape tuple as an array of some other c-types
        type. For example: ``self.shape_as(ctypes.c_short)``.
        """
        if self._zerod:
            return None
        return (obj*self._arr.ndim)(*self._arr.shape)

    def strides_as(self, obj):
        """
        Return the strides tuple as an array of some other
        c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.
        """
        if self._zerod:
            return None
        return (obj*self._arr.ndim)(*self._arr.strides)

    @property
    def data(self):
        """
        A pointer to the memory area of the array as a Python integer.
        This memory area may contain data that is not aligned, or not in
        correct byte-order. The memory area may not even be writeable.
        The array flags and data-type of this array should be respected
        when passing this attribute to arbitrary C-code to avoid trouble
        that can include Python crashing. User Beware! The value of this
        attribute is exactly the same as:
        ``self._array_interface_['data'][0]``.

        Note that unlike ``data_as``, a reference won't be kept to the array:
        code like ``ctypes.c_void_p((a + b).ctypes.data)`` will result in a
        pointer to a deallocated array, and should be spelt
        ``(a + b).ctypes.data_as(ctypes.c_void_p)``
        """
        return self._data.value

    @property
    def shape(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the C-integer corresponding to ``dtype('p')`` on this
        platform (see `~numpy.ctypeslib.c_intp`). This base-type could be
        `ctypes.c_int`, `ctypes.c_long`, or `ctypes.c_longlong` depending on
        the platform. The ctypes array contains the shape of
        the underlying array.
        """
        return self.shape_as(_getintp_ctype())

    @property
    def strides(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the same as for the shape attribute. This ctypes
        array contains the strides information from the underlying array.
        This strides information is important for showing how many bytes
        must be jumped to get to the next element in the array.
        """
        return self.strides_as(_getintp_ctype())

    @property
    def _as_parameter_(self):
        """
        Overrides the ctypes semi-magic method

        Enables `c_func(some_array.ctypes)`
        """
        return self.data_as(ctypes.c_void_p)

    # Numpy 1.21.0, 2021-05-18

    def get_data(self):
        """Deprecated getter for the `_ctypes.data` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_data" is deprecated. Use "data" instead',
                      DeprecationWarning, stacklevel=2)
        return self.data

    def get_shape(self):
        """Deprecated getter for the `_ctypes.shape` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_shape" is deprecated. Use "shape" instead',
                      DeprecationWarning, stacklevel=2)
        return self.shape

    def get_strides(self):
        """Deprecated getter for the `_ctypes.strides` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_strides" is deprecated. Use "strides" instead',
                      DeprecationWarning, stacklevel=2)
        return self.strides

    def get_as_parameter(self):
        """Deprecated getter for the `_ctypes._as_parameter_` property.

        .. deprecated:: 1.21
        """
        warnings.warn(
            '"get_as_parameter" is deprecated. Use "_as_parameter_" instead',
            DeprecationWarning, stacklevel=2,
        )
        return self._as_parameter_


def _newnames(datatype, order):
    """
    Given a datatype and an order object, return a new names tuple, with the
    order indicated
    """
    oldnames = datatype.names
    nameslist = list(oldnames)
    if isinstance(order, str):
        order = [order]
    seen = set()
    if isinstance(order, (list, tuple)):
        for name in order:
            try:
                nameslist.remove(name)
            except ValueError:
                if name in seen:
                    raise ValueError(f"duplicate field name: {name}") from None
                else:
                    raise ValueError(f"unknown field name: {name}") from None
            seen.add(name)
        return tuple(list(order) + nameslist)
    raise ValueError(f"unsupported order value: {order}")

def _copy_fields(ary):
    """Return copy of structured array with padding between fields removed.

    Parameters
    ----------
    ary : ndarray
       Structured array from which to remove padding bytes

    Returns
    -------
    ary_copy : ndarray
       Copy of ary with padding bytes removed
    """
    dt = ary.dtype
    copy_dtype = {'names': dt.names,
                  'formats': [dt.fields[name][0] for name in dt.names]}
    return array(ary, dtype=copy_dtype, copy=True)

def _promote_fields(dt1, dt2):
    """ Perform type promotion for two structured dtypes.

    Parameters
    ----------
    dt1 : structured dtype
        First dtype.
    dt2 : structured dtype
        Second dtype.

    Returns
    -------
    out : dtype
        The promoted dtype

    Notes
    -----
    If one of the inputs is aligned, the result will be.  The titles of
    both descriptors must match (point to the same field).
    """
    # Both must be structured and have the same names in the same order
    if (dt1.names is None or dt2.names is None) or dt1.names != dt2.names:
        raise DTypePromotionError(
                f"field names `{dt1.names}` and `{dt2.names}` mismatch.")

    # if both are identical, we can (maybe!) just return the same dtype.
    identical = dt1 is dt2
    new_fields = []
    for name in dt1.names:
        field1 = dt1.fields[name]
        field2 = dt2.fields[name]
        new_descr = promote_types(field1[0], field2[0])
        identical = identical and new_descr is field1[0]

        # Check that the titles match (if given):
        if field1[2:] != field2[2:]:
            raise DTypePromotionError(
                    f"field titles of field '{name}' mismatch")
        if len(field1) == 2:
            new_fields.append((name, new_descr))
        else:
            new_fields.append(((field1[2], name), new_descr))

    res = dtype(new_fields, align=dt1.isalignedstruct or dt2.isalignedstruct)

    # Might as well preserve identity (and metadata) if the dtype is identical
    # and the itemsize, offsets are also unmodified.  This could probably be
    # sped up, but also probably just be removed entirely.
    if identical and res.itemsize == dt1.itemsize:
        for name in dt1.names:
            if dt1.fields[name][1] != res.fields[name][1]:
                return res  # the dtype changed.
        return dt1

    return res


def _getfield_is_safe(oldtype, newtype, offset):
    """ Checks safety of getfield for object arrays.

    As in _view_is_safe, we need to check that memory containing objects is not
    reinterpreted as a non-object datatype and vice versa.

    Parameters
    ----------
    oldtype : data-type
        Data type of the original ndarray.
    newtype : data-type
        Data type of the field being accessed by ndarray.getfield
    offset : int
        Offset of the field being accessed by ndarray.getfield

    Raises
    ------
    TypeError
        If the field access is invalid

    """
    if newtype.hasobject or oldtype.hasobject:
        if offset == 0 and newtype == oldtype:
            return
        if oldtype.names is not None:
            for name in oldtype.names:
                if (oldtype.fields[name][1] == offset and
                        oldtype.fields[name][0] == newtype):
                    return
        raise TypeError("Cannot get/set field of an object array")
    return

def _view_is_safe(oldtype, newtype):
    """ Checks safety of a view involving object arrays, for example when
    doing::

        np.zeros(10, dtype=oldtype).view(newtype)

    Parameters
    ----------
    oldtype : data-type
        Data type of original ndarray
    newtype : data-type
        Data type of the view

    Raises
    ------
    TypeError
        If the new type is incompatible with the old type.

    """

    # if the types are equivalent, there is no problem.
    # for example: dtype((np.record, 'i4,i4')) == dtype((np.void, 'i4,i4'))
    if oldtype == newtype:
        return

    if newtype.hasobject or oldtype.hasobject:
        raise TypeError("Cannot change data-type for array of references.")
    return


# Given a string containing a PEP 3118 format specifier,
# construct a NumPy dtype

_pep3118_native_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'h',
    'H': 'H',
    'i': 'i',
    'I': 'I',
    'l': 'l',
    'L': 'L',
    'q': 'q',
    'Q': 'Q',
    'e': 'e',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())

_pep3118_standard_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'i2',
    'H': 'u2',
    'i': 'i4',
    'I': 'u4',
    'l': 'i4',
    'L': 'u4',
    'q': 'i8',
    'Q': 'u8',
    'e': 'f2',
    'f': 'f',
    'd': 'd',
    'Zf': 'F',
    'Zd': 'D',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())

_pep3118_unsupported_map = {
    'u': 'UCS-2 strings',
    '&': 'pointers',
    't': 'bitfields',
    'X': 'function pointers',
}

class _Stream:
    def __init__(self, s):
        self.s = s
        self.byteorder = '@'

    def advance(self, n):
        res = self.s[:n]
        self.s = self.s[n:]
        return res

    def consume(self, c):
        if self.s[:len(c)] == c:
            self.advance(len(c))
            return True
        return False

    def consume_until(self, c):
        if callable(c):
            i = 0
            while i < len(self.s) and not c(self.s[i]):
                i = i + 1
            return self.advance(i)
        else:
            i = self.s.index(c)
            res = self.advance(i)
            self.advance(len(c))
            return res

    @property
    def next(self):
        return self.s[0]

    def __bool__(self):
        return bool(self.s)


def _dtype_from_pep3118(spec):
    stream = _Stream(spec)
    dtype, align = __dtype_from_pep3118(stream, is_subdtype=False)
    return dtype

def __dtype_from_pep3118(stream, is_subdtype):
    field_spec = dict(
        names=[],
        formats=[],
        offsets=[],
        itemsize=0
    )
    offset = 0
    common_alignment = 1
    is_padding = False

    # Parse spec
    while stream:
        value = None

        # End of structure, bail out to upper level
        if stream.consume('}'):
            break

        # Sub-arrays (1)
        shape = None
        if stream.consume('('):
            shape = stream.consume_until(')')
            shape = tuple(map(int, shape.split(',')))

        # Byte order
        if stream.next in ('@', '=', '<', '>', '^', '!'):
            byteorder = stream.advance(1)
            if byteorder == '!':
                byteorder = '>'
            stream.byteorder = byteorder

        # Byte order characters also control native vs. standard type sizes
        if stream.byteorder in ('@', '^'):
            type_map = _pep3118_native_map
            type_map_chars = _pep3118_native_typechars
        else:
            type_map = _pep3118_standard_map
            type_map_chars = _pep3118_standard_typechars

        # Item sizes
        itemsize_str = stream.consume_until(lambda c: not c.isdigit())
        if itemsize_str:
            itemsize = int(itemsize_str)
        else:
            itemsize = 1

        # Data types
        is_padding = False

        if stream.consume('T{'):
            value, align = __dtype_from_pep3118(
                stream, is_subdtype=True)
        elif stream.next in type_map_chars:
            if stream.next == 'Z':
                typechar = stream.advance(2)
            else:
                typechar = stream.advance(1)

            is_padding = (typechar == 'x')
            dtypechar = type_map[typechar]
            if dtypechar in 'USV':
                dtypechar += '%d' % itemsize
                itemsize = 1
            numpy_byteorder = {'@': '=', '^': '='}.get(
                stream.byteorder, stream.byteorder)
            value = dtype(numpy_byteorder + dtypechar)
            align = value.alignment
        elif stream.next in _pep3118_unsupported_map:
            desc = _pep3118_unsupported_map[stream.next]
            raise NotImplementedError(
                "Unrepresentable PEP 3118 data type {!r} ({})"
                .format(stream.next, desc))
        else:
            raise ValueError(
                "Unknown PEP 3118 data type specifier %r" % stream.s
            )

        #
        # Native alignment may require padding
        #
        # Here we assume that the presence of a '@' character implicitly
        # implies that the start of the array is *already* aligned.
        #
        extra_offset = 0
        if stream.byteorder == '@':
            start_padding = (-offset) % align
            intra_padding = (-value.itemsize) % align

            offset += start_padding

            if intra_padding != 0:
                if itemsize > 1 or (shape is not None and _prod(shape) > 1):
                    # Inject internal padding to the end of the sub-item
                    value = _add_trailing_padding(value, intra_padding)
                else:
                    # We can postpone the injection of internal padding,
                    # as the item appears at most once
                    extra_offset += intra_padding

            # Update common alignment
            common_alignment = _lcm(align, common_alignment)

        # Convert itemsize to sub-array
        if itemsize != 1:
            value = dtype((value, (itemsize,)))

        # Sub-arrays (2)
        if shape is not None:
            value = dtype((value, shape))

        # Field name
        if stream.consume(':'):
            name = stream.consume_until(':')
        else:
            name = None

        if not (is_padding and name is None):
            if name is not None and name in field_spec['names']:
                raise RuntimeError(
                    f"Duplicate field name '{name}' in PEP3118 format"
                )
            field_spec['names'].append(name)
            field_spec['formats'].append(value)
            field_spec['offsets'].append(offset)

        offset += value.itemsize
        offset += extra_offset

        field_spec['itemsize'] = offset

    # extra final padding for aligned types
    if stream.byteorder == '@':
        field_spec['itemsize'] += (-offset) % common_alignment

    # Check if this was a simple 1-item type, and unwrap it
    if (field_spec['names'] == [None]
            and field_spec['offsets'][0] == 0
            and field_spec['itemsize'] == field_spec['formats'][0].itemsize
            and not is_subdtype):
        ret = field_spec['formats'][0]
    else:
        _fix_names(field_spec)
        ret = dtype(field_spec)

    # Finished
    return ret, common_alignment

def _fix_names(field_spec):
    """ Replace names which are None with the next unused f%d name """
    names = field_spec['names']
    for i, name in enumerate(names):
        if name is not None:
            continue

        j = 0
        while True:
            name = f'f{j}'
            if name not in names:
                break
            j = j + 1
        names[i] = name

def _add_trailing_padding(value, padding):
    """Inject the specified number of padding bytes at the end of a dtype"""
    if value.fields is None:
        field_spec = dict(
            names=['f0'],
            formats=[value],
            offsets=[0],
            itemsize=value.itemsize
        )
    else:
        fields = value.fields
        names = value.names
        field_spec = dict(
            names=names,
            formats=[fields[name][0] for name in names],
            offsets=[fields[name][1] for name in names],
            itemsize=value.itemsize
        )

    field_spec['itemsize'] += padding
    return dtype(field_spec)

def _prod(a):
    p = 1
    for x in a:
        p *= x
    return p

def _gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('Can only find greatest common divisor of '
                         f'finite arguments, found "{a}" and "{b}"')
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a // _gcd(a, b) * b

def array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs):
    """ Format the error message for when __array_ufunc__ gives up. """
    args_string = ', '.join(['{!r}'.format(arg) for arg in inputs] +
                            ['{}={!r}'.format(k, v)
                             for k, v in kwargs.items()])
    args = inputs + kwargs.get('out', ())
    types_string = ', '.join(repr(type(arg).__name__) for arg in args)
    return ('operand type(s) all returned NotImplemented from '
            '__array_ufunc__({!r}, {!r}, {}): {}'
            .format(ufunc, method, args_string, types_string))


def array_function_errmsg_formatter(public_api, types):
    """ Format the error message for when __array_ufunc__ gives up. """
    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    return ("no implementation found for '{}' on types that implement "
            '__array_function__: {}'.format(func_name, list(types)))


def _ufunc_doc_signature_formatter(ufunc):
    """
    Builds a signature string which resembles PEP 457

    This is used to construct the first line of the docstring
    """

    # input arguments are simple
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        in_args = ', '.join(f'x{i+1}' for i in range(ufunc.nin))

    # output arguments are both keyword or positional
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        out_args = '[, {positional}], / [, out={default}]'.format(
            positional=', '.join(
                'out{}'.format(i+1) for i in range(ufunc.nout)),
            default=repr((None,)*ufunc.nout)
        )

    # keyword only args depend on whether this is a gufunc
    kwargs = (
        ", casting='same_kind'"
        ", order='K'"
        ", dtype=None"
        ", subok=True"
    )

    # NOTE: gufuncs may or may not support the `axis` parameter
    if ufunc.signature is None:
        kwargs = f", where=True{kwargs}[, signature]"
    else:
        kwargs += "[, signature, axes, axis]"

    # join all the parts together
    return '{name}({in_args}{out_args}, *{kwargs})'.format(
        name=ufunc.__name__,
        in_args=in_args,
        out_args=out_args,
        kwargs=kwargs
    )


def npy_ctypes_check(cls):
    # determine if a class comes from ctypes, in order to work around
    # a bug in the buffer protocol for those objects, bpo-10746
    try:
        # ctypes class are new-style, so have an __mro__. This probably fails
        # for ctypes classes with multiple inheritance.
        if IS_PYPY:
            # (..., _ctypes.basics._CData, Bufferable, object)
            ctype_base = cls.__mro__[-3]
        else:
            # # (..., _ctypes._CData, object)
            ctype_base = cls.__mro__[-2]
        # right now, they're part of the _ctypes module
        return '_ctypes' in ctype_base.__module__
    except Exception:
        return False

# used to handle the _NoValue default argument for na_object
# in the C implementation of the __reduce__ method for stringdtype
def _convert_to_stringdtype_kwargs(coerce, na_object=_NoValue):
    if na_object is _NoValue:
        return StringDType(coerce=coerce)
    return StringDType(coerce=coerce, na_object=na_object)
<|endoftext|>__all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'block', 'hstack',
           'stack', 'unstack', 'vstack']

import functools
import itertools
import operator

from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _atleast_1d_dispatcher(*arys):
    return arys


@array_function_dispatch(_atleast_1d_dispatcher)
def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> import numpy as np
    >>> np.atleast_1d(1.0)
    array([1.])

    >>> x = np.arange(9.0).reshape(3,3)
    >>> np.atleast_1d(x)
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.atleast_1d(x) is x
    True

    >>> np.atleast_1d(1, [3, 4])
    (array([1]), array([3, 4]))

    """
    if len(arys) == 1:
        result = asanyarray(arys[0])
        if result.ndim == 0:
            result = result.reshape(1)
        return result
    res = []
    for ary in arys:
        result = asanyarray(ary)
        if result.ndim == 0:
            result = result.reshape(1)
        res.append(result)
    return tuple(res)


def _atleast_2d_dispatcher(*arys):
    return arys


@array_function_dispatch(_atleast_2d_dispatcher)
def atleast_2d(*arys):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> import numpy as np
    >>> np.atleast_2d(3.0)
    array([[3.]])

    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> np.atleast_2d(x).base is x
    True

    >>> np.atleast_2d(1, [1, 2], [[1, 2]])
    (array([[1]]), array([[1, 2]]), array([[1, 2]]))

    """
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def _atleast_3d_dispatcher(*arys):
    return arys


@array_function_dispatch(_atleast_3d_dispatcher)
def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    >>> import numpy as np
    >>> np.atleast_3d(3.0)
    array([[[3.]]])

    >>> x = np.arange(3.0)
    >>> np.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = np.arange(12.0).reshape(4,3)
    >>> np.atleast_3d(x).shape
    (4, 3, 1)
    >>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
    True

    >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
    ...     print(arr, arr.shape) # doctest: +SKIP
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)

    """
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :, _nx.newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, _nx.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def _arrays_for_stack_dispatcher(arrays):
    if not hasattr(arrays, "__getitem__"):
        raise TypeError('arrays to stack must be passed as a "sequence" type '
                        'such as list or tuple.')

    return tuple(arrays)


def _vhstack_dispatcher(tup, *, dtype=None, casting=None):
    return _arrays_for_stack_dispatcher(tup)


@array_function_dispatch(_vhstack_dispatcher)
def vstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length. In the case of a single
        array_like input, it will be treated as a sequence of arrays; i.e.,
        each element along the zeroth axis is treated as a separate array.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.vstack((a,b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[4], [5], [6]])
    >>> np.vstack((a,b))
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])

    """
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)


@array_function_dispatch(_vhstack_dispatcher)
def hstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length. In the case of a single
        array_like input, it will be treated as a sequence of arrays; i.e.,
        each element along the zeroth axis is treated as a separate array.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    hsplit : Split an array into multiple sub-arrays
             horizontally (column-wise).
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((4,5,6))
    >>> np.hstack((a,b))
    array([1, 2, 3, 4, 5, 6])
    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[4],[5],[6]])
    >>> np.hstack((a,b))
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    else:
        return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)


def _stack_dispatcher(arrays, axis=None, out=None, *,
                      dtype=None, casting=None):
    arrays = _arrays_for_stack_dispatcher(arrays)
    if out is not None:
        # optimize for the typical case where only arrays is provided
        arrays = list(arrays)
        arrays.append(out)
    return arrays


@array_function_dispatch(_stack_dispatcher)
def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    """
    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of ndarrays
        Each array must have the same shape. In the case of a single ndarray
        array_like input, it will be treated as a sequence of arrays; i.e.,
        each element along the zeroth axis is treated as a separate array.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24


    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    block : Assemble an nd-array from nested lists of blocks.
    split : Split array into a list of multiple sub-arrays of equal size.
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> arrays = [rng.normal(size=(3,4)) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)

    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)

    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.stack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.stack((a, b), axis=-1)
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """
    arrays = [asanyarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    result_ndim = arrays[0].ndim + 1
    axis = normalize_axis_index(axis, result_ndim)

    sl = (slice(None),) * axis + (_nx.newaxis,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return _nx.concatenate(expanded_arrays, axis=axis, out=out,
                           dtype=dtype, casting=casting)

def _unstack_dispatcher(x, /, *, axis=None):
    return (x,)

@array_function_dispatch(_unstack_dispatcher)
def unstack(x, /, *, axis=0):
    """
    Split an array into a sequence of arrays along the given axis.

    The ``axis`` parameter specifies the dimension along which the array will
    be split. For example, if ``axis=0`` (the default) it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    The result is a tuple of arrays split along ``axis``.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    x : ndarray
        The array to be unstacked.
    axis : int, optional
        Axis along which the array will be split. Default: ``0``.

    Returns
    -------
    unstacked : tuple of ndarrays
        The unstacked arrays.

    See Also
    --------
    stack : Join a sequence of arrays along a new axis.
    concatenate : Join a sequence of arrays along an existing axis.
    block : Assemble an nd-array from nested lists of blocks.
    split : Split array into a list of multiple sub-arrays of equal size.

    Notes
    -----
    ``unstack`` serves as the reverse operation of :py:func:`stack`, i.e.,
    ``stack(unstack(x, axis=axis), axis=axis) == x``.

    This function is equivalent to ``tuple(np.moveaxis(x, axis, 0))``, since
    iterating on an array iterates along the first axis.

    Examples
    --------
    >>> arr = np.arange(24).reshape((2, 3, 4))
    >>> np.unstack(arr)
    (array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]),
     array([[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]))
    >>> np.unstack(arr, axis=1)
    (array([[ 0,  1,  2,  3],
            [12, 13, 14, 15]]),
     array([[ 4,  5,  6,  7],
            [16, 17, 18, 19]]),
     array([[ 8,  9, 10, 11],
            [20, 21, 22, 23]]))
    >>> arr2 = np.stack(np.unstack(arr, axis=1), axis=1)
    >>> arr2.shape
    (2, 3, 4)
    >>> np.all(arr == arr2)
    np.True_

    """
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    return tuple(_nx.moveaxis(x, axis, 0))

# Internal functions to eliminate the overhead of repeated dispatch in one of
# the two possible paths inside np.block.
# Use getattr to protect against __array_function__ being disabled.
_size = getattr(_from_nx.size, '__wrapped__', _from_nx.size)
_ndim = getattr(_from_nx.ndim, '__wrapped__', _from_nx.ndim)
_concatenate = getattr(_from_nx.concatenate,
                       '__wrapped__', _from_nx.concatenate)


def _block_format_index(index):
    """
    Convert a list of indices ``[0, 1, 2]`` into ``"arrays[0][1][2]"``.
    """
    idx_str = ''.join('[{}]'.format(i) for i in index if i is not None)
    return 'arrays' + idx_str


def _block_check_depths_match(arrays, parent_index=[]):
    """
    Recursive function checking that the depths of nested lists in `arrays`
    all match. Mismatch raises a ValueError as described in the block
    docstring below.

    The entire index (rather than just the depth) needs to be calculated
    for each innermost list, in case an error needs to be raised, so that
    the index of the offending list can be printed as part of the error.

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    parent_index : list of int
        The full index of `arrays` within the nested lists passed to
        `_block_check_depths_match` at the top of the recursion.

    Returns
    -------
    first_index : list of int
        The full index of an element from the bottom of the nesting in
        `arrays`. If any element at the bottom is an empty list, this will
        refer to it, and the last index along the empty axis will be None.
    max_arr_ndim : int
        The maximum of the ndims of the arrays nested in `arrays`.
    final_size: int
        The number of elements in the final array. This is used the motivate
        the choice of algorithm used using benchmarking wisdom.

    """
    if type(arrays) is tuple:
        # not strictly necessary, but saves us from:
        #  - more than one way to do things - no point treating tuples like
        #    lists
        #  - horribly confusing behaviour that results when tuples are
        #    treated like ndarray
        raise TypeError(
            '{} is a tuple. '
            'Only lists can be used to arrange blocks, and np.block does '
            'not allow implicit conversion from tuple to ndarray.'.format(
                _block_format_index(parent_index)
            )
        )
    elif type(arrays) is list and len(arrays) > 0:
        idxs_ndims = (_block_check_depths_match(arr, parent_index + [i])
                      for i, arr in enumerate(arrays))

        first_index, max_arr_ndim, final_size = next(idxs_ndims)
        for index, ndim, size in idxs_ndims:
            final_size += size
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            if len(index) != len(first_index):
                raise ValueError(
                    "List depths are mismatched. First element was at depth "
                    "{}, but there is an element at depth {} ({})".format(
                        len(first_index),
                        len(index),
                        _block_format_index(index)
                    )
                )
            # propagate our flag that indicates an empty list at the bottom
            if index[-1] is None:
                first_index = index

        return first_index, max_arr_ndim, final_size
    elif type(arrays) is list and len(arrays) == 0:
        # We've 'bottomed out' on an empty list
        return parent_index + [None], 0, 0
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        size = _size(arrays)
        return parent_index, _ndim(arrays), size


def _atleast_nd(a, ndim):
    # Ensures `a` has at least `ndim` dimensions by prepending
    # ones to `a.shape` as necessary
