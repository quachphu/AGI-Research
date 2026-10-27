    :ref:`how-to-partition`

    Examples
    --------
    >>> import numpy as np
    >>> np.mgrid[0:5, 0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    >>> np.mgrid[0:4].shape
    (4,)
    >>> np.mgrid[0:4, 0:5].shape
    (2, 4, 5)
    >>> np.mgrid[0:4, 0:5, 0:6].shape
    (3, 4, 5, 6)

    """
    __slots__ = ()

    def __init__(self):
        super().__init__(sparse=False)


mgrid = MGridClass()


class OGridClass(nd_grid):
    """
    An instance which returns an open multi-dimensional "meshgrid".

    An instance which returns an open (i.e. not fleshed out) mesh-grid
    when indexed, so that only one dimension of each returned array is
    greater than 1.  The dimension and number of the output arrays are
    equal to the number of indexing dimensions.  If the step length is
    not a complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid : ndarray or tuple of ndarrays
        If the input is a single slice, returns an array.
        If the input is multiple slices, returns a tuple of arrays, with
        only one dimension not equal to 1.

    See Also
    --------
    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5, 0:5]
    (array([[0],
            [1],
            [2],
            [3],
            [4]]),
     array([[0, 1, 2, 3, 4]]))

    """
    __slots__ = ()

    def __init__(self):
        super().__init__(sparse=True)


ogrid = OGridClass()


class AxisConcatenator:
    """
    Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see `r_`.
    """
    __slots__ = ('axis', 'matrix', 'trans1d', 'ndmin')

    # allow ma.mr_ to override this
    concatenate = staticmethod(_nx.concatenate)
    makemat = staticmethod(matrixlib.matrix)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self, key):
        # handle matrix builder syntax
        if isinstance(key, str):
            frame = sys._getframe().f_back
            mymat = matrixlib.bmat(key, frame.f_globals, frame.f_locals)
            return mymat

        if not isinstance(key, tuple):
            key = (key,)

        # copy attributes, since they can be overridden in the first argument
        trans1d = self.trans1d
        ndmin = self.ndmin
        matrix = self.matrix
        axis = self.axis

        objs = []
        # dtypes or scalars for weak scalar handling in result_type
        result_type_objs = []

        for k, item in enumerate(key):
            scalar = False
            if isinstance(item, slice):
                step = item.step
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    size = int(abs(step))
                    newobj = linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
                if ndmin > 1:
                    newobj = array(newobj, copy=None, ndmin=ndmin)
                    if trans1d != -1:
                        newobj = newobj.swapaxes(-1, trans1d)
            elif isinstance(item, str):
                if k != 0:
                    raise ValueError("special directives must be the "
                                     "first entry.")
                if item in ('r', 'c'):
                    matrix = True
                    col = (item == 'c')
                    continue
                if ',' in item:
                    vec = item.split(',')
                    try:
                        axis, ndmin = [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except Exception as e:
                        raise ValueError(
                            "unknown special directive {!r}".format(item)
                        ) from e
                try:
                    axis = int(item)
                    continue
                except (ValueError, TypeError) as e:
                    raise ValueError("unknown special directive") from e
            elif type(item) in ScalarType:
                scalar = True
                newobj = item
            else:
                item_ndim = np.ndim(item)
                newobj = array(item, copy=None, subok=True, ndmin=ndmin)
                if trans1d != -1 and item_ndim < ndmin:
                    k2 = ndmin - item_ndim
                    k1 = trans1d
                    if k1 < 0:
                        k1 += k2 + 1
                    defaxes = list(range(ndmin))
                    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
                    newobj = newobj.transpose(axes)

            objs.append(newobj)
            if scalar:
                result_type_objs.append(item)
            else:
                result_type_objs.append(newobj.dtype)

        # Ensure that scalars won't up-cast unless warranted, for 0, drops
        # through to error in concatenate.
        if len(result_type_objs) != 0:
            final_dtype = _nx.result_type(*result_type_objs)
            # concatenate could do cast, but that can be overridden:
            objs = [array(obj, copy=None, subok=True,
                          ndmin=ndmin, dtype=final_dtype) for obj in objs]

        res = self.concatenate(tuple(objs), axis=axis)

        if matrix:
            oldndim = res.ndim
            res = self.makemat(res)
            if oldndim == 1 and col:
                res = res.T
        return res

    def __len__(self):
        return 0

# separate classes are used here instead of just making r_ = concatenator(0),
# etc. because otherwise we couldn't get the doc string to come out right
# in help(r_)


class RClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the first axis.

    This is a simple way to build up arrays quickly. There are two use cases.

    1. If the index expression contains comma separated arrays, then stack
       them along their first axis.
    2. If the index expression contains slice notation or scalars then create
       a 1-D array with a range indicated by the slice notation.

    If slice notation is used, the syntax ``start:stop:step`` is equivalent
    to ``np.arange(start, stop, step)`` inside of the brackets. However, if
    ``step`` is an imaginary number (i.e. 100j) then its integer portion is
    interpreted as a number-of-points desired and the start and stop are
    inclusive. In other words ``start:stop:stepj`` is interpreted as
    ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
    After expansion of slice notation, all comma separated sequences are
    concatenated together.

    Optional character strings placed as the first element of the index
    expression can be used to change the output. The strings 'r' or 'c' result
    in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
    matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
    (column) matrix is produced. If the result is 2-D then both provide the
    same matrix result.

    A string integer specifies which axis to stack multiple comma separated
    arrays along. A string of two comma-separated integers allows indication
    of the minimum number of dimensions to force each entry into as the
    second integer (the axis to concatenate along is still the first integer).

    A string with three comma-separated integers allows specification of the
    axis to concatenate along, the minimum number of dimensions to force the
    entries to, and which axis should contain the start of the arrays which
    are less than the specified number of dimensions. In other words the third
    integer allows you to specify where the 1's should be placed in the shape
    of the arrays that have their shapes upgraded. By default, they are placed
    in the front of the shape tuple. The third argument allows you to specify
    where the start of the array should be instead. Thus, a third argument of
    '0' would place the 1's at the end of the array shape. Negative integers
    specify where in the new shape tuple the last dimension of upgraded arrays
    should be placed, so the default is '-1'.

    Parameters
    ----------
    Not a function, so takes no parameters


    Returns
    -------
    A concatenated ndarray or matrix.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    c_ : Translates slice objects to concatenation along the second axis.

    Examples
    --------
    >>> import numpy as np
    >>> np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
    array([1, 2, 3, ..., 4, 5, 6])
    >>> np.r_[-1:1:6j, [0]*3, 5, 6]
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])

    String integers specify the axis to concatenate along or the minimum
    number of dimensions to force entries into.

    >>> a = np.array([[0, 1, 2], [3, 4, 5]])
    >>> np.r_['-1', a, a] # concatenate along last axis
    array([[0, 1, 2, 0, 1, 2],
           [3, 4, 5, 3, 4, 5]])
    >>> np.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, dim>=2
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.r_['0,2,0', [1,2,3], [4,5,6]]
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])
    >>> np.r_['1,2,0', [1,2,3], [4,5,6]]
    array([[1, 4],
           [2, 5],
           [3, 6]])

    Using 'r' or 'c' as a first string argument creates a matrix.

    >>> np.r_['r',[1,2,3], [4,5,6]]
    matrix([[1, 2, 3, 4, 5, 6]])

    """
    __slots__ = ()

    def __init__(self):
        AxisConcatenator.__init__(self, 0)


r_ = RClass()


class CClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the second axis.

    This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
    useful because of its common occurrence. In particular, arrays will be
    stacked along their last axis after being upgraded to at least 2-D with
    1's post-pended to the shape (column vectors made out of 1-D arrays).

    See Also
    --------
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    r_ : For more detailed documentation.

    Examples
    --------
    >>> import numpy as np
    >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    array([[1, 2, 3, ..., 4, 5, 6]])

    """
    __slots__ = ()

    def __init__(self):
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)


c_ = CClass()


@set_module('numpy')
class ndenumerate:
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values.

    Parameters
    ----------
    arr : ndarray
      Input array.

    See Also
    --------
    ndindex, flatiter

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> for index, x in np.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4

    """

    def __init__(self, arr):
        self.iter = np.asarray(arr).flat

    def __next__(self):
        """
        Standard iterator method, returns the index tuple and array value.

        Returns
        -------
        coords : tuple of ints
            The indices of the current iteration.
        val : scalar
            The array element of the current iteration.

        """
        return self.iter.coords, next(self.iter)

    def __iter__(self):
        return self


@set_module('numpy')
class ndindex:
    """
    An N-dimensional iterator object to index arrays.

    Given the shape of an array, an `ndindex` instance iterates over
    the N-dimensional index of the array. At each iteration a tuple
    of indices is returned, the last dimension is iterated over first.

    Parameters
    ----------
    shape : ints, or a single tuple of ints
        The size of each dimension of the array can be passed as
        individual parameters or as the elements of a tuple.

    See Also
    --------
    ndenumerate, flatiter

    Examples
    --------
    >>> import numpy as np

    Dimensions as individual arguments

    >>> for index in np.ndindex(3, 2, 1):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    Same dimensions - but in a tuple ``(3, 2, 1)``

    >>> for index in np.ndindex((3, 2, 1)):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """

    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        x = as_strided(_nx.zeros(1), shape=shape,
                       strides=_nx.zeros_like(shape))
        self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'],
                              order='C')

    def __iter__(self):
        return self

    def ndincr(self):
        """
        Increment the multi-dimensional index by one.

        This method is for backward compatibility only: do not use.

        .. deprecated:: 1.20.0
            This method has been advised against since numpy 1.8.0, but only
            started emitting DeprecationWarning as of this version.
        """
        # NumPy 1.20.0, 2020-09-08
        warnings.warn(
            "`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead",
            DeprecationWarning, stacklevel=2)
        next(self)

    def __next__(self):
        """
        Standard iterator method, updates the index and returns the index
        tuple.

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the indices of the current
            iteration.

        """
        next(self._it)
        return self._it.multi_index


# You can do all this with slice() plus a few special objects,
# but there's a lot to remember. This version is simpler because
# it uses the standard array indexing syntax.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-23
#
# Cosmetic changes by T. Oliphant 2001
#
#

class IndexExpression:
    """
    A nicer way to build up index tuples for arrays.

    .. note::
       Use one of the two predefined instances ``index_exp`` or `s_`
       rather than directly using `IndexExpression`.

    For any index combination, including slicing and axis insertion,
    ``a[indices]`` is the same as ``a[np.index_exp[indices]]`` for any
    array `a`. However, ``np.index_exp[indices]`` can be used anywhere
    in Python code and returns a tuple of slice objects that can be
    used in the construction of complex index expressions.

    Parameters
    ----------
    maketuple : bool
        If True, always returns a tuple.

    See Also
    --------
    s_ : Predefined instance without tuple conversion:
       `s_ = IndexExpression(maketuple=False)`.
       The ``index_exp`` is another predefined instance that
       always returns a tuple:
       `index_exp = IndexExpression(maketuple=True)`.

    Notes
    -----
    You can do all this with :class:`slice` plus a few special objects,
    but there's a lot to remember and this version is simpler because
    it uses the standard array indexing syntax.

    Examples
    --------
    >>> import numpy as np
    >>> np.s_[2::2]
    slice(2, None, 2)
    >>> np.index_exp[2::2]
    (slice(2, None, 2),)

    >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]
    array([2, 4])

    """
    __slots__ = ('maketuple',)

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)

# End contribution from Konrad.


# The following functions complement those in twodim_base, but are
# applicable to N-dimensions.


def _fill_diagonal_dispatcher(a, val, wrap=None):
    return (a,)


@array_function_dispatch(_fill_diagonal_dispatcher)
def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array of any dimensionality.

    For an array `a` with ``a.ndim >= 2``, the diagonal is the list of
    values ``a[i, ..., i]`` with indices ``i`` all identical.  This function
    modifies the input array in-place without returning a value.

    Parameters
    ----------
    a : array, at least 2-D.
      Array whose diagonal is to be filled in-place.
    val : scalar or array_like
      Value(s) to write on the diagonal. If `val` is scalar, the value is
      written along the diagonal. If array-like, the flattened `val` is
      written along the diagonal, repeating if necessary to fill all
      diagonal entries.

    wrap : bool
      For tall matrices in NumPy version up to 1.6.2, the
      diagonal "wrapped" after N columns. You can have this behavior
      with this option. This affects only tall matrices.

    See also
    --------
    diag_indices, diag_indices_from

    Notes
    -----
    This functionality can be obtained via `diag_indices`, but internally
    this version uses a much faster implementation that never constructs the
    indices and uses simple slicing.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), int)
    >>> np.fill_diagonal(a, 5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-D array:

    >>> a = np.zeros((3, 3, 3, 3), int)
    >>> np.fill_diagonal(a, 4)

    We only show a few blocks for clarity:

    >>> a[0, 0]
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1, 1]
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2, 2]
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    The wrap option affects only tall matrices:

    >>> # tall matrices no wrap
    >>> a = np.zeros((5, 3), int)
    >>> np.fill_diagonal(a, 4)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [0, 0, 0]])

    >>> # tall matrices wrap
    >>> a = np.zeros((5, 3), int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [4, 0, 0]])

    >>> # wide matrices
    >>> a = np.zeros((3, 5), int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 4, 0, 0]])

    The anti-diagonal can be filled by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = np.zeros((3, 3), int);
    >>> np.fill_diagonal(np.fliplr(a), [1,2,3])  # Horizontal flip
    >>> a
    array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    >>> np.fill_diagonal(np.flipud(a), [1,2,3])  # Vertical flip
    >>> a
    array([[0, 0, 3],
           [0, 2, 0],
           [1, 0, 0]])

    Note that the order in which the diagonal is filled varies depending
    on the flip function.
    """
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not np.all(diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + (np.cumprod(a.shape[:-1])).sum()

    # Write the value out into the diagonal.
    a.flat[:end:step] = val


@set_module('numpy')
def diag_indices(n, ndim=2):
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
    for ``i = [0..n-1]``.

    Parameters
    ----------
    n : int
      The size, along each dimension, of the arrays for which the returned
      indices can be used.

    ndim : int, optional
      The number of dimensions.

    See Also
    --------
    diag_indices_from

    Examples
    --------
    >>> import numpy as np

    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> di = np.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Now, we create indices to manipulate a 3-D array:

    >>> d3 = np.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = np.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
           [[0, 0],
            [0, 1]]])

    """
    idx = np.arange(n)
    return (idx,) * ndim


def _diag_indices_from(arr):
    return (arr,)


@array_function_dispatch(_diag_indices_from)
def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    See `diag_indices` for full details.

    Parameters
    ----------
    arr : array, at least 2-D

    See Also
    --------
    diag_indices

    Examples
    --------
    >>> import numpy as np

    Create a 4 by 4 array.

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Get the indices of the diagonal elements.

    >>> di = np.diag_indices_from(a)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    >>> a[di]
    array([ 0,  5, 10, 15])

    This is simply syntactic sugar for diag_indices.

    >>> np.diag_indices(a.shape[0])
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """

    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    if not np.all(diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(arr.shape[0], arr.ndim)
<|endoftext|>"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.

"""
__all__ = ['fix', 'isneginf', 'isposinf']

import numpy._core.numeric as nx
from numpy._core.overrides import array_function_dispatch


def _dispatcher(x, out=None):
    return (x, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def fix(x, out=None):
    """
    Round to nearest integer towards zero.

    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values have the same data-type as the input.

    Parameters
    ----------
    x : array_like
        An array to be rounded
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the input broadcasts to. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    out : ndarray of floats
        An array with the same dimensions and data-type as the input.
        If second argument is not supplied then a new array is returned
        with the rounded values.

        If a second argument is supplied the result is stored there.
        The return value ``out`` is then a reference to that array.

    See Also
    --------
    rint, trunc, floor, ceil
    around : Round to given number of decimals

    Examples
    --------
    >>> import numpy as np
    >>> np.fix(3.14)
    3.0
    >>> np.fix(3)
    3
    >>> np.fix([2.1, 2.9, -2.1, -2.9])
    array([ 2.,  2., -2., -2.])

    """
    # promote back to an array if flattened
    res = nx.asanyarray(nx.ceil(x, out=out))
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))

    # when no out argument is passed and no subclasses are involved, flatten
    # scalars
    if out is None and type(res) is nx.ndarray:
        res = res[()]
    return res


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a boolean array is returned
        with values True where the corresponding element of the input is
        positive infinity and values False where the element of the input is
        not positive infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as zeros
        and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isneginf, isfinite, isnan

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values

    Examples
    --------
    >>> import numpy as np
    >>> np.isposinf(np.inf)
    True
    >>> np.isposinf(-np.inf)
    False
    >>> np.isposinf([-np.inf, 0., np.inf])
    array([False, False,  True])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isposinf(x, y)
    array([0, 0, 1])
    >>> y
    array([0, 0, 1])

    """
    is_inf = nx.isinf(x)
    try:
        signbit = ~nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True. The
        return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isposinf, isnan, isfinite

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values.

    Examples
    --------
    >>> import numpy as np
    >>> np.isneginf(-np.inf)
    True
    >>> np.isneginf(np.inf)
    False
    >>> np.isneginf([-np.inf, 0., np.inf])
    array([ True, False, False])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isneginf(x, y)
    array([1, 0, 0])
    >>> y
    array([1, 0, 0])

    """
    is_inf = nx.isinf(x)
    try:
        signbit = nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)
<|endoftext|>"""
Miscellaneous utils.
"""
from numpy._core import asarray
from numpy._core.numeric import normalize_axis_tuple, normalize_axis_index
from numpy._utils import set_module

__all__ = ["byte_bounds", "normalize_axis_tuple", "normalize_axis_index"]


@set_module("numpy.lib.array_utils")
def byte_bounds(a):
    """
    Returns pointers to the end-points of an array.

    Parameters
    ----------
    a : ndarray
        Input array. It must conform to the Python-side of the array
        interface.

    Returns
    -------
    (low, high) : tuple of 2 integers
        The first integer is the first byte of the array, the second
        integer is just past the last byte of the array.  If `a` is not
        contiguous it will not use every byte between the (`low`, `high`)
        values.

    Examples
    --------
    >>> import numpy as np
    >>> I = np.eye(2, dtype='f'); I.dtype
    dtype('float32')
    >>> low, high = np.lib.array_utils.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True
    >>> I = np.eye(2); I.dtype
    dtype('float64')
    >>> low, high = np.lib.array_utils.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True

    """
    ai = a.__array_interface__
    a_data = ai['data'][0]
    astrides = ai['strides']
    ashape = ai['shape']
    bytes_a = asarray(a).dtype.itemsize

    a_low = a_high = a_data
    if astrides is None:
        # contiguous case
        a_high += a.size * bytes_a
    else:
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape-1)*stride
            else:
                a_high += (shape-1)*stride
        a_high += bytes_a
    return a_low, a_high
<|endoftext|>"""
Functions that ignore NaN.

Functions
---------

- `nanmin` -- minimum non-NaN value
- `nanmax` -- maximum non-NaN value
- `nanargmin` -- index of minimum non-NaN value
- `nanargmax` -- index of maximum non-NaN value
- `nansum` -- sum of non-NaN values
- `nanprod` -- product of non-NaN values
- `nancumsum` -- cumulative sum of non-NaN values
- `nancumprod` -- cumulative product of non-NaN values
- `nanmean` -- mean of non-NaN values
- `nanvar` -- variance of non-NaN values
- `nanstd` -- standard deviation of non-NaN values
- `nanmedian` -- median of non-NaN values
- `nanquantile` -- qth quantile of non-NaN values
- `nanpercentile` -- qth percentile of non-NaN values

"""
import functools
import warnings
import numpy as np
import numpy._core.numeric as _nx
from numpy.lib import _function_base_impl as fnb
from numpy.lib._function_base_impl import _weights_are_valid
from numpy._core import overrides


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
    'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod',
    'nancumsum', 'nancumprod', 'nanquantile'
    ]


def _nan_mask(a, out=None):
    """
    Parameters
    ----------
    a : array-like
        Input array with at least 1 dimension.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output and will prevent the allocation of a new array.

    Returns
    -------
    y : bool ndarray or True
        A bool array where ``np.nan`` positions are marked with ``False``
        and other positions are marked with ``True``. If the type of ``a``
        is such that it can't possibly contain ``np.nan``, returns ``True``.
    """
    # we assume that a is an array for this private function

    if a.dtype.kind not in 'fc':
        return True

    y = np.isnan(a, out=out)
    y = np.invert(y, out=y)
    return y

def _replace_nan(a, val):
    """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Note that scalars will end up as array scalars, which is important
    for using the result as the value of the out argument in some
    operations.

    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.

    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.

    """
    a = np.asanyarray(a)

    if a.dtype == np.object_:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        mask = np.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, np.inexact):
        mask = np.isnan(a)
    else:
        mask = None

    if mask is not None:
        a = np.array(a, subok=True, copy=True)
        np.copyto(a, val, where=mask)

    return a, mask


def _copyto(a, val, mask):
    """
    Replace values in `a` with NaN where `mask` is True.  This differs from
    copyto in that it will deal with the case where `a` is a numpy scalar.

    Parameters
    ----------
    a : ndarray or numpy scalar
        Array or numpy scalar some of whose values are to be replaced
        by val.
    val : numpy scalar
        Value used a replacement.
    mask : ndarray, scalar
        Boolean array. Where True the corresponding element of `a` is
        replaced by `val`. Broadcasts.

    Returns
    -------
    res : ndarray, scalar
        Array with elements replaced or scalar `val`.

    """
    if isinstance(a, np.ndarray):
        np.copyto(a, val, where=mask, casting='unsafe')
    else:
        a = a.dtype.type(val)
    return a


def _remove_nan_1d(arr1d, second_arr1d=None, overwrite_input=False):
    """
    Equivalent to arr1d[~arr1d.isnan()], but in a different order

    Presumably faster as it incurs fewer copies

    Parameters
    ----------
    arr1d : ndarray
        Array to remove nans from
    second_arr1d : ndarray or None
        A second array which will have the same positions removed as arr1d.
    overwrite_input : bool
        True if `arr1d` can be modified in place

    Returns
    -------
    res : ndarray
        Array with nan elements removed
    second_res : ndarray or None
        Second array with nan element positions of first array removed.
    overwrite_input : bool
        True if `res` can be modified in place, given the constraint on the
        input
    """
    if arr1d.dtype == object:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = np.isnan(arr1d)

    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=6)
        if second_arr1d is None:
            return arr1d[:0], None, True
        else:
            return arr1d[:0], second_arr1d[:0], True
    elif s.size == 0:
        return arr1d, second_arr1d, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        # select non-nans at end of array
        enonan = arr1d[-s.size:][~c[-s.size:]]
        # fill nans in beginning of array with non-nans of end
        arr1d[s[:enonan.size]] = enonan

        if second_arr1d is None:
            return arr1d[:-s.size], None, True
        else:
            if not overwrite_input:
                second_arr1d = second_arr1d.copy()
            enonan = second_arr1d[-s.size:][~c[-s.size:]]
            second_arr1d[s[:enonan.size]] = enonan

            return arr1d[:-s.size], second_arr1d[:-s.size], True


def _divide_by_count(a, b, out=None):
    """
    Compute a/b ignoring invalid results. If `a` is an array the division
    is done in place. If `a` is a scalar, then its type is preserved in the
    output. If out is None, then a is used instead so that the division
    is in place. Note that this is only called with `a` an inexact type.

    Parameters
    ----------
    a : {ndarray, numpy scalar}
        Numerator. Expected to be of inexact type but not checked.
    b : {ndarray, numpy scalar}
        Denominator.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.

    Returns
    -------
    ret : {ndarray, numpy scalar}
        The return value is a/b. If `a` was an ndarray the division is done
        in place. If `a` is a numpy scalar, the division preserves its type.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        if isinstance(a, np.ndarray):
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                # Precaution against reduced object arrays
                try:
                    return a.dtype.type(a / b)
                except AttributeError:
                    return a / b
            else:
                # This is questionable, but currently a numpy scalar can
                # be output to a zero dimensional array.
                return np.divide(a, b, out=out, casting='unsafe')


def _nanmin_dispatcher(a, axis=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nanmin_dispatcher)
def nanmin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
           where=np._NoValue):
    """
    Return minimum of an array or minimum along an axis, ignoring any NaNs.
    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
    Nan is returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the minimum is computed. The default is to compute
        the minimum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `min` method
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanmin : ndarray
        An array with the same shape as `a`, with the specified axis
        removed.  If `a` is a 0-d array, or if axis is None, an ndarray
        scalar is returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amax, fmax, maximum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.min.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmin(a)
    1.0
    >>> np.nanmin(a, axis=0)
    array([1.,  2.])
    >>> np.nanmin(a, axis=1)
    array([1.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmin([1, 2, np.nan, np.inf])
    1.0
    >>> np.nanmin([1, 2, np.nan, -np.inf])
    -inf

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if initial is not np._NoValue:
        kwargs['initial'] = initial
    if where is not np._NoValue:
        kwargs['where'] = where

    if type(a) is np.ndarray and a.dtype != np.object_:
        # Fast, but not safe for subclasses of ndarray, or object arrays,
        # which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=2)
    else:
        # Slow, but safe for subclasses of ndarray
        a, mask = _replace_nan(a, +np.inf)
        res = np.amin(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # Check for all-NaN axis
        kwargs.pop("initial", None)
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=2)
    return res


def _nanmax_dispatcher(a, axis=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nanmax_dispatcher)
def nanmax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
           where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis, ignoring any
    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
    raised and NaN is returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the maximum is computed. The default is to compute
        the maximum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If the value is anything but the default, then
        `keepdims` will be passed through to the `max` method
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanmax : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an ndarray scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amin, fmin, minimum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.max.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmax(a)
    3.0
    >>> np.nanmax(a, axis=0)
    array([3.,  2.])
    >>> np.nanmax(a, axis=1)
    array([2.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmax([1, 2, np.nan, -np.inf])
    2.0
    >>> np.nanmax([1, 2, np.nan, np.inf])
    inf

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if initial is not np._NoValue:
        kwargs['initial'] = initial
    if where is not np._NoValue:
        kwargs['where'] = where

    if type(a) is np.ndarray and a.dtype != np.object_:
        # Fast, but not safe for subclasses of ndarray, or object arrays,
        # which do not implement isnan (gh-9009), or fmax correctly (gh-8975)
        res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=2)
    else:
        # Slow, but safe for subclasses of ndarray
        a, mask = _replace_nan(a, -np.inf)
        res = np.amax(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # Check for all-NaN axis
        kwargs.pop("initial", None)
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=2)
    return res


def _nanargmin_dispatcher(a, axis=None, out=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_nanargmin_dispatcher)
def nanargmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
    cannot be trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

        .. versionadded:: 1.22.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmin, nanargmax

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmin(a)
    0
    >>> np.nanargmin(a)
    2
    >>> np.nanargmin(a, axis=0)
    array([1, 1])
    >>> np.nanargmin(a, axis=1)
    array([1, 0])

    """
    a, mask = _replace_nan(a, np.inf)
    if mask is not None and mask.size:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    res = np.argmin(a, axis=axis, out=out, keepdims=keepdims)
    return res


def _nanargmax_dispatcher(a, axis=None, out=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_nanargmax_dispatcher)
def nanargmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the
    results cannot be trusted if a slice contains only NaNs and -Infs.


    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

        .. versionadded:: 1.22.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmax, nanargmin

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmax(a)
    0
    >>> np.nanargmax(a)
    1
    >>> np.nanargmax(a, axis=0)
    array([1, 0])
    >>> np.nanargmax(a, axis=1)
    array([1, 1])

    """
    a, mask = _replace_nan(a, -np.inf)
    if mask is not None and mask.size:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    res = np.argmax(a, axis=axis, out=out, keepdims=keepdims)
    return res


def _nansum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nansum_dispatcher)
def nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
           initial=np._NoValue, where=np._NoValue):
    """
    Return the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    In NumPy versions <= 1.9.0 Nan is returned for slices that are all-NaN or
    empty. In later versions zero is returned.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the sum is computed. The default is to compute the
        sum of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nansum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    numpy.sum : Sum across array propagating NaNs.
    isnan : Show which elements are NaN.
    isfinite : Show which elements are not NaN or +/-inf.

    Notes
    -----
    If both positive and negative infinity are present, the sum will be Not
    A Number (NaN).

    Examples
    --------
    >>> import numpy as np
    >>> np.nansum(1)
    1
    >>> np.nansum([1])
    1
    >>> np.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> np.nansum(a)
    3.0
    >>> np.nansum(a, axis=0)
    array([2.,  1.])
    >>> np.nansum([1, np.nan, np.inf])
    inf
    >>> np.nansum([1, np.nan, -np.inf])
    -inf
    >>> from numpy.testing import suppress_warnings
    >>> with np.errstate(invalid="ignore"):
    ...     np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present
    np.float64(nan)

    """
    a, mask = _replace_nan(a, 0)
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                  initial=initial, where=where)


def _nanprod_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                        initial=None, where=None):
    return (a, out)


@array_function_dispatch(_nanprod_dispatcher)
def nanprod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
            initial=np._NoValue, where=np._NoValue):
    """
    Return the product of array elements over a given axis treating Not a
    Numbers (NaNs) as ones.

    One is returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose product is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the product is computed. The default is to compute
        the product of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will
        broadcast correctly against the original `arr`.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to include in the product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    numpy.prod : Product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> np.nanprod(1)
    1
    >>> np.nanprod([1])
    1
    >>> np.nanprod([1, np.nan])
    1.0
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanprod(a)
    6.0
    >>> np.nanprod(a, axis=0)
    array([3., 2.])

    """
    a, mask = _replace_nan(a, 1)
    return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                   initial=initial, where=where)


def _nancumsum_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_nancumsum_dispatcher)
def nancumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.  The cumulative sum does not change when NaNs are
    encountered and leading NaNs are replaced by zeros.

    Zeros are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See :ref:`ufuncs-output-type` for
        more details.

    Returns
    -------
    nancumsum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    numpy.cumsum : Cumulative sum across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> np.nancumsum(1)
    array([1])
    >>> np.nancumsum([1])
    array([1])
    >>> np.nancumsum([1, np.nan])
    array([1.,  1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumsum(a)
    array([1.,  3.,  6.,  6.])
    >>> np.nancumsum(a, axis=0)
    array([[1.,  2.],
           [4.,  2.]])
    >>> np.nancumsum(a, axis=1)
    array([[1.,  3.],
           [3.,  3.]])

    """
    a, mask = _replace_nan(a, 0)
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)


def _nancumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_nancumprod_dispatcher)
def nancumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of array elements over a given axis treating Not a
    Numbers (NaNs) as one.  The cumulative product does not change when NaNs are
    encountered and leading NaNs are replaced by ones.

    Ones are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed.  By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If *dtype* is not specified, it
        defaults to the dtype of `a`, unless `a` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Returns
    -------
    nancumprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    numpy.cumprod : Cumulative product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> np.nancumprod(1)
    array([1])
    >>> np.nancumprod([1])
    array([1])
    >>> np.nancumprod([1, np.nan])
    array([1.,  1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumprod(a)
    array([1.,  2.,  6.,  6.])
    >>> np.nancumprod(a, axis=0)
    array([[1.,  2.],
           [3.,  2.]])
    >>> np.nancumprod(a, axis=1)
    array([[1.,  2.],
           [3.,  3.]])

    """
    a, mask = _replace_nan(a, 1)
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)


def _nanmean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                        *, where=None):
    return (a, out)


@array_function_dispatch(_nanmean_dispatcher)
def nanmean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
            *, where=np._NoValue):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the input
        dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32`.  Specifying a
    higher-precision accumulator using the `dtype` keyword can alleviate
    this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanmean(a)
    2.6666666666666665
    >>> np.nanmean(a, axis=0)
    array([2.,  4.])
    >>> np.nanmean(a, axis=1)
    array([1.,  3.5]) # may vary

    """
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                       where=where)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims,
                 where=where)
    tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)
    avg = _divide_by_count(tot, cnt, out=out)

    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        # NaN is the only possible bad value, so no further
        # action is needed to handle bad results.
    return avg


def _nanmedian1d(arr1d, overwrite_input=False):
    """
    Private function for rank 1 arrays. Compute the median ignoring NaNs.
    See nanmedian for parameter usage
    """
    arr1d_parsed, _, overwrite_input = _remove_nan_1d(
        arr1d, overwrite_input=overwrite_input,
    )

    if arr1d_parsed.size == 0:
        # Ensure that a nan-esque scalar of the appropriate type (and unit)
        # is returned for `timedelta64` and `complexfloating`
        return arr1d[-1]

    return np.median(arr1d_parsed, overwrite_input=overwrite_input)


def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanmedian for parameter usage

    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        else:
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        # for small medians use sort + indexing which is still faster than
        # apply_along_axis
        # benchmarked with shuffled (50, 50, x) containing a few NaN
        if a.shape[axis] < 600:
            return _nanmedian_small(a, axis, out, overwrite_input)
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        if out is not None:
            out[...] = result
        return result


def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    """
    sort + indexing median, faster for small medians along multiple
    dimensions due to the high overhead of apply_along_axis

    see nanmedian for parameter usage
    """
    a = np.ma.masked_array(a, np.isnan(a))
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=5)

    fill_value = np.timedelta64("NaT") if m.dtype.kind == "m" else np.nan
    if out is not None:
        out[...] = m.filled(fill_value)
        return out
    return m.filled(fill_value)


def _nanmedian_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)


@array_function_dispatch(_nanmedian_dispatcher)
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue):
    """
    Compute the median along the specified axis, while ignoring NaNs.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
        A sequence of axes is supported since version 1.9.0.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean, median, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,
    ``V_sorted[(N-1)/2]``, when ``N`` is odd and the average of the two
    middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10.0, 7, 4], [3, 2, 1]])
    >>> a[0, 1] = np.nan
    >>> a
    array([[10., nan,  4.],
           [ 3.,  2.,  1.]])
    >>> np.median(a)
    np.float64(nan)
    >>> np.nanmedian(a)
    3.0
    >>> np.nanmedian(a, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.median(a, axis=1)
    array([nan,  2.])
    >>> b = a.copy()
    >>> np.nanmedian(b, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.nanmedian(b, axis=None, overwrite_input=True)
    3.0
    >>> assert not np.all(a==b)

    """
    a = np.asanyarray(a)
    # apply_along_axis in _nanmedian doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)

    return fnb._ureduce(a, func=_nanmedian, keepdims=keepdims,
                        axis=axis, out=out,
                        overwrite_input=overwrite_input)


def _nanpercentile_dispatcher(
        a, q, axis=None, out=None, overwrite_input=None,
        method=None, keepdims=None, *, weights=None, interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_nanpercentile_dispatcher)
def nanpercentile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        *,
        weights=None,
        interpolation=None,
):
    """
    Compute the qth percentile of the data along the specified axis,
    while ignoring nan values.

    Returns the qth percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be
        between 0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default
        is to compute the percentile(s) along a flattened version of the
        array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the percentile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    nanmean
    nanmedian : equivalent to ``nanpercentile(..., 50)``
    percentile, median, mean
    nanquantile : equivalent to nanpercentile, except q in range [0, 1].

    Notes
    -----
    The behavior of `numpy.nanpercentile` with percentage `q` is that of
    `numpy.quantile` with argument ``q/100`` (ignoring nan values).
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = np.nan
    >>> a
    array([[10.,  nan,   4.],
          [ 3.,   2.,   1.]])
    >>> np.percentile(a, 50)
    np.float64(nan)
    >>> np.nanpercentile(a, 50)
    3.0
    >>> np.nanpercentile(a, 50, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.nanpercentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.nanpercentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.nanpercentile(a, 50, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5,  2. ,  2.5])

    >>> b = a.copy()
    >>> np.nanpercentile(b, 50, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    if interpolation is not None:
        method = fnb._check_interpolation_as_method(
            method, interpolation, "nanpercentile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)
    # undo any decay that the ufunc performed (see gh-13105)
    q = np.asanyarray(q)
    if not fnb._quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _nanquantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                            method=None, keepdims=None, *, weights=None,
                            interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_nanquantile_dispatcher)
def nanquantile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        *,
        weights=None,
        interpolation=None,
):
    """
    Compute the qth quantile of the data along the specified axis,
    while ignoring nan values.
    Returns the qth quantile(s) of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored
    q : array_like of float
        Probability or sequence of probabilities for the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the quantile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single probability and `axis=None`, then the result
        is a scalar. If multiple probability levels are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    quantile
    nanmean, nanmedian
    nanmedian : equivalent to ``nanquantile(..., 0.5)``
    nanpercentile : same as nanquantile, but with q in the range [0, 100].

    Notes
    -----
    The behavior of `numpy.nanquantile` is the same as that of
    `numpy.quantile` (ignoring nan values).
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = np.nan
    >>> a
    array([[10.,  nan,   4.],
          [ 3.,   2.,   1.]])
    >>> np.quantile(a, 0.5)
    np.float64(nan)
    >>> np.nanquantile(a, 0.5)
    3.0
    >>> np.nanquantile(a, 0.5, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.nanquantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.nanquantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.nanquantile(a, 0.5, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5,  2. ,  2.5])
    >>> b = a.copy()
    >>> np.nanquantile(b, 0.5, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """

    if interpolation is not None:
        method = fnb._check_interpolation_as_method(
            method, interpolation, "nanquantile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float).
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    if not fnb._quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _nanquantile_unchecked(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        weights=None,
):
    """Assumes that q is in [0, 1], and is an ndarray"""
    # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    return fnb._ureduce(a,
                        func=_nanquantile_ureduce_func,
                        q=q,
                        weights=weights,
                        keepdims=keepdims,
                        axis=axis,
                        out=out,
                        overwrite_input=overwrite_input,
                        method=method)


def _nanquantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int | None = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        wgt = None if weights is None else weights.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, method, weights=wgt)
    else:
        # Note that this code could try to fill in `out` right away
        if weights is None:
            result = np.apply_along_axis(_nanquantile_1d, axis, a, q,
                                         overwrite_input, method, weights)
            # apply_along_axis fills in collapsed axis with results.
            # Move those axes to the beginning to match percentile's
            # convention.
            if q.ndim != 0:
                from_ax = [axis + i for i in range(q.ndim)]
                result = np.moveaxis(result, from_ax, list(range(q.ndim)))
        else:
            # We need to apply along axis over 2 arrays, a and weights.
            # move operation axes to end for simplicity:
            a = np.moveaxis(a, axis, -1)
            if weights is not None:
                weights = np.moveaxis(weights, axis, -1)
            if out is not None:
                result = out
            else:
                # weights are limited to `inverted_cdf` so the result dtype
                # is known to be identical to that of `a` here:
                result = np.empty_like(a, shape=q.shape + a.shape[:-1])

            for ii in np.ndindex(a.shape[:-1]):
                result[(...,) + ii] = _nanquantile_1d(
                        a[ii], q, weights=weights[ii],
                        overwrite_input=overwrite_input, method=method,
                )
            # This path dealt with `out` already...
            return result

    if out is not None:
        out[...] = result
    return result


def _nanquantile_1d(
    arr1d, q, overwrite_input=False, method="linear", weights=None,
):
    """
    Private function for rank 1 arrays. Compute quantile ignoring NaNs.
    See nanpercentile for parameter usage
    """
    # TODO: What to do when arr1d = [1, np.nan] and weights = [0, 1]?
    arr1d, weights, overwrite_input = _remove_nan_1d(arr1d,
        second_arr1d=weights, overwrite_input=overwrite_input)
    if arr1d.size == 0:
        # convert to scalar
        return np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]

    return fnb._quantile_unchecked(
        arr1d,
        q,
        overwrite_input=overwrite_input,
        method=method,
        weights=weights,
    )


def _nanvar_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                       keepdims=None, *, where=None, mean=None,
                       correction=None):
    return (a, out)


@array_function_dispatch(_nanvar_dispatcher)
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue,
           *, where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of
    a distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float64`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : {int, float}, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
    where : array_like of bool, optional
        Elements to include in the variance. See `~numpy.ufunc.reduce` for
        details.

        .. versionadded:: 1.22.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this var function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If `out` is None, return a new array containing the variance,
        otherwise return a reference to the output array. If ddof is >= the
        number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd, nanmean
    :ref:`ufuncs-output-type`

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite
    population.  ``ddof=0`` provides a maximum likelihood estimate of the
    variance for normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    For this function to work on sub-classes of ndarray, they must define
    `sum` with the kwarg `keepdims`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanvar(a)
    1.5555555555555554
    >>> np.nanvar(a, axis=0)
    array([1.,  0.])
    >>> np.nanvar(a, axis=1)
    array([0.,  0.25])  # may vary

    """
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof,
                      keepdims=keepdims, where=where, mean=mean,
                      correction=correction)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    # Compute mean
    if type(arr) is np.matrix:
        _keepdims = np._NoValue
    else:
        _keepdims = True

    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=_keepdims,
                     where=where)

    if mean is not np._NoValue:
        avg = mean
    else:
        # we need to special case matrix for reverse compatibility
        # in order for this to work, these sums need to be called with
        # keepdims=True, however matrix now raises an error in this case, but
        # the reason that it drops the keepdims kwarg is to force keepdims=True
        # so this used to work by serendipity.
        avg = np.sum(arr, axis=axis, dtype=dtype,
                     keepdims=_keepdims, where=where)
        avg = _divide_by_count(avg, cnt)

    # Compute squared deviation from mean.
    np.subtract(arr, avg, out=arr, casting='unsafe', where=where)
    arr = _copyto(arr, 0, mask)
    if issubclass(arr.dtype.type, np.complexfloating):
        sqr = np.multiply(arr, arr.conj(), out=arr, where=where).real
    else:
        sqr = np.multiply(arr, arr, out=arr, where=where)

    # Compute variance.
    var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)

    # Precaution against reduced object arrays
    try:
        var_ndim = var.ndim
    except AttributeError:
        var_ndim = np.ndim(var)
    if var_ndim < cnt.ndim:
        # Subclasses of ndarray may ignore keepdims, so check here.
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    var = _divide_by_count(var, dof)

    isbad = (dof <= 0)
    if np.any(isbad):
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning,
                      stacklevel=2)
        # NaN, inf, or negative numbers are all possible bad
        # values, so explicitly replace them with NaN.
        var = _copyto(var, np.nan, isbad)
    return var


def _nanstd_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                       keepdims=None, *, where=None, mean=None,
                       correction=None):
    return (a, out)


@array_function_dispatch(_nanstd_dispatcher)
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue,
           *, where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    """
    Compute the standard deviation along the specified axis, while
    ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a
    distribution, of the non-NaN array elements. The standard deviation is
    computed for the flattened array by default, otherwise over the
    specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of the non-NaN values.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it
        is the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the
        calculated values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non-NaN
        elements.  By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If this value is anything but the default it is passed through
        as-is to the relevant functions of the sub-classes.  If these
        functions do not have a `keepdims` kwarg, a RuntimeError will
        be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this std function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard
        deviation, otherwise return a reference to the output array. If
        ddof is >= the number of non-NaN elements in a slice or the slice
        contains only NaNs, then the result for that slice is NaN.

    See Also
    --------
    var, mean, std
    nanvar, nanmean
    :ref:`ufuncs-output-type`

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is
    specified, the divisor ``N - ddof`` is used instead. In standard
    statistical practice, ``ddof=1`` provides an unbiased estimator of the
    variance of the infinite population. ``ddof=0`` provides a maximum
    likelihood estimate of the variance for normally distributed variables.
    The standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Note that, for complex numbers, `std` takes the absolute value before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the *std* is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example
    below).  Specifying a higher-accuracy accumulator using the `dtype`
    keyword can alleviate this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanstd(a)
    1.247219128924647
    >>> np.nanstd(a, axis=0)
    array([1., 0.])
    >>> np.nanstd(a, axis=1)
    array([0.,  0.5]) # may vary

    """
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims, where=where, mean=mean,
                 correction=correction)
    if isinstance(var, np.ndarray):
        std = np.sqrt(var, out=var)
    elif hasattr(var, 'dtype'):
        std = var.dtype.type(np.sqrt(var))
    else:
        std = np.sqrt(var)
    return std
<|endoftext|>"""A file interface for handling local and remote data files.

The goal of datasource is to abstract some of the file system operations
when dealing with data files so the researcher doesn't have to know all the
low-level details.  Through datasource, a researcher can obtain and use a
file with one function call, regardless of location of the file.

DataSource is meant to augment standard python libraries, not replace them.
It should work seamlessly with standard file IO operations and the os
module.

DataSource files can originate locally or remotely:

- local files : '/home/guido/src/local/data.txt'
- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'

DataSource files can also be compressed or uncompressed.  Currently only
gzip, bz2 and xz are supported.

Example::

    >>> # Create a DataSource, use os.curdir (default) for local storage.
    >>> from numpy import DataSource
    >>> ds = DataSource()
    >>>
    >>> # Open a remote file.
    >>> # DataSource downloads the file, stores it locally in:
    >>> #     './www.google.com/index.html'
    >>> # opens the file and returns a file object.
    >>> fp = ds.open('http://www.google.com/') # doctest: +SKIP
    >>>
    >>> # Use the file as you normally would
    >>> fp.read() # doctest: +SKIP
    >>> fp.close() # doctest: +SKIP

"""
import os

from .._utils import set_module


_open = open


def _check_mode(mode, encoding, newline):
    """Check mode and that encoding and newline are compatible.

    Parameters
    ----------
    mode : str
        File open mode.
    encoding : str
        File encoding.
    newline : str
        Newline for text files.

    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")


# Using a class instead of a module-level dictionary
# to reduce the initial 'import numpy' overhead by
# deferring the import of lzma, bz2 and gzip until needed

# TODO: .zip support, .tar support?
class _FileOpeners:
    """
    Container for different methods to open (un-)compressed files.

    `_FileOpeners` contains a dictionary that holds one method for each
    supported file format. Attribute lookup is implemented in such a way
    that an instance of `_FileOpeners` itself can be indexed with the keys
    of that dictionary. Currently uncompressed files as well as files
    compressed with ``gzip``, ``bz2`` or ``xz`` compression are supported.

    Notes
    -----
    `_file_openers`, an instance of `_FileOpeners`, is made available for
    use in the `_datasource` module.

    Examples
    --------
    >>> import gzip
    >>> np.lib._datasource._file_openers.keys()
    [None, '.bz2', '.gz', '.xz', '.lzma']
    >>> np.lib._datasource._file_openers['.gz'] is gzip.open
    True

    """

    def __init__(self):
        self._loaded = False
        self._file_openers = {None: open}

    def _load(self):
        if self._loaded:
            return

        try:
            import bz2
            self._file_openers[".bz2"] = bz2.open
        except ImportError:
            pass

        try:
            import gzip
            self._file_openers[".gz"] = gzip.open
        except ImportError:
            pass

        try:
            import lzma
            self._file_openers[".xz"] = lzma.open
            self._file_openers[".lzma"] = lzma.open
        except (ImportError, AttributeError):
            # There are incompatible backports of lzma that do not have the
            # lzma.open attribute, so catch that as well as ImportError.
            pass

        self._loaded = True

    def keys(self):
        """
        Return the keys of currently supported file openers.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list
            The keys are None for uncompressed files and the file extension
            strings (i.e. ``'.gz'``, ``'.xz'``) for supported compression
            methods.

        """
        self._load()
        return list(self._file_openers.keys())

    def __getitem__(self, key):
        self._load()
        return self._file_openers[key]

_file_openers = _FileOpeners()

def open(path, mode='r', destpath=os.curdir, encoding=None, newline=None):
    """
    Open `path` with `mode` and return the file object.

    If ``path`` is an URL, it will be downloaded, stored in the
    `DataSource` `destpath` directory and opened from there.

    Parameters
    ----------
    path : str or pathlib.Path
        Local file path or URL to open.
    mode : str, optional
        Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to
        append. Available modes depend on the type of object specified by
        path.  Default is 'r'.
    destpath : str, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.
    encoding : {None, str}, optional
        Open text file with given encoding. The default encoding will be
        what `open` uses.
    newline : {None, str}, optional
        Newline to use when reading text file.

    Returns
    -------
    out : file object
        The opened file.

    Notes
    -----
    This is a convenience function that instantiates a `DataSource` and
    returns the file object from ``DataSource.open(path)``.

    """

    ds = DataSource(destpath)
    return ds.open(path, mode, encoding=encoding, newline=newline)


@set_module('numpy.lib.npyio')
class DataSource:
    """
    DataSource(destpath='.')

    A generic data source file (file, http, ftp, ...).

    DataSources can be local files or remote files/URLs.  The files may
    also be compressed or uncompressed. DataSource hides some of the
    low-level details of downloading the file, allowing you to simply pass
    in a valid file path (or URL) and obtain a file object.

    Parameters
    ----------
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Notes
    -----
    URLs require a scheme string (``http://``) to be used, without it they
    will fail::

        >>> repos = np.lib.npyio.DataSource()
        >>> repos.exists('www.google.com/index.html')
        False
        >>> repos.exists('http://www.google.com/index.html')
        True

    Temporary directories are deleted when the DataSource is deleted.

    Examples
    --------
    ::

        >>> ds = np.lib.npyio.DataSource('/home/guido')
        >>> urlname = 'http://www.google.com/'
        >>> gfile = ds.open('http://www.google.com/')
        >>> ds.abspath(urlname)
        '/home/guido/www.google.com/index.html'

        >>> ds = np.lib.npyio.DataSource(None)  # use with temporary file
        >>> ds.open('/home/guido/foobar.txt')
        <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>
        >>> ds.abspath('/home/guido/foobar.txt')
        '/tmp/.../home/guido/foobar.txt'

    """

    def __init__(self, destpath=os.curdir):
        """Create a DataSource with a local path at destpath."""
        if destpath:
            self._destpath = os.path.abspath(destpath)
            self._istmpdest = False
        else:
            import tempfile  # deferring import to improve startup time
            self._destpath = tempfile.mkdtemp()
            self._istmpdest = True

    def __del__(self):
        # Remove temp directories
        if hasattr(self, '_istmpdest') and self._istmpdest:
            import shutil

            shutil.rmtree(self._destpath)

    def _iszip(self, filename):
        """Test if the filename is a zip file by looking at the file extension.

        """
        fname, ext = os.path.splitext(filename)
        return ext in _file_openers.keys()

    def _iswritemode(self, mode):
        """Test if the given mode will open a file for writing."""

        # Currently only used to test the bz2 files.
        _writemodes = ("w", "+")
        return any(c in _writemodes for c in mode)

    def _splitzipext(self, filename):
        """Split zip extension from filename and return filename.

        Returns
        -------
        base, zip_ext : {tuple}

        """

        if self._iszip(filename):
            return os.path.splitext(filename)
        else:
            return filename, None

    def _possible_names(self, filename):
        """Return a tuple containing compressed filename variations."""
        names = [filename]
        if not self._iszip(filename):
            for zipext in _file_openers.keys():
                if zipext:
                    names.append(filename+zipext)
        return names

    def _isurl(self, path):
        """Test if path is a net location.  Tests the scheme and netloc."""

        # We do this here to reduce the 'import numpy' initial import time.
        from urllib.parse import urlparse

        # BUG : URLs require a scheme string ('http://') to be used.
        #       www.google.com will fail.
        #       Should we prepend the scheme for those that don't have it and
        #       test that also?  Similar to the way we append .gz and test for
        #       for compressed versions of files.

        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        return bool(scheme and netloc)

    def _cache(self, path):
        """Cache the file specified by path.

        Creates a copy of the file in the datasource cache.

        """
        # We import these here because importing them is slow and
        # a significant fraction of numpy's total import time.
        import shutil
        from urllib.request import urlopen

        upath = self.abspath(path)

        # ensure directory exists
        if not os.path.exists(os.path.dirname(upath)):
            os.makedirs(os.path.dirname(upath))

        # TODO: Doesn't handle compressed files!
        if self._isurl(path):
            with urlopen(path) as openedurl:
                with _open(upath, 'wb') as f:
                    shutil.copyfileobj(openedurl, f)
        else:
            shutil.copyfile(path, upath)
        return upath

    def _findfile(self, path):
        """Searches for ``path`` and returns full path if found.

        If path is an URL, _findfile will cache a local copy and return the
        path to the cached file.  If path is a local file, _findfile will
        return a path to that local file.

        The search will include possible compressed versions of the file
        and return the first occurrence found.

        """

        # Build list of possible local file paths
        if not self._isurl(path):
            # Valid local paths
            filelist = self._possible_names(path)
            # Paths in self._destpath
            filelist += self._possible_names(self.abspath(path))
        else:
            # Cached URLs in self._destpath
            filelist = self._possible_names(self.abspath(path))
            # Remote URLs
            filelist = filelist + self._possible_names(path)

        for name in filelist:
            if self.exists(name):
                if self._isurl(name):
                    name = self._cache(name)
                return name
        return None

    def abspath(self, path):
        """
        Return absolute path of file in the DataSource directory.

        If `path` is an URL, then `abspath` will return either the location
        the file exists locally or the location it would exist when opened
        using the `open` method.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL.

        Returns
        -------
        out : str
            Complete path, including the `DataSource` destination directory.

        Notes
        -----
        The functionality is based on `os.path.abspath`.

        """
        # We do this here to reduce the 'import numpy' initial import time.
        from urllib.parse import urlparse

        # TODO:  This should be more robust.  Handles case where path includes
        #        the destpath, but not other sub-paths. Failing case:
        #        path = /home/guido/datafile.txt
        #        destpath = /home/alex/
        #        upath = self.abspath(path)
        #        upath == '/home/alex/home/guido/datafile.txt'

        # handle case where path includes self._destpath
        splitpath = path.split(self._destpath, 2)
        if len(splitpath) > 1:
            path = splitpath[1]
        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        netloc = self._sanitize_relative_path(netloc)
        upath = self._sanitize_relative_path(upath)
        return os.path.join(self._destpath, netloc, upath)

    def _sanitize_relative_path(self, path):
        """Return a sanitised relative path for which
        os.path.abspath(os.path.join(base, path)).startswith(base)
        """
        last = None
        path = os.path.normpath(path)
        while path != last:
            last = path
            # Note: os.path.join treats '/' as os.sep on Windows
            path = path.lstrip(os.sep).lstrip('/')
            path = path.lstrip(os.pardir).removeprefix('..')
            drive, path = os.path.splitdrive(path)  # for Windows
        return path

    def exists(self, path):
        """
        Test if path exists.

        Test if `path` exists as (and in this order):

        - a local file.
        - a remote URL that has been downloaded and stored locally in the
          `DataSource` directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL.

        Returns
        -------
        out : bool
            True if `path` exists.

        Notes
        -----
        When `path` is an URL, `exists` will return True if it's either
        stored locally in the `DataSource` directory, or is a valid remote
        URL.  `DataSource` does not discriminate between the two, the file
        is accessible if it exists in either location.

        """

        # First test for local path
        if os.path.exists(path):
            return True

        # We import this here because importing urllib is slow and
        # a significant fraction of numpy's total import time.
        from urllib.request import urlopen
        from urllib.error import URLError

        # Test cached url
        upath = self.abspath(path)
        if os.path.exists(upath):
            return True

        # Test remote url
        if self._isurl(path):
            try:
                netfile = urlopen(path)
                netfile.close()
                del(netfile)
                return True
            except URLError:
                return False
        return False

    def open(self, path, mode='r', encoding=None, newline=None):
        """
        Open and return file-like object.

        If `path` is an URL, it will be downloaded, stored in the
        `DataSource` directory and opened from there.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to open.
        mode : {'r', 'w', 'a'}, optional
            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
            'a' to append. Available modes depend on the type of object
            specified by `path`. Default is 'r'.
        encoding : {None, str}, optional
            Open text file with given encoding. The default encoding will be
            what `open` uses.
        newline : {None, str}, optional
            Newline to use when reading text file.

        Returns
        -------
        out : file object
            File object.

        """

        # TODO: There is no support for opening a file for writing which
        #       doesn't exist yet (creating a file).  Should there be?

        # TODO: Add a ``subdir`` parameter for specifying the subdirectory
        #       used to store URLs in self._destpath.

        if self._isurl(path) and self._iswritemode(mode):
            raise ValueError("URLs are not writeable")

        # NOTE: _findfile will fail on a new file opened for writing.
        found = self._findfile(path)
        if found:
            _fname, ext = self._splitzipext(found)
            if ext == 'bz2':
                mode.replace("+", "")
            return _file_openers[ext](found, mode=mode,
                                      encoding=encoding, newline=newline)
        else:
            raise FileNotFoundError(f"{path} not found.")


class Repository (DataSource):
    """
    Repository(baseurl, destpath='.')

    A data repository where multiple DataSource's share a base
    URL/directory.

    `Repository` extends `DataSource` by prepending a base URL (or
    directory) to all the files it handles. Use `Repository` when you will
    be working with multiple files from one base URL.  Initialize
    `Repository` with the base URL, then refer to each file by its filename
    only.

    Parameters
    ----------
    baseurl : str
        Path to the local directory or remote location that contains the
        data files.
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Examples
    --------
    To analyze all files in the repository, do something like this
    (note: this is not self-contained code)::

        >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')
        >>> for filename in filelist:
        ...     fp = repos.open(filename)
        ...     fp.analyze()
        ...     fp.close()

    Similarly you could use a URL for a repository::

        >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')

    """

    def __init__(self, baseurl, destpath=os.curdir):
        """Create a Repository with a shared url or directory of baseurl."""
        DataSource.__init__(self, destpath=destpath)
        self._baseurl = baseurl

    def __del__(self):
        DataSource.__del__(self)

    def _fullpath(self, path):
        """Return complete path for path.  Prepends baseurl if necessary."""
        splitpath = path.split(self._baseurl, 2)
        if len(splitpath) == 1:
            result = os.path.join(self._baseurl, path)
        else:
            result = path    # path contains baseurl already
        return result

    def _findfile(self, path):
        """Extend DataSource method to prepend baseurl to ``path``."""
        return DataSource._findfile(self, self._fullpath(path))

    def abspath(self, path):
        """
        Return absolute path of file in the Repository directory.

        If `path` is an URL, then `abspath` will return either the location
        the file exists locally or the location it would exist when opened
        using the `open` method.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL. This may, but does not
            have to, include the `baseurl` with which the `Repository` was
            initialized.

        Returns
        -------
        out : str
            Complete path, including the `DataSource` destination directory.

        """
        return DataSource.abspath(self, self._fullpath(path))

    def exists(self, path):
        """
        Test if path exists prepending Repository base URL to path.

        Test if `path` exists as (and in this order):

        - a local file.
        - a remote URL that has been downloaded and stored locally in the
          `DataSource` directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL. This may, but does not
            have to, include the `baseurl` with which the `Repository` was
            initialized.

        Returns
        -------
        out : bool
            True if `path` exists.

        Notes
        -----
        When `path` is an URL, `exists` will return True if it's either
        stored locally in the `DataSource` directory, or is a valid remote
        URL.  `DataSource` does not discriminate between the two, the file
        is accessible if it exists in either location.

        """
        return DataSource.exists(self, self._fullpath(path))

    def open(self, path, mode='r', encoding=None, newline=None):
        """
        Open and return file-like object prepending Repository base URL.

        If `path` is an URL, it will be downloaded, stored in the
        DataSource directory and opened from there.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to open. This may, but does not have to,
            include the `baseurl` with which the `Repository` was
            initialized.
        mode : {'r', 'w', 'a'}, optional
            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
            'a' to append. Available modes depend on the type of object
            specified by `path`. Default is 'r'.
        encoding : {None, str}, optional
            Open text file with given encoding. The default encoding will be
            what `open` uses.
        newline : {None, str}, optional
            Newline to use when reading text file.

        Returns
        -------
        out : file object
            File object.

        """
        return DataSource.open(self, self._fullpath(path), mode,
                               encoding=encoding, newline=newline)

    def listdir(self):
        """
        List files in the source Repository.

        Returns
        -------
        files : list of str or pathlib.Path
            List of file names (not containing a directory part).

        Notes
        -----
        Does not currently work for remote repositories.

        """
        if self._isurl(self._baseurl):
            raise NotImplementedError(
                  "Directory listing of URLs, not supported yet.")
        else:
            return os.listdir(self._baseurl)
<|endoftext|>import builtins
import collections.abc
import functools
import re
import sys
import warnings

import numpy as np
import numpy._core.numeric as _nx
from numpy._core import transpose, overrides
from numpy._core.numeric import (
    ones, zeros_like, arange, concatenate, array, asarray, asanyarray, empty,
    ndarray, take, dot, where, intp, integer, isscalar, absolute
    )
from numpy._core.umath import (
    pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, not_equal, subtract, minimum
    )
from numpy._core.fromnumeric import (
    ravel, nonzero, partition, mean, any, sum
    )
from numpy._core.numerictypes import typecodes
from numpy.lib._twodim_base_impl import diag
from numpy._core.multiarray import (
    _place, bincount, normalize_axis_index, _monotonicity,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy._core._multiarray_umath import _array_converter
from numpy._utils import set_module

# needed in this module for compatibility
from numpy.lib._histograms_impl import histogram, histogramdd  # noqa: F401


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'select', 'piecewise', 'trim_zeros', 'copy', 'iterable', 'percentile',
    'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'flip',
    'rot90', 'extract', 'place', 'vectorize', 'asarray_chkfinite', 'average',
    'bincount', 'digitize', 'cov', 'corrcoef',
    'median', 'sinc', 'hamming', 'hanning', 'bartlett',
    'blackman', 'kaiser', 'trapezoid', 'trapz', 'i0',
    'meshgrid', 'delete', 'insert', 'append', 'interp',
    'quantile'
    ]

# _QuantileMethods is a dictionary listing all the supported methods to
# compute quantile/percentile.
#
# Below virtual_index refers to the index of the element where the percentile
# would be found in the sorted sample.
# When the sample contains exactly the percentile wanted, the virtual_index is
# an integer to the index of this element.
# When the percentile wanted is in between two elements, the virtual_index
# is made of a integer part (a.k.a 'i' or 'left') and a fractional part
# (a.k.a 'g' or 'gamma')
#
# Each method in _QuantileMethods has two properties
# get_virtual_index : Callable
#   The function used to compute the virtual_index.
# fix_gamma : Callable
#   A function used for discrete methods to force the index to a specific value.
_QuantileMethods = dict(
    # --- HYNDMAN and FAN METHODS
    # Discrete methods
    inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: _inverted_cdf(n, quantiles),
        fix_gamma=None,  # should never be called
    ),
    averaged_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: (n * quantiles) - 1,
        fix_gamma=lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape,
            default_value=1.,
            conditioned_value=0.5,
            where=gamma == 0),
    ),
    closest_observation=dict(
        get_virtual_index=lambda n, quantiles: _closest_observation(n,
                                                                    quantiles),
        fix_gamma=None,  # should never be called
    ),
    # Continuous methods
    interpolated_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0, 1),
        fix_gamma=lambda gamma, _: gamma,
    ),
    hazen=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0.5, 0.5),
        fix_gamma=lambda gamma, _: gamma,
    ),
    weibull=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0, 0),
        fix_gamma=lambda gamma, _: gamma,
    ),
    # Default method.
    # To avoid some rounding issues, `(n-1) * quantiles` is preferred to
    # `_compute_virtual_index(n, quantiles, 1, 1)`.
    # They are mathematically equivalent.
    linear=dict(
        get_virtual_index=lambda n, quantiles: (n - 1) * quantiles,
        fix_gamma=lambda gamma, _: gamma,
    ),
    median_unbiased=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 1 / 3.0, 1 / 3.0),
        fix_gamma=lambda gamma, _: gamma,
    ),
    normal_unbiased=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 3 / 8.0, 3 / 8.0),
        fix_gamma=lambda gamma, _: gamma,
    ),
    # --- OTHER METHODS
    lower=dict(
        get_virtual_index=lambda n, quantiles: np.floor(
            (n - 1) * quantiles).astype(np.intp),
        fix_gamma=None,  # should never be called, index dtype is int
    ),
    higher=dict(
        get_virtual_index=lambda n, quantiles: np.ceil(
            (n - 1) * quantiles).astype(np.intp),
        fix_gamma=None,  # should never be called, index dtype is int
    ),
    midpoint=dict(
        get_virtual_index=lambda n, quantiles: 0.5 * (
                np.floor((n - 1) * quantiles)
                + np.ceil((n - 1) * quantiles)),
        fix_gamma=lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape,
            default_value=0.5,
            conditioned_value=0.,
            where=index % 1 == 0),
    ),
    nearest=dict(
        get_virtual_index=lambda n, quantiles: np.around(
            (n - 1) * quantiles).astype(np.intp),
        fix_gamma=None,
        # should never be called, index dtype is int
    ))


def _rot90_dispatcher(m, k=None, axes=None):
    return (m,)


@array_function_dispatch(_rot90_dispatcher)
def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.

    Rotation direction is from the first towards the second axis.
    This means for a 2D array with the default `k` and `axes`, the
    rotation will be counterclockwise.

    Parameters
    ----------
    m : array_like
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Returns
    -------
    y : ndarray
        A rotated view of `m`.

    See Also
    --------
    flip : Reverse the order of elements in an array along the given axis.
    fliplr : Flip an array horizontally.
    flipud : Flip an array vertically.

    Notes
    -----
    ``rot90(m, k=1, axes=(1,0))``  is the reverse of
    ``rot90(m, k=1, axes=(0,1))``

    ``rot90(m, k=1, axes=(1,0))`` is equivalent to
    ``rot90(m, k=-1, axes=(0,1))``

    Examples
    --------
    >>> import numpy as np
    >>> m = np.array([[1,2],[3,4]], int)
    >>> m
    array([[1, 2],
           [3, 4]])
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]])
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]])
    >>> m = np.arange(8).reshape((2,2,2))
    >>> np.rot90(m, 1, (1,2))
    array([[[1, 3],
            [0, 2]],
           [[5, 7],
            [4, 6]]])

    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    m = asanyarray(m)

    if axes[0] == axes[1] or absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if (axes[0] >= m.ndim or axes[0] < -m.ndim
        or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError("Axes={} out of range for array of ndim={}."
            .format(axes, m.ndim))

    k %= 4

    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])

    axes_list = arange(0, m.ndim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])

    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        # k == 3
        return flip(transpose(m, axes_list), axes[1])


def _flip_dispatcher(m, axis=None):
    return (m,)


@array_function_dispatch(_flip_dispatcher)
def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip over. The default,
         axis=None, will flip over all of the axes of the input array.
         If axis is negative it counts from the last to the first axis.

         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.

    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).

    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).

    flip(m, 1) is equivalent to fliplr(m).

    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.

    flip(m) corresponds to ``m[::-1,::-1,...,::-1]`` with ``::-1`` at all
    positions.

    flip(m, (0, 1)) corresponds to ``m[::-1,::-1,...]`` with ``::-1`` at
    position 0 and position 1.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> np.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    >>> rng = np.random.default_rng()
    >>> A = rng.normal(size=(3,4,5))
    >>> np.all(np.flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    if axis is None:
        indexer = (np.s_[::-1],) * m.ndim
    else:
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        indexer = [np.s_[:]] * m.ndim
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        indexer = tuple(indexer)
    return m[indexer]


@set_module('numpy')
def iterable(y):
    """
    Check whether or not an object can be iterated over.

    Parameters
    ----------
    y : object
      Input object.

    Returns
    -------
    b : bool
      Return ``True`` if the object has an iterator method or is a
      sequence and ``False`` otherwise.


    Examples
    --------
    >>> import numpy as np
    >>> np.iterable([1, 2, 3])
    True
    >>> np.iterable(2)
    False

    Notes
    -----
    In most cases, the results of ``np.iterable(obj)`` are consistent with
    ``isinstance(obj, collections.abc.Iterable)``. One notable exception is
    the treatment of 0-dimensional arrays::

        >>> from collections.abc import Iterable
        >>> a = np.array(1.0)  # 0-dimensional numpy array
        >>> isinstance(a, Iterable)
        True
        >>> np.iterable(a)
        False

    """
    try:
        iter(y)
    except TypeError:
        return False
    return True


def _weights_are_valid(weights, a, axis):
    """Validate weights array.

    We assume, weights is not None.
    """
    wgt = np.asanyarray(weights)

    # Sanity checks
    if a.shape != wgt.shape:
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of a and weights "
                "differ.")
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            raise ValueError(
                "Shape of weights must be consistent with "
                "shape of a along specified axis.")

        # setup wgt to broadcast along axis
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                for ax, s in enumerate(a.shape)))
    return wgt


def _average_dispatcher(a, axis=None, weights=None, returned=None, *,
                        keepdims=None):
    return (a, weights)


@array_function_dispatch(_average_dispatcher)
def average(a, axis=None, weights=None, returned=False, *,
            keepdims=np._NoValue):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        `axis=None`, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The array of weights must be the same shape as `a` if no axis is
        specified, otherwise the weights must have dimensions and shape
        consistent with `a` along the specified axis.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        The calculation is::

            avg = sum(a * weights) / sum(weights)

        where the sum is over all included elements.
        The only constraint on the values of `weights` is that `sum(weights)`
        must not be 0.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        *Note:* `keepdims` will not work with instances of `numpy.matrix`
        or other classes whose methods do not support `keepdims`.

        .. versionadded:: 1.23.0

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        Return the average along the specified axis. When `returned` is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. `sum_of_weights` is of the
        same type as `retval`. The result dtype follows a general pattern.
        If `weights` is None, the result dtype will be that of `a` , or ``float64``
        if `a` is integral. Otherwise, if `weights` is not None and `a` is non-
        integral, the result type will be the type of lowest precision capable of
        representing values of both `a` and `weights`. If `a` happens to be
        integral, the previous rules still applies but the result dtype will
        at least be ``float64``.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When `weights` does not have the same shape as `a`, and `axis=None`.
    ValueError
        When `weights` does not have dimensions and shape consistent with `a`
        along specified `axis`.

    See Also
    --------
    mean

    ma.average : average for masked arrays -- useful if your data contains
                 "missing" values
    numpy.result_type : Returns the type that results from applying the
                        numpy type promotion rules to the arguments.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(1, 5)
    >>> data
    array([1, 2, 3, 4])
    >>> np.average(data)
    2.5
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    4.0

    >>> data = np.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> np.average(data, weights=[1./4, 3./4])
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of a and weights differ.

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.average(data, axis=1, keepdims=True)
    array([[0.5],
           [2.5],
           [4.5]])

    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    array([3.4, 4.4])
    >>> np.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.
    """
    a = np.asanyarray(a)

    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")

    if keepdims is np._NoValue:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        avg_as_array = np.asanyarray(avg)
        scl = avg_as_array.dtype.type(a.size/avg_as_array.size)
    else:
        wgt = _weights_are_valid(weights=weights, a=a, axis=axis)

        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = avg_as_array = np.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg_as_array.shape:
            scl = np.broadcast_to(scl, avg_as_array.shape).copy()
        return avg, scl
    else:
        return avg


@set_module('numpy')
def asarray_chkfinite(a, dtype=None, order=None):
    """Convert the input to an array, checking for NaNs or Infs.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.  Success requires no NaNs or Infs.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of ndarray, a base
        class ndarray is returned.

    Raises
    ------
    ValueError
        Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

    See Also
    --------
    asarray : Create and array.
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    >>> import numpy as np

    Convert a list into an array. If all elements are finite, then
    ``asarray_chkfinite`` is identical to ``asarray``.

    >>> a = [1, 2]
    >>> np.asarray_chkfinite(a, dtype=float)
    array([1., 2.])

    Raises ValueError if array_like contains Nans or Infs.

    >>> a = [1, 2, np.inf]
    >>> try:
    ...     np.asarray_chkfinite(a)
    ... except ValueError:
    ...     print('ValueError')
    ...
    ValueError

    """
    a = asarray(a, dtype=dtype, order=order)
    if a.dtype.char in typecodes['AllFloat'] and not np.isfinite(a).all():
        raise ValueError(
            "array must not contain infs or NaNs")
    return a


def _piecewise_dispatcher(x, condlist, funclist, *args, **kw):
    yield x
    # support the undocumented behavior of allowing scalars
    if np.iterable(condlist):
        yield from condlist


@array_function_dispatch(_piecewise_dispatcher)
def piecewise(x, condlist, funclist, *args, **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    x : ndarray or scalar
        The input domain.
    condlist : list of bool arrays or bool scalars
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x)` is used as the output value.

        Each boolean array in `condlist` selects a piece of `x`,
        and should therefore be of the same shape as `x`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) == len(condlist) + 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(x,*args,**kw), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take a 1d array as input and give an 1d
        array or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    args : tuple, optional
        Any further arguments given to `piecewise` are passed to the functions
        upon execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then
        each function is called as ``f(x, 1, 'a')``.
    kw : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., alpha=1)``, then each function is called as
        ``f(x, alpha=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have a default value of 0.


    See Also
    --------
    choose, select, where

    Notes
    -----
    This is similar to choose or select, except that functions are
    evaluated on elements of `x` that satisfy the corresponding condition from
    `condlist`.

    The result is::

            |--
            |funclist[0](x[condlist[0]])
      out = |funclist[1](x[condlist[1]])
            |...
            |funclist[n2](x[condlist[n2]])
            |--

    Examples
    --------
    >>> import numpy as np

    Define the signum function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

    >>> x = np.linspace(-2.5, 2.5, 6)
    >>> np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    array([-1., -1., -1.,  1.,  1.,  1.])

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
    ``x >= 0``.

    >>> np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    array([2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    Apply the same function to a scalar value.

    >>> y = -2
    >>> np.piecewise(y, [y < 0, y >= 0], [lambda x: -x, lambda x: x])
    array(2)

    """
    x = asanyarray(x)
    n2 = len(funclist)

    # undocumented: single condition is promoted to a list of one condition
    if isscalar(condlist) or (
            not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        condlist = [condlist]

    condlist = asarray(condlist, dtype=bool)
    n = len(condlist)

    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
            .format(n, n, n+1)
        )

    y = zeros_like(x)
    for cond, func in zip(condlist, funclist):
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)

    return y


def _select_dispatcher(condlist, choicelist, default=None):
    yield from condlist
    yield from choicelist


@array_function_dispatch(_select_dispatcher)
def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    choicelist : list of ndarrays
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : scalar, optional
        The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : ndarray
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is True.

    See Also
    --------
    where : Return elements from one of two arrays depending on condition.
    take, choose, compress, diag, diagonal

    Examples
    --------
    >>> import numpy as np

    Beginning with an array of integers from 0 to 5 (inclusive),
    elements less than ``3`` are negated, elements greater than ``3``
    are squared, and elements not meeting either of these conditions
    (exactly ``3``) are replaced with a `default` value of ``42``.

    >>> x = np.arange(6)
    >>> condlist = [x<3, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 42)
    array([ 0,  1,  2, 42, 16, 25])

    When multiple conditions are satisfied, the first one encountered in
    `condlist` is used.

    >>> condlist = [x<=4, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 55)
    array([ 0,  1,  2,  3,  4, 25])

    """
    # Check the size of condlist and choicelist are the same, or abort.
    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    # Now that the dtype is known, handle the deprecated select([], []) case
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    # TODO: This preserves the Python int, float, complex manually to get the
    #       right `result_type` with NEP 50.  Most likely we will grow a better
    #       way to spell this (and this can be replaced).
    choicelist = [
        choice if type(choice) in (int, float, complex) else np.asarray(choice)
        for choice in choicelist]
    choicelist.append(default if type(default) in (int, float, complex)
                      else np.asarray(default))

    try:
        dtype = np.result_type(*choicelist)
    except TypeError as e:
        msg = f'Choicelist and default value do not have a common dtype: {e}'
        raise TypeError(msg) from None

    # Convert conditions to arrays and broadcast conditions and choices
    # as the shape is needed for the result. Doing it separately optimizes
    # for example when all choices are scalars.
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)

    # If cond array is not an ndarray in boolean format or scalar bool, abort.
    for i, cond in enumerate(condlist):
        if cond.dtype.type is not np.bool:
            raise TypeError(
                'invalid entry {} in condlist: should be boolean ndarray'.format(i))

    if choicelist[0].ndim == 0:
        # This may be common, so avoid the call.
        result_shape = condlist[0].shape
    else:
        result_shape = np.broadcast_arrays(condlist[0], choicelist[0])[0].shape

    result = np.full(result_shape, choicelist[-1], dtype)

    # Use np.copyto to burn each choicelist array onto result, using the
    # corresponding condlist as a boolean mask. This is done in reverse
    # order since the first choice should take precedence.
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)

    return result


def _copy_dispatcher(a, order=None, subok=None):
    return (a,)


@array_function_dispatch(_copy_dispatcher)
def copy(a, order='K', subok=False):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :meth:`ndarray.copy` are very
        similar, but have different default values for their order=
        arguments.)
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array (defaults to False).

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    See Also
    --------
    ndarray.copy : Preferred method for creating an array copy

    Notes
    -----
    This is equivalent to:

    >>> np.array(a, copy=True)  #doctest: +SKIP

    The copy made of the data is shallow, i.e., for arrays with object dtype,
    the new array will point to the same objects.
    See Examples from `ndarray.copy`.

    Examples
    --------
    >>> import numpy as np

    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> x[0] == y[0]
    True
    >>> x[0] == z[0]
    False

    Note that, np.copy clears previously set WRITEABLE=False flag.

    >>> a = np.array([1, 2, 3])
    >>> a.flags["WRITEABLE"] = False
    >>> b = np.copy(a)
    >>> b.flags["WRITEABLE"]
    True
    >>> b[0] = 3
    >>> b
    array([3, 2, 3])
    """
    return array(a, order=order, subok=subok, copy=True)

# Basic operations


def _gradient_dispatcher(f, *varargs, axis=None, edge_order=None):
    yield f
    yield from varargs


@array_function_dispatch(_gradient_dispatcher)
def gradient(f, *varargs, axis=None, edge_order=1):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    varargs : list of scalar or array, optional
        Spacing between f values. Default unitary spacing for all dimensions.
        Spacing can be specified using:

        1. single scalar to specify a sample distance for all dimensions.
        2. N scalars to specify a constant sample distance for each dimension.
           i.e. `dx`, `dy`, `dz`, ...
        3. N arrays to specify the coordinates of the values along each
           dimension of F. The length of the array must match the size of
           the corresponding dimension
        4. Any combination of N scalars/arrays with the meaning of 2. and 3.

        If `axis` is given, the number of varargs must equal the number of axes.
        Default: 1. (see Examples below).

    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries. Default: 1.
    axis : None or int or tuple of ints, optional
        Gradient is calculated only along the given axis or axes
        The default (axis = None) is to calculate the gradient for all the axes
        of the input array. axis may be negative, in which case it counts from
        the last to the first axis.

    Returns
    -------
    gradient : ndarray or tuple of ndarray
        A tuple of ndarrays (or a single ndarray if there is only one
        dimension) corresponding to the derivatives of f with respect
        to each dimension. Each derivative has the same shape as f.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.array([1, 2, 4, 7, 11, 16])
    >>> np.gradient(f)
    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    >>> np.gradient(f, 2)
    array([0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

    Spacing can be also specified with an array that represents the coordinates
    of the values F along the dimensions.
    For instance a uniform spacing:

    >>> x = np.arange(f.size)
    >>> np.gradient(f, x)
    array([1. ,  1.5,  2.5,  3.5,  4.5,  5. ])

    Or a non uniform one:

    >>> x = np.array([0., 1., 1.5, 3.5, 4., 6.])
    >>> np.gradient(f, x)
    array([1. ,  3. ,  3.5,  6.7,  6.9,  2.5])

    For two dimensional arrays, the return will be two arrays ordered by
    axis. In this example the first array stands for the gradient in
    rows and the second one in columns direction:

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]))
    (array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]),
     array([[1. , 2.5, 4. ],
            [1. , 1. , 1. ]]))

    In this example the spacing is also specified:
    uniform for axis=0 and non uniform for axis=1

    >>> dx = 2.
    >>> y = [1., 1.5, 3.5]
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]), dx, y)
    (array([[ 1. ,  1. , -0.5],
            [ 1. ,  1. , -0.5]]),
     array([[2. , 2. , 2. ],
            [2. , 1.7, 0.5]]))

    It is possible to specify how boundaries are treated using `edge_order`

    >>> x = np.array([0, 1, 2, 3, 4])
    >>> f = x**2
    >>> np.gradient(f, edge_order=1)
    array([1.,  2.,  4.,  6.,  7.])
    >>> np.gradient(f, edge_order=2)
    array([0., 2., 4., 6., 8.])

    The `axis` keyword can be used to specify a subset of axes of which the
    gradient is calculated

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])

    The `varargs` argument defines the spacing between sample points in the
    input array. It can take two forms:

    1. An array, specifying coordinates, which may be unevenly spaced:

    >>> x = np.array([0., 2., 3., 6., 8.])
    >>> y = x ** 2
    >>> np.gradient(y, x, edge_order=2)
    array([ 0.,  4.,  6., 12., 16.])

    2. A scalar, representing the fixed sample distance:

    >>> dx = 2
    >>> x = np.array([0., 2., 4., 6., 8.])
    >>> y = x ** 2
    >>> np.gradient(y, dx, edge_order=2)
    array([ 0.,  4.,  8., 12., 16.])

    It's possible to provide different data for spacing along each dimension.
    The number of arguments must match the number of dimensions in the input
    data.

    >>> dx = 2
    >>> dy = 3
    >>> x = np.arange(0, 6, dx)
    >>> y = np.arange(0, 9, dy)
    >>> xs, ys = np.meshgrid(x, y)
    >>> zs = xs + 2 * ys
    >>> np.gradient(zs, dy, dx)  # Passing two scalars
    (array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]),
     array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]))

    Mixing scalars and arrays is also allowed:

    >>> np.gradient(zs, y, dx)  # Passing one array and one scalar
    (array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]),
     array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]))

    Notes
    -----
    Assuming that :math:`f\\in C^{3}` (i.e., :math:`f` has at least 3 continuous
    derivatives) and let :math:`h_{*}` be a non-homogeneous stepsize, we
    minimize the "consistency error" :math:`\\eta_{i}` between the true gradient
    and its estimate from a linear combination of the neighboring grid-points:

    .. math::

        \\eta_{i} = f_{i}^{\\left(1\\right)} -
                    \\left[ \\alpha f\\left(x_{i}\\right) +
                            \\beta f\\left(x_{i} + h_{d}\\right) +
                            \\gamma f\\left(x_{i}-h_{s}\\right)
                    \\right]

    By substituting :math:`f(x_{i} + h_{d})` and :math:`f(x_{i} - h_{s})`
    with their Taylor series expansion, this translates into solving
    the following the linear system:

    .. math::

        \\left\\{
            \\begin{array}{r}
                \\alpha+\\beta+\\gamma=0 \\\\
                \\beta h_{d}-\\gamma h_{s}=1 \\\\
                \\beta h_{d}^{2}+\\gamma h_{s}^{2}=0
            \\end{array}
        \\right.

    The resulting approximation of :math:`f_{i}^{(1)}` is the following:

    .. math::

        \\hat f_{i}^{(1)} =
            \\frac{
                h_{s}^{2}f\\left(x_{i} + h_{d}\\right)
                + \\left(h_{d}^{2} - h_{s}^{2}\\right)f\\left(x_{i}\\right)
                - h_{d}^{2}f\\left(x_{i}-h_{s}\\right)}
                { h_{s}h_{d}\\left(h_{d} + h_{s}\\right)}
            + \\mathcal{O}\\left(\\frac{h_{d}h_{s}^{2}
                                + h_{s}h_{d}^{2}}{h_{d}
                                + h_{s}}\\right)

    It is worth noting that if :math:`h_{s}=h_{d}`
    (i.e., data are evenly spaced)
    we find the standard second order approximation:

    .. math::

        \\hat f_{i}^{(1)}=
            \\frac{f\\left(x_{i+1}\\right) - f\\left(x_{i-1}\\right)}{2h}
            + \\mathcal{O}\\left(h^{2}\\right)

    With a similar procedure the forward/backward approximations used for
    boundaries can be derived.

    References
    ----------
    .. [1]  Quarteroni A., Sacco R., Saleri F. (2007) Numerical Mathematics
            (Texts in Applied Mathematics). New York: Springer.
    .. [2]  Durran D. R. (1999) Numerical Methods for Wave Equations
            in Geophysical Fluid Dynamics. New York: Springer.
    .. [3]  Fornberg B. (1988) Generation of Finite Difference Formulas on
            Arbitrarily Spaced Grids,
            Mathematics of Computation 51, no. 184 : 699-706.
            `PDF <https://www.ams.org/journals/mcom/1988-51-184/
            S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_.
    """
    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype
    if otype.type is np.datetime64:
        # the timedelta dtype with the same unit information
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        # view as timedelta to allow addition
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        # result allocation
        out = np.empty_like(f, dtype=otype)

        # spacing for the current axis
        uniform_spacing = np.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2)/(dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = - dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = - (dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)


def _diff_dispatcher(a, n=None, axis=None, prepend=None, append=None):
    return (a, prepend, append)


@array_function_dispatch(_diff_dispatcher)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    See Also
    --------
    gradient, ediff1d, cumsum

    Notes
    -----
    Type is preserved for boolean arrays, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer arrays, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly:

    >>> u8_arr = np.array([1, 0], dtype=np.uint8)
    >>> np.diff(u8_arr)
    array([255], dtype=uint8)
    >>> u8_arr[1,...] - u8_arr[0,...]
    np.uint8(255)

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    >>> i16_arr = u8_arr.astype(np.int16)
    >>> np.diff(i16_arr)
    array([-1], dtype=int16)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    >>> x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
    >>> np.diff(x)
    array([1, 1], dtype='timedelta64[D]')

    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))

    a = asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = normalize_axis_index(axis, nd)

    combined = []
    if prepend is not np._NoValue:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not np._NoValue:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = not_equal if a.dtype == np.bool else subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


def _interp_dispatcher(x, xp, fp, left=None, right=None, period=None):
    return (x, xp, fp)


@array_function_dispatch(_interp_dispatcher)
def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.

    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.

    fp : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.

    left : optional float or complex corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.

    right : optional float or complex corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.

    period : None or float, optional
        A period for the x-coordinates. This parameter allows the proper
        interpolation of angular x-coordinates. Parameters `left` and `right`
        are ignored if `period` is specified.

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length
        If `xp` or `fp` are not 1-D sequences
        If `period == 0`

    See Also
    --------
    scipy.interpolate

    Warnings
    --------
    The x-coordinate sequence is expected to be increasing, but this is not
    explicitly enforced.  However, if the sequence `xp` is non-increasing,
    interpolation results are meaningless.

    Note that, since NaN is unsortable, `xp` also cannot contain NaNs.

    A simple check for `xp` being strictly increasing is::

        np.all(np.diff(xp) > 0)

    Examples
    --------
    >>> import numpy as np
    >>> xp = [1, 2, 3]
    >>> fp = [3, 2, 0]
    >>> np.interp(2.5, xp, fp)
    1.0
    >>> np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
    array([3.  , 3.  , 2.5 , 0.56, 0.  ])
    >>> UNDEF = -99.0
    >>> np.interp(3.14, xp, fp, right=UNDEF)
    -99.0

    Plot an interpolant to the sine function:

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.sin(x)
    >>> xvals = np.linspace(0, 2*np.pi, 50)
    >>> yinterp = np.interp(xvals, x, y)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(xvals, yinterp, '-x')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    Interpolation with periodic x-coordinates:

    >>> x = [-180, -170, -185, 185, -10, -5, 0, 365]
    >>> xp = [190, -190, 350, -350]
    >>> fp = [5, 10, 3, 4]
    >>> np.interp(x, xp, fp, period=360)
    array([7.5 , 5.  , 8.75, 6.25, 3.  , 3.25, 3.5 , 3.75])

    Complex interpolation:

    >>> x = [1.5, 4.0]
    >>> xp = [2,3,5]
    >>> fp = [1.0j, 0, 2+3j]
    >>> np.interp(x, xp, fp)
    array([0.+1.j , 1.+1.5j])

    """

    fp = np.asarray(fp)

    if np.iscomplexobj(fp):
        interp_func = compiled_interp_complex
        input_dtype = np.complex128
    else:
        interp_func = compiled_interp
        input_dtype = np.float64

    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None

        x = np.asarray(x, dtype=np.float64)
        xp = np.asarray(xp, dtype=np.float64)
        fp = np.asarray(fp, dtype=input_dtype)

        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        # normalizing periodic boundaries
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))

    return interp_func(x, xp, fp, left, right)


def _angle_dispatcher(z, deg=None):
    return (z,)


@array_function_dispatch(_angle_dispatcher)
def angle(z, deg=False):
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.

    See Also
    --------
    arctan2
    absolute

    Notes
    -----
    This function passes the imaginary and real parts of the argument to
    `arctan2` to compute the result; consequently, it follows the convention
    of `arctan2` when the magnitude of the argument is zero. See example.

    Examples
    --------
    >>> import numpy as np
    >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
    array([ 0.        ,  1.57079633,  0.78539816]) # may vary
    >>> np.angle(1+1j, deg=True)                  # in degrees
    45.0
    >>> np.angle([0., -0., complex(0., -0.), complex(-0., -0.)])  # convention
    array([ 0.        ,  3.14159265, -0.        , -3.14159265])

    """
    z = asanyarray(z)
    if issubclass(z.dtype.type, _nx.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z

    a = arctan2(zimag, zreal)
    if deg:
        a *= 180/pi
    return a


def _unwrap_dispatcher(p, discont=None, axis=None, *, period=None):
    return (p,)


@array_function_dispatch(_unwrap_dispatcher)
def unwrap(p, discont=None, axis=-1, *, period=2*pi):
    r"""
    Unwrap by taking the complement of large deltas with respect to the period.

    This unwraps a signal `p` by changing elements which have an absolute
    difference from their predecessor of more than ``max(discont, period/2)``
    to their `period`-complementary values.

    For the default case where `period` is :math:`2\pi` and `discont` is
    :math:`\pi`, this unwraps a radian phase `p` such that adjacent differences
    are never greater than :math:`\pi` by adding :math:`2k\pi` for some
    integer :math:`k`.

    Parameters
    ----------
    p : array_like
        Input array.
    discont : float, optional
        Maximum discontinuity between values, default is ``period/2``.
        Values below ``period/2`` are treated as if they were ``period/2``.
        To have an effect different from the default, `discont` should be
        larger than ``period/2``.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.
    period : float, optional
        Size of the range over which the input wraps. By default, it is
        ``2 pi``.

        .. versionadded:: 1.21.0

    Returns
    -------
    out : ndarray
        Output array.

    See Also
    --------
    rad2deg, deg2rad

    Notes
    -----
    If the discontinuity in `p` is smaller than ``period/2``,
    but larger than `discont`, no unwrapping is done because taking
    the complement would only make the discontinuity larger.

    Examples
    --------
    >>> import numpy as np
    >>> phase = np.linspace(0, np.pi, num=5)
    >>> phase[3:] += np.pi
    >>> phase
    array([ 0.        ,  0.78539816,  1.57079633,  5.49778714,  6.28318531]) # may vary
    >>> np.unwrap(phase)
    array([ 0.        ,  0.78539816,  1.57079633, -0.78539816,  0.        ]) # may vary
    >>> np.unwrap([0, 1, 2, -1, 0], period=4)
    array([0, 1, 2, 3, 4])
    >>> np.unwrap([ 1, 2, 3, 4, 5, 6, 1, 2, 3], period=6)
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.unwrap([2, 3, 4, 5, 2, 3, 4, 5], period=4)
    array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> phase_deg = np.mod(np.linspace(0 ,720, 19), 360) - 180
    >>> np.unwrap(phase_deg, period=360)
    array([-180., -140., -100.,  -60.,  -20.,   20.,   60.,  100.,  140.,
            180.,  220.,  260.,  300.,  340.,  380.,  420.,  460.,  500.,
            540.])
    """
    p = asarray(p)
    nd = p.ndim
    dd = diff(p, axis=axis)
    if discont is None:
        discont = period/2
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    dtype = np.result_type(dd, period)
    if _nx.issubdtype(dtype, _nx.integer):
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        # for `mask = (abs(dd) == period/2)`, the above line made
        # `ddmod[mask] == -period/2`. correct these such that
        # `ddmod[mask] == sign(dd[mask])*period/2`.
        _nx.copyto(ddmod, interval_high,
                   where=(ddmod == interval_low) & (dd > 0))
    ph_correct = ddmod - dd
    _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    up = array(p, copy=True, dtype=dtype)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up


def _sort_complex(a):
    return (a,)


@array_function_dispatch(_sort_complex)
def sort_complex(a):
    """
    Sort a complex array using the real part first, then the imaginary part.

    Parameters
    ----------
    a : array_like
        Input array

    Returns
    -------
    out : complex ndarray
        Always returns a sorted complex array.

    Examples
    --------
    >>> import numpy as np
    >>> np.sort_complex([5, 3, 6, 2, 1])
    array([1.+0.j, 2.+0.j, 3.+0.j, 5.+0.j, 6.+0.j])

    >>> np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
    array([1.+2.j,  2.-1.j,  3.-3.j,  3.-2.j,  3.+5.j])

    """
    b = array(a, copy=True)
    b.sort()
    if not issubclass(b.dtype.type, _nx.complexfloating):
        if b.dtype.char in 'bhBH':
            return b.astype('F')
        elif b.dtype.char == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b


def _arg_trim_zeros(filt):
    """Return indices of the first and last non-zero element.

    Parameters
    ----------
    filt : array_like
        Input array.

    Returns
    -------
    start, stop : ndarray
        Two arrays containing the indices of the first and last non-zero
        element in each dimension.

    See also
    --------
    trim_zeros

    Examples
    --------
    >>> import numpy as np
    >>> _arg_trim_zeros(np.array([0, 0, 1, 1, 0]))
    (array([2]), array([3]))
    """
    nonzero = (
        np.argwhere(filt)
        if filt.dtype != np.object_
        # Historically, `trim_zeros` treats `None` in an object array
        # as non-zero while argwhere doesn't, account for that
        else np.argwhere(filt != 0)
    )
    if nonzero.size == 0:
        start = stop = np.array([], dtype=np.intp)
    else:
        start = nonzero.min(axis=0)
        stop = nonzero.max(axis=0)
    return start, stop


def _trim_zeros(filt, trim=None, axis=None):
    return (filt,)


@array_function_dispatch(_trim_zeros)
def trim_zeros(filt, trim='fb', axis=None):
    """Remove values along a dimension which are zero along all other.

    Parameters
    ----------
    filt : array_like
        Input array.
    trim : {"fb", "f", "b"}, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. By default, zeros are trimmed on both sides.
        Front and back refer to the edges of a dimension, with "front" refering
        to the side with the lowest index 0, and "back" refering to the highest
        index (or index -1).
    axis : int or sequence, optional
        If None, `filt` is cropped such, that the smallest bounding box is
        returned that still contains all values which are not zero.
        If an axis is specified, `filt` will be sliced in that dimension only
        on the sides specified by `trim`. The remaining area will be the
        smallest that still contains all values wich are not zero.

    Returns
    -------
    trimmed : ndarray or sequence
        The result of trimming the input. The number of dimensions and the
        input data type are preserved.

    Notes
    -----
    For all-zero arrays, the first axis is trimmed first.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_zeros(a)
    array([1, 2, 3, 0, 2, 1])

    >>> np.trim_zeros(a, trim='b')
    array([0, 0, 0, ..., 0, 2, 1])

    Multiple dimensions are supported.

    >>> b = np.array([[0, 0, 2, 3, 0, 0],
    ...               [0, 1, 0, 3, 0, 0],
    ...               [0, 0, 0, 0, 0, 0]])
    >>> np.trim_zeros(b)
    array([[0, 2, 3],
           [1, 0, 3]])

    >>> np.trim_zeros(b, axis=-1)
    array([[0, 2, 3],
           [1, 0, 3],
           [0, 0, 0]])

    The input data type is preserved, list/tuple in means list/tuple out.

    >>> np.trim_zeros([0, 1, 2, 0])
    [1, 2]

    """
    filt_ = np.asarray(filt)

    trim = trim.lower()
    if trim not in {"fb", "bf", "f", "b"}:
        raise ValueError(f"unexpected character(s) in `trim`: {trim!r}")

    start, stop = _arg_trim_zeros(filt_)
    stop += 1  # Adjust for slicing

    if start.size == 0:
        # filt is all-zero -> assign same values to start and stop so that
        # resulting slice will be empty
        start = stop = np.zeros(filt_.ndim, dtype=np.intp)
    else:
        if 'f' not in trim:
            start = (None,) * filt_.ndim
        if 'b' not in trim:
            stop = (None,) * filt_.ndim

    if len(start) == 1:
        # filt is 1D -> don't use multi-dimensional slicing to preserve
        # non-array input types
        sl = slice(start[0], stop[0])
    elif axis is None:
        # trim all axes
        sl = tuple(slice(*x) for x in zip(start, stop))
    else:
        # only trim single axis
        axis = normalize_axis_index(axis, filt_.ndim)
        sl = (slice(None),) * axis + (slice(start[axis], stop[axis]),) + (...,)

    trimmed = filt[sl]
    return trimmed



def _extract_dispatcher(condition, arr):
    return (condition, arr)


@array_function_dispatch(_extract_dispatcher)
def extract(condition, arr):
    """
    Return the elements of an array that satisfy some condition.

    This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.  If
    `condition` is boolean ``np.extract`` is equivalent to ``arr[condition]``.

    Note that `place` does the exact opposite of `extract`.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of `arr`
        to extract.
    arr : array_like
        Input array of the same size as `condition`.

    Returns
    -------
    extract : ndarray
        Rank 1 array of values from `arr` where `condition` is True.

    See Also
    --------
    take, put, copyto, compress, place

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(12).reshape((3, 4))
    >>> arr
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> condition = np.mod(arr, 3)==0
    >>> condition
    array([[ True, False, False,  True],
           [False, False,  True, False],
           [False,  True, False, False]])
    >>> np.extract(condition, arr)
    array([0, 3, 6, 9])


    If `condition` is boolean:

    >>> arr[condition]
    array([0, 3, 6, 9])

    """
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])


def _place_dispatcher(arr, mask, vals):
    return (arr, mask, vals)


@array_function_dispatch(_place_dispatcher)
def place(arr, mask, vals):
    """
    Change elements of an array based on conditional and input values.

    Similar to ``np.copyto(arr, vals, where=mask)``, the difference is that
    `place` uses the first N elements of `vals`, where N is the number of
    True values in `mask`, while `copyto` uses the elements where `mask`
    is True.

    Note that `extract` does the exact opposite of `place`.

    Parameters
    ----------
    arr : ndarray
        Array to put data into.
    mask : array_like
        Boolean mask array. Must have the same size as `a`.
    vals : 1-D sequence
        Values to put into `a`. Only the first N elements are used, where
        N is the number of True values in `mask`. If `vals` is smaller
        than N, it will be repeated, and if elements of `a` are to be masked,
        this sequence must be non-empty.

    See Also
    --------
    copyto, put, take, extract

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(6).reshape(2, 3)
    >>> np.place(arr, arr>2, [44, 55])
    >>> arr
    array([[ 0,  1,  2],
           [44, 55, 44]])

    """
    return _place(arr, mask, vals)


def disp(mesg, device=None, linefeed=True):
    """
    Display a message on a device.

    .. deprecated:: 2.0
        Use your own printing function instead.

    Parameters
    ----------
    mesg : str
        Message to display.
    device : object
        Device to write message. If None, defaults to ``sys.stdout`` which is
        very similar to ``print``. `device` needs to have ``write()`` and
        ``flush()`` methods.
    linefeed : bool, optional
        Option whether to print a line feed or not. Defaults to True.

    Raises
    ------
    AttributeError
        If `device` does not have a ``write()`` or ``flush()`` method.

    Examples
    --------
    >>> import numpy as np

    Besides ``sys.stdout``, a file-like object can also be used as it has
    both required methods:

    >>> from io import StringIO
    >>> buf = StringIO()
    >>> np.disp('"Display" in a file', device=buf)
    >>> buf.getvalue()
    '"Display" in a file\\n'

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`disp` is deprecated, "
        "use your own printing function instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    if device is None:
        device = sys.stdout
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    device.flush()
    return


# See https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)


def _parse_gufunc_signature(signature):
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]].
    """
    signature = re.sub(r'\s+', '', signature)

    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                  for arg in re.findall(_ARGUMENT, arg_list)]
                 for arg_list in signature.split('->'))


def _update_dim_sizes(dim_sizes, arg, core_dims):
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
    dim_sizes : Dict[str, int]
        Sizes of existing core dimensions. Will be updated in-place.
    arg : ndarray
        Argument to examine.
    core_dims : Tuple[str, ...]
        Core dimensions for this argument.
    """
    if not core_dims:
        return

    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))

    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib._stride_tricks_impl._broadcast_shape(
        *broadcast_args
    )
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    """Helper for calculating broadcast shapes with core dimensions."""
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes,
                   results=None):
    """Helper for creating output arrays in vectorize."""
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    if dtypes is None:
        dtypes = [None] * len(shapes)
    if results is None:
        arrays = tuple(np.empty(shape=shape, dtype=dtype)
                       for shape, dtype in zip(shapes, dtypes))
    else:
        arrays = tuple(np.empty_like(result, shape=shape, dtype=dtype)
                       for result, shape, dtype
                       in zip(results, shapes, dtypes))
    return arrays


def _get_vectorize_dtype(dtype):
    if dtype.char in "SU":
        return dtype.char
    return dtype


@set_module('numpy')
class vectorize:
    """
    vectorize(pyfunc=np._NoValue, otypes=None, doc=None, excluded=None,
    cache=False, signature=None)

    Returns an object that acts like pyfunc, but takes arrays as input.

    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns a single numpy array or a tuple of numpy
