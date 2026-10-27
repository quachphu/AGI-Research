    The sort order of ``np.nan`` is bigger than ``np.inf``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([7, 1, 7, 7, 1, 5, 7, 2, 3, 2, 6, 2, 3, 0])
    >>> p = np.partition(a, 4)
    >>> p
    array([0, 1, 2, 1, 2, 5, 2, 3, 3, 6, 7, 7, 7, 7]) # may vary

    ``p[4]`` is 2;  all elements in ``p[:4]`` are less than or equal
    to ``p[4]``, and all elements in ``p[5:]`` are greater than or
    equal to ``p[4]``.  The partition is::

        [0, 1, 2, 1], [2], [5, 2, 3, 3, 6, 7, 7, 7, 7]

    The next example shows the use of multiple values passed to `kth`.

    >>> p2 = np.partition(a, (4, 8))
    >>> p2
    array([0, 1, 2, 1, 2, 3, 3, 2, 5, 6, 7, 7, 7, 7])

    ``p2[4]`` is 2  and ``p2[8]`` is 5.  All elements in ``p2[:4]``
    are less than or equal to ``p2[4]``, all elements in ``p2[5:8]``
    are greater than or equal to ``p2[4]`` and less than or equal to
    ``p2[8]``, and all elements in ``p2[9:]`` are greater than or
    equal to ``p2[8]``.  The partition is::

        [0, 1, 2, 1], [2], [3, 3, 2], [5], [6, 7, 7, 7, 7]
    """
    if axis is None:
        # flatten returns (1, N) for np.matrix, so always use the last axis
        a = asanyarray(a).flatten()
        axis = -1
    else:
        a = asanyarray(a).copy(order="K")
    a.partition(kth, axis=axis, kind=kind, order=order)
    return a


def _argpartition_dispatcher(a, kth, axis=None, kind=None, order=None):
    return (a,)


@array_function_dispatch(_argpartition_dispatcher)
def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Perform an indirect partition along the given axis using the
    algorithm specified by the `kind` keyword. It returns an array of
    indices of the same shape as `a` that index data along the given
    axis in partitioned order.

    Parameters
    ----------
    a : array_like
        Array to sort.
    kth : int or sequence of ints
        Element index to partition by. The k-th element will be in its
        final sorted position and all smaller elements will be moved
        before it and all larger elements behind it. The order of all
        elements in the partitions is undefined. If provided with a
        sequence of k-th it will partition all of them into their sorted
        position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int or None, optional
        Axis along which to sort. The default is -1 (the last axis). If
        None, the flattened array is used.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument
        specifies which fields to compare first, second, etc. A single
        field can be specified as a string, and not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    index_array : ndarray, int
        Array of indices that partition `a` along the specified axis.
        If `a` is one-dimensional, ``a[index_array]`` yields a partitioned `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the partitioned `a`, irrespective of dimensionality.

    See Also
    --------
    partition : Describes partition algorithms used.
    ndarray.partition : Inplace partition.
    argsort : Full indirect sort.
    take_along_axis : Apply ``index_array`` from argpartition
                      to an array as if by calling partition.

    Notes
    -----
    The returned indices are not guaranteed to be sorted according to
    the values. Furthermore, the default selection algorithm ``introselect``
    is unstable, and hence the returned indices are not guaranteed
    to be the earliest/latest occurrence of the element.

    `argpartition` works for real/complex inputs with nan values,
    see `partition` for notes on the enhanced sort order and
    different selection algorithms.

    Examples
    --------
    One dimensional array:

    >>> import numpy as np
    >>> x = np.array([3, 4, 2, 1])
    >>> x[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary
    >>> x[np.argpartition(x, (1, 3))]
    array([1, 2, 3, 4]) # may vary

    >>> x = [3, 4, 2, 1]
    >>> np.array(x)[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary

    Multi-dimensional array:

    >>> x = np.array([[3, 4, 2], [1, 3, 1]])
    >>> index_array = np.argpartition(x, kth=1, axis=-1)
    >>> # below is the same as np.partition(x, kth=1)
    >>> np.take_along_axis(x, index_array, axis=-1)
    array([[2, 3, 4],
           [1, 1, 3]])

    """
    return _wrapfunc(a, 'argpartition', kth, axis=axis, kind=kind, order=order)


def _sort_dispatcher(a, axis=None, kind=None, order=None, *, stable=None):
    return (a,)


@array_function_dispatch(_sort_dispatcher)
def sort(a, axis=-1, kind=None, order=None, *, stable=None):
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort or radix sort under the covers and,
        in general, the actual implementation will vary with data type.
        The 'mergesort' option is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    stable : bool, optional
        Sort stability. If ``True``, the returned array will maintain
        the relative order of ``a`` values which compare as equal.
        If ``False`` or ``None``, this is not guaranteed. Internally,
        this option selects ``kind='stable'``. Default: ``None``.

        .. versionadded:: 2.0.0

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted array.
    partition : Partial sort.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The four algorithms implemented in NumPy have the following
    properties:

    =========== ======= ============= ============ ========
       kind      speed   worst case    work space   stable
    =========== ======= ============= ============ ========
    'quicksort'    1     O(n^2)            0          no
    'heapsort'     3     O(n*log(n))       0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'timsort'      2     O(n*log(n))      ~n/2        yes
    =========== ======= ============= ============ ========

    .. note:: The datatype determines which of 'mergesort' or 'timsort'
       is actually used, even if 'mergesort' is specified. User selection
       at a finer scale is not currently available.

    For performance, ``sort`` makes a temporary copy if needed to make the data
    `contiguous <https://numpy.org/doc/stable/glossary.html#term-contiguous>`_
    in memory along the sort axis. For even better performance and reduced
    memory consumption, ensure that the array is already contiguous along the
    sort axis.

    The sort order for complex numbers is lexicographic. If both the real
    and imaginary parts are non-nan then the order is determined by the
    real parts except when they are equal, in which case the order is
    determined by the imaginary parts.

    Previous to numpy 1.4.0 sorting real and complex arrays containing nan
    values led to undefined behaviour. In numpy versions >= 1.4.0 nan
    values are sorted to the end. The extended sort order is:

      * Real: [R, nan]
      * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]

    where R is a non-nan real value. Complex values with the same nan
    placements are sorted according to the non-nan part if it exists.
    Non-nan values are sorted as before.

    quicksort has been changed to:
    `introsort <https://en.wikipedia.org/wiki/Introsort>`_.
    When sorting does not make enough progress it switches to
    `heapsort <https://en.wikipedia.org/wiki/Heapsort>`_.
    This implementation makes quicksort O(n*log(n)) in the worst case.

    'stable' automatically chooses the best stable sorting algorithm
    for the data type being sorted.
    It, along with 'mergesort' is currently mapped to
    `timsort <https://en.wikipedia.org/wiki/Timsort>`_
    or `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
    depending on the data type.
    API forward compatibility currently limits the
    ability to select the implementation and it is hardwired for the different
    data types.

    Timsort is added for better performance on already or nearly
    sorted data. On random data timsort is almost identical to
    mergesort. It is now used for stable sort while quicksort is still the
    default sort if none is chosen. For timsort details, refer to
    `CPython listsort.txt
    <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_
    'mergesort' and 'stable' are mapped to radix sort for integer data types.
    Radix sort is an O(n) sort instead of O(n log n).

    NaT now sorts to the end of arrays for consistency with NaN.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = np.array(values, dtype=dtype)       # create a structured array
    >>> np.sort(a, order='height')                        # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    Sort by age, then height if ages are equal:

    >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    """
    if axis is None:
        # flatten returns (1, N) for np.matrix, so always use the last axis
        a = asanyarray(a).flatten()
        axis = -1
    else:
        a = asanyarray(a).copy(order="K")
    a.sort(axis=axis, kind=kind, order=order, stable=stable)
    return a


def _argsort_dispatcher(a, axis=None, kind=None, order=None, *, stable=None):
    return (a,)


@array_function_dispatch(_argsort_dispatcher)
def argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm specified
    by the `kind` keyword. It returns an array of indices of the same shape as
    `a` that index data along the given axis in sorted order.

    Parameters
    ----------
    a : array_like
        Array to sort.
    axis : int or None, optional
        Axis along which to sort.  The default is -1 (the last axis). If None,
        the flattened array is used.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort under the covers and, in general, the
        actual implementation will vary with data type. The 'mergesort' option
        is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    stable : bool, optional
        Sort stability. If ``True``, the returned array will maintain
        the relative order of ``a`` values which compare as equal.
        If ``False`` or ``None``, this is not guaranteed. Internally,
        this option selects ``kind='stable'``. Default: ``None``.

        .. versionadded:: 2.0.0

    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified `axis`.
        If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the sorted `a`, irrespective of dimensionality.

    See Also
    --------
    sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    ndarray.sort : Inplace sort.
    argpartition : Indirect partial sort.
    take_along_axis : Apply ``index_array`` from argsort
                      to an array as if by calling sort.

    Notes
    -----
    See `sort` for notes on the different sorting algorithms.

    As of NumPy 1.4.0 `argsort` works with real/complex arrays containing
    nan values. The enhanced sort order is documented in `sort`.

    Examples
    --------
    One dimensional array:

    >>> import numpy as np
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    Two-dimensional array:

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    >>> ind = np.argsort(x, axis=0)  # sorts along first axis (down)
    >>> ind
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
    array([[0, 2],
           [2, 3]])

    >>> ind = np.argsort(x, axis=1)  # sorts along last axis (across)
    >>> ind
    array([[0, 1],
           [0, 1]])
    >>> np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
    array([[0, 3],
           [2, 2]])

    Indices of the sorted elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
    >>> ind
    (array([0, 1, 1, 0]), array([0, 0, 1, 1]))
    >>> x[ind]  # same as np.sort(x, axis=None)
    array([0, 2, 2, 3])

    Sorting with keys:

    >>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
    >>> x
    array([(1, 0), (0, 1)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    >>> np.argsort(x, order=('x','y'))
    array([1, 0])

    >>> np.argsort(x, order=('y','x'))
    array([0, 1])

    """
    return _wrapfunc(
        a, 'argsort', axis=axis, kind=kind, order=order, stable=stable
    )

def _argmax_dispatcher(a, axis=None, out=None, *, keepdims=np._NoValue):
    return (a, out)


@array_function_dispatch(_argmax_dispatcher)
def argmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as ``a.shape``
        with the dimension along `axis` removed. If `keepdims` is set to True,
        then the size of `axis` will be 1 with the resulting array having same
        shape as ``a.shape``.

    See Also
    --------
    ndarray.argmax, argmin
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                      from argmax to an array as if by calling max.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmax(a)
    5
    >>> np.argmax(a, axis=0)
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)
    array([2, 2])

    Indexes of the maximal elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    >>> ind
    (1, 2)
    >>> a[ind]
    15

    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    1

    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> index_array = np.argmax(x, axis=-1)
    >>> # Same as np.amax(x, axis=-1, keepdims=True)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    array([[4],
           [3]])
    >>> # Same as np.amax(x, axis=-1)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
    ...     axis=-1).squeeze(axis=-1)
    array([4, 3])

    Setting `keepdims` to `True`,

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmax(x, axis=1, keepdims=True)
    >>> res.shape
    (2, 1, 4)
    """
    kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
    return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)


def _argmin_dispatcher(a, axis=None, out=None, *, keepdims=np._NoValue):
    return (a, out)


@array_function_dispatch(_argmin_dispatcher)
def argmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed. If `keepdims` is set to True,
        then the size of `axis` will be 1 with the resulting array having same
        shape as `a.shape`.

    See Also
    --------
    ndarray.argmin, argmax
    amin : The minimum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                      from argmin to an array as if by calling min.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmin(a)
    0
    >>> np.argmin(a, axis=0)
    array([0, 0, 0])
    >>> np.argmin(a, axis=1)
    array([0, 0])

    Indices of the minimum elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
    >>> ind
    (0, 0)
    >>> a[ind]
    10

    >>> b = np.arange(6) + 10
    >>> b[4] = 10
    >>> b
    array([10, 11, 12, 13, 10, 15])
    >>> np.argmin(b)  # Only the first occurrence is returned.
    0

    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> index_array = np.argmin(x, axis=-1)
    >>> # Same as np.amin(x, axis=-1, keepdims=True)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    array([[2],
           [0]])
    >>> # Same as np.amax(x, axis=-1)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
    ...     axis=-1).squeeze(axis=-1)
    array([2, 0])

    Setting `keepdims` to `True`,

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmin(x, axis=1, keepdims=True)
    >>> res.shape
    (2, 1, 4)
    """
    kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)


def _searchsorted_dispatcher(a, v, side=None, sorter=None):
    return (a, v, sorter)


@array_function_dispatch(_searchsorted_dispatcher)
def searchsorted(a, v, side='left', sorter=None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    a : 1-D array_like
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    indices : int or array of ints
        Array of insertion points with the same shape as `v`,
        or an integer if `v` is a scalar.

    See Also
    --------
    sort : Return a sorted copy of an array.
    histogram : Produce histogram from 1-D data.

    Notes
    -----
    Binary search is used to find the required insertion points.

    As of NumPy 1.4.0 `searchsorted` works with real/complex arrays containing
    `nan` values. The enhanced sort order is documented in `sort`.

    This function uses the same algorithm as the builtin python
    `bisect.bisect_left` (``side='left'``) and `bisect.bisect_right`
    (``side='right'``) functions, which is also vectorized
    in the `v` argument.

    Examples
    --------
    >>> import numpy as np
    >>> np.searchsorted([11,12,13,14,15], 13)
    2
    >>> np.searchsorted([11,12,13,14,15], 13, side='right')
    3
    >>> np.searchsorted([11,12,13,14,15], [-10, 20, 12, 13])
    array([0, 5, 1, 2])

    When `sorter` is used, the returned indices refer to the sorted
    array of `a` and not `a` itself:

    >>> a = np.array([40, 10, 20, 30])
    >>> sorter = np.argsort(a)
    >>> sorter
    array([1, 2, 3, 0])  # Indices that would sort the array 'a'
    >>> result = np.searchsorted(a, 25, sorter=sorter)
    >>> result
    2
    >>> a[sorter[result]]
    30  # The element at index 2 of the sorted array is 30.
    """
    return _wrapfunc(a, 'searchsorted', v, side=side, sorter=sorter)


def _resize_dispatcher(a, new_shape):
    return (a,)


@array_function_dispatch(_resize_dispatcher)
def resize(a, new_shape):
    """
    Return a new array with the specified shape.

    If the new array is larger than the original array, then the new
    array is filled with repeated copies of `a`.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of `a`.

    Parameters
    ----------
    a : array_like
        Array to be resized.

    new_shape : int or tuple of int
        Shape of resized array.

    Returns
    -------
    reshaped_array : ndarray
        The new array is formed from the data in the old array, repeated
        if necessary to fill out the required number of elements.  The
        data are repeated iterating over the array in C-order.

    See Also
    --------
    numpy.reshape : Reshape an array without changing the total size.
    numpy.pad : Enlarge and pad an array.
    numpy.repeat : Repeat elements of an array.
    ndarray.resize : resize an array in-place.

    Notes
    -----
    When the total size of the array does not change `~numpy.reshape` should
    be used.  In most other cases either indexing (to reduce the size)
    or padding (to increase the size) may be a more appropriate solution.

    Warning: This functionality does **not** consider axes separately,
    i.e. it does not apply interpolation/extrapolation.
    It fills the return array with the required number of elements, iterating
    over `a` in C-order, disregarding axes (and cycling back from the start if
    the new shape is larger).  This functionality is therefore not suitable to
    resize images, or data where each axis represents a separate and distinct
    entity.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[0,1],[2,3]])
    >>> np.resize(a,(2,3))
    array([[0, 1, 2],
           [3, 0, 1]])
    >>> np.resize(a,(1,4))
    array([[0, 1, 2, 3]])
    >>> np.resize(a,(2,4))
    array([[0, 1, 2, 3],
           [0, 1, 2, 3]])

    """
    if isinstance(new_shape, (int, nt.integer)):
        new_shape = (new_shape,)

    a = ravel(a)

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError(
                'all elements of `new_shape` must be non-negative'
            )

    if a.size == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return np.zeros_like(a, shape=new_shape)

    repeats = -(-new_size // a.size)  # ceil division
    a = concatenate((a,) * repeats)[:new_size]

    return reshape(a, new_shape)


def _squeeze_dispatcher(a, axis=None):
    return (a,)


@array_function_dispatch(_squeeze_dispatcher)
def squeeze(a, axis=None):
    """
    Remove axes of length one from `a`.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the entries of length one in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`. Note that if all axes are squeezed,
        the result is a 0d array and not a scalar.

    Raises
    ------
    ValueError
        If `axis` is not None, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding entries of length one
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size
    not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    >>> x = np.array([[1234]])
    >>> x.shape
    (1, 1)
    >>> np.squeeze(x)
    array(1234)  # 0d array
    >>> np.squeeze(x).shape
    ()
    >>> np.squeeze(x)[()]
    1234

    """
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze', axis=axis)
    if axis is None:
        return squeeze()
    else:
        return squeeze(axis=axis)


def _diagonal_dispatcher(a, offset=None, axis1=None, axis2=None):
    return (a,)


@array_function_dispatch(_diagonal_dispatcher)
def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.  If
    `a` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned.  The shape of the resulting array can be determined by
    removing `axis1` and `axis2` and appending an index to the right equal
    to the size of the resulting diagonals.

    In versions of NumPy prior to 1.7, this function always returned a new,
    independent array containing a copy of the values in the diagonal.

    In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal,
    but depending on this fact is deprecated. Writing to the resulting
    array continues to work as it used to, but a FutureWarning is issued.

    Starting in NumPy 1.9 it returns a read-only view on the original array.
    Attempting to write to the resulting array will produce an error.

    In some future release, it will return a read/write view and writing to
    the returned array will alter your original array.  The returned array
    will have the same type as the input array.

    If you don't write to the array returned by this function, then you can
    just ignore all of the above.

    If you depend on the current behavior, then we suggest copying the
    returned array explicitly, i.e., use ``np.diagonal(a).copy()`` instead
    of just ``np.diagonal(a)``. This will work with both past and future
    versions of NumPy.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from which
        the diagonals should be taken.  Defaults to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, then a 1-D array containing the diagonal and of the
        same type as `a` is returned unless `a` is a `matrix`, in which case
        a 1-D array rather than a (2-D) `matrix` is returned in order to
        maintain backward compatibility.

        If ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2`
        are removed, and a new axis inserted at the end corresponding to the
        diagonal.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    diag : MATLAB work-a-like for 1-D and 2-D arrays.
    diagflat : Create diagonal arrays.
    trace : Sum along diagonals.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = np.arange(8).reshape(2,2,2); a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> a.diagonal(0,  # Main diagonals of two arrays created by skipping
    ...            0,  # across the outer(left)-most axis last and
    ...            1)  # the "middle" (row) axis first.
    array([[0, 6],
           [1, 7]])

    The sub-arrays whose main diagonals we just obtained; note that each
    corresponds to fixing the right-most (column) axis, and that the
    diagonals are "packed" in rows.

    >>> a[:,:,0]  # main diagonal is [0 6]
    array([[0, 2],
           [4, 6]])
    >>> a[:,:,1]  # main diagonal is [1 7]
    array([[1, 3],
           [5, 7]])

    The anti-diagonal can be obtained by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.fliplr(a).diagonal()  # Horizontal flip
    array([2, 4, 6])
    >>> np.flipud(a).diagonal()  # Vertical flip
    array([6, 4, 2])

    Note that the order in which the diagonal is retrieved varies depending
    on the flip function.
    """
    if isinstance(a, np.matrix):
        # Make diagonal of matrix 1-D to preserve backward compatibility.
        return asarray(a).diagonal(offset=offset, axis1=axis1, axis2=axis2)
    else:
        return asanyarray(a).diagonal(offset=offset, axis1=axis1, axis2=axis2)


def _trace_dispatcher(
        a, offset=None, axis1=None, axis2=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_trace_dispatcher)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    Return the sum along diagonals of the array.

    If `a` is 2-D, the sum along its diagonal with the given offset
    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1`
    and `axis2` removed.

    Parameters
    ----------
    a : array_like
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the first two
        axes of `a`.
    dtype : dtype, optional
        Determines the data-type of the returned array and of the accumulator
        where the elements are summed. If dtype has the value None and `a` is
        of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise,
        the precision is the same as that of `a`.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
        larger dimensions, then an array of sums along diagonals is returned.

    See Also
    --------
    diag, diagonal, diagflat

    Examples
    --------
    >>> import numpy as np
    >>> np.trace(np.eye(3))
    3.0
    >>> a = np.arange(8).reshape((2,2,2))
    >>> np.trace(a)
    array([6, 8])

    >>> a = np.arange(24).reshape((2,2,2,3))
    >>> np.trace(a).shape
    (2, 3)

    """
    if isinstance(a, np.matrix):
        # Get trace of matrix via an array to preserve backward compatibility.
        return asarray(a).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )
    else:
        return asanyarray(a).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )


def _ravel_dispatcher(a, order=None):
    return (a,)


@array_function_dispatch(_ravel_dispatcher)
def ravel(a, order='C'):
    """Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    As of NumPy 1.10, the returned array will have the same type as the input
    array. (for example, a masked array will be returned for a masked array
    input)

    Parameters
    ----------
    a : array_like
        Input array.  The elements in `a` are read in the order specified by
        `order`, and packed as a 1-D array.
    order : {'C','F', 'A', 'K'}, optional

        The elements of `a` are read using this index order. 'C' means
        to index the elements in row-major, C-style order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest.  'F' means to index the elements
        in column-major, Fortran-style order, with the
        first index changing fastest, and the last index changing
        slowest. Note that the 'C' and 'F' options take no account of
        the memory layout of the underlying array, and only refer to
        the order of axis indexing.  'A' means to read the elements in
        Fortran-like index order if `a` is Fortran *contiguous* in
        memory, C-like order otherwise.  'K' means to read the
        elements in the order they occur in memory, except for
        reversing the data when strides are negative.  By default, 'C'
        index order is used.

    Returns
    -------
    y : array_like
        y is a contiguous 1-D array of the same subtype as `a`,
        with shape ``(a.size,)``.
        Note that matrices are special cased for backward compatibility,
        if `a` is a matrix, then y is a 1-D ndarray.

    See Also
    --------
    ndarray.flat : 1-D iterator over an array.
    ndarray.flatten : 1-D array copy of the elements of an array
                      in row-major order.
    ndarray.reshape : Change the shape of an array without changing its data.

    Notes
    -----
    In row-major, C-style order, in two dimensions, the row index
    varies the slowest, and the column index the quickest.  This can
    be generalized to multiple dimensions, where row-major order
    implies that the index along the first axis varies slowest, and
    the index along the last quickest.  The opposite holds for
    column-major, Fortran-style index ordering.

    When a view is desired in as many cases as possible, ``arr.reshape(-1)``
    may be preferable. However, ``ravel`` supports ``K`` in the optional
    ``order`` argument while ``reshape`` does not.

    Examples
    --------
    It is equivalent to ``reshape(-1, order=order)``.

    >>> import numpy as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ravel(x)
    array([1, 2, 3, 4, 5, 6])

    >>> x.reshape(-1)
    array([1, 2, 3, 4, 5, 6])

    >>> np.ravel(x, order='F')
    array([1, 4, 2, 5, 3, 6])

    When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:

    >>> np.ravel(x.T)
    array([1, 4, 2, 5, 3, 6])
    >>> np.ravel(x.T, order='A')
    array([1, 2, 3, 4, 5, 6])

    When ``order`` is 'K', it will preserve orderings that are neither 'C'
    nor 'F', but won't reverse axes:

    >>> a = np.arange(3)[::-1]; a
    array([2, 1, 0])
    >>> a.ravel(order='C')
    array([2, 1, 0])
    >>> a.ravel(order='K')
    array([2, 1, 0])

    >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
    array([[[ 0,  2,  4],
            [ 1,  3,  5]],
           [[ 6,  8, 10],
            [ 7,  9, 11]]])
    >>> a.ravel(order='C')
    array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
    >>> a.ravel(order='K')
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    """
    if isinstance(a, np.matrix):
        return asarray(a).ravel(order=order)
    else:
        return asanyarray(a).ravel(order=order)


def _nonzero_dispatcher(a):
    return (a,)


@array_function_dispatch(_nonzero_dispatcher)
def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned in
    row-major, C-style order.

    To group the indices by element, rather than dimension, use `argwhere`,
    which returns a row for each non-zero element.

    .. note::

       When called on a zero-d array or scalar, ``nonzero(a)`` is treated
       as ``nonzero(atleast_1d(a))``.

       .. deprecated:: 1.17.0

          Use `atleast_1d` explicitly if this behavior is deliberate.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        array.
    ndarray.nonzero :
        Equivalent ndarray method.
    count_nonzero :
        Counts the number of non-zero elements in the input array.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.transpose(np.nonzero(x))
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]])

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """
    return _wrapfunc(a, 'nonzero')


def _shape_dispatcher(a):
    return (a,)


@array_function_dispatch(_shape_dispatcher)
def shape(a):
    """
    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
          ``N>=1``.
    ndarray.shape : Equivalent array method.

    Examples
    --------
    >>> import numpy as np
    >>> np.shape(np.eye(3))
    (3, 3)
    >>> np.shape([[1, 3]])
    (1, 2)
    >>> np.shape([0])
    (1,)
    >>> np.shape(0)
    ()

    >>> a = np.array([(1, 2), (3, 4), (5, 6)],
    ...              dtype=[('x', 'i4'), ('y', 'i4')])
    >>> np.shape(a)
    (3,)
    >>> a.shape
    (3,)

    """
    try:
        result = a.shape
    except AttributeError:
        result = asarray(a).shape
    return result


def _compress_dispatcher(condition, a, axis=None, out=None):
    return (condition, a, out)


@array_function_dispatch(_compress_dispatcher)
def compress(condition, a, axis=None, out=None):
    """
    Return selected slices of an array along given axis.

    When working along a given axis, a slice along that axis is returned in
    `output` for each index where `condition` evaluates to True. When
    working on a 1-D array, `compress` is equivalent to `extract`.

    Parameters
    ----------
    condition : 1-D array of bools
        Array that selects which entries to return. If len(condition)
        is less than the size of `a` along the given axis, then output is
        truncated to the length of the condition array.
    a : array_like
        Array from which to extract a part.
    axis : int, optional
        Axis along which to take slices. If None (default), work on the
        flattened array.
    out : ndarray, optional
        Output array.  Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : ndarray
        A copy of `a` without the slices along axis for which `condition`
        is false.

    See Also
    --------
    take, choose, diag, diagonal, select
    ndarray.compress : Equivalent method in ndarray
    extract : Equivalent method when working on 1-D arrays
    :ref:`ufuncs-output-type`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4], [5, 6]])
    >>> a
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.compress([0, 1], a, axis=0)
    array([[3, 4]])
    >>> np.compress([False, True, True], a, axis=0)
    array([[3, 4],
           [5, 6]])
    >>> np.compress([False, True], a, axis=1)
    array([[2],
           [4],
           [6]])

    Working on the flattened array does not return slices along an axis but
    selects elements.

    >>> np.compress([False, True], a)
    array([2])

    """
    return _wrapfunc(a, 'compress', condition, axis=axis, out=out)


def _clip_dispatcher(a, a_min=None, a_max=None, out=None, *, min=None,
                     max=None, **kwargs):
    return (a, a_min, a_max, out, min, max)


@array_function_dispatch(_clip_dispatcher)
def clip(a, a_min=np._NoValue, a_max=np._NoValue, out=None, *,
         min=np._NoValue, max=np._NoValue, **kwargs):
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Equivalent to but faster than ``np.minimum(a_max, np.maximum(a, a_min))``.

    No check is performed to ensure ``a_min < a_max``.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min, a_max : array_like or None
        Minimum and maximum value. If ``None``, clipping is not performed on
        the corresponding edge. If both ``a_min`` and ``a_max`` are ``None``,
        the elements of the returned array stay the same. Both are broadcasted
        against ``a``.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.
    min, max : array_like or None
        Array API compatible alternatives for ``a_min`` and ``a_max``
        arguments. Either ``a_min`` and ``a_max`` or ``min`` and ``max``
        can be passed at the same time. Default: ``None``.

        .. versionadded:: 2.1.0
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    See Also
    --------
    :ref:`ufuncs-output-type`

    Notes
    -----
    When `a_min` is greater than `a_max`, `clip` returns an
    array in which all values are equal to `a_max`,
    as shown in the second example.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> np.clip(a, 8, 1)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    if a_min is np._NoValue and a_max is np._NoValue:
        a_min = None if min is np._NoValue else min
        a_max = None if max is np._NoValue else max
    elif a_min is np._NoValue:
        raise TypeError("clip() missing 1 required positional "
                        "argument: 'a_min'")
    elif a_max is np._NoValue:
        raise TypeError("clip() missing 1 required positional "
                        "argument: 'a_max'")
    elif min is not np._NoValue or max is not np._NoValue:
        raise ValueError("Passing `min` or `max` keyword argument when "
                         "`a_min` and `a_max` are provided is forbidden.")

    return _wrapfunc(a, 'clip', a_min, a_max, out=out, **kwargs)


def _sum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                    initial=None, where=None):
    return (a, out)


@array_function_dispatch(_sum_dispatcher)
def sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
        initial=np._NoValue, where=np._NoValue):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis. If
        axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.
    add: ``numpy.add.reduce`` equivalent function.
    cumsum : Cumulative sum of array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    The sum of an empty array is the neutral element 0:

    >>> np.sum([])
    0.0

    For floating point numbers the numerical precision of sum (and
    ``np.add.reduce``) is in general limited by directly adding each number
    individually to the result causing rounding errors in every step.
    However, often numpy will use a  numerically better approach (partial
    pairwise summation) leading to improved precision in many use-cases.
    This improved precision is always provided when no ``axis`` is given.
    When ``axis`` is given, it will depend on which axis is summed.
    Technically, to provide the best speed possible, the improved precision
    is only used when the summation is along the fast axis in memory.
    Note that the exact precision may vary depending on other parameters.
    In contrast to NumPy, Python's ``math.fsum`` function uses a slower but
    more precise approach to summation.
    Especially when summing a large number of lower precision floating point
    numbers, such as ``float32``, numerical errors can become significant.
    In such cases it can be advisable to use `dtype="float64"` to use a higher
    precision for the output.

    Examples
    --------
    >>> import numpy as np
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    np.int32(1)
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    >>> np.sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1)
    array([1., 5.])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    np.int8(-128)

    You can also start the sum with a value other than zero:

    >>> np.sum([10], initial=5)
    15
    """
    if isinstance(a, _gentype):
        # 2018-02-25, 1.15.0
        warnings.warn(
            "Calling np.sum(generator) is deprecated, and in the future will "
            "give a different result. Use np.sum(np.fromiter(generator)) or "
            "the python sum builtin instead.",
            DeprecationWarning, stacklevel=2
        )

        res = _sum_(a)
        if out is not None:
            out[...] = res
            return out
        return res

    return _wrapreduction(
        a, np.add, 'sum', axis, dtype, out,
        keepdims=keepdims, initial=initial, where=where
    )


def _any_dispatcher(a, axis=None, out=None, keepdims=None, *,
                    where=np._NoValue):
    return (a, where, out)


@array_function_dispatch(_any_dispatcher)
def any(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
    """
    Test whether any array element along a given axis evaluates to True.

    Returns single boolean if `axis` is ``None``

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis. If this
        is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in checking for any `True` values.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    any : bool or ndarray
        A new boolean or `ndarray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    ndarray.any : equivalent method

    all : Test whether all elements along a given axis evaluate to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to `True` because these are not equal to zero.

    .. versionchanged:: 2.0
       Before NumPy 2.0, ``any`` did not return booleans for object dtype
       input arrays.
       This behavior is still available via ``np.logical_or.reduce``.

    Examples
    --------
    >>> import numpy as np
    >>> np.any([[True, False], [True, True]])
    True

    >>> np.any([[True,  False, True ],
    ...         [False, False, False]], axis=0)
    array([ True, False, True])

    >>> np.any([-1, 0, 5])
    True

    >>> np.any([[np.nan], [np.inf]], axis=1, keepdims=True)
    array([[ True],
           [ True]])

    >>> np.any([[True, False], [False, False]], where=[[False], [True]])
    False

    >>> a = np.array([[1, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 0, 0]])
    >>> np.any(a, axis=0)
    array([ True, False,  True])
    >>> np.any(a, axis=1)
    array([ True,  True, False])

    >>> o=np.array(False)
    >>> z=np.any([-1, 4, 5], out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that z is a reference to o
    >>> z is o
    True
    >>> id(z), id(o) # identity of z and o              # doctest: +SKIP
    (191614240, 191614240)

    """
    return _wrapreduction_any_all(a, np.logical_or, 'any', axis, out,
                                  keepdims=keepdims, where=where)


def _all_dispatcher(a, axis=None, out=None, keepdims=None, *,
                    where=None):
    return (a, where, out)


@array_function_dispatch(_all_dispatcher)
def all(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis. If this
        is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's). See :ref:`ufuncs-output-type`
        for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in checking for all `True` values.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    all : ndarray, bool
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    ndarray.all : equivalent method

    any : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    .. versionchanged:: 2.0
       Before NumPy 2.0, ``all`` did not return booleans for object dtype
       input arrays.
       This behavior is still available via ``np.logical_and.reduce``.

    Examples
    --------
    >>> import numpy as np
    >>> np.all([[True,False],[True,True]])
    False

    >>> np.all([[True,False],[True,True]], axis=0)
    array([ True, False])

    >>> np.all([-1, 4, 5])
    True

    >>> np.all([1.0, np.nan])
    True

    >>> np.all([[True, True], [False, True]], where=[[True], [False]])
    True

    >>> o=np.array(False)
    >>> z=np.all([-1, 4, 5], out=o)
    >>> id(z), id(o), z
    (28293632, 28293632, array(True)) # may vary

    """
    return _wrapreduction_any_all(a, np.logical_and, 'all', axis, out,
                                  keepdims=keepdims, where=where)


def _cumulative_func(x, func, axis, dtype, out, include_initial):
    x = np.atleast_1d(x)
    x_ndim = x.ndim
    if axis is None:
        if x_ndim >= 2:
            raise ValueError("For arrays which have more than one dimension "
                            "``axis`` argument is required.")
        axis = 0

    if out is not None and include_initial:
        item = [slice(None)] * x_ndim
        item[axis] = slice(1, None)
        func.accumulate(x, axis=axis, dtype=dtype, out=out[tuple(item)])
        item[axis] = 0
        out[tuple(item)] = func.identity
        return out

    res = func.accumulate(x, axis=axis, dtype=dtype, out=out)
    if include_initial:
        initial_shape = list(x.shape)
        initial_shape[axis] = 1
        res = np.concat(
            [np.full_like(res, func.identity, shape=initial_shape), res],
            axis=axis,
        )

    return res


def _cumulative_prod_dispatcher(x, /, *, axis=None, dtype=None, out=None,
                                include_initial=None):
    return (x, out)


@array_function_dispatch(_cumulative_prod_dispatcher)
def cumulative_prod(x, /, *, axis=None, dtype=None, out=None,
                    include_initial=False):
    """
    Return the cumulative product of elements along a given axis.

    This function is an Array API compatible alternative to `numpy.cumprod`.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. The default
        (None) is only allowed for one-dimensional arrays. For arrays
        with more than one dimension ``axis`` is required.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If ``dtype`` is not specified, it
        defaults to the dtype of ``x``, unless ``x`` has an integer dtype
        with a precision less than that of the default platform integer.
        In that case, the default platform integer is used instead.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    include_initial : bool, optional
        Boolean indicating whether to include the initial value (ones) as
        the first value in the output. With ``include_initial=True``
        the shape of the output is different than the shape of the input.
        Default: ``False``.

    Returns
    -------
    cumulative_prod_along_axis : ndarray
        A new array holding the result is returned unless ``out`` is
        specified, in which case a reference to ``out`` is returned. The
        result has the same shape as ``x`` if ``include_initial=False``.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> np.cumulative_prod(a)  # intermediate results 1, 1*2
    ...                        # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = np.array([1, 2, 3, 4, 5, 6])
    >>> np.cumulative_prod(a, dtype=float) # specify type of output
    array([   1.,    2.,    6.,   24.,  120.,  720.])

    The cumulative product for each column (i.e., over the rows) of ``b``:

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumulative_prod(b, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of ``b``:

    >>> np.cumulative_prod(b, axis=1)
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """
    return _cumulative_func(x, um.multiply, axis, dtype, out, include_initial)


def _cumulative_sum_dispatcher(x, /, *, axis=None, dtype=None, out=None,
                               include_initial=None):
    return (x, out)


@array_function_dispatch(_cumulative_sum_dispatcher)
def cumulative_sum(x, /, *, axis=None, dtype=None, out=None,
                   include_initial=False):
    """
    Return the cumulative sum of the elements along a given axis.

    This function is an Array API compatible alternative to `numpy.cumsum`.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is only allowed for one-dimensional arrays. For arrays
        with more than one dimension ``axis`` is required.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If ``dtype`` is not specified, it defaults
        to the dtype of ``x``, unless ``x`` has an integer dtype with
        a precision less than that of the default platform integer.
        In that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.
    include_initial : bool, optional
        Boolean indicating whether to include the initial value (zeros) as
        the first value in the output. With ``include_initial=True``
        the shape of the output is different than the shape of the input.
        Default: ``False``.

    Returns
    -------
    cumulative_sum_along_axis : ndarray
        A new array holding the result is returned unless ``out`` is
        specified, in which case a reference to ``out`` is returned. The
        result has the same shape as ``x`` if ``include_initial=False``.

    See Also
    --------
    sum : Sum array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.
    diff : Calculate the n-th discrete difference along given axis.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    ``cumulative_sum(a)[-1]`` may not be equal to ``sum(a)`` for
    floating-point values since ``sum`` may use a pairwise summation routine,
    reducing the roundoff-error. See `sum` for more information.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5, 6])
    >>> a
    array([1, 2, 3, 4, 5, 6])
    >>> np.cumulative_sum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumulative_sum(a, dtype=float)  # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumulative_sum(b,axis=0)  # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumulative_sum(b,axis=1)  # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    ``cumulative_sum(c)[-1]`` may not be equal to ``sum(c)``

    >>> c = np.array([1, 2e-9, 3e-9] * 1000000)
    >>> np.cumulative_sum(c)[-1]
    1000000.0050045159
    >>> c.sum()
    1000000.0050000029

    """
    return _cumulative_func(x, um.add, axis, dtype, out, include_initial)


def _cumsum_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_cumsum_dispatcher)
def cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

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
        but the type will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.

    See Also
    --------
    cumulative_sum : Array API compatible alternative for ``cumsum``.
    sum : Sum array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.
    diff : Calculate the n-th discrete difference along given axis.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    ``cumsum(a)[-1]`` may not be equal to ``sum(a)`` for floating-point
    values since ``sum`` may use a pairwise summation routine, reducing
    the roundoff-error. See `sum` for more information.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])

    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    ``cumsum(b)[-1]`` may not be equal to ``sum(b)``

    >>> b = np.array([1, 2e-9, 3e-9] * 1000000)
    >>> b.cumsum()[-1]
    1000000.0050045159
    >>> b.sum()
    1000000.0050000029

    """
    return _wrapfunc(a, 'cumsum', axis=axis, dtype=dtype, out=out)


def _ptp_dispatcher(a, axis=None, out=None, keepdims=None):
    return (a, out)


@array_function_dispatch(_ptp_dispatcher)
def ptp(a, axis=None, out=None, keepdims=np._NoValue):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for 'peak to peak'.

    .. warning::
        `ptp` preserves the data type of the array. This means the
        return value for an input of signed integers with n bits
        (e.g. `numpy.int8`, `numpy.int16`, etc) is also a signed integer
        with n bits.  In that case, peak-to-peak values greater than
        ``2**(n-1)-1`` will be returned as negative values. An example
        with a work-around is shown below.

    Parameters
    ----------
    a : array_like
        Input values.
    axis : None or int or tuple of ints, optional
        Axis along which to find the peaks.  By default, flatten the
        array.  `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : array_like
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type of the output values will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `ptp` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    ptp : ndarray or scalar
        The range of a given array - `scalar` if array is one-dimensional
        or a new array holding the result along the given axis

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])

    >>> np.ptp(x, axis=1)
    array([8, 6])

    >>> np.ptp(x, axis=0)
    array([2, 0, 5, 2])

    >>> np.ptp(x)
    10

    This example shows that a negative value can be returned when
    the input is an array of signed integers.

    >>> y = np.array([[1, 127],
    ...               [0, 127],
    ...               [-1, 127],
    ...               [-2, 127]], dtype=np.int8)
    >>> np.ptp(y, axis=1)
    array([ 126,  127, -128, -127], dtype=int8)

    A work-around is to use the `view()` method to view the result as
    unsigned integers with the same bit width:

    >>> np.ptp(y, axis=1).view(np.uint8)
    array([126, 127, 128, 129], dtype=uint8)

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    return _methods._ptp(a, axis=axis, out=out, **kwargs)


def _max_dispatcher(a, axis=None, out=None, keepdims=None, initial=None,
                    where=None):
    return (a, out)


@array_function_dispatch(_max_dispatcher)
@set_module('numpy')
def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used. If this is a tuple of ints, the maximum is selected over
        multiple axes, instead of a single axis or all the axes as before.

    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``max`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    max : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    See Also
    --------
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `~numpy.max` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``max(a, axis=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.max(a)           # Maximum of the flattened array
    3
    >>> np.max(a, axis=0)   # Maxima along the first axis
    array([2, 3])
    >>> np.max(a, axis=1)   # Maxima along the second axis
    array([1, 3])
    >>> np.max(a, where=[False, True], initial=-1, axis=0)
    array([-1,  3])
    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.max(b)
    np.float64(nan)
    >>> np.max(b, where=~np.isnan(b), initial=-1)
    4.0
    >>> np.nanmax(b)
    4.0

    You can use an initial value to compute the maximum of an empty slice, or
    to initialize it to a different value:

    >>> np.max([[-50], [10]], axis=-1, initial=0)
    array([ 0, 10])

    Notice that the initial value is used as one of the elements for which the
    maximum is determined, unlike for the default argument Python's max
    function, which is only used for empty iterables.

    >>> np.max([5], initial=6)
    6
    >>> max([5], default=6)
    5
    """
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


@array_function_dispatch(_max_dispatcher)
def amax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    `amax` is an alias of `~numpy.max`.

    See Also
    --------
    max : alias of this function
    ndarray.max : equivalent method
    """
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


def _min_dispatcher(a, axis=None, out=None, keepdims=None, initial=None,
                    where=None):
    return (a, out)


@array_function_dispatch(_min_dispatcher)
def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
        where=np._NoValue):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``min`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    min : ndarray or scalar
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``.  If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    See Also
    --------
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmin :
        Return the indices of the minimum values.

    nanmax, maximum, fmax

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding min value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmin.

    Don't use `~numpy.min` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``min(a, axis=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.min(a)           # Minimum of the flattened array
    0
    >>> np.min(a, axis=0)   # Minima along the first axis
    array([0, 1])
    >>> np.min(a, axis=1)   # Minima along the second axis
    array([0, 2])
    >>> np.min(a, where=[False, True], initial=10, axis=0)
    array([10,  1])

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.min(b)
    np.float64(nan)
    >>> np.min(b, where=~np.isnan(b), initial=10)
    0.0
    >>> np.nanmin(b)
    0.0

    >>> np.min([[-50], [10]], axis=-1, initial=0)
    array([-50,   0])

    Notice that the initial value is used as one of the elements for which the
    minimum is determined, unlike for the default argument Python's max
    function, which is only used for empty iterables.

    Notice that this isn't the same as Python's ``default`` argument.

    >>> np.min([6], initial=5)
    5
    >>> min([6], default=5)
    6
    """
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


@array_function_dispatch(_min_dispatcher)
def amin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the minimum of an array or minimum along an axis.

    `amin` is an alias of `~numpy.min`.

    See Also
    --------
    min : alias of this function
    ndarray.min : equivalent method
    """
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


def _prod_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                     initial=None, where=None):
    return (a, out)


@array_function_dispatch(_prod_dispatcher)
def prod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
         initial=np._NoValue, where=np._NoValue):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce`
        for details.
    where : array_like of bool, optional
        Elements to include in the product. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    ndarray.prod : equivalent method
    :ref:`ufuncs-output-type`

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> x = np.array([536870910, 536870910, 536870910, 536870910])
    >>> np.prod(x)
    16 # may vary

    The product of an empty array is the neutral element 1:

    >>> np.prod([])
    1.0

    Examples
    --------
    By default, calculate the product of all elements:

    >>> import numpy as np
    >>> np.prod([1.,2.])
    2.0

    Even when the input array is two-dimensional:

    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> np.prod(a)
    24.0

    But we can also specify the axis over which to multiply:

    >>> np.prod(a, axis=1)
    array([  2.,  12.])
    >>> np.prod(a, axis=0)
    array([3., 8.])

    Or select specific elements to include:

    >>> np.prod([1., np.nan, 3.], where=[True, False, True])
    3.0

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == int
    True

    You can also start the product with a value other than one:

    >>> np.prod([1, 2], initial=5)
    10
    """
    return _wrapreduction(a, np.multiply, 'prod', axis, dtype, out,
                          keepdims=keepdims, initial=initial, where=where)


def _cumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_cumprod_dispatcher)
def cumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of elements along a given axis.

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
    cumprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to out is returned.

    See Also
    --------
    cumulative_prod : Array API compatible alternative for ``cumprod``.
    :ref:`ufuncs-output-type`

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> np.cumprod(a) # intermediate results 1, 1*2
    ...               # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumprod(a, dtype=float) # specify type of output
    array([   1.,    2.,    6.,   24.,  120.,  720.])

    The cumulative product for each column (i.e., over the rows) of `a`:

    >>> np.cumprod(a, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of `a`:

    >>> np.cumprod(a,axis=1)
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """
    return _wrapfunc(a, 'cumprod', axis=axis, dtype=dtype, out=out)


def _ndim_dispatcher(a):
    return (a,)


@array_function_dispatch(_ndim_dispatcher)
def ndim(a):
    """
    Return the number of dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Examples
    --------
    >>> import numpy as np
    >>> np.ndim([[1,2,3],[4,5,6]])
    2
    >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
    2
    >>> np.ndim(1)
    0

    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim


def _size_dispatcher(a, axis=None):
    return (a,)


@array_function_dispatch(_size_dispatcher)
def size(a, axis=None):
    """
    Return the number of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    shape : dimensions of array
    ndarray.shape : dimensions of array
    ndarray.size : number of elements in array

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> np.size(a)
    6
    >>> np.size(a,1)
    3
    >>> np.size(a,0)
    2

    """
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return asarray(a).size
    else:
        try:
            return a.shape[axis]
        except AttributeError:
            return asarray(a).shape[axis]


def _round_dispatcher(a, decimals=None, out=None):
    return (a, out)


@array_function_dispatch(_round_dispatcher)
def round(a, decimals=0, out=None):
    """
    Evenly round to the given number of decimals.

    Parameters
    ----------
    a : array_like
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as `a`, containing the rounded values.
        Unless `out` was specified, a new array is created.  A reference to
        the result is returned.

        The real and imaginary parts of complex numbers are rounded
        separately.  The result of rounding a float is a float.

    See Also
    --------
    ndarray.round : equivalent method
    around : an alias for this function
    ceil, fix, floor, rint, trunc


    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc.

    ``np.round`` uses a fast but sometimes inexact algorithm to round
    floating-point datatypes. For positive `decimals` it is equivalent to
    ``np.true_divide(np.rint(a * 10**decimals), 10**decimals)``, which has
    error due to the inexact representation of decimal fractions in the IEEE
    floating point standard [1]_ and errors introduced when scaling by powers
    of ten. For instance, note the extra "1" in the following:

        >>> np.round(56294995342131.5, 3)
        56294995342131.51

    If your goal is to print such values with a fixed number of decimals, it is
    preferable to use numpy's float printing routines to limit the number of
    printed decimals:

        >>> np.format_float_positional(56294995342131.5, precision=3)
        '56294995342131.5'

    The float printing routines use an accurate but much more computationally
    demanding algorithm to compute the number of digits after the decimal
    point.

    Alternatively, Python's builtin `round` function uses a more accurate
    but slower algorithm for 64-bit floating point values:

        >>> round(56294995342131.5, 3)
        56294995342131.5
        >>> np.round(16.055, 2), round(16.055, 2)  # equals 16.0549999999999997
        (16.06, 16.05)


    References
    ----------
    .. [1] "Lecture Notes on the Status of IEEE 754", William Kahan,
           https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF

    Examples
    --------
    >>> import numpy as np
    >>> np.round([0.37, 1.64])
    array([0., 2.])
    >>> np.round([0.37, 1.64], decimals=1)
    array([0.4, 1.6])
    >>> np.round([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
    array([0., 2., 2., 4., 4.])
    >>> np.round([1,2,3,11], decimals=1) # ndarray of ints is returned
    array([ 1,  2,  3, 11])
    >>> np.round([1,2,3,11], decimals=-1)
    array([ 0,  0,  0, 10])

    """
    return _wrapfunc(a, 'round', decimals=decimals, out=out)


@array_function_dispatch(_round_dispatcher)
def around(a, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    `around` is an alias of `~numpy.round`.

    See Also
    --------
    ndarray.round : equivalent method
    round : alias for this function
    ceil, fix, floor, rint, trunc

    """
    return _wrapfunc(a, 'round', decimals=decimals, out=out)


def _mean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None, *,
                     where=None):
    return (a, where, out)


@array_function_dispatch(_mean_dispatcher)
def mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue, *,
         where=np._NoValue):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.

    Note that for floating-point input, the mean is computed using the
    same precision the input has.  Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32` (see
    example below).  Specifying a higher-precision accumulator using the
    `dtype` keyword can alleviate this issue.

    By default, `float16` results are computed using `float32` intermediates
    for extra precision.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    2.5
    >>> np.mean(a, axis=0)
    array([2., 3.])
    >>> np.mean(a, axis=1)
    array([1.5, 3.5])

    In single precision, `mean` can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.mean(a)
    np.float32(0.54999924)

    Computing the mean in float64 is more accurate:

    >>> np.mean(a, dtype=np.float64)
    0.55000000074505806 # may vary

    Computing the mean in timedelta64 is available:

    >>> b = np.array([1, 3], dtype="timedelta64[D]")
    >>> np.mean(b)
    np.timedelta64(2,'D')

    Specifying a where argument:

    >>> a = np.array([[5, 9, 13], [14, 10, 12], [11, 15, 19]])
    >>> np.mean(a)
    12.0
    >>> np.mean(a, where=[[True], [False], [False]])
    9.0

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if type(a) is not mu.ndarray:
        try:
            mean = a.mean
        except AttributeError:
            pass
        else:
            return mean(axis=axis, dtype=dtype, out=out, **kwargs)

    return _methods._mean(a, axis=axis, dtype=dtype,
                          out=out, **kwargs)


def _std_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                    keepdims=None, *, where=None, mean=None, correction=None):
    return (a, where, out, mean)


@array_function_dispatch(_std_dispatcher)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, *,
        where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    r"""
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero. See Notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

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
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

    See Also
    --------
    var, mean, nanmean, nanstd, nanvar
    :ref:`ufuncs-output-type`

    Notes
    -----
    There are several common variants of the array standard deviation
    calculation. Assuming the input `a` is a one-dimensional NumPy array
    and ``mean`` is either provided as an argument or computed as
    ``a.mean()``, NumPy computes the standard deviation of an array as::

        N = len(a)
        d2 = abs(a - mean)**2  # abs is for complex `a`
        var = d2.sum() / (N - ddof)  # note use of `ddof`
        std = var**0.5

    Different values of the argument `ddof` are useful in different
    contexts. NumPy's default ``ddof=0`` corresponds with the expression:

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N}}

    which is sometimes called the "population standard deviation" in the field
    of statistics because it applies the definition of standard deviation to
    `a` as if `a` were a complete population of possible observations.

    Many other libraries define the standard deviation of an array
    differently, e.g.:

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N - 1}}

    In statistics, the resulting quantity is sometimes called the "sample
    standard deviation" because if `a` is a random sample from a larger
    population, this calculation provides the square root of an unbiased
    estimate of the variance of the population. The use of :math:`N-1` in the
    denominator is often called "Bessel's correction" because it corrects for
    bias (toward lower values) in the variance estimate introduced when the
    sample mean of `a` is used in place of the true mean of the population.
    The resulting estimate of the standard deviation is still biased, but less
    than it would have been without the correction. For this quantity, use
    ``ddof=1``.

    Note that, for complex numbers, `std` takes the absolute
    value before squaring, so that the result is always real and nonnegative.

    For floating-point input, the standard deviation is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example below).
    Specifying a higher-accuracy accumulator using the `dtype` keyword can
    alleviate this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949 # may vary
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])

    In single precision, std() can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.std(a)
    np.float32(0.45000005)

    Computing the standard deviation in float64 is more accurate:

    >>> np.std(a, dtype=np.float64)
    0.44999999925494177 # may vary

    Specifying a where argument:

    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> np.std(a)
    2.614064523559687 # may vary
    >>> np.std(a, where=[[True], [True], [False]])
    2.0

    Using the mean keyword to save computation time:

    >>> import numpy as np
    >>> from timeit import timeit
    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> mean = np.mean(a, axis=1, keepdims=True)
    >>>
    >>> g = globals()
    >>> n = 10000
    >>> t1 = timeit("std = np.std(a, axis=1, mean=mean)", globals=g, number=n)
    >>> t2 = timeit("std = np.std(a, axis=1)", globals=g, number=n)
    >>> print(f'Percentage execution time saved {100*(t2-t1)/t2:.0f}%')
    #doctest: +SKIP
    Percentage execution time saved 30%

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if mean is not np._NoValue:
        kwargs['mean'] = mean

    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    if type(a) is not mu.ndarray:
        try:
            std = a.std
        except AttributeError:
            pass
        else:
            return std(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)

    return _methods._std(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                         **kwargs)


def _var_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                    keepdims=None, *, where=None, mean=None, correction=None):
    return (a, where, out, mean)


@array_function_dispatch(_var_dispatcher)
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, *,
        where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    r"""
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.
        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
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
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero. See notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the variance. See `~numpy.ufunc.reduce` for
        details.

        .. versionadded:: 1.20.0

    mean : array like, optional
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
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

    See Also
    --------
    std, mean, nanmean, nanstd, nanvar
    :ref:`ufuncs-output-type`

    Notes
    -----
    There are several common variants of the array variance calculation.
    Assuming the input `a` is a one-dimensional NumPy array and ``mean`` is
    either provided as an argument or computed as ``a.mean()``, NumPy
    computes the variance of an array as::

        N = len(a)
        d2 = abs(a - mean)**2  # abs is for complex `a`
        var = d2.sum() / (N - ddof)  # note use of `ddof`

    Different values of the argument `ddof` are useful in different
    contexts. NumPy's default ``ddof=0`` corresponds with the expression:

    .. math::

        \frac{\sum_i{|a_i - \bar{a}|^2 }}{N}

    which is sometimes called the "population variance" in the field of
    statistics because it applies the definition of variance to `a` as if `a`
    were a complete population of possible observations.

    Many other libraries define the variance of an array differently, e.g.:

    .. math::

        \frac{\sum_i{|a_i - \bar{a}|^2}}{N - 1}

    In statistics, the resulting quantity is sometimes called the "sample
    variance" because if `a` is a random sample from a larger population,
    this calculation provides an unbiased estimate of the variance of the
    population.  The use of :math:`N-1` in the denominator is often called
    "Bessel's correction" because it corrects for bias (toward lower values)
    in the variance estimate introduced when the sample mean of `a` is used
    in place of the true mean of the population. For this quantity, use
    ``ddof=1``.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    1.25
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    In single precision, var() can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.var(a)
    np.float32(0.20250003)

    Computing the variance in float64 is more accurate:

    >>> np.var(a, dtype=np.float64)
    0.20249999932944759 # may vary
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025

    Specifying a where argument:

    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> np.var(a)
    6.833333333333333 # may vary
    >>> np.var(a, where=[[True], [True], [False]])
    4.0

    Using the mean keyword to save computation time:

    >>> import numpy as np
    >>> from timeit import timeit
    >>>
    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> mean = np.mean(a, axis=1, keepdims=True)
    >>>
    >>> g = globals()
    >>> n = 10000
    >>> t1 = timeit("var = np.var(a, axis=1, mean=mean)", globals=g, number=n)
    >>> t2 = timeit("var = np.var(a, axis=1)", globals=g, number=n)
    >>> print(f'Percentage execution time saved {100*(t2-t1)/t2:.0f}%')
    #doctest: +SKIP
    Percentage execution time saved 32%

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if mean is not np._NoValue:
        kwargs['mean'] = mean

    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    if type(a) is not mu.ndarray:
        try:
            var = a.var

        except AttributeError:
            pass
        else:
            return var(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)

    return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                         **kwargs)
<|endoftext|>"""
Conversion from ctypes to dtype.

In an ideal world, we could achieve this through the PEP3118 buffer protocol,
something like::

    def dtype_from_ctypes_type(t):
        # needed to ensure that the shape of `t` is within memoryview.format
        class DummyStruct(ctypes.Structure):
            _fields_ = [('a', t)]

        # empty to avoid memory allocation
        ctype_0 = (DummyStruct * 0)()
        mv = memoryview(ctype_0)

        # convert the struct, and slice back out the field
        return _dtype_from_pep3118(mv.format)['a']

Unfortunately, this fails because:

* ctypes cannot handle length-0 arrays with PEP3118 (bpo-32782)
* PEP3118 cannot represent unions, but both numpy and ctypes can
* ctypes cannot handle big-endian structs with PEP3118 (bpo-32780)
"""

# We delay-import ctypes for distributions that do not include it.
# While this module is not used unless the user passes in ctypes
# members, it is eagerly imported from numpy/_core/__init__.py.
import numpy as np


def _from_ctypes_array(t):
    return np.dtype((dtype_from_ctypes_type(t._type_), (t._length_,)))


def _from_ctypes_structure(t):
    for item in t._fields_:
        if len(item) > 2:
            raise TypeError(
                "ctypes bitfields have no dtype equivalent")

    if hasattr(t, "_pack_"):
        import ctypes
        formats = []
        offsets = []
        names = []
        current_offset = 0
        for fname, ftyp in t._fields_:
            names.append(fname)
            formats.append(dtype_from_ctypes_type(ftyp))
            # Each type has a default offset, this is platform dependent
            # for some types.
            effective_pack = min(t._pack_, ctypes.alignment(ftyp))
            current_offset = (
                (current_offset + effective_pack - 1) // effective_pack
            ) * effective_pack
            offsets.append(current_offset)
            current_offset += ctypes.sizeof(ftyp)

        return np.dtype(dict(
            formats=formats,
            offsets=offsets,
            names=names,
            itemsize=ctypes.sizeof(t)))
    else:
        fields = []
        for fname, ftyp in t._fields_:
            fields.append((fname, dtype_from_ctypes_type(ftyp)))

        # by default, ctypes structs are aligned
        return np.dtype(fields, align=True)


def _from_ctypes_scalar(t):
    """
    Return the dtype type with endianness included if it's the case
    """
    if getattr(t, '__ctype_be__', None) is t:
        return np.dtype('>' + t._type_)
    elif getattr(t, '__ctype_le__', None) is t:
        return np.dtype('<' + t._type_)
    else:
        return np.dtype(t._type_)


def _from_ctypes_union(t):
    import ctypes
    formats = []
    offsets = []
    names = []
    for fname, ftyp in t._fields_:
        names.append(fname)
        formats.append(dtype_from_ctypes_type(ftyp))
        offsets.append(0)  # Union fields are offset to 0

    return np.dtype(dict(
        formats=formats,
        offsets=offsets,
        names=names,
        itemsize=ctypes.sizeof(t)))


def dtype_from_ctypes_type(t):
    """
    Construct a dtype object from a ctypes type
    """
    import _ctypes
    if issubclass(t, _ctypes.Array):
        return _from_ctypes_array(t)
    elif issubclass(t, _ctypes._Pointer):
        raise TypeError("ctypes pointers have no dtype equivalent")
    elif issubclass(t, _ctypes.Structure):
        return _from_ctypes_structure(t)
    elif issubclass(t, _ctypes.Union):
        return _from_ctypes_union(t)
    elif isinstance(getattr(t, '_type_', None), str):
        return _from_ctypes_scalar(t)
    else:
        raise NotImplementedError(
            "Unknown ctypes type {}".format(t.__name__))
<|endoftext|>"""
Create the numpy._core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

"""

import numpy
from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
# These imports are needed for backward compatibility,
# do not change them. issue gh-11862
# _ones_like is semi-public, on purpose not added to __all__
from ._multiarray_umath import (
    _UFUNC_API, _add_newdoc_ufunc, _ones_like, _get_extobj_dict, _make_extobj,
    _extobj_contextvar)
# These imports are needed for the strip & replace implementations
from ._multiarray_umath import (
    _replace, _strip_whitespace, _lstrip_whitespace, _rstrip_whitespace,
    _strip_chars, _lstrip_chars, _rstrip_chars, _expandtabs_length,
    _expandtabs, _center, _ljust, _rjust, _zfill, _partition, _partition_index,
    _rpartition, _rpartition_index)

__all__ = [
    'absolute', 'add',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj',
    'conjugate', 'copysign', 'cos', 'cosh', 'bitwise_count', 'deg2rad',
    'degrees', 'divide', 'divmod', 'e', 'equal', 'euler_gamma', 'exp', 'exp2',
    'expm1', 'fabs', 'floor', 'floor_divide', 'float_power', 'fmax', 'fmin',
    'fmod', 'frexp', 'frompyfunc', 'gcd', 'greater', 'greater_equal',
    'heaviside', 'hypot', 'invert', 'isfinite', 'isinf', 'isnan', 'isnat',
    'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'log', 'log10',
    'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not',
    'logical_or', 'logical_xor', 'matvec', 'maximum', 'minimum', 'mod', 'modf',
    'multiply', 'negative', 'nextafter', 'not_equal', 'pi', 'positive',
    'power', 'rad2deg', 'radians', 'reciprocal', 'remainder', 'right_shift',
    'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square',
    'subtract', 'tan', 'tanh', 'true_divide', 'trunc', 'vecdot', 'vecmat']
<|endoftext|>def __getattr__(attr_name):
    import warnings
    from numpy.linalg import _linalg
    ret = getattr(_linalg, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.linalg.linalg' has no attribute {attr_name}")
    warnings.warn(
        "The numpy.linalg.linalg has been made private and renamed to "
        "numpy.linalg._linalg. All public functions exported by it are "
        f"available from numpy.linalg. Please use numpy.linalg.{attr_name} "
        "instead.",
        DeprecationWarning,
        stacklevel=3
    )
    return ret
<|endoftext|>"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
contains high-level Python interface to the LAPACK library.  The lite
version only accesses the following LAPACK functions: dgesv, zgesv,
dgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,
zgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.
"""

__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv',
           'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det',
           'svd', 'svdvals', 'eig', 'eigh', 'lstsq', 'norm', 'qr', 'cond',
           'matrix_rank', 'LinAlgError', 'multi_dot', 'trace', 'diagonal',
           'cross', 'outer', 'tensordot', 'matmul', 'matrix_transpose',
           'matrix_norm', 'vector_norm', 'vecdot']

import functools
import operator
import warnings
from typing import NamedTuple, Any

from numpy._utils import set_module
from numpy._core import (
    array, asarray, zeros, empty, empty_like, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, all, inf, dot,
    add, multiply, sqrt, sum, isfinite, finfo, errstate, moveaxis, amin,
    amax, prod, abs, atleast_2d, intp, asanyarray, object_,
    swapaxes, divide, count_nonzero, isnan, sign, argsort, sort,
    reciprocal, overrides, diagonal as _core_diagonal, trace as _core_trace,
    cross as _core_cross, outer as _core_outer, tensordot as _core_tensordot,
    matmul as _core_matmul, matrix_transpose as _core_matrix_transpose,
    transpose as _core_transpose, vecdot as _core_vecdot,
)
from numpy._globals import _NoValue
from numpy.lib._twodim_base_impl import triu, eye
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from numpy.linalg import _umath_linalg

from numpy._typing import NDArray

class EigResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class EighResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class QRResult(NamedTuple):
    Q: NDArray[Any]
    R: NDArray[Any]

class SlogdetResult(NamedTuple):
    sign: NDArray[Any]
    logabsdet: NDArray[Any]

class SVDResult(NamedTuple):
    U: NDArray[Any]
    S: NDArray[Any]
    Vh: NDArray[Any]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.linalg'
)


fortran_int = intc


@set_module('numpy.linalg')
class LinAlgError(ValueError):
    """
    Generic Python-exception-derived object raised by linalg functions.

    General purpose exception class, derived from Python's ValueError
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> LA.inv(np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "...linalg.py", line 350,
        in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
      File "...linalg.py", line 249,
        in solve
        raise LinAlgError('Singular matrix')
    numpy.linalg.LinAlgError: Singular matrix

    """


def _raise_linalgerror_singular(err, flag):
    raise LinAlgError("Singular matrix")

def _raise_linalgerror_nonposdef(err, flag):
    raise LinAlgError("Matrix is not positive definite")

def _raise_linalgerror_eigenvalues_nonconvergence(err, flag):
    raise LinAlgError("Eigenvalues did not converge")

def _raise_linalgerror_svd_nonconvergence(err, flag):
    raise LinAlgError("SVD did not converge")

def _raise_linalgerror_lstsq(err, flag):
    raise LinAlgError("SVD did not converge in Linear Least Squares")

def _raise_linalgerror_qr(err, flag):
    raise LinAlgError("Incorrect argument found while performing "
                      "QR factorization")


def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap

def isComplexType(t):
    return issubclass(t, complexfloating)


_real_types_map = {single: single,
                   double: double,
                   csingle: single,
                   cdouble: double}

_complex_types_map = {single: csingle,
                      double: cdouble,
                      csingle: csingle,
                      cdouble: cdouble}

def _realType(t, default=double):
    return _real_types_map.get(t, default)

def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)

def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        type_ = a.dtype.type
        if issubclass(type_, inexact):
            if isComplexType(type_):
                is_complex = True
            rt = _realType(type_, default=None)
            if rt is double:
                result_type = double
            elif rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            result_type = double
    if is_complex:
        result_type = _complex_types_map[result_type]
        return cdouble, result_type
    else:
        return double, result_type


def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ('=', '|'):
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'two-dimensional' % a.ndim)

def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)

def _assert_stacked_square(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinAlgError('Last 2 dimensions of the array must be square')

def _assert_finite(*arrays):
    for a in arrays:
        if not isfinite(a).all():
            raise LinAlgError("Array must not contain infs or NaNs")

def _is_empty_2d(arr):
    # check size first for efficiency
    return arr.size == 0 and prod(arr.shape[-2:]) == 0


def transpose(a):
    """
    Transpose each matrix in a stack of matrices.

    Unlike np.transpose, this only swaps the last two axes, rather than all of
    them

    Parameters
    ----------
    a : (...,M,N) array_like

    Returns
    -------
    aT : (...,N,M) ndarray
    """
    return swapaxes(a, -1, -2)

# Linear equations

def _tensorsolve_dispatcher(a, b, axes=None):
    return (a, b)


@array_function_dispatch(_tensorsolve_dispatcher)
def tensorsolve(a, b, axes=None):
    """
    Solve the tensor equation ``a x = b`` for x.

    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=x.ndim)``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : ndarray, shape Q

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    numpy.tensordot, tensorinv, numpy.einsum

    Examples
    --------
    >>> import numpy as np
    >>> a = np.eye(2*3*4)
    >>> a.shape = (2*3, 4, 2, 3, 4)
    >>> rng = np.random.default_rng()
    >>> b = rng.normal(size=(2*3, 4))
    >>> x = np.linalg.tensorsolve(a, b)
    >>> x.shape
    (2, 3, 4)
    >>> np.allclose(np.tensordot(a, x, axes=3), b)
    True

    """
    a, wrap = _makearray(a)
    b = asarray(b)
    an = a.ndim

    if axes is not None:
        allaxes = list(range(0, an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)

    oldshape = a.shape[-(an-b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k

    if a.size != prod ** 2:
        raise LinAlgError(
            "Input arrays must satisfy the requirement \
            prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
        )

    a = a.reshape(prod, prod)
    b = b.ravel()
    res = wrap(solve(a, b))
    res.shape = oldshape
    return res


def _solve_dispatcher(a, b):
    return (a, b)


@array_function_dispatch(_solve_dispatcher)
def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.

    Parameters
    ----------
    a : (..., M, M) array_like
        Coefficient matrix.
    b : {(M,), (..., M, K)}, array_like
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : {(..., M,), (..., M, K)} ndarray
        Solution to the system a x = b.  Returned shape is (..., M) if b is
        shape (M,) and (..., M, K) if b is (..., M, K), where the "..." part is
        broadcasted between a and b.

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

    See Also
    --------
    scipy.linalg.solve : Similar function in SciPy.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The solutions are computed using LAPACK routine ``_gesv``.

    `a` must be square and of full-rank, i.e., all rows (or, equivalently,
    columns) must be linearly independent; if either is not true, use
    `lstsq` for the least-squares best "solution" of the
    system/equation.

    .. versionchanged:: 2.0

       The b array is only treated as a shape (M,) column vector if it is
       exactly 1-dimensional. In all other instances it is treated as a stack
       of (M, K) matrices. Previously b would be treated as a stack of (M,)
       vectors if b.ndim was equal to a.ndim - 1.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 22.

    Examples
    --------
    Solve the system of equations:
    ``x0 + 2 * x1 = 1`` and
    ``3 * x0 + 5 * x1 = 2``:

    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 5]])
    >>> b = np.array([1, 2])
    >>> x = np.linalg.solve(a, b)
    >>> x
    array([-1.,  1.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True

    """
    a, _ = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    b, wrap = _makearray(b)
    t, result_t = _commonType(a, b)

    # We use the b = (..., M,) logic, only if the number of extra dimensions
    # match exactly
    if b.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve

    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    with errstate(call=_raise_linalgerror_singular, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        r = gufunc(a, b, signature=signature)

    return wrap(r.astype(result_t, copy=False))


def _tensorinv_dispatcher(a, ind=None):
    return (a,)


@array_function_dispatch(_tensorinv_dispatcher)
def tensorinv(a, ind=2):
    """
    Compute the 'inverse' of an N-dimensional array.

    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    numpy.tensordot, tensorsolve

    Examples
    --------
    >>> import numpy as np
    >>> a = np.eye(4*6)
    >>> a.shape = (4, 6, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)
    >>> rng = np.random.default_rng()
    >>> b = rng.normal(size=(4, 6))
    >>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))
    True

    >>> a = np.eye(4*6)
    >>> a.shape = (24, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=1)
    >>> ainv.shape
    (8, 3, 24)
    >>> rng = np.random.default_rng()
    >>> b = rng.normal(size=24)
    >>> np.allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))
    True

    """
    a = asarray(a)
    oldshape = a.shape
    prod = 1
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        for k in oldshape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = a.reshape(prod, -1)
    ia = inv(a)
    return ia.reshape(*invshape)


# Matrix inversion

def _unary_dispatcher(a):
    return (a,)


@array_function_dispatch(_unary_dispatcher)
def inv(a):
    """
    Compute the inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``a @ ainv = ainv @ a = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.

    Returns
    -------
    ainv : (..., M, M) ndarray or matrix
        Inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.

    See Also
    --------
    scipy.linalg.inv : Similar function in SciPy.
    numpy.linalg.cond : Compute the condition number of a matrix.
    numpy.linalg.svd : Compute the singular value decomposition of a matrix.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    If `a` is detected to be singular, a `LinAlgError` is raised. If `a` is
    ill-conditioned, a `LinAlgError` may or may not be raised, and results may
    be inaccurate due to floating-point errors.

    References
    ----------
    .. [1] Wikipedia, "Condition number",
           https://en.wikipedia.org/wiki/Condition_number

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.linalg import inv
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = inv(a)
    >>> np.allclose(a @ ainv, np.eye(2))
    True
    >>> np.allclose(ainv @ a, np.eye(2))
    True

    If a is a matrix object, then the return value is a matrix as well:

    >>> ainv = inv(np.matrix(a))
    >>> ainv
    matrix([[-2. ,  1. ],
            [ 1.5, -0.5]])

    Inverses of several matrices can be computed at once:

    >>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    >>> inv(a)
    array([[[-2.  ,  1.  ],
            [ 1.5 , -0.5 ]],
           [[-1.25,  0.75],
            [ 0.75, -0.25]]])

    If a matrix is close to singular, the computed inverse may not satisfy
    ``a @ ainv = ainv @ a = eye(a.shape[0])`` even if a `LinAlgError`
    is not raised:

    >>> a = np.array([[2,4,6],[2,0,2],[6,8,14]])
    >>> inv(a)  # No errors raised
    array([[-1.12589991e+15, -5.62949953e+14,  5.62949953e+14],
       [-1.12589991e+15, -5.62949953e+14,  5.62949953e+14],
       [ 1.12589991e+15,  5.62949953e+14, -5.62949953e+14]])
    >>> a @ inv(a)
    array([[ 0.   , -0.5  ,  0.   ],  # may vary
           [-0.5  ,  0.625,  0.25 ],
           [ 0.   ,  0.   ,  1.   ]])

    To detect ill-conditioned matrices, you can use `numpy.linalg.cond` to
    compute its *condition number* [1]_. The larger the condition number, the
    more ill-conditioned the matrix is. As a rule of thumb, if the condition
    number ``cond(a) = 10**k``, then you may lose up to ``k`` digits of
    accuracy on top of what would be lost to the numerical method due to loss
    of precision from arithmetic methods.

    >>> from numpy.linalg import cond
    >>> cond(a)
    np.float64(8.659885634118668e+17)  # may vary

    It is also possible to detect ill-conditioning by inspecting the matrix's
    singular values directly. The ratio between the largest and the smallest
    singular value is the condition number:

    >>> from numpy.linalg import svd
    >>> sigma = svd(a, compute_uv=False)  # Do not compute singular vectors
    >>> sigma.max()/sigma.min()
    8.659885634118668e+17  # may vary

    """
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)

    signature = 'D->D' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_singular, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        ainv = _umath_linalg.inv(a, signature=signature)
    return wrap(ainv.astype(result_t, copy=False))


def _matrix_power_dispatcher(a, n):
    return (a,)


@array_function_dispatch(_matrix_power_dispatcher)
def matrix_power(a, n):
    """
    Raise a square matrix to the (integer) power `n`.

    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If ``n == 0``, the identity matrix
    of the same shape as M is returned. If ``n < 0``, the inverse
    is computed and then raised to the ``abs(n)``.

    .. note:: Stacks of object matrices are not currently supported.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be "powered".
    n : int
        The exponent can be any integer or long integer, positive,
        negative, or zero.

    Returns
    -------
    a**n : (..., M, M) ndarray or matrix object
        The return value is the same shape and type as `M`;
        if the exponent is positive or zero then the type of the
        elements is the same as those of `M`. If the exponent is
        negative the elements are floating-point.

    Raises
    ------
    LinAlgError
        For matrices that are not square or that (for negative powers) cannot
        be inverted numerically.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.linalg import matrix_power
    >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit
    >>> matrix_power(i, 3) # should = -i
    array([[ 0, -1],
           [ 1,  0]])
    >>> matrix_power(i, 0)
    array([[1, 0],
           [0, 1]])
    >>> matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements
    array([[ 0.,  1.],
           [-1.,  0.]])

    Somewhat more sophisticated example

    >>> q = np.zeros((4, 4))
    >>> q[0:2, 0:2] = -i
    >>> q[2:4, 2:4] = i
    >>> q # one of the three quaternion units not equal to 1
    array([[ 0., -1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])
    >>> matrix_power(q, 2) # = -np.eye(4)
    array([[-1.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0.,  0., -1.]])

    """
    a = asanyarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)

    try:
        n = operator.index(n)
    except TypeError as e:
        raise TypeError("exponent must be an integer") from e

    # Fall back on dot for object arrays. Object arrays are not supported by
    # the current implementation of matmul using einsum
    if a.dtype != object:
        fmatmul = matmul
    elif a.ndim == 2:
        fmatmul = dot
    else:
        raise NotImplementedError(
            "matrix_power not supported for stacks of object arrays")

    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a

    elif n < 0:
        a = inv(a)
        n = abs(n)

    # short-cuts.
    if n == 1:
        return a

    elif n == 2:
        return fmatmul(a, a)

    elif n == 3:
        return fmatmul(fmatmul(a, a), a)

    # Use binary decomposition to reduce the number of matrix multiplications.
    # Here, we iterate over the bits of n, from LSB to MSB, raise `a` to
    # increasing powers of 2, and multiply into the result as needed.
    z = result = None
    while n > 0:
        z = a if z is None else fmatmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else fmatmul(result, z)

    return result


# Cholesky decomposition

def _cholesky_dispatcher(a, /, *, upper=None):
    return (a,)


@array_function_dispatch(_cholesky_dispatcher)
def cholesky(a, /, *, upper=False):
    """
    Cholesky decomposition.

    Return the lower or upper Cholesky decomposition, ``L * L.H`` or
    ``U.H * U``, of the square matrix ``a``, where ``L`` is lower-triangular,
    ``U`` is upper-triangular, and ``.H`` is the conjugate transpose operator
    (which is the ordinary transpose if ``a`` is real-valued). ``a`` must be
    Hermitian (symmetric if real-valued) and positive-definite. No checking is
    performed to verify whether ``a`` is Hermitian or not. In addition, only
    the lower or upper-triangular and diagonal elements of ``a`` are used.
    Only ``L`` or ``U`` is actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.
    upper : bool
        If ``True``, the result must be the upper-triangular Cholesky factor.
        If ``False``, the result must be the lower-triangular Cholesky factor.
        Default: ``False``.

    Returns
    -------
    L : (..., M, M) array_like
        Lower or upper-triangular Cholesky factor of `a`. Returns a matrix
        object if `a` is a matrix object.

    Raises
    ------
    LinAlgError
       If the decomposition fails, for example, if `a` is not
       positive-definite.

    See Also
    --------
    scipy.linalg.cholesky : Similar function in SciPy.
    scipy.linalg.cholesky_banded : Cholesky decompose a banded Hermitian
                                   positive-definite matrix.
    scipy.linalg.cho_factor : Cholesky decomposition of a matrix, to use in
                              `scipy.linalg.cho_solve`.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \\mathbf{x} = \\mathbf{b}

    (when `A` is both Hermitian/symmetric and positive-definite).

    First, we solve for :math:`\\mathbf{y}` in

    .. math:: L \\mathbf{y} = \\mathbf{b},

    and then for :math:`\\mathbf{x}` in

    .. math:: L^{H} \\mathbf{x} = \\mathbf{y}.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1,-2j],[2j,5]])
    >>> A
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[1.+0.j, 0.+0.j],
           [0.+2.j, 1.+0.j]])
    >>> np.dot(L, L.T.conj()) # verify that L * L.H = A
    array([[1.+0.j, 0.-2.j],
           [0.+2.j, 5.+0.j]])
    >>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?
    >>> np.linalg.cholesky(A) # an ndarray object is returned
    array([[1.+0.j, 0.+0.j],
           [0.+2.j, 1.+0.j]])
    >>> # But a matrix object is returned if A is a matrix object
    >>> np.linalg.cholesky(np.matrix(A))
    matrix([[ 1.+0.j,  0.+0.j],
            [ 0.+2.j,  1.+0.j]])
    >>> # The upper-triangular Cholesky factor can also be obtained.
    >>> np.linalg.cholesky(A, upper=True)
    array([[1.-0.j, 0.-2.j],
           [0.-0.j, 1.-0.j]])

    """
    gufunc = _umath_linalg.cholesky_up if upper else _umath_linalg.cholesky_lo
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_nonposdef, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        r = gufunc(a, signature=signature)
    return wrap(r.astype(result_t, copy=False))


# outer product


def _outer_dispatcher(x1, x2):
    return (x1, x2)


@array_function_dispatch(_outer_dispatcher)
def outer(x1, x2, /):
    """
    Compute the outer product of two vectors.

    This function is Array API compatible. Compared to ``np.outer``
    it accepts 1-dimensional inputs only.

    Parameters
    ----------
    x1 : (M,) array_like
        One-dimensional input array of size ``N``.
        Must have a numeric data type.
    x2 : (N,) array_like
        One-dimensional input array of size ``M``.
        Must have a numeric data type.

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``

    See also
    --------
    outer

    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> rl = np.linalg.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.linalg.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
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
    >>> np.linalg.outer(x, [1, 2, 3])
    array([['a', 'aa', 'aaa'],
           ['b', 'bb', 'bbb'],
           ['c', 'cc', 'ccc']], dtype=object)

    """
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "Input arrays must be one-dimensional, but they are "
            f"{x1.ndim=} and {x2.ndim=}."
        )
    return _core_outer(x1, x2, out=None)


# QR decomposition


def _qr_dispatcher(a, mode=None):
    return (a,)


@array_function_dispatch(_qr_dispatcher)
def qr(a, mode='reduced'):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    Parameters
    ----------
    a : array_like, shape (..., M, N)
        An array-like object with the dimensionality of at least 2.
    mode : {'reduced', 'complete', 'r', 'raw'}, optional, default: 'reduced'
        If K = min(M, N), then

        * 'reduced'  : returns Q, R with dimensions (..., M, K), (..., K, N)
        * 'complete' : returns Q, R with dimensions (..., M, M), (..., M, N)
        * 'r'        : returns R only with dimensions (..., K, N)
        * 'raw'      : returns h, tau with dimensions (..., N, M), (..., K,)

        The options 'reduced', 'complete, and 'raw' are new in numpy 1.8,
        see the notes for more information. The default is 'reduced', and to
        maintain backward compatibility with earlier versions of numpy both
        it and the old default 'full' can be omitted. Note that array h
        returned in 'raw' mode is transposed for calling Fortran. The
        'economic' mode is deprecated.  The modes 'full' and 'economic' may
        be passed using only the first letter for backwards compatibility,
        but all others must be spelled out. See the Notes for more
        explanation.


    Returns
    -------
    When mode is 'reduced' or 'complete', the result will be a namedtuple with
    the attributes `Q` and `R`.

    Q : ndarray of float or complex, optional
        A matrix with orthonormal columns. When mode = 'complete' the
        result is an orthogonal/unitary matrix depending on whether or not
        a is real/complex. The determinant may be either +/- 1 in that
        case. In case the number of dimensions in the input array is
        greater than 2 then a stack of the matrices with above properties
        is returned.
    R : ndarray of float or complex, optional
        The upper-triangular matrix or a stack of upper-triangular
        matrices if the number of dimensions in the input array is greater
        than 2.
    (h, tau) : ndarrays of np.double or np.cdouble, optional
        The array h contains the Householder reflectors that generate q
        along with r. The tau array contains scaling factors for the
        reflectors. In the deprecated  'economic' mode only h is returned.

    Raises
    ------
    LinAlgError
        If factoring fails.

    See Also
    --------
    scipy.linalg.qr : Similar function in SciPy.
    scipy.linalg.rq : Compute RQ decomposition of a matrix.

    Notes
    -----
    This is an interface to the LAPACK routines ``dgeqrf``, ``zgeqrf``,
    ``dorgqr``, and ``zungqr``.

    For more information on the qr factorization, see for example:
    https://en.wikipedia.org/wiki/QR_factorization

    Subclasses of `ndarray` are preserved except for the 'raw' mode. So if
    `a` is of type `matrix`, all the return values will be matrices too.

    New 'reduced', 'complete', and 'raw' options for mode were added in
    NumPy 1.8.0 and the old option 'full' was made an alias of 'reduced'.  In
    addition the options 'full' and 'economic' were deprecated.  Because
    'full' was the previous default and 'reduced' is the new default,
    backward compatibility can be maintained by letting `mode` default.
    The 'raw' option was added so that LAPACK routines that can multiply
    arrays by q using the Householder reflectors can be used. Note that in
    this case the returned arrays are of type np.double or np.cdouble and
    the h array is transposed to be FORTRAN compatible.  No routines using
    the 'raw' return are currently exposed by numpy, but some are available
    in lapack_lite and just await the necessary work.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> a = rng.normal(size=(9, 6))
    >>> Q, R = np.linalg.qr(a)
    >>> np.allclose(a, np.dot(Q, R))  # a does equal QR
    True
    >>> R2 = np.linalg.qr(a, mode='r')
    >>> np.allclose(R, R2)  # mode='r' returns the same R as mode='full'
    True
    >>> a = np.random.normal(size=(3, 2, 2)) # Stack of 2 x 2 matrices as input
    >>> Q, R = np.linalg.qr(a)
    >>> Q.shape
    (3, 2, 2)
    >>> R.shape
    (3, 2, 2)
    >>> np.allclose(a, np.matmul(Q, R))
    True

    Example illustrating a common use of `qr`: solving of least squares
    problems

    What are the least-squares-best `m` and `y0` in ``y = y0 + mx`` for
    the following data: {(0,1), (1,0), (1,2), (2,1)}. (Graph the points
    and you'll see that it should be y0 = 0, m = 1.)  The answer is provided
    by solving the over-determined matrix equation ``Ax = b``, where::

      A = array([[0, 1], [1, 1], [1, 1], [2, 1]])
      x = array([[y0], [m]])
      b = array([[1], [0], [2], [1]])

    If A = QR such that Q is orthonormal (which is always possible via
    Gram-Schmidt), then ``x = inv(R) * (Q.T) * b``.  (In numpy practice,
    however, we simply use `lstsq`.)

    >>> A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])
    >>> A
    array([[0, 1],
           [1, 1],
           [1, 1],
           [2, 1]])
    >>> b = np.array([1, 2, 2, 3])
    >>> Q, R = np.linalg.qr(A)
    >>> p = np.dot(Q.T, b)
    >>> np.dot(np.linalg.inv(R), p)
    array([  1.,   1.])

    """
    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full'):
            # 2013-04-01, 1.8
            msg = (
                "The 'full' option is deprecated in favor of 'reduced'.\n"
                "For backward compatibility let mode default."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = 'reduced'
        elif mode in ('e', 'economic'):
            # 2013-04-01, 1.8
            msg = "The 'economic' option is deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = 'economic'
        else:
            raise ValueError(f"Unrecognized mode '{mode}'")

    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    m, n = a.shape[-2:]
    t, result_t = _commonType(a)
    a = a.astype(t, copy=True)
    a = _to_native_byte_order(a)
    mn = min(m, n)

    signature = 'D->D' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_qr, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        tau = _umath_linalg.qr_r_raw(a, signature=signature)

    # handle modes that don't return q
    if mode == 'r':
        r = triu(a[..., :mn, :])
        r = r.astype(result_t, copy=False)
        return wrap(r)

    if mode == 'raw':
        q = transpose(a)
        q = q.astype(result_t, copy=False)
        tau = tau.astype(result_t, copy=False)
        return wrap(q), tau

    if mode == 'economic':
        a = a.astype(result_t, copy=False)
        return wrap(a)

    # mc is the number of columns in the resulting q
    # matrix. If the mode is complete then it is
    # same as number of rows, and if the mode is reduced,
    # then it is the minimum of number of rows and columns.
    if mode == 'complete' and m > n:
        mc = m
        gufunc = _umath_linalg.qr_complete
    else:
        mc = mn
        gufunc = _umath_linalg.qr_reduced

    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    with errstate(call=_raise_linalgerror_qr, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        q = gufunc(a, tau, signature=signature)
    r = triu(a[..., :mc, :])

    q = q.astype(result_t, copy=False)
    r = r.astype(result_t, copy=False)

    return QRResult(wrap(q), wrap(r))

# Eigenvalues


@array_function_dispatch(_unary_dispatcher)
def eigvals(a):
    """
    Compute the eigenvalues of a general matrix.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered, nor are they necessarily
        real for real matrices.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays
    eigvalsh : eigenvalues of real symmetric or complex Hermitian
               (conjugate symmetric) arrays.
    eigh : eigenvalues and eigenvectors of real symmetric or complex
           Hermitian (conjugate symmetric) arrays.
    scipy.linalg.eigvals : Similar function in SciPy.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    This is implemented using the ``_geev`` LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    Examples
    --------
    Illustration, using the fact that the eigenvalues of a diagonal matrix
    are its diagonal elements, that multiplying a matrix on the left
    by an orthogonal matrix, `Q`, and on the right by `Q.T` (the transpose
    of `Q`), preserves the eigenvalues of the "middle" matrix. In other words,
    if `Q` is orthogonal, then ``Q * A * Q.T`` has the same eigenvalues as
    ``A``:

    >>> import numpy as np
    >>> from numpy import linalg as LA
    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])
    (1.0, 1.0, 0.0)

    Now multiply a diagonal matrix by ``Q`` on one side and
    by ``Q.T`` on the other:

    >>> D = np.diag((-1,1))
    >>> LA.eigvals(D)
    array([-1.,  1.])
    >>> A = np.dot(Q, D)
    >>> A = np.dot(A, Q.T)
    >>> LA.eigvals(A)
    array([ 1., -1.]) # random

    """
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)

    signature = 'D->D' if isComplexType(t) else 'd->D'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w = _umath_linalg.eigvals(a, signature=signature)

    if not isComplexType(t):
        if all(w.imag == 0):
            w = w.real
            result_t = _realType(result_t)
        else:
            result_t = _complexType(result_t)

    return w.astype(result_t, copy=False)


def _eigvalsh_dispatcher(a, UPLO=None):
    return (a,)


@array_function_dispatch(_eigvalsh_dispatcher)
def eigvalsh(a, UPLO='L'):
    """
    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Main difference from eigh: the eigenvectors are not computed.

    Parameters
    ----------
    a : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues are to be
        computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigh : eigenvalues and eigenvectors of real symmetric or complex Hermitian
           (conjugate symmetric) arrays.
    eigvals : eigenvalues of general real or complex arrays.
    eig : eigenvalues and right eigenvectors of general real or complex
          arrays.
    scipy.linalg.eigvalsh : Similar function in SciPy.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The eigenvalues are computed using LAPACK routines ``_syevd``, ``_heevd``.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> LA.eigvalsh(a)
    array([ 0.17157288,  5.82842712]) # may vary

    >>> # demonstrate the treatment of the imaginary part of the diagonal
    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # with UPLO='L' this is numerically equivalent to using LA.eigvals()
    >>> # with:
    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa = LA.eigvalsh(a)
    >>> wb = LA.eigvals(b)
    >>> wa; wb
    array([1., 6.])
    array([6.+0.j, 1.+0.j])

    """
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    if UPLO == 'L':
        gufunc = _umath_linalg.eigvalsh_lo
    else:
        gufunc = _umath_linalg.eigvalsh_up

    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->d' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w = gufunc(a, signature=signature)
    return w.astype(_realType(result_t), copy=False)

def _convertarray(a):
    t, result_t = _commonType(a)
    a = a.astype(t).T.copy()
    return a, t, result_t


# Eigenvectors


@array_function_dispatch(_unary_dispatcher)
def eig(a):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed

    Returns
    -------
    A namedtuple with the following attributes:

    eigenvalues : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be of complex type, unless the imaginary part is
        zero in which case it will be cast to a real type. When `a`
        is real the resulting eigenvalues will be real (0 imaginary
        part) or occur in conjugate pairs

    eigenvectors : (..., M, M) array
        The normalized (unit "length") eigenvectors, such that the
        column ``eigenvectors[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``eigenvalues[i]``.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvals : eigenvalues of a non-symmetric array.
    eigh : eigenvalues and eigenvectors of a real symmetric or complex
           Hermitian (conjugate symmetric) array.
    eigvalsh : eigenvalues of a real symmetric or complex Hermitian
               (conjugate symmetric) array.
    scipy.linalg.eig : Similar function in SciPy that also solves the
                       generalized eigenvalue problem.
    scipy.linalg.schur : Best choice for unitary and other non-Hermitian
                         normal matrices.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    This is implemented using the ``_geev`` LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    The number `w` is an eigenvalue of `a` if there exists a vector `v` such
    that ``a @ v = w * v``. Thus, the arrays `a`, `eigenvalues`, and
    `eigenvectors` satisfy the equations ``a @ eigenvectors[:,i] =
    eigenvalues[i] * eigenvectors[:,i]`` for :math:`i \\in \\{0,...,M-1\\}`.

    The array `eigenvectors` may not be of maximum rank, that is, some of the
    columns may be linearly dependent, although round-off error may obscure
    that fact. If the eigenvalues are all different, then theoretically the
    eigenvectors are linearly independent and `a` can be diagonalized by a
    similarity transformation using `eigenvectors`, i.e, ``inv(eigenvectors) @
    a @ eigenvectors`` is diagonal.

    For non-Hermitian normal matrices the SciPy function `scipy.linalg.schur`
    is preferred because the matrix `eigenvectors` is guaranteed to be
    unitary, which is not the case when using `eig`. The Schur factorization
    produces an upper triangular matrix rather than a diagonal matrix, but for
    normal matrices only the diagonal of the upper triangular matrix is
    needed, the rest is roundoff error.

    Finally, it is emphasized that `eigenvectors` consists of the *right* (as
    in right-hand side) eigenvectors of `a`. A vector `y` satisfying ``y.T @ a
    = z * y.T`` for some number `z` is called a *left* eigenvector of `a`,
    and, in general, the left and right eigenvectors of a matrix are not
    necessarily the (perhaps conjugate) transposes of each other.

    References
    ----------
    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,
    Academic Press, Inc., 1980, Various pp.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import linalg as LA

    (Almost) trivial example with real eigenvalues and eigenvectors.

    >>> eigenvalues, eigenvectors = LA.eig(np.diag((1, 2, 3)))
    >>> eigenvalues
    array([1., 2., 3.])
    >>> eigenvectors
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Real matrix possessing complex eigenvalues and eigenvectors;
    note that the eigenvalues are complex conjugates of each other.

    >>> eigenvalues, eigenvectors = LA.eig(np.array([[1, -1], [1, 1]]))
    >>> eigenvalues
    array([1.+1.j, 1.-1.j])
    >>> eigenvectors
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])

    Complex-valued matrix with real eigenvalues (but complex-valued
    eigenvectors); note that ``a.conj().T == a``, i.e., `a` is Hermitian.

    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> eigenvalues, eigenvectors = LA.eig(a)
    >>> eigenvalues
    array([2.+0.j, 0.+0.j])
    >>> eigenvectors
    array([[ 0.        +0.70710678j,  0.70710678+0.j        ], # may vary
           [ 0.70710678+0.j        , -0.        +0.70710678j]])

    Be careful about round-off error!

    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. eigenvalues are 1 +/- 1e-9
    >>> eigenvalues, eigenvectors = LA.eig(a)
    >>> eigenvalues
    array([1., 1.])
    >>> eigenvectors
    array([[1., 0.],
           [0., 1.]])

    """
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)

    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w, vt = _umath_linalg.eig(a, signature=signature)

    if not isComplexType(t) and all(w.imag == 0.0):
        w = w.real
        vt = vt.real
        result_t = _realType(result_t)
    else:
        result_t = _complexType(result_t)

    vt = vt.astype(result_t, copy=False)
    return EigResult(w.astype(result_t, copy=False), wrap(vt))


@array_function_dispatch(_eigvalsh_dispatcher)
def eigh(a, UPLO='L'):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Parameters
    ----------
    a : (..., M, M) array
        Hermitian or real symmetric matrices whose eigenvalues and
        eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    A namedtuple with the following attributes:

    eigenvalues : (..., M) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    eigenvectors : {(..., M, M) ndarray, (..., M, M) matrix}
        The column ``eigenvectors[:, i]`` is the normalized eigenvector
        corresponding to the eigenvalue ``eigenvalues[i]``.  Will return a
        matrix object if `a` is a matrix object.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of real symmetric or complex Hermitian
               (conjugate symmetric) arrays.
    eig : eigenvalues and right eigenvectors for non-symmetric arrays.
    eigvals : eigenvalues of non-symmetric arrays.
    scipy.linalg.eigh : Similar function in SciPy (but also solves the
                        generalized eigenvalue problem).

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``,
    ``_heevd``.

    The eigenvalues of real symmetric or complex Hermitian matrices are always
    real. [1]_ The array `eigenvalues` of (column) eigenvectors is unitary and
    `a`, `eigenvalues`, and `eigenvectors` satisfy the equations ``dot(a,
    eigenvectors[:, i]) = eigenvalues[i] * eigenvectors[:, i]``.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 222.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> a
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> eigenvalues, eigenvectors = LA.eigh(a)
    >>> eigenvalues
    array([0.17157288, 5.82842712])
    >>> eigenvectors
    array([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary
           [ 0.        +0.38268343j,  0.        -0.92387953j]])

    >>> (np.dot(a, eigenvectors[:, 0]) -
    ... eigenvalues[0] * eigenvectors[:, 0])  # verify 1st eigenval/vec pair
    array([5.55111512e-17+0.0000000e+00j, 0.00000000e+00+1.2490009e-16j])
    >>> (np.dot(a, eigenvectors[:, 1]) -
    ... eigenvalues[1] * eigenvectors[:, 1])  # verify 2nd eigenval/vec pair
    array([0.+0.j, 0.+0.j])

    >>> A = np.matrix(a) # what happens if input is a matrix object
    >>> A
    matrix([[ 1.+0.j, -0.-2.j],
            [ 0.+2.j,  5.+0.j]])
    >>> eigenvalues, eigenvectors = LA.eigh(A)
    >>> eigenvalues
    array([0.17157288, 5.82842712])
    >>> eigenvectors
    matrix([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary
            [ 0.        +0.38268343j,  0.        -0.92387953j]])

    >>> # demonstrate the treatment of the imaginary part of the diagonal
    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # with UPLO='L' this is numerically equivalent to using LA.eig() with:
    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa, va = LA.eigh(a)
    >>> wb, vb = LA.eig(b)
    >>> wa
    array([1., 6.])
    >>> wb
    array([6.+0.j, 1.+0.j])
    >>> va
    array([[-0.4472136 +0.j        , -0.89442719+0.j        ], # may vary
           [ 0.        +0.89442719j,  0.        -0.4472136j ]])
    >>> vb
    array([[ 0.89442719+0.j       , -0.        +0.4472136j],
           [-0.        +0.4472136j,  0.89442719+0.j       ]])

    """
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)

    if UPLO == 'L':
        gufunc = _umath_linalg.eigh_lo
    else:
        gufunc = _umath_linalg.eigh_up

    signature = 'D->dD' if isComplexType(t) else 'd->dd'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w, vt = gufunc(a, signature=signature)
    w = w.astype(_realType(result_t), copy=False)
    vt = vt.astype(result_t, copy=False)
    return EighResult(w, wrap(vt))


# Singular value decomposition

def _svd_dispatcher(a, full_matrices=None, compute_uv=None, hermitian=None):
    return (a,)


@array_function_dispatch(_svd_dispatcher)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    Singular Value Decomposition.

    When `a` is a 2D array, and ``full_matrices=False``, then it is
    factorized as ``u @ np.diag(s) @ vh = (u * s) @ vh``, where
    `u` and the Hermitian transpose of `vh` are 2D arrays with
    orthonormal columns and `s` is a 1D array of `a`'s singular
    values. When `a` is higher-dimensional, SVD is applied in
    stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) array_like
        A real or complex array with ``a.ndim >= 2``.
    full_matrices : bool, optional
        If True (default), `u` and `vh` have the shapes ``(..., M, M)`` and
        ``(..., N, N)``, respectively.  Otherwise, the shapes are
        ``(..., M, K)`` and ``(..., K, N)``, respectively, where
        ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether or not to compute `u` and `vh` in addition to `s`.  True
        by default.
    hermitian : bool, optional
        If True, `a` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.

    Returns
    -------
    When `compute_uv` is True, the result is a namedtuple with the following
    attribute names:

    U : { (..., M, M), (..., M, K) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    S : (..., K) array
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    Vh : { (..., N, N), (..., K, N) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    scipy.linalg.svd : Similar function in SciPy.
    scipy.linalg.svdvals : Compute singular values of a matrix.

    Notes
    -----
    The decomposition is performed using LAPACK routine ``_gesdd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U S V^H`, where :math:`A = a`, :math:`U= u`,
    :math:`S= \\mathtt{np.diag}(s)` and :math:`V^H = vh`. The 1D array `s`
    contains the singular values of `a` and `u` and `vh` are unitary. The rows
    of `vh` are the eigenvectors of :math:`A^H A` and the columns of `u` are
    the eigenvectors of :math:`A A^H`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    If `a` has more than two dimensions, then broadcasting rules apply, as
    explained in :ref:`routines.linalg-broadcasting`. This means that SVD is
    working in "stacked" mode: it iterates over all indices of the first
    ``a.ndim - 2`` dimensions and for each combination SVD is applied to the
    last two indices. The matrix `a` can be reconstructed from the
    decomposition with either ``(u * s[..., None, :]) @ vh`` or
    ``u @ (s[..., None] * vh)``. (The ``@`` operator can be replaced by the
    function ``np.matmul`` for python versions below 3.5.)

    If `a` is a ``matrix`` object (as opposed to an ``ndarray``), then so are
    all the return values.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> a = rng.normal(size=(9, 6)) + 1j*rng.normal(size=(9, 6))
    >>> b = rng.normal(size=(2, 7, 8, 3)) + 1j*rng.normal(size=(2, 7, 8, 3))


    Reconstruction based on full SVD, 2D case:

    >>> U, S, Vh = np.linalg.svd(a, full_matrices=True)
    >>> U.shape, S.shape, Vh.shape
    ((9, 9), (6,), (6, 6))
    >>> np.allclose(a, np.dot(U[:, :6] * S, Vh))
    True
    >>> smat = np.zeros((9, 6), dtype=complex)
    >>> smat[:6, :6] = np.diag(S)
    >>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))
    True

    Reconstruction based on reduced SVD, 2D case:

    >>> U, S, Vh = np.linalg.svd(a, full_matrices=False)
    >>> U.shape, S.shape, Vh.shape
    ((9, 6), (6,), (6, 6))
    >>> np.allclose(a, np.dot(U * S, Vh))
    True
    >>> smat = np.diag(S)
    >>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))
    True

    Reconstruction based on full SVD, 4D case:

    >>> U, S, Vh = np.linalg.svd(b, full_matrices=True)
    >>> U.shape, S.shape, Vh.shape
    ((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(U[..., :3] * S[..., None, :], Vh))
    True
    >>> np.allclose(b, np.matmul(U[..., :3], S[..., None] * Vh))
    True

    Reconstruction based on reduced SVD, 4D case:

    >>> U, S, Vh = np.linalg.svd(b, full_matrices=False)
    >>> U.shape, S.shape, Vh.shape
    ((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(U * S[..., None, :], Vh))
    True
    >>> np.allclose(b, np.matmul(U, S[..., None] * Vh))
    True

    """
    import numpy as _nx
    a, wrap = _makearray(a)

    if hermitian:
        # note: lapack svd returns eigenvalues with s ** 2 sorted descending,
        # but eig returns s sorted ascending, so we re-order the eigenvalues
        # and related arrays to have the correct order
        if compute_uv:
            s, u = eigh(a)
            sgn = sign(s)
            s = abs(s)
            sidx = argsort(s)[..., ::-1]
            sgn = _nx.take_along_axis(sgn, sidx, axis=-1)
            s = _nx.take_along_axis(s, sidx, axis=-1)
            u = _nx.take_along_axis(u, sidx[..., None, :], axis=-1)
            # singular values are unsigned, move the sign into v
            vt = transpose(u * sgn[..., None, :]).conjugate()
            return SVDResult(wrap(u), s, wrap(vt))
        else:
            s = eigvalsh(a)
            s = abs(s)
            return sort(s)[..., ::-1]

    _assert_stacked_2d(a)
    t, result_t = _commonType(a)

    m, n = a.shape[-2:]
    if compute_uv:
        if full_matrices:
            gufunc = _umath_linalg.svd_f
        else:
            gufunc = _umath_linalg.svd_s

        signature = 'D->DdD' if isComplexType(t) else 'd->ddd'
        with errstate(call=_raise_linalgerror_svd_nonconvergence,
                      invalid='call', over='ignore', divide='ignore',
                      under='ignore'):
            u, s, vh = gufunc(a, signature=signature)
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)
        return SVDResult(wrap(u), s, wrap(vh))
    else:
        signature = 'D->d' if isComplexType(t) else 'd->d'
        with errstate(call=_raise_linalgerror_svd_nonconvergence,
                      invalid='call', over='ignore', divide='ignore',
                      under='ignore'):
            s = _umath_linalg.svd(a, signature=signature)
        s = s.astype(_realType(result_t), copy=False)
        return s


def _svdvals_dispatcher(x):
    return (x,)


@array_function_dispatch(_svdvals_dispatcher)
def svdvals(x, /):
    """
    Returns the singular values of a matrix (or a stack of matrices) ``x``.
    When x is a stack of matrices, the function will compute the singular
    values for each matrix in the stack.

    This function is Array API compatible.

    Calling ``np.svdvals(x)`` to get singular values is the same as
    ``np.svd(x, compute_uv=False, hermitian=False)``.

    Parameters
    ----------
    x : (..., M, N) array_like
        Input array having shape (..., M, N) and whose last two
        dimensions form matrices on which to perform singular value
        decomposition. Should have a floating-point data type.

    Returns
    -------
    out : ndarray
        An array with shape (..., K) that contains the vector(s)
        of singular values of length K, where K = min(M, N).

    See Also
    --------
    scipy.linalg.svdvals : Compute singular values of a matrix.

    Examples
    --------

    >>> np.linalg.svdvals([[1, 2, 3, 4, 5],
    ...                    [1, 4, 9, 16, 25],
    ...                    [1, 8, 27, 64, 125]])
    array([146.68862757,   5.57510612,   0.60393245])

    Determine the rank of a matrix using singular values:

    >>> s = np.linalg.svdvals([[1, 2, 3],
    ...                        [2, 4, 6],
    ...                        [-1, 1, -1]]); s
    array([8.38434191e+00, 1.64402274e+00, 2.31534378e-16])
    >>> np.count_nonzero(s > 1e-10)  # Matrix of rank 2
    2

    """
    return svd(x, compute_uv=False, hermitian=False)


def _cond_dispatcher(x, p=None):
    return (x,)


@array_function_dispatch(_cond_dispatcher)
def cond(x, p=None):
    """
    Compute the condition number of a matrix.

    This function is capable of returning the condition number using
    one of seven different norms, depending on the value of `p` (see
    Parameters below).

    Parameters
    ----------
    x : (..., M, N) array_like
        The matrix whose condition number is sought.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional
        Order of the norm used in the condition number computation:

        =====  ============================
        p      norm for matrices
        =====  ============================
        None   2-norm, computed directly using the ``SVD``
        'fro'  Frobenius norm
        inf    max(sum(abs(x), axis=1))
        -inf   min(sum(abs(x), axis=1))
        1      max(sum(abs(x), axis=0))
        -1     min(sum(abs(x), axis=0))
        2      2-norm (largest sing. value)
        -2     smallest singular value
        =====  ============================

        inf means the `numpy.inf` object, and the Frobenius norm is
        the root-of-sum-of-squares norm.

    Returns
    -------
    c : {float, inf}
        The condition number of the matrix. May be infinite.

    See Also
    --------
    numpy.linalg.norm

    Notes
    -----
    The condition number of `x` is defined as the norm of `x` times the
    norm of the inverse of `x` [1]_; the norm can be the usual L2-norm
    (root-of-sum-of-squares) or one of a number of other matrix norms.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, Orlando, FL,
           Academic Press, Inc., 1980, pg. 285.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> a
    array([[ 1,  0, -1],
           [ 0,  1,  0],
           [ 1,  0,  1]])
    >>> LA.cond(a)
    1.4142135623730951
    >>> LA.cond(a, 'fro')
    3.1622776601683795
    >>> LA.cond(a, np.inf)
    2.0
    >>> LA.cond(a, -np.inf)
    1.0
    >>> LA.cond(a, 1)
    2.0
    >>> LA.cond(a, -1)
    1.0
    >>> LA.cond(a, 2)
    1.4142135623730951
    >>> LA.cond(a, -2)
    0.70710678118654746 # may vary
    >>> (min(LA.svd(a, compute_uv=False)) *
    ... min(LA.svd(LA.inv(a), compute_uv=False)))
    0.70710678118654746 # may vary

    """
    x = asarray(x)  # in case we have a matrix
    if _is_empty_2d(x):
        raise LinAlgError("cond is not defined on empty arrays")
    if p is None or p == 2 or p == -2:
        s = svd(x, compute_uv=False)
        with errstate(all='ignore'):
            if p == -2:
                r = s[..., -1] / s[..., 0]
            else:
                r = s[..., 0] / s[..., -1]
    else:
        # Call inv(x) ignoring errors. The result array will
        # contain nans in the entries where inversion failed.
        _assert_stacked_2d(x)
        _assert_stacked_square(x)
        t, result_t = _commonType(x)
        signature = 'D->D' if isComplexType(t) else 'd->d'
        with errstate(all='ignore'):
            invx = _umath_linalg.inv(x, signature=signature)
            r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
        r = r.astype(result_t, copy=False)

    # Convert nans to infs unless the original array had nan entries
    r = asarray(r)
    nan_mask = isnan(r)
    if nan_mask.any():
        nan_mask &= ~isnan(x).any(axis=(-2, -1))
        if r.ndim > 0:
            r[nan_mask] = inf
        elif nan_mask:
            r[()] = inf

    # Convention is to return scalars instead of 0d arrays
    if r.ndim == 0:
        r = r[()]

    return r


def _matrix_rank_dispatcher(A, tol=None, hermitian=None, *, rtol=None):
    return (A,)


@array_function_dispatch(_matrix_rank_dispatcher)
def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    """
    Return matrix rank of array using SVD method

    Rank of the array is the number of singular values of the array that are
    greater than `tol`.

    Parameters
    ----------
    A : {(M,), (..., M, N)} array_like
        Input vector or stack of matrices.
    tol : (...) array_like, float, optional
        Threshold below which SVD values are considered zero. If `tol` is
        None, and ``S`` is an array with singular values for `M`, and
        ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
        set to ``S.max() * max(M, N) * eps``.
    hermitian : bool, optional
        If True, `A` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.
    rtol : (...) array_like, float, optional
        Parameter for the relative tolerance component. Only ``tol`` or
        ``rtol`` can be set at a time. Defaults to ``max(M, N) * eps``.

        .. versionadded:: 2.0.0

    Returns
    -------
    rank : (...) array_like
        Rank of A.

    Notes
    -----
    The default threshold to detect rank deficiency is a test on the magnitude
    of the singular values of `A`.  By default, we identify singular values
    less than ``S.max() * max(M, N) * eps`` as indicating rank deficiency
    (with the symbols defined above). This is the algorithm MATLAB uses [1].
    It also appears in *Numerical recipes* in the discussion of SVD solutions
    for linear least squares [2].

    This default threshold is designed to detect rank deficiency accounting
    for the numerical errors of the SVD computation. Imagine that there
    is a column in `A` that is an exact (in floating point) linear combination
    of other columns in `A`. Computing the SVD on `A` will not produce
    a singular value exactly equal to 0 in general: any difference of
    the smallest SVD value from 0 will be caused by numerical imprecision
    in the calculation of the SVD. Our threshold for small SVD values takes
    this numerical imprecision into account, and the default threshold will
    detect such numerical rank deficiency. The threshold may declare a matrix
    `A` rank deficient even if the linear combination of some columns of `A`
    is not exactly equal to another column of `A` but only numerically very
    close to another column of `A`.

    We chose our default threshold because it is in wide use. Other thresholds
    are possible.  For example, elsewhere in the 2007 edition of *Numerical
    recipes* there is an alternative threshold of ``S.max() *
    np.finfo(A.dtype).eps / 2. * np.sqrt(m + n + 1.)``. The authors describe
    this threshold as being based on "expected roundoff error" (p 71).

    The thresholds above deal with floating point roundoff error in the
    calculation of the SVD.  However, you may have more information about
    the sources of error in `A` that would make you consider other tolerance
    values to detect *effective* rank deficiency. The most useful measure
    of the tolerance depends on the operations you intend to use on your
    matrix. For example, if your data come from uncertain measurements with
    uncertainties greater than floating point epsilon, choosing a tolerance
    near that uncertainty may be preferable. The tolerance may be absolute
    if the uncertainties are absolute rather than relative.

    References
    ----------
    .. [1] MATLAB reference documentation, "Rank"
           https://www.mathworks.com/help/techdoc/ref/rank.html
    .. [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
           "Numerical Recipes (3rd edition)", Cambridge University Press, 2007,
           page 795.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.linalg import matrix_rank
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0
    """
    if rtol is not None and tol is not None:
        raise ValueError("`tol` and `rtol` can't be both set.")

    A = asarray(A)
    if A.ndim < 2:
        return int(not all(A == 0))
    S = svd(A, compute_uv=False, hermitian=hermitian)

    if tol is None:
        if rtol is None:
            rtol = max(A.shape[-2:]) * finfo(S.dtype).eps
        else:
            rtol = asarray(rtol)[..., newaxis]
        tol = S.max(axis=-1, keepdims=True) * rtol
    else:
        tol = asarray(tol)[..., newaxis]

    return count_nonzero(S > tol, axis=-1)


# Generalized inverse

def _pinv_dispatcher(a, rcond=None, hermitian=None, *, rtol=None):
    return (a,)


@array_function_dispatch(_pinv_dispatcher)
def pinv(a, rcond=None, hermitian=False, *, rtol=_NoValue):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Parameters
    ----------
    a : (..., M, N) array_like
        Matrix or stack of matrices to be pseudo-inverted.
    rcond : (...) array_like of float, optional
        Cutoff for small singular values.
        Singular values less than or equal to
        ``rcond * largest_singular_value`` are set to zero.
        Broadcasts against the stack of matrices. Default: ``1e-15``.
    hermitian : bool, optional
        If True, `a` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.
    rtol : (...) array_like of float, optional
        Same as `rcond`, but it's an Array API compatible parameter name.
        Only `rcond` or `rtol` can be set at a time. If none of them are
        provided then NumPy's ``1e-15`` default is used. If ``rtol=None``
        is passed then the API standard default is used.

        .. versionadded:: 2.0.0

    Returns
    -------
    B : (..., N, M) ndarray
        The pseudo-inverse of `a`. If `a` is a `matrix` instance, then so
        is `B`.

    Raises
    ------
    LinAlgError
        If the SVD computation does not converge.

    See Also
    --------
    scipy.linalg.pinv : Similar function in SciPy.
    scipy.linalg.pinvh : Compute the (Moore-Penrose) pseudo-inverse of a
                         Hermitian matrix.

    Notes
    -----
    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
    value decomposition of A, then
    :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
    orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
    of A's so-called singular values, (followed, typically, by
    zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
    consisting of the reciprocals of A's singular values
    (again, followed by zeros). [1]_

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pp. 139-142.

    Examples
    --------
    The following example checks that ``a * a+ * a == a`` and
    ``a+ * a * a+ == a+``:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> a = rng.normal(size=(9, 6))
    >>> B = np.linalg.pinv(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    """
    a, wrap = _makearray(a)
    if rcond is None:
        if rtol is _NoValue:
            rcond = 1e-15
        elif rtol is None:
            rcond = max(a.shape[-2:]) * finfo(a.dtype).eps
        else:
            rcond = rtol
    elif rtol is not _NoValue:
        raise ValueError("`rtol` and `rcond` can't be both set.")
    else:
        # NOTE: Deprecate `rcond` in a few versions.
        pass

    rcond = asarray(rcond)
    if _is_empty_2d(a):
        m, n = a.shape[-2:]
        res = empty(a.shape[:-2] + (n, m), dtype=a.dtype)
        return wrap(res)
    a = a.conjugate()
    u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)

    # discard small singular values
    cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = divide(1, s, where=large, out=s)
    s[~large] = 0

    res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
    return wrap(res)


# Determinant


@array_function_dispatch(_unary_dispatcher)
def slogdet(a):
    """
    Compute the sign and (natural) logarithm of the determinant of an array.

    If an array has a very small or very large determinant, then a call to
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself.

    Parameters
    ----------
    a : (..., M, M) array_like
        Input array, has to be a square 2-D array.

    Returns
    -------
    A namedtuple with the following attributes:

    sign : (...) array_like
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1. For a complex matrix, this is a complex number
        with absolute value 1 (i.e., it is on the unit circle), or else 0.
    logabsdet : (...) array_like
        The natural log of the absolute value of the determinant.

    If the determinant is zero, then `sign` will be 0 and `logabsdet`
    will be -inf. In all cases, the determinant is equal to
    ``sign * np.exp(logabsdet)``.

    See Also
    --------
    det

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The determinant is computed via LU factorization using the LAPACK
    routine ``z/dgetrf``.

    Examples
    --------
    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:

    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logabsdet) = np.linalg.slogdet(a)
    >>> (sign, logabsdet)
    (-1, 0.69314718055994529) # may vary
    >>> sign * np.exp(logabsdet)
    -2.0

    Computing log-determinants for a stack of matrices:

    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)
    >>> sign, logabsdet = np.linalg.slogdet(a)
    >>> (sign, logabsdet)
    (array([-1., -1., -1.]), array([ 0.69314718,  1.09861229,  2.07944154]))
    >>> sign * np.exp(logabsdet)
    array([-2., -3., -8.])

    This routine succeeds where ordinary `det` does not:

    >>> np.linalg.det(np.eye(500) * 0.1)
    0.0
    >>> np.linalg.slogdet(np.eye(500) * 0.1)
    (1, -1151.2925464970228)

    """
    a = asarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    real_t = _realType(result_t)
    signature = 'D->Dd' if isComplexType(t) else 'd->dd'
    sign, logdet = _umath_linalg.slogdet(a, signature=signature)
    sign = sign.astype(result_t, copy=False)
    logdet = logdet.astype(real_t, copy=False)
    return SlogdetResult(sign, logdet)


@array_function_dispatch(_unary_dispatcher)
def det(a):
    """
    Compute the determinant of an array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Input array to compute determinants for.

    Returns
    -------
    det : (...) array_like
        Determinant of `a`.

    See Also
    --------
    slogdet : Another way to represent the determinant, more suitable
      for large matrices where underflow/overflow may occur.
    scipy.linalg.det : Similar function in SciPy.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The determinant is computed via LU factorization using the LAPACK
    routine ``z/dgetrf``.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.linalg.det(a)
    -2.0 # may vary

    Computing determinants for a stack of matrices:

