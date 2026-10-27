    >>> np.set_printoptions(precision=4)

    >>> np.emath.log10(10**1)
    1.0

    >>> np.emath.log10([-10**1, -10**2, 10**2])
    array([1.+1.3644j, 2.+1.3644j, 2.+0.j    ])

    """
    x = _fix_real_lt_zero(x)
    return nx.log10(x)


def _logn_dispatcher(n, x):
    return (n, x,)


@set_module('numpy.lib.scimath')
@array_function_dispatch(_logn_dispatcher)
def logn(n, x):
    """
    Take log base n of x.

    If `x` contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    n : array_like
       The integer base(s) in which the log is taken.
    x : array_like
       The value(s) whose log base `n` is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The log base `n` of the `x` value(s). If `x` was a scalar, so is
       `out`, otherwise an array is returned.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.logn(2, [4, 8])
    array([2., 3.])
    >>> np.emath.logn(2, [-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return nx.log(x)/nx.log(n)


@set_module('numpy.lib.scimath')
@array_function_dispatch(_unary_dispatcher)
def log2(x):
    """
    Compute the logarithm base 2 of `x`.

    Return the "principal value" (for a description of this, see
    `numpy.log2`) of :math:`log_2(x)`. For real `x > 0`, this is
    a real number (``log2(0)`` returns ``-inf`` and ``log2(np.inf)`` returns
    ``inf``). Otherwise, the complex principle value is returned.

    Parameters
    ----------
    x : array_like
       The value(s) whose log base 2 is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The log base 2 of the `x` value(s). If `x` was a scalar, so is `out`,
       otherwise an array is returned.

    See Also
    --------
    numpy.log2

    Notes
    -----
    For a log2() that returns ``NAN`` when real `x < 0`, use `numpy.log2`
    (note, however, that otherwise `numpy.log2` and this `log2` are
    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,
    and, notably, the complex principle value if ``x.imag != 0``).

    Examples
    --------

    We set the printing precision so the example can be auto-tested:

    >>> np.set_printoptions(precision=4)

    >>> np.emath.log2(8)
    3.0
    >>> np.emath.log2([-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    """
    x = _fix_real_lt_zero(x)
    return nx.log2(x)


def _power_dispatcher(x, p):
    return (x, p)


@set_module('numpy.lib.scimath')
@array_function_dispatch(_power_dispatcher)
def power(x, p):
    """
    Return x to the power p, (x**p).

    If `x` contains negative values, the output is converted to the
    complex domain.

    Parameters
    ----------
    x : array_like
        The input value(s).
    p : array_like of ints
        The power(s) to which `x` is raised. If `x` contains multiple values,
        `p` has to either be a scalar, or contain the same number of values
        as `x`. In the latter case, the result is
        ``x[0]**p[0], x[1]**p[1], ...``.

    Returns
    -------
    out : ndarray or scalar
        The result of ``x**p``. If `x` and `p` are scalars, so is `out`,
        otherwise an array is returned.

    See Also
    --------
    numpy.power

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.power(2, 2)
    4

    >>> np.emath.power([2, 4], 2)
    array([ 4, 16])

    >>> np.emath.power([2, 4], -2)
    array([0.25  ,  0.0625])

    >>> np.emath.power([-2, 4], 2)
    array([ 4.-0.j, 16.+0.j])

    >>> np.emath.power([2, 4], [2, 4])
    array([ 4, 256])

    """
    x = _fix_real_lt_zero(x)
    p = _fix_int_lt_zero(p)
    return nx.power(x, p)


@set_module('numpy.lib.scimath')
@array_function_dispatch(_unary_dispatcher)
def arccos(x):
    """
    Compute the inverse cosine of x.

    Return the "principal value" (for a description of this, see
    `numpy.arccos`) of the inverse cosine of `x`. For real `x` such that
    `abs(x) <= 1`, this is a real number in the closed interval
    :math:`[0, \\pi]`.  Otherwise, the complex principle value is returned.

    Parameters
    ----------
    x : array_like or scalar
       The value(s) whose arccos is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The inverse cosine(s) of the `x` value(s). If `x` was a scalar, so
       is `out`, otherwise an array object is returned.

    See Also
    --------
    numpy.arccos

    Notes
    -----
    For an arccos() that returns ``NAN`` when real `x` is not in the
    interval ``[-1,1]``, use `numpy.arccos`.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.arccos(1) # a scalar is returned
    0.0

    >>> np.emath.arccos([1,2])
    array([0.-0.j   , 0.-1.317j])

    """
    x = _fix_real_abs_gt_1(x)
    return nx.arccos(x)


@set_module('numpy.lib.scimath')
@array_function_dispatch(_unary_dispatcher)
def arcsin(x):
    """
    Compute the inverse sine of x.

    Return the "principal value" (for a description of this, see
    `numpy.arcsin`) of the inverse sine of `x`. For real `x` such that
    `abs(x) <= 1`, this is a real number in the closed interval
    :math:`[-\\pi/2, \\pi/2]`.  Otherwise, the complex principle value is
    returned.

    Parameters
    ----------
    x : array_like or scalar
       The value(s) whose arcsin is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The inverse sine(s) of the `x` value(s). If `x` was a scalar, so
       is `out`, otherwise an array object is returned.

    See Also
    --------
    numpy.arcsin

    Notes
    -----
    For an arcsin() that returns ``NAN`` when real `x` is not in the
    interval ``[-1,1]``, use `numpy.arcsin`.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.arcsin(0)
    0.0

    >>> np.emath.arcsin([0,1])
    array([0.    , 1.5708])

    """
    x = _fix_real_abs_gt_1(x)
    return nx.arcsin(x)


@set_module('numpy.lib.scimath')
@array_function_dispatch(_unary_dispatcher)
def arctanh(x):
    """
    Compute the inverse hyperbolic tangent of `x`.

    Return the "principal value" (for a description of this, see
    `numpy.arctanh`) of ``arctanh(x)``. For real `x` such that
    ``abs(x) < 1``, this is a real number.  If `abs(x) > 1`, or if `x` is
    complex, the result is complex. Finally, `x = 1` returns``inf`` and
    ``x=-1`` returns ``-inf``.

    Parameters
    ----------
    x : array_like
       The value(s) whose arctanh is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The inverse hyperbolic tangent(s) of the `x` value(s). If `x` was
       a scalar so is `out`, otherwise an array is returned.


    See Also
    --------
    numpy.arctanh

    Notes
    -----
    For an arctanh() that returns ``NAN`` when real `x` is not in the
    interval ``(-1,1)``, use `numpy.arctanh` (this latter, however, does
    return +/-inf for ``x = +/-1``).

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.arctanh(0.5)
    0.5493061443340549

    >>> from numpy.testing import suppress_warnings
    >>> with suppress_warnings() as sup:
    ...     sup.filter(RuntimeWarning)
    ...     np.emath.arctanh(np.eye(2))
    array([[inf,  0.],
           [ 0., inf]])
    >>> np.emath.arctanh([1j])
    array([0.+0.7854j])

    """
    x = _fix_real_abs_gt_1(x)
    return nx.arctanh(x)
<|endoftext|>"""
The arraypad module contains a group of functions to pad values onto the edges
of an n-dimensional array.

"""
import numpy as np
from numpy._core.overrides import array_function_dispatch
from numpy.lib._index_tricks_impl import ndindex


__all__ = ['pad']


###############################################################################
# Private utility functions.


def _round_if_needed(arr, dtype):
    """
    Rounds arr inplace if destination dtype is integer.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dtype : dtype
        The dtype of the destination array.
    """
    if np.issubdtype(dtype, np.integer):
        arr.round(out=arr)


def _slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> np._slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), (...,))
    """
    return (slice(None),) * axis + (sl,) + (...,)


def _view_roi(array, original_area_slice, axis):
    """
    Get a view of the current region of interest during iterative padding.

    When padding multiple dimensions iteratively corner values are
    unnecessarily overwritten multiple times. This function reduces the
    working area for the first dimensions so that corners are excluded.

    Parameters
    ----------
    array : ndarray
        The array with the region of interest.
    original_area_slice : tuple of slices
        Denotes the area with original values of the unpadded array.
    axis : int
        The currently padded dimension assuming that `axis` is padded before
        `axis` + 1.

    Returns
    -------
    roi : ndarray
        The region of interest of the original `array`.
    """
    axis += 1
    sl = (slice(None),) * axis + original_area_slice[axis:]
    return array[sl]


def _pad_simple(array, pad_width, fill_value=None):
    """
    Pad array on all sides with either a single value or undefined values.

    Parameters
    ----------
    array : ndarray
        Array to grow.
    pad_width : sequence of tuple[int, int]
        Pad width on both sides for each dimension in `arr`.
    fill_value : scalar, optional
        If provided the padded area is filled with this value, otherwise
        the pad area left undefined.

    Returns
    -------
    padded : ndarray
        The padded array with the same dtype as`array`. Its order will default
        to C-style if `array` is not F-contiguous.
    original_area_slice : tuple
        A tuple of slices pointing to the area of the original array.
    """
    # Allocate grown array
    new_shape = tuple(
        left + size + right
        for size, (left, right) in zip(array.shape, pad_width)
    )
    order = 'F' if array.flags.fnc else 'C'  # Fortran and not also C-order
    padded = np.empty(new_shape, dtype=array.dtype, order=order)

    if fill_value is not None:
        padded.fill(fill_value)

    # Copy old array into correct space
    original_area_slice = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(array.shape, pad_width)
    )
    padded[original_area_slice] = array

    return padded, original_area_slice


def _set_pad_area(padded, axis, width_pair, value_pair):
    """
    Set empty-padded area in given dimension.

    Parameters
    ----------
    padded : ndarray
        Array with the pad area which is modified inplace.
    axis : int
        Dimension with the pad area to set.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    value_pair : tuple of scalars or ndarrays
        Values inserted into the pad area on each side. It must match or be
        broadcastable to the shape of `arr`.
    """
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    padded[left_slice] = value_pair[0]

    right_slice = _slice_at_axis(
        slice(padded.shape[axis] - width_pair[1], None), axis)
    padded[right_slice] = value_pair[1]


def _get_edges(padded, axis, width_pair):
    """
    Retrieve edge values from empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the edges are considered.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.

    Returns
    -------
    left_edge, right_edge : ndarray
        Edge values of the valid area in `padded` in the given dimension. Its
        shape will always match `padded` except for the dimension given by
        `axis` which will have a length of 1.
    """
    left_index = width_pair[0]
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]

    right_index = padded.shape[axis] - width_pair[1]
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]

    return left_edge, right_edge


def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    """
    Construct linear ramps for empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the ramps are constructed.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    end_value_pair : (scalar, scalar)
        End values for the linear ramps which form the edge of the fully padded
        array. These values are included in the linear ramps.

    Returns
    -------
    left_ramp, right_ramp : ndarray
        Linear ramps to set on both sides of `padded`.
    """
    edge_pair = _get_edges(padded, axis, width_pair)

    left_ramp, right_ramp = (
        np.linspace(
            start=end_value,
            stop=edge.squeeze(axis), # Dimension is replaced by linspace
            num=width,
            endpoint=False,
            dtype=padded.dtype,
            axis=axis
        )
        for end_value, edge, width in zip(
            end_value_pair, edge_pair, width_pair
        )
    )

    # Reverse linear space in appropriate dimension
    right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]

    return left_ramp, right_ramp


def _get_stats(padded, axis, width_pair, length_pair, stat_func):
    """
    Calculate statistic for the empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the statistic is calculated.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    length_pair : 2-element sequence of None or int
        Gives the number of values in valid area from each side that is
        taken into account when calculating the statistic. If None the entire
        valid area in `padded` is considered.
    stat_func : function
        Function to compute statistic. The expected signature is
        ``stat_func(x: ndarray, axis: int, keepdims: bool) -> ndarray``.

    Returns
    -------
    left_stat, right_stat : ndarray
        Calculated statistic for both sides of `padded`.
    """
    # Calculate indices of the edges of the area with original values
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    # as well as its length
    max_length = right_index - left_index

    # Limit stat_lengths to max_length
    left_length, right_length = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length

    if (left_length == 0 or right_length == 0) \
            and stat_func in {np.amax, np.amin}:
        # amax and amin can't operate on an empty array,
        # raise a more descriptive warning here instead of the default one
        raise ValueError("stat_length of 0 yields no value for padding")

    # Calculate statistic for the left side
    left_slice = _slice_at_axis(
        slice(left_index, left_index + left_length), axis)
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    _round_if_needed(left_stat, padded.dtype)

    if left_length == right_length == max_length:
        # return early as right_stat must be identical to left_stat
        return left_stat, left_stat

    # Calculate statistic for the right side
    right_slice = _slice_at_axis(
        slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
    _round_if_needed(right_stat, padded.dtype)

    return left_stat, right_stat


def _set_reflect_both(padded, axis, width_pair, method,
                      original_period, include_edge=False):
    """
    Pad `axis` of `arr` with reflection.

    Parameters
    ----------
    padded : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    method : str
        Controls method of reflection; options are 'even' or 'odd'.
    original_period : int
        Original length of data on `axis` of `arr`.
    include_edge : bool
        If true, edge value is included in reflection, otherwise the edge
        value forms the symmetric axis to the reflection.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension.
    """
    left_pad, right_pad = width_pair
    old_length = padded.shape[axis] - right_pad - left_pad

    if include_edge:
        # Avoid wrapping with only a subset of the original area
        # by ensuring period can only be a multiple of the original
        # area's length.
        old_length = old_length // original_period * original_period
        # Edge is included, we need to offset the pad amount by 1
        edge_offset = 1
    else:
        # Avoid wrapping with only a subset of the original area
        # by ensuring period can only be a multiple of the original
        # area's length.
        old_length = ((old_length - 1) // (original_period - 1)
            * (original_period - 1) + 1)
        edge_offset = 0  # Edge is not included, no need to offset pad amount
        old_length -= 1  # but must be omitted from the chunk

    if left_pad > 0:
        # Pad with reflected values on left side:
        # First limit chunk size which can't be larger than pad area
        chunk_length = min(old_length, left_pad)
        # Slice right to left, stop on or next to edge, start relative to stop
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]

        if method == "odd":
            # Negate chunk and align with edge
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk

        # Insert chunk into padded area
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        # Adjust pointer to left edge for next iteration
        left_pad -= chunk_length

    if right_pad > 0:
        # Pad with reflected values on right side:
        # First limit chunk size which can't be larger than pad area
        chunk_length = min(old_length, right_pad)
        # Slice right to left, start on or next to edge, stop relative to start
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]

        if method == "odd":
            # Negate chunk and align with edge
            edge_slice = _slice_at_axis(
                slice(-right_pad - 1, -right_pad), axis)
            right_chunk = 2 * padded[edge_slice] - right_chunk

        # Insert chunk into padded area
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        # Adjust pointer to right edge for next iteration
        right_pad -= chunk_length

    return left_pad, right_pad


def _set_wrap_both(padded, axis, width_pair, original_period):
    """
    Pad `axis` of `arr` with wrapped values.

    Parameters
    ----------
    padded : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    original_period : int
        Original length of data on `axis` of `arr`.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension.
    """
    left_pad, right_pad = width_pair
    period = padded.shape[axis] - right_pad - left_pad
    # Avoid wrapping with only a subset of the original area by ensuring period
    # can only be a multiple of the original area's length.
    period = period // original_period * original_period

    # If the current dimension of `arr` doesn't contain enough valid values
    # (not part of the undefined pad area) we need to pad multiple times.
    # Each time the pad area shrinks on both sides which is communicated with
    # these variables.
    new_left_pad = 0
    new_right_pad = 0

    if left_pad > 0:
        # Pad with wrapped values on left side
        # First slice chunk from left side of the non-pad area.
        # Use min(period, left_pad) to ensure that chunk is not larger than
        # pad area.
        slice_end = left_pad + period
        slice_start = slice_end - min(period, left_pad)
        right_slice = _slice_at_axis(slice(slice_start, slice_end), axis)
        right_chunk = padded[right_slice]

        if left_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(slice(left_pad - period, left_pad), axis)
            new_left_pad = left_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(slice(None, left_pad), axis)
        padded[pad_area] = right_chunk

    if right_pad > 0:
        # Pad with wrapped values on right side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, right_pad) to ensure that chunk is not larger than
        # pad area.
        slice_start = -right_pad - period
        slice_end = slice_start + min(period, right_pad)
        left_slice = _slice_at_axis(slice(slice_start, slice_end), axis)
        left_chunk = padded[left_slice]

        if right_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                slice(-right_pad, -right_pad + period), axis)
            new_right_pad = right_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(slice(-right_pad, None), axis)
        padded[pad_area] = left_chunk

    return new_left_pad, new_right_pad


def _as_pairs(x, ndim, as_index=False):
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.

    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2).
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to an integer
        (dtype `np.intp`) and ensure every element is positive.

    Returns
    -------
    pairs : nested iterables, shape (`ndim`, 2)
        The broadcasted version of `x`.

    Raises
    ------
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape (`ndim`, 2).
    """
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `np.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


def _pad_dispatcher(array, pad_width, mode=None, **kwargs):
    return (array,)


###############################################################################
# Public functions


@array_function_dispatch(_pad_dispatcher, module='numpy')
def pad(array, pad_width, mode='constant', **kwargs):
    """
    Pad an array.

    Parameters
    ----------
    array : array_like of rank N
        The array to pad.
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width
        for all axes.
    mode : str or function, optional
        One of the following string values or a user supplied function.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        'empty'
            Pads with undefined values.

        <function>
            Padding function, see Notes.
    stat_length : sequence or int, optional
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ``((before_1, after_1), ... (before_N, after_N))`` unique statistic
        lengths for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after statistic lengths for each axis.

        ``(stat_length,)`` or ``int`` is a shortcut for
        ``before = after = statistic`` length for all axes.

        Default is ``None``, to use the entire axis.
    constant_values : sequence or scalar, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after constants for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.
    end_values : sequence or scalar, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ``((before_1, after_1), ... (before_N, after_N))`` unique end values
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after end values for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.
    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

    Notes
    -----
    For an array with rank greater than 1, some of the padding of later
    axes is calculated from padding of previous axes.  This is easiest to
    think about with a rank 2 array where the corners of the padded array
    are calculated by using padded values from the first axis.

    The padding function, if used, should modify a rank 1 array in-place. It
    has the following signature::

        padding_func(vector, iaxis_pad_width, iaxis, kwargs)

    where

    vector : ndarray
        A rank 1 array already padded with zeros.  Padded values are
        vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:].
    iaxis_pad_width : tuple
        A 2-tuple of ints, iaxis_pad_width[0] represents the number of
        values padded at the beginning of vector where
        iaxis_pad_width[1] represents the number of values padded at
        the end of vector.
    iaxis : int
        The axis currently being calculated.
    kwargs : dict
        Any keyword arguments the function requires.

    Examples
    --------
    >>> import numpy as np
    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
    array([4, 4, 1, ..., 6, 6, 6])

    >>> np.pad(a, (2, 3), 'edge')
    array([1, 1, 1, ..., 5, 5, 5])

    >>> np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    >>> np.pad(a, (2,), 'maximum')
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> np.pad(a, (2,), 'mean')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> np.pad(a, (2,), 'median')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> a = [[1, 2], [3, 4]]
    >>> np.pad(a, ((3, 2), (2, 3)), 'minimum')
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'reflect')
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> np.pad(a, (2, 3), 'symmetric')
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> np.pad(a, (2, 3), 'wrap')
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    >>> def pad_with(vector, pad_width, iaxis, kwargs):
    ...     pad_value = kwargs.get('padder', 10)
    ...     vector[:pad_width[0]] = pad_value
    ...     vector[-pad_width[1]:] = pad_value
    >>> a = np.arange(6)
    >>> a = a.reshape((2, 3))
    >>> np.pad(a, 2, pad_with)
    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10,  0,  1,  2, 10, 10],
           [10, 10,  3,  4,  5, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])
    >>> np.pad(a, 2, pad_with, padder=100)
    array([[100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100,   0,   1,   2, 100, 100],
           [100, 100,   3,   4,   5, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100]])
    """
    array = np.asarray(array)
    pad_width = np.asarray(pad_width)

    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    # Broadcast to shape (array.ndim, 2)
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    if callable(mode):
        # Old behavior: Use user-supplied function with np.apply_along_axis
        function = mode
        # Create a new zero padded array
        padded, _ = _pad_simple(array, pad_width, fill_value=0)
        # And apply along each axis

        for axis in range(padded.ndim):
            # Iterate using ndindex as in apply_along_axis, but assuming that
            # function operates inplace on the padded array.

            # view with the iteration axis at the end
            view = np.moveaxis(padded, axis, -1)

            # compute indices for the iteration axes, and append a trailing
            # ellipsis to prevent 0d arrays decaying to scalars (gh-8642)
            inds = ndindex(view.shape[:-1])
            inds = (ind + (Ellipsis,) for ind in inds)
            for ind in inds:
                function(view[ind], pad_width[axis], axis, kwargs)

        return padded

    # Make sure that no unsupported keywords were passed for the current mode
    allowed_kwargs = {
        'empty': [], 'edge': [], 'wrap': [],
        'constant': ['constant_values'],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
    }
    try:
        unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    except KeyError:
        raise ValueError("mode '{}' is not supported".format(mode)) from None
    if unsupported_kwargs:
        raise ValueError("unsupported keyword arguments for mode '{}': {}"
                         .format(mode, unsupported_kwargs))

    stat_functions = {"maximum": np.amax, "minimum": np.amin,
                      "mean": np.mean, "median": np.median}

    # Create array with final shape and original values
    # (padded area is undefined)
    padded, original_area_slice = _pad_simple(array, pad_width)
    # And prepare iteration over all dimensions
    # (zipping may be more readable than using enumerate)
    axes = range(padded.ndim)

    if mode == "constant":
        values = kwargs.get("constant_values", 0)
        values = _as_pairs(values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, values):
            roi = _view_roi(padded, original_area_slice, axis)
            _set_pad_area(roi, axis, width_pair, value_pair)

    elif mode == "empty":
        pass  # Do nothing as _pad_simple already returned the correct result

    elif array.size == 0:
        # Only modes "constant" and "empty" can extend empty axes, all other
        # modes depend on `array` not being empty
        # -> ensure every empty axis is only "padded with 0"
        for axis, width_pair in zip(axes, pad_width):
            if array.shape[axis] == 0 and any(width_pair):
                raise ValueError(
                    "can't extend empty axis {} using modes other than "
                    "'constant' or 'empty'".format(axis)
                )
        # passed, don't need to do anything more as _pad_simple already
        # returned the correct result

    elif mode == "edge":
        for axis, width_pair in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            edge_pair = _get_edges(roi, axis, width_pair)
            _set_pad_area(roi, axis, width_pair, edge_pair)

    elif mode == "linear_ramp":
        end_values = kwargs.get("end_values", 0)
        end_values = _as_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, end_values):
            roi = _view_roi(padded, original_area_slice, axis)
            ramp_pair = _get_linear_ramps(roi, axis, width_pair, value_pair)
            _set_pad_area(roi, axis, width_pair, ramp_pair)

    elif mode in stat_functions:
        func = stat_functions[mode]
        length = kwargs.get("stat_length", None)
        length = _as_pairs(length, padded.ndim, as_index=True)
        for axis, width_pair, length_pair in zip(axes, pad_width, length):
            roi = _view_roi(padded, original_area_slice, axis)
            stat_pair = _get_stats(roi, axis, width_pair, length_pair, func)
            _set_pad_area(roi, axis, width_pair, stat_pair)

    elif mode in {"reflect", "symmetric"}:
        method = kwargs.get("reflect_type", "even")
        include_edge = mode == "symmetric"
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                _set_pad_area(
                    padded, axis, (left_index, right_index), edge_pair)
                continue

            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_reflect_both(
                    roi, axis, (left_index, right_index),
                    method, array.shape[axis], include_edge
                )

    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            original_period = padded.shape[axis] - right_index - left_index
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_wrap_both(
                    roi, axis, (left_index, right_index), original_period)

    return padded
<|endoftext|>"""Utility to compare (NumPy) version strings.

The NumpyVersion class allows properly comparing numpy version strings.
The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.

"""
import re


__all__ = ['NumpyVersion']


class NumpyVersion:
    """Parse and compare numpy version strings.

    NumPy has the following versioning scheme (numbers given are examples; they
    can be > 9 in principle):

    - Released version: '1.8.0', '1.8.1', etc.
    - Alpha: '1.8.0a1', '1.8.0a2', etc.
    - Beta: '1.8.0b1', '1.8.0b2', etc.
    - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
    - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
    - Development versions after a1: '1.8.0a1.dev-f1234afa',
                                     '1.8.0b2.dev-f1234afa',
                                     '1.8.1rc1.dev-f1234afa', etc.
    - Development versions (no git hash available): '1.8.0.dev-Unknown'

    Comparing needs to be done against a valid version string or other
    `NumpyVersion` instance. Note that all development versions of the same
    (pre-)release compare equal.

    Parameters
    ----------
    vstring : str
        NumPy version string (``np.__version__``).

    Examples
    --------
    >>> from numpy.lib import NumpyVersion
    >>> if NumpyVersion(np.__version__) < '1.7.0':
    ...     print('skip')
    >>> # skip

    >>> NumpyVersion('1.7')  # raises ValueError, add ".0"
    Traceback (most recent call last):
        ...
    ValueError: Not a valid numpy version string

    """

    __module__ = "numpy.lib"

    def __init__(self, vstring):
        self.vstring = vstring
        ver_main = re.match(r'\d+\.\d+\.\d+', vstring)
        if not ver_main:
            raise ValueError("Not a valid numpy version string")

        self.version = ver_main.group()
        self.major, self.minor, self.bugfix = [int(x) for x in
            self.version.split('.')]
        if len(vstring) == ver_main.end():
            self.pre_release = 'final'
        else:
            alpha = re.match(r'a\d', vstring[ver_main.end():])
            beta = re.match(r'b\d', vstring[ver_main.end():])
            rc = re.match(r'rc\d', vstring[ver_main.end():])
            pre_rel = [m for m in [alpha, beta, rc] if m is not None]
            if pre_rel:
                self.pre_release = pre_rel[0].group()
            else:
                self.pre_release = ''

        self.is_devversion = bool(re.search(r'.dev', vstring))

    def _compare_version(self, other):
        """Compare major.minor.bugfix"""
        if self.major == other.major:
            if self.minor == other.minor:
                if self.bugfix == other.bugfix:
                    vercmp = 0
                elif self.bugfix > other.bugfix:
                    vercmp = 1
                else:
                    vercmp = -1
            elif self.minor > other.minor:
                vercmp = 1
            else:
                vercmp = -1
        elif self.major > other.major:
            vercmp = 1
        else:
            vercmp = -1

        return vercmp

    def _compare_pre_release(self, other):
        """Compare alpha/beta/rc/final."""
        if self.pre_release == other.pre_release:
            vercmp = 0
        elif self.pre_release == 'final':
            vercmp = 1
        elif other.pre_release == 'final':
            vercmp = -1
        elif self.pre_release > other.pre_release:
            vercmp = 1
        else:
            vercmp = -1

        return vercmp

    def _compare(self, other):
        if not isinstance(other, (str, NumpyVersion)):
            raise ValueError("Invalid object to compare with NumpyVersion.")

        if isinstance(other, str):
            other = NumpyVersion(other)

        vercmp = self._compare_version(other)
        if vercmp == 0:
            # Same x.y.z version, check for alpha/beta/rc
            vercmp = self._compare_pre_release(other)
            if vercmp == 0:
                # Same version and same pre-release, check if dev version
                if self.is_devversion is other.is_devversion:
                    vercmp = 0
                elif self.is_devversion:
                    vercmp = -1
                else:
                    vercmp = 1

        return vercmp

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __eq__(self, other):
        return self._compare(other) == 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __repr__(self):
        return "NumpyVersion(%s)" % self.vstring
<|endoftext|>"""A collection of functions designed to help I/O with ascii files.

"""
__docformat__ = "restructuredtext en"

import numpy as np
import numpy._core.numeric as nx
from numpy._utils import asbytes, asunicode


def _decode_line(line, encoding=None):
    """Decode bytes from binary input streams.

    Defaults to decoding from 'latin1'. That differs from the behavior of
    np.compat.asunicode that decodes from 'ascii'.

    Parameters
    ----------
    line : str or bytes
         Line to be decoded.
    encoding : str
         Encoding used to decode `line`.

    Returns
    -------
    decoded_line : str

    """
    if type(line) is bytes:
        if encoding is None:
            encoding = "latin1"
        line = line.decode(encoding)

    return line


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def _is_bytes_like(obj):
    """
    Check whether obj behaves like a bytes object.
    """
    try:
        obj + b''
    except (TypeError, ValueError):
        return False
    return True


def has_nested_fields(ndtype):
    """
    Returns whether one or several fields of a dtype are nested.

    Parameters
    ----------
    ndtype : dtype
        Data-type of a structured array.

    Raises
    ------
    AttributeError
        If `ndtype` does not have a `names` attribute.

    Examples
    --------
    >>> import numpy as np
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])
    >>> np.lib._iotools.has_nested_fields(dt)
    False

    """
    return any(ndtype[name].names is not None for name in ndtype.names or ())


def flatten_dtype(ndtype, flatten_base=False):
    """
    Unpack a structured data-type by collapsing nested fields and/or fields
    with a shape.

    Note that the field names are lost.

    Parameters
    ----------
    ndtype : dtype
        The datatype to collapse
    flatten_base : bool, optional
       If True, transform a field with a shape into several fields. Default is
       False.

    Examples
    --------
    >>> import numpy as np
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
    ...                ('block', int, (2, 3))])
    >>> np.lib._iotools.flatten_dtype(dt)
    [dtype('S4'), dtype('float64'), dtype('float64'), dtype('int64')]
    >>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)
    [dtype('S4'),
     dtype('float64'),
     dtype('float64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64')]

    """
    names = ndtype.names
    if names is None:
        if flatten_base:
            return [ndtype.base] * int(np.prod(ndtype.shape))
        return [ndtype.base]
    else:
        types = []
        for field in names:
            info = ndtype.fields[field]
            flat_dt = flatten_dtype(info[0], flatten_base)
            types.extend(flat_dt)
        return types


class LineSplitter:
    """
    Object to split a string at a given delimiter or at given places.

    Parameters
    ----------
    delimiter : str, int, or sequence of ints, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    comments : str, optional
        Character used to mark the beginning of a comment. Default is '#'.
    autostrip : bool, optional
        Whether to strip each individual field. Default is True.

    """

    def autostrip(self, method):
        """
        Wrapper to strip each member of the output of `method`.

        Parameters
        ----------
        method : function
            Function that takes a single argument and returns a sequence of
            strings.

        Returns
        -------
        wrapped : function
            The result of wrapping `method`. `wrapped` takes a single input
            argument and returns a list of strings that are stripped of
            white-space.

        """
        return lambda input: [_.strip() for _ in method(input)]

    def __init__(self, delimiter=None, comments='#', autostrip=True,
                 encoding=None):
        delimiter = _decode_line(delimiter)
        comments = _decode_line(comments)

        self.comments = comments

        # Delimiter is a character
        if (delimiter is None) or isinstance(delimiter, str):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0] + list(delimiter))
            delimiter = [slice(i, j) for (i, j) in zip(idx[:-1], idx[1:])]
        # Delimiter is a single integer
        elif int(delimiter):
            (_handyman, delimiter) = (
                    self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
        self.encoding = encoding

    def _delimited_splitter(self, line):
        """Chop off comments, strip, and split at delimiter. """
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip(" \r\n")
        if not line:
            return []
        return line.split(self.delimiter)

    def _fixedwidth_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip("\r\n")
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
        return [line[s] for s in slices]

    def _variablewidth_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]

    def __call__(self, line):
        return self._handyman(_decode_line(line, self.encoding))


class NameValidator:
    """
    Object to validate a list of strings to use as field names.

    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by '_'. During instantiation, the user can define a list
    of names to exclude, as well as a list of invalid characters. Names in
    the exclusion list are appended a '_' character.

    Once an instance has been created, it can be called with a list of
    names, and a list of valid names will be created.  The `__call__`
    method accepts an optional keyword "default" that sets the default name
    in case of ambiguity. By default this is 'f', so that names will
    default to `f0`, `f1`, etc.

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default
        list ['return', 'file', 'print']. Excluded names are appended an
        underscore: for example, `file` becomes `file_` if supplied.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        * If True, field names are case-sensitive.
        * If False or 'upper', field names are converted to upper case.
        * If 'lower', field names are converted to lower case.

        The default value is True.
    replace_space : '_', optional
        Character(s) used in replacement of white spaces.

    Notes
    -----
    Calling an instance of `NameValidator` is the same as calling its
    method `validate`.

    Examples
    --------
    >>> import numpy as np
    >>> validator = np.lib._iotools.NameValidator()
    >>> validator(['file', 'field2', 'with space', 'CaSe'])
    ('file_', 'field2', 'with_space', 'CaSe')

    >>> validator = np.lib._iotools.NameValidator(excludelist=['excl'],
    ...                                           deletechars='q',
    ...                                           case_sensitive=False)
    >>> validator(['excl', 'field2', 'no_q', 'with space', 'CaSe'])
    ('EXCL', 'FIELD2', 'NO_Q', 'WITH_SPACE', 'CASE')

    """

    defaultexcludelist = ['return', 'file', 'print']
    defaultdeletechars = set(r"""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")

    def __init__(self, excludelist=None, deletechars=None,
                 case_sensitive=None, replace_space='_'):
        # Process the exclusion list ..
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        # Process the list of characters to delete
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        # Process the case option .....
        if (case_sensitive is None) or (case_sensitive is True):
            self.case_converter = lambda x: x
        elif (case_sensitive is False) or case_sensitive.startswith('u'):
            self.case_converter = lambda x: x.upper()
        elif case_sensitive.startswith('l'):
            self.case_converter = lambda x: x.lower()
        else:
            msg = 'unrecognized case_sensitive value %s.' % case_sensitive
            raise ValueError(msg)

        self.replace_space = replace_space

    def validate(self, names, defaultfmt="f%i", nbfields=None):
        """
        Validate a list of strings as field names for a structured array.

        Parameters
        ----------
        names : sequence of str
            Strings to be validated.
        defaultfmt : str, optional
            Default format string, used if validating a given string
            reduces its length to zero.
        nbfields : integer, optional
            Final number of validated names, used to expand or shrink the
            initial list of names.

        Returns
        -------
        validatednames : list of str
            The list of validated field names.

        Notes
        -----
        A `NameValidator` instance can be called directly, which is the
        same as calling `validate`. For examples, see `NameValidator`.

        """
        # Initial checks ..............
        if (names is None):
            if (nbfields is None):
                return None
            names = []
        if isinstance(names, str):
            names = [names, ]
        if nbfields is not None:
            nbnames = len(names)
            if (nbnames < nbfields):
                names = list(names) + [''] * (nbfields - nbnames)
            elif (nbnames > nbfields):
                names = names[:nbfields]
        # Set some shortcuts ...........
        deletechars = self.deletechars
        excludelist = self.excludelist
        case_converter = self.case_converter
        replace_space = self.replace_space
        # Initializes some variables ...
        validatednames = []
        seen = dict()
        nbempty = 0

        for item in names:
            item = case_converter(item).strip()
            if replace_space:
                item = item.replace(' ', replace_space)
            item = ''.join([c for c in item if c not in deletechars])
            if item == '':
                item = defaultfmt % nbempty
                while item in names:
                    nbempty += 1
                    item = defaultfmt % nbempty
                nbempty += 1
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt + 1
        return tuple(validatednames)

    def __call__(self, names, defaultfmt="f%i", nbfields=None):
        return self.validate(names, defaultfmt=defaultfmt, nbfields=nbfields)


def str2bool(value):
    """
    Tries to transform a string supposed to represent a boolean to a boolean.

    Parameters
    ----------
    value : str
        The string that is transformed to a boolean.

    Returns
    -------
    boolval : bool
        The boolean representation of `value`.

    Raises
    ------
    ValueError
        If the string is not 'True' or 'False' (case independent)

    Examples
    --------
    >>> import numpy as np
    >>> np.lib._iotools.str2bool('TRUE')
    True
    >>> np.lib._iotools.str2bool('false')
    False

    """
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError("Invalid boolean")


class ConverterError(Exception):
    """
    Exception raised when an error occurs in a converter for string values.

    """
    pass


class ConverterLockError(ConverterError):
    """
    Exception raised when an attempt is made to upgrade a locked converter.

    """
    pass


class ConversionWarning(UserWarning):
    """
    Warning issued when a string converter has a problem.

    Notes
    -----
    In `genfromtxt` a `ConversionWarning` is issued if raising exceptions
    is explicitly suppressed with the "invalid_raise" keyword.

    """
    pass


class StringConverter:
    """
    Factory class for function transforming a string into another object
    (int, float).

    After initialization, an instance can be called to transform a string
    into another object. If the string is recognized as representing a
    missing value, a default value is returned.

    Attributes
    ----------
    func : function
        Function used for the conversion.
    default : any
        Default value to return when the input corresponds to a missing
        value.
    type : type
        Type of the output.
    _status : int
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in
        order.
    _locked : bool
        Holds `locked` parameter.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        If a `dtype`, specifies the input data type, used to define a basic
        function and a default value for missing data. For example, when
        `dtype` is float, the `func` attribute is set to `float` and the
        default value to `np.nan`.  If a function, this function is used to
        convert a string to another object. In this case, it is recommended
        to give an associated default value as input.
    default : any, optional
        Value to return by default, that is, when the string to be
        converted is flagged as missing. If not given, `StringConverter`
        tries to supply a reasonable default value.
    missing_values : {None, sequence of str}, optional
        ``None`` or sequence of strings indicating a missing value. If ``None``
        then missing values are indicated by empty entries. The default is
        ``None``.
    locked : bool, optional
        Whether the StringConverter should be locked to prevent automatic
        upgrade or not. Default is False.

    """
    _mapper = [(nx.bool, str2bool, False),
               (nx.int_, int, -1),]

    # On 32-bit systems, we need to make sure that we explicitly include
    # nx.int64 since ns.int_ is nx.int32.
    if nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize:
        _mapper.append((nx.int64, int, -1))

    _mapper.extend([(nx.float64, float, nx.nan),
                    (nx.complex128, complex, nx.nan + 0j),
                    (nx.longdouble, nx.longdouble, nx.nan),
                    # If a non-default dtype is passed, fall back to generic
                    # ones (should only be used for the converter)
                    (nx.integer, int, -1),
                    (nx.floating, float, nx.nan),
                    (nx.complexfloating, complex, nx.nan + 0j),
                    # Last, try with the string types (must be last, because
                    # `_mapper[-1]` is used as default in some cases)
                    (nx.str_, asunicode, '???'),
                    (nx.bytes_, asbytes, '???'),
                    ])

    @classmethod
    def _getdtype(cls, val):
        """Returns the dtype of the input variable."""
        return np.array(val).dtype

    @classmethod
    def _getsubdtype(cls, val):
        """Returns the type of the dtype of the input variable."""
        return np.array(val).dtype.type

    @classmethod
    def _dtypeortype(cls, dtype):
        """Returns dtype for datetime64 and type of dtype otherwise."""

        # This is a bit annoying. We want to return the "general" type in most
        # cases (ie. "string" rather than "S10"), but we want to return the
        # specific type for datetime64 (ie. "datetime64[us]" rather than
        # "datetime64").
        if dtype.type == np.datetime64:
            return dtype
        return dtype.type

    @classmethod
    def upgrade_mapper(cls, func, default=None):
        """
        Upgrade the mapper of a StringConverter by adding a new function and
        its corresponding default.

        The input function (or sequence of functions) and its associated
        default value (if any) is inserted in penultimate position of the
        mapper.  The corresponding type is estimated from the dtype of the
        default value.

        Parameters
        ----------
        func : var
            Function, or sequence of functions

        Examples
        --------
        >>> import dateutil.parser
        >>> import datetime
        >>> dateparser = dateutil.parser.parse
        >>> defaultdate = datetime.date(2000, 1, 1)
        >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
        """
        # Func is a single functions
        if callable(func):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func) - len(default)))
            for fct, dft in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))

    @classmethod
    def _find_map_entry(cls, dtype):
        # if a converter for the specific dtype is available use that
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if dtype.type == deftype:
                return i, (deftype, func, default_def)

        # otherwise find an inexact match
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if np.issubdtype(dtype.type, deftype):
                return i, (deftype, func, default_def)

        raise LookupError

    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        # Defines a lock for upgrade
        self._locked = bool(locked)
        # No input dtype: minimal initialization
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default or False
            dtype = np.dtype('bool')
        else:
            # Is the input a np.dtype ?
            try:
                self.func = None
                dtype = np.dtype(dtype_or_func)
            except TypeError:
                # dtype_or_func must be a function, then
                if not callable(dtype_or_func):
                    errmsg = ("The input argument `dtype` is neither a"
                              " function nor a dtype (got '%s' instead)")
                    raise TypeError(errmsg % type(dtype_or_func))
                # Set the function
                self.func = dtype_or_func
                # If we don't have a default, try to guess it or set it to
                # None
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                dtype = self._getdtype(default)

            # find the best match in our mapper
            try:
                self._status, (_, func, default_def) = self._find_map_entry(dtype)
            except LookupError:
                # no match
                self.default = default
                _, func, _ = self._mapper[-1]
                self._status = 0
            else:
                # use the found default only if we did not already have one
                if default is None:
                    self.default = default_def
                else:
                    self.default = default

            # If the input was a dtype, set the function to the last we saw
            if self.func is None:
                self.func = func

            # If the status is 1 (int), change the function to
            # something more robust.
            if self.func == self._mapper[1][1]:
                if issubclass(dtype.type, np.uint64):
                    self.func = np.uint64
                elif issubclass(dtype.type, np.int64):
                    self.func = np.int64
                else:
                    self.func = lambda x: int(float(x))
        # Store the list of strings corresponding to missing values.
        if missing_values is None:
            self.missing_values = {''}
        else:
            if isinstance(missing_values, str):
                missing_values = missing_values.split(",")
            self.missing_values = set(list(missing_values) + [''])

        self._callingfunction = self._strict_call
        self.type = self._dtypeortype(dtype)
        self._checked = False
        self._initial_default = default

    def _loose_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            return self.default

    def _strict_call(self, value):
        try:

            # We check if we can convert the value using the current function
            new_value = self.func(value)

            # In addition to having to check whether func can convert the
            # value, we also have to make sure that we don't get overflow
            # errors for integers.
            if self.func is int:
                try:
                    np.array(value, dtype=self.type)
                except OverflowError:
                    raise ValueError

            # We're still here so we can now return the new value
            return new_value

        except ValueError:
            if value.strip() in self.missing_values:
                if not self._status:
                    self._checked = False
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)

    def __call__(self, value):
        return self._callingfunction(value)

    def _do_upgrade(self):
        # Raise an exception if we locked the converter...
        if self._locked:
            errmsg = "Converter is locked and cannot be upgraded"
            raise ConverterLockError(errmsg)
        _statusmax = len(self._mapper)
        # Complains if we try to upgrade by the maximum
        _status = self._status
        if _status == _statusmax:
            errmsg = "Could not find a valid conversion function"
            raise ConverterError(errmsg)
        elif _status < _statusmax - 1:
            _status += 1
        self.type, self.func, default = self._mapper[_status]
        self._status = _status
        if self._initial_default is not None:
            self.default = self._initial_default
        else:
            self.default = default

    def upgrade(self, value):
        """
        Find the best converter for a given string, and return the result.

        The supplied string `value` is converted by testing different
        converters in order. First the `func` method of the
        `StringConverter` instance is tried, if this fails other available
        converters are tried.  The order in which these other converters
        are tried is determined by the `_status` attribute of the instance.

        Parameters
        ----------
        value : str
            The string to convert.

        Returns
        -------
        out : any
            The result of converting `value` with the appropriate converter.

        """
        self._checked = True
        try:
            return self._strict_call(value)
        except ValueError:
            self._do_upgrade()
            return self.upgrade(value)

    def iterupgrade(self, value):
        self._checked = True
        if not hasattr(value, '__iter__'):
            value = (value,)
        _strict_call = self._strict_call
        try:
            for _m in value:
                _strict_call(_m)
        except ValueError:
            self._do_upgrade()
            self.iterupgrade(value)

    def update(self, func, default=None, testing_value=None,
               missing_values='', locked=False):
        """
        Set StringConverter attributes directly.

        Parameters
        ----------
        func : function
            Conversion function.
        default : any, optional
            Value to return by default, that is, when the string to be
            converted is flagged as missing. If not given,
            `StringConverter` tries to supply a reasonable default value.
        testing_value : str, optional
            A string representing a standard input value of the converter.
            This string is used to help defining a reasonable default
            value.
        missing_values : {sequence of str, None}, optional
            Sequence of strings indicating a missing value. If ``None``, then
            the existing `missing_values` are cleared. The default is ``''``.
        locked : bool, optional
            Whether the StringConverter should be locked to prevent
            automatic upgrade or not. Default is False.

        Notes
        -----
        `update` takes the same parameters as the constructor of
        `StringConverter`, except that `func` does not accept a `dtype`
        whereas `dtype_or_func` in the constructor does.

        """
        self.func = func
        self._locked = locked

        # Don't reset the default to None if we can avoid it
        if default is not None:
            self.default = default
            self.type = self._dtypeortype(self._getdtype(default))
        else:
            try:
                tester = func(testing_value or '1')
            except (TypeError, ValueError):
                tester = None
            self.type = self._dtypeortype(self._getdtype(tester))

        # Add the missing values to the existing set or clear it.
        if missing_values is None:
            # Clear all missing values even though the ctor initializes it to
            # set(['']) when the argument is None.
            self.missing_values = set()
        else:
            if not np.iterable(missing_values):
                missing_values = [missing_values]
            if not all(isinstance(v, str) for v in missing_values):
                raise TypeError("missing_values must be strings or unicode")
            self.missing_values.update(missing_values)


def easy_dtype(ndtype, names=None, defaultfmt="f%i", **validationargs):
    """
    Convenience function to create a `np.dtype` object.

    The function processes the input `dtype` and matches it with the given
    names.

    Parameters
    ----------
    ndtype : var
        Definition of the dtype. Can be any string or dictionary recognized
        by the `np.dtype` function, or a sequence of types.
    names : str or sequence, optional
        Sequence of strings to use as field names for a structured dtype.
        For convenience, `names` can be a string of a comma-separated list
        of names.
    defaultfmt : str, optional
        Format string used to define missing names, such as ``"f%i"``
        (default) or ``"fields_%02i"``.
    validationargs : optional
        A series of optional arguments used to initialize a
        `NameValidator`.

    Examples
    --------
    >>> import numpy as np
    >>> np.lib._iotools.easy_dtype(float)
    dtype('float64')
    >>> np.lib._iotools.easy_dtype("i4, f8")
    dtype([('f0', '<i4'), ('f1', '<f8')])
    >>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")
    dtype([('field_000', '<i4'), ('field_001', '<f8')])

    >>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")
    dtype([('a', '<i8'), ('b', '<f8'), ('c', '<f8')])
    >>> np.lib._iotools.easy_dtype(float, names="a,b,c")
    dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

    """
    try:
        ndtype = np.dtype(ndtype)
    except TypeError:
        validate = NameValidator(**validationargs)
        nbfields = len(ndtype)
        if names is None:
            names = [''] * len(ndtype)
        elif isinstance(names, str):
            names = names.split(",")
        names = validate(names, nbfields=nbfields, defaultfmt=defaultfmt)
        ndtype = np.dtype(dict(formats=ndtype, names=names))
    else:
        # Explicit names
        if names is not None:
            validate = NameValidator(**validationargs)
            if isinstance(names, str):
                names = names.split(",")
            # Simple dtype: repeat to match the nb of names
            if ndtype.names is None:
                formats = tuple([ndtype.type] * len(names))
                names = validate(names, defaultfmt=defaultfmt)
                ndtype = np.dtype(list(zip(names, formats)))
            # Structured dtype: just validate the names as needed
            else:
                ndtype.names = validate(names, nbfields=len(ndtype.names),
                                        defaultfmt=defaultfmt)
        # No implicit names
        elif ndtype.names is not None:
            validate = NameValidator(**validationargs)
            # Default initial names : should we change the format ?
            numbered_names = tuple("f%i" % i for i in range(len(ndtype.names)))
            if ((ndtype.names == numbered_names) and (defaultfmt != "f%i")):
                ndtype.names = validate([''] * len(ndtype.names),
                                        defaultfmt=defaultfmt)
            # Explicit initial names : just validate
            else:
                ndtype.names = validate(ndtype.names, defaultfmt=defaultfmt)
    return ndtype
<|endoftext|>"""
Mixin classes for custom array types that don't inherit from ndarray.
"""
from numpy._core import umath as um


__all__ = ['NDArrayOperatorsMixin']


def _disables_array_ufunc(obj):
    """True when __array_ufunc__ is set to None."""
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False


def _binary_method(ufunc, name):
    """Implement a forward binary method with a ufunc, e.g., __add__."""
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(self, other)
    func.__name__ = '__{}__'.format(name)
    return func


def _reflected_binary_method(ufunc, name):
    """Implement a reflected binary method with a ufunc, e.g., __radd__."""
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(other, self)
    func.__name__ = '__r{}__'.format(name)
    return func


def _inplace_binary_method(ufunc, name):
    """Implement an in-place binary method with a ufunc, e.g., __iadd__."""
    def func(self, other):
        return ufunc(self, other, out=(self,))
    func.__name__ = '__i{}__'.format(name)
    return func


def _numeric_methods(ufunc, name):
    """Implement forward, reflected and inplace binary methods with a ufunc."""
    return (_binary_method(ufunc, name),
            _reflected_binary_method(ufunc, name),
            _inplace_binary_method(ufunc, name))


def _unary_method(ufunc, name):
    """Implement a unary special method with a ufunc."""
    def func(self):
        return ufunc(self)
    func.__name__ = '__{}__'.format(name)
    return func


class NDArrayOperatorsMixin:
    """Mixin defining all operator special methods using __array_ufunc__.

    This class implements the special methods for almost all of Python's
    builtin operators defined in the `operator` module, including comparisons
    (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by
    deferring to the ``__array_ufunc__`` method, which subclasses must
    implement.

    It is useful for writing classes that do not inherit from `numpy.ndarray`,
    but that should support arithmetic and numpy universal functions like
    arrays as described in `A Mechanism for Overriding Ufuncs
    <https://numpy.org/neps/nep-0013-ufunc-overrides.html>`_.

    As an trivial example, consider this implementation of an ``ArrayLike``
    class that simply wraps a NumPy array and ensures that the result of any
    arithmetic operation is also an ``ArrayLike`` object:

        >>> import numbers
        >>> class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
        ...     def __init__(self, value):
        ...         self.value = np.asarray(value)
        ...
        ...     # One might also consider adding the built-in list type to this
        ...     # list, to support operations like np.add(array_like, list)
        ...     _HANDLED_TYPES = (np.ndarray, numbers.Number)
        ...
        ...     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ...         out = kwargs.get('out', ())
        ...         for x in inputs + out:
        ...             # Only support operations with instances of
        ...             # _HANDLED_TYPES. Use ArrayLike instead of type(self)
        ...             # for isinstance to allow subclasses that don't
        ...             # override __array_ufunc__ to handle ArrayLike objects.
        ...             if not isinstance(
        ...                 x, self._HANDLED_TYPES + (ArrayLike,)
        ...             ):
        ...                 return NotImplemented
        ...
        ...         # Defer to the implementation of the ufunc
        ...         # on unwrapped values.
        ...         inputs = tuple(x.value if isinstance(x, ArrayLike) else x
        ...                     for x in inputs)
        ...         if out:
        ...             kwargs['out'] = tuple(
        ...                 x.value if isinstance(x, ArrayLike) else x
        ...                 for x in out)
        ...         result = getattr(ufunc, method)(*inputs, **kwargs)
        ...
        ...         if type(result) is tuple:
        ...             # multiple return values
        ...             return tuple(type(self)(x) for x in result)
        ...         elif method == 'at':
        ...             # no return value
        ...             return None
        ...         else:
        ...             # one return value
        ...             return type(self)(result)
        ...
        ...     def __repr__(self):
        ...         return '%s(%r)' % (type(self).__name__, self.value)

    In interactions between ``ArrayLike`` objects and numbers or numpy arrays,
    the result is always another ``ArrayLike``:

        >>> x = ArrayLike([1, 2, 3])
        >>> x - 1
        ArrayLike(array([0, 1, 2]))
        >>> 1 - x
        ArrayLike(array([ 0, -1, -2]))
        >>> np.arange(3) - x
        ArrayLike(array([-1, -1, -1]))
        >>> x - np.arange(3)
        ArrayLike(array([1, 1, 1]))

    Note that unlike ``numpy.ndarray``, ``ArrayLike`` does not allow operations
    with arbitrary, unrecognized types. This ensures that interactions with
    ArrayLike preserve a well-defined casting hierarchy.

    """
    __slots__ = ()
    # Like np.ndarray, this mixin class implements "Option 1" from the ufunc
    # overrides NEP.

    # comparisons don't have reflected and in-place versions
    __lt__ = _binary_method(um.less, 'lt')
    __le__ = _binary_method(um.less_equal, 'le')
    __eq__ = _binary_method(um.equal, 'eq')
    __ne__ = _binary_method(um.not_equal, 'ne')
    __gt__ = _binary_method(um.greater, 'gt')
    __ge__ = _binary_method(um.greater_equal, 'ge')

    # numeric methods
    __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(
        um.matmul, 'matmul')
    # Python 3 does not use __div__, __rdiv__, or __idiv__
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
        um.true_divide, 'truediv')
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
        um.floor_divide, 'floordiv')
    __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
    __divmod__ = _binary_method(um.divmod, 'divmod')
    __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
    # __idivmod__ does not exist
    # TODO: handle the optional third argument for __pow__?
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
        um.left_shift, 'lshift')
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(
        um.right_shift, 'rshift')
    __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
    __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')

    # unary methods
    __neg__ = _unary_method(um.negative, 'neg')
    __pos__ = _unary_method(um.positive, 'pos')
    __abs__ = _unary_method(um.absolute, 'abs')
    __invert__ = _unary_method(um.invert, 'invert')
<|endoftext|>"""
A buffered iterator for big arrays.

This module solves the problem of iterating over a big file-based array
without having to read it into memory. The `Arrayterator` class wraps
an array object, and when iterated it will return sub-arrays with at most
a user-specified number of elements.

"""
from operator import mul
from functools import reduce

__all__ = ['Arrayterator']


class Arrayterator:
    """
    Buffered iterator for big arrays.

    `Arrayterator` creates a buffered iterator for reading big arrays in small
    contiguous blocks. The class is useful for objects stored in the
    file system. It allows iteration over the object *without* reading
    everything in memory; instead, small blocks are read and iterated over.

    `Arrayterator` can be used with any object that supports multidimensional
    slices. This includes NumPy arrays, but also variables from
    Scientific.IO.NetCDF or pynetcdf for example.

    Parameters
    ----------
    var : array_like
        The object to iterate over.
    buf_size : int, optional
        The buffer size. If `buf_size` is supplied, the maximum amount of
        data that will be read into memory is `buf_size` elements.
        Default is None, which will read as many element as possible
        into memory.

    Attributes
    ----------
    var
    buf_size
    start
    stop
    step
    shape
    flat

    See Also
    --------
    numpy.ndenumerate : Multidimensional array iterator.
    numpy.flatiter : Flat array iterator.
    numpy.memmap : Create a memory-map to an array stored
                   in a binary file on disk.

    Notes
    -----
    The algorithm works by first finding a "running dimension", along which
    the blocks will be extracted. Given an array of dimensions
    ``(d1, d2, ..., dn)``, e.g. if `buf_size` is smaller than ``d1``, the
    first dimension will be used. If, on the other hand,
    ``d1 < buf_size < d1*d2`` the second dimension will be used, and so on.
    Blocks are extracted along this dimension, and when the last block is
    returned the process continues from the next dimension, until all
    elements have been read.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    >>> a_itor = np.lib.Arrayterator(a, 2)
    >>> a_itor.shape
    (3, 4, 5, 6)

    Now we can iterate over ``a_itor``, and it will return arrays of size
    two. Since `buf_size` was smaller than any dimension, the first
    dimension will be iterated over first:

    >>> for subarr in a_itor:
    ...     if not subarr.all():
    ...         print(subarr, subarr.shape) # doctest: +SKIP
    >>> # [[[[0 1]]]] (1, 1, 1, 2)

    """

    __module__ = "numpy.lib"

    def __init__(self, var, buf_size=None):
        self.var = var
        self.buf_size = buf_size

        self.start = [0 for dim in var.shape]
        self.stop = list(var.shape)
        self.step = [1 for dim in var.shape]

    def __getattr__(self, attr):
        return getattr(self.var, attr)

    def __getitem__(self, index):
        """
        Return a new arrayterator.

        """
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), self.ndim
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims-length+1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_+1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims-len(index))

        # Return a new arrayterator object.
        out = self.__class__(self.var, self.buf_size)
        for i, (start, stop, step, slice_) in enumerate(
                zip(self.start, self.stop, self.step, index)):
            out.start[i] = start + (slice_.start or 0)
            out.step[i] = step * (slice_.step or 1)
            out.stop[i] = start + (slice_.stop or stop-start)
            out.stop[i] = min(stop, out.stop[i])
        return out

    def __array__(self, dtype=None, copy=None):
        """
        Return corresponding data.

        """
        slice_ = tuple(slice(*t) for t in zip(
                self.start, self.stop, self.step))
        return self.var[slice_]

    @property
    def flat(self):
        """
        A 1-D flat iterator for Arrayterator objects.

        This iterator returns elements of the array to be iterated over in
        `~lib.Arrayterator` one by one.
        It is similar to `flatiter`.

        See Also
        --------
        lib.Arrayterator
        flatiter

        Examples
        --------
        >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        >>> a_itor = np.lib.Arrayterator(a, 2)

        >>> for subarr in a_itor.flat:
        ...     if not subarr:
        ...         print(subarr, type(subarr))
        ...
        0 <class 'numpy.int64'>

        """
        for block in self:
            yield from block.flat

    @property
    def shape(self):
        """
        The shape of the array to be iterated over.

        For an example, see `Arrayterator`.

        """
        return tuple(((stop-start-1)//step+1) for start, stop, step in
                zip(self.start, self.stop, self.step))

    def __iter__(self):
        # Skip arrays with degenerate dimensions
        if [dim for dim in self.shape if dim <= 0]:
            return

        start = self.start[:]
        stop = self.stop[:]
        step = self.step[:]
        ndims = self.var.ndim

        while True:
            count = self.buf_size or reduce(mul, self.shape)

            # iterate over each dimension, looking for the
            # running dimension (ie, the dimension along which
            # the blocks will be built from)
            rundim = 0
            for i in range(ndims-1, -1, -1):
                # if count is zero we ran out of elements to read
                # along higher dimensions, so we read only a single position
                if count == 0:
                    stop[i] = start[i]+1
                elif count <= self.shape[i]:
                    # limit along this dimension
                    stop[i] = start[i] + count*step[i]
                    rundim = i
                else:
                    # read everything along this dimension
                    stop[i] = self.stop[i]
                stop[i] = min(self.stop[i], stop[i])
                count = count//self.shape[i]

            # yield a block
            slice_ = tuple(slice(*t) for t in zip(start, stop, step))
            yield self.var[slice_]

            # Update start position, taking care of overflow to
            # other dimensions
            start[rundim] = stop[rundim]  # start where we stopped
            for i in range(ndims-1, 0, -1):
                if start[i] >= self.stop[i]:
                    start[i] = self.start[i]
                    start[i-1] += self.step[i-1]
            if start[0] >= self.stop[0]:
                return
<|endoftext|>import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
    compiler_class, gen_lib_options, get_default_compiler, new_compiler,
    CCompiler
)
from distutils.errors import (
    DistutilsExecError, DistutilsModuleError, DistutilsPlatformError,
    CompileError, UnknownFileError
)
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

from numpy.distutils import log
from numpy.distutils.exec_command import (
    filepath_from_subprocess_output, forward_bytes_to_stdout
)
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
                                      get_num_build_jobs, \
                                      _commandline_dep_string, \
                                      sanitize_cxx_flags

# globals for parallel build management
import threading

_job_semaphore = None
_global_lock = threading.Lock()
_processing_files = set()


def _needs_build(obj, cc_args, extra_postargs, pp_opts):
    """
    Check if an objects needs to be rebuild based on its dependencies

    Parameters
    ----------
    obj : str
        object file

    Returns
    -------
    bool
    """
    # defined in unixcompiler.py
    dep_file = obj + '.d'
    if not os.path.exists(dep_file):
        return True

    # dep_file is a makefile containing 'object: dependencies'
    # formatted like posix shell (spaces escaped, \ line continuations)
    # the last line contains the compiler commandline arguments as some
    # projects may compile an extension multiple times with different
    # arguments
    with open(dep_file) as f:
        lines = f.readlines()

    cmdline =_commandline_dep_string(cc_args, extra_postargs, pp_opts)
    last_cmdline = lines[-1]
    if last_cmdline != cmdline:
        return True

    contents = ''.join(lines[:-1])
    deps = [x for x in shlex.split(contents, posix=True)
            if x != "\n" and not x.endswith(":")]

    try:
        t_obj = os.stat(obj).st_mtime

        # check if any of the dependencies is newer than the object
        # the dependencies includes the source used to create the object
        for f in deps:
            if os.stat(f).st_mtime > t_obj:
                return True
    except OSError:
        # no object counts as newer (shouldn't happen if dep_file exists)
        return True

    return False


def replace_method(klass, method_name, func):
    # Py3k does not have unbound method anymore, MethodType does not work
    m = lambda self, *args, **kw: func(self, *args, **kw)
    setattr(klass, method_name, m)


######################################################################
## Method that subclasses may redefine. But don't call this method,
## it i private to CCompiler class and may return unexpected
## results if used elsewhere. So, you have been warned..

def CCompiler_find_executables(self):
    """
    Does nothing here, but is called by the get_version method and can be
    overridden by subclasses. In particular it is redefined in the `FCompiler`
    class where more documentation can be found.

    """
    pass


replace_method(CCompiler, 'find_executables', CCompiler_find_executables)


# Using customized CCompiler.spawn.
def CCompiler_spawn(self, cmd, display=None, env=None):
    """
    Execute a command in a sub-process.

    Parameters
    ----------
    cmd : str
        The command to execute.
    display : str or sequence of str, optional
        The text to add to the log file kept by `numpy.distutils`.
        If not given, `display` is equal to `cmd`.
    env : a dictionary for environment variables, optional

    Returns
    -------
    None

    Raises
    ------
    DistutilsExecError
        If the command failed, i.e. the exit status was not 0.

    """
    env = env if env is not None else dict(os.environ)
    if display is None:
        display = cmd
        if is_sequence(display):
            display = ' '.join(list(display))
    log.info(display)
    try:
        if self.verbose:
            subprocess.check_output(cmd, env=env)
        else:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        o = exc.output
        s = exc.returncode
    except OSError as e:
        # OSError doesn't have the same hooks for the exception
        # output, but exec_command() historically would use an
        # empty string for EnvironmentError (base class for
        # OSError)
        # o = b''
        # still that would make the end-user lost in translation!
        o = f"\n\n{e}\n\n\n"
        try:
            o = o.encode(sys.stdout.encoding)
        except AttributeError:
            o = o.encode('utf8')
        # status previously used by exec_command() for parent
        # of OSError
        s = 127
    else:
        # use a convenience return here so that any kind of
        # caught exception will execute the default code after the
        # try / except block, which handles various exceptions
        return None

    if is_sequence(cmd):
        cmd = ' '.join(list(cmd))

    if self.verbose:
        forward_bytes_to_stdout(o)

    if re.search(b'Too many open files', o):
        msg = '\nTry rerunning setup command until build succeeds.'
    else:
        msg = ''
    raise DistutilsExecError('Command "%s" failed with exit status %d%s' %
                            (cmd, s, msg))

replace_method(CCompiler, 'spawn', CCompiler_spawn)

def CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
    """
    Return the name of the object files for the given source files.

    Parameters
    ----------
    source_filenames : list of str
        The list of paths to source files. Paths can be either relative or
        absolute, this is handled transparently.
    strip_dir : bool, optional
        Whether to strip the directory from the returned paths. If True,
        the file name prepended by `output_dir` is returned. Default is False.
    output_dir : str, optional
        If given, this path is prepended to the returned paths to the
        object files.

    Returns
    -------
    obj_names : list of str
        The list of paths to the object files corresponding to the source
        files in `source_filenames`.

    """
    if output_dir is None:
        output_dir = ''
    obj_names = []
    for src_name in source_filenames:
        base, ext = os.path.splitext(os.path.normpath(src_name))
        base = os.path.splitdrive(base)[1] # Chop off the drive
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        if base.startswith('..'):
            # Resolve starting relative path components, middle ones
            # (if any) have been handled by os.path.normpath above.
            i = base.rfind('..')+2
            d = base[:i]
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        if ext not in self.src_extensions:
            raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
        if strip_dir:
            base = os.path.basename(base)
        obj_name = os.path.join(output_dir, base + self.obj_extension)
        obj_names.append(obj_name)
    return obj_names

replace_method(CCompiler, 'object_filenames', CCompiler_object_filenames)

def CCompiler_compile(self, sources, output_dir=None, macros=None,
                      include_dirs=None, debug=0, extra_preargs=None,
                      extra_postargs=None, depends=None):
    """
    Compile one or more source files.

    Please refer to the Python distutils API reference for more details.

    Parameters
    ----------
    sources : list of str
        A list of filenames
    output_dir : str, optional
        Path to the output directory.
    macros : list of tuples
        A list of macro definitions.
    include_dirs : list of str, optional
        The directories to add to the default include file search path for
        this compilation only.
    debug : bool, optional
        Whether or not to output debug symbols in or alongside the object
        file(s).
    extra_preargs, extra_postargs : ?
        Extra pre- and post-arguments.
    depends : list of str, optional
        A list of file names that all targets depend on.

    Returns
    -------
    objects : list of str
        A list of object file names, one per source file `sources`.

    Raises
    ------
    CompileError
        If compilation fails.

    """
    global _job_semaphore

    jobs = get_num_build_jobs()

    # setup semaphore to not exceed number of compile jobs when parallelized at
    # extension level (python >= 3.5)
    with _global_lock:
        if _job_semaphore is None:
            _job_semaphore = threading.Semaphore(jobs)

    if not sources:
        return []
    from numpy.distutils.fcompiler import (FCompiler,
                                           FORTRAN_COMMON_FIXED_EXTENSIONS,
                                           has_f90_header)
    if isinstance(self, FCompiler):
        display = []
        for fc in ['f77', 'f90', 'fix']:
            fcomp = getattr(self, 'compiler_'+fc)
            if fcomp is None:
                continue
            display.append("Fortran %s compiler: %s" % (fc, ' '.join(fcomp)))
        display = '\n'.join(display)
    else:
        ccomp = self.compiler_so
        display = "C compiler: %s\n" % (' '.join(ccomp),)
    log.info(display)
    macros, objects, extra_postargs, pp_opts, build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    display = "compile options: '%s'" % (' '.join(cc_args))
    if extra_postargs:
        display += "\nextra options: '%s'" % (' '.join(extra_postargs))
    log.info(display)

    def single_compile(args):
        obj, (src, ext) = args
        if not _needs_build(obj, cc_args, extra_postargs, pp_opts):
            return

        # check if we are currently already processing the same object
        # happens when using the same source in multiple extensions
        while True:
            # need explicit lock as there is no atomic check and add with GIL
            with _global_lock:
                # file not being worked on, start working
                if obj not in _processing_files:
                    _processing_files.add(obj)
                    break
            # wait for the processing to end
            time.sleep(0.1)

        try:
            # retrieve slot from our #job semaphore and build
            with _job_semaphore:
                self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        finally:
            # register being done processing
            with _global_lock:
                _processing_files.remove(obj)


    if isinstance(self, FCompiler):
        objects_to_build = list(build.keys())
        f77_objects, other_objects = [], []
        for obj in objects:
            if obj in objects_to_build:
                src, ext = build[obj]
                if self.compiler_type=='absoft':
                    obj = cyg2win32(obj)
                    src = cyg2win32(src)
                if Path(src).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS \
                   and not has_f90_header(src):
                    f77_objects.append((obj, (src, ext)))
                else:
                    other_objects.append((obj, (src, ext)))

        # f77 objects can be built in parallel
        build_items = f77_objects
        # build f90 modules serial, module files are generated during
        # compilation and may be used by files later in the list so the
        # ordering is important
        for o in other_objects:
            single_compile(o)
    else:
        build_items = build.items()

    if len(build) > 1 and jobs > 1:
        # build parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs) as pool:
            res = pool.map(single_compile, build_items)
        list(res)  # access result to raise errors
    else:
        # build serial
        for o in build_items:
            single_compile(o)

    # Return *all* object filenames, not just the ones we just built.
    return objects

replace_method(CCompiler, 'compile', CCompiler_compile)

def CCompiler_customize_cmd(self, cmd, ignore=()):
    """
    Customize compiler using distutils command.

    Parameters
    ----------
    cmd : class instance
        An instance inheriting from ``distutils.cmd.Command``.
    ignore : sequence of str, optional
        List of ``distutils.ccompiler.CCompiler`` commands (without ``'set_'``) that should not be
        altered. Strings that are checked for are:
        ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',
        'rpath', 'link_objects')``.

    Returns
    -------
    None

    """
    log.info('customize %s using %s' % (self.__class__.__name__,
                                        cmd.__class__.__name__))

    if (
        hasattr(self, 'compiler') and
        'clang' in self.compiler[0] and
        not (platform.machine() == 'arm64' and sys.platform == 'darwin')
    ):
        # clang defaults to a non-strict floating error point model.
        # However, '-ftrapping-math' is not currently supported (2023-04-08)
        # for macosx_arm64.
        # Since NumPy and most Python libs give warnings for these, override:
        self.compiler.append('-ftrapping-math')
        self.compiler_so.append('-ftrapping-math')

    def allow(attr):
        return getattr(cmd, attr, None) is not None and attr not in ignore

    if allow('include_dirs'):
        self.set_include_dirs(cmd.include_dirs)
    if allow('define'):
        for (name, value) in cmd.define:
            self.define_macro(name, value)
    if allow('undef'):
        for macro in cmd.undef:
            self.undefine_macro(macro)
    if allow('libraries'):
        self.set_libraries(self.libraries + cmd.libraries)
    if allow('library_dirs'):
        self.set_library_dirs(self.library_dirs + cmd.library_dirs)
    if allow('rpath'):
        self.set_runtime_library_dirs(cmd.rpath)
    if allow('link_objects'):
        self.set_link_objects(cmd.link_objects)

replace_method(CCompiler, 'customize_cmd', CCompiler_customize_cmd)

def _compiler_to_string(compiler):
    props = []
    mx = 0
    keys = list(compiler.executables.keys())
    for key in ['version', 'libraries', 'library_dirs',
                'object_switch', 'compile_switch',
                'include_dirs', 'define', 'undef', 'rpath', 'link_objects']:
        if key not in keys:
            keys.append(key)
    for key in keys:
        if hasattr(compiler, key):
            v = getattr(compiler, key)
            mx = max(mx, len(key))
            props.append((key, repr(v)))
    fmt = '%-' + repr(mx+1) + 's = %s'
    lines = [fmt % prop for prop in props]
    return '\n'.join(lines)

def CCompiler_show_customization(self):
    """
    Print the compiler customizations to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Printing is only done if the distutils log threshold is < 2.

    """
    try:
        self.get_version()
    except Exception:
        pass
    if log._global_log.threshold<2:
        print('*'*80)
        print(self.__class__)
        print(_compiler_to_string(self))
        print('*'*80)

replace_method(CCompiler, 'show_customization', CCompiler_show_customization)

def CCompiler_customize(self, dist, need_cxx=0):
    """
    Do any platform-specific customization of a compiler instance.

    This method calls ``distutils.sysconfig.customize_compiler`` for
    platform-specific customization, as well as optionally remove a flag
    to suppress spurious warnings in case C++ code is being compiled.

    Parameters
    ----------
    dist : object
        This parameter is not used for anything.
    need_cxx : bool, optional
        Whether or not C++ has to be compiled. If so (True), the
        ``"-Wstrict-prototypes"`` option is removed to prevent spurious
        warnings. Default is False.

    Returns
    -------
    None

    Notes
    -----
    All the default options used by distutils can be extracted with::

      from distutils import sysconfig
      sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
                                'CCSHARED', 'LDSHARED', 'SO')

    """
    # See FCompiler.customize for suggested usage.
    log.info('customize %s' % (self.__class__.__name__))
    customize_compiler(self)
    if need_cxx:
        # In general, distutils uses -Wstrict-prototypes, but this option is
        # not valid for C++ code, only for C.  Remove it if it's there to
        # avoid a spurious warning on every compilation.
        try:
            self.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass

        if hasattr(self, 'compiler') and 'cc' in self.compiler[0]:
            if not self.compiler_cxx:
                if self.compiler[0].startswith('gcc'):
                    a, b = 'gcc', 'g++'
                else:
                    a, b = 'cc', 'c++'
                self.compiler_cxx = [self.compiler[0].replace(a, b)]\
                                    + self.compiler[1:]
        else:
            if hasattr(self, 'compiler'):
                log.warn("#### %s #######" % (self.compiler,))
            if not hasattr(self, 'compiler_cxx'):
                log.warn('Missing compiler_cxx fix for ' + self.__class__.__name__)


    # check if compiler supports gcc style automatic dependencies
    # run on every extension so skip for known good compilers
    if hasattr(self, 'compiler') and ('gcc' in self.compiler[0] or
                                      'g++' in self.compiler[0] or
                                      'clang' in self.compiler[0]):
        self._auto_depends = True
    elif os.name == 'posix':
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            fn = os.path.join(tmpdir, "file.c")
            with open(fn, "w") as f:
                f.write("int a;\n")
            self.compile([fn], output_dir=tmpdir,
                         extra_preargs=['-MMD', '-MF', fn + '.d'])
            self._auto_depends = True
        except CompileError:
            self._auto_depends = False
        finally:
            shutil.rmtree(tmpdir)

    return

replace_method(CCompiler, 'customize', CCompiler_customize)

def simple_version_match(pat=r'[-.\d]+', ignore='', start=''):
    """
    Simple matching of version numbers, for use in CCompiler and FCompiler.

    Parameters
    ----------
    pat : str, optional
        A regular expression matching version numbers.
        Default is ``r'[-.\\d]+'``.
    ignore : str, optional
        A regular expression matching patterns to skip.
        Default is ``''``, in which case nothing is skipped.
    start : str, optional
        A regular expression matching the start of where to start looking
        for version numbers.
        Default is ``''``, in which case searching is started at the
        beginning of the version string given to `matcher`.

    Returns
    -------
    matcher : callable
        A function that is appropriate to use as the ``.version_match``
        attribute of a ``distutils.ccompiler.CCompiler`` class. `matcher` takes a single parameter,
        a version string.

    """
    def matcher(self, version_string):
        # version string may appear in the second line, so getting rid
        # of new lines:
        version_string = version_string.replace('\n', ' ')
        pos = 0
        if start:
            m = re.match(start, version_string)
            if not m:
                return None
            pos = m.end()
        while True:
            m = re.search(pat, version_string[pos:])
            if not m:
                return None
            if ignore and re.match(ignore, m.group(0)):
                pos = m.end()
                continue
            break
        return m.group(0)
    return matcher

def CCompiler_get_version(self, force=False, ok_status=[0]):
    """
    Return compiler version, or None if compiler is not available.

    Parameters
    ----------
    force : bool, optional
        If True, force a new determination of the version, even if the
        compiler already has a version attribute. Default is False.
    ok_status : list of int, optional
        The list of status values returned by the version look-up process
        for which a version string is returned. If the status value is not
        in `ok_status`, None is returned. Default is ``[0]``.

    Returns
    -------
    version : str or None
        Version string, in the format of ``distutils.version.LooseVersion``.

    """
    if not force and hasattr(self, 'version'):
        return self.version
    self.find_executables()
    try:
        version_cmd = self.version_cmd
    except AttributeError:
        return None
    if not version_cmd or not version_cmd[0]:
        return None
    try:
        matcher = self.version_match
    except AttributeError:
        try:
            pat = self.version_pattern
        except AttributeError:
            return None
        def matcher(version_string):
            m = re.match(pat, version_string)
            if not m:
                return None
            version = m.group('version')
            return version

    try:
        output = subprocess.check_output(version_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output
        status = exc.returncode
    except OSError:
        # match the historical returns for a parent
        # exception class caught by exec_command()
        status = 127
        output = b''
    else:
        # output isn't actually a filepath but we do this
        # for now to match previous distutils behavior
        output = filepath_from_subprocess_output(output)
        status = 0

    version = None
    if status in ok_status:
        version = matcher(output)
        if version:
            version = LooseVersion(version)
    self.version = version
    return version

replace_method(CCompiler, 'get_version', CCompiler_get_version)

def CCompiler_cxx_compiler(self):
    """
    Return the C++ compiler.

    Parameters
    ----------
    None

    Returns
    -------
    cxx : class instance
        The C++ compiler, as a ``distutils.ccompiler.CCompiler`` instance.

    """
    if self.compiler_type in ('msvc', 'intelw', 'intelemw'):
        return self

    cxx = copy(self)
    cxx.compiler_cxx = cxx.compiler_cxx
    cxx.compiler_so = [cxx.compiler_cxx[0]] + \
                      sanitize_cxx_flags(cxx.compiler_so[1:])
    if (sys.platform.startswith(('aix', 'os400')) and
            'ld_so_aix' in cxx.linker_so[0]):
        # AIX needs the ld_so_aix script included with Python
        cxx.linker_so = [cxx.linker_so[0], cxx.compiler_cxx[0]] \
                        + cxx.linker_so[2:]
    if sys.platform.startswith('os400'):
        #This is required by i 7.4 and prievous for PRId64 in printf() call.
        cxx.compiler_so.append('-D__STDC_FORMAT_MACROS')
        #This a bug of gcc10.3, which failed to handle the TLS init.
        cxx.compiler_so.append('-fno-extern-tls-init')
        cxx.linker_so.append('-fno-extern-tls-init')
    else:
        cxx.linker_so = [cxx.compiler_cxx[0]] + cxx.linker_so[1:]
    return cxx

replace_method(CCompiler, 'cxx_compiler', CCompiler_cxx_compiler)

compiler_class['intel'] = ('intelccompiler', 'IntelCCompiler',
                           "Intel C Compiler for 32-bit applications")
compiler_class['intele'] = ('intelccompiler', 'IntelItaniumCCompiler',
                            "Intel C Itanium Compiler for Itanium-based applications")
compiler_class['intelem'] = ('intelccompiler', 'IntelEM64TCCompiler',
                             "Intel C Compiler for 64-bit applications")
compiler_class['intelw'] = ('intelccompiler', 'IntelCCompilerW',
                            "Intel C Compiler for 32-bit applications on Windows")
compiler_class['intelemw'] = ('intelccompiler', 'IntelEM64TCCompilerW',
                              "Intel C Compiler for 64-bit applications on Windows")
compiler_class['pathcc'] = ('pathccompiler', 'PathScaleCCompiler',
                            "PathScale Compiler for SiCortex-based applications")
compiler_class['arm'] = ('armccompiler', 'ArmCCompiler',
                            "Arm C Compiler")
compiler_class['fujitsu'] = ('fujitsuccompiler', 'FujitsuCCompiler',
                            "Fujitsu C Compiler")

ccompiler._default_compilers += (('linux.*', 'intel'),
                                 ('linux.*', 'intele'),
                                 ('linux.*', 'intelem'),
                                 ('linux.*', 'pathcc'),
                                 ('nt', 'intelw'),
                                 ('nt', 'intelemw'))

if sys.platform == 'win32':
    compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler',
                                 "Mingw32 port of GNU C Compiler for Win32"\
                                 "(for MSC built Python)")
    if mingw32():
        # On windows platforms, we want to default to mingw32 (gcc)
        # because msvc can't build blitz stuff.
        log.info('Setting mingw32 as default compiler for nt.')
        ccompiler._default_compilers = (('nt', 'mingw32'),) \
                                       + ccompiler._default_compilers


_distutils_new_compiler = new_compiler
def new_compiler (plat=None,
                  compiler=None,
                  verbose=None,
                  dry_run=0,
                  force=0):
    # Try first C compilers from numpy.distutils.
    if verbose is None:
        verbose = log.get_threshold() <= log.INFO
    if plat is None:
        plat = os.name
    try:
        if compiler is None:
            compiler = get_default_compiler(plat)
        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError(msg)
    module_name = "numpy.distutils." + module_name
    try:
        __import__ (module_name)
    except ImportError as e:
        msg = str(e)
        log.info('%s in numpy.distutils; trying from distutils',
                 str(msg))
        module_name = module_name[6:]
        try:
            __import__(module_name)
        except ImportError as e:
            msg = str(e)
            raise DistutilsModuleError("can't compile C/C++ code: unable to load module '%s'" % \
                  module_name)
    try:
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except KeyError:
        raise DistutilsModuleError(("can't compile C/C++ code: unable to find class '%s' " +
               "in module '%s'") % (class_name, module_name))
    compiler = klass(None, dry_run, force)
    compiler.verbose = verbose
    log.debug('new_compiler returns %s' % (klass))
    return compiler

ccompiler.new_compiler = new_compiler

_distutils_gen_lib_options = gen_lib_options
def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
    # the version of this function provided by CPython allows the following
    # to return lists, which are unpacked automatically:
    # - compiler.runtime_library_dir_option
    # our version extends the behavior to:
    # - compiler.library_dir_option
    # - compiler.library_option
    # - compiler.find_library_file
    r = _distutils_gen_lib_options(compiler, library_dirs,
                                   runtime_library_dirs, libraries)
    lib_opts = []
    for i in r:
        if is_sequence(i):
            lib_opts.extend(list(i))
        else:
            lib_opts.append(i)
    return lib_opts
ccompiler.gen_lib_options = gen_lib_options

# Also fix up the various compiler modules, which do
# from distutils.ccompiler import gen_lib_options
# Don't bother with mwerks, as we don't support Classic Mac.
for _cc in ['msvc9', 'msvc', '_msvc', 'bcpp', 'cygwinc', 'emxc', 'unixc']:
    _m = sys.modules.get('distutils.' + _cc + 'compiler')
    if _m is not None:
        setattr(_m, 'gen_lib_options', gen_lib_options)

<|endoftext|>import os
from distutils.msvc9compiler import MSVCCompiler as _MSVCCompiler

from .system_info import platform_bits


def _merge(old, new):
    """Concatenate two environment paths avoiding repeats.

    Here `old` is the environment string before the base class initialize
    function is called and `new` is the string after the call. The new string
    will be a fixed string if it is not obtained from the current environment,
    or the same as the old string if obtained from the same environment. The aim
    here is not to append the new string if it is already contained in the old
    string so as to limit the growth of the environment string.

    Parameters
    ----------
    old : string
        Previous environment string.
    new : string
        New environment string.

    Returns
    -------
    ret : string
        Updated environment string.

    """
    if not old:
        return new
    if new in old:
        return old

    # Neither new nor old is empty. Give old priority.
    return ';'.join([old, new])


class MSVCCompiler(_MSVCCompiler):
    def __init__(self, verbose=0, dry_run=0, force=0):
        _MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self, plat_name=None):
        # The 'lib' and 'include' variables may be overwritten
        # by MSVCCompiler.initialize, so save them for later merge.
        environ_lib = os.getenv('lib')
        environ_include = os.getenv('include')
        _MSVCCompiler.initialize(self, plat_name)

        # Merge current and previous values of 'lib' and 'include'
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])
        os.environ['include'] = _merge(environ_include, os.environ['include'])

        # msvc9 building for 32 bits requires SSE2 to work around a
        # compiler bug.
        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']

    def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
        ld_args.append('/MANIFEST')
        _MSVCCompiler.manifest_setup_ldargs(self, output_filename,
                                            build_temp, ld_args)
<|endoftext|>"""
Helper functions for interacting with the shell, and consuming shell-style
parameters provided in config files.
"""
import os
import shlex
import subprocess

__all__ = ['WindowsParser', 'PosixParser', 'NativeParser']


class CommandLineParser:
    """
    An object that knows how to split and join command-line arguments.

    It must be true that ``argv == split(join(argv))`` for all ``argv``.
    The reverse neednt be true - `join(split(cmd))` may result in the addition
    or removal of unnecessary escaping.
    """
    @staticmethod
    def join(argv):
        """ Join a list of arguments into a command line string """
        raise NotImplementedError

    @staticmethod
    def split(cmd):
        """ Split a command line string into a list of arguments """
        raise NotImplementedError


class WindowsParser:
    """
    The parsing behavior used by `subprocess.call("string")` on Windows, which
    matches the Microsoft C/C++ runtime.

    Note that this is _not_ the behavior of cmd.
    """
    @staticmethod
    def join(argv):
        # note that list2cmdline is specific to the windows syntax
        return subprocess.list2cmdline(argv)

    @staticmethod
    def split(cmd):
        import ctypes  # guarded import for systems without ctypes
        try:
            ctypes.windll
        except AttributeError:
            raise NotImplementedError

        # Windows has special parsing rules for the executable (no quotes),
        # that we do not care about - insert a dummy element
        if not cmd:
            return []
        cmd = 'dummy ' + cmd

        CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
        CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))

        nargs = ctypes.c_int()
        lpargs = CommandLineToArgvW(cmd, ctypes.byref(nargs))
        args = [lpargs[i] for i in range(nargs.value)]
        assert not ctypes.windll.kernel32.LocalFree(lpargs)

        # strip the element we inserted
        assert args[0] == "dummy"
        return args[1:]


class PosixParser:
    """
    The parsing behavior used by `subprocess.call("string", shell=True)` on Posix.
    """
    @staticmethod
    def join(argv):
        return ' '.join(shlex.quote(arg) for arg in argv)

    @staticmethod
    def split(cmd):
        return shlex.split(cmd, posix=True)


if os.name == 'nt':
    NativeParser = WindowsParser
elif os.name == 'posix':
    NativeParser = PosixParser
<|endoftext|>"""
exec_command

Implements exec_command function that is (almost) equivalent to
commands.getstatusoutput function but on NT, DOS systems the
returned status is actually correct (though, the returned status
values may be different by a factor). In addition, exec_command
takes keyword arguments for (re-)defining environment variables.

Provides functions:

  exec_command  --- execute command in a specified directory and
                    in the modified environment.
  find_executable --- locate a command using info from environment
                    variable PATH. Equivalent to posix `which`
                    command.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: 11 January 2003

Requires: Python 2.x

Successfully tested on:

========  ============  =================================================
os.name   sys.platform  comments
========  ============  =================================================
posix     linux2        Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
                        PyCrust 0.9.3, Idle 1.0.2
posix     linux2        Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
posix     sunos5        SunOS 5.9, Python 2.2, 2.3.2
posix     darwin        Darwin 7.2.0, Python 2.3
nt        win32         Windows Me
                        Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
                        Python 2.1.1 Idle 0.8
nt        win32         Windows 98, Python 2.1.1. Idle 0.8
nt        win32         Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
                        fail i.e. redefining environment variables may
                        not work. FIXED: don't use cygwin echo!
                        Comment: also `cmd /c echo` will not work
                        but redefining environment variables do work.
posix     cygwin        Cygwin 98-4.10, Python 2.3.3(cygming special)
nt        win32         Windows XP, Python 2.3.3
========  ============  =================================================

Known bugs:

* Tests, that send messages to stderr, fail when executed from MSYS prompt
  because the messages are lost at some point.

"""
__all__ = ['exec_command', 'find_executable']

import os
import sys
import subprocess
import locale
import warnings

from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log

def filepath_from_subprocess_output(output):
    """
    Convert `bytes` in the encoding used by a subprocess into a filesystem-appropriate `str`.

    Inherited from `exec_command`, and possibly incorrect.
    """
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    output = output.decode(mylocale, errors='replace')
    output = output.replace('\r\n', '\n')
    # Another historical oddity
    if output[-1:] == '\n':
        output = output[:-1]
    return output


def forward_bytes_to_stdout(val):
    """
    Forward bytes from a subprocess call to the console, without attempting to
    decode them.

    The assumption is that the subprocess call already returned bytes in
    a suitable encoding.
    """
    if hasattr(sys.stdout, 'buffer'):
        # use the underlying binary output if there is one
        sys.stdout.buffer.write(val)
    elif hasattr(sys.stdout, 'encoding'):
        # round-trip the encoding if necessary
        sys.stdout.write(val.decode(sys.stdout.encoding))
    else:
        # make a best-guess at the encoding
        sys.stdout.write(val.decode('utf8', errors='replace'))


def temp_file_name():
    # 2019-01-30, 1.17
    warnings.warn('temp_file_name is deprecated since NumPy v1.17, use '
                  'tempfile.mkstemp instead', DeprecationWarning, stacklevel=1)
    fo, name = make_temp_file()
    fo.close()
    return name

def get_pythonexe():
    pythonexe = sys.executable
    if os.name in ['nt', 'dos']:
        fdir, fn = os.path.split(pythonexe)
        fn = fn.upper().replace('PYTHONW', 'PYTHON')
        pythonexe = os.path.join(fdir, fn)
        assert os.path.isfile(pythonexe), '%r is not a file' % (pythonexe,)
    return pythonexe

def find_executable(exe, path=None, _cache={}):
    """Return full path of a executable or None.

    Symbolic links are not followed.
    """
    key = exe, path
    try:
        return _cache[key]
    except KeyError:
        pass
    log.debug('find_executable(%r)' % exe)
    orig_exe = exe

    if path is None:
        path = os.environ.get('PATH', os.defpath)
    if os.name=='posix':
        realpath = os.path.realpath
    else:
        realpath = lambda a:a

    if exe.startswith('"'):
        exe = exe[1:-1]

    suffixes = ['']
    if os.name in ['nt', 'dos', 'os2']:
        fn, ext = os.path.splitext(exe)
        extra_suffixes = ['.exe', '.com', '.bat']
        if ext.lower() not in extra_suffixes:
            suffixes = extra_suffixes

    if os.path.isabs(exe):
        paths = ['']
    else:
        paths = [ os.path.abspath(p) for p in path.split(os.pathsep) ]

    for path in paths:
        fn = os.path.join(path, exe)
        for s in suffixes:
            f_ext = fn+s
            if not os.path.islink(f_ext):
                f_ext = realpath(f_ext)
            if os.path.isfile(f_ext) and os.access(f_ext, os.X_OK):
                log.info('Found executable %s' % f_ext)
                _cache[key] = f_ext
                return f_ext

    log.warn('Could not locate executable %s' % orig_exe)
    return None

############################################################

def _preserve_environment( names ):
    log.debug('_preserve_environment(%r)' % (names))
    env = {name: os.environ.get(name) for name in names}
    return env

def _update_environment( **env ):
    log.debug('_update_environment(...)')
    for name, value in env.items():
        os.environ[name] = value or ''

def exec_command(command, execute_in='', use_shell=None, use_tee=None,
                 _with_python = 1, **env ):
    """
    Return (status,output) of executed command.

    .. deprecated:: 1.17
        Use subprocess.Popen instead

    Parameters
    ----------
    command : str
        A concatenated string of executable and arguments.
    execute_in : str
        Before running command ``cd execute_in`` and after ``cd -``.
    use_shell : {bool, None}, optional
        If True, execute ``sh -c command``. Default None (True)
    use_tee : {bool, None}, optional
        If True use tee. Default None (True)


    Returns
    -------
    res : str
        Both stdout and stderr messages.

    Notes
    -----
    On NT, DOS systems the returned status is correct for external commands.
    Wild cards will not work for non-posix systems or when use_shell=0.

    """
    # 2019-01-30, 1.17
    warnings.warn('exec_command is deprecated since NumPy v1.17, use '
                  'subprocess.Popen instead', DeprecationWarning, stacklevel=1)
    log.debug('exec_command(%r,%s)' % (command,
         ','.join(['%s=%r'%kv for kv in env.items()])))

    if use_tee is None:
        use_tee = os.name=='posix'
    if use_shell is None:
        use_shell = os.name=='posix'
    execute_in = os.path.abspath(execute_in)
    oldcwd = os.path.abspath(os.getcwd())

    if __name__[-12:] == 'exec_command':
        exec_dir = os.path.dirname(os.path.abspath(__file__))
    elif os.path.isfile('exec_command.py'):
        exec_dir = os.path.abspath('.')
    else:
        exec_dir = os.path.abspath(sys.argv[0])
        if os.path.isfile(exec_dir):
            exec_dir = os.path.dirname(exec_dir)

    if oldcwd!=execute_in:
        os.chdir(execute_in)
        log.debug('New cwd: %s' % execute_in)
    else:
        log.debug('Retaining cwd: %s' % oldcwd)

    oldenv = _preserve_environment( list(env.keys()) )
    _update_environment( **env )

    try:
        st = _exec_command(command,
                           use_shell=use_shell,
                           use_tee=use_tee,
                           **env)
    finally:
        if oldcwd!=execute_in:
            os.chdir(oldcwd)
            log.debug('Restored cwd to %s' % oldcwd)
        _update_environment(**oldenv)

    return st


def _exec_command(command, use_shell=None, use_tee = None, **env):
    """
    Internal workhorse for exec_command().
    """
    if use_shell is None:
        use_shell = os.name=='posix'
    if use_tee is None:
        use_tee = os.name=='posix'

    if os.name == 'posix' and use_shell:
        # On POSIX, subprocess always uses /bin/sh, override
        sh = os.environ.get('SHELL', '/bin/sh')
        if is_sequence(command):
            command = [sh, '-c', ' '.join(command)]
        else:
            command = [sh, '-c', command]
        use_shell = False

    elif os.name == 'nt' and is_sequence(command):
        # On Windows, join the string for CreateProcess() ourselves as
        # subprocess does it a bit differently
        command = ' '.join(_quote_arg(arg) for arg in command)

    # Inherit environment by default
    env = env or None
    try:
        # text is set to False so that communicate()
        # will return bytes. We need to decode the output ourselves
        # so that Python will not raise a UnicodeDecodeError when
        # it encounters an invalid character; rather, we simply replace it
        proc = subprocess.Popen(command, shell=use_shell, env=env, text=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    except OSError:
        # Return 127, as os.spawn*() and /bin/sh do
        return 127, ''

    text, err = proc.communicate()
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    text = text.decode(mylocale, errors='replace')
    text = text.replace('\r\n', '\n')
    # Another historical oddity
    if text[-1:] == '\n':
        text = text[:-1]

    if use_tee and text:
        print(text)
    return proc.returncode, text


def _quote_arg(arg):
    """
    Quote the argument for safe use in a shell command line.
    """
    # If there is a quote in the string, assume relevant parts of the
    # string are already quoted (e.g. '-I"C:\\Program Files\\..."')
    if '"' not in arg and ' ' in arg:
        return '"%s"' % arg
    return arg

############################################################
<|endoftext|>"""distutils.extension

Provides the Extension class, used to describe C/C++ extension
modules in setup scripts.

Overridden to support f2py.

"""
import re
from distutils.extension import Extension as old_Extension


cxx_ext_re = re.compile(r'.*\.(cpp|cxx|cc)\Z', re.I).match
fortran_pyf_ext_re = re.compile(r'.*\.(f90|f95|f77|for|ftn|f|pyf)\Z', re.I).match


class Extension(old_Extension):
    """
    Parameters
    ----------
    name : str
        Extension name.
    sources : list of str
        List of source file locations relative to the top directory of
        the package.
    extra_compile_args : list of str
        Extra command line arguments to pass to the compiler.
    extra_f77_compile_args : list of str
        Extra command line arguments to pass to the fortran77 compiler.
    extra_f90_compile_args : list of str
        Extra command line arguments to pass to the fortran90 compiler.
    """
    def __init__(
            self, name, sources,
            include_dirs=None,
            define_macros=None,
            undef_macros=None,
            library_dirs=None,
            libraries=None,
            runtime_library_dirs=None,
            extra_objects=None,
            extra_compile_args=None,
            extra_link_args=None,
            export_symbols=None,
            swig_opts=None,
            depends=None,
            language=None,
            f2py_options=None,
            module_dirs=None,
            extra_c_compile_args=None,
            extra_cxx_compile_args=None,
            extra_f77_compile_args=None,
            extra_f90_compile_args=None,):

        old_Extension.__init__(
                self, name, [],
                include_dirs=include_dirs,
                define_macros=define_macros,
                undef_macros=undef_macros,
                library_dirs=library_dirs,
                libraries=libraries,
                runtime_library_dirs=runtime_library_dirs,
                extra_objects=extra_objects,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                export_symbols=export_symbols)

        # Avoid assert statements checking that sources contains strings:
        self.sources = sources

        # Python 2.4 distutils new features
        self.swig_opts = swig_opts or []
        # swig_opts is assumed to be a list. Here we handle the case where it
        # is specified as a string instead.
        if isinstance(self.swig_opts, str):
            import warnings
            msg = "swig_opts is specified as a string instead of a list"
            warnings.warn(msg, SyntaxWarning, stacklevel=2)
            self.swig_opts = self.swig_opts.split()

        # Python 2.3 distutils new features
        self.depends = depends or []
        self.language = language

        # numpy_distutils features
        self.f2py_options = f2py_options or []
        self.module_dirs = module_dirs or []
        self.extra_c_compile_args = extra_c_compile_args or []
        self.extra_cxx_compile_args = extra_cxx_compile_args or []
        self.extra_f77_compile_args = extra_f77_compile_args or []
        self.extra_f90_compile_args = extra_f90_compile_args or []

        return

    def has_cxx_sources(self):
        return any(cxx_ext_re(str(source)) for source in self.sources)

    def has_f2py_sources(self):
        return any(fortran_pyf_ext_re(source) for source in self.sources)

# class Extension
<|endoftext|>from distutils.unixccompiler import UnixCCompiler

class FujitsuCCompiler(UnixCCompiler):

    """
    Fujitsu compiler.
    """

    compiler_type = 'fujitsu'
    cc_exe = 'fcc'
    cxx_exe = 'FCC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        cc_compiler = self.cc_exe
        cxx_compiler = self.cxx_exe
        self.set_executables(
            compiler=cc_compiler +
            ' -O3 -Nclang -fPIC',
            compiler_so=cc_compiler +
            ' -O3 -Nclang -fPIC',
            compiler_cxx=cxx_compiler +
            ' -O3 -Nclang -fPIC',
            linker_exe=cc_compiler +
            ' -lfj90i -lfj90f -lfjsrcinfo -lelf -shared',
            linker_so=cc_compiler +
            ' -lfj90i -lfj90f -lfjsrcinfo -lelf -shared'
            )
<|endoftext|># XXX: Handle setuptools ?
from distutils.core import Distribution

# This class is used because we add new files (sconscripts, and so on) with the
# scons command
class NumpyDistribution(Distribution):
    def __init__(self, attrs = None):
        # A list of (sconscripts, pre_hook, post_hook, src, parent_names)
        self.scons_data = []
        # A list of installable libraries
        self.installed_libraries = []
        # A dict of pkg_config files to generate/install
        self.installed_pkg_config = {}
        Distribution.__init__(self, attrs)

    def has_scons_scripts(self):
        return bool(self.scons_data)
<|endoftext|>""" Functions for converting from DOS to UNIX line endings

"""
import os
import re
import sys


def dos2unix(file):
    "Replace CRLF with LF in argument files.  Print names of changed files."
    if os.path.isdir(file):
        print(file, "Directory!")
        return

    with open(file, "rb") as fp:
        data = fp.read()
    if '\0' in data:
        print(file, "Binary!")
        return

    newdata = re.sub("\r\n", "\n", data)
    if newdata != data:
        print('dos2unix:', file)
        with open(file, "wb") as f:
            f.write(newdata)
        return file
    else:
        print(file, 'ok')

def dos2unix_one_dir(modified_files, dir_name, file_names):
    for file in file_names:
        full_path = os.path.join(dir_name, file)
        file = dos2unix(full_path)
        if file is not None:
            modified_files.append(file)

def dos2unix_dir(dir_name):
    modified_files = []
    os.path.walk(dir_name, dos2unix_one_dir, modified_files)
    return modified_files
#----------------------------------

def unix2dos(file):
    "Replace LF with CRLF in argument files.  Print names of changed files."
    if os.path.isdir(file):
        print(file, "Directory!")
        return

    with open(file, "rb") as fp:
        data = fp.read()
    if '\0' in data:
        print(file, "Binary!")
        return
    newdata = re.sub("\r\n", "\n", data)
    newdata = re.sub("\n", "\r\n", newdata)
    if newdata != data:
        print('unix2dos:', file)
        with open(file, "wb") as f:
            f.write(newdata)
        return file
    else:
        print(file, 'ok')

def unix2dos_one_dir(modified_files, dir_name, file_names):
    for file in file_names:
        full_path = os.path.join(dir_name, file)
        unix2dos(full_path)
        if file is not None:
            modified_files.append(file)

def unix2dos_dir(dir_name):
    modified_files = []
    os.path.walk(dir_name, unix2dos_one_dir, modified_files)
    return modified_files

if __name__ == "__main__":
    dos2unix_dir(sys.argv[1])
<|endoftext|>"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
    # 3. Force windows to use g77

"""
import os
import sys
import subprocess
import re
import textwrap

# Overwrite certain distutils.ccompiler functions:
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
# NT stuff
# 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
# 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
#    --> this is done in numpy/distutils/ccompiler.py
# 3. Force windows to use g77

import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler

try:
    from distutils.msvccompiler import get_build_version as get_build_msvc_version
except ImportError:
    def get_build_msvc_version():
        return None

from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
                                       msvc_runtime_version,
                                       msvc_runtime_major,
                                       get_build_architecture)

def get_msvcr_replacement():
    """Replacement for outdated version of get_msvcr from cygwinccompiler"""
    msvcr = msvc_runtime_library()
    return [] if msvcr is None else [msvcr]


# Useful to generate table of symbols from a dll
_START = re.compile(r'\[Ordinal/Name Pointer\] Table')
_TABLE = re.compile(r'^\s+\[([\s*[0-9]*)\] ([a-zA-Z0-9_]*)')

# the same as cygwin plus some additional parameters
class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """ A modified MingW32 compiler compatible with an MSVC built Python.

    """

    compiler_type = 'mingw32'

    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):

        distutils.cygwinccompiler.CygwinCCompiler.__init__ (self, verbose,
                                                            dry_run, force)

        # **changes: eric jones 4/11/01
        # 1. Check for import library on Windows.  Build if it doesn't exist.

        build_import_library()

        # Check for custom msvc runtime library on Windows. Build if it doesn't exist.
        msvcr_success = build_msvcr_library()
        msvcr_dbg_success = build_msvcr_library(debug=True)
        if msvcr_success or msvcr_dbg_success:
            # add preprocessor statement for using customized msvcr lib
            self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')

        # Define the MSVC version as hint for MinGW
        msvcr_version = msvc_runtime_version()
        if msvcr_version:
            self.define_macro('__MSVCRT_VERSION__', '0x%04i' % msvcr_version)

        # MS_WIN64 should be defined when building for amd64 on windows,
        # but python headers define it only for MS compilers, which has all
        # kind of bad consequences, like using Py_ModuleInit4 instead of
        # Py_ModuleInit4_64, etc... So we add it here
        if get_build_architecture() == 'AMD64':
            self.set_executables(
                compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall',
                compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall '
                            '-Wstrict-prototypes',
                linker_exe='gcc -g',
                linker_so='gcc -g -shared')
        else:
            self.set_executables(
                compiler='gcc -O2 -Wall',
                compiler_so='gcc -O2 -Wall -Wstrict-prototypes',
                linker_exe='g++ ',
                linker_so='g++ -shared')
        # added for python2.3 support
        # we can't pass it through set_executables because pre 2.2 would fail
        self.compiler_cxx = ['g++']

        # Maybe we should also append -mthreads, but then the finished dlls
        # need another dll (mingwm10.dll see Mingw32 docs) (-mthreads: Support
        # thread-safe exception handling on `Mingw32')

        # no additional libraries needed
        #self.dll_libraries=[]
        return

    # __init__ ()

    def link(self,
             target_desc,
             objects,
             output_filename,
             output_dir,
             libraries,
             library_dirs,
             runtime_library_dirs,
             export_symbols = None,
             debug=0,
             extra_preargs=None,
             extra_postargs=None,
             build_temp=None,
             target_lang=None):
        # Include the appropriate MSVC runtime library if Python was built
        # with MSVC >= 7.0 (MinGW standard is msvcrt)
        runtime_library = msvc_runtime_library()
        if runtime_library:
            if not libraries:
                libraries = []
            libraries.append(runtime_library)
        args = (self,
                target_desc,
                objects,
                output_filename,
                output_dir,
                libraries,
                library_dirs,
                runtime_library_dirs,
                None, #export_symbols, we do this in our def-file
                debug,
                extra_preargs,
                extra_postargs,
                build_temp,
                target_lang)
        func = UnixCCompiler.link
        func(*args[:func.__code__.co_argcount])
        return

    def object_filenames (self,
                          source_filenames,
                          strip_dir=0,
                          output_dir=''):
        if output_dir is None: output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            # use normcase to make sure '.rc' is really '.rc' and not '.RC'
            (base, ext) = os.path.splitext (os.path.normcase(src_name))

            # added these lines to strip off windows drive letters
            # without it, .o files are placed next to .c files
            # instead of the build directory
            drv, base = os.path.splitdrive(base)
            if drv:
                base = base[1:]

            if ext not in (self.src_extensions + ['.rc', '.res']):
                raise UnknownFileError(
                      "unknown file type '%s' (from '%s')" % \
                      (ext, src_name))
            if strip_dir:
                base = os.path.basename (base)
            if ext == '.res' or ext == '.rc':
                # these need to be compiled to object files
                obj_names.append (os.path.join (output_dir,
                                                base + ext + self.obj_extension))
            else:
                obj_names.append (os.path.join (output_dir,
                                                base + self.obj_extension))
        return obj_names

    # object_filenames ()


def find_python_dll():
    # We can't do much here:
    # - find it in the virtualenv (sys.prefix)
    # - find it in python main dir (sys.base_prefix, if in a virtualenv)
    # - in system32,
    # - otherwise (Sxs), I don't know how to get it.
    stems = [sys.prefix]
    if sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)

    sub_dirs = ['', 'lib', 'bin']
    # generate possible combinations of directory trees and sub-directories
    lib_dirs = []
    for stem in stems:
        for folder in sub_dirs:
            lib_dirs.append(os.path.join(stem, folder))

    # add system directory as well
    if 'SYSTEMROOT' in os.environ:
        lib_dirs.append(os.path.join(os.environ['SYSTEMROOT'], 'System32'))

    # search in the file system for possible candidates
    major_version, minor_version = tuple(sys.version_info[:2])
    implementation = sys.implementation.name
    if implementation == 'cpython':
        dllname = f'python{major_version}{minor_version}.dll'
    elif implementation == 'pypy':
        dllname = f'libpypy{major_version}.{minor_version}-c.dll'
    else:
        dllname = f'Unknown platform {implementation}' 
    print("Looking for %s" % dllname)
    for folder in lib_dirs:
        dll = os.path.join(folder, dllname)
        if os.path.exists(dll):
            return dll
 
    raise ValueError("%s not found in %s" % (dllname, lib_dirs))

def dump_table(dll):
    st = subprocess.check_output(["objdump.exe", "-p", dll])
    return st.split(b'\n')

def generate_def(dll, dfile):
    """Given a dll file location,  get all its exported symbols and dump them
    into the given def file.

    The .def file will be overwritten"""
    dump = dump_table(dll)
    for i in range(len(dump)):
        if _START.match(dump[i].decode()):
            break
    else:
        raise ValueError("Symbol table not found")

    syms = []
    for j in range(i+1, len(dump)):
        m = _TABLE.match(dump[j].decode())
        if m:
            syms.append((int(m.group(1).strip()), m.group(2)))
        else:
            break

    if len(syms) == 0:
        log.warn('No symbols found in %s' % dll)

    with open(dfile, 'w') as d:
        d.write('LIBRARY        %s\n' % os.path.basename(dll))
        d.write(';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
        d.write(';DATA          PRELOAD SINGLE\n')
        d.write('\nEXPORTS\n')
        for s in syms:
            #d.write('@%d    %s\n' % (s[0], s[1]))
            d.write('%s\n' % s[1])

def find_dll(dll_name):

    arch = {'AMD64' : 'amd64',
            'Intel' : 'x86'}[get_build_architecture()]

    def _find_dll_in_winsxs(dll_name):
        # Walk through the WinSxS directory to find the dll.
        winsxs_path = os.path.join(os.environ.get('WINDIR', r'C:\WINDOWS'),
                                   'winsxs')
        if not os.path.exists(winsxs_path):
            return None
        for root, dirs, files in os.walk(winsxs_path):
            if dll_name in files and arch in root:
                return os.path.join(root, dll_name)
        return None

    def _find_dll_in_path(dll_name):
        # First, look in the Python directory, then scan PATH for
        # the given dll name.
        for path in [sys.prefix] + os.environ['PATH'].split(';'):
            filepath = os.path.join(path, dll_name)
            if os.path.exists(filepath):
                return os.path.abspath(filepath)

    return _find_dll_in_winsxs(dll_name) or _find_dll_in_path(dll_name)

def build_msvcr_library(debug=False):
    if os.name != 'nt':
        return False

    # If the version number is None, then we couldn't find the MSVC runtime at
    # all, because we are running on a Python distribution which is customed
    # compiled; trust that the compiler is the same as the one available to us
    # now, and that it is capable of linking with the correct runtime without
    # any extra options.
    msvcr_ver = msvc_runtime_major()
    if msvcr_ver is None:
        log.debug('Skip building import library: '
                  'Runtime is not compiled with MSVC')
        return False

    # Skip using a custom library for versions < MSVC 8.0
    if msvcr_ver < 80:
        log.debug('Skip building msvcr library:'
                  ' custom functionality not present')
        return False

    msvcr_name = msvc_runtime_library()
    if debug:
        msvcr_name += 'd'

    # Skip if custom library already exists
    out_name = "lib%s.a" % msvcr_name
    out_file = os.path.join(sys.prefix, 'libs', out_name)
    if os.path.isfile(out_file):
        log.debug('Skip building msvcr library: "%s" exists' %
                  (out_file,))
        return True

    # Find the msvcr dll
    msvcr_dll_name = msvcr_name + '.dll'
    dll_file = find_dll(msvcr_dll_name)
    if not dll_file:
        log.warn('Cannot build msvcr library: "%s" not found' %
                 msvcr_dll_name)
        return False

    def_name = "lib%s.def" % msvcr_name
    def_file = os.path.join(sys.prefix, 'libs', def_name)

    log.info('Building msvcr library: "%s" (from %s)' \
             % (out_file, dll_file))

    # Generate a symbol definition file from the msvcr dll
    generate_def(dll_file, def_file)

    # Create a custom mingw library for the given symbol definitions
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    retcode = subprocess.call(cmd)

    # Clean up symbol definitions
    os.remove(def_file)

    return (not retcode)

def build_import_library():
    if os.name != 'nt':
        return

    arch = get_build_architecture()
    if arch == 'AMD64':
        return _build_import_library_amd64()
    elif arch == 'Intel':
        return _build_import_library_x86()
    else:
        raise ValueError("Unhandled arch %s" % arch)

def _check_for_import_lib():
    """Check if an import library for the Python runtime already exists."""
    major_version, minor_version = tuple(sys.version_info[:2])

    # patterns for the file name of the library itself
    patterns = ['libpython%d%d.a',
                'libpython%d%d.dll.a',
                'libpython%d.%d.dll.a']

    # directory trees that may contain the library
    stems = [sys.prefix]
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    elif hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix:
        stems.append(sys.real_prefix)

    # possible subdirectories within those trees where it is placed
    sub_dirs = ['libs', 'lib']

    # generate a list of candidate locations
    candidates = []
    for pat in patterns:
        filename = pat % (major_version, minor_version)
        for stem_dir in stems:
            for folder in sub_dirs:
                candidates.append(os.path.join(stem_dir, folder, filename))

    # test the filesystem to see if we can find any of these
    for fullname in candidates:
        if os.path.isfile(fullname):
            # already exists, in location given
            return (True, fullname)

    # needs to be built, preferred location given first
    return (False, candidates[0])

def _build_import_library_amd64():
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return

    # get the runtime dll for which we are building import library
    dll_file = find_python_dll()
    log.info('Building import library (arch=AMD64): "%s" (from %s)' %
             (out_file, dll_file))

    # generate symbol list from this library
    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    generate_def(dll_file, def_file)

    # generate import library from this symbol list
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    subprocess.check_call(cmd)

def _build_import_library_x86():
    """ Build the import libraries for Mingw32-gcc on Windows
    """
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return

    lib_name = "python%d%d.lib" % tuple(sys.version_info[:2])
    lib_file = os.path.join(sys.prefix, 'libs', lib_name)
    if not os.path.isfile(lib_file):
        # didn't find library file in virtualenv, try base distribution, too,
        # and use that instead if found there. for Python 2.7 venvs, the base
        # directory is in attribute real_prefix instead of base_prefix.
        if hasattr(sys, 'base_prefix'):
            base_lib = os.path.join(sys.base_prefix, 'libs', lib_name)
        elif hasattr(sys, 'real_prefix'):
            base_lib = os.path.join(sys.real_prefix, 'libs', lib_name)
        else:
            base_lib = ''  # os.path.isfile('') == False

        if os.path.isfile(base_lib):
            lib_file = base_lib
        else:
            log.warn('Cannot build import library: "%s" not found', lib_file)
            return
    log.info('Building import library (ARCH=x86): "%s"', out_file)

    from numpy.distutils import lib2def

    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    nm_output = lib2def.getnm(
            lib2def.DEFAULT_NM + [lib_file], shell=False)
    dlist, flist = lib2def.parse_nm(nm_output)
    with open(def_file, 'w') as fid:
        lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, fid)

    dll_name = find_python_dll ()

    cmd = ["dlltool",
           "--dllname", dll_name,
           "--def", def_file,
           "--output-lib", out_file]
    status = subprocess.check_output(cmd)
    if status:
        log.warn('Failed to build import library for gcc. Linking will fail.')
    return

#=====================================
# Dealing with Visual Studio MANIFESTS
#=====================================

# Functions to deal with visual studio manifests. Manifest are a mechanism to
# enforce strong DLL versioning on windows, and has nothing to do with
# distutils MANIFEST. manifests are XML files with version info, and used by
# the OS loader; they are necessary when linking against a DLL not in the
# system path; in particular, official python 2.6 binary is built against the
# MS runtime 9 (the one from VS 2008), which is not available on most windows
# systems; python 2.6 installer does install it in the Win SxS (Side by side)
# directory, but this requires the manifest for this to work. This is a big
# mess, thanks MS for a wonderful system.

# XXX: ideally, we should use exactly the same version as used by python. I
# submitted a patch to get this version, but it was only included for python
# 2.6.1 and above. So for versions below, we use a "best guess".
_MSVCRVER_TO_FULLVER = {}
if sys.platform == 'win32':
    try:
        import msvcrt
        # I took one version in my SxS directory: no idea if it is the good
        # one, and we can't retrieve it from python
        _MSVCRVER_TO_FULLVER['80'] = "8.0.50727.42"
        _MSVCRVER_TO_FULLVER['90'] = "9.0.21022.8"
        # Value from msvcrt.CRT_ASSEMBLY_VERSION under Python 3.3.0
        # on Windows XP:
        _MSVCRVER_TO_FULLVER['100'] = "10.0.30319.460"
        crt_ver = getattr(msvcrt, 'CRT_ASSEMBLY_VERSION', None)
        if crt_ver is not None:  # Available at least back to Python 3.3
            maj, min = re.match(r'(\d+)\.(\d)', crt_ver).groups()
            _MSVCRVER_TO_FULLVER[maj + min] = crt_ver
            del maj, min
        del crt_ver
    except ImportError:
        # If we are here, means python was not built with MSVC. Not sure what
        # to do in that case: manifest building will fail, but it should not be
        # used in that case anyway
        log.warn('Cannot import msvcrt: using manifest will not be possible')

def msvc_manifest_xml(maj, min):
    """Given a major and minor version of the MSVCR, returns the
    corresponding XML file."""
    try:
        fullver = _MSVCRVER_TO_FULLVER[str(maj * 10 + min)]
    except KeyError:
        raise ValueError("Version %d,%d of MSVCRT not supported yet" %
                         (maj, min)) from None
    # Don't be fooled, it looks like an XML, but it is not. In particular, it
    # should not have any space before starting, and its size should be
    # divisible by 4, most likely for alignment constraints when the xml is
    # embedded in the binary...
    # This template was copied directly from the python 2.6 binary (using
    # strings.exe from mingw on python.exe).
    template = textwrap.dedent("""\
        <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
          <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
            <security>
              <requestedPrivileges>
                <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
              </requestedPrivileges>
            </security>
          </trustInfo>
          <dependency>
            <dependentAssembly>
              <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>
            </dependentAssembly>
          </dependency>
        </assembly>""")

    return template % {'fullver': fullver, 'maj': maj, 'min': min}

def manifest_rc(name, type='dll'):
    """Return the rc file used to generate the res file which will be embedded
    as manifest for given manifest file name, of given type ('dll' or
    'exe').

    Parameters
    ----------
    name : str
            name of the manifest file to embed
    type : str {'dll', 'exe'}
            type of the binary which will embed the manifest

    """
    if type == 'dll':
        rctype = 2
    elif type == 'exe':
        rctype = 1
    else:
        raise ValueError("Type %s not supported" % type)

    return """\
#include "winuser.h"
%d RT_MANIFEST %s""" % (rctype, name)

def check_embedded_msvcr_match_linked(msver):
    """msver is the ms runtime version used for the MANIFEST."""
    # check msvcr major version are the same for linking and
    # embedding
    maj = msvc_runtime_major()
    if maj:
        if not maj == int(msver):
            raise ValueError(
                  "Discrepancy between linked msvcr " \
                  "(%d) and the one about to be embedded " \
                  "(%d)" % (int(msver), maj))

def configtest_name(config):
    base = os.path.basename(config._gen_temp_sourcefile("yo", [], "c"))
    return os.path.splitext(base)[0]

def manifest_name(config):
    # Get configest name (including suffix)
    root = configtest_name(config)
    exext = config.compiler.exe_extension
    return root + exext + ".manifest"

def rc_name(config):
    # Get configtest name (including suffix)
    root = configtest_name(config)
    return root + ".rc"

def generate_manifest(config):
    msver = get_build_msvc_version()
    if msver is not None:
        if msver >= 8:
            check_embedded_msvcr_match_linked(msver)
            ma_str, mi_str = str(msver).split('.')
            # Write the manifest file
            manxml = msvc_manifest_xml(int(ma_str), int(mi_str))
            with open(manifest_name(config), "w") as man:
                config.temp_files.append(manifest_name(config))
                man.write(manxml)
<|endoftext|>import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce

import distutils
from distutils.errors import DistutilsError

# stores temporary directory of each thread to only create one per thread
_tdata = tlocal()

# store all created temporary directories so they can be deleted on exit
_tmpdirs = []
def clean_up_temporary_directory():
    if _tmpdirs is not None:
        for d in _tmpdirs:
            try:
                shutil.rmtree(d)
            except OSError:
                pass

atexit.register(clean_up_temporary_directory)

__all__ = ['Configuration', 'get_numpy_include_dirs', 'default_config_dict',
           'dict_append', 'appendpath', 'generate_config_py',
           'get_cmd', 'allpath', 'get_mathlibs',
           'terminal_has_colors', 'red_text', 'green_text', 'yellow_text',
           'blue_text', 'cyan_text', 'cyg2win32', 'mingw32', 'all_strings',
           'has_f_sources', 'has_cxx_sources', 'filter_sources',
           'get_dependencies', 'is_local_src_dir', 'get_ext_source_files',
           'get_script_files', 'get_lib_source_files', 'get_data_files',
           'dot_join', 'get_frame', 'minrelpath', 'njoin',
           'is_sequence', 'is_string', 'as_list', 'gpaths', 'get_language',
           'get_build_architecture', 'get_info', 'get_pkg_info',
           'get_num_build_jobs', 'sanitize_cxx_flags',
           'exec_mod_from_location']

class InstallableLib:
    """
    Container to hold information on an installable library.

    Parameters
    ----------
    name : str
        Name of the installed library.
    build_info : dict
        Dictionary holding build information.
    target_dir : str
        Absolute path specifying where to install the library.

    See Also
    --------
    Configuration.add_installed_library

    Notes
    -----
    The three parameters are stored as attributes with the same names.

    """
    def __init__(self, name, build_info, target_dir):
        self.name = name
        self.build_info = build_info
        self.target_dir = target_dir


def get_num_build_jobs():
    """
    Get number of parallel build jobs set by the --parallel command line
    argument of setup.py
    If the command did not receive a setting the environment variable
    NPY_NUM_BUILD_JOBS is checked. If that is unset, return the number of
    processors on the system, with a maximum of 8 (to prevent
    overloading the system if there a lot of CPUs).

    Returns
    -------
    out : int
        number of parallel jobs that can be run

    """
    from numpy.distutils.core import get_distribution
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = multiprocessing.cpu_count()
    cpu_count = min(cpu_count, 8)
    envjobs = int(os.environ.get("NPY_NUM_BUILD_JOBS", cpu_count))
    dist = get_distribution()
    # may be None during configuration
    if dist is None:
        return envjobs

    # any of these three may have the job set, take the largest
    cmdattr = (getattr(dist.get_command_obj('build'), 'parallel', None),
               getattr(dist.get_command_obj('build_ext'), 'parallel', None),
               getattr(dist.get_command_obj('build_clib'), 'parallel', None))
    if all(x is None for x in cmdattr):
        return envjobs
    else:
        return max(x for x in cmdattr if x is not None)

def quote_args(args):
    """Quote list of arguments.

    .. deprecated:: 1.22.
    """
    import warnings
    warnings.warn('"quote_args" is deprecated.',
                  DeprecationWarning, stacklevel=2)
    # don't used _nt_quote_args as it does not check if
    # args items already have quotes or not.
    args = list(args)
    for i in range(len(args)):
        a = args[i]
        if ' ' in a and a[0] not in '"\'':
            args[i] = '"%s"' % (a)
    return args

def allpath(name):
    "Convert a /-separated pathname to one using the OS's path separator."
    split = name.split('/')
    return os.path.join(*split)

def rel_path(path, parent_path):
    """Return path relative to parent_path."""
    # Use realpath to avoid issues with symlinked dirs (see gh-7707)
    pd = os.path.realpath(os.path.abspath(parent_path))
    apath = os.path.realpath(os.path.abspath(path))
    if len(apath) < len(pd):
        return path
    if apath == pd:
        return ''
    if pd == apath[:len(pd)]:
        assert apath[len(pd)] in [os.sep], repr((path, apath[len(pd)]))
        path = apath[len(pd)+1:]
    return path

def get_path_from_frame(frame, parent_path=None):
    """Return path of the module given a frame object from the call stack.

    Returned path is relative to parent_path when given,
    otherwise it is absolute path.
    """

    # First, try to find if the file name is in the frame.
    try:
        caller_file = eval('__file__', frame.f_globals, frame.f_locals)
        d = os.path.dirname(os.path.abspath(caller_file))
    except NameError:
        # __file__ is not defined, so let's try __name__. We try this second
        # because setuptools spoofs __name__ to be '__main__' even though
        # sys.modules['__main__'] might be something else, like easy_install(1).
        caller_name = eval('__name__', frame.f_globals, frame.f_locals)
        __import__(caller_name)
        mod = sys.modules[caller_name]
        if hasattr(mod, '__file__'):
            d = os.path.dirname(os.path.abspath(mod.__file__))
        else:
            # we're probably running setup.py as execfile("setup.py")
            # (likely we're building an egg)
            d = os.path.abspath('.')

    if parent_path is not None:
        d = rel_path(d, parent_path)

    return d or '.'

def njoin(*path):
    """Join two or more pathname components +
    - convert a /-separated pathname to one using the OS's path separator.
    - resolve `..` and `.` from path.

    Either passing n arguments as in njoin('a','b'), or a sequence
    of n names as in njoin(['a','b']) is handled, or a mixture of such arguments.
    """
    paths = []
    for p in path:
        if is_sequence(p):
            # njoin(['a', 'b'], 'c')
            paths.append(njoin(*p))
        else:
            assert is_string(p)
            paths.append(p)
    path = paths
    if not path:
        # njoin()
        joined = ''
    else:
        # njoin('a', 'b')
        joined = os.path.join(*path)
    if os.path.sep != '/':
        joined = joined.replace('/', os.path.sep)
    return minrelpath(joined)

def get_mathlibs(path=None):
    """Return the MATHLIB line from numpyconfig.h
    """
    if path is not None:
        config_file = os.path.join(path, '_numpyconfig.h')
    else:
        # Look for the file in each of the numpy include directories.
        dirs = get_numpy_include_dirs()
        for path in dirs:
            fn = os.path.join(path, '_numpyconfig.h')
            if os.path.exists(fn):
                config_file = fn
                break
        else:
            raise DistutilsError('_numpyconfig.h not found in numpy include '
                'dirs %r' % (dirs,))

    with open(config_file) as fid:
        mathlibs = []
        s = '#define MATHLIB'
        for line in fid:
            if line.startswith(s):
                value = line[len(s):].strip()
                if value:
                    mathlibs.extend(value.split(','))
    return mathlibs

def minrelpath(path):
    """Resolve `..` and '.' from path.
    """
    if not is_string(path):
        return path
    if '.' not in path:
        return path
    l = path.split(os.sep)
    while l:
        try:
            i = l.index('.', 1)
        except ValueError:
            break
        del l[i]
    j = 1
    while l:
        try:
            i = l.index('..', j)
        except ValueError:
            break
        if l[i-1]=='..':
            j += 1
        else:
            del l[i], l[i-1]
            j = 1
    if not l:
        return ''
    return os.sep.join(l)

def sorted_glob(fileglob):
    """sorts output of python glob for https://bugs.python.org/issue30461
    to allow extensions to have reproducible build results"""
    return sorted(glob.glob(fileglob))

def _fix_paths(paths, local_path, include_non_existing):
    assert is_sequence(paths), repr(type(paths))
    new_paths = []
    assert not is_string(paths), repr(paths)
    for n in paths:
        if is_string(n):
            if '*' in n or '?' in n:
                p = sorted_glob(n)
                p2 = sorted_glob(njoin(local_path, n))
                if p2:
                    new_paths.extend(p2)
                elif p:
                    new_paths.extend(p)
                else:
                    if include_non_existing:
                        new_paths.append(n)
                    print('could not resolve pattern in %r: %r' %
                            (local_path, n))
            else:
                n2 = njoin(local_path, n)
                if os.path.exists(n2):
                    new_paths.append(n2)
                else:
                    if os.path.exists(n):
                        new_paths.append(n)
                    elif include_non_existing:
                        new_paths.append(n)
                    if not os.path.exists(n):
                        print('non-existing path in %r: %r' %
                                (local_path, n))

        elif is_sequence(n):
            new_paths.extend(_fix_paths(n, local_path, include_non_existing))
        else:
            new_paths.append(n)
    return [minrelpath(p) for p in new_paths]

def gpaths(paths, local_path='', include_non_existing=True):
    """Apply glob to paths and prepend local_path if needed.
    """
    if is_string(paths):
        paths = (paths,)
    return _fix_paths(paths, local_path, include_non_existing)

def make_temp_file(suffix='', prefix='', text=True):
    if not hasattr(_tdata, 'tempdir'):
        _tdata.tempdir = tempfile.mkdtemp()
        _tmpdirs.append(_tdata.tempdir)
    fid, name = tempfile.mkstemp(suffix=suffix,
                                 prefix=prefix,
                                 dir=_tdata.tempdir,
                                 text=text)
    fo = os.fdopen(fid, 'w')
    return fo, name

# Hooks for colored terminal output.
# See also https://web.archive.org/web/20100314204946/http://www.livinglogic.de/Python/ansistyle
def terminal_has_colors():
    if sys.platform=='cygwin' and 'USE_COLOR' not in os.environ:
        # Avoid importing curses that causes illegal operation
        # with a message:
        #  PYTHON2 caused an invalid page fault in
        #  module CYGNURSES7.DLL as 015f:18bbfc28
        # Details: Python 2.3.3 [GCC 3.3.1 (cygming special)]
        #          ssh to Win32 machine from debian
        #          curses.version is 2.2
        #          CYGWIN_98-4.10, release 1.5.7(0.109/3/2))
        return 0
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        try:
            import curses
            curses.setupterm()
            if (curses.tigetnum("colors") >= 0
                and curses.tigetnum("pairs") >= 0
                and ((curses.tigetstr("setf") is not None
                      and curses.tigetstr("setb") is not None)
                     or (curses.tigetstr("setaf") is not None
                         and curses.tigetstr("setab") is not None)
                     or curses.tigetstr("scp") is not None)):
                return 1
        except Exception:
            pass
    return 0

if terminal_has_colors():
    _colour_codes = dict(black=0, red=1, green=2, yellow=3,
                         blue=4, magenta=5, cyan=6, white=7, default=9)
    def colour_text(s, fg=None, bg=None, bold=False):
        seq = []
        if bold:
            seq.append('1')
        if fg:
            fgcode = 30 + _colour_codes.get(fg.lower(), 0)
            seq.append(str(fgcode))
        if bg:
            bgcode = 40 + _colour_codes.get(bg.lower(), 7)
            seq.append(str(bgcode))
        if seq:
            return '\x1b[%sm%s\x1b[0m' % (';'.join(seq), s)
        else:
            return s
else:
    def colour_text(s, fg=None, bg=None):
        return s

def default_text(s):
    return colour_text(s, 'default')
def red_text(s):
    return colour_text(s, 'red')
def green_text(s):
    return colour_text(s, 'green')
def yellow_text(s):
    return colour_text(s, 'yellow')
def cyan_text(s):
    return colour_text(s, 'cyan')
def blue_text(s):
    return colour_text(s, 'blue')

#########################

def cyg2win32(path: str) -> str:
    """Convert a path from Cygwin-native to Windows-native.

    Uses the cygpath utility (part of the Base install) to do the
    actual conversion.  Falls back to returning the original path if
    this fails.

    Handles the default ``/cygdrive`` mount prefix as well as the
    ``/proc/cygdrive`` portable prefix, custom cygdrive prefixes such
    as ``/`` or ``/mnt``, and absolute paths such as ``/usr/src/`` or
    ``/home/username``

    Parameters
    ----------
    path : str
       The path to convert

    Returns
    -------
    converted_path : str
        The converted path

    Notes
    -----
    Documentation for cygpath utility:
    https://cygwin.com/cygwin-ug-net/cygpath.html
    Documentation for the C function it wraps:
    https://cygwin.com/cygwin-api/func-cygwin-conv-path.html

    """
    if sys.platform != "cygwin":
        return path
    return subprocess.check_output(
        ["/usr/bin/cygpath", "--windows", path], text=True
    )


def mingw32():
    """Return true when using mingw32 environment.
    """
    if sys.platform=='win32':
        if os.environ.get('OSTYPE', '')=='msys':
            return True
        if os.environ.get('MSYSTEM', '')=='MINGW32':
            return True
    return False

def msvc_runtime_version():
    "Return version of MSVC runtime library, as defined by __MSC_VER__ macro"
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = int(sys.version[msc_pos+6:msc_pos+10])
    else:
        msc_ver = None
    return msc_ver

def msvc_runtime_library():
    "Return name of MSVC runtime library if Python was built with MSVC >= 7"
    ver = msvc_runtime_major ()
    if ver:
        if ver < 140:
            return "msvcr%i" % ver
        else:
            return "vcruntime%i" % ver
    else:
        return None

def msvc_runtime_major():
    "Return major version of MSVC runtime coded like get_build_msvc_version"
    major = {1300:  70,  # MSVC 7.0
             1310:  71,  # MSVC 7.1
             1400:  80,  # MSVC 8
             1500:  90,  # MSVC 9  (aka 2008)
             1600: 100,  # MSVC 10 (aka 2010)
             1900: 140,  # MSVC 14 (aka 2015)
    }.get(msvc_runtime_version(), None)
    return major

#########################

#XXX need support for .C that is also C++
cxx_ext_match = re.compile(r'.*\.(cpp|cxx|cc)\Z', re.I).match
fortran_ext_match = re.compile(r'.*\.(f90|f95|f77|for|ftn|f)\Z', re.I).match
f90_ext_match = re.compile(r'.*\.(f90|f95)\Z', re.I).match
f90_module_name_match = re.compile(r'\s*module\s*(?P<name>[\w_]+)', re.I).match
def _get_f90_modules(source):
    """Return a list of Fortran f90 module names that
    given source file defines.
    """
    if not f90_ext_match(source):
        return []
    modules = []
    with open(source) as f:
        for line in f:
            m = f90_module_name_match(line)
            if m:
                name = m.group('name')
                modules.append(name)
                # break  # XXX can we assume that there is one module per file?
    return modules

def is_string(s):
    return isinstance(s, str)

def all_strings(lst):
    """Return True if all items in lst are string objects. """
    return all(is_string(item) for item in lst)

def is_sequence(seq):
    if is_string(seq):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True

def is_glob_pattern(s):
    return is_string(s) and ('*' in s or '?' in s)

def as_list(seq):
    if is_sequence(seq):
        return list(seq)
    else:
        return [seq]

def get_language(sources):
    # not used in numpy/scipy packages, use build_ext.detect_language instead
    """Determine language value (c,f77,f90) from sources """
    language = None
    for source in sources:
        if isinstance(source, str):
            if f90_ext_match(source):
                language = 'f90'
                break
            elif fortran_ext_match(source):
                language = 'f77'
    return language

def has_f_sources(sources):
    """Return True if sources contains Fortran files """
    return any(fortran_ext_match(source) for source in sources)

def has_cxx_sources(sources):
    """Return True if sources contains C++ files """
    return any(cxx_ext_match(source) for source in sources)

def filter_sources(sources):
    """Return four lists of filenames containing
    C, C++, Fortran, and Fortran 90 module sources,
    respectively.
    """
    c_sources = []
    cxx_sources = []
    f_sources = []
    fmodule_sources = []
    for source in sources:
        if fortran_ext_match(source):
            modules = _get_f90_modules(source)
            if modules:
                fmodule_sources.append(source)
            else:
                f_sources.append(source)
        elif cxx_ext_match(source):
            cxx_sources.append(source)
        else:
            c_sources.append(source)
    return c_sources, cxx_sources, f_sources, fmodule_sources


def _get_headers(directory_list):
    # get *.h files from list of directories
    headers = []
    for d in directory_list:
        head = sorted_glob(os.path.join(d, "*.h")) #XXX: *.hpp files??
        headers.extend(head)
    return headers

def _get_directories(list_of_sources):
    # get unique directories from list of sources.
    direcs = []
    for f in list_of_sources:
        d = os.path.split(f)
        if d[0] != '' and not d[0] in direcs:
            direcs.append(d[0])
    return direcs

def _commandline_dep_string(cc_args, extra_postargs, pp_opts):
    """
    Return commandline representation used to determine if a file needs
    to be recompiled
    """
    cmdline = 'commandline: '
    cmdline += ' '.join(cc_args)
    cmdline += ' '.join(extra_postargs)
    cmdline += ' '.join(pp_opts) + '\n'
    return cmdline


def get_dependencies(sources):
    #XXX scan sources for include statements
    return _get_headers(_get_directories(sources))

def is_local_src_dir(directory):
    """Return true if directory is local directory.
    """
    if not is_string(directory):
        return False
    abs_dir = os.path.abspath(directory)
    c = os.path.commonprefix([os.getcwd(), abs_dir])
    new_dir = abs_dir[len(c):].split(os.sep)
    if new_dir and not new_dir[0]:
        new_dir = new_dir[1:]
    if new_dir and new_dir[0]=='build':
        return False
    new_dir = os.sep.join(new_dir)
    return os.path.isdir(new_dir)

def general_source_files(top_path):
    pruned_directories = {'CVS':1, '.svn':1, 'build':1}
    prune_file_pat = re.compile(r'(?:[~#]|\.py[co]|\.o)$')
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for f in filenames:
            if not prune_file_pat.search(f):
                yield os.path.join(dirpath, f)

def general_source_directories_files(top_path):
    """Return a directory name relative to top_path and
    files contained.
    """
    pruned_directories = ['CVS', '.svn', 'build']
    prune_file_pat = re.compile(r'(?:[~#]|\.py[co]|\.o)$')
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for d in dirnames:
            dpath = os.path.join(dirpath, d)
            rpath = rel_path(dpath, top_path)
            files = []
            for f in os.listdir(dpath):
                fn = os.path.join(dpath, f)
                if os.path.isfile(fn) and not prune_file_pat.search(fn):
                    files.append(fn)
            yield rpath, files
    dpath = top_path
    rpath = rel_path(dpath, top_path)
    filenames = [os.path.join(dpath, f) for f in os.listdir(dpath) \
                 if not prune_file_pat.search(f)]
    files = [f for f in filenames if os.path.isfile(f)]
    yield rpath, files


def get_ext_source_files(ext):
    # Get sources and any include files in the same directory.
    filenames = []
    sources = [_m for _m in ext.sources if is_string(_m)]
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    for d in ext.depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames

def get_script_files(scripts):
    scripts = [_m for _m in scripts if is_string(_m)]
    return scripts

def get_lib_source_files(lib):
    filenames = []
    sources = lib[1].get('sources', [])
    sources = [_m for _m in sources if is_string(_m)]
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    depends = lib[1].get('depends', [])
    for d in depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames

def get_shared_lib_extension(is_python_ext=False):
    """Return the correct file extension for shared libraries.

    Parameters
    ----------
    is_python_ext : bool, optional
        Whether the shared library is a Python extension.  Default is False.

    Returns
    -------
    so_ext : str
        The shared library extension.

    Notes
    -----
    For Python shared libs, `so_ext` will typically be '.so' on Linux and OS X,
    and '.pyd' on Windows.  For Python >= 3.2 `so_ext` has a tag prepended on
    POSIX systems according to PEP 3149.

    """
    confvars = distutils.sysconfig.get_config_vars()
    so_ext = confvars.get('EXT_SUFFIX', '')

    if not is_python_ext:
        # hardcode known values, config vars (including SHLIB_SUFFIX) are
        # unreliable (see #3182)
        # darwin, windows and debug linux are wrong in 3.3.1 and older
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            # fall back to config vars for unknown platforms
            # fix long extension for Python >=3.2, see PEP 3149.
            if 'SOABI' in confvars:
                # Does nothing unless SOABI config var exists
                so_ext = so_ext.replace('.' + confvars.get('SOABI'), '', 1)

    return so_ext

def get_data_files(data):
    if is_string(data):
        return [data]
    sources = data[1]
    filenames = []
    for s in sources:
        if hasattr(s, '__call__'):
            continue
        if is_local_src_dir(s):
            filenames.extend(list(general_source_files(s)))
        elif is_string(s):
            if os.path.isfile(s):
                filenames.append(s)
            else:
                print('Not existing data file:', s)
        else:
            raise TypeError(repr(s))
    return filenames

def dot_join(*args):
    return '.'.join([a for a in args if a])

def get_frame(level=0):
    """Return frame object from call stack with given level.
    """
    try:
        return sys._getframe(level+1)
    except AttributeError:
        frame = sys.exc_info()[2].tb_frame
        for _ in range(level+1):
            frame = frame.f_back
        return frame


######################

class Configuration:

    _list_keys = ['packages', 'ext_modules', 'data_files', 'include_dirs',
                  'libraries', 'headers', 'scripts', 'py_modules',
                  'installed_libraries', 'define_macros']
    _dict_keys = ['package_dir', 'installed_pkg_config']
    _extra_keys = ['name', 'version']

    numpy_include_dirs = []

    def __init__(self,
                 package_name=None,
                 parent_name=None,
                 top_path=None,
                 package_path=None,
                 caller_level=1,
                 setup_name='setup.py',
                 **attrs):
        """Construct configuration instance of a package.

        package_name -- name of the package
                        Ex.: 'distutils'
        parent_name  -- name of the parent package
                        Ex.: 'numpy'
        top_path     -- directory of the toplevel package
                        Ex.: the directory where the numpy package source sits
        package_path -- directory of package. Will be computed by magic from the
                        directory of the caller module if not specified
                        Ex.: the directory where numpy.distutils is
        caller_level -- frame level to caller namespace, internal parameter.
        """
        self.name = dot_join(parent_name, package_name)
        self.version = None

        caller_frame = get_frame(caller_level)
        self.local_path = get_path_from_frame(caller_frame, top_path)
        # local_path -- directory of a file (usually setup.py) that
        #               defines a configuration() function.
        # local_path -- directory of a file (usually setup.py) that
        #               defines a configuration() function.
        if top_path is None:
            top_path = self.local_path
            self.local_path = ''
        if package_path is None:
            package_path = self.local_path
        elif os.path.isdir(njoin(self.local_path, package_path)):
            package_path = njoin(self.local_path, package_path)
        if not os.path.isdir(package_path or '.'):
            raise ValueError("%r is not a directory" % (package_path,))
        self.top_path = top_path
        self.package_path = package_path
        # this is the relative path in the installed package
        self.path_in_package = os.path.join(*self.name.split('.'))

        self.list_keys = self._list_keys[:]
        self.dict_keys = self._dict_keys[:]

        for n in self.list_keys:
            v = copy.copy(attrs.get(n, []))
            setattr(self, n, as_list(v))

        for n in self.dict_keys:
            v = copy.copy(attrs.get(n, {}))
            setattr(self, n, v)

        known_keys = self.list_keys + self.dict_keys
        self.extra_keys = self._extra_keys[:]
        for n in attrs.keys():
            if n in known_keys:
                continue
            a = attrs[n]
            setattr(self, n, a)
            if isinstance(a, list):
                self.list_keys.append(n)
            elif isinstance(a, dict):
                self.dict_keys.append(n)
            else:
                self.extra_keys.append(n)

        if os.path.exists(njoin(package_path, '__init__.py')):
            self.packages.append(self.name)
            self.package_dir[self.name] = package_path

        self.options = dict(
            ignore_setup_xxx_py = False,
            assume_default_configuration = False,
            delegate_options_to_subpackages = False,
            quiet = False,
            )

        caller_instance = None
        for i in range(1, 3):
            try:
                f = get_frame(i)
            except ValueError:
                break
            try:
                caller_instance = eval('self', f.f_globals, f.f_locals)
                break
            except NameError:
                pass
        if isinstance(caller_instance, self.__class__):
            if caller_instance.options['delegate_options_to_subpackages']:
                self.set_options(**caller_instance.options)

        self.setup_name = setup_name

    def todict(self):
        """
        Return a dictionary compatible with the keyword arguments of distutils
        setup function.

        Examples
        --------
        >>> setup(**config.todict())                           #doctest: +SKIP
        """

        self._optimize_data_files()
        d = {}
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for n in known_keys:
            a = getattr(self, n)
            if a:
                d[n] = a
        return d

    def info(self, message):
        if not self.options['quiet']:
            print(message)

    def warn(self, message):
        sys.stderr.write('Warning: %s\n' % (message,))

    def set_options(self, **options):
        """
        Configure Configuration instance.

        The following options are available:
         - ignore_setup_xxx_py
         - assume_default_configuration
         - delegate_options_to_subpackages
         - quiet

        """
        for key, value in options.items():
            if key in self.options:
                self.options[key] = value
            else:
                raise ValueError('Unknown option: '+key)

    def get_distribution(self):
        """Return the distutils distribution object for self."""
        from numpy.distutils.core import get_distribution
        return get_distribution()

    def _wildcard_get_subpackage(self, subpackage_name,
                                 parent_name,
                                 caller_level = 1):
        l = subpackage_name.split('.')
        subpackage_path = njoin([self.local_path]+l)
        dirs = [_m for _m in sorted_glob(subpackage_path) if os.path.isdir(_m)]
        config_list = []
        for d in dirs:
            if not os.path.isfile(njoin(d, '__init__.py')):
                continue
            if 'build' in d.split(os.sep):
                continue
            n = '.'.join(d.split(os.sep)[-len(l):])
            c = self.get_subpackage(n,
                                    parent_name = parent_name,
                                    caller_level = caller_level+1)
            config_list.extend(c)
        return config_list

    def _get_configuration_from_setup_py(self, setup_py,
                                         subpackage_name,
                                         subpackage_path,
                                         parent_name,
                                         caller_level = 1):
        # In case setup_py imports local modules:
        sys.path.insert(0, os.path.dirname(setup_py))
        try:
            setup_name = os.path.splitext(os.path.basename(setup_py))[0]
            n = dot_join(self.name, subpackage_name, setup_name)
            setup_module = exec_mod_from_location(
                                '_'.join(n.split('.')), setup_py)
            if not hasattr(setup_module, 'configuration'):
                if not self.options['assume_default_configuration']:
                    self.warn('Assuming default configuration '\
                              '(%s does not define configuration())'\
                              % (setup_module))
                config = Configuration(subpackage_name, parent_name,
                                       self.top_path, subpackage_path,
                                       caller_level = caller_level + 1)
            else:
                pn = dot_join(*([parent_name] + subpackage_name.split('.')[:-1]))
                args = (pn,)
                if setup_module.configuration.__code__.co_argcount > 1:
                    args = args + (self.top_path,)
                config = setup_module.configuration(*args)
            if config.name!=dot_join(parent_name, subpackage_name):
                self.warn('Subpackage %r configuration returned as %r' % \
                          (dot_join(parent_name, subpackage_name), config.name))
        finally:
            del sys.path[0]
        return config

    def get_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       parent_name=None,
                       caller_level = 1):
        """Return list of subpackage configurations.

        Parameters
        ----------
        subpackage_name : str or None
            Name of the subpackage to get the configuration. '*' in
            subpackage_name is handled as a wildcard.
        subpackage_path : str
            If None, then the path is assumed to be the local path plus the
            subpackage_name. If a setup.py file is not found in the
            subpackage_path, then a default configuration is used.
        parent_name : str
            Parent name.
        """
        if subpackage_name is None:
            if subpackage_path is None:
                raise ValueError(
                    "either subpackage_name or subpackage_path must be specified")
            subpackage_name = os.path.basename(subpackage_path)

        # handle wildcards
        l = subpackage_name.split('.')
        if subpackage_path is None and '*' in subpackage_name:
            return self._wildcard_get_subpackage(subpackage_name,
                                                 parent_name,
                                                 caller_level = caller_level+1)
        assert '*' not in subpackage_name, repr((subpackage_name, subpackage_path, parent_name))
        if subpackage_path is None:
            subpackage_path = njoin([self.local_path] + l)
        else:
            subpackage_path = njoin([subpackage_path] + l[:-1])
            subpackage_path = self.paths([subpackage_path])[0]
        setup_py = njoin(subpackage_path, self.setup_name)
        if not self.options['ignore_setup_xxx_py']:
            if not os.path.isfile(setup_py):
                setup_py = njoin(subpackage_path,
                                 'setup_%s.py' % (subpackage_name))
        if not os.path.isfile(setup_py):
            if not self.options['assume_default_configuration']:
                self.warn('Assuming default configuration '\
                          '(%s/{setup_%s,setup}.py was not found)' \
                          % (os.path.dirname(setup_py), subpackage_name))
            config = Configuration(subpackage_name, parent_name,
                                   self.top_path, subpackage_path,
                                   caller_level = caller_level+1)
        else:
            config = self._get_configuration_from_setup_py(
                setup_py,
                subpackage_name,
                subpackage_path,
                parent_name,
                caller_level = caller_level + 1)
        if config:
            return [config]
        else:
            return []

    def add_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       standalone = False):
        """Add a sub-package to the current Configuration instance.

        This is useful in a setup.py script for adding sub-packages to a
        package.

        Parameters
        ----------
        subpackage_name : str
            name of the subpackage
        subpackage_path : str
            if given, the subpackage path such as the subpackage is in
            subpackage_path / subpackage_name. If None,the subpackage is
            assumed to be located in the local path / subpackage_name.
        standalone : bool
        """

        if standalone:
            parent_name = None
        else:
            parent_name = self.name
        config_list = self.get_subpackage(subpackage_name, subpackage_path,
                                          parent_name = parent_name,
                                          caller_level = 2)
        if not config_list:
            self.warn('No configuration returned, assuming unavailable.')
        for config in config_list:
            d = config
            if isinstance(config, Configuration):
                d = config.todict()
            assert isinstance(d, dict), repr(type(d))

            self.info('Appending %s configuration to %s' \
                      % (d.get('name'), self.name))
            self.dict_append(**d)

        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add a subpackage '+ subpackage_name)

    def add_data_dir(self, data_path):
        """Recursively add files under data_path to data_files list.

        Recursively add files under data_path to the list of data_files to be
