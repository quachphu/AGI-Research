    return array(a, ndmin=ndim, copy=None, subok=True)


def _accumulate(values):
    return list(itertools.accumulate(values))


def _concatenate_shapes(shapes, axis):
    """Given array shapes, return the resulting shape and slices prefixes.

    These help in nested concatenation.

    Returns
    -------
    shape: tuple of int
        This tuple satisfies::

            shape, _ = _concatenate_shapes([arr.shape for shape in arrs], axis)
            shape == concatenate(arrs, axis).shape

    slice_prefixes: tuple of (slice(start, end), )
        For a list of arrays being concatenated, this returns the slice
        in the larger array at axis that needs to be sliced into.

        For example, the following holds::

            ret = concatenate([a, b, c], axis)
            _, (sl_a, sl_b, sl_c) = concatenate_slices([a, b, c], axis)

            ret[(slice(None),) * axis + sl_a] == a
            ret[(slice(None),) * axis + sl_b] == b
            ret[(slice(None),) * axis + sl_c] == c

        These are called slice prefixes since they are used in the recursive
        blocking algorithm to compute the left-most slices during the
        recursion. Therefore, they must be prepended to rest of the slice
        that was computed deeper in the recursion.

        These are returned as tuples to ensure that they can quickly be added
        to existing slice tuple without creating a new tuple every time.

    """
    # Cache a result that will be reused.
    shape_at_axis = [shape[axis] for shape in shapes]

    # Take a shape, any shape
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis+1:]

    if any(shape[:axis] != first_shape_pre or
           shape[axis+1:] != first_shape_post for shape in shapes):
        raise ValueError(
            'Mismatched array shapes in block along axis {}.'.format(axis))

    shape = (first_shape_pre + (sum(shape_at_axis),) + first_shape[axis+1:])

    offsets_at_axis = _accumulate(shape_at_axis)
    slice_prefixes = [(slice(start, end),)
                      for start, end in zip([0] + offsets_at_axis,
                                            offsets_at_axis)]
    return shape, slice_prefixes


def _block_info_recursion(arrays, max_depth, result_ndim, depth=0):
    """
    Returns the shape of the final array, along with a list
    of slices and a list of arrays that can be used for assignment inside the
    new array

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    max_depth : list of int
        The number of nested lists
    result_ndim : int
        The number of dimensions in thefinal array.

    Returns
    -------
    shape : tuple of int
        The shape that the final array will take on.
    slices: list of tuple of slices
        The slices into the full array required for assignment. These are
        required to be prepended with ``(Ellipsis, )`` to obtain to correct
        final index.
    arrays: list of ndarray
        The data to assign to each slice of the full array

    """
    if depth < max_depth:
        shapes, slices, arrays = zip(
            *[_block_info_recursion(arr, max_depth, result_ndim, depth+1)
              for arr in arrays])

        axis = result_ndim - max_depth + depth
        shape, slice_prefixes = _concatenate_shapes(shapes, axis)

        # Prepend the slice prefix and flatten the slices
        slices = [slice_prefix + the_slice
                  for slice_prefix, inner_slices in zip(slice_prefixes, slices)
                  for the_slice in inner_slices]

        # Flatten the array list
        arrays = functools.reduce(operator.add, arrays)

        return shape, slices, arrays
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        # Return the slice and the array inside a list to be consistent with
        # the recursive case.
        arr = _atleast_nd(arrays, result_ndim)
        return arr.shape, [()], [arr]


def _block(arrays, max_depth, result_ndim, depth=0):
    """
    Internal implementation of block based on repeated concatenation.
    `arrays` is the argument passed to
    block. `max_depth` is the depth of nested lists within `arrays` and
    `result_ndim` is the greatest of the dimensions of the arrays in
    `arrays` and the depth of the lists in `arrays` (see block docstring
    for details).
    """
    if depth < max_depth:
        arrs = [_block(arr, max_depth, result_ndim, depth+1)
                for arr in arrays]
        return _concatenate(arrs, axis=-(max_depth-depth))
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        return _atleast_nd(arrays, result_ndim)


def _block_dispatcher(arrays):
    # Use type(...) is list to match the behavior of np.block(), which special
    # cases list specifically rather than allowing for generic iterables or
    # tuple. Also, we know that list.__array_function__ will never exist.
    if type(arrays) is list:
        for subarrays in arrays:
            yield from _block_dispatcher(subarrays)
    else:
        yield arrays


@array_function_dispatch(_block_dispatcher)
def block(arrays):
    """
    Assemble an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see `concatenate`) along
    the last dimension (-1), then these are concatenated along the
    second-last dimension (-2), and so on until the outermost list is reached.

    Blocks can be of any dimension, but will not be broadcasted using
    the normal rules. Instead, leading axes of size 1 are inserted,
    to make ``block.ndim`` the same for all blocks. This is primarily useful
    for working with scalars, and means that code like ``np.block([v, 1])``
    is valid, where ``v.ndim == 1``.

    When the nested list is two levels deep, this allows block matrices to be
    constructed from their components.

    Parameters
    ----------
    arrays : nested list of array_like or scalars (but not tuples)
        If passed a single ndarray or scalar (a nested list of depth 0), this
        is returned unmodified (and not copied).

        Elements shapes must match along the appropriate axes (without
        broadcasting), but leading 1s will be prepended to the shape as
        necessary to make the dimensions match.

    Returns
    -------
    block_array : ndarray
        The array assembled from the given blocks.

        The dimensionality of the output is equal to the greatest of:

        * the dimensionality of all the inputs
        * the depth to which the input list is nested

    Raises
    ------
    ValueError
        * If list depths are mismatched - for instance, ``[[a, b], c]`` is
          illegal, and should be spelt ``[[a, b], [c]]``
        * If lists are empty - for instance, ``[[a, b], []]``

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Notes
    -----
    When called with only scalars, ``np.block`` is equivalent to an ndarray
    call. So ``np.block([[1, 2], [3, 4]])`` is equivalent to
    ``np.array([[1, 2], [3, 4]])``.

    This function does not enforce that the blocks lie on a fixed grid.
    ``np.block([[a, b], [c, d]])`` is not restricted to arrays of the form::

        AAAbb
        AAAbb
        cccDD

    But is also allowed to produce, for some ``a, b, c, d``::

        AAAbb
        AAAbb
        cDDDD

    Since concatenation happens along the last axis first, `block` is *not*
    capable of producing the following directly::

        AAAbb
        cccbb
        cccDD

    Matlab's "square bracket stacking", ``[A, B, ...; p, q, ...]``, is
    equivalent to ``np.block([[A, B, ...], [p, q, ...]])``.

    Examples
    --------
    The most common use of this function is to build a block matrix:

    >>> import numpy as np
    >>> A = np.eye(2) * 2
    >>> B = np.eye(3) * 3
    >>> np.block([
    ...     [A,               np.zeros((2, 3))],
    ...     [np.ones((3, 2)), B               ]
    ... ])
    array([[2., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],
           [1., 1., 3., 0., 0.],
           [1., 1., 0., 3., 0.],
           [1., 1., 0., 0., 3.]])

    With a list of depth 1, `block` can be used as `hstack`:

    >>> np.block([1, 2, 3])              # hstack([1, 2, 3])
    array([1, 2, 3])

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.block([a, b, 10])             # hstack([a, b, 10])
    array([ 1,  2,  3,  4,  5,  6, 10])

    >>> A = np.ones((2, 2), int)
    >>> B = 2 * A
    >>> np.block([A, B])                 # hstack([A, B])
    array([[1, 1, 2, 2],
           [1, 1, 2, 2]])

    With a list of depth 2, `block` can be used in place of `vstack`:

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.block([[a], [b]])             # vstack([a, b])
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> A = np.ones((2, 2), int)
    >>> B = 2 * A
    >>> np.block([[A], [B]])             # vstack([A, B])
    array([[1, 1],
           [1, 1],
           [2, 2],
           [2, 2]])

    It can also be used in place of `atleast_1d` and `atleast_2d`:

    >>> a = np.array(0)
    >>> b = np.array([1])
    >>> np.block([a])                    # atleast_1d(a)
    array([0])
    >>> np.block([b])                    # atleast_1d(b)
    array([1])

    >>> np.block([[a]])                  # atleast_2d(a)
    array([[0]])
    >>> np.block([[b]])                  # atleast_2d(b)
    array([[1]])


    """
    arrays, list_ndim, result_ndim, final_size = _block_setup(arrays)

    # It was found through benchmarking that making an array of final size
    # around 256x256 was faster by straight concatenation on a
    # i7-7700HQ processor and dual channel ram 2400MHz.
    # It didn't seem to matter heavily on the dtype used.
    #
    # A 2D array using repeated concatenation requires 2 copies of the array.
    #
    # The fastest algorithm will depend on the ratio of CPU power to memory
    # speed.
    # One can monitor the results of the benchmark
    # https://pv.github.io/numpy-bench/#bench_shape_base.Block2D.time_block2d
    # to tune this parameter until a C version of the `_block_info_recursion`
    # algorithm is implemented which would likely be faster than the python
    # version.
    if list_ndim * final_size > (2 * 512 * 512):
        return _block_slicing(arrays, list_ndim, result_ndim)
    else:
        return _block_concatenate(arrays, list_ndim, result_ndim)


# These helper functions are mostly used for testing.
# They allow us to write tests that directly call `_block_slicing`
# or `_block_concatenate` without blocking large arrays to force the wisdom
# to trigger the desired path.
def _block_setup(arrays):
    """
    Returns
    (`arrays`, list_ndim, result_ndim, final_size)
    """
    bottom_index, arr_ndim, final_size = _block_check_depths_match(arrays)
    list_ndim = len(bottom_index)
    if bottom_index and bottom_index[-1] is None:
        raise ValueError(
            'List at {} cannot be empty'.format(
                _block_format_index(bottom_index)
            )
        )
    result_ndim = max(arr_ndim, list_ndim)
    return arrays, list_ndim, result_ndim, final_size


def _block_slicing(arrays, list_ndim, result_ndim):
    shape, slices, arrays = _block_info_recursion(
        arrays, list_ndim, result_ndim)
    dtype = _nx.result_type(*[arr.dtype for arr in arrays])

    # Test preferring F only in the case that all input arrays are F
    F_order = all(arr.flags['F_CONTIGUOUS'] for arr in arrays)
    C_order = all(arr.flags['C_CONTIGUOUS'] for arr in arrays)
    order = 'F' if F_order and not C_order else 'C'
    result = _nx.empty(shape=shape, dtype=dtype, order=order)
    # Note: In a c implementation, the function
    # PyArray_CreateMultiSortedStridePerm could be used for more advanced
    # guessing of the desired order.

    for the_slice, arr in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    return result


def _block_concatenate(arrays, list_ndim, result_ndim):
    result = _block(arrays, list_ndim, result_ndim)
    if list_ndim == 0:
        # Catch an edge case where _block returns a view because
        # `arrays` is a single numpy array and not a list of numpy arrays.
        # This might copy scalars or lists twice, but this isn't a likely
        # usecase for those interested in performance
        result = result.copy()
    return result
<|endoftext|>"""
Functions for changing global ufunc configuration

This provides helpers which wrap `_get_extobj_dict` and `_make_extobj`, and
`_extobj_contextvar` from umath.
"""
import contextlib
import contextvars
import functools

from .._utils import set_module
from .umath import _make_extobj, _get_extobj_dict, _extobj_contextvar

__all__ = [
    "seterr", "geterr", "setbufsize", "getbufsize", "seterrcall", "geterrcall",
    "errstate"
]


@set_module('numpy')
def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """
    Set how floating-point errors are handled.

    Note that operations on integer scalar types (such as `int16`) are
    handled like floating point, and are affected by these settings.

    Parameters
    ----------
    all : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Set treatment for all types of floating-point errors at once:

        - ignore: Take no action when the exception occurs.
        - warn: Print a :exc:`RuntimeWarning` (via the Python `warnings`
          module).
        - raise: Raise a :exc:`FloatingPointError`.
        - call: Call a function specified using the `seterrcall` function.
        - print: Print a warning directly to ``stdout``.
        - log: Record error in a Log object specified by `seterrcall`.

        The default is not to change the current behavior.
    divide : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for division by zero.
    over : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for floating-point overflow.
    under : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for floating-point underflow.
    invalid : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for invalid floating-point operation.

    Returns
    -------
    old_settings : dict
        Dictionary containing the old settings.

    See also
    --------
    seterrcall : Set a callback function for the 'call' mode.
    geterr, geterrcall, errstate

    Notes
    -----
    The floating-point exceptions are defined in the IEEE 754 standard [1]_:

    - Division by zero: infinite result obtained from finite numbers.
    - Overflow: result too large to be expressed.
    - Underflow: result so close to zero that some precision
      was lost.
    - Invalid operation: result is not an expressible number, typically
      indicates that a NaN was produced.

    .. [1] https://en.wikipedia.org/wiki/IEEE_754

    Examples
    --------
    >>> import numpy as np
    >>> orig_settings = np.seterr(all='ignore')  # seterr to known value
    >>> np.int16(32000) * np.int16(3)
    np.int16(30464)
    >>> np.seterr(over='raise')
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> old_settings = np.seterr(all='warn', over='raise')
    >>> np.int16(32000) * np.int16(3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    FloatingPointError: overflow encountered in scalar multiply

    >>> old_settings = np.seterr(all='print')
    >>> np.geterr()
    {'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}
    >>> np.int16(32000) * np.int16(3)
    np.int16(30464)
    >>> np.seterr(**orig_settings)  # restore original
    {'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}

    """

    old = _get_extobj_dict()
    # The errstate doesn't include call and bufsize, so pop them:
    old.pop("call", None)
    old.pop("bufsize", None)

    extobj = _make_extobj(
            all=all, divide=divide, over=over, under=under, invalid=invalid)
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def geterr():
    """
    Get the current way of handling floating-point errors.

    Returns
    -------
    res : dict
        A dictionary with keys "divide", "over", "under", and "invalid",
        whose values are from the strings "ignore", "print", "log", "warn",
        "raise", and "call". The keys represent possible floating-point
        exceptions, and the values define how these exceptions are handled.

    See Also
    --------
    geterrcall, seterr, seterrcall

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> import numpy as np
    >>> np.geterr()
    {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
    >>> np.arange(3.) / np.arange(3.)  # doctest: +SKIP
    array([nan,  1.,  1.])
    RuntimeWarning: invalid value encountered in divide

    >>> oldsettings = np.seterr(all='warn', invalid='raise')
    >>> np.geterr()
    {'divide': 'warn', 'over': 'warn', 'under': 'warn', 'invalid': 'raise'}
    >>> np.arange(3.) / np.arange(3.)
    Traceback (most recent call last):
      ...
    FloatingPointError: invalid value encountered in divide
    >>> oldsettings = np.seterr(**oldsettings)  # restore original

    """
    res = _get_extobj_dict()
    # The "geterr" doesn't include call and bufsize,:
    res.pop("call", None)
    res.pop("bufsize", None)
    return res


@set_module('numpy')
def setbufsize(size):
    """
    Set the size of the buffer used in ufuncs.

    .. versionchanged:: 2.0
        The scope of setting the buffer is tied to the `numpy.errstate`
        context.  Exiting a ``with errstate():`` will also restore the bufsize.

    Parameters
    ----------
    size : int
        Size of buffer.

    Returns
    -------
    bufsize : int
        Previous size of ufunc buffer in bytes.

    Examples
    --------
    When exiting a `numpy.errstate` context manager the bufsize is restored:

    >>> import numpy as np
    >>> with np.errstate():
    ...     np.setbufsize(4096)
    ...     print(np.getbufsize())
    ...
    8192
    4096
    >>> np.getbufsize()
    8192

    """
    old = _get_extobj_dict()["bufsize"]
    extobj = _make_extobj(bufsize=size)
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def getbufsize():
    """
    Return the size of the buffer used in ufuncs.

    Returns
    -------
    getbufsize : int
        Size of ufunc buffer in bytes.

    Examples
    --------
    >>> import numpy as np
    >>> np.getbufsize()
    8192

    """
    return _get_extobj_dict()["bufsize"]


@set_module('numpy')
def seterrcall(func):
    """
    Set the floating-point error callback function or log object.

    There are two ways to capture floating-point error messages.  The first
    is to set the error-handler to 'call', using `seterr`.  Then, set
    the function to call using this function.

    The second is to set the error-handler to 'log', using `seterr`.
    Floating-point errors then trigger a call to the 'write' method of
    the provided object.

    Parameters
    ----------
    func : callable f(err, flag) or object with write method
        Function to call upon floating-point errors ('call'-mode) or
        object whose 'write' method is used to log such message ('log'-mode).

        The call function takes two arguments. The first is a string describing
        the type of error (such as "divide by zero", "overflow", "underflow",
        or "invalid value"), and the second is the status flag.  The flag is a
        byte, whose four least-significant bits indicate the type of error, one
        of "divide", "over", "under", "invalid"::

          [0 0 0 0 divide over under invalid]

        In other words, ``flags = divide + 2*over + 4*under + 8*invalid``.

        If an object is provided, its write method should take one argument,
        a string.

    Returns
    -------
    h : callable, log instance or None
        The old error handler.

    See Also
    --------
    seterr, geterr, geterrcall

    Examples
    --------
    Callback upon error:

    >>> def err_handler(type, flag):
    ...     print("Floating point error (%s), with flag %s" % (type, flag))
    ...

    >>> import numpy as np

    >>> orig_handler = np.seterrcall(err_handler)
    >>> orig_err = np.seterr(all='call')

    >>> np.array([1, 2, 3]) / 0.0
    Floating point error (divide by zero), with flag 1
    array([inf, inf, inf])

    >>> np.seterrcall(orig_handler)
    <function err_handler at 0x...>
    >>> np.seterr(**orig_err)
    {'divide': 'call', 'over': 'call', 'under': 'call', 'invalid': 'call'}

    Log error message:

    >>> class Log:
    ...     def write(self, msg):
    ...         print("LOG: %s" % msg)
    ...

    >>> log = Log()
    >>> saved_handler = np.seterrcall(log)
    >>> save_err = np.seterr(all='log')

    >>> np.array([1, 2, 3]) / 0.0
    LOG: Warning: divide by zero encountered in divide
    array([inf, inf, inf])

    >>> np.seterrcall(orig_handler)
    <numpy.Log object at 0x...>
    >>> np.seterr(**orig_err)
    {'divide': 'log', 'over': 'log', 'under': 'log', 'invalid': 'log'}

    """
    old = _get_extobj_dict()["call"]
    extobj = _make_extobj(call=func)
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def geterrcall():
    """
    Return the current callback function used on floating-point errors.

    When the error handling for a floating-point error (one of "divide",
    "over", "under", or "invalid") is set to 'call' or 'log', the function
    that is called or the log instance that is written to is returned by
    `geterrcall`. This function or log instance has been set with
    `seterrcall`.

    Returns
    -------
    errobj : callable, log instance or None
        The current error handler. If no handler was set through `seterrcall`,
        ``None`` is returned.

    See Also
    --------
    seterrcall, seterr, geterr

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> import numpy as np
    >>> np.geterrcall()  # we did not yet set a handler, returns None

    >>> orig_settings = np.seterr(all='call')
    >>> def err_handler(type, flag):
    ...     print("Floating point error (%s), with flag %s" % (type, flag))
    >>> old_handler = np.seterrcall(err_handler)
    >>> np.array([1, 2, 3]) / 0.0
    Floating point error (divide by zero), with flag 1
    array([inf, inf, inf])

    >>> cur_handler = np.geterrcall()
    >>> cur_handler is err_handler
    True
    >>> old_settings = np.seterr(**orig_settings)  # restore original
    >>> old_handler = np.seterrcall(None)  # restore original

    """
    return _get_extobj_dict()["call"]


class _unspecified:
    pass


_Unspecified = _unspecified()


@set_module('numpy')
class errstate:
    """
    errstate(**kwargs)

    Context manager for floating-point error handling.

    Using an instance of `errstate` as a context manager allows statements in
    that context to execute with a known error handling behavior. Upon entering
    the context the error handling is set with `seterr` and `seterrcall`, and
    upon exiting it is reset to what it was before.

    ..  versionchanged:: 1.17.0
        `errstate` is also usable as a function decorator, saving
        a level of indentation if an entire function is wrapped.

    .. versionchanged:: 2.0
        `errstate` is now fully thread and asyncio safe, but may not be
        entered more than once.
        It is not safe to decorate async functions using ``errstate``.

    Parameters
    ----------
    kwargs : {divide, over, under, invalid}
        Keyword arguments. The valid keywords are the possible floating-point
        exceptions. Each keyword should have a string value that defines the
        treatment for the particular error. Possible values are
        {'ignore', 'warn', 'raise', 'call', 'print', 'log'}.

    See Also
    --------
    seterr, geterr, seterrcall, geterrcall

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> import numpy as np
    >>> olderr = np.seterr(all='ignore')  # Set error handling to known state.

    >>> np.arange(3) / 0.
    array([nan, inf, inf])
    >>> with np.errstate(divide='ignore'):
    ...     np.arange(3) / 0.
    array([nan, inf, inf])

    >>> np.sqrt(-1)
    np.float64(nan)
    >>> with np.errstate(invalid='raise'):
    ...     np.sqrt(-1)
    Traceback (most recent call last):
      File "<stdin>", line 2, in <module>
    FloatingPointError: invalid value encountered in sqrt

    Outside the context the error handling behavior has not changed:

    >>> np.geterr()
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> olderr = np.seterr(**olderr)  # restore original state

    """
    __slots__ = (
        "_call", "_all", "_divide", "_over", "_under", "_invalid", "_token")

    def __init__(self, *, call=_Unspecified,
                 all=None, divide=None, over=None, under=None, invalid=None):
        self._token = None
        self._call = call
        self._all = all
        self._divide = divide
        self._over = over
        self._under = under
        self._invalid = invalid

    def __enter__(self):
        # Note that __call__ duplicates much of this logic
        if self._token is not None:
            raise TypeError("Cannot enter `np.errstate` twice.")
        if self._call is _Unspecified:
            extobj = _make_extobj(
                    all=self._all, divide=self._divide, over=self._over,
                    under=self._under, invalid=self._invalid)
        else:
            extobj = _make_extobj(
                    call=self._call,
                    all=self._all, divide=self._divide, over=self._over,
                    under=self._under, invalid=self._invalid)

        self._token = _extobj_contextvar.set(extobj)

    def __exit__(self, *exc_info):
        _extobj_contextvar.reset(self._token)

    def __call__(self, func):
        # We need to customize `__call__` compared to `ContextDecorator`
        # because we must store the token per-thread so cannot store it on
        # the instance (we could create a new instance for this).
        # This duplicates the code from `__enter__`.
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if self._call is _Unspecified:
                extobj = _make_extobj(
                        all=self._all, divide=self._divide, over=self._over,
                        under=self._under, invalid=self._invalid)
            else:
                extobj = _make_extobj(
                        call=self._call,
                        all=self._all, divide=self._divide, over=self._over,
                        under=self._under, invalid=self._invalid)

            _token = _extobj_contextvar.set(extobj)
            try:
                # Call the original, decorated, function:
                return func(*args, **kwargs)
            finally:
                _extobj_contextvar.reset(_token)

        return inner
<|endoftext|>"""
Contains the core of NumPy: ndarray, ufuncs, dtypes, etc.

Please note that this module is private.  All functions and objects
are available in the main ``numpy`` namespace - use that instead.

"""

import os

from numpy.version import version as __version__


# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)

try:
    from . import multiarray
except ImportError as exc:
    import sys
    msg = """

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python%d.%d from "%s"
  * The NumPy version is: "%s"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: %s
""" % (sys.version_info[0], sys.version_info[1], sys.executable,
        __version__, exc)
    raise ImportError(msg)
finally:
    for envkey in env_added:
        del os.environ[envkey]
del envkey
del env_added
del os

from . import umath

# Check that multiarray,umath are pure python modules wrapping
# _multiarray_umath and not either of the old c-extension modules
if not (hasattr(multiarray, '_multiarray_umath') and
        hasattr(umath, '_multiarray_umath')):
    import sys
    path = sys.modules['numpy'].__path__
    msg = ("Something is wrong with the numpy installation. "
        "While importing we detected an older version of "
        "numpy in {}. One method of fixing this is to repeatedly uninstall "
        "numpy until none is found, then reinstall this version.")
    raise ImportError(msg.format(path))

from . import numerictypes as nt
from .numerictypes import sctypes, sctypeDict
multiarray.set_typeDict(nt.sctypeDict)
from . import numeric
from .numeric import *
from . import fromnumeric
from .fromnumeric import *
from .records import record, recarray
# Note: module name memmap is overwritten by a class with same name
from .memmap import *
from . import function_base
from .function_base import *
from . import _machar
from . import getlimits
from .getlimits import *
from . import shape_base
from .shape_base import *
from . import einsumfunc
from .einsumfunc import *
del nt

from .numeric import absolute as abs

# do this after everything else, to minimize the chance of this misleadingly
# appearing in an import-time traceback
from . import _add_newdocs
from . import _add_newdocs_scalars
# add these for module-freeze analysis (like PyInstaller)
from . import _dtype_ctypes
from . import _internal
from . import _dtype
from . import _methods

acos = numeric.arccos
acosh = numeric.arccosh
asin = numeric.arcsin
asinh = numeric.arcsinh
atan = numeric.arctan
atanh = numeric.arctanh
atan2 = numeric.arctan2
concat = numeric.concatenate
bitwise_left_shift = numeric.left_shift
bitwise_invert = numeric.invert
bitwise_right_shift = numeric.right_shift
permute_dims = numeric.transpose
pow = numeric.power

__all__ = [
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2",
    "bitwise_invert", "bitwise_left_shift", "bitwise_right_shift", "concat",
    "pow", "permute_dims", "memmap", "sctypeDict", "record", "recarray"
]
__all__ += numeric.__all__
__all__ += function_base.__all__
__all__ += getlimits.__all__
__all__ += shape_base.__all__
__all__ += einsumfunc.__all__


def _ufunc_reduce(func):
    # Report the `__name__`. pickle will try to find the module. Note that
    # pickle supports for this `__name__` to be a `__qualname__`. It may
    # make sense to add a `__qualname__` to ufuncs, to allow this more
    # explicitly (Numba has ufuncs as attributes).
    # See also: https://github.com/dask/distributed/issues/3450
    return func.__name__


def _DType_reconstruct(scalar_type):
    # This is a work-around to pickle type(np.dtype(np.float64)), etc.
    # and it should eventually be replaced with a better solution, e.g. when
    # DTypes become HeapTypes.
    return type(dtype(scalar_type))


def _DType_reduce(DType):
    # As types/classes, most DTypes can simply be pickled by their name:
    if not DType._legacy or DType.__module__ == "numpy.dtypes":
        return DType.__name__

    # However, user defined legacy dtypes (like rational) do not end up in
    # `numpy.dtypes` as module and do not have a public class at all.
    # For these, we pickle them by reconstructing them from the scalar type:
    scalar_type = DType.type
    return _DType_reconstruct, (scalar_type,)


def __getattr__(name):
    # Deprecated 2022-11-22, NumPy 1.25.
    if name == "MachAr":
        import warnings
        warnings.warn(
            "The `np._core.MachAr` is considered private API (NumPy 1.24)",
            DeprecationWarning, stacklevel=2,
        )
        return _machar.MachAr
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")


import copyreg

copyreg.pickle(ufunc, _ufunc_reduce)
copyreg.pickle(type(dtype), _DType_reduce, _DType_reconstruct)

# Unclutter namespace (must keep _*_reconstruct for unpickling)
del copyreg, _ufunc_reduce, _DType_reduce

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""
Various richly-typed exceptions, that also help us deal with string formatting
in python where it's easier.

By putting the formatting in `__str__`, we also avoid paying the cost for
users who silence the exceptions.
"""
from .._utils import set_module

def _unpack_tuple(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def _display_as_base(cls):
    """
    A decorator that makes an exception class look like its base.

    We use this to hide subclasses that are implementation details - the user
    should catch the base type, which is what the traceback will show them.

    Classes decorated with this decorator are subject to removal without a
    deprecation warning.
    """
    assert issubclass(cls, Exception)
    cls.__name__ = cls.__base__.__name__
    return cls


class UFuncTypeError(TypeError):
    """ Base class for all ufunc exceptions """
    def __init__(self, ufunc):
        self.ufunc = ufunc


@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    """ Thrown when a ufunc loop cannot be found """
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)

    def __str__(self):
        return (
            "ufunc {!r} did not contain a loop with signature matching types "
            "{!r} -> {!r}"
        ).format(
            self.ufunc.__name__,
            _unpack_tuple(self.dtypes[:self.ufunc.nin]),
            _unpack_tuple(self.dtypes[self.ufunc.nin:])
        )


@_display_as_base
class _UFuncBinaryResolutionError(_UFuncNoLoopError):
    """ Thrown when a binary resolution fails """
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc, dtypes)
        assert len(self.dtypes) == 2

    def __str__(self):
        return (
            "ufunc {!r} cannot use operands with types {!r} and {!r}"
        ).format(
            self.ufunc.__name__, *self.dtypes
        )


@_display_as_base
class _UFuncCastingError(UFuncTypeError):
    def __init__(self, ufunc, casting, from_, to):
        super().__init__(ufunc)
        self.casting = casting
        self.from_ = from_
        self.to = to


@_display_as_base
class _UFuncInputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc input cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.in_i = i

    def __str__(self):
        # only show the number if more than one input exists
        i_str = "{} ".format(self.in_i) if self.ufunc.nin != 1 else ""
        return (
            "Cannot cast ufunc {!r} input {}from {!r} to {!r} with casting "
            "rule {!r}"
        ).format(
            self.ufunc.__name__, i_str, self.from_, self.to, self.casting
        )


@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc output cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.out_i = i

    def __str__(self):
        # only show the number if more than one output exists
        i_str = "{} ".format(self.out_i) if self.ufunc.nout != 1 else ""
        return (
            "Cannot cast ufunc {!r} output {}from {!r} to {!r} with casting "
            "rule {!r}"
        ).format(
            self.ufunc.__name__, i_str, self.from_, self.to, self.casting
        )


@_display_as_base
class _ArrayMemoryError(MemoryError):
    """ Thrown when an array cannot be allocated"""
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def _total_size(self):
        num_bytes = self.dtype.itemsize
        for dim in self.shape:
            num_bytes *= dim
        return num_bytes

    @staticmethod
    def _size_to_string(num_bytes):
        """ Convert a number of bytes into a binary size string """

        # https://en.wikipedia.org/wiki/Binary_prefix
        LOG2_STEP = 10
        STEP = 1024
        units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']

        unit_i = max(num_bytes.bit_length() - 1, 1) // LOG2_STEP
        unit_val = 1 << (unit_i * LOG2_STEP)
        n_units = num_bytes / unit_val
        del unit_val

        # ensure we pick a unit that is correct after rounding
        if round(n_units) == STEP:
            unit_i += 1
            n_units /= STEP

        # deal with sizes so large that we don't have units for them
        if unit_i >= len(units):
            new_unit_i = len(units) - 1
            n_units *= 1 << ((unit_i - new_unit_i) * LOG2_STEP)
            unit_i = new_unit_i

        unit_name = units[unit_i]
        # format with a sensible number of digits
        if unit_i == 0:
            # no decimal point on bytes
            return '{:.0f} {}'.format(n_units, unit_name)
        elif round(n_units) < 1000:
            # 3 significant figures, if none are dropped to the left of the .
            return '{:#.3g} {}'.format(n_units, unit_name)
        else:
            # just give all the digits otherwise
            return '{:#.0f} {}'.format(n_units, unit_name)

    def __str__(self):
        size_str = self._size_to_string(self._total_size)
        return (
            "Unable to allocate {} for an array with shape {} and data type {}"
            .format(size_str, self.shape, self.dtype)
        )
<|endoftext|>"""
This file is separate from ``_add_newdocs.py`` so that it can be mocked out by
our sphinx ``conf.py`` during doc builds, where we want to avoid showing
platform-dependent information.
"""
import sys
import os
from numpy._core import dtype
from numpy._core import numerictypes as _numerictypes
from numpy._core.function_base import add_newdoc

##############################################################################
#
# Documentation for concrete scalar classes
#
##############################################################################

def numeric_type_aliases(aliases):
    def type_aliases_gen():
        for alias, doc in aliases:
            try:
                alias_type = getattr(_numerictypes, alias)
            except AttributeError:
                # The set of aliases that actually exist varies between platforms
                pass
            else:
                yield (alias_type, alias, doc)
    return list(type_aliases_gen())


possible_aliases = numeric_type_aliases([
    ('int8', '8-bit signed integer (``-128`` to ``127``)'),
    ('int16', '16-bit signed integer (``-32_768`` to ``32_767``)'),
    ('int32', '32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``)'),
    ('int64', '64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``)'),
    ('intp', 'Signed integer large enough to fit pointer, compatible with C ``intptr_t``'),
    ('uint8', '8-bit unsigned integer (``0`` to ``255``)'),
    ('uint16', '16-bit unsigned integer (``0`` to ``65_535``)'),
    ('uint32', '32-bit unsigned integer (``0`` to ``4_294_967_295``)'),
    ('uint64', '64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``)'),
    ('uintp', 'Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``'),
    ('float16', '16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa'),
    ('float32', '32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa'),
    ('float64', '64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa'),
    ('float96', '96-bit extended-precision floating-point number type'),
    ('float128', '128-bit extended-precision floating-point number type'),
    ('complex64', 'Complex number type composed of 2 32-bit-precision floating-point numbers'),
    ('complex128', 'Complex number type composed of 2 64-bit-precision floating-point numbers'),
    ('complex192', 'Complex number type composed of 2 96-bit extended-precision floating-point numbers'),
    ('complex256', 'Complex number type composed of 2 128-bit extended-precision floating-point numbers'),
    ])


def _get_platform_and_machine():
    try:
        system, _, _, _, machine = os.uname()
    except AttributeError:
        system = sys.platform
        if system == 'win32':
            machine = os.environ.get('PROCESSOR_ARCHITEW6432', '') \
                    or os.environ.get('PROCESSOR_ARCHITECTURE', '')
        else:
            machine = 'unknown'
    return system, machine


_system, _machine = _get_platform_and_machine()
_doc_alias_string = f":Alias on this platform ({_system} {_machine}):"


def add_newdoc_for_scalar_type(obj, fixed_aliases, doc):
    # note: `:field: value` is rST syntax which renders as field lists.
    o = getattr(_numerictypes, obj)

    character_code = dtype(o).char
    canonical_name_doc = "" if obj == o.__name__ else \
                        f":Canonical name: `numpy.{obj}`\n    "
    if fixed_aliases:
        alias_doc = ''.join(f":Alias: `numpy.{alias}`\n    "
                            for alias in fixed_aliases)
    else:
        alias_doc = ''
    alias_doc += ''.join(f"{_doc_alias_string} `numpy.{alias}`: {doc}.\n    "
                         for (alias_type, alias, doc) in possible_aliases if alias_type is o)

    docstring = f"""
    {doc.strip()}

    :Character code: ``'{character_code}'``
    {canonical_name_doc}{alias_doc}
    """

    add_newdoc('numpy._core.numerictypes', obj, docstring)


_bool_docstring = (
    """
    Boolean type (True or False), stored as a byte.

    .. warning::

       The :class:`bool` type is not a subclass of the :class:`int_` type
       (the :class:`bool` is not even a number type). This is different
       than Python's default implementation of :class:`bool` as a
       sub-class of :class:`int`.
    """
)

add_newdoc_for_scalar_type('bool', [], _bool_docstring)

add_newdoc_for_scalar_type('bool_', [], _bool_docstring)

add_newdoc_for_scalar_type('byte', [],
    """
    Signed integer type, compatible with C ``char``.
    """)

add_newdoc_for_scalar_type('short', [],
    """
    Signed integer type, compatible with C ``short``.
    """)

add_newdoc_for_scalar_type('intc', [],
    """
    Signed integer type, compatible with C ``int``.
    """)

# TODO: These docs probably need an if to highlight the default rather than
#       the C-types (and be correct).
add_newdoc_for_scalar_type('int_', [],
    """
    Default signed integer type, 64bit on 64bit systems and 32bit on 32bit
    systems.
    """)

add_newdoc_for_scalar_type('longlong', [],
    """
    Signed integer type, compatible with C ``long long``.
    """)

add_newdoc_for_scalar_type('ubyte', [],
    """
    Unsigned integer type, compatible with C ``unsigned char``.
    """)

add_newdoc_for_scalar_type('ushort', [],
    """
    Unsigned integer type, compatible with C ``unsigned short``.
    """)

add_newdoc_for_scalar_type('uintc', [],
    """
    Unsigned integer type, compatible with C ``unsigned int``.
    """)

add_newdoc_for_scalar_type('uint', [],
    """
    Unsigned signed integer type, 64bit on 64bit systems and 32bit on 32bit
    systems.
    """)

add_newdoc_for_scalar_type('ulonglong', [],
    """
    Signed integer type, compatible with C ``unsigned long long``.
    """)

add_newdoc_for_scalar_type('half', [],
    """
    Half-precision floating-point number type.
    """)

add_newdoc_for_scalar_type('single', [],
    """
    Single-precision floating-point number type, compatible with C ``float``.
    """)

add_newdoc_for_scalar_type('double', [],
    """
    Double-precision floating-point number type, compatible with Python
    :class:`float` and C ``double``.
    """)

add_newdoc_for_scalar_type('longdouble', [],
    """
    Extended-precision floating-point number type, compatible with C
    ``long double`` but not necessarily with IEEE 754 quadruple-precision.
    """)

add_newdoc_for_scalar_type('csingle', [],
    """
    Complex number type composed of two single-precision floating-point
    numbers.
    """)

add_newdoc_for_scalar_type('cdouble', [],
    """
    Complex number type composed of two double-precision floating-point
    numbers, compatible with Python :class:`complex`.
    """)

add_newdoc_for_scalar_type('clongdouble', [],
    """
    Complex number type composed of two extended-precision floating-point
    numbers.
    """)

add_newdoc_for_scalar_type('object_', [],
    """
    Any Python object.
    """)

add_newdoc_for_scalar_type('str_', [],
    r"""
    A unicode string.

    This type strips trailing null codepoints.

    >>> s = np.str_("abc\x00")
    >>> s
    'abc'

    Unlike the builtin :class:`str`, this supports the
    :ref:`python:bufferobjects`, exposing its contents as UCS4:

    >>> m = memoryview(np.str_("abc"))
    >>> m.format
    '3w'
    >>> m.tobytes()
    b'a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00'
    """)

add_newdoc_for_scalar_type('bytes_', [],
    r"""
    A byte string.

    When used in arrays, this type strips trailing null bytes.
    """)

add_newdoc_for_scalar_type('void', [],
    r"""
    np.void(length_or_data, /, dtype=None)

    Create a new structured or unstructured void scalar.

    Parameters
    ----------
    length_or_data : int, array-like, bytes-like, object
       One of multiple meanings (see notes).  The length or
       bytes data of an unstructured void.  Or alternatively,
       the data to be stored in the new scalar when `dtype`
       is provided.
       This can be an array-like, in which case an array may
       be returned.
    dtype : dtype, optional
       If provided the dtype of the new scalar.  This dtype must
       be "void" dtype (i.e. a structured or unstructured void,
       see also :ref:`defining-structured-types`).

       .. versionadded:: 1.24

    Notes
    -----
    For historical reasons and because void scalars can represent both
    arbitrary byte data and structured dtypes, the void constructor
    has three calling conventions:

    1. ``np.void(5)`` creates a ``dtype="V5"`` scalar filled with five
       ``\0`` bytes.  The 5 can be a Python or NumPy integer.
    2. ``np.void(b"bytes-like")`` creates a void scalar from the byte string.
       The dtype itemsize will match the byte string length, here ``"V10"``.
    3. When a ``dtype=`` is passed the call is roughly the same as an
       array creation.  However, a void scalar rather than array is returned.

    Please see the examples which show all three different conventions.

    Examples
    --------
    >>> np.void(5)
    np.void(b'\x00\x00\x00\x00\x00')
    >>> np.void(b'abcd')
    np.void(b'\x61\x62\x63\x64')
    >>> np.void((3.2, b'eggs'), dtype="d,S5")
    np.void((3.2, b'eggs'), dtype=[('f0', '<f8'), ('f1', 'S5')])
    >>> np.void(3, dtype=[('x', np.int8), ('y', np.int8)])
    np.void((3, 3), dtype=[('x', 'i1'), ('y', 'i1')])

    """)

add_newdoc_for_scalar_type('datetime64', [],
    """
    If created from a 64-bit integer, it represents an offset from
    ``1970-01-01T00:00:00``.
    If created from string, the string can be in ISO 8601 date
    or datetime format.

    When parsing a string to create a datetime object, if the string contains
    a trailing timezone (A 'Z' or a timezone offset), the timezone will be
    dropped and a User Warning is given.

    Datetime64 objects should be considered to be UTC and therefore have an
    offset of +0000.

    >>> np.datetime64(10, 'Y')
    np.datetime64('1980')
    >>> np.datetime64('1980', 'Y')
    np.datetime64('1980')
    >>> np.datetime64(10, 'D')
    np.datetime64('1970-01-11')

    See :ref:`arrays.datetime` for more information.
    """)

add_newdoc_for_scalar_type('timedelta64', [],
    """
    A timedelta stored as a 64-bit integer.

    See :ref:`arrays.datetime` for more information.
    """)

add_newdoc('numpy._core.numerictypes', "integer", ('is_integer',
    """
    integer.is_integer() -> bool

    Return ``True`` if the number is finite with integral value.

    .. versionadded:: 1.22

    Examples
    --------
    >>> import numpy as np
    >>> np.int64(-2).is_integer()
    True
    >>> np.uint32(5).is_integer()
    True
    """))

# TODO: work out how to put this on the base class, np.floating
for float_name in ('half', 'single', 'double', 'longdouble'):
    add_newdoc('numpy._core.numerictypes', float_name, ('as_integer_ratio',
        """
        {ftype}.as_integer_ratio() -> (int, int)

        Return a pair of integers, whose ratio is exactly equal to the original
        floating point number, and with a positive denominator.
        Raise `OverflowError` on infinities and a `ValueError` on NaNs.

        >>> np.{ftype}(10.0).as_integer_ratio()
        (10, 1)
        >>> np.{ftype}(0.0).as_integer_ratio()
        (0, 1)
        >>> np.{ftype}(-.25).as_integer_ratio()
        (-1, 4)
        """.format(ftype=float_name)))

    add_newdoc('numpy._core.numerictypes', float_name, ('is_integer',
        f"""
        {float_name}.is_integer() -> bool

        Return ``True`` if the floating point number is finite with integral
        value, and ``False`` otherwise.

        .. versionadded:: 1.22

        Examples
        --------
        >>> np.{float_name}(-2.0).is_integer()
        True
        >>> np.{float_name}(3.2).is_integer()
        False
        """))

for int_name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
        'int64', 'uint64', 'int64', 'uint64', 'int64', 'uint64'):
    # Add negative examples for signed cases by checking typecode
    add_newdoc('numpy._core.numerictypes', int_name, ('bit_count',
        f"""
        {int_name}.bit_count() -> int

        Computes the number of 1-bits in the absolute value of the input.
        Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

        Examples
        --------
        >>> np.{int_name}(127).bit_count()
        7""" +
        (f"""
        >>> np.{int_name}(-127).bit_count()
        7
        """ if dtype(int_name).char.islower() else "")))
<|endoftext|>"""Machine limits for Float32 and Float64 and (long double) if available...

"""
__all__ = ['finfo', 'iinfo']

import types
import warnings

from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, nan
from .umath import log10, exp2, nextafter, isnan


def _fr0(a):
    """fix rank-0 --> rank-1"""
    if a.ndim == 0:
        a = a.copy()
        a.shape = (1,)
    return a


def _fr1(a):
    """fix rank > 0 --> rank-0"""
    if a.size == 1:
        a = a.copy()
        a.shape = ()
    return a


class MachArLike:
    """ Object to simulate MachAr instance """
    def __init__(self, ftype, *, eps, epsneg, huge, tiny,
                 ibeta, smallest_subnormal=None, **kwargs):
        self.params = _MACHAR_PARAMS[ftype]
        self.ftype = ftype
        self.title = self.params['title']
        # Parameter types same as for discovered MachAr object.
        if not smallest_subnormal:
            self._smallest_subnormal = nextafter(
                self.ftype(0), self.ftype(1), dtype=self.ftype)
        else:
            self._smallest_subnormal = smallest_subnormal
        self.epsilon = self.eps = self._float_to_float(eps)
        self.epsneg = self._float_to_float(epsneg)
        self.xmax = self.huge = self._float_to_float(huge)
        self.xmin = self._float_to_float(tiny)
        self.smallest_normal = self.tiny = self._float_to_float(tiny)
        self.ibeta = self.params['itype'](ibeta)
        self.__dict__.update(kwargs)
        self.precision = int(-log10(self.eps))
        self.resolution = self._float_to_float(
            self._float_conv(10) ** (-self.precision))
        self._str_eps = self._float_to_str(self.eps)
        self._str_epsneg = self._float_to_str(self.epsneg)
        self._str_xmin = self._float_to_str(self.xmin)
        self._str_xmax = self._float_to_str(self.xmax)
        self._str_resolution = self._float_to_str(self.resolution)
        self._str_smallest_normal = self._float_to_str(self.xmin)

    @property
    def smallest_subnormal(self):
        """Return the value for the smallest subnormal.

        Returns
        -------
        smallest_subnormal : float
            value for the smallest subnormal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest subnormal is zero.
        """
        # Check that the calculated value is not zero, in case it raises a
        # warning.
        value = self._smallest_subnormal
        if self.ftype(0) == value:
            warnings.warn(
                'The value of the smallest subnormal for {} type '
                'is zero.'.format(self.ftype), UserWarning, stacklevel=2)

        return self._float_to_float(value)

    @property
    def _str_smallest_subnormal(self):
        """Return the string representation of the smallest subnormal."""
        return self._float_to_str(self.smallest_subnormal)

    def _float_to_float(self, value):
        """Converts float to float.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        return _fr1(self._float_conv(value))

    def _float_conv(self, value):
        """Converts float to conv.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        return array([value], self.ftype)

    def _float_to_str(self, value):
        """Converts float to str.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        return self.params['fmt'] % array(_fr0(value)[0], self.ftype)


_convert_to_float = {
    ntypes.csingle: ntypes.single,
    ntypes.complex128: ntypes.float64,
    ntypes.clongdouble: ntypes.longdouble
    }

# Parameters for creating MachAr / MachAr-like objects
_title_fmt = 'numpy {} precision floating point number'
_MACHAR_PARAMS = {
    ntypes.double: dict(
        itype = ntypes.int64,
        fmt = '%24.16e',
        title = _title_fmt.format('double')),
    ntypes.single: dict(
        itype = ntypes.int32,
        fmt = '%15.7e',
        title = _title_fmt.format('single')),
    ntypes.longdouble: dict(
        itype = ntypes.longlong,
        fmt = '%s',
        title = _title_fmt.format('long double')),
    ntypes.half: dict(
        itype = ntypes.int16,
        fmt = '%12.5e',
        title = _title_fmt.format('half'))}

# Key to identify the floating point type.  Key is result of
#
#    ftype = np.longdouble        # or float64, float32, etc.
#    v = (ftype(-1.0) / ftype(10.0))
#    v.view(v.dtype.newbyteorder('<')).tobytes()
#
# Uses division to work around deficiencies in strtold on some platforms.
# See:
# https://perl5.git.perl.org/perl.git/blob/3118d7d684b56cbeb702af874f4326683c45f045:/Configure

_KNOWN_TYPES = {}
def _register_type(machar, bytepat):
    _KNOWN_TYPES[bytepat] = machar


_float_ma = {}


def _register_known_types():
    # Known parameters for float16
    # See docstring of MachAr class for description of parameters.
    f16 = ntypes.float16
    float16_ma = MachArLike(f16,
                            machep=-10,
                            negep=-11,
                            minexp=-14,
                            maxexp=16,
                            it=10,
                            iexp=5,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f16(-10)),
                            epsneg=exp2(f16(-11)),
                            huge=f16(65504),
                            tiny=f16(2 ** -14))
    _register_type(float16_ma, b'f\xae')
    _float_ma[16] = float16_ma

    # Known parameters for float32
    f32 = ntypes.float32
    float32_ma = MachArLike(f32,
                            machep=-23,
                            negep=-24,
                            minexp=-126,
                            maxexp=128,
                            it=23,
                            iexp=8,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f32(-23)),
                            epsneg=exp2(f32(-24)),
                            huge=f32((1 - 2 ** -24) * 2**128),
                            tiny=exp2(f32(-126)))
    _register_type(float32_ma, b'\xcd\xcc\xcc\xbd')
    _float_ma[32] = float32_ma

    # Known parameters for float64
    f64 = ntypes.float64
    epsneg_f64 = 2.0 ** -53.0
    tiny_f64 = 2.0 ** -1022.0
    float64_ma = MachArLike(f64,
                            machep=-52,
                            negep=-53,
                            minexp=-1022,
                            maxexp=1024,
                            it=52,
                            iexp=11,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=2.0 ** -52.0,
                            epsneg=epsneg_f64,
                            huge=(1.0 - epsneg_f64) / tiny_f64 * f64(4),
                            tiny=tiny_f64)
    _register_type(float64_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _float_ma[64] = float64_ma

    # Known parameters for IEEE 754 128-bit binary float
    ld = ntypes.longdouble
    epsneg_f128 = exp2(ld(-113))
    tiny_f128 = exp2(ld(-16382))
    # Ignore runtime error when this is not f128
    with numeric.errstate(all='ignore'):
        huge_f128 = (ld(1) - epsneg_f128) / tiny_f128 * ld(4)
    float128_ma = MachArLike(ld,
                             machep=-112,
                             negep=-113,
                             minexp=-16382,
                             maxexp=16384,
                             it=112,
                             iexp=15,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-112)),
                             epsneg=epsneg_f128,
                             huge=huge_f128,
                             tiny=tiny_f128)
    # IEEE 754 128-bit binary float
    _register_type(float128_ma,
        b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    _float_ma[128] = float128_ma

    # Known parameters for float80 (Intel 80-bit extended precision)
    epsneg_f80 = exp2(ld(-64))
    tiny_f80 = exp2(ld(-16382))
    # Ignore runtime error when this is not f80
    with numeric.errstate(all='ignore'):
        huge_f80 = (ld(1) - epsneg_f80) / tiny_f80 * ld(4)
    float80_ma = MachArLike(ld,
                            machep=-63,
                            negep=-64,
                            minexp=-16382,
                            maxexp=16384,
                            it=63,
                            iexp=15,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(ld(-63)),
                            epsneg=epsneg_f80,
                            huge=huge_f80,
                            tiny=tiny_f80)
    # float80, first 10 bytes containing actual storage
    _register_type(float80_ma, b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf')
    _float_ma[80] = float80_ma

    # Guessed / known parameters for double double; see:
    # https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic
    # These numbers have the same exponent range as float64, but extended
    # number of digits in the significand.
    huge_dd = nextafter(ld(inf), ld(0), dtype=ld)
    # As the smallest_normal in double double is so hard to calculate we set
    # it to NaN.
    smallest_normal_dd = nan
    # Leave the same value for the smallest subnormal as double
    smallest_subnormal_dd = ld(nextafter(0., 1.))
    float_dd_ma = MachArLike(ld,
                             machep=-105,
                             negep=-106,
                             minexp=-1022,
                             maxexp=1024,
                             it=105,
                             iexp=11,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-105)),
                             epsneg=exp2(ld(-106)),
                             huge=huge_dd,
                             tiny=smallest_normal_dd,
                             smallest_subnormal=smallest_subnormal_dd)
    # double double; low, high order (e.g. PPC 64)
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99Y<\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    # double double; high, low order (e.g. PPC 64 le)
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99\xb9\xbf\x9a\x99\x99\x99\x99\x99Y<')
    _float_ma['dd'] = float_dd_ma


def _get_machar(ftype):
    """ Get MachAr instance or MachAr-like instance

    Get parameters for floating point type, by first trying signatures of
    various known floating point types, then, if none match, attempting to
    identify parameters by analysis.

    Parameters
    ----------
    ftype : class
        Numpy floating point type class (e.g. ``np.float64``)

    Returns
    -------
    ma_like : instance of :class:`MachAr` or :class:`MachArLike`
        Object giving floating point parameters for `ftype`.

    Warns
    -----
    UserWarning
        If the binary signature of the float type is not in the dictionary of
        known float types.
    """
    params = _MACHAR_PARAMS.get(ftype)
    if params is None:
        raise ValueError(repr(ftype))
    # Detect known / suspected types
    # ftype(-1.0) / ftype(10.0) is better than ftype('-0.1') because stold
    # may be deficient
    key = (ftype(-1.0) / ftype(10.))
    key = key.view(key.dtype.newbyteorder("<")).tobytes()
    ma_like = None
    if ftype == ntypes.longdouble:
        # Could be 80 bit == 10 byte extended precision, where last bytes can
        # be random garbage.
        # Comparing first 10 bytes to pattern first to avoid branching on the
        # random garbage.
        ma_like = _KNOWN_TYPES.get(key[:10])
    if ma_like is None:
        # see if the full key is known.
        ma_like = _KNOWN_TYPES.get(key)
    if ma_like is None and len(key) == 16:
        # machine limits could be f80 masquerading as np.float128,
        # find all keys with length 16 and make new dict, but make the keys
        # only 10 bytes long, the last bytes can be random garbage
        _kt = {k[:10]: v for k, v in _KNOWN_TYPES.items() if len(k) == 16}
        ma_like = _kt.get(key[:10])
    if ma_like is not None:
        return ma_like
    # Fall back to parameter discovery
    warnings.warn(
        f'Signature {key} for {ftype} does not match any known type: '
        'falling back to type probe function.\n'
        'This warnings indicates broken support for the dtype!',
        UserWarning, stacklevel=2)
    return _discovered_machar(ftype)


def _discovered_machar(ftype):
    """ Create MachAr instance with found information on float types

    TODO: MachAr should be retired completely ideally.  We currently only
          ever use it system with broken longdouble (valgrind, WSL).
    """
    params = _MACHAR_PARAMS[ftype]
    return MachAr(lambda v: array([v], ftype),
                  lambda v: _fr0(v.astype(params['itype']))[0],
                  lambda v: array(_fr0(v)[0], ftype),
                  lambda v: params['fmt'] % array(_fr0(v)[0], ftype),
                  params['title'])


@set_module('numpy')
class finfo:
    """
    finfo(dtype)

    Machine limits for floating point types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `finfo` returns information. For complex
        input, the returned dtype is the associated ``float*`` dtype for its
        real and complex components.
    eps : float
        The difference between 1.0 and the next smallest representable float
        larger than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``eps = 2**-52``, approximately 2.22e-16.
    epsneg : float
        The difference between 1.0 and the next smallest representable float
        less than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``epsneg = 2**-53``, approximately 1.11e-16.
    iexp : int
        The number of bits in the exponent portion of the floating point
        representation.
    machep : int
        The exponent that yields `eps`.
    max : floating point number of the appropriate type
        The largest representable number.
    maxexp : int
        The smallest positive power of the base (2) that causes overflow.
    min : floating point number of the appropriate type
        The smallest representable number, typically ``-max``.
    minexp : int
        The most negative power of the base (2) consistent with there
        being no leading 0's in the mantissa.
    negep : int
        The exponent that yields `epsneg`.
    nexp : int
        The number of bits in the exponent including its sign and bias.
    nmant : int
        The number of bits in the mantissa.
    precision : int
        The approximate number of decimal digits to which this kind of
        float is precise.
    resolution : floating point number of the appropriate type
        The approximate decimal resolution of this type, i.e.,
        ``10**-precision``.
    tiny : float
        An alias for `smallest_normal`, kept for backwards compatibility.
    smallest_normal : float
        The smallest positive floating point number with 1 as leading bit in
        the mantissa following IEEE-754 (see Notes).
    smallest_subnormal : float
        The smallest positive floating point number with 0 as leading bit in
        the mantissa following IEEE-754.

    Parameters
    ----------
    dtype : float, dtype, or instance
        Kind of floating point or complex floating point
        data-type about which to get information.

    See Also
    --------
    iinfo : The equivalent for integer data types.
    spacing : The distance between a value and the nearest adjacent number
    nextafter : The next floating point value after x1 towards x2

    Notes
    -----
    For developers of NumPy: do not instantiate this at the module level.
    The initial calculation of these parameters is expensive and negatively
    impacts import times.  These objects are cached, so calling ``finfo()``
    repeatedly inside your functions is not a problem.

    Note that ``smallest_normal`` is not actually the smallest positive
    representable value in a NumPy floating point type. As in the IEEE-754
    standard [1]_, NumPy floating point types make use of subnormal numbers to
    fill the gap between 0 and ``smallest_normal``. However, subnormal numbers
    may have significantly reduced precision [2]_.

    This function can also be used for complex data types as well. If used,
    the output will be the same as the corresponding real float type
    (e.g. numpy.finfo(numpy.csingle) is the same as numpy.finfo(numpy.single)).
    However, the output is true for the real and imaginary components.

    References
    ----------
    .. [1] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2008,
           pp.1-70, 2008, https://doi.org/10.1109/IEEESTD.2008.4610935
    .. [2] Wikipedia, "Denormal Numbers",
           https://en.wikipedia.org/wiki/Denormal_number

    Examples
    --------
    >>> import numpy as np
    >>> np.finfo(np.float64).dtype
    dtype('float64')
    >>> np.finfo(np.complex64).dtype
    dtype('float32')

    """

    _finfo_cache = {}

    __class_getitem__ = classmethod(types.GenericAlias)

    def __new__(cls, dtype):
        try:
            obj = cls._finfo_cache.get(dtype)  # most common path
            if obj is not None:
                return obj
        except TypeError:
            pass

        if dtype is None:
            # Deprecated in NumPy 1.25, 2023-01-16
            warnings.warn(
                "finfo() dtype cannot be None. This behavior will "
                "raise an error in the future. (Deprecated in NumPy 1.25)",
                DeprecationWarning,
                stacklevel=2
            )

        try:
            dtype = numeric.dtype(dtype)
        except TypeError:
            # In case a float instance was given
            dtype = numeric.dtype(type(dtype))

        obj = cls._finfo_cache.get(dtype)
        if obj is not None:
            return obj
        dtypes = [dtype]
        newdtype = ntypes.obj2sctype(dtype)
        if newdtype is not dtype:
            dtypes.append(newdtype)
            dtype = newdtype
        if not issubclass(dtype, numeric.inexact):
            raise ValueError("data type %r not inexact" % (dtype))
        obj = cls._finfo_cache.get(dtype)
        if obj is not None:
            return obj
        if not issubclass(dtype, numeric.floating):
            newdtype = _convert_to_float[dtype]
            if newdtype is not dtype:
                # dtype changed, for example from complex128 to float64
                dtypes.append(newdtype)
                dtype = newdtype

                obj = cls._finfo_cache.get(dtype, None)
                if obj is not None:
                    # the original dtype was not in the cache, but the new
                    # dtype is in the cache. we add the original dtypes to
                    # the cache and return the result
                    for dt in dtypes:
                        cls._finfo_cache[dt] = obj
                    return obj
        obj = object.__new__(cls)._init(dtype)
        for dt in dtypes:
            cls._finfo_cache[dt] = obj
        return obj

    def _init(self, dtype):
        self.dtype = numeric.dtype(dtype)
        machar = _get_machar(dtype)

        for word in ['precision', 'iexp',
                     'maxexp', 'minexp', 'negep',
                     'machep']:
            setattr(self, word, getattr(machar, word))
        for word in ['resolution', 'epsneg', 'smallest_subnormal']:
            setattr(self, word, getattr(machar, word).flat[0])
        self.bits = self.dtype.itemsize * 8
        self.max = machar.huge.flat[0]
        self.min = -self.max
        self.eps = machar.eps.flat[0]
        self.nexp = machar.iexp
        self.nmant = machar.it
        self._machar = machar
        self._str_tiny = machar._str_xmin.strip()
        self._str_max = machar._str_xmax.strip()
        self._str_epsneg = machar._str_epsneg.strip()
        self._str_eps = machar._str_eps.strip()
        self._str_resolution = machar._str_resolution.strip()
        self._str_smallest_normal = machar._str_smallest_normal.strip()
        self._str_smallest_subnormal = machar._str_smallest_subnormal.strip()
        return self

    def __str__(self):
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'precision = %(precision)3s   resolution = %(_str_resolution)s\n'
            'machep = %(machep)6s   eps =        %(_str_eps)s\n'
            'negep =  %(negep)6s   epsneg =     %(_str_epsneg)s\n'
            'minexp = %(minexp)6s   tiny =       %(_str_tiny)s\n'
            'maxexp = %(maxexp)6s   max =        %(_str_max)s\n'
            'nexp =   %(nexp)6s   min =        -max\n'
            'smallest_normal = %(_str_smallest_normal)s   '
            'smallest_subnormal = %(_str_smallest_subnormal)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % self.__dict__

    def __repr__(self):
        c = self.__class__.__name__
        d = self.__dict__.copy()
        d['klass'] = c
        return (("%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s,"
                 " max=%(_str_max)s, dtype=%(dtype)s)") % d)

    @property
    def smallest_normal(self):
        """Return the value for the smallest normal.

        Returns
        -------
        smallest_normal : float
            Value for the smallest normal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest normal is requested for
            double-double.
        """
        # This check is necessary because the value for smallest_normal is
        # platform dependent for longdouble types.
        if isnan(self._machar.smallest_normal.flat[0]):
            warnings.warn(
                'The value of smallest normal is undefined for double double',
                UserWarning, stacklevel=2)
        return self._machar.smallest_normal.flat[0]

    @property
    def tiny(self):
        """Return the value for tiny, alias of smallest_normal.

        Returns
        -------
        tiny : float
            Value for the smallest normal, alias of smallest_normal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest normal is requested for
            double-double.
        """
        return self.smallest_normal


@set_module('numpy')
class iinfo:
    """
    iinfo(type)

    Machine limits for integer types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `iinfo` returns information.
    min : int
        The smallest integer expressible by the type.
    max : int
        The largest integer expressible by the type.

    Parameters
    ----------
    int_type : integer type, dtype, or instance
        The kind of integer data type to get information about.

    See Also
    --------
    finfo : The equivalent for floating point data types.

    Examples
    --------
    With types:

    >>> import numpy as np
    >>> ii16 = np.iinfo(np.int16)
    >>> ii16.min
    -32768
    >>> ii16.max
    32767
    >>> ii32 = np.iinfo(np.int32)
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    With instances:

    >>> ii32 = np.iinfo(np.int32(10))
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    """

    _min_vals = {}
    _max_vals = {}

    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, int_type):
        try:
            self.dtype = numeric.dtype(int_type)
        except TypeError:
            self.dtype = numeric.dtype(type(int_type))
        self.kind = self.dtype.kind
        self.bits = self.dtype.itemsize * 8
        self.key = "%s%d" % (self.kind, self.bits)
        if self.kind not in 'iu':
            raise ValueError("Invalid integer data type %r." % (self.kind,))

    @property
    def min(self):
        """Minimum value of given dtype."""
        if self.kind == 'u':
            return 0
        else:
            try:
                val = iinfo._min_vals[self.key]
            except KeyError:
                val = int(-(1 << (self.bits-1)))
                iinfo._min_vals[self.key] = val
            return val

    @property
    def max(self):
        """Maximum value of given dtype."""
        try:
            val = iinfo._max_vals[self.key]
        except KeyError:
            if self.kind == 'u':
                val = int((1 << self.bits) - 1)
            else:
                val = int((1 << (self.bits-1)) - 1)
            iinfo._max_vals[self.key] = val
        return val

    def __str__(self):
        """String representation."""
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'min = %(min)s\n'
            'max = %(max)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % {'dtype': self.dtype, 'min': self.min, 'max': self.max}

    def __repr__(self):
        return "%s(min=%s, max=%s, dtype=%s)" % (self.__class__.__name__,
                                    self.min, self.max, self.dtype)
<|endoftext|>"""
Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

"""
import os
import pickle
import warnings
from contextlib import nullcontext

import numpy as np
from numpy._core import multiarray as mu
from numpy._core import umath as um
from numpy._core.multiarray import asanyarray
from numpy._core import numerictypes as nt
from numpy._core import _exceptions
from numpy._globals import _NoValue

# save those O(100) nanoseconds!
bool_dt = mu.dtype("bool")
umr_maximum = um.maximum.reduce
umr_minimum = um.minimum.reduce
umr_sum = um.add.reduce
umr_prod = um.multiply.reduce
umr_bitwise_count = um.bitwise_count
umr_any = um.logical_or.reduce
umr_all = um.logical_and.reduce

# Complex types to -> (2,)float view for fast-path computation in _var()
_complex_to_float = {
    nt.dtype(nt.csingle) : nt.dtype(nt.single),
    nt.dtype(nt.cdouble) : nt.dtype(nt.double),
}
# Special case for windows: ensure double takes precedence
if nt.dtype(nt.longdouble) != nt.dtype(nt.double):
    _complex_to_float.update({
        nt.dtype(nt.clongdouble) : nt.dtype(nt.longdouble),
    })

# avoid keyword arguments to speed up parsing, saves about 15%-20% for very
# small reductions
def _amax(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_maximum(a, axis, None, out, keepdims, initial, where)

def _amin(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_minimum(a, axis, None, out, keepdims, initial, where)

def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         initial=_NoValue, where=True):
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)

def _prod(a, axis=None, dtype=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_prod(a, axis, dtype, out, keepdims, initial, where)

def _any(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # By default, return a boolean for any and all
    if dtype is None:
        dtype = bool_dt
    # Parsing keyword arguments is currently fairly slow, so avoid it for now
    if where is True:
        return umr_any(a, axis, dtype, out, keepdims)
    return umr_any(a, axis, dtype, out, keepdims, where=where)

def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # By default, return a boolean for any and all
    if dtype is None:
        dtype = bool_dt
    # Parsing keyword arguments is currently fairly slow, so avoid it for now
    if where is True:
        return umr_all(a, axis, dtype, out, keepdims)
    return umr_all(a, axis, dtype, out, keepdims, where=where)

def _count_reduce_items(arr, axis, keepdims=False, where=True):
    # fast-path for the default case
    if where is True:
        # no boolean mask given, calculate items according to axis
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
        items = nt.intp(items)
    else:
        # TODO: Optimize case when `where` is broadcast along a non-reduction
        # axis and full sum is more excessive than needed.

        # guarded to protect circular imports
        from numpy.lib._stride_tricks_impl import broadcast_to
        # count True values in (potentially broadcasted) boolean mask
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None,
                        keepdims)
    return items

def _clip(a, min=None, max=None, out=None, **kwargs):
    if a.dtype.kind in "iu":
        # If min/max is a Python integer, deal with out-of-bound values here.
        # (This enforces NEP 50 rules as no value based promotion is done.)
        if type(min) is int and min <= np.iinfo(a.dtype).min:
            min = None
        if type(max) is int and max >= np.iinfo(a.dtype).max:
            max = None

    if min is None and max is None:
        # return identity
        return um.positive(a, out=out, **kwargs)
    elif min is None:
        return um.minimum(a, max, out=out, **kwargs)
    elif max is None:
        return um.maximum(a, min, out=out, **kwargs)
    else:
        return um.clip(a, min, max, out=out, **kwargs)

def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    arr = asanyarray(a)

    is_float16_result = False

    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if rcount == 0 if where is True else umr_any(rcount == 0, axis=None):
        warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None:
        if issubclass(arr.dtype.type, (nt.integer, nt.bool)):
            dtype = mu.dtype('f8')
        elif issubclass(arr.dtype.type, nt.float16):
            dtype = mu.dtype('f4')
            is_float16_result = True

    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
        if is_float16_result and out is None:
            ret = arr.dtype.type(ret)
    elif hasattr(ret, 'dtype'):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True, mean=None):
    arr = asanyarray(a)

    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    # Make this warning show up on top.
    if ddof >= rcount if where is True else umr_any(ddof >= rcount, axis=None):
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                      stacklevel=2)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool)):
        dtype = mu.dtype('f8')

    if mean is not None:
        arrmean = mean
    else:
        # Compute the mean.
        # Note that if dtype is not of inexact type then arraymean will
        # not be either.
        arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
        # The shape of rcount has to match arrmean to not change the shape of
        # out in broadcasting. Otherwise, it cannot be stored back to arrmean.
        if rcount.ndim == 0:
            # fast-path for default case when where is True
            div = rcount
        else:
            # matching rcount to arrmean when where is specified as array
            div = rcount.reshape(arrmean.shape)
        if isinstance(arrmean, mu.ndarray):
            arrmean = um.true_divide(arrmean, div, out=arrmean,
                                     casting='unsafe', subok=False)
        elif hasattr(arrmean, "dtype"):
            arrmean = arrmean.dtype.type(arrmean / rcount)
        else:
            arrmean = arrmean / rcount

    # Compute sum of squared deviations from mean
    # Note that x may not be inexact and that we need it to be an array,
    # not a scalar.
    x = asanyarray(arr - arrmean)

    if issubclass(arr.dtype.type, (nt.floating, nt.integer)):
        x = um.multiply(x, x, out=x)
    # Fast-paths for built-in complex types
    elif x.dtype in _complex_to_float:
        xv = x.view(dtype=(_complex_to_float[x.dtype], (2,)))
        um.multiply(xv, xv, out=xv)
        x = um.add(xv[..., 0], xv[..., 1], out=x.real).real
    # Most general case; includes handling object arrays containing imaginary
    # numbers and complex types with non-native byteorder
    else:
        x = um.multiply(x, um.conjugate(x), out=x).real

    ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)

    # Compute degrees of freedom and make sure it is not negative.
    rcount = um.maximum(rcount - ddof, 0)

    # divide by degrees of freedom
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True, mean=None):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims, where=where, mean=mean)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)

    return ret

def _ptp(a, axis=None, out=None, keepdims=False):
    return um.subtract(
        umr_maximum(a, axis, None, out, keepdims),
        umr_minimum(a, axis, None, None, keepdims),
        out
    )

def _dump(self, file, protocol=2):
    if hasattr(file, 'write'):
        ctx = nullcontext(file)
    else:
        ctx = open(os.fspath(file), "wb")
    with ctx as f:
        pickle.dump(self, f, protocol=protocol)

def _dumps(self, protocol=2):
    return pickle.dumps(self, protocol=protocol)

def _bitwise_count(a, out=None, *, where=True, casting='same_kind',
          order='K', dtype=None, subok=True):
    return umr_bitwise_count(a, out, where=where, casting=casting,
            order=order, dtype=dtype, subok=subok)
<|endoftext|>"""
Create the numpy._core.multiarray namespace for backward compatibility.
In v1.16 the multiarray and umath c-extension modules were merged into
a single _multiarray_umath extension module. So we replicate the old
namespace by importing from the extension module.

"""

import functools
from . import overrides
from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
# These imports are needed for backward compatibility,
# do not change them. issue gh-15518
# _get_ndarray_c_version is semi-public, on purpose not added to __all__
from ._multiarray_umath import (
    _flagdict, from_dlpack, _place, _reconstruct,
    _vec_string, _ARRAY_API, _monotonicity, _get_ndarray_c_version,
    _get_madvise_hugepage, _set_madvise_hugepage,
    )

__all__ = [
    '_ARRAY_API', 'ALLOW_THREADS', 'BUFSIZE', 'CLIP', 'DATETIMEUNITS',
    'ITEM_HASOBJECT', 'ITEM_IS_POINTER', 'LIST_PICKLE', 'MAXDIMS',
    'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'NEEDS_INIT', 'NEEDS_PYAPI',
    'RAISE', 'USE_GETITEM', 'USE_SETITEM', 'WRAP',
    '_flagdict', 'from_dlpack', '_place', '_reconstruct', '_vec_string',
    '_monotonicity', 'add_docstring', 'arange', 'array', 'asarray',
    'asanyarray', 'ascontiguousarray', 'asfortranarray', 'bincount',
    'broadcast', 'busday_count', 'busday_offset', 'busdaycalendar', 'can_cast',
    'compare_chararrays', 'concatenate', 'copyto', 'correlate', 'correlate2',
    'count_nonzero', 'c_einsum', 'datetime_as_string', 'datetime_data',
    'dot', 'dragon4_positional', 'dragon4_scientific', 'dtype',
    'empty', 'empty_like', 'error', 'flagsobj', 'flatiter', 'format_longfloat',
    'frombuffer', 'fromfile', 'fromiter', 'fromstring',
    'get_handler_name', 'get_handler_version', 'inner', 'interp',
    'interp_complex', 'is_busday', 'lexsort', 'matmul', 'vecdot',
    'may_share_memory', 'min_scalar_type', 'ndarray', 'nditer', 'nested_iters',
    'normalize_axis_index', 'packbits', 'promote_types', 'putmask',
    'ravel_multi_index', 'result_type', 'scalar', 'set_datetimeparse_function',
    'set_typeDict', 'shares_memory', 'typeinfo',
    'unpackbits', 'unravel_index', 'vdot', 'where', 'zeros']

# For backward compatibility, make sure pickle imports
# these functions from here
_reconstruct.__module__ = 'numpy._core.multiarray'
scalar.__module__ = 'numpy._core.multiarray'


from_dlpack.__module__ = 'numpy'
arange.__module__ = 'numpy'
array.__module__ = 'numpy'
asarray.__module__ = 'numpy'
asanyarray.__module__ = 'numpy'
ascontiguousarray.__module__ = 'numpy'
asfortranarray.__module__ = 'numpy'
datetime_data.__module__ = 'numpy'
empty.__module__ = 'numpy'
frombuffer.__module__ = 'numpy'
fromfile.__module__ = 'numpy'
fromiter.__module__ = 'numpy'
frompyfunc.__module__ = 'numpy'
fromstring.__module__ = 'numpy'
may_share_memory.__module__ = 'numpy'
nested_iters.__module__ = 'numpy'
promote_types.__module__ = 'numpy'
zeros.__module__ = 'numpy'
normalize_axis_index.__module__ = 'numpy.lib.array_utils'
add_docstring.__module__ = 'numpy.lib'
compare_chararrays.__module__ = 'numpy.char'


def _override___module__():
    namespace_names = globals()
    for ufunc_name in [
        'absolute', 'arccos', 'arccosh', 'add', 'arcsin', 'arcsinh', 'arctan',
        'arctan2', 'arctanh', 'bitwise_and', 'bitwise_count', 'invert',
        'left_shift', 'bitwise_or', 'right_shift', 'bitwise_xor', 'cbrt',
        'ceil', 'conjugate', 'copysign', 'cos', 'cosh', 'deg2rad', 'degrees',
        'divide', 'divmod', 'equal', 'exp', 'exp2', 'expm1', 'fabs',
        'float_power', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod',
        'frexp', 'gcd', 'greater', 'greater_equal', 'heaviside', 'hypot',
        'isfinite', 'isinf', 'isnan', 'isnat', 'lcm', 'ldexp', 'less',
        'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp',
        'logaddexp2', 'logical_and', 'logical_not', 'logical_or',
        'logical_xor', 'matmul', 'matvec', 'maximum', 'minimum', 'remainder',
        'modf', 'multiply', 'negative', 'nextafter', 'not_equal', 'positive',
        'power', 'rad2deg', 'radians', 'reciprocal', 'rint', 'sign', 'signbit',
        'sin', 'sinh', 'spacing', 'sqrt', 'square', 'subtract', 'tan', 'tanh',
        'trunc', 'vecdot', 'vecmat',
    ]:
        ufunc = namespace_names[ufunc_name]
        ufunc.__module__ = "numpy"
        ufunc.__qualname__ = ufunc_name


_override___module__()


# We can't verify dispatcher signatures because NumPy's C functions don't
# support introspection.
array_function_from_c_func_and_dispatcher = functools.partial(
    overrides.array_function_from_dispatcher,
    module='numpy', docs_from_dispatcher=True, verify=False)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.empty_like)
def empty_like(
    prototype, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    """
    empty_like(prototype, dtype=None, order='K', subok=True, shape=None, *,
               device=None)

    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `prototype` is Fortran
        contiguous, 'C' otherwise. 'K' means match the layout of `prototype`
        as closely as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `prototype`, otherwise it will be a base-class array. Defaults
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
        Array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.

    Notes
    -----
    Unlike other array creation functions (e.g. `zeros_like`, `ones_like`,
    `full_like`), `empty_like` does not initialize the values of the array,
    and may therefore be marginally faster. However, the values stored in the
    newly allocated array are arbitrary. For reproducible behavior, be sure
    to set each element of the array before reading.

    Examples
    --------
    >>> import numpy as np
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    # uninitialized
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], # uninitialized
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])

    """   # NOQA
    return (prototype,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.concatenate)
def concatenate(arrays, axis=None, out=None, *, dtype=None, casting=None):
    """
    concatenate(
        (a1, a2, ...),
        axis=0,
        out=None,
        dtype=None,
        casting="same_kind"
    )

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.20.0

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.
        For a description of the options, please see :term:`casting`.

        .. versionadded:: 1.20.0

    Returns
    -------
    res : ndarray
        The concatenated array.

    See Also
    --------
    ma.concatenate : Concatenate function that preserves input masks.
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.
    split : Split array into a list of multiple sub-arrays of equal size.
    hsplit : Split array into multiple sub-arrays horizontally (column wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    stack : Stack a sequence of arrays along a new axis.
    block : Assemble arrays from blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    column_stack : Stack 1-D arrays as columns into a 2-D array.

    Notes
    -----
    When one or more of the arrays to be concatenated is a MaskedArray,
    this function will return a MaskedArray object instead of an ndarray,
    but the input masks are *not* preserved. In cases where a MaskedArray
    is expected as input, use the ma.concatenate function from the masked
    array module instead.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])
    >>> np.concatenate((a, b), axis=None)
    array([1, 2, 3, 4, 5, 6])

    This function will not preserve masking of MaskedArray inputs.

    >>> a = np.ma.arange(3)
    >>> a[1] = np.ma.masked
    >>> b = np.arange(2, 5)
    >>> a
    masked_array(data=[0, --, 2],
                 mask=[False,  True, False],
           fill_value=999999)
    >>> b
    array([2, 3, 4])
    >>> np.concatenate([a, b])
    masked_array(data=[0, 1, 2, 2, 3, 4],
                 mask=False,
           fill_value=999999)
    >>> np.ma.concatenate([a, b])
    masked_array(data=[0, --, 2, 2, 3, 4],
                 mask=[False,  True, False, False, False, False],
           fill_value=999999)

    """
    if out is not None:
        # optimize for the typical case where only arrays is provided
        arrays = list(arrays)
        arrays.append(out)
    return arrays


@array_function_from_c_func_and_dispatcher(_multiarray_umath.inner)
def inner(a, b):
    """
    inner(a, b, /)

    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : array_like
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : ndarray
        If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        ``out.shape = (*a.shape[:-1], *b.shape[:-1])``

    Raises
    ------
    ValueError
        If both `a` and `b` are nonscalar and their last dimensions have
        different sizes.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    vecdot : Vector dot product of two arrays.
    einsum : Einstein summation convention.

    Notes
    -----
    For vectors (1-D arrays) it computes the ordinary inner-product::

        np.inner(a, b) = sum(a[:]*b[:])

    More generally, if ``ndim(a) = r > 0`` and ``ndim(b) = s > 0``::

        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))

    or explicitly::

        np.inner(a, b)[i0,...,ir-2,j0,...,js-2]
             = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])

    In addition `a` or `b` may be scalars, in which case::

       np.inner(a,b) = a*b

    Examples
    --------
    Ordinary inner product for vectors:

    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2

    Some multidimensional examples:

    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> c = np.inner(a, b)
    >>> c.shape
    (2, 3)
    >>> c
    array([[ 14,  38,  62],
           [ 86, 110, 134]])

    >>> a = np.arange(2).reshape((1,1,2))
    >>> b = np.arange(6).reshape((3,2))
    >>> c = np.inner(a, b)
    >>> c.shape
    (1, 1, 3)
    >>> c
    array([[[1, 3, 5]]])

    An example where `b` is a scalar:

    >>> np.inner(np.eye(2), 7)
    array([[7., 0.],
           [0., 7.]])

    """
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.where)
def where(condition, x=None, y=None):
    """
    where(condition, [x, y], /)

    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only `condition` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
        preferred, as it behaves correctly for subclasses. The rest of this
        documentation covers only the case where all three arguments are
        provided.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : ndarray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    See Also
    --------
    choose
    nonzero : The function that is called when x and y are omitted

    Notes
    -----
    If all the arrays are 1-D, `where` is equivalent to::

        [xv if c else yv
         for c, xv, yv in zip(condition, x, y)]

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    This can be used on multidimensional arrays too:

    >>> np.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    array([[1, 8],
           [3, 4]])

    The shapes of x, y, and the condition are broadcast together:

    >>> x, y = np.ogrid[:3, :4]
    >>> np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
    array([[10,  0,  0,  0],
           [10, 11,  1,  1],
           [10, 11, 12,  2]])

    >>> a = np.array([[0, 1, 2],
    ...               [0, 2, 4],
    ...               [0, 3, 6]])
    >>> np.where(a < 4, a, -1)  # -1 is broadcast
    array([[ 0,  1,  2],
           [ 0,  2, -1],
           [ 0,  3, -1]])
    """
    return (condition, x, y)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.lexsort)
def lexsort(keys, axis=None):
    """
    lexsort(keys, axis=-1)

    Perform an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, lexsort returns an array of integer indices
    that describes the sort order by multiple keys. The last key in the
    sequence is used for the primary sort order, ties are broken by the
    second-to-last key, and so on.

    Parameters
    ----------
    keys : (k, m, n, ...) array-like
        The `k` keys to be sorted. The *last* key (e.g, the last
        row if `keys` is a 2D array) is the primary sort key.
        Each element of `keys` along the zeroth axis must be
        an array-like object of the same shape.
    axis : int, optional
        Axis to be indirectly sorted. By default, sort over the last axis
        of each sequence. Separate slices along `axis` sorted over
        independently; see last example.

    Returns
    -------
    indices : (m, n, ...) ndarray of ints
        Array of indices that sort the keys along the specified axis.

    See Also
    --------
    argsort : Indirect sort.
    ndarray.sort : In-place sort.
    sort : Return a sorted copy of an array.

    Examples
    --------
    Sort names: first by surname, then by name.

    >>> import numpy as np
    >>> surnames =    ('Hertz',    'Galilei', 'Hertz')
    >>> first_names = ('Heinrich', 'Galileo', 'Gustav')
    >>> ind = np.lexsort((first_names, surnames))
    >>> ind
    array([1, 2, 0])

    >>> [surnames[i] + ", " + first_names[i] for i in ind]
    ['Galilei, Galileo', 'Hertz, Gustav', 'Hertz, Heinrich']

    Sort according to two numerical keys, first by elements
    of ``a``, then breaking ties according to elements of ``b``:

    >>> a = [1, 5, 1, 4, 3, 4, 4]  # First sequence
    >>> b = [9, 4, 0, 4, 0, 2, 1]  # Second sequence
    >>> ind = np.lexsort((b, a))  # Sort by `a`, then by `b`
    >>> ind
    array([2, 0, 4, 6, 5, 3, 1])
    >>> [(a[i], b[i]) for i in ind]
    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]

    Compare against `argsort`, which would sort each key independently.

    >>> np.argsort((b, a), kind='stable')
    array([[2, 4, 6, 5, 1, 3, 0],
           [0, 2, 4, 3, 5, 6, 1]])

    To sort lexicographically with `argsort`, we would need to provide a
    structured array.

    >>> x = np.array([(ai, bi) for ai, bi in zip(a, b)],
    ...              dtype = np.dtype([('x', int), ('y', int)]))
    >>> np.argsort(x)  # or np.argsort(x, order=('x', 'y'))
    array([2, 0, 4, 6, 5, 3, 1])

    The zeroth axis of `keys` always corresponds with the sequence of keys,
    so 2D arrays are treated just like other sequences of keys.

    >>> arr = np.asarray([b, a])
    >>> ind2 = np.lexsort(arr)
    >>> np.testing.assert_equal(ind2, ind)

    Accordingly, the `axis` parameter refers to an axis of *each* key, not of
    the `keys` argument itself. For instance, the array ``arr`` is treated as
    a sequence of two 1-D keys, so specifying ``axis=0`` is equivalent to
    using the default axis, ``axis=-1``.

    >>> np.testing.assert_equal(np.lexsort(arr, axis=0),
    ...                         np.lexsort(arr, axis=-1))

    For higher-dimensional arrays, the axis parameter begins to matter. The
    resulting array has the same shape as each key, and the values are what
    we would expect if `lexsort` were performed on corresponding slices
    of the keys independently. For instance,

    >>> x = [[1, 2, 3, 4],
    ...      [4, 3, 2, 1],
    ...      [2, 1, 4, 3]]
    >>> y = [[2, 2, 1, 1],
    ...      [1, 2, 1, 2],
    ...      [1, 1, 2, 1]]
    >>> np.lexsort((x, y), axis=1)
    array([[2, 3, 0, 1],
           [2, 0, 3, 1],
           [1, 0, 3, 2]])

    Each row of the result is what we would expect if we were to perform
    `lexsort` on the corresponding row of the keys:

    >>> for i in range(3):
    ...     print(np.lexsort((x[i], y[i])))
    [2 3 0 1]
    [2 0 3 1]
    [1 0 3 2]

    """
    if isinstance(keys, tuple):
        return keys
    else:
        return (keys,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.can_cast)
def can_cast(from_, to, casting=None):
    """
    can_cast(from_, to, casting='safe')

    Returns True if cast between data types can occur according to the
    casting rule.

    Parameters
    ----------
    from_ : dtype, dtype specifier, NumPy scalar, or array
        Data type, NumPy scalar, or array to cast from.
    to : dtype or dtype specifier
        Data type to cast to.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.

    Returns
    -------
    out : bool
        True if cast can occur according to the casting rule.

    Notes
    -----
    .. versionchanged:: 2.0
       This function does not support Python scalars anymore and does not
       apply any value-based logic for 0-D arrays and NumPy scalars.

    See also
    --------
    dtype, result_type

    Examples
    --------
    Basic examples

    >>> import numpy as np
    >>> np.can_cast(np.int32, np.int64)
    True
    >>> np.can_cast(np.float64, complex)
    True
    >>> np.can_cast(complex, float)
    False

    >>> np.can_cast('i8', 'f8')
    True
    >>> np.can_cast('i8', 'f4')
    False
    >>> np.can_cast('i4', 'S4')
    False

    """
    return (from_,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.min_scalar_type)
def min_scalar_type(a):
    """
    min_scalar_type(a, /)

    For scalar ``a``, returns the data type with the smallest size
    and smallest scalar kind which can hold its value.  For non-scalar
    array ``a``, returns the vector's dtype unmodified.

    Floating point values are not demoted to integers,
    and complex values are not demoted to floats.

    Parameters
    ----------
    a : scalar or array_like
        The value whose minimal data type is to be found.

    Returns
    -------
    out : dtype
        The minimal data type.

    See Also
    --------
    result_type, promote_types, dtype, can_cast

    Examples
    --------
    >>> import numpy as np
    >>> np.min_scalar_type(10)
    dtype('uint8')

    >>> np.min_scalar_type(-260)
    dtype('int16')

    >>> np.min_scalar_type(3.1)
    dtype('float16')

    >>> np.min_scalar_type(1e50)
    dtype('float64')

    >>> np.min_scalar_type(np.arange(4,dtype='f8'))
    dtype('float64')

    """
    return (a,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.result_type)
def result_type(*arrays_and_dtypes):
    """
    result_type(*arrays_and_dtypes)

    Returns the type that results from applying the NumPy
    type promotion rules to the arguments.

    Type promotion in NumPy works similarly to the rules in languages
    like C++, with some slight differences.  When both scalars and
    arrays are used, the array's type takes precedence and the actual value
    of the scalar is taken into account.

    For example, calculating 3*a, where a is an array of 32-bit floats,
    intuitively should result in a 32-bit float output.  If the 3 is a
    32-bit integer, the NumPy rules indicate it can't convert losslessly
    into a 32-bit float, so a 64-bit float should be the result type.
    By examining the value of the constant, '3', we see that it fits in
    an 8-bit integer, which can be cast losslessly into the 32-bit float.

    Parameters
    ----------
    arrays_and_dtypes : list of arrays and dtypes
        The operands of some operation whose result type is needed.

    Returns
    -------
    out : dtype
        The result type.

    See also
    --------
    dtype, promote_types, min_scalar_type, can_cast

    Notes
    -----
    The specific algorithm used is as follows.

    Categories are determined by first checking which of boolean,
    integer (int/uint), or floating point (float/complex) the maximum
    kind of all the arrays and the scalars are.

    If there are only scalars or the maximum category of the scalars
    is higher than the maximum category of the arrays,
    the data types are combined with :func:`promote_types`
    to produce the return value.

    Otherwise, `min_scalar_type` is called on each scalar, and
    the resulting data types are all combined with :func:`promote_types`
    to produce the return value.

    The set of int values is not a subset of the uint values for types
    with the same number of bits, something not reflected in
    :func:`min_scalar_type`, but handled as a special case in `result_type`.

    Examples
    --------
    >>> import numpy as np
    >>> np.result_type(3, np.arange(7, dtype='i1'))
    dtype('int8')

    >>> np.result_type('i4', 'c8')
    dtype('complex128')

    >>> np.result_type(3.0, -2)
    dtype('float64')

    """
    return arrays_and_dtypes


@array_function_from_c_func_and_dispatcher(_multiarray_umath.dot)
def dot(a, b, out=None):
    """
    dot(a, b, out=None)

    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to
      :func:`multiply` and using ``numpy.multiply(a, b)`` or ``a * b`` is
      preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of
      `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    It uses an optimized BLAS library when possible (see `numpy.linalg`).

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    vecdot : Vector dot product of two arrays.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.
    matmul : '@' operator as method with out parameter.
    linalg.multi_dot : Chained dot product.

    Examples
    --------
    >>> import numpy as np
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it is the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128

    """
    return (a, b, out)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.vdot)
def vdot(a, b):
    r"""
    vdot(a, b, /)

    Return the dot product of two vectors.

    The `vdot` function handles complex numbers differently than `dot`:
    if the first argument is complex, it is replaced by its complex conjugate
    in the dot product calculation. `vdot` also handles multidimensional
    arrays differently than `dot`: it does not perform a matrix product, but
    flattens the arguments to 1-D arrays before taking a vector dot product.

    Consequently, when the arguments are 2-D arrays of the same shape, this
    function effectively returns their
    `Frobenius inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`_
    (also known as the *trace inner product* or the *standard inner product*
    on a vector space of matrices).

    Parameters
    ----------
    a : array_like
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : array_like
        Second argument to the dot product.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`.  Can be an int, float, or
        complex depending on the types of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
          first argument.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+2j,3+4j])
    >>> b = np.array([5+6j,7+8j])
    >>> np.vdot(a, b)
    (70-8j)
    >>> np.vdot(b, a)
    (70+8j)

    Note that higher-dimensional arrays are flattened!

    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.vdot(a, b)
    30
    >>> np.vdot(b, a)
    30
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30

    """  # noqa: E501
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.bincount)
def bincount(x, weights=None, minlength=None):
    """
    bincount(x, /, weights=None, minlength=0)

    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : ndarray of ints
        The result of binning the input array.
        The length of `out` is equal to ``np.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    histogram, digitize, unique

    Examples
    --------
    >>> import numpy as np
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x)+1
    True

    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:

    >>> np.bincount(np.arange(5, dtype=float))
    Traceback (most recent call last):
      ...
    TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
    according to the rule 'safe'

    A possible use of ``bincount`` is to perform sums over
    variable-size chunks of an array, using the ``weights`` keyword.

    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = np.array([0, 1, 1, 2, 2, 2])
    >>> np.bincount(x,  weights=w)
    array([ 0.3,  0.7,  1.1])

    """
    return (x, weights)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode=None, order=None):
    """
    ravel_multi_index(multi_index, dims, mode='raise', order='C')

    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : ndarray
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[3,6,6],[4,5,1]])
    >>> np.ravel_multi_index(arr, (7,6))
    array([22, 41, 37])
    >>> np.ravel_multi_index(arr, (7,6), order='F')
    array([31, 41, 13])
    >>> np.ravel_multi_index(arr, (4,6), mode='clip')
    array([22, 23, 19])
    >>> np.ravel_multi_index(arr, (4,4), mode=('clip','wrap'))
    array([12, 13, 13])

    >>> np.ravel_multi_index((3,1,4,1), (6,7,8,9))
    1621
    """
    return multi_index


@array_function_from_c_func_and_dispatcher(_multiarray_umath.unravel_index)
def unravel_index(indices, shape=None, order=None):
    """
    unravel_index(indices, shape, order='C')

    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened
        version of an array of dimensions ``shape``. Before version 1.6.0,
        this function accepted just one index value.
    shape : tuple of ints
        The shape of the array to use for unraveling ``indices``.
    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the ``indices``
        array.

    See Also
    --------
    ravel_multi_index

    Examples
    --------
    >>> import numpy as np
    >>> np.unravel_index([22, 41, 37], (7,6))
    (array([3, 6, 6]), array([4, 5, 1]))
    >>> np.unravel_index([31, 41, 13], (7,6), order='F')
    (array([3, 6, 6]), array([4, 5, 1]))

    >>> np.unravel_index(1621, (6,7,8,9))
    (3, 1, 4, 1)

    """
    return (indices,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.copyto)
def copyto(dst, src, casting=None, where=None):
    """
    copyto(dst, src, casting='same_kind', where=True)

    Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    Parameters
    ----------
    dst : ndarray
        The array into which values are copied.
    src : array_like
        The array from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions
        of `dst`, and selects elements to copy from `src` to `dst`
        wherever it contains the value True.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([4, 5, 6])
    >>> B = [1, 2, 3]
    >>> np.copyto(A, B)
    >>> A
    array([1, 2, 3])

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> B = [[4, 5, 6], [7, 8, 9]]
    >>> np.copyto(A, B)
    >>> A
    array([[4, 5, 6],
           [7, 8, 9]])

    """
    return (dst, src, where)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.putmask)
def putmask(a, /, mask, values):
    """
    putmask(a, mask, values)

    Changes elements of an array based on conditional and input values.

    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.

    If `values` is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from ``a[mask] = values``.

    Parameters
    ----------
    a : ndarray
        Target array.
    mask : array_like
        Boolean mask array. It has to be the same shape as `a`.
    values : array_like
        Values to put into `a` where `mask` is True. If `values` is smaller
        than `a` it will be repeated.

    See Also
    --------
    place, put, take, copyto

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> np.putmask(x, x>2, x**2)
    >>> x
    array([[ 0,  1,  2],
           [ 9, 16, 25]])

    If `values` is smaller than `a` it is repeated:

    >>> x = np.arange(5)
    >>> np.putmask(x, x>1, [-33, -44])
    >>> x
    array([  0,   1, -33, -44, -33])

    """
    return (a, mask, values)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.packbits)
def packbits(a, axis=None, bitorder='big'):
    """
    packbits(a, /, axis=None, bitorder='big')

    Packs the elements of a binary-valued array into bits in a uint8 array.

    The result is padded to full bytes by inserting zero bits at the end.

    Parameters
    ----------
    a : array_like
        An array of integers or booleans whose elements should be packed to
        bits.
    axis : int, optional
        The dimension over which bit-packing is done.
        ``None`` implies packing the flattened array.
    bitorder : {'big', 'little'}, optional
        The order of the input bits. 'big' will mimic bin(val),
        ``[0, 0, 0, 0, 0, 0, 1, 1] => 3 = 0b00000011``, 'little' will
        reverse the order so ``[1, 1, 0, 0, 0, 0, 0, 0] => 3``.
        Defaults to 'big'.

    Returns
    -------
    packed : ndarray
        Array of type uint8 whose elements represent bits corresponding to the
        logical (0 or nonzero) value of the input elements. The shape of
        `packed` has the same number of dimensions as the input (unless `axis`
        is None, in which case the output is 1-D).

    See Also
    --------
    unpackbits: Unpacks elements of a uint8 array into a binary-valued output
                array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[[1,0,1],
    ...                [0,1,0]],
    ...               [[1,1,0],
    ...                [0,0,1]]])
    >>> b = np.packbits(a, axis=-1)
    >>> b
    array([[[160],
            [ 64]],
           [[192],
            [ 32]]], dtype=uint8)

    Note that in binary 160 = 1010 0000, 64 = 0100 0000, 192 = 1100 0000,
    and 32 = 0010 0000.

    """
    return (a,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.unpackbits)
def unpackbits(a, axis=None, count=None, bitorder='big'):
    """
    unpackbits(a, /, axis=None, count=None, bitorder='big')

    Unpacks elements of a uint8 array into a binary-valued output array.

    Each element of `a` represents a bit-field that should be unpacked
    into a binary-valued output array. The shape of the output array is
    either 1-D (if `axis` is ``None``) or the same shape as the input
    array with unpacking done along the axis specified.

    Parameters
    ----------
    a : ndarray, uint8 type
       Input array.
    axis : int, optional
        The dimension over which bit-unpacking is done.
        ``None`` implies unpacking the flattened array.
    count : int or None, optional
        The number of elements to unpack along `axis`, provided as a way
        of undoing the effect of packing a size that is not a multiple
        of eight. A non-negative number means to only unpack `count`
        bits. A negative number means to trim off that many bits from
        the end. ``None`` means to unpack the entire array (the
        default). Counts larger than the available number of bits will
        add zero padding to the output. Negative counts must not
        exceed the available number of bits.
    bitorder : {'big', 'little'}, optional
        The order of the returned bits. 'big' will mimic bin(val),
        ``3 = 0b00000011 => [0, 0, 0, 0, 0, 0, 1, 1]``, 'little' will reverse
        the order to ``[1, 1, 0, 0, 0, 0, 0, 0]``.
        Defaults to 'big'.

    Returns
    -------
    unpacked : ndarray, uint8 type
       The elements are binary-valued (0 or 1).

    See Also
    --------
    packbits : Packs the elements of a binary-valued array into bits in
               a uint8 array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[2], [7], [23]], dtype=np.uint8)
    >>> a
    array([[ 2],
           [ 7],
           [23]], dtype=uint8)
    >>> b = np.unpackbits(a, axis=1)
    >>> b
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)
    >>> c = np.unpackbits(a, axis=1, count=-3)
    >>> c
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0]], dtype=uint8)

    >>> p = np.packbits(b, axis=0)
    >>> np.unpackbits(p, axis=0)
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> np.array_equal(b, np.unpackbits(p, axis=0, count=b.shape[0]))
    True

    """
    return (a,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.shares_memory)
def shares_memory(a, b, max_work=None):
    """
    shares_memory(a, b, /, max_work=None)

    Determine if two arrays share memory.

    .. warning::

       This function can be exponentially slow for some inputs, unless
       `max_work` is set to zero or a positive integer.
       If in doubt, use `numpy.may_share_memory` instead.

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    max_work : int, optional
        Effort to spend on solving the overlap problem (maximum number
        of candidate solutions to consider). The following special
        values are recognized:

        max_work=-1 (default)
            The problem is solved exactly. In this case, the function returns
            True only if there is an element shared between the arrays. Finding
            the exact solution may take extremely long in some cases.
        max_work=0
            Only the memory bounds of a and b are checked.
            This is equivalent to using ``may_share_memory()``.

    Raises
    ------
    numpy.exceptions.TooHardError
        Exceeded max_work.

    Returns
    -------
    out : bool

    See Also
    --------
    may_share_memory

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> np.shares_memory(x, np.array([5, 6, 7]))
    False
    >>> np.shares_memory(x[::2], x)
    True
    >>> np.shares_memory(x[::2], x[1::2])
    False

    Checking whether two arrays share memory is NP-complete, and
    runtime may increase exponentially in the number of
    dimensions. Hence, `max_work` should generally be set to a finite
    number, as it is possible to construct examples that take
    extremely long to run:

    >>> from numpy.lib.stride_tricks import as_strided
    >>> x = np.zeros([192163377], dtype=np.int8)
    >>> x1 = as_strided(
    ...     x, strides=(36674, 61119, 85569), shape=(1049, 1049, 1049))
    >>> x2 = as_strided(
    ...     x[64023025:], strides=(12223, 12224, 1), shape=(1049, 1049, 1))
    >>> np.shares_memory(x1, x2, max_work=1000)
    Traceback (most recent call last):
    ...
    numpy.exceptions.TooHardError: Exceeded max_work

    Running ``np.shares_memory(x1, x2)`` without `max_work` set takes
    around 1 minute for this case. It is possible to find problems
    that take still significantly longer.

    """
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.may_share_memory)
def may_share_memory(a, b, max_work=None):
    """
    may_share_memory(a, b, /, max_work=None)

    Determine if two arrays might share memory

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.

    Only the memory bounds of a and b are checked by default.

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    max_work : int, optional
        Effort to spend on solving the overlap problem.  See
        `shares_memory` for details.  Default for ``may_share_memory``
        is to do a bounds check.

    Returns
    -------
    out : bool

    See Also
    --------
    shares_memory

    Examples
    --------
    >>> import numpy as np
    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
    False
    >>> x = np.zeros([3, 4])
    >>> np.may_share_memory(x[:,0], x[:,1])
    True

    """
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.is_busday)
def is_busday(dates, weekmask=None, holidays=None, busdaycal=None, out=None):
    """
    is_busday(
        dates,
        weekmask='1111100',
        holidays=None,
        busdaycal=None,
        out=None
    )

    Calculates which of the given dates are valid days, and which are not.

    Parameters
    ----------
    dates : array_like of datetime64[D]
        The array of dates to process.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of bool, optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of bool
        An array with the same shape as ``dates``, containing True for
        each valid day, and False for each invalid day.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    busday_offset : Applies an offset counted in valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Examples
    --------
    >>> import numpy as np
    >>> # The weekdays are Friday, Saturday, and Monday
    ... np.is_busday(['2011-07-01', '2011-07-02', '2011-07-18'],
    ...                 holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
    array([False, False,  True])
    """
    return (dates, weekmask, holidays, out)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_offset)
def busday_offset(dates, offsets, roll=None, weekmask=None, holidays=None,
                  busdaycal=None, out=None):
    """
    busday_offset(
        dates,
        offsets,
        roll='raise',
        weekmask='1111100',
        holidays=None,
        busdaycal=None,
        out=None
    )

    First adjusts the date to fall on a valid day according to
    the ``roll`` rule, then applies offsets to the given dates
    counted in valid days.

    Parameters
    ----------
    dates : array_like of datetime64[D]
        The array of dates to process.
    offsets : array_like of int
        The array of offsets, which is broadcast with ``dates``.
    roll : {'raise', 'nat', 'forward', 'following', 'backward', 'preceding', \
        'modifiedfollowing', 'modifiedpreceding'}, optional
        How to treat dates that do not fall on a valid day. The default
        is 'raise'.

        * 'raise' means to raise an exception for an invalid day.
        * 'nat' means to return a NaT (not-a-time) for an invalid day.
        * 'forward' and 'following' mean to take the first valid day
          later in time.
        * 'backward' and 'preceding' mean to take the first valid day
          earlier in time.
        * 'modifiedfollowing' means to take the first valid day
          later in time unless it is across a Month boundary, in which
          case to take the first valid day earlier in time.
        * 'modifiedpreceding' means to take the first valid day
          earlier in time unless it is across a Month boundary, in which
          case to take the first valid day later in time.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of datetime64[D], optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of datetime64[D]
        An array with a shape from broadcasting ``dates`` and ``offsets``
        together, containing the dates with offsets applied.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    is_busday : Returns a boolean array indicating valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Examples
    --------
    >>> import numpy as np
    >>> # First business day in October 2011 (not accounting for holidays)
    ... np.busday_offset('2011-10', 0, roll='forward')
    np.datetime64('2011-10-03')
    >>> # Last business day in February 2012 (not accounting for holidays)
    ... np.busday_offset('2012-03', -1, roll='forward')
    np.datetime64('2012-02-29')
    >>> # Third Wednesday in January 2011
    ... np.busday_offset('2011-01', 2, roll='forward', weekmask='Wed')
    np.datetime64('2011-01-19')
    >>> # 2012 Mother's Day in Canada and the U.S.
    ... np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    np.datetime64('2012-05-13')

    >>> # First business day on or after a date
    ... np.busday_offset('2011-03-20', 0, roll='forward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 0, roll='forward')
    np.datetime64('2011-03-22')
    >>> # First business day after a date
    ... np.busday_offset('2011-03-20', 1, roll='backward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 1, roll='backward')
    np.datetime64('2011-03-23')
    """
    return (dates, offsets, weekmask, holidays, out)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_count)
def busday_count(begindates, enddates, weekmask=None, holidays=None,
                 busdaycal=None, out=None):
    """
    busday_count(
        begindates,
        enddates,
        weekmask='1111100',
        holidays=[],
        busdaycal=None,
        out=None
    )

    Counts the number of valid days between `begindates` and
    `enddates`, not including the day of `enddates`.

    If ``enddates`` specifies a date value that is earlier than the
    corresponding ``begindates`` date value, the count will be negative.

    Parameters
    ----------
    begindates : array_like of datetime64[D]
        The array of the first dates for counting.
    enddates : array_like of datetime64[D]
        The array of the end dates for counting, which are excluded
        from the count themselves.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of int, optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of int
        An array with a shape from broadcasting ``begindates`` and ``enddates``
        together, containing the number of valid days between
        the begin and end dates.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    is_busday : Returns a boolean array indicating valid days.
    busday_offset : Applies an offset counted in valid days.

    Examples
    --------
    >>> import numpy as np
    >>> # Number of weekdays in January 2011
    ... np.busday_count('2011-01', '2011-02')
    21
    >>> # Number of weekdays in 2011
    >>> np.busday_count('2011', '2012')
    260
    >>> # Number of Saturdays in 2011
    ... np.busday_count('2011', '2012', weekmask='Sat')
    53
    """
    return (begindates, enddates, weekmask, holidays, out)


@array_function_from_c_func_and_dispatcher(
    _multiarray_umath.datetime_as_string)
def datetime_as_string(arr, unit=None, timezone=None, casting=None):
    """
    datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind')

    Convert an array of datetimes into an array of strings.

    Parameters
    ----------
    arr : array_like of datetime64
        The array of UTC timestamps to format.
    unit : str
        One of None, 'auto', or
        a :ref:`datetime unit <arrays.dtypes.dateunits>`.
    timezone : {'naive', 'UTC', 'local'} or tzinfo
        Timezone information to use when displaying the datetime. If 'UTC',
        end with a Z to indicate UTC time. If 'local', convert to the local
        timezone first, and suffix with a +-#### timezone offset. If a tzinfo
        object, then do as with 'local', but use the specified timezone.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}
        Casting to allow when changing between datetime units.

    Returns
    -------
    str_arr : ndarray
        An array of strings the same shape as `arr`.

    Examples
    --------
    >>> import numpy as np
    >>> import pytz
    >>> d = np.arange('2002-10-27T04:30', 4*60, 60, dtype='M8[m]')
    >>> d
    array(['2002-10-27T04:30', '2002-10-27T05:30', '2002-10-27T06:30',
           '2002-10-27T07:30'], dtype='datetime64[m]')

    Setting the timezone to UTC shows the same information, but with a Z suffix

    >>> np.datetime_as_string(d, timezone='UTC')
    array(['2002-10-27T04:30Z', '2002-10-27T05:30Z', '2002-10-27T06:30Z',
           '2002-10-27T07:30Z'], dtype='<U35')

    Note that we picked datetimes that cross a DST boundary. Passing in a
    ``pytz`` timezone object will print the appropriate offset

    >>> np.datetime_as_string(d, timezone=pytz.timezone('US/Eastern'))
    array(['2002-10-27T00:30-0400', '2002-10-27T01:30-0400',
           '2002-10-27T01:30-0500', '2002-10-27T02:30-0500'], dtype='<U39')

    Passing in a unit will change the precision

    >>> np.datetime_as_string(d, unit='h')
    array(['2002-10-27T04', '2002-10-27T05', '2002-10-27T06', '2002-10-27T07'],
          dtype='<U32')
    >>> np.datetime_as_string(d, unit='s')
    array(['2002-10-27T04:30:00', '2002-10-27T05:30:00', '2002-10-27T06:30:00',
           '2002-10-27T07:30:00'], dtype='<U38')

    'casting' can be used to specify whether precision can be changed

    >>> np.datetime_as_string(d, unit='h', casting='safe')
    Traceback (most recent call last):
        ...
    TypeError: Cannot create a datetime string as units 'h' from a NumPy
    datetime with units 'm' according to the rule 'safe'
    """
    return (arr,)
<|endoftext|>"""
String-handling utilities to avoid locale-dependence.

Used primarily to generate type name aliases.
"""
# "import string" is costly to import!
# Construct the translation tables directly
#   "A" = chr(65), "a" = chr(97)
_all_chars = tuple(map(chr, range(256)))
_ascii_upper = _all_chars[65:65+26]
_ascii_lower = _all_chars[97:97+26]
LOWER_TABLE = _all_chars[:65] + _ascii_lower + _all_chars[65+26:]
UPPER_TABLE = _all_chars[:97] + _ascii_upper + _all_chars[97+26:]


def english_lower(s):
    """ Apply English case rules to convert ASCII strings to all lower case.

    This is an internal utility function to replace calls to str.lower() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "I".lower() != "i" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    lowered : str

    Examples
    --------
    >>> from numpy._core.numerictypes import english_lower
    >>> english_lower('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
    'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_lower('')
    ''
    """
    lowered = s.translate(LOWER_TABLE)
    return lowered


def english_upper(s):
    """ Apply English case rules to convert ASCII strings to all upper case.

    This is an internal utility function to replace calls to str.upper() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    uppered : str

    Examples
    --------
    >>> from numpy._core.numerictypes import english_upper
    >>> english_upper('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


def english_capitalize(s):
    """ Apply English case rules to convert the first character of an ASCII
    string to upper case.

    This is an internal utility function to replace calls to str.capitalize()
    such that we can avoid changing behavior with changing locales.

    Parameters
    ----------
    s : str

    Returns
    -------
    capitalized : str

    Examples
    --------
    >>> from numpy._core.numerictypes import english_capitalize
    >>> english_capitalize('int8')
    'Int8'
    >>> english_capitalize('Int8')
    'Int8'
    >>> english_capitalize('')
    ''
    """
    if s:
        return english_upper(s[0]) + s[1:]
    else:
        return s
<|endoftext|>"""Simple script to compute the api hash of the current API.

The API has is defined by numpy_api_order and ufunc_api_order.

"""
from os.path import dirname

from code_generators.genapi import fullapi_hash
from code_generators.numpy_api import full_api

if __name__ == '__main__':
    curdir = dirname(__file__)
    print(fullapi_hash(full_api))
<|endoftext|>import functools
import warnings
import operator
import types

import numpy as np
from . import numeric as _nx
from .numeric import result_type, nan, asanyarray, ndim
from numpy._core.multiarray import add_docstring
from numpy._core._multiarray_umath import _array_converter
from numpy._core import overrides

__all__ = ['logspace', 'linspace', 'geomspace']


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _linspace_dispatcher(start, stop, num=None, endpoint=None, retstep=None,
                         dtype=None, axis=None, *, device=None):
    return (start, stop)


@array_function_dispatch(_linspace_dispatcher)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0, *, device=None):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    .. versionchanged:: 1.20.0
        Values are rounded towards ``-inf`` instead of ``0`` when an
        integer ``dtype`` is specified. The old behavior can
        still be obtained with ``np.linspace(start, stop, num).astype(int)``

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).
    geomspace : Similar to `linspace`, but with numbers spaced evenly on a log
                scale (a geometric progression).
    logspace : Similar to `geomspace`, but with the end points specified as
               logarithms.
    :ref:`how-to-partition`

    Examples
    --------
    >>> import numpy as np
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    num = operator.index(num)
    if num < 0:
        raise ValueError(
            "Number of samples, %s, must be non-negative." % num
        )
    div = (num - 1) if endpoint else num

    conv = _array_converter(start, stop)
    start, stop = conv.as_arrays()
    dt = conv.result_type(ensure_inexact=True)

    if dtype is None:
        dtype = dt
        integer_dtype = False
    else:
        integer_dtype = _nx.issubdtype(dtype, _nx.integer)

    # Use `dtype=type(dt)` to enforce a floating point evaluation:
    delta = np.subtract(stop, start, dtype=type(dt))
    y = _nx.arange(
        0, num, dtype=dt, device=device
    ).reshape((-1,) + (1,) * ndim(delta))

    # In-place multiplication y *= delta/div is faster, but prevents
    # the multiplicant from overriding what class is produced, and thus
    # prevents, e.g. use of Quantities, see gh-7142. Hence, we multiply
    # in place only for standard scalar types.
    if div > 0:
        _mult_inplace = _nx.isscalar(delta)
        step = delta / div
        any_step_zero = (
            step == 0 if _mult_inplace else _nx.asanyarray(step == 0).any())
        if any_step_zero:
            # Special handling for denormal numbers, gh-5437
            y /= div
            if _mult_inplace:
                y *= delta
            else:
                y = y * delta
        else:
            if _mult_inplace:
                y *= step
            else:
                y = y * step
    else:
        # sequences with 0 items or 1 item with endpoint=True (i.e. div <= 0)
        # have an undefined step
        step = nan
        # Multiply with delta to allow possible override of output class.
        y = y * delta

    y += start

    if endpoint and num > 1:
        y[-1, ...] = stop

    if axis != 0:
        y = _nx.moveaxis(y, 0, axis)

    if integer_dtype:
        _nx.floor(y, out=y)

    y = conv.wrap(y.astype(dtype, copy=False))
    if retstep:
        return y, step
    else:
        return y


def _logspace_dispatcher(start, stop, num=None, endpoint=None, base=None,
                         dtype=None, axis=None):
    return (start, stop, base)


@array_function_dispatch(_logspace_dispatcher)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis=0):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    .. versionchanged:: 1.25.0
        Non-scalar 'base` is now supported

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : array_like, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred type will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start,
        stop, or base are array-like.  By default (0), the samples will be
        along a new axis inserted at the beginning. Use -1 to get an axis at
        the end.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.
    geomspace : Similar to logspace, but with endpoints specified directly.
    :ref:`how-to-partition`

    Notes
    -----
    If base is a scalar, logspace is equivalent to the code

    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
    ... # doctest: +SKIP
    >>> power(base, y).astype(dtype)
    ... # doctest: +SKIP

    Examples
    --------
    >>> import numpy as np
    >>> np.logspace(2.0, 3.0, num=4)
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])
    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
    array([100.        ,  177.827941  ,  316.22776602,  562.34132519])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
    array([4.        ,  5.0396842 ,  6.34960421,  8.        ])
    >>> np.logspace(2.0, 3.0, num=4, base=[2.0, 3.0], axis=-1)
    array([[ 4.        ,  5.0396842 ,  6.34960421,  8.        ],
           [ 9.        , 12.98024613, 18.72075441, 27.        ]])

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
    >>> y = np.zeros(N)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    if not isinstance(base, (float, int)) and np.ndim(base):
        # If base is non-scalar, broadcast it with the others, since it
        # may influence how axis is interpreted.
        ndmax = np.broadcast(start, stop, base).ndim
        start, stop, base = (
            np.array(a, copy=None, subok=True, ndmin=ndmax)
            for a in (start, stop, base)
        )
        base = np.expand_dims(base, axis=axis)
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    if dtype is None:
        return _nx.power(base, y)
    return _nx.power(base, y).astype(dtype, copy=False)


def _geomspace_dispatcher(start, stop, num=None, endpoint=None, dtype=None,
                          axis=None):
    return (start, stop)


@array_function_dispatch(_geomspace_dispatcher)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    This is similar to `logspace`, but with endpoints specified directly.
    Each output sample is a constant multiple of the previous.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The final value of the sequence, unless `endpoint` is False.
        In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    logspace : Similar to geomspace, but with endpoints specified using log
               and base.
    linspace : Similar to geomspace, but with arithmetic instead of geometric
               progression.
    arange : Similar to linspace, with the step size specified instead of the
             number of samples.
    :ref:`how-to-partition`

    Notes
    -----
    If the inputs or dtype are complex, the output will follow a logarithmic
    spiral in the complex plane.  (There are an infinite number of spirals
    passing through two points; the output will follow the shortest such path.)

    Examples
    --------
    >>> import numpy as np
    >>> np.geomspace(1, 1000, num=4)
    array([    1.,    10.,   100.,  1000.])
    >>> np.geomspace(1, 1000, num=3, endpoint=False)
    array([   1.,   10.,  100.])
    >>> np.geomspace(1, 1000, num=4, endpoint=False)
    array([   1.        ,    5.62341325,   31.6227766 ,  177.827941  ])
    >>> np.geomspace(1, 256, num=9)
    array([   1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.])

    Note that the above may not produce exact integers:

    >>> np.geomspace(1, 256, num=9, dtype=int)
    array([  1,   2,   4,   7,  16,  32,  63, 127, 256])
    >>> np.around(np.geomspace(1, 256, num=9)).astype(int)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256])

    Negative, decreasing, and complex inputs are allowed:

    >>> np.geomspace(1000, 1, num=4)
    array([1000.,  100.,   10.,    1.])
    >>> np.geomspace(-1000, -1, num=4)
    array([-1000.,  -100.,   -10.,    -1.])
    >>> np.geomspace(1j, 1000j, num=4)  # Straight line
    array([0.   +1.j, 0.  +10.j, 0. +100.j, 0.+1000.j])
    >>> np.geomspace(-1+0j, 1+0j, num=5)  # Circle
    array([-1.00000000e+00+1.22464680e-16j, -7.07106781e-01+7.07106781e-01j,
            6.12323400e-17+1.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
            1.00000000e+00+0.00000000e+00j])

    Graphical illustration of `endpoint` parameter:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> y = np.zeros(N)
    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=True), y + 1, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=False), y + 2, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.axis([0.5, 2000, 0, 3])
    [0.5, 2000, 0, 3]
    >>> plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')
    >>> plt.show()

    """
    start = asanyarray(start)
    stop = asanyarray(stop)
    if _nx.any(start == 0) or _nx.any(stop == 0):
        raise ValueError('Geometric sequence cannot include zero')

    dt = result_type(start, stop, float(num), _nx.zeros((), dtype))
    if dtype is None:
        dtype = dt
    else:
        # complex to dtype('complex128'), for instance
        dtype = _nx.dtype(dtype)

    # Promote both arguments to the same dtype in case, for instance, one is
    # complex and another is negative and log would produce NaN otherwise.
    # Copy since we may change things in-place further down.
    start = start.astype(dt, copy=True)
    stop = stop.astype(dt, copy=True)

    # Allow negative real values and ensure a consistent result for complex
    # (including avoiding negligible real or imaginary parts in output) by
    # rotating start to positive real, calculating, then undoing rotation.
    out_sign = _nx.sign(start)
    start /= out_sign
    stop = stop / out_sign

    log_start = _nx.log10(start)
    log_stop = _nx.log10(stop)
    result = logspace(log_start, log_stop, num=num,
                      endpoint=endpoint, base=10.0, dtype=dt)

    # Make sure the endpoints match the start and stop arguments. This is
    # necessary because np.exp(np.log(x)) is not necessarily equal to x.
    if num > 0:
        result[0] = start
        if num > 1 and endpoint:
            result[-1] = stop

    result *= out_sign

    if axis != 0:
        result = _nx.moveaxis(result, 0, axis)

    return result.astype(dtype, copy=False)


def _needs_add_docstring(obj):
    """
    Returns true if the only way to set the docstring of `obj` from python is
    via add_docstring.

    This function errs on the side of being overly conservative.
    """
    Py_TPFLAGS_HEAPTYPE = 1 << 9

    if isinstance(obj, (types.FunctionType, types.MethodType, property)):
        return False

    if isinstance(obj, type) and obj.__flags__ & Py_TPFLAGS_HEAPTYPE:
        return False

    return True


def _add_docstring(obj, doc, warn_on_python):
    if warn_on_python and not _needs_add_docstring(obj):
        warnings.warn(
            "add_newdoc was used on a pure-python object {}. "
            "Prefer to attach it directly to the source."
            .format(obj),
            UserWarning,
            stacklevel=3)
    try:
        add_docstring(obj, doc)
    except Exception:
        pass


def add_newdoc(place, obj, doc, warn_on_python=True):
    """
    Add documentation to an existing object, typically one defined in C

    The purpose is to allow easier editing of the docstrings without requiring
    a re-compile. This exists primarily for internal use within numpy itself.

    Parameters
    ----------
    place : str
        The absolute name of the module to import from
    obj : str or None
        The name of the object to add documentation to, typically a class or
        function name.
    doc : {str, Tuple[str, str], List[Tuple[str, str]]}
        If a string, the documentation to apply to `obj`

        If a tuple, then the first element is interpreted as an attribute
        of `obj` and the second as the docstring to apply -
        ``(method, docstring)``

        If a list, then each element of the list should be a tuple of length
        two - ``[(method1, docstring1), (method2, docstring2), ...]``
    warn_on_python : bool
        If True, the default, emit `UserWarning` if this is used to attach
        documentation to a pure-python object.

    Notes
    -----
    This routine never raises an error if the docstring can't be written, but
    will raise an error if the object being documented does not exist.

    This routine cannot modify read-only docstrings, as appear
    in new-style classes or built-in functions. Because this
    routine never raises an error the caller must check manually
    that the docstrings were changed.

    Since this function grabs the ``char *`` from a c-level str object and puts
    it into the ``tp_doc`` slot of the type of `obj`, it violates a number of
    C-API best-practices, by:

    - modifying a `PyTypeObject` after calling `PyType_Ready`
    - calling `Py_INCREF` on the str and losing the reference, so the str
      will never be released

    If possible it should be avoided.
    """
    new = getattr(__import__(place, globals(), {}, [obj]), obj)
    if isinstance(doc, str):
        if "${ARRAY_FUNCTION_LIKE}" in doc:
            doc = overrides.get_array_function_like_doc(new, doc)
        _add_docstring(new, doc.strip(), warn_on_python)
    elif isinstance(doc, tuple):
        attr, docstring = doc
        _add_docstring(getattr(new, attr), docstring.strip(), warn_on_python)
    elif isinstance(doc, list):
        for attr, docstring in doc:
            _add_docstring(
                getattr(new, attr), docstring.strip(), warn_on_python
            )
<|endoftext|>"""
Functions in the ``as*array`` family that promote array-likes into arrays.

`require` fits this category despite its name not matching this pattern.
"""
from .overrides import (
    array_function_dispatch,
    finalize_array_function_like,
    set_module,
)
from .multiarray import array, asanyarray


__all__ = ["require"]


POSSIBLE_FLAGS = {
    'C': 'C', 'C_CONTIGUOUS': 'C', 'CONTIGUOUS': 'C',
    'F': 'F', 'F_CONTIGUOUS': 'F', 'FORTRAN': 'F',
    'A': 'A', 'ALIGNED': 'A',
    'W': 'W', 'WRITEABLE': 'W',
    'O': 'O', 'OWNDATA': 'O',
    'E': 'E', 'ENSUREARRAY': 'E'
}


@finalize_array_function_like
@set_module('numpy')
def require(a, dtype=None, requirements=None, *, like=None):
    """
    Return an ndarray of the provided type that satisfies requirements.

    This function is useful to be sure that an array with the correct flags
    is returned for passing to compiled code (perhaps through ctypes).

    Parameters
    ----------
    a : array_like
       The object to be converted to a type-and-requirement-satisfying array.
    dtype : data-type
       The required data-type. If None preserve the current dtype. If your
       application requires the data to be in native byteorder, include
       a byteorder specification as a part of the dtype specification.
    requirements : str or sequence of str
       The requirements list can be any of the following

       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
       * 'ALIGNED' ('A')      - ensure a data-type aligned array
       * 'WRITEABLE' ('W')    - ensure a writable array
       * 'OWNDATA' ('O')      - ensure an array that owns its own data
       * 'ENSUREARRAY', ('E') - ensure a base array, instead of a subclass
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array with specified requirements and type if given.

    See Also
    --------
    asarray : Convert input to an ndarray.
    asanyarray : Convert to an ndarray, but pass through ndarray subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    ndarray.flags : Information about the memory layout of the array.

    Notes
    -----
    The returned array will be guaranteed to have the listed requirements
    by making a copy if needed.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2,3)
    >>> x.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : False
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False

    >>> y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
    >>> y.flags
      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False

    """
    if like is not None:
        return _require_with_like(
            like,
            a,
            dtype=dtype,
            requirements=requirements,
        )

    if not requirements:
        return asanyarray(a, dtype=dtype)

    requirements = {POSSIBLE_FLAGS[x.upper()] for x in requirements}

    if 'E' in requirements:
        requirements.remove('E')
        subok = False
    else:
        subok = True

    order = 'A'
    if requirements >= {'C', 'F'}:
        raise ValueError('Cannot specify both "C" and "F" order')
    elif 'F' in requirements:
        order = 'F'
        requirements.remove('F')
    elif 'C' in requirements:
        order = 'C'
        requirements.remove('C')

    arr = array(a, dtype=dtype, order=order, copy=None, subok=subok)

    for prop in requirements:
        if not arr.flags[prop]:
            return arr.copy(order)
    return arr


_require_with_like = array_function_dispatch()(require)
<|endoftext|>"""Module containing non-deprecated functions borrowed from Numeric.

"""
import functools
import types
import warnings

import numpy as np
from .._utils import set_module
from . import multiarray as mu
from . import overrides
from . import umath as um
from . import numerictypes as nt
from .multiarray import asarray, array, asanyarray, concatenate
from ._multiarray_umath import _array_converter
from . import _methods

_dt_ = nt.sctype2char

# functions that are methods
__all__ = [
    'all', 'amax', 'amin', 'any', 'argmax',
    'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip',
    'compress', 'cumprod', 'cumsum', 'cumulative_prod', 'cumulative_sum',
    'diagonal', 'mean', 'max', 'min', 'matrix_transpose',
    'ndim', 'nonzero', 'partition', 'prod', 'ptp', 'put',
    'ravel', 'repeat', 'reshape', 'resize', 'round',
    'searchsorted', 'shape', 'size', 'sort', 'squeeze',
    'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var',
]

_gentype = types.GeneratorType
# save away Python sum
_sum_ = sum

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


# functions that are now methods
def _wrapit(obj, method, *args, **kwds):
    conv = _array_converter(obj)
    # As this already tried the method, subok is maybe quite reasonable here
    # but this follows what was done before. TODO: revisit this.
    arr, = conv.as_arrays(subok=False)
    result = getattr(arr, method)(*args, **kwds)

    return conv.wrap(result, to_scalar=False)


def _wrapfunc(obj, method, *args, **kwds):
    bound = getattr(obj, method, None)
    if bound is None:
        return _wrapit(obj, method, *args, **kwds)

    try:
        return bound(*args, **kwds)
    except TypeError:
        # A TypeError occurs if the object does have such a method in its
        # class, but its signature is not identical to that of NumPy's. This
        # situation has occurred in the case of a downstream library like
        # 'pandas'.
        #
        # Call _wrapit from within the except clause to ensure a potential
        # exception has a traceback chain.
        return _wrapit(obj, method, *args, **kwds)


def _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs):
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}

    if type(obj) is not mu.ndarray:
        try:
            reduction = getattr(obj, method)
        except AttributeError:
            pass
        else:
            # This branch is needed for reductions like any which don't
            # support a dtype.
            if dtype is not None:
                return reduction(axis=axis, dtype=dtype, out=out, **passkwargs)
            else:
                return reduction(axis=axis, out=out, **passkwargs)

    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)


def _wrapreduction_any_all(obj, ufunc, method, axis, out, **kwargs):
    # Same as above function, but dtype is always bool (but never passed on)
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}

    if type(obj) is not mu.ndarray:
        try:
            reduction = getattr(obj, method)
        except AttributeError:
            pass
        else:
            return reduction(axis=axis, out=out, **passkwargs)

    return ufunc.reduce(obj, axis, bool, out, **passkwargs)


def _take_dispatcher(a, indices, axis=None, out=None, mode=None):
    return (a, out)


@array_function_dispatch(_take_dispatcher)
def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : array_like (Ni..., M, Nk...)
        The source array.
    indices : array_like (Nj...)
        The indices of the values to extract.
        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional (Ni..., Nj..., Nk...)
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if `mode='raise'`; use other modes for better performance.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray (Ni..., Nj..., Nk...)
        The returned array has the same type as `a`.

    See Also
    --------
    compress : Take elements using a boolean mask
    ndarray.take : equivalent method
    take_along_axis : Take elements by matching the array and the index arrays

    Notes
    -----
    By eliminating the inner loop in the description above, and using `s_` to
    build simple slice objects, `take` can be expressed  in terms of applying
    fancy indexing to each 1-d slice::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nj):
                out[ii + s_[...,] + kk] = a[ii + s_[:,] + kk][indices]

    For this reason, it is equivalent to (but faster than) the following use
    of `apply_along_axis`::

        out = np.apply_along_axis(lambda a_1d: a_1d[indices], axis, a)

    Examples
    --------
    >>> import numpy as np
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> np.take(a, indices)
    array([4, 3, 6])

    In this example if `a` is an ndarray, "fancy" indexing can be used.

    >>> a = np.array(a)
    >>> a[indices]
    array([4, 3, 6])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(a, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])
    """
    return _wrapfunc(a, 'take', indices, axis=axis, out=out, mode=mode)


def _reshape_dispatcher(a, /, shape=None, order=None, *, newshape=None,
                        copy=None):
    return (a,)


@array_function_dispatch(_reshape_dispatcher)
def reshape(a, /, shape=None, order='C', *, newshape=None, copy=None):
    """
    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    shape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
        Read the elements of ``a`` using this index order, and place the
        elements into the reshaped array using this index order. 'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if ``a`` is Fortran *contiguous* in memory, C-like order
        otherwise.
    newshape : int or tuple of ints
        .. deprecated:: 2.1
            Replaced by ``shape`` argument. Retained for backward
            compatibility.
    copy : bool, optional
        If ``True``, then the array data is copied. If ``None``, a copy will
        only be made if it's required by ``order``. For ``False`` it raises
        a ``ValueError`` if a copy cannot be avoided. Default: ``None``.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    See Also
    --------
    ndarray.reshape : Equivalent method.

    Notes
    -----
    It is not always possible to change the shape of an array without copying
    the data.

    The ``order`` keyword gives the index ordering both for *fetching*
    the values from ``a``, and then *placing* the values into the output
    array. For example, let's say you have an array:

    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the array (using the given
    index order), then inserting the elements from the raveled array into the
    new array using the same kind of index ordering as was used for the
    raveling.

    >>> np.reshape(a, (2, 3)) # C-like index ordering
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    array([[0, 4, 3],
           [2, 1, 5]])
    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    array([[0, 4, 3],
           [2, 1, 5]])

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])

    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    if newshape is None and shape is None:
        raise TypeError(
            "reshape() missing 1 required positional argument: 'shape'")
    if newshape is not None:
        if shape is not None:
            raise TypeError(
                "You cannot specify 'newshape' and 'shape' arguments "
                "at the same time.")
        # Deprecated in NumPy 2.1, 2024-04-18
        warnings.warn(
            "`newshape` keyword argument is deprecated, "
            "use `shape=...` or pass shape positionally instead. "
            "(deprecated in NumPy 2.1)",
            DeprecationWarning,
            stacklevel=2,
        )
        shape = newshape
    if copy is not None:
        return _wrapfunc(a, 'reshape', shape, order=order, copy=copy)
    return _wrapfunc(a, 'reshape', shape, order=order)


def _choose_dispatcher(a, choices, out=None, mode=None):
    yield a
    yield from choices
    yield out


@array_function_dispatch(_choose_dispatcher)
def choose(a, choices, out=None, mode='raise'):
    """
    Construct an array from an index array and a list of arrays to choose from.

    First of all, if confused or uncertain, definitely look at the Examples -
    in its full generality, this function is less simple than it might
    seem from the following code description::

        np.choose(a,c) == np.array([c[a[I]][I] for I in np.ndindex(a.shape)])

    But this omits some subtleties.  Here is a fully general summary:

    Given an "index" array (`a`) of integers and a sequence of ``n`` arrays
    (`choices`), `a` and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =
    0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
    for each ``i``.  Then, a new array with shape ``Ba.shape`` is created as
    follows:

    * if ``mode='raise'`` (the default), then, first of all, each element of
      ``a`` (and thus ``Ba``) must be in the range ``[0, n-1]``; now, suppose
      that ``i`` (in that range) is the value at the ``(j0, j1, ..., jm)``
      position in ``Ba`` - then the value at the same position in the new array
      is the value in ``Bchoices[i]`` at that same position;

    * if ``mode='wrap'``, values in `a` (and thus `Ba`) may be any (signed)
      integer; modular arithmetic is used to map integers outside the range
      `[0, n-1]` back into that range; and then the new array is constructed
      as above;

    * if ``mode='clip'``, values in `a` (and thus ``Ba``) may be any (signed)
      integer; negative integers are mapped to 0; values greater than ``n-1``
      are mapped to ``n-1``; and then the new array is constructed as above.

    Parameters
    ----------
    a : int array
        This array must contain integers in ``[0, n-1]``, where ``n`` is the
        number of choices, unless ``mode=wrap`` or ``mode=clip``, in which
        cases any integers are permissible.
    choices : sequence of arrays
        Choice arrays. `a` and all of the choices must be broadcastable to the
        same shape.  If `choices` is itself an array (not recommended), then
        its outermost dimension (i.e., the one corresponding to
        ``choices.shape[0]``) is taken as defining the "sequence".
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if ``mode='raise'``; use other modes for better performance.
    mode : {'raise' (default), 'wrap', 'clip'}, optional
        Specifies how indices outside ``[0, n-1]`` will be treated:

        * 'raise' : an exception is raised
        * 'wrap' : value becomes value mod ``n``
        * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array : array
        The merged result.

    Raises
    ------
    ValueError: shape mismatch
        If `a` and each choice array are not all broadcastable to the same
        shape.

    See Also
    --------
    ndarray.choose : equivalent method
    numpy.take_along_axis : Preferable if `choices` is an array

    Notes
    -----
    To reduce the chance of misinterpretation, even though the following
    "abuse" is nominally supported, `choices` should neither be, nor be
    thought of as, a single array, i.e., the outermost sequence-like container
    should be either a list or a tuple.

    Examples
    --------

    >>> import numpy as np
    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
    ...   [20, 21, 22, 23], [30, 31, 32, 33]]
    >>> np.choose([2, 3, 1, 0], choices
    ... # the first element of the result will be the first element of the
    ... # third (2+1) "array" in choices, namely, 20; the second element
    ... # will be the second element of the fourth (3+1) choice array, i.e.,
    ... # 31, etc.
    ... )
    array([20, 31, 12,  3])
    >>> np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
    array([20, 31, 12,  3])
    >>> # because there are 4 choice arrays
    >>> np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4)
    array([20,  1, 12,  3])
    >>> # i.e., 0

    A couple examples illustrating how choose broadcasts:

    >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    >>> choices = [-10, 10]
    >>> np.choose(a, choices)
    array([[ 10, -10,  10],
           [-10,  10, -10],
           [ 10, -10,  10]])

    >>> # With thanks to Anne Archibald
    >>> a = np.array([0, 1]).reshape((2,1,1))
    >>> c1 = np.array([1, 2, 3]).reshape((1,3,1))
    >>> c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
    >>> np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
    array([[[ 1,  1,  1,  1,  1],
            [ 2,  2,  2,  2,  2],
            [ 3,  3,  3,  3,  3]],
           [[-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5]]])

    """
    return _wrapfunc(a, 'choose', choices, out=out, mode=mode)


def _repeat_dispatcher(a, repeats, axis=None):
    return (a,)


@array_function_dispatch(_repeat_dispatcher)
def repeat(a, repeats, axis=None):
    """
    Repeat each element of an array after themselves

    Parameters
    ----------
    a : array_like
        Input array.
    repeats : int or array of ints
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.

    See Also
    --------
    tile : Tile an array.
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> np.repeat(3, 4)
    array([3, 3, 3, 3])
    >>> x = np.array([[1,2],[3,4]])
    >>> np.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    return _wrapfunc(a, 'repeat', repeats, axis=axis)


def _put_dispatcher(a, ind, v, mode=None):
    return (a, ind, v)


@array_function_dispatch(_put_dispatcher)
def put(a, ind, v, mode='raise'):
    """
    Replaces specified elements of an array with given values.

    The indexing works on the flattened target array. `put` is roughly
    equivalent to:

    ::

        a.flat[ind] = v

    Parameters
    ----------
    a : ndarray
        Target array.
    ind : array_like
        Target indices, interpreted as integers.
    v : array_like
        Values to place in `a` at target indices. If `v` is shorter than
        `ind` it will be repeated as necessary.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers. In 'raise' mode,
        if an exception occurs the target array may still be modified.

    See Also
    --------
    putmask, place
    put_along_axis : Put elements by matching the array and the index arrays

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(5)
    >>> np.put(a, [0, 2], [-44, -55])
    >>> a
    array([-44,   1, -55,   3,   4])

    >>> a = np.arange(5)
    >>> np.put(a, 22, -5, mode='clip')
    >>> a
    array([ 0,  1,  2,  3, -5])

    """
    try:
        put = a.put
    except AttributeError as e:
        raise TypeError("argument 1 must be numpy.ndarray, "
                        "not {name}".format(name=type(a).__name__)) from e

    return put(ind, v, mode=mode)


def _swapaxes_dispatcher(a, axis1, axis2):
    return (a,)


@array_function_dispatch(_swapaxes_dispatcher)
def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        For NumPy >= 1.10.0, if `a` is an ndarray, then a view of `a` is
        returned; otherwise a new array is created. For earlier NumPy
        versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1,2,3]])
    >>> np.swapaxes(x,0,1)
    array([[1],
           [2],
           [3]])

    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> np.swapaxes(x,0,2)
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])

    """
    return _wrapfunc(a, 'swapaxes', axis1, axis2)


def _transpose_dispatcher(a, axes=None):
    return (a,)


@array_function_dispatch(_transpose_dispatcher)
def transpose(a, axes=None):
    """
    Returns an array with axes transposed.

    For a 1-D array, this returns an unchanged view of the original array, as a
    transposed vector is simply the same vector.
    To convert a 1-D array into a 2-D column vector, an additional dimension
    must be added, e.g., ``np.atleast_2d(a).T`` achieves this, as does
    ``a[:, np.newaxis]``.
    For a 2-D array, this is the standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided, then
    ``transpose(a).shape == a.shape[::-1]``.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation
        of [0, 1, ..., N-1] where N is the number of axes of `a`. Negative
        indices can also be used to specify axes. The i-th axis of the returned
        array will correspond to the axis numbered ``axes[i]`` of the input.
        If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses
        the order of the axes.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted. A view is returned whenever possible.

    See Also
    --------
    ndarray.transpose : Equivalent method.
    moveaxis : Move axes of an array to new positions.
    argsort : Return the indices that would sort an array.

    Notes
    -----
    Use ``transpose(a, argsort(axes))`` to invert the transposition of tensors
    when using the `axes` keyword argument.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> np.transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> np.transpose(a)
    array([1, 2, 3, 4])

    >>> a = np.ones((1, 2, 3))
    >>> np.transpose(a, (1, 0, 2)).shape
    (2, 1, 3)

    >>> a = np.ones((2, 3, 4, 5))
    >>> np.transpose(a).shape
    (5, 4, 3, 2)

    >>> a = np.arange(3*4*5).reshape((3, 4, 5))
    >>> np.transpose(a, (-1, 0, -2)).shape
    (5, 3, 4)

    """
    return _wrapfunc(a, 'transpose', axes)


def _matrix_transpose_dispatcher(x):
    return (x,)

@array_function_dispatch(_matrix_transpose_dispatcher)
def matrix_transpose(x, /):
    """
    Transposes a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.

    Returns
    -------
    out : ndarray
        An array containing the transpose for each matrix and having shape
        (..., N, M).

    See Also
    --------
    transpose : Generic transpose method.

    Examples
    --------
    >>> import numpy as np
    >>> np.matrix_transpose([[1, 2], [3, 4]])
    array([[1, 3],
           [2, 4]])

    >>> np.matrix_transpose([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array([[[1, 3],
            [2, 4]],
           [[5, 7],
            [6, 8]]])

    """
    x = asanyarray(x)
    if x.ndim < 2:
        raise ValueError(
            f"Input array must be at least 2-dimensional, but it is {x.ndim}"
        )
    return swapaxes(x, -1, -2)


def _partition_dispatcher(a, kth, axis=None, kind=None, order=None):
    return (a,)


@array_function_dispatch(_partition_dispatcher)
def partition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Return a partitioned copy of an array.

    Creates a copy of the array and partially sorts it in such a way that
    the value of the element in k-th position is in the position it would be
    in a sorted array. In the output array, all elements smaller than the k-th
    element are located to the left of this element and all equal or greater
    are located to its right. The ordering of the elements in the two
    partitions on the either side of the k-th element in the output array is
    undefined.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    kth : int or sequence of ints
        Element index to partition by. The k-th value of the element
        will be in its final sorted position and all smaller elements
        will be moved before it and all equal or greater elements behind
        it. The order of all elements in the partitions is undefined. If
        provided with a sequence of k-th it will partition all elements
        indexed by k-th  of them into their sorted position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument
        specifies which fields to compare first, second, etc.  A single
        field can be specified as a string.  Not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    partitioned_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.partition : Method to sort an array in-place.
    argpartition : Indirect partition.
    sort : Full sorting

    Notes
    -----
    The various selection algorithms are characterized by their average
    speed, worst case performance, work space size, and whether they are
    stable. A stable sort keeps items with the same key in the same
    relative order. The available algorithms have the following
    properties:

    ================= ======= ============= ============ =======
       kind            speed   worst case    work space  stable
    ================= ======= ============= ============ =======
    'introselect'        1        O(n)           0         no
    ================= ======= ============= ============ =======

    All the partition algorithms make temporary copies of the data when
    partitioning along any but the last axis.  Consequently,
    partitioning along the last axis is faster and uses less space than
    partitioning along any other axis.

    The sort order for complex numbers is lexicographic. If both the
    real and imaginary parts are non-nan then the order is determined by
    the real parts except when they are equal, in which case the order
    is determined by the imaginary parts.

