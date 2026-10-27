        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'ignore', then invalid parsing will return the input.

    Returns
    -------
    timedelta
        If parsing succeeded.
        Return type depends on input:

        - list-like: TimedeltaIndex of timedelta64 dtype
        - Series: Series of timedelta64 dtype
        - scalar: Timedelta

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_datetime : Convert argument to datetime.
    convert_dtypes : Convert dtypes.

    Notes
    -----
    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds for string inputs.

    Examples
    --------
    Parsing a single string to a Timedelta:

    >>> pd.to_timedelta('1 days 06:05:01.00003')
    Timedelta('1 days 06:05:01.000030')
    >>> pd.to_timedelta('15.5us')
    Timedelta('0 days 00:00:00.000015500')

    Parsing a list or array of strings:

    >>> pd.to_timedelta(['1 days 06:05:01.00003', '15.5us', 'nan'])
    TimedeltaIndex(['1 days 06:05:01.000030', '0 days 00:00:00.000015500', NaT],
                   dtype='timedelta64[ns]', freq=None)

    Converting numbers by specifying the `unit` keyword argument:

    >>> pd.to_timedelta(np.arange(5), unit='s')
    TimedeltaIndex(['0 days 00:00:00', '0 days 00:00:01', '0 days 00:00:02',
                    '0 days 00:00:03', '0 days 00:00:04'],
                   dtype='timedelta64[ns]', freq=None)
    >>> pd.to_timedelta(np.arange(5), unit='d')
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq=None)
    """
    if unit is not None:
        unit = parse_timedelta_unit(unit)
        disallow_ambiguous_unit(unit)

    if errors not in ("ignore", "raise", "coerce"):
        raise ValueError("errors must be one of 'ignore', 'raise', or 'coerce'.")
    if errors == "ignore":
        # GH#54467
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a future version. "
            "Use to_timedelta without passing `errors` and catch exceptions "
            "explicitly instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if arg is None:
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, unit=unit, errors=errors)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, unit=unit, errors=errors, name=arg.name)
    elif isinstance(arg, np.ndarray) and arg.ndim == 0:
        # extract array scalar and process below
        # error: Incompatible types in assignment (expression has type "object",
        # variable has type "Union[str, int, float, timedelta, List[Any],
        # Tuple[Any, ...], Union[Union[ExtensionArray, ndarray[Any, Any]], Index,
        # Series]]")  [assignment]
        arg = lib.item_from_zerodim(arg)  # type: ignore[assignment]
    elif is_list_like(arg) and getattr(arg, "ndim", 1) == 1:
        return _convert_listlike(arg, unit=unit, errors=errors)
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, timedelta, list, tuple, 1-d array, or Series"
        )

    if isinstance(arg, str) and unit is not None:
        raise ValueError("unit must not be specified if the input is/contains a str")

    # ...so it must be a scalar value. Return scalar.
    return _coerce_scalar_to_timedelta_type(arg, unit=unit, errors=errors)


def _coerce_scalar_to_timedelta_type(
    r, unit: UnitChoices | None = "ns", errors: DateTimeErrorChoices = "raise"
):
    """Convert string 'r' to a timedelta object."""
    result: Timedelta | NaTType

    try:
        result = Timedelta(r, unit)
    except ValueError:
        if errors == "raise":
            raise
        if errors == "ignore":
            return r

        # coerce
        result = NaT

    return result


def _convert_listlike(
    arg,
    unit: UnitChoices | None = None,
    errors: DateTimeErrorChoices = "raise",
    name: Hashable | None = None,
):
    """Convert a list of objects to a timedelta index object."""
    arg_dtype = getattr(arg, "dtype", None)
    if isinstance(arg, (list, tuple)) or arg_dtype is None:
        # This is needed only to ensure that in the case where we end up
        #  returning arg (errors == "ignore"), and where the input is a
        #  generator, we return a useful list-like instead of a
        #  used-up generator
        if not hasattr(arg, "__array__"):
            arg = list(arg)
        arg = np.array(arg, dtype=object)
    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.kind == "m":
        return arg

    try:
        td64arr = sequence_to_td64ns(arg, unit=unit, errors=errors, copy=False)[0]
    except ValueError:
        if errors == "ignore":
            return arg
        else:
            # This else-block accounts for the cases when errors='raise'
            # and errors='coerce'. If errors == 'raise', these errors
            # should be raised. If errors == 'coerce', we shouldn't
            # expect any errors to be raised, since all parsing errors
            # cause coercion to pd.NaT. However, if an error / bug is
            # introduced that causes an Exception to be raised, we would
            # like to surface it.
            raise

    from pandas import TimedeltaIndex

    value = TimedeltaIndex(td64arr, name=name)
    return value
<|endoftext|>""" define the IntervalIndex """
from __future__ import annotations

from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    Interval,
    IntervalMixin,
    IntervalTree,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
    Appender,
    cache_readonly,
)
from pandas.util._exceptions import rewrite_exception

from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    maybe_box_datetimelike,
    maybe_downcast_numeric,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
    IntervalArray,
    _interval_shared_docs,
)
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    ensure_index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.extension import (
    ExtensionIndex,
    inherit_names,
)
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Self,
        npt,
    )
_index_doc_kwargs = dict(ibase._index_doc_kwargs)

_index_doc_kwargs.update(
    {
        "klass": "IntervalIndex",
        "qualname": "IntervalIndex",
        "target_klass": "IntervalIndex or list of Intervals",
        "name": textwrap.dedent(
            """\
         name : object, optional
              Name to be stored in the index.
         """
        ),
    }
)


def _get_next_label(label):
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label + np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label + 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {repr(type(label))}")


def _get_prev_label(label):
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label - np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label - 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, -np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {repr(type(label))}")


def _new_IntervalIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """
    return cls.from_arrays(**d)


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalIndex",
        "summary": "Immutable index of intervals that are closed on the same side.",
        "name": _index_doc_kwargs["name"],
        "extra_attributes": "is_overlapping\nvalues\n",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalIndex`` is typically constructed using
    :func:`interval_range`:

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalIndex.from_arrays`,
    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.

    See further examples in the doc strings of ``interval_range`` and the
    mentioned constructor methods.
    """
        ),
    }
)
@inherit_names(["set_closed", "to_tuples"], IntervalArray, wrap=True)
@inherit_names(
    [
        "__array__",
        "overlaps",
        "contains",
        "closed_left",
        "closed_right",
        "open_left",
        "open_right",
        "is_empty",
    ],
    IntervalArray,
)
@inherit_names(["is_non_overlapping_monotonic", "closed"], IntervalArray, cache=True)
class IntervalIndex(ExtensionIndex):
    _typ = "intervalindex"

    # annotate properties pinned via inherit_names
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool

    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings = False
    _data_cls = IntervalArray

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
        verify_integrity: bool = True,
    ) -> Self:
        name = maybe_extract_name(name, data, cls)

        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray(
                data,
                closed=closed,
                copy=copy,
                dtype=dtype,
                verify_integrity=verify_integrity,
            )

        return cls._simple_new(array, name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_breaks(
                breaks, closed=closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_arrays(
                left, right, closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
        IntervalIndex([(0, 1], (1, 2]],
                       dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(arr, name=name)

    # --------------------------------------------------------------------
    # error: Return type "IntervalTree" of "_engine" incompatible with return type
    # "Union[IndexEngine, ExtensionEngine]" in supertype "Index"
    @cache_readonly
    def _engine(self) -> IntervalTree:  # type: ignore[override]
        # IntervalTree does not supports numpy array unless they are 64 bit
        left = self._maybe_convert_i8(self.left)
        left = maybe_upcast_numeric_to_64bit(left)
        right = self._maybe_convert_i8(self.right)
        right = maybe_upcast_numeric_to_64bit(right)
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key: Any) -> bool:
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
        hash(key)
        if not isinstance(key, Interval):
            if is_valid_na_for_dtype(key, self.dtype):
                return self.hasnans
            return False

        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        return type(self)._simple_new(res, name=self._name)

    @cache_readonly
    def _multiindex(self) -> MultiIndex:
        return MultiIndex.from_arrays([self.left, self.right], names=["left", "right"])

    def __reduce__(self):
        d = {
            "left": self.left,
            "right": self.right,
            "closed": self.closed,
            "name": self.name,
        }
        return _new_IntervalIndex, (type(self), d), None

    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values"""
        return "interval"

    # Cannot determine type of "memory_usage"
    @Appender(Index.memory_usage.__doc__)  # type: ignore[has-type]
    def memory_usage(self, deep: bool = False) -> int:
        # we don't use an explicit engine
        # so return the bytes here
        return self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep)

    # IntervalTree doesn't have a is_monotonic_decreasing, so have to override
    #  the Index implementation
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if the IntervalIndex is monotonic decreasing (only equal or
        decreasing values), else False
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def is_unique(self) -> bool:
        """
        Return True if the IntervalIndex contains unique elements, else False.
        """
        left = self.left
        right = self.right

        if self.isna().sum() > 1:
            return False

        if left.is_unique or right.is_unique:
            return True

        seen_pairs = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)

        return True

    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Returns
        -------
        bool
            Boolean indicating if the IntervalIndex has overlapping intervals.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.
        IntervalIndex.overlaps : Check an IntervalIndex elementwise for
            overlaps.

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        Intervals that share closed endpoints overlap:

        >>> index = pd.interval_range(0, 3, closed='both')
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> index = pd.interval_range(0, 3, closed='left')
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
        """
        # GH 23309
        return self._engine.is_overlapping

    def _needs_i8_conversion(self, key) -> bool:
        """
        Check if a given key needs i8 conversion. Conversion is necessary for
        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An
        Interval-like requires conversion if its endpoints are one of the
        aforementioned types.

        Assumes that any list-like data has already been cast to an Index.

        Parameters
        ----------
        key : scalar or Index-like
            The key that should be checked for i8 conversion

        Returns
        -------
        bool
        """
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            return self._needs_i8_conversion(key.left)

        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        return isinstance(key, i8_types)

    def _maybe_convert_i8(self, key):
        """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
        if is_list_like(key):
            key = ensure_index(key)
            key = maybe_upcast_numeric_to_64bit(key)

        if not self._needs_i8_conversion(key):
            return key

        scalar = is_scalar(key)
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            # convert left/right and reconstruct
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            constructor = Interval if scalar else IntervalIndex.from_arrays
            # error: "object" not callable
            return constructor(
                left, right, closed=self.closed
            )  # type: ignore[operator]

        if scalar:
            # Timestamp/Timedelta
            key_dtype, key_i8 = infer_dtype_from_scalar(key)
            if isinstance(key, Period):
                key_i8 = key.ordinal
            elif isinstance(key_i8, Timestamp):
                key_i8 = key_i8._value
            elif isinstance(key_i8, (np.datetime64, np.timedelta64)):
                key_i8 = key_i8.view("i8")
        else:
            # DatetimeIndex/TimedeltaIndex
            key_dtype, key_i8 = key.dtype, Index(key.asi8)
            if key.hasnans:
                # convert NaT from its i8 value to np.nan so it's not viewed
                # as a valid value, maybe causing errors (e.g. is_overlapping)
                key_i8 = key_i8.where(~key._isnan)

        # ensure consistency with IntervalIndex subtype
        # error: Item "ExtensionDtype"/"dtype[Any]" of "Union[dtype[Any],
        # ExtensionDtype]" has no attribute "subtype"
        subtype = self.dtype.subtype  # type: ignore[union-attr]

        if subtype != key_dtype:
            raise ValueError(
                f"Cannot index an IntervalIndex of subtype {subtype} with "
                f"values of dtype {key_dtype}"
            )

        return key_i8

    def _searchsorted_monotonic(self, label, side: Literal["left", "right"] = "left"):
        if not self.is_non_overlapping_monotonic:
            raise KeyError(
                "can only get slices from an IntervalIndex if bounds are "
                "non-overlapping and all monotonic increasing or decreasing"
            )

        if isinstance(label, (IntervalMixin, IntervalIndex)):
            raise NotImplementedError("Interval objects are not currently supported")

        # GH 20921: "not is_monotonic_increasing" for the second condition
        # instead of "is_monotonic_decreasing" to account for single element
        # indexes being both increasing and decreasing
        if (side == "left" and self.left.is_monotonic_increasing) or (
            side == "right" and not self.left.is_monotonic_increasing
        ):
            sub_idx = self.right
            if self.open_right:
                label = _get_next_label(label)
        else:
            sub_idx = self.left
            if self.open_left:
                label = _get_prev_label(label)

        return sub_idx._searchsorted_monotonic(label, side)

    # --------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key) -> int | slice | np.ndarray:
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)
        >>> index = pd.IntervalIndex([i1, i2])
        >>> index.get_loc(1)
        0

        You can also supply a point inside an interval.

        >>> index.get_loc(1.5)
        1

        If a label is in several intervals, you get the locations of all the
        relevant intervals.

        >>> i3 = pd.Interval(0, 2)
        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])
        >>> overlapping_index.get_loc(0.5)
        array([ True, False,  True])

        Only exact matches will be returned if an interval is provided.

        >>> index.get_loc(pd.Interval(0, 1))
        0
        """
        self._check_indexing_error(key)

        if isinstance(key, Interval):
            if self.closed != key.closed:
                raise KeyError(key)
            mask = (self.left == key.left) & (self.right == key.right)
        elif is_valid_na_for_dtype(key, self.dtype):
            mask = self.isna()
        else:
            # assume scalar
            op_left = le if self.closed_left else lt
            op_right = le if self.closed_right else lt
            try:
                mask = op_left(self.left, key) & op_right(key, self.right)
            except TypeError as err:
                # scalar is not comparable to II subtype --> invalid label
                raise KeyError(key) from err

        matches = mask.sum()
        if matches == 0:
            raise KeyError(key)
        if matches == 1:
            return mask.argmax()

        res = lib.maybe_booleans_to_slice(mask.view("u1"))
        if isinstance(res, slice) and res.stop is None:
            # TODO: DO this in maybe_booleans_to_slice?
            res = slice(res.start, len(self), res.step)
        return res

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance: Any | None = None,
    ) -> npt.NDArray[np.intp]:
        if isinstance(target, IntervalIndex):
            # We only get here with not self.is_overlapping
            # -> at most one match per interval in target
            # want exact matches -> need both left/right to match, so defer to
            # left/right get_indexer, compare elementwise, equality -> match
            indexer = self._get_indexer_unique_sides(target)

        elif not (is_object_dtype(target.dtype) or is_string_dtype(target.dtype)):
            # homogeneous scalar index: use IntervalTree
            # we should always have self._should_partial_index(target) here
            target = self._maybe_convert_i8(target)
            indexer = self._engine.get_indexer(target.values)
        else:
            # heterogeneous scalar index: defer elementwise to get_loc
            # we should always have self._should_partial_index(target) here
            return self._get_indexer_pointwise(target)[0]

        return ensure_platform_int(indexer)

    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        target = ensure_index(target)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # e.g. IntervalIndex with different closed or incompatible subtype
            #  -> no matches
            return self._get_indexer_non_comparable(target, None, unique=False)

        elif isinstance(target, IntervalIndex):
            if self.left.is_unique and self.right.is_unique:
                # fastpath available even if we don't have self._index_as_unique
                indexer = self._get_indexer_unique_sides(target)
                missing = (indexer == -1).nonzero()[0]
            else:
                return self._get_indexer_pointwise(target)

        elif is_object_dtype(target.dtype) or not self._should_partial_index(target):
            # target might contain intervals: defer elementwise to get_loc
            return self._get_indexer_pointwise(target)

        else:
            # Note: this case behaves differently from other Index subclasses
            #  because IntervalIndex does partial-int indexing
            target = self._maybe_convert_i8(target)
            indexer, missing = self._engine.get_indexer_non_unique(target.values)

        return ensure_platform_int(indexer), ensure_platform_int(missing)

    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
        # Caller is responsible for checking
        #  `self.left.is_unique and self.right.is_unique`

        left_indexer = self.left.get_indexer(target.left)
        right_indexer = self.right.get_indexer(target.right)
        indexer = np.where(left_indexer == right_indexer, left_indexer, -1)
        return indexer

    def _get_indexer_pointwise(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
        indexer, missing = [], []
        for i, key in enumerate(target):
            try:
                locs = self.get_loc(key)
                if isinstance(locs, slice):
                    # Only needed for get_indexer_non_unique
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype="intp")
                elif lib.is_integer(locs):
                    locs = np.array(locs, ndmin=1)
                else:
                    # otherwise we have ndarray[bool]
                    locs = np.where(locs)[0]
            except KeyError:
                missing.append(i)
                locs = np.array([-1])
            except InvalidIndexError:
                # i.e. non-scalar key e.g. a tuple.
                # see test_append_different_columns_types_raises
                missing.append(i)
                locs = np.array([-1])

            indexer.append(locs)

        indexer = np.concatenate(indexer)
        return ensure_platform_int(indexer), ensure_platform_int(missing)

    @cache_readonly
    def _index_as_unique(self) -> bool:
        return not self.is_overlapping and self._engine._na_count < 2

    _requires_unique_msg = (
        "cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique"
    )

    def _convert_slice_indexer(self, key: slice, kind: Literal["loc", "getitem"]):
        if not (key.step is None or key.step == 1):
            # GH#31658 if label-based, we require step == 1,
            #  if positional, we disallow float start/stop
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            if kind == "loc":
                raise ValueError(msg)
            if kind == "getitem":
                if not is_valid_positional_slice(key):
                    # i.e. this cannot be interpreted as a positional slice
                    raise ValueError(msg)

        return super()._convert_slice_indexer(key, kind)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # integer lookups in Series.__getitem__ are unambiguously
        #  positional in this case
        # error: Item "ExtensionDtype"/"dtype[Any]" of "Union[dtype[Any],
        # ExtensionDtype]" has no attribute "subtype"
        return self.dtype.subtype.kind in "mM"  # type: ignore[union-attr]

    def _maybe_cast_slice_bound(self, label, side: str):
        return getattr(self, side)._maybe_cast_slice_bound(label, side)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        if not isinstance(dtype, IntervalDtype):
            return False
        common_subtype = find_common_type([self.dtype, dtype])
        return not is_object_dtype(common_subtype)

    # --------------------------------------------------------------------

    @cache_readonly
    def left(self) -> Index:
        return Index(self._data.left, copy=False)

    @cache_readonly
    def right(self) -> Index:
        return Index(self._data.right, copy=False)

    @cache_readonly
    def mid(self) -> Index:
        return Index(self._data.mid, copy=False)

    @property
    def length(self) -> Index:
        return Index(self._data.length, copy=False)

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other, sort):
        """
        intersection specialized to the case with matching dtypes.
        """
        # For IntervalIndex we also know other.closed == self.closed
        if self.left.is_unique and self.right.is_unique:
            taken = self._intersection_unique(other)
        elif other.left.is_unique and other.right.is_unique and self.isna().sum() <= 1:
            # Swap other/self if other is unique and self does not have
            # multiple NaNs
            taken = other._intersection_unique(self)
        else:
            # duplicates
            taken = self._intersection_non_unique(other)

        if sort is None:
            taken = taken.sort_values()

        return taken

    def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does not have any common endpoint,
        no matter left or right.
        Return the intersection with another IntervalIndex.
        Parameters
        ----------
        other : IntervalIndex
        Returns
        -------
        IntervalIndex
        """
        # Note: this is much more performant than super()._intersection(other)
        lindexer = self.left.get_indexer(other.left)
        rindexer = self.right.get_indexer(other.right)

        match = (lindexer == rindexer) & (lindexer != -1)
        indexer = lindexer.take(match.nonzero()[0])
        indexer = unique(indexer)

        return self.take(indexer)

    def _intersection_non_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does have some common endpoints,
        on either sides.
        Return the intersection with another IntervalIndex.

        Parameters
        ----------
        other : IntervalIndex

        Returns
        -------
        IntervalIndex
        """
        # Note: this is about 3.25x faster than super()._intersection(other)
        #  in IntervalIndexMethod.time_intersection_both_duplicate(1000)
        mask = np.zeros(len(self), dtype=bool)

        if self.hasnans and other.hasnans:
            first_nan_loc = np.arange(len(self))[self.isna()][0]
            mask[first_nan_loc] = True

        other_tups = set(zip(other.left, other.right))
        for i, tup in enumerate(zip(self.left, self.right)):
            if tup in other_tups:
                mask[i] = True

        return self[mask]

    # --------------------------------------------------------------------

    def _get_engine_target(self) -> np.ndarray:
        # Note: we _could_ use libjoin functions by either casting to object
        #  dtype or constructing tuples (faster than constructing Intervals)
        #  but the libjoin fastpaths are no longer fast in these cases.
        raise NotImplementedError(
            "IntervalIndex does not use libjoin fastpaths or pass values to "
            "IndexEngine objects"
        )

    def _from_join_target(self, result):
        raise NotImplementedError("IntervalIndex does not use libjoin fastpaths")

    # TODO: arithmetic operations


def _is_valid_endpoint(endpoint) -> bool:
    """
    Helper for interval_range to check if start/end are valid types.
    """
    return any(
        [
            is_number(endpoint),
            isinstance(endpoint, Timestamp),
            isinstance(endpoint, Timedelta),
            endpoint is None,
        ]
    )


def _is_type_compatible(a, b) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
    is_ts_compat = lambda x: isinstance(x, (Timestamp, BaseOffset))
    is_td_compat = lambda x: isinstance(x, (Timedelta, BaseOffset))
    return (
        (is_number(a) and is_number(b))
        or (is_ts_compat(a) and is_ts_compat(b))
        or (is_td_compat(a) and is_td_compat(b))
        or com.any_none(a, b)
    )


def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name: Hashable | None = None,
    closed: IntervalClosedType = "right",
) -> IntervalIndex:
    """
    Return a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric or datetime-like, default None
        Left bound for generating intervals.
    end : numeric or datetime-like, default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate.
    freq : numeric, str, Timedelta, datetime.timedelta, or DateOffset, default None
        The length of each interval. Must be consistent with the type of start
        and end, e.g. 2 for numeric, or '5H' for datetime-like.  Default is 1
        for numeric and 'D' for datetime-like.
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both
        or neither.

    Returns
    -------
    IntervalIndex

    See Also
    --------
    IntervalIndex : An Index of intervals that are all closed on the same side.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``IntervalIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end``, inclusively.

    To learn more about datetime-like frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    Numeric ``start`` and  ``end`` is supported.

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    Additionally, datetime-like input is also supported.

    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),
    ...                   end=pd.Timestamp('2017-01-04'))
    IntervalIndex([(2017-01-01 00:00:00, 2017-01-02 00:00:00],
                   (2017-01-02 00:00:00, 2017-01-03 00:00:00],
                   (2017-01-03 00:00:00, 2017-01-04 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    The ``freq`` parameter specifies the frequency between the left and right.
    endpoints of the individual intervals within the ``IntervalIndex``.  For
    numeric ``start`` and ``end``, the frequency must also be numeric.

    >>> pd.interval_range(start=0, periods=4, freq=1.5)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
                  dtype='interval[float64, right]')

    Similarly, for datetime-like ``start`` and ``end``, the frequency must be
    convertible to a DateOffset.

    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),
    ...                   periods=3, freq='MS')
    IntervalIndex([(2017-01-01 00:00:00, 2017-02-01 00:00:00],
                   (2017-02-01 00:00:00, 2017-03-01 00:00:00],
                   (2017-03-01 00:00:00, 2017-04-01 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.interval_range(start=0, end=6, periods=4)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
              dtype='interval[float64, right]')

    The ``closed`` parameter specifies which endpoints of the individual
    intervals within the ``IntervalIndex`` are closed.

    >>> pd.interval_range(end=5, periods=4, closed='both')
    IntervalIndex([[1, 2], [2, 3], [3, 4], [4, 5]],
                  dtype='interval[int64, both]')
    """
    start = maybe_box_datetimelike(start)
    end = maybe_box_datetimelike(end)
    endpoint = start if start is not None else end

    if freq is None and com.any_none(periods, start, end):
        freq = 1 if is_number(endpoint) else "D"

    if com.count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

    if not _is_valid_endpoint(start):
        raise ValueError(f"start must be numeric or datetime-like, got {start}")
    if not _is_valid_endpoint(end):
        raise ValueError(f"end must be numeric or datetime-like, got {end}")

    periods = validate_periods(periods)

    if freq is not None and not is_number(freq):
        try:
            freq = to_offset(freq)
        except ValueError as err:
            raise ValueError(
                f"freq must be numeric or convertible to DateOffset, got {freq}"
            ) from err

    # verify type compatibility
    if not all(
        [
            _is_type_compatible(start, end),
            _is_type_compatible(start, freq),
            _is_type_compatible(end, freq),
        ]
    ):
        raise TypeError("start, end, freq need to be type compatible")

    # +1 to convert interval count to breaks count (n breaks = n-1 intervals)
    if periods is not None:
        periods += 1

    breaks: np.ndarray | TimedeltaIndex | DatetimeIndex

    if is_number(endpoint):
        if com.all_not_none(start, end, freq):
            # 0.1 ensures we capture end
            breaks = np.arange(start, end + (freq * 0.1), freq)
        else:
            # compute the period/start/end if unspecified (at most one)
            if periods is None:
                periods = int((end - start) // freq) + 1
            elif start is None:
                start = end - (periods - 1) * freq
            elif end is None:
                end = start + (periods - 1) * freq

            breaks = np.linspace(start, end, periods)
        if all(is_integer(x) for x in com.not_none(start, end, freq)):
            # np.linspace always produces float output

            # error: Argument 1 to "maybe_downcast_numeric" has incompatible type
            # "Union[ndarray[Any, Any], TimedeltaIndex, DatetimeIndex]";
            # expected "ndarray[Any, Any]"  [
            breaks = maybe_downcast_numeric(
                breaks,  # type: ignore[arg-type]
                np.dtype("int64"),
            )
    else:
        # delegate to the appropriate range function
        if isinstance(endpoint, Timestamp):
            breaks = date_range(start=start, end=end, periods=periods, freq=freq)
        else:
            breaks = timedelta_range(start=start, end=end, periods=periods, freq=freq)

    return IntervalIndex.from_breaks(breaks, name=name, closed=closed)
<|endoftext|>"""
Base and utility classes for tseries type pandas objects.
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._config import using_copy_on_write

from pandas._libs import (
    NaT,
    Timedelta,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Resolution,
    Tick,
    parsing,
    to_offset,
)
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.compat.numpy import function as nv
from pandas.errors import (
    InvalidIndexError,
    NullFrequencyError,
)
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
)
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pandas._typing import (
        Axis,
        Self,
        npt,
    )

    from pandas import CategoricalIndex

_index_doc_kwargs = dict(ibase._index_doc_kwargs)


class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """

    _can_hold_strings = False
    _data: DatetimeArray | TimedeltaArray | PeriodArray

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> BaseOffset | None:
        return self._data.freq

    @freq.setter
    def freq(self, value) -> None:
        # error: Property "freq" defined in "PeriodArray" is read-only  [misc]
        self._data.freq = value  # type: ignore[misc]

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> str:
        from pandas import PeriodIndex

        if self._data.freqstr is not None and isinstance(
            self._data, (PeriodArray, PeriodIndex)
        ):
            freq = freq_to_period_freqstr(self._data.freq.n, self._data.freq.name)
            return freq
        else:
            return self._data.freqstr  # type: ignore[return-value]

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> str:
        return self._data.resolution

    # ------------------------------------------------------------------------

    @cache_readonly
    def hasnans(self) -> bool:
        return self._data._hasna

    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in "iufc":
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast("CategoricalIndex", other)
                should_try = other.categories.inferred_type in inferable

            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    # e.g.
                    #  ValueError -> cannot parse str entry, or OutOfBoundsDatetime
                    #  TypeError  -> trying to convert IntervalIndex to DatetimeIndex
                    #  OverflowError -> Index([very_large_timedeltas])
                    return False

        if self.dtype != other.dtype:
            # have different timezone
            return False

        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)

    # --------------------------------------------------------------------
    # Rendering Methods
    _default_na_rep = "NaT"

    def format(
        self,
        name: bool = False,
        formatter: Callable | None = None,
        na_rep: str = "NaT",
        date_format: str | None = None,
    ) -> list[str]:
        """
        Render a string representation of the Index.
        """
        warnings.warn(
            # GH#55413
            f"{type(self).__name__}.format is deprecated and will be removed "
            "in a future version. Convert using index.astype(str) or "
            "index.map(formatter) instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        header = []
        if name:
            header.append(
                ibase.pprint_thing(self.name, escape_chars=("\t", "\r", "\n"))
                if self.name is not None
                else ""
            )

        if formatter is not None:
            return header + list(self.map(formatter))

        return self._format_with_header(
            header=header, na_rep=na_rep, date_format=date_format
        )

    def _format_with_header(
        self, *, header: list[str], na_rep: str, date_format: str | None = None
    ) -> list[str]:
        # TODO: not reached in tests 2023-10-11
        # matches base class except for whitespace padding and date_format
        return header + list(
            self._get_values_for_csv(na_rep=na_rep, date_format=date_format)
        )

    @property
    def _formatter_func(self):
        return self._data._formatter()

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            # iterating over _attributes prevents us from doing this for PeriodIndex
            if attrib == "freq":
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)  # e.g. D -> 'D'
                attrs.append(("freq", freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name=None) -> str:
        result = super()._summary(name=name)
        if self.freq:
            result += f"\nFreq: {self.freqstr}"

        return result

    # --------------------------------------------------------------------
    # Indexing Methods

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        # e.g. test_getitem_setitem_periodindex
        # History of conversation GH#3452, GH#3931, GH#2369, GH#14826
        return reso > self._resolution_obj
        # NB: for DTI/PI, not TDI

    def _parsed_string_to_bounds(self, reso: Resolution, parsed):
        raise NotImplementedError

    def _parse_with_reso(self, label: str):
        # overridden by TimedeltaIndex
        try:
            if self.freq is None or hasattr(self.freq, "rule_code"):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, "freqstr", getattr(self, "inferred_freq", None))

        freqstr: str | None
        if freq is not None and not isinstance(freq, str):
            freqstr = freq.rule_code
        else:
            freqstr = freq

        if isinstance(label, np.str_):
            # GH#45580
            label = str(label)

        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return parsed, reso

    def _get_string_slice(self, key: str):
        # overridden by TimedeltaIndex
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(
        self,
        reso: Resolution,
        parsed: datetime,
    ) -> slice | npt.NDArray[np.intp]:
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        if not self._can_partial_date_slice(reso):
            raise ValueError

        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox

        if self.is_monotonic_increasing:
            if len(self) and (
                (t1 < self[0] and t2 < self[0]) or (t1 > self[-1] and t2 > self[-1])
            ):
                # we are out of range
                raise KeyError

            # TODO: does this depend on being monotonic _increasing_?

            # a monotonic (sorted) series can be sliced
            left = vals.searchsorted(unbox(t1), side="left")
            right = vals.searchsorted(unbox(t2), side="right")
            return slice(left, right)

        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)

            # try to find the dates
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label, side: str):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                # DTI -> parsing.DateParseError
                # TDI -> 'unit abbreviation w/o a number'
                # PI -> string cannot be parsed as datetime-like
                self._raise_invalid_indexer("slice", label, err)

            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer("slice", label)

        return label

    # --------------------------------------------------------------------
    # Arithmetic Methods

    def shift(self, periods: int = 1, freq=None) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr):
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                # e.g. we don't want to cast DTA to ndarray[object]
                res = com.asarray_tuplesafe(keyarr)
                # TODO: com.asarray_tuplesafe shouldn't cast e.g. DatetimeArray
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)


class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """

    _data: DatetimeArray | TimedeltaArray
    _comparables = ["name", "freq"]
    _attributes = ["name", "freq"]

    # Compat for frequency inference, see GH#23789
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def unit(self) -> str:
        return self._data.unit

    def as_unit(self, unit: str) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(['2020-01-02 01:02:03.004005006'])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit('s')
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(['1 day 3 min 2 us 42 ns'])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit('s')
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def _with_freq(self, freq):
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        # NB: For Datetime64TZ this is lossy
        data = self._data._ndarray
        if using_copy_on_write():
            data = data.view()
            data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq=None) -> Self:
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset

        if periods == 0 or len(self) == 0:
            # GH#14811 empty case
            return self.copy()

        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")

        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq

        # Note: in the DatetimeTZ case, _generate_range will infer the
        #  appropriate timezone from `start` and `end`, so tz does not need
        #  to be passed explicitly.
        result = self._data._generate_range(
            start=start, end=end, periods=None, freq=self.freq, unit=self.unit
        )
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def inferred_freq(self) -> str | None:
        return self._data.inferred_freq

    # --------------------------------------------------------------------
    # Set Operation Methods

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        # Convert our i8 representations to RangeIndex
        # Caller is responsible for checking isinstance(self.freq, Tick)
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit("ns")._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other) -> bool:
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other, res_i8) -> Self:
        new_freq = None
        if not len(res_i8):
            # RangeIndex defaults to step=1, which we don't want.
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(Timedelta(res_i8.step))

        # TODO(GH#41493): we cannot just do
        #  type(self._data)(res_i8.values, dtype=self.dtype, freq=new_freq)
        # because test_setops_preserve_freq fails with _validate_frequency raising.
        # This raising is incorrect, as 'on_freq' is incorrect. This will
        # be fixed by GH#41493
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(
            # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
            # incompatible type "Union[dtype[Any], ExtensionDtype]"; expected
            # "Union[dtype[datetime64], DatetimeTZDtype]"
            res_values,
            dtype=self.dtype,  # type: ignore[arg-type]
            freq=new_freq,  # type: ignore[arg-type]
        )
        return cast("Self", self._wrap_setop_result(other, result))

    def _range_intersect(self, other, sort) -> Self:
        # Dispatch to RangeIndex intersection logic.
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other, sort) -> Self:
        # Dispatch to RangeIndex union logic.
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _intersection(self, other: Index, sort: bool = False) -> Index:
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
        other = cast("DatetimeTimedeltaMixin", other)

        if self._can_range_setop(other):
            return self._range_intersect(other, sort=sort)

        if not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            # We need to invalidate the freq because Index._intersection
            #  uses _shallow_copy on a view of self._data, which will preserve
            #  self.freq if we're not careful.
            # At this point we should have result.dtype == self.dtype
            #  and type(result) is type(self._data)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq("infer")

        else:
            return self._fast_intersect(other, sort)

    def _fast_intersect(self, other, sort):
        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        # after sorting, the intersection always starts with the right index
        # and ends with the index of which the last elements is smallest
        end = min(left[-1], right[-1])
        start = right[0]

        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]

        return result

    def _can_fast_intersect(self, other: Self) -> bool:
        # Note: we only get here with len(self) > 0 and len(other) > 0
        if self.freq is None:
            return False

        elif other.freq != self.freq:
            return False

        elif not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            return False

        # this along with matching freqs ensure that we "line up",
        #  so intersection will preserve freq
        # Note we are assuming away Ticks, as those go through _range_intersect
        # GH#42104
        return self.freq.n == 1

    def _can_fast_union(self, other: Self) -> bool:
        # Assumes that type(self) == type(other), as per the annotation
        # The ability to fast_union also implies that `freq` should be
        #  retained on union.
        freq = self.freq

        if freq is None or freq != other.freq:
            return False

        if not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            # TODO: do union on the reversed indexes?
            return False

        if len(self) == 0 or len(other) == 0:
            # only reached via union_many
            return True

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        right_start = right[0]
        left_end = left[-1]

        # Only need to "adjoin", not overlap
        return (right_start == left_end + freq) or right_start in left

    def _fast_union(self, other: Self, sort=None) -> Self:
        # Caller is responsible for ensuring self and other are non-empty

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        elif sort is False:
            # TDIs are not in the "correct" order and we don't want
            #  to sort but want to remove overlaps
            left, right = self, other
            left_start = left[0]
            loc = right.searchsorted(left_start, side="left")
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result
        else:
            left, right = other, self

        left_end = left[-1]
        right_end = right[-1]

        # concatenate
        if left_end < right_end:
            loc = right.searchsorted(left_end, side="right")
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            # The can_fast_union check ensures that the result.freq
            #  should match self.freq
            assert isinstance(dates, type(self._data))
            # error: Item "ExtensionArray" of "ExtensionArray |
            # ndarray[Any, Any]" has no attribute "_freq"
            assert dates._freq == self.freq  # type: ignore[union-attr]
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other, sort):
        # We are called by `union`, which is responsible for this validation
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype

        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)

        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            # in the case with sort=None, the _can_fast_union check ensures
            #  that result.freq == self.freq
            return result
        else:
            return super()._union(other, sort)._with_freq("infer")

    # --------------------------------------------------------------------
    # Join Methods

    def _get_join_freq(self, other):
        """
        Get the freq to attach to the result of a join operation.
        """
        freq = None
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def _wrap_joined_index(
        self, joined, other, lidx: npt.NDArray[np.intp], ridx: npt.NDArray[np.intp]
    ):
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        result = super()._wrap_joined_index(joined, other, lidx, ridx)
        result._data._freq = self._get_join_freq(other)
        return result

    def _get_engine_target(self) -> np.ndarray:
        # engine methods and libjoin methods need dt64/td64 values cast to i8
        return self._data._ndarray.view("i8")

    def _from_join_target(self, result: np.ndarray):
        # view e.g. i8 back to M8[ns]
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    # --------------------------------------------------------------------
    # List-like Methods

    def _get_delete_freq(self, loc: int | slice | Sequence[int]):
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        if self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    # error: Incompatible types in assignment (expression has
                    # type "Union[slice, ndarray]", variable has type
                    # "Union[int, slice, Sequence[int]]")
                    loc = lib.maybe_indices_to_slice(  # type: ignore[assignment]
                        np.asarray(loc, dtype=np.intp), len(self)
                    )
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc: int, item):
        """
        Find the `freq` for self.insert(loc, item).
        """
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)

        freq = None
        if self.freq is not None:
            # freq can be preserved on edge cases
            if self.size:
                if item is NaT:
                    pass
                elif loc in (0, -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif (loc == len(self)) and item - self.freq == self[-1]:
                    freq = self.freq
            else:
                # Adding a single item to an empty index may preserve freq
                if isinstance(self.freq, Tick):
                    # all TimedeltaIndex cases go through here; is_on_offset
                    #  would raise TypeError
                    freq = self.freq
                elif self.freq.is_on_offset(item):
                    freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc) -> Self:
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item):
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            # i.e. parent class method did not cast
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    # --------------------------------------------------------------------
    # NDArray-Like Methods

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)

        result = NDArrayBackedExtensionIndex.take(
            self, indices, axis, allow_fill, fill_value, **kwargs
        )

        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
<|endoftext|>"""
Shared methods for Index subclasses backed by ExtensionArray.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    TypeVar,
)

from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.generic import ABCDataFrame

from pandas.core.indexes.base import Index

if TYPE_CHECKING:
    import numpy as np

    from pandas._typing import (
        ArrayLike,
        npt,
    )

    from pandas.core.arrays import IntervalArray
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

_ExtensionIndexT = TypeVar("_ExtensionIndexT", bound="ExtensionIndex")


def _inherit_from_data(
    name: str, delegate: type, cache: bool = False, wrap: bool = False
):
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
    attr = getattr(delegate, name)

    if isinstance(attr, property) or type(attr).__name__ == "getset_descriptor":
        # getset_descriptor i.e. property defined in cython class
        if cache:

            def cached(self):
                return getattr(self._data, name)

            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            method = cache_readonly(cached)

        else:

            def fget(self):
                result = getattr(self._data, name)
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name, dtype=result.dtype)
                return result

            def fset(self, value) -> None:
                setattr(self._data, name, value)

            fget.__name__ = name
            fget.__doc__ = attr.__doc__

            method = property(fget, fset)

    elif not callable(attr):
        # just a normal attribute, no wrapping
        method = attr

    else:
        # error: Incompatible redefinition (redefinition with type "Callable[[Any,
        # VarArg(Any), KwArg(Any)], Any]", original type "property")
        def method(self, *args, **kwargs):  # type: ignore[misc]
            if "inplace" in kwargs:
                raise ValueError(f"cannot use inplace with {type(self).__name__}")
            result = attr(self._data, *args, **kwargs)
            if wrap:
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                return Index(result, name=self.name, dtype=result.dtype)
            return result

        # error: "property" has no attribute "__name__"
        method.__name__ = name  # type: ignore[attr-defined]
        method.__doc__ = attr.__doc__
    return method


def inherit_names(
    names: list[str], delegate: type, cache: bool = False, wrap: bool = False
) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """

    def wrapper(cls: type[_ExtensionIndexT]) -> type[_ExtensionIndexT]:
        for name in names:
            meth = _inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)

        return cls

    return wrapper


class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """

    # The base class already passes through to _data:
    #  size, __len__, dtype

    _data: IntervalArray | NDArrayBackedExtensionArray

    # ---------------------------------------------------------------------

    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """
        return self._data._validate_setitem_value(value)

    @cache_readonly
    def _isnan(self) -> npt.NDArray[np.bool_]:
        # error: Incompatible return value type (got "ExtensionArray", expected
        # "ndarray")
        return self._data.isna()  # type: ignore[return-value]


class NDArrayBackedExtensionIndex(ExtensionIndex):
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """

    _data: NDArrayBackedExtensionArray

    def _get_engine_target(self) -> np.ndarray:
        return self._data._ndarray

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        assert result.dtype == self._data._ndarray.dtype
        return self._data._from_backing_data(result)
<|endoftext|>from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import index as libindex
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    Period,
    Resolution,
    Tick,
)
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
    raise_on_incompatible,
    validate_dtype_freq,
)
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    Index,
)
from pandas.core.indexes.extension import inherit_names

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )


_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({"target_klass": "PeriodIndex or list of Periods"})
_shared_doc_kwargs = {
    "klass": "PeriodArray",
}

# --- Period index sketch


def _new_PeriodIndex(cls, **d):
    # GH13277 for unpickling
    values = d.pop("data")
    if values.dtype == "int64":
        freq = d.pop("freq", None)
        dtype = PeriodDtype(freq)
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


@inherit_names(
    ["strftime", "start_time", "end_time"] + PeriodArray._field_ops,
    PeriodArray,
    wrap=True,
)
@inherit_names(["is_leap_year"], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    copy : bool
        Make a copy of input ndarray.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    year : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    month : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    quarter : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    day : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    hour : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    minute : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    second : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    dtype : str or PeriodDtype, default None

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp
    from_fields
    from_ordinals

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex.from_fields(year=[2000, 2002], quarter=[1, 3])
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """

    _typ = "periodindex"

    _data: PeriodArray
    freq: BaseOffset
    dtype: PeriodDtype

    _data_cls = PeriodArray
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]:
        return libindex.PeriodEngine

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        # for compat with DatetimeIndex
        return self.dtype._resolution_obj

    # --------------------------------------------------------------------
    # methods that dispatch to array and wrap result in Index
    # These are defined here instead of via inherit_names for mypy

    @doc(
        PeriodArray.asfreq,
        other="pandas.arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs,
    )
    def asfreq(self, freq=None, how: str = "E") -> Self:
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeIndex:
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Index:
        return Index(self._data.hour, name=self.name)

    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Index:
        return Index(self._data.minute, name=self.name)

    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Index:
        return Index(self._data.second, name=self.name)

    # ------------------------------------------------------------------------
    # Index Constructors

    def __new__(
        cls,
        data=None,
        ordinal=None,
        freq=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
        **fields,
    ) -> Self:
        valid_field_set = {
            "year",
            "month",
            "day",
            "quarter",
            "hour",
            "minute",
            "second",
        }

        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        if not set(fields).issubset(valid_field_set):
            argument = next(iter(set(fields) - valid_field_set))
            raise TypeError(f"__new__() got an unexpected keyword argument {argument}")
        elif len(fields):
            # GH#55960
            warnings.warn(
                "Constructing PeriodIndex from fields is deprecated. Use "
                "PeriodIndex.from_fields instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if ordinal is not None:
            # GH#55960
            warnings.warn(
                "The 'ordinal' keyword in PeriodIndex is deprecated and will "
                "be removed in a future version. Use PeriodIndex.from_ordinals "
                "instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        name = maybe_extract_name(name, data, cls)

        if data is None and ordinal is None:
            # range-based.
            if not fields:
                # test_pickle_compat_construction
                cls._raise_scalar_data_error(None)
            data = cls.from_fields(**fields, freq=freq)._data
            copy = False

        elif fields:
            if data is not None:
                raise ValueError("Cannot pass both data and fields")
            raise ValueError("Cannot pass both ordinal and fields")

        else:
            freq = validate_dtype_freq(dtype, freq)

            # PeriodIndex allow PeriodIndex(period_index, freq=different)
            # Let's not encourage that kind of behavior in PeriodArray.

            if freq and isinstance(data, cls) and data.freq != freq:
                # TODO: We can do some of these with no-copy / coercion?
                # e.g. D -> 2D seems to be OK
                data = data.asfreq(freq)

            if data is None and ordinal is not None:
                ordinal = np.asarray(ordinal, dtype=np.int64)
                dtype = PeriodDtype(freq)
                data = PeriodArray(ordinal, dtype=dtype)
            elif data is not None and ordinal is not None:
                raise ValueError("Cannot pass both data and ordinal")
            else:
                # don't pass copy here, since we copy later.
                data = period_array(data=data, freq=freq)

        if copy:
            data = data.copy()

        return cls._simple_new(data, name=name, refs=refs)

    @classmethod
    def from_fields(
        cls,
        *,
        year=None,
        quarter=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        freq=None,
    ) -> Self:
        fields = {
            "year": year,
            "quarter": quarter,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        arr = PeriodArray._from_fields(fields=fields, freq=freq)
        return cls._simple_new(arr)

    @classmethod
    def from_ordinals(cls, ordinals, *, freq, name=None) -> Self:
        ordinals = np.asarray(ordinals, dtype=np.int64)
        dtype = PeriodDtype(freq)
        data = PeriodArray._simple_new(ordinals, dtype=dtype)
        return cls._simple_new(data, name=name)

    # ------------------------------------------------------------------------
    # Data

    @property
    def values(self) -> npt.NDArray[np.object_]:
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other) -> int | npt.NDArray[np.int64]:
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                # _check_timedeltalike_freq_compat will raise if incompatible
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n

            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            assert isinstance(other, int)
            return other

        # raise when input doesn't have freq
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return self.dtype == dtype

    # ------------------------------------------------------------------------
    # Index Methods

    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> np.ndarray:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        elif not isinstance(where, PeriodIndex):
            raise TypeError("asof_locs `where` must be DatetimeIndex or PeriodIndex")

        return super().asof_locs(where, mask)

    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise ValueError("Index is not monotonic")
        values = self.asi8
        return bool(((values[1:] - values[:-1]) < 2).all())

    @property
    def inferred_type(self) -> str:
        # b/c data is represented as ints make sure we can't have ambiguous
        # indexing
        return "period"

    # ------------------------------------------------------------------------
    # Indexing Methods

    def _convert_tolerance(self, tolerance, target):
        # Returned tolerance must be in dtype/units so that
        #  `|self._get_engine_target() - target._engine_target()| <= tolerance`
        #  is meaningful.  Since PeriodIndex returns int64 for engine_target,
        #  we may need to convert timedelta64 tolerance to int64.
        tolerance = super()._convert_tolerance(tolerance, target)

        if self.dtype == target.dtype:
            # convert tolerance to i8
            tolerance = self._maybe_convert_timedelta(tolerance)

        return tolerance

    def get_loc(self, key):
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]

        Raises
        ------
        KeyError
            Key is not present in the index.
        TypeError
            If key is listlike or otherwise not hashable.
        """
        orig_key = key

        self._check_indexing_error(key)

        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except ValueError as err:
                # A string with invalid format
                raise KeyError(f"Cannot interpret '{key}' as period") from err

            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err

            if reso == self._resolution_obj:
                # the reso < self._resolution_obj case goes
                #  through _get_string_slice
                key = self._cast_partial_indexing_scalar(parsed)
            else:
                raise KeyError(key)

        elif isinstance(key, Period):
            self._disallow_mismatched_indexing(key)

        elif isinstance(key, datetime):
            key = self._cast_partial_indexing_scalar(key)

        else:
            # in particular integer, which Period constructor would cast to string
            raise KeyError(key)

        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _disallow_mismatched_indexing(self, key: Period) -> None:
        if key._dtype != self.dtype:
            raise KeyError(key)

    def _cast_partial_indexing_scalar(self, label: datetime) -> Period:
        try:
            period = Period(label, freq=self.freq)
        except ValueError as err:
            # we cannot construct the Period
            raise KeyError(label) from err
        return period

    @doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        if isinstance(label, datetime):
            label = self._cast_partial_indexing_scalar(label)

        return super()._maybe_cast_slice_bound(label, side)

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime):
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        iv = Period(parsed, freq=freq)
        return (iv.asfreq(self.freq, how="start"), iv.asfreq(self.freq, how="end"))

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq=None) -> Self:
        if freq is not None:
            raise TypeError(
                f"`freq` argument is not supported for {type(self).__name__}.shift"
            )
        return self + periods


def period_range(
    start=None,
    end=None,
    periods: int | None = None,
    freq=None,
    name: Hashable | None = None,
) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period('2017Q1', freq='Q'),
    ...                 end=pd.Period('2017Q2', freq='Q'), freq='M')
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
    if freq is None and (not isinstance(start, Period) and not isinstance(end, Period)):
        freq = "D"

    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterator,
)
from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.algos import unique_deltas
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    deprecate_nonkeyword_arguments,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
)
from pandas.core.dtypes.generic import ABCTimedeltaIndex

from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Dtype,
        NaPosition,
        Self,
        npt,
    )
_empty_range = range(0)
_dtype_int64 = np.dtype(np.int64)


class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
    step : int (default: 1)
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    """

    _typ = "rangeindex"
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")
    _range: range
    _values: np.ndarray

    @property
    def _engine_type(self) -> type[libindex.Int64Engine]:
        return libindex.Int64Engine

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        start=None,
        stop=None,
        step=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)

        # RangeIndex
        if isinstance(start, cls):
            return start.copy(name=name)
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)

        # validate the arguments
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")

        start = ensure_python_int(start) if start is not None else 0

        if stop is None:
            start, stop = 0, start
        else:
            stop = ensure_python_int(stop)

        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError("Step must not be zero")

        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(cls, data: range, name=None, dtype: Dtype | None = None) -> Self:
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.

        Returns
        -------
        RangeIndex

        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)

        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a "
                f"range, {repr(data)} was passed"
            )
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    #  error: Argument 1 of "_simple_new" is incompatible with supertype "Index";
    #  supertype defines the argument type as
    #  "Union[ExtensionArray, ndarray[Any, Any]]"  [override]
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls, values: range, name: Hashable | None = None
    ) -> Self:
        result = object.__new__(cls)

        assert isinstance(values, range)

        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        result._references = None
        return result

    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None:
        if dtype is None:
            return

        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    # --------------------------------------------------------------------

    # error: Return type "Type[Index]" of "_constructor" incompatible with return
    # type "Type[RangeIndex]" in supertype "Index"
    @cache_readonly
    def _constructor(self) -> type[Index]:  # type: ignore[override]
        """return the class to use for construction"""
        return Index

    # error: Signature of "_data" incompatible with supertype "Index"
    @cache_readonly
    def _data(self) -> np.ndarray:  # type: ignore[override]
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> list[tuple[str, int]]:
        """return a list of tuples of start, stop, step"""
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self):
        d = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------
    # Rendering Methods

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs = cast("list[tuple[str, str | int]]", self._get_data_as_items())
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]:
        # Equivalent to Index implementation, but faster
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))

        return header + [f"{x:<{max_length}}" for x in self._range]

    # --------------------------------------------------------------------

    @property
    def start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
        # GH 25710
        return self._range.start

    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
        return self._range.stop

    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
        # GH 25710
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            key = ensure_python_int(key)
        except TypeError:
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        return "integer"

    # --------------------------------------------------------------------
    # Indexing Methods

    @doc(Index.get_loc)
    def get_loc(self, key) -> int:
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key = int(key)
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> npt.NDArray[np.intp]:
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target, method=method, tolerance=tolerance, limit=limit
            )

        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # GH 28678: work on reversed range for simplicity
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        target_array = np.asarray(target)
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step

        if step != self.step:
            # We reversed this range: transform to original locs
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return False

    # --------------------------------------------------------------------

    def tolist(self) -> list[int]:
        return list(self._range)

    @doc(Index.__iter__)
    def __iter__(self) -> Iterator[int]:
        yield from self._range

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values, name: Hashable = no_default):
        name = self._name if name is no_default else name

        if values.dtype.kind == "f":
            return Index(values, name=name, dtype=np.float64)
        # GH 46675 & 43885: If values is equally spaced, return a
        # more memory-compact RangeIndex instead of Index with 64-bit dtype
        unique_diffs = unique_deltas(values)
        if len(unique_diffs) == 1 and unique_diffs[0] != 0:
            diff = unique_diffs[0]
            new_range = range(values[0], values[-1] + diff, diff)
            return type(self)._simple_new(new_range, name=name)
        else:
            return self._constructor._simple_new(values, name=name)

    def _view(self) -> Self:
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    @doc(Index.copy)
    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self:
        name = self._validate_names(name=name, deep=deep)[0]
        new_index = self._rename(name=name)
        return new_index

    def _minmax(self, meth: str):
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start

        return self.start + self.step * no_steps

    def min(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        """The maximum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        ascending = kwargs.pop("ascending", True)  # EA compat
        kwargs.pop("kind", None)  # e.g. "mergesort" is irrelevant
        nv.validate_argsort(args, kwargs)

        if self._range.step > 0:
            result = np.arange(len(self), dtype=np.intp)
        else:
            result = np.arange(len(self) - 1, -1, -1, dtype=np.intp)

        if not ascending:
            result = result[::-1]
        return result

    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], RangeIndex]:
        codes = np.arange(len(self), dtype=np.intp)
        uniques = self
        if sort and self.step < 0:
            codes = codes[::-1]
            uniques = uniques[::-1]
        return codes, uniques

    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    # error: Signature of "sort_values" incompatible with supertype "Index"
    @overload  # type: ignore[override]
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self:
        ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> tuple[Self, np.ndarray | RangeIndex]:
        ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]:
        ...

    @deprecate_nonkeyword_arguments(
        version="3.0", allowed_args=["self"], name="sort_values"
    )
    def sort_values(
        self,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable | None = None,
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]:
        if key is not None:
            return super().sort_values(
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        else:
            sorted_index = self
            inverse_indexer = False
            if ascending:
                if self.step < 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True
            else:
                if self.step > 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True

        if return_indexer:
            if inverse_indexer:
                rng = range(len(self) - 1, -1, -1)
            else:
                rng = range(len(self))
            return sorted_index, RangeIndex(rng)
        else:
            return sorted_index

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other: Index, sort: bool = False):
        # caller is responsible for checking self and other are both non-empty

        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)

        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # check whether intervals intersect
        # deals with in- and decreasing ranges
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return self._simple_new(_empty_range)

        # Method hint: linear Diophantine equation
        # solve intersection problem
        # performance hint: for identical step sizes, could use
        # cheaper alternative
        gcd, s, _ = self._extended_gcd(first.step, second.step)

        # check whether element sets intersect
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)

        # calculate parameters for the RangeIndex describing the
        # intersection disregarding the lower bounds
        tmp_start = first.start + (second.start - first.start) * first.step // gcd * s
        new_step = first.step * second.step // gcd
        new_range = range(tmp_start, int_high, new_step)
        new_index = self._simple_new(new_range)

        # adjust index to limiting interval
        new_start = new_index._min_fitting_element(int_low)
        new_range = range(new_start, new_index.stop, new_index.step)
        new_index = self._simple_new(new_range)

        if (self.step < 0 and other.step < 0) is not (new_index.step < 0):
            new_index = new_index[::-1]

        if sort is None:
            new_index = new_index.sort_values()

        return new_index

    def _min_fitting_element(self, lower_limit: int) -> int:
        """Returns the smallest element greater than or equal to the limit"""
        no_steps = -(-(lower_limit - self.start) // abs(self.step))
        return self.start + abs(self.step) * no_steps

    def _extended_gcd(self, a: int, b: int) -> tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        return old_r, old_s, old_t

    def _range_in_self(self, other: range) -> bool:
        """Check if other range is contained in self"""
        # https://stackoverflow.com/a/32481015
        if not other:
            return True
        if not self._range:
            return False
        if len(other) > 1 and other.step % self._range.step:
            return False
        return other.start in self._range and other[-1] in self._range

    def _union(self, other: Index, sort: bool | None):
        """
        Form the union of two Index objects and sorts if possible

        Parameters
        ----------
        other : Index or array-like

        sort : bool or None, default None
            Whether to sort (monotonically increasing) the resulting index.
            ``sort=None|True`` returns a ``RangeIndex`` if possible or a sorted
            ``Index`` with a int64 dtype if not.
            ``sort=False`` can return a ``RangeIndex`` if self is monotonically
            increasing and other is fully contained in self. Otherwise, returns
            an unsorted ``Index`` with an int64 dtype.

        Returns
        -------
        union : Index
        """
        if isinstance(other, RangeIndex):
            if sort in (None, True) or (
                sort is False and self.step > 0 and self._range_in_self(other._range)
            ):
                # GH 47557: Can still return a RangeIndex
                # if other range in self and sort=False
                start_s, step_s = self.start, self.step
                end_s = self.start + self.step * (len(self) - 1)
                start_o, step_o = other.start, other.step
                end_o = other.start + other.step * (len(other) - 1)
                if self.step < 0:
                    start_s, step_s, end_s = end_s, -step_s, start_s
                if other.step < 0:
                    start_o, step_o, end_o = end_o, -step_o, start_o
                if len(self) == 1 and len(other) == 1:
                    step_s = step_o = abs(self.start - other.start)
                elif len(self) == 1:
                    step_s = step_o
                elif len(other) == 1:
                    step_o = step_s
                start_r = min(start_s, start_o)
                end_r = max(end_s, end_o)
                if step_o == step_s:
                    if (
                        (start_s - start_o) % step_s == 0
                        and (start_s - end_o) <= step_s
                        and (start_o - end_s) <= step_s
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                    if (
                        (step_s % 2 == 0)
                        and (abs(start_s - start_o) == step_s / 2)
                        and (abs(end_s - end_o) == step_s / 2)
                    ):
                        # e.g. range(0, 10, 2) and range(1, 11, 2)
                        #  but not range(0, 20, 4) and range(1, 21, 4) GH#44019
                        return type(self)(start_r, end_r + step_s / 2, step_s / 2)

                elif step_o % step_s == 0:
                    if (
                        (start_o - start_s) % step_s == 0
                        and (start_o + step_s >= start_s)
                        and (end_o - step_s <= end_s)
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                elif step_s % step_o == 0:
                    if (
                        (start_s - start_o) % step_o == 0
                        and (start_s + step_o >= start_o)
                        and (end_s - step_o <= end_o)
                    ):
                        return type(self)(start_r, end_r + step_o, step_o)

        return super()._union(other, sort=sort)

    def _difference(self, other, sort=None):
        # optimized set operation if we have another RangeIndex
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        if not isinstance(other, RangeIndex):
            return super()._difference(other, sort=sort)

        if sort is not False and self.step < 0:
            return self[::-1]._difference(other)

        res_name = ops.get_op_result_name(self, other)

        first = self._range[::-1] if self.step < 0 else self._range
        overlap = self.intersection(other)
        if overlap.step < 0:
            overlap = overlap[::-1]

        if len(overlap) == 0:
            return self.rename(name=res_name)
        if len(overlap) == len(self):
            return self[:0].rename(res_name)

        # overlap.step will always be a multiple of self.step (see _intersection)

        if len(overlap) == 1:
            if overlap[0] == self[0]:
                return self[1:]

            elif overlap[0] == self[-1]:
                return self[:-1]

            elif len(self) == 3 and overlap[0] == self[1]:
                return self[::2]

            else:
                return super()._difference(other, sort=sort)

        elif len(overlap) == 2 and overlap[0] == first[0] and overlap[-1] == first[-1]:
            # e.g. range(-8, 20, 7) and range(13, -9, -3)
            return self[1:-1]

        if overlap.step == first.step:
            if overlap[0] == first.start:
                # The difference is everything after the intersection
                new_rng = range(overlap[-1] + first.step, first.stop, first.step)
            elif overlap[-1] == first[-1]:
                # The difference is everything before the intersection
                new_rng = range(first.start, overlap[0], first.step)
            elif overlap._range == first[1:-1]:
                # e.g. range(4) and range(1, 3)
                step = len(first) - 1
                new_rng = first[::step]
            else:
                # The difference is not range-like
                # e.g. range(1, 10, 1) and range(3, 7, 1)
                return super()._difference(other, sort=sort)

        else:
            # We must have len(self) > 1, bc we ruled out above
            #  len(overlap) == 0 and len(overlap) == len(self)
            assert len(self) > 1

            if overlap.step == first.step * 2:
                if overlap[0] == first[0] and overlap[-1] in (first[-1], first[-2]):
                    # e.g. range(1, 10, 1) and range(1, 10, 2)
                    new_rng = first[1::2]

                elif overlap[0] == first[1] and overlap[-1] in (first[-1], first[-2]):
                    # e.g. range(1, 10, 1) and range(2, 10, 2)
                    new_rng = first[::2]

                else:
                    # We can get here with  e.g. range(20) and range(0, 10, 2)
                    return super()._difference(other, sort=sort)

            else:
                # e.g. range(10) and range(0, 10, 3)
                return super()._difference(other, sort=sort)

        new_index = type(self)._simple_new(new_rng, name=res_name)
        if first is not self._range:
            new_index = new_index[::-1]

        return new_index

    def symmetric_difference(
        self, other, result_name: Hashable | None = None, sort=None
    ):
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)

        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)

        if result_name is not None:
            result = result.rename(result_name)
        return result

    # --------------------------------------------------------------------

    # error: Return type "Index" of "delete" incompatible with return type
    #  "RangeIndex" in supertype "Index"
    def delete(self, loc) -> Index:  # type: ignore[override]
        # In some cases we can retain RangeIndex, see also
        #  DatetimeTimedeltaMixin._get_delete_Freq
        if is_integer(loc):
            if loc in (0, -len(self)):
                return self[1:]
            if loc in (-1, len(self) - 1):
                return self[:-1]
            if len(self) == 3 and loc in (1, -2):
                return self[::2]

        elif lib.is_list_like(loc):
            slc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))

            if isinstance(slc, slice):
                # defer to RangeIndex._difference, which is optimized to return
                #  a RangeIndex whenever possible
                other = self[slc]
                return self.difference(other, sort=False)

        return super().delete(loc)

    def insert(self, loc: int, item) -> Index:
        if len(self) and (is_integer(item) or is_float(item)):
            # We can retain RangeIndex is inserting at the beginning or end,
            #  or right in the middle.
            rng = self._range
            if loc == 0 and item == self[0] - self.step:
                new_rng = range(rng.start - rng.step, rng.stop, rng.step)
                return type(self)._simple_new(new_rng, name=self._name)

            elif loc == len(self) and item == self[-1] + self.step:
                new_rng = range(rng.start, rng.stop + rng.step, rng.step)
                return type(self)._simple_new(new_rng, name=self._name)

            elif len(self) == 2 and item == self[0] + self.step / 2:
                # e.g. inserting 1 into [0, 2]
                step = int(self.step / 2)
                new_rng = range(self.start, self.stop, step)
                return type(self)._simple_new(new_rng, name=self._name)

        return super().insert(loc, item)

    def _concat(self, indexes: list[Index], name: Hashable) -> Index:
        """
        Overriding parent method for the case of all RangeIndex instances.

        When all members of "indexes" are of type RangeIndex: result will be
        RangeIndex if possible, Index with a int64 dtype otherwise. E.g.:
        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)
        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Index([0,1,2,4,5], dtype='int64')
        """
        if not all(isinstance(x, RangeIndex) for x in indexes):
            return super()._concat(indexes, name)

        elif len(indexes) == 1:
            return indexes[0]

        rng_indexes = cast(list[RangeIndex], indexes)

        start = step = next_ = None

        # Filter the empty indexes
        non_empty_indexes = [obj for obj in rng_indexes if len(obj)]

        for obj in non_empty_indexes:
            rng = obj._range

            if start is None:
                # This is set by the first non-empty index
                start = rng.start
                if step is None and len(rng) > 1:
                    step = rng.step
            elif step is None:
                # First non-empty index had only one element
                if rng.start == start:
                    values = np.concatenate([x._values for x in rng_indexes])
                    result = self._constructor(values)
                    return result.rename(name)

                step = rng.start - start

            non_consecutive = (step != rng.step and len(rng) > 1) or (
                next_ is not None and rng.start != next_
            )
            if non_consecutive:
                result = self._constructor(
                    np.concatenate([x._values for x in rng_indexes])
                )
                return result.rename(name)

            if step is not None:
                next_ = rng[-1] + step

        if non_empty_indexes:
            # Get the stop value from "next" or alternatively
            # from the last non-empty index
            stop = non_empty_indexes[-1].stop if next_ is None else next_
            return RangeIndex(start, stop, step).rename(name)

        # Here all "indexes" had 0 length, i.e. were empty.
        # In this case return an empty range index.
        return RangeIndex(0, 0).rename(name)

    def __len__(self) -> int:
        """
        return the length of the RangeIndex
        """
        return len(self._range)

    @property
    def size(self) -> int:
        return len(self)

    def __getitem__(self, key):
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        if isinstance(key, slice):
            return self._getitem_slice(key)
        elif is_integer(key):
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(key):
            raise IndexError(
                "only integers, slices (`:`), "
                "ellipsis (`...`), numpy.newaxis (`None`) "
                "and integer or boolean "
                "arrays are valid indices"
            )
        return super().__getitem__(key)

    def _getitem_slice(self, slobj: slice) -> Self:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._range[slobj]
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        if is_integer(other) and other != 0:
            if len(self) == 0 or self.start % other == 0 and self.step % other == 0:
                start = self.start // other
                step = self.step // other
                stop = start + len(self) * step
                new_range = range(start, stop, step or 1)
                return self._simple_new(new_range, name=self._name)
            if len(self) == 1:
                start = self.start // other
                new_range = range(start, start + 1, 1)
                return self._simple_new(new_range, name=self._name)

        return super().__floordiv__(other)

    # --------------------------------------------------------------------
    # Reductions

    def all(self, *args, **kwargs) -> bool:
        return 0 not in self._range

    def any(self, *args, **kwargs) -> bool:
        return any(self._range)

    # --------------------------------------------------------------------

    def _cmp_method(self, other, op):
        if isinstance(other, RangeIndex) and self._range == other._range:
            # Both are immutable so if ._range attr. are equal, shortcut is possible
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def _arith_method(self, other, op):
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
            perform the binary op
        """

        if isinstance(other, ABCTimedeltaIndex):
            # Defer to TimedeltaIndex implementation
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            # GH#19333 is_integer evaluated True on timedelta64,
            # so we need to catch these explicitly
            return super()._arith_method(other, op)
        elif lib.is_np_dtype(getattr(other, "dtype", None), "m"):
            # Must be an np.ndarray; GH#22390
            return super()._arith_method(other, op)

        if op in [
            operator.pow,
            ops.rpow,
            operator.mod,
            ops.rmod,
            operator.floordiv,
            ops.rfloordiv,
            divmod,
            ops.rdivmod,
        ]:
            return super()._arith_method(other, op)

        step: Callable | None = None
        if op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]:
            step = op

        # TODO: if other is a RangeIndex we may have more efficient options
        right = extract_array(other, extract_numpy=True, extract_range=True)
        left = self

        try:
            # apply if we have an override
            if step:
                with np.errstate(all="ignore"):
                    rstep = step(left.step, right)

                # we don't have a representable op
                # so return a base index
                if not is_integer(rstep) or not rstep:
                    raise ValueError

            # GH#53255
            else:
                rstep = -left.step if op == ops.rsub else left.step

            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)

            res_name = ops.get_op_result_name(self, other)
            result = type(self)(rstart, rstop, rstep, name=res_name)

            # for compat with numpy / Index with int64 dtype
            # even if we can represent as a RangeIndex, return
            # as a float64 Index if we have float-like descriptors
            if not all(is_integer(x) for x in [rstart, rstop, rstep]):
                result = result.astype("float64")

            return result

        except (ValueError, TypeError, ZeroDivisionError):
            # test_arithmetic_explicit_conversions
            return super()._arith_method(other, op)

    # error: Return type "Index" of "take" incompatible with return type
    # "RangeIndex" in supertype "Index"
    def take(  # type: ignore[override]
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Index:
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)

        # raise an exception if allow_fill is True and fill_value is not None
        self._maybe_disallow_fill(allow_fill, fill_value, indices)

        if len(indices) == 0:
            taken = np.array([], dtype=self.dtype)
        else:
            ind_max = indices.max()
            if ind_max >= len(self):
                raise IndexError(
                    f"index {ind_max} is out of bounds for axis 0 with size {len(self)}"
                )
            ind_min = indices.min()
            if ind_min < -len(self):
                raise IndexError(
                    f"index {ind_min} is out of bounds for axis 0 with size {len(self)}"
                )
            taken = indices.astype(self.dtype, casting="safe")
            if ind_min < 0:
                taken %= len(self)
            if self.step != 1:
                taken *= self.step
            if self.start != 0:
                taken += self.start

        # _constructor so RangeIndex-> Index with an int64 dtype
        return self._constructor._simple_new(taken, name=self.name)
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
from functools import wraps
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    algos as libalgos,
    index as libindex,
    lib,
)
from pandas._libs.hashtable import duplicated
from pandas._typing import (
    AnyAll,
    AnyArrayLike,
    Axis,
    DropKeep,
    DtypeObj,
    F,
    IgnoreRaise,
    IndexLabel,
    Scalar,
    Self,
    Shape,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    InvalidIndexError,
    PerformanceWarning,
    UnsortedIndexError,
)
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
    array_equivalent,
    isna,
)

import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
from pandas.core.arrays.categorical import (
    factorize_from_iterables,
    recode_for_categories,
)
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    ensure_index,
    get_unanimous_names,
)
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
    get_group_index,
    lexsort_indexer,
)

from pandas.io.formats.printing import (
    get_adjustment,
    pprint_thing,
)

if TYPE_CHECKING:
    from pandas import (
        CategoricalIndex,
        DataFrame,
        Series,
    )

_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update(
    {"klass": "MultiIndex", "target_klass": "MultiIndex or list of tuples"}
)


class MultiIndexUIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    """
    This class manages a MultiIndex by mapping label combinations to positive
    integers.
    """

    _base = libindex.UInt64Engine

    def _codes_to_ints(self, codes):
        """
        Transform combination(s) of uint64 in one uint64 (each), in a strictly
        monotonic way (i.e. respecting the lexicographic order of integer
        combinations): see BaseMultiIndexCodesEngine documentation.

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint64
            Combinations of integers (one per row)

        Returns
        -------
        scalar or 1-dimensional array, of dtype uint64
            Integer(s) representing one combination (each).
        """
        # Shift the representation of each level by the pre-calculated number
        # of bits:
        codes <<= self.offsets

        # Now sum and OR are in fact interchangeable. This is a simple
        # composition of the (disjunct) significant bits of each level (i.e.
        # each column in "codes") in a single positive integer:
        if codes.ndim == 1:
            # Single key
            return np.bitwise_or.reduce(codes)

        # Multiple keys
        return np.bitwise_or.reduce(codes, axis=1)


class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    """
    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """

    _base = libindex.ObjectEngine

    def _codes_to_ints(self, codes):
        """
        Transform combination(s) of uint64 in one Python integer (each), in a
        strictly monotonic way (i.e. respecting the lexicographic order of
        integer combinations): see BaseMultiIndexCodesEngine documentation.

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint64
            Combinations of integers (one per row)

        Returns
        -------
        int, or 1-dimensional array of dtype object
            Integer(s) representing one combination (each).
        """
        # Shift the representation of each level by the pre-calculated number
        # of bits. Since this can overflow uint64, first make sure we are
        # working with Python integers:
        codes = codes.astype("object") << self.offsets

        # Now sum and OR are in fact interchangeable. This is a simple
        # composition of the (disjunct) significant bits of each level (i.e.
        # each column in "codes") in a single positive integer (per row):
        if codes.ndim == 1:
            # Single key
            return np.bitwise_or.reduce(codes)

        # Multiple keys
        return np.bitwise_or.reduce(codes, axis=1)


def names_compat(meth: F) -> F:
    """
    A decorator to allow either `name` or `names` keyword but not both.

    This makes it easier to share code with base class.
    """

    @wraps(meth)
    def new_meth(self_or_cls, *args, **kwargs):
        if "name" in kwargs and "names" in kwargs:
            raise TypeError("Can only provide one of `names` and `name`")
        if "name" in kwargs:
            kwargs["names"] = kwargs.pop("name")

        return meth(self_or_cls, *args, **kwargs)

    return cast(F, new_meth)


class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape
    dtypes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_indexer
    get_loc
    get_locs
    get_loc_level
    drop

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : The base pandas Index type.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
    for more.

    Examples
    --------
    A new ``MultiIndex`` is typically constructed using one of the helper
    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`
    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):

    >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
    >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               names=['number', 'color'])

    See further examples for how to construct a MultiIndex in the doc strings
    of the mentioned helper methods.
    """

    _hidden_attrs = Index._hidden_attrs | frozenset()

    # initialize to zero-length tuples to make everything work
    _typ = "multiindex"
    _names: list[Hashable | None] = []
    _levels = FrozenList()
    _codes = FrozenList()
    _comparables = ["names"]

    sortorder: int | None

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy: bool = False,
        name=None,
        verify_integrity: bool = True,
    ) -> Self:
        # compat with Index
        if name is not None:
            names = name
        if levels is None or codes is None:
            raise TypeError("Must pass both levels and codes")
        if len(levels) != len(codes):
            raise ValueError("Length of levels and codes must be the same.")
        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")

        result = object.__new__(cls)
        result._cache = {}

        # we've already validated levels and codes, so shortcut here
        result._set_levels(levels, copy=copy, validate=False)
        result._set_codes(codes, copy=copy, validate=False)

        result._names = [None] * len(levels)
        if names is not None:
            # handles name validation
            result._set_names(names)

        if sortorder is not None:
            result.sortorder = int(sortorder)
        else:
            result.sortorder = sortorder

        if verify_integrity:
            new_codes = result._verify_integrity()
            result._codes = new_codes

        result._reset_identity()
        result._references = None

        return result

    def _validate_codes(self, level: list, code: list):
        """
        Reassign code values as -1 if their corresponding levels are NaN.

        Parameters
        ----------
        code : list
            Code to reassign.
        level : list
            Level to check for missing values (NaN, NaT, None).

        Returns
        -------
        new code where code value = -1 if it corresponds
        to a level with missing values (NaN, NaT, None).
        """
        null_mask = isna(level)
        if np.any(null_mask):
            # error: Incompatible types in assignment
            # (expression has type "ndarray[Any, dtype[Any]]",
            # variable has type "List[Any]")
            code = np.where(null_mask[code], -1, code)  # type: ignore[assignment]
        return code

    def _verify_integrity(
        self,
        codes: list | None = None,
        levels: list | None = None,
        levels_to_verify: list[int] | range | None = None,
    ):
        """
        Parameters
        ----------
        codes : optional list
            Codes to check for validity. Defaults to current codes.
        levels : optional list
            Levels to check for validity. Defaults to current levels.
        levels_to_validate: optional list
            Specifies the levels to verify.

        Raises
        ------
        ValueError
            If length of levels and codes don't match, if the codes for any
            level would exceed level bounds, or there are any duplicate levels.

        Returns
        -------
        new codes where code value = -1 if it corresponds to a
        NaN level.
        """
        # NOTE: Currently does not check, among other things, that cached
        # nlevels matches nor that sortorder matches actually sortorder.
        codes = codes or self.codes
        levels = levels or self.levels
        if levels_to_verify is None:
            levels_to_verify = range(len(levels))

        if len(levels) != len(codes):
            raise ValueError(
                "Length of levels and codes must match. NOTE: "
                "this index is in an inconsistent state."
            )
        codes_length = len(codes[0])
        for i in levels_to_verify:
            level = levels[i]
            level_codes = codes[i]

            if len(level_codes) != codes_length:
                raise ValueError(
                    f"Unequal code lengths: {[len(code_) for code_ in codes]}"
                )
            if len(level_codes) and level_codes.max() >= len(level):
                raise ValueError(
                    f"On level {i}, code max ({level_codes.max()}) >= length of "
                    f"level ({len(level)}). NOTE: this index is in an "
                    "inconsistent state"
                )
            if len(level_codes) and level_codes.min() < -1:
                raise ValueError(f"On level {i}, code value ({level_codes.min()}) < -1")
            if not level.is_unique:
                raise ValueError(
                    f"Level values must be unique: {list(level)} on level {i}"
                )
        if self.sortorder is not None:
            if self.sortorder > _lexsort_depth(self.codes, self.nlevels):
                raise ValueError(
                    "Value for sortorder must be inferior or equal to actual "
                    f"lexsort_depth: sortorder {self.sortorder} "
                    f"with lexsort_depth {_lexsort_depth(self.codes, self.nlevels)}"
                )

        result_codes = []
        for i in range(len(levels)):
            if i in levels_to_verify:
                result_codes.append(self._validate_codes(levels[i], codes[i]))
            else:
                result_codes.append(codes[i])

        new_codes = FrozenList(result_codes)
        return new_codes

    @classmethod
    def from_arrays(
        cls,
        arrays,
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | lib.NoDefault = lib.no_default,
    ) -> MultiIndex:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        error_msg = "Input must be a list / sequence of array-likes."
        if not is_list_like(arrays):
            raise TypeError(error_msg)
        if is_iterator(arrays):
            arrays = list(arrays)

        # Check if elements of array are list-like
        for array in arrays:
            if not is_list_like(array):
                raise TypeError(error_msg)

        # Check if lengths of all arrays are equal or not,
        # raise ValueError, if not
        for i in range(1, len(arrays)):
            if len(arrays[i]) != len(arrays[i - 1]):
                raise ValueError("all arrays must be same length")

        codes, levels = factorize_from_iterables(arrays)
        if names is lib.no_default:
            names = [getattr(arr, "name", None) for arr in arrays]

        return cls(
            levels=levels,
            codes=codes,
            sortorder=sortorder,
            names=names,
            verify_integrity=False,
        )

    @classmethod
    @names_compat
    def from_tuples(
        cls,
        tuples: Iterable[tuple[Hashable, ...]],
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | None = None,
    ) -> MultiIndex:
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        if not is_list_like(tuples):
            raise TypeError("Input must be a list / sequence of tuple-likes.")
        if is_iterator(tuples):
            tuples = list(tuples)
        tuples = cast(Collection[tuple[Hashable, ...]], tuples)

        # handling the empty tuple cases
        if len(tuples) and all(isinstance(e, tuple) and not e for e in tuples):
            codes = [np.zeros(len(tuples))]
            levels = [Index(com.asarray_tuplesafe(tuples, dtype=np.dtype("object")))]
            return cls(
                levels=levels,
                codes=codes,
                sortorder=sortorder,
                names=names,
                verify_integrity=False,
            )

        arrays: list[Sequence[Hashable]]
        if len(tuples) == 0:
            if names is None:
                raise TypeError("Cannot infer number of levels from empty list")
            # error: Argument 1 to "len" has incompatible type "Hashable";
            # expected "Sized"
            arrays = [[]] * len(names)  # type: ignore[arg-type]
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = np.asarray(tuples._values)

            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            arrs = zip(*tuples)
            arrays = cast(list[Sequence[Hashable]], arrs)

        return cls.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(
        cls,
        iterables: Sequence[Iterable[Hashable]],
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | lib.NoDefault = lib.no_default,
    ) -> MultiIndex:
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> pd.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        from pandas.core.reshape.util import cartesian_product

        if not is_list_like(iterables):
            raise TypeError("Input must be a list / sequence of iterables.")
        if is_iterator(iterables):
            iterables = list(iterables)

        codes, levels = factorize_from_iterables(iterables)
        if names is lib.no_default:
            names = [getattr(it, "name", None) for it in iterables]

        # codes are all ndarrays, so cartesian_product is lossless
        codes = cartesian_product(codes)
        return cls(levels, codes, sortorder=sortorder, names=names)

    @classmethod
    def from_frame(
        cls,
        df: DataFrame,
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | None = None,
    ) -> MultiIndex:
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = pd.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> pd.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        if not isinstance(df, ABCDataFrame):
            raise TypeError("Input must be a DataFrame")

        column_names, columns = zip(*df.items())
        names = column_names if names is None else names
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    # --------------------------------------------------------------------

    @cache_readonly
    def _values(self) -> np.ndarray:
        # We override here, since our parent uses _data, which we don't use.
        values = []

        for i in range(self.nlevels):
            index = self.levels[i]
            codes = self.codes[i]

            vals = index
            if isinstance(vals.dtype, CategoricalDtype):
                vals = cast("CategoricalIndex", vals)
                vals = vals._data._internal_get_values()

            if isinstance(vals.dtype, ExtensionDtype) or lib.is_np_dtype(
                vals.dtype, "mM"
            ):
                vals = vals.astype(object)

            vals = np.asarray(vals)
            vals = algos.take_nd(vals, codes, fill_value=index._na_value)
            values.append(vals)

        arr = lib.fast_zip(values)
        return arr

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def array(self):
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
        raise ValueError(
            "MultiIndex has no single backing array. Use "
            "'MultiIndex.to_numpy()' to get a NumPy array of tuples."
        )

    @cache_readonly
    def dtypes(self) -> Series:
        """
        Return the dtypes as a Series for the underlying MultiIndex.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=['number', 'color'])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.dtypes
        number     int64
        color     object
        dtype: object
        """
        from pandas import Series

        names = com.fill_missing_names([level.name for level in self.levels])
        return Series([level.dtype for level in self.levels], index=Index(names))

    def __len__(self) -> int:
        return len(self.codes[0])

    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.
        """
        # override Index.size to avoid materializing _values
        return len(self)

    # --------------------------------------------------------------------
    # Levels Methods

    @cache_readonly
    def levels(self) -> FrozenList:
        """
        Levels of the MultiIndex.

        Levels refer to the different hierarchical levels or layers in a MultiIndex.
        In a MultiIndex, each level represents a distinct dimension or category of
        the index.

        To access the levels, you can use the levels attribute of the MultiIndex,
        which returns a tuple of Index objects. Each Index object represents a
        level in the MultiIndex and contains the unique values found in that
        specific level.

        If a MultiIndex is created with levels A, B, C, and the DataFrame using
        it filters out all rows of the level C, MultiIndex.levels will still
        return A, B, C.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product([['mammal'],
        ...                                     ('goat', 'human', 'cat', 'dog')],
        ...                                    names=['Category', 'Animals'])
        >>> leg_num = pd.DataFrame(data=(4, 2, 4, 4), index=index, columns=['Legs'])
        >>> leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 human       2
                 cat         4
                 dog         4

        >>> leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])

        MultiIndex levels will not change even if the DataFrame using the MultiIndex
        does not contain all them anymore.
        See how "human" is not in the DataFrame, but it is still in levels:

        >>> large_leg_num = leg_num[leg_num.Legs > 2]
        >>> large_leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 cat         4
                 dog         4

        >>> large_leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])
        """
        # Use cache_readonly to ensure that self.get_locs doesn't repeatedly
        # create new IndexEngine
        # https://github.com/pandas-dev/pandas/issues/31648
        result = [x._rename(name=name) for x, name in zip(self._levels, self._names)]
        for level in result:
            # disallow midx.levels[0].name = "foo"
            level._no_setting_name = True
        return FrozenList(result)

    def _set_levels(
        self,
        levels,
        *,
        level=None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False,
    ) -> None:
        # This is NOT part of the levels property because it should be
        # externally not allowed to set levels. User beware if you change
        # _levels directly
        if validate:
            if len(levels) == 0:
                raise ValueError("Must set non-zero number of levels.")
            if level is None and len(levels) != self.nlevels:
                raise ValueError("Length of levels must match number of levels.")
            if level is not None and len(levels) != len(level):
                raise ValueError("Length of levels must match length of level.")

        if level is None:
            new_levels = FrozenList(
                ensure_index(lev, copy=copy)._view() for lev in levels
            )
            level_numbers = list(range(len(new_levels)))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_levels_list = list(self._levels)
            for lev_num, lev in zip(level_numbers, levels):
                new_levels_list[lev_num] = ensure_index(lev, copy=copy)._view()
            new_levels = FrozenList(new_levels_list)

        if verify_integrity:
            new_codes = self._verify_integrity(
                levels=new_levels, levels_to_verify=level_numbers
            )
            self._codes = new_codes

        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)

        self._reset_cache()

    def set_levels(
        self, levels, *, level=None, verify_integrity: bool = True
    ) -> MultiIndex:
        """
        Set new levels on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        levels : sequence or list of sequence
            New level(s) to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, "one"),
        ...         (1, "two"),
        ...         (2, "one"),
        ...         (2, "two"),
        ...         (3, "one"),
        ...         (3, "two")
        ...     ],
        ...     names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two'),
            (3, 'one'),
            (3, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_levels([['a', 'b', 'c'], [1, 2]])
        MultiIndex([('a', 1),
                    ('a', 2),
                    ('b', 1),
                    ('b', 2),
                    ('c', 1),
                    ('c', 2)],
                   names=['foo', 'bar'])
        >>> idx.set_levels(['a', 'b', 'c'], level=0)
        MultiIndex([('a', 'one'),
                    ('a', 'two'),
                    ('b', 'one'),
                    ('b', 'two'),
                    ('c', 'one'),
                    ('c', 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_levels(['a', 'b'], level='bar')
        MultiIndex([(1, 'a'),
                    (1, 'b'),
                    (2, 'a'),
                    (2, 'b'),
                    (3, 'a'),
                    (3, 'b')],
                   names=['foo', 'bar'])

        If any of the levels passed to ``set_levels()`` exceeds the
        existing length, all of the values from that argument will
        be stored in the MultiIndex levels, though the values will
        be truncated in the MultiIndex output.

        >>> idx.set_levels([['a', 'b', 'c'], [1, 2, 3, 4]], level=[0, 1])
        MultiIndex([('a', 1),
            ('a', 2),
            ('b', 1),
            ('b', 2),
            ('c', 1),
            ('c', 2)],
           names=['foo', 'bar'])
        >>> idx.set_levels([['a', 'b', 'c'], [1, 2, 3, 4]], level=[0, 1]).levels
        FrozenList([['a', 'b', 'c'], [1, 2, 3, 4]])
        """

        if isinstance(levels, Index):
            pass
        elif is_array_like(levels):
            levels = Index(levels)
        elif is_list_like(levels):
            levels = list(levels)

        level, levels = _require_listlike(level, levels, "Levels")
        idx = self._view()
        idx._reset_identity()
        idx._set_levels(
            levels, level=level, validate=True, verify_integrity=verify_integrity
        )
        return idx

    @property
    def nlevels(self) -> int:
        """
        Integer number of levels in this MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.nlevels
        3
        """
        return len(self._levels)

    @property
    def levshape(self) -> Shape:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.levshape
        (1, 1, 1)
        """
        return tuple(len(x) for x in self.levels)

    # --------------------------------------------------------------------
    # Codes Methods

    @property
    def codes(self) -> FrozenList:
        return self._codes

    def _set_codes(
        self,
        codes,
        *,
        level=None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False,
    ) -> None:
        if validate:
            if level is None and len(codes) != self.nlevels:
                raise ValueError("Length of codes must match number of levels")
            if level is not None and len(codes) != len(level):
                raise ValueError("Length of codes must match length of levels.")

        level_numbers: list[int] | range
        if level is None:
            new_codes = FrozenList(
                _coerce_indexer_frozen(level_codes, lev, copy=copy).view()
                for lev, level_codes in zip(self._levels, codes)
            )
            level_numbers = range(len(new_codes))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_codes_list = list(self._codes)
            for lev_num, level_codes in zip(level_numbers, codes):
                lev = self.levels[lev_num]
                new_codes_list[lev_num] = _coerce_indexer_frozen(
                    level_codes, lev, copy=copy
                )
            new_codes = FrozenList(new_codes_list)

        if verify_integrity:
            new_codes = self._verify_integrity(
                codes=new_codes, levels_to_verify=level_numbers
            )

        self._codes = new_codes

        self._reset_cache()

    def set_codes(
        self, codes, *, level=None, verify_integrity: bool = True
    ) -> MultiIndex:
        """
        Set new codes on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        codes : sequence or list of sequence
            New codes to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([1, 0, 1, 0], level=0)
        MultiIndex([(2, 'one'),
                    (1, 'two'),
                    (2, 'one'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([0, 0, 1, 1], level='bar')
        MultiIndex([(1, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (2, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        """

        level, codes = _require_listlike(level, codes, "Codes")
        idx = self._view()
        idx._reset_identity()
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        return idx

    # --------------------------------------------------------------------
    # Index Internals

    @cache_readonly
    def _engine(self):
        # Calculate the number of bits needed to represent labels in each
        # level, as log2 of their sizes:
        # NaN values are shifted to 1 and missing values in other while
        # calculating the indexer are shifted to 0
        sizes = np.ceil(
            np.log2(
                [len(level) + libindex.multiindex_nulls_shift for level in self.levels]
            )
        )

        # Sum bit counts, starting from the _right_....
        lev_bits = np.cumsum(sizes[::-1])[::-1]

        # ... in order to obtain offsets such that sorting the combination of
        # shifted codes (one for each level, resulting in a unique integer) is
        # equivalent to sorting lexicographically the codes themselves. Notice
        # that each level needs to be shifted by the number of bits needed to
        # represent the _previous_ ones:
        offsets = np.concatenate([lev_bits[1:], [0]]).astype("uint64")

        # Check the total number of bits needed for our representation:
        if lev_bits[0] > 64:
            # The levels would overflow a 64 bit uint - use Python integers:
            return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
        return MultiIndexUIntEngine(self.levels, self.codes, offsets)

    # Return type "Callable[..., MultiIndex]" of "_constructor" incompatible with return
    # type "Type[MultiIndex]" in supertype "Index"
    @property
    def _constructor(self) -> Callable[..., MultiIndex]:  # type: ignore[override]
        return type(self).from_tuples

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values: np.ndarray, name=lib.no_default) -> MultiIndex:
        names = name if name is not lib.no_default else self.names

        return type(self).from_tuples(values, sortorder=None, names=names)

    def _view(self) -> MultiIndex:
        result = type(self)(
            levels=self.levels,
            codes=self.codes,
            sortorder=self.sortorder,
            names=self.names,
            verify_integrity=False,
        )
        result._cache = self._cache.copy()
        result._cache.pop("levels", None)  # GH32669
        return result

    # --------------------------------------------------------------------

    # error: Signature of "copy" incompatible with supertype "Index"
    def copy(  # type: ignore[override]
        self,
        names=None,
        deep: bool = False,
        name=None,
    ) -> Self:
        """
        Make a copy of this object.

        Names, dtype, levels and codes can be passed and will be set on new copy.

        Parameters
        ----------
        names : sequence, optional
        deep : bool, default False
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.copy()
        MultiIndex([('a', 'b', 'c')],
                   )
        """
        names = self._validate_names(name=name, names=names, deep=deep)
        keep_id = not deep
        levels, codes = None, None

        if deep:
            from copy import deepcopy

            levels = deepcopy(self.levels)
            codes = deepcopy(self.codes)

        levels = levels if levels is not None else self.levels
        codes = codes if codes is not None else self.codes

        new_index = type(self)(
            levels=levels,
            codes=codes,
            sortorder=self.sortorder,
            names=names,
            verify_integrity=False,
        )
        new_index._cache = self._cache.copy()
        new_index._cache.pop("levels", None)  # GH32669
        if keep_id:
            new_index._id = self._id
        return new_index

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """the array interface, return my values"""
        if copy is False:
            # self.values is always a newly construct array, so raise.
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if copy is True:
            # explicit np.array call to ensure a copy is made and unique objects
            # are returned, because self.values is cached
            return np.array(self.values, dtype=dtype)
        return self.values

    def view(self, cls=None) -> Self:
        """this is defined as a copy with the same identity"""
        result = self.copy()
        result._id = self._id
        return result

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @cache_readonly
    def dtype(self) -> np.dtype:
        return np.dtype("O")

    def _is_memory_usage_qualified(self) -> bool:
        """return a boolean if we need a qualified .info display"""

        def f(dtype) -> bool:
            return is_object_dtype(dtype) or (
                is_string_dtype(dtype) and dtype.storage == "python"
            )

        return any(f(level.dtype) for level in self.levels)

    # Cannot determine type of "memory_usage"
    @doc(Index.memory_usage)  # type: ignore[has-type]
    def memory_usage(self, deep: bool = False) -> int:
        # we are overwriting our base class to avoid
        # computing .values here which could materialize
        # a tuple representation unnecessarily
        return self._nbytes(deep)

    @cache_readonly
    def nbytes(self) -> int:
        """return the number of bytes in the underlying data"""
        return self._nbytes(False)

    def _nbytes(self, deep: bool = False) -> int:
        """
        return the number of bytes in the underlying data
        deeply introspect the level data if deep=True

        include the engine hashtable

        *this is in internal routine*

        """
        # for implementations with no useful getsizeof (PyPy)
        objsize = 24

        level_nbytes = sum(i.memory_usage(deep=deep) for i in self.levels)
        label_nbytes = sum(i.nbytes for i in self.codes)
        names_nbytes = sum(getsizeof(i, objsize) for i in self.names)
        result = level_nbytes + label_nbytes + names_nbytes

        # include our engine hashtable
        result += self._engine.sizeof(deep=deep)
        return result

    # --------------------------------------------------------------------
    # Rendering Methods

    def _formatter_func(self, tup):
        """
        Formats each item in tup according to its level's formatter function.
        """
        formatter_funcs = [level._formatter_func for level in self.levels]
        return tuple(func(val) for func, val in zip(formatter_funcs, tup))

    def _get_values_for_csv(
        self, *, na_rep: str = "nan", **kwargs
    ) -> npt.NDArray[np.object_]:
        new_levels = []
        new_codes = []

        # go through the levels and format them
        for level, level_codes in zip(self.levels, self.codes):
            level_strs = level._get_values_for_csv(na_rep=na_rep, **kwargs)
            # add nan values, if there are any
            mask = level_codes == -1
            if mask.any():
                nan_index = len(level_strs)
                # numpy 1.21 deprecated implicit string casting
                level_strs = level_strs.astype(str)
                level_strs = np.append(level_strs, na_rep)
                assert not level_codes.flags.writeable  # i.e. copy is needed
                level_codes = level_codes.copy()  # make writeable
                level_codes[mask] = nan_index
            new_levels.append(level_strs)
            new_codes.append(level_codes)

        if len(new_levels) == 1:
            # a single-level multi-index
            return Index(new_levels[0].take(new_codes[0]))._get_values_for_csv()
        else:
            # reconstruct the multi-index
            mi = MultiIndex(
                levels=new_levels,
                codes=new_codes,
                names=self.names,
                sortorder=self.sortorder,
                verify_integrity=False,
            )
            return mi._values

    def format(
        self,
        name: bool | None = None,
        formatter: Callable | None = None,
        na_rep: str | None = None,
        names: bool = False,
        space: int = 2,
        sparsify=None,
        adjoin: bool = True,
    ) -> list:
        warnings.warn(
            # GH#55413
            f"{type(self).__name__}.format is deprecated and will be removed "
            "in a future version. Convert using index.astype(str) or "
            "index.map(formatter) instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

        if name is not None:
            names = name

        if len(self) == 0:
            return []

        stringified_levels = []
        for lev, level_codes in zip(self.levels, self.codes):
            na = na_rep if na_rep is not None else _get_na_rep(lev.dtype)

            if len(lev) > 0:
                formatted = lev.take(level_codes).format(formatter=formatter)

                # we have some NA
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()

            else:
                # weird all NA case
                formatted = [
                    pprint_thing(na if isna(x) else x, escape_chars=("\t", "\r", "\n"))
                    for x in algos.take_nd(lev._values, level_codes)
                ]
            stringified_levels.append(formatted)

        result_levels = []
        for lev, lev_name in zip(stringified_levels, self.names):
            level = []

            if names:
                level.append(
                    pprint_thing(lev_name, escape_chars=("\t", "\r", "\n"))
                    if lev_name is not None
                    else ""
                )

            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)

        if sparsify is None:
            sparsify = get_option("display.multi_sparse")

        if sparsify:
            sentinel: Literal[""] | bool | lib.NoDefault = ""
            # GH3547 use value of sparsify as sentinel if it's "Falsey"
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify in [False, lib.no_default]:
                sentinel = sparsify
            # little bit of a kludge job for #1217
            result_levels = sparsify_labels(
                result_levels, start=int(names), sentinel=sentinel
            )

        if adjoin:
            adj = get_adjustment()
            return adj.adjoin(space, *result_levels).split("\n")
        else:
            return result_levels

    def _format_multi(
        self,
        *,
        include_names: bool,
        sparsify: bool | None | lib.NoDefault,
        formatter: Callable | None = None,
    ) -> list:
        if len(self) == 0:
            return []

        stringified_levels = []
        for lev, level_codes in zip(self.levels, self.codes):
            na = _get_na_rep(lev.dtype)

            if len(lev) > 0:
                taken = formatted = lev.take(level_codes)
                formatted = taken._format_flat(include_name=False, formatter=formatter)

                # we have some NA
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()

            else:
                # weird all NA case
                formatted = [
                    pprint_thing(na if isna(x) else x, escape_chars=("\t", "\r", "\n"))
                    for x in algos.take_nd(lev._values, level_codes)
                ]
            stringified_levels.append(formatted)

        result_levels = []
        for lev, lev_name in zip(stringified_levels, self.names):
            level = []

            if include_names:
                level.append(
                    pprint_thing(lev_name, escape_chars=("\t", "\r", "\n"))
                    if lev_name is not None
                    else ""
                )

            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)

        if sparsify is None:
            sparsify = get_option("display.multi_sparse")

        if sparsify:
            sentinel: Literal[""] | bool | lib.NoDefault = ""
            # GH3547 use value of sparsify as sentinel if it's "Falsey"
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify is lib.no_default:
                sentinel = sparsify
            # little bit of a kludge job for #1217
            result_levels = sparsify_labels(
                result_levels, start=int(include_names), sentinel=sentinel
            )

        return result_levels

    # --------------------------------------------------------------------
    # Names Methods

    def _get_names(self) -> FrozenList:
        return FrozenList(self._names)

    def _set_names(self, names, *, level=None, validate: bool = True):
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None
        validate : bool, default True
            validate that the names match level lengths

        Raises
        ------
        TypeError if each name is not hashable.

        Notes
        -----
        sets names on levels. WARNING: mutates!

        Note that you generally want to set this *after* changing levels, so
        that it only acts on copies
        """
        # GH 15110
        # Don't allow a single string for names in a MultiIndex
        if names is not None and not is_list_like(names):
            raise ValueError("Names should be list-like for a MultiIndex")
        names = list(names)

        if validate:
            if level is not None and len(names) != len(level):
                raise ValueError("Length of names must match length of level.")
            if level is None and len(names) != self.nlevels:
                raise ValueError(
                    "Length of names must match number of levels in MultiIndex."
                )

        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._get_level_number(lev) for lev in level]

        # set the name
        for lev, name in zip(level, names):
            if name is not None:
                # GH 20527
                # All items in 'names' need to be hashable:
                if not is_hashable(name):
                    raise TypeError(
                        f"{type(self).__name__}.name must be a hashable type"
                    )
            self._names[lev] = name

        # If .levels has been accessed, the names in our cache will be stale.
        self._reset_cache()

    names = property(
        fset=_set_names,
        fget=_get_names,
        doc="""
        Names of levels in MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])
        >>> mi.names
        FrozenList(['x', 'y', 'z'])
        """,
    )

    # --------------------------------------------------------------------

    @cache_readonly
    def inferred_type(self) -> str:
        return "mixed"

    def _get_level_number(self, level) -> int:
        count = self.names.count(level)
        if (count > 1) and not is_integer(level):
            raise ValueError(
                f"The name {level} occurs multiple times, use a level number"
            )
        try:
            level = self.names.index(level)
        except ValueError as err:
            if not is_integer(level):
                raise KeyError(f"Level {level} not found") from err
            if level < 0:
                level += self.nlevels
                if level < 0:
                    orig_level = level - self.nlevels
                    raise IndexError(
                        f"Too many levels: Index has only {self.nlevels} levels, "
                        f"{orig_level} is not a valid level number"
                    ) from err
            # Note: levels are zero-based
            elif level >= self.nlevels:
                raise IndexError(
                    f"Too many levels: Index has only {self.nlevels} levels, "
                    f"not {level + 1}"
                ) from err
        return level

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        """
        Return a boolean if the values are equal or increasing.
        """
        if any(-1 in code for code in self.codes):
            return False

        if all(level.is_monotonic_increasing for level in self.levels):
            # If each level is sorted, we can operate on the codes directly. GH27495
            return libalgos.is_lexsorted(
                [x.astype("int64", copy=False) for x in self.codes]
            )

        # reversed() because lexsort() wants the most significant key last.
        values = [
            self._get_level_values(i)._values for i in reversed(range(len(self.levels)))
        ]
        try:
            # error: Argument 1 to "lexsort" has incompatible type
            # "List[Union[ExtensionArray, ndarray[Any, Any]]]";
            # expected "Union[_SupportsArray[dtype[Any]],
            # _NestedSequence[_SupportsArray[dtype[Any]]], bool,
            # int, float, complex, str, bytes, _NestedSequence[Union
            # [bool, int, float, complex, str, bytes]]]"
            sort_order = np.lexsort(values)  # type: ignore[arg-type]
            return Index(sort_order).is_monotonic_increasing
        except TypeError:
            # we have mixed types and np.lexsort is not happy
            return Index(self._values).is_monotonic_increasing

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.
        """
        # monotonic decreasing if and only if reverse is monotonic increasing
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def _inferred_type_levels(self) -> list[str]:
        """return a list of the inferred types, one for each level"""
        return [i.inferred_type for i in self.levels]

    @doc(Index.duplicated)
    def duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        shape = tuple(len(lev) for lev in self.levels)
        ids = get_group_index(self.codes, shape, sort=False, xnull=False)

        return duplicated(ids, keep)

    # error: Cannot override final attribute "_duplicated"
    # (previously declared in base class "IndexOpsMixin")
    _duplicated = duplicated  # type: ignore[misc]

    def fillna(self, value=None, downcast=None):
        """
        fillna is not implemented for MultiIndex
        """
        raise NotImplementedError("isna is not defined for MultiIndex")

    @doc(Index.dropna)
    def dropna(self, how: AnyAll = "any") -> MultiIndex:
        nans = [level_codes == -1 for level_codes in self.codes]
        if how == "any":
            indexer = np.any(nans, axis=0)
        elif how == "all":
            indexer = np.all(nans, axis=0)
        else:
            raise ValueError(f"invalid how option: {how}")

        new_codes = [level_codes[~indexer] for level_codes in self.codes]
        return self.set_codes(codes=new_codes)

    def _get_level_values(self, level: int, unique: bool = False) -> Index:
        """
        Return vector of label values for requested level,
        equal to the length of the index

        **this is an internal method**

        Parameters
        ----------
        level : int
        unique : bool, default False
            if True, drop duplicated values

        Returns
        -------
        Index
        """
        lev = self.levels[level]
        level_codes = self.codes[level]
        name = self._names[level]
        if unique:
            level_codes = algos.unique(level_codes)
        filled = algos.take_nd(lev._values, level_codes, fill_value=lev._na_value)
        return lev._shallow_copy(filled, name=name)

    # error: Signature of "get_level_values" incompatible with supertype "Index"
    def get_level_values(self, level) -> Index:  # type: ignore[override]
        """
        Return vector of label values for requested level.

        Length of returned vector is equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Notes
        -----
        If the level contains missing values, the result may be casted to
        ``float`` with missing values specified as ``NaN``. This is because
        the level is converted to a regular ``Index``.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))
        >>> mi.names = ['level_1', 'level_2']

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.get_level_values('level_2')
        Index(['d', 'e', 'f'], dtype='object', name='level_2')

        If a level contains missing values, the return type of the level
        may be cast to ``float``.

        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).dtypes
        level_0    int64
        level_1    int64
        dtype: object
        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).get_level_values(0)
        Index([1.0, nan, 2.0], dtype='float64')
        """
        level = self._get_level_number(level)
        values = self._get_level_values(level)
        return values

    @doc(Index.unique)
    def unique(self, level=None):
        if level is None:
            return self.drop_duplicates()
        else:
            level = self._get_level_number(level)
            return self._get_level_values(level=level, unique=True)

    def to_frame(
        self,
        index: bool = True,
        name=lib.no_default,
        allow_duplicates: bool = False,
    ) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b'], ['c', 'd']])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=['x', 'y'])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        from pandas import DataFrame

        if name is not lib.no_default:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence of column names.")

            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have same length as number of levels on index."
                )
            idx_names = name
