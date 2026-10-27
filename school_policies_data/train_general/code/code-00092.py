        raise IndexError(
            # GH#34193
            "Indexing with a float is no longer supported. Manually convert "
            "to an integer key instead."
        )
    return val


def not_none(*args):
    """
    Returns a generator consisting of the arguments that are not None.
    """
    return (arg for arg in args if arg is not None)


def any_none(*args) -> bool:
    """
    Returns a boolean indicating if any argument is None.
    """
    return any(arg is None for arg in args)


def all_none(*args) -> bool:
    """
    Returns a boolean indicating if all arguments are None.
    """
    return all(arg is None for arg in args)


def any_not_none(*args) -> bool:
    """
    Returns a boolean indicating if any argument is not None.
    """
    return any(arg is not None for arg in args)


def all_not_none(*args) -> bool:
    """
    Returns a boolean indicating if all arguments are not None.
    """
    return all(arg is not None for arg in args)


def count_not_none(*args) -> int:
    """
    Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


@overload
def asarray_tuplesafe(
    values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...
) -> np.ndarray:
    # ExtensionArray can only be returned when values is an Index, all other iterables
    # will return np.ndarray. Unfortunately "all other" cannot be encoded in a type
    # signature, so instead we special-case some common types.
    ...


@overload
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike:
    ...


def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = None) -> ArrayLike:
    if not (isinstance(values, (list, tuple)) or hasattr(values, "__array__")):
        values = list(values)
    elif isinstance(values, ABCIndex):
        return values._values
    elif isinstance(values, ABCSeries):
        return values._values

    if isinstance(values, list) and dtype in [np.object_, object]:
        return construct_1d_object_array_from_listlike(values)

    try:
        with warnings.catch_warnings():
            # Can remove warning filter once NumPy 1.24 is min version
            if not np_version_gte1p24:
                warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
            result = np.asarray(values, dtype=dtype)
    except ValueError:
        # Using try/except since it's more performant than checking is_list_like
        # over each element
        # error: Argument 1 to "construct_1d_object_array_from_listlike"
        # has incompatible type "Iterable[Any]"; expected "Sized"
        return construct_1d_object_array_from_listlike(values)  # type: ignore[arg-type]

    if issubclass(result.dtype.type, str):
        result = np.asarray(values, dtype=object)

    if result.ndim == 2:
        # Avoid building an array of arrays:
        values = [tuple(x) for x in values]
        result = construct_1d_object_array_from_listlike(values)

    return result


def index_labels_to_array(
    labels: np.ndarray | Iterable, dtype: NpDtype | None = None
) -> np.ndarray:
    """
    Transform label or iterable of labels to array, for use in Index.

    Parameters
    ----------
    dtype : dtype
        If specified, use as dtype of the resulting array, otherwise infer.

    Returns
    -------
    array
    """
    if isinstance(labels, (str, tuple)):
        labels = [labels]

    if not isinstance(labels, (list, np.ndarray)):
        try:
            labels = list(labels)
        except TypeError:  # non-iterable
            labels = [labels]

    labels = asarray_tuplesafe(labels, dtype=dtype)

    return labels


def maybe_make_list(obj):
    if obj is not None and not isinstance(obj, (tuple, list)):
        return [obj]
    return obj


def maybe_iterable_to_list(obj: Iterable[T] | T) -> Collection[T] | T:
    """
    If obj is Iterable but not list-like, consume into list.
    """
    if isinstance(obj, abc.Iterable) and not isinstance(obj, abc.Sized):
        return list(obj)
    obj = cast(Collection, obj)
    return obj


def is_null_slice(obj) -> bool:
    """
    We have a null slice.
    """
    return (
        isinstance(obj, slice)
        and obj.start is None
        and obj.stop is None
        and obj.step is None
    )


def is_empty_slice(obj) -> bool:
    """
    We have an empty slice, e.g. no values are selected.
    """
    return (
        isinstance(obj, slice)
        and obj.start is not None
        and obj.stop is not None
        and obj.start == obj.stop
    )


def is_true_slices(line) -> list[bool]:
    """
    Find non-trivial slices in "line": return a list of booleans with same length.
    """
    return [isinstance(k, slice) and not is_null_slice(k) for k in line]


# TODO: used only once in indexing; belongs elsewhere?
def is_full_slice(obj, line: int) -> bool:
    """
    We have a full length slice.
    """
    return (
        isinstance(obj, slice)
        and obj.start == 0
        and obj.stop == line
        and obj.step is None
    )


def get_callable_name(obj):
    # typical case has name
    if hasattr(obj, "__name__"):
        return getattr(obj, "__name__")
    # some objects don't; could recurse
    if isinstance(obj, partial):
        return get_callable_name(obj.func)
    # fall back to class name
    if callable(obj):
        return type(obj).__name__
    # everything failed (probably because the argument
    # wasn't actually callable); we return None
    # instead of the empty string in this case to allow
    # distinguishing between no name and a name of ''
    return None


def apply_if_callable(maybe_callable, obj, **kwargs):
    """
    Evaluate possibly callable input using obj and kwargs if it is callable,
    otherwise return as it is.

    Parameters
    ----------
    maybe_callable : possibly a callable
    obj : NDFrame
    **kwargs
    """
    if callable(maybe_callable):
        return maybe_callable(obj, **kwargs)

    return maybe_callable


def standardize_mapping(into):
    """
    Helper function to standardize a supplied mapping.

    Parameters
    ----------
    into : instance or subclass of collections.abc.Mapping
        Must be a class, an initialized collections.defaultdict,
        or an instance of a collections.abc.Mapping subclass.

    Returns
    -------
    mapping : a collections.abc.Mapping subclass or other constructor
        a callable object that can accept an iterator to create
        the desired Mapping.

    See Also
    --------
    DataFrame.to_dict
    Series.to_dict
    """
    if not inspect.isclass(into):
        if isinstance(into, defaultdict):
            return partial(defaultdict, into.default_factory)
        into = type(into)
    if not issubclass(into, abc.Mapping):
        raise TypeError(f"unsupported type: {into}")
    if into == defaultdict:
        raise TypeError("to_dict() only accepts initialized defaultdicts")
    return into


@overload
def random_state(state: np.random.Generator) -> np.random.Generator:
    ...


@overload
def random_state(
    state: int | np.ndarray | np.random.BitGenerator | np.random.RandomState | None,
) -> np.random.RandomState:
    ...


def random_state(state: RandomState | None = None):
    """
    Helper function for processing random_state arguments.

    Parameters
    ----------
    state : int, array-like, BitGenerator, Generator, np.random.RandomState, None.
        If receives an int, array-like, or BitGenerator, passes to
        np.random.RandomState() as seed.
        If receives an np.random RandomState or Generator, just returns that unchanged.
        If receives `None`, returns np.random.
        If receives anything else, raises an informative ValueError.

        Default None.

    Returns
    -------
    np.random.RandomState or np.random.Generator. If state is None, returns np.random

    """
    if is_integer(state) or isinstance(state, (np.ndarray, np.random.BitGenerator)):
        return np.random.RandomState(state)
    elif isinstance(state, np.random.RandomState):
        return state
    elif isinstance(state, np.random.Generator):
        return state
    elif state is None:
        return np.random
    else:
        raise ValueError(
            "random_state must be an integer, array-like, a BitGenerator, Generator, "
            "a numpy RandomState, or None"
        )


def pipe(
    obj, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
) -> T:
    """
    Apply a function ``func`` to object ``obj`` either by passing obj as the
    first argument to the function or, in the case that the func is a tuple,
    interpret the first element of the tuple as a function and pass the obj to
    that function as a keyword argument whose key is the value of the second
    element of the tuple.

    Parameters
    ----------
    func : callable or tuple of (callable, str)
        Function to apply to this object or, alternatively, a
        ``(callable, data_keyword)`` tuple where ``data_keyword`` is a
        string indicating the keyword of ``callable`` that expects the
        object.
    *args : iterable, optional
        Positional arguments passed into ``func``.
    **kwargs : dict, optional
        A dictionary of keyword arguments passed into ``func``.

    Returns
    -------
    object : the return type of ``func``.
    """
    if isinstance(func, tuple):
        func, target = func
        if target in kwargs:
            msg = f"{target} is both the pipe target and a keyword argument"
            raise ValueError(msg)
        kwargs[target] = obj
        return func(*args, **kwargs)
    else:
        return func(obj, *args, **kwargs)


def get_rename_function(mapper):
    """
    Returns a function that will map names/labels, dependent if mapper
    is a dict, Series or just a function.
    """

    def f(x):
        if x in mapper:
            return mapper[x]
        else:
            return x

    return f if isinstance(mapper, (abc.Mapping, ABCSeries)) else mapper


def convert_to_list_like(
    values: Hashable | Iterable | AnyArrayLike,
) -> list | AnyArrayLike:
    """
    Convert list-like or scalar input to list-like. List, numpy and pandas array-like
    inputs are returned unmodified whereas others are converted to list.
    """
    if isinstance(values, (list, np.ndarray, ABCIndex, ABCSeries, ABCExtensionArray)):
        return values
    elif isinstance(values, abc.Iterable) and not isinstance(values, str):
        return list(values)

    return [values]


@contextlib.contextmanager
def temp_setattr(
    obj, attr: str, value, condition: bool = True
) -> Generator[None, None, None]:
    """
    Temporarily set attribute on an object.

    Parameters
    ----------
    obj : object
        Object whose attribute will be modified.
    attr : str
        Attribute to modify.
    value : Any
        Value to temporarily set attribute to.
    condition : bool, default True
        Whether to set the attribute. Provided in order to not have to
        conditionally use this context manager.

    Yields
    ------
    object : obj with modified attribute.
    """
    if condition:
        old_value = getattr(obj, attr)
        setattr(obj, attr, value)
    try:
        yield obj
    finally:
        if condition:
            setattr(obj, attr, old_value)


def require_length_match(data, index: Index) -> None:
    """
    Check the length of data matches the length of the index.
    """
    if len(data) != len(index):
        raise ValueError(
            "Length of values "
            f"({len(data)}) "
            "does not match length of index "
            f"({len(index)})"
        )


# the ufuncs np.maximum.reduce and np.minimum.reduce default to axis=0,
#  whereas np.min and np.max (which directly call obj.min and obj.max)
#  default to axis=None.
_builtin_table = {
    builtins.sum: np.sum,
    builtins.max: np.maximum.reduce,
    builtins.min: np.minimum.reduce,
}

# GH#53425: Only for deprecation
_builtin_table_alias = {
    builtins.sum: "np.sum",
    builtins.max: "np.maximum.reduce",
    builtins.min: "np.minimum.reduce",
}

_cython_table = {
    builtins.sum: "sum",
    builtins.max: "max",
    builtins.min: "min",
    np.all: "all",
    np.any: "any",
    np.sum: "sum",
    np.nansum: "sum",
    np.mean: "mean",
    np.nanmean: "mean",
    np.prod: "prod",
    np.nanprod: "prod",
    np.std: "std",
    np.nanstd: "std",
    np.var: "var",
    np.nanvar: "var",
    np.median: "median",
    np.nanmedian: "median",
    np.max: "max",
    np.nanmax: "max",
    np.min: "min",
    np.nanmin: "min",
    np.cumprod: "cumprod",
    np.nancumprod: "cumprod",
    np.cumsum: "cumsum",
    np.nancumsum: "cumsum",
}


def get_cython_func(arg: Callable) -> str | None:
    """
    if we define an internal function for this argument, return it
    """
    return _cython_table.get(arg)


def is_builtin_func(arg):
    """
    if we define a builtin function for this argument, return it,
    otherwise return the arg
    """
    return _builtin_table.get(arg, arg)


def fill_missing_names(names: Sequence[Hashable | None]) -> list[Hashable]:
    """
    If a name is missing then replace it by level_n, where n is the count

    .. versionadded:: 1.4.0

    Parameters
    ----------
    names : list-like
        list of column names or None values.

    Returns
    -------
    list
        list of column names with the None values replaced.
    """
    return [f"level_{i}" if name is None else name for i, name in enumerate(names)]
<|endoftext|>from __future__ import annotations

import copy
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
    final,
    no_type_check,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)
from pandas.util._exceptions import (
    find_stack_level,
    rewrite_warning,
)

from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.algorithms as algos
from pandas.core.apply import (
    ResamplerWindowApply,
    warn_alias_replacement,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
import pandas.core.common as com
from pandas.core.generic import (
    NDFrame,
    _shared_docs,
)
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
    BaseGroupBy,
    GroupBy,
    _apply_groupings_depr,
    _pipe_template,
    get_groupby,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.period import (
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)
from pandas.tseries.offsets import (
    Day,
    Tick,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        AnyArrayLike,
        Axis,
        AxisInt,
        Frequency,
        IndexLabel,
        InterpolateOptions,
        T,
        TimedeltaConvertibleTypes,
        TimeGrouperOrigin,
        TimestampConvertibleTypes,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )

_shared_docs_kwargs: dict[str, str] = {}


class Resampler(BaseGroupBy, PandasObject):
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
    groupby : TimeGrouper
    axis : int, default 0
    kind : str or None
        'period', 'timestamp' to override default index treatment

    Returns
    -------
    a Resampler of the appropriate type

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """

    _grouper: BinGrouper
    _timegrouper: TimeGrouper
    binner: DatetimeIndex | TimedeltaIndex | PeriodIndex  # depends on subclass
    exclusions: frozenset[Hashable] = frozenset()  # for SelectionMixin compat
    _internal_names_set = set({"obj", "ax", "_indexer"})

    # to the groupby descriptor
    _attributes = [
        "freq",
        "axis",
        "closed",
        "label",
        "convention",
        "kind",
        "origin",
        "offset",
    ]

    def __init__(
        self,
        obj: NDFrame,
        timegrouper: TimeGrouper,
        axis: Axis = 0,
        kind=None,
        *,
        gpr_index: Index,
        group_keys: bool = False,
        selection=None,
        include_groups: bool = True,
    ) -> None:
        self._timegrouper = timegrouper
        self.keys = None
        self.sort = True
        self.axis = obj._get_axis_number(axis)
        self.kind = kind
        self.group_keys = group_keys
        self.as_index = True
        self.include_groups = include_groups

        self.obj, self.ax, self._indexer = self._timegrouper._set_grouper(
            self._convert_obj(obj), sort=True, gpr_index=gpr_index
        )
        self.binner, self._grouper = self._get_binner()
        self._selection = selection
        if self._timegrouper.key is not None:
            self.exclusions = frozenset([self._timegrouper.key])
        else:
            self.exclusions = frozenset()

    @final
    def __str__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        attrs = (
            f"{k}={getattr(self._timegrouper, k)}"
            for k in self._attributes
            if getattr(self._timegrouper, k, None) is not None
        )
        return f"{type(self).__name__} [{', '.join(attrs)}]"

    @final
    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self._attributes:
            return getattr(self._timegrouper, attr)
        if attr in self.obj:
            return self[attr]

        return object.__getattribute__(self, attr)

    @final
    @property
    def _from_selection(self) -> bool:
        """
        Is the resampling from a DataFrame column or MultiIndex level.
        """
        # upsampling and PeriodIndex resampling do not work
        # with selection, this state used to catch and raise an error
        return self._timegrouper is not None and (
            self._timegrouper.key is not None or self._timegrouper.level is not None
        )

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        """
        Provide any conversions for the object in order to correctly handle.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Series or DataFrame
        """
        return obj._consolidate()

    def _get_binner_for_time(self):
        raise AbstractMethodError(self)

    @final
    def _get_binner(self):
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
        binner, bins, binlabels = self._get_binner_for_time()
        assert len(bins) == len(binlabels)
        bin_grouper = BinGrouper(bins, binlabels, indexer=self._indexer)
        return binner, bin_grouper

    @final
    @Substitution(
        klass="Resampler",
        examples="""
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},
    ...                   index=pd.date_range('2012-08-02', periods=4))
    >>> df
                A
    2012-08-02  1
    2012-08-03  2
    2012-08-04  3
    2012-08-05  4

    To get the difference between each 2-day period's maximum and minimum
    value in one pass, you can do

    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())
                A
    2012-08-02  1
    2012-08-04  1""",
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args,
        **kwargs,
    ) -> T:
        return super().pipe(func, *args, **kwargs)

    _agg_see_also_doc = dedent(
        """
    See Also
    --------
    DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
        or list of string/callables.
    DataFrame.resample.transform : Transforms the Series on each group
        based on the given function.
    DataFrame.aggregate: Aggregate using one or more
        operations over the specified axis.
    """
    )

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5],
    ...               index=pd.date_range('20130101', periods=5, freq='s'))
    >>> s
    2013-01-01 00:00:00    1
    2013-01-01 00:00:01    2
    2013-01-01 00:00:02    3
    2013-01-01 00:00:03    4
    2013-01-01 00:00:04    5
    Freq: s, dtype: int64

    >>> r = s.resample('2s')

    >>> r.agg("sum")
    2013-01-01 00:00:00    3
    2013-01-01 00:00:02    7
    2013-01-01 00:00:04    5
    Freq: 2s, dtype: int64

    >>> r.agg(['sum', 'mean', 'max'])
                         sum  mean  max
    2013-01-01 00:00:00    3   1.5    2
    2013-01-01 00:00:02    7   3.5    4
    2013-01-01 00:00:04    5   5.0    5

    >>> r.agg({'result': lambda x: x.mean() / x.std(),
    ...        'total': "sum"})
                           result  total
    2013-01-01 00:00:00  2.121320      3
    2013-01-01 00:00:02  4.949747      7
    2013-01-01 00:00:04       NaN      5

    >>> r.agg(average="mean", total="sum")
                             average  total
    2013-01-01 00:00:00      1.5      3
    2013-01-01 00:00:02      3.5      7
    2013-01-01 00:00:04      5.0      5
    """
    )

    @final
    @doc(
        _shared_docs["aggregate"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
        klass="DataFrame",
        axis="",
    )
    def aggregate(self, func=None, *args, **kwargs):
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            how = func
            result = self._groupby_and_aggregate(how, *args, **kwargs)

        return result

    agg = aggregate
    apply = aggregate

    @final
    def transform(self, arg, *args, **kwargs):
        """
        Call function producing a like-indexed Series on each group.

        Return a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([1, 2],
        ...               index=pd.date_range('20180101',
        ...                                   periods=2,
        ...                                   freq='1h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> resampled = s.resample('15min')
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        2018-01-01 00:00:00   NaN
        2018-01-01 01:00:00   NaN
        Freq: h, dtype: float64
        """
        return self._selected_obj.groupby(self._timegrouper).transform(
            arg, *args, **kwargs
        )

    def _downsample(self, f, **kwargs):
        raise AbstractMethodError(self)

    def _upsample(self, f, limit: int | None = None, fill_value=None):
        raise AbstractMethodError(self)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        grouper = self._grouper
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                # reached via Apply.agg_dict_like with selection=None and ndim=1
                assert subset.ndim == 1
        if ndim == 1:
            assert subset.ndim == 1

        grouped = get_groupby(
            subset, by=None, grouper=grouper, axis=self.axis, group_keys=self.group_keys
        )
        return grouped

    def _groupby_and_aggregate(self, how, *args, **kwargs):
        """
        Re-evaluate the obj with a groupby aggregation.
        """
        grouper = self._grouper

        # Excludes `on` column when provided
        obj = self._obj_with_exclusions

        grouped = get_groupby(
            obj, by=None, grouper=grouper, axis=self.axis, group_keys=self.group_keys
        )

        try:
            if callable(how):
                # TODO: test_resample_apply_with_additional_args fails if we go
                #  through the non-lambda path, not clear that it should.
                func = lambda x: how(x, *args, **kwargs)
                result = grouped.aggregate(func)
            else:
                result = grouped.aggregate(how, *args, **kwargs)
        except (AttributeError, KeyError):
            # we have a non-reducing function; try to evaluate
            # alternatively we want to evaluate only a column of the input

            # test_apply_to_one_column_of_df the function being applied references
            #  a DataFrame column, but aggregate_item_by_item operates column-wise
            #  on Series, raising AttributeError or KeyError
            #  (depending on whether the column lookup uses getattr/__getitem__)
            result = _apply(
                grouped, how, *args, include_groups=self.include_groups, **kwargs
            )

        except ValueError as err:
            if "Must produce aggregated value" in str(err):
                # raised in _aggregate_named
                # see test_apply_without_aggregation, test_apply_with_mutated_index
                pass
            else:
                raise

            # we have a non-reducing function
            # try to evaluate
            result = _apply(
                grouped, how, *args, include_groups=self.include_groups, **kwargs
            )

        return self._wrap_result(result)

    @final
    def _get_resampler_for_grouping(
        self, groupby: GroupBy, key, include_groups: bool = True
    ):
        """
        Return the correct class for resampling with groupby.
        """
        return self._resampler_for_grouping(
            groupby=groupby, key=key, parent=self, include_groups=include_groups
        )

    def _wrap_result(self, result):
        """
        Potentially wrap any results.
        """
        # GH 47705
        obj = self.obj
        if (
            isinstance(result, ABCDataFrame)
            and len(result) == 0
            and not isinstance(result.index, PeriodIndex)
        ):
            result = result.set_index(
                _asfreq_compat(obj.index[:0], freq=self.freq), append=True
            )

        if isinstance(result, ABCSeries) and self._selection is not None:
            result.name = self._selection

        if isinstance(result, ABCSeries) and result.empty:
            # When index is all NaT, result is empty but index is not
            result.index = _asfreq_compat(obj.index[:0], freq=self.freq)
            result.name = getattr(obj, "name", None)

        if self._timegrouper._arrow_dtype is not None:
            result.index = result.index.astype(self._timegrouper._arrow_dtype)

        return result

    @final
    def ffill(self, limit: int | None = None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        An upsampled Series.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.

        Examples
        --------
        Here we only create a ``Series``.

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        Example for ``ffill`` with downsampling (we have fewer dates after resampling):

        >>> ser.resample('MS').ffill()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64

        Example for ``ffill`` with upsampling (fill the new dates with
        the previous value):

        >>> ser.resample('W').ffill()
        2023-01-01    1
        2023-01-08    1
        2023-01-15    2
        2023-01-22    2
        2023-01-29    2
        2023-02-05    3
        2023-02-12    3
        2023-02-19    4
        Freq: W-SUN, dtype: int64

        With upsampling and limiting (only fill the first new date with the
        previous value):

        >>> ser.resample('W').ffill(limit=1)
        2023-01-01    1.0
        2023-01-08    1.0
        2023-01-15    2.0
        2023-01-22    2.0
        2023-01-29    NaN
        2023-02-05    3.0
        2023-02-12    NaN
        2023-02-19    4.0
        Freq: W-SUN, dtype: float64
        """
        return self._upsample("ffill", limit=limit)

    @final
    def nearest(self, limit: int | None = None):
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        backfill : Backward fill the new missing values in the resampled data.
        pad : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2],
        ...               index=pd.date_range('20180101',
        ...                                   periods=2,
        ...                                   freq='1h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> s.resample('15min').nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15min, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample('15min').nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15min, dtype: float64
        """
        return self._upsample("nearest", limit=limit)

    @final
    def bfill(self, limit: int | None = None):
        """
        Backward fill the new missing values in the resampled data.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency). The backward fill will replace NaN values that appeared in
        the resampled data with the next value in the original sequence.
        Missing values that existed in the original data will not be modified.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series, DataFrame
            An upsampled Series or DataFrame with backward filled NaN values.

        See Also
        --------
        bfill : Alias of backfill.
        fillna : Fill NaN values using the specified method, which can be
            'backfill'.
        nearest : Fill NaN values with nearest neighbor starting from center.
        ffill : Forward fill NaN values.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'backfill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'backfill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: h, dtype: int64

        >>> s.resample('30min').bfill()
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample('15min').bfill(limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15min, dtype: float64

        Resampling a DataFrame that has missing values:

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').bfill()
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('15min').bfill(limit=2)
                               a    b
        2018-01-01 00:00:00  2.0  1.0
        2018-01-01 00:15:00  NaN  NaN
        2018-01-01 00:30:00  NaN  3.0
        2018-01-01 00:45:00  NaN  3.0
        2018-01-01 01:00:00  NaN  3.0
        2018-01-01 01:15:00  NaN  NaN
        2018-01-01 01:30:00  6.0  5.0
        2018-01-01 01:45:00  6.0  5.0
        2018-01-01 02:00:00  6.0  5.0
        """
        return self._upsample("bfill", limit=limit)

    @final
    def fillna(self, method, limit: int | None = None):
        """
        Fill missing values introduced by upsampling.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency).

        Missing values that existed in the original data will
        not be modified.

        Parameters
        ----------
        method : {'pad', 'backfill', 'ffill', 'bfill', 'nearest'}
            Method to use for filling holes in resampled data

            * 'pad' or 'ffill': use previous valid observation to fill gap
              (forward fill).
            * 'backfill' or 'bfill': use next valid observation to fill gap.
            * 'nearest': use nearest valid observation to fill gap.

        limit : int, optional
            Limit of how many consecutive missing values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with missing values filled.

        See Also
        --------
        bfill : Backward fill NaN values in the resampled data.
        ffill : Forward fill NaN values in the resampled data.
        nearest : Fill NaN values in the resampled data
            with nearest neighbor starting from center.
        interpolate : Fill NaN values using interpolation.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'bfill' and 'ffill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'bfill' and 'ffill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: h, dtype: int64

        Without filling the missing values you get:

        >>> s.resample("30min").asfreq()
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    2.0
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        >>> s.resample('30min').fillna("backfill")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample('15min').fillna("backfill", limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15min, dtype: float64

        >>> s.resample('30min').fillna("pad")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    1
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    2
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample('30min').fillna("nearest")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        Missing values present before the upsampling are not affected.

        >>> sm = pd.Series([1, None, 3],
        ...                index=pd.date_range('20180101', periods=3, freq='h'))
        >>> sm
        2018-01-01 00:00:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: h, dtype: float64

        >>> sm.resample('30min').fillna('backfill')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        >>> sm.resample('30min').fillna('pad')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        >>> sm.resample('30min').fillna('nearest')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        DataFrame resampling is done column-wise. All the same options are
        available.

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').fillna("bfill")
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5
        """
        warnings.warn(
            f"{type(self).__name__}.fillna is deprecated and will be removed "
            "in a future version. Use obj.ffill(), obj.bfill(), "
            "or obj.nearest() instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._upsample(method, limit=limit)

    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area=None,
        downcast=lib.no_default,
        **kwargs,
    ):
        """
        Interpolate values between target timestamps according to different methods.

        The original index is first reindexed to target timestamps
        (see :meth:`core.resample.Resampler.asfreq`),
        then the interpolation of ``NaN`` values via :meth:`DataFrame.interpolate`
        happens.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. One of:

            * 'linear': Ignore the index and treat the values as equally
              spaced. This is the only method supported on MultiIndexes.
            * 'time': Works on daily and higher resolution data to interpolate
              given length of interval.
            * 'index', 'values': use the actual numerical values of the index.
            * 'pad': Fill in NaNs using existing values.
            * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
              'barycentric', 'polynomial': Passed to
              `scipy.interpolate.interp1d`, whereas 'spline' is passed to
              `scipy.interpolate.UnivariateSpline`. These methods use the numerical
              values of the index.  Both 'polynomial' and 'spline' require that
              you also specify an `order` (int), e.g.
              ``df.interpolate(method='polynomial', order=5)``. Note that,
              `slinear` method in Pandas refers to the Scipy first order `spline`
              instead of Pandas first order `spline`.
            * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
              'cubicspline': Wrappers around the SciPy interpolation methods of
              similar names. See `Notes`.
            * 'from_derivatives': Refers to
              `scipy.interpolate.BPoly.from_derivatives`.

        axis : {{0 or 'index', 1 or 'columns', None}}, default None
            Axis to interpolate along. For `Series` this parameter is unused
            and defaults to 0.
        limit : int, optional
            Maximum number of consecutive NaNs to fill. Must be greater than
            0.
        inplace : bool, default False
            Update the data in place if possible.
        limit_direction : {{'forward', 'backward', 'both'}}, Optional
            Consecutive NaNs will be filled in this direction.

            If limit is specified:
                * If 'method' is 'pad' or 'ffill', 'limit_direction' must be 'forward'.
                * If 'method' is 'backfill' or 'bfill', 'limit_direction' must be
                  'backwards'.

            If 'limit' is not specified:
                * If 'method' is 'backfill' or 'bfill', the default is 'backward'
                * else the default is 'forward'

                raises ValueError if `limit_direction` is 'forward' or 'both' and
                    method is 'backfill' or 'bfill'.
                raises ValueError if `limit_direction` is 'backward' or 'both' and
                    method is 'pad' or 'ffill'.

        limit_area : {{`None`, 'inside', 'outside'}}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

        downcast : optional, 'infer' or None, defaults to None
            Downcast dtypes if possible.

            .. deprecated:: 2.1.0

        ``**kwargs`` : optional
            Keyword arguments to pass on to the interpolating function.

        Returns
        -------
        DataFrame or Series
            Interpolated values at the specified freq.

        See Also
        --------
        core.resample.Resampler.asfreq: Return the values at the new freq,
            essentially a reindex.
        DataFrame.interpolate: Fill NaN values using an interpolation method.

        Notes
        -----
        For high-frequent or non-equidistant time-series with timestamps
        the reindexing followed by interpolation may lead to information loss
        as shown in the last example.

        Examples
        --------

        >>> start = "2023-03-01T07:00:00"
        >>> timesteps = pd.date_range(start, periods=5, freq="s")
        >>> series = pd.Series(data=[1, -1, 2, 1, 3], index=timesteps)
        >>> series
        2023-03-01 07:00:00    1
        2023-03-01 07:00:01   -1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:03    1
        2023-03-01 07:00:04    3
        Freq: s, dtype: int64

        Upsample the dataframe to 0.5Hz by providing the period time of 2s.

        >>> series.resample("2s").interpolate("linear")
        2023-03-01 07:00:00    1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:04    3
        Freq: 2s, dtype: int64

        Downsample the dataframe to 2Hz by providing the period time of 500ms.

        >>> series.resample("500ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.500    0.0
        2023-03-01 07:00:01.000   -1.0
        2023-03-01 07:00:01.500    0.5
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.500    1.5
        2023-03-01 07:00:03.000    1.0
        2023-03-01 07:00:03.500    2.0
        2023-03-01 07:00:04.000    3.0
        Freq: 500ms, dtype: float64

        Internal reindexing with ``asfreq()`` prior to interpolation leads to
        an interpolated timeseries on the basis the reindexed timestamps (anchors).
        Since not all datapoints from original series become anchors,
        it can lead to misleading interpolation results as in the following example:

        >>> series.resample("400ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.400    1.2
        2023-03-01 07:00:00.800    1.4
        2023-03-01 07:00:01.200    1.6
        2023-03-01 07:00:01.600    1.8
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.400    2.2
        2023-03-01 07:00:02.800    2.4
        2023-03-01 07:00:03.200    2.6
        2023-03-01 07:00:03.600    2.8
        2023-03-01 07:00:04.000    3.0
        Freq: 400ms, dtype: float64

        Note that the series erroneously increases between two anchors
        ``07:00:00`` and ``07:00:02``.
        """
        assert downcast is lib.no_default  # just checking coverage
        result = self._upsample("asfreq")
        return result.interpolate(
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )

    @final
    def asfreq(self, fill_value=None):
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.

        Examples
        --------

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-31', '2023-02-01', '2023-02-28']))
        >>> ser
        2023-01-01    1
        2023-01-31    2
        2023-02-01    3
        2023-02-28    4
        dtype: int64
        >>> ser.resample('MS').asfreq()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._upsample("asfreq", fill_value=fill_value)

    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        *args,
        **kwargs,
    ):
        """
        Compute sum of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').sum()
        2023-01-01    3
        2023-02-01    7
        Freq: MS, dtype: int64
        """
        maybe_warn_args_and_kwargs(type(self), "sum", args, kwargs)
        nv.validate_resampler_func("sum", args, kwargs)
        return self._downsample("sum", numeric_only=numeric_only, min_count=min_count)

    @final
    def prod(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        *args,
        **kwargs,
    ):
        """
        Compute prod of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed prod of values within each group.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').prod()
        2023-01-01    2
        2023-02-01   12
        Freq: MS, dtype: int64
        """
        maybe_warn_args_and_kwargs(type(self), "prod", args, kwargs)
        nv.validate_resampler_func("prod", args, kwargs)
        return self._downsample("prod", numeric_only=numeric_only, min_count=min_count)

    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        *args,
        **kwargs,
    ):
        """
        Compute min value of group.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').min()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """

        maybe_warn_args_and_kwargs(type(self), "min", args, kwargs)
        nv.validate_resampler_func("min", args, kwargs)
        return self._downsample("min", numeric_only=numeric_only, min_count=min_count)

    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        *args,
        **kwargs,
    ):
        """
        Compute max value of group.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').max()
        2023-01-01    2
        2023-02-01    4
        Freq: MS, dtype: int64
        """
        maybe_warn_args_and_kwargs(type(self), "max", args, kwargs)
        nv.validate_resampler_func("max", args, kwargs)
        return self._downsample("max", numeric_only=numeric_only, min_count=min_count)

    @final
    @doc(GroupBy.first)
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
        *args,
        **kwargs,
    ):
        maybe_warn_args_and_kwargs(type(self), "first", args, kwargs)
        nv.validate_resampler_func("first", args, kwargs)
        return self._downsample(
            "first", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    @doc(GroupBy.last)
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
        *args,
        **kwargs,
    ):
        maybe_warn_args_and_kwargs(type(self), "last", args, kwargs)
        nv.validate_resampler_func("last", args, kwargs)
        return self._downsample(
            "last", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    @doc(GroupBy.median)
    def median(self, numeric_only: bool = False, *args, **kwargs):
        maybe_warn_args_and_kwargs(type(self), "median", args, kwargs)
        nv.validate_resampler_func("median", args, kwargs)
        return self._downsample("median", numeric_only=numeric_only)

    @final
    def mean(
        self,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Mean of values within each group.

        Examples
        --------

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').mean()
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        maybe_warn_args_and_kwargs(type(self), "mean", args, kwargs)
        nv.validate_resampler_func("mean", args, kwargs)
        return self._downsample("mean", numeric_only=numeric_only)

    @final
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ):
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Standard deviation of values within each group.

        Examples
        --------

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').std()
        2023-01-01    1.000000
        2023-02-01    2.645751
        Freq: MS, dtype: float64
        """
        maybe_warn_args_and_kwargs(type(self), "std", args, kwargs)
        nv.validate_resampler_func("std", args, kwargs)
        return self._downsample("std", ddof=ddof, numeric_only=numeric_only)

    @final
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ):
        """
        Compute variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Variance of values within each group.

        Examples
        --------

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').var()
        2023-01-01    1.0
        2023-02-01    7.0
        Freq: MS, dtype: float64

        >>> ser.resample('MS').var(ddof=0)
        2023-01-01    0.666667
        2023-02-01    4.666667
        Freq: MS, dtype: float64
        """
        maybe_warn_args_and_kwargs(type(self), "var", args, kwargs)
        nv.validate_resampler_func("var", args, kwargs)
        return self._downsample("var", ddof=ddof, numeric_only=numeric_only)

    @final
    @doc(GroupBy.sem)
    def sem(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ):
        maybe_warn_args_and_kwargs(type(self), "sem", args, kwargs)
        nv.validate_resampler_func("sem", args, kwargs)
        return self._downsample("sem", ddof=ddof, numeric_only=numeric_only)

    @final
    @doc(GroupBy.ohlc)
    def ohlc(
        self,
        *args,
        **kwargs,
    ):
        maybe_warn_args_and_kwargs(type(self), "ohlc", args, kwargs)
        nv.validate_resampler_func("ohlc", args, kwargs)

        ax = self.ax
        obj = self._obj_with_exclusions
        if len(ax) == 0:
            # GH#42902
            obj = obj.copy()
            obj.index = _asfreq_compat(obj.index, self.freq)
            if obj.ndim == 1:
                obj = obj.to_frame()
                obj = obj.reindex(["open", "high", "low", "close"], axis=1)
            else:
                mi = MultiIndex.from_product(
                    [obj.columns, ["open", "high", "low", "close"]]
                )
                obj = obj.reindex(mi, axis=1)
            return obj

        return self._downsample("ohlc")

    @final
    @doc(SeriesGroupBy.nunique)
    def nunique(
        self,
        *args,
        **kwargs,
    ):
        maybe_warn_args_and_kwargs(type(self), "nunique", args, kwargs)
        nv.validate_resampler_func("nunique", args, kwargs)
        return self._downsample("nunique")

    @final
    @doc(GroupBy.size)
    def size(self):
        result = self._downsample("size")

        # If the result is a non-empty DataFrame we stack to get a Series
        # GH 46826
        if isinstance(result, ABCDataFrame) and not result.empty:
            result = result.stack(future_stack=True)

        if not len(self.ax):
            from pandas import Series

            if self._selected_obj.ndim == 1:
                name = self._selected_obj.name
            else:
                name = None
            result = Series([], index=result.index, dtype="int64", name=name)
        return result

    @final
    @doc(GroupBy.count)
    def count(self):
        result = self._downsample("count")
        if not len(self.ax):
            if self._selected_obj.ndim == 1:
                result = type(self._selected_obj)(
                    [], index=result.index, dtype="int64", name=self._selected_obj.name
                )
            else:
                from pandas import DataFrame

                result = DataFrame(
                    [], index=result.index, columns=result.columns, dtype="int64"
                )

        return result

    @final
    def quantile(self, q: float | list[float] | AnyArrayLike = 0.5, **kwargs):
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the columns are groupby columns,
            and the values are its quantiles.

        Examples
        --------

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').quantile()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64

        >>> ser.resample('MS').quantile(.25)
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        return self._downsample("quantile", q=q, **kwargs)


class _GroupByMixin(PandasObject, SelectionMixin):
    """
    Provide the groupby facilities.
    """

    _attributes: list[str]  # in practice the same as Resampler._attributes
    _selection: IndexLabel | None = None
    _groupby: GroupBy
    _timegrouper: TimeGrouper

    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key=None,
        selection: IndexLabel | None = None,
        include_groups: bool = False,
    ) -> None:
        # reached via ._gotitem and _get_resampler_for_grouping

        assert isinstance(groupby, GroupBy), type(groupby)

        # parent is always a Resampler, sometimes a _GroupByMixin
        assert isinstance(parent, Resampler), type(parent)

        # initialize our GroupByMixin object with
        # the resampler attributes
        for attr in self._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection

        self.binner = parent.binner
        self.key = key

        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)

        self.ax = parent.ax
        self.obj = parent.obj
        self.include_groups = include_groups

    @no_type_check
    def _apply(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """

        def func(x):
            x = self._resampler_cls(x, timegrouper=self._timegrouper, gpr_index=self.ax)

            if isinstance(f, str):
                return getattr(x, f)(**kwargs)

            return x.apply(f, *args, **kwargs)

        result = _apply(self._groupby, func, include_groups=self.include_groups)
        return self._wrap_result(result)

    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

    @final
    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # create a new object to prevent aliasing
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                # reached via Apply.agg_dict_like with selection=None, ndim=1
                assert subset.ndim == 1

        # Try to select from a DataFrame, falling back to a Series
        try:
            if isinstance(key, list) and self.key not in key and self.key is not None:
                key.append(self.key)
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby

        selection = self._infer_selection(key, subset)

        new_rs = type(self)(
            groupby=groupby,
            parent=cast(Resampler, self),
            selection=selection,
        )
        return new_rs


class DatetimeIndexResampler(Resampler):
    ax: DatetimeIndex

    @property
    def _resampler_for_grouping(self):
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self):
        # this is how we are actually creating the bins
        if self.kind == "period":
            return self._timegrouper._get_time_period_bins(self.ax)
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        orig_how = how
        how = com.get_cython_func(how) or how
        if orig_how != how:
            warn_alias_replacement(self, orig_how, how)
        ax = self.ax

        # Excludes `on` column when provided
        obj = self._obj_with_exclusions

        if not len(ax):
            # reset to the new freq
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert obj.index.freq == self.freq, (obj.index.freq, self.freq)
            return obj

        # do we have a regular frequency

        # error: Item "None" of "Optional[Any]" has no attribute "binlabels"
        if (
            (ax.freq is not None or ax.inferred_freq is not None)
            and len(self._grouper.binlabels) > len(ax)
            and how is None
        ):
            # let's do an asfreq
            return self.asfreq()

        # we are downsampling
        # we want to call the actual grouper method here
        if self.axis == 0:
            result = obj.groupby(self._grouper).aggregate(how, **kwargs)
        else:
            # test_resample_axis1
            result = obj.T.groupby(self._grouper).aggregate(how, **kwargs).T

        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
        if self.closed == "right":
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner

    def _upsample(self, method, limit: int | None = None, fill_value=None):
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
        if self.axis:
            raise AssertionError("axis must be 0")
        if self._from_selection:
            raise ValueError(
                "Upsampling from level= or on= selection "
                "is not supported, use .set_index(...) "
                "to explicitly set index to datetime-like"
            )

        ax = self.ax
        obj = self._selected_obj
        binner = self.binner
        res_index = self._adjust_binner_for_upsample(binner)

        # if we have the same frequency as our axis, then we are equal sampling
        if (
            limit is None
            and to_offset(ax.inferred_freq) == self.freq
            and len(obj) == len(res_index)
        ):
            result = obj.copy()
            result.index = res_index
        else:
            if method == "asfreq":
                method = None
            result = obj.reindex(
                res_index, method=method, limit=limit, fill_value=fill_value
            )

        return self._wrap_result(result)

    def _wrap_result(self, result):
        result = super()._wrap_result(result)

        # we may have a different kind that we were asked originally
        # convert if needed
        if self.kind == "period" and not isinstance(result.index, PeriodIndex):
            if isinstance(result.index, MultiIndex):
                # GH 24103 - e.g. groupby resample
                if not isinstance(result.index.levels[-1], PeriodIndex):
                    new_level = result.index.levels[-1].to_period(self.freq)
                    result.index = result.index.set_levels(new_level, level=-1)
            else:
                result.index = result.index.to_period(self.freq)
        return result


# error: Definition of "ax" in base class "_GroupByMixin" is incompatible
# with definition in base class "DatetimeIndexResampler"
class DatetimeIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, DatetimeIndexResampler
):
    """
    Provides a resample of a groupby implementation
    """

    @property
    def _resampler_cls(self):
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    # error: Incompatible types in assignment (expression has type "PeriodIndex", base
    # class "DatetimeIndexResampler" defined the type as "DatetimeIndex")
    ax: PeriodIndex  # type: ignore[assignment]

    @property
    def _resampler_for_grouping(self):
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self):
        if self.kind == "timestamp":
            return super()._get_binner_for_time()
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        obj = super()._convert_obj(obj)

        if self._from_selection:
            # see GH 14008, GH 12871
            msg = (
                "Resampling from level= or on= selection "
                "with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )
            raise NotImplementedError(msg)

        # convert to timestamp
        if self.kind == "timestamp":
            obj = obj.to_timestamp(how=self.convention)

        return obj

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        # we may need to actually resample as if we are timestamps
        if self.kind == "timestamp":
            return super()._downsample(how, **kwargs)

        orig_how = how
        how = com.get_cython_func(how) or how
        if orig_how != how:
            warn_alias_replacement(self, orig_how, how)
        ax = self.ax

        if is_subperiod(ax.freq, self.freq):
            # Downsampling
            return self._groupby_and_aggregate(how, **kwargs)
        elif is_superperiod(ax.freq, self.freq):
            if how == "ohlc":
                # GH #13083
                # upsampling to subperiods is handled as an asfreq, which works
                # for pure aggregating/reducing methods
                # OHLC reduces along the time dimension, but creates multiple
                # values for each period -> handle by _groupby_and_aggregate()
                return self._groupby_and_aggregate(how)
            return self.asfreq()
        elif ax.freq == self.freq:
            return self.asfreq()

        raise IncompatibleFrequency(
            f"Frequency {ax.freq} cannot be resampled to {self.freq}, "
            "as they are not sub or super periods"
        )

    def _upsample(self, method, limit: int | None = None, fill_value=None):
        """
        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
        # we may need to actually resample as if we are timestamps
        if self.kind == "timestamp":
            return super()._upsample(method, limit=limit, fill_value=fill_value)

        ax = self.ax
        obj = self.obj
        new_index = self.binner

        # Start vs. end of period
        memb = ax.asfreq(self.freq, how=self.convention)

        # Get the fill indexer
        if method == "asfreq":
            method = None
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        new_obj = _take_new_index(
            obj,
            indexer,
            new_index,
            axis=self.axis,
        )
        return self._wrap_result(new_obj)


# error: Definition of "ax" in base class "_GroupByMixin" is incompatible with
# definition in base class "PeriodIndexResampler"
class PeriodIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, PeriodIndexResampler
):
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _resampler_cls(self):
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    # error: Incompatible types in assignment (expression has type "TimedeltaIndex",
    # base class "DatetimeIndexResampler" defined the type as "DatetimeIndex")
    ax: TimedeltaIndex  # type: ignore[assignment]

    @property
    def _resampler_for_grouping(self):
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self):
        return self._timegrouper._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index is allowed to be greater than original range
        so we don't need to change the length of a binner, GH 13022
        """
        return binner


# error: Definition of "ax" in base class "_GroupByMixin" is incompatible with
# definition in base class "DatetimeIndexResampler"
class TimedeltaIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, TimedeltaIndexResampler
):
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _resampler_cls(self):
        return TimedeltaIndexResampler


def get_resampler(obj: Series | DataFrame, kind=None, **kwds) -> Resampler:
    """
    Create a TimeGrouper and return our resampler.
    """
    tg = TimeGrouper(obj, **kwds)  # type: ignore[arg-type]
    return tg._get_resampler(obj, kind=kind)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby: GroupBy,
    rule,
    how=None,
    fill_method=None,
    limit: int | None = None,
    kind=None,
    on=None,
    include_groups: bool = True,
    **kwargs,
) -> Resampler:
    """
    Return our appropriate resampler when grouping as well.
    """
    # .resample uses 'on' similar to how .groupby uses 'key'
    tg = TimeGrouper(freq=rule, key=on, **kwargs)
    resampler = tg._get_resampler(groupby.obj, kind=kind)
    return resampler._get_resampler_for_grouping(
        groupby=groupby, include_groups=include_groups, key=tg.key
    )


class TimeGrouper(Grouper):
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """

    _attributes = Grouper._attributes + (
        "closed",
        "label",
        "how",
        "kind",
        "convention",
        "origin",
        "offset",
    )

    origin: TimeGrouperOrigin

    def __init__(
        self,
        obj: Grouper | None = None,
        freq: Frequency = "Min",
        key: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        how: str = "mean",
        axis: Axis = 0,
        fill_method=None,
        limit: int | None = None,
        kind: str | None = None,
        convention: Literal["start", "end", "e", "s"] | None = None,
        origin: Literal["epoch", "start", "start_day", "end", "end_day"]
        | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
        **kwargs,
    ) -> None:
        # Check for correctness of the keyword arguments which would
        # otherwise silently use the default if misspelled
        if label not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {label} for `label`")
        if closed not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {closed} for `closed`")
        if convention not in {None, "start", "end", "e", "s"}:
            raise ValueError(f"Unsupported value {convention} for `convention`")

        if (
            key is None
            and obj is not None
            and isinstance(obj.index, PeriodIndex)  # type: ignore[attr-defined]
            or (
                key is not None
                and obj is not None
                and getattr(obj[key], "dtype", None) == "period"  # type: ignore[index]
            )
        ):
            freq = to_offset(freq, is_period=True)
        else:
            freq = to_offset(freq)

        end_types = {"ME", "YE", "QE", "BME", "BYE", "BQE", "W"}
        rule = freq.rule_code
        if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        else:
            # The backward resample sets ``closed`` to ``'right'`` by default
            # since the last value should be considered as the edge point for
            # the last bin. When origin in "end" or "end_day", the value for a
            # specific ``Timestamp`` index stands for the resample result from
            # the current ``Timestamp`` minus ``freq`` to the current
            # ``Timestamp`` with a right close.
            if origin in ["end", "end_day"]:
                if closed is None:
                    closed = "right"
                if label is None:
                    label = "right"
            else:
                if closed is None:
                    closed = "left"
                if label is None:
                    label = "left"

        self.closed = closed
        self.label = label
        self.kind = kind
        self.convention = convention if convention is not None else "e"
        self.how = how
        self.fill_method = fill_method
        self.limit = limit
        self.group_keys = group_keys
        self._arrow_dtype: ArrowDtype | None = None

        if origin in ("epoch", "start", "start_day", "end", "end_day"):
            # error: Incompatible types in assignment (expression has type "Union[Union[
            # Timestamp, datetime, datetime64, signedinteger[_64Bit], float, str],
            # Literal['epoch', 'start', 'start_day', 'end', 'end_day']]", variable has
            # type "Union[Timestamp, Literal['epoch', 'start', 'start_day', 'end',
            # 'end_day']]")
            self.origin = origin  # type: ignore[assignment]
        else:
            try:
                self.origin = Timestamp(origin)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "'origin' should be equal to 'epoch', 'start', 'start_day', "
                    "'end', 'end_day' or "
                    f"should be a Timestamp convertible type. Got '{origin}' instead."
                ) from err

        try:
            self.offset = Timedelta(offset) if offset is not None else None
        except (ValueError, TypeError) as err:
            raise ValueError(
                "'offset' should be a Timedelta convertible type. "
                f"Got '{offset}' instead."
            ) from err

        # always sort time groupers
        kwargs["sort"] = True

        super().__init__(freq=freq, key=key, axis=axis, **kwargs)

    def _get_resampler(self, obj: NDFrame, kind=None) -> Resampler:
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : Series or DataFrame
        kind : string, optional
            'period','timestamp','timedelta' are valid

        Returns
        -------
        Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
        _, ax, _ = self._set_grouper(obj, gpr_index=None)
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(
                obj,
                timegrouper=self,
                kind=kind,
                axis=self.axis,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, PeriodIndex) or kind == "period":
            if not isinstance(ax, PeriodIndex):
                warnings.warn(
                    "Resampling with kind='period' is deprecated.  "
                    "Use datetime paths instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return PeriodIndexResampler(
                obj,
                timegrouper=self,
                kind=kind,
                axis=self.axis,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(
                obj,
                timegrouper=self,
                axis=self.axis,
                group_keys=self.group_keys,
                gpr_index=ax,
            )

        raise TypeError(
            "Only valid with DatetimeIndex, "
            "TimedeltaIndex or PeriodIndex, "
            f"but got an instance of '{type(ax).__name__}'"
        )

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
    ) -> tuple[BinGrouper, NDFrameT]:
        # create the resampler and return our binner
        r = self._get_resampler(obj)
        return r._grouper, cast(NDFrameT, r.obj)

    def _get_time_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if len(ax) == 0:
            binner = labels = DatetimeIndex(
                data=[], freq=self.freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels

        first, last = _get_timestamp_range_edges(
            ax.min(),
            ax.max(),
            self.freq,
            unit=ax.unit,
            closed=self.closed,
            origin=self.origin,
            offset=self.offset,
        )
        # GH #12037
        # use first/last directly instead of call replace() on them
        # because replace() will swallow the nanosecond part
        # thus last bin maybe slightly before the end if the end contains
        # nanosecond part and lead to `Values falls after last bin` error
        # GH 25758: If DST lands at midnight (e.g. 'America/Havana'), user feedback
        # has noted that ambiguous=True provides the most sensible result
        binner = labels = date_range(
            freq=self.freq,
            start=first,
            end=last,
            tz=ax.tz,
            name=ax.name,
            ambiguous=True,
            nonexistent="shift_forward",
            unit=ax.unit,
        )

        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)

        # general version, knowing nothing about relative frequencies
        bins = lib.generate_bins_dt64(
            ax_values, bin_edges, self.closed, hasnans=ax.hasnans
        )

        if self.closed == "right":
            labels = binner
            if self.label == "right":
                labels = labels[1:]
        elif self.label == "right":
            labels = labels[1:]

        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)

        # if we end up with more labels than bins
        # adjust the labels
        # GH4076
        if len(bins) < len(labels):
            labels = labels[: len(bins)]

        return binner, bins, labels

    def _adjust_bin_edges(
        self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]
    ) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]:
        # Some hacks for > daily data, see #1471, #1458, #1483

        if self.freq.name in ("BME", "ME", "W") or self.freq.name.split("-")[0] in (
            "BQE",
            "BYE",
            "QE",
            "YE",
            "W",
        ):
            # If the right end-point is on the last day of the month, roll forwards
            # until the last moment of that day. Note that we only do this for offsets
            # which correspond to the end of a super-daily period - "month start", for
            # example, is excluded.
            if self.closed == "right":
                # GH 21459, GH 9119: Adjust the bins relative to the wall time
                edges_dti = binner.tz_localize(None)
                edges_dti = (
                    edges_dti
                    + Timedelta(days=1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                    - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                )
                bin_edges = edges_dti.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8

            # intraday values on last day
            if bin_edges[-2] > ax_values.max():
                bin_edges = bin_edges[:-1]
                binner = binner[:-1]
        else:
            bin_edges = binner.asi8
        return binner, bin_edges

    def _get_time_delta_bins(self, ax: TimedeltaIndex):
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(
                "axis must be a TimedeltaIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if not isinstance(self.freq, Tick):
            # GH#51896
            raise ValueError(
                "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
                f"e.g. '24h' or '3D', not {self.freq}"
            )

        if not len(ax):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels

        start, end = ax.min(), ax.max()

        if self.closed == "right":
            end += self.freq

        labels = binner = timedelta_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        end_stamps = labels
        if self.closed == "left":
            end_stamps += self.freq

        bins = ax.searchsorted(end_stamps, side=self.closed)

        if self.offset:
            # GH 10530 & 31809
            labels += self.offset

        return binner, bins, labels

    def _get_time_period_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        freq = self.freq

        if len(ax) == 0:
            binner = labels = PeriodIndex(
                data=[], freq=freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels

        labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)

        end_stamps = (labels + freq).asfreq(freq, "s").to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side="left")

        return binner, bins, labels

    def _get_period_bins(self, ax: PeriodIndex):
        if not isinstance(ax, PeriodIndex):
            raise TypeError(
                "axis must be a PeriodIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        memb = ax.asfreq(self.freq, how=self.convention)

        # NaT handling as in pandas._lib.lib.generate_bins_dt64()
        nat_count = 0
        if memb.hasnans:
            # error: Incompatible types in assignment (expression has type
            # "bool_", variable has type "int")  [assignment]
            nat_count = np.sum(memb._isnan)  # type: ignore[assignment]
            memb = memb[~memb._isnan]

        if not len(memb):
            # index contains no valid (non-NaT) values
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            if len(ax) > 0:
                # index is all NaT
                binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
            return binner, bins, labels

        freq_mult = self.freq.n

        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how="end")
        bin_shift = 0

        if isinstance(self.freq, Tick):
            # GH 23882 & 31809: get adjusted bin edge labels with 'origin'
            # and 'origin' support. This call only makes sense if the freq is a
            # Tick since offset and origin are only used in those cases.
            # Not doing this check could create an extra empty bin.
            p_start, end = _get_period_range_edges(
                start,
                end,
                self.freq,
                closed=self.closed,
                origin=self.origin,
                offset=self.offset,
            )

            # Get offset for bin edge (not label edge) adjustment
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            # error: Item "Period" of "Union[Period, Any]" has no attribute "n"
            bin_shift = start_offset.n % freq_mult  # type: ignore[union-attr]
            start = p_start

        labels = binner = period_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        i8 = memb.asi8

        # when upsampling to subperiods, we need to generate enough bins
        expected_bins_count = len(binner) * freq_mult
        i8_extend = expected_bins_count - (i8[-1] - i8[0])
        rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
        rng += freq_mult
        # adjust bin edge indexes to account for base
        rng -= bin_shift

        # Wrap in PeriodArray for PeriodArray.searchsorted
        prng = type(memb._data)(rng, dtype=memb.dtype)
        bins = memb.searchsorted(prng, side="left")

        if nat_count > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)

        return binner, bins, labels

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
        if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in "Mm":
            self._arrow_dtype = ax.dtype
            ax = Index(
                cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array()
            )
        return obj, ax, indexer


def _take_new_index(
    obj: NDFrameT, indexer: npt.NDArray[np.intp], new_index: Index, axis: AxisInt = 0
) -> NDFrameT:
    if isinstance(obj, ABCSeries):
        new_values = algos.take_nd(obj._values, indexer)
        # error: Incompatible return value type (got "Series", expected "NDFrameT")
        return obj._constructor(  # type: ignore[return-value]
            new_values, index=new_index, name=obj.name
        )
    elif isinstance(obj, ABCDataFrame):
        if axis == 1:
            raise NotImplementedError("axis 1 is not supported")
        new_mgr = obj._mgr.reindex_indexer(new_axis=new_index, indexer=indexer, axis=1)
        # error: Incompatible return value type (got "DataFrame", expected "NDFrameT")
        return obj._constructor_from_mgr(new_mgr, axes=new_mgr.axes)  # type: ignore[return-value]
    else:
        raise ValueError("'obj' should be either a Series or a DataFrame")


def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: str,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    """
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'} or Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    """
    if isinstance(freq, Tick):
        index_tz = first.tz
        if isinstance(origin, Timestamp) and (origin.tz is None) != (index_tz is None):
            raise ValueError("The origin must have the same timezone as the index.")
        if origin == "epoch":
            # set the epoch based on the timezone to have similar bins results when
            # resampling on the same kind of indexes on different timezones
            origin = Timestamp("1970-01-01", tz=index_tz)

        if isinstance(freq, Day):
            # _adjust_dates_anchored assumes 'D' means 24h, but first/last
            # might contain a DST transition (23h, 24h, or 25h).
            # So "pretend" the dates are naive when adjusting the endpoints
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            if isinstance(origin, Timestamp):
                origin = origin.tz_localize(None)

        first, last = _adjust_dates_anchored(
            first, last, freq, closed=closed, origin=origin, offset=offset, unit=unit
        )
        if isinstance(freq, Day):
            first = first.tz_localize(index_tz)
            last = last.tz_localize(index_tz)
    else:
        first = first.normalize()
        last = last.normalize()

        if closed == "left":
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp(first - freq)

        last = Timestamp(last + freq)

    return first, last


def _get_period_range_edges(
    first: Period,
    last: Period,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Period, Period]:
    """
    Adjust the provided `first` and `last` Periods to the respective Period of
    the given offset that encompasses them.

    Parameters
    ----------
    first : pd.Period
        The beginning Period of the range to be adjusted.
    last : pd.Period
        The ending Period of the range to be adjusted.
    freq : pd.DateOffset
        The freq to which the Periods will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'}, Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.

        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Period objects.
    """
    if not all(isinstance(obj, Period) for obj in [first, last]):
        raise TypeError("'first' and 'last' must be instances of type Period")

    # GH 23882
    first_ts = first.to_timestamp()
    last_ts = last.to_timestamp()
    adjust_first = not freq.is_on_offset(first_ts)
    adjust_last = freq.is_on_offset(last_ts)

    first_ts, last_ts = _get_timestamp_range_edges(
        first_ts, last_ts, freq, unit="ns", closed=closed, origin=origin, offset=offset
    )

    first = (first_ts + int(adjust_first) * freq).to_period(freq)
    last = (last_ts - int(adjust_last) * freq).to_period(freq)
    return first, last


def _insert_nat_bin(
    binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int
) -> tuple[PeriodIndex, np.ndarray, PeriodIndex]:
    # NaT handling as in pandas._lib.lib.generate_bins_dt64()
    # shift bins by the number of NaT
    assert nat_count > 0
    bins += nat_count
    bins = np.insert(bins, 0, nat_count)

    # Incompatible types in assignment (expression has type "Index", variable
    # has type "PeriodIndex")
    binner = binner.insert(0, NaT)  # type: ignore[assignment]
    # Incompatible types in assignment (expression has type "Index", variable
    # has type "PeriodIndex")
    labels = labels.insert(0, NaT)  # type: ignore[assignment]
    return binner, bins, labels


def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
    unit: str = "ns",
) -> tuple[Timestamp, Timestamp]:
    # First and last offsets should be calculated from the start day to fix an
    # error cause by resampling across multiple days when a one day period is
    # not a multiple of the frequency. See GH 8683
    # To handle frequencies that are not multiple or divisible by a day we let
    # the possibility to define a fixed origin timestamp. See GH 31809
    first = first.as_unit(unit)
    last = last.as_unit(unit)
    if offset is not None:
        offset = offset.as_unit(unit)

    freq_value = Timedelta(freq).as_unit(unit)._value

    origin_timestamp = 0  # origin == "epoch"
    if origin == "start_day":
        origin_timestamp = first.normalize()._value
    elif origin == "start":
        origin_timestamp = first._value
    elif isinstance(origin, Timestamp):
        origin_timestamp = origin.as_unit(unit)._value
    elif origin in ["end", "end_day"]:
        origin_last = last if origin == "end" else last.ceil("D")
        sub_freq_times = (origin_last._value - first._value) // freq_value
        if closed == "left":
            sub_freq_times += 1
        first = origin_last - sub_freq_times * freq
        origin_timestamp = first._value
    origin_timestamp += offset._value if offset else 0

    # GH 10117 & GH 19375. If first and last contain timezone information,
    # Perform the calculation in UTC in order to avoid localizing on an
    # Ambiguous or Nonexistent time.
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert("UTC")
    if last_tzinfo is not None:
        last = last.tz_convert("UTC")

    foffset = (first._value - origin_timestamp) % freq_value
    loffset = (last._value - origin_timestamp) % freq_value

    if closed == "right":
        if foffset > 0:
            # roll back
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value - freq_value

        if loffset > 0:
            # roll forward
            lresult_int = last._value + (freq_value - loffset)
        else:
            # already the end of the road
            lresult_int = last._value
    else:  # closed == 'left'
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            # start of the road
            fresult_int = first._value

        if loffset > 0:
            # roll forward
            lresult_int = last._value + (freq_value - loffset)
        else:
            lresult_int = last._value + freq_value
    fresult = Timestamp(fresult_int, unit=unit)
    lresult = Timestamp(lresult_int, unit=unit)
    if first_tzinfo is not None:
        fresult = fresult.tz_localize("UTC").tz_convert(first_tzinfo)
    if last_tzinfo is not None:
        lresult = lresult.tz_localize("UTC").tz_convert(last_tzinfo)
    return fresult, lresult


def asfreq(
    obj: NDFrameT,
    freq,
    method=None,
    how=None,
    normalize: bool = False,
    fill_value=None,
) -> NDFrameT:
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
    if isinstance(obj.index, PeriodIndex):
        if method is not None:
            raise NotImplementedError("'method' argument is not supported")

        if how is None:
            how = "E"

        if isinstance(freq, BaseOffset):
            if hasattr(freq, "_period_dtype_code"):
                freq = freq_to_period_freqstr(freq.n, freq.name)
            else:
                raise ValueError(
                    f"Invalid offset: '{freq.base}' for converting time series "
                    f"with PeriodIndex."
                )

        new_obj = obj.copy()
        new_obj.index = obj.index.asfreq(freq, how=how)

    elif len(obj.index) == 0:
        new_obj = obj.copy()

        new_obj.index = _asfreq_compat(obj.index, freq)
    else:
        unit = None
        if isinstance(obj.index, DatetimeIndex):
            # TODO: should we disallow non-DatetimeIndex?
            unit = obj.index.unit
        dti = date_range(obj.index.min(), obj.index.max(), freq=freq, unit=unit)
        dti.name = obj.index.name
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        if normalize:
            new_obj.index = new_obj.index.normalize()

    return new_obj


def _asfreq_compat(index: DatetimeIndex | PeriodIndex | TimedeltaIndex, freq):
    """
    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.

    Parameters
    ----------
    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex
    freq : DateOffset

    Returns
    -------
    same type as index
    """
    if len(index) != 0:
        # This should never be reached, always checked by the caller
        raise ValueError(
            "Can only set arbitrary freq for empty DatetimeIndex or TimedeltaIndex"
        )
    new_index: Index
    if isinstance(index, PeriodIndex):
        new_index = index.asfreq(freq=freq)
    elif isinstance(index, DatetimeIndex):
        new_index = DatetimeIndex([], dtype=index.dtype, freq=freq, name=index.name)
    elif isinstance(index, TimedeltaIndex):
        new_index = TimedeltaIndex([], dtype=index.dtype, freq=freq, name=index.name)
    else:  # pragma: no cover
        raise TypeError(type(index))
    return new_index


def maybe_warn_args_and_kwargs(cls, kernel: str, args, kwargs) -> None:
    """
    Warn for deprecation of args and kwargs in resample functions.

    Parameters
    ----------
    cls : type
        Class to warn about.
    kernel : str
        Operation name.
    args : tuple or None
        args passed by user. Will be None if and only if kernel does not have args.
    kwargs : dict or None
        kwargs passed by user. Will be None if and only if kernel does not have kwargs.
    """
    warn_args = args is not None and len(args) > 0
    warn_kwargs = kwargs is not None and len(kwargs) > 0
    if warn_args and warn_kwargs:
        msg = "args and kwargs"
    elif warn_args:
        msg = "args"
    elif warn_kwargs:
        msg = "kwargs"
    else:
        return
    warnings.warn(
        f"Passing additional {msg} to {cls.__name__}.{kernel} has "
        "no impact on the result and is deprecated. This will "
        "raise a TypeError in a future version of pandas.",
        category=FutureWarning,
        stacklevel=find_stack_level(),
    )


def _apply(
    grouped: GroupBy, how: Callable, *args, include_groups: bool, **kwargs
) -> DataFrame:
    # GH#7155 - rewrite warning to appear as if it came from `.resample`
    target_message = "DataFrameGroupBy.apply operated on the grouping columns"
    new_message = _apply_groupings_depr.format("DataFrameGroupBy", "resample")
    with rewrite_warning(
        target_message=target_message,
        target_category=FutureWarning,
        new_message=new_message,
    ):
        result = grouped.apply(how, *args, include_groups=include_groups, **kwargs)
    return result
<|endoftext|>"""
This module is imported from the pandas package __init__.py file
in order to ensure that the core.config options registered here will
be available as soon as the user loads the package. if register_option
is invoked inside specific modules, they will not be registered until that
module is imported, which may or may not be a problem.

If you need to make sure options are available even before a certain
module is imported, register them here rather than in the module.

"""
from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
)

import pandas._config.config as cf
from pandas._config.config import (
    is_bool,
    is_callable,
    is_instance_factory,
    is_int,
    is_nonnegative_int,
    is_one_of_factory,
    is_str,
    is_text,
)

# compute

use_bottleneck_doc = """
: bool
    Use the bottleneck library to accelerate if it is installed,
    the default is True
    Valid values: False,True
"""


def use_bottleneck_cb(key) -> None:
    from pandas.core import nanops

    nanops.set_use_bottleneck(cf.get_option(key))


use_numexpr_doc = """
: bool
    Use the numexpr library to accelerate computation if it is installed,
    the default is True
    Valid values: False,True
"""


def use_numexpr_cb(key) -> None:
    from pandas.core.computation import expressions

    expressions.set_use_numexpr(cf.get_option(key))


use_numba_doc = """
: bool
    Use the numba engine option for select operations if it is installed,
    the default is False
    Valid values: False,True
"""


def use_numba_cb(key) -> None:
    from pandas.core.util import numba_

    numba_.set_use_numba(cf.get_option(key))


with cf.config_prefix("compute"):
    cf.register_option(
        "use_bottleneck",
        True,
        use_bottleneck_doc,
        validator=is_bool,
        cb=use_bottleneck_cb,
    )
    cf.register_option(
        "use_numexpr", True, use_numexpr_doc, validator=is_bool, cb=use_numexpr_cb
    )
    cf.register_option(
        "use_numba", False, use_numba_doc, validator=is_bool, cb=use_numba_cb
    )
#
# options from the "display" namespace

pc_precision_doc = """
: int
    Floating point output precision in terms of number of places after the
    decimal, for regular formatting as well as scientific notation. Similar
    to ``precision`` in :meth:`numpy.set_printoptions`.
"""

pc_colspace_doc = """
: int
    Default space for DataFrame columns.
"""

pc_max_rows_doc = """
: int
    If max_rows is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 and pandas will auto-detect
    the height of the terminal and print a truncated object which fits
    the screen height. The IPython notebook, IPython qtconsole, or
    IDLE do not run in a terminal and hence it is not possible to do
    correct auto-detection.
"""

pc_min_rows_doc = """
: int
    The numbers of rows to show in a truncated view (when `max_rows` is
    exceeded). Ignored when `max_rows` is set to None or 0. When set to
    None, follows the value of `max_rows`.
"""

pc_max_cols_doc = """
: int
    If max_cols is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 or None and pandas will auto-detect
    the width of the terminal and print a truncated object which fits
    the screen width. The IPython notebook, IPython qtconsole, or IDLE
    do not run in a terminal and hence it is not possible to do
    correct auto-detection and defaults to 20.
"""

pc_max_categories_doc = """
: int
    This sets the maximum number of categories pandas should output when
    printing out a `Categorical` or a Series of dtype "category".
"""

pc_max_info_cols_doc = """
: int
    max_info_columns is used in DataFrame.info method to decide if
    per column information will be printed.
"""

pc_nb_repr_h_doc = """
: boolean
    When True, IPython notebook will use html representation for
    pandas objects (if it is available).
"""

pc_pprint_nest_depth = """
: int
    Controls the number of nested levels to process when pretty-printing
"""

pc_multi_sparse_doc = """
: boolean
    "sparsify" MultiIndex display (don't display repeated
    elements in outer levels within groups)
"""

float_format_doc = """
: callable
    The callable should accept a floating point number and return
    a string with the desired format of the number. This is used
    in some places like SeriesFormatter.
    See formats.format.EngFormatter for an example.
"""

max_colwidth_doc = """
: int or None
    The maximum width in characters of a column in the repr of
    a pandas data structure. When the column overflows, a "..."
    placeholder is embedded in the output. A 'None' value means unlimited.
"""

colheader_justify_doc = """
: 'left'/'right'
    Controls the justification of column headers. used by DataFrameFormatter.
"""

pc_expand_repr_doc = """
: boolean
    Whether to print out the full DataFrame repr for wide DataFrames across
    multiple lines, `max_columns` is still respected, but the output will
    wrap-around across multiple "pages" if its width exceeds `display.width`.
"""

pc_show_dimensions_doc = """
: boolean or 'truncate'
    Whether to print out dimensions at the end of DataFrame repr.
    If 'truncate' is specified, only print out the dimensions if the
    frame is truncated (e.g. not display all rows and/or columns)
"""

pc_east_asian_width_doc = """
: boolean
    Whether to use the Unicode East Asian Width to calculate the display text
    width.
    Enabling this may affect to the performance (default: False)
"""

pc_ambiguous_as_wide_doc = """
: boolean
    Whether to handle Unicode characters belong to Ambiguous as Wide (width=2)
    (default: False)
"""

pc_table_schema_doc = """
: boolean
    Whether to publish a Table Schema representation for frontends
    that support it.
    (default: False)
"""

pc_html_border_doc = """
: int
    A ``border=value`` attribute is inserted in the ``<table>`` tag
    for the DataFrame HTML repr.
"""

pc_html_use_mathjax_doc = """\
: boolean
    When True, Jupyter notebook will process table contents using MathJax,
    rendering mathematical expressions enclosed by the dollar symbol.
    (default: True)
"""

pc_max_dir_items = """\
: int
    The number of items that will be added to `dir(...)`. 'None' value means
    unlimited. Because dir is cached, changing this option will not immediately
    affect already existing dataframes until a column is deleted or added.

    This is for instance used to suggest columns from a dataframe to tab
    completion.
"""

pc_width_doc = """
: int
    Width of the display in characters. In case python/IPython is running in
    a terminal this can be set to None and pandas will correctly auto-detect
    the width.
    Note that the IPython notebook, IPython qtconsole, or IDLE do not run in a
    terminal and hence it is not possible to correctly detect the width.
"""

pc_chop_threshold_doc = """
: float or None
    if set to a float value, all float values smaller than the given threshold
    will be displayed as exactly 0 by repr and friends.
"""

pc_max_seq_items = """
: int or None
    When pretty-printing a long sequence, no more then `max_seq_items`
    will be printed. If items are omitted, they will be denoted by the
    addition of "..." to the resulting string.

    If set to None, the number of items to be printed is unlimited.
"""

pc_max_info_rows_doc = """
: int
    df.info() will usually show null-counts for each column.
    For large frames this can be quite slow. max_info_rows and max_info_cols
    limit this null check only to frames with smaller dimensions than
    specified.
"""

pc_large_repr_doc = """
: 'truncate'/'info'
    For DataFrames exceeding max_rows/max_cols, the repr (and HTML repr) can
    show a truncated table, or switch to the view from
    df.info() (the behaviour in earlier versions of pandas).
"""

pc_memory_usage_doc = """
: bool, string or None
    This specifies if the memory usage of a DataFrame should be displayed when
    df.info() is called. Valid values True,False,'deep'
"""


def table_schema_cb(key) -> None:
    from pandas.io.formats.printing import enable_data_resource_formatter

    enable_data_resource_formatter(cf.get_option(key))


def is_terminal() -> bool:
    """
    Detect if Python is running in a terminal.

    Returns True if Python is running in a terminal or False if not.
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:  # assume standard Python interpreter in a terminal
        return True
    else:
        if hasattr(ip, "kernel"):  # IPython as a Jupyter kernel
            return False
        else:  # IPython in a terminal
            return True


with cf.config_prefix("display"):
    cf.register_option("precision", 6, pc_precision_doc, validator=is_nonnegative_int)
    cf.register_option(
        "float_format",
        None,
        float_format_doc,
        validator=is_one_of_factory([None, is_callable]),
    )
    cf.register_option(
        "max_info_rows",
        1690785,
        pc_max_info_rows_doc,
        validator=is_int,
    )
    cf.register_option("max_rows", 60, pc_max_rows_doc, validator=is_nonnegative_int)
    cf.register_option(
        "min_rows",
        10,
        pc_min_rows_doc,
        validator=is_instance_factory([type(None), int]),
    )
    cf.register_option("max_categories", 8, pc_max_categories_doc, validator=is_int)

    cf.register_option(
        "max_colwidth",
        50,
        max_colwidth_doc,
        validator=is_nonnegative_int,
    )
    if is_terminal():
        max_cols = 0  # automatically determine optimal number of columns
    else:
        max_cols = 20  # cannot determine optimal number of columns
    cf.register_option(
        "max_columns", max_cols, pc_max_cols_doc, validator=is_nonnegative_int
    )
    cf.register_option(
        "large_repr",
        "truncate",
        pc_large_repr_doc,
        validator=is_one_of_factory(["truncate", "info"]),
    )
    cf.register_option("max_info_columns", 100, pc_max_info_cols_doc, validator=is_int)
    cf.register_option(
        "colheader_justify", "right", colheader_justify_doc, validator=is_text
    )
    cf.register_option("notebook_repr_html", True, pc_nb_repr_h_doc, validator=is_bool)
    cf.register_option("pprint_nest_depth", 3, pc_pprint_nest_depth, validator=is_int)
    cf.register_option("multi_sparse", True, pc_multi_sparse_doc, validator=is_bool)
    cf.register_option("expand_frame_repr", True, pc_expand_repr_doc)
    cf.register_option(
        "show_dimensions",
        "truncate",
        pc_show_dimensions_doc,
        validator=is_one_of_factory([True, False, "truncate"]),
    )
    cf.register_option("chop_threshold", None, pc_chop_threshold_doc)
    cf.register_option("max_seq_items", 100, pc_max_seq_items)
    cf.register_option(
        "width", 80, pc_width_doc, validator=is_instance_factory([type(None), int])
    )
    cf.register_option(
        "memory_usage",
        True,
        pc_memory_usage_doc,
        validator=is_one_of_factory([None, True, False, "deep"]),
    )
    cf.register_option(
        "unicode.east_asian_width", False, pc_east_asian_width_doc, validator=is_bool
    )
    cf.register_option(
        "unicode.ambiguous_as_wide", False, pc_east_asian_width_doc, validator=is_bool
    )
    cf.register_option(
        "html.table_schema",
        False,
        pc_table_schema_doc,
        validator=is_bool,
        cb=table_schema_cb,
    )
    cf.register_option("html.border", 1, pc_html_border_doc, validator=is_int)
    cf.register_option(
        "html.use_mathjax", True, pc_html_use_mathjax_doc, validator=is_bool
    )
    cf.register_option(
        "max_dir_items", 100, pc_max_dir_items, validator=is_nonnegative_int
    )

tc_sim_interactive_doc = """
: boolean
    Whether to simulate interactive mode for purposes of testing
"""

with cf.config_prefix("mode"):
    cf.register_option("sim_interactive", False, tc_sim_interactive_doc)

use_inf_as_na_doc = """
: boolean
    True means treat None, NaN, INF, -INF as NA (old way),
    False means None and NaN are null, but INF, -INF are not NA
    (new way).

    This option is deprecated in pandas 2.1.0 and will be removed in 3.0.
"""

# We don't want to start importing everything at the global context level
# or we'll hit circular deps.


def use_inf_as_na_cb(key) -> None:
    # TODO(3.0): enforcing this deprecation will close GH#52501
    from pandas.core.dtypes.missing import _use_inf_as_na

    _use_inf_as_na(key)


with cf.config_prefix("mode"):
    cf.register_option("use_inf_as_na", False, use_inf_as_na_doc, cb=use_inf_as_na_cb)

cf.deprecate_option(
    # GH#51684
    "mode.use_inf_as_na",
    "use_inf_as_na option is deprecated and will be removed in a future "
    "version. Convert inf values to NaN before operating instead.",
)

data_manager_doc = """
: string
    Internal data manager type; can be "block" or "array". Defaults to "block",
    unless overridden by the 'PANDAS_DATA_MANAGER' environment variable (needs
    to be set before pandas is imported).
"""


with cf.config_prefix("mode"):
    cf.register_option(
        "data_manager",
        # Get the default from an environment variable, if set, otherwise defaults
        # to "block". This environment variable can be set for testing.
        os.environ.get("PANDAS_DATA_MANAGER", "block"),
        data_manager_doc,
        validator=is_one_of_factory(["block", "array"]),
    )

cf.deprecate_option(
    # GH#55043
    "mode.data_manager",
    "data_manager option is deprecated and will be removed in a future "
    "version. Only the BlockManager will be available.",
)


# TODO better name?
copy_on_write_doc = """
: bool
    Use new copy-view behaviour using Copy-on-Write. Defaults to False,
    unless overridden by the 'PANDAS_COPY_ON_WRITE' environment variable
    (if set to "1" for True, needs to be set before pandas is imported).
"""


with cf.config_prefix("mode"):
    cf.register_option(
        "copy_on_write",
        # Get the default from an environment variable, if set, otherwise defaults
        # to False. This environment variable can be set for testing.
        "warn"
        if os.environ.get("PANDAS_COPY_ON_WRITE", "0") == "warn"
        else os.environ.get("PANDAS_COPY_ON_WRITE", "0") == "1",
        copy_on_write_doc,
        validator=is_one_of_factory([True, False, "warn"]),
    )


# user warnings
chained_assignment = """
: string
    Raise an exception, warn, or no action if trying to use chained assignment,
    The default is warn
"""

with cf.config_prefix("mode"):
    cf.register_option(
        "chained_assignment",
        "warn",
        chained_assignment,
        validator=is_one_of_factory([None, "warn", "raise"]),
    )


string_storage_doc = """
: string
    The default storage for StringDtype.
"""


def is_valid_string_storage(value: Any) -> None:
    legal_values = ["auto", "python", "pyarrow"]
    if value not in legal_values:
        msg = "Value must be one of python|pyarrow"
        if value == "pyarrow_numpy":
            # TODO: we can remove extra message after 3.0
            msg += (
                ". 'pyarrow_numpy' was specified, but this option should be "
                "enabled using pandas.options.future.infer_string instead"
            )
        raise ValueError(msg)


with cf.config_prefix("mode"):
    cf.register_option(
        "string_storage",
        "auto",
        string_storage_doc,
        # validator=is_one_of_factory(["python", "pyarrow"]),
        validator=is_valid_string_storage,
    )


# Set up the io.excel specific reader configuration.
reader_engine_doc = """
: string
    The default Excel reader engine for '{ext}' files. Available options:
    auto, {others}.
"""

_xls_options = ["xlrd", "calamine"]
_xlsm_options = ["xlrd", "openpyxl", "calamine"]
_xlsx_options = ["xlrd", "openpyxl", "calamine"]
_ods_options = ["odf", "calamine"]
_xlsb_options = ["pyxlsb", "calamine"]


with cf.config_prefix("io.excel.xls"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xls", others=", ".join(_xls_options)),
        validator=is_one_of_factory(_xls_options + ["auto"]),
    )

with cf.config_prefix("io.excel.xlsm"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsm", others=", ".join(_xlsm_options)),
        validator=is_one_of_factory(_xlsm_options + ["auto"]),
    )


with cf.config_prefix("io.excel.xlsx"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsx", others=", ".join(_xlsx_options)),
        validator=is_one_of_factory(_xlsx_options + ["auto"]),
    )


with cf.config_prefix("io.excel.ods"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="ods", others=", ".join(_ods_options)),
        validator=is_one_of_factory(_ods_options + ["auto"]),
    )

with cf.config_prefix("io.excel.xlsb"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsb", others=", ".join(_xlsb_options)),
        validator=is_one_of_factory(_xlsb_options + ["auto"]),
    )

# Set up the io.excel specific writer configuration.
writer_engine_doc = """
: string
    The default Excel writer engine for '{ext}' files. Available options:
    auto, {others}.
"""

_xlsm_options = ["openpyxl"]
_xlsx_options = ["openpyxl", "xlsxwriter"]
_ods_options = ["odf"]


with cf.config_prefix("io.excel.xlsm"):
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="xlsm", others=", ".join(_xlsm_options)),
        validator=str,
    )


with cf.config_prefix("io.excel.xlsx"):
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="xlsx", others=", ".join(_xlsx_options)),
        validator=str,
    )


with cf.config_prefix("io.excel.ods"):
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="ods", others=", ".join(_ods_options)),
        validator=str,
    )


# Set up the io.parquet specific configuration.
parquet_engine_doc = """
: string
    The default parquet reader/writer engine. Available options:
    'auto', 'pyarrow', 'fastparquet', the default is 'auto'
"""

with cf.config_prefix("io.parquet"):
    cf.register_option(
        "engine",
        "auto",
        parquet_engine_doc,
        validator=is_one_of_factory(["auto", "pyarrow", "fastparquet"]),
    )


# Set up the io.sql specific configuration.
sql_engine_doc = """
: string
    The default sql reader/writer engine. Available options:
    'auto', 'sqlalchemy', the default is 'auto'
"""

with cf.config_prefix("io.sql"):
    cf.register_option(
        "engine",
        "auto",
        sql_engine_doc,
        validator=is_one_of_factory(["auto", "sqlalchemy"]),
    )

# --------
# Plotting
# ---------

plotting_backend_doc = """
: str
    The plotting backend to use. The default value is "matplotlib", the
    backend provided with pandas. Other backends can be specified by
    providing the name of the module that implements the backend.
"""


def register_plotting_backend_cb(key) -> None:
    if key == "matplotlib":
        # We defer matplotlib validation, since it's the default
        return
    from pandas.plotting._core import _get_plot_backend

    _get_plot_backend(key)


with cf.config_prefix("plotting"):
    cf.register_option(
        "backend",
        defval="matplotlib",
        doc=plotting_backend_doc,
        validator=register_plotting_backend_cb,
    )


register_converter_doc = """
: bool or 'auto'.
    Whether to register converters with matplotlib's units registry for
    dates, times, datetimes, and Periods. Toggling to False will remove
    the converters, restoring any converters that pandas overwrote.
"""


def register_converter_cb(key) -> None:
    from pandas.plotting import (
        deregister_matplotlib_converters,
        register_matplotlib_converters,
    )

    if cf.get_option(key):
        register_matplotlib_converters()
    else:
        deregister_matplotlib_converters()


with cf.config_prefix("plotting.matplotlib"):
    cf.register_option(
        "register_converters",
        "auto",
        register_converter_doc,
        validator=is_one_of_factory(["auto", True, False]),
        cb=register_converter_cb,
    )

# ------
# Styler
# ------

styler_sparse_index_doc = """
: bool
    Whether to sparsify the display of a hierarchical index. Setting to False will
    display each explicit level element in a hierarchical key for each row.
"""

styler_sparse_columns_doc = """
: bool
    Whether to sparsify the display of hierarchical columns. Setting to False will
    display each explicit level element in a hierarchical key for each column.
"""

styler_render_repr = """
: str
    Determine which output to use in Jupyter Notebook in {"html", "latex"}.
"""

styler_max_elements = """
: int
    The maximum number of data-cell (<td>) elements that will be rendered before
    trimming will occur over columns, rows or both if needed.
"""

styler_max_rows = """
: int, optional
    The maximum number of rows that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
"""

styler_max_columns = """
: int, optional
    The maximum number of columns that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
"""

styler_precision = """
: int
    The precision for floats and complex numbers.
"""

styler_decimal = """
: str
    The character representation for the decimal separator for floats and complex.
"""

styler_thousands = """
: str, optional
    The character representation for thousands separator for floats, int and complex.
"""

styler_na_rep = """
: str, optional
    The string representation for values identified as missing.
"""

styler_escape = """
: str, optional
    Whether to escape certain characters according to the given context; html or latex.
"""

styler_formatter = """
: str, callable, dict, optional
    A formatter object to be used as default within ``Styler.format``.
"""

styler_multirow_align = """
: {"c", "t", "b"}
    The specifier for vertical alignment of sparsified LaTeX multirows.
"""

styler_multicol_align = r"""
: {"r", "c", "l", "naive-l", "naive-r"}
    The specifier for horizontal alignment of sparsified LaTeX multicolumns. Pipe
    decorators can also be added to non-naive values to draw vertical
    rules, e.g. "\|r" will draw a rule on the left side of right aligned merged cells.
"""

styler_hrules = """
: bool
    Whether to add horizontal rules on top and bottom and below the headers.
"""

styler_environment = """
: str
    The environment to replace ``\\begin{table}``. If "longtable" is used results
    in a specific longtable environment format.
"""

styler_encoding = """
: str
    The encoding used for output HTML and LaTeX files.
"""

styler_mathjax = """
: bool
    If False will render special CSS classes to table attributes that indicate Mathjax
    will not be used in Jupyter Notebook.
"""

with cf.config_prefix("styler"):
    cf.register_option("sparse.index", True, styler_sparse_index_doc, validator=is_bool)

    cf.register_option(
        "sparse.columns", True, styler_sparse_columns_doc, validator=is_bool
    )

    cf.register_option(
        "render.repr",
        "html",
        styler_render_repr,
        validator=is_one_of_factory(["html", "latex"]),
    )

    cf.register_option(
        "render.max_elements",
        2**18,
        styler_max_elements,
        validator=is_nonnegative_int,
    )

    cf.register_option(
        "render.max_rows",
        None,
        styler_max_rows,
        validator=is_nonnegative_int,
    )

    cf.register_option(
        "render.max_columns",
        None,
        styler_max_columns,
        validator=is_nonnegative_int,
    )

    cf.register_option("render.encoding", "utf-8", styler_encoding, validator=is_str)

    cf.register_option("format.decimal", ".", styler_decimal, validator=is_str)

    cf.register_option(
        "format.precision", 6, styler_precision, validator=is_nonnegative_int
    )

    cf.register_option(
        "format.thousands",
        None,
        styler_thousands,
        validator=is_instance_factory([type(None), str]),
    )

    cf.register_option(
        "format.na_rep",
        None,
        styler_na_rep,
        validator=is_instance_factory([type(None), str]),
    )

    cf.register_option(
        "format.escape",
        None,
        styler_escape,
        validator=is_one_of_factory([None, "html", "latex", "latex-math"]),
    )

    cf.register_option(
        "format.formatter",
        None,
        styler_formatter,
        validator=is_instance_factory([type(None), dict, Callable, str]),
    )

    cf.register_option("html.mathjax", True, styler_mathjax, validator=is_bool)

    cf.register_option(
        "latex.multirow_align",
        "c",
        styler_multirow_align,
        validator=is_one_of_factory(["c", "t", "b", "naive"]),
    )

    val_mca = ["r", "|r|", "|r", "r|", "c", "|c|", "|c", "c|", "l", "|l|", "|l", "l|"]
    val_mca += ["naive-l", "naive-r"]
    cf.register_option(
        "latex.multicol_align",
        "r",
        styler_multicol_align,
        validator=is_one_of_factory(val_mca),
    )

    cf.register_option("latex.hrules", False, styler_hrules, validator=is_bool)

    cf.register_option(
        "latex.environment",
        None,
        styler_environment,
        validator=is_instance_factory([type(None), str]),
    )


with cf.config_prefix("future"):
    cf.register_option(
        "infer_string",
        True if os.environ.get("PANDAS_FUTURE_INFER_STRING", "0") == "1" else False,
        "Whether to infer sequence of str objects as pyarrow string "
        "dtype, which will be the default in pandas 3.0 "
        "(at which point this option will be deprecated).",
        validator=is_one_of_factory([True, False]),
    )

    cf.register_option(
        "no_silent_downcasting",
        False,
        "Whether to opt-in to the future behavior which will *not* silently "
        "downcast results from Series and DataFrame `where`, `mask`, and `clip` "
        "methods. "
        "Silent downcasting will be removed in pandas 3.0 "
        "(at which point this option will be deprecated).",
        validator=is_one_of_factory([True, False]),
    )
<|endoftext|>from __future__ import annotations

import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import option_context

from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
    AggFuncType,
    AggFuncTypeBase,
    AggFuncTypeDict,
    AggObjType,
    Axis,
    AxisInt,
    NDFrameT,
    npt,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_numeric_dtype,
    is_sequence,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCNDFrame,
    ABCSeries,
)

from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Hashable,
        Iterable,
        MutableMapping,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby import GroupBy
    from pandas.core.resample import Resampler
    from pandas.core.window.rolling import BaseWindow


ResType = dict[int, Any]


def frame_apply(
    obj: DataFrame,
    func: AggFuncType,
    axis: Axis = 0,
    raw: bool = False,
    result_type: str | None = None,
    by_row: Literal[False, "compat"] = "compat",
    engine: str = "python",
    engine_kwargs: dict[str, bool] | None = None,
    args=None,
    kwargs=None,
) -> FrameApply:
    """construct and return a row or column based frame apply object"""
    axis = obj._get_axis_number(axis)
    klass: type[FrameApply]
    if axis == 0:
        klass = FrameRowApply
    elif axis == 1:
        klass = FrameColumnApply

    _, func, _, _ = reconstruct_func(func, **kwargs)
    assert func is not None

    return klass(
        obj,
        func,
        raw=raw,
        result_type=result_type,
        by_row=by_row,
        engine=engine,
        engine_kwargs=engine_kwargs,
        args=args,
        kwargs=kwargs,
    )


class Apply(metaclass=abc.ABCMeta):
    axis: AxisInt

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: str | None,
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        engine: str = "python",
        engine_kwargs: dict[str, bool] | None = None,
        args,
        kwargs,
    ) -> None:
        self.obj = obj
        self.raw = raw

        assert by_row is False or by_row in ["compat", "_compat"]
        self.by_row = by_row

        self.args = args or ()
        self.kwargs = kwargs or {}

        self.engine = engine
        self.engine_kwargs = {} if engine_kwargs is None else engine_kwargs

        if result_type not in [None, "reduce", "broadcast", "expand"]:
            raise ValueError(
                "invalid value for result_type, must be one "
                "of {None, 'reduce', 'broadcast', 'expand'}"
            )

        self.result_type = result_type

        self.func = func

    @abc.abstractmethod
    def apply(self) -> DataFrame | Series:
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        pass

    def agg(self) -> DataFrame | Series | None:
        """
        Provide an implementation for the aggregators.

        Returns
        -------
        Result of aggregation, or None if agg cannot be performed by
        this method.
        """
        obj = self.obj
        func = self.func
        args = self.args
        kwargs = self.kwargs

        if isinstance(func, str):
            return self.apply_str()

        if is_dict_like(func):
            return self.agg_dict_like()
        elif is_list_like(func):
            # we require a list, but not a 'str'
            return self.agg_list_like()

        if callable(func):
            f = com.get_cython_func(func)
            if f and not args and not kwargs:
                warn_alias_replacement(obj, func, f)
                return getattr(obj, f)()

        # caller can react
        return None

    def transform(self) -> DataFrame | Series:
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
        obj = self.obj
        func = self.func
        axis = self.axis
        args = self.args
        kwargs = self.kwargs

        is_series = obj.ndim == 1

        if obj._get_axis_number(axis) == 1:
            assert not is_series
            return obj.T.transform(func, 0, *args, **kwargs).T

        if is_list_like(func) and not is_dict_like(func):
            func = cast(list[AggFuncTypeBase], func)
            # Convert func equivalent dict
            if is_series:
                func = {com.get_callable_name(v) or v: v for v in func}
            else:
                func = {col: func for col in obj}

        if is_dict_like(func):
            func = cast(AggFuncTypeDict, func)
            return self.transform_dict_like(func)

        # func is either str or callable
        func = cast(AggFuncTypeBase, func)
        try:
            result = self.transform_str_or_callable(func)
        except TypeError:
            raise
        except Exception as err:
            raise ValueError("Transform function failed") from err

        # Functions that transform may return empty Series/DataFrame
        # when the dtype is not appropriate
        if (
            isinstance(result, (ABCSeries, ABCDataFrame))
            and result.empty
            and not obj.empty
        ):
            raise ValueError("Transform function failed")
        # error: Argument 1 to "__get__" of "AxisProperty" has incompatible type
        # "Union[Series, DataFrame, GroupBy[Any], SeriesGroupBy,
        # DataFrameGroupBy, BaseWindow, Resampler]"; expected "Union[DataFrame,
        # Series]"
        if not isinstance(result, (ABCSeries, ABCDataFrame)) or not result.index.equals(
            obj.index  # type: ignore[arg-type]
        ):
            raise ValueError("Function did not transform")

        return result

    def transform_dict_like(self, func) -> DataFrame:
        """
        Compute transform in the case of a dict-like func
        """
        from pandas.core.reshape.concat import concat

        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        # transform is currently only for Series/DataFrame
        assert isinstance(obj, ABCNDFrame)

        if len(func) == 0:
            raise ValueError("No transform functions were provided")

        func = self.normalize_dictlike_arg("transform", obj, func)

        results: dict[Hashable, DataFrame | Series] = {}
        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)
            results[name] = colg.transform(how, 0, *args, **kwargs)
        return concat(results, axis=1)

    def transform_str_or_callable(self, func) -> DataFrame | Series:
        """
        Compute transform in the case of a string or callable func
        """
        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        if isinstance(func, str):
            return self._apply_str(obj, func, *args, **kwargs)

        if not args and not kwargs:
            f = com.get_cython_func(func)
            if f:
                warn_alias_replacement(obj, func, f)
                return getattr(obj, f)()

        # Two possible ways to use a UDF - apply or call directly
        try:
            return obj.apply(func, args=args, **kwargs)
        except Exception:
            return func(obj, *args, **kwargs)

    def agg_list_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_list_like(op_name="agg")

    def compute_list_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable] | Index, list[Any]]:
        """
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        """
        func = cast(list[AggFuncTypeBase], self.func)
        obj = self.obj

        results = []
        keys = []

        # degenerate case
        if selected_obj.ndim == 1:
            for a in func:
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)

                # make sure we find a good name
                name = com.get_callable_name(a) or a
                keys.append(name)

        else:
            indices = []
            for index, col in enumerate(selected_obj):
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            # error: Incompatible types in assignment (expression has type "Any |
            # Index", variable has type "list[Any | Callable[..., Any] | str]")
            keys = selected_obj.columns.take(indices)  # type: ignore[assignment]

        return keys, results

    def wrap_results_list_like(
        self, keys: Iterable[Hashable], results: list[Series | DataFrame]
    ):
        from pandas.core.reshape.concat import concat

        obj = self.obj

        try:
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError as err:
            # we are concatting non-NDFrame objects,
            # e.g. a list of scalars
            from pandas import Series

            result = Series(results, index=keys, name=obj.name)
            if is_nested_object(result):
                raise ValueError(
                    "cannot combine transform and aggregation operations"
                ) from err
            return result

    def agg_dict_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_dict_like(op_name="agg")

    def compute_dict_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        selection: Hashable | Sequence[Hashable],
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable], list[Any]]:
        """
        Compute agg/apply results for dict-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        selection : hashable or sequence of hashables
            Used by GroupBy, Window, and Resample if selection is applied to the object.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[hashable]
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python object.
        """
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        obj = self.obj
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        func = cast(AggFuncTypeDict, self.func)
        func = self.normalize_dictlike_arg(op_name, selected_obj, func)

        is_non_unique_col = (
            selected_obj.ndim == 2
            and selected_obj.columns.nunique() < len(selected_obj.columns)
        )

        if selected_obj.ndim == 1:
            # key only used for output
            colg = obj._gotitem(selection, ndim=1)
            results = [getattr(colg, op_name)(how, **kwargs) for _, how in func.items()]
            keys = list(func.keys())
        elif not is_groupby and is_non_unique_col:
            # key used for column selection and output
            # GH#51099
            results = []
            keys = []
            for key, how in func.items():
                indices = selected_obj.columns.get_indexer_for([key])
                labels = selected_obj.columns.take(indices)
                label_to_indices = defaultdict(list)
                for index, label in zip(indices, labels):
                    label_to_indices[label].append(index)

                key_data = [
                    getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs)
                    for label, indices in label_to_indices.items()
                    for indice in indices
                ]

                keys += [key] * len(key_data)
                results += key_data
        else:
            # key used for column selection and output
            results = [
                getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
                for key, how in func.items()
            ]
            keys = list(func.keys())

        return keys, results

    def wrap_results_dict_like(
        self,
        selected_obj: Series | DataFrame,
        result_index: list[Hashable],
        result_data: list,
    ):
        from pandas import Index
        from pandas.core.reshape.concat import concat

        obj = self.obj

        # Avoid making two isinstance calls in all and any below
        is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]

        if all(is_ndframe):
            results = dict(zip(result_index, result_data))
            keys_to_use: Iterable[Hashable]
            keys_to_use = [k for k in result_index if not results[k].empty]
            # Have to check, if at least one DataFrame is not empty.
            keys_to_use = keys_to_use if keys_to_use != [] else result_index
            if selected_obj.ndim == 2:
                # keys are columns, so we can preserve names
                ktu = Index(keys_to_use)
                ktu._set_names(selected_obj.columns.names)
                keys_to_use = ktu

            axis: AxisInt = 0 if isinstance(obj, ABCSeries) else 1
            result = concat(
                {k: results[k] for k in keys_to_use},
                axis=axis,
                keys=keys_to_use,
            )
        elif any(is_ndframe):
            # There is a mix of NDFrames and scalars
            raise ValueError(
                "cannot perform both aggregation "
                "and transformation operations "
                "simultaneously"
            )
        else:
            from pandas import Series

            # we have a list of scalars
            # GH 36212 use name only if obj is a series
            if obj.ndim == 1:
                obj = cast("Series", obj)
                name = obj.name
            else:
                name = None

            result = Series(result_data, index=result_index, name=name)

        return result

    def apply_str(self) -> DataFrame | Series:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
        # Caller is responsible for checking isinstance(self.f, str)
        func = cast(str, self.func)

        obj = self.obj

        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        # Support for `frame.transform('method')`
        # Some methods (shift, etc.) require the axis argument, others
        # don't, so inspect and insert if necessary.
        method = getattr(obj, func, None)
        if callable(method):
            sig = inspect.getfullargspec(method)
            arg_names = (*sig.args, *sig.kwonlyargs)
            if self.axis != 0 and (
                "axis" not in arg_names or func in ("corrwith", "skew")
            ):
                raise ValueError(f"Operation {func} does not support axis=1")
            if "axis" in arg_names:
                if isinstance(obj, (SeriesGroupBy, DataFrameGroupBy)):
                    # Try to avoid FutureWarning for deprecated axis keyword;
                    # If self.axis matches the axis we would get by not passing
                    #  axis, we safely exclude the keyword.

                    default_axis = 0
                    if func in ["idxmax", "idxmin"]:
                        # DataFrameGroupBy.idxmax, idxmin axis defaults to self.axis,
                        # whereas other axis keywords default to 0
                        default_axis = self.obj.axis

                    if default_axis != self.axis:
                        self.kwargs["axis"] = self.axis
                else:
                    self.kwargs["axis"] = self.axis
        return self._apply_str(obj, func, *self.args, **self.kwargs)

    def apply_list_or_dict_like(self) -> DataFrame | Series:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """

        if self.engine == "numba":
            raise NotImplementedError(
                "The 'numba' engine doesn't support list-like/"
                "dict likes of callables yet."
            )

        if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
            return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T

        func = self.func
        kwargs = self.kwargs

        if is_dict_like(func):
            result = self.agg_or_apply_dict_like(op_name="apply")
        else:
            result = self.agg_or_apply_list_like(op_name="apply")

        result = reconstruct_and_relabel_result(result, func, **kwargs)

        return result

    def normalize_dictlike_arg(
        self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict
    ) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ("apply", "agg", "transform")

        # Can't use func.values(); wouldn't work for a Series
        if (
            how == "agg"
            and isinstance(obj, ABCSeries)
            and any(is_list_like(v) for _, v in func.items())
        ) or (any(is_dict_like(v) for _, v in func.items())):
            # GH 15931 - deprecation of renaming keys
            raise SpecificationError("nested renamer is not supported")

        if obj.ndim != 1:
            # Check for missing columns on a frame
            from pandas import Index

            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                raise KeyError(f"Column(s) {list(cols)} do not exist")

        aggregator_types = (list, tuple, dict)

        # if we have a dict of any non-scalars
        # eg. {'A' : ['mean']}, normalize all to
        # be list-likes
        # Cannot use func.values() because arg may be a Series
        if any(isinstance(x, aggregator_types) for _, x in func.items()):
            new_func: AggFuncTypeDict = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func
        return func

    def _apply_str(self, obj, func: str, *args, **kwargs):
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """
        assert isinstance(func, str)

        if hasattr(obj, func):
            f = getattr(obj, func)
            if callable(f):
                return f(*args, **kwargs)

            # people may aggregate on a non-callable attribute
            # but don't let them think they can pass args to it
            assert len(args) == 0
            assert len([kwarg for kwarg in kwargs if kwarg not in ["axis"]]) == 0
            return f
        elif hasattr(np, func) and hasattr(obj, "__array__"):
            # in particular exclude Window
            f = getattr(np, func)
            return f(obj, *args, **kwargs)
        else:
            msg = f"'{func}' is not a valid function for '{type(obj).__name__}' object"
            raise AttributeError(msg)


class NDFrameApply(Apply):
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """

    obj: DataFrame | Series

    @property
    def index(self) -> Index:
        return self.obj.index

    @property
    def agg_axis(self) -> Index:
        return self.obj._get_agg_axis(self.axis)

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs

        if op_name == "apply":
            if isinstance(self, FrameApply):
                by_row = self.by_row

            elif isinstance(self, SeriesApply):
                by_row = "_compat" if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, "by_row": by_row}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        keys, results = self.compute_list_like(op_name, obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        assert op_name in ["agg", "apply"]
        obj = self.obj

        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selection = None
        result_index, result_data = self.compute_dict_like(
            op_name, obj, selection, kwargs
        )
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        return result


class FrameApply(NDFrameApply):
    obj: DataFrame

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: str | None,
        *,
        by_row: Literal[False, "compat"] = False,
        engine: str = "python",
        engine_kwargs: dict[str, bool] | None = None,
        args,
        kwargs,
    ) -> None:
        if by_row is not False and by_row != "compat":
            raise ValueError(f"by_row={by_row} not allowed")
        super().__init__(
            obj,
            func,
            raw,
            result_type,
            by_row=by_row,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    # ---------------------------------------------------------------
    # Abstract Methods

    @property
    @abc.abstractmethod
    def result_index(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def series_generator(self) -> Generator[Series, None, None]:
        pass

    @staticmethod
    @functools.cache
    @abc.abstractmethod
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        pass

    @abc.abstractmethod
    def apply_with_numba(self):
        pass

    def validate_values_for_numba(self):
        # Validate column dtyps all OK
        for colname, dtype in self.obj.dtypes.items():
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Column {colname} must have a numeric dtype. "
                    f"Found '{dtype}' instead"
                )
            if is_extension_array_dtype(dtype):
                raise ValueError(
                    f"Column {colname} is backed by an extension array, "
                    f"which is not supported by the numba engine."
                )

    @abc.abstractmethod
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        pass

    # ---------------------------------------------------------------

    @property
    def res_columns(self) -> Index:
        return self.result_columns

    @property
    def columns(self) -> Index:
        return self.obj.columns

    @cache_readonly
    def values(self):
        return self.obj.values

    def apply(self) -> DataFrame | Series:
        """compute the results"""

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support lists of callables yet"
                )
            return self.apply_list_or_dict_like()

        # all empty
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()

        # string dispatch
        if isinstance(self.func, str):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support using "
                    "a string as the callable function"
                )
            return self.apply_str()

        # ufunc
        elif isinstance(self.func, np.ufunc):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support "
                    "using a numpy ufunc as the callable function"
                )
            with np.errstate(all="ignore"):
                results = self.obj._mgr.apply("apply", func=self.func)
            # _constructor will retain self.index and self.columns
            return self.obj._constructor_from_mgr(results, axes=results.axes)

        # broadcasting
        if self.result_type == "broadcast":
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support result_type='broadcast'"
                )
            return self.apply_broadcast(self.obj)

        # one axis empty
        elif not all(self.obj.shape):
            return self.apply_empty_result()

        # raw
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)

        return self.apply_standard()

    def agg(self):
        obj = self.obj
        axis = self.axis

        # TODO: Avoid having to change state
        self.obj = self.obj if self.axis == 0 else self.obj.T
        self.axis = 0

        result = None
        try:
            result = super().agg()
        finally:
            self.obj = obj
            self.axis = axis

        if axis == 1:
            result = result.T if result is not None else result

        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)

        return result

    def apply_empty_result(self):
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        assert callable(self.func)

        # we are not asked to reduce or infer reduction
        # so just return a copy of the existing object
        if self.result_type not in ["reduce", None]:
            return self.obj.copy()

        # we may need to infer
        should_reduce = self.result_type == "reduce"

        from pandas import Series

        if not should_reduce:
            try:
                if self.axis == 0:
                    r = self.func(
                        Series([], dtype=np.float64), *self.args, **self.kwargs
                    )
                else:
                    r = self.func(
                        Series(index=self.columns, dtype=np.float64),
                        *self.args,
                        **self.kwargs,
                    )
            except Exception:
                pass
            else:
                should_reduce = not isinstance(r, Series)

        if should_reduce:
            if len(self.agg_axis):
                r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
            else:
                r = np.nan

            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            return self.obj.copy()

    def apply_raw(self, engine="python", engine_kwargs=None):
        """apply to the values as a numpy array"""

        def wrap_function(func):
            """
            Wrap user supplied function to work around numpy issue.

            see https://github.com/numpy/numpy/issues/8352
            """

            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result

            return wrapper

        if engine == "numba":
            engine_kwargs = {} if engine_kwargs is None else engine_kwargs

            # error: Argument 1 to "__call__" of "_lru_cache_wrapper" has
            # incompatible type "Callable[..., Any] | str | list[Callable
            # [..., Any] | str] | dict[Hashable,Callable[..., Any] | str |
            # list[Callable[..., Any] | str]]"; expected "Hashable"
            nb_looper = generate_apply_looper(
                self.func, **engine_kwargs  # type: ignore[arg-type]
            )
            result = nb_looper(self.values, self.axis)
            # If we made the result 2-D, squeeze it back to 1-D
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(
                wrap_function(self.func),
                self.axis,
                self.values,
                *self.args,
                **self.kwargs,
            )

        # TODO: mixed type case
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        assert callable(self.func)

        result_values = np.empty_like(target.values)

        # axis which we want to compare compliance
        result_compare = target.shape[0]

        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim

            # must be a scalar or 1d
            if ares > 1:
                raise ValueError("too many dims to broadcast")
            if ares == 1:
                # must match return dim
                if result_compare != len(res):
                    raise ValueError("cannot broadcast result")

            result_values[:, i] = res

        # we *always* preserve the original index / columns
        result = self.obj._constructor(
            result_values, index=target.index, columns=target.columns
        )
        return result

    def apply_standard(self):
        if self.engine == "python":
            results, res_index = self.apply_series_generator()
        else:
            results, res_index = self.apply_series_numba()

        # wrap results
        return self.wrap_results(results, res_index)

    def apply_series_generator(self) -> tuple[ResType, Index]:
        assert callable(self.func)

        series_gen = self.series_generator
        res_index = self.result_index

        results = {}

        with option_context("mode.chained_assignment", None):
            for i, v in enumerate(series_gen):
                # ignore SettingWithCopy here in case the user mutates
                results[i] = self.func(v, *self.args, **self.kwargs)
                if isinstance(results[i], ABCSeries):
                    # If we have a view on v, we need to make a copy because
                    #  series_generator will swap out the underlying data
                    results[i] = results[i].copy(deep=False)

        return results, res_index

    def apply_series_numba(self):
        if self.engine_kwargs.get("parallel", False):
            raise NotImplementedError(
                "Parallel apply is not supported when raw=False and engine='numba'"
            )
        if not self.obj.index.is_unique or not self.columns.is_unique:
            raise NotImplementedError(
                "The index/columns must be unique when raw=False and engine='numba'"
            )
        self.validate_values_for_numba()
        results = self.apply_with_numba()
        return results, self.result_index

    def wrap_results(self, results: ResType, res_index: Index) -> DataFrame | Series:
        from pandas import Series

        # see if we can infer the results
        if len(results) > 0 and 0 in results and is_sequence(results[0]):
            return self.wrap_results_for_axis(results, res_index)

        # dict of scalars

        # the default dtype of an empty Series is `object`, but this
        # code can be hit by df.mean() where the result should have dtype
        # float64 even if it's an empty Series.
        constructor_sliced = self.obj._constructor_sliced
        if len(results) == 0 and constructor_sliced is Series:
            result = constructor_sliced(results, dtype=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index

        return result

    def apply_str(self) -> DataFrame | Series:
        # Caller is responsible for checking isinstance(self.func, str)
        # TODO: GH#39993 - Avoid special-casing by replacing with lambda
        if self.func == "size":
            # Special-cased because DataFrame.size returns a single scalar
            obj = self.obj
            value = obj.shape[self.axis]
            return obj._constructor_sliced(value, index=self.agg_axis)
        return super().apply_str()


class FrameRowApply(FrameApply):
    axis: AxisInt = 0

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        numba = import_optional_dependency("numba")
        from pandas import Series

        # Import helper from extensions to cast string object -> np strings
        # Note: This also has the side effect of loading our numba extensions
        from pandas.core._numba.extensions import maybe_cast_str

        jitted_udf = numba.extending.register_jitable(func)

        # Currently the parallel argument doesn't get passed through here
        # (it's disabled) since the dicts in numba aren't thread-safe.
        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names, df_index):
            results = {}
            for j in range(values.shape[1]):
                # Create the series
                ser = Series(
                    values[:, j], index=df_index, name=maybe_cast_str(col_names[j])
                )
                results[j] = jitted_udf(ser)
            return results

        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        nb_func = self.generate_numba_apply_func(
            cast(Callable, self.func), **self.engine_kwargs
        )
        from pandas.core._numba.extensions import set_numba_data

        index = self.obj.index
        columns = self.obj.columns

        # Convert from numba dict to regular dict
        # Our isinstance checks in the df constructor don't pass for numbas typed dict
        with set_numba_data(index) as index, set_numba_data(columns) as columns:
            res = dict(nb_func(self.values, columns, index))
        return res

    @property
    def result_index(self) -> Index:
        return self.columns

    @property
    def result_columns(self) -> Index:
        return self.index

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        """return the results for the rows"""

        if self.result_type == "reduce":
            # e.g. test_apply_dict GH#8735
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res

        elif self.result_type is None and all(
            isinstance(x, dict) for x in results.values()
        ):
            # Our operation was a to_dict op e.g.
            #  test_apply_dict GH#8735, test_apply_reduce_to_dict GH#25196 #37544
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res

        try:
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if "All arrays must be of the same length" in str(err):
                # e.g. result = [[2, 3], [1.5], ['foo', 'bar']]
                #  see test_agg_listlike_result GH#29587
                res = self.obj._constructor_sliced(results)
                res.index = res_index
                return res
            else:
                raise

        if not isinstance(results[0], ABCSeries):
            if len(result.index) == len(self.res_columns):
                result.index = self.res_columns

        if len(result.columns) == len(res_index):
            result.columns = res_index

        return result


class FrameColumnApply(FrameApply):
    axis: AxisInt = 1

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        values = self.values
        values = ensure_wrapped_if_datetimelike(values)
        assert len(values) > 0

        # We create one Series object, and will swap out the data inside
        #  of it.  Kids: don't do this at home.
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr

        is_view = mgr.blocks[0].refs.has_reference()  # type: ignore[union-attr]

        if isinstance(ser.dtype, ExtensionDtype):
            # values will be incorrect for this block
            # TODO(EA2D): special case would be unnecessary with 2D EAs
            obj = self.obj
            for i in range(len(obj)):
                yield obj._ixs(i, axis=0)

        else:
            for arr, name in zip(values, self.index):
                # GH#35462 re-pin mgr in case setitem changed it
                ser._mgr = mgr
                mgr.set_values(arr)
                object.__setattr__(ser, "_name", name)
                if not is_view:
                    # In apply_series_generator we store the a shallow copy of the
                    # result, which potentially increases the ref count of this reused
                    # `ser` object (depending on the result of the applied function)
                    # -> if that happened and `ser` is already a copy, then we reset
                    # the refs here to avoid triggering a unnecessary CoW inside the
                    # applied function (https://github.com/pandas-dev/pandas/pull/56212)
                    mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])  # type: ignore[union-attr]
                yield ser

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        numba = import_optional_dependency("numba")
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str

        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names_index, index):
            results = {}
            # Currently the parallel argument doesn't get passed through here
            # (it's disabled) since the dicts in numba aren't thread-safe.
            for i in range(values.shape[0]):
                # Create the series
                # TODO: values corrupted without the copy
                ser = Series(
                    values[i].copy(),
                    index=col_names_index,
                    name=maybe_cast_str(index[i]),
                )
                results[i] = jitted_udf(ser)

            return results

        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        nb_func = self.generate_numba_apply_func(
            cast(Callable, self.func), **self.engine_kwargs
        )

        from pandas.core._numba.extensions import set_numba_data

        # Convert from numba dict to regular dict
        # Our isinstance checks in the df constructor don't pass for numbas typed dict
        with set_numba_data(self.obj.index) as index, set_numba_data(
            self.columns
        ) as columns:
            res = dict(nb_func(self.values, columns, index))

        return res

    @property
    def result_index(self) -> Index:
        return self.index

    @property
    def result_columns(self) -> Index:
        return self.columns

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        """return the results for the columns"""
        result: DataFrame | Series

        # we have requested to expand
        if self.result_type == "expand":
            result = self.infer_to_same_shape(results, res_index)

        # we have a non-series and don't want inference
        elif not isinstance(results[0], ABCSeries):
            result = self.obj._constructor_sliced(results)
            result.index = res_index

        # we may want to infer results
        else:
            result = self.infer_to_same_shape(results, res_index)

        return result

    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        """infer the results to the same shape as the input object"""
        result = self.obj._constructor(data=results)
        result = result.T

        # set the index
        result.index = res_index

        # infer dtypes
        result = result.infer_objects(copy=False)

        return result


class SeriesApply(NDFrameApply):
    obj: Series
    axis: AxisInt = 0
    by_row: Literal[False, "compat", "_compat"]  # only relevant for apply()

    def __init__(
        self,
        obj: Series,
        func: AggFuncType,
        *,
        convert_dtype: bool | lib.NoDefault = lib.no_default,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        args,
        kwargs,
    ) -> None:
        if convert_dtype is lib.no_default:
            convert_dtype = True
        else:
            warnings.warn(
                "the convert_dtype parameter is deprecated and will be removed in a "
                "future version.  Do ``ser.astype(object).apply()`` "
                "instead if you want ``convert_dtype=False``.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        self.convert_dtype = convert_dtype

        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        )

    def apply(self) -> DataFrame | Series:
        obj = self.obj

        if len(obj) == 0:
            return self.apply_empty_result()

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()

        if isinstance(self.func, str):
            # if we are a string, try to dispatch
            return self.apply_str()

        if self.by_row == "_compat":
            return self.apply_compat()

        # self.func is Callable
        return self.apply_standard()

    def agg(self):
        result = super().agg()
        if result is None:
            obj = self.obj
            func = self.func
            # string, list-like, and dict-like are entirely handled in super
            assert callable(func)

            # GH53325: The setup below is just to keep current behavior while emitting a
            # deprecation message. In the future this will all be replaced with a simple
            # `result = f(self.obj, *self.args, **self.kwargs)`.
            try:
                result = obj.apply(func, args=self.args, **self.kwargs)
            except (ValueError, AttributeError, TypeError):
                result = func(obj, *self.args, **self.kwargs)
            else:
                msg = (
                    f"using {func} in {type(obj).__name__}.agg cannot aggregate and "
                    f"has been deprecated. Use {type(obj).__name__}.transform to "
                    f"keep behavior unchanged."
                )
                warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())

        return result

    def apply_empty_result(self) -> Series:
        obj = self.obj
        return obj._constructor(dtype=obj.dtype, index=obj.index).__finalize__(
            obj, method="apply"
        )

    def apply_compat(self):
        """compat apply method for funcs in listlikes and dictlikes.

         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.

        .. versionadded:: 2.1.0
        """
        obj = self.obj
        func = self.func

        if callable(func):
            f = com.get_cython_func(func)
            if f and not self.args and not self.kwargs:
                return obj.apply(func, by_row=False)

        try:
            result = obj.apply(func, by_row="compat")
        except (ValueError, AttributeError, TypeError):
            result = obj.apply(func, by_row=False)
        return result

    def apply_standard(self) -> DataFrame | Series:
        # caller is responsible for ensuring that f is Callable
        func = cast(Callable, self.func)
        obj = self.obj

        if isinstance(func, np.ufunc):
            with np.errstate(all="ignore"):
                return func(obj, *self.args, **self.kwargs)
        elif not self.by_row:
            return func(obj, *self.args, **self.kwargs)

        if self.args or self.kwargs:
            # _map_values does not support args/kwargs
            def curried(x):
                return func(x, *self.args, **self.kwargs)

        else:
            curried = func

        # row-wise access
        # apply doesn't have a `na_action` keyword and for backward compat reasons
        # we need to give `na_action="ignore"` for categorical data.
        # TODO: remove the `na_action="ignore"` when that default has been changed in
        #  Categorical (GH51645).
        action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
        mapped = obj._map_values(
            mapper=curried, na_action=action, convert=self.convert_dtype
        )

        if len(mapped) and isinstance(mapped[0], ABCSeries):
            # GH#43986 Need to do list(mapped) in order to get treated as nested
            #  See also GH#25959 regarding EA support
            return obj._constructor_expanddim(list(mapped), index=obj.index)
        else:
            return obj._constructor(mapped, index=obj.index).__finalize__(
                obj, method="apply"
            )


class GroupByApply(Apply):
    obj: GroupBy | Resampler | BaseWindow

    def __init__(
        self,
        obj: GroupBy[NDFrameT],
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        kwargs = kwargs.copy()
        self.axis = obj.obj._get_axis_number(kwargs.get("axis", 0))
        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    def apply(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs
        if op_name == "apply":
            kwargs = {**kwargs, "by_row": False}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        if obj._selected_obj.ndim == 1:
            # For SeriesGroupBy this matches _obj_with_exclusions
            selected_obj = obj._selected_obj
        else:
            selected_obj = obj._obj_with_exclusions

        # Only set as_index=True on groupby objects, not Window or Resample
        # that inherit from this class.
        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        assert op_name in ["agg", "apply"]

        obj = self.obj
        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selected_obj = obj._selected_obj
        selection = obj._selection

        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))

        # Numba Groupby engine/engine-kwargs passthrough
        if is_groupby:
            engine = self.kwargs.get("engine", None)
            engine_kwargs = self.kwargs.get("engine_kwargs", None)
            kwargs.update({"engine": engine, "engine_kwargs": engine_kwargs})

        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            result_index, result_data = self.compute_dict_like(
                op_name, selected_obj, selection, kwargs
            )
        result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
        return result


class ResamplerWindowApply(GroupByApply):
    axis: AxisInt = 0
    obj: Resampler | BaseWindow

    def __init__(
        self,
        obj: Resampler | BaseWindow,
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        super(GroupByApply, self).__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    def apply(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType, tuple[str, ...] | None, npt.NDArray[np.intp] | None]:
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class 'list'>, {'col': ['min']}), ('foo',), array([0]))

    >>> reconstruct_func("min")
    (False, 'min', None, None)
    """
    relabeling = func is None and is_multi_agg_with_relabel(**kwargs)
    columns: tuple[str, ...] | None = None
    order: npt.NDArray[np.intp] | None = None

    if not relabeling:
        if isinstance(func, list) and len(func) > len(set(func)):
            # GH 28426 will raise error if duplicated function names are used and
            # there is no reassigned name
            raise SpecificationError(
                "Function names must be unique if there is no new column names "
                "assigned"
            )
        if func is None:
            # nicer error message
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")

    if relabeling:
        # error: Incompatible types in assignment (expression has type
        # "MutableMapping[Hashable, list[Callable[..., Any] | str]]", variable has type
        # "Callable[..., Any] | str | list[Callable[..., Any] | str] |
        # MutableMapping[Hashable, Callable[..., Any] | str | list[Callable[..., Any] |
        # str]] | None")
        func, columns, order = normalize_keyword_aggregation(  # type: ignore[assignment]
            kwargs
        )
    assert func is not None

    return relabeling, func, columns, order


def is_multi_agg_with_relabel(**kwargs) -> bool:
    """
    Check whether kwargs passed to .agg look like multi-agg with relabeling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool

    Examples
    --------
    >>> is_multi_agg_with_relabel(a="max")
    False
    >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))
    True
    >>> is_multi_agg_with_relabel()
    False
    """
    return all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values()) and (
        len(kwargs) > 0
    )


def normalize_keyword_aggregation(
    kwargs: dict,
) -> tuple[
    MutableMapping[Hashable, list[AggFuncTypeBase]],
    tuple[str, ...],
    npt.NDArray[np.intp],
]:
    """
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : tuple[str, ...]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.

    Examples
    --------
    >>> normalize_keyword_aggregation({"output": ("input", "sum")})
    (defaultdict(<class 'list'>, {'input': ['sum']}), ('output',), array([0]))
    """
    from pandas.core.indexes.base import Index

    # Normalize the aggregation functions as Mapping[column, List[func]],
    # process normally, then fixup the names.
    # TODO: aggspec type: typing.Dict[str, List[AggScalar]]
    aggspec = defaultdict(list)
    order = []
    columns, pairs = list(zip(*kwargs.items()))

    for column, aggfunc in pairs:
        aggspec[column].append(aggfunc)
        order.append((column, com.get_callable_name(aggfunc) or aggfunc))

    # uniquify aggfunc name if duplicated in order list
    uniquified_order = _make_unique_kwarg_list(order)

    # GH 25719, due to aggspec will change the order of assigned columns in aggregation
    # uniquified_aggspec will store uniquified order list and will compare it with order
    # based on index
    aggspec_order = [
        (column, com.get_callable_name(aggfunc) or aggfunc)
        for column, aggfuncs in aggspec.items()
        for aggfunc in aggfuncs
    ]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)

    # get the new index of columns by comparison
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return aggspec, columns, col_idx_order


def _make_unique_kwarg_list(
    seq: Sequence[tuple[Any, Any]]
) -> Sequence[tuple[Any, Any]]:
    """
    Uniquify aggfunc name of the pairs in the order list

    Examples:
    --------
    >>> kwarg_list = [('a', '<lambda>'), ('a', '<lambda>'), ('b', '<lambda>')]
    >>> _make_unique_kwarg_list(kwarg_list)
    [('a', '<lambda>_0'), ('a', '<lambda>_1'), ('b', '<lambda>')]
    """
    return [
        (pair[0], f"{pair[1]}_{seq[:i].count(pair)}") if seq.count(pair) > 1 else pair
        for i, pair in enumerate(seq)
    ]


def relabel_result(
    result: DataFrame | Series,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]:
    """
    Internal function to reorder result if relabelling is True for
    dataframe.agg, and return the reordered result in dict.

    Parameters:
    ----------
    result: Result from aggregation
    func: Dict of (column name, funcs)
    columns: New columns name for relabelling
    order: New order for relabelling

    Examples
    --------
    >>> from pandas.core.apply import relabel_result
    >>> result = pd.DataFrame(
    ...     {"A": [np.nan, 2, np.nan], "C": [6, np.nan, np.nan], "B": [np.nan, 4, 2.5]},
    ...     index=["max", "mean", "min"]
    ... )
    >>> funcs = {"A": ["max"], "C": ["max"], "B": ["mean", "min"]}
    >>> columns = ("foo", "aab", "bar", "dat")
    >>> order = [0, 1, 2, 3]
    >>> result_in_dict = relabel_result(result, funcs, columns, order)
    >>> pd.DataFrame(result_in_dict, index=columns)
           A    C    B
    foo  2.0  NaN  NaN
    aab  NaN  6.0  NaN
    bar  NaN  NaN  4.0
    dat  NaN  NaN  2.5
    """
    from pandas.core.indexes.base import Index

    reordered_indexes = [
        pair[0] for pair in sorted(zip(columns, order), key=lambda t: t[1])
    ]
    reordered_result_in_dict: dict[Hashable, Series] = {}
    idx = 0

    reorder_mask = not isinstance(result, ABCSeries) and len(result.columns) > 1
    for col, fun in func.items():
        s = result[col].dropna()

        # In the `_aggregate`, the callable names are obtained and used in `result`, and
        # these names are ordered alphabetically. e.g.
        #           C2   C1
        # <lambda>   1  NaN
        # amax     NaN  4.0
        # max      NaN  4.0
        # sum     18.0  6.0
        # Therefore, the order of functions for each column could be shuffled
        # accordingly so need to get the callable name if it is not parsed names, and
        # reorder the aggregated result for each column.
        # e.g. if df.agg(c1=("C2", sum), c2=("C2", lambda x: min(x))), correct order is
        # [sum, <lambda>], but in `result`, it will be [<lambda>, sum], and we need to
        # reorder so that aggregated values map to their functions regarding the order.

        # However there is only one column being used for aggregation, not need to
        # reorder since the index is not sorted, and keep as is in `funcs`, e.g.
        #         A
        # min   1.0
        # mean  1.5
        # mean  1.5
        if reorder_mask:
            fun = [
                com.get_callable_name(f) if not isinstance(f, str) else f for f in fun
            ]
            col_idx_order = Index(s.index).get_indexer(fun)
            s = s.iloc[col_idx_order]

        # assign the new user-provided "named aggregation" as index names, and reindex
        # it based on the whole user-provided names.
        s.index = reordered_indexes[idx : idx + len(fun)]
        reordered_result_in_dict[col] = s.reindex(columns, copy=False)
        idx = idx + len(fun)
    return reordered_result_in_dict


def reconstruct_and_relabel_result(result, func, **kwargs) -> DataFrame | Series:
    from pandas import DataFrame

    relabeling, func, columns, order = reconstruct_func(func, **kwargs)

    if relabeling:
        # This is to keep the order to columns occurrence unchanged, and also
        # keep the order of new columns occurrence unchanged

        # For the return values of reconstruct_func, if relabeling is
        # False, columns and order will be None.
        assert columns is not None
        assert order is not None

        result_in_dict = relabel_result(result, func, columns, order)
        result = DataFrame(result_in_dict, index=columns)

    return result


# TODO: Can't use, because mypy doesn't like us setting __name__
#   error: "partial[Any]" has no attribute "__name__"
# the type is:
#   typing.Sequence[Callable[..., ScalarResult]]
#     -> typing.Sequence[Callable[..., ScalarResult]]:


def _managle_lambda_list(aggfuncs: Sequence[Any]) -> Sequence[Any]:
    """
    Possibly mangle a list of aggfuncs.

    Parameters
    ----------
    aggfuncs : Sequence

    Returns
    -------
    mangled: list-like
        A new AggSpec sequence, where lambdas have been converted
        to have unique names.

    Notes
    -----
    If just one aggfunc is passed, the name will not be mangled.
    """
    if len(aggfuncs) <= 1:
        # don't mangle for .agg([lambda x: .])
        return aggfuncs
    i = 0
    mangled_aggfuncs = []
    for aggfunc in aggfuncs:
        if com.get_callable_name(aggfunc) == "<lambda>":
            aggfunc = partial(aggfunc)
            aggfunc.__name__ = f"<lambda_{i}>"
            i += 1
        mangled_aggfuncs.append(aggfunc)

    return mangled_aggfuncs


def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    """
    Make new lambdas with unique names.

    Parameters
    ----------
    agg_spec : Any
        An argument to GroupBy.agg.
        Non-dict-like `agg_spec` are pass through as is.
