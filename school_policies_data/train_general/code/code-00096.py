# -----------------------------------------------------------
# Indexer Validation


def check_setitem_lengths(indexer, value, values) -> bool:
    """
    Validate that value and indexer are the same length.

    An special-case is allowed for when the indexer is a boolean array
    and the number of true values equals the length of ``value``. In
    this case, no exception is raised.

    Parameters
    ----------
    indexer : sequence
        Key for the setitem.
    value : array-like
        Value for the setitem.
    values : array-like
        Values being set into.

    Returns
    -------
    bool
        Whether this is an empty listlike setting which is a no-op.

    Raises
    ------
    ValueError
        When the indexer is an ndarray or list and the lengths don't match.
    """
    no_op = False

    if isinstance(indexer, (np.ndarray, list)):
        # We can ignore other listlikes because they are either
        #  a) not necessarily 1-D indexers, e.g. tuple
        #  b) boolean indexers e.g. BoolArray
        if is_list_like(value):
            if len(indexer) != len(value) and values.ndim == 1:
                # boolean with truth values == len of the value is ok too
                if isinstance(indexer, list):
                    indexer = np.array(indexer)
                if not (
                    isinstance(indexer, np.ndarray)
                    and indexer.dtype == np.bool_
                    and indexer.sum() == len(value)
                ):
                    raise ValueError(
                        "cannot set using a list-like indexer "
                        "with a different length than the value"
                    )
            if not len(indexer):
                no_op = True

    elif isinstance(indexer, slice):
        if is_list_like(value):
            if len(value) != length_of_indexer(indexer, values) and values.ndim == 1:
                # In case of two dimensional value is used row-wise and broadcasted
                raise ValueError(
                    "cannot set using a slice indexer with a "
                    "different length than the value"
                )
            if not len(value):
                no_op = True

    return no_op


def validate_indices(indices: np.ndarray, n: int) -> None:
    """
    Perform bounds-checking for an indexer.

    -1 is allowed for indicating missing values.

    Parameters
    ----------
    indices : ndarray
    n : int
        Length of the array being indexed.

    Raises
    ------
    ValueError

    Examples
    --------
    >>> validate_indices(np.array([1, 2]), 3) # OK

    >>> validate_indices(np.array([1, -2]), 3)
    Traceback (most recent call last):
        ...
    ValueError: negative dimensions are not allowed

    >>> validate_indices(np.array([1, 2, 3]), 3)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds

    >>> validate_indices(np.array([-1, -1]), 0) # OK

    >>> validate_indices(np.array([0, 1]), 0)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds
    """
    if len(indices):
        min_idx = indices.min()
        if min_idx < -1:
            msg = f"'indices' contains values less than allowed ({min_idx} < -1)"
            raise ValueError(msg)

        max_idx = indices.max()
        if max_idx >= n:
            raise IndexError("indices are out-of-bounds")


# -----------------------------------------------------------
# Indexer Conversion


def maybe_convert_indices(indices, n: int, verify: bool = True) -> np.ndarray:
    """
    Attempt to convert indices into valid, positive indices.

    If we have negative indices, translate to positive here.
    If we have indices that are out-of-bounds, raise an IndexError.

    Parameters
    ----------
    indices : array-like
        Array of indices that we are to convert.
    n : int
        Number of elements in the array that we are indexing.
    verify : bool, default True
        Check that all entries are between 0 and n - 1, inclusive.

    Returns
    -------
    array-like
        An array-like of positive indices that correspond to the ones
        that were passed in initially to this function.

    Raises
    ------
    IndexError
        One of the converted indices either exceeded the number of,
        elements (specified by `n`), or was still negative.
    """
    if isinstance(indices, list):
        indices = np.array(indices)
        if len(indices) == 0:
            # If `indices` is empty, np.array will return a float,
            # and will cause indexing errors.
            return np.empty(0, dtype=np.intp)

    mask = indices < 0
    if mask.any():
        indices = indices.copy()
        indices[mask] += n

    if verify:
        mask = (indices >= n) | (indices < 0)
        if mask.any():
            raise IndexError("indices are out-of-bounds")
    return indices


# -----------------------------------------------------------
# Unsorted


def length_of_indexer(indexer, target=None) -> int:
    """
    Return the expected length of target[indexer]

    Returns
    -------
    int
    """
    if target is not None and isinstance(indexer, slice):
        target_len = len(target)
        start = indexer.start
        stop = indexer.stop
        step = indexer.step
        if start is None:
            start = 0
        elif start < 0:
            start += target_len
        if stop is None or stop > target_len:
            stop = target_len
        elif stop < 0:
            stop += target_len
        if step is None:
            step = 1
        elif step < 0:
            start, stop = stop + 1, start + 1
            step = -step
        return (stop - start + step - 1) // step
    elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
        if isinstance(indexer, list):
            indexer = np.array(indexer)

        if indexer.dtype == bool:
            # GH#25774
            return indexer.sum()
        return len(indexer)
    elif isinstance(indexer, range):
        return (indexer.stop - indexer.start) // indexer.step
    elif not is_list_like_indexer(indexer):
        return 1
    raise AssertionError("cannot find the length of the indexer")


def disallow_ndim_indexing(result) -> None:
    """
    Helper function to disallow multi-dimensional indexing on 1D Series/Index.

    GH#27125 indexer like idx[:, None] expands dim, but we cannot do that
    and keep an index, so we used to return ndarray, which was deprecated
    in GH#30588.
    """
    if np.ndim(result) > 1:
        raise ValueError(
            "Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer "
            "supported. Convert to a numpy array before indexing instead."
        )


def unpack_1tuple(tup):
    """
    If we have a length-1 tuple/list that contains a slice, unpack to just
    the slice.

    Notes
    -----
    The list case is deprecated.
    """
    if len(tup) == 1 and isinstance(tup[0], slice):
        # if we don't have a MultiIndex, we may still be able to handle
        #  a 1-tuple.  see test_1tuple_without_multiindex

        if isinstance(tup, list):
            # GH#31299
            raise ValueError(
                "Indexing with a single-item list containing a "
                "slice is not allowed. Pass a tuple instead.",
            )

        return tup[0]
    return tup


def check_key_length(columns: Index, key, value: DataFrame) -> None:
    """
    Checks if a key used as indexer has the same length as the columns it is
    associated with.

    Parameters
    ----------
    columns : Index The columns of the DataFrame to index.
    key : A list-like of keys to index with.
    value : DataFrame The value to set for the keys.

    Raises
    ------
    ValueError: If the length of key is not equal to the number of columns in value
                or if the number of columns referenced by key is not equal to number
                of columns.
    """
    if columns.is_unique:
        if len(value.columns) != len(key):
            raise ValueError("Columns must be same length as key")
    else:
        # Missing keys in columns are represented as -1
        if len(columns.get_indexer_non_unique(key)[0]) != len(value.columns):
            raise ValueError("Columns must be same length as key")


def unpack_tuple_and_ellipses(item: tuple):
    """
    Possibly unpack arr[..., n] to arr[n]
    """
    if len(item) > 1:
        # Note: we are assuming this indexing is being done on a 1D arraylike
        if item[0] is Ellipsis:
            item = item[1:]
        elif item[-1] is Ellipsis:
            item = item[:-1]

    if len(item) > 1:
        raise IndexError("too many indices for array.")

    item = item[0]
    return item


# -----------------------------------------------------------
# Public indexer validation


def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
    """
    Check if `indexer` is a valid array indexer for `array`.

    For a boolean mask, `array` and `indexer` are checked to have the same
    length. The dtype is validated, and if it is an integer or boolean
    ExtensionArray, it is checked if there are missing values present, and
    it is converted to the appropriate numpy array. Other dtypes will raise
    an error.

    Non-array indexers (integer, slice, Ellipsis, tuples, ..) are passed
    through as is.

    Parameters
    ----------
    array : array-like
        The array that is being indexed (only used for the length).
    indexer : array-like or list-like
        The array-like that's used to index. List-like input that is not yet
        a numpy array or an ExtensionArray is converted to one. Other input
        types are passed through as is.

    Returns
    -------
    numpy.ndarray
        The validated indexer as a numpy array that can be used to index.

    Raises
    ------
    IndexError
        When the lengths don't match.
    ValueError
        When `indexer` cannot be converted to a numpy ndarray to index
        (e.g. presence of missing values).

    See Also
    --------
    api.types.is_bool_dtype : Check if `key` is of boolean dtype.

    Examples
    --------
    When checking a boolean mask, a boolean ndarray is returned when the
    arguments are all valid.

    >>> mask = pd.array([True, False])
    >>> arr = pd.array([1, 2])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    An IndexError is raised when the lengths don't match.

    >>> mask = pd.array([True, False, True])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    Traceback (most recent call last):
    ...
    IndexError: Boolean index has wrong length: 3 instead of 2.

    NA values in a boolean array are treated as False.

    >>> mask = pd.array([True, pd.NA])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    A numpy boolean mask will get passed through (if the length is correct):

    >>> mask = np.array([True, False])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    Similarly for integer indexers, an integer ndarray is returned when it is
    a valid indexer, otherwise an error is  (for integer indexers, a matching
    length is not required):

    >>> indexer = pd.array([0, 2], dtype="Int64")
    >>> arr = pd.array([1, 2, 3])
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    array([0, 2])

    >>> indexer = pd.array([0, pd.NA], dtype="Int64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    ValueError: Cannot index with an integer indexer containing NA values

    For non-integer/boolean dtypes, an appropriate error is raised:

    >>> indexer = np.array([0., 2.], dtype="float64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    IndexError: arrays used as indices must be of integer or boolean type
    """
    from pandas.core.construction import array as pd_array

    # whatever is not an array-like is returned as-is (possible valid array
    # indexers that are not array-like: integer, slice, Ellipsis, None)
    # In this context, tuples are not considered as array-like, as they have
    # a specific meaning in indexing (multi-dimensional indexing)
    if is_list_like(indexer):
        if isinstance(indexer, tuple):
            return indexer
    else:
        return indexer

    # convert list-likes to array
    if not is_array_like(indexer):
        indexer = pd_array(indexer)
        if len(indexer) == 0:
            # empty list is converted to float array by pd.array
            indexer = np.array([], dtype=np.intp)

    dtype = indexer.dtype
    if is_bool_dtype(dtype):
        if isinstance(dtype, ExtensionDtype):
            indexer = indexer.to_numpy(dtype=bool, na_value=False)
        else:
            indexer = np.asarray(indexer, dtype=bool)

        # GH26658
        if len(indexer) != len(array):
            raise IndexError(
                f"Boolean index has wrong length: "
                f"{len(indexer)} instead of {len(array)}"
            )
    elif is_integer_dtype(dtype):
        try:
            indexer = np.asarray(indexer, dtype=np.intp)
        except ValueError as err:
            raise ValueError(
                "Cannot index with an integer indexer containing NA values"
            ) from err
    else:
        raise IndexError("arrays used as indices must be of integer or boolean type")

    return indexer
<|endoftext|>"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
from __future__ import annotations

from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    Interval,
    lib,
)
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_int64,
    is_bool,
    is_dict_like,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import algorithms
from pandas.core.apply import (
    GroupByApply,
    maybe_mangle_lambdas,
    reconstruct_func,
    validate_func_kwargs,
    warn_alias_replacement,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
    base,
    ops,
)
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
    _agg_template_frame,
    _agg_template_series,
    _apply_docs,
    _transform_template,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    all_indexes_same,
    default_index,
)
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba

from pandas.plotting import boxplot_frame_groupby

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        Axis,
        AxisInt,
        CorrelationMethod,
        FillnaOptions,
        IndexLabel,
        Manager,
        Manager2D,
        SingleManager,
        TakeIndexer,
    )

    from pandas import Categorical
    from pandas.core.generic import NDFrame

# TODO(typing) the return value on this callable should be any *scalar*.
AggScalar = Union[str, Callable[..., Any]]
# TODO: validate types on ScalarResult and move to _typing
# Blocked from using by https://github.com/python/mypy/issues/1484
# See note at _mangle_lambda_list
ScalarResult = TypeVar("ScalarResult")


class NamedAgg(NamedTuple):
    """
    Helper for column specific aggregation with control over output column names.

    Subclass of typing.NamedTuple.

    Parameters
    ----------
    column : Hashable
        Column label in the DataFrame to apply aggfunc.
    aggfunc : function or str
        Function to apply to the provided column. If string, the name of a built-in
        pandas function.

    Examples
    --------
    >>> df = pd.DataFrame({"key": [1, 1, 2], "a": [-1, 0, 1], 1: [10, 11, 12]})
    >>> agg_a = pd.NamedAgg(column="a", aggfunc="min")
    >>> agg_1 = pd.NamedAgg(column=1, aggfunc=lambda x: np.mean(x))
    >>> df.groupby("key").agg(result_a=agg_a, result_1=agg_1)
         result_a  result_1
    key
    1          -1      10.5
    2           1      12.0
    """

    column: Hashable
    aggfunc: AggScalar


class SeriesGroupBy(GroupBy[Series]):
    def _wrap_agged_manager(self, mgr: Manager) -> Series:
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: str | None = None
    ) -> SingleManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and not is_numeric_dtype(ser.dtype):
            # GH#41291 match Series behavior
            kwd_name = "numeric_only"
            raise TypeError(
                f"Cannot use {kwd_name}=True with "
                f"{type(self).__name__}.{name} and non-numeric dtypes."
            )
        return single

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])

    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).min()
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg('min')
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])
       min  max
    1    1    2
    2    3    4

    The output column names can be controlled by passing
    the desired column names and aggregations as keyword arguments.

    >>> s.groupby([1, 1, 2, 2]).agg(
    ...     minimum='min',
    ...     maximum='max',
    ... )
       minimum  maximum
    1        1        2
    2        3        4

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
    1    1.0
    2    3.0
    dtype: float64
    """
    )

    @Appender(
        _apply_docs["template"].format(
            input="series", examples=_apply_docs["series_examples"]
        )
    )
    def apply(self, func, *args, **kwargs) -> Series:
        return super().apply(func, *args, **kwargs)

    @doc(_agg_template_series, examples=_agg_examples_doc, klass="Series")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        relabeling = func is None
        columns = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}

        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                # Not all agg functions support numba, only propagate numba kwargs
                # if user asks for numba, and engine is not None
                # (if engine is None, the called function will handle the case where
                # numba is requested via the global option)
                kwargs["engine"] = engine
            if engine_kwargs is not None:
                kwargs["engine_kwargs"] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)

        elif isinstance(func, abc.Iterable):
            # Catch instances of lists / tuples
            # but not the class list / tuple itself.
            func = maybe_mangle_lambdas(func)
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            if relabeling:
                # columns is not narrowed by mypy from relabeling flag
                assert columns is not None  # for mypy
                ret.columns = columns
            if not self.as_index:
                ret = ret.reset_index()
            return ret

        else:
            cyfunc = com.get_cython_func(func)
            if cyfunc and not args and not kwargs:
                warn_alias_replacement(self, func, cyfunc)
                return getattr(self, cyfunc)()

            if maybe_use_numba(engine):
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )

            if self.ngroups == 0:
                # e.g. test_evaluate_with_empty_groups without any groups to
                #  iterate over, we have no output on which to do dtype
                #  inference. We default to using the existing dtype.
                #  xref GH#51445
                obj = self._obj_with_exclusions
                return self.obj._constructor(
                    [],
                    name=self.obj.name,
                    index=self._grouper.result_index,
                    dtype=obj.dtype,
                )

            if self._grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)

            try:
                return self._python_agg_general(func, *args, **kwargs)
            except KeyError:
                # KeyError raised in test_groupby.test_basic is bc the func does
                #  a dictionary lookup on group.name, but group name is not
                #  pinned in _python_agg_general, only in _aggregate_named
                result = self._aggregate_named(func, *args, **kwargs)

                warnings.warn(
                    "Pinning the groupby key to each group in "
                    f"{type(self).__name__}.agg is deprecated, and cases that "
                    "relied on it will raise in a future version. "
                    "If your operation requires utilizing the groupby keys, "
                    "iterate over the groupby object instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

                # result is a dict whose keys are the elements of result_index
                result = Series(result, index=self._grouper.result_index)
                result = self._wrap_aggregated_output(result)
                return result

    agg = aggregate

    def _python_agg_general(self, func, *args, **kwargs):
        orig_func = func
        func = com.is_builtin_func(func)
        if orig_func != func:
            alias = com._builtin_table_alias[func]
            warn_alias_replacement(self, orig_func, alias)
        f = lambda x: func(x, *args, **kwargs)

        obj = self._obj_with_exclusions
        result = self._grouper.agg_series(obj, f)
        res = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(self, arg, *args, **kwargs) -> DataFrame:
        if isinstance(arg, dict):
            if self.as_index:
                # GH 15931
                raise SpecificationError("nested renamer is not supported")
            else:
                # GH#50684 - This accidentally worked in 1.x
                msg = (
                    "Passing a dictionary to SeriesGroupBy.agg is deprecated "
                    "and will raise in a future version of pandas. Pass a list "
                    "of aggregations instead."
                )
                warnings.warn(
                    message=msg,
                    category=FutureWarning,
                    stacklevel=find_stack_level(),
                )
                arg = list(arg.items())
        elif any(isinstance(x, (tuple, list)) for x in arg):
            arg = [(x, x) if not isinstance(x, (tuple, list)) else x for x in arg]
        else:
            # list of functions / function names
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)

        results: dict[base.OutputKey, DataFrame | Series] = {}
        with com.temp_setattr(self, "as_index", True):
            # Combine results using the index, need to adjust index after
            # if as_index=False (GH#50724)
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)

        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results]
            )
            return res_df

        indexed_output = {key.position: val for key, val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)

        return output

    def _wrap_applied_output(
        self,
        data: Series,
        values: list[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> DataFrame | Series:
        """
        Wrap the output of SeriesGroupBy.apply into the expected result.

        Parameters
        ----------
        data : Series
            Input data for groupby operation.
        values : List[Any]
            Applied output for each group.
        not_indexed_same : bool, default False
            Whether the applied outputs are not indexed the same as the group axes.

        Returns
        -------
        DataFrame or Series
        """
        if len(values) == 0:
            # GH #6265
            if is_transform:
                # GH#47787 see test_group_on_empty_multiindex
                res_index = data.index
            else:
                res_index = self._grouper.result_index

            return self.obj._constructor(
                [],
                name=self.obj.name,
                index=res_index,
                dtype=data.dtype,
            )
        assert values is not None

        if isinstance(values[0], dict):
            # GH #823 #24880
            index = self._grouper.result_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_df = self._reindex_output(res_df)
            # if self.observed is False,
            # keep all-NaN rows created while re-indexing
            res_ser = res_df.stack(future_stack=True)
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                is_transform=is_transform,
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            # GH #6265 #24880
            result = self.obj._constructor(
                data=values, index=self._grouper.result_index, name=self.obj.name
            )
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return self._reindex_output(result)

    def _aggregate_named(self, func, *args, **kwargs):
        # Note: this is very similar to _aggregate_series_pure_python,
        #  but that does not pin group.name
        result = {}
        initialized = False

        for name, group in self._grouper.get_iterator(
            self._obj_with_exclusions, axis=self.axis
        ):
            # needed for pandas/tests/groupby/test_groupby.py::test_basic_aggregations
            object.__setattr__(group, "name", name)

            output = func(group, *args, **kwargs)
            output = ops.extract_result(output)
            if not initialized:
                # We only do this validation on the first iteration
                ops.check_result_array(output, group.dtype)
                initialized = True
            result[name] = output

        return result

    __examples_series_doc = dedent(
        """
    >>> ser = pd.Series([390.0, 350.0, 30.0, 20.0],
    ...                 index=["Falcon", "Falcon", "Parrot", "Parrot"],
    ...                 name="Max Speed")
    >>> grouped = ser.groupby([1, 1, 2, 2])
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
        Falcon    0.707107
        Falcon   -0.707107
        Parrot    0.707107
        Parrot   -0.707107
        Name: Max Speed, dtype: float64

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min())
    Falcon    40.0
    Falcon    40.0
    Parrot    10.0
    Parrot    10.0
    Name: Max Speed, dtype: float64

    >>> grouped.transform("mean")
    Falcon    370.0
    Falcon    370.0
    Parrot     25.0
    Parrot     25.0
    Name: Max Speed, dtype: float64

    .. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    for example:

    >>> grouped.transform(lambda x: x.astype(int).max())
    Falcon    390
    Falcon    390
    Parrot     30
    Parrot     30
    Name: Max Speed, dtype: int64
    """
    )

    @Substitution(klass="Series", example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _cython_transform(
        self, how: str, numeric_only: bool = False, axis: AxisInt = 0, **kwargs
    ):
        assert axis == 0  # handled by caller

        obj = self._obj_with_exclusions

        try:
            result = self._grouper._cython_operation(
                "transform", obj._values, how, axis, **kwargs
            )
        except NotImplementedError as err:
            # e.g. test_groupby_raises_string
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err

        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(
        self, func: Callable, engine, engine_kwargs, *args, **kwargs
    ) -> Series:
        """
        Transform with a callable `func`.
        """
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        assert callable(func)
        klass = type(self.obj)

        results = []
        for name, group in self._grouper.get_iterator(
            self._obj_with_exclusions, axis=self.axis
        ):
            # this setattr is needed for test_transform_lambda_with_datetimetz
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)

            results.append(klass(res, index=group.index))

        # check for empty "results" to avoid concat ValueError
        if results:
            from pandas.core.reshape.concat import concat

            concatenated = concat(results)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)

        result.name = self.obj.name
        return result

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.

        Returns
        -------
        Series

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> df.groupby('A').B.filter(lambda x: x.mean() > 3.)
        1    2
        3    4
        5    6
        Name: B, dtype: int64
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # Interpret np.nan as False.
        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return notna(b) and b

        try:
            indices = [
                self._get_index(name)
                for name, group in self._grouper.get_iterator(
                    self._obj_with_exclusions, axis=self.axis
                )
                if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err

        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Series | DataFrame:
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.

        Examples
        --------
        For SeriesGroupby:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    3
        dtype: int64
        >>> ser.groupby(level=0).nunique()
        a    2
        b    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 3], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    3
        dtype: int64
        >>> ser.resample('MS').nunique()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        ids, _, ngroups = self._grouper.group_info
        val = self.obj._values
        codes, uniques = algorithms.factorize(val, use_na_sentinel=dropna, sort=False)

        if self._grouper.has_dropped_na:
            mask = ids >= 0
            ids = ids[mask]
            codes = codes[mask]

        group_index = get_group_index(
            labels=[ids, codes],
            shape=(ngroups, len(uniques)),
            sort=False,
            xnull=dropna,
        )

        if dropna:
            mask = group_index >= 0
            if (~mask).any():
                ids = ids[mask]
                group_index = group_index[mask]

        mask = duplicated(group_index, "first")
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)

        ri = self._grouper.result_index
        result: Series | DataFrame = self.obj._constructor(
            res, index=ri, name=self.obj.name
        )
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return self._reindex_output(result, fill_value=0)

    @doc(Series.describe)
    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        return super().describe(
            percentiles=percentiles, include=include, exclude=exclude
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series | DataFrame:
        name = "proportion" if normalize else "count"

        if bins is None:
            result = self._value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            )
            result.name = name
            return result

        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut

        ids, _, _ = self._grouper.group_info
        val = self.obj._values

        index_names = self._grouper.names + [self.obj.name]

        if isinstance(val.dtype, CategoricalDtype) or (
            bins is not None and not np.iterable(bins)
        ):
            # scalar bins cannot be done at top level
            # in a backward compatible way
            # GH38672 relates to categorical dtype
            ser = self.apply(
                Series.value_counts,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
            )
            ser.name = name
            ser.index.names = index_names
            return ser

        # groupby removes null keys from groupings
        mask = ids != -1
        ids, val = ids[mask], val[mask]

        lab: Index | np.ndarray
        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:
            # lab is a Categorical with categories an IntervalIndex
            cat_ser = cut(Series(val, copy=False), bins, include_lowest=True)
            cat_obj = cast("Categorical", cat_ser._values)
            lev = cat_obj.categories
            lab = lev.take(
                cat_obj.codes,
                allow_fill=True,
                fill_value=lev._na_value,
            )
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]

        if isinstance(lab.dtype, IntervalDtype):
            # TODO: should we do this inside II?
            lab_interval = cast(Interval, lab)

            sorter = np.lexsort((lab_interval.left, lab_interval.right, ids))
        else:
            sorter = np.lexsort((lab, ids))

        ids, lab = ids[sorter], lab[sorter]

        # group boundaries are where group ids change
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = idchanges

        # new values are where sorted labels change
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = lchanges
        inc[idx] = True  # group boundaries are also new values
        out = np.diff(np.nonzero(np.r_[inc, True])[0])  # value counts

        # num. of times each group should be repeated
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))

        # multi-index components
        codes = self._grouper.reconstructed_codes
        codes = [rep(level_codes) for level_codes in codes] + [llab(lab, inc)]
        levels = [ping._group_index for ping in self._grouper.groupings] + [lev]

        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = out[mask], [level_codes[mask] for level_codes in codes]

        if normalize:
            out = out.astype("float")
            d = np.diff(np.r_[idx, len(ids)])
            if dropna:
                m = ids[lab == -1]
                np.add.at(d, m, -1)
                acc = rep(d)[mask]
            else:
                acc = rep(d)
            out /= acc

        if sort and bins is None:
            cat = ids[inc][mask] if dropna else ids[inc]
            sorter = np.lexsort((out if ascending else -out, cat))
            out, codes[-1] = out[sorter], codes[-1][sorter]

        if bins is not None:
            # for compat. with libgroupby.value_counts need to ensure every
            # bin is present at every index level, null filled with zeros
            diff = np.zeros(len(out), dtype="bool")
            for level_codes in codes[:-1]:
                diff |= np.r_[True, level_codes[1:] != level_codes[:-1]]

            ncat, nbin = diff.sum(), len(levels[-1])

            left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]

            right = [diff.cumsum() - 1, codes[-1]]

            # error: Argument 1 to "get_join_indexers" has incompatible type
            # "List[ndarray[Any, Any]]"; expected "List[Union[Union[ExtensionArray,
            # ndarray[Any, Any]], Index, Series]]
            _, idx = get_join_indexers(
                left, right, sort=False, how="left"  # type: ignore[arg-type]
            )
            if idx is not None:
                out = np.where(idx != -1, out[idx], 0)

            if sort:
                sorter = np.lexsort((out if ascending else -out, left[0]))
                out, left[-1] = out[sorter], left[-1][sorter]

            # build the multi-index w/ full levels
            def build_codes(lev_codes: np.ndarray) -> np.ndarray:
                return np.repeat(lev_codes[diff], nbin)

            codes = [build_codes(lev_codes) for lev_codes in codes[:-1]]
            codes.append(left[-1])

        mi = MultiIndex(
            levels=levels, codes=codes, names=index_names, verify_integrity=False
        )

        if is_integer_dtype(out.dtype):
            out = ensure_int64(out)
        result = self.obj._constructor(out, index=mi, name=name)
        if not self.as_index:
            result = result.reset_index()
        return result

    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        inplace: bool = False,
        limit: int | None = None,
        downcast: dict | None | lib.NoDefault = lib.no_default,
    ) -> Series | None:
        """
        Fill NA/NaN values using the specified method within groups.

        .. deprecated:: 2.2.0
            This method is deprecated and will be removed in a future version.
            Use the :meth:`.SeriesGroupBy.ffill` or :meth:`.SeriesGroupBy.bfill`
            for forward or backward filling instead. If you want to fill with a
            single value, use :meth:`Series.fillna` instead.

        Parameters
        ----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list. Users wanting to use the ``value`` argument and not ``method``
            should prefer :meth:`.Series.fillna` as this
            will produce the same result and be more performant.
        method : {{'bfill', 'ffill', None}}, default None
            Method to use for filling holes. ``'ffill'`` will propagate
            the last valid observation forward within a group.
            ``'bfill'`` will use next valid observation to fill the gap.
        axis : {0 or 'index', 1 or 'columns'}
            Unused, only for compatibility with :meth:`DataFrameGroupBy.fillna`.
        inplace : bool, default False
            Broken. Do not set to True.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill within a group. In other words,
            if there is a gap with more than this number of consecutive NaNs,
            it will only be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

        Returns
        -------
        Series
            Object with missing values filled within groups.

        See Also
        --------
        ffill : Forward fill values within a group.
        bfill : Backward fill values within a group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['cat', 'cat', 'cat', 'mouse', 'mouse']
        >>> ser = pd.Series([1, None, None, 2, None], index=lst)
        >>> ser
        cat    1.0
        cat    NaN
        cat    NaN
        mouse  2.0
        mouse  NaN
        dtype: float64
        >>> ser.groupby(level=0).fillna(0, limit=1)
        cat    1.0
        cat    0.0
        cat    NaN
        mouse  2.0
        mouse  0.0
        dtype: float64
        """
        warnings.warn(
            f"{type(self).__name__}.fillna is deprecated and "
            "will be removed in a future version. Use obj.ffill() or obj.bfill() "
            "for forward or backward filling instead. If you want to fill with a "
            f"single value, use {type(self.obj).__name__}.fillna instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        result = self._op_via_apply(
            "fillna",
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )
        return result

    def take(
        self,
        indices: TakeIndexer,
        axis: Axis | lib.NoDefault = lib.no_default,
        **kwargs,
    ) -> Series:
        """
        Return the elements in the given *positional* indices in each group.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        If a requested index does not exist for some group, this method will raise.
        To get similar behavior that ignores indices that don't exist, see
        :meth:`.SeriesGroupBy.nth`.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take in each group.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.
            For `SeriesGroupBy` this parameter is unused and defaults to 0.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.

        Returns
        -------
        Series
            A Series containing the elements taken from each group.

        See Also
        --------
        Series.take : Take elements from a Series along an axis.
        Series.loc : Select a subset of a DataFrame by labels.
        Series.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.
        SeriesGroupBy.nth : Similar to take, won't raise if indices don't exist.

        Examples
        --------
        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan),
        ...                    ('rabbit', 'mammal', 15.0)],
        ...                   columns=['name', 'class', 'max_speed'],
        ...                   index=[4, 3, 2, 1, 0])
        >>> df
             name   class  max_speed
        4  falcon    bird      389.0
        3  parrot    bird       24.0
        2    lion  mammal       80.5
        1  monkey  mammal        NaN
        0  rabbit  mammal       15.0
        >>> gb = df["name"].groupby([1, 1, 2, 2, 2])

        Take elements at positions 0 and 1 along the axis 0 in each group (default).

        >>> gb.take([0, 1])
        1  4    falcon
           3    parrot
        2  2      lion
           1    monkey
        Name: name, dtype: object

        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.

        >>> gb.take([-1, -2])
        1  3    parrot
           4    falcon
        2  0    rabbit
           1    monkey
        Name: name, dtype: object
        """
        result = self._op_via_apply("take", indices=indices, axis=axis, **kwargs)
        return result

    def skew(
        self,
        axis: Axis | lib.NoDefault = lib.no_default,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series:
        """
        Return unbiased skew within groups.

        Normalized by N-1.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis for the function to be applied on.
            This parameter is only for compatibility with DataFrame and is unused.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series

        See Also
        --------
        Series.skew : Return unbiased skew over requested axis.

        Examples
        --------
        >>> ser = pd.Series([390., 350., 357., np.nan, 22., 20., 30.],
        ...                 index=['Falcon', 'Falcon', 'Falcon', 'Falcon',
        ...                        'Parrot', 'Parrot', 'Parrot'],
        ...                 name="Max Speed")
        >>> ser
        Falcon    390.0
        Falcon    350.0
        Falcon    357.0
        Falcon      NaN
        Parrot     22.0
        Parrot     20.0
        Parrot     30.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).skew()
        Falcon    1.525174
        Parrot    1.457863
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).skew(skipna=False)
        Falcon         NaN
        Parrot    1.457863
        Name: Max Speed, dtype: float64
        """
        if axis is lib.no_default:
            axis = 0

        if axis != 0:
            result = self._op_via_apply(
                "skew",
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )
            return result

        def alt(obj):
            # This should not be reached since the cython path should raise
            #  TypeError and not NotImplementedError.
            raise TypeError(f"'skew' is not supported for dtype={obj.dtype}")

        return self._cython_agg_general(
            "skew", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @property
    @doc(Series.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(Series.nlargest.__doc__)
    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f = partial(Series.nlargest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.nsmallest.__doc__)
    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f = partial(Series.nsmallest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.idxmin.__doc__)
    def idxmin(
        self, axis: Axis | lib.NoDefault = lib.no_default, skipna: bool = True
    ) -> Series:
        return self._idxmax_idxmin("idxmin", axis=axis, skipna=skipna)

    @doc(Series.idxmax.__doc__)
    def idxmax(
        self, axis: Axis | lib.NoDefault = lib.no_default, skipna: bool = True
    ) -> Series:
        return self._idxmax_idxmin("idxmax", axis=axis, skipna=skipna)

    @doc(Series.corr.__doc__)
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> Series:
        result = self._op_via_apply(
            "corr", other=other, method=method, min_periods=min_periods
        )
        return result

    @doc(Series.cov.__doc__)
    def cov(
        self, other: Series, min_periods: int | None = None, ddof: int | None = 1
    ) -> Series:
        result = self._op_via_apply(
            "cov", other=other, min_periods=min_periods, ddof=ddof
        )
        return result

    @property
    def is_monotonic_increasing(self) -> Series:
        """
        Return whether each group's values are monotonically increasing.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([2, 1, 3, 4], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'])
        >>> s.groupby(level=0).is_monotonic_increasing
        Falcon    False
        Parrot     True
        dtype: bool
        """
        return self.apply(lambda ser: ser.is_monotonic_increasing)

    @property
    def is_monotonic_decreasing(self) -> Series:
        """
        Return whether each group's values are monotonically decreasing.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([2, 1, 3, 4], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'])
        >>> s.groupby(level=0).is_monotonic_decreasing
        Falcon     True
        Parrot    False
        dtype: bool
        """
        return self.apply(lambda ser: ser.is_monotonic_decreasing)

    @doc(Series.hist.__doc__)
    def hist(
        self,
        by=None,
        ax=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        figsize: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        result = self._op_via_apply(
            "hist",
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    @property
    @doc(Series.dtype.__doc__)
    def dtype(self) -> Series:
        return self.apply(lambda ser: ser.dtype)

    def unique(self) -> Series:
        """
        Return unique values for each group.

        It returns unique values for each of the grouped values. Returned in
        order of appearance. Hash table-based unique, therefore does NOT sort.

        Returns
        -------
        Series
            Unique values for each of the grouped values.

        See Also
        --------
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> df = pd.DataFrame([('Chihuahua', 'dog', 6.1),
        ...                    ('Beagle', 'dog', 15.2),
        ...                    ('Chihuahua', 'dog', 6.9),
        ...                    ('Persian', 'cat', 9.2),
        ...                    ('Chihuahua', 'dog', 7),
        ...                    ('Persian', 'cat', 8.8)],
        ...                   columns=['breed', 'animal', 'height_in'])
        >>> df
               breed     animal   height_in
        0  Chihuahua        dog         6.1
        1     Beagle        dog        15.2
        2  Chihuahua        dog         6.9
        3    Persian        cat         9.2
        4  Chihuahua        dog         7.0
        5    Persian        cat         8.8
        >>> ser = df.groupby('animal')['breed'].unique()
        >>> ser
        animal
        cat              [Persian]
        dog    [Chihuahua, Beagle]
        Name: breed, dtype: object
        """
        result = self._op_via_apply("unique")
        return result


class DataFrameGroupBy(GroupBy[DataFrame]):
    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> data = {"A": [1, 1, 2, 2],
    ...         "B": [1, 2, 3, 4],
    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860]}
    >>> df = pd.DataFrame(data)
    >>> df
       A  B         C
    0  1  1  0.362838
    1  1  2  0.227877
    2  2  3  1.267767
    3  2  4 -0.562860

    The aggregation is for each column.

    >>> df.groupby('A').agg('min')
       B         C
    A
    1  1  0.227877
    2  3 -0.562860

    Multiple aggregations

    >>> df.groupby('A').agg(['min', 'max'])
        B             C
      min max       min       max
    A
    1   1   2  0.227877  0.362838
    2   3   4 -0.562860  1.267767

    Select a column for aggregation

    >>> df.groupby('A').B.agg(['min', 'max'])
       min  max
    A
    1    1    2
    2    3    4

    User-defined function for aggregation

    >>> df.groupby('A').agg(lambda x: sum(x) + 2)
        B	       C
    A
    1	5	2.590715
    2	9	2.704907

    Different aggregations per column

    >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
        B             C
      min max       sum
    A
    1   1   2  0.590715
    2   3   4  0.704907

    To control the output names with different aggregations per column,
    pandas supports "named aggregation"

    >>> df.groupby("A").agg(
    ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
    ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum")
    ... )
       b_min     c_sum
    A
    1      1  0.590715
    2      3  0.704907

    - The keywords are the *output* column names
    - The values are tuples whose first element is the column to select
      and the second element is the aggregation to apply to that column.
      Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
      ``['column', 'aggfunc']`` to make it clearer what the arguments are.
      As usual, the aggregation can be a callable or a string alias.

    See :ref:`groupby.aggregate.named` for more.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
          B
    A
    1   1.0
    2   3.0
    """
    )

    @doc(_agg_template_frame, examples=_agg_examples_doc, klass="DataFrame")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)

        if maybe_use_numba(engine):
            # Not all agg functions support numba, only propagate numba kwargs
            # if user asks for numba
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs

        op = GroupByApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        if not is_dict_like(func) and result is not None:
            # GH #52849
            if not self.as_index and is_list_like(func):
                return result.reset_index()
            else:
                return result
        elif relabeling:
            # this should be the only (non-raising) case with relabeling
            # used reordered index of columns
            result = cast(DataFrame, result)
            result = result.iloc[:, order]
            result = cast(DataFrame, result)
            # error: Incompatible types in assignment (expression has type
            # "Optional[List[str]]", variable has type
            # "Union[Union[Union[ExtensionArray, ndarray[Any, Any]],
            # Index, Series], Sequence[Any]]")
            result.columns = columns  # type: ignore[assignment]

        if result is None:
            # Remove the kwargs we inserted
            # (already stored in engine, engine_kwargs arguments)
            if "engine" in kwargs:
                del kwargs["engine"]
                del kwargs["engine_kwargs"]
            # at this point func is not a str, list-like, dict-like,
            # or a known callable(e.g. sum)
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )
            # grouper specific aggregations
            if self._grouper.nkeys > 1:
                # test_groupby_as_index_series_scalar gets here with 'not self.as_index'
                return self._python_agg_general(func, *args, **kwargs)
            elif args or kwargs:
                # test_pass_args_kwargs gets here (with and without as_index)
                # can't return early
                result = self._aggregate_frame(func, *args, **kwargs)

            elif self.axis == 1:
                # _aggregate_multiple_funcs does not allow self.axis == 1
                # Note: axis == 1 precludes 'not self.as_index', see __init__
                result = self._aggregate_frame(func)
                return result

            else:
                # try to treat as if we are passing a list
                gba = GroupByApply(self, [func], args=(), kwargs={})
                try:
                    result = gba.agg()

                except ValueError as err:
                    if "No objects to concatenate" not in str(err):
                        raise
                    # _aggregate_frame can fail with e.g. func=Series.mode,
                    # where it expects 1D values but would be getting 2D values
                    # In other tests, using aggregate_frame instead of GroupByApply
                    #  would give correct values but incorrect dtypes
                    #  object vs float64 in test_cython_agg_empty_buckets
                    #  float64 vs int64 in test_category_order_apply
                    result = self._aggregate_frame(func)

                else:
                    # GH#32040, GH#35246
                    # e.g. test_groupby_as_index_select_column_sum_empty_df
                    result = cast(DataFrame, result)
                    result.columns = self._obj_with_exclusions.columns.copy()

        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))

        return result

    agg = aggregate

    def _python_agg_general(self, func, *args, **kwargs):
        orig_func = func
        func = com.is_builtin_func(func)
        if orig_func != func:
            alias = com._builtin_table_alias[func]
            warn_alias_replacement(self, orig_func, alias)
        f = lambda x: func(x, *args, **kwargs)

        if self.ngroups == 0:
            # e.g. test_evaluate_with_empty_groups different path gets different
            #  result dtype in empty case.
            return self._python_apply_general(f, self._selected_obj, is_agg=True)

        obj = self._obj_with_exclusions
        if self.axis == 1:
            obj = obj.T

        if not len(obj.columns):
            # e.g. test_margins_no_values_no_cols
            return self._python_apply_general(f, self._selected_obj)

        output: dict[int, ArrayLike] = {}
        for idx, (name, ser) in enumerate(obj.items()):
            result = self._grouper.agg_series(ser, f)
            output[idx] = result

        res = self.obj._constructor(output)
        res.columns = obj.columns.copy(deep=False)
        return self._wrap_aggregated_output(res)

    def _aggregate_frame(self, func, *args, **kwargs) -> DataFrame:
        if self._grouper.nkeys != 1:
            raise AssertionError("Number of keys must be 1")

        obj = self._obj_with_exclusions

        result: dict[Hashable, NDFrame | np.ndarray] = {}
        for name, grp_df in self._grouper.get_iterator(obj, self.axis):
            fres = func(grp_df, *args, **kwargs)
            result[name] = fres

        result_index = self._grouper.result_index
        other_ax = obj.axes[1 - self.axis]
        out = self.obj._constructor(result, index=other_ax, columns=result_index)
        if self.axis == 0:
            out = out.T

        return out

    def _wrap_applied_output(
        self,
        data: DataFrame,
        values: list,
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ):
        if len(values) == 0:
            if is_transform:
                # GH#47787 see test_group_on_empty_multiindex
                res_index = data.index
            else:
                res_index = self._grouper.result_index

            result = self.obj._constructor(index=res_index, columns=data.columns)
            result = result.astype(data.dtypes, copy=False)
            return result

        # GH12824
        # using values[0] here breaks test_groupby_apply_none_first
        first_not_none = next(com.not_none(*values), None)

        if first_not_none is None:
            # GH9684 - All values are None, return an empty frame.
            return self.obj._constructor()
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                is_transform=is_transform,
            )

        key_index = self._grouper.result_index if self.as_index else None

        if isinstance(first_not_none, (np.ndarray, Index)):
            # GH#1738: values is list of arrays of unequal lengths
            #  fall through to the outer else clause
            # TODO: sure this is right?  we used to do this
            #  after raising AttributeError above
            # GH 18930
            if not is_hashable(self._selection):
                # error: Need type annotation for "name"
                name = tuple(self._selection)  # type: ignore[var-annotated, arg-type]
            else:
                # error: Incompatible types in assignment
                # (expression has type "Hashable", variable
                # has type "Tuple[Any, ...]")
                name = self._selection  # type: ignore[assignment]
            return self.obj._constructor_sliced(values, index=key_index, name=name)
        elif not isinstance(first_not_none, Series):
            # values are not series or array-like but scalars
            # self._selection not passed through to Series as the
            # result should not take the name of original selection
            # of columns
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(values, columns=[self._selection])
                result = self._insert_inaxis_grouper(result)
                return result
        else:
            # values are Series
            return self._wrap_applied_output_series(
                values,
                not_indexed_same,
                first_not_none,
                key_index,
                is_transform,
            )

    def _wrap_applied_output_series(
        self,
        values: list[Series],
        not_indexed_same: bool,
        first_not_none,
        key_index: Index | None,
        is_transform: bool,
    ) -> DataFrame | Series:
        kwargs = first_not_none._construct_axes_dict()
        backup = Series(**kwargs)
        values = [x if (x is not None) else backup for x in values]

        all_indexed_same = all_indexes_same(x.index for x in values)

        if not all_indexed_same:
            # GH 8467
            return self._concat_objects(
                values,
                not_indexed_same=True,
                is_transform=is_transform,
            )

        # Combine values
        # vstack+constructor is faster than concat and handles MI-columns
        stacked_values = np.vstack([np.asarray(v) for v in values])

        if self.axis == 0:
            index = key_index
            columns = first_not_none.index.copy()
            if columns.name is None:
                # GH6124 - propagate name of Series when it's consistent
                names = {v.name for v in values}
                if len(names) == 1:
                    columns.name = next(iter(names))
        else:
            index = first_not_none.index
            columns = key_index
            stacked_values = stacked_values.T

        if stacked_values.dtype == object:
            # We'll have the DataFrame constructor do inference
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(stacked_values, index=index, columns=columns)

        if not self.as_index:
            result = self._insert_inaxis_grouper(result)

        return self._reindex_output(result)

    def _cython_transform(
        self,
        how: str,
        numeric_only: bool = False,
        axis: AxisInt = 0,
        **kwargs,
    ) -> DataFrame:
        assert axis == 0  # handled by caller

        # With self.axis == 0, we have multi-block tests
        #  e.g. test_rank_min_int, test_cython_transform_frame
        #  test_transform_numeric_ret
        # With self.axis == 1, _get_data_to_aggregate does a transpose
        #  so we always have a single block.
        mgr: Manager2D = self._get_data_to_aggregate(
            numeric_only=numeric_only, name=how
        )

        def arr_func(bvalues: ArrayLike) -> ArrayLike:
            return self._grouper._cython_operation(
                "transform", bvalues, how, 1, **kwargs
            )

        # We could use `mgr.apply` here and not have to set_axis, but
        #  we would have to do shape gymnastics for ArrayManager compat
        res_mgr = mgr.grouped_reduce(arr_func)
        res_mgr.set_axis(1, mgr.axes[1])

        res_df = self.obj._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        res_df = self._maybe_transpose_result(res_df)
        return res_df

    def _transform_general(self, func, engine, engine_kwargs, *args, **kwargs):
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        from pandas.core.reshape.concat import concat

        applied = []
        obj = self._obj_with_exclusions
        gen = self._grouper.get_iterator(obj, axis=self.axis)
        fast_path, slow_path = self._define_paths(func, *args, **kwargs)

        # Determine whether to use slow or fast path by evaluating on the first group.
        # Need to handle the case of an empty generator and process the result so that
        # it does not need to be computed again.
        try:
            name, group = next(gen)
        except StopIteration:
            pass
        else:
            # 2023-02-27 No tests broken by disabling this pinning
            object.__setattr__(group, "name", name)
            try:
                path, res = self._choose_path(fast_path, slow_path, group)
            except ValueError as err:
                # e.g. test_transform_with_non_scalar_group
                msg = "transform must return a scalar value for each group"
                raise ValueError(msg) from err
            if group.size > 0:
                res = _wrap_transform_general_frame(self.obj, group, res)
                applied.append(res)

        # Compute and process with the remaining groups
        for name, group in gen:
            if group.size == 0:
                continue
            # 2023-02-27 No tests broken by disabling this pinning
            object.__setattr__(group, "name", name)
            res = path(group)

            res = _wrap_transform_general_frame(self.obj, group, res)
            applied.append(res)

        concat_index = obj.columns if self.axis == 0 else obj.index
        other_axis = 1 if self.axis == 0 else 0  # switches between 0 & 1
        concatenated = concat(applied, axis=self.axis, verify_integrity=False)
        concatenated = concatenated.reindex(concat_index, axis=other_axis, copy=False)
        return self._set_result_index_ordered(concatenated)

    __examples_dataframe_doc = dedent(
        """
    >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
    ...                           'foo', 'bar'],
    ...                    'B' : ['one', 'one', 'two', 'three',
    ...                           'two', 'two'],
    ...                    'C' : [1, 5, 5, 2, 5, 5],
    ...                    'D' : [2.0, 5., 8., 1., 2., 9.]})
    >>> grouped = df.groupby('A')[['C', 'D']]
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
            C         D
    0 -1.154701 -0.577350
    1  0.577350  0.000000
    2  0.577350  1.154701
    3 -1.154701 -1.000000
    4  0.577350 -0.577350
    5  0.577350  1.000000

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min())
        C    D
    0  4.0  6.0
    1  3.0  8.0
    2  4.0  6.0
    3  3.0  8.0
    4  4.0  6.0
    5  3.0  8.0

    >>> grouped.transform("mean")
        C    D
    0  3.666667  4.0
    1  4.000000  5.0
    2  3.666667  4.0
    3  4.000000  5.0
    4  3.666667  4.0
    5  4.000000  5.0

    .. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    for example:

    >>> grouped.transform(lambda x: x.astype(int).max())
    C  D
    0  5  8
    1  5  9
    2  5  8
    3  5  9
    4  5  8
    5  5  9
    """
    )

    @Substitution(klass="DataFrame", example=__examples_dataframe_doc)
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _define_paths(self, func, *args, **kwargs):
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(
                lambda x: getattr(x, func)(*args, **kwargs), axis=self.axis
            )
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(
                lambda x: func(x, *args, **kwargs), axis=self.axis
            )
        return fast_path, slow_path

    def _choose_path(self, fast_path: Callable, slow_path: Callable, group: DataFrame):
        path = slow_path
        res = slow_path(group)

        if self.ngroups == 1:
            # no need to evaluate multiple paths when only
            # a single group exists
            return path, res

        # if we make it here, test if we can use the fast path
        try:
            res_fast = fast_path(group)
        except AssertionError:
            raise  # pragma: no cover
        except Exception:
            # GH#29631 For user-defined function, we can't predict what may be
            #  raised; see test_transform.test_transform_fastpath_raises
            return path, res

        # verify fast path returns either:
        # a DataFrame with columns equal to group.columns
        # OR a Series with index equal to group.columns
        if isinstance(res_fast, DataFrame):
            if not res_fast.columns.equals(group.columns):
                return path, res
        elif isinstance(res_fast, Series):
            if not res_fast.index.equals(group.columns):
                return path, res
        else:
            return path, res

        if res_fast.equals(res):
            path = fast_path

        return path, res

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.

        Returns
        -------
        DataFrame

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
        """
        indices = []

        obj = self._selected_obj
        gen = self._grouper.get_iterator(obj, axis=self.axis)

        for name, group in gen:
            # 2023-02-27 no tests are broken this pinning, but it is documented in the
            #  docstring above.
            object.__setattr__(group, "name", name)

            res = func(group, *args, **kwargs)

            try:
                res = res.squeeze()
            except AttributeError:  # allow e.g., scalars and frames to pass
                pass

            # interpret the result of the filter
            if is_bool(res) or (is_scalar(res) and isna(res)):
                if notna(res) and res:
                    indices.append(self._get_index(name))
            else:
                # non scalars aren't allowed
                raise TypeError(
                    f"filter function returned a {type(res).__name__}, "
                    "but expected a scalar bool"
                )

        return self._apply_filter(indices, dropna)

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        if self.axis == 1:
            # GH 37725
            raise ValueError("Cannot subset columns when using axis=1")
        # per GH 23566
        if isinstance(key, tuple) and len(key) > 1:
            # if len == 1, then it becomes a SeriesGroupBy and this is actually
            # valid syntax, so don't raise
            raise ValueError(
                "Cannot subset columns with a tuple with more than one element. "
                "Use a list instead."
            )
        return super().__getitem__(key)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return DataFrameGroupBy(
                subset,
                self.keys,
                axis=self.axis,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return SeriesGroupBy(
                subset,
                self.keys,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )

        raise AssertionError("invalid ndim for _gotitem")

    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: str | None = None
    ) -> Manager2D:
        obj = self._obj_with_exclusions
        if self.axis == 1:
            mgr = obj.T._mgr
        else:
            mgr = obj._mgr

        if numeric_only:
            mgr = mgr.get_numeric_data()
        return mgr

    def _wrap_agged_manager(self, mgr: Manager2D) -> DataFrame:
        return self.obj._constructor_from_mgr(mgr, axes=mgr.axes)

    def _apply_to_column_groupbys(self, func) -> DataFrame:
        from pandas.core.reshape.concat import concat

        obj = self._obj_with_exclusions
        columns = obj.columns
        sgbs = [
            SeriesGroupBy(
                obj.iloc[:, i],
                selection=colname,
                grouper=self._grouper,
                exclusions=self.exclusions,
                observed=self.observed,
            )
            for i, colname in enumerate(obj.columns)
        ]
        results = [func(sgb) for sgb in sgbs]

        if not len(results):
            # concat would raise
            res_df = DataFrame([], columns=columns, index=self._grouper.result_index)
        else:
            res_df = concat(results, keys=columns, axis=1)

        if not self.as_index:
            res_df.index = default_index(len(res_df))
            res_df = self._insert_inaxis_grouper(res_df)
        return res_df

    def nunique(self, dropna: bool = True) -> DataFrame:
        """
        Return DataFrame with counts of unique elements in each position.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        nunique: DataFrame

        Examples
        --------
        >>> df = pd.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',
        ...                           'ham', 'ham'],
        ...                    'value1': [1, 5, 5, 2, 5, 5],
        ...                    'value2': list('abbaxy')})
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y

        >>> df.groupby('id').nunique()
              value1  value2
        id
        egg        1       1
        ham        1       2
        spam       2       1

        Check for rows with the same id but conflicting values:

        >>> df.groupby('id').filter(lambda g: (g.nunique() > 1).any())
             id  value1 value2
        0  spam       1      a
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        """

        if self.axis != 0:
            # see test_groupby_crash_on_nunique
            return self._python_apply_general(
                lambda sgb: sgb.nunique(dropna), self._obj_with_exclusions, is_agg=True
            )

        return self._apply_to_column_groupbys(lambda sgb: sgb.nunique(dropna))

    def idxmax(
        self,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Return index of first occurrence of maximum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.

            .. versionchanged:: 2.0.0

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series
            Indexes of maxima along the specified axis.

        Raises
        ------
        ValueError
            * If the row/column is empty

        See Also
        --------
        Series.idxmax : Return index of the maximum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmax``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48],
        ...                    'co2_emissions': [37.2, 19.66, 1712]},
        ...                   index=['Pork', 'Wheat Products', 'Beef'])

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the maximum value in each column.

        >>> df.idxmax()
        consumption     Wheat Products
        co2_emissions             Beef
        dtype: object

        To return the index for the maximum value in each row, use ``axis="columns"``.

        >>> df.idxmax(axis="columns")
        Pork              co2_emissions
        Wheat Products     consumption
        Beef              co2_emissions
        dtype: object
        """
        return self._idxmax_idxmin(
            "idxmax", axis=axis, numeric_only=numeric_only, skipna=skipna
        )

    def idxmin(
        self,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Return index of first occurrence of minimum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.

            .. versionchanged:: 2.0.0

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series
            Indexes of minima along the specified axis.

        Raises
        ------
        ValueError
            * If the row/column is empty

        See Also
        --------
        Series.idxmin : Return index of the minimum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmin``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48],
        ...                    'co2_emissions': [37.2, 19.66, 1712]},
        ...                   index=['Pork', 'Wheat Products', 'Beef'])

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the minimum value in each column.

        >>> df.idxmin()
        consumption                Pork
        co2_emissions    Wheat Products
        dtype: object

        To return the index for the minimum value in each row, use ``axis="columns"``.

        >>> df.idxmin(axis="columns")
        Pork                consumption
        Wheat Products    co2_emissions
        Beef                consumption
        dtype: object
        """
        return self._idxmax_idxmin(
            "idxmin", axis=axis, numeric_only=numeric_only, skipna=skipna
        )

    boxplot = boxplot_frame_groupby

    def value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame | Series:
        """
        Return a Series or DataFrame containing counts of unique rows.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        subset : list-like, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don't include counts of rows that contain NA values.

        Returns
        -------
        Series or DataFrame
            Series if the groupby as_index is True, otherwise DataFrame.

        See Also
        --------
        Series.value_counts: Equivalent method on Series.
        DataFrame.value_counts: Equivalent method on DataFrame.
        SeriesGroupBy.value_counts: Equivalent method on SeriesGroupBy.

        Notes
        -----
        - If the groupby as_index is True then the returned Series will have a
          MultiIndex with one level per input column.
        - If the groupby as_index is False then the returned DataFrame will have an
          additional column with the value_counts. The column is labelled 'count' or
          'proportion', depending on the ``normalize`` parameter.

        By default, rows that contain any NA values are omitted from
        the result.

        By default, the result will be in descending order so that the
        first element of each group is the most frequently-occurring row.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        ...     'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        ...     'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
        ... })

        >>> df
                gender  education   country
        0       male    low         US
        1       male    medium      FR
        2       female  high        US
        3       male    low         FR
        4       female  high        FR
        5       male    low         FR

        >>> df.groupby('gender').value_counts()
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        FR         2
                           US         1
                medium     FR         1
        Name: count, dtype: int64

        >>> df.groupby('gender').value_counts(ascending=True)
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        US         1
                medium     FR         1
                low        FR         2
        Name: count, dtype: int64

        >>> df.groupby('gender').value_counts(normalize=True)
        gender  education  country
        female  high       FR         0.50
                           US         0.50
        male    low        FR         0.50
                           US         0.25
                medium     FR         0.25
        Name: proportion, dtype: float64

        >>> df.groupby('gender', as_index=False).value_counts()
           gender education country  count
        0  female      high      FR      1
        1  female      high      US      1
        2    male       low      FR      2
        3    male       low      US      1
        4    male    medium      FR      1

        >>> df.groupby('gender', as_index=False).value_counts(normalize=True)
           gender education country  proportion
        0  female      high      FR        0.50
        1  female      high      US        0.50
        2    male       low      FR        0.50
        3    male       low      US        0.25
        4    male    medium      FR        0.25
        """
        return self._value_counts(subset, normalize, sort, ascending, dropna)

    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame | None = None,
        method: FillnaOptions | None = None,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        inplace: bool = False,
        limit: int | None = None,
        downcast=lib.no_default,
    ) -> DataFrame | None:
        """
        Fill NA/NaN values using the specified method within groups.

        .. deprecated:: 2.2.0
            This method is deprecated and will be removed in a future version.
            Use the :meth:`.DataFrameGroupBy.ffill` or :meth:`.DataFrameGroupBy.bfill`
            for forward or backward filling instead. If you want to fill with a
            single value, use :meth:`DataFrame.fillna` instead.

        Parameters
        ----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list. Users wanting to use the ``value`` argument and not ``method``
            should prefer :meth:`.DataFrame.fillna` as this
            will produce the same result and be more performant.
        method : {{'bfill', 'ffill', None}}, default None
            Method to use for filling holes. ``'ffill'`` will propagate
            the last valid observation forward within a group.
            ``'bfill'`` will use next valid observation to fill the gap.
        axis : {0 or 'index', 1 or 'columns'}
            Axis along which to fill missing values. When the :class:`DataFrameGroupBy`
            ``axis`` argument is ``0``, using ``axis=1`` here will produce
            the same results as :meth:`.DataFrame.fillna`. When the
            :class:`DataFrameGroupBy` ``axis`` argument is ``1``, using ``axis=0``
            or ``axis=1`` here will produce the same results.
        inplace : bool, default False
            Broken. Do not set to True.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill within a group. In other words,
            if there is a gap with more than this number of consecutive NaNs,
            it will only be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

        Returns
        -------
        DataFrame
            Object with missing values filled.

        See Also
        --------
        ffill : Forward fill values within a group.
        bfill : Backward fill values within a group.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "key": [0, 0, 1, 1, 1],
        ...         "A": [np.nan, 2, np.nan, 3, np.nan],
        ...         "B": [2, 3, np.nan, np.nan, np.nan],
        ...         "C": [np.nan, np.nan, 2, np.nan, np.nan],
        ...     }
        ... )
        >>> df
           key    A    B   C
        0    0  NaN  2.0 NaN
        1    0  2.0  3.0 NaN
        2    1  NaN  NaN 2.0
        3    1  3.0  NaN NaN
        4    1  NaN  NaN NaN

        Propagate non-null values forward or backward within each group along columns.

        >>> df.groupby("key").fillna(method="ffill")
             A    B   C
        0  NaN  2.0 NaN
        1  2.0  3.0 NaN
        2  NaN  NaN 2.0
        3  3.0  NaN 2.0
        4  3.0  NaN 2.0

        >>> df.groupby("key").fillna(method="bfill")
             A    B   C
        0  2.0  2.0 NaN
        1  2.0  3.0 NaN
        2  3.0  NaN 2.0
        3  3.0  NaN NaN
        4  NaN  NaN NaN

        Propagate non-null values forward or backward within each group along rows.

        >>> df.T.groupby(np.array([0, 0, 1, 1])).fillna(method="ffill").T
           key    A    B    C
        0  0.0  0.0  2.0  2.0
        1  0.0  2.0  3.0  3.0
        2  1.0  1.0  NaN  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  1.0  NaN  NaN

        >>> df.T.groupby(np.array([0, 0, 1, 1])).fillna(method="bfill").T
           key    A    B    C
        0  0.0  NaN  2.0  NaN
        1  0.0  2.0  3.0  NaN
        2  1.0  NaN  2.0  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  NaN  NaN  NaN

        Only replace the first NaN element within a group along rows.

        >>> df.groupby("key").fillna(method="ffill", limit=1)
             A    B    C
        0  NaN  2.0  NaN
        1  2.0  3.0  NaN
        2  NaN  NaN  2.0
        3  3.0  NaN  2.0
        4  3.0  NaN  NaN
        """
        warnings.warn(
            f"{type(self).__name__}.fillna is deprecated and "
            "will be removed in a future version. Use obj.ffill() or obj.bfill() "
            "for forward or backward filling instead. If you want to fill with a "
            f"single value, use {type(self.obj).__name__}.fillna instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

        result = self._op_via_apply(
            "fillna",
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )
        return result

    def take(
        self,
        indices: TakeIndexer,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        **kwargs,
    ) -> DataFrame:
        """
        Return the elements in the given *positional* indices in each group.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        If a requested index does not exist for some group, this method will raise.
        To get similar behavior that ignores indices that don't exist, see
        :meth:`.DataFrameGroupBy.nth`.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.

        Returns
        -------
        DataFrame
            An DataFrame containing the elements taken from each group.

        See Also
        --------
        DataFrame.take : Take elements from a Series along an axis.
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.

        Examples
        --------
        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan),
        ...                    ('rabbit', 'mammal', 15.0)],
        ...                   columns=['name', 'class', 'max_speed'],
        ...                   index=[4, 3, 2, 1, 0])
        >>> df
             name   class  max_speed
        4  falcon    bird      389.0
        3  parrot    bird       24.0
        2    lion  mammal       80.5
        1  monkey  mammal        NaN
        0  rabbit  mammal       15.0
        >>> gb = df.groupby([1, 1, 2, 2, 2])

        Take elements at positions 0 and 1 along the axis 0 (default).

        Note how the indices selected in the result do not correspond to
        our input indices 0 and 1. That's because we are selecting the 0th
        and 1st rows, not rows whose indices equal 0 and 1.

        >>> gb.take([0, 1])
               name   class  max_speed
        1 4  falcon    bird      389.0
          3  parrot    bird       24.0
        2 2    lion  mammal       80.5
          1  monkey  mammal        NaN

        The order of the specified indices influences the order in the result.
        Here, the order is swapped from the previous example.

        >>> gb.take([1, 0])
               name   class  max_speed
        1 3  parrot    bird       24.0
          4  falcon    bird      389.0
        2 1  monkey  mammal        NaN
          2    lion  mammal       80.5

        Take elements at indices 1 and 2 along the axis 1 (column selection).

        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.

        >>> gb.take([-1, -2])
               name   class  max_speed
        1 3  parrot    bird       24.0
          4  falcon    bird      389.0
        2 0  rabbit  mammal       15.0
          1  monkey  mammal        NaN
        """
        result = self._op_via_apply("take", indices=indices, axis=axis, **kwargs)
        return result

    def skew(
        self,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> DataFrame:
        """
        Return unbiased skew within groups.

        Normalized by N-1.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis for the function to be applied on.

            Specifying ``axis=None`` will apply the aggregation across both axes.

            .. versionadded:: 2.0.0

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.skew : Return unbiased skew over requested axis.

        Examples
        --------
        >>> arrays = [['falcon', 'parrot', 'cockatoo', 'kiwi',
        ...            'lion', 'monkey', 'rabbit'],
        ...           ['bird', 'bird', 'bird', 'bird',
        ...            'mammal', 'mammal', 'mammal']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=('name', 'class'))
        >>> df = pd.DataFrame({'max_speed': [389.0, 24.0, 70.0, np.nan,
        ...                                  80.5, 21.5, 15.0]},
        ...                   index=index)
        >>> df
                        max_speed
        name     class
        falcon   bird        389.0
        parrot   bird         24.0
        cockatoo bird         70.0
        kiwi     bird          NaN
        lion     mammal       80.5
        monkey   mammal       21.5
        rabbit   mammal       15.0
        >>> gb = df.groupby(["class"])
        >>> gb.skew()
                max_speed
        class
        bird     1.628296
        mammal   1.669046
        >>> gb.skew(skipna=False)
                max_speed
        class
        bird          NaN
        mammal   1.669046
        """
        if axis is lib.no_default:
            axis = 0

        if axis != 0:
            result = self._op_via_apply(
                "skew",
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )
            return result

        def alt(obj):
            # This should not be reached since the cython path should raise
            #  TypeError and not NotImplementedError.
            raise TypeError(f"'skew' is not supported for dtype={obj.dtype}")

        return self._cython_agg_general(
            "skew", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @property
    @doc(DataFrame.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(DataFrame.corr.__doc__)
    def corr(
        self,
        method: str | Callable[[np.ndarray, np.ndarray], float] = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result = self._op_via_apply(
            "corr", method=method, min_periods=min_periods, numeric_only=numeric_only
        )
        return result

    @doc(DataFrame.cov.__doc__)
    def cov(
        self,
        min_periods: int | None = None,
        ddof: int | None = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result = self._op_via_apply(
            "cov", min_periods=min_periods, ddof=ddof, numeric_only=numeric_only
        )
        return result

    @doc(DataFrame.hist.__doc__)
    def hist(
        self,
        column: IndexLabel | None = None,
        by=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        ax=None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[int, int] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        result = self._op_via_apply(
            "hist",
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    @property
    @doc(DataFrame.dtypes.__doc__)
    def dtypes(self) -> Series:
        # GH#51045
        warnings.warn(
            f"{type(self).__name__}.dtypes is deprecated and will be removed in "
            "a future version. Check the dtypes on the base object instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

        # error: Incompatible return value type (got "DataFrame", expected "Series")
        return self._python_apply_general(  # type: ignore[return-value]
            lambda df: df.dtypes, self._selected_obj
        )

    @doc(DataFrame.corrwith.__doc__)
    def corrwith(
        self,
        other: DataFrame | Series,
        axis: Axis | lib.NoDefault = lib.no_default,
        drop: bool = False,
        method: CorrelationMethod = "pearson",
        numeric_only: bool = False,
    ) -> DataFrame:
        result = self._op_via_apply(
            "corrwith",
            other=other,
            axis=axis,
            drop=drop,
            method=method,
            numeric_only=numeric_only,
        )
        return result


def _wrap_transform_general_frame(
    obj: DataFrame, group: DataFrame, res: DataFrame | Series
) -> DataFrame:
    from pandas import concat

    if isinstance(res, Series):
        # we need to broadcast across the
        # other dimension; this will preserve dtypes
        # GH14457
        if res.index.is_(obj.index):
            res_frame = concat([res] * len(group.columns), axis=1)
            res_frame.columns = group.columns
            res_frame.index = group.index
        else:
            res_frame = obj._constructor(
                np.tile(res.values, (len(group.index), 1)),
                columns=group.columns,
                index=group.index,
            )
        assert isinstance(res_frame, DataFrame)
        return res_frame
    elif isinstance(res, DataFrame) and not res.index.is_(group.index):
        return res._align_frame(group)[0]
    else:
        return res
<|endoftext|>from pandas.core.groupby.generic import (
    DataFrameGroupBy,
    NamedAgg,
    SeriesGroupBy,
)
from pandas.core.groupby.groupby import GroupBy
from pandas.core.groupby.grouper import Grouper

__all__ = [
    "DataFrameGroupBy",
    "NamedAgg",
    "SeriesGroupBy",
    "GroupBy",
    "Grouper",
]
<|endoftext|>"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    final,
)
import warnings

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import lib
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core import algorithms
from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
)
from pandas.core.series import Series

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas._typing import (
        ArrayLike,
        Axis,
        NDFrameT,
        npt,
    )

    from pandas.core.generic import NDFrame


class Grouper:
    """
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level and/or axis parameters are given, a level of the index of the target
    object.

    If `axis` and/or `level` are passed as keywords to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    axis : str, int, defaults to 0
        Number/name of the axis.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {'left' or 'right'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {'left' or 'right'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {'start', 'end', 'e', 's'}
        If grouper is PeriodIndex and `freq` parameter is passed.

    origin : Timestamp or str, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If string, must be one of the following:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries

        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day

        .. versionadded:: 1.3.0

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

    dropna : bool, default True
        If True, and if group keys contain NA values, NA values together with
        row/column will be dropped. If False, NA values will also be treated as
        the key in groups.

    Returns
    -------
    Grouper or pandas.api.typing.TimeGrouper
        A TimeGrouper is returned if ``freq`` is not ``None``. Otherwise, a Grouper
        is returned.

    Examples
    --------
    ``df.groupby(pd.Grouper(key="Animal"))`` is equivalent to ``df.groupby('Animal')``

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column 'Publish date'

    >>> df = pd.DataFrame(
    ...    {
    ...        "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16")
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40]
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = '2000-10-01 23:30:00', '2000-10-02 00:30:00'
    >>> rng = pd.date_range(start, end, freq='7min')
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
    2000-10-02 00:05:00    15
    2000-10-02 00:12:00    18
    2000-10-02 00:19:00    21
    2000-10-02 00:26:00    24
    Freq: 7min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min')).sum()
    2000-10-01 23:14:00     0
    2000-10-01 23:31:00     9
    2000-10-01 23:48:00    21
    2000-10-02 00:05:00    54
    2000-10-02 00:22:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', origin='epoch')).sum()
    2000-10-01 23:18:00     0
    2000-10-01 23:35:00    18
    2000-10-01 23:52:00    27
    2000-10-02 00:09:00    39
    2000-10-02 00:26:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', origin='2000-01-01')).sum()
    2000-10-01 23:24:00     3
    2000-10-01 23:41:00    15
    2000-10-01 23:58:00    45
    2000-10-02 00:15:00    45
    Freq: 17min, dtype: int64

    If you want to adjust the start of the bins with an `offset` Timedelta, the two
    following lines are equivalent:

    >>> ts.groupby(pd.Grouper(freq='17min', origin='start')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', offset='23h30min')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    To replace the use of the deprecated `base` argument, you can now use `offset`,
    in this example it is equivalent to have `base=2`:

    >>> ts.groupby(pd.Grouper(freq='17min', offset='2min')).sum()
    2000-10-01 23:16:00     0
    2000-10-01 23:33:00     9
    2000-10-01 23:50:00    36
    2000-10-02 00:07:00    39
    2000-10-02 00:24:00    24
    Freq: 17min, dtype: int64
    """

    sort: bool
    dropna: bool
    _gpr_index: Index | None
    _grouper: Index | None

    _attributes: tuple[str, ...] = ("key", "level", "freq", "axis", "sort", "dropna")

    def __new__(cls, *args, **kwargs):
        if kwargs.get("freq") is not None:
            from pandas.core.resample import TimeGrouper

            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(
        self,
        key=None,
        level=None,
        freq=None,
        axis: Axis | lib.NoDefault = lib.no_default,
        sort: bool = False,
        dropna: bool = True,
    ) -> None:
        if type(self) is Grouper:
            # i.e. not TimeGrouper
            if axis is not lib.no_default:
                warnings.warn(
                    "Grouper axis keyword is deprecated and will be removed in a "
                    "future version. To group on axis=1, use obj.T.groupby(...) "
                    "instead",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            else:
                axis = 0
        if axis is lib.no_default:
            axis = 0

        self.key = key
        self.level = level
        self.freq = freq
        self.axis = axis
        self.sort = sort
        self.dropna = dropna

        self._grouper_deprecated = None
        self._indexer_deprecated: npt.NDArray[np.intp] | None = None
        self._obj_deprecated = None
        self._gpr_index = None
        self.binner = None
        self._grouper = None
        self._indexer: npt.NDArray[np.intp] | None = None

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
    ) -> tuple[ops.BaseGrouper, NDFrameT]:
        """
        Parameters
        ----------
        obj : Series or DataFrame
        validate : bool, default True
            if True, validate the grouper

        Returns
        -------
        a tuple of grouper, obj (possibly sorted)
        """
        obj, _, _ = self._set_grouper(obj)
        grouper, _, obj = get_grouper(
            obj,
            [self.key],
            axis=self.axis,
            level=self.level,
            sort=self.sort,
            validate=validate,
            dropna=self.dropna,
        )
        # Without setting this, subsequent lookups to .groups raise
        # error: Incompatible types in assignment (expression has type "BaseGrouper",
        # variable has type "None")
        self._grouper_deprecated = grouper  # type: ignore[assignment]

        return grouper, obj

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        """
        given an object and the specifications, setup the internal grouper
        for this particular specification

        Parameters
        ----------
        obj : Series or DataFrame
        sort : bool, default False
            whether the resulting grouper should be sorted
        gpr_index : Index or None, default None

        Returns
        -------
        NDFrame
        Index
        np.ndarray[np.intp] | None
        """
        assert obj is not None

        if self.key is not None and self.level is not None:
            raise ValueError("The Grouper cannot specify both a key and a level!")

        # Keep self._grouper value before overriding
        if self._grouper is None:
            # TODO: What are we assuming about subsequent calls?
            self._grouper = gpr_index
            self._indexer = self._indexer_deprecated

        # the key must be a valid info item
        if self.key is not None:
            key = self.key
            # The 'on' is already defined
            if getattr(gpr_index, "name", None) == key and isinstance(obj, Series):
                # Sometimes self._grouper will have been resorted while
                # obj has not. In this case there is a mismatch when we
                # call self._grouper.take(obj.index) so we need to undo the sorting
                # before we call _grouper.take.
                assert self._grouper is not None
                if self._indexer is not None:
                    reverse_indexer = self._indexer.argsort()
                    unsorted_ax = self._grouper.take(reverse_indexer)
                    ax = unsorted_ax.take(obj.index)
                else:
                    ax = self._grouper.take(obj.index)
            else:
                if key not in obj._info_axis:
                    raise KeyError(f"The grouper name {key} is not found")
                ax = Index(obj[key], name=key)

        else:
            ax = obj._get_axis(self.axis)
            if self.level is not None:
                level = self.level

                # if a level is given it must be a mi level or
                # equivalent to the axis name
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax._get_level_values(level), name=ax.names[level])

                else:
                    if level not in (0, ax.name):
                        raise ValueError(f"The level {level} is not valid")

        # possibly sort
        indexer: npt.NDArray[np.intp] | None = None
        if (self.sort or sort) and not ax.is_monotonic_increasing:
            # use stable sort to support first, last, nth
            # TODO: why does putting na_position="first" fix datetimelike cases?
            indexer = self._indexer_deprecated = ax.array.argsort(
                kind="mergesort", na_position="first"
            )
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=self.axis)

        # error: Incompatible types in assignment (expression has type
        # "NDFrameT", variable has type "None")
        self._obj_deprecated = obj  # type: ignore[assignment]
        self._gpr_index = ax
        return obj, ax, indexer

    @final
    @property
    def ax(self) -> Index:
        warnings.warn(
            f"{type(self).__name__}.ax is deprecated and will be removed in a "
            "future version. Use Resampler.ax instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        index = self._gpr_index
        if index is None:
            raise ValueError("_set_grouper must be called before ax is accessed")
        return index

    @final
    @property
    def indexer(self):
        warnings.warn(
            f"{type(self).__name__}.indexer is deprecated and will be removed "
            "in a future version. Use Resampler.indexer instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._indexer_deprecated

    @final
    @property
    def obj(self):
        # TODO(3.0): enforcing these deprecations on Grouper should close
        #  GH#25564, GH#41930
        warnings.warn(
            f"{type(self).__name__}.obj is deprecated and will be removed "
            "in a future version. Use GroupBy.indexer instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._obj_deprecated

    @final
    @property
    def grouper(self):
        warnings.warn(
            f"{type(self).__name__}.grouper is deprecated and will be removed "
            "in a future version. Use GroupBy.grouper instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._grouper_deprecated

    @final
    @property
    def groups(self):
        warnings.warn(
            f"{type(self).__name__}.groups is deprecated and will be removed "
            "in a future version. Use GroupBy.groups instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        # error: "None" has no attribute "groups"
        return self._grouper_deprecated.groups  # type: ignore[attr-defined]

    @final
    def __repr__(self) -> str:
        attrs_list = (
            f"{attr_name}={repr(getattr(self, attr_name))}"
            for attr_name in self._attributes
            if getattr(self, attr_name) is not None
        )
        attrs = ", ".join(attrs_list)
        cls_name = type(self).__name__
        return f"{cls_name}({attrs})"


@final
class Grouping:
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
    grouper :
    obj : DataFrame or Series
    name : Label
    level :
    observed : bool, default False
        If we are a Categorical, use the observed values
    in_axis : if the Grouping is a column in self.obj and hence among
        Groupby.exclusions list
    dropna : bool, default True
        Whether to drop NA groups.
    uniques : Array-like, optional
        When specified, will be used for unique values. Enables including empty groups
        in the result for a BinGrouper. Must not contain duplicates.

    Attributes
    -------
    indices : dict
        Mapping of {group -> index_list}
    codes : ndarray
        Group codes
    group_index : Index or None
        unique groups
    groups : dict
        Mapping of {group -> label_list}
    """

    _codes: npt.NDArray[np.signedinteger] | None = None
    _all_grouper: Categorical | None
    _orig_cats: Index | None
    _index: Index

    def __init__(
        self,
        index: Index,
        grouper=None,
        obj: NDFrame | None = None,
        level=None,
        sort: bool = True,
        observed: bool = False,
        in_axis: bool = False,
        dropna: bool = True,
        uniques: ArrayLike | None = None,
    ) -> None:
        self.level = level
        self._orig_grouper = grouper
        grouping_vector = _convert_grouper(index, grouper)
        self._all_grouper = None
        self._orig_cats = None
        self._index = index
        self._sort = sort
        self.obj = obj
        self._observed = observed
        self.in_axis = in_axis
        self._dropna = dropna
        self._uniques = uniques

        # we have a single grouper which may be a myriad of things,
        # some of which are dependent on the passing in level

        ilevel = self._ilevel
        if ilevel is not None:
            # In extant tests, the new self.grouping_vector matches
            #  `index.get_level_values(ilevel)` whenever
            #  mapper is None and isinstance(index, MultiIndex)
            if isinstance(index, MultiIndex):
                index_level = index.get_level_values(ilevel)
            else:
                index_level = index

            if grouping_vector is None:
                grouping_vector = index_level
            else:
                mapper = grouping_vector
                grouping_vector = index_level.map(mapper)

        # a passed Grouper like, directly get the grouper in the same way
        # as single grouper groupby, use the group_info to get codes
        elif isinstance(grouping_vector, Grouper):
            # get the new grouper; we already have disambiguated
            # what key/level refer to exactly, don't need to
            # check again as we have by this point converted these
            # to an actual value (rather than a pd.Grouper)
            assert self.obj is not None  # for mypy
            newgrouper, newobj = grouping_vector._get_grouper(self.obj, validate=False)
            self.obj = newobj

            if isinstance(newgrouper, ops.BinGrouper):
                # TODO: can we unwrap this and get a tighter typing
                #  for self.grouping_vector?
                grouping_vector = newgrouper
            else:
                # ops.BaseGrouper
                # TODO: 2023-02-03 no test cases with len(newgrouper.groupings) > 1.
                #  If that were to occur, would we be throwing out information?
                # error: Cannot determine type of "grouping_vector"  [has-type]
                ng = newgrouper.groupings[0].grouping_vector  # type: ignore[has-type]
                # use Index instead of ndarray so we can recover the name
                grouping_vector = Index(ng, name=newgrouper.result_index.name)

        elif not isinstance(
            grouping_vector, (Series, Index, ExtensionArray, np.ndarray)
        ):
            # no level passed
            if getattr(grouping_vector, "ndim", 1) != 1:
                t = str(type(grouping_vector))
                raise ValueError(f"Grouper for '{t}' not 1-dimensional")

            grouping_vector = index.map(grouping_vector)

            if not (
                hasattr(grouping_vector, "__len__")
                and len(grouping_vector) == len(index)
            ):
                grper = pprint_thing(grouping_vector)
                errmsg = (
                    "Grouper result violates len(labels) == "
                    f"len(data)\nresult: {grper}"
                )
                raise AssertionError(errmsg)

        if isinstance(grouping_vector, np.ndarray):
            if grouping_vector.dtype.kind in "mM":
                # if we have a date/time-like grouper, make sure that we have
                # Timestamps like
                # TODO 2022-10-08 we only have one test that gets here and
                #  values are already in nanoseconds in that case.
                grouping_vector = Series(grouping_vector).to_numpy()
        elif isinstance(getattr(grouping_vector, "dtype", None), CategoricalDtype):
            # a passed Categorical
            self._orig_cats = grouping_vector.categories
            grouping_vector, self._all_grouper = recode_for_groupby(
                grouping_vector, sort, observed
            )

        self.grouping_vector = grouping_vector

    def __repr__(self) -> str:
        return f"Grouping({self.name})"

    def __iter__(self) -> Iterator:
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        dtype = getattr(self.grouping_vector, "dtype", None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Hashable:
        ilevel = self._ilevel
        if ilevel is not None:
            return self._index.names[ilevel]

        if isinstance(self._orig_grouper, (Index, Series)):
            return self._orig_grouper.name

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.result_index.name

        elif isinstance(self.grouping_vector, Index):
            return self.grouping_vector.name

        # otherwise we have ndarray or ExtensionArray -> no name
        return None

    @cache_readonly
    def _ilevel(self) -> int | None:
        """
        If necessary, converted index level name to index level position.
        """
        level = self.level
        if level is None:
            return None
        if not isinstance(level, int):
            index = self._index
            if level not in index.names:
                raise AssertionError(f"Level {level} not in index")
            return index.names.index(level)
        return level

    @property
    def ngroups(self) -> int:
        return len(self._group_index)

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        # we have a list of groupers
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.indices

        values = Categorical(self.grouping_vector)
        return values._reverse_indexer()

    @property
    def codes(self) -> npt.NDArray[np.signedinteger]:
        return self._codes_and_uniques[0]

    @cache_readonly
    def _group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
        if self._all_grouper is not None:
            # retain dtype for categories, including unobserved ones
            return self._result_index._values

        elif self._passed_categorical:
            return self._group_index._values

        return self._codes_and_uniques[1]

    @property
    def group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
        warnings.warn(
            "group_arraylike is deprecated and will be removed in a future "
            "version of pandas",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._group_arraylike

    @cache_readonly
    def _result_index(self) -> Index:
        # result_index retains dtype for categories, including unobserved ones,
        #  which group_index does not
        if self._all_grouper is not None:
            group_idx = self._group_index
            assert isinstance(group_idx, CategoricalIndex)
            cats = self._orig_cats
            # set_categories is dynamically added
            return group_idx.set_categories(cats)  # type: ignore[attr-defined]
        return self._group_index

    @property
    def result_index(self) -> Index:
        warnings.warn(
            "result_index is deprecated and will be removed in a future "
            "version of pandas",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._result_index

    @cache_readonly
    def _group_index(self) -> Index:
        codes, uniques = self._codes_and_uniques
        if not self._dropna and self._passed_categorical:
            assert isinstance(uniques, Categorical)
            if self._sort and (codes == len(uniques)).any():
                # Add NA value on the end when sorting
                uniques = Categorical.from_codes(
                    np.append(uniques.codes, [-1]), uniques.categories, validate=False
                )
            elif len(codes) > 0:
                # Need to determine proper placement of NA value when not sorting
                cat = self.grouping_vector
                na_idx = (cat.codes < 0).argmax()
                if cat.codes[na_idx] < 0:
                    # count number of unique codes that comes before the nan value
                    na_unique_idx = algorithms.nunique_ints(cat.codes[:na_idx])
                    new_codes = np.insert(uniques.codes, na_unique_idx, -1)
                    uniques = Categorical.from_codes(
                        new_codes, uniques.categories, validate=False
                    )
        return Index._with_infer(uniques, name=self.name)

    @property
    def group_index(self) -> Index:
        warnings.warn(
            "group_index is deprecated and will be removed in a future "
            "version of pandas",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._group_index

    @cache_readonly
    def _codes_and_uniques(self) -> tuple[npt.NDArray[np.signedinteger], ArrayLike]:
        uniques: ArrayLike
        if self._passed_categorical:
            # we make a CategoricalIndex out of the cat grouper
            # preserving the categories / ordered attributes;
            # doesn't (yet - GH#46909) handle dropna=False
            cat = self.grouping_vector
            categories = cat.categories

            if self._observed:
                ucodes = algorithms.unique1d(cat.codes)
                ucodes = ucodes[ucodes != -1]
                if self._sort:
                    ucodes = np.sort(ucodes)
            else:
                ucodes = np.arange(len(categories))

            uniques = Categorical.from_codes(
                codes=ucodes, categories=categories, ordered=cat.ordered, validate=False
            )

            codes = cat.codes
            if not self._dropna:
                na_mask = codes < 0
                if np.any(na_mask):
                    if self._sort:
                        # Replace NA codes with `largest code + 1`
                        na_code = len(categories)
                        codes = np.where(na_mask, na_code, codes)
                    else:
                        # Insert NA code into the codes based on first appearance
                        # A negative code must exist, no need to check codes[na_idx] < 0
                        na_idx = na_mask.argmax()
                        # count number of unique codes that comes before the nan value
                        na_code = algorithms.nunique_ints(codes[:na_idx])
                        codes = np.where(codes >= na_code, codes + 1, codes)
                        codes = np.where(na_mask, na_code, codes)

            if not self._observed:
                uniques = uniques.reorder_categories(self._orig_cats)

            return codes, uniques

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            # we have a list of groupers
            codes = self.grouping_vector.codes_info
            uniques = self.grouping_vector.result_index._values
        elif self._uniques is not None:
            # GH#50486 Code grouping_vector using _uniques; allows
            # including uniques that are not present in grouping_vector.
            cat = Categorical(self.grouping_vector, categories=self._uniques)
            codes = cat.codes
            uniques = self._uniques
        else:
            # GH35667, replace dropna=False with use_na_sentinel=False
            # error: Incompatible types in assignment (expression has type "Union[
            # ndarray[Any, Any], Index]", variable has type "Categorical")
            codes, uniques = algorithms.factorize(  # type: ignore[assignment]
                self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna
            )
        return codes, uniques

    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]:
        cats = Categorical.from_codes(self.codes, self._group_index, validate=False)
        return self._index.groupby(cats)


def get_grouper(
    obj: NDFrameT,
    key=None,
    axis: Axis = 0,
    level=None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True,
) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to axis,level,sort, while
    the passed in axis, level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    group_axis = obj._get_axis(axis)

    # validate that the passed single level is compatible with the passed
    # axis of the object
    if level is not None:
        # TODO: These if-block and else-block are almost same.
        # MultiIndex instance check is removable, but it seems that there are
        # some processes only for non-MultiIndex in else-block,
        # eg. `obj.index.name != level`. We have to consider carefully whether
        # these are applicable for MultiIndex. Even if these are applicable,
        # we need to check if it makes no side effect to subsequent processes
        # on the outside of this condition.
        # (GH 17621)
        if isinstance(group_axis, MultiIndex):
            if is_list_like(level) and len(level) == 1:
                level = level[0]

            if key is None and is_scalar(level):
                # Get the level values from group_axis
                key = group_axis.get_level_values(level)
                level = None

        else:
            # allow level to be a length-one list-like object
            # (e.g., level=[0])
            # GH 13901
            if is_list_like(level):
                nlevels = len(level)
                if nlevels == 1:
                    level = level[0]
                elif nlevels == 0:
                    raise ValueError("No group keys passed!")
                else:
                    raise ValueError("multiple levels only valid with MultiIndex")

            if isinstance(level, str):
                if obj._get_axis(axis).name != level:
                    raise ValueError(
                        f"level name {level} is not the name "
                        f"of the {obj._get_axis_name(axis)}"
                    )
            elif level > 0 or level < -1:
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")

            # NOTE: `group_axis` and `group_axis.get_level_values(level)`
            # are same in this section.
            level = None
            key = group_axis

    # a passed-in Grouper, directly convert
    if isinstance(key, Grouper):
        grouper, obj = key._get_grouper(obj, validate=False)
        if key.key is None:
            return grouper, frozenset(), obj
        else:
            return grouper, frozenset({key.key}), obj

    # already have a BaseGrouper, just return it
    elif isinstance(key, ops.BaseGrouper):
        return key, frozenset(), obj

    if not isinstance(key, list):
        keys = [key]
        match_axis_length = False
    else:
        keys = key
        match_axis_length = len(keys) == len(group_axis)

    # what are we after, exactly?
    any_callable = any(callable(g) or isinstance(g, dict) for g in keys)
    any_groupers = any(isinstance(g, (Grouper, Grouping)) for g in keys)
    any_arraylike = any(
        isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys
    )

    # is this an index replacement?
    if (
        not any_callable
        and not any_arraylike
        and not any_groupers
        and match_axis_length
        and level is None
    ):
        if isinstance(obj, DataFrame):
            all_in_columns_index = all(
                g in obj.columns or g in obj.index.names for g in keys
            )
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all(g in obj.index.names for g in keys)

        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]

    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        levels = [level] * len(keys)

    groupings: list[Grouping] = []
    exclusions: set[Hashable] = set()

    # if the actual grouper should be obj[key]
    def is_in_axis(key) -> bool:
        if not _is_label_like(key):
            if obj.ndim == 1:
                return False

            # items -> .columns for DataFrame, .index for Series
            items = obj.axes[-1]
            try:
                items.get_loc(key)
            except (KeyError, TypeError, InvalidIndexError):
                # TypeError shows up here if we pass e.g. an Index
                return False

        return True

    # if the grouper is obj[name]
    def is_in_obj(gpr) -> bool:
        if not hasattr(gpr, "name"):
            return False
        if using_copy_on_write() or warn_copy_on_write():
            # For the CoW case, we check the references to determine if the
            # series is part of the object
            try:
                obj_gpr_column = obj[gpr.name]
            except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
                return False
            if isinstance(gpr, Series) and isinstance(obj_gpr_column, Series):
                return gpr._mgr.references_same_values(  # type: ignore[union-attr]
                    obj_gpr_column._mgr, 0  # type: ignore[arg-type]
                )
            return False
        try:
            return gpr is obj[gpr.name]
        except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
            # IndexError reached in e.g. test_skip_group_keys when we pass
            #  lambda here
            # InvalidIndexError raised on key-types inappropriate for index,
            #  e.g. DatetimeIndex.get_loc(tuple())
            # OutOfBoundsDatetime raised when obj is a Series with DatetimeIndex
            # and gpr.name is month str
            return False

    for gpr, level in zip(keys, levels):
        if isinstance(obj, DataFrame) and is_in_obj(gpr):  # df.groupby(df['name'])
            in_axis = True
            exclusions.add(gpr.name)

        elif is_in_axis(gpr):  # df.groupby('name')
            if obj.ndim != 1 and gpr in obj:
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=axis)
                in_axis, name, gpr = True, gpr, obj[gpr]
                if gpr.ndim != 1:
                    # non-unique columns; raise here to get the name in the
                    # exception message
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=axis):
                in_axis, level, gpr = False, gpr, None
            else:
                raise KeyError(gpr)
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            # Add key to exclusions
            exclusions.add(gpr.key)
            in_axis = True
        else:
            in_axis = False

        # create the Grouping
        # allow us to passing the actual Grouping as the gpr
        ping = (
            Grouping(
                group_axis,
                gpr,
                obj=obj,
                level=level,
                sort=sort,
                observed=observed,
                in_axis=in_axis,
                dropna=dropna,
            )
            if not isinstance(gpr, Grouping)
            else gpr
        )

        groupings.append(ping)

    if len(groupings) == 0 and len(obj):
        raise ValueError("No group keys passed!")
    if len(groupings) == 0:
        groupings.append(Grouping(Index([], dtype="int"), np.array([], dtype=np.intp)))

    # create the internals grouper
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, dropna=dropna)
    return grouper, frozenset(exclusions), obj


def _is_label_like(val) -> bool:
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))


def _convert_grouper(axis: Index, grouper):
    if isinstance(grouper, dict):
        return grouper.get
    elif isinstance(grouper, Series):
        if grouper.index.equals(axis):
            return grouper._values
        else:
            return grouper.reindex(axis)._values
    elif isinstance(grouper, MultiIndex):
        return grouper._values
    elif isinstance(grouper, (list, tuple, Index, Categorical, np.ndarray)):
        if len(grouper) != len(axis):
            raise ValueError("Grouper and axis must be same length")

        if isinstance(grouper, (list, tuple)):
            grouper = com.asarray_tuplesafe(grouper)
        return grouper
    else:
        return grouper
<|endoftext|>"""Common utilities for Numba operations with groupby ops"""
from __future__ import annotations

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.util.numba_ import (
    NumbaUtilError,
    jit_user_function,
)

if TYPE_CHECKING:
    from pandas._typing import Scalar


def validate_udf(func: Callable) -> None:
    """
    Validate user defined function for ops when using Numba with groupby ops.

    The first signature arguments should include:

    def f(values, index, ...):
        ...

    Parameters
    ----------
    func : function, default False
        user defined function

    Returns
    -------
    None

    Raises
    ------
    NumbaUtilError
    """
    if not callable(func):
        raise NotImplementedError(
            "Numba engine can only be used with a single function."
        )
    udf_signature = list(inspect.signature(func).parameters.keys())
    expected_args = ["values", "index"]
    min_number_args = len(expected_args)
    if (
        len(udf_signature) < min_number_args
        or udf_signature[:min_number_args] != expected_args
    ):
        raise NumbaUtilError(
            f"The first {min_number_args} arguments to {func.__name__} must be "
            f"{expected_args}"
        )


@functools.cache
def generate_numba_agg_func(
    func: Callable[..., Scalar],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    """
    Generate a numba jitted agg function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby agg function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each group and will be JITed
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_agg(
        values: np.ndarray,
        index: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        num_columns: int,
        *args: Any,
    ) -> np.ndarray:
        assert len(begin) == len(end)
        num_groups = len(begin)

        result = np.empty((num_groups, num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i] : end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i] : end[i], j]
                result[i, j] = numba_func(group, group_index, *args)
        return result

    return group_agg


@functools.cache
def generate_numba_transform_func(
    func: Callable[..., np.ndarray],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    """
    Generate a numba jitted transform function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby transform function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each window and will be JITed
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_transform(
        values: np.ndarray,
        index: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        num_columns: int,
        *args: Any,
    ) -> np.ndarray:
        assert len(begin) == len(end)
        num_groups = len(begin)

        result = np.empty((len(values), num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i] : end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i] : end[i], j]
                result[begin[i] : end[i], j] = numba_func(group, group_index, *args)
        return result

    return group_transform
<|endoftext|>from __future__ import annotations

import numpy as np

from pandas.core.algorithms import unique1d
from pandas.core.arrays.categorical import (
    Categorical,
    CategoricalDtype,
    recode_for_categories,
)


def recode_for_groupby(
    c: Categorical, sort: bool, observed: bool
) -> tuple[Categorical, Categorical | None]:
    """
    Code the categories to ensure we can groupby for categoricals.

    If observed=True, we return a new Categorical with the observed
    categories only.

    If sort=False, return a copy of self, coded with categories as
    returned by .unique(), followed by any categories not appearing in
    the data. If sort=True, return self.

    This method is needed solely to ensure the categorical index of the
    GroupBy result has categories in the order of appearance in the data
    (GH-8868).

    Parameters
    ----------
    c : Categorical
    sort : bool
        The value of the sort parameter groupby was called with.
    observed : bool
        Account only for the observed values

    Returns
    -------
    Categorical
        If sort=False, the new categories are set to the order of
        appearance in codes (unless ordered=True, in which case the
        original order is preserved), followed by any unrepresented
        categories in the original order.
    Categorical or None
        If we are observed, return the original categorical, otherwise None
    """
    # we only care about observed values
    if observed:
        # In cases with c.ordered, this is equivalent to
        #  return c.remove_unused_categories(), c

        unique_codes = unique1d(c.codes)

        take_codes = unique_codes[unique_codes != -1]
        if sort:
            take_codes = np.sort(take_codes)

        # we recode according to the uniques
        categories = c.categories.take(take_codes)
        codes = recode_for_categories(c.codes, c.categories, categories)

        # return a new categorical that maps our new codes
        # and categories
        dtype = CategoricalDtype(categories, ordered=c.ordered)
        return Categorical._simple_new(codes, dtype=dtype), c

    # Already sorted according to c.categories; all is fine
    if sort:
        return c, None

    # sort=False should order groups in as-encountered order (GH-8868)

    # xref GH:46909: Re-ordering codes faster than using (set|add|reorder)_categories
    all_codes = np.arange(c.categories.nunique())
    # GH 38140: exclude nan from indexer for categories
    unique_notnan_codes = unique1d(c.codes[c.codes != -1])
    if sort:
        unique_notnan_codes = np.sort(unique_notnan_codes)
    if len(all_codes) > len(unique_notnan_codes):
        # GH 13179: All categories need to be present, even if missing from the data
        missing_codes = np.setdiff1d(all_codes, unique_notnan_codes, assume_unique=True)
        take_codes = np.concatenate((unique_notnan_codes, missing_codes))
    else:
        take_codes = unique_notnan_codes

    return Categorical(c, c.unique().categories.take(take_codes)), None
<|endoftext|>"""
Provide classes to perform the groupby aggregate operations.

These are not exposed to the user and provide implementations of the grouping
operations, primarily in cython. These classes (BaseGrouper and BinGrouper)
are contained *in* the SeriesGroupBy and DataFrameGroupBy objects.
"""
from __future__ import annotations

import collections
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    final,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
import pandas._libs.groupby as libgroupby
from pandas._typing import (
    ArrayLike,
    AxisInt,
    NDFrameT,
    Shape,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.cast import (
    maybe_cast_pointwise_result,
    maybe_downcast_to_dtype,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_int64,
    ensure_platform_int,
    ensure_uint64,
    is_1d_only_ea_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    maybe_fill,
)

from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
    ensure_index,
)
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_flattened_list,
    get_group_index,
    get_group_index_sorter,
    get_indexer_dict,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Sequence,
    )

    from pandas.core.generic import NDFrame


def check_result_array(obj, dtype) -> None:
    # Our operation is supposed to be an aggregation/reduction. If
    #  it returns an ndarray, this likely means an invalid operation has
    #  been passed. See test_apply_without_aggregation, test_agg_must_agg
    if isinstance(obj, np.ndarray):
        if dtype != object:
            # If it is object dtype, the function can be a reduction/aggregation
            #  and still return an ndarray e.g. test_agg_over_numpy_arrays
            raise ValueError("Must produce aggregated value")


def extract_result(res):
    """
    Extract the result object, it might be a 0-dim ndarray
    or a len-1 0-dim, or a scalar
    """
    if hasattr(res, "_values"):
        # Preserve EA
        res = res._values
        if res.ndim == 1 and len(res) == 1:
            # see test_agg_lambda_with_timezone, test_resampler_grouper.py::test_apply
            res = res[0]
    return res


class WrappedCythonOp:
    """
    Dispatch logic for functions defined in _libs.groupby

    Parameters
    ----------
    kind: str
        Whether the operation is an aggregate or transform.
    how: str
        Operation name, e.g. "mean".
    has_dropped_na: bool
        True precisely when dropna=True and the grouper contains a null value.
    """

    # Functions for which we do _not_ attempt to cast the cython result
    #  back to the original dtype.
    cast_blocklist = frozenset(
        ["any", "all", "rank", "count", "size", "idxmin", "idxmax"]
    )

    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na

    _CYTHON_FUNCTIONS: dict[str, dict] = {
        "aggregate": {
            "any": functools.partial(libgroupby.group_any_all, val_test="any"),
            "all": functools.partial(libgroupby.group_any_all, val_test="all"),
            "sum": "group_sum",
            "prod": "group_prod",
            "idxmin": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmin"),
            "idxmax": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmax"),
            "min": "group_min",
            "max": "group_max",
            "mean": "group_mean",
            "median": "group_median_float64",
            "var": "group_var",
            "std": functools.partial(libgroupby.group_var, name="std"),
            "sem": functools.partial(libgroupby.group_var, name="sem"),
            "skew": "group_skew",
            "first": "group_nth",
            "last": "group_last",
            "ohlc": "group_ohlc",
        },
        "transform": {
            "cumprod": "group_cumprod",
            "cumsum": "group_cumsum",
            "cummin": "group_cummin",
            "cummax": "group_cummax",
            "rank": "group_rank",
        },
    }

    _cython_arity = {"ohlc": 4}  # OHLC

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS["aggregate"]:
            return "aggregate"
        return "transform"

    # Note: we make this a classmethod and pass kind+how so that caching
    #  works at the class level and not the instance level
    @classmethod
    @functools.cache
    def _get_cython_function(
        cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool
    ):
        dtype_str = dtype.name
        ftype = cls._CYTHON_FUNCTIONS[kind][how]

        # see if there is a fused-type version of function
        # only valid for numeric
        if callable(ftype):
            f = ftype
        else:
            f = getattr(libgroupby, ftype)
        if is_numeric:
            return f
        elif dtype == np.dtype(object):
            if how in ["median", "cumprod"]:
                # no fused types -> no __signatures__
                raise NotImplementedError(
                    f"function is not implemented for this dtype: "
                    f"[how->{how},dtype->{dtype_str}]"
                )
            elif how in ["std", "sem", "idxmin", "idxmax"]:
                # We have a partial object that does not have __signatures__
                return f
            elif how == "skew":
                # _get_cython_vals will convert to float64
                pass
            elif "object" not in f.__signatures__:
                # raise NotImplementedError here rather than TypeError later
                raise NotImplementedError(
                    f"function is not implemented for this dtype: "
                    f"[how->{how},dtype->{dtype_str}]"
                )
            return f
        else:
            raise NotImplementedError(
                "This should not be reached. Please report a bug at "
                "github.com/pandas-dev/pandas/",
                dtype,
            )

    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
        how = self.how

        if how in ["median", "std", "sem", "skew"]:
            # median only has a float64 implementation
            # We should only get here with is_numeric, as non-numeric cases
            #  should raise in _get_cython_function
            values = ensure_float64(values)

        elif values.dtype.kind in "iu":
            if how in ["var", "mean"] or (
                self.kind == "transform" and self.has_dropped_na
            ):
                # has_dropped_na check need for test_null_group_str_transformer
                # result may still include NaN, so we have to cast
                values = ensure_float64(values)

            elif how in ["sum", "ohlc", "prod", "cumsum", "cumprod"]:
                # Avoid overflow during group op
                if values.dtype.kind == "i":
                    values = ensure_int64(values)
                else:
                    values = ensure_uint64(values)

        return values

    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape:
        how = self.how
        kind = self.kind

        arity = self._cython_arity.get(how, 1)

        out_shape: Shape
        if how == "ohlc":
            out_shape = (ngroups, arity)
        elif arity > 1:
            raise NotImplementedError(
                "arity of more than 1 is not supported for the 'how' argument"
            )
        elif kind == "transform":
            out_shape = values.shape
        else:
            out_shape = (ngroups,) + values.shape[1:]
        return out_shape

    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        how = self.how

        if how == "rank":
            out_dtype = "float64"
        elif how in ["idxmin", "idxmax"]:
            # The Cython implementation only produces the row number; we'll take
            # from the index using this in post processing
            out_dtype = "intp"
        else:
            if dtype.kind in "iufcb":
                out_dtype = f"{dtype.kind}{dtype.itemsize}"
            else:
                out_dtype = "object"
        return np.dtype(out_dtype)

    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        """
        Get the desired dtype of a result based on the
        input dtype and how it was computed.

        Parameters
        ----------
        dtype : np.dtype

        Returns
        -------
        np.dtype
            The desired dtype of the result.
        """
        how = self.how

        if how in ["sum", "cumsum", "sum", "prod", "cumprod"]:
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ["mean", "median", "var", "std", "sem"]:
            if dtype.kind in "fc":
                return dtype
            elif dtype.kind in "iub":
                return np.dtype(np.float64)
        return dtype

    @final
    def _cython_op_ndim_compat(
        self,
        values: np.ndarray,
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None = None,
        result_mask: npt.NDArray[np.bool_] | None = None,
        initial: Any = 0,
        **kwargs,
    ) -> np.ndarray:
        if values.ndim == 1:
            # expand to 2d, dispatch, then squeeze if appropriate
            values2d = values[None, :]
            if mask is not None:
                mask = mask[None, :]
            if result_mask is not None:
                result_mask = result_mask[None, :]
            res = self._call_cython_op(
                values2d,
                min_count=min_count,
                ngroups=ngroups,
                comp_ids=comp_ids,
                mask=mask,
                result_mask=result_mask,
                initial=initial,
                **kwargs,
            )
            if res.shape[0] == 1:
                return res[0]

            # otherwise we have OHLC
            return res.T

        return self._call_cython_op(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=mask,
            result_mask=result_mask,
            initial=initial,
            **kwargs,
        )

    @final
    def _call_cython_op(
        self,
        values: np.ndarray,  # np.ndarray[ndim=2]
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None,
        result_mask: npt.NDArray[np.bool_] | None,
        initial: Any = 0,
        **kwargs,
    ) -> np.ndarray:  # np.ndarray[ndim=2]
        orig_values = values

        dtype = values.dtype
        is_numeric = dtype.kind in "iufcb"

        is_datetimelike = dtype.kind in "mM"

        if is_datetimelike:
            values = values.view("int64")
            is_numeric = True
        elif dtype.kind == "b":
            values = values.view("uint8")
        if values.dtype == "float16":
            values = values.astype(np.float32)

        if self.how in ["any", "all"]:
            if mask is None:
                mask = isna(values)
            if dtype == object:
                if kwargs["skipna"]:
                    # GH#37501: don't raise on pd.NA when skipna=True
                    if mask.any():
                        # mask on original values computed separately
                        values = values.copy()
                        values[mask] = True
            values = values.astype(bool, copy=False).view(np.int8)
            is_numeric = True

        values = values.T
        if mask is not None:
            mask = mask.T
            if result_mask is not None:
                result_mask = result_mask.T

        out_shape = self._get_output_shape(ngroups, values)
        func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
        values = self._get_cython_vals(values)
        out_dtype = self._get_out_dtype(values.dtype)

        result = maybe_fill(np.empty(out_shape, dtype=out_dtype))
        if self.kind == "aggregate":
            counts = np.zeros(ngroups, dtype=np.int64)
            if self.how in [
                "idxmin",
                "idxmax",
                "min",
                "max",
                "mean",
                "last",
                "first",
                "sum",
            ]:
                if self.how == "sum":
                    # pass in through kwargs only for sum (other functions don't have
                    # the keyword)
                    kwargs["initial"] = initial
                func(
                    out=result,
                    counts=counts,
                    values=values,
                    labels=comp_ids,
                    min_count=min_count,
                    mask=mask,
                    result_mask=result_mask,
                    is_datetimelike=is_datetimelike,
                    **kwargs,
                )
            elif self.how in ["sem", "std", "var", "ohlc", "prod", "median"]:
                if self.how in ["std", "sem"]:
                    kwargs["is_datetimelike"] = is_datetimelike
                func(
                    result,
                    counts,
                    values,
                    comp_ids,
                    min_count=min_count,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
            elif self.how in ["any", "all"]:
                func(
                    out=result,
                    values=values,
                    labels=comp_ids,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
                result = result.astype(bool, copy=False)
            elif self.how in ["skew"]:
                func(
                    out=result,
                    counts=counts,
                    values=values,
                    labels=comp_ids,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
                if dtype == object:
                    result = result.astype(object)

            else:
                raise NotImplementedError(f"{self.how} is not implemented")
        else:
            # TODO: min_count
            if self.how != "rank":
                # TODO: should rank take result_mask?
                kwargs["result_mask"] = result_mask
            func(
                out=result,
                values=values,
                labels=comp_ids,
                ngroups=ngroups,
                is_datetimelike=is_datetimelike,
                mask=mask,
                **kwargs,
            )

        if self.kind == "aggregate" and self.how not in ["idxmin", "idxmax"]:
            # i.e. counts is defined.  Locations where count<min_count
            # need to have the result set to np.nan, which may require casting,
            # see GH#40767. For idxmin/idxmax is handled specially via post-processing
            if result.dtype.kind in "iu" and not is_datetimelike:
                # if the op keeps the int dtypes, we have to use 0
                cutoff = max(0 if self.how in ["sum", "prod"] else 1, min_count)
                empty_groups = counts < cutoff
                if empty_groups.any():
                    if result_mask is not None:
                        assert result_mask[empty_groups].all()
                    else:
                        # Note: this conversion could be lossy, see GH#40767
                        result = result.astype("float64")
                        result[empty_groups] = np.nan

        result = result.T

        if self.how not in self.cast_blocklist:
            # e.g. if we are int64 and need to restore to datetime64/timedelta64
            # "rank" is the only member of cast_blocklist we get here
            # Casting only needed for float16, bool, datetimelike,
            #  and self.how in ["sum", "prod", "ohlc", "cumprod"]
            res_dtype = self._get_result_dtype(orig_values.dtype)
            op_result = maybe_downcast_to_dtype(result, res_dtype)
        else:
            op_result = result

        return op_result

    @final
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None:
        if values.ndim > 2:
            raise NotImplementedError("number of dimensions is currently limited to 2")
        if values.ndim == 2:
            assert axis == 1, axis
        elif not is_1d_only_ea_dtype(values.dtype):
            # Note: it is *not* the case that axis is always 0 for 1-dim values,
            #  as we can have 1D ExtensionArrays that we need to treat as 2D
            assert axis == 0

    @final
    def cython_operation(
        self,
        *,
        values: ArrayLike,
        axis: AxisInt,
        min_count: int = -1,
        comp_ids: np.ndarray,
        ngroups: int,
        **kwargs,
    ) -> ArrayLike:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """
        self._validate_axis(axis, values)

        if not isinstance(values, np.ndarray):
            # i.e. ExtensionArray
            return values._groupby_op(
                how=self.how,
                has_dropped_na=self.has_dropped_na,
                min_count=min_count,
                ngroups=ngroups,
                ids=comp_ids,
                **kwargs,
            )

        return self._cython_op_ndim_compat(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=None,
            **kwargs,
        )


class BaseGrouper:
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """

    axis: Index

    def __init__(
        self,
        axis: Index,
        groupings: Sequence[grouper.Grouping],
        sort: bool = True,
        dropna: bool = True,
    ) -> None:
        assert isinstance(axis, Index), axis

        self.axis = axis
        self._groupings: list[grouper.Grouping] = list(groupings)
        self._sort = sort
        self.dropna = dropna

    @property
    def groupings(self) -> list[grouper.Grouping]:
        return self._groupings

    @property
    def shape(self) -> Shape:
        return tuple(ping.ngroups for ping in self.groupings)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        return len(self.groupings)

    def get_iterator(
        self, data: NDFrameT, axis: AxisInt = 0
    ) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        splitter = self._get_splitter(data, axis=axis)
        keys = self.group_keys_seq
        yield from zip(keys, splitter)

    @final
    def _get_splitter(self, data: NDFrame, axis: AxisInt = 0) -> DataSplitter:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
        ids, _, ngroups = self.group_info
        return _get_splitter(
            data,
            ids,
            ngroups,
            sorted_ids=self._sorted_ids,
            sort_idx=self._sort_idx,
            axis=axis,
        )

    @final
    @cache_readonly
    def group_keys_seq(self):
        if len(self.groupings) == 1:
            return self.levels[0]
        else:
            ids, _, ngroups = self.group_info

            # provide "flattened" iterator for multi-group setting
            return get_flattened_list(ids, ngroups, self.levels, self.codes)

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """dict {group name -> group indices}"""
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            # This shows unused categories in indices GH#38642
            return self.groupings[0].indices
        codes_list = [ping.codes for ping in self.groupings]
        keys = [ping._group_index for ping in self.groupings]
        return get_indexer_dict(codes_list, keys)

    @final
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        """
        Get the original integer locations of result_index in the input.
        """
        # Original indices are where group_index would go via sorting.
        # But when dropna is true, we need to remove null values while accounting for
        # any gaps that then occur because of them.
        group_index = get_group_index(
            self.codes, self.shape, sort=self._sort, xnull=True
        )
        group_index, _ = compress_group_index(group_index, sort=self._sort)

        if self.has_dropped_na:
            mask = np.where(group_index >= 0)
            # Count how many gaps are caused by previous null values for each position
            null_gaps = np.cumsum(group_index == -1)[mask]
            group_index = group_index[mask]

        result = get_group_index_sorter(group_index, self.ngroups)

        if self.has_dropped_na:
            # Shift by the number of prior null gaps
            result += np.take(null_gaps, result)

        return result

    @final
    @property
    def codes(self) -> list[npt.NDArray[np.signedinteger]]:
        return [ping.codes for ping in self.groupings]

    @property
    def levels(self) -> list[Index]:
        return [ping._group_index for ping in self.groupings]

    @property
    def names(self) -> list[Hashable]:
        return [ping.name for ping in self.groupings]

    @final
    def size(self) -> Series:
        """
        Compute group sizes.
        """
        ids, _, ngroups = self.group_info
        out: np.ndarray | list
        if ngroups:
            out = np.bincount(ids[ids != -1], minlength=ngroups)
        else:
            out = []
        return Series(out, index=self.result_index, dtype="int64", copy=False)

    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]:
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups
        else:
            to_groupby = []
            for ping in self.groupings:
                gv = ping.grouping_vector
                if not isinstance(gv, BaseGrouper):
                    to_groupby.append(gv)
                else:
                    to_groupby.append(gv.groupings[0].grouping_vector)
            index = MultiIndex.from_arrays(to_groupby)
            return self.axis.groupby(index)

    @final
    @cache_readonly
    def is_monotonic(self) -> bool:
        # return if my group orderings are monotonic
        return Index(self.group_info[0]).is_monotonic_increasing

    @final
    @cache_readonly
    def has_dropped_na(self) -> bool:
        """
        Whether grouper has null value(s) that are dropped.
        """
        return bool((self.group_info[0] < 0).any())

    @cache_readonly
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
        comp_ids, obs_group_ids = self._get_compressed_codes()

        ngroups = len(obs_group_ids)
        comp_ids = ensure_platform_int(comp_ids)

        return comp_ids, obs_group_ids, ngroups

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # return the codes of items in original grouped axis
        ids, _, _ = self.group_info
        return ids

    @final
    def _get_compressed_codes(
        self,
    ) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.intp]]:
        # The first returned ndarray may have any signed integer dtype
        if len(self.groupings) > 1:
            group_index = get_group_index(self.codes, self.shape, sort=True, xnull=True)
            return compress_group_index(group_index, sort=self._sort)
            # FIXME: compress_group_index's second return value is int64, not intp

        ping = self.groupings[0]
        return ping.codes, np.arange(len(ping._group_index), dtype=np.intp)

    @final
    @cache_readonly
    def ngroups(self) -> int:
        return len(self.result_index)

    @property
    def reconstructed_codes(self) -> list[npt.NDArray[np.intp]]:
        codes = self.codes
        ids, obs_ids, _ = self.group_info
        return decons_obs_group_ids(ids, obs_ids, self.shape, codes, xnull=True)

    @cache_readonly
    def result_index(self) -> Index:
        if len(self.groupings) == 1:
            return self.groupings[0]._result_index.rename(self.names[0])

        codes = self.reconstructed_codes
        levels = [ping._result_index for ping in self.groupings]
        return MultiIndex(
            levels=levels, codes=codes, verify_integrity=False, names=self.names
        )

    @final
    def get_group_levels(self) -> list[ArrayLike]:
        # Note: only called from _insert_inaxis_grouper, which
        #  is only called for BaseGrouper, never for BinGrouper
        if len(self.groupings) == 1:
            return [self.groupings[0]._group_arraylike]

        name_list = []
        for ping, codes in zip(self.groupings, self.reconstructed_codes):
            codes = ensure_platform_int(codes)
            levels = ping._group_arraylike.take(codes)

            name_list.append(levels)

        return name_list

    # ------------------------------------------------------------
    # Aggregation functions

    @final
    def _cython_operation(
        self,
        kind: str,
        values,
        how: str,
        axis: AxisInt,
        min_count: int = -1,
        **kwargs,
    ) -> ArrayLike:
        """
        Returns the values of a cython operation.
        """
        assert kind in ["transform", "aggregate"]

        cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)

        ids, _, _ = self.group_info
        ngroups = self.ngroups
        return cy_op.cython_operation(
            values=values,
            axis=axis,
            min_count=min_count,
            comp_ids=ids,
            ngroups=ngroups,
            **kwargs,
        )

    @final
    def agg_series(
        self, obj: Series, func: Callable, preserve_dtype: bool = False
    ) -> ArrayLike:
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """

        if not isinstance(obj._values, np.ndarray):
            # we can preserve a little bit more aggressively with EA dtype
            #  because maybe_cast_pointwise_result will do a try/except
            #  with _from_sequence.  NB we are assuming here that _from_sequence
            #  is sufficiently strict that it casts appropriately.
            preserve_dtype = True

        result = self._aggregate_series_pure_python(obj, func)

        npvalues = lib.maybe_convert_objects(result, try_float=False)
        if preserve_dtype:
            out = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
        else:
            out = npvalues
        return out

    @final
    def _aggregate_series_pure_python(
        self, obj: Series, func: Callable
    ) -> npt.NDArray[np.object_]:
        _, _, ngroups = self.group_info

        result = np.empty(ngroups, dtype="O")
        initialized = False

        splitter = self._get_splitter(obj, axis=0)

        for i, group in enumerate(splitter):
            res = func(group)
            res = extract_result(res)

            if not initialized:
                # We only do this validation on the first iteration
                check_result_array(res, group.dtype)
                initialized = True

            result[i] = res

        return result

    @final
    def apply_groupwise(
        self, f: Callable, data: DataFrame | Series, axis: AxisInt = 0
    ) -> tuple[list, bool]:
        mutated = False
        splitter = self._get_splitter(data, axis=axis)
        group_keys = self.group_keys_seq
        result_values = []

        # This calls DataSplitter.__iter__
        zipped = zip(group_keys, splitter)

        for key, group in zipped:
            # Pinning name is needed for
            #  test_group_apply_once_per_group,
            #  test_inconsistent_return_type, test_set_group_name,
            #  test_group_name_available_in_inference_pass,
            #  test_groupby_multi_timezone
            object.__setattr__(group, "name", key)

            # group might be modified
            group_axes = group.axes
            res = f(group)
            if not mutated and not _is_indexed_like(res, group_axes, axis):
                mutated = True
            result_values.append(res)
        # getattr pattern for __name__ is needed for functools.partial objects
        if len(group_keys) == 0 and getattr(f, "__name__", None) in [
            "skew",
            "sum",
            "prod",
        ]:
            #  If group_keys is empty, then no function calls have been made,
            #  so we will not have raised even if this is an invalid dtype.
            #  So do one dummy call here to raise appropriate TypeError.
            f(data.iloc[:0])

        return result_values, mutated

    # ------------------------------------------------------------
    # Methods for sorting subsets of our GroupBy's object

    @final
    @cache_readonly
    def _sort_idx(self) -> npt.NDArray[np.intp]:
        # Counting sort indexer
        ids, _, ngroups = self.group_info
        return get_group_index_sorter(ids, ngroups)

    @final
    @cache_readonly
    def _sorted_ids(self) -> npt.NDArray[np.intp]:
        ids, _, _ = self.group_info
        return ids.take(self._sort_idx)


class BinGrouper(BaseGrouper):
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """

    bins: npt.NDArray[np.int64]
    binlabels: Index

    def __init__(
        self,
        bins,
        binlabels,
        indexer=None,
    ) -> None:
        self.bins = ensure_int64(bins)
        self.binlabels = ensure_index(binlabels)
        self.indexer = indexer

        # These lengths must match, otherwise we could call agg_series
        #  with empty self.bins, which would raise later.
        assert len(self.binlabels) == len(self.bins)

    @cache_readonly
    def groups(self):
        """dict {group name -> group labels}"""
        # this is mainly for compat
        # GH 3881
        result = {
            key: value
            for key, value in zip(self.binlabels, self.bins)
            if key is not NaT
        }
        return result

    @property
    def nkeys(self) -> int:
        # still matches len(self.groupings), but we can hard-code
        return 1

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # return the codes of items in original grouped axis
        ids, _, _ = self.group_info
        if self.indexer is not None:
            sorter = np.lexsort((ids, self.indexer))
            ids = ids[sorter]
        return ids

    def get_iterator(self, data: NDFrame, axis: AxisInt = 0):
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        if axis == 0:
            slicer = lambda start, edge: data.iloc[start:edge]
        else:
            slicer = lambda start, edge: data.iloc[:, start:edge]

        length = len(data.axes[axis])

        start = 0
        for edge, label in zip(self.bins, self.binlabels):
            if label is not NaT:
                yield label, slicer(start, edge)
            start = edge

        if start < length:
            yield self.binlabels[-1], slicer(start, None)

    @cache_readonly
    def indices(self):
        indices = collections.defaultdict(list)

        i = 0
        for label, bin in zip(self.binlabels, self.bins):
            if i < bin:
                if label is not NaT:
                    indices[label] = list(range(i, bin))
                i = bin
        return indices

    @cache_readonly
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
        ngroups = self.ngroups
        obs_group_ids = np.arange(ngroups, dtype=np.intp)
        rep = np.diff(np.r_[0, self.bins])

        rep = ensure_platform_int(rep)
        if ngroups == len(self.bins):
            comp_ids = np.repeat(np.arange(ngroups), rep)
        else:
            comp_ids = np.repeat(np.r_[-1, np.arange(ngroups)], rep)

        return (
            ensure_platform_int(comp_ids),
            obs_group_ids,
            ngroups,
        )

    @cache_readonly
    def reconstructed_codes(self) -> list[np.ndarray]:
        # get unique result indices, and prepend 0 as groupby starts from the first
        return [np.r_[0, np.flatnonzero(self.bins[1:] != self.bins[:-1]) + 1]]

    @cache_readonly
    def result_index(self) -> Index:
        if len(self.binlabels) != 0 and isna(self.binlabels[0]):
            return self.binlabels[1:]

        return self.binlabels

    @property
    def levels(self) -> list[Index]:
        return [self.binlabels]

    @property
    def names(self) -> list[Hashable]:
        return [self.binlabels.name]

    @property
    def groupings(self) -> list[grouper.Grouping]:
        lev = self.binlabels
        codes = self.group_info[0]
        labels = lev.take(codes)
        ping = grouper.Grouping(
            labels, labels, in_axis=False, level=None, uniques=lev._values
        )
        return [ping]


def _is_indexed_like(obj, axes, axis: AxisInt) -> bool:
    if isinstance(obj, Series):
        if len(axes) > 1:
            return False
        return obj.axes[axis].equals(axes[axis])
    elif isinstance(obj, DataFrame):
        return obj.axes[axis].equals(axes[axis])

    return False


# ----------------------------------------------------------------------
# Splitting / application


class DataSplitter(Generic[NDFrameT]):
    def __init__(
        self,
        data: NDFrameT,
        labels: npt.NDArray[np.intp],
        ngroups: int,
        *,
        sort_idx: npt.NDArray[np.intp],
        sorted_ids: npt.NDArray[np.intp],
        axis: AxisInt = 0,
    ) -> None:
        self.data = data
        self.labels = ensure_platform_int(labels)  # _should_ already be np.intp
        self.ngroups = ngroups

        self._slabels = sorted_ids
        self._sort_idx = sort_idx

        self.axis = axis
        assert isinstance(axis, int), axis

    def __iter__(self) -> Iterator:
        sdata = self._sorted_data

        if self.ngroups == 0:
            # we are inside a generator, rather than raise StopIteration
            # we merely return signal the end
            return

        starts, ends = lib.generate_slices(self._slabels, self.ngroups)

        for start, end in zip(starts, ends):
            yield self._chop(sdata, slice(start, end))

    @cache_readonly
    def _sorted_data(self) -> NDFrameT:
        return self.data.take(self._sort_idx, axis=self.axis)

    def _chop(self, sdata, slice_obj: slice) -> NDFrame:
        raise AbstractMethodError(self)


class SeriesSplitter(DataSplitter):
    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        # fastpath equivalent to `sdata.iloc[slice_obj]`
        mgr = sdata._mgr.get_slice(slice_obj)
        ser = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        ser._name = sdata.name
        return ser.__finalize__(sdata, method="groupby")


class FrameSplitter(DataSplitter):
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        # Fastpath equivalent to:
        # if self.axis == 0:
        #     return sdata.iloc[slice_obj]
        # else:
        #     return sdata.iloc[:, slice_obj]
        mgr = sdata._mgr.get_slice(slice_obj, axis=1 - self.axis)
        df = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(sdata, method="groupby")


def _get_splitter(
    data: NDFrame,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    *,
    sort_idx: npt.NDArray[np.intp],
    sorted_ids: npt.NDArray[np.intp],
    axis: AxisInt = 0,
) -> DataSplitter:
    if isinstance(data, Series):
        klass: type[DataSplitter] = SeriesSplitter
    else:
        # i.e. DataFrame
        klass = FrameSplitter

    return klass(
        data, labels, ngroups, sort_idx=sort_idx, sorted_ids=sorted_ids, axis=axis
    )
<|endoftext|>"""
Provide basic components for groupby.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable


@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable
    position: int


# special case to prevent duplicate plots when catching exceptions when
# forwarding methods from NDFrames
plotting_methods = frozenset(["plot", "hist"])

# cythonized transformations or canned "agg+broadcast", which do not
# require postprocessing of the result by transform.
cythonized_kernels = frozenset(["cumprod", "cumsum", "shift", "cummin", "cummax"])

# List of aggregation/reduction functions.
# These map each group to a single numeric value
reduction_kernels = frozenset(
    [
        "all",
        "any",
        "corrwith",
        "count",
        "first",
        "idxmax",
        "idxmin",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "nunique",
        "prod",
        # as long as `quantile`'s signature accepts only
        # a single quantile value, it's a reduction.
        # GH#27526 might change that.
        "quantile",
        "sem",
        "size",
        "skew",
        "std",
        "sum",
        "var",
    ]
)

# List of transformation functions.
# a transformation is a function that, for each group,
# produces a result that has the same shape as the group.


transformation_kernels = frozenset(
    [
        "bfill",
        "cumcount",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "diff",
        "ffill",
        "fillna",
        "ngroup",
        "pct_change",
        "rank",
        "shift",
    ]
)

# these are all the public methods on Grouper which don't belong
# in either of the above lists
groupby_other_methods = frozenset(
    [
        "agg",
        "aggregate",
        "apply",
        "boxplot",
        # corr and cov return ngroups*ncolumns rows, so they
        # are neither a transformation nor a reduction
        "corr",
        "cov",
        "describe",
        "dtypes",
        "expanding",
        "ewm",
        "filter",
        "get_group",
        "groups",
        "head",
        "hist",
        "indices",
        "ndim",
        "ngroups",
        "nth",
        "ohlc",
        "pipe",
        "plot",
        "resample",
        "rolling",
        "tail",
        "take",
        "transform",
        "sample",
        "value_counts",
    ]
)
# Valid values  of `name` for `groupby.transform(name)`
# NOTE: do NOT edit this directly. New additions should be inserted
# into the appropriate list above.
transform_kernel_allowlist = reduction_kernels | transformation_kernels
<|endoftext|>from __future__ import annotations

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)

import numpy as np

from pandas.util._decorators import (
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)

if TYPE_CHECKING:
    from pandas._typing import PositionalIndexer

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.groupby import groupby


class GroupByIndexingMixin:
    """
    Mixin for adding ._positional_selector to GroupBy.
    """

    @cache_readonly
    def _positional_selector(self) -> GroupByPositionalSelector:
        """
        Return positional selection for each group.

        ``groupby._positional_selector[i:j]`` is similar to
        ``groupby.apply(lambda x: x.iloc[i:j])``
        but much faster and preserves the original index and order.

        ``_positional_selector[]`` is compatible with and extends :meth:`~GroupBy.head`
        and :meth:`~GroupBy.tail`. For example:

        - ``head(5)``
        - ``_positional_selector[5:-5]``
        - ``tail(5)``

        together return all the rows.
