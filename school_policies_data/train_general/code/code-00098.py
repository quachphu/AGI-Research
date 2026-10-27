    invalid_op.__name__ = name
    return invalid_op
<|endoftext|>"""
Implementation of nlargest and nsmallest.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    cast,
    final,
)

import numpy as np

from pandas._libs import algos as libalgos

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import BaseMaskedDtype

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        IndexLabel,
    )

    from pandas import (
        DataFrame,
        Series,
    )


class SelectN:
    def __init__(self, obj, n: int, keep: str) -> None:
        self.obj = obj
        self.n = n
        self.keep = keep

        if self.keep not in ("first", "last", "all"):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> DataFrame | Series:
        raise NotImplementedError

    @final
    def nlargest(self):
        return self.compute("nlargest")

    @final
    def nsmallest(self):
        return self.compute("nsmallest")

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        if is_numeric_dtype(dtype):
            return not is_complex_dtype(dtype)
        return needs_i8_conversion(dtype)


class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series:
        from pandas.core.reshape.concat import concat

        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

        if n <= 0:
            return self.obj[[]]

        dropped = self.obj.dropna()
        nan_index = self.obj.drop(dropped.index)

        # slow method
        if n >= len(self.obj):
            ascending = method == "nsmallest"
            return self.obj.sort_values(ascending=ascending).head(n)

        # fast method
        new_dtype = dropped.dtype

        # Similar to algorithms._ensure_data
        arr = dropped._values
        if needs_i8_conversion(arr.dtype):
            arr = arr.view("i8")
        elif isinstance(arr.dtype, BaseMaskedDtype):
            arr = arr._data
        else:
            arr = np.asarray(arr)
        if arr.dtype.kind == "b":
            arr = arr.view(np.uint8)

        if method == "nlargest":
            arr = -arr
            if is_integer_dtype(new_dtype):
                # GH 21426: ensure reverse ordering at boundaries
                arr -= 1

            elif is_bool_dtype(new_dtype):
                # GH 26154: ensure False is smaller than True
                arr = 1 - (-arr)

        if self.keep == "last":
            arr = arr[::-1]

        nbase = n
        narr = len(arr)
        n = min(n, narr)

        # arr passed into kth_smallest must be contiguous. We copy
        # here because kth_smallest will modify its input
        # avoid OOB access with kth_smallest_c when n <= 0
        if len(arr) > 0:
            kth_val = libalgos.kth_smallest(arr.copy(order="C"), n - 1)
        else:
            kth_val = np.nan
        (ns,) = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind="mergesort")]

        if self.keep != "all":
            inds = inds[:n]
            findex = nbase
        else:
            if len(inds) < nbase <= len(nan_index) + len(inds):
                findex = len(nan_index) + len(inds)
            else:
                findex = len(inds)

        if self.keep == "last":
            # reverse indices
            inds = narr - 1 - inds

        return concat([dropped.iloc[inds], nan_index]).iloc[:findex]


class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj: DataFrame, n: int, keep: str, columns: IndexLabel) -> None:
        super().__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]

        columns = cast(Sequence[Hashable], columns)
        columns = list(columns)
        self.columns = columns

    def compute(self, method: str) -> DataFrame:
        from pandas.core.api import Index

        n = self.n
        frame = self.obj
        columns = self.columns

        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(
                    f"Column {repr(column)} has dtype {dtype}, "
                    f"cannot use method {repr(method)} with this dtype"
                )

        def get_indexer(current_indexer, other_indexer):
            """
            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            if method == "nsmallest":
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)

        # Below we save and reset the index in case index contains duplicates
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Index([], dtype=np.int64)

        for i, column in enumerate(columns):
            # For each column we apply method to cur_frame[column].
            # If it's the last column or if we have the number of
            # results desired we are done.
            # Otherwise there are duplicates of the largest/smallest
            # value and we need to look at the rest of the columns
            # to determine which of the rows with the largest/smallest
            # value in the column to keep.
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(
                cur_n, keep=self.keep if is_last_column else "all"
            )

            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break

            # Now find all values which are equal to
            # the (nsmallest: largest)/(nlargest: smallest)
            # from our series.
            border_value = values == values[values.index[-1]]

            # Some of these values are among the top-n
            # some aren't.
            unsafe_values = values[border_value]

            # These values are definitely among the top-n
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)

            # Go on and separate the unsafe_values on the remaining
            # columns.
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)

        frame = frame.take(indexer)

        # Restore the index on frame
        frame.index = original_index.take(indexer)

        # If there is only one column, the frame is already sorted.
        if len(columns) == 1:
            return frame

        ascending = method == "nsmallest"

        return frame.sort_values(columns, ascending=ascending, kind="mergesort")
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import maybe_box_native
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    ExtensionDtype,
)

from pandas.core import common as com

if TYPE_CHECKING:
    from pandas._typing import MutableMappingT

    from pandas import DataFrame


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> MutableMappingT:
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> list[MutableMappingT]:
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> dict:
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> list[dict]:
    ...


# error: Incompatible default for argument "into" (default has type "type[dict
# [Any, Any]]", argument has type "type[MutableMappingT] | MutableMappingT")
def to_dict(
    df: DataFrame,
    orient: Literal[
        "dict", "list", "series", "split", "tight", "records", "index"
    ] = "dict",
    *,
    into: type[MutableMappingT] | MutableMappingT = dict,  # type: ignore[assignment]
    index: bool = True,
) -> MutableMappingT | list[MutableMappingT]:
    """
    Convert the DataFrame to a dictionary.

    The type of the key-value pairs can be customized with the parameters
    (see below).

    Parameters
    ----------
    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        Determines the type of the values of the dictionary.

        - 'dict' (default) : dict like {column -> {index -> value}}
        - 'list' : dict like {column -> [values]}
        - 'series' : dict like {column -> Series(values)}
        - 'split' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
        - 'tight' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
          'index_names' -> [index.names], 'column_names' -> [column.names]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}

        .. versionadded:: 1.4.0
            'tight' as an allowed value for the ``orient`` argument

    into : class, default dict
        The collections.abc.MutableMapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    index : bool, default True
        Whether to include the index item (and index_names item if `orient`
        is 'tight') in the returned dictionary. Can only be ``False``
        when `orient` is 'split' or 'tight'.

        .. versionadded:: 2.0.0

    Returns
    -------
    dict, list or collections.abc.Mapping
        Return a collections.abc.MutableMapping object representing the
        DataFrame. The resulting transformation depends on the `orient` parameter.
    """
    if not df.columns.is_unique:
        warnings.warn(
            "DataFrame columns are not unique, some columns will be omitted.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
    # GH16122
    into_c = com.standardize_mapping(into)

    #  error: Incompatible types in assignment (expression has type "str",
    # variable has type "Literal['dict', 'list', 'series', 'split', 'tight',
    # 'records', 'index']")
    orient = orient.lower()  # type: ignore[assignment]

    if not index and orient not in ["split", "tight"]:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split' or 'tight'"
        )

    if orient == "series":
        # GH46470 Return quickly if orient series to avoid creating dtype objects
        return into_c((k, v) for k, v in df.items())

    box_native_indices = [
        i
        for i, col_dtype in enumerate(df.dtypes.values)
        if col_dtype == np.dtype(object) or isinstance(col_dtype, ExtensionDtype)
    ]
    box_na_values = [
        lib.no_default if not isinstance(col_dtype, BaseMaskedDtype) else libmissing.NA
        for i, col_dtype in enumerate(df.dtypes.values)
    ]
    are_all_object_dtype_cols = len(box_native_indices) == len(df.dtypes)

    if orient == "dict":
        return into_c((k, v.to_dict(into=into)) for k, v in df.items())

    elif orient == "list":
        object_dtype_indices_as_set: set[int] = set(box_native_indices)
        return into_c(
            (
                k,
                list(map(maybe_box_native, v.to_numpy(na_value=box_na_values[i])))
                if i in object_dtype_indices_as_set
                else list(map(maybe_box_native, v.to_numpy())),
            )
            for i, (k, v) in enumerate(df.items())
        )

    elif orient == "split":
        data = df._create_data_for_split_and_tight_to_dict(
            are_all_object_dtype_cols, box_native_indices
        )

        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                ("data", data),
            )
        )

    elif orient == "tight":
        data = df._create_data_for_split_and_tight_to_dict(
            are_all_object_dtype_cols, box_native_indices
        )

        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                (
                    "data",
                    [
                        list(map(maybe_box_native, t))
                        for t in df.itertuples(index=False, name=None)
                    ],
                ),
            )
            + ((("index_names", list(df.index.names)),) if index else ())
            + (("column_names", list(df.columns.names)),)
        )

    elif orient == "records":
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            rows = (
                dict(zip(columns, row)) for row in df.itertuples(index=False, name=None)
            )
            return [
                into_c((k, maybe_box_native(v)) for k, v in row.items()) for row in rows
            ]
        else:
            data = [
                into_c(zip(columns, t)) for t in df.itertuples(index=False, name=None)
            ]
            if box_native_indices:
                object_dtype_indices_as_set = set(box_native_indices)
                object_dtype_cols = {
                    col
                    for i, col in enumerate(df.columns)
                    if i in object_dtype_indices_as_set
                }
                for row in data:
                    for col in object_dtype_cols:
                        row[col] = maybe_box_native(row[col])
            return data

    elif orient == "index":
        if not df.index.is_unique:
            raise ValueError("DataFrame index must be unique for orient='index'.")
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            return into_c(
                (t[0], dict(zip(df.columns, map(maybe_box_native, t[1:]))))
                for t in df.itertuples(name=None)
            )
        elif box_native_indices:
            object_dtype_indices_as_set = set(box_native_indices)
            is_object_dtype_by_index = [
                i in object_dtype_indices_as_set for i in range(len(df.columns))
            ]
            return into_c(
                (
                    t[0],
                    {
                        columns[i]: maybe_box_native(v)
                        if is_object_dtype_by_index[i]
                        else v
                        for i, v in enumerate(t[1:])
                    },
                )
                for t in df.itertuples(name=None)
            )
        else:
            return into_c(
                (t[0], dict(zip(df.columns, t[1:]))) for t in df.itertuples(name=None)
            )

    else:
        raise ValueError(f"orient '{orient}' not understood")
<|endoftext|>"""
Module responsible for execution of NDFrame.describe() method.

Method NDFrame.describe() delegates actual execution to function describe_ndframe().
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    cast,
)

import numpy as np

from pandas._libs.tslibs import Timestamp
from pandas._typing import (
    DtypeObj,
    NDFrameT,
    npt,
)
from pandas.util._validators import validate_percentile

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)

from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat

from pandas.io.formats.format import format_percentiles

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Series,
    )


def describe_ndframe(
    *,
    obj: NDFrameT,
    include: str | Sequence[str] | None,
    exclude: str | Sequence[str] | None,
    percentiles: Sequence[float] | np.ndarray | None,
) -> NDFrameT:
    """Describe series or dataframe.

    Called from pandas.core.generic.NDFrame.describe()

    Parameters
    ----------
    obj: DataFrame or Series
        Either dataframe or series to be described.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored for ``Series``.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored for ``Series``.
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should fall between 0 and 1.
        The default is ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

    Returns
    -------
    Dataframe or series description.
    """
    percentiles = _refine_percentiles(percentiles)

    describer: NDFrameDescriberAbstract
    if obj.ndim == 1:
        describer = SeriesDescriber(
            obj=cast("Series", obj),
        )
    else:
        describer = DataFrameDescriber(
            obj=cast("DataFrame", obj),
            include=include,
            exclude=exclude,
        )

    result = describer.describe(percentiles=percentiles)
    return cast(NDFrameT, result)


class NDFrameDescriberAbstract(ABC):
    """Abstract class for describing dataframe or series.

    Parameters
    ----------
    obj : Series or DataFrame
        Object to be described.
    """

    def __init__(self, obj: DataFrame | Series) -> None:
        self.obj = obj

    @abstractmethod
    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame | Series:
        """Do describe either series or dataframe.

        Parameters
        ----------
        percentiles : list-like of numbers
            The percentiles to include in the output.
        """


class SeriesDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating series description."""

    obj: Series

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> Series:
        describe_func = select_describe_func(
            self.obj,
        )
        return describe_func(self.obj, percentiles)


class DataFrameDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating dataobj description.

    Parameters
    ----------
    obj : DataFrame
        DataFrame to be described.
    include : 'all', list-like of dtypes or None
        A white list of data types to include in the result.
    exclude : list-like of dtypes or None
        A black list of data types to omit from the result.
    """

    obj: DataFrame

    def __init__(
        self,
        obj: DataFrame,
        *,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
    ) -> None:
        self.include = include
        self.exclude = exclude

        if obj.ndim == 2 and obj.columns.size == 0:
            raise ValueError("Cannot describe a DataFrame without columns")

        super().__init__(obj)

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame:
        data = self._select_data()

        ldesc: list[Series] = []
        for _, series in data.items():
            describe_func = select_describe_func(series)
            ldesc.append(describe_func(series, percentiles))

        col_names = reorder_columns(ldesc)
        d = concat(
            [x.reindex(col_names, copy=False) for x in ldesc],
            axis=1,
            sort=False,
        )
        d.columns = data.columns.copy()
        return d

    def _select_data(self) -> DataFrame:
        """Select columns to be described."""
        if (self.include is None) and (self.exclude is None):
            # when some numerics are found, keep only numerics
            default_include: list[npt.DTypeLike] = [np.number, "datetime"]
            data = self.obj.select_dtypes(include=default_include)
            if len(data.columns) == 0:
                data = self.obj
        elif self.include == "all":
            if self.exclude is not None:
                msg = "exclude must be None when include is 'all'"
                raise ValueError(msg)
            data = self.obj
        else:
            data = self.obj.select_dtypes(
                include=self.include,
                exclude=self.exclude,
            )
        return data


def reorder_columns(ldesc: Sequence[Series]) -> list[Hashable]:
    """Set a convenient order for rows for display."""
    names: list[Hashable] = []
    seen_names: set[Hashable] = set()
    ldesc_indexes = sorted((x.index for x in ldesc), key=len)
    for idxnames in ldesc_indexes:
        for name in idxnames:
            if name not in seen_names:
                seen_names.add(name)
                names.append(name)
    return names


def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing numerical data.

    Parameters
    ----------
    series : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    stat_index = ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]
    d = (
        [series.count(), series.mean(), series.std(), series.min()]
        + series.quantile(percentiles).tolist()
        + [series.max()]
    )
    # GH#48340 - always return float on non-complex numeric data
    dtype: DtypeObj | None
    if isinstance(series.dtype, ExtensionDtype):
        if isinstance(series.dtype, ArrowDtype):
            if series.dtype.kind == "m":
                # GH53001: describe timedeltas with object dtype
                dtype = None
            else:
                import pyarrow as pa

                dtype = ArrowDtype(pa.float64())
        else:
            dtype = Float64Dtype()
    elif series.dtype.kind in "iufb":
        # i.e. numeric but exclude complex dtype
        dtype = np.dtype("float")
    else:
        dtype = None
    return Series(d, index=stat_index, name=series.name, dtype=dtype)


def describe_categorical_1d(
    data: Series,
    percentiles_ignored: Sequence[float],
) -> Series:
    """Describe series containing categorical data.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ["count", "unique", "top", "freq"]
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    if count_unique > 0:
        top, freq = objcounts.index[0], objcounts.iloc[0]
        dtype = None
    else:
        # If the DataFrame is empty, set 'top' and 'freq' to None
        # to maintain output shape consistency
        top, freq = np.nan, np.nan
        dtype = "object"

    result = [data.count(), count_unique, top, freq]

    from pandas import Series

    return Series(result, index=names, name=data.name, dtype=dtype)


def describe_timestamp_as_categorical_1d(
    data: Series,
    percentiles_ignored: Sequence[float],
) -> Series:
    """Describe series containing timestamp data treated as categorical.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ["count", "unique"]
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    result: list[float | Timestamp] = [data.count(), count_unique]
    dtype = None
    if count_unique > 0:
        top, freq = objcounts.index[0], objcounts.iloc[0]
        tz = data.dt.tz
        asint = data.dropna().values.view("i8")
        top = Timestamp(top)
        if top.tzinfo is not None and tz is not None:
            # Don't tz_localize(None) if key is already tz-aware
            top = top.tz_convert(tz)
        else:
            top = top.tz_localize(tz)
        names += ["top", "freq", "first", "last"]
        result += [
            top,
            freq,
            Timestamp(asint.min(), tz=tz),
            Timestamp(asint.max(), tz=tz),
        ]

    # If the DataFrame is empty, set 'top' and 'freq' to None
    # to maintain output shape consistency
    else:
        names += ["top", "freq"]
        result += [np.nan, np.nan]
        dtype = "object"

    from pandas import Series

    return Series(result, index=names, name=data.name, dtype=dtype)


def describe_timestamp_1d(data: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing datetime64 dtype.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    # GH-30164
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    stat_index = ["count", "mean", "min"] + formatted_percentiles + ["max"]
    d = (
        [data.count(), data.mean(), data.min()]
        + data.quantile(percentiles).tolist()
        + [data.max()]
    )
    return Series(d, index=stat_index, name=data.name)


def select_describe_func(
    data: Series,
) -> Callable:
    """Select proper function for describing series based on data type.

    Parameters
    ----------
    data : Series
        Series to be described.
    """
    if is_bool_dtype(data.dtype):
        return describe_categorical_1d
    elif is_numeric_dtype(data):
        return describe_numeric_1d
    elif data.dtype.kind == "M" or isinstance(data.dtype, DatetimeTZDtype):
        return describe_timestamp_1d
    elif data.dtype.kind == "m":
        return describe_numeric_1d
    else:
        return describe_categorical_1d


def _refine_percentiles(
    percentiles: Sequence[float] | np.ndarray | None,
) -> npt.NDArray[np.float64]:
    """
    Ensure that percentiles are unique and sorted.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output.
    """
    if percentiles is None:
        return np.array([0.25, 0.5, 0.75])

    # explicit conversion of `percentiles` to list
    percentiles = list(percentiles)

    # get them all to be in [0, 1]
    validate_percentile(percentiles)

    # median should always be included
    if 0.5 not in percentiles:
        percentiles.append(0.5)

    percentiles = np.asarray(percentiles)

    # sort and check for duplicates
    unique_pcts = np.unique(percentiles)
    assert percentiles is not None
    if len(unique_pcts) < len(percentiles):
        raise ValueError("percentiles cannot contain duplicates")

    return unique_pcts
<|endoftext|>"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""
from __future__ import annotations

from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from numpy import ma

from pandas._config import using_string_dtype

from pandas._libs import lib

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    dict_compat,
    maybe_cast_to_datetime,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_integer_dtype,
    is_list_like,
    is_named_tuple,
    is_object_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core import (
    algorithms,
    common as com,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
    range_to_ndarray,
    sanitize_array,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    TimedeltaIndex,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    union_indexes,
)
from pandas.core.internals.array_manager import (
    ArrayManager,
    SingleArrayManager,
)
from pandas.core.internals.blocks import (
    BlockPlacement,
    ensure_block_shape,
    new_block,
    new_block_2d,
)
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
    create_block_manager_from_blocks,
    create_block_manager_from_column_arrays,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        Manager,
        npt,
    )
# ---------------------------------------------------------------------
# BlockManager Interface


def arrays_to_mgr(
    arrays,
    columns: Index,
    index,
    *,
    dtype: DtypeObj | None = None,
    verify_integrity: bool = True,
    typ: str | None = None,
    consolidate: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    if verify_integrity:
        # figure out the index, if necessary
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)

        # don't force copy because getting jammed in an ndarray anyway
        arrays, refs = _homogenize(arrays, index, dtype)
        # _homogenize ensures
        #  - all(len(x) == len(index) for x in arrays)
        #  - all(x.ndim == 1 for x in arrays)
        #  - all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
        #  - all(type(x) is not NumpyExtensionArray for x in arrays)

    else:
        index = ensure_index(index)
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        # with _from_arrays, the passed arrays should never be Series objects
        refs = [None] * len(arrays)

        # Reached via DataFrame._from_arrays; we do minimal validation here
        for arr in arrays:
            if (
                not isinstance(arr, (np.ndarray, ExtensionArray))
                or arr.ndim != 1
                or len(arr) != len(index)
            ):
                raise ValueError(
                    "Arrays must be 1-dimensional np.ndarray or ExtensionArray "
                    "with length matching len(index)"
                )

    columns = ensure_index(columns)
    if len(columns) != len(arrays):
        raise ValueError("len(arrays) must match len(columns)")

    # from BlockManager perspective
    axes = [columns, index]

    if typ == "block":
        return create_block_manager_from_column_arrays(
            arrays, axes, consolidate=consolidate, refs=refs
        )
    elif typ == "array":
        return ArrayManager(arrays, [index, columns])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")


def rec_array_to_mgr(
    data: np.rec.recarray | np.ndarray,
    index,
    columns,
    dtype: DtypeObj | None,
    copy: bool,
    typ: str,
) -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
    # essentially process a record array then fill it
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)

    if columns is not None:
        columns = ensure_index(columns)
    arrays, arr_columns = to_arrays(fdata, columns)

    # create the manager

    arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns

    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ)

    if copy:
        mgr = mgr.copy()
    return mgr


def mgr_to_mgr(mgr, typ: str, copy: bool = True) -> Manager:
    """
    Convert to specific type of Manager. Does not copy if the type is already
    correct. Does not guarantee a copy otherwise. `copy` keyword only controls
    whether conversion from Block->ArrayManager copies the 1D arrays.
    """
    new_mgr: Manager

    if typ == "block":
        if isinstance(mgr, BlockManager):
            new_mgr = mgr
        else:
            if mgr.ndim == 2:
                new_mgr = arrays_to_mgr(
                    mgr.arrays, mgr.axes[0], mgr.axes[1], typ="block"
                )
            else:
                new_mgr = SingleBlockManager.from_array(mgr.arrays[0], mgr.index)
    elif typ == "array":
        if isinstance(mgr, ArrayManager):
            new_mgr = mgr
        else:
            if mgr.ndim == 2:
                arrays = [mgr.iget_values(i) for i in range(len(mgr.axes[0]))]
                if copy:
                    arrays = [arr.copy() for arr in arrays]
                new_mgr = ArrayManager(arrays, [mgr.axes[1], mgr.axes[0]])
            else:
                array = mgr.internal_values()
                if copy:
                    array = array.copy()
                new_mgr = SingleArrayManager([array], [mgr.index])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")
    return new_mgr


# ---------------------------------------------------------------------
# DataFrame Constructor Interface


def ndarray_to_mgr(
    values, index, columns, dtype: DtypeObj | None, copy: bool, typ: str
) -> Manager:
    # used in DataFrame.__init__
    # input must be a ndarray, list, Series, Index, ExtensionArray

    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = Index([values.name])
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)

        # zero len case (GH #2234)
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)

    # if the array preparation does a copy -> avoid this for ArrayManager,
    # since the copy is done on conversion to 1D arrays
    copy_on_sanitize = False if typ == "array" else copy

    vdtype = getattr(values, "dtype", None)
    refs = None
    if is_1d_only_ea_dtype(vdtype) or is_1d_only_ea_dtype(dtype):
        # GH#19157

        if isinstance(values, (np.ndarray, ExtensionArray)) and values.ndim > 1:
            # GH#12513 a EA dtype passed with a 2D array, split into
            #  multiple EAs that view the values
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[slice, int]"
            values = [
                values[:, n]  # type: ignore[call-overload]
                for n in range(values.shape[1])
            ]
        else:
            values = [values]

        if columns is None:
            columns = Index(range(len(values)))
        else:
            columns = ensure_index(columns)

        return arrays_to_mgr(values, columns, index, dtype=dtype, typ=typ)

    elif isinstance(vdtype, ExtensionDtype):
        # i.e. Datetime64TZ, PeriodDtype; cases with is_1d_only_ea_dtype(vdtype)
        #  are already caught above
        values = extract_array(values, extract_numpy=True)
        if copy:
            values = values.copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)

    elif isinstance(values, (ABCSeries, Index)):
        if not copy_on_sanitize and (
            dtype is None or astype_is_view(values.dtype, dtype)
        ):
            refs = values._references

        if copy_on_sanitize:
            values = values._values.copy()
        else:
            values = values._values

        values = _ensure_2d(values)

    elif isinstance(values, (np.ndarray, ExtensionArray)):
        # drop subclass info
        if copy_on_sanitize and (dtype is None or astype_is_view(values.dtype, dtype)):
            # only force a copy now if copy=True was requested
            # and a subsequent `astype` will not already result in a copy
            values = np.array(values, copy=True, order="F")
        else:
            values = np.asarray(values)
        values = _ensure_2d(values)

    else:
        # by definition an array here
        # the dtypes will be coerced to a single dtype
        values = _prep_ndarraylike(values, copy=copy_on_sanitize)

    if dtype is not None and values.dtype != dtype:
        # GH#40110 see similar check inside sanitize_array
        values = sanitize_array(
            values,
            None,
            dtype=dtype,
            copy=copy_on_sanitize,
            allow_2d=True,
        )

    # _prep_ndarraylike ensures that values.ndim == 2 at this point
    index, columns = _get_axes(
        values.shape[0], values.shape[1], index=index, columns=columns
    )

    _check_values_indices_shape_match(values, index, columns)

    if typ == "array":
        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)

        if dtype is None and is_object_dtype(values.dtype):
            arrays = [
                ensure_wrapped_if_datetimelike(
                    maybe_infer_to_datetimelike(values[:, i])
                )
                for i in range(values.shape[1])
            ]
        else:
            if lib.is_np_dtype(values.dtype, "mM"):
                values = ensure_wrapped_if_datetimelike(values)
            arrays = [values[:, i] for i in range(values.shape[1])]

        if copy:
            arrays = [arr.copy() for arr in arrays]

        return ArrayManager(arrays, [index, columns], verify_integrity=False)

    values = values.T

    # if we don't have a dtype specified, then try to convert objects
    # on the entire block; this is to convert if we have datetimelike's
    # embedded in an object type
    if dtype is None and is_object_dtype(values.dtype):
        obj_columns = list(values)
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        # don't convert (and copy) the objects if no type inference occurs
        if any(x is not y for x, y in zip(obj_columns, maybe_datetime)):
            dvals_list = [ensure_block_shape(dval, 2) for dval in maybe_datetime]
            block_values = [
                new_block_2d(dvals_list[n], placement=BlockPlacement(n))
                for n in range(len(dvals_list))
            ]
        else:
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == "U" and using_string_dtype():
        dtype = StringDtype(na_value=np.nan)

        obj_columns = list(values)
        block_values = [
            new_block(
                dtype.construct_array_type()._from_sequence(data, dtype=dtype),
                BlockPlacement(slice(i, i + 1)),
                ndim=2,
            )
            for i, data in enumerate(obj_columns)
        ]

    else:
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]

    if len(columns) == 0:
        # TODO: check len(values) == 0?
        block_values = []

    return create_block_manager_from_blocks(
        block_values, [columns, index], verify_integrity=False
    )


def _check_values_indices_shape_match(
    values: np.ndarray, index: Index, columns: Index
) -> None:
    """
    Check that the shape implied by our axes matches the actual shape of the
    data.
    """
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        # Could let this raise in Block constructor, but we get a more
        #  helpful exception message this way.
        if values.shape[0] == 0 < len(index):
            raise ValueError("Empty data passed with indices specified.")

        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


def dict_to_mgr(
    data: dict,
    index,
    columns,
    *,
    dtype: DtypeObj | None = None,
    typ: str = "block",
    copy: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    arrays: Sequence[Any] | Series

    if columns is not None:
        from pandas.core.series import Series

        arrays = Series(data, index=columns, dtype=object)
        missing = arrays.isna()
        if index is None:
            # GH10856
            # raise ValueError if only scalars in dict
            index = _extract_index(arrays[~missing])
        else:
            index = ensure_index(index)

        # no obvious "empty" int column
        if missing.any() and not is_integer_dtype(dtype):
            nan_dtype: DtypeObj

            if dtype is not None:
                # calling sanitize_array ensures we don't mix-and-match
                #  NA dtypes
                midxs = missing.values.nonzero()[0]
                for i in midxs:
                    arr = sanitize_array(arrays.iat[i], index, dtype=dtype)
                    arrays.iat[i] = arr
            else:
                # GH#1783
                nan_dtype = np.dtype("object")
                val = construct_1d_arraylike_from_scalar(np.nan, len(index), nan_dtype)
                nmissing = missing.sum()
                if copy:
                    rhs = [val] * nmissing
                else:
                    # GH#45369
                    rhs = [val.copy() for _ in range(nmissing)]
                arrays.loc[missing] = rhs

        arrays = list(arrays)
        columns = ensure_index(columns)

    else:
        keys = list(data.keys())
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]

    if copy:
        if typ == "block":
            # We only need to copy arrays that will not get consolidated, i.e.
            #  only EA arrays
            arrays = [
                x.copy()
                if isinstance(x, ExtensionArray)
                else x.copy(deep=True)
                if (
                    isinstance(x, Index)
                    or isinstance(x, ABCSeries)
                    and is_1d_only_ea_dtype(x.dtype)
                )
                else x
                for x in arrays
            ]
        else:
            # dtype check to exclude e.g. range objects, scalars
            arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]

    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)


def nested_data_to_arrays(
    data: Sequence,
    columns: Index | None,
    index: Index | None,
    dtype: DtypeObj | None,
) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    # By the time we get here we have already checked treat_as_nested(data)

    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)

    arrays, columns = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)

    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))

    return arrays, columns, index


def treat_as_nested(data) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
    return (
        len(data) > 0
        and is_list_like(data[0])
        and getattr(data[0], "ndim", 1) == 1
        and not (isinstance(data, ExtensionArray) and data.ndim == 2)
    )


# ---------------------------------------------------------------------


def _prep_ndarraylike(values, copy: bool = True) -> np.ndarray:
    # values is specifically _not_ ndarray, EA, Index, or Series
    # We only get here with `not treat_as_nested(values)`

    if len(values) == 0:
        # TODO: check for length-zero range, in which case return int64 dtype?
        # TODO: reuse anything in try_cast?
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v):
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v

        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        # We don't do maybe_infer_to_datetimelike here bc we will end up doing
        #  it column-by-column in ndarray_to_mgr
        return res

    # we could have a 1-dim or 2-dim list here
    # this is equiv of np.asarray, but does object conversion
    # and platform dtype preservation
    # does not convert e.g. [1, "a", True] to ["1", "a", "True"] like
    #  np.asarray would
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        # GH#21861 see test_constructor_list_of_lists
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)

    return _ensure_2d(values)


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """
    Reshape 1D values, raise on anything else other than 2D.
    """
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f"Must pass 2-d input. shape={values.shape}")
    return values


def _homogenize(
    data, index: Index, dtype: DtypeObj | None
) -> tuple[list[ArrayLike], list[Any]]:
    oindex = None
    homogenized = []
    # if the original array-like in `data` is a Series, keep track of this Series' refs
    refs: list[Any] = []

    for val in data:
        if isinstance(val, (ABCSeries, Index)):
            if dtype is not None:
                val = val.astype(dtype, copy=False)
            if isinstance(val, ABCSeries) and val.index is not index:
                # Forces alignment. No need to copy data since we
                # are putting it into an ndarray later
                val = val.reindex(index, copy=False)
            refs.append(val._references)
            val = val._values
        else:
            if isinstance(val, dict):
                # GH#41785 this _should_ be equivalent to (but faster than)
                #  val = Series(val, index=index)._values
                if oindex is None:
                    oindex = index.astype("O")

                if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                    # see test_constructor_dict_datetime64_index
                    val = dict_compat(val)
                else:
                    # see test_constructor_subclass_dict
                    val = dict(val)
                val = lib.fast_multiget(val, oindex._values, default=np.nan)

            val = sanitize_array(val, index, dtype=dtype, copy=False)
            com.require_length_match(val, index)
            refs.append(None)

        homogenized.append(val)

    return homogenized, refs


def _extract_index(data) -> Index:
    """
    Try to infer an Index from the passed data, raise ValueError on failure.
    """
    index: Index
    if len(data) == 0:
        return default_index(0)

    raw_lengths = []
    indexes: list[list[Hashable] | Index] = []

    have_raw_arrays = False
    have_series = False
    have_dicts = False

    for val in data:
        if isinstance(val, ABCSeries):
            have_series = True
            indexes.append(val.index)
        elif isinstance(val, dict):
            have_dicts = True
            indexes.append(list(val.keys()))
        elif is_list_like(val) and getattr(val, "ndim", 1) == 1:
            have_raw_arrays = True
            raw_lengths.append(len(val))
        elif isinstance(val, np.ndarray) and val.ndim > 1:
            raise ValueError("Per-column arrays must each be 1-dimensional")

    if not indexes and not raw_lengths:
        raise ValueError("If using all scalar values, you must pass an index")

    if have_series:
        index = union_indexes(indexes)
    elif have_dicts:
        index = union_indexes(indexes, sort=False)

    if have_raw_arrays:
        lengths = list(set(raw_lengths))
        if len(lengths) > 1:
            raise ValueError("All arrays must be of the same length")

        if have_dicts:
            raise ValueError(
                "Mixing dicts with non-Series may lead to ambiguous ordering."
            )

        if have_series:
            if lengths[0] != len(index):
                msg = (
                    f"array length {lengths[0]} does not match index "
                    f"length {len(index)}"
                )
                raise ValueError(msg)
        else:
            index = default_index(lengths[0])

    return ensure_index(index)


def reorder_arrays(
    arrays: list[ArrayLike], arr_columns: Index, columns: Index | None, length: int
) -> tuple[list[ArrayLike], Index]:
    """
    Pre-emptively (cheaply) reindex arrays with new columns.
    """
    # reorder according to the columns
    if columns is not None:
        if not columns.equals(arr_columns):
            # if they are equal, there is nothing to do
            new_arrays: list[ArrayLike] = []
            indexer = arr_columns.get_indexer(columns)
            for i, k in enumerate(indexer):
                if k == -1:
                    # by convention default is all-NaN object dtype
                    arr = np.empty(length, dtype=object)
                    arr.fill(np.nan)
                else:
                    arr = arrays[k]
                new_arrays.append(arr)

            arrays = new_arrays
            arr_columns = columns

    return arrays, arr_columns


def _get_names_from_index(data) -> Index:
    has_some_name = any(getattr(s, "name", None) is not None for s in data)
    if not has_some_name:
        return default_index(len(data))

    index: list[Hashable] = list(range(len(data)))
    count = 0
    for i, s in enumerate(data):
        n = getattr(s, "name", None)
        if n is not None:
            index[i] = n
        else:
            index[i] = f"Unnamed {count}"
            count += 1

    return Index(index)


def _get_axes(
    N: int, K: int, index: Index | None, columns: Index | None
) -> tuple[Index, Index]:
    # helper to create the axes as indexes
    # return axes or defaults

    if index is None:
        index = default_index(N)
    else:
        index = ensure_index(index)

    if columns is None:
        columns = default_index(K)
    else:
        columns = ensure_index(columns)
    return index, columns


def dataclasses_to_dicts(data):
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]

    Returns
    --------
    list_dict : List[dict]

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> dataclasses_to_dicts([Point(1, 2), Point(2, 3)])
    [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]

    """
    from dataclasses import asdict

    return list(map(asdict, data))


# ---------------------------------------------------------------------
# Conversion of Inputs to Arrays


def to_arrays(
    data, columns: Index | None, dtype: DtypeObj | None = None
) -> tuple[list[ArrayLike], Index]:
    """
    Return list of arrays, columns.

    Returns
    -------
    list[ArrayLike]
        These will become columns in a DataFrame.
    Index
        This will become frame.columns.

    Notes
    -----
    Ensures that len(result_arrays) == len(result_index).
    """

    if not len(data):
        if isinstance(data, np.ndarray):
            if data.dtype.names is not None:
                # i.e. numpy structured array
                columns = ensure_index(data.dtype.names)
                arrays = [data[name] for name in columns]

                if len(data) == 0:
                    # GH#42456 the indexing above results in list of 2D ndarrays
                    # TODO: is that an issue with numpy?
                    for i, arr in enumerate(arrays):
                        if arr.ndim == 2:
                            arrays[i] = arr[:, 0]

                return arrays, columns
        return [], ensure_index([])

    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        # e.g. recarray
        columns = Index(list(data.dtype.names))
        arrays = [data[k] for k in columns]
        return arrays, columns

    if isinstance(data[0], (list, tuple)):
        arr = _list_to_arrays(data)
    elif isinstance(data[0], abc.Mapping):
        arr, columns = _list_of_dict_to_arrays(data, columns)
    elif isinstance(data[0], ABCSeries):
        arr, columns = _list_of_series_to_arrays(data, columns)
    else:
        # last ditch effort
        data = [tuple(x) for x in data]
        arr = _list_to_arrays(data)

    content, columns = _finalize_columns_and_data(arr, columns, dtype)
    return content, columns


def _list_to_arrays(data: list[tuple | list]) -> np.ndarray:
    # Returned np.ndarray has ndim = 2
    # Note: we already check len(data) > 0 before getting hre
    if isinstance(data[0], tuple):
        content = lib.to_object_array_tuples(data)
    else:
        # list of lists
        content = lib.to_object_array(data)
    return content


def _list_of_series_to_arrays(
    data: list,
    columns: Index | None,
) -> tuple[np.ndarray, Index]:
    # returned np.ndarray has ndim == 2

    if columns is None:
        # We know pass_data is non-empty because data[0] is a Series
        pass_data = [x for x in data if isinstance(x, (ABCSeries, ABCDataFrame))]
        columns = get_objs_combined_axis(pass_data, sort=False)

    indexer_cache: dict[int, np.ndarray] = {}

    aligned_values = []
    for s in data:
        index = getattr(s, "index", None)
        if index is None:
            index = default_index(len(s))

        if id(index) in indexer_cache:
            indexer = indexer_cache[id(index)]
        else:
            indexer = indexer_cache[id(index)] = index.get_indexer(columns)

        values = extract_array(s, extract_numpy=True)
        aligned_values.append(algorithms.take_nd(values, indexer))

    content = np.vstack(aligned_values)
    return content, columns


def _list_of_dict_to_arrays(
    data: list[dict],
    columns: Index | None,
) -> tuple[np.ndarray, Index]:
    """
    Convert list of dicts to numpy arrays

    if `columns` is not passed, column names are inferred from the records
    - for OrderedDict and dicts, the column names match
      the key insertion-order from the first record to the last.
    - For other kinds of dict-likes, the keys are lexically sorted.

    Parameters
    ----------
    data : iterable
        collection of records (OrderedDict, dict)
    columns: iterables or None

    Returns
    -------
    content : np.ndarray[object, ndim=2]
    columns : Index
    """
    if columns is None:
        gen = (list(x.keys()) for x in data)
        sort = not any(isinstance(d, dict) for d in data)
        pre_cols = lib.fast_unique_multiple_list_gen(gen, sort=sort)
        columns = ensure_index(pre_cols)

    # assure that they are of the base dict class and not of derived
    # classes
    data = [d if type(d) is dict else dict(d) for d in data]  # noqa: E721

    content = lib.dicts_to_array(data, list(columns))
    return content, columns


def _finalize_columns_and_data(
    content: np.ndarray,  # ndim == 2
    columns: Index | None,
    dtype: DtypeObj | None,
) -> tuple[list[ArrayLike], Index]:
    """
    Ensure we have valid columns, cast object dtypes if possible.
    """
    contents = list(content.T)

    try:
        columns = _validate_or_indexify_columns(contents, columns)
    except AssertionError as err:
        # GH#26429 do not raise user-facing AssertionError
        raise ValueError(err) from err

    if len(contents) and contents[0].dtype == np.object_:
        contents = convert_object_array(contents, dtype=dtype)

    return contents, columns


def _validate_or_indexify_columns(
    content: list[np.ndarray], columns: Index | None
) -> Index:
    """
    If columns is None, make numbers as column names; Otherwise, validate that
    columns have valid length.

    Parameters
    ----------
    content : list of np.ndarrays
    columns : Index or None

    Returns
    -------
    Index
        If columns is None, assign positional column index value as columns.

    Raises
    ------
    1. AssertionError when content is not composed of list of lists, and if
        length of columns is not equal to length of content.
    2. ValueError when content is list of lists, but length of each sub-list
        is not equal
    3. ValueError when content is list of lists, but length of sub-list is
        not equal to length of content
    """
    if columns is None:
        columns = default_index(len(content))
    else:
        # Add mask for data which is composed of list of lists
        is_mi_list = isinstance(columns, list) and all(
            isinstance(col, list) for col in columns
        )

        if not is_mi_list and len(columns) != len(content):  # pragma: no cover
            # caller's responsibility to check for this...
            raise AssertionError(
                f"{len(columns)} columns passed, passed data had "
                f"{len(content)} columns"
            )
        if is_mi_list:
            # check if nested list column, length of each sub-list should be equal
            if len({len(col) for col in columns}) > 1:
                raise ValueError(
                    "Length of columns passed for MultiIndex columns is different"
                )

            # if columns is not empty and length of sublist is not equal to content
            if columns and len(columns[0]) != len(content):
                raise ValueError(
                    f"{len(columns[0])} columns passed, passed data had "
                    f"{len(content)} columns"
                )
    return columns


def convert_object_array(
    content: list[npt.NDArray[np.object_]],
    dtype: DtypeObj | None,
    dtype_backend: str = "numpy",
    coerce_float: bool = False,
) -> list[ArrayLike]:
    """
    Internal function to convert object array.

    Parameters
    ----------
    content: List[np.ndarray]
    dtype: np.dtype or ExtensionDtype
    dtype_backend: Controls if nullable/pyarrow dtypes are returned.
    coerce_float: Cast floats that are integers to int.

    Returns
    -------
    List[ArrayLike]
    """
    # provide soft conversion of object dtypes

    def convert(arr):
        if dtype != np.dtype("O"):
            arr = lib.maybe_convert_objects(
                arr,
                try_float=coerce_float,
                convert_to_nullable_dtype=dtype_backend != "numpy",
            )
            # Notes on cases that get here 2023-02-15
            # 1) we DO get here when arr is all Timestamps and dtype=None
            # 2) disabling this doesn't break the world, so this must be
            #    getting caught at a higher level
            # 3) passing convert_non_numeric to maybe_convert_objects get this right
            # 4) convert_non_numeric?

            if dtype is None:
                if arr.dtype == np.dtype("O"):
                    # i.e. maybe_convert_objects didn't convert
                    convert_to_nullable_dtype = dtype_backend != "numpy"
                    arr = maybe_infer_to_datetimelike(arr, convert_to_nullable_dtype)
                    if convert_to_nullable_dtype and arr.dtype == np.dtype("O"):
                        new_dtype = StringDtype()
                        arr_cls = new_dtype.construct_array_type()
                        arr = arr_cls._from_sequence(arr, dtype=new_dtype)
                elif dtype_backend != "numpy" and isinstance(arr, np.ndarray):
                    if arr.dtype.kind in "iufb":
                        arr = pd_array(arr, copy=False)

            elif isinstance(dtype, ExtensionDtype):
                # TODO: test(s) that get here
                # TODO: try to de-duplicate this convert function with
                #  core.construction functions
                cls = dtype.construct_array_type()
                arr = cls._from_sequence(arr, dtype=dtype, copy=False)
            elif dtype.kind in "mM":
                # This restriction is harmless bc these are the only cases
                #  where maybe_cast_to_datetime is not a no-op.
                # Here we know:
                #  1) dtype.kind in "mM" and
                #  2) arr is either object or numeric dtype
                arr = maybe_cast_to_datetime(arr, dtype)

        return arr

    arrays = [convert(arr) for arr in content]

    return arrays
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
import itertools
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import (
    internals as libinternals,
    lib,
)
from pandas._libs.internals import (
    BlockPlacement,
    BlockValuesRefs,
)
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    array_equals,
    isna,
)

import pandas.core.algorithms as algos
from pandas.core.arrays import (
    ArrowExtensionArray,
    ArrowStringArray,
    DatetimeArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
    Index,
    ensure_index,
)
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
    ensure_np_dtype,
    interleaved_dtype,
)
from pandas.core.internals.blocks import (
    COW_WARNING_GENERAL_MSG,
    COW_WARNING_SETITEM_MSG,
    Block,
    NumpyBlock,
    ensure_block_shape,
    extend_blocks,
    get_block_type,
    maybe_coerce_values,
    new_block,
    new_block_2d,
)
from pandas.core.internals.ops import (
    blockwise_all,
    operate_blockwise,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        QuantileInterpolation,
        Self,
        Shape,
        npt,
    )

    from pandas.api.extensions import ExtensionArray


class BaseBlockManager(DataManager):
    """
    Core internal data structure to implement DataFrame, Series, etc.

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Attributes
    ----------
    shape
    ndim
    axes
    values
    items

    Methods
    -------
    set_axis(axis, new_labels)
    copy(deep=True)

    get_dtypes

    apply(func, axes, block_filter_fn)

    get_bool_data
    get_numeric_data

    get_slice(slice_like, axis)
    get(label)
    iget(loc)

    take(indexer, axis)
    reindex_axis(new_labels, axis)
    reindex_indexer(new_labels, indexer, axis)

    delete(label)
    insert(loc, label, value)
    set(label, value)

    Parameters
    ----------
    blocks: Sequence of Block
    axes: Sequence of Index
    verify_integrity: bool, default True

    Notes
    -----
    This is *not* a public API class
    """

    __slots__ = ()

    _blknos: npt.NDArray[np.intp]
    _blklocs: npt.NDArray[np.intp]
    blocks: tuple[Block, ...]
    axes: list[Index]

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    _known_consolidated: bool
    _is_consolidated: bool

    def __init__(self, blocks, axes, verify_integrity: bool = True) -> None:
        raise NotImplementedError

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        raise NotImplementedError

    @property
    def blknos(self) -> npt.NDArray[np.intp]:
        """
        Suppose we want to find the array corresponding to our i'th column.

        blknos[i] identifies the block from self.blocks that contains this column.

        blklocs[i] identifies the column of interest within
        self.blocks[self.blknos[i]]
        """
        if self._blknos is None:
            # Note: these can be altered by other BlockManager methods.
            self._rebuild_blknos_and_blklocs()

        return self._blknos

    @property
    def blklocs(self) -> npt.NDArray[np.intp]:
        """
        See blknos.__doc__
        """
        if self._blklocs is None:
            # Note: these can be altered by other BlockManager methods.
            self._rebuild_blknos_and_blklocs()

        return self._blklocs

    def make_empty(self, axes=None) -> Self:
        """return an empty BlockManager with the items axis of len 0"""
        if axes is None:
            axes = [Index([])] + self.axes[1:]

        # preserve dtype if possible
        if self.ndim == 1:
            assert isinstance(self, SingleBlockManager)  # for mypy
            blk = self.blocks[0]
            arr = blk.values[:0]
            bp = BlockPlacement(slice(0, 0))
            nb = blk.make_block_same_class(arr, placement=bp)
            blocks = [nb]
        else:
            blocks = []
        return type(self).from_blocks(blocks, axes)

    def __nonzero__(self) -> bool:
        return True

    # Python3 compat
    __bool__ = __nonzero__

    def _normalize_axis(self, axis: AxisInt) -> int:
        # switch axis to follow BlockManager logic
        if self.ndim == 2:
            axis = 1 if axis == 0 else 0
        return axis

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        self._validate_set_axis(axis, new_labels)
        self.axes[axis] = new_labels

    @property
    def is_single_block(self) -> bool:
        # Assumes we are 2D; overridden by SingleBlockManager
        return len(self.blocks) == 1

    @property
    def items(self) -> Index:
        return self.axes[0]

    def _has_no_reference(self, i: int) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        blkno = self.blknos[i]
        return self._has_no_reference_block(blkno)

    def _has_no_reference_block(self, blkno: int) -> bool:
        """
        Check for block `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the block has no references.
        """
        return not self.blocks[blkno].refs.has_reference()

    def add_references(self, mgr: BaseBlockManager) -> None:
        """
        Adds the references from one manager to another. We assume that both
        managers have the same block structure.
        """
        if len(self.blocks) != len(mgr.blocks):
            # If block structure changes, then we made a copy
            return
        for i, blk in enumerate(self.blocks):
            blk.refs = mgr.blocks[i].refs
            blk.refs.add_reference(blk)

    def references_same_values(self, mgr: BaseBlockManager, blkno: int) -> bool:
        """
        Checks if two blocks from two different block managers reference the
        same underlying values.
        """
        blk = self.blocks[blkno]
        return any(blk is ref() for ref in mgr.blocks[blkno].refs.referenced_blocks)

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        dtypes = np.array([blk.dtype for blk in self.blocks], dtype=object)
        return dtypes.take(self.blknos)

    @property
    def arrays(self) -> list[ArrayLike]:
        """
        Quick access to the backing arrays of the Blocks.

        Only for compatibility with ArrayManager for testing convenience.
        Not to be used in actual code, and return value is not the same as the
        ArrayManager method (list of 1D arrays vs iterator of 2D ndarrays / 1D EAs).

        Warning! The returned arrays don't handle Copy-on-Write, so this should
        be used with caution (only in read-mode).
        """
        return [blk.values for blk in self.blocks]

    def __repr__(self) -> str:
        output = type(self).__name__
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += f"\nItems: {ax}"
            else:
                output += f"\nAxis {i}: {ax}"

        for block in self.blocks:
            output += f"\n{block}"
        return output

    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
        assert "filter" not in kwargs

        align_keys = align_keys or []
        result_blocks: list[Block] = []
        # fillna: Series/DataFrame is responsible for making sure value is aligned

        aligned_args = {k: kwargs[k] for k in align_keys}

        for b in self.blocks:
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[b.mgr_locs.indexer]._values
                        else:
                            kwargs[k] = obj.iloc[:, b.mgr_locs.indexer]._values
                    else:
                        # otherwise we have an ndarray
                        kwargs[k] = obj[b.mgr_locs.indexer]

            if callable(f):
                applied = b.apply(f, **kwargs)
            else:
                applied = getattr(b, f)(**kwargs)
            result_blocks = extend_blocks(applied, result_blocks)

        out = type(self).from_blocks(result_blocks, self.axes)
        return out

    # Alias so we can share code with ArrayManager
    apply_with_block = apply

    def setitem(self, indexer, value, warn: bool = True) -> Self:
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")

        if warn and warn_copy_on_write() and not self._has_no_reference(0):
            warnings.warn(
                COW_WARNING_GENERAL_MSG,
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        elif using_copy_on_write() and not self._has_no_reference(0):
            # this method is only called if there is a single block -> hardcoded 0
            # Split blocks to only copy the columns we want to modify
            if self.ndim == 2 and isinstance(indexer, tuple):
                blk_loc = self.blklocs[indexer[1]]
                if is_list_like(blk_loc) and blk_loc.ndim == 2:
                    blk_loc = np.squeeze(blk_loc, axis=0)
                elif not is_list_like(blk_loc):
                    # Keep dimension and copy data later
                    blk_loc = [blk_loc]  # type: ignore[assignment]
                if len(blk_loc) == 0:
                    return self.copy(deep=False)

                values = self.blocks[0].values
                if values.ndim == 2:
                    values = values[blk_loc]
                    # "T" has no attribute "_iset_split_block"
                    self._iset_split_block(  # type: ignore[attr-defined]
                        0, blk_loc, values
                    )

                    indexer = list(indexer)
                    # first block equals values we are setting to -> set to all columns
                    if lib.is_integer(indexer[1]):
                        col_indexer = 0
                    elif len(blk_loc) > 1:
                        col_indexer = slice(None)  # type: ignore[assignment]
                    else:
                        col_indexer = np.arange(len(blk_loc))  # type: ignore[assignment]
                    indexer[1] = col_indexer

                    row_indexer = indexer[0]
                    if isinstance(row_indexer, np.ndarray) and row_indexer.ndim == 2:
                        # numpy cannot handle a 2d indexer in combo with a slice
                        row_indexer = np.squeeze(row_indexer, axis=1)
                    if isinstance(row_indexer, np.ndarray) and len(row_indexer) == 0:
                        # numpy does not like empty indexer combined with slice
                        # and we are setting nothing anyway
                        return self
                    indexer[0] = row_indexer
                    self.blocks[0].setitem(tuple(indexer), value)
                    return self
            # No need to split if we either set all columns or on a single block
            # manager
            self = self.copy()

        return self.apply("setitem", indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        # only reached with self.ndim == 2
        return self.apply("diff", n=n)

    def astype(self, dtype, copy: bool | None = False, errors: str = "raise") -> Self:
        if copy is None:
            if using_copy_on_write():
                copy = False
            else:
                copy = True
        elif using_copy_on_write():
            copy = False

        return self.apply(
            "astype",
            dtype=dtype,
            copy=copy,
            errors=errors,
            using_cow=using_copy_on_write(),
        )

    def convert(self, copy: bool | None) -> Self:
        if copy is None:
            if using_copy_on_write():
                copy = False
            else:
                copy = True
        elif using_copy_on_write():
            copy = False

        return self.apply("convert", copy=copy, using_cow=using_copy_on_write())

    def convert_dtypes(self, **kwargs):
        if using_copy_on_write():
            copy = False
        else:
            copy = True

        return self.apply(
            "convert_dtypes", copy=copy, using_cow=using_copy_on_write(), **kwargs
        )

    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Self:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        return self.apply(
            "get_values_for_csv",
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        return any(block.is_extension for block in self.blocks)

    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        if len(self.blocks) == 1:
            return self.blocks[0].is_view

        # It is technically possible to figure out which blocks are views
        # e.g. [ b.values.base is not None for b in self.blocks ]
        # but then we have the case of possibly some blocks being a view
        # and some blocks not. setting in theory is possible on the non-view
        # blocks w/o causing a SettingWithCopy raise/warn. But this is a bit
        # complicated

        return False

    def _get_data_subset(self, predicate: Callable) -> Self:
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        return self._combine(blocks)

    def get_bool_data(self) -> Self:
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.
        """

        new_blocks = []

        for blk in self.blocks:
            if blk.dtype == bool:
                new_blocks.append(blk)

            elif blk.is_object:
                nbs = blk._split()
                new_blocks.extend(nb for nb in nbs if nb.is_bool)

        return self._combine(new_blocks)

    def get_numeric_data(self) -> Self:
        numeric_blocks = [blk for blk in self.blocks if blk.is_numeric]
        if len(numeric_blocks) == len(self.blocks):
            # Avoid somewhat expensive _combine
            return self
        return self._combine(numeric_blocks)

    def _combine(self, blocks: list[Block], index: Index | None = None) -> Self:
        """return a new manager with the blocks"""
        if len(blocks) == 0:
            if self.ndim == 2:
                # retain our own Index dtype
                if index is not None:
                    axes = [self.items[:0], index]
                else:
                    axes = [self.items[:0]] + self.axes[1:]
                return self.make_empty(axes)
            return self.make_empty()

        # FIXME: optimization potential
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])

        new_blocks: list[Block] = []
        for b in blocks:
            nb = b.copy(deep=False)
            nb.mgr_locs = BlockPlacement(inv_indexer[nb.mgr_locs.indexer])
            new_blocks.append(nb)

        axes = list(self.axes)
        if index is not None:
            axes[-1] = index
        axes[0] = self.items.take(indexer)

        return type(self).from_blocks(new_blocks, axes)

    @property
    def nblocks(self) -> int:
        return len(self.blocks)

    def copy(self, deep: bool | None | Literal["all"] = True) -> Self:
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : bool, string or None, default True
            If False or None, return a shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        if deep is None:
            if using_copy_on_write():
                # use shallow copy
                deep = False
            else:
                # preserve deep copy for BlockManager with copy=None
                deep = True

        # this preserves the notion of view copying of axes
        if deep:
            # hit in e.g. tests.io.json.test_pandas

            def copy_func(ax):
                return ax.copy(deep=True) if deep == "all" else ax.view()

            new_axes = [copy_func(ax) for ax in self.axes]
        else:
            if using_copy_on_write():
                new_axes = [ax.view() for ax in self.axes]
            else:
                new_axes = list(self.axes)

        res = self.apply("copy", deep=deep)
        res.axes = new_axes

        if self.ndim > 1:
            # Avoid needing to re-compute these
            blknos = self._blknos
            if blknos is not None:
                res._blknos = blknos.copy()
                res._blklocs = self._blklocs.copy()

        if deep:
            res._consolidate_inplace()
        return res

    def consolidate(self) -> Self:
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        if self.is_consolidated():
            return self

        bm = type(self)(self.blocks, self.axes, verify_integrity=False)
        bm._is_consolidated = False
        bm._consolidate_inplace()
        return bm

    def reindex_indexer(
        self,
        new_axis: Index,
        indexer: npt.NDArray[np.intp] | None,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool | None = True,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool or None, default True
            If None, regard as False to get shallow copy.
        only_slice : bool, default False
            Whether to take views, not copies, along columns.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.

        pandas-indexer with -1's only.
        """
        if copy is None:
            if using_copy_on_write():
                # use shallow copy
                copy = False
            else:
                # preserve deep copy for BlockManager with copy=None
                copy = True

        if indexer is None:
            if new_axis is self.axes[axis] and not copy:
                return self

            result = self.copy(deep=copy)
            result.axes = list(self.axes)
            result.axes[axis] = new_axis
            return result

        # Should be intp, but in some cases we get int64 on 32bit builds
        assert isinstance(indexer, np.ndarray)

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self.axes[axis]._validate_can_reindex(indexer)

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 0:
            new_blocks = self._slice_take_blocks_ax0(
                indexer,
                fill_value=fill_value,
                only_slice=only_slice,
                use_na_proxy=use_na_proxy,
            )
        else:
            new_blocks = [
                blk.take_nd(
                    indexer,
                    axis=1,
                    fill_value=(
                        fill_value if fill_value is not None else blk.fill_value
                    ),
                )
                for blk in self.blocks
            ]

        new_axes = list(self.axes)
        new_axes[axis] = new_axis

        new_mgr = type(self).from_blocks(new_blocks, new_axes)
        if axis == 1:
            # We can avoid the need to rebuild these
            new_mgr._blknos = self.blknos.copy()
            new_mgr._blklocs = self.blklocs.copy()
        return new_mgr

    def _slice_take_blocks_ax0(
        self,
        slice_or_indexer: slice | np.ndarray,
        fill_value=lib.no_default,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
        ref_inplace_op: bool = False,
    ) -> list[Block]:
        """
        Slice/take blocks along axis=0.

        Overloaded for SingleBlock

        Parameters
        ----------
        slice_or_indexer : slice or np.ndarray[int64]
        fill_value : scalar, default lib.no_default
        only_slice : bool, default False
            If True, we always return views on existing arrays, never copies.
            This is used when called from ops.blockwise.operate_blockwise.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.
        ref_inplace_op: bool, default False
            Don't track refs if True because we operate inplace

        Returns
        -------
        new_blocks : list of Block
        """
        allow_fill = fill_value is not lib.no_default

        sl_type, slobj, sllen = _preprocess_slice_or_indexer(
            slice_or_indexer, self.shape[0], allow_fill=allow_fill
        )

        if self.is_single_block:
            blk = self.blocks[0]

            if sl_type == "slice":
                # GH#32959 EABlock would fail since we can't make 0-width
                # TODO(EA2D): special casing unnecessary with 2D EAs
                if sllen == 0:
                    return []
                bp = BlockPlacement(slice(0, sllen))
                return [blk.getitem_block_columns(slobj, new_mgr_locs=bp)]
            elif not allow_fill or self.ndim == 1:
                if allow_fill and fill_value is None:
                    fill_value = blk.fill_value

                if not allow_fill and only_slice:
                    # GH#33597 slice instead of take, so we get
                    #  views instead of copies
                    blocks = [
                        blk.getitem_block_columns(
                            slice(ml, ml + 1),
                            new_mgr_locs=BlockPlacement(i),
                            ref_inplace_op=ref_inplace_op,
                        )
                        for i, ml in enumerate(slobj)
                    ]
                    return blocks
                else:
                    bp = BlockPlacement(slice(0, sllen))
                    return [
                        blk.take_nd(
                            slobj,
                            axis=0,
                            new_mgr_locs=bp,
                            fill_value=fill_value,
                        )
                    ]

        if sl_type == "slice":
            blknos = self.blknos[slobj]
            blklocs = self.blklocs[slobj]
        else:
            blknos = algos.take_nd(
                self.blknos, slobj, fill_value=-1, allow_fill=allow_fill
            )
            blklocs = algos.take_nd(
                self.blklocs, slobj, fill_value=-1, allow_fill=allow_fill
            )

        # When filling blknos, make sure blknos is updated before appending to
        # blocks list, that way new blkno is exactly len(blocks).
        blocks = []
        group = not only_slice
        for blkno, mgr_locs in libinternals.get_blkno_placements(blknos, group=group):
            if blkno == -1:
                # If we've got here, fill_value was not lib.no_default

                blocks.append(
                    self._make_na_block(
                        placement=mgr_locs,
                        fill_value=fill_value,
                        use_na_proxy=use_na_proxy,
                    )
                )
            else:
                blk = self.blocks[blkno]

                # Otherwise, slicing along items axis is necessary.
                if not blk._can_consolidate and not blk._validate_ndim:
                    # i.e. we dont go through here for DatetimeTZBlock
                    # A non-consolidatable block, it's easy, because there's
                    # only one item and each mgr loc is a copy of that single
                    # item.
                    deep = not (only_slice or using_copy_on_write())
                    for mgr_loc in mgr_locs:
                        newblk = blk.copy(deep=deep)
                        newblk.mgr_locs = BlockPlacement(slice(mgr_loc, mgr_loc + 1))
                        blocks.append(newblk)

                else:
                    # GH#32779 to avoid the performance penalty of copying,
                    #  we may try to only slice
                    taker = blklocs[mgr_locs.indexer]
                    max_len = max(len(mgr_locs), taker.max() + 1)
                    if only_slice or using_copy_on_write():
                        taker = lib.maybe_indices_to_slice(taker, max_len)

                    if isinstance(taker, slice):
                        nb = blk.getitem_block_columns(taker, new_mgr_locs=mgr_locs)
                        blocks.append(nb)
                    elif only_slice:
                        # GH#33597 slice instead of take, so we get
                        #  views instead of copies
                        for i, ml in zip(taker, mgr_locs):
                            slc = slice(i, i + 1)
                            bp = BlockPlacement(ml)
                            nb = blk.getitem_block_columns(slc, new_mgr_locs=bp)
                            # We have np.shares_memory(nb.values, blk.values)
                            blocks.append(nb)
                    else:
                        nb = blk.take_nd(taker, axis=0, new_mgr_locs=mgr_locs)
                        blocks.append(nb)

        return blocks

    def _make_na_block(
        self, placement: BlockPlacement, fill_value=None, use_na_proxy: bool = False
    ) -> Block:
        # Note: we only get here with self.ndim == 2

        if use_na_proxy:
            assert fill_value is None
            shape = (len(placement), self.shape[1])
            vals = np.empty(shape, dtype=np.void)
            nb = NumpyBlock(vals, placement, ndim=2)
            return nb

        if fill_value is None:
            fill_value = np.nan

        shape = (len(placement), self.shape[1])

        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        block_values = make_na_array(dtype, shape, fill_value)
        return new_block_2d(block_values, placement=placement)

    def take(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt = 1,
        verify: bool = True,
    ) -> Self:
        """
        Take items along any axis.

        indexer : np.ndarray[np.intp]
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
        # Caller is responsible for ensuring indexer annotation is accurate

        n = self.shape[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)

        new_labels = self.axes[axis].take(indexer)
        return self.reindex_indexer(
            new_axis=new_labels,
            indexer=indexer,
            axis=axis,
            allow_dups=True,
            copy=None,
        )


class BlockManager(libinternals.BlockManager, BaseBlockManager):
    """
    BaseBlockManager that holds 2D blocks.
    """

    ndim = 2

    # ----------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        blocks: Sequence[Block],
        axes: Sequence[Index],
        verify_integrity: bool = True,
    ) -> None:
        if verify_integrity:
            # Assertion disabled for performance
            # assert all(isinstance(x, Index) for x in axes)

            for block in blocks:
                if self.ndim != block.ndim:
                    raise AssertionError(
                        f"Number of Block dimensions ({block.ndim}) must equal "
                        f"number of axes ({self.ndim})"
                    )
                # As of 2.0, the caller is responsible for ensuring that
                #  DatetimeTZBlock with block.ndim == 2 has block.values.ndim ==2;
                #  previously there was a special check for fastparquet compat.

            self._verify_integrity()

    def _verify_integrity(self) -> None:
        mgr_shape = self.shape
        tot_items = sum(len(x.mgr_locs) for x in self.blocks)
        for block in self.blocks:
            if block.shape[1:] != mgr_shape[1:]:
                raise_construction_error(tot_items, block.shape[1:], self.axes)
        if len(self.items) != tot_items:
            raise AssertionError(
                "Number of manager items must equal union of "
                f"block items\n# manager items: {len(self.items)}, # "
                f"tot_items: {tot_items}"
            )

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        return cls(blocks, axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Indexing

    def fast_xs(self, loc: int) -> SingleBlockManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if len(self.blocks) == 1:
            # TODO: this could be wrong if blk.mgr_locs is not slice(None)-like;
            #  is this ruled out in the general case?
            result = self.blocks[0].iget((slice(None), loc))
            # in the case of a single block, the new block is a view
            bp = BlockPlacement(slice(0, len(result)))
            block = new_block(
                result,
                placement=bp,
                ndim=1,
                refs=self.blocks[0].refs,
            )
            return SingleBlockManager(block, self.axes[0])

        dtype = interleaved_dtype([blk.dtype for blk in self.blocks])

        n = len(self)

        if isinstance(dtype, ExtensionDtype):
            # TODO: use object dtype as workaround for non-performant
            #  EA.__setitem__ methods. (primarily ArrowExtensionArray.__setitem__
            #  when iteratively setting individual values)
            #  https://github.com/pandas-dev/pandas/pull/54508#issuecomment-1675827918
            result = np.empty(n, dtype=object)
        else:
            result = np.empty(n, dtype=dtype)
            result = ensure_wrapped_if_datetimelike(result)

        for blk in self.blocks:
            # Such assignment may incorrectly coerce NaT to None
            # result[blk.mgr_locs] = blk._slice((slice(None), loc))
            for i, rl in enumerate(blk.mgr_locs):
                result[rl] = blk.iget((i, loc))

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)

        bp = BlockPlacement(slice(0, len(result)))
        block = new_block(result, placement=bp, ndim=1)
        return SingleBlockManager(block, self.axes[0])

    def iget(self, i: int, track_ref: bool = True) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])

        # shortcut for select a single-dim from a 2-dim BM
        bp = BlockPlacement(slice(0, len(values)))
        nb = type(block)(
            values, placement=bp, ndim=1, refs=block.refs if track_ref else None
        )
        return SingleBlockManager(nb, self.axes[1])

    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
        # TODO(CoW) making the arrays read-only might make this safer to use?
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return values

    @property
    def column_arrays(self) -> list[np.ndarray]:
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
        # This is an optimized equivalent to
        #  result = [self.iget_values(i) for i in range(len(self.items))]
        result: list[np.ndarray | None] = [None] * len(self.items)

        for blk in self.blocks:
            mgr_locs = blk._mgr_locs
            values = blk.array_values._values_for_json()
            if values.ndim == 1:
                # TODO(EA2D): special casing not needed with 2D EAs
                result[mgr_locs[0]] = values

            else:
                for i, loc in enumerate(mgr_locs):
                    result[loc] = values[i]

        # error: Incompatible return value type (got "List[None]",
        # expected "List[ndarray[Any, Any]]")
        return result  # type: ignore[return-value]

    def iset(
        self,
        loc: int | slice | np.ndarray,
        value: ArrayLike,
        inplace: bool = False,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """

        # FIXME: refactor, clearly separate broadcasting & zip-like assignment
        #        can prob also fix the various if tests for sparse/categorical
        if self._blklocs is None and self.ndim > 1:
            self._rebuild_blknos_and_blklocs()

        # Note: we exclude DTA/TDA here
        value_is_extension_type = is_1d_only_ea_dtype(value.dtype)
        if not value_is_extension_type:
            if value.ndim == 2:
                value = value.T
            else:
                value = ensure_block_shape(value, ndim=2)

            if value.shape[1:] != self.shape[1:]:
                raise AssertionError(
                    "Shape of new values must be compatible with manager shape"
                )

        if lib.is_integer(loc):
            # We have 6 tests where loc is _not_ an int.
            # In this case, get_blkno_placements will yield only one tuple,
            #  containing (self._blknos[loc], BlockPlacement(slice(0, 1, 1)))

            # Check if we can use _iset_single fastpath
            loc = cast(int, loc)
            blkno = self.blknos[loc]
            blk = self.blocks[blkno]
            if len(blk._mgr_locs) == 1:  # TODO: fastest way to check this?
                return self._iset_single(
                    loc,
                    value,
                    inplace=inplace,
                    blkno=blkno,
                    blk=blk,
                    refs=refs,
                )

            # error: Incompatible types in assignment (expression has type
            # "List[Union[int, slice, ndarray]]", variable has type "Union[int,
            # slice, ndarray]")
            loc = [loc]  # type: ignore[assignment]

        # categorical/sparse/datetimetz
        if value_is_extension_type:

            def value_getitem(placement):
                return value

        else:

            def value_getitem(placement):
                return value[placement.indexer]

        # Accessing public blknos ensures the public versions are initialized
        blknos = self.blknos[loc]
        blklocs = self.blklocs[loc].copy()

        unfit_mgr_locs = []
        unfit_val_locs = []
        removed_blknos = []
        for blkno_l, val_locs in libinternals.get_blkno_placements(blknos, group=True):
            blk = self.blocks[blkno_l]
            blk_locs = blklocs[val_locs.indexer]
            if inplace and blk.should_store(value):
                # Updating inplace -> check if we need to do Copy-on-Write
                if using_copy_on_write() and not self._has_no_reference_block(blkno_l):
                    self._iset_split_block(
                        blkno_l, blk_locs, value_getitem(val_locs), refs=refs
                    )
                else:
                    blk.set_inplace(blk_locs, value_getitem(val_locs))
                    continue
            else:
                unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
                unfit_val_locs.append(val_locs)

                # If all block items are unfit, schedule the block for removal.
                if len(val_locs) == len(blk.mgr_locs):
                    removed_blknos.append(blkno_l)
                    continue
                else:
                    # Defer setting the new values to enable consolidation
                    self._iset_split_block(blkno_l, blk_locs, refs=refs)

        if len(removed_blknos):
            # Remove blocks & update blknos accordingly
            is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
            is_deleted[removed_blknos] = True

            new_blknos = np.empty(self.nblocks, dtype=np.intp)
            new_blknos.fill(-1)
            new_blknos[~is_deleted] = np.arange(self.nblocks - len(removed_blknos))
            self._blknos = new_blknos[self._blknos]
            self.blocks = tuple(
                blk for i, blk in enumerate(self.blocks) if i not in set(removed_blknos)
            )

        if unfit_val_locs:
            unfit_idxr = np.concatenate(unfit_mgr_locs)
            unfit_count = len(unfit_idxr)

            new_blocks: list[Block] = []
            if value_is_extension_type:
                # This code (ab-)uses the fact that EA blocks contain only
                # one item.
                # TODO(EA2D): special casing unnecessary with 2D EAs
                new_blocks.extend(
                    new_block_2d(
                        values=value,
                        placement=BlockPlacement(slice(mgr_loc, mgr_loc + 1)),
                        refs=refs,
                    )
                    for mgr_loc in unfit_idxr
                )

                self._blknos[unfit_idxr] = np.arange(unfit_count) + len(self.blocks)
                self._blklocs[unfit_idxr] = 0

            else:
                # unfit_val_locs contains BlockPlacement objects
                unfit_val_items = unfit_val_locs[0].append(unfit_val_locs[1:])

                new_blocks.append(
                    new_block_2d(
                        values=value_getitem(unfit_val_items),
                        placement=BlockPlacement(unfit_idxr),
                        refs=refs,
                    )
                )

                self._blknos[unfit_idxr] = len(self.blocks)
                self._blklocs[unfit_idxr] = np.arange(unfit_count)

            self.blocks += tuple(new_blocks)

            # Newly created block's dtype may already be present.
            self._known_consolidated = False

    def _iset_split_block(
        self,
        blkno_l: int,
        blk_locs: np.ndarray | list[int],
        value: ArrayLike | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determinint the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
        blk = self.blocks[blkno_l]

        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()

        nbs_tup = tuple(blk.delete(blk_locs))
        if value is not None:
            locs = blk.mgr_locs.as_array[blk_locs]
            first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
        else:
            first_nb = nbs_tup[0]
            nbs_tup = tuple(nbs_tup[1:])

        nr_blocks = len(self.blocks)
        blocks_tup = (
            self.blocks[:blkno_l] + (first_nb,) + self.blocks[blkno_l + 1 :] + nbs_tup
        )
        self.blocks = blocks_tup

        if not nbs_tup and value is not None:
            # No need to update anything if split did not happen
            return

        self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))

        for i, nb in enumerate(nbs_tup):
            self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
            self._blknos[nb.mgr_locs.indexer] = i + nr_blocks

    def _iset_single(
        self,
        loc: int,
        value: ArrayLike,
        inplace: bool,
        blkno: int,
        blk: Block,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
        # Caller is responsible for verifying value.shape

        if inplace and blk.should_store(value):
            copy = False
            if using_copy_on_write() and not self._has_no_reference_block(blkno):
                # perform Copy-on-Write and clear the reference
                copy = True
            iloc = self.blklocs[loc]
            blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
            return

        nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
        old_blocks = self.blocks
        new_blocks = old_blocks[:blkno] + (nb,) + old_blocks[blkno + 1 :]
        self.blocks = new_blocks
        return

    def column_setitem(
        self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = False
    ) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        needs_to_warn = False
        if warn_copy_on_write() and not self._has_no_reference(loc):
            if not isinstance(
                self.blocks[self.blknos[loc]].values,
                (ArrowExtensionArray, ArrowStringArray),
            ):
                # We might raise if we are in an expansion case, so defer
                # warning till we actually updated
                needs_to_warn = True

        elif using_copy_on_write() and not self._has_no_reference(loc):
            blkno = self.blknos[loc]
            # Split blocks to only copy the column we want to modify
            blk_loc = self.blklocs[loc]
            # Copy our values
            values = self.blocks[blkno].values
            if values.ndim == 1:
                values = values.copy()
            else:
                # Use [blk_loc] as indexer to keep ndim=2, this already results in a
                # copy
                values = values[[blk_loc]]
            self._iset_split_block(blkno, [blk_loc], values)

        # this manager is only created temporarily to mutate the values in place
        # so don't track references, otherwise the `setitem` would perform CoW again
        col_mgr = self.iget(loc, track_ref=False)
        if inplace_only:
            col_mgr.setitem_inplace(idx, value)
        else:
            new_mgr = col_mgr.setitem((idx,), value)
            self.iset(loc, new_mgr._block.values, inplace=True)

        if needs_to_warn:
            warnings.warn(
                COW_WARNING_GENERAL_MSG,
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        refs : The reference tracking object of the value to set.
        """
        with warnings.catch_warnings():
            # TODO: re-issue this with setitem-specific message?
            warnings.filterwarnings(
                "ignore",
                "The behavior of Index.insert with object-dtype is deprecated",
                category=FutureWarning,
            )
            new_axis = self.items.insert(loc, item)

        if value.ndim == 2:
            value = value.T
            if len(value) > 1:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.T.shape}"
                )
        else:
            value = ensure_block_shape(value, ndim=self.ndim)

        bp = BlockPlacement(slice(loc, loc + 1))
        block = new_block_2d(values=value, placement=bp, refs=refs)

        if not len(self.blocks):
            # Fastpath
            self._blklocs = np.array([0], dtype=np.intp)
            self._blknos = np.array([0], dtype=np.intp)
        else:
            self._insert_update_mgr_locs(loc)
            self._insert_update_blklocs_and_blknos(loc)

        self.axes[0] = new_axis
        self.blocks += (block,)

        self._known_consolidated = False

        if sum(not block.is_extension for block in self.blocks) > 100:
            warnings.warn(
                "DataFrame is highly fragmented.  This is usually the result "
                "of calling `frame.insert` many times, which has poor performance.  "
                "Consider joining all columns at once using pd.concat(axis=1) "
                "instead. To get a de-fragmented frame, use `newframe = frame.copy()`",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

    def _insert_update_mgr_locs(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we increment
        all of the mgr_locs of blocks above that by one.
        """
        for blkno, count in _fast_count_smallints(self.blknos[loc:]):
            # .620 this way, .326 of which is in increment_above
            blk = self.blocks[blkno]
            blk._mgr_locs = blk._mgr_locs.increment_above(loc)

    def _insert_update_blklocs_and_blknos(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """

        # Accessing public blklocs ensures the public versions are initialized
        if loc == self.blklocs.shape[0]:
            # np.append is a lot faster, let's use it if we can.
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        elif loc == 0:
            # np.append is a lot faster, let's use it if we can.
            self._blklocs = np.append(self._blklocs[::-1], 0)[::-1]
            self._blknos = np.append(self._blknos[::-1], len(self.blocks))[::-1]
        else:
            new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(
                self.blklocs, self.blknos, loc, len(self.blocks)
            )
            self._blklocs = new_blklocs
            self._blknos = new_blknos

    def idelete(self, indexer) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        taker = (~is_deleted).nonzero()[0]

        nbs = self._slice_take_blocks_ax0(taker, only_slice=True, ref_inplace_op=True)
        new_columns = self.items[~is_deleted]
        axes = [new_columns, self.axes[1]]
        return type(self)(tuple(nbs), axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Block-wise Operation

    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
        result_blocks: list[Block] = []

        for blk in self.blocks:
            if blk.is_object:
                # split on object-dtype blocks bc some columns may raise
                #  while others do not.
                for sb in blk._split():
                    applied = sb.apply(func)
                    result_blocks = extend_blocks(applied, result_blocks)
            else:
                applied = blk.apply(func)
                result_blocks = extend_blocks(applied, result_blocks)

        if len(result_blocks) == 0:
            nrows = 0
        else:
            nrows = result_blocks[0].values.shape[-1]
        index = Index(range(nrows))

        return type(self).from_blocks(result_blocks, [self.axes[0], index])

    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        BlockManager
        """
        # If 2D, we assume that we're operating column-wise
        assert self.ndim == 2

        res_blocks: list[Block] = []
        for blk in self.blocks:
            nbs = blk.reduce(func)
            res_blocks.extend(nbs)

        index = Index([None])  # placeholder
        new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return new_mgr

    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        return operate_blockwise(self, other, array_op)

    def _equal_values(self: BlockManager, other: BlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        return blockwise_all(self, other, array_equals)

    def quantile(
        self,
        *,
        qs: Index,  # with dtype float 64
        interpolation: QuantileInterpolation = "linear",
    ) -> Self:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
        # Series dispatches to DataFrame for quantile, which allows us to
        #  simplify some of the code here and in the blocks
        assert self.ndim >= 2
        assert is_list_like(qs)  # caller is responsible for this

        new_axes = list(self.axes)
        new_axes[1] = Index(qs, dtype=np.float64)

        blocks = [
            blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
        ]

        return type(self)(blocks, new_axes)

    # ----------------------------------------------------------------

    def unstack(self, unstacker, fill_value) -> BlockManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        new_columns = unstacker.get_new_columns(self.items)
        new_index = unstacker.new_index

        allow_fill = not unstacker.mask_all
        if allow_fill:
            # calculating the full mask once and passing it to Block._unstack is
            #  faster than letting calculating it in each repeated call
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        else:
            needs_masking = np.zeros(unstacker.full_shape[1], dtype=bool)

        new_blocks: list[Block] = []
        columns_mask: list[np.ndarray] = []

        if len(self.items) == 0:
            factor = 1
        else:
            fac = len(new_columns) / len(self.items)
            assert fac == int(fac)
            factor = int(fac)

        for blk in self.blocks:
            mgr_locs = blk.mgr_locs
            new_placement = mgr_locs.tile_for_unstack(factor)

            blocks, mask = blk._unstack(
                unstacker,
                fill_value,
                new_placement=new_placement,
                needs_masking=needs_masking,
            )

            new_blocks.extend(blocks)
            columns_mask.extend(mask)

            # Block._unstack should ensure this holds,
            assert mask.sum() == sum(len(nb._mgr_locs) for nb in blocks)
            # In turn this ensures that in the BlockManager call below
            #  we have len(new_columns) == sum(x.shape[0] for x in new_blocks)
            #  which suffices to allow us to pass verify_inegrity=False

        new_columns = new_columns[columns_mask]

        bm = BlockManager(new_blocks, [new_columns, new_index], verify_integrity=False)
        return bm

    def to_dict(self) -> dict[str, Self]:
        """
        Return a dict of str(dtype) -> BlockManager

        Returns
        -------
        values : a dict of dtype -> BlockManager
        """

        bd: dict[str, list[Block]] = {}
        for b in self.blocks:
            bd.setdefault(str(b.dtype), []).append(b)

        # TODO(EA2D): the combine will be unnecessary with 2D EAs
        return {dtype: self._combine(blocks) for dtype, blocks in bd.items()}

    def as_array(
        self,
        dtype: np.dtype | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        passed_nan = lib.is_float(na_value) and isna(na_value)

        if len(self.blocks) == 0:
            arr = np.empty(self.shape, dtype=float)
            return arr.transpose()

        if self.is_single_block:
            blk = self.blocks[0]

            if na_value is not lib.no_default:
                # We want to copy when na_value is provided to avoid
                # mutating the original object
                if lib.is_np_dtype(blk.dtype, "f") and passed_nan:
                    # We are already numpy-float and na_value=np.nan
                    pass
                else:
                    copy = True

            if blk.is_extension:
                # Avoid implicit conversion of extension blocks to object

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                    copy=copy,
                ).reshape(blk.shape)
            elif not copy:
                arr = np.asarray(blk.values, dtype=dtype)
            else:
                arr = np.array(blk.values, dtype=dtype, copy=copy)

            if using_copy_on_write() and not copy:
                arr = arr.view()
                arr.flags.writeable = False
        else:
            arr = self._interleave(dtype=dtype, na_value=na_value)
            # The underlying data was copied within _interleave, so no need
            # to further copy if copy=True or setting na_value

        if na_value is lib.no_default:
            pass
        elif arr.dtype.kind == "f" and passed_nan:
            pass
        else:
            arr[isna(arr)] = na_value

        return arr.transpose()

    def _interleave(
        self,
        dtype: np.dtype | None = None,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        if not dtype:
            # Incompatible types in assignment (expression has type
            # "Optional[Union[dtype[Any], ExtensionDtype]]", variable has
            # type "Optional[dtype[Any]]")
            dtype = interleaved_dtype(  # type: ignore[assignment]
                [blk.dtype for blk in self.blocks]
            )

        # error: Argument 1 to "ensure_np_dtype" has incompatible type
        # "Optional[dtype[Any]]"; expected "Union[dtype[Any], ExtensionDtype]"
        dtype = ensure_np_dtype(dtype)  # type: ignore[arg-type]
        result = np.empty(self.shape, dtype=dtype)

        itemmask = np.zeros(self.shape[0])

        if dtype == np.dtype("object") and na_value is lib.no_default:
            # much more performant than using to_numpy below
            for blk in self.blocks:
                rl = blk.mgr_locs
                arr = blk.get_values(dtype)
                result[rl.indexer] = arr
                itemmask[rl.indexer] = 1
            return result

        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                # Avoid implicit conversion of extension blocks to object

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                )
            else:
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            itemmask[rl.indexer] = 1

        if not itemmask.all():
            raise AssertionError("Some items were not contained in blocks")

        return result

    # ----------------------------------------------------------------
    # Consolidation

    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self) -> None:
        if len(self.blocks) == 1:
            # fastpath
            self._is_consolidated = True
            self._known_consolidated = True
            return
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True

    def _consolidate_inplace(self) -> None:
        # In general, _consolidate_inplace should only be called via
        #  DataFrame._consolidate_inplace, otherwise we will fail to invalidate
        #  the DataFrame's _item_cache. The exception is for newly-created
        #  BlockManager objects not yet attached to a DataFrame.
        if not self.is_consolidated():
            self.blocks = _consolidate(self.blocks)
            self._is_consolidated = True
            self._known_consolidated = True
            self._rebuild_blknos_and_blklocs()

    # ----------------------------------------------------------------
    # Concatenation

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
        offset = 0
        blocks: list[Block] = []
        for mgr in mgrs:
            for blk in mgr.blocks:
                # We need to do getitem_block here otherwise we would be altering
                #  blk.mgr_locs in place, which would render it invalid. This is only
                #  relevant in the copy=False case.
                nb = blk.slice_block_columns(slice(None))
                nb._mgr_locs = nb._mgr_locs.add(offset)
                blocks.append(nb)

            offset += len(mgr.items)

        new_mgr = cls(tuple(blocks), axes)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
        raise NotImplementedError("This logic lives (for now) in internals.concat")


class SingleBlockManager(BaseBlockManager, SingleDataManager):
    """manage a single block with"""

    @property
    def ndim(self) -> Literal[1]:
        return 1

    _is_consolidated = True
    _known_consolidated = True
    __slots__ = ()
    is_single_block = True

    def __init__(
        self,
        block: Block,
        axis: Index,
        verify_integrity: bool = False,
    ) -> None:
        # Assertions disabled for performance
        # assert isinstance(block, Block), type(block)
        # assert isinstance(axis, Index), type(axis)

        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(
        cls,
        blocks: list[Block],
        axes: list[Index],
    ) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        assert len(blocks) == 1
        assert len(axes) == 1
        return cls(blocks[0], axes[0], verify_integrity=False)

    @classmethod
    def from_array(
        cls, array: ArrayLike, index: Index, refs: BlockValuesRefs | None = None
    ) -> SingleBlockManager:
        """
        Constructor for if we have an array that is not yet a Block.
        """
        array = maybe_coerce_values(array)
        bp = BlockPlacement(slice(0, len(index)))
        block = new_block(array, placement=bp, ndim=1, refs=refs)
        return cls(block, index)

    def to_2d_mgr(self, columns: Index) -> BlockManager:
        """
        Manager analogue of Series.to_frame
        """
        blk = self.blocks[0]
        arr = ensure_block_shape(blk.values, ndim=2)
        bp = BlockPlacement(0)
        new_blk = type(blk)(arr, placement=bp, ndim=2, refs=blk.refs)
        axes = [columns, self.axes[0]]
        return BlockManager([new_blk], axes=axes, verify_integrity=False)

    def _has_no_reference(self, i: int = 0) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        return not self.blocks[0].refs.has_reference()

    def __getstate__(self):
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)

        extra_state = {
            "0.14.1": {
                "axes": axes_array,
                "blocks": [
                    {"values": b.values, "mgr_locs": b.mgr_locs.indexer}
                    for b in self.blocks
                ],
            }
        }

        # First three elements of the state are to maintain forward
        # compatibility with 0.13.1.
        return axes_array, block_values, block_items, extra_state

    def __setstate__(self, state) -> None:
        def unpickle_block(values, mgr_locs, ndim: int) -> Block:
            # TODO(EA2D): ndim would be unnecessary with 2D EAs
            # older pickles may store e.g. DatetimeIndex instead of DatetimeArray
            values = extract_array(values, extract_numpy=True)
            if not isinstance(mgr_locs, BlockPlacement):
                mgr_locs = BlockPlacement(mgr_locs)

            values = maybe_coerce_values(values)
            return new_block(values, placement=mgr_locs, ndim=ndim)

        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            state = state[3]["0.14.1"]
            self.axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(self.axes)
            self.blocks = tuple(
                unpickle_block(b["values"], b["mgr_locs"], ndim=ndim)
                for b in state["blocks"]
            )
        else:
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        self._post_setstate()

    def _post_setstate(self) -> None:
        pass

    @cache_readonly
    def _block(self) -> Block:
        return self.blocks[0]

    @property
    def _blknos(self):
        """compat with BlockManager"""
        return None

    @property
    def _blklocs(self):
        """compat with BlockManager"""
        return None

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self:
        # similar to get_slice, but not restricted to slice indexer
        blk = self._block
        if using_copy_on_write() and len(indexer) > 0 and indexer.all():
            return type(self)(blk.copy(deep=False), self.index)
        array = blk.values[indexer]

        if isinstance(indexer, np.ndarray) and indexer.dtype.kind == "b":
            # boolean indexing always gives a copy with numpy
            refs = None
        else:
            # TODO(CoW) in theory only need to track reference if new_array is a view
            refs = blk.refs

        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(array, placement=bp, ndim=1, refs=refs)

        new_idx = self.index[indexer]
        return type(self)(block, new_idx)

    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> SingleBlockManager:
        # Assertion disabled for performance
        # assert isinstance(slobj, slice), type(slobj)
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        blk = self._block
        array = blk.values[slobj]
        bp = BlockPlacement(slice(0, len(array)))
        # TODO this method is only used in groupby SeriesSplitter at the moment,
        # so passing refs is not yet covered by the tests
        block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
        new_index = self.index._getitem_slice(slobj)
        return type(self)(block, new_index)

    @property
    def index(self) -> Index:
        return self.axes[0]

    @property
    def dtype(self) -> DtypeObj:
        return self._block.dtype

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        return np.array([self._block.dtype], dtype=object)

    def external_values(self):
        """The array that Series.values returns"""
        return self._block.external_values()

    def internal_values(self):
        """The array that Series._values returns"""
        return self._block.values

    def array_values(self) -> ExtensionArray:
        """The array that Series.array returns"""
        return self._block.array_values

    def get_numeric_data(self) -> Self:
        if self._block.is_numeric:
            return self.copy(deep=False)
        return self.make_empty()

    @property
    def _can_hold_na(self) -> bool:
        return self._block._can_hold_na

    def setitem_inplace(self, indexer, value, warn: bool = True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        using_cow = using_copy_on_write()
        warn_cow = warn_copy_on_write()
        if (using_cow or warn_cow) and not self._has_no_reference(0):
            if using_cow:
                self.blocks = (self._block.copy(),)
                self._cache.clear()
            elif warn_cow and warn:
                warnings.warn(
                    COW_WARNING_SETITEM_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        super().setitem_inplace(indexer, value)

    def idelete(self, indexer) -> SingleBlockManager:
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
        nb = self._block.delete(indexer)[0]
        self.blocks = (nb,)
        self.axes[0] = self.axes[0].delete(indexer)
        self._cache.clear()
        return self

    def fast_xs(self, loc):
        """
        fast path for getting a cross-section
        return a view of the data
        """
        raise NotImplementedError("Use series._values[loc] instead")

    def set_values(self, values: ArrayLike) -> None:
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc),
        and this does not properly keep track of references.
        """
        # NOTE(CoW) Currently this is only used for FrameColumnApply.series_generator
        # which handles CoW by setting the refs manually if necessary
        self.blocks[0].values = values
        self.blocks[0]._mgr_locs = BlockPlacement(slice(len(values)))

    def _equal_values(self, other: Self) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        # For SingleBlockManager (i.e.Series)
        if other.ndim != 1:
            return False
        left = self.blocks[0].values
        right = other.blocks[0].values
        return array_equals(left, right)


# --------------------------------------------------------------------
# Constructor Helpers


def create_block_manager_from_blocks(
    blocks: list[Block],
    axes: list[Index],
    consolidate: bool = True,
    verify_integrity: bool = True,
) -> BlockManager:
    # If verify_integrity=False, then caller is responsible for checking
    #  all(x.shape[-1] == len(axes[1]) for x in blocks)
    #  sum(x.shape[0] for x in blocks) == len(axes[0])
    #  set(x for blk in blocks for x in blk.mgr_locs) == set(range(len(axes[0])))
    #  all(blk.ndim == 2 for blk in blocks)
    # This allows us to safely pass verify_integrity=False

    try:
        mgr = BlockManager(blocks, axes, verify_integrity=verify_integrity)

    except ValueError as err:
        arrays = [blk.values for blk in blocks]
        tot_items = sum(arr.shape[0] for arr in arrays)
        raise_construction_error(tot_items, arrays[0].shape[1:], axes, err)

    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def create_block_manager_from_column_arrays(
    arrays: list[ArrayLike],
    axes: list[Index],
    consolidate: bool,
    refs: list,
) -> BlockManager:
    # Assertions disabled for performance (caller is responsible for verifying)
    # assert isinstance(axes, list)
    # assert all(isinstance(x, Index) for x in axes)
    # assert all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
    # assert all(type(x) is not NumpyExtensionArray for x in arrays)
    # assert all(x.ndim == 1 for x in arrays)
    # assert all(len(x) == len(axes[1]) for x in arrays)
    # assert len(arrays) == len(axes[0])
    # These last three are sufficient to allow us to safely pass
    #  verify_integrity=False below.

    try:
        blocks = _form_blocks(arrays, consolidate, refs)
        mgr = BlockManager(blocks, axes, verify_integrity=False)
    except ValueError as e:
        raise_construction_error(len(arrays), arrays[0].shape, axes, e)
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def raise_construction_error(
    tot_items: int,
    block_shape: Shape,
    axes: list[Index],
    e: ValueError | None = None,
):
    """raise a helpful message about our construction"""
    passed = tuple(map(int, [tot_items] + list(block_shape)))
    # Correcting the user facing error message during dataframe construction
    if len(passed) <= 2:
        passed = passed[::-1]

    implied = tuple(len(ax) for ax in axes)
    # Correcting the user facing error message during dataframe construction
    if len(implied) <= 2:
        implied = implied[::-1]

    # We return the exception object instead of raising it so that we
    #  can raise it in the caller; mypy plays better with that
    if passed == implied and e is not None:
        raise e
    if block_shape[0] == 0:
        raise ValueError("Empty data passed with indices specified.")
    raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


# -----------------------------------------------------------------------


def _grouping_func(tup: tuple[int, ArrayLike]) -> tuple[int, DtypeObj]:
    dtype = tup[1].dtype

    if is_1d_only_ea_dtype(dtype):
        # We know these won't be consolidated, so don't need to group these.
        # This avoids expensive comparisons of CategoricalDtype objects
        sep = id(dtype)
    else:
        sep = 0

    return sep, dtype


def _form_blocks(arrays: list[ArrayLike], consolidate: bool, refs: list) -> list[Block]:
    tuples = list(enumerate(arrays))

    if not consolidate:
        return _tuples_to_blocks_no_consolidate(tuples, refs)

    # when consolidating, we can ignore refs (either stacking always copies,
    # or the EA is already copied in the calling dict_to_mgr)

    # group by dtype
    grouper = itertools.groupby(tuples, _grouping_func)

    nbs: list[Block] = []
    for (_, dtype), tup_block in grouper:
        block_type = get_block_type(dtype)

        if isinstance(dtype, np.dtype):
            is_dtlike = dtype.kind in "mM"

            if issubclass(dtype.type, (str, bytes)):
                dtype = np.dtype(object)

            values, placement = _stack_arrays(list(tup_block), dtype)
            if is_dtlike:
                values = ensure_wrapped_if_datetimelike(values)
            blk = block_type(values, placement=BlockPlacement(placement), ndim=2)
            nbs.append(blk)

        elif is_1d_only_ea_dtype(dtype):
            dtype_blocks = [
                block_type(x[1], placement=BlockPlacement(x[0]), ndim=2)
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)

        else:
            dtype_blocks = [
                block_type(
                    ensure_block_shape(x[1], 2), placement=BlockPlacement(x[0]), ndim=2
                )
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)
    return nbs


def _tuples_to_blocks_no_consolidate(tuples, refs) -> list[Block]:
    # tuples produced within _form_blocks are of the form (placement, array)
    return [
        new_block_2d(
            ensure_block_shape(arr, ndim=2), placement=BlockPlacement(i), refs=ref
        )
        for ((i, arr), ref) in zip(tuples, refs)
    ]


def _stack_arrays(tuples, dtype: np.dtype):
    placement, arrays = zip(*tuples)

    first = arrays[0]
    shape = (len(arrays),) + first.shape

    stacked = np.empty(shape, dtype=dtype)
    for i, arr in enumerate(arrays):
        stacked[i] = arr

    return stacked, placement


def _consolidate(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
    """
    Merge blocks having same dtype, exclude non-consolidating blocks
    """
    # sort by _can_consolidate, dtype
    gkey = lambda x: x._consolidate_key
    grouper = itertools.groupby(sorted(blocks, key=gkey), gkey)

    new_blocks: list[Block] = []
    for (_can_consolidate, dtype), group_blocks in grouper:
        merged_blocks, _ = _merge_blocks(
            list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
        )
        new_blocks = extend_blocks(merged_blocks, new_blocks)
    return tuple(new_blocks)


def _merge_blocks(
    blocks: list[Block], dtype: DtypeObj, can_consolidate: bool
) -> tuple[list[Block], bool]:
    if len(blocks) == 1:
        return blocks, False

    if can_consolidate:
        # TODO: optimization potential in case all mgrs contain slices and
        # combination of those slices is a slice, too.
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])

        new_values: ArrayLike

        if isinstance(blocks[0].dtype, np.dtype):
            # error: List comprehension has incompatible type List[Union[ndarray,
            # ExtensionArray]]; expected List[Union[complex, generic,
            # Sequence[Union[int, float, complex, str, bytes, generic]],
            # Sequence[Sequence[Any]], SupportsArray]]
            new_values = np.vstack([b.values for b in blocks])  # type: ignore[misc]
        else:
            bvals = [blk.values for blk in blocks]
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)

        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]

        bp = BlockPlacement(new_mgr_locs)
        return [new_block_2d(new_values, placement=bp)], True

    # can't consolidate --> no merge
    return blocks, False


def _fast_count_smallints(arr: npt.NDArray[np.intp]):
    """Faster version of set(arr) for sequences of small numbers."""
    counts = np.bincount(arr)
    nz = counts.nonzero()[0]
    # Note: list(zip(...) outperforms list(np.c_[nz, counts[nz]]) here,
    #  in one benchmark by a factor of 11
    return zip(nz, counts[nz])


def _preprocess_slice_or_indexer(
    slice_or_indexer: slice | np.ndarray, length: int, allow_fill: bool
):
    if isinstance(slice_or_indexer, slice):
        return (
            "slice",
            slice_or_indexer,
            libinternals.slice_len(slice_or_indexer, length),
        )
    else:
        if (
            not isinstance(slice_or_indexer, np.ndarray)
            or slice_or_indexer.dtype.kind != "i"
        ):
            dtype = getattr(slice_or_indexer, "dtype", None)
            raise TypeError(type(slice_or_indexer), dtype)

        indexer = ensure_platform_int(slice_or_indexer)
        if not allow_fill:
            indexer = maybe_convert_indices(indexer, length)
        return "fancy", indexer, len(indexer)


def make_na_array(dtype: DtypeObj, shape: Shape, fill_value) -> ArrayLike:
    if isinstance(dtype, DatetimeTZDtype):
        # NB: exclude e.g. pyarrow[dt64tz] dtypes
        ts = Timestamp(fill_value).as_unit(dtype.unit)
        i8values = np.full(shape, ts._value)
        dt64values = i8values.view(f"M8[{dtype.unit}]")
        return DatetimeArray._simple_new(dt64values, dtype=dtype)

    elif is_1d_only_ea_dtype(dtype):
        dtype = cast(ExtensionDtype, dtype)
        cls = dtype.construct_array_type()

        missing_arr = cls._from_sequence([], dtype=dtype)
        ncols, nrows = shape
        assert ncols == 1, ncols
        empty_arr = -1 * np.ones((nrows,), dtype=np.intp)
        return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
    elif isinstance(dtype, ExtensionDtype):
        # TODO: no tests get here, a handful would if we disabled
        #  the dt64tz special-case above (which is faster)
        cls = dtype.construct_array_type()
        missing_arr = cls._empty(shape=shape, dtype=dtype)
        missing_arr[:] = fill_value
        return missing_arr
    else:
        # NB: we should never get here with dtype integer or bool;
        #  if we did, the missing_arr.fill would cast to gibberish
        missing_arr = np.empty(shape, dtype=dtype)
        missing_arr.fill(fill_value)

        if dtype.kind in "mM":
            missing_arr = ensure_wrapped_if_datetimelike(missing_arr)
        return missing_arr
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    NaT,
    algos as libalgos,
    internals as libinternals,
    lib,
)
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    ensure_dtype_can_hold_na,
    find_common_type,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_scalar,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    SparseDtype,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    isna_all,
)

from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
    ensure_block_shape,
    new_block_2d,
)
from pandas.core.internals.managers import (
    BlockManager,
    make_na_array,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        Manager2D,
        Shape,
    )

    from pandas import Index
    from pandas.core.internals.blocks import (
        Block,
        BlockPlacement,
    )


def _concatenate_array_managers(
    mgrs: list[ArrayManager], axes: list[Index], concat_axis: AxisInt
) -> Manager2D:
    """
    Concatenate array managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (ArrayManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int

    Returns
    -------
    ArrayManager
    """
    if concat_axis == 1:
        return mgrs[0].concat_vertical(mgrs, axes)
    else:
        # concatting along the columns -> combine reindexed arrays in a single manager
        assert concat_axis == 0
        return mgrs[0].concat_horizontal(mgrs, axes)


def concatenate_managers(
    mgrs_indexers, axes: list[Index], concat_axis: AxisInt, copy: bool
) -> Manager2D:
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int
    copy : bool

    Returns
    -------
    BlockManager
    """

    needs_copy = copy and concat_axis == 0

    # TODO(ArrayManager) this assumes that all managers are of the same type
    if isinstance(mgrs_indexers[0][0], ArrayManager):
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        # error: Argument 1 to "_concatenate_array_managers" has incompatible
        # type "List[BlockManager]"; expected "List[Union[ArrayManager,
        # SingleArrayManager, BlockManager, SingleBlockManager]]"
        return _concatenate_array_managers(
            mgrs, axes, concat_axis  # type: ignore[arg-type]
        )

    # Assertions disabled for performance
    # for tup in mgrs_indexers:
    #    # caller is responsible for ensuring this
    #    indexers = tup[1]
    #    assert concat_axis not in indexers

    if concat_axis == 0:
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        return mgrs[0].concat_horizontal(mgrs, axes)

    if len(mgrs_indexers) > 0 and mgrs_indexers[0][0].nblocks > 0:
        first_dtype = mgrs_indexers[0][0].blocks[0].dtype
        if first_dtype in [np.float64, np.float32]:
            # TODO: support more dtypes here.  This will be simpler once
            #  JoinUnit.is_na behavior is deprecated.
            if (
                all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in mgrs_indexers)
                and len(mgrs_indexers) > 1
            ):
                # Fastpath!
                # Length restriction is just to avoid having to worry about 'copy'
                shape = tuple(len(x) for x in axes)
                nb = _concat_homogeneous_fastpath(mgrs_indexers, shape, first_dtype)
                return BlockManager((nb,), axes)

    mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)

    if len(mgrs) == 1:
        mgr = mgrs[0]
        out = mgr.copy(deep=False)
        out.axes = axes
        return out

    concat_plan = _get_combined_plan(mgrs)

    blocks = []
    values: ArrayLike

    for placement, join_units in concat_plan:
        unit = join_units[0]
        blk = unit.block

        if _is_uniform_join_units(join_units):
            vals = [ju.block.values for ju in join_units]

            if not blk.is_extension:
                # _is_uniform_join_units ensures a single dtype, so
                #  we can use np.concatenate, which is more performant
                #  than concat_compat
                # error: Argument 1 to "concatenate" has incompatible type
                # "List[Union[ndarray[Any, Any], ExtensionArray]]";
                # expected "Union[_SupportsArray[dtype[Any]],
                # _NestedSequence[_SupportsArray[dtype[Any]]]]"
                values = np.concatenate(vals, axis=1)  # type: ignore[arg-type]
            elif is_1d_only_ea_dtype(blk.dtype):
                # TODO(EA2D): special-casing not needed with 2D EAs
                values = concat_compat(vals, axis=0, ea_compat_axis=True)
                values = ensure_block_shape(values, ndim=2)
            else:
                values = concat_compat(vals, axis=1)

            values = ensure_wrapped_if_datetimelike(values)

            fastpath = blk.values.dtype == values.dtype
        else:
            values = _concatenate_join_units(join_units, copy=copy)
            fastpath = False

        if fastpath:
            b = blk.make_block_same_class(values, placement=placement)
        else:
            b = new_block_2d(values, placement=placement)

        blocks.append(b)

    return BlockManager(tuple(blocks), axes)


def _maybe_reindex_columns_na_proxy(
    axes: list[Index],
    mgrs_indexers: list[tuple[BlockManager, dict[int, np.ndarray]]],
    needs_copy: bool,
) -> list[BlockManager]:
    """
    Reindex along columns so that all of the BlockManagers being concatenated
    have matching columns.

    Columns added in this reindexing have dtype=np.void, indicating they
    should be ignored when choosing a column's final dtype.
    """
    new_mgrs = []

    for mgr, indexers in mgrs_indexers:
        # For axis=0 (i.e. columns) we use_na_proxy and only_slice, so this
        #  is a cheap reindexing.
        for i, indexer in indexers.items():
            mgr = mgr.reindex_indexer(
                axes[i],
                indexers[i],
                axis=i,
                copy=False,
                only_slice=True,  # only relevant for i==0
                allow_dups=True,
                use_na_proxy=True,  # only relevant for i==0
            )
        if needs_copy and not indexers:
            mgr = mgr.copy()

        new_mgrs.append(mgr)
    return new_mgrs


def _is_homogeneous_mgr(mgr: BlockManager, first_dtype: DtypeObj) -> bool:
    """
    Check if this Manager can be treated as a single ndarray.
    """
    if mgr.nblocks != 1:
        return False
    blk = mgr.blocks[0]
    if not (blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1):
        return False

    return blk.dtype == first_dtype


def _concat_homogeneous_fastpath(
    mgrs_indexers, shape: Shape, first_dtype: np.dtype
) -> Block:
    """
    With single-Block managers with homogeneous dtypes (that can already hold nan),
    we avoid [...]
    """
    # assumes
    #  all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in in mgrs_indexers)

    if all(not indexers for _, indexers in mgrs_indexers):
        # https://github.com/pandas-dev/pandas/pull/52685#issuecomment-1523287739
        arrs = [mgr.blocks[0].values.T for mgr, _ in mgrs_indexers]
        arr = np.concatenate(arrs).T
        bp = libinternals.BlockPlacement(slice(shape[0]))
        nb = new_block_2d(arr, bp)
        return nb

    arr = np.empty(shape, dtype=first_dtype)

    if first_dtype == np.float64:
        take_func = libalgos.take_2d_axis0_float64_float64
    else:
        take_func = libalgos.take_2d_axis0_float32_float32

    start = 0
    for mgr, indexers in mgrs_indexers:
        mgr_len = mgr.shape[1]
        end = start + mgr_len

        if 0 in indexers:
            take_func(
                mgr.blocks[0].values,
                indexers[0],
                arr[:, start:end],
            )
        else:
            # No reindexing necessary, we can copy values directly
            arr[:, start:end] = mgr.blocks[0].values

        start += mgr_len

    bp = libinternals.BlockPlacement(slice(shape[0]))
    nb = new_block_2d(arr, bp)
    return nb


def _get_combined_plan(
    mgrs: list[BlockManager],
) -> list[tuple[BlockPlacement, list[JoinUnit]]]:
    plan = []

    max_len = mgrs[0].shape[0]

    blknos_list = [mgr.blknos for mgr in mgrs]
    pairs = libinternals.get_concat_blkno_indexers(blknos_list)
    for ind, (blknos, bp) in enumerate(pairs):
        # assert bp.is_slice_like
        # assert len(bp) > 0

        units_for_bp = []
        for k, mgr in enumerate(mgrs):
            blkno = blknos[k]

            nb = _get_block_for_concat_plan(mgr, bp, blkno, max_len=max_len)
            unit = JoinUnit(nb)
            units_for_bp.append(unit)

        plan.append((bp, units_for_bp))

    return plan


def _get_block_for_concat_plan(
    mgr: BlockManager, bp: BlockPlacement, blkno: int, *, max_len: int
) -> Block:
    blk = mgr.blocks[blkno]
    # Assertions disabled for performance:
    #  assert bp.is_slice_like
    #  assert blkno != -1
    #  assert (mgr.blknos[bp] == blkno).all()

    if len(bp) == len(blk.mgr_locs) and (
        blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1
    ):
        nb = blk
    else:
        ax0_blk_indexer = mgr.blklocs[bp.indexer]

        slc = lib.maybe_indices_to_slice(ax0_blk_indexer, max_len)
        # TODO: in all extant test cases 2023-04-08 we have a slice here.
        #  Will this always be the case?
        if isinstance(slc, slice):
            nb = blk.slice_block_columns(slc)
        else:
            nb = blk.take_block_columns(slc)

    # assert nb.shape == (len(bp), mgr.shape[1])
    return nb


class JoinUnit:
    def __init__(self, block: Block) -> None:
        self.block = block

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.block)})"

    def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
        """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
        if not self.is_na:
            return False

        blk = self.block
        if blk.dtype.kind == "V":
            return True

        if blk.dtype == object:
            values = blk.values
            return all(is_valid_na_for_dtype(x, dtype) for x in values.ravel(order="K"))

        na_value = blk.fill_value
        if na_value is NaT and blk.dtype != dtype:
            # e.g. we are dt64 and other is td64
            # fill_values match but we should not cast blk.values to dtype
            # TODO: this will need updating if we ever have non-nano dt64/td64
            return False

        if na_value is NA and needs_i8_conversion(dtype):
            # FIXME: kludge; test_append_empty_frame_with_timedelta64ns_nat
            #  e.g. blk.dtype == "Int64" and dtype is td64, we dont want
            #  to consider these as matching
            return False

        # TODO: better to use can_hold_element?
        return is_valid_na_for_dtype(na_value, dtype)

    @cache_readonly
    def is_na(self) -> bool:
        blk = self.block
        if blk.dtype.kind == "V":
            return True

        if not blk._can_hold_na:
            return False

        values = blk.values
        if values.size == 0:
            # GH#39122 this case will return False once deprecation is enforced
            return True

        if isinstance(values.dtype, SparseDtype):
            return False

        if values.ndim == 1:
            # TODO(EA2D): no need for special case with 2D EAs
            val = values[0]
            if not is_scalar(val) or not isna(val):
                # ideally isna_all would do this short-circuiting
                return False
            return isna_all(values)
        else:
            val = values[0][0]
            if not is_scalar(val) or not isna(val):
                # ideally isna_all would do this short-circuiting
                return False
            return all(isna_all(row) for row in values)

    @cache_readonly
    def is_na_after_size_and_isna_all_deprecation(self) -> bool:
        """
        Will self.is_na be True after values.size == 0 deprecation and isna_all
        deprecation are enforced?
        """
        blk = self.block
        if blk.dtype.kind == "V":
            return True
        return False

    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike:
        values: ArrayLike

        if upcasted_na is None and self.block.dtype.kind != "V":
            # No upcasting is necessary
            return self.block.values
        else:
            fill_value = upcasted_na

            if self._is_valid_na_for(empty_dtype):
                # note: always holds when self.block.dtype.kind == "V"
                blk_dtype = self.block.dtype

                if blk_dtype == np.dtype("object"):
                    # we want to avoid filling with np.nan if we are
                    # using None; we already know that we are all
                    # nulls
                    values = cast(np.ndarray, self.block.values)
                    if values.size and values[0, 0] is None:
                        fill_value = None

                return make_na_array(empty_dtype, self.block.shape, fill_value)

            return self.block.values


def _concatenate_join_units(join_units: list[JoinUnit], copy: bool) -> ArrayLike:
    """
    Concatenate values from several join units along axis=1.
    """
    empty_dtype, empty_dtype_future = _get_empty_dtype(join_units)

    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)
    upcasted_na = _dtype_to_na_value(empty_dtype, has_none_blocks)

    to_concat = [
        ju.get_reindexed_values(empty_dtype=empty_dtype, upcasted_na=upcasted_na)
        for ju in join_units
    ]

    if any(is_1d_only_ea_dtype(t.dtype) for t in to_concat):
        # TODO(EA2D): special case not needed if all EAs used HybridBlocks

        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[int, slice]"
        to_concat = [
            t
            if is_1d_only_ea_dtype(t.dtype)
            else t[0, :]  # type: ignore[call-overload]
            for t in to_concat
        ]
        concat_values = concat_compat(to_concat, axis=0, ea_compat_axis=True)
        concat_values = ensure_block_shape(concat_values, 2)

    else:
        concat_values = concat_compat(to_concat, axis=1)

    if empty_dtype != empty_dtype_future:
        if empty_dtype == concat_values.dtype:
            # GH#39122, GH#40893
            warnings.warn(
                "The behavior of DataFrame concatenation with empty or all-NA "
                "entries is deprecated. In a future version, this will no longer "
                "exclude empty or all-NA columns when determining the result dtypes. "
                "To retain the old behavior, exclude the relevant entries before "
                "the concat operation.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
    return concat_values


def _dtype_to_na_value(dtype: DtypeObj, has_none_blocks: bool):
    """
    Find the NA value to go with this dtype.
    """
    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        return dtype.type("NaT")
    elif dtype.kind in "fc":
        return dtype.type("NaN")
    elif dtype.kind == "b":
        # different from missing.na_value_for_dtype
        return None
    elif dtype.kind in "iu":
        if not has_none_blocks:
            # different from missing.na_value_for_dtype
            return None
        return np.nan
    elif dtype.kind == "O":
        return np.nan
    raise NotImplementedError


def _get_empty_dtype(join_units: Sequence[JoinUnit]) -> tuple[DtypeObj, DtypeObj]:
    """
    Return dtype and N/A values to use when concatenating specified units.

    Returned N/A value may be None which means there was no casting involved.

    Returns
    -------
    dtype
    """
    if lib.dtypes_all_equal([ju.block.dtype for ju in join_units]):
        empty_dtype = join_units[0].block.dtype
        return empty_dtype, empty_dtype

    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)

    dtypes = [unit.block.dtype for unit in join_units if not unit.is_na]
    if not len(dtypes):
        dtypes = [
            unit.block.dtype for unit in join_units if unit.block.dtype.kind != "V"
        ]

    dtype = find_common_type(dtypes)
    if has_none_blocks:
        dtype = ensure_dtype_can_hold_na(dtype)

    dtype_future = dtype
    if len(dtypes) != len(join_units):
        dtypes_future = [
            unit.block.dtype
            for unit in join_units
            if not unit.is_na_after_size_and_isna_all_deprecation
        ]
        if not len(dtypes_future):
            dtypes_future = [
                unit.block.dtype for unit in join_units if unit.block.dtype.kind != "V"
            ]

        if len(dtypes) != len(dtypes_future):
            dtype_future = find_common_type(dtypes_future)
            if has_none_blocks:
                dtype_future = ensure_dtype_can_hold_na(dtype_future)

    return dtype, dtype_future


def _is_uniform_join_units(join_units: list[JoinUnit]) -> bool:
    """
    Check if the join units consist of blocks of uniform type that can
    be concatenated using Block.concat_same_type instead of the generic
    _concatenate_join_units (which uses `concat_compat`).

    """
    first = join_units[0].block
    if first.dtype.kind == "V":
        return False
    return (
        # exclude cases where a) ju.block is None or b) we have e.g. Int64+int64
        all(type(ju.block) is type(first) for ju in join_units)
        and
        # e.g. DatetimeLikeBlock can be dt64 or td64, but these are not uniform
        all(
            ju.block.dtype == first.dtype
            # GH#42092 we only want the dtype_equal check for non-numeric blocks
            #  (for now, may change but that would need a deprecation)
            or ju.block.dtype.kind in "iub"
            for ju in join_units
        )
        and
        # no blocks that would get missing values (can lead to type upcasts)
        # unless we're an extension dtype.
        all(not ju.is_na or ju.block.is_extension for ju in join_units)
    )
<|endoftext|>from pandas.core.internals.api import make_block  # 2023-09-18 pyarrow uses this
from pandas.core.internals.array_manager import (
    ArrayManager,
    SingleArrayManager,
)
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
)
from pandas.core.internals.concat import concatenate_managers
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
)

__all__ = [
    "Block",  # pylint: disable=undefined-all-variable
    "DatetimeTZBlock",  # pylint: disable=undefined-all-variable
    "ExtensionBlock",  # pylint: disable=undefined-all-variable
    "make_block",
    "DataManager",
    "ArrayManager",
    "BlockManager",
    "SingleDataManager",
    "SingleBlockManager",
    "SingleArrayManager",
    "concatenate_managers",
]


def __getattr__(name: str):
    # GH#55139
    import warnings

    if name == "create_block_manager_from_blocks":
        # GH#33892
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )
        from pandas.core.internals.managers import create_block_manager_from_blocks

        return create_block_manager_from_blocks

    if name in [
        "NumericBlock",
        "ObjectBlock",
        "Block",
        "ExtensionBlock",
        "DatetimeTZBlock",
    ]:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )
        if name == "NumericBlock":
            from pandas.core.internals.blocks import NumericBlock

            return NumericBlock
        elif name == "DatetimeTZBlock":
            from pandas.core.internals.blocks import DatetimeTZBlock

            return DatetimeTZBlock
        elif name == "ExtensionBlock":
            from pandas.core.internals.blocks import ExtensionBlock

            return ExtensionBlock
        elif name == "Block":
            from pandas.core.internals.blocks import Block

            return Block
        else:
            from pandas.core.internals.blocks import ObjectBlock

            return ObjectBlock

    raise AttributeError(f"module 'pandas.core.internals' has no attribute '{name}'")
<|endoftext|>from __future__ import annotations

from functools import wraps
import inspect
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    final,
)
import warnings
import weakref

import numpy as np

from pandas._config import (
    get_option,
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import (
    NaT,
    internals as libinternals,
    lib,
)
from pandas._libs.internals import (
    BlockPlacement,
    BlockValuesRefs,
)
from pandas._libs.missing import NA
from pandas._typing import (
    ArrayLike,
    AxisInt,
    DtypeBackend,
    DtypeObj,
    F,
    FillnaOptions,
    IgnoreRaise,
    InterpolateOptions,
    QuantileInterpolation,
    Self,
    Shape,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.astype import (
    astype_array_safe,
    astype_is_view,
)
from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    convert_dtypes,
    find_result_type,
    maybe_downcast_to_dtype,
    np_can_hold_element,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    NumpyEADtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCNumpyExtensionArray,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_re
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
    extract_bool_array,
    putmask_inplace,
    putmask_without_repeat,
    setitem_datetimelike_compat,
    validate_putmask,
)
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
    compare_or_regex_search,
    replace_regex,
    should_use_regex,
)
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    from pandas.core.api import Index
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

# comparison is faster than is_object_dtype
_dtype_obj = np.dtype("object")


COW_WARNING_GENERAL_MSG = """\
Setting a value on a view: behaviour will change in pandas 3.0.
You are mutating a Series or DataFrame object, and currently this mutation will
also have effect on other Series or DataFrame objects that share data with this
object. In pandas 3.0 (with Copy-on-Write), updating one Series or DataFrame object
will never modify another.
"""


COW_WARNING_SETITEM_MSG = """\
Setting a value on a view: behaviour will change in pandas 3.0.
Currently, the mutation will also have effect on the object that shares data
with this object. For example, when setting a value in a Series that was
extracted from a column of a DataFrame, that DataFrame will also be updated:

    ser = df["col"]
    ser[0] = 0     <--- in pandas 2, this also updates `df`

In pandas 3.0 (with Copy-on-Write), updating one Series/DataFrame will never
modify another, and thus in the example above, `df` will not be changed.
"""


def maybe_split(meth: F) -> F:
    """
    If we have a multi-column block, split and operate block-wise.  Otherwise
    use the original method.
    """

    @wraps(meth)
    def newfunc(self, *args, **kwargs) -> list[Block]:
        if self.ndim == 1 or self.shape[0] == 1:
            return meth(self, *args, **kwargs)
        else:
            # Split and operate column-by-column
            return self.split_and_operate(meth, *args, **kwargs)

    return cast(F, newfunc)


class Block(PandasObject, libinternals.Block):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """

    values: np.ndarray | ExtensionArray
    ndim: int
    refs: BlockValuesRefs
    __init__: Callable

    __slots__ = ()
    is_numeric = False

    @final
    @cache_readonly
    def _validate_ndim(self) -> bool:
        """
        We validate dimension for blocks that can hold 2D values, which for now
        means numpy dtypes or DatetimeTZDtype.
        """
        dtype = self.dtype
        return not isinstance(dtype, ExtensionDtype) or isinstance(
            dtype, DatetimeTZDtype
        )

    @final
    @cache_readonly
    def is_object(self) -> bool:
        return self.values.dtype == _dtype_obj

    @final
    @cache_readonly
    def is_extension(self) -> bool:
        return not lib.is_np_dtype(self.values.dtype)

    @final
    @cache_readonly
    def _can_consolidate(self) -> bool:
        # We _could_ consolidate for DatetimeTZDtype but don't for now.
        return not self.is_extension

    @final
    @cache_readonly
    def _consolidate_key(self):
        return self._can_consolidate, self.dtype.name

    @final
    @cache_readonly
    def _can_hold_na(self) -> bool:
        """
        Can we store NA values in this Block?
        """
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in "iub"
        return dtype._can_hold_na

    @final
    @property
    def is_bool(self) -> bool:
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
        return self.values.dtype == np.dtype(bool)

    @final
    def external_values(self):
        return external_values(self.values)

    @final
    @cache_readonly
    def fill_value(self):
        # Used in reindex_indexer
        return na_value_for_dtype(self.dtype, compat=False)

    @final
    def _standardize_fill_value(self, value):
        # if we are passed a scalar None, convert it here
        if self.dtype != _dtype_obj and is_valid_na_for_dtype(value, self.dtype):
            value = self.fill_value
        return value

    @property
    def mgr_locs(self) -> BlockPlacement:
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: BlockPlacement) -> None:
        self._mgr_locs = new_mgr_locs

    @final
    def make_block(
        self,
        values,
        placement: BlockPlacement | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
        if placement is None:
            placement = self._mgr_locs
        if self.is_extension:
            values = ensure_block_shape(values, ndim=self.ndim)

        return new_block(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def make_block_same_class(
        self,
        values,
        placement: BlockPlacement | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> Self:
        """Wrap given values in a block of same type as self."""
        # Pre-2.0 we called ensure_wrapped_if_datetimelike because fastparquet
        #  relied on it, as of 2.0 the caller is responsible for this.
        if placement is None:
            placement = self._mgr_locs

        # We assume maybe_coerce_values has already been called
        return type(self)(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def __repr__(self) -> str:
        # don't want to print out all of the items here
        name = type(self).__name__
        if self.ndim == 1:
            result = f"{name}: {len(self)} dtype: {self.dtype}"
        else:
            shape = " x ".join([str(s) for s in self.shape])
            result = f"{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}"

        return result

    @final
    def __len__(self) -> int:
        return len(self.values)

    @final
    def slice_block_columns(self, slc: slice) -> Self:
        """
        Perform __getitem__-like, return result as block.
        """
        new_mgr_locs = self._mgr_locs[slc]

        new_values = self._slice(slc)
        refs = self.refs
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def take_block_columns(self, indices: npt.NDArray[np.intp]) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        # Note: only called from is from internals.concat, and we can verify
        #  that never happens with 1-column blocks, i.e. never for ExtensionBlock.

        new_mgr_locs = self._mgr_locs[indices]

        new_values = self._slice(indices)
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=None)

    @final
    def getitem_block_columns(
        self, slicer: slice, new_mgr_locs: BlockPlacement, ref_inplace_op: bool = False
    ) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_values = self._slice(slicer)
        refs = self.refs if not ref_inplace_op or self.refs.has_reference() else None
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def _can_hold_element(self, element: Any) -> bool:
        """require the same dtype as ourselves"""
        element = extract_array(element, extract_numpy=True)
        return can_hold_element(self.values, element)

    @final
    def should_store(self, value: ArrayLike) -> bool:
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
        return value.dtype == self.dtype

    # ---------------------------------------------------------------------
    # Apply/Reduce and Helpers

    @final
    def apply(self, func, **kwargs) -> list[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
        result = func(self.values, **kwargs)

        result = maybe_coerce_values(result)
        return self._split_op_result(result)

    @final
    def reduce(self, func) -> list[Block]:
        # We will apply the function and reshape the result into a single-row
        #  Block with the same mgr_locs; squeezing will be done at a higher level
        assert self.ndim == 2

        result = func(self.values)

        if self.values.ndim == 1:
            res_values = result
        else:
            res_values = result.reshape(-1, 1)

        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result: ArrayLike) -> list[Block]:
        # See also: split_and_operate
        if result.ndim > 1 and isinstance(result.dtype, ExtensionDtype):
            # TODO(EA2D): unnecessary with 2D EAs
            # if we get a 2D ExtensionArray, we need to split it into 1D pieces
            nbs = []
            for i, loc in enumerate(self._mgr_locs):
                if not is_1d_only_ea_dtype(result.dtype):
                    vals = result[i : i + 1]
                else:
                    vals = result[i]

                bp = BlockPlacement(loc)
                block = self.make_block(values=vals, placement=bp)
                nbs.append(block)
            return nbs

        nb = self.make_block(result)

        return [nb]

    @final
    def _split(self) -> list[Block]:
        """
        Split a block into a list of single-column blocks.
        """
        assert self.ndim == 2

        new_blocks = []
        for i, ref_loc in enumerate(self._mgr_locs):
            vals = self.values[slice(i, i + 1)]

            bp = BlockPlacement(ref_loc)
            nb = type(self)(vals, placement=bp, ndim=2, refs=self.refs)
            new_blocks.append(nb)
        return new_blocks

    @final
    def split_and_operate(self, func, *args, **kwargs) -> list[Block]:
        """
        Split the block and apply func column-by-column.

        Parameters
        ----------
        func : Block method
        *args
        **kwargs

        Returns
        -------
        List[Block]
        """
        assert self.ndim == 2 and self.shape[0] != 1

        res_blocks = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks

    # ---------------------------------------------------------------------
    # Up/Down-casting

    @final
    def coerce_to_target_dtype(
        self, other, warn_on_upcast: bool = False, using_cow: bool = False
    ) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
        new_dtype = find_result_type(self.values.dtype, other)
        if new_dtype == self.dtype:
            # GH#52927 avoid RecursionError
            raise AssertionError(
                "Something has gone wrong, please report a bug at "
                "https://github.com/pandas-dev/pandas/issues"
            )

        # In a future version of pandas, the default will be that
        # setting `nan` into an integer series won't raise.
        if (
            is_scalar(other)
            and is_integer_dtype(self.values.dtype)
            and isna(other)
            and other is not NaT
            and not (
                isinstance(other, (np.datetime64, np.timedelta64)) and np.isnat(other)
            )
        ):
            warn_on_upcast = False
        elif (
            isinstance(other, np.ndarray)
            and other.ndim == 1
            and is_integer_dtype(self.values.dtype)
            and is_float_dtype(other.dtype)
            and lib.has_only_ints_or_nan(other)
        ):
            warn_on_upcast = False

        if warn_on_upcast:
            warnings.warn(
                f"Setting an item of incompatible dtype is deprecated "
                "and will raise an error in a future version of pandas. "
                f"Value '{other}' has dtype incompatible with {self.values.dtype}, "
                "please explicitly cast to a compatible dtype first.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if self.values.dtype == new_dtype:
            raise AssertionError(
                f"Did not expect new dtype {new_dtype} to equal self.dtype "
                f"{self.values.dtype}. Please report a bug at "
                "https://github.com/pandas-dev/pandas/issues."
            )
        copy = False
        if (
            not using_cow
            and isinstance(self.dtype, StringDtype)
            and self.dtype.storage == "python"
        ):
            copy = True
        return self.astype(new_dtype, copy=copy, using_cow=using_cow)

    @final
    def _maybe_downcast(
        self,
        blocks: list[Block],
        downcast,
        using_cow: bool,
        caller: str,
    ) -> list[Block]:
        if downcast is False:
            return blocks

        if self.dtype == _dtype_obj:
            # TODO: does it matter that self.dtype might not match blocks[i].dtype?
            # GH#44241 We downcast regardless of the argument;
            #  respecting 'downcast=None' may be worthwhile at some point,
            #  but ATM it breaks too much existing code.
            # split and convert the blocks

            if caller == "fillna" and get_option("future.no_silent_downcasting"):
                return blocks

            nbs = extend_blocks(
                [
                    blk.convert(
                        using_cow=using_cow, copy=not using_cow, convert_string=False
                    )
                    for blk in blocks
                ]
            )
            if caller == "fillna":
                if len(nbs) != len(blocks) or not all(
                    x.dtype == y.dtype for x, y in zip(nbs, blocks)
                ):
                    # GH#54261
                    warnings.warn(
                        "Downcasting object dtype arrays on .fillna, .ffill, .bfill "
                        "is deprecated and will change in a future version. "
                        "Call result.infer_objects(copy=False) instead. "
                        "To opt-in to the future "
                        "behavior, set "
                        "`pd.set_option('future.no_silent_downcasting', True)`",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )

            return nbs

        elif downcast is None:
            return blocks
        elif caller == "where" and get_option("future.no_silent_downcasting") is True:
            return blocks
        else:
            nbs = extend_blocks([b._downcast_2d(downcast, using_cow) for b in blocks])

        # When _maybe_downcast is called with caller="where", it is either
        #  a) with downcast=False, which is a no-op (the desired future behavior)
        #  b) with downcast="infer", which is _not_ passed by the user.
        # In the latter case the future behavior is to stop doing inference,
        #  so we issue a warning if and only if some inference occurred.
        if caller == "where":
            # GH#53656
            if len(blocks) != len(nbs) or any(
                left.dtype != right.dtype for left, right in zip(blocks, nbs)
            ):
                # In this case _maybe_downcast was _not_ a no-op, so the behavior
                #  will change, so we issue a warning.
                warnings.warn(
                    "Downcasting behavior in Series and DataFrame methods 'where', "
                    "'mask', and 'clip' is deprecated. In a future "
                    "version this will not infer object dtypes or cast all-round "
                    "floats to integers. Instead call "
                    "result.infer_objects(copy=False) for object inference, "
                    "or cast round floats explicitly. To opt-in to the future "
                    "behavior, set "
                    "`pd.set_option('future.no_silent_downcasting', True)`",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        return nbs

    @final
    @maybe_split
    def _downcast_2d(self, dtype, using_cow: bool = False) -> list[Block]:
        """
        downcast specialized to 2D case post-validation.

        Refactored to allow use of maybe_split.
        """
        new_values = maybe_downcast_to_dtype(self.values, dtype=dtype)
        new_values = maybe_coerce_values(new_values)
        refs = self.refs if new_values is self.values else None
        return [self.make_block(new_values, refs=refs)]

    @final
    def convert(
        self,
        *,
        copy: bool = True,
        using_cow: bool = False,
        convert_string: bool = True,
    ) -> list[Block]:
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
        if not self.is_object:
            if not copy and using_cow:
                return [self.copy(deep=False)]
            return [self.copy()] if copy else [self]

        if self.ndim != 1 and self.shape[0] != 1:
            blocks = self.split_and_operate(
                Block.convert,
                copy=copy,
                using_cow=using_cow,
                convert_string=convert_string,
            )
            if all(blk.dtype.kind == "O" for blk in blocks):
                # Avoid fragmenting the block if convert is a no-op
                if using_cow:
                    return [self.copy(deep=False)]
                return [self.copy()] if copy else [self]
            return blocks

        values = self.values
        if values.ndim == 2:
            # the check above ensures we only get here with values.shape[0] == 1,
            # avoid doing .ravel as that might make a copy
            values = values[0]

        res_values = lib.maybe_convert_objects(
            values,  # type: ignore[arg-type]
            convert_non_numeric=True,
            convert_string=convert_string,
        )
        refs = None
        if (
            copy
            and res_values is values
            or isinstance(res_values, NumpyExtensionArray)
            and res_values._ndarray is values
        ):
            res_values = res_values.copy()
        elif res_values is values:
            refs = self.refs

        res_values = ensure_block_shape(res_values, self.ndim)
        res_values = maybe_coerce_values(res_values)
        return [self.make_block(res_values, refs=refs)]

    def convert_dtypes(
        self,
        copy: bool,
        using_cow: bool,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: DtypeBackend = "numpy_nullable",
    ) -> list[Block]:
        if infer_objects and self.is_object:
            blks = self.convert(copy=False, using_cow=using_cow)
        else:
            blks = [self]

        if not any(
            [convert_floating, convert_integer, convert_boolean, convert_string]
        ):
            return [b.copy(deep=copy) for b in blks]

        rbs = []
        for blk in blks:
            # Determine dtype column by column
            sub_blks = [blk] if blk.ndim == 1 or self.shape[0] == 1 else blk._split()
            dtypes = [
                convert_dtypes(
                    b.values,
                    convert_string,
                    convert_integer,
                    convert_boolean,
                    convert_floating,
                    infer_objects,
                    dtype_backend,
                )
                for b in sub_blks
            ]
            if all(dtype == self.dtype for dtype in dtypes):
                # Avoid block splitting if no dtype changes
                rbs.append(blk.copy(deep=copy))
                continue

            for dtype, b in zip(dtypes, sub_blks):
                rbs.append(b.astype(dtype=dtype, copy=copy, squeeze=b.ndim != 1))
        return rbs

    # ---------------------------------------------------------------------
    # Array-Like Methods

    @final
    @cache_readonly
    def dtype(self) -> DtypeObj:
        return self.values.dtype

    @final
    def astype(
        self,
        dtype: DtypeObj,
        copy: bool = False,
        errors: IgnoreRaise = "raise",
        using_cow: bool = False,
        squeeze: bool = False,
    ) -> Block:
        """
        Coerce to the new dtype.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
        copy : bool, default False
            copy if indicated
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object
        using_cow: bool, default False
            Signaling if copy on write copy logic is used.
        squeeze : bool, default False
            squeeze values to ndim=1 if only one column is given

        Returns
        -------
        Block
        """
        values = self.values
        if squeeze and values.ndim == 2 and is_1d_only_ea_dtype(dtype):
            if values.shape[0] != 1:
                raise ValueError("Can not squeeze with more than one column.")
            values = values[0, :]  # type: ignore[call-overload]

        new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)

        new_values = maybe_coerce_values(new_values)

        refs = None
        if (using_cow or not copy) and astype_is_view(values.dtype, new_values.dtype):
            refs = self.refs

        newb = self.make_block(new_values, refs=refs)
        if newb.shape != self.shape:
            raise TypeError(
                f"cannot set astype for copy = [{copy}] for dtype "
                f"({self.dtype.name} [{self.shape}]) to different shape "
                f"({newb.dtype.name} [{newb.shape}])"
            )
        return newb

    @final
    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Block:
        """convert to our native types format"""
        result = get_values_for_csv(
            self.values,
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )
        return self.make_block(result)

    @final
    def copy(self, deep: bool = True) -> Self:
        """copy constructor"""
        values = self.values
        refs: BlockValuesRefs | None
        if deep:
            values = values.copy()
            refs = None
        else:
            refs = self.refs
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim, refs=refs)

    # ---------------------------------------------------------------------
    # Copy-on-Write Helpers

    @final
    def _maybe_copy(self, using_cow: bool, inplace: bool) -> Self:
        if using_cow and inplace:
            deep = self.refs.has_reference()
            blk = self.copy(deep=deep)
        else:
            blk = self if inplace else self.copy()
        return blk

    @final
    def _get_refs_and_copy(self, using_cow: bool, inplace: bool):
        refs = None
        copy = not inplace
        if inplace:
            if using_cow and self.refs.has_reference():
                copy = True
            else:
                refs = self.refs
        return copy, refs

    # ---------------------------------------------------------------------
    # Replace

    @final
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = False,
        # mask may be pre-computed if we're called from replace_list
        mask: npt.NDArray[np.bool_] | None = None,
        using_cow: bool = False,
        already_warned=None,
        convert_string=None,
    ) -> list[Block]:
        """
        replace the to_replace value with value, possible to create new
        blocks here this is just a call to putmask.
        """

        # Note: the checks we do in NDFrame.replace ensure we never get
        #  here with listlike to_replace or value, as those cases
        #  go through replace_list
        values = self.values

        if isinstance(values, Categorical):
            # TODO: avoid special-casing
            # GH49404
            blk = self._maybe_copy(using_cow, inplace)
            values = cast(Categorical, blk.values)
            values._replace(to_replace=to_replace, value=value, inplace=True)
            return [blk]

        if not self._can_hold_element(to_replace):
            # We cannot hold `to_replace`, so we know immediately that
            #  replacing it is a no-op.
            # Note: If to_replace were a list, NDFrame.replace would call
            #  replace_list instead of replace.
            if using_cow:
                return [self.copy(deep=False)]
            else:
                return [self] if inplace else [self.copy()]

        if mask is None:
            mask = missing.mask_missing(values, to_replace)
        if not mask.any():
            # Note: we get here with test_replace_extension_other incorrectly
            #  bc _can_hold_element is incorrect.
            if using_cow:
                return [self.copy(deep=False)]
            else:
                return [self] if inplace else [self.copy()]

        elif self._can_hold_element(value) or (self.dtype == "string" and is_re(value)):
            # TODO(CoW): Maybe split here as well into columns where mask has True
            # and rest?
            blk = self._maybe_copy(using_cow, inplace)
            putmask_inplace(blk.values, mask, value)
            if (
                inplace
                and warn_copy_on_write()
                and already_warned is not None
                and not already_warned.warned_already
            ):
                if self.refs.has_reference():
                    warnings.warn(
                        COW_WARNING_GENERAL_MSG,
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )
                    already_warned.warned_already = True

            if not (self.is_object and value is None):
                # if the user *explicitly* gave None, we keep None, otherwise
                #  may downcast to NaN
                if get_option("future.no_silent_downcasting") is True:
                    blocks = [blk]
                else:
                    blocks = blk.convert(
                        copy=False,
                        using_cow=using_cow,
                        convert_string=convert_string or self.dtype == "string",
                    )
                    if len(blocks) > 1 or blocks[0].dtype != blk.dtype:
                        warnings.warn(
                            # GH#54710
                            "Downcasting behavior in `replace` is deprecated and "
                            "will be removed in a future version. To retain the old "
                            "behavior, explicitly call "
                            "`result.infer_objects(copy=False)`. "
                            "To opt-in to the future "
                            "behavior, set "
                            "`pd.set_option('future.no_silent_downcasting', True)`",
                            FutureWarning,
                            stacklevel=find_stack_level(),
                        )
            else:
                blocks = [blk]
            return blocks

        elif self.ndim == 1 or self.shape[0] == 1:
            if value is None or value is NA:
                blk = self.astype(np.dtype(object))
            else:
                blk = self.coerce_to_target_dtype(value, using_cow=using_cow)
            return blk.replace(
                to_replace=to_replace,
                value=value,
                inplace=True,
                mask=mask,
                using_cow=using_cow,
                convert_string=convert_string,
            )

        else:
            # split so that we only upcast where necessary
            blocks = []
            for i, nb in enumerate(self._split()):
                blocks.extend(
                    type(self).replace(
                        nb,
                        to_replace=to_replace,
                        value=value,
                        inplace=True,
                        mask=mask[i : i + 1],
                        using_cow=using_cow,
                        convert_string=convert_string,
                    )
                )
            return blocks

    @final
    def _replace_regex(
        self,
        to_replace,
        value,
        inplace: bool = False,
        mask=None,
        using_cow: bool = False,
        convert_string=None,
        already_warned=None,
    ) -> list[Block]:
        """
        Replace elements by the given value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        inplace : bool, default False
            Perform inplace modification.
        mask : array-like of bool, optional
            True indicate corresponding element is ignored.
        using_cow: bool, default False
            Specifying if copy on write is enabled.

        Returns
        -------
        List[Block]
        """
        if not is_re(to_replace) and not self._can_hold_element(to_replace):
            # i.e. only if self.is_object is True, but could in principle include a
            #  String ExtensionBlock
            if using_cow:
                return [self.copy(deep=False)]
