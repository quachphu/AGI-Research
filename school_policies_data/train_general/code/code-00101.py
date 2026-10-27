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
    def roll_table(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
        *args: Any,
    ):
        result = np.empty((len(begin), values.shape[1]))
        min_periods_mask = np.empty(result.shape)
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window), axis=0)
            sub_result = numba_func(window, *args)
            nan_mask = len(window) - count_nan >= minimum_periods
            min_periods_mask[i, :] = nan_mask
            result[i, :] = sub_result
        result = np.where(min_periods_mask, result, np.nan)
        return result

    return roll_table


# This function will no longer be needed once numba supports
# axis for all np.nan* agg functions
# https://github.com/numba/numba/issues/1269
@functools.cache
def generate_manual_numpy_nan_agg_with_axis(nan_func):
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=True, nogil=True, parallel=True)
    def nan_agg_with_axis(table):
        result = np.empty(table.shape[1])
        for i in numba.prange(table.shape[1]):
            partition = table[:, i]
            result[i] = nan_func(partition)
        return result

    return nan_agg_with_axis


@functools.cache
def generate_numba_ewm_table_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: tuple,
    normalize: bool,
):
    """
    Generate a numba jitted ewm mean or sum function applied table wise specified
    by values from engine_kwargs.

    Parameters
    ----------
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit
    com : float
    adjust : bool
    ignore_na : bool
    deltas : tuple
    normalize: bool

    Returns
    -------
    Numba function
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm_table(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
    ) -> np.ndarray:
        alpha = 1.0 / (1.0 + com)
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0 if adjust else alpha
        old_wt = np.ones(values.shape[1])

        result = np.empty(values.shape)
        weighted = values[0].copy()
        nobs = (~np.isnan(weighted)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted, np.nan)
        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted[j]):
                    if is_observations[j] or not ignore_na:
                        if normalize:
                            # note that len(deltas) = len(vals) - 1 and deltas[i]
                            # is to be used in conjunction with vals[i+1]
                            old_wt[j] *= old_wt_factor ** deltas[i - 1]
                        else:
                            weighted[j] = old_wt_factor * weighted[j]
                        if is_observations[j]:
                            if normalize:
                                # avoid numerical errors on constant series
                                if weighted[j] != cur[j]:
                                    weighted[j] = (
                                        old_wt[j] * weighted[j] + new_wt * cur[j]
                                    )
                                    if normalize:
                                        weighted[j] = weighted[j] / (old_wt[j] + new_wt)
                                if adjust:
                                    old_wt[j] += new_wt
                                else:
                                    old_wt[j] = 1.0
                            else:
                                weighted[j] += cur[j]
                elif is_observations[j]:
                    weighted[j] = cur[j]

            result[i] = np.where(nobs >= minimum_periods, weighted, np.nan)

        return result

    return ewm_table
<|endoftext|>"""
Provide a generic structure to support window functions,
similar to how we have a Groupby object.
"""
from __future__ import annotations

import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
)

import numpy as np

from pandas._libs.tslibs import (
    BaseOffset,
    Timedelta,
    to_offset,
)
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
    deprecate_kwarg,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_float64,
    is_bool,
    is_integer,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna

from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
    BaseIndexer,
    FixedWindowIndexer,
    GroupbyIndexer,
    VariableWindowIndexer,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    TimedeltaIndex,
)
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
)
from pandas.core.window.common import (
    flex_binary_moment,
    zsqrt,
)
from pandas.core.window.doc import (
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    kwargs_scipy,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
    window_apply_parameters,
)
from pandas.core.window.numba_ import (
    generate_manual_numpy_nan_agg_with_axis,
    generate_numba_apply_func,
    generate_numba_table_func,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Sized,
    )

    from pandas._typing import (
        ArrayLike,
        Axis,
        NDFrameT,
        QuantileInterpolation,
        WindowingRankType,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.ops import BaseGrouper

from pandas.core.arrays.datetimelike import dtype_to_unit


class BaseWindow(SelectionMixin):
    """Provides utilities for performing windowing operations."""

    _attributes: list[str] = []
    exclusions: frozenset[Hashable] = frozenset()
    _on: Index

    def __init__(
        self,
        obj: NDFrame,
        window=None,
        min_periods: int | None = None,
        center: bool | None = False,
        win_type: str | None = None,
        axis: Axis = 0,
        on: str | Index | None = None,
        closed: str | None = None,
        step: int | None = None,
        method: str = "single",
        *,
        selection=None,
    ) -> None:
        self.obj = obj
        self.on = on
        self.closed = closed
        self.step = step
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.axis = obj._get_axis_number(axis) if axis is not None else None
        self.method = method
        self._win_freq_i8: int | None = None
        if self.on is None:
            if self.axis == 0:
                self._on = self.obj.index
            else:
                # i.e. self.axis == 1
                self._on = self.obj.columns
        elif isinstance(self.on, Index):
            self._on = self.on
        elif isinstance(self.obj, ABCDataFrame) and self.on in self.obj.columns:
            self._on = Index(self.obj[self.on])
        else:
            raise ValueError(
                f"invalid on specified as {self.on}, "
                "must be a column (of DataFrame), an Index or None"
            )

        self._selection = selection
        self._validate()

    def _validate(self) -> None:
        if self.center is not None and not is_bool(self.center):
            raise ValueError("center must be a boolean")
        if self.min_periods is not None:
            if not is_integer(self.min_periods):
                raise ValueError("min_periods must be an integer")
            if self.min_periods < 0:
                raise ValueError("min_periods must be >= 0")
            if is_integer(self.window) and self.min_periods > self.window:
                raise ValueError(
                    f"min_periods {self.min_periods} must be <= window {self.window}"
                )
        if self.closed is not None and self.closed not in [
            "right",
            "both",
            "left",
            "neither",
        ]:
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        if not isinstance(self.obj, (ABCSeries, ABCDataFrame)):
            raise TypeError(f"invalid type: {type(self)}")
        if isinstance(self.window, BaseIndexer):
            # Validate that the passed BaseIndexer subclass has
            # a get_window_bounds with the correct signature.
            get_window_bounds_signature = inspect.signature(
                self.window.get_window_bounds
            ).parameters.keys()
            expected_signature = inspect.signature(
                BaseIndexer().get_window_bounds
            ).parameters.keys()
            if get_window_bounds_signature != expected_signature:
                raise ValueError(
                    f"{type(self.window).__name__} does not implement "
                    f"the correct signature for get_window_bounds"
                )
        if self.method not in ["table", "single"]:
            raise ValueError("method must be 'table' or 'single")
        if self.step is not None:
            if not is_integer(self.step):
                raise ValueError("step must be an integer")
            if self.step < 0:
                raise ValueError("step must be >= 0")

    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        if len(start) != len(end):
            raise ValueError(
                f"start ({len(start)}) and end ({len(end)}) bounds must be the "
                f"same length"
            )
        if len(start) != (num_vals + (self.step or 1) - 1) // (self.step or 1):
            raise ValueError(
                f"start and end bounds ({len(start)}) must be the same length "
                f"as the object ({num_vals}) divided by the step ({self.step}) "
                f"if given and rounded up"
            )

    def _slice_axis_for_step(self, index: Index, result: Sized | None = None) -> Index:
        """
        Slices the index for a given result and the preset step.
        """
        return (
            index
            if result is None or len(result) == len(index)
            else index[:: self.step]
        )

    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        """
        Validate numeric_only argument, raising if invalid for the input.

        Parameters
        ----------
        name : str
            Name of the operator (kernel).
        numeric_only : bool
            Value passed by user.
        """
        if (
            self._selected_obj.ndim == 1
            and numeric_only
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            raise NotImplementedError(
                f"{type(self).__name__}.{name} does not implement numeric_only"
            )

    def _make_numeric_only(self, obj: NDFrameT) -> NDFrameT:
        """Subset DataFrame to numeric columns.

        Parameters
        ----------
        obj : DataFrame

        Returns
        -------
        obj subset to numeric-only columns.
        """
        result = obj.select_dtypes(include=["number"], exclude=["timedelta"])
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
        # filter out the on from the object
        if self.on is not None and not isinstance(self.on, Index) and obj.ndim == 2:
            obj = obj.reindex(columns=obj.columns.difference([self.on]), copy=False)
        if obj.ndim > 1 and (numeric_only or self.axis == 1):
            # GH: 20649 in case of mixed dtype and axis=1 we have to convert everything
            # to float to calculate the complete row at once. We exclude all non-numeric
            # dtypes.
            obj = self._make_numeric_only(obj)
        if self.axis == 1:
            obj = obj.astype("float64", copy=False)
            obj._mgr = obj._mgr.consolidate()
        return obj

    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # create a new object to prevent aliasing
        if subset is None:
            subset = self.obj

        # we need to make a shallow copy of ourselves
        # with the same groupby
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}

        selection = self._infer_selection(key, subset)
        new_win = type(self)(subset, selection=selection, **kwargs)
        return new_win

    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def _dir_additions(self):
        return self.obj._dir_additions()

    def __repr__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)}"
            for attr_name in self._attributes
            if getattr(self, attr_name, None) is not None and attr_name[0] != "_"
        )
        attrs = ",".join(attrs_list)
        return f"{type(self).__name__} [{attrs}]"

    def __iter__(self) -> Iterator:
        obj = self._selected_obj.set_axis(self._on)
        obj = self._create_data(obj)
        indexer = self._get_window_indexer()

        start, end = indexer.get_window_bounds(
            num_values=len(obj),
            min_periods=self.min_periods,
            center=self.center,
            closed=self.closed,
            step=self.step,
        )
        self._check_window_bounds(start, end, len(obj))

        for s, e in zip(start, end):
            result = obj.iloc[slice(s, e)]
            yield result

    def _prep_values(self, values: ArrayLike) -> np.ndarray:
        """Convert input to numpy arrays for Cython routines"""
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(
                f"ops for {type(self).__name__} for this "
                f"dtype {values.dtype} are not implemented"
            )
        # GH #12373 : rolling functions error on float32 data
        # make sure the data is coerced to float64
        try:
            if isinstance(values, ExtensionArray):
                values = values.to_numpy(np.float64, na_value=np.nan)
            else:
                values = ensure_float64(values)
        except (ValueError, TypeError) as err:
            raise TypeError(f"cannot handle this type -> {values.dtype}") from err

        # Convert inf to nan for C funcs
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)

        return values

    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None:
        # if we have an 'on' column we want to put it back into
        # the results in the same location
        from pandas import Series

        if self.on is not None and not self._on.equals(obj.index):
            name = self._on.name
            extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
            if name in result.columns:
                # TODO: sure we want to overwrite results?
                result[name] = extra_col
            elif name in result.index.names:
                pass
            elif name in self._selected_obj.columns:
                # insert in the same location as we had in _selected_obj
                old_cols = self._selected_obj.columns
                new_cols = result.columns
                old_loc = old_cols.get_loc(name)
                overlap = new_cols.intersection(old_cols[:old_loc])
                new_loc = len(overlap)
                result.insert(new_loc, name, extra_col)
            else:
                # insert at the end
                result[name] = extra_col

    @property
    def _index_array(self) -> npt.NDArray[np.int64] | None:
        # TODO: why do we get here with e.g. MultiIndex?
        if isinstance(self._on, (PeriodIndex, DatetimeIndex, TimedeltaIndex)):
            return self._on.asi8
        elif isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM":
            return self._on.to_numpy(dtype=np.int64)
        return None

    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        """Validate and finalize result."""
        if out.shape[1] == 0 and obj.shape[1] > 0:
            raise DataError("No numeric types to aggregate")
        if out.shape[1] == 0:
            return obj.astype("float64")

        self._insert_on_column(out, obj)
        return out

    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        if isinstance(self.window, BaseIndexer):
            return self.window
        if self._win_freq_i8 is not None:
            return VariableWindowIndexer(
                index_array=self._index_array,
                window_size=self._win_freq_i8,
                center=self.center,
            )
        return FixedWindowIndexer(window_size=self.window)

    def _apply_series(
        self, homogeneous_func: Callable[..., ArrayLike], name: str | None = None
    ) -> Series:
        """
        Series version of _apply_columnwise
        """
        obj = self._create_data(self._selected_obj)

        if name == "count":
            # GH 12541: Special case for count where we support date-like types
            obj = notna(obj).astype(int)
        try:
            values = self._prep_values(obj._values)
        except (TypeError, NotImplementedError) as err:
            raise DataError("No numeric types to aggregate") from err

        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        return obj._constructor(result, index=index, name=obj.name)

    def _apply_columnwise(
        self,
        homogeneous_func: Callable[..., ArrayLike],
        name: str,
        numeric_only: bool = False,
    ) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
        self._validate_numeric_only(name, numeric_only)
        if self._selected_obj.ndim == 1:
            return self._apply_series(homogeneous_func, name)

        obj = self._create_data(self._selected_obj, numeric_only)
        if name == "count":
            # GH 12541: Special case for count where we support date-like types
            obj = notna(obj).astype(int)
            obj._mgr = obj._mgr.consolidate()

        if self.axis == 1:
            obj = obj.T

        taker = []
        res_values = []
        for i, arr in enumerate(obj._iter_column_arrays()):
            # GH#42736 operate column-wise instead of block-wise
            # As of 2.0, hfunc will raise for nuisance columns
            try:
                arr = self._prep_values(arr)
            except (TypeError, NotImplementedError) as err:
                raise DataError(
                    f"Cannot aggregate non-numeric type: {arr.dtype}"
                ) from err
            res = homogeneous_func(arr)
            res_values.append(res)
            taker.append(i)

        index = self._slice_axis_for_step(
            obj.index, res_values[0] if len(res_values) > 0 else None
        )
        df = type(obj)._from_arrays(
            res_values,
            index=index,
            columns=obj.columns.take(taker),
            verify_integrity=False,
        )

        if self.axis == 1:
            df = df.T

        return self._resolve_output(df, obj)

    def _apply_tablewise(
        self,
        homogeneous_func: Callable[..., ArrayLike],
        name: str | None = None,
        numeric_only: bool = False,
    ) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame across the entire object
        """
        if self._selected_obj.ndim == 1:
            raise ValueError("method='table' not applicable for Series objects.")
        obj = self._create_data(self._selected_obj, numeric_only)
        values = self._prep_values(obj.to_numpy())
        values = values.T if self.axis == 1 else values
        result = homogeneous_func(values)
        result = result.T if self.axis == 1 else result
        index = self._slice_axis_for_step(obj.index, result)
        columns = (
            obj.columns
            if result.shape[1] == len(obj.columns)
            else obj.columns[:: self.step]
        )
        out = obj._constructor(result, index=index, columns=columns)

        return self._resolve_output(out, obj)

    def _apply_pairwise(
        self,
        target: DataFrame | Series,
        other: DataFrame | Series | None,
        pairwise: bool | None,
        func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series],
        numeric_only: bool,
    ) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
        target = self._create_data(target, numeric_only)
        if other is None:
            other = target
            # only default unset
            pairwise = True if pairwise is None else pairwise
        elif not isinstance(other, (ABCDataFrame, ABCSeries)):
            raise ValueError("other must be a DataFrame or Series")
        elif other.ndim == 2 and numeric_only:
            other = self._make_numeric_only(other)

        return flex_binary_moment(target, other, func, pairwise=bool(pairwise))

    def _apply(
        self,
        func: Callable[..., Any],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
    ):
        """
        Rolling statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
        name : str,
        numba_args : tuple
            args to be passed when func is a numba func
        **kwargs
            additional arguments for rolling function and window function

        Returns
        -------
        y : type of input
        """
        window_indexer = self._get_window_indexer()
        min_periods = (
            self.min_periods
            if self.min_periods is not None
            else window_indexer.window_size
        )

        def homogeneous_func(values: np.ndarray):
            # calculation function

            if values.size == 0:
                return values.copy()

            def calc(x):
                start, end = window_indexer.get_window_bounds(
                    num_values=len(x),
                    min_periods=min_periods,
                    center=self.center,
                    closed=self.closed,
                    step=self.step,
                )
                self._check_window_bounds(start, end, len(x))

                return func(x, start, end, min_periods, *numba_args)

            with np.errstate(all="ignore"):
                result = calc(values)

            return result

        if self.method == "single":
            return self._apply_columnwise(homogeneous_func, name, numeric_only)
        else:
            return self._apply_tablewise(homogeneous_func, name, numeric_only)

    def _numba_apply(
        self,
        func: Callable[..., Any],
        engine_kwargs: dict[str, bool] | None = None,
        **func_kwargs,
    ):
        window_indexer = self._get_window_indexer()
        min_periods = (
            self.min_periods
            if self.min_periods is not None
            else window_indexer.window_size
        )
        obj = self._create_data(self._selected_obj)
        if self.axis == 1:
            obj = obj.T
        values = self._prep_values(obj.to_numpy())
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        start, end = window_indexer.get_window_bounds(
            num_values=len(values),
            min_periods=min_periods,
            center=self.center,
            closed=self.closed,
            step=self.step,
        )
        self._check_window_bounds(start, end, len(values))
        # For now, map everything to float to match the Cython impl
        # even though it is wrong
        # TODO: Could preserve correct dtypes in future
        # xref #53214
        dtype_mapping = executor.float_dtype_mapping
        aggregator = executor.generate_shared_aggregator(
            func,
            dtype_mapping,
            is_grouped_kernel=False,
            **get_jit_arguments(engine_kwargs),
        )
        result = aggregator(
            values.T, start=start, end=end, min_periods=min_periods, **func_kwargs
        ).T
        result = result.T if self.axis == 1 else result
        index = self._slice_axis_for_step(obj.index, result)
        if obj.ndim == 1:
            result = result.squeeze()
            out = obj._constructor(result, index=index, name=obj.name)
            return out
        else:
            columns = self._slice_axis_for_step(obj.columns, result.T)
            out = obj._constructor(result, index=index, columns=columns)
            return self._resolve_output(out, obj)

    def aggregate(self, func, *args, **kwargs):
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            return self.apply(func, raw=False, args=args, kwargs=kwargs)
        return result

    agg = aggregate


class BaseWindowGroupby(BaseWindow):
    """
    Provide the groupby windowing facilities.
    """

    _grouper: BaseGrouper
    _as_index: bool
    _attributes: list[str] = ["_grouper"]

    def __init__(
        self,
        obj: DataFrame | Series,
        *args,
        _grouper: BaseGrouper,
        _as_index: bool = True,
        **kwargs,
    ) -> None:
        from pandas.core.groupby.ops import BaseGrouper

        if not isinstance(_grouper, BaseGrouper):
            raise ValueError("Must pass a BaseGrouper object.")
        self._grouper = _grouper
        self._as_index = _as_index
        # GH 32262: It's convention to keep the grouping column in
        # groupby.<agg_func>, but unexpected to users in
        # groupby.rolling.<agg_func>
        obj = obj.drop(columns=self._grouper.names, errors="ignore")
        # GH 15354
        if kwargs.get("step") is not None:
            raise NotImplementedError("step not implemented for groupby")
        super().__init__(obj, *args, **kwargs)

    def _apply(
        self,
        func: Callable[..., Any],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
    ) -> DataFrame | Series:
        result = super()._apply(
            func,
            name,
            numeric_only,
            numba_args,
            **kwargs,
        )
        # Reconstruct the resulting MultiIndex
        # 1st set of levels = group by labels
        # 2nd set of levels = original DataFrame/Series index
        grouped_object_index = self.obj.index
        grouped_index_name = [*grouped_object_index.names]
        groupby_keys = copy.copy(self._grouper.names)
        result_index_names = groupby_keys + grouped_index_name

        drop_columns = [
            key
            for key in self._grouper.names
            if key not in self.obj.index.names or key is None
        ]

        if len(drop_columns) != len(groupby_keys):
            # Our result will have still kept the column in the result
            result = result.drop(columns=drop_columns, errors="ignore")

        codes = self._grouper.codes
        levels = copy.copy(self._grouper.levels)

        group_indices = self._grouper.indices.values()
        if group_indices:
            indexer = np.concatenate(list(group_indices))
        else:
            indexer = np.array([], dtype=np.intp)
        codes = [c.take(indexer) for c in codes]

        # if the index of the original dataframe needs to be preserved, append
        # this index (but reordered) to the codes/levels from the groupby
        if grouped_object_index is not None:
            idx = grouped_object_index.take(indexer)
            if not isinstance(idx, MultiIndex):
                idx = MultiIndex.from_arrays([idx])
            codes.extend(list(idx.codes))
            levels.extend(list(idx.levels))

        result_index = MultiIndex(
            levels, codes, names=result_index_names, verify_integrity=False
        )

        result.index = result_index
        if not self._as_index:
            result = result.reset_index(level=list(range(len(groupby_keys))))
        return result

    def _apply_pairwise(
        self,
        target: DataFrame | Series,
        other: DataFrame | Series | None,
        pairwise: bool | None,
        func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series],
        numeric_only: bool,
    ) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
        # Manually drop the grouping column first
        target = target.drop(columns=self._grouper.names, errors="ignore")
        result = super()._apply_pairwise(target, other, pairwise, func, numeric_only)
        # 1) Determine the levels + codes of the groupby levels
        if other is not None and not all(
            len(group) == len(other) for group in self._grouper.indices.values()
        ):
            # GH 42915
            # len(other) != len(any group), so must reindex (expand) the result
            # from flex_binary_moment to a "transform"-like result
            # per groupby combination
            old_result_len = len(result)
            result = concat(
                [
                    result.take(gb_indices).reindex(result.index)
                    for gb_indices in self._grouper.indices.values()
                ]
            )

            gb_pairs = (
                com.maybe_make_list(pair) for pair in self._grouper.indices.keys()
            )
            groupby_codes = []
            groupby_levels = []
            # e.g. [[1, 2], [4, 5]] as [[1, 4], [2, 5]]
            for gb_level_pair in map(list, zip(*gb_pairs)):
                labels = np.repeat(np.array(gb_level_pair), old_result_len)
                codes, levels = factorize(labels)
                groupby_codes.append(codes)
                groupby_levels.append(levels)
        else:
            # pairwise=True or len(other) == len(each group), so repeat
            # the groupby labels by the number of columns in the original object
            groupby_codes = self._grouper.codes
            # error: Incompatible types in assignment (expression has type
            # "List[Index]", variable has type "List[Union[ndarray, Index]]")
            groupby_levels = self._grouper.levels  # type: ignore[assignment]

            group_indices = self._grouper.indices.values()
            if group_indices:
                indexer = np.concatenate(list(group_indices))
            else:
                indexer = np.array([], dtype=np.intp)

            if target.ndim == 1:
                repeat_by = 1
            else:
                repeat_by = len(target.columns)
            groupby_codes = [
                np.repeat(c.take(indexer), repeat_by) for c in groupby_codes
            ]
        # 2) Determine the levels + codes of the result from super()._apply_pairwise
        if isinstance(result.index, MultiIndex):
            result_codes = list(result.index.codes)
            result_levels = list(result.index.levels)
            result_names = list(result.index.names)
        else:
            idx_codes, idx_levels = factorize(result.index)
            result_codes = [idx_codes]
            result_levels = [idx_levels]
            result_names = [result.index.name]

        # 3) Create the resulting index by combining 1) + 2)
        result_codes = groupby_codes + result_codes
        result_levels = groupby_levels + result_levels
        result_names = self._grouper.names + result_names

        result_index = MultiIndex(
            result_levels, result_codes, names=result_names, verify_integrity=False
        )
        result.index = result_index
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
        # Ensure the object we're rolling over is monotonically sorted relative
        # to the groups
        # GH 36197
        if not obj.empty:
            groupby_order = np.concatenate(list(self._grouper.indices.values())).astype(
                np.int64
            )
            obj = obj.take(groupby_order)
        return super()._create_data(obj, numeric_only)

    def _gotitem(self, key, ndim, subset=None):
        # we are setting the index on the actual object
        # here so our index is carried through to the selected obj
        # when we do the splitting for the groupby
        if self.on is not None:
            # GH 43355
            subset = self.obj.set_index(self._on)
        return super()._gotitem(key, ndim, subset=subset)


class Window(BaseWindow):
    """
    Provide rolling window calculations.

    Parameters
    ----------
    window : int, timedelta, str, offset, or BaseIndexer subclass
        Size of the moving window.

        If an integer, the fixed number of observations used for
        each window.

        If a timedelta, str, or offset, the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes.
        To learn more about the offsets & frequency strings, please see `this link
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

        If a BaseIndexer subclass, the window boundaries
        based on the defined ``get_window_bounds`` method. Additional rolling
        keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
        ``step`` will be passed to ``get_window_bounds``.

    min_periods : int, default None
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

        For a window that is specified by an offset, ``min_periods`` will default to 1.

        For a window that is specified by an integer, ``min_periods`` will default
        to the size of the window.

    center : bool, default False
        If False, set the window labels as the right edge of the window index.

        If True, set the window labels as the center of the window index.

    win_type : str, default None
        If ``None``, all points are evenly weighted.

        If a string, it must be a valid `scipy.signal window function
        <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

        Certain Scipy window types require additional parameters to be passed
        in the aggregation function. The additional parameters must match
        the keywords specified in the Scipy window type method signature.

    on : str, optional
        For a DataFrame, a column label or Index level on which
        to calculate the rolling window, rather than the DataFrame's index.

        Provided integer column is ignored and excluded from result since
        an integer index is not used to calculate the rolling window.

    axis : int or str, default 0
        If ``0`` or ``'index'``, roll across the rows.

        If ``1`` or ``'columns'``, roll across the columns.

        For `Series` this parameter is unused and defaults to 0.

        .. deprecated:: 2.1.0

            The axis keyword is deprecated. For ``axis=1``,
            transpose the DataFrame first instead.

    closed : str, default None
        If ``'right'``, the first point in the window is excluded from calculations.

        If ``'left'``, the last point in the window is excluded from calculations.

        If ``'both'``, the no points in the window are excluded from calculations.

        If ``'neither'``, the first and last points in the window are excluded
        from calculations.

        Default ``None`` (``'right'``).

    step : int, default None

        .. versionadded:: 1.5.0

        Evaluate the window at every ``step`` result, equivalent to slicing as
        ``[::step]``. ``window`` must be an integer. Using a step argument other
        than None or 1 will produce a result with a different shape than the input.

    method : str {'single', 'table'}, default 'single'

        .. versionadded:: 1.3.0

        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

    Returns
    -------
    pandas.api.typing.Window or pandas.api.typing.Rolling
        An instance of Window is returned if ``win_type`` is passed. Otherwise,
        an instance of Rolling is returned.

    See Also
    --------
    expanding : Provides expanding transformations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    See :ref:`Windowing Operations <window.generic>` for further usage details
    and examples.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    **window**

    Rolling sum with a window length of 2 observations.

    >>> df.rolling(2).sum()
         B
    0  NaN
    1  1.0
    2  3.0
    3  NaN
    4  NaN

    Rolling sum with a window span of 2 seconds.

    >>> df_time = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]},
    ...                        index=[pd.Timestamp('20130101 09:00:00'),
    ...                               pd.Timestamp('20130101 09:00:02'),
    ...                               pd.Timestamp('20130101 09:00:03'),
    ...                               pd.Timestamp('20130101 09:00:05'),
    ...                               pd.Timestamp('20130101 09:00:06')])

    >>> df_time
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  2.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    >>> df_time.rolling('2s').sum()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  3.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    Rolling sum with forward looking windows with 2 observations.

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0

    **min_periods**

    Rolling sum with a window length of 2 observations, but only needs a minimum of 1
    observation to calculate a value.

    >>> df.rolling(2, min_periods=1).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  2.0
    4  4.0

    **center**

    Rolling sum with the result assigned to the center of the window index.

    >>> df.rolling(3, min_periods=1, center=True).sum()
         B
    0  1.0
    1  3.0
    2  3.0
    3  6.0
    4  4.0

    >>> df.rolling(3, min_periods=1, center=False).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  3.0
    4  6.0

    **step**

    Rolling sum with a window length of 2 observations, minimum of 1 observation to
    calculate a value, and a step of 2.

    >>> df.rolling(2, min_periods=1, step=2).sum()
         B
    0  0.0
    2  3.0
    4  4.0

    **win_type**

    Rolling sum with a window length of 2, using the Scipy ``'gaussian'``
    window type. ``std`` is required in the aggregation function.

    >>> df.rolling(2, win_type='gaussian').sum(std=3)
              B
    0       NaN
    1  0.986207
    2  2.958621
    3       NaN
    4       NaN

    **on**

    Rolling sum with a window length of 2 days.

    >>> df = pd.DataFrame({
    ...     'A': [pd.to_datetime('2020-01-01'),
    ...           pd.to_datetime('2020-01-01'),
    ...           pd.to_datetime('2020-01-02'),],
    ...     'B': [1, 2, 3], },
    ...     index=pd.date_range('2020', periods=3))

    >>> df
                        A  B
    2020-01-01 2020-01-01  1
    2020-01-02 2020-01-01  2
    2020-01-03 2020-01-02  3

    >>> df.rolling('2D', on='A').sum()
                        A    B
    2020-01-01 2020-01-01  1.0
    2020-01-02 2020-01-01  3.0
    2020-01-03 2020-01-02  6.0
    """

    _attributes = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "axis",
        "on",
        "closed",
        "step",
        "method",
    ]

    def _validate(self):
        super()._validate()

        if not isinstance(self.win_type, str):
            raise ValueError(f"Invalid win_type {self.win_type}")
        signal = import_optional_dependency(
            "scipy.signal.windows", extra="Scipy is required to generate window weight."
        )
        self._scipy_weight_generator = getattr(signal, self.win_type, None)
        if self._scipy_weight_generator is None:
            raise ValueError(f"Invalid win_type {self.win_type}")

        if isinstance(self.window, BaseIndexer):
            raise NotImplementedError(
                "BaseIndexer subclasses not implemented with win_types."
            )
        if not is_integer(self.window) or self.window < 0:
            raise ValueError("window must be an integer 0 or greater")

        if self.method != "single":
            raise NotImplementedError("'single' is the only supported method type.")

    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        """
        Center the result in the window for weighted rolling aggregations.
        """
        if offset > 0:
            lead_indexer = [slice(offset, None)]
            result = np.copy(result[tuple(lead_indexer)])
        return result

    def _apply(
        self,
        func: Callable[[np.ndarray, int, int], np.ndarray],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
    ):
        """
        Rolling with weights statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
        name : str,
        numeric_only : bool, default False
            Whether to only operate on bool, int, and float columns
        numba_args : tuple
            unused
        **kwargs
            additional arguments for scipy windows if necessary

        Returns
        -------
        y : type of input
        """
        # "None" not callable  [misc]
        window = self._scipy_weight_generator(  # type: ignore[misc]
            self.window, **kwargs
        )
        offset = (len(window) - 1) // 2 if self.center else 0

        def homogeneous_func(values: np.ndarray):
            # calculation function

            if values.size == 0:
                return values.copy()

            def calc(x):
                additional_nans = np.array([np.nan] * offset)
                x = np.concatenate((x, additional_nans))
                return func(
                    x,
                    window,
                    self.min_periods if self.min_periods is not None else len(window),
                )

            with np.errstate(all="ignore"):
                # Our weighted aggregations return memoryviews
                result = np.asarray(calc(values))

            if self.center:
                result = self._center_window(result, offset)

            return result

        return self._apply_columnwise(homogeneous_func, name, numeric_only)[
            :: self.step
        ]

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.DataFrame.aggregate : Similar DataFrame method.
        pandas.Series.aggregate : Similar Series method.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2, win_type="boxcar").agg("mean")
             A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
        """
        ),
        klass="Series/DataFrame",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            # these must apply directly
            result = func(self)

        return result

    agg = aggregate

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method
        (`sum` in this case):

        >>> ser.rolling(2, win_type='gaussian').sum(std=3)
        0         NaN
        1    0.986207
        2    5.917243
        3    6.903450
        4    9.862071
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window sum",
        agg_method="sum",
    )
    def sum(self, numeric_only: bool = False, **kwargs):
        window_func = window_aggregations.roll_weighted_sum
        # error: Argument 1 to "_apply" of "Window" has incompatible type
        # "Callable[[ndarray, ndarray, int], ndarray]"; expected
        # "Callable[[ndarray, int, int], ndarray]"
        return self._apply(
            window_func,  # type: ignore[arg-type]
            name="sum",
            numeric_only=numeric_only,
            **kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').mean(std=3)
        0    NaN
        1    0.5
        2    3.0
        3    3.5
        4    5.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window mean",
        agg_method="mean",
    )
    def mean(self, numeric_only: bool = False, **kwargs):
        window_func = window_aggregations.roll_weighted_mean
        # error: Argument 1 to "_apply" of "Window" has incompatible type
        # "Callable[[ndarray, ndarray, int], ndarray]"; expected
        # "Callable[[ndarray, int, int], ndarray]"
        return self._apply(
            window_func,  # type: ignore[arg-type]
            name="mean",
            numeric_only=numeric_only,
            **kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').var(std=3)
        0     NaN
        1     0.5
        2     8.0
        3     4.5
        4    18.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window variance",
        agg_method="var",
    )
    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs):
        window_func = partial(window_aggregations.roll_weighted_var, ddof=ddof)
        kwargs.pop("name", None)
        return self._apply(window_func, name="var", numeric_only=numeric_only, **kwargs)

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').std(std=3)
        0         NaN
        1    0.707107
        2    2.828427
        3    2.121320
        4    4.242641
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window standard deviation",
        agg_method="std",
    )
    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs):
        return zsqrt(
            self.var(ddof=ddof, name="std", numeric_only=numeric_only, **kwargs)
        )


class RollingAndExpandingMixin(BaseWindow):
    def count(self, numeric_only: bool = False):
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name="count", numeric_only=numeric_only)

    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        if not is_bool(raw):
            raise ValueError("raw parameter must be `True` or `False`")

        numba_args: tuple[Any, ...] = ()
        if maybe_use_numba(engine):
            if raw is False:
                raise ValueError("raw must be `True` when using the numba engine")
            numba_args = args
            if self.method == "single":
                apply_func = generate_numba_apply_func(
                    func, **get_jit_arguments(engine_kwargs, kwargs)
                )
            else:
                apply_func = generate_numba_table_func(
                    func, **get_jit_arguments(engine_kwargs, kwargs)
                )
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")
            apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

        return self._apply(
            apply_func,
            name="apply",
            numba_args=numba_args,
        )

    def _generate_cython_apply_func(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        raw: bool | np.bool_,
        function: Callable[..., Any],
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
        from pandas import Series

        window_func = partial(
            window_aggregations.roll_apply,
            args=args,
            kwargs=kwargs,
            raw=raw,
            function=function,
        )

        def apply_func(values, begin, end, min_periods, raw=raw):
            if not raw:
                # GH 45912
                values = Series(values, index=self._on, copy=False)
            return window_func(values, begin, end, min_periods)

        return apply_func

    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nansum)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_sum

                return self._numba_apply(sliding_sum, engine_kwargs)
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name="sum", numeric_only=numeric_only)

    def max(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmax)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_min_max

                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=True)
        window_func = window_aggregations.roll_max
        return self._apply(window_func, name="max", numeric_only=numeric_only)

    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmin)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_min_max

                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=False)
        window_func = window_aggregations.roll_min
        return self._apply(window_func, name="min", numeric_only=numeric_only)

    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_mean

                return self._numba_apply(sliding_mean, engine_kwargs)
        window_func = window_aggregations.roll_mean
        return self._apply(window_func, name="mean", numeric_only=numeric_only)

    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmedian)
            else:
                func = np.nanmedian

            return self.apply(
                func,
                raw=True,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )
        window_func = window_aggregations.roll_median_c
        return self._apply(window_func, name="median", numeric_only=numeric_only)

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                raise NotImplementedError("std not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var

            return zsqrt(self._numba_apply(sliding_var, engine_kwargs, ddof=ddof))
        window_func = window_aggregations.roll_var

        def zsqrt_func(values, begin, end, min_periods):
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))

        return self._apply(
            zsqrt_func,
            name="std",
            numeric_only=numeric_only,
        )

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            if self.method == "table":
                raise NotImplementedError("var not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var

            return self._numba_apply(sliding_var, engine_kwargs, ddof=ddof)
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        return self._apply(
            window_func,
            name="var",
            numeric_only=numeric_only,
        )

    def skew(self, numeric_only: bool = False):
        window_func = window_aggregations.roll_skew
        return self._apply(
            window_func,
            name="skew",
            numeric_only=numeric_only,
        )

    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # Raise here so error message says sem instead of std
        self._validate_numeric_only("sem", numeric_only)
        return self.std(numeric_only=numeric_only) / (
            self.count(numeric_only=numeric_only) - ddof
        ).pow(0.5)

    def kurt(self, numeric_only: bool = False):
        window_func = window_aggregations.roll_kurt
        return self._apply(
            window_func,
            name="kurt",
            numeric_only=numeric_only,
        )

    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = "linear",
        numeric_only: bool = False,
    ):
        if q == 1.0:
            window_func = window_aggregations.roll_max
        elif q == 0.0:
            window_func = window_aggregations.roll_min
        else:
            window_func = partial(
                window_aggregations.roll_quantile,
                quantile=q,
                interpolation=interpolation,
            )

        return self._apply(window_func, name="quantile", numeric_only=numeric_only)

    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        window_func = partial(
            window_aggregations.roll_rank,
            method=method,
            ascending=ascending,
            percentile=pct,
        )

        return self._apply(window_func, name="rank", numeric_only=numeric_only)

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        if self.step is not None:
            raise NotImplementedError("step not implemented for cov")
        self._validate_numeric_only("cov", numeric_only)

        from pandas import Series

        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            self._check_window_bounds(start, end, len(x_array))

            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(
                    x_array * y_array, start, end, min_periods
                )
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(
                    notna(x_array + y_array).astype(np.float64), start, end, 0
                )
                result = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        if self.step is not None:
            raise NotImplementedError("step not implemented for corr")
        self._validate_numeric_only("corr", numeric_only)

        from pandas import Series

        def corr_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            self._check_window_bounds(start, end, len(x_array))

            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(
                    x_array * y_array, start, end, min_periods
                )
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(
                    notna(x_array + y_array).astype(np.float64), start, end, 0
                )
                x_var = window_aggregations.roll_var(
                    x_array, start, end, min_periods, ddof
                )
                y_var = window_aggregations.roll_var(
                    y_array, start, end, min_periods, ddof
                )
                numerator = (mean_x_y - mean_x * mean_y) * (
                    count_x_y / (count_x_y - ddof)
                )
                denominator = (x_var * y_var) ** 0.5
                result = numerator / denominator
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, corr_func, numeric_only
        )


class Rolling(RollingAndExpandingMixin):
    _attributes: list[str] = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "axis",
        "on",
        "closed",
        "step",
        "method",
    ]

    def _validate(self):
        super()._validate()

        # we allow rolling on a datetimelike index
        if (
            self.obj.empty
            or isinstance(self._on, (DatetimeIndex, TimedeltaIndex, PeriodIndex))
            or (isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM")
        ) and isinstance(self.window, (str, BaseOffset, timedelta)):
            self._validate_datetimelike_monotonic()

            # this will raise ValueError on non-fixed freqs
            try:
                freq = to_offset(self.window)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"passed window {self.window} is not "
                    "compatible with a datetimelike index"
                ) from err
            if isinstance(self._on, PeriodIndex):
                # error: Incompatible types in assignment (expression has type
                # "float", variable has type "Optional[int]")
                self._win_freq_i8 = freq.nanos / (  # type: ignore[assignment]
                    self._on.freq.nanos / self._on.freq.n
                )
            else:
                try:
                    unit = dtype_to_unit(self._on.dtype)  # type: ignore[arg-type]
                except TypeError:
                    # if not a datetime dtype, eg for empty dataframes
                    unit = "ns"
                self._win_freq_i8 = Timedelta(freq.nanos).as_unit(unit)._value

            # min_periods must be an integer
            if self.min_periods is None:
                self.min_periods = 1

            if self.step is not None:
                raise NotImplementedError(
                    "step is not supported with frequency windows"
                )

        elif isinstance(self.window, BaseIndexer):
            # Passed BaseIndexer subclass should handle all other rolling kwargs
            pass
        elif not is_integer(self.window) or self.window < 0:
            raise ValueError("window must be an integer 0 or greater")

    def _validate_datetimelike_monotonic(self) -> None:
        """
        Validate self._on is monotonic (increasing or decreasing) and has
        no NaT values for frequency windows.
        """
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        if not (self._on.is_monotonic_increasing or self._on.is_monotonic_decreasing):
            self._raise_monotonic_error("values must be monotonic")

    def _raise_monotonic_error(self, msg: str):
        on = self.on
        if on is None:
            if self.axis == 0:
                on = "index"
            else:
                on = "column"
        raise ValueError(f"{on} {msg}")

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.Series.rolling : Calling object with Series data.
        pandas.DataFrame.rolling : Calling object with DataFrame data.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0

        >>> df.rolling(2).agg({"A": "sum", "B": "min"})
             A    B
        0  NaN  NaN
        1  3.0  4.0
        2  5.0  5.0
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([2, 3, np.nan, 10])
        >>> s.rolling(2).count()
        0    NaN
        1    2.0
        2    1.0
        3    1.0
        dtype: float64
        >>> s.rolling(3).count()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        dtype: float64
        >>> s.rolling(4).count()
        0    NaN
        1    NaN
        2    NaN
        3    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="count of non NaN observations",
        agg_method="count",
    )
    def count(self, numeric_only: bool = False):
        return super().count(numeric_only)

    @doc(
        template_header,
        create_section_header("Parameters"),
        window_apply_parameters,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 6, 5, 4])
        >>> ser.rolling(2).apply(lambda s: s.sum() - s.min())
        0    NaN
        1    6.0
        2    6.0
        3    5.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="custom aggregation function",
        agg_method="apply",
    )
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        return super().apply(
            func,
            raw=raw,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> s
        0    1
        1    2
        2    3
        3    4
        4    5
        dtype: int64

        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3     9.0
        4    12.0
        dtype: float64

        >>> s.rolling(3, center=True).sum()
        0     NaN
        1     6.0
        2     9.0
        3    12.0
        4     NaN
        dtype: float64

        For DataFrame, each sum is computed column-wise.

        >>> df = pd.DataFrame({{"A": s, "B": s ** 2}})
        >>> df
           A   B
        0  1   1
        1  2   4
        2  3   9
        3  4  16
        4  5  25

        >>> df.rolling(3).sum()
              A     B
        0   NaN   NaN
        1   NaN   NaN
        2   6.0  14.0
        3   9.0  29.0
        4  12.0  50.0
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="sum",
        agg_method="sum",
    )
    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().sum(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.rolling(2).max()
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="maximum",
        agg_method="max",
    )
    def max(
        self,
        numeric_only: bool = False,
        *args,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs,
    ):
        return super().max(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        Performing a rolling minimum with a window size of 3.

        >>> s = pd.Series([4, 3, 5, 2, 6])
        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="minimum",
        agg_method="min",
    )
    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().min(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        The below examples will show rolling mean calculations with window sizes of
        two and three, respectively.

        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).mean()
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64

        >>> s.rolling(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="mean",
        agg_method="mean",
    )
    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().mean(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        Compute the rolling median of a series with a window size of 3.

        >>> s = pd.Series([0, 1, 2, 3, 4])
        >>> s.rolling(3).median()
        0    NaN
        1    NaN
        2    1.0
        3    2.0
        4    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="median",
        agg_method="median",
    )
    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().median(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        window_agg_numba_parameters("1.4"),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "numpy.std : Equivalent method for NumPy array.\n",
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.

        A minimum of one period is required for the rolling calculation.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    1.000000
        4    1.000000
        5    1.154701
        6    0.000000
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="standard deviation",
        agg_method="std",
    )
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().std(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        window_agg_numba_parameters("1.4"),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "numpy.var : Equivalent method for NumPy array.\n",
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.

        A minimum of one period is required for the rolling calculation.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    1.000000
        4    1.000000
        5    1.333333
        6    0.000000
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="variance",
        agg_method="var",
    )
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().var(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "scipy.stats.skew : Third moment of a probability density.\n",
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        A minimum of three periods is required for the rolling calculation.\n
        """
        ),
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 5, 2, 7, 15, 6])
        >>> ser.rolling(3).skew().round(6)
        0         NaN
        1         NaN
        2    1.293343
        3   -0.585583
        4    0.670284
        5    1.652317
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="unbiased skewness",
        agg_method="skew",
    )
    def skew(self, numeric_only: bool = False):
        return super().skew(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        "A minimum of one period is required for the calculation.\n\n",
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([0, 1, 2, 3])
        >>> s.rolling(2, min_periods=1).sem()
        0         NaN
        1    0.707107
        2    0.707107
        3    0.707107
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="standard error of mean",
        agg_method="sem",
    )
    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # Raise here so error message says sem instead of std
        self._validate_numeric_only("sem", numeric_only)
        return self.std(numeric_only=numeric_only) / (
            self.count(numeric_only) - ddof
        ).pow(0.5)

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "scipy.stats.kurtosis : Reference SciPy method.\n",
        template_see_also,
        create_section_header("Notes"),
        "A minimum of four periods is required for the calculation.\n\n",
        create_section_header("Examples"),
        dedent(
            """
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.

        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}}")
        -1.200000
        >>> print(f"{{scipy.stats.kurtosis(arr[1:], bias=False):.6f}}")
        3.999946
        >>> s = pd.Series(arr)
        >>> s.rolling(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    3.999946
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="Fisher's definition of kurtosis without bias",
        agg_method="kurt",
    )
    def kurt(self, numeric_only: bool = False):
        return super().kurt(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        quantile : float
            Quantile to compute. 0 <= quantile <= 1.

            .. deprecated:: 2.1.0
                This will be renamed to 'q' in a future version.
        interpolation : {{'linear', 'lower', 'higher', 'midpoint', 'nearest'}}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).quantile(.4, interpolation='lower')
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

        >>> s.rolling(2).quantile(.4, interpolation='midpoint')
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="quantile",
        agg_method="quantile",
    )
    @deprecate_kwarg(old_arg_name="quantile", new_arg_name="q")
    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = "linear",
        numeric_only: bool = False,
    ):
        return super().quantile(
            q=q,
            interpolation=interpolation,
            numeric_only=numeric_only,
        )

    @doc(
        template_header,
        ".. versionadded:: 1.4.0 \n\n",
        create_section_header("Parameters"),
        dedent(
            """
        method : {{'average', 'min', 'max'}}, default 'average'
            How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group

        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.rolling(3).rank()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.5
        dtype: float64

        >>> s.rolling(3).rank(method="max")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    2.0
        dtype: float64

        >>> s.rolling(3).rank(method="min")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="rank",
        agg_method="rank",
    )
    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        return super().rank(
            method=method,
            ascending=ascending,
            pct=pct,
            numeric_only=numeric_only,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([1, 4, 5, 8])
        >>> ser1.rolling(2).cov(ser2)
        0    NaN
        1    1.5
        2    0.5
        3    1.5
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="sample covariance",
        agg_method="cov",
    )
    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        return super().cov(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        dedent(
            """
        cov : Similar method to calculate covariance.
        numpy.corrcoef : NumPy Pearson's correlation calculation.
        """
        ).replace("\n", "", 1),
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        This function uses Pearson's definition of correlation
        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

        When `other` is not specified, the output will be self correlation (e.g.
        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`
        set to `True`.

        Function will return ``NaN`` for correlations of equal valued sequences;
        this is the result of a 0/0 division error.

        When `pairwise` is set to `False`, only matching columns between `self` and
        `other` will be used.

        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame
        with the original index on the first level, and the `other` DataFrame
        columns on the second level.

        In the case of missing elements, only complete pairwise observations
        will be used.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        dedent(
            """
        The below example shows a rolling calculation with a window size of
        four matching the equivalent function call using :meth:`numpy.corrcoef`.

        >>> v1 = [3, 3, 3, 5, 8]
        >>> v2 = [3, 4, 4, 4, 8]
        >>> np.corrcoef(v1[:-1], v2[:-1])
        array([[1.        , 0.33333333],
               [0.33333333, 1.        ]])
        >>> np.corrcoef(v1[1:], v2[1:])
        array([[1.       , 0.9169493],
               [0.9169493, 1.       ]])
        >>> s1 = pd.Series(v1)
        >>> s2 = pd.Series(v2)
        >>> s1.rolling(4).corr(s2)
        0         NaN
        1         NaN
        2         NaN
        3    0.333333
        4    0.916949
        dtype: float64

        The below example shows a similar rolling calculation on a
        DataFrame using the pairwise option.

        >>> matrix = np.array([[51., 35.],
        ...                    [49., 30.],
        ...                    [47., 32.],
        ...                    [46., 31.],
        ...                    [50., 36.]])
        >>> np.corrcoef(matrix[:-1, 0], matrix[:-1, 1])
        array([[1.       , 0.6263001],
               [0.6263001, 1.       ]])
        >>> np.corrcoef(matrix[1:, 0], matrix[1:, 1])
        array([[1.        , 0.55536811],
               [0.55536811, 1.        ]])
        >>> df = pd.DataFrame(matrix, columns=['X', 'Y'])
        >>> df
              X     Y
        0  51.0  35.0
        1  49.0  30.0
        2  47.0  32.0
        3  46.0  31.0
        4  50.0  36.0
        >>> df.rolling(4).corr(pairwise=True)
                    X         Y
        0 X       NaN       NaN
          Y       NaN       NaN
        1 X       NaN       NaN
          Y       NaN       NaN
        2 X       NaN       NaN
          Y       NaN       NaN
        3 X  1.000000  0.626300
          Y  0.626300  1.000000
        4 X  1.000000  0.555368
          Y  0.555368  1.000000
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="correlation",
        agg_method="corr",
    )
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        return super().corr(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )


Rolling.__doc__ = Window.__doc__


class RollingGroupby(BaseWindowGroupby, Rolling):
    """
    Provide a rolling groupby implementation.
    """

    _attributes = Rolling._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        rolling_indexer: type[BaseIndexer]
        indexer_kwargs: dict[str, Any] | None = None
        index_array = self._index_array
        if isinstance(self.window, BaseIndexer):
            rolling_indexer = type(self.window)
            indexer_kwargs = self.window.__dict__.copy()
            assert isinstance(indexer_kwargs, dict)  # for mypy
            # We'll be using the index of each group later
            indexer_kwargs.pop("index_array", None)
            window = self.window
        elif self._win_freq_i8 is not None:
            rolling_indexer = VariableWindowIndexer
            # error: Incompatible types in assignment (expression has type
            # "int", variable has type "BaseIndexer")
            window = self._win_freq_i8  # type: ignore[assignment]
        else:
            rolling_indexer = FixedWindowIndexer
            window = self.window
        window_indexer = GroupbyIndexer(
            index_array=index_array,
            window_size=window,
            groupby_indices=self._grouper.indices,
            window_indexer=rolling_indexer,
            indexer_kwargs=indexer_kwargs,
        )
        return window_indexer

    def _validate_datetimelike_monotonic(self):
        """
        Validate that each group in self._on is monotonic
        """
        # GH 46061
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        for group_indices in self._grouper.indices.values():
            group_on = self._on.take(group_indices)
            if not (
                group_on.is_monotonic_increasing or group_on.is_monotonic_decreasing
            ):
                on = "index" if self.on is None else self.on
                raise ValueError(
                    f"Each group within {on} must be monotonic. "
                    f"Sort the values in {on} first."
                )
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency


def generate_online_numba_ewma_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    """
    Generate a numba jitted groupby ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
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
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def online_ewma(
        values: np.ndarray,
        deltas: np.ndarray,
        minimum_periods: int,
        old_wt_factor: float,
        new_wt: float,
        old_wt: np.ndarray,
        adjust: bool,
        ignore_na: bool,
    ):
        """
        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        """
        result = np.empty(values.shape)
        weighted_avg = values[0].copy()
        nobs = (~np.isnan(weighted_avg)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)

        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted_avg[j]):
                    if is_observations[j] or not ignore_na:
                        # note that len(deltas) = len(vals) - 1 and deltas[i] is to be
                        # used in conjunction with vals[i+1]
                        old_wt[j] *= old_wt_factor ** deltas[j - 1]
                        if is_observations[j]:
                            # avoid numerical errors on constant series
                            if weighted_avg[j] != cur[j]:
                                weighted_avg[j] = (
                                    (old_wt[j] * weighted_avg[j]) + (new_wt * cur[j])
                                ) / (old_wt[j] + new_wt)
                            if adjust:
                                old_wt[j] += new_wt
                            else:
                                old_wt[j] = 1.0
                elif is_observations[j]:
                    weighted_avg[j] = cur[j]

            result[i] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)

        return result, old_wt

    return online_ewma


class EWMMeanState:
    def __init__(self, com, adjust, ignore_na, axis, shape) -> None:
        alpha = 1.0 / (1.0 + com)
        self.axis = axis
        self.shape = shape
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.new_wt = 1.0 if adjust else alpha
        self.old_wt_factor = 1.0 - alpha
        self.old_wt = np.ones(self.shape[self.axis - 1])
        self.last_ewm = None

    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func):
        result, old_wt = ewm_func(
            weighted_avg,
            deltas,
            min_periods,
            self.old_wt_factor,
            self.new_wt,
            self.old_wt,
            self.adjust,
            self.ignore_na,
        )
        self.old_wt = old_wt
        self.last_ewm = result[-1]
        return result

    def reset(self) -> None:
        self.old_wt = np.ones(self.shape[self.axis - 1])
        self.last_ewm = None
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.common import is_list_like

if TYPE_CHECKING:
    from pandas._typing import NumpyIndexT


def cartesian_product(X) -> list[np.ndarray]:
    """
    Numpy version of itertools.product.
    Sometimes faster (for large inputs)...

    Parameters
    ----------
    X : list-like of list-likes

    Returns
    -------
    product : list of ndarrays

    Examples
    --------
    >>> cartesian_product([list('ABC'), [1, 2]])
    [array(['A', 'A', 'B', 'B', 'C', 'C'], dtype='<U1'), array([1, 2, 1, 2, 1, 2])]

    See Also
    --------
    itertools.product : Cartesian product of input iterables.  Equivalent to
        nested for-loops.
    """
    msg = "Input must be a list-like of list-likes"
    if not is_list_like(X):
        raise TypeError(msg)
    for x in X:
        if not is_list_like(x):
            raise TypeError(msg)

    if len(X) == 0:
        return []

    lenX = np.fromiter((len(x) for x in X), dtype=np.intp)
    cumprodX = np.cumprod(lenX)

    if np.any(cumprodX < 0):
        raise ValueError("Product space too large to allocate arrays!")

    a = np.roll(cumprodX, 1)
    a[0] = 1

    if cumprodX[-1] != 0:
        b = cumprodX[-1] / cumprodX
    else:
        # if any factor is empty, the cartesian product is empty
        b = np.zeros_like(cumprodX)

    # error: Argument of type "int_" cannot be assigned to parameter "num" of
    # type "int" in function "tile_compat"
    return [
        tile_compat(
            np.repeat(x, b[i]),
            np.prod(a[i]),
        )
        for i, x in enumerate(X)
    ]


def tile_compat(arr: NumpyIndexT, num: int) -> NumpyIndexT:
    """
    Index compat for np.tile.

    Notes
    -----
    Does not support multi-dimensional `num`.
    """
    if isinstance(arr, np.ndarray):
        return np.tile(arr, num)

    # Otherwise we have an Index
    taker = np.tile(np.arange(len(arr)), num)
    return arr.take(taker)
<|endoftext|>from __future__ import annotations

from collections import defaultdict
from collections.abc import (
    Hashable,
    Iterable,
)
import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import missing as libmissing
from pandas._libs.sparse import IntIndex

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)

from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.arrays.string_ import StringDtype
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    default_index,
)
from pandas.core.series import Series

if TYPE_CHECKING:
    from pandas._typing import NpDtype


def get_dummies(
    data,
    prefix=None,
    prefix_sep: str | Iterable[str] | dict[str, str] = "_",
    dummy_na: bool = False,
    columns=None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: NpDtype | None = None,
) -> DataFrame:
    """
    Convert categorical variable into dummy/indicator variables.

    Each variable is converted in as many 0/1 variables as there are different
    values. Columns in the output are each named after a value; if the input is
    a DataFrame, the name of the original variable is prepended to the value.

    Parameters
    ----------
    data : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, list of str, or dict of str, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : str, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix`.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object`, `string`, or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    dtype : dtype, default bool
        Data type for new columns. Only a single dtype is allowed.

    Returns
    -------
    DataFrame
        Dummy-coded data. If `data` contains other columns than the
        dummy-coded one(s), these will be prepended, unaltered, to the result.

    See Also
    --------
    Series.str.get_dummies : Convert Series of strings to dummy codes.
    :func:`~pandas.from_dummies` : Convert dummy codes to categorical ``DataFrame``.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.dummies>` for more examples.

    Examples
    --------
    >>> s = pd.Series(list('abca'))

    >>> pd.get_dummies(s)
           a      b      c
    0   True  False  False
    1  False   True  False
    2  False  False   True
    3   True  False  False

    >>> s1 = ['a', 'b', np.nan]

    >>> pd.get_dummies(s1)
           a      b
    0   True  False
    1  False   True
    2  False  False

    >>> pd.get_dummies(s1, dummy_na=True)
           a      b    NaN
    0   True  False  False
    1  False   True  False
    2  False  False   True

    >>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    ...                    'C': [1, 2, 3]})

    >>> pd.get_dummies(df, prefix=['col1', 'col2'])
       C  col1_a  col1_b  col2_a  col2_b  col2_c
    0  1    True   False   False    True   False
    1  2   False    True    True   False   False
    2  3    True   False   False   False    True

    >>> pd.get_dummies(pd.Series(list('abcaa')))
           a      b      c
    0   True  False  False
    1  False   True  False
    2  False  False   True
    3   True  False  False
    4   True  False  False

    >>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
           b      c
    0  False  False
    1   True  False
    2  False   True
    3  False  False
    4  False  False

    >>> pd.get_dummies(pd.Series(list('abc')), dtype=float)
         a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    from pandas.core.reshape.concat import concat

    dtypes_to_encode = ["object", "string", "category"]

    if isinstance(data, DataFrame):
        # determine columns being encoded
        if columns is None:
            data_to_encode = data.select_dtypes(include=dtypes_to_encode)
        elif not is_list_like(columns):
            raise TypeError("Input must be a list-like for parameter `columns`")
        else:
            data_to_encode = data[columns]

        # validate prefixes and separator to avoid silently dropping cols
        def check_len(item, name: str):
            if is_list_like(item):
                if not len(item) == data_to_encode.shape[1]:
                    len_msg = (
                        f"Length of '{name}' ({len(item)}) did not match the "
                        "length of the columns being encoded "
                        f"({data_to_encode.shape[1]})."
                    )
                    raise ValueError(len_msg)

        check_len(prefix, "prefix")
        check_len(prefix_sep, "prefix_sep")

        if isinstance(prefix, str):
            prefix = itertools.cycle([prefix])
        if isinstance(prefix, dict):
            prefix = [prefix[col] for col in data_to_encode.columns]

        if prefix is None:
            prefix = data_to_encode.columns

        # validate separators
        if isinstance(prefix_sep, str):
            prefix_sep = itertools.cycle([prefix_sep])
        elif isinstance(prefix_sep, dict):
            prefix_sep = [prefix_sep[col] for col in data_to_encode.columns]

        with_dummies: list[DataFrame]
        if data_to_encode.shape == data.shape:
            # Encoding the entire df, do not prepend any dropped columns
            with_dummies = []
        elif columns is not None:
            # Encoding only cols specified in columns. Get all cols not in
            # columns to prepend to result.
            with_dummies = [data.drop(columns, axis=1)]
        else:
            # Encoding only object and category dtype columns. Get remaining
            # columns to prepend to result.
            with_dummies = [data.select_dtypes(exclude=dtypes_to_encode)]

        for col, pre, sep in zip(data_to_encode.items(), prefix, prefix_sep):
            # col is (column_name, column), use just column data here
            dummy = _get_dummies_1d(
                col[1],
                prefix=pre,
                prefix_sep=sep,
                dummy_na=dummy_na,
                sparse=sparse,
                drop_first=drop_first,
                dtype=dtype,
            )
            with_dummies.append(dummy)
        result = concat(with_dummies, axis=1)
    else:
        result = _get_dummies_1d(
            data,
            prefix,
            prefix_sep,
            dummy_na,
            sparse=sparse,
            drop_first=drop_first,
            dtype=dtype,
        )
    return result


def _get_dummies_1d(
    data,
    prefix,
    prefix_sep: str | Iterable[str] | dict[str, str] = "_",
    dummy_na: bool = False,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: NpDtype | None = None,
) -> DataFrame:
    from pandas.core.reshape.concat import concat

    # Series avoids inconsistent NaN handling
    codes, levels = factorize_from_iterable(Series(data, copy=False))

    if dtype is None and hasattr(data, "dtype"):
        input_dtype = data.dtype
        if isinstance(input_dtype, CategoricalDtype):
            input_dtype = input_dtype.categories.dtype

        if isinstance(input_dtype, ArrowDtype):
            import pyarrow as pa

            dtype = ArrowDtype(pa.bool_())  # type: ignore[assignment]
        elif (
            isinstance(input_dtype, StringDtype)
            and input_dtype.na_value is libmissing.NA
        ):
            dtype = pandas_dtype("boolean")  # type: ignore[assignment]
        else:
            dtype = np.dtype(bool)
    elif dtype is None:
        dtype = np.dtype(bool)

    _dtype = pandas_dtype(dtype)

    if is_object_dtype(_dtype):
        raise ValueError("dtype=object is not a valid dtype for get_dummies")

    def get_empty_frame(data) -> DataFrame:
        index: Index | np.ndarray
        if isinstance(data, Series):
            index = data.index
        else:
            index = default_index(len(data))
        return DataFrame(index=index)

    # if all NaN
    if not dummy_na and len(levels) == 0:
        return get_empty_frame(data)

    codes = codes.copy()
    if dummy_na:
        codes[codes == -1] = len(levels)
        levels = levels.insert(len(levels), np.nan)

    # if dummy_na, we just fake a nan level. drop_first will drop it again
    if drop_first and len(levels) == 1:
        return get_empty_frame(data)

    number_of_cols = len(levels)

    if prefix is None:
        dummy_cols = levels
    else:
        dummy_cols = Index([f"{prefix}{prefix_sep}{level}" for level in levels])

    index: Index | None
    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if sparse:
        fill_value: bool | float
        if is_integer_dtype(dtype):
            fill_value = 0
        elif dtype == np.dtype(bool):
            fill_value = False
        else:
            fill_value = 0.0

        sparse_series = []
        N = len(data)
        sp_indices: list[list] = [[] for _ in range(len(dummy_cols))]
        mask = codes != -1
        codes = codes[mask]
        n_idx = np.arange(N)[mask]

        for ndx, code in zip(n_idx, codes):
            sp_indices[code].append(ndx)

        if drop_first:
            # remove first categorical level to avoid perfect collinearity
            # GH12042
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]
        for col, ixs in zip(dummy_cols, sp_indices):
            sarr = SparseArray(
                np.ones(len(ixs), dtype=dtype),
                sparse_index=IntIndex(N, ixs),
                fill_value=fill_value,
                dtype=dtype,
            )
            sparse_series.append(Series(data=sarr, index=index, name=col, copy=False))

        return concat(sparse_series, axis=1, copy=False)

    else:
        # ensure ndarray layout is column-major
        shape = len(codes), number_of_cols
        dummy_dtype: NpDtype
        if isinstance(_dtype, np.dtype):
            dummy_dtype = _dtype
        else:
            dummy_dtype = np.bool_
        dummy_mat = np.zeros(shape=shape, dtype=dummy_dtype, order="F")
        dummy_mat[np.arange(len(codes)), codes] = 1

        if not dummy_na:
            # reset NaN GH4446
            dummy_mat[codes == -1] = 0

        if drop_first:
            # remove first GH12042
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]
        return DataFrame(dummy_mat, index=index, columns=dummy_cols, dtype=_dtype)


def from_dummies(
    data: DataFrame,
    sep: None | str = None,
    default_category: None | Hashable | dict[str, Hashable] = None,
) -> DataFrame:
    """
    Create a categorical ``DataFrame`` from a ``DataFrame`` of dummy variables.

    Inverts the operation performed by :func:`~pandas.get_dummies`.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    data : DataFrame
        Data which contains dummy-coded variables in form of integer columns of
        1's and 0's.
    sep : str, default None
        Separator used in the column names of the dummy categories they are
        character indicating the separation of the categorical names from the prefixes.
        For example, if your column names are 'prefix_A' and 'prefix_B',
        you can strip the underscore by specifying sep='_'.
    default_category : None, Hashable or dict of Hashables, default None
        The default category is the implied category when a value has none of the
        listed categories specified with a one, i.e. if all dummies in a row are
        zero. Can be a single value for all variables or a dict directly mapping
        the default categories to a prefix of a variable.

    Returns
    -------
    DataFrame
        Categorical data decoded from the dummy input-data.

    Raises
    ------
    ValueError
        * When the input ``DataFrame`` ``data`` contains NA values.
        * When the input ``DataFrame`` ``data`` contains column names with separators
          that do not match the separator specified with ``sep``.
        * When a ``dict`` passed to ``default_category`` does not include an implied
          category for each prefix.
        * When a value in ``data`` has more than one category assigned to it.
        * When ``default_category=None`` and a value in ``data`` has no category
          assigned to it.
    TypeError
        * When the input ``data`` is not of type ``DataFrame``.
        * When the input ``DataFrame`` ``data`` contains non-dummy data.
        * When the passed ``sep`` is of a wrong data type.
        * When the passed ``default_category`` is of a wrong data type.

    See Also
    --------
    :func:`~pandas.get_dummies` : Convert ``Series`` or ``DataFrame`` to dummy codes.
    :class:`~pandas.Categorical` : Represent a categorical variable in classic.

    Notes
    -----
    The columns of the passed dummy data should only include 1's and 0's,
    or boolean values.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0],
    ...                    "c": [0, 0, 1, 0]})

    >>> df
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> pd.from_dummies(df)
    0     a
    1     b
    2     c
    3     a

    >>> df = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0],
    ...                    "col2_a": [0, 1, 0], "col2_b": [1, 0, 0],
    ...                    "col2_c": [0, 0, 1]})

    >>> df
          col1_a  col1_b  col2_a  col2_b  col2_c
    0       1       0       0       1       0
    1       0       1       1       0       0
    2       1       0       0       0       1

    >>> pd.from_dummies(df, sep="_")
        col1    col2
    0    a       b
    1    b       a
    2    a       c

    >>> df = pd.DataFrame({"col1_a": [1, 0, 0], "col1_b": [0, 1, 0],
    ...                    "col2_a": [0, 1, 0], "col2_b": [1, 0, 0],
    ...                    "col2_c": [0, 0, 0]})

    >>> df
          col1_a  col1_b  col2_a  col2_b  col2_c
    0       1       0       0       1       0
    1       0       1       1       0       0
    2       0       0       0       0       0

    >>> pd.from_dummies(df, sep="_", default_category={"col1": "d", "col2": "e"})
        col1    col2
    0    a       b
    1    b       a
    2    d       e
    """
    from pandas.core.reshape.concat import concat

    if not isinstance(data, DataFrame):
        raise TypeError(
            "Expected 'data' to be a 'DataFrame'; "
            f"Received 'data' of type: {type(data).__name__}"
        )

    col_isna_mask = cast(Series, data.isna().any())

    if col_isna_mask.any():
        raise ValueError(
            "Dummy DataFrame contains NA value in column: "
            f"'{col_isna_mask.idxmax()}'"
        )

    # index data with a list of all columns that are dummies
    try:
        data_to_decode = data.astype("boolean", copy=False)
    except TypeError:
        raise TypeError("Passed DataFrame contains non-dummy data")

    # collect prefixes and get lists to slice data for each prefix
    variables_slice = defaultdict(list)
    if sep is None:
        variables_slice[""] = list(data.columns)
    elif isinstance(sep, str):
        for col in data_to_decode.columns:
            prefix = col.split(sep)[0]
            if len(prefix) == len(col):
                raise ValueError(f"Separator not specified for column: {col}")
            variables_slice[prefix].append(col)
    else:
        raise TypeError(
            "Expected 'sep' to be of type 'str' or 'None'; "
            f"Received 'sep' of type: {type(sep).__name__}"
        )

    if default_category is not None:
        if isinstance(default_category, dict):
            if not len(default_category) == len(variables_slice):
                len_msg = (
                    f"Length of 'default_category' ({len(default_category)}) "
                    f"did not match the length of the columns being encoded "
                    f"({len(variables_slice)})"
                )
                raise ValueError(len_msg)
        elif isinstance(default_category, Hashable):
            default_category = dict(
                zip(variables_slice, [default_category] * len(variables_slice))
            )
        else:
            raise TypeError(
                "Expected 'default_category' to be of type "
                "'None', 'Hashable', or 'dict'; "
                "Received 'default_category' of type: "
                f"{type(default_category).__name__}"
            )

    cat_data = {}
    for prefix, prefix_slice in variables_slice.items():
        if sep is None:
            cats = prefix_slice.copy()
        else:
            cats = [col[len(prefix + sep) :] for col in prefix_slice]
        assigned = data_to_decode.loc[:, prefix_slice].sum(axis=1)
        if any(assigned > 1):
            raise ValueError(
                "Dummy DataFrame contains multi-assignment(s); "
                f"First instance in row: {assigned.idxmax()}"
            )
        if any(assigned == 0):
            if isinstance(default_category, dict):
                cats.append(default_category[prefix])
            else:
                raise ValueError(
                    "Dummy DataFrame contains unassigned value(s); "
                    f"First instance in row: {assigned.idxmin()}"
                )
            data_slice = concat(
                (data_to_decode.loc[:, prefix_slice], assigned == 0), axis=1
            )
        else:
            data_slice = data_to_decode.loc[:, prefix_slice]
        cats_array = data._constructor_sliced(cats, dtype=data.columns.dtype)
        # get indices of True entries along axis=1
        true_values = data_slice.idxmax(axis=1)
        indexer = data_slice.columns.get_indexer_for(true_values)
        cat_data[prefix] = cats_array.take(indexer).set_axis(data.index)

    result = DataFrame(cat_data)
    if sep is not None:
        result.columns = result.columns.astype(data.columns.dtype)
    return result
<|endoftext|>"""
Concat routines.
"""
from __future__ import annotations

from collections import abc
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_copy_on_write

from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool,
    is_iterator,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays.categorical import (
    factorize_from_iterable,
    factorize_from_iterables,
)
import pandas.core.common as com
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    all_indexes_same,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    get_unanimous_names,
)
from pandas.core.internals import concatenate_managers

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
    )

    from pandas._typing import (
        Axis,
        AxisInt,
        HashableT,
    )

    from pandas import (
        DataFrame,
        Series,
    )

# ---------------------------------------------------------------------
# Concatenate DataFrame objects


@overload
def concat(
    objs: Iterable[DataFrame] | Mapping[HashableT, DataFrame],
    *,
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels=...,
    names: list[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | None = ...,
) -> DataFrame:
    ...


@overload
def concat(
    objs: Iterable[Series] | Mapping[HashableT, Series],
    *,
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels=...,
    names: list[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | None = ...,
) -> Series:
    ...


@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels=...,
    names: list[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | None = ...,
) -> DataFrame | Series:
    ...


@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Literal[1, "columns"],
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels=...,
    names: list[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | None = ...,
) -> DataFrame:
    ...


@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Axis = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels=...,
    names: list[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | None = ...,
) -> DataFrame | Series:
    ...


def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Axis = 0,
    join: str = "outer",
    ignore_index: bool = False,
    keys: Iterable[Hashable] | None = None,
    levels=None,
    names: list[HashableT] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool | None = None,
) -> DataFrame | Series:
    """
    Concatenate pandas objects along a particular axis.

    Allows optional set logic along the other axes.

    Can also add a layer of hierarchical indexing on the concatenation axis,
    which may be useful if the labels are the same (or overlapping) on
    the passed axis number.

    Parameters
    ----------
    objs : a sequence or mapping of Series or DataFrame objects
        If a mapping is passed, the sorted keys will be used as the `keys`
        argument, unless it is passed, in which case the values will be
        selected (see below). Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised.
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index : bool, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level.
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys.
    names : list, default None
        Names for the levels in the resulting hierarchical index.
    verify_integrity : bool, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation.
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned. One exception to
        this is when the non-concatentation axis is a DatetimeIndex and join='outer'
        and the axis is not already aligned. In that case, the non-concatenation
        axis is always sorted lexicographically.
    copy : bool, default True
        If False, do not copy data unnecessarily.

    Returns
    -------
    object, type of objs
        When concatenating all ``Series`` along the index (axis=0), a
        ``Series`` is returned. When ``objs`` contains at least one
        ``DataFrame``, a ``DataFrame`` is returned. When concatenating along
        the columns (axis=1), a ``DataFrame`` is returned.

    See Also
    --------
    DataFrame.join : Join DataFrames using indexes.
    DataFrame.merge : Merge DataFrames by indexes or columns.

    Notes
    -----
    The keys, levels, and names arguments are all optional.

    A walkthrough of how this method fits in with other tools for combining
    pandas objects can be found `here
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__.

    It is not recommended to build DataFrames by adding single rows in a
    for loop. Build a list of rows and make a DataFrame in a single concat.

    Examples
    --------
    Combine two ``Series``.

    >>> s1 = pd.Series(['a', 'b'])
    >>> s2 = pd.Series(['c', 'd'])
    >>> pd.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: object

    Clear the existing index and reset it in the result
    by setting the ``ignore_index`` option to ``True``.

    >>> pd.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: object

    Add a hierarchical index at the outermost level of
    the data with the ``keys`` option.

    >>> pd.concat([s1, s2], keys=['s1', 's2'])
    s1  0    a
        1    b
    s2  0    c
        1    d
    dtype: object

    Label the index keys you create with the ``names`` option.

    >>> pd.concat([s1, s2], keys=['s1', 's2'],
    ...           names=['Series name', 'Row ID'])
    Series name  Row ID
    s1           0         a
                 1         b
    s2           0         c
                 1         d
    dtype: object

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = pd.DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = pd.DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    >>> pd.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``NaN`` values.

    >>> df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> pd.concat([df1, df3], sort=False)
      letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> pd.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects horizontally along the x axis by
    passing in ``axis=1``.

    >>> df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> pd.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    Prevent the result from including duplicate index values with the
    ``verify_integrity`` option.

    >>> df5 = pd.DataFrame([1], index=['a'])
    >>> df5
       0
    a  1
    >>> df6 = pd.DataFrame([2], index=['a'])
    >>> df6
       0
    a  2
    >>> pd.concat([df5, df6], verify_integrity=True)
    Traceback (most recent call last):
        ...
    ValueError: Indexes have overlapping values: ['a']

    Append a single row to the end of a ``DataFrame`` object.

    >>> df7 = pd.DataFrame({'a': 1, 'b': 2}, index=[0])
    >>> df7
        a   b
    0   1   2
    >>> new_row = pd.Series({'a': 3, 'b': 4})
    >>> new_row
    a    3
    b    4
    dtype: int64
    >>> pd.concat([df7, new_row.to_frame().T], ignore_index=True)
        a   b
    0   1   2
    1   3   4
    """
    if copy is None:
        if using_copy_on_write():
            copy = False
        else:
            copy = True
    elif copy and using_copy_on_write():
        copy = False

    op = _Concatenator(
        objs,
        axis=axis,
        ignore_index=ignore_index,
        join=join,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        copy=copy,
        sort=sort,
    )

    return op.get_result()


class _Concatenator:
    """
    Orchestrates a concatenation operation for BlockManagers
    """

    sort: bool

    def __init__(
        self,
        objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
        axis: Axis = 0,
        join: str = "outer",
        keys: Iterable[Hashable] | None = None,
        levels=None,
        names: list[HashableT] | None = None,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        copy: bool = True,
        sort: bool = False,
    ) -> None:
        if isinstance(objs, (ABCSeries, ABCDataFrame, str)):
            raise TypeError(
                "first argument must be an iterable of pandas "
                f'objects, you passed an object of type "{type(objs).__name__}"'
            )

        if join == "outer":
            self.intersect = False
        elif join == "inner":
            self.intersect = True
        else:  # pragma: no cover
            raise ValueError(
                "Only can inner (intersect) or outer (union) join the other axis"
            )

        if not is_bool(sort):
            raise ValueError(
                f"The 'sort' keyword only accepts boolean values; {sort} was passed."
            )
        # Incompatible types in assignment (expression has type "Union[bool, bool_]",
        # variable has type "bool")
        self.sort = sort  # type: ignore[assignment]

        self.ignore_index = ignore_index
        self.verify_integrity = verify_integrity
        self.copy = copy

        objs, keys = self._clean_keys_and_objs(objs, keys)

        # figure out what our result ndim is going to be
        ndims = self._get_ndims(objs)
        sample, objs = self._get_sample_object(objs, ndims, keys, names, levels)

        # Standardize axis parameter to int
        if sample.ndim == 1:
            from pandas import DataFrame

            axis = DataFrame._get_axis_number(axis)
            self._is_frame = False
            self._is_series = True
        else:
            axis = sample._get_axis_number(axis)
            self._is_frame = True
            self._is_series = False

            # Need to flip BlockManager axis in the DataFrame special case
            axis = sample._get_block_manager_axis(axis)

        # if we have mixed ndims, then convert to highest ndim
        # creating column numbers as needed
        if len(ndims) > 1:
            objs = self._sanitize_mixed_ndim(objs, sample, ignore_index, axis)

        self.objs = objs

        # note: this is the BlockManager axis (since DataFrame is transposed)
        self.bm_axis = axis
        self.axis = 1 - self.bm_axis if self._is_frame else 0
        self.keys = keys
        self.names = names or getattr(keys, "names", None)
        self.levels = levels

    def _get_ndims(self, objs: list[Series | DataFrame]) -> set[int]:
        # figure out what our result ndim is going to be
        ndims = set()
        for obj in objs:
            if not isinstance(obj, (ABCSeries, ABCDataFrame)):
                msg = (
                    f"cannot concatenate object of type '{type(obj)}'; "
                    "only Series and DataFrame objs are valid"
                )
                raise TypeError(msg)

            ndims.add(obj.ndim)
        return ndims

    def _clean_keys_and_objs(
        self,
        objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
        keys,
    ) -> tuple[list[Series | DataFrame], Index | None]:
        if isinstance(objs, abc.Mapping):
            if keys is None:
                keys = list(objs.keys())
            objs_list = [objs[k] for k in keys]
        else:
            objs_list = list(objs)

        if len(objs_list) == 0:
            raise ValueError("No objects to concatenate")

        if keys is None:
            objs_list = list(com.not_none(*objs_list))
        else:
            # GH#1649
            clean_keys = []
            clean_objs = []
            if is_iterator(keys):
                keys = list(keys)
            if len(keys) != len(objs_list):
                # GH#43485
                warnings.warn(
                    "The behavior of pd.concat with len(keys) != len(objs) is "
                    "deprecated. In a future version this will raise instead of "
                    "truncating to the smaller of the two sequences",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            for k, v in zip(keys, objs_list):
                if v is None:
                    continue
                clean_keys.append(k)
                clean_objs.append(v)
            objs_list = clean_objs

            if isinstance(keys, MultiIndex):
                # TODO: retain levels?
                keys = type(keys).from_tuples(clean_keys, names=keys.names)
            else:
                name = getattr(keys, "name", None)
                keys = Index(clean_keys, name=name, dtype=getattr(keys, "dtype", None))

        if len(objs_list) == 0:
            raise ValueError("All objects passed were None")

        return objs_list, keys

    def _get_sample_object(
        self,
        objs: list[Series | DataFrame],
        ndims: set[int],
        keys,
        names,
        levels,
    ) -> tuple[Series | DataFrame, list[Series | DataFrame]]:
        # get the sample
        # want the highest ndim that we have, and must be non-empty
        # unless all objs are empty
        sample: Series | DataFrame | None = None
        if len(ndims) > 1:
            max_ndim = max(ndims)
            for obj in objs:
                if obj.ndim == max_ndim and np.sum(obj.shape):
                    sample = obj
                    break

        else:
            # filter out the empties if we have not multi-index possibilities
            # note to keep empty Series as it affect to result columns / name
            non_empties = [obj for obj in objs if sum(obj.shape) > 0 or obj.ndim == 1]

            if len(non_empties) and (
                keys is None and names is None and levels is None and not self.intersect
            ):
                objs = non_empties
                sample = objs[0]

        if sample is None:
            sample = objs[0]
        return sample, objs

    def _sanitize_mixed_ndim(
        self,
        objs: list[Series | DataFrame],
        sample: Series | DataFrame,
        ignore_index: bool,
        axis: AxisInt,
    ) -> list[Series | DataFrame]:
        # if we have mixed ndims, then convert to highest ndim
        # creating column numbers as needed

        new_objs = []

        current_column = 0
        max_ndim = sample.ndim
        for obj in objs:
            ndim = obj.ndim
            if ndim == max_ndim:
                pass

            elif ndim != max_ndim - 1:
                raise ValueError(
                    "cannot concatenate unaligned mixed dimensional NDFrame objects"
                )

            else:
                name = getattr(obj, "name", None)
                if ignore_index or name is None:
                    if axis == 1:
                        # doing a row-wise concatenation so need everything
                        # to line up
                        name = 0
                    else:
                        # doing a column-wise concatenation so need series
                        # to have unique names
                        name = current_column
                        current_column += 1

                obj = sample._constructor({name: obj}, copy=False)

            new_objs.append(obj)

        return new_objs

    def get_result(self):
        cons: Callable[..., DataFrame | Series]
        sample: DataFrame | Series

        # series only
        if self._is_series:
            sample = cast("Series", self.objs[0])

            # stack blocks
            if self.bm_axis == 0:
                name = com.consensus_name_attr(self.objs)
                cons = sample._constructor

                arrs = [ser._values for ser in self.objs]

                res = concat_compat(arrs, axis=0)

                new_index: Index
                if self.ignore_index:
                    # We can avoid surprisingly-expensive _get_concat_axis
                    new_index = default_index(len(res))
                else:
                    new_index = self.new_axes[0]

                mgr = type(sample._mgr).from_array(res, index=new_index)

                result = sample._constructor_from_mgr(mgr, axes=mgr.axes)
                result._name = name
                return result.__finalize__(self, method="concat")

            # combine as columns in a frame
            else:
                data = dict(zip(range(len(self.objs)), self.objs))

                # GH28330 Preserves subclassed objects through concat
                cons = sample._constructor_expanddim

                index, columns = self.new_axes
                df = cons(data, index=index, copy=self.copy)
                df.columns = columns
                return df.__finalize__(self, method="concat")

        # combine block managers
        else:
            sample = cast("DataFrame", self.objs[0])

            mgrs_indexers = []
            for obj in self.objs:
                indexers = {}
                for ax, new_labels in enumerate(self.new_axes):
                    # ::-1 to convert BlockManager ax to DataFrame ax
                    if ax == self.bm_axis:
                        # Suppress reindexing on concat axis
                        continue

                    # 1-ax to convert BlockManager axis to DataFrame axis
                    obj_labels = obj.axes[1 - ax]
                    if not new_labels.equals(obj_labels):
                        indexers[ax] = obj_labels.get_indexer(new_labels)

                mgrs_indexers.append((obj._mgr, indexers))

            new_data = concatenate_managers(
                mgrs_indexers, self.new_axes, concat_axis=self.bm_axis, copy=self.copy
            )
            if not self.copy and not using_copy_on_write():
                new_data._consolidate_inplace()

            out = sample._constructor_from_mgr(new_data, axes=new_data.axes)
            return out.__finalize__(self, method="concat")

    def _get_result_dim(self) -> int:
        if self._is_series and self.bm_axis == 1:
            return 2
        else:
            return self.objs[0].ndim

    @cache_readonly
    def new_axes(self) -> list[Index]:
        ndim = self._get_result_dim()
        return [
            self._get_concat_axis if i == self.bm_axis else self._get_comb_axis(i)
            for i in range(ndim)
        ]

    def _get_comb_axis(self, i: AxisInt) -> Index:
        data_axis = self.objs[0]._get_block_manager_axis(i)
        return get_objs_combined_axis(
            self.objs,
            axis=data_axis,
            intersect=self.intersect,
            sort=self.sort,
            copy=self.copy,
        )

    @cache_readonly
    def _get_concat_axis(self) -> Index:
        """
        Return index to be used along concatenation axis.
        """
        if self._is_series:
            if self.bm_axis == 0:
                indexes = [x.index for x in self.objs]
            elif self.ignore_index:
                idx = default_index(len(self.objs))
                return idx
            elif self.keys is None:
                names: list[Hashable] = [None] * len(self.objs)
                num = 0
                has_names = False
                for i, x in enumerate(self.objs):
                    if x.ndim != 1:
                        raise TypeError(
                            f"Cannot concatenate type 'Series' with "
                            f"object of type '{type(x).__name__}'"
                        )
                    if x.name is not None:
                        names[i] = x.name
                        has_names = True
                    else:
                        names[i] = num
                        num += 1
                if has_names:
                    return Index(names)
                else:
                    return default_index(len(self.objs))
            else:
                return ensure_index(self.keys).set_names(self.names)
        else:
            indexes = [x.axes[self.axis] for x in self.objs]

        if self.ignore_index:
            idx = default_index(sum(len(i) for i in indexes))
            return idx

        if self.keys is None:
            if self.levels is not None:
                raise ValueError("levels supported only when keys is not None")
            concat_axis = _concat_indexes(indexes)
        else:
            concat_axis = _make_concat_multiindex(
                indexes, self.keys, self.levels, self.names
            )

        self._maybe_check_integrity(concat_axis)

        return concat_axis

    def _maybe_check_integrity(self, concat_index: Index):
        if self.verify_integrity:
            if not concat_index.is_unique:
                overlap = concat_index[concat_index.duplicated()].unique()
                raise ValueError(f"Indexes have overlapping values: {overlap}")


def _concat_indexes(indexes) -> Index:
    return indexes[0].append(indexes[1:])


def _make_concat_multiindex(indexes, keys, levels=None, names=None) -> MultiIndex:
    if (levels is None and isinstance(keys[0], tuple)) or (
        levels is not None and len(levels) > 1
    ):
        zipped = list(zip(*keys))
        if names is None:
            names = [None] * len(zipped)

        if levels is None:
            _, levels = factorize_from_iterables(zipped)
        else:
            levels = [ensure_index(x) for x in levels]
    else:
        zipped = [keys]
        if names is None:
            names = [None]

        if levels is None:
            levels = [ensure_index(keys).unique()]
        else:
            levels = [ensure_index(x) for x in levels]

    for level in levels:
        if not level.is_unique:
            raise ValueError(f"Level values not unique: {level.tolist()}")

    if not all_indexes_same(indexes) or not all(level.is_unique for level in levels):
        codes_list = []

        # things are potentially different sizes, so compute the exact codes
        # for each level and pass those to MultiIndex.from_arrays

        for hlevel, level in zip(zipped, levels):
            to_concat = []
            if isinstance(hlevel, Index) and hlevel.equals(level):
                lens = [len(idx) for idx in indexes]
                codes_list.append(np.repeat(np.arange(len(hlevel)), lens))
            else:
                for key, index in zip(hlevel, indexes):
                    # Find matching codes, include matching nan values as equal.
                    mask = (isna(level) & isna(key)) | (level == key)
                    if not mask.any():
                        raise ValueError(f"Key {key} not in level {level}")
                    i = np.nonzero(mask)[0][0]

                    to_concat.append(np.repeat(i, len(index)))
                codes_list.append(np.concatenate(to_concat))

        concat_index = _concat_indexes(indexes)

        # these go at the end
        if isinstance(concat_index, MultiIndex):
            levels.extend(concat_index.levels)
            codes_list.extend(concat_index.codes)
        else:
            codes, categories = factorize_from_iterable(concat_index)
            levels.append(categories)
            codes_list.append(codes)

        if len(names) == len(levels):
            names = list(names)
        else:
            # make sure that all of the passed indices have the same nlevels
            if not len({idx.nlevels for idx in indexes}) == 1:
                raise AssertionError(
                    "Cannot concat indices that do not have the same number of levels"
                )

            # also copies
            names = list(names) + list(get_unanimous_names(*indexes))

        return MultiIndex(
            levels=levels, codes=codes_list, names=names, verify_integrity=False
        )

    new_index = indexes[0]
    n = len(new_index)
    kpieces = len(indexes)

    # also copies
    new_names = list(names)
    new_levels = list(levels)

    # construct codes
    new_codes = []

    # do something a bit more speedy

    for hlevel, level in zip(zipped, levels):
        hlevel_index = ensure_index(hlevel)
        mapped = level.get_indexer(hlevel_index)

        mask = mapped == -1
        if mask.any():
            raise ValueError(
                f"Values not found in passed level: {hlevel_index[mask]!s}"
            )

        new_codes.append(np.repeat(mapped, n))

    if isinstance(new_index, MultiIndex):
        new_levels.extend(new_index.levels)
        new_codes.extend([np.tile(lab, kpieces) for lab in new_index.codes])
    else:
        new_levels.append(new_index.unique())
        single_codes = new_index.unique().get_indexer(new_index)
        new_codes.append(np.tile(single_codes, kpieces))

    if len(new_names) < len(new_levels):
        new_names.extend(new_index.names)

    return MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )
<|endoftext|>from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from pandas.util._decorators import Appender

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna

import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import tile_compat
from pandas.core.shared_docs import _shared_docs
from pandas.core.tools.numeric import to_numeric

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import AnyArrayLike

    from pandas import DataFrame


def ensure_list_vars(arg_vars, variable: str, columns) -> list:
    if arg_vars is not None:
        if not is_list_like(arg_vars):
            return [arg_vars]
        elif isinstance(columns, MultiIndex) and not isinstance(arg_vars, list):
            raise ValueError(
                f"{variable} must be a list of tuples when columns are a MultiIndex"
            )
        else:
            return list(arg_vars)
    else:
        return []


@Appender(_shared_docs["melt"] % {"caller": "pd.melt(df, ", "other": "DataFrame.melt"})
def melt(
    frame: DataFrame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name: Hashable = "value",
    col_level=None,
    ignore_index: bool = True,
) -> DataFrame:
    if value_name in frame.columns:
        raise ValueError(
            f"value_name ({value_name}) cannot match an element in "
            "the DataFrame columns."
        )
    id_vars = ensure_list_vars(id_vars, "id_vars", frame.columns)
    value_vars_was_not_none = value_vars is not None
    value_vars = ensure_list_vars(value_vars, "value_vars", frame.columns)

    if id_vars or value_vars:
        if col_level is not None:
            level = frame.columns.get_level_values(col_level)
        else:
            level = frame.columns
        labels = id_vars + value_vars
        idx = level.get_indexer_for(labels)
        missing = idx == -1
        if missing.any():
            missing_labels = [
                lab for lab, not_found in zip(labels, missing) if not_found
            ]
            raise KeyError(
                "The following id_vars or value_vars are not present in "
                f"the DataFrame: {missing_labels}"
            )
        if value_vars_was_not_none:
            frame = frame.iloc[:, algos.unique(idx)]
        else:
            frame = frame.copy()
    else:
        frame = frame.copy()

    if col_level is not None:  # allow list or other?
        # frame is a copy
        frame.columns = frame.columns.get_level_values(col_level)

    if var_name is None:
        if isinstance(frame.columns, MultiIndex):
            if len(frame.columns.names) == len(set(frame.columns.names)):
                var_name = frame.columns.names
            else:
                var_name = [f"variable_{i}" for i in range(len(frame.columns.names))]
        else:
            var_name = [
                frame.columns.name if frame.columns.name is not None else "variable"
            ]
    elif is_list_like(var_name):
        raise ValueError(f"{var_name=} must be a scalar.")
    else:
        var_name = [var_name]

    num_rows, K = frame.shape
    num_cols_adjusted = K - len(id_vars)

    mdata: dict[Hashable, AnyArrayLike] = {}
    for col in id_vars:
        id_data = frame.pop(col)
        if not isinstance(id_data.dtype, np.dtype):
            # i.e. ExtensionDtype
            if num_cols_adjusted > 0:
                mdata[col] = concat([id_data] * num_cols_adjusted, ignore_index=True)
            else:
                # We can't concat empty list. (GH 46044)
                mdata[col] = type(id_data)([], name=id_data.name, dtype=id_data.dtype)
        else:
            mdata[col] = np.tile(id_data._values, num_cols_adjusted)

    mcolumns = id_vars + var_name + [value_name]

    if frame.shape[1] > 0 and not any(
        not isinstance(dt, np.dtype) and dt._supports_2d for dt in frame.dtypes
    ):
        mdata[value_name] = concat(
            [frame.iloc[:, i] for i in range(frame.shape[1])]
        ).values
    else:
        mdata[value_name] = frame._values.ravel("F")
    for i, col in enumerate(var_name):
        mdata[col] = frame.columns._get_level_values(i).repeat(num_rows)

    result = frame._constructor(mdata, columns=mcolumns)

    if not ignore_index:
        result.index = tile_compat(frame.index, num_cols_adjusted)

    return result


def lreshape(data: DataFrame, groups: dict, dropna: bool = True) -> DataFrame:
    """
    Reshape wide-format data to long. Generalized inverse of DataFrame.pivot.

    Accepts a dictionary, ``groups``, in which each key is a new column name
    and each value is a list of old column names that will be "melted" under
    the new column name as part of the reshape.

    Parameters
    ----------
    data : DataFrame
        The wide-format DataFrame.
    groups : dict
        {new_name : list_of_columns}.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.

    Returns
    -------
    DataFrame
        Reshaped DataFrame.

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Examples
    --------
    >>> data = pd.DataFrame({'hr1': [514, 573], 'hr2': [545, 526],
    ...                      'team': ['Red Sox', 'Yankees'],
    ...                      'year1': [2007, 2007], 'year2': [2008, 2008]})
    >>> data
       hr1  hr2     team  year1  year2
    0  514  545  Red Sox   2007   2008
    1  573  526  Yankees   2007   2008

    >>> pd.lreshape(data, {'year': ['year1', 'year2'], 'hr': ['hr1', 'hr2']})
          team  year   hr
    0  Red Sox  2007  514
    1  Yankees  2007  573
    2  Red Sox  2008  545
    3  Yankees  2008  526
    """
    mdata = {}
    pivot_cols = []
    all_cols: set[Hashable] = set()
    K = len(next(iter(groups.values())))
    for target, names in groups.items():
        if len(names) != K:
            raise ValueError("All column lists must be same length")
        to_concat = [data[col]._values for col in names]

        mdata[target] = concat_compat(to_concat)
        pivot_cols.append(target)
        all_cols = all_cols.union(names)

    id_cols = list(data.columns.difference(all_cols))
    for col in id_cols:
        mdata[col] = np.tile(data[col]._values, K)

    if dropna:
        mask = np.ones(len(mdata[pivot_cols[0]]), dtype=bool)
        for c in pivot_cols:
            mask &= notna(mdata[c])
        if not mask.all():
            mdata = {k: v[mask] for k, v in mdata.items()}

    return data._constructor(mdata, columns=id_cols + pivot_cols)


def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = "", suffix: str = r"\d+"
) -> DataFrame:
    r"""
    Unpivot a DataFrame from wide to long format.

    Less flexible but more user-friendly than melt.

    With stubnames ['A', 'B'], this function expects to find one or more
    group of columns with format
    A-suffix1, A-suffix2,..., B-suffix1, B-suffix2,...
    You specify what you want to call this suffix in the resulting long format
    with `j` (for example `j='year'`)

    Each row of these wide variables are assumed to be uniquely identified by
    `i` (can be a single column name or a list of column names)

    All remaining variables in the data frame are left intact.

    Parameters
    ----------
    df : DataFrame
        The wide-format DataFrame.
    stubnames : str or list-like
        The stub name(s). The wide format variables are assumed to
        start with the stub names.
    i : str or list-like
        Column(s) to use as id variable(s).
    j : str
        The name of the sub-observation variable. What you wish to name your
        suffix in the long format.
    sep : str, default ""
        A character indicating the separation of the variable names
        in the wide format, to be stripped from the names in the long format.
        For example, if your column names are A-suffix1, A-suffix2, you
        can strip the hyphen by specifying `sep='-'`.
    suffix : str, default '\\d+'
        A regular expression capturing the wanted suffixes. '\\d+' captures
        numeric suffixes. Suffixes with no numbers could be specified with the
        negated character class '\\D+'. You can also further disambiguate
        suffixes, for example, if your wide variables are of the form A-one,
        B-two,.., and you have an unrelated column A-rating, you can ignore the
        last one by specifying `suffix='(!?one|two)'`. When all suffixes are
        numeric, they are cast to int64/float64.

    Returns
    -------
    DataFrame
        A DataFrame that contains each stub name as a variable, with new index
        (i, j).

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.

    Notes
    -----
    All extra variables are left untouched. This simply uses
    `pandas.melt` under the hood, but is hard-coded to "do the right thing"
    in a typical case.

    Examples
    --------
    >>> np.random.seed(123)
    >>> df = pd.DataFrame({"A1970" : {0 : "a", 1 : "b", 2 : "c"},
    ...                    "A1980" : {0 : "d", 1 : "e", 2 : "f"},
    ...                    "B1970" : {0 : 2.5, 1 : 1.2, 2 : .7},
    ...                    "B1980" : {0 : 3.2, 1 : 1.3, 2 : .1},
    ...                    "X"     : dict(zip(range(3), np.random.randn(3)))
    ...                   })
    >>> df["id"] = df.index
    >>> df
      A1970 A1980  B1970  B1980         X  id
    0     a     d    2.5    3.2 -1.085631   0
    1     b     e    1.2    1.3  0.997345   1
    2     c     f    0.7    0.1  0.282978   2
    >>> pd.wide_to_long(df, ["A", "B"], i="id", j="year")
    ... # doctest: +NORMALIZE_WHITESPACE
                    X  A    B
    id year
    0  1970 -1.085631  a  2.5
    1  1970  0.997345  b  1.2
    2  1970  0.282978  c  0.7
    0  1980 -1.085631  d  3.2
    1  1980  0.997345  e  1.3
    2  1980  0.282978  f  0.1

    With multiple id columns

    >>> df = pd.DataFrame({
    ...     'famid': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'birth': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ...     'ht1': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
    ...     'ht2': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]
    ... })
    >>> df
       famid  birth  ht1  ht2
    0      1      1  2.8  3.4
    1      1      2  2.9  3.8
    2      1      3  2.2  2.9
    3      2      1  2.0  3.2
    4      2      2  1.8  2.8
    5      2      3  1.9  2.4
    6      3      1  2.2  3.3
    7      3      2  2.3  3.4
    8      3      3  2.1  2.9
    >>> l = pd.wide_to_long(df, stubnames='ht', i=['famid', 'birth'], j='age')
    >>> l
    ... # doctest: +NORMALIZE_WHITESPACE
                      ht
    famid birth age
    1     1     1    2.8
                2    3.4
          2     1    2.9
                2    3.8
          3     1    2.2
                2    2.9
    2     1     1    2.0
                2    3.2
          2     1    1.8
                2    2.8
          3     1    1.9
                2    2.4
    3     1     1    2.2
                2    3.3
          2     1    2.3
                2    3.4
          3     1    2.1
                2    2.9

    Going from long back to wide just takes some creative use of `unstack`

    >>> w = l.unstack()
    >>> w.columns = w.columns.map('{0[0]}{0[1]}'.format)
    >>> w.reset_index()
       famid  birth  ht1  ht2
    0      1      1  2.8  3.4
    1      1      2  2.9  3.8
    2      1      3  2.2  2.9
    3      2      1  2.0  3.2
    4      2      2  1.8  2.8
    5      2      3  1.9  2.4
    6      3      1  2.2  3.3
    7      3      2  2.3  3.4
    8      3      3  2.1  2.9

    Less wieldy column names are also handled

    >>> np.random.seed(0)
    >>> df = pd.DataFrame({'A(weekly)-2010': np.random.rand(3),
    ...                    'A(weekly)-2011': np.random.rand(3),
    ...                    'B(weekly)-2010': np.random.rand(3),
    ...                    'B(weekly)-2011': np.random.rand(3),
    ...                    'X' : np.random.randint(3, size=3)})
    >>> df['id'] = df.index
    >>> df # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
       A(weekly)-2010  A(weekly)-2011  B(weekly)-2010  B(weekly)-2011  X  id
    0        0.548814        0.544883        0.437587        0.383442  0   0
    1        0.715189        0.423655        0.891773        0.791725  1   1
    2        0.602763        0.645894        0.963663        0.528895  1   2

    >>> pd.wide_to_long(df, ['A(weekly)', 'B(weekly)'], i='id',
    ...                 j='year', sep='-')
    ... # doctest: +NORMALIZE_WHITESPACE
             X  A(weekly)  B(weekly)
    id year
    0  2010  0   0.548814   0.437587
    1  2010  1   0.715189   0.891773
    2  2010  1   0.602763   0.963663
    0  2011  0   0.544883   0.383442
    1  2011  1   0.423655   0.791725
    2  2011  1   0.645894   0.528895

    If we have many columns, we could also use a regex to find our
    stubnames and pass that list on to wide_to_long

    >>> stubnames = sorted(
    ...     set([match[0] for match in df.columns.str.findall(
    ...         r'[A-B]\(.*\)').values if match != []])
    ... )
    >>> list(stubnames)
    ['A(weekly)', 'B(weekly)']

    All of the above examples have integers as suffixes. It is possible to
    have non-integers as suffixes.

    >>> df = pd.DataFrame({
    ...     'famid': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'birth': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ...     'ht_one': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
    ...     'ht_two': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]
    ... })
    >>> df
       famid  birth  ht_one  ht_two
    0      1      1     2.8     3.4
    1      1      2     2.9     3.8
    2      1      3     2.2     2.9
    3      2      1     2.0     3.2
    4      2      2     1.8     2.8
    5      2      3     1.9     2.4
    6      3      1     2.2     3.3
    7      3      2     2.3     3.4
    8      3      3     2.1     2.9

    >>> l = pd.wide_to_long(df, stubnames='ht', i=['famid', 'birth'], j='age',
    ...                     sep='_', suffix=r'\w+')
    >>> l
    ... # doctest: +NORMALIZE_WHITESPACE
                      ht
    famid birth age
    1     1     one  2.8
                two  3.4
          2     one  2.9
                two  3.8
          3     one  2.2
                two  2.9
    2     1     one  2.0
                two  3.2
          2     one  1.8
                two  2.8
          3     one  1.9
                two  2.4
    3     1     one  2.2
                two  3.3
          2     one  2.3
                two  3.4
          3     one  2.1
                two  2.9
    """

    def get_var_names(df, stub: str, sep: str, suffix: str):
        regex = rf"^{re.escape(stub)}{re.escape(sep)}{suffix}$"
        return df.columns[df.columns.str.match(regex)]

    def melt_stub(df, stub: str, i, j, value_vars, sep: str):
        newdf = melt(
            df,
            id_vars=i,
            value_vars=value_vars,
            value_name=stub.rstrip(sep),
            var_name=j,
        )
        newdf[j] = newdf[j].str.replace(re.escape(stub + sep), "", regex=True)

        # GH17627 Cast numerics suffixes to int/float
        try:
            newdf[j] = to_numeric(newdf[j])
        except (TypeError, ValueError, OverflowError):
            # TODO: anything else to catch?
            pass

        return newdf.set_index(i + [j])

    if not is_list_like(stubnames):
        stubnames = [stubnames]
    else:
        stubnames = list(stubnames)

    if df.columns.isin(stubnames).any():
        raise ValueError("stubname can't be identical to a column name")

    if not is_list_like(i):
        i = [i]
    else:
        i = list(i)

    if df[i].duplicated().any():
        raise ValueError("the id variables need to uniquely identify each row")

    _melted = []
    value_vars_flattened = []
    for stub in stubnames:
        value_var = get_var_names(df, stub, sep, suffix)
        value_vars_flattened.extend(value_var)
        _melted.append(melt_stub(df, stub, i, j, value_var, sep))

    melted = concat(_melted, axis=1)
    id_vars = df.columns.difference(value_vars_flattened)
    new = df[id_vars]

    if len(i) == 1:
        return new.set_index(i).join(melted)
    else:
        return new.merge(melted.reset_index(), on=i).set_index(i + [j])
<|endoftext|>"""
SQL-style merge routines
"""
from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
import datetime
from functools import partial
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    final,
)
import uuid
import warnings

import numpy as np

from pandas._libs import (
    Timedelta,
    hashtable as libhashtable,
    join as libjoin,
    lib,
)
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    IndexLabel,
    JoinHow,
    MergeHow,
    Shape,
    Suffixes,
    npt,
)
from pandas.errors import MergeError
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_object,
    is_bool,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas import (
    ArrowDtype,
    Categorical,
    Index,
    MultiIndex,
    Series,
)
import pandas.core.algorithms as algos
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    ExtensionArray,
)
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
    get_group_index,
    is_int64_overflow_possible,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.core import groupby
    from pandas.core.arrays import DatetimeArray
    from pandas.core.indexes.frozen import FrozenList

_factorizers = {
    np.int64: libhashtable.Int64Factorizer,
    np.longlong: libhashtable.Int64Factorizer,
    np.int32: libhashtable.Int32Factorizer,
    np.int16: libhashtable.Int16Factorizer,
    np.int8: libhashtable.Int8Factorizer,
    np.uint64: libhashtable.UInt64Factorizer,
    np.uint32: libhashtable.UInt32Factorizer,
    np.uint16: libhashtable.UInt16Factorizer,
    np.uint8: libhashtable.UInt8Factorizer,
    np.bool_: libhashtable.UInt8Factorizer,
    np.float64: libhashtable.Float64Factorizer,
    np.float32: libhashtable.Float32Factorizer,
    np.complex64: libhashtable.Complex64Factorizer,
    np.complex128: libhashtable.Complex128Factorizer,
    np.object_: libhashtable.ObjectFactorizer,
}

# See https://github.com/pandas-dev/pandas/issues/52451
if np.intc is not np.int32:
    _factorizers[np.intc] = libhashtable.Int64Factorizer

_known = (np.ndarray, ExtensionArray, Index, ABCSeries)


@Substitution("\nleft : DataFrame or named Series")
@Appender(_merge_doc, indents=0)
def merge(
    left: DataFrame | Series,
    right: DataFrame | Series,
    how: MergeHow = "inner",
    on: IndexLabel | AnyArrayLike | None = None,
    left_on: IndexLabel | AnyArrayLike | None = None,
    right_on: IndexLabel | AnyArrayLike | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    copy: bool | None = None,
    indicator: str | bool = False,
    validate: str | None = None,
) -> DataFrame:
    left_df = _validate_operand(left)
    right_df = _validate_operand(right)
    if how == "cross":
        return _cross_merge(
            left_df,
            right_df,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
            copy=copy,
        )
    else:
        op = _MergeOperation(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
        )
        return op.get_result(copy=copy)


def _cross_merge(
    left: DataFrame,
    right: DataFrame,
    on: IndexLabel | AnyArrayLike | None = None,
    left_on: IndexLabel | AnyArrayLike | None = None,
    right_on: IndexLabel | AnyArrayLike | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    copy: bool | None = None,
    indicator: str | bool = False,
    validate: str | None = None,
) -> DataFrame:
    """
    See merge.__doc__ with how='cross'
    """

    if (
        left_index
        or right_index
        or right_on is not None
        or left_on is not None
        or on is not None
    ):
        raise MergeError(
            "Can not pass on, right_on, left_on or set right_index=True or "
            "left_index=True"
        )

    cross_col = f"_cross_{uuid.uuid4()}"
    left = left.assign(**{cross_col: 1})
    right = right.assign(**{cross_col: 1})

    left_on = right_on = [cross_col]

    res = merge(
        left,
        right,
        how="inner",
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        indicator=indicator,
        validate=validate,
        copy=copy,
    )
    del res[cross_col]
    return res


def _groupby_and_merge(
    by, left: DataFrame | Series, right: DataFrame | Series, merge_pieces
):
    """
    groupby & merge; we are always performing a left-by type operation

    Parameters
    ----------
    by: field to group
    left: DataFrame
    right: DataFrame
    merge_pieces: function for merging
    """
    pieces = []
    if not isinstance(by, (list, tuple)):
        by = [by]

    lby = left.groupby(by, sort=False)
    rby: groupby.DataFrameGroupBy | groupby.SeriesGroupBy | None = None

    # if we can groupby the rhs
    # then we can get vastly better perf
    if all(item in right.columns for item in by):
        rby = right.groupby(by, sort=False)

    for key, lhs in lby._grouper.get_iterator(lby._selected_obj, axis=lby.axis):
        if rby is None:
            rhs = right
        else:
            try:
                rhs = right.take(rby.indices[key])
            except KeyError:
                # key doesn't exist in left
                lcols = lhs.columns.tolist()
                cols = lcols + [r for r in right.columns if r not in set(lcols)]
                merged = lhs.reindex(columns=cols)
                merged.index = range(len(merged))
                pieces.append(merged)
                continue

        merged = merge_pieces(lhs, rhs)

        # make sure join keys are in the merged
        # TODO, should merge_pieces do this?
        merged[by] = key

        pieces.append(merged)

    # preserve the original order
    # if we have a missing piece this can be reset
    from pandas.core.reshape.concat import concat

    result = concat(pieces, ignore_index=True)
    result = result.reindex(columns=pieces[0].columns, copy=False)
    return result, lby


def merge_ordered(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: IndexLabel | None = None,
    left_on: IndexLabel | None = None,
    right_on: IndexLabel | None = None,
    left_by=None,
    right_by=None,
    fill_method: str | None = None,
    suffixes: Suffixes = ("_x", "_y"),
    how: JoinHow = "outer",
) -> DataFrame:
    """
    Perform a merge for ordered data with optional filling/interpolation.

    Designed for ordered data like time series data. Optionally
    perform group-wise merge (see examples).

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
    on : label or list
        Field names to join on. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector or list of
        vectors of the length of the DataFrame to use a particular vector as
        the join key instead of columns.
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of vectors per
        left_on docs.
    left_by : column name or list of column names
        Group left DataFrame by group columns and merge piece by piece with
        right DataFrame. Must be None if either left or right are a Series.
    right_by : column name or list of column names
        Group right DataFrame by group columns and merge piece by piece with
        left DataFrame. Must be None if either left or right are a Series.
    fill_method : {'ffill', None}, default None
        Interpolation method for data.
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.

    how : {'left', 'right', 'outer', 'inner'}, default 'outer'
        * left: use only keys from left frame (SQL: left outer join)
        * right: use only keys from right frame (SQL: right outer join)
        * outer: use union of keys from both frames (SQL: full outer join)
        * inner: use intersection of keys from both frames (SQL: inner join).

    Returns
    -------
    DataFrame
        The merged DataFrame output type will be the same as
        'left', if it is a subclass of DataFrame.

    See Also
    --------
    merge : Merge with a database-style join.
    merge_asof : Merge on nearest keys.

    Examples
    --------
    >>> from pandas import merge_ordered
    >>> df1 = pd.DataFrame(
    ...     {
    ...         "key": ["a", "c", "e", "a", "c", "e"],
    ...         "lvalue": [1, 2, 3, 1, 2, 3],
    ...         "group": ["a", "a", "a", "b", "b", "b"]
    ...     }
    ... )
    >>> df1
      key  lvalue group
    0   a       1     a
    1   c       2     a
    2   e       3     a
    3   a       1     b
    4   c       2     b
    5   e       3     b

    >>> df2 = pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
    >>> df2
      key  rvalue
    0   b       1
    1   c       2
    2   d       3

    >>> merge_ordered(df1, df2, fill_method="ffill", left_by="group")
      key  lvalue group  rvalue
    0   a       1     a     NaN
    1   b       1     a     1.0
    2   c       2     a     2.0
    3   d       2     a     3.0
    4   e       3     a     3.0
    5   a       1     b     NaN
    6   b       1     b     1.0
    7   c       2     b     2.0
    8   d       2     b     3.0
    9   e       3     b     3.0
    """

    def _merger(x, y) -> DataFrame:
        # perform the ordered merge operation
        op = _OrderedMerge(
            x,
            y,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            fill_method=fill_method,
            how=how,
        )
        return op.get_result()

    if left_by is not None and right_by is not None:
        raise ValueError("Can only group either left or right frames")
    if left_by is not None:
        if isinstance(left_by, str):
            left_by = [left_by]
        check = set(left_by).difference(left.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in left columns")
        result, _ = _groupby_and_merge(left_by, left, right, lambda x, y: _merger(x, y))
    elif right_by is not None:
        if isinstance(right_by, str):
            right_by = [right_by]
        check = set(right_by).difference(right.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in right columns")
        result, _ = _groupby_and_merge(
            right_by, right, left, lambda x, y: _merger(y, x)
        )
    else:
        result = _merger(left, right)
    return result


def merge_asof(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: IndexLabel | None = None,
    left_on: IndexLabel | None = None,
    right_on: IndexLabel | None = None,
    left_index: bool = False,
    right_index: bool = False,
    by=None,
    left_by=None,
    right_by=None,
    suffixes: Suffixes = ("_x", "_y"),
    tolerance: int | Timedelta | None = None,
    allow_exact_matches: bool = True,
    direction: str = "backward",
) -> DataFrame:
    """
    Perform a merge by key distance.

    This is similar to a left-join except that we match on nearest
    key rather than equal keys. Both DataFrames must be sorted by the key.

    For each row in the left DataFrame:

      - A "backward" search selects the last row in the right DataFrame whose
        'on' key is less than or equal to the left's key.

      - A "forward" search selects the first row in the right DataFrame whose
        'on' key is greater than or equal to the left's key.

      - A "nearest" search selects the row in the right DataFrame whose 'on'
        key is closest in absolute distance to the left's key.

    Optionally match on equivalent keys with 'by' before searching with 'on'.

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
    on : label
        Field name to join on. Must be found in both DataFrames.
        The data MUST be ordered. Furthermore this must be a numeric column,
        such as datetimelike, integer, or float. On or left_on/right_on
        must be given.
    left_on : label
        Field name to join on in left DataFrame.
    right_on : label
        Field name to join on in right DataFrame.
    left_index : bool
        Use the index of the left DataFrame as the join key.
    right_index : bool
        Use the index of the right DataFrame as the join key.
    by : column name or list of column names
        Match on these columns before performing merge operation.
    left_by : column name
        Field names to match on in the left DataFrame.
    right_by : column name
        Field names to match on in the right DataFrame.
    suffixes : 2-length sequence (tuple, list, ...)
        Suffix to apply to overlapping column names in the left and right
        side, respectively.
    tolerance : int or Timedelta, optional, default None
        Select asof tolerance within this range; must be compatible
        with the merge index.
    allow_exact_matches : bool, default True

        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than).

    direction : 'backward' (default), 'forward', or 'nearest'
        Whether to search for prior, subsequent, or closest matches.

    Returns
    -------
    DataFrame

    See Also
    --------
    merge : Merge with a database-style join.
    merge_ordered : Merge with optional filling/interpolation.

    Examples
    --------
    >>> left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    >>> left
        a left_val
    0   1        a
    1   5        b
    2  10        c

    >>> right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})
    >>> right
       a  right_val
    0  1          1
    1  2          2
    2  3          3
    3  6          6
    4  7          7

    >>> pd.merge_asof(left, right, on="a")
        a left_val  right_val
    0   1        a          1
    1   5        b          3
    2  10        c          7

    >>> pd.merge_asof(left, right, on="a", allow_exact_matches=False)
        a left_val  right_val
    0   1        a        NaN
    1   5        b        3.0
    2  10        c        7.0

    >>> pd.merge_asof(left, right, on="a", direction="forward")
        a left_val  right_val
    0   1        a        1.0
    1   5        b        6.0
    2  10        c        NaN

    >>> pd.merge_asof(left, right, on="a", direction="nearest")
        a left_val  right_val
    0   1        a          1
    1   5        b          6
    2  10        c          7

    We can use indexed DataFrames as well.

    >>> left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
    >>> left
       left_val
    1         a
    5         b
    10        c

    >>> right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    >>> right
       right_val
    1          1
    2          2
    3          3
    6          6
    7          7

    >>> pd.merge_asof(left, right, left_index=True, right_index=True)
       left_val  right_val
    1         a          1
    5         b          3
    10        c          7

    Here is a real-world times-series example

    >>> quotes = pd.DataFrame(
    ...     {
    ...         "time": [
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.030"),
    ...             pd.Timestamp("2016-05-25 13:30:00.041"),
    ...             pd.Timestamp("2016-05-25 13:30:00.048"),
    ...             pd.Timestamp("2016-05-25 13:30:00.049"),
    ...             pd.Timestamp("2016-05-25 13:30:00.072"),
    ...             pd.Timestamp("2016-05-25 13:30:00.075")
    ...         ],
    ...         "ticker": [
    ...                "GOOG",
    ...                "MSFT",
    ...                "MSFT",
    ...                "MSFT",
    ...                "GOOG",
    ...                "AAPL",
    ...                "GOOG",
    ...                "MSFT"
    ...            ],
    ...            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
    ...            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03]
    ...     }
    ... )
    >>> quotes
                         time ticker     bid     ask
    0 2016-05-25 13:30:00.023   GOOG  720.50  720.93
    1 2016-05-25 13:30:00.023   MSFT   51.95   51.96
    2 2016-05-25 13:30:00.030   MSFT   51.97   51.98
    3 2016-05-25 13:30:00.041   MSFT   51.99   52.00
    4 2016-05-25 13:30:00.048   GOOG  720.50  720.93
    5 2016-05-25 13:30:00.049   AAPL   97.99   98.01
    6 2016-05-25 13:30:00.072   GOOG  720.50  720.88
    7 2016-05-25 13:30:00.075   MSFT   52.01   52.03

    >>> trades = pd.DataFrame(
    ...        {
    ...            "time": [
    ...                pd.Timestamp("2016-05-25 13:30:00.023"),
    ...                pd.Timestamp("2016-05-25 13:30:00.038"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048")
    ...            ],
    ...            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
    ...            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
    ...            "quantity": [75, 155, 100, 100, 100]
    ...        }
    ...    )
    >>> trades
                         time ticker   price  quantity
    0 2016-05-25 13:30:00.023   MSFT   51.95        75
    1 2016-05-25 13:30:00.038   MSFT   51.95       155
    2 2016-05-25 13:30:00.048   GOOG  720.77       100
    3 2016-05-25 13:30:00.048   GOOG  720.92       100
    4 2016-05-25 13:30:00.048   AAPL   98.00       100

    By default we are taking the asof of the quotes

    >>> pd.merge_asof(trades, quotes, on="time", by="ticker")
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN

    We only asof within 2ms between the quote time and the trade time

    >>> pd.merge_asof(
    ...     trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms")
    ... )
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
    1 2016-05-25 13:30:00.038   MSFT   51.95       155     NaN     NaN
    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN

    We only asof within 10ms between the quote time and the trade time
    and we exclude exact matches on time. However *prior* data will
    propagate forward

    >>> pd.merge_asof(
    ...     trades,
    ...     quotes,
    ...     on="time",
    ...     by="ticker",
    ...     tolerance=pd.Timedelta("10ms"),
    ...     allow_exact_matches=False
    ... )
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75     NaN     NaN
    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
    2 2016-05-25 13:30:00.048   GOOG  720.77       100     NaN     NaN
    3 2016-05-25 13:30:00.048   GOOG  720.92       100     NaN     NaN
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN
    """
    op = _AsOfMerge(
        left,
        right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        by=by,
        left_by=left_by,
        right_by=right_by,
        suffixes=suffixes,
        how="asof",
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
        direction=direction,
    )
    return op.get_result()


# TODO: transformations??
# TODO: only copy DataFrames when modification necessary
class _MergeOperation:
    """
    Perform a database (SQL) merge operation between two DataFrame or Series
    objects using either columns as keys or their row indexes
    """

    _merge_type = "merge"
    how: JoinHow | Literal["asof"]
    on: IndexLabel | None
    # left_on/right_on may be None when passed, but in validate_specification
    #  get replaced with non-None.
    left_on: Sequence[Hashable | AnyArrayLike]
    right_on: Sequence[Hashable | AnyArrayLike]
    left_index: bool
    right_index: bool
    sort: bool
    suffixes: Suffixes
    copy: bool
    indicator: str | bool
    validate: str | None
    join_names: list[Hashable]
    right_join_keys: list[ArrayLike]
    left_join_keys: list[ArrayLike]

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        how: JoinHow | Literal["asof"] = "inner",
        on: IndexLabel | AnyArrayLike | None = None,
        left_on: IndexLabel | AnyArrayLike | None = None,
        right_on: IndexLabel | AnyArrayLike | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = True,
        suffixes: Suffixes = ("_x", "_y"),
        indicator: str | bool = False,
        validate: str | None = None,
    ) -> None:
        _left = _validate_operand(left)
        _right = _validate_operand(right)
        self.left = self.orig_left = _left
        self.right = self.orig_right = _right
        self.how = how

        self.on = com.maybe_make_list(on)

        self.suffixes = suffixes
        self.sort = sort or how == "outer"

        self.left_index = left_index
        self.right_index = right_index

        self.indicator = indicator

        if not is_bool(left_index):
            raise ValueError(
                f"left_index parameter must be of type bool, not {type(left_index)}"
            )
        if not is_bool(right_index):
            raise ValueError(
                f"right_index parameter must be of type bool, not {type(right_index)}"
            )

        # GH 40993: raise when merging between different levels; enforced in 2.0
        if _left.columns.nlevels != _right.columns.nlevels:
            msg = (
                "Not allowed to merge between different levels. "
                f"({_left.columns.nlevels} levels on the left, "
                f"{_right.columns.nlevels} on the right)"
            )
            raise MergeError(msg)

        self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)

        (
            self.left_join_keys,
            self.right_join_keys,
            self.join_names,
            left_drop,
            right_drop,
        ) = self._get_merge_keys()

        if left_drop:
            self.left = self.left._drop_labels_or_levels(left_drop)

        if right_drop:
            self.right = self.right._drop_labels_or_levels(right_drop)

        self._maybe_require_matching_dtypes(self.left_join_keys, self.right_join_keys)
        self._validate_tolerance(self.left_join_keys)

        # validate the merge keys dtypes. We may need to coerce
        # to avoid incompatible dtypes
        self._maybe_coerce_merge_keys()

        # If argument passed to validate,
        # check if columns specified as unique
        # are in fact unique.
        if validate is not None:
            self._validate_validate_kwd(validate)

    def _maybe_require_matching_dtypes(
        self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]
    ) -> None:
        # Overridden by AsOfMerge
        pass

    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
        # Overridden by AsOfMerge
        pass

    @final
    def _reindex_and_concat(
        self,
        join_index: Index,
        left_indexer: npt.NDArray[np.intp] | None,
        right_indexer: npt.NDArray[np.intp] | None,
        copy: bool | None,
    ) -> DataFrame:
        """
        reindex along index and concat along columns.
        """
        # Take views so we do not alter the originals
        left = self.left[:]
        right = self.right[:]

        llabels, rlabels = _items_overlap_with_suffix(
            self.left._info_axis, self.right._info_axis, self.suffixes
        )

        if left_indexer is not None and not is_range_indexer(left_indexer, len(left)):
            # Pinning the index here (and in the right code just below) is not
            #  necessary, but makes the `.take` more performant if we have e.g.
            #  a MultiIndex for left.index.
            lmgr = left._mgr.reindex_indexer(
                join_index,
                left_indexer,
                axis=1,
                copy=False,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            left = left._constructor_from_mgr(lmgr, axes=lmgr.axes)
        left.index = join_index

        if right_indexer is not None and not is_range_indexer(
            right_indexer, len(right)
        ):
            rmgr = right._mgr.reindex_indexer(
                join_index,
                right_indexer,
                axis=1,
                copy=False,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            right = right._constructor_from_mgr(rmgr, axes=rmgr.axes)
        right.index = join_index

        from pandas import concat

        left.columns = llabels
        right.columns = rlabels
        result = concat([left, right], axis=1, copy=copy)
        return result

    def get_result(self, copy: bool | None = True) -> DataFrame:
        if self.indicator:
            self.left, self.right = self._indicator_pre_merge(self.left, self.right)

        join_index, left_indexer, right_indexer = self._get_join_info()

        result = self._reindex_and_concat(
            join_index, left_indexer, right_indexer, copy=copy
        )
        result = result.__finalize__(self, method=self._merge_type)

        if self.indicator:
            result = self._indicator_post_merge(result)

        self._maybe_add_join_keys(result, left_indexer, right_indexer)

        self._maybe_restore_index_levels(result)

        return result.__finalize__(self, method="merge")

    @final
    @cache_readonly
    def _indicator_name(self) -> str | None:
        if isinstance(self.indicator, str):
            return self.indicator
        elif isinstance(self.indicator, bool):
            return "_merge" if self.indicator else None
