    def _raise(left, right, err_msg) -> NoReturn:
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(
                    obj, f"{obj} shapes are different", left.shape, right.shape
                )

            diff = 0
            for left_arr, right_arr in zip(left, right):
                # count up differences
                if not array_equivalent(left_arr, right_arr, strict_nan=strict_nan):
                    diff += 1

            diff = diff * 100.0 / left.size
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    # compare shape and values
    if not array_equivalent(left, right, strict_nan=strict_nan):
        _raise(left, right, err_msg)

    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            assert_attr_equal("dtype", left, right, obj=obj)


def assert_extension_array_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    index_values=None,
    check_exact: bool | lib.NoDefault = lib.no_default,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "ExtensionArray",
) -> None:
    """
    Check that left and right ExtensionArrays are equal.

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : Index | numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'ExtensionArray'
        Specify object name being compared, internally used to show appropriate
        assertion message.

        .. versionadded:: 2.0.0

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> tm.assert_extension_array_equal(b, c)
    """
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype)
            and not is_float_dtype(left.dtype)
            or is_numeric_dtype(right.dtype)
            and not is_float_dtype(right.dtype)
        )
    elif check_exact is lib.no_default:
        check_exact = False

    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    assert isinstance(left, ExtensionArray), "left is not an ExtensionArray"
    assert isinstance(right, ExtensionArray), "right is not an ExtensionArray"
    if check_dtype:
        assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    if (
        isinstance(left, DatetimeLikeArrayMixin)
        and isinstance(right, DatetimeLikeArrayMixin)
        and type(right) == type(left)
    ):
        # GH 52449
        if not check_dtype and left.dtype.kind in "mM":
            if not isinstance(left.dtype, np.dtype):
                l_unit = cast(DatetimeTZDtype, left.dtype).unit
            else:
                l_unit = np.datetime_data(left.dtype)[0]
            if not isinstance(right.dtype, np.dtype):
                r_unit = cast(DatetimeTZDtype, right.dtype).unit
            else:
                r_unit = np.datetime_data(right.dtype)[0]
            if (
                l_unit != r_unit
                and compare_mismatched_resolutions(
                    left._ndarray, right._ndarray, operator.eq
                ).all()
            ):
                return
        # Avoid slow object-dtype comparisons
        # np.asarray for case where we have a np.MaskedArray
        assert_numpy_array_equal(
            np.asarray(left.asi8),
            np.asarray(right.asi8),
            index_values=index_values,
            obj=obj,
        )
        return

    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    assert_numpy_array_equal(
        left_na, right_na, obj=f"{obj} NA mask", index_values=index_values
    )

    # Specifically for StringArrayNumpySemantics, validate here we have a valid array
    if (
        isinstance(left.dtype, StringDtype)
        and left.dtype.storage == "python"
        and left.dtype.na_value is np.nan
    ):
        assert np.all(
            [np.isnan(val) for val in left._ndarray[left_na]]  # type: ignore[attr-defined]
        ), "wrong missing value sentinels"
    if (
        isinstance(right.dtype, StringDtype)
        and right.dtype.storage == "python"
        and right.dtype.na_value is np.nan
    ):
        assert np.all(
            [np.isnan(val) for val in right._ndarray[right_na]]  # type: ignore[attr-defined]
        ), "wrong missing value sentinels"

    left_valid = left[~left_na].to_numpy(dtype=object)
    right_valid = right[~right_na].to_numpy(dtype=object)
    if check_exact:
        assert_numpy_array_equal(
            left_valid, right_valid, obj=obj, index_values=index_values
        )
    else:
        _testing.assert_almost_equal(
            left_valid,
            right_valid,
            check_dtype=bool(check_dtype),
            rtol=rtol,
            atol=atol,
            obj=obj,
            index_values=index_values,
        )


# This could be refactored to use the NDFrame.equals method
def assert_series_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
    right : Series
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.

        .. versionadded:: 1.3.0
    check_like : bool, default False
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: same labels must be with the same data.

        .. versionadded:: 1.5.0

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> tm.assert_series_equal(a, b)
    """
    __tracebackhide__ = True
    check_exact_index = False if check_exact is lib.no_default else check_exact
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype)
            and not is_float_dtype(left.dtype)
            or is_numeric_dtype(right.dtype)
            and not is_float_dtype(right.dtype)
        )
    elif check_exact is lib.no_default:
        check_exact = False

    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    if not check_index and check_like:
        raise ValueError("check_like must be False if check_index is False")

    # instance validation
    _check_isinstance(left, right, Series)

    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    if check_flags:
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"

    if check_index:
        # GH #38183
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=check_exact_index,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )

    if check_like:
        left = left.reindex_like(right)

    if check_freq and isinstance(left.index, (DatetimeIndex, TimedeltaIndex)):
        lidx = left.index
        ridx = right.index
        assert lidx.freq == ridx.freq, (lidx.freq, ridx.freq)

    if check_dtype:
        # We want to skip exact dtype checking when `check_categorical`
        # is False. We'll still raise if only one is a `Categorical`,
        # regardless of `check_categorical`
        if (
            isinstance(left.dtype, CategoricalDtype)
            and isinstance(right.dtype, CategoricalDtype)
            and not check_categorical
        ):
            pass
        else:
            assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    if check_exact:
        left_values = left._values
        right_values = right._values
        # Only check exact if dtype is numeric
        if isinstance(left_values, ExtensionArray) and isinstance(
            right_values, ExtensionArray
        ):
            assert_extension_array_equal(
                left_values,
                right_values,
                check_dtype=check_dtype,
                index_values=left.index,
                obj=str(obj),
            )
        else:
            # convert both to NumPy if not, check_dtype would raise earlier
            lv, rv = left_values, right_values
            if isinstance(left_values, ExtensionArray):
                lv = left_values.to_numpy()
            if isinstance(right_values, ExtensionArray):
                rv = right_values.to_numpy()
            assert_numpy_array_equal(
                lv,
                rv,
                check_dtype=check_dtype,
                obj=str(obj),
                index_values=left.index,
            )
    elif check_datetimelike_compat and (
        needs_i8_conversion(left.dtype) or needs_i8_conversion(right.dtype)
    ):
        # we want to check only if we have compat dtypes
        # e.g. integer and M|m are NOT compat, but we can simply check
        # the values in that case

        # datetimelike may have different objects (e.g. datetime.datetime
        # vs Timestamp) but will compare equal
        if not Index(left._values).equals(Index(right._values)):
            msg = (
                f"[datetimelike_compat=True] {left._values} "
                f"is not equal to {right._values}."
            )
            raise AssertionError(msg)
    elif isinstance(left.dtype, IntervalDtype) and isinstance(
        right.dtype, IntervalDtype
    ):
        assert_interval_array_equal(left.array, right.array)
    elif isinstance(left.dtype, CategoricalDtype) or isinstance(
        right.dtype, CategoricalDtype
    ):
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )
    elif isinstance(left.dtype, ExtensionDtype) and isinstance(
        right.dtype, ExtensionDtype
    ):
        assert_extension_array_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    elif is_extension_array_dtype_and_needs_i8_conversion(
        left.dtype, right.dtype
    ) or is_extension_array_dtype_and_needs_i8_conversion(right.dtype, left.dtype):
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    elif needs_i8_conversion(left.dtype) and needs_i8_conversion(right.dtype):
        # DatetimeArray or TimedeltaArray
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    else:
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("name", left, right, obj=obj)

    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            assert_categorical_equal(
                left._values,
                right._values,
                obj=f"{obj} category",
                check_category_order=check_category_order,
            )


# This could be refactored to use the NDFrame.equals method
def assert_frame_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_column_type: bool | Literal["equiv"] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "DataFrame",
) -> None:
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True
    _rtol = rtol if rtol is not lib.no_default else 1.0e-5
    _atol = atol if atol is not lib.no_default else 1.0e-8
    _check_exact = check_exact if check_exact is not lib.no_default else False

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(
            obj, f"{obj} shape mismatch", f"{repr(left.shape)}", f"{repr(right.shape)}"
        )

    if check_flags:
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.index",
    )

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        left = left.reindex_like(right)

    # compare by blocks
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            assert_frame_equal(
                lblocks[dtype], rblocks[dtype], check_dtype=check_dtype, obj=obj
            )

    # compare by columns
    else:
        for i, col in enumerate(left.columns):
            # We have already checked that columns match, so we can do
            #  fast location-based lookups
            lcol = left._ixs(i, axis=1)
            rcol = right._ixs(i, axis=1)

            # GH #38183
            # use check_index=False, because we do not want to run
            # assert_index_equal for each column,
            # as we already checked it for the whole dataframe before.
            assert_series_equal(
                lcol,
                rcol,
                check_dtype=check_dtype,
                check_index_type=check_index_type,
                check_exact=check_exact,
                check_names=check_names,
                check_datetimelike_compat=check_datetimelike_compat,
                check_categorical=check_categorical,
                check_freq=check_freq,
                obj=f'{obj}.iloc[:, {i}] (column name="{col}")',
                rtol=rtol,
                atol=atol,
                check_index=False,
                check_flags=False,
            )


def assert_equal(left, right, **kwargs) -> None:
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    __tracebackhide__ = True

    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert left.freq == right.freq, (left.freq, right.freq)
    elif isinstance(left, Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        assert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        assert_numpy_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        assert_almost_equal(left, right)


def assert_sp_array_equal(left, right) -> None:
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    _check_isinstance(left, right, pd.arrays.SparseArray)

    assert_numpy_array_equal(left.sp_values, right.sp_values)

    # SparseIndex comparison
    assert isinstance(left.sp_index, SparseIndex)
    assert isinstance(right.sp_index, SparseIndex)

    left_index = left.sp_index
    right_index = right.sp_index

    if not left_index.equals(right_index):
        raise_assert_detail(
            "SparseArray.index", "index are not equal", left_index, right_index
        )
    else:
        # Just ensure a
        pass

    assert_attr_equal("fill_value", left, right)
    assert_attr_equal("dtype", left, right)
    assert_numpy_array_equal(left.to_dense(), right.to_dense())


def assert_contains_all(iterable, dic) -> None:
    for k in iterable:
        assert k in dic, f"Did not contain item: {repr(k)}"


def assert_copy(iter1, iter2, **eql_kwargs) -> None:
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    for elem1, elem2 in zip(iter1, iter2):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = (
            f"Expected object {repr(type(elem1))} and object {repr(type(elem2))} to be "
            "different objects, but they were the same object."
        )
        assert elem1 is not elem2, msg


def is_extension_array_dtype_and_needs_i8_conversion(
    left_dtype: DtypeObj, right_dtype: DtypeObj
) -> bool:
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    return isinstance(left_dtype, ExtensionDtype) and needs_i8_conversion(right_dtype)


def assert_indexing_slices_equivalent(ser: Series, l_slc: slice, i_slc: slice) -> None:
    """
    Check that ser.iloc[i_slc] matches ser.loc[l_slc] and, if applicable,
    ser[l_slc].
    """
    expected = ser.iloc[i_slc]

    assert_series_equal(ser.loc[l_slc], expected)

    if not is_integer_dtype(ser.index):
        # For integer indices, .loc and plain getitem are position-based.
        assert_series_equal(ser[l_slc], expected)


def assert_metadata_equivalent(
    left: DataFrame | Series, right: DataFrame | Series | None = None
) -> None:
    """
    Check that ._metadata attributes are equivalent.
    """
    for attr in left._metadata:
        val = getattr(left, attr, None)
        if right is None:
            assert val is None
        else:
            assert val == getattr(right, attr, None)
<|endoftext|>"""
Helpers for sharing tests between DataFrame/Series
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


def get_dtype(obj) -> DtypeObj:
    if isinstance(obj, DataFrame):
        # Note: we are assuming only one column
        return obj.dtypes.iat[0]
    else:
        return obj.dtype


def get_obj(df: DataFrame, klass):
    """
    For sharing tests using frame_or_series, either return the DataFrame
    unchanged or return it's first column as a Series.
    """
    if klass is DataFrame:
        return df
    return df._ixs(0, axis=1)
<|endoftext|>from __future__ import annotations

import gzip
import io
import pathlib
import tarfile
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)
import uuid
import zipfile

from pandas.compat import (
    get_bz2_file,
    get_lzma_file,
)
from pandas.compat._optional import import_optional_dependency

import pandas as pd
from pandas._testing.contexts import ensure_clean

if TYPE_CHECKING:
    from pandas._typing import (
        FilePath,
        ReadPickleBuffer,
    )

    from pandas import (
        DataFrame,
        Series,
    )

# ------------------------------------------------------------------
# File-IO


def round_trip_pickle(
    obj: Any, path: FilePath | ReadPickleBuffer | None = None
) -> DataFrame | Series:
    """
    Pickle an object and then read it again.

    Parameters
    ----------
    obj : any object
        The object to pickle and then re-read.
    path : str, path object or file-like object, default None
        The path where the pickled object is written and then read.

    Returns
    -------
    pandas object
        The original object that was pickled and then re-read.
    """
    _path = path
    if _path is None:
        _path = f"__{uuid.uuid4()}__.pickle"
    with ensure_clean(_path) as temp_path:
        pd.to_pickle(obj, temp_path)
        return pd.read_pickle(temp_path)


def round_trip_pathlib(writer, reader, path: str | None = None):
    """
    Write an object to file specified by a pathlib.Path and read it back

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
    Path = pathlib.Path
    if path is None:
        path = "___pathlib___"
    with ensure_clean(path) as path:
        writer(Path(path))  # type: ignore[arg-type]
        obj = reader(Path(path))  # type: ignore[arg-type]
    return obj


def round_trip_localpath(writer, reader, path: str | None = None):
    """
    Write an object to file specified by a py.path LocalPath and read it back.

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
    import pytest

    LocalPath = pytest.importorskip("py.path").local
    if path is None:
        path = "___localpath___"
    with ensure_clean(path) as path:
        writer(LocalPath(path))
        obj = reader(LocalPath(path))
    return obj


def write_to_compressed(compression, path, data, dest: str = "test") -> None:
    """
    Write data to a compressed file.

    Parameters
    ----------
    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd'}
        The compression type to use.
    path : str
        The file path to write the data.
    data : str
        The data to write.
    dest : str, default "test"
        The destination file (for ZIP only)

    Raises
    ------
    ValueError : An invalid compression value was passed in.
    """
    args: tuple[Any, ...] = (data,)
    mode = "wb"
    method = "write"
    compress_method: Callable

    if compression == "zip":
        compress_method = zipfile.ZipFile
        mode = "w"
        args = (dest, data)
        method = "writestr"
    elif compression == "tar":
        compress_method = tarfile.TarFile
        mode = "w"
        file = tarfile.TarInfo(name=dest)
        bytes = io.BytesIO(data)
        file.size = len(data)
        args = (file, bytes)
        method = "addfile"
    elif compression == "gzip":
        compress_method = gzip.GzipFile
    elif compression == "bz2":
        compress_method = get_bz2_file()
    elif compression == "zstd":
        compress_method = import_optional_dependency("zstandard").open
    elif compression == "xz":
        compress_method = get_lzma_file()
    else:
        raise ValueError(f"Unrecognized compression type: {compression}")

    with compress_method(path, mode=mode) as f:
        getattr(f, method)(*args)
<|endoftext|>from __future__ import annotations

from decimal import Decimal
import operator
import os
from sys import byteorder
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype
from pandas._config.localization import (
    can_set_locale,
    get_locales,
    set_locale,
)

from pandas.compat import pa_version_under10p1

import pandas as pd
from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
)
from pandas._testing._io import (
    round_trip_localpath,
    round_trip_pathlib,
    round_trip_pickle,
    write_to_compressed,
)
from pandas._testing._warnings import (
    assert_produces_warning,
    maybe_produces_warning,
)
from pandas._testing.asserters import (
    assert_almost_equal,
    assert_attr_equal,
    assert_categorical_equal,
    assert_class_equal,
    assert_contains_all,
    assert_copy,
    assert_datetime_array_equal,
    assert_dict_equal,
    assert_equal,
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_indexing_slices_equivalent,
    assert_interval_array_equal,
    assert_is_sorted,
    assert_is_valid_plot_return_object,
    assert_metadata_equivalent,
    assert_numpy_array_equal,
    assert_period_array_equal,
    assert_series_equal,
    assert_sp_array_equal,
    assert_timedelta_array_equal,
    raise_assert_detail,
)
from pandas._testing.compat import (
    get_dtype,
    get_obj,
)
from pandas._testing.contexts import (
    assert_cow_warning,
    decompress_file,
    ensure_clean,
    raises_chained_assignment_error,
    set_timezone,
    use_numexpr,
    with_csv_dialect,
)
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    NumpyExtensionArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from pandas._typing import (
        Dtype,
        NpDtype,
    )


UNSIGNED_INT_NUMPY_DTYPES: list[NpDtype] = ["uint8", "uint16", "uint32", "uint64"]
UNSIGNED_INT_EA_DTYPES: list[Dtype] = ["UInt8", "UInt16", "UInt32", "UInt64"]
SIGNED_INT_NUMPY_DTYPES: list[NpDtype] = [int, "int8", "int16", "int32", "int64"]
SIGNED_INT_EA_DTYPES: list[Dtype] = ["Int8", "Int16", "Int32", "Int64"]
ALL_INT_NUMPY_DTYPES = UNSIGNED_INT_NUMPY_DTYPES + SIGNED_INT_NUMPY_DTYPES
ALL_INT_EA_DTYPES = UNSIGNED_INT_EA_DTYPES + SIGNED_INT_EA_DTYPES
ALL_INT_DTYPES: list[Dtype] = [*ALL_INT_NUMPY_DTYPES, *ALL_INT_EA_DTYPES]

FLOAT_NUMPY_DTYPES: list[NpDtype] = [float, "float32", "float64"]
FLOAT_EA_DTYPES: list[Dtype] = ["Float32", "Float64"]
ALL_FLOAT_DTYPES: list[Dtype] = [*FLOAT_NUMPY_DTYPES, *FLOAT_EA_DTYPES]

COMPLEX_DTYPES: list[Dtype] = [complex, "complex64", "complex128"]
if using_string_dtype():
    STRING_DTYPES: list[Dtype] = ["U"]
else:
    STRING_DTYPES: list[Dtype] = [str, "str", "U"]  # type: ignore[no-redef]
COMPLEX_FLOAT_DTYPES: list[Dtype] = [*COMPLEX_DTYPES, *FLOAT_NUMPY_DTYPES]

DATETIME64_DTYPES: list[Dtype] = ["datetime64[ns]", "M8[ns]"]
TIMEDELTA64_DTYPES: list[Dtype] = ["timedelta64[ns]", "m8[ns]"]

BOOL_DTYPES: list[Dtype] = [bool, "bool"]
BYTES_DTYPES: list[Dtype] = [bytes, "bytes"]
OBJECT_DTYPES: list[Dtype] = [object, "object"]

ALL_REAL_NUMPY_DTYPES = FLOAT_NUMPY_DTYPES + ALL_INT_NUMPY_DTYPES
ALL_REAL_EXTENSION_DTYPES = FLOAT_EA_DTYPES + ALL_INT_EA_DTYPES
ALL_REAL_DTYPES: list[Dtype] = [*ALL_REAL_NUMPY_DTYPES, *ALL_REAL_EXTENSION_DTYPES]
ALL_NUMERIC_DTYPES: list[Dtype] = [*ALL_REAL_DTYPES, *COMPLEX_DTYPES]

ALL_NUMPY_DTYPES = (
    ALL_REAL_NUMPY_DTYPES
    + COMPLEX_DTYPES
    + STRING_DTYPES
    + DATETIME64_DTYPES
    + TIMEDELTA64_DTYPES
    + BOOL_DTYPES
    + OBJECT_DTYPES
    + BYTES_DTYPES
)

NARROW_NP_DTYPES = [
    np.float16,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]

PYTHON_DATA_TYPES = [
    str,
    int,
    float,
    complex,
    list,
    tuple,
    range,
    dict,
    set,
    frozenset,
    bool,
    bytes,
    bytearray,
    memoryview,
]

ENDIAN = {"little": "<", "big": ">"}[byteorder]

NULL_OBJECTS = [None, np.nan, pd.NaT, float("nan"), pd.NA, Decimal("NaN")]
NP_NAT_OBJECTS = [
    cls("NaT", unit)
    for cls in [np.datetime64, np.timedelta64]
    for unit in [
        "Y",
        "M",
        "W",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "ps",
        "fs",
        "as",
    ]
]

if not pa_version_under10p1:
    import pyarrow as pa

    UNSIGNED_INT_PYARROW_DTYPES = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
    SIGNED_INT_PYARROW_DTYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
    ALL_INT_PYARROW_DTYPES = UNSIGNED_INT_PYARROW_DTYPES + SIGNED_INT_PYARROW_DTYPES
    ALL_INT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in ALL_INT_PYARROW_DTYPES
    ]

    # pa.float16 doesn't seem supported
    # https://github.com/apache/arrow/blob/master/python/pyarrow/src/arrow/python/helpers.cc#L86
    FLOAT_PYARROW_DTYPES = [pa.float32(), pa.float64()]
    FLOAT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in FLOAT_PYARROW_DTYPES
    ]
    DECIMAL_PYARROW_DTYPES = [pa.decimal128(7, 3)]
    STRING_PYARROW_DTYPES = [pa.string()]
    BINARY_PYARROW_DTYPES = [pa.binary()]

    TIME_PYARROW_DTYPES = [
        pa.time32("s"),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
    ]
    DATE_PYARROW_DTYPES = [pa.date32(), pa.date64()]
    DATETIME_PYARROW_DTYPES = [
        pa.timestamp(unit=unit, tz=tz)
        for unit in ["s", "ms", "us", "ns"]
        for tz in [None, "UTC", "US/Pacific", "US/Eastern"]
    ]
    TIMEDELTA_PYARROW_DTYPES = [pa.duration(unit) for unit in ["s", "ms", "us", "ns"]]

    BOOL_PYARROW_DTYPES = [pa.bool_()]

    # TODO: Add container like pyarrow types:
    #  https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
    ALL_PYARROW_DTYPES = (
        ALL_INT_PYARROW_DTYPES
        + FLOAT_PYARROW_DTYPES
        + DECIMAL_PYARROW_DTYPES
        + STRING_PYARROW_DTYPES
        + BINARY_PYARROW_DTYPES
        + TIME_PYARROW_DTYPES
        + DATE_PYARROW_DTYPES
        + DATETIME_PYARROW_DTYPES
        + TIMEDELTA_PYARROW_DTYPES
        + BOOL_PYARROW_DTYPES
    )
    ALL_REAL_PYARROW_DTYPES_STR_REPR = (
        ALL_INT_PYARROW_DTYPES_STR_REPR + FLOAT_PYARROW_DTYPES_STR_REPR
    )
else:
    FLOAT_PYARROW_DTYPES_STR_REPR = []
    ALL_INT_PYARROW_DTYPES_STR_REPR = []
    ALL_PYARROW_DTYPES = []
    ALL_REAL_PYARROW_DTYPES_STR_REPR = []

ALL_REAL_NULLABLE_DTYPES = (
    FLOAT_NUMPY_DTYPES + ALL_REAL_EXTENSION_DTYPES + ALL_REAL_PYARROW_DTYPES_STR_REPR
)

arithmetic_dunder_methods = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]

comparison_dunder_methods = ["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"]


# -----------------------------------------------------------------------------
# Comparators


def box_expected(expected, box_cls, transpose: bool = True):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
    if box_cls is pd.array:
        if isinstance(expected, RangeIndex):
            # pd.array would return an IntegerArray
            expected = NumpyExtensionArray(np.asarray(expected._values))
        else:
            expected = pd.array(expected, copy=False)
    elif box_cls is Index:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Dtype inference", category=FutureWarning)
            expected = Index(expected)
    elif box_cls is Series:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Dtype inference", category=FutureWarning)
            expected = Series(expected)
    elif box_cls is DataFrame:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Dtype inference", category=FutureWarning)
            expected = Series(expected).to_frame()
        if transpose:
            # for vector operations, we need a DataFrame to be a single-row,
            #  not a single-column, in order to operate against non-DataFrame
            #  vectors of the same length. But convert to two rows to avoid
            #  single-row special cases in datetime arithmetic
            expected = expected.T
            expected = pd.concat([expected] * 2, ignore_index=True)
    elif box_cls is np.ndarray or box_cls is np.array:
        expected = np.array(expected)
    elif box_cls is to_array:
        expected = to_array(expected)
    else:
        raise NotImplementedError(box_cls)
    return expected


def to_array(obj):
    """
    Similar to pd.array, but does not cast numpy dtypes to nullable dtypes.
    """
    # temporary implementation until we get pd.array in place
    dtype = getattr(obj, "dtype", None)

    if dtype is None:
        return np.asarray(obj)

    return extract_array(obj, extract_numpy=True)


class SubclassedSeries(Series):
    _metadata = ["testattr", "name"]

    @property
    def _constructor(self):
        # For testing, those properties return a generic callable, and not
        # the actual class. In this case that is equivalent, but it is to
        # ensure we don't rely on the property returning a class
        # See https://github.com/pandas-dev/pandas/pull/46018 and
        # https://github.com/pandas-dev/pandas/issues/32638 and linked issues
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)

    @property
    def _constructor_expanddim(self):
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)


class SubclassedDataFrame(DataFrame):
    _metadata = ["testattr"]

    @property
    def _constructor(self):
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)

    @property
    def _constructor_sliced(self):
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)


def convert_rows_list_to_csv_str(rows_list: list[str]) -> str:
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
    sep = os.linesep
    return sep.join(rows_list) + sep


def external_error_raised(expected_exception: type[Exception]) -> ContextManager:
    """
    Helper function to mark pytest.raises that have an external error message.

    Parameters
    ----------
    expected_exception : Exception
        Expected error to raise.

    Returns
    -------
    Callable
        Regular `pytest.raises` function with `match` equal to `None`.
    """
    import pytest

    return pytest.raises(expected_exception, match=None)


cython_table = pd.core.common._cython_table.items()


def get_cython_table_params(ndframe, func_names_and_expected):
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of a NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
    results = []
    for func_name, expected in func_names_and_expected:
        results.append((ndframe, func_name, expected))
        results += [
            (ndframe, func, expected)
            for func, name in cython_table
            if name == func_name
        ]
    return results


def get_op_from_name(op_name: str) -> Callable:
    """
    The operator function for a given op name.

    Parameters
    ----------
    op_name : str
        The op name, in form of "add" or "__add__".

    Returns
    -------
    function
        A function performing the operation.
    """
    short_opname = op_name.strip("_")
    try:
        op = getattr(operator, short_opname)
    except AttributeError:
        # Assume it is the reverse operator
        rop = getattr(operator, short_opname[1:])
        op = lambda x, y: rop(y, x)

    return op


# -----------------------------------------------------------------------------
# Indexing test helpers


def getitem(x):
    return x


def setitem(x):
    return x


def loc(x):
    return x.loc


def iloc(x):
    return x.iloc


def at(x):
    return x.at


def iat(x):
    return x.iat


# -----------------------------------------------------------------------------

_UNITS = ["s", "ms", "us", "ns"]


def get_finest_unit(left: str, right: str):
    """
    Find the higher of two datetime64 units.
    """
    if _UNITS.index(left) >= _UNITS.index(right):
        return left
    return right


def shares_memory(left, right) -> bool:
    """
    Pandas-compat for np.shares_memory.
    """
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.shares_memory(left, right)
    elif isinstance(left, np.ndarray):
        # Call with reversed args to get to unpacking logic below.
        return shares_memory(right, left)

    if isinstance(left, RangeIndex):
        return False
    if isinstance(left, MultiIndex):
        return shares_memory(left._codes, right)
    if isinstance(left, (Index, Series)):
        if isinstance(right, (Index, Series)):
            return shares_memory(left._values, right._values)
        return shares_memory(left._values, right)

    if isinstance(left, NDArrayBackedExtensionArray):
        return shares_memory(left._ndarray, right)
    if isinstance(left, pd.core.arrays.SparseArray):
        return shares_memory(left.sp_values, right)
    if isinstance(left, pd.core.arrays.IntervalArray):
        return shares_memory(left._left, right) or shares_memory(left._right, right)

    if isinstance(left, ArrowExtensionArray):
        if isinstance(right, ArrowExtensionArray):
            # https://github.com/pandas-dev/pandas/pull/43930#discussion_r736862669
            left_pa_data = left._pa_array
            right_pa_data = right._pa_array
            left_buf1 = left_pa_data.chunk(0).buffers()[1]
            right_buf1 = right_pa_data.chunk(0).buffers()[1]
            return left_buf1.address == right_buf1.address
        else:
            # if we have one one ArrowExtensionArray and one other array, assume
            # they can only share memory if they share the same numpy buffer
            return np.shares_memory(left, right)

    if isinstance(left, BaseMaskedArray) and isinstance(right, BaseMaskedArray):
        # By convention, we'll say these share memory if they share *either*
        #  the _data or the _mask
        return np.shares_memory(left._data, right._data) or np.shares_memory(
            left._mask, right._mask
        )

    if isinstance(left, DataFrame) and len(left._mgr.arrays) == 1:
        arr = left._mgr.arrays[0]
        return shares_memory(arr, right)

    raise NotImplementedError(type(left), type(right))


__all__ = [
    "ALL_INT_EA_DTYPES",
    "ALL_INT_NUMPY_DTYPES",
    "ALL_NUMPY_DTYPES",
    "ALL_REAL_NUMPY_DTYPES",
    "assert_almost_equal",
    "assert_attr_equal",
    "assert_categorical_equal",
    "assert_class_equal",
    "assert_contains_all",
    "assert_copy",
    "assert_datetime_array_equal",
    "assert_dict_equal",
    "assert_equal",
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_indexing_slices_equivalent",
    "assert_interval_array_equal",
    "assert_is_sorted",
    "assert_is_valid_plot_return_object",
    "assert_metadata_equivalent",
    "assert_numpy_array_equal",
    "assert_period_array_equal",
    "assert_produces_warning",
    "assert_series_equal",
    "assert_sp_array_equal",
    "assert_timedelta_array_equal",
    "assert_cow_warning",
    "at",
    "BOOL_DTYPES",
    "box_expected",
    "BYTES_DTYPES",
    "can_set_locale",
    "COMPLEX_DTYPES",
    "convert_rows_list_to_csv_str",
    "DATETIME64_DTYPES",
    "decompress_file",
    "ENDIAN",
    "ensure_clean",
    "external_error_raised",
    "FLOAT_EA_DTYPES",
    "FLOAT_NUMPY_DTYPES",
    "get_cython_table_params",
    "get_dtype",
    "getitem",
    "get_locales",
    "get_finest_unit",
    "get_obj",
    "get_op_from_name",
    "iat",
    "iloc",
    "loc",
    "maybe_produces_warning",
    "NARROW_NP_DTYPES",
    "NP_NAT_OBJECTS",
    "NULL_OBJECTS",
    "OBJECT_DTYPES",
    "raise_assert_detail",
    "raises_chained_assignment_error",
    "round_trip_localpath",
    "round_trip_pathlib",
    "round_trip_pickle",
    "setitem",
    "set_locale",
    "set_timezone",
    "shares_memory",
    "SIGNED_INT_EA_DTYPES",
    "SIGNED_INT_NUMPY_DTYPES",
    "STRING_DTYPES",
    "SubclassedDataFrame",
    "SubclassedSeries",
    "TIMEDELTA64_DTYPES",
    "to_array",
    "UNSIGNED_INT_EA_DTYPES",
    "UNSIGNED_INT_NUMPY_DTYPES",
    "use_numexpr",
    "with_csv_dialect",
    "write_to_compressed",
]
<|endoftext|>from __future__ import annotations

from contextlib import (
    contextmanager,
    nullcontext,
)
import inspect
import re
import sys
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
import warnings

from pandas.compat import PY311

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Sequence,
    )


@contextmanager
def assert_produces_warning(
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None = Warning,
    filter_level: Literal[
        "error", "ignore", "always", "default", "module", "once"
    ] = "always",
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: str | None = None,
) -> Generator[list[warnings.WarningMessage], None, None]:
    """
    Context manager for running code expected to either raise a specific warning,
    multiple specific warnings, or not raise any warnings. Verifies that the code
    raises the expected warning(s), and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Parameters
    ----------
    expected_warning : {Warning, False, tuple[Warning, ...], None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To raise multiple types of exceptions,
        pass them as a tuple. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str or None, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:

        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated

    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.
    raise_on_extra_warnings : bool, default True
        Whether extra warnings not of the type `expected_warning` should
        cause the test to fail.
    match : str, optional
        Match warning message.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    ...
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    ...
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'.

    ..warn:: This is *not* thread-safe.
    """
    __tracebackhide__ = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        try:
            yield w
        finally:
            if expected_warning:
                expected_warning = cast(type[Warning], expected_warning)
                _assert_caught_expected_warning(
                    caught_warnings=w,
                    expected_warning=expected_warning,
                    match=match,
                    check_stacklevel=check_stacklevel,
                )
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(
                    caught_warnings=w,
                    expected_warning=expected_warning,
                )


def maybe_produces_warning(warning: type[Warning], condition: bool, **kwargs):
    """
    Return a context manager that possibly checks a warning based on the condition
    """
    if condition:
        return assert_produces_warning(warning, **kwargs)
    else:
        return nullcontext()


def _assert_caught_expected_warning(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning],
    match: str | None,
    check_stacklevel: bool,
) -> None:
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning = False
    matched_message = False
    unmatched_messages = []

    for actual_warning in caught_warnings:
        if issubclass(actual_warning.category, expected_warning):
            saw_warning = True

            if check_stacklevel:
                _assert_raised_with_correct_stacklevel(actual_warning)

            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True
                else:
                    unmatched_messages.append(actual_warning.message)

    if not saw_warning:
        raise AssertionError(
            f"Did not see expected warning of class "
            f"{repr(expected_warning.__name__)}"
        )

    if match and not matched_message:
        raise AssertionError(
            f"Did not see warning {repr(expected_warning.__name__)} "
            f"matching '{match}'. The emitted warning messages are "
            f"{unmatched_messages}"
        )


def _assert_caught_no_extra_warnings(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None,
) -> None:
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings = []

    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            # GH#38630 pytest.filterwarnings does not suppress these.
            if actual_warning.category == ResourceWarning:
                # GH 44732: Don't make the CI flaky by filtering SSL-related
                # ResourceWarning from dependencies
                if "unclosed <ssl.SSLSocket" in str(actual_warning.message):
                    continue
                # GH 44844: Matplotlib leaves font files open during the entire process
                # upon import. Don't make CI flaky if ResourceWarning raised
                # due to these open files.
                if any("matplotlib" in mod for mod in sys.modules):
                    continue
            if PY311 and actual_warning.category == EncodingWarning:
                # EncodingWarnings are checked in the CI
                # pyproject.toml errors on EncodingWarnings in pandas
                # Ignore EncodingWarnings from other libraries
                continue
            extra_warnings.append(
                (
                    actual_warning.category.__name__,
                    actual_warning.message,
                    actual_warning.filename,
                    actual_warning.lineno,
                )
            )

    if extra_warnings:
        raise AssertionError(f"Caused unexpected warning(s): {repr(extra_warnings)}")


def _is_unexpected_warning(
    actual_warning: warnings.WarningMessage,
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None,
) -> bool:
    """Check if the actual warning issued is unexpected."""
    if actual_warning and not expected_warning:
        return True
    expected_warning = cast(type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning))


def _assert_raised_with_correct_stacklevel(
    actual_warning: warnings.WarningMessage,
) -> None:
    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back  # type: ignore[union-attr]
    try:
        caller_filename = inspect.getfile(frame)  # type: ignore[arg-type]
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    msg = (
        "Warning not set with correct stacklevel. "
        f"File where warning is raised: {actual_warning.filename} != "
        f"{caller_filename}. Warning message: {actual_warning.message}"
    )
    assert actual_warning.filename == caller_filename, msg
<|endoftext|>"""
Hypothesis data generator helpers.
"""
from datetime import datetime

from hypothesis import strategies as st
from hypothesis.extra.dateutil import timezones as dateutil_timezones
from hypothesis.extra.pytz import timezones as pytz_timezones

from pandas.compat import is_platform_windows

import pandas as pd

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)

OPTIONAL_INTS = st.lists(st.one_of(st.integers(), st.none()), max_size=10, min_size=3)

OPTIONAL_FLOATS = st.lists(st.one_of(st.floats(), st.none()), max_size=10, min_size=3)

OPTIONAL_TEXT = st.lists(st.one_of(st.none(), st.text()), max_size=10, min_size=3)

OPTIONAL_DICTS = st.lists(
    st.one_of(st.none(), st.dictionaries(st.text(), st.integers())),
    max_size=10,
    min_size=3,
)

OPTIONAL_LISTS = st.lists(
    st.one_of(st.none(), st.lists(st.text(), max_size=10, min_size=3)),
    max_size=10,
    min_size=3,
)

OPTIONAL_ONE_OF_ALL = st.one_of(
    OPTIONAL_DICTS, OPTIONAL_FLOATS, OPTIONAL_INTS, OPTIONAL_LISTS, OPTIONAL_TEXT
)

if is_platform_windows():
    DATETIME_NO_TZ = st.datetimes(min_value=datetime(1900, 1, 1))
else:
    DATETIME_NO_TZ = st.datetimes()

DATETIME_JAN_1_1900_OPTIONAL_TZ = st.datetimes(
    min_value=pd.Timestamp(
        1900, 1, 1
    ).to_pydatetime(),  # pyright: ignore[reportGeneralTypeIssues]
    max_value=pd.Timestamp(
        1900, 1, 1
    ).to_pydatetime(),  # pyright: ignore[reportGeneralTypeIssues]
    timezones=st.one_of(st.none(), dateutil_timezones(), pytz_timezones()),
)

DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ = st.datetimes(
    min_value=pd.Timestamp.min.to_pydatetime(warn=False),
    max_value=pd.Timestamp.max.to_pydatetime(warn=False),
)

INT_NEG_999_TO_POS_999 = st.integers(-999, 999)

# The strategy for each type is registered in conftest.py, as they don't carry
# enough runtime information (e.g. type hints) to infer how to build them.
YQM_OFFSET = st.one_of(
    *map(
        st.from_type,
        [
            MonthBegin,
            MonthEnd,
            BMonthBegin,
            BMonthEnd,
            QuarterBegin,
            QuarterEnd,
            BQuarterBegin,
            BQuarterEnd,
            YearBegin,
            YearEnd,
            BYearBegin,
            BYearEnd,
        ],
    )
)
<|endoftext|>from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
)
import uuid

from pandas._config import using_copy_on_write

from pandas.compat import (
    PYPY,
    WARNING_CHECK_DISABLED,
)
from pandas.errors import ChainedAssignmentError

from pandas import set_option

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas._typing import (
        BaseBuffer,
        CompressionOptions,
        FilePath,
    )


@contextmanager
def decompress_file(
    path: FilePath | BaseBuffer, compression: CompressionOptions
) -> Generator[IO[bytes], None, None]:
    """
    Open a compressed file and return a file object.

    Parameters
    ----------
    path : str
        The path where the file is read from.

    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd', None}
        Name of the decompression to use

    Returns
    -------
    file object
    """
    with get_handle(path, "rb", compression=compression, is_text=False) as handle:
        yield handle.handle


@contextmanager
def set_timezone(tz: str) -> Generator[None, None, None]:
    """
    Context manager for temporarily setting a timezone.

    Parameters
    ----------
    tz : str
        A string representing a valid timezone.

    Examples
    --------
    >>> from datetime import datetime
    >>> from dateutil.tz import tzlocal
    >>> tzlocal().tzname(datetime(2021, 1, 1))  # doctest: +SKIP
    'IST'

    >>> with set_timezone('US/Eastern'):
    ...     tzlocal().tzname(datetime(2021, 1, 1))
    ...
    'EST'
    """
    import time

    def setTZ(tz) -> None:
        if hasattr(time, "tzset"):
            if tz is None:
                try:
                    del os.environ["TZ"]
                except KeyError:
                    pass
            else:
                os.environ["TZ"] = tz
                time.tzset()

    orig_tz = os.environ.get("TZ")
    setTZ(tz)
    try:
        yield
    finally:
        setTZ(orig_tz)


@contextmanager
def ensure_clean(
    filename=None, return_filelike: bool = False, **kwargs: Any
) -> Generator[Any, None, None]:
    """
    Gets a temporary path and agrees to remove on close.

    This implementation does not use tempfile.mkstemp to avoid having a file handle.
    If the code using the returned path wants to delete the file itself, windows
    requires that no program has a file handle to it.

    Parameters
    ----------
    filename : str (optional)
        suffix of the created file.
    return_filelike : bool (default False)
        if True, returns a file-like which is *always* cleaned. Necessary for
        savefig and other functions which want to append extensions.
    **kwargs
        Additional keywords are passed to open().

    """
    folder = Path(tempfile.gettempdir())

    if filename is None:
        filename = ""
    filename = str(uuid.uuid4()) + filename
    path = folder / filename

    path.touch()

    handle_or_str: str | IO = str(path)
    encoding = kwargs.pop("encoding", None)
    if return_filelike:
        kwargs.setdefault("mode", "w+b")
        if encoding is None and "b" not in kwargs["mode"]:
            encoding = "utf-8"
        handle_or_str = open(path, encoding=encoding, **kwargs)

    try:
        yield handle_or_str
    finally:
        if not isinstance(handle_or_str, str):
            handle_or_str.close()
        if path.is_file():
            path.unlink()


@contextmanager
def with_csv_dialect(name: str, **kwargs) -> Generator[None, None, None]:
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
    import csv

    _BUILTIN_DIALECTS = {"excel", "excel-tab", "unix"}

    if name in _BUILTIN_DIALECTS:
        raise ValueError("Cannot override builtin dialect.")

    csv.register_dialect(name, **kwargs)
    try:
        yield
    finally:
        csv.unregister_dialect(name)


@contextmanager
def use_numexpr(use, min_elements=None) -> Generator[None, None, None]:
    from pandas.core.computation import expressions as expr

    if min_elements is None:
        min_elements = expr._MIN_ELEMENTS

    olduse = expr.USE_NUMEXPR
    oldmin = expr._MIN_ELEMENTS
    set_option("compute.use_numexpr", use)
    expr._MIN_ELEMENTS = min_elements
    try:
        yield
    finally:
        expr._MIN_ELEMENTS = oldmin
        set_option("compute.use_numexpr", olduse)


def raises_chained_assignment_error(warn=True, extra_warnings=(), extra_match=()):
    from pandas._testing import assert_produces_warning

    if not warn:
        from contextlib import nullcontext

        return nullcontext()

    if (PYPY or WARNING_CHECK_DISABLED) and not extra_warnings:
        from contextlib import nullcontext

        return nullcontext()
    elif (PYPY or WARNING_CHECK_DISABLED) and extra_warnings:
        return assert_produces_warning(
            extra_warnings,
            match="|".join(extra_match),
        )
    else:
        if using_copy_on_write():
            warning = ChainedAssignmentError
            match = (
                "A value is trying to be set on a copy of a DataFrame or Series "
                "through chained assignment"
            )
        else:
            warning = FutureWarning  # type: ignore[assignment]
            # TODO update match
            match = "ChainedAssignmentError"
        if extra_warnings:
            warning = (warning, *extra_warnings)  # type: ignore[assignment]
        return assert_produces_warning(
            warning,
            match="|".join((match, *extra_match)),
        )


def assert_cow_warning(warn=True, match=None, **kwargs):
    """
    Assert that a warning is raised in the CoW warning mode.

    Parameters
    ----------
    warn : bool, default True
        By default, check that a warning is raised. Can be turned off by passing False.
    match : str
        The warning message to match against, if different from the default.
    kwargs
        Passed through to assert_produces_warning
    """
    from pandas._testing import assert_produces_warning

    if not warn or WARNING_CHECK_DISABLED:
        from contextlib import nullcontext

        return nullcontext()

    if not match:
        match = "Setting a value on a view"

    return assert_produces_warning(FutureWarning, match=match, **kwargs)
<|endoftext|>"""
All of pandas' ExtensionArrays.

See :ref:`extending.extension-types` for more.
"""
from pandas.core.arrays import (
    ArrowExtensionArray,
    ArrowStringArray,
    BooleanArray,
    Categorical,
    DatetimeArray,
    FloatingArray,
    IntegerArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    SparseArray,
    StringArray,
    TimedeltaArray,
)

__all__ = [
    "ArrowExtensionArray",
    "ArrowStringArray",
    "BooleanArray",
    "Categorical",
    "DatetimeArray",
    "FloatingArray",
    "IntegerArray",
    "IntervalArray",
    "NumpyExtensionArray",
    "PeriodArray",
    "SparseArray",
    "StringArray",
    "TimedeltaArray",
]


def __getattr__(name: str) -> type[NumpyExtensionArray]:
    if name == "PandasArray":
        # GH#53694
        import warnings

        from pandas.util._exceptions import find_stack_level

        warnings.warn(
            "PandasArray has been renamed NumpyExtensionArray. Use that "
            "instead. This alias will be removed in a future version.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return NumpyExtensionArray
    raise AttributeError(f"module 'pandas.arrays' has no attribute '{name}'")
<|endoftext|>"""
config for datetime formatting
"""
from __future__ import annotations

from pandas._config import config as cf

pc_date_dayfirst_doc = """
: boolean
    When True, prints and parses dates with the day first, eg 20/01/2005
"""

pc_date_yearfirst_doc = """
: boolean
    When True, prints and parses dates with the year first, eg 2005/01/20
"""

with cf.config_prefix("display"):
    # Needed upstream of `_libs` because these are used in tslibs.parsing
    cf.register_option(
        "date_dayfirst", False, pc_date_dayfirst_doc, validator=cf.is_bool
    )
    cf.register_option(
        "date_yearfirst", False, pc_date_yearfirst_doc, validator=cf.is_bool
    )
<|endoftext|>"""
Unopinionated display configuration.
"""

from __future__ import annotations

import locale
import sys

from pandas._config import config as cf

# -----------------------------------------------------------------------------
# Global formatting options
_initial_defencoding: str | None = None


def detect_console_encoding() -> str:
    """
    Try to find the most capable encoding supported by the console.
    slightly modified from the way IPython handles the same issue.
    """
    global _initial_defencoding

    encoding = None
    try:
        encoding = sys.stdout.encoding or sys.stdin.encoding
    except (AttributeError, OSError):
        pass

    # try again for something better
    if not encoding or "ascii" in encoding.lower():
        try:
            encoding = locale.getpreferredencoding()
        except locale.Error:
            # can be raised by locale.setlocale(), which is
            #  called by getpreferredencoding
            #  (on some systems, see stdlib locale docs)
            pass

    # when all else fails. this will usually be "ascii"
    if not encoding or "ascii" in encoding.lower():
        encoding = sys.getdefaultencoding()

    # GH#3360, save the reported defencoding at import time
    # MPL backends may change it. Make available for debugging.
    if not _initial_defencoding:
        _initial_defencoding = sys.getdefaultencoding()

    return encoding


pc_encoding_doc = """
: str/unicode
    Defaults to the detected encoding of the console.
    Specifies the encoding to be used for strings returned by to_string,
    these are generally strings meant to be displayed on the console.
"""

with cf.config_prefix("display"):
    cf.register_option(
        "encoding", detect_console_encoding(), pc_encoding_doc, validator=cf.is_text
    )
<|endoftext|>"""
The config module holds package-wide configurables and provides
a uniform API for working with them.

Overview
========

This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options can be registered at init-time (via core.config_init)
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be deprecated, in which case referencing them
  should produce a warning.
- deprecated options can optionally be rerouted to a replacement
  so that accessing a deprecated option reroutes to a differently
  named option.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- all options in a certain sub - namespace can be reset at once.
- the user can set / get / reset or ask for the description of an option.
- a developer can register and mark an option as deprecated.
- you can register a callback to be invoked when the option value
  is set or reset. Changing the stored value is considered misuse, but
  is not verboten.

Implementation
==============

- Data is stored using nested dictionaries, and should be accessed
  through the provided API.

- "Registered options" and "Deprecated options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

- the config_init module is imported by the package's __init__.py file.
  placing any register_option() calls there will ensure those options
  are available as soon as pandas is loaded. If you use register_option
  in a module, it will only be available after that module is imported,
  which you should be aware of.

- `config_prefix` is a context_manager (for use with the `with` keyword)
  which can save developers some typing, see the docstring.

"""

from __future__ import annotations

from contextlib import (
    ContextDecorator,
    contextmanager,
)
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NamedTuple,
    cast,
)
import warnings

from pandas._typing import (
    F,
    T,
)
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
    )


class DeprecatedOption(NamedTuple):
    key: str
    msg: str | None
    rkey: str | None
    removal_ver: str | None


class RegisteredOption(NamedTuple):
    key: str
    defval: object
    doc: str
    validator: Callable[[object], Any] | None
    cb: Callable[[str], Any] | None


# holds deprecated option metadata
_deprecated_options: dict[str, DeprecatedOption] = {}

# holds registered option metadata
_registered_options: dict[str, RegisteredOption] = {}

# holds the current values for registered options
_global_config: dict[str, Any] = {}

# keys which have a special meaning
_reserved_keys: list[str] = ["all"]


class OptionError(AttributeError, KeyError):
    """
    Exception raised for pandas.options.

    Backwards compatible with KeyError checks.

    Examples
    --------
    >>> pd.options.context
    Traceback (most recent call last):
    OptionError: No such option
    """


#
# User API


def _get_single_key(pat: str, silent: bool) -> str:
    keys = _select_options(pat)
    if len(keys) == 0:
        if not silent:
            _warn_if_deprecated(pat)
        raise OptionError(f"No such keys(s): {repr(pat)}")
    if len(keys) > 1:
        raise OptionError("Pattern matched multiple keys")
    key = keys[0]

    if not silent:
        _warn_if_deprecated(key)

    key = _translate_key(key)

    return key


def _get_option(pat: str, silent: bool = False) -> Any:
    key = _get_single_key(pat, silent)

    # walk the nested dict
    root, k = _get_root(key)
    return root[k]


def _set_option(*args, **kwargs) -> None:
    # must at least 1 arg deal with constraints later
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    # default to false
    silent = kwargs.pop("silent", False)

    if kwargs:
        kwarg = next(iter(kwargs.keys()))
        raise TypeError(f'_set_option() got an unexpected keyword argument "{kwarg}"')

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k, silent)

        o = _get_registered_option(key)
        if o and o.validator:
            o.validator(v)

        # walk the nested dict
        root, k_root = _get_root(key)
        root[k_root] = v

        if o.cb:
            if silent:
                with warnings.catch_warnings(record=True):
                    o.cb(key)
            else:
                o.cb(key)


def _describe_option(pat: str = "", _print_desc: bool = True) -> str | None:
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    s = "\n".join([_build_option_description(k) for k in keys])

    if _print_desc:
        print(s)
        return None
    return s


def _reset_option(pat: str, silent: bool = False) -> None:
    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default value'
        )

    for k in keys:
        _set_option(k, _registered_options[k].defval, silent=silent)


def get_default_val(pat: str):
    key = _get_single_key(pat, silent=True)
    return _get_registered_option(key).defval


class DictWrapper:
    """provide attribute-style access to a nested dict"""

    d: dict[str, Any]

    def __init__(self, d: dict[str, Any], prefix: str = "") -> None:
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        # you can't set new keys
        # can you can't overwrite subtrees
        if key in self.d and not isinstance(self.d[key], dict):
            _set_option(prefix, val)
        else:
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key: str):
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError as err:
            raise OptionError("No such option") from err
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return _get_option(prefix)

    def __dir__(self) -> list[str]:
        return list(self.d.keys())


# For user convenience,  we'd like to have the available options described
# in the docstring. For dev convenience we'd like to generate the docstrings
# dynamically instead of maintaining them by hand. To this, we use the
# class below which wraps functions inside a callable, and converts
# __doc__ into a property function. The doctsrings below are templates
# using the py2.6+ advanced formatting syntax to plug in a concise list
# of options, and option descriptions.


class CallableDynamicDoc(Generic[T]):
    def __init__(self, func: Callable[..., T], doc_tmpl: str) -> None:
        self.__doc_tmpl__ = doc_tmpl
        self.__func__ = func

    def __call__(self, *args, **kwds) -> T:
        return self.__func__(*args, **kwds)

    # error: Signature of "__doc__" incompatible with supertype "object"
    @property
    def __doc__(self) -> str:  # type: ignore[override]
        opts_desc = _describe_option("all", _print_desc=False)
        opts_list = pp_options_list(list(_registered_options.keys()))
        return self.__doc_tmpl__.format(opts_desc=opts_desc, opts_list=opts_list)


_get_option_tmpl = """
get_option(pat)

Retrieves the value of the specified option.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.

Returns
-------
result : the value of the option

Raises
------
OptionError : if no such option exists

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.get_option('display.max_columns')  # doctest: +SKIP
4
"""

_set_option_tmpl = """
set_option(pat, value)

Sets the value of the specified option.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.
value : object
    New value of option.

Returns
-------
None

Raises
------
OptionError if no such option exists

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.set_option('display.max_columns', 4)
>>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> df
   0  1  ...  3   4
0  1  2  ...  4   5
1  6  7  ...  9  10
[2 rows x 5 columns]
>>> pd.reset_option('display.max_columns')
"""

_describe_option_tmpl = """
describe_option(pat, _print_desc=False)

Prints the description for one or more registered options.

Call with no arguments to get a listing for all registered options.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp pattern. All matching keys will have their description displayed.
_print_desc : bool, default True
    If True (default) the description(s) will be printed to stdout.
    Otherwise, the description(s) will be returned as a unicode string
    (for testing).

Returns
-------
None by default, the description(s) as a unicode string if _print_desc
is False

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.describe_option('display.max_columns')  # doctest: +SKIP
display.max_columns : int
    If max_cols is exceeded, switch to truncate view...
"""

_reset_option_tmpl = """
reset_option(pat)

Reset one or more options to their default value.

Pass "all" as argument to reset all options.

Available options:

{opts_list}

Parameters
----------
pat : str/regex
    If specified only options matching `prefix*` will be reset.
    Note: partial matches are supported for convenience, but unless you
    use the full option name (e.g. x.y.z.option_name), your code may break
    in future versions if new options with similar names are introduced.

Returns
-------
None

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.reset_option('display.max_columns')  # doctest: +SKIP
"""

# bind the functions with their docstrings into a Callable
# and use that as the functions exposed in pd.api
get_option = CallableDynamicDoc(_get_option, _get_option_tmpl)
set_option = CallableDynamicDoc(_set_option, _set_option_tmpl)
reset_option = CallableDynamicDoc(_reset_option, _reset_option_tmpl)
describe_option = CallableDynamicDoc(_describe_option, _describe_option_tmpl)
options = DictWrapper(_global_config)

#
# Functions for use by pandas developers, in addition to User - api


class option_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> from pandas import option_context
    >>> with option_context('display.max_rows', 10, 'display.max_columns', 5):
    ...     pass
    """

    def __init__(self, *args) -> None:
        if len(args) % 2 != 0 or len(args) < 2:
            raise ValueError(
                "Need to invoke as option_context(pat, val, [(pat, val), ...])."
            )

        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self) -> None:
        self.undo = [(pat, _get_option(pat)) for pat, val in self.ops]

        for pat, val in self.ops:
            _set_option(pat, val, silent=True)

    def __exit__(self, *args) -> None:
        if self.undo:
            for pat, val in self.undo:
                _set_option(pat, val, silent=True)


def register_option(
    key: str,
    defval: object,
    doc: str = "",
    validator: Callable[[object], Any] | None = None,
    cb: Callable[[str], Any] | None = None,
) -> None:
    """
    Register an option in the package-wide pandas config object

    Parameters
    ----------
    key : str
        Fully-qualified key, e.g. "x.y.option - z".
    defval : object
        Default value of the option.
    doc : str
        Description of the option.
    validator : Callable, optional
        Function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    cb
        a function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.

    Raises
    ------
    ValueError if `validator` is specified and `defval` is not a valid value.

    """
    import keyword
    import tokenize

    key = key.lower()

    if key in _registered_options:
        raise OptionError(f"Option '{key}' has already been registered")
    if key in _reserved_keys:
        raise OptionError(f"Option '{key}' is a reserved key")

    # the default value should be legal
    if validator:
        validator(defval)

    # walk the nested dict, creating dicts as needed along the path
    path = key.split(".")

    for k in path:
        if not re.match("^" + tokenize.Name + "$", k):
            raise ValueError(f"{k} is not a valid identifier")
        if keyword.iskeyword(k):
            raise ValueError(f"{k} is a python keyword")

    cursor = _global_config
    msg = "Path prefix to option '{option}' is already an option"

    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(msg.format(option=".".join(path[:i])))
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]

    if not isinstance(cursor, dict):
        raise OptionError(msg.format(option=".".join(path[:-1])))

    cursor[path[-1]] = defval  # initialize

    # save the option metadata
    _registered_options[key] = RegisteredOption(
        key=key, defval=defval, doc=doc, validator=validator, cb=cb
    )


def deprecate_option(
    key: str,
    msg: str | None = None,
    rkey: str | None = None,
    removal_ver: str | None = None,
) -> None:
    """
    Mark option `key` as deprecated, if code attempts to access this option,
    a warning will be produced, using `msg` if given, or a default message
    if not.
    if `rkey` is given, any access to the key will be re-routed to `rkey`.

    Neither the existence of `key` nor that if `rkey` is checked. If they
    do not exist, any subsequence access will fail as usual, after the
    deprecation warning is given.

    Parameters
    ----------
    key : str
        Name of the option to be deprecated.
        must be a fully-qualified option name (e.g "x.y.z.rkey").
    msg : str, optional
        Warning message to output when the key is referenced.
        if no message is given a default message will be emitted.
    rkey : str, optional
        Name of an option to reroute access to.
        If specified, any referenced `key` will be
        re-routed to `rkey` including set/get/reset.
        rkey must be a fully-qualified option name (e.g "x.y.z.rkey").
        used by the default message if no `msg` is specified.
    removal_ver : str, optional
        Specifies the version in which this option will
        be removed. used by the default message if no `msg` is specified.

    Raises
    ------
    OptionError
        If the specified key has already been deprecated.
    """
    key = key.lower()

    if key in _deprecated_options:
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")

    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)


#
# functions internal to the module


def _select_options(pat: str) -> list[str]:
    """
    returns a list of keys matching `pat`

    if pat=="all", returns all registered options
    """
    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == "all":  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> tuple[dict[str, Any], str]:
    path = key.split(".")
    cursor = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _is_deprecated(key: str) -> bool:
    """Returns True if the given option has been deprecated"""
    key = key.lower()
    return key in _deprecated_options


def _get_deprecated_option(key: str):
    """
    Retrieves the metadata for a deprecated option, if `key` is deprecated.

    Returns
    -------
    DeprecatedOption (namedtuple) if key is deprecated, None otherwise
    """
    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d


def _get_registered_option(key: str):
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)


def _translate_key(key: str) -> str:
    """
    if key id deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as - is
    """
    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key


def _warn_if_deprecated(key: str) -> bool:
    """
    Checks if `key` is a deprecated option and if so, prints a warning.

    Returns
    -------
    bool - True if `key` is deprecated, False otherwise.
    """
    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            warnings.warn(
                d.msg,
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        else:
            msg = f"'{key}' is deprecated"
            if d.removal_ver:
                msg += f" and will be removed in {d.removal_ver}"
            if d.rkey:
                msg += f", please use '{d.rkey}' instead."
            else:
                msg += ", please refrain from using it."

            warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())
        return True
    return False


def _build_option_description(k: str) -> str:
    """Builds a formatted description of a registered option and prints it"""
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)

    s = f"{k} "

    if o.doc:
        s += "\n".join(o.doc.strip().split("\n"))
    else:
        s += "No description available."

    if o:
        s += f"\n    [default: {o.defval}] [currently: {_get_option(k, True)}]"

    if d:
        rkey = d.rkey or ""
        s += "\n    (Deprecated"
        s += f", use `{rkey}` instead."
        s += ")"

    return s


def pp_options_list(keys: Iterable[str], width: int = 80, _print: bool = False):
    """Builds a concise listing of available options, grouped by prefix"""
    from itertools import groupby
    from textwrap import wrap

    def pp(name: str, ks: Iterable[str]) -> list[str]:
        pfx = "- " + name + ".[" if name else ""
        ls = wrap(
            ", ".join(ks),
            width,
            initial_indent=pfx,
            subsequent_indent="  ",
            break_long_words=False,
        )
        if ls and ls[-1] and name:
            ls[-1] = ls[-1] + "]"
        return ls

    ls: list[str] = []
    singles = [x for x in sorted(keys) if x.find(".") < 0]
    if singles:
        ls += pp("", singles)
    keys = [x for x in keys if x.find(".") >= 0]

    for k, g in groupby(sorted(keys), lambda x: x[: x.rfind(".")]):
        ks = [x[len(k) + 1 :] for x in list(g)]
        ls += pp(k, ks)
    s = "\n".join(ls)
    if _print:
        print(s)
    else:
        return s


#
# helpers


@contextmanager
def config_prefix(prefix: str) -> Generator[None, None, None]:
    """
    contextmanager for multiple invocations of API with a common prefix

    supported API functions: (register / get / set )__option

    Warning: This is not thread - safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.

    Example
    -------
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...

        etc'

    will register options "display.font.color", "display.font.size", set the
    value of "display.font.size"... and so on.
    """
    # Note: reset_option relies on set_option, and on key directly
    # it does not fit in to this monkey-patching scheme

    global register_option, get_option, set_option

    def wrap(func: F) -> F:
        def inner(key: str, *args, **kwds):
            pkey = f"{prefix}.{key}"
            return func(pkey, *args, **kwds)

        return cast(F, inner)

    _register_option = register_option
    _get_option = get_option
    _set_option = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    try:
        yield
    finally:
        set_option = _set_option
        get_option = _get_option
        register_option = _register_option


# These factories and methods are handy for use as the validator
# arg in register_option


def is_type_factory(_type: type[Any]) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`

    """

    def inner(x) -> None:
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")

    return inner


def is_instance_factory(_type) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - the type to be checked against

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if x is not an instance of `_type`

    """
    if isinstance(_type, (tuple, list)):
        _type = tuple(_type)
        type_repr = "|".join(map(str, _type))
    else:
        type_repr = f"'{_type}'"

    def inner(x) -> None:
        if not isinstance(x, _type):
            raise ValueError(f"Value must be an instance of {type_repr}")

    return inner


def is_one_of_factory(legal_values) -> Callable[[Any], None]:
    callables = [c for c in legal_values if callable(c)]
    legal_values = [c for c in legal_values if not callable(c)]

    def inner(x) -> None:
        if x not in legal_values:
            if not any(c(x) for c in callables):
                uvals = [str(lval) for lval in legal_values]
                pp_values = "|".join(uvals)
                msg = f"Value must be one of {pp_values}"
                if len(callables):
                    msg += " or a callable"
                raise ValueError(msg)

    return inner


def is_nonnegative_int(value: object) -> None:
    """
    Verify that value is None or a positive int.

    Parameters
    ----------
    value : None or int
            The `value` to be checked.

    Raises
    ------
    ValueError
        When the value is not None or is a negative integer
    """
    if value is None:
        return

    elif isinstance(value, int):
        if value >= 0:
            return

    msg = "Value must be a nonnegative integer or None"
    raise ValueError(msg)


# common type validators, for convenience
# usage: register_option(... , validator = is_int)
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))


def is_callable(obj) -> bool:
    """

    Parameters
    ----------
    `obj` - the object to be checked

    Returns
    -------
    validator - returns True if object is callable
        raises ValueError otherwise.

    """
    if not callable(obj):
        raise ValueError("Value must be a callable")
    return True
<|endoftext|>"""
pandas._config is considered explicitly upstream of everything else in pandas,
should have no intra-pandas dependencies.

importing `dates` and `display` ensures that keys needed by _libs
are initialized.
"""
__all__ = [
    "config",
    "detect_console_encoding",
    "get_option",
    "set_option",
    "reset_option",
    "describe_option",
    "option_context",
    "options",
    "using_copy_on_write",
    "warn_copy_on_write",
]
from pandas._config import config
from pandas._config import dates  # pyright: ignore[reportUnusedImport]  # noqa: F401
from pandas._config.config import (
    _global_config,
    describe_option,
    get_option,
    option_context,
    options,
    reset_option,
    set_option,
)
from pandas._config.display import detect_console_encoding


def using_copy_on_write() -> bool:
    _mode_options = _global_config["mode"]
    return (
        _mode_options["copy_on_write"] is True
        and _mode_options["data_manager"] == "block"
    )


def warn_copy_on_write() -> bool:
    _mode_options = _global_config["mode"]
    return (
        _mode_options["copy_on_write"] == "warn"
        and _mode_options["data_manager"] == "block"
    )


def using_nullable_dtypes() -> bool:
    _mode_options = _global_config["mode"]
    return _mode_options["nullable_dtypes"]


def using_string_dtype() -> bool:
    _mode_options = _global_config["future"]
    return _mode_options["infer_string"]
<|endoftext|>"""
Helpers for configuring locale settings.

Name `localization` is chosen to avoid overlap with builtin `locale` module.
"""
from __future__ import annotations

from contextlib import contextmanager
import locale
import platform
import re
import subprocess
from typing import TYPE_CHECKING

from pandas._config.config import options

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def set_locale(
    new_locale: str | tuple[str, str], lc_var: int = locale.LC_ALL
) -> Generator[str | tuple[str, str], None, None]:
    """
    Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    # getlocale is not always compliant with setlocale, use setlocale. GH#46595
    current_locale = locale.setlocale(lc_var)

    try:
        locale.setlocale(lc_var, new_locale)
        normalized_code, normalized_encoding = locale.getlocale()
        if normalized_code is not None and normalized_encoding is not None:
            yield f"{normalized_code}.{normalized_encoding}"
        else:
            yield new_locale
    finally:
        locale.setlocale(lc_var, current_locale)


def can_set_locale(lc: str, lc_var: int = locale.LC_ALL) -> bool:
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError, locale.Error):
        # horrible name for a Exception subclass
        return False
    else:
        return True


def _valid_locales(locales: list[str] | str, normalize: bool) -> list[str]:
    """
    Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    return [
        loc
        for loc in (
            locale.normalize(loc.strip()) if normalize else loc.strip()
            for loc in locales
        )
        if can_set_locale(loc)
    ]


def get_locales(
    prefix: str | None = None,
    normalize: bool = True,
) -> list[str]:
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return an empty list (no locale available, e.g. Windows)

    """
    if platform.system() in ("Linux", "Darwin"):
        raw_locales = subprocess.check_output(["locale", "-a"])
    else:
        # Other platforms e.g. windows platforms don't define "locale -a"
        #  Note: is_platform_windows causes circular import here
        return []

    try:
        # raw_locales is "\n" separated list of locales
        # it may contain non-decodable parts, so split
        # extract what we can and then rejoin.
        split_raw_locales = raw_locales.split(b"\n")
        out_locales = []
        for x in split_raw_locales:
            try:
                out_locales.append(str(x, encoding=options.display.encoding))
            except UnicodeError:
                # 'locale -a' is used to populated 'raw_locales' and on
                # Redhat 7 Linux (and maybe others) prints locale names
                # using windows-1252 encoding.  Bug only triggered by
                # a few special characters and when there is an
                # extensive list of installed locales.
                out_locales.append(str(x, encoding="windows-1252"))

    except TypeError:
        pass

    if prefix is None:
        return _valid_locales(out_locales, normalize)

    pattern = re.compile(f"{prefix}.*")
    found = pattern.findall("\n".join(out_locales))
    return _valid_locales(found, normalize)
<|endoftext|>from __future__ import annotations

import importlib
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

from pandas._config import get_option

from pandas.util._decorators import (
    Appender,
    Substitution,
)

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )
    import types

    from matplotlib.axes import Axes
    import numpy as np

    from pandas._typing import IndexLabel

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.groupby.generic import DataFrameGroupBy


def hist_series(
    self: Series,
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
    """
    Draw histogram of the input series using matplotlib.

    Parameters
    ----------
    by : object, optional
        If passed, then used to form histograms for separate groups.
    ax : matplotlib axis object
        If not passed, uses gca().
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels.
    figsize : tuple, default None
        Figure size in inches by default.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    legend : bool, default False
        Whether to show the legend.

    **kwargs
        To be passed to the actual plotting function.

    Returns
    -------
    matplotlib.AxesSubplot
        A histogram plot.

    See Also
    --------
    matplotlib.axes.Axes.hist : Plot a histogram using matplotlib.

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> hist = ser.hist()

    For Groupby:

    .. plot::
        :context: close-figs

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> hist = ser.groupby(level=0).hist()
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(
        self,
        by=by,
        ax=ax,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        figsize=figsize,
        bins=bins,
        legend=legend,
        **kwargs,
    )


def hist_frame(
    data: DataFrame,
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
    """
    Make a histogram of the DataFrame's columns.

    A `histogram`_ is a representation of the distribution of data.
    This function calls :meth:`matplotlib.pyplot.hist`, on each series in
    the DataFrame, resulting in one histogram per column.

    .. _histogram: https://en.wikipedia.org/wiki/Histogram

    Parameters
    ----------
    data : DataFrame
        The pandas object holding the data.
    column : str or sequence, optional
        If passed, will be used to limit data to a subset of columns.
    by : object, optional
        If passed, then used to form histograms for separate groups.
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels. For example, a value of 90 displays the
        x labels rotated 90 degrees clockwise.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels. For example, a value of 90 displays the
        y labels rotated 90 degrees clockwise.
    ax : Matplotlib axes object, default None
        The axes to plot the histogram on.
    sharex : bool, default True if ax is None else False
        In case subplots=True, share x axis and set some x axis labels to
        invisible; defaults to True if ax is None otherwise False if an ax
        is passed in.
        Note that passing in both an ax and sharex=True will alter all x axis
        labels for all subplots in a figure.
    sharey : bool, default False
        In case subplots=True, share y axis and set some y axis labels to
        invisible.
    figsize : tuple, optional
        The size in inches of the figure to create. Uses the value in
        `matplotlib.rcParams` by default.
    layout : tuple, optional
        Tuple of (rows, columns) for the layout of the histograms.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.

    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.

    legend : bool, default False
        Whether to show the legend.

    **kwargs
        All other plotting keyword arguments to be passed to
        :meth:`matplotlib.pyplot.hist`.

    Returns
    -------
    matplotlib.AxesSubplot or numpy.ndarray of them

    See Also
    --------
    matplotlib.pyplot.hist : Plot a histogram using matplotlib.

    Examples
    --------
    This example draws a histogram based on the length and width of
    some animals, displayed in three bins

    .. plot::
        :context: close-figs

        >>> data = {'length': [1.5, 0.5, 1.2, 0.9, 3],
        ...         'width': [0.7, 0.2, 0.15, 0.2, 1.1]}
        >>> index = ['pig', 'rabbit', 'duck', 'chicken', 'horse']
        >>> df = pd.DataFrame(data, index=index)
        >>> hist = df.hist(bins=3)
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(
        data,
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
        legend=legend,
        bins=bins,
        **kwargs,
    )


_boxplot_doc = """
Make a box plot from DataFrame columns.

Make a box-and-whisker plot from DataFrame columns, optionally grouped
by some other columns. A box plot is a method for graphically depicting
groups of numerical data through their quartiles.
The box extends from the Q1 to Q3 quartile values of the data,
with a line at the median (Q2). The whiskers extend from the edges
of box to show the range of the data. By default, they extend no more than
`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
data point within that interval. Outliers are plotted as separate dots.

For further details see
Wikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.

Parameters
----------
%(data)s\
column : str or list of str, optional
    Column name or list of names, or vector.
    Can be any valid input to :meth:`pandas.DataFrame.groupby`.
by : str or array-like, optional
    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.
    One box-plot will be done per value of columns in `by`.
ax : object of class matplotlib.axes.Axes, optional
    The matplotlib axes to be used by boxplot.
fontsize : float or str
    Tick label font size in points or as a string (e.g., `large`).
rot : float, default 0
    The rotation angle of labels (in degrees)
    with respect to the screen coordinate system.
grid : bool, default True
    Setting this to True will show the grid.
figsize : A tuple (width, height) in inches
    The size of the figure to create in matplotlib.
layout : tuple (rows, columns), optional
    For example, (3, 5) will display the subplots
    using 3 rows and 5 columns, starting from the top-left.
return_type : {'axes', 'dict', 'both'} or None, default 'axes'
    The kind of object to return. The default is ``axes``.

    * 'axes' returns the matplotlib axes the boxplot is drawn on.
    * 'dict' returns a dictionary whose values are the matplotlib
      Lines of the boxplot.
    * 'both' returns a namedtuple with the axes and dict.
    * when grouping with ``by``, a Series mapping columns to
      ``return_type`` is returned.

      If ``return_type`` is `None`, a NumPy array
      of axes with the same shape as ``layout`` is returned.
%(backend)s\

**kwargs
    All other plotting keyword arguments to be passed to
    :func:`matplotlib.pyplot.boxplot`.

Returns
-------
result
    See Notes.

See Also
--------
pandas.Series.plot.hist: Make a histogram.
matplotlib.pyplot.boxplot : Matplotlib equivalent plot.

Notes
-----
The return type depends on the `return_type` parameter:

* 'axes' : object of class matplotlib.axes.Axes
* 'dict' : dict of matplotlib.lines.Line2D objects
* 'both' : a namedtuple with structure (ax, lines)

For data grouped with ``by``, return a Series of the above or a numpy
array:

* :class:`~pandas.Series`
* :class:`~numpy.array` (for ``return_type = None``)

Use ``return_type='dict'`` when you want to tweak the appearance
of the lines after plotting. In this case a dict containing the Lines
making up the boxes, caps, fliers, medians, and whiskers is returned.

Examples
--------

Boxplots can be created for every column in the dataframe
by ``df.boxplot()`` or indicating the columns to be used:

.. plot::
    :context: close-figs

    >>> np.random.seed(1234)
    >>> df = pd.DataFrame(np.random.randn(10, 4),
    ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])
    >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP

Boxplots of variables distributions grouped by the values of a third
variable can be created using the option ``by``. For instance:

.. plot::
    :context: close-figs

    >>> df = pd.DataFrame(np.random.randn(10, 2),
    ...                   columns=['Col1', 'Col2'])
    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
    ...                      'B', 'B', 'B', 'B', 'B'])
    >>> boxplot = df.boxplot(by='X')

A list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot
in order to group the data by combination of the variables in the x-axis:

.. plot::
    :context: close-figs

    >>> df = pd.DataFrame(np.random.randn(10, 3),
    ...                   columns=['Col1', 'Col2', 'Col3'])
    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
    ...                      'B', 'B', 'B', 'B', 'B'])
    >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',
    ...                      'B', 'A', 'B', 'A', 'B'])
    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])

The layout of boxplot can be adjusted giving a tuple to ``layout``:

.. plot::
    :context: close-figs

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      layout=(2, 1))

Additional formatting can be done to the boxplot, like suppressing the grid
(``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)
or changing the fontsize (i.e. ``fontsize=15``):

.. plot::
    :context: close-figs

    >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP

The parameter ``return_type`` can be used to select the type of element
returned by `boxplot`.  When ``return_type='axes'`` is selected,
the matplotlib axes on which the boxplot is drawn are returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')
    >>> type(boxplot)
    <class 'matplotlib.axes._axes.Axes'>

When grouping with ``by``, a Series mapping columns to ``return_type``
is returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      return_type='axes')
    >>> type(boxplot)
    <class 'pandas.core.series.Series'>

If ``return_type`` is `None`, a NumPy array of axes with the same shape
as ``layout`` is returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      return_type=None)
    >>> type(boxplot)
    <class 'numpy.ndarray'>
"""

_backend_doc = """\
backend : str, default None
    Backend to use instead of the backend specified in the option
    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
    specify the ``plotting.backend`` for the whole session, set
    ``pd.options.plotting.backend``.
"""


_bar_or_line_doc = """
        Parameters
        ----------
        x : label or position, optional
            Allows plotting of one column versus another. If not specified,
            the index of the DataFrame is used.
        y : label or position, optional
            Allows plotting of one column versus another. If not specified,
            all numerical columns are used.
        color : str, array-like, or dict, optional
            The color for each of the DataFrame's columns. Possible values are:

            - A single color string referred to by name, RGB or RGBA code,
                for instance 'red' or '#a98d19'.

            - A sequence of color strings referred to by name, RGB or RGBA
                code, which will be used for each column recursively. For
                instance ['green','yellow'] each column's %(kind)s will be filled in
                green or yellow, alternatively. If there is only a single column to
                be plotted, then only the first color from the color list will be
                used.

            - A dict of the form {column name : color}, so that each column will be
                colored accordingly. For example, if your columns are called `a` and
                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for
                column `a` in green and %(kind)ss for column `b` in red.

        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            An ndarray is returned with one :class:`matplotlib.axes.Axes`
            per column when ``subplots=True``.
"""


@Substitution(data="data : DataFrame\n    The data to visualize.\n", backend="")
@Appender(_boxplot_doc)
def boxplot(
    data: DataFrame,
    column: str | list[str] | None = None,
    by: str | list[str] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: str | None = None,
    **kwargs,
):
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.boxplot(
        data,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


@Substitution(data="", backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: DataFrame,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    backend=None,
    **kwargs,
):
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


def boxplot_frame_groupby(
    grouped: DataFrameGroupBy,
    subplots: bool = True,
    column=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax=None,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = True,
    backend=None,
    **kwargs,
):
    """
    Make box plots from DataFrameGroupBy data.

    Parameters
    ----------
    grouped : Grouped DataFrame
    subplots : bool
        * ``False`` - no subplots will be used
        * ``True`` - create a subplot for each group.

    column : column name or list of names, or vector
        Can be any valid input to groupby.
    fontsize : float or str
    rot : label rotation angle
    grid : Setting this to True will show the grid
    ax : Matplotlib axis object, default None
    figsize : A tuple (width, height) in inches
    layout : tuple (optional)
        The layout of the plot: (rows, columns).
    sharex : bool, default False
        Whether x-axes will be shared among subplots.
    sharey : bool, default True
        Whether y-axes will be shared among subplots.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        All other plotting keyword arguments to be passed to
        matplotlib's boxplot function.

    Returns
    -------
    dict of key/value = group key/DataFrame.boxplot return value
    or DataFrame.boxplot return value in case subplots=figures=False

    Examples
    --------
    You can create boxplots for grouped data and show them as separate subplots:

    .. plot::
        :context: close-figs

        >>> import itertools
        >>> tuples = [t for t in itertools.product(range(1000), range(4))]
        >>> index = pd.MultiIndex.from_tuples(tuples, names=['lvl0', 'lvl1'])
        >>> data = np.random.randn(len(index), 4)
        >>> df = pd.DataFrame(data, columns=list('ABCD'), index=index)
        >>> grouped = df.groupby(level='lvl1')
        >>> grouped.boxplot(rot=45, fontsize=12, figsize=(8, 10))  # doctest: +SKIP

    The ``subplots=False`` option shows the boxplots in a single figure.

    .. plot::
        :context: close-figs

        >>> grouped.boxplot(subplots=False, rot=45, fontsize=12)  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(
        grouped,
        subplots=subplots,
        column=column,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        ax=ax,
        figsize=figsize,
        layout=layout,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )


class PlotAccessor(PandasObject):
    """
    Make plots of Series or DataFrame.

    Uses the backend specified by the
    option ``plotting.backend``. By default, matplotlib is used.

    Parameters
    ----------
    data : Series or DataFrame
        The object for which the method is called.
    x : label or position, default None
        Only used if data is a DataFrame.
    y : label, position or list of label, positions, default None
        Allows plotting of one column versus another. Only used if data is a
        DataFrame.
    kind : str
        The kind of plot to produce:

        - 'line' : line plot (default)
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : boxplot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot (DataFrame only)
        - 'hexbin' : hexbin plot (DataFrame only)
    ax : matplotlib axes object, default None
        An axes of the current figure.
    subplots : bool or sequence of iterables, default False
        Whether to group columns into subplots:

        - ``False`` : No subplots will be used
        - ``True`` : Make separate subplots for each column.
        - sequence of iterables of column labels: Create a subplot for each
          group of columns. For example `[('a', 'c'), ('b', 'd')]` will
          create 2 subplots: one with columns 'a' and 'c', and one
          with columns 'b' and 'd'. Remaining columns that aren't specified
          will be plotted in additional subplots (one per column).

          .. versionadded:: 1.5.0

    sharex : bool, default True if ax is None else False
        In case ``subplots=True``, share x axis and set some x axis labels
        to invisible; defaults to True if ax is None otherwise False if
        an ax is passed in; Be aware, that passing in both an ax and
        ``sharex=True`` will alter all x axis labels for all axis in a figure.
    sharey : bool, default False
        In case ``subplots=True``, share y axis and set some y axis labels to invisible.
    layout : tuple, optional
        (rows, columns) for the layout of subplots.
    figsize : a tuple (width, height) in inches
        Size of a figure object.
    use_index : bool, default True
        Use index as ticks for x axis.
    title : str or list
        Title to use for the plot. If a string is passed, print the string
        at the top of the figure. If a list is passed and `subplots` is
        True, print each item in the list above the corresponding subplot.
    grid : bool, default None (matlab style default)
        Axis grid lines.
    legend : bool or {'reverse'}
        Place legend on axis subplots.
    style : list or dict
        The matplotlib line style per column.
    logx : bool or 'sym', default False
        Use log scaling or symlog scaling on x axis.

    logy : bool or 'sym' default False
        Use log scaling or symlog scaling on y axis.

    loglog : bool or 'sym', default False
        Use log scaling or symlog scaling on both x and y axes.

    xticks : sequence
        Values to use for the xticks.
    yticks : sequence
        Values to use for the yticks.
    xlim : 2-tuple/list
        Set the x limits of the current axes.
    ylim : 2-tuple/list
        Set the y limits of the current axes.
    xlabel : label, optional
        Name to use for the xlabel on x-axis. Default uses index name as xlabel, or the
        x-column name for planar plots.

        .. versionchanged:: 2.0.0

            Now applicable to histograms.

    ylabel : label, optional
        Name to use for the ylabel on y-axis. Default will show no ylabel, or the
        y-column name for planar plots.

        .. versionchanged:: 2.0.0

            Now applicable to histograms.

    rot : float, default None
        Rotation for ticks (xticks for vertical, yticks for horizontal
        plots).
    fontsize : float, default None
        Font size for xticks and yticks.
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that
        name from matplotlib.
    colorbar : bool, optional
        If True, plot colorbar (only relevant for 'scatter' and 'hexbin'
        plots).
    position : float
        Specify relative alignments for bar plot layout.
        From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
        (center).
    table : bool, Series or DataFrame, default False
        If True, draw a table using the data in the DataFrame and the data
        will be transposed to meet matplotlib's default layout.
        If a Series or DataFrame is passed, use passed data to draw a
        table.
    yerr : DataFrame, Series, array-like, dict and str
        See :ref:`Plotting with Error Bars <visualization.errorbars>` for
        detail.
    xerr : DataFrame, Series, array-like, dict and str
        Equivalent to yerr.
    stacked : bool, default False in line and bar plots, and True in area plot
        If True, create stacked plot.
    secondary_y : bool or sequence, default False
        Whether to plot on the secondary y-axis if a list/tuple, which
        columns to plot on secondary y-axis.
    mark_right : bool, default True
        When using a secondary_y axis, automatically mark the column
        labels with "(right)" in the legend.
    include_bool : bool, default is False
        If True, boolean values can be plotted.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        If the backend is not the default matplotlib one, the return value
        will be the object returned by the backend.

    Notes
    -----
    - See matplotlib documentation online for more on this subject
    - If `kind` = 'bar' or 'barh', you can specify relative alignments
      for bar plot layout by `position` keyword.
      From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
      (center)

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> ser = pd.Series([1, 2, 3, 3])
        >>> plot = ser.plot(kind='hist', title="My plot")

    For DataFrame:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame({'length': [1.5, 0.5, 1.2, 0.9, 3],
        ...                   'width': [0.7, 0.2, 0.15, 0.2, 1.1]},
        ...                   index=['pig', 'rabbit', 'duck', 'chicken', 'horse'])
        >>> plot = df.plot(title="DataFrame Plot")

    For SeriesGroupBy:

    .. plot::
        :context: close-figs

        >>> lst = [-1, -2, -3, 1, 2, 3]
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> plot = ser.groupby(lambda x: x > 0).plot(title="SeriesGroupBy Plot")

    For DataFrameGroupBy:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame({"col1" : [1, 2, 3, 4],
        ...                   "col2" : ["A", "B", "A", "B"]})
        >>> plot = df.groupby("col2").plot(kind="bar", title="DataFrameGroupBy Plot")
    """

    _common_kinds = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    _series_kinds = ("pie",)
    _dataframe_kinds = ("scatter", "hexbin")
    _kind_aliases = {"density": "kde"}
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Series | DataFrame) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(backend_name: str, data: Series | DataFrame, args, kwargs):
        """
        This function makes calls to this accessor `__call__` method compatible
        with the previous `SeriesPlotMethods.__call__` and
        `DataFramePlotMethods.__call__`. Those had slightly different
        signatures, since `DataFramePlotMethods` accepted `x` and `y`
        parameters.
        """
        if isinstance(data, ABCSeries):
            arg_def = [
                ("kind", "line"),
                ("ax", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", False),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("label", None),
                ("secondary_y", False),
                ("xlabel", None),
                ("ylabel", None),
            ]
        elif isinstance(data, ABCDataFrame):
            arg_def = [
                ("x", None),
                ("y", None),
                ("kind", "line"),
                ("ax", None),
                ("subplots", False),
                ("sharex", None),
                ("sharey", False),
                ("layout", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", True),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("secondary_y", False),
                ("xlabel", None),
                ("ylabel", None),
            ]
        else:
            raise TypeError(
                f"Called plot accessor for type {type(data).__name__}, "
                "expected Series or DataFrame"
            )

        if args and isinstance(data, ABCSeries):
            positional_args = str(args)[1:-1]
            keyword_args = ", ".join(
                [f"{name}={repr(value)}" for (name, _), value in zip(arg_def, args)]
            )
            msg = (
                "`Series.plot()` should not be called with positional "
                "arguments, only keyword arguments. The order of "
                "positional arguments will change in the future. "
                f"Use `Series.plot({keyword_args})` instead of "
                f"`Series.plot({positional_args})`."
            )
            raise TypeError(msg)

        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == "pandas.plotting._matplotlib":
            kwargs = dict(arg_def, **pos_args, **kwargs)
        else:
            kwargs = dict(pos_args, **kwargs)

        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        kind = kwargs.pop("kind", "line")
        return x, y, kind, kwargs

    def __call__(self, *args, **kwargs):
        plot_backend = _get_plot_backend(kwargs.pop("backend", None))

        x, y, kind, kwargs = self._get_call_args(
            plot_backend.__name__, self._parent, args, kwargs
        )

        kind = self._kind_aliases.get(kind, kind)

        # when using another backend, get out of the way
        if plot_backend.__name__ != "pandas.plotting._matplotlib":
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)

        if kind not in self._all_kinds:
            raise ValueError(
                f"{kind} is not a valid plot kind "
                f"Valid plot kinds: {self._all_kinds}"
            )

        # The original data structured can be transformed before passed to the
        # backend. For example, for DataFrame is common to set the index as the
        # `x` parameter, and return a Series with the parameter `y` as values.
        data = self._parent.copy()

        if isinstance(data, ABCSeries):
            kwargs["reuse_plot"] = True

        if kind in self._dataframe_kinds:
            if isinstance(data, ABCDataFrame):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            else:
                raise ValueError(f"plot kind {kind} can only be used for data frames")
        elif kind in self._series_kinds:
            if isinstance(data, ABCDataFrame):
                if y is None and kwargs.get("subplots") is False:
                    raise ValueError(
                        f"{kind} requires either y column or 'subplots=True'"
                    )
                if y is not None:
                    if is_integer(y) and not data.columns._holds_integer():
                        y = data.columns[y]
                    # converted to series actually. copy to not modify
                    data = data[y].copy()
                    data.index.name = y
        elif isinstance(data, ABCDataFrame):
            data_cols = data.columns
            if x is not None:
                if is_integer(x) and not data.columns._holds_integer():
                    x = data_cols[x]
                elif not isinstance(data[x], ABCSeries):
                    raise ValueError("x must be a label or position")
                data = data.set_index(x)
            if y is not None:
                # check if we have y as int or list of ints
                int_ylist = is_list_like(y) and all(is_integer(c) for c in y)
                int_y_arg = is_integer(y) or int_ylist
                if int_y_arg and not data.columns._holds_integer():
                    y = data_cols[y]

                label_kw = kwargs["label"] if "label" in kwargs else False
                for kw in ["xerr", "yerr"]:
                    if kw in kwargs and (
                        isinstance(kwargs[kw], str) or is_integer(kwargs[kw])
                    ):
                        try:
                            kwargs[kw] = data[kwargs[kw]]
                        except (IndexError, KeyError, TypeError):
                            pass

                # don't overwrite
                data = data[y].copy()

                if isinstance(data, ABCSeries):
                    label_name = label_kw or y
                    data.name = label_name
                else:
                    match = is_list_like(label_kw) and len(label_kw) == len(y)
                    if label_kw and not match:
                        raise ValueError(
                            "label should be list-like and same length as y"
                        )
                    label_name = label_kw or data.columns
                    data.columns = label_name

        return plot_backend.plot(data, kind=kind, **kwargs)

    __call__.__doc__ = __doc__

    @Appender(
        """
        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 3, 2])
            >>> s.plot.line()  # doctest: +SKIP

        .. plot::
            :context: close-figs

            The following example shows the populations for some animals
            over the years.

            >>> df = pd.DataFrame({
            ...    'pig': [20, 18, 489, 675, 1776],
            ...    'horse': [4, 25, 281, 600, 1900]
            ...    }, index=[1990, 1997, 2003, 2009, 2014])
            >>> lines = df.plot.line()

        .. plot::
           :context: close-figs

           An example with subplots, so an array of axes is returned.

           >>> axes = df.plot.line(subplots=True)
           >>> type(axes)
           <class 'numpy.ndarray'>

        .. plot::
           :context: close-figs

           Let's repeat the same example, but specifying colors for
           each column (in this case, for each animal).

           >>> axes = df.plot.line(
           ...     subplots=True, color={"pig": "pink", "horse": "#742802"}
           ... )

        .. plot::
            :context: close-figs

            The following example shows the relationship between both
            populations.

            >>> lines = df.plot.line(x='pig', y='horse')
        """
    )
    @Substitution(kind="line")
    @Appender(_bar_or_line_doc)
    def line(
        self, x: Hashable | None = None, y: Hashable | None = None, **kwargs
    ) -> PlotAccessor:
        """
        Plot Series or DataFrame as lines.

        This function is useful to plot lines using DataFrame's values
        as coordinates.
        """
        return self(kind="line", x=x, y=y, **kwargs)

    @Appender(
        """
        See Also
        --------
        DataFrame.plot.barh : Horizontal bar plot.
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.bar : Make a bar plot with matplotlib.

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)

        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)

        Plot stacked bar charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(stacked=True)

        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        If you don't like the default colours, you can specify how you'd
        like each column to be colored.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(
            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}
            ... )
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        Plot a single column.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(y='speed', rot=0)

        Plot only selected categories for the DataFrame.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(x='lifespan', rot=0)
    """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def bar(  # pylint: disable=disallowed-name
        self, x: Hashable | None = None, y: Hashable | None = None, **kwargs
    ) -> PlotAccessor:
        """
        Vertical bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        return self(kind="bar", x=x, y=y, **kwargs)

    @Appender(
        """
        See Also
        --------
        DataFrame.plot.bar: Vertical bar plot.
        DataFrame.plot : Make plots of DataFrame using matplotlib.
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        Basic example

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.barh(x='lab', y='val')

        Plot a whole DataFrame to a horizontal bar plot

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh()

        Plot stacked barh charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.barh(stacked=True)

        We can specify colors for each column

        .. plot::
            :context: close-figs

            >>> ax = df.plot.barh(color={"speed": "red", "lifespan": "green"})

        Plot a column of the DataFrame to a horizontal bar plot

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(y='speed')

        Plot DataFrame versus the desired column

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(x='lifespan')
    """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def barh(
        self, x: Hashable | None = None, y: Hashable | None = None, **kwargs
    ) -> PlotAccessor:
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        return self(kind="barh", x=x, y=y, **kwargs)

    def box(self, by: IndexLabel | None = None, **kwargs) -> PlotAccessor:
        r"""
        Make a box plot of the DataFrame columns.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.
        The box extends from the Q1 to Q3 quartile values of the data,
        with a line at the median (Q2). The whiskers extend from the edges
        of box to show the range of the data. The position of the whiskers
        is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the
        box. Outlier points are those past the end of the whiskers.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        A consideration when using this chart is that the box and the whiskers
        can overlap, which is very common when plotting small sets of data.

        Parameters
        ----------
        by : str or sequence
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        **kwargs
            Additional keywords are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        DataFrame.boxplot: Another method to draw a box plot.
        Series.plot.box: Draw a box plot from a Series object.
        matplotlib.pyplot.boxplot: Draw a box plot in matplotlib.

        Examples
        --------
        Draw a box plot from a DataFrame with four columns of randomly
        generated data.

        .. plot::
            :context: close-figs

            >>> data = np.random.randn(25, 4)
            >>> df = pd.DataFrame(data, columns=list('ABCD'))
            >>> ax = df.plot.box()

        You can also generate groupings if you specify the `by` parameter (which
        can take a column name, or a list or tuple of column names):

        .. versionchanged:: 1.4.0

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.box(column="age", by="gender", figsize=(10, 8))
        """
        return self(kind="box", by=by, **kwargs)

    def hist(
        self, by: IndexLabel | None = None, bins: int = 10, **kwargs
    ) -> PlotAccessor:
        """
        Draw one histogram of the DataFrame's columns.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.
        This is useful when the DataFrame's Series are in a similar scale.

        Parameters
        ----------
        by : str or sequence, optional
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        bins : int, default 10
            Number of histogram bins to be used.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        class:`matplotlib.AxesSubplot`
            Return a histogram plot.

        See Also
        --------
        DataFrame.hist : Draw histograms per DataFrame's Series.
        Series.hist : Draw a histogram with Series' data.

        Examples
        --------
        When we roll a die 6000 times, we expect to get each value around 1000
        times. But when we roll two dice and sum the result, the distribution
        is going to be quite different. A histogram illustrates those
        distributions.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randint(1, 7, 6000), columns=['one'])
            >>> df['two'] = df['one'] + np.random.randint(1, 7, 6000)
            >>> ax = df.plot.hist(bins=12, alpha=0.5)

        A grouped histogram can be generated by providing the parameter `by` (which
        can be a column name, or a list of column names):

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.hist(column=["age"], by="gender", figsize=(10, 8))
        """
        return self(kind="hist", by=by, bins=bins, **kwargs)

    def kde(
        self,
        bw_method: Literal["scott", "silverman"] | float | Callable | None = None,
        ind: np.ndarray | int | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Generate Kernel Density Estimate plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth. This can be
            'scott', 'silverman', a scalar constant or a callable.
            If None (default), 'scott' is used.
            See :class:`scipy.stats.gaussian_kde` for more information.
        ind : NumPy array or int, optional
            Evaluation points for the estimated PDF. If None (default),
            1000 equally spaced points are used. If `ind` is a NumPy array, the
            KDE is evaluated at the points passed. If `ind` is an integer,
            `ind` number of equally spaced points are used.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        See Also
        --------
        scipy.stats.gaussian_kde : Representation of a kernel-density
            estimate using Gaussian kernels. This is the function used
            internally to estimate the PDF.

        Examples
        --------
        Given a Series of points randomly sampled from an unknown
        distribution, estimate its PDF using KDE with automatic
        bandwidth determination and plot the results, evaluating them at
        1000 equally spaced points (default):

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> ax = s.plot.kde()

        A scalar bandwidth can be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=3)

        Finally, the `ind` parameter determines the evaluation points for the
        plot of the estimated PDF:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(ind=[1, 2, 3, 4, 5])

        For DataFrame, it works in the same way:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 2.5, 3, 3.5, 4, 5],
            ...     'y': [4, 4, 4.5, 5, 5.5, 6, 6],
            ... })
            >>> ax = df.plot.kde()

        A scalar bandwidth can be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=3)

        Finally, the `ind` parameter determines the evaluation points for the
        plot of the estimated PDF:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(ind=[1, 2, 3, 4, 5, 6])
        """
        return self(kind="kde", bw_method=bw_method, ind=ind, **kwargs)

    density = kde

    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        **kwargs,
    ) -> PlotAccessor:
        """
        Draw a stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the matplotlib area function.

        Parameters
        ----------
        x : label or position, optional
            Coordinates for the X axis. By default uses the index.
        y : label or position, optional
            Column to plot. By default uses all columns.
        stacked : bool, default True
            Area plots are stacked by default. Set to False to create a
            unstacked plot.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Area plot, or array of area plots if subplots is True.

        See Also
        --------
        DataFrame.plot : Make plots of DataFrame using matplotlib / pylab.

        Examples
        --------
        Draw an area plot based on basic business metrics:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({
            ...     'sales': [3, 2, 3, 9, 10, 6],
            ...     'signups': [5, 5, 6, 12, 14, 13],
            ...     'visits': [20, 42, 28, 62, 81, 50],
            ... }, index=pd.date_range(start='2018/01/01', end='2018/07/01',
            ...                        freq='ME'))
            >>> ax = df.plot.area()

        Area plots are stacked by default. To produce an unstacked plot,
        pass ``stacked=False``:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(stacked=False)

        Draw an area plot for a single column:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(y='sales')

        Draw with a different `x`:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({
            ...     'sales': [3, 2, 3],
            ...     'visits': [20, 42, 28],
            ...     'day': [1, 2, 3],
            ... })
            >>> ax = df.plot.area(x='day')
        """
        return self(kind="area", x=x, y=y, stacked=stacked, **kwargs)

    def pie(self, **kwargs) -> PlotAccessor:
        """
        Generate a pie plot.

        A pie plot is a proportional representation of the numerical data in a
        column. This function wraps :meth:`matplotlib.pyplot.pie` for the
        specified column. If no column reference is passed and
        ``subplots=True`` a pie plot is drawn for each numerical column
        independently.

        Parameters
        ----------
        y : int or label, optional
            Label or position of the column to plot.
            If not provided, ``subplots=True`` argument must be passed.
        **kwargs
            Keyword arguments to pass on to :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            A NumPy array is returned when `subplots` is True.

        See Also
        --------
        Series.plot.pie : Generate a pie plot for a Series.
        DataFrame.plot : Make plots of a DataFrame.

        Examples
        --------
        In the example below we have a DataFrame with the information about
        planet's mass and radius. We pass the 'mass' column to the
        pie function to get a pie plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'mass': [0.330, 4.87 , 5.97],
            ...                    'radius': [2439.7, 6051.8, 6378.1]},
            ...                   index=['Mercury', 'Venus', 'Earth'])
            >>> plot = df.plot.pie(y='mass', figsize=(5, 5))

        .. plot::
            :context: close-figs

            >>> plot = df.plot.pie(subplots=True, figsize=(11, 6))
        """
        if (
            isinstance(self._parent, ABCDataFrame)
            and kwargs.get("y", None) is None
            and not kwargs.get("subplots", False)
        ):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind="pie", **kwargs)

    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[Hashable] | None = None,
        c: Hashable | Sequence[Hashable] | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Create a scatter plot with varying marker point size and color.

        The coordinates of each point are defined by two dataframe columns and
        filled circles are used to represent each point. This kind of plot is
        useful to see complex correlations between two variables. Points could
        be for instance natural 2D coordinates like longitude and latitude in
        a map or, in general, any pair of metrics that can be plotted against
        each other.

        Parameters
        ----------
        x : int or str
            The column name or column position to be used as horizontal
            coordinates for each point.
        y : int or str
            The column name or column position to be used as vertical
            coordinates for each point.
        s : str, scalar or array-like, optional
            The size of each point. Possible values are:

            - A string with the name of the column to be used for marker's size.

            - A single scalar so all points have the same size.

            - A sequence of scalars, which will be used for each point's size
              recursively. For instance, when passing [2,14] all points size
              will be either 2 or 14, alternatively.

        c : str, int or array-like, optional
            The color of each point. Possible values are:

            - A single color string referred to by name, RGB or RGBA code,
              for instance 'red' or '#a98d19'.

            - A sequence of color strings referred to by name, RGB or RGBA
              code, which will be used for each point's color recursively. For
              instance ['green','yellow'] all points will be filled in green or
              yellow, alternatively.

            - A column name or position whose values will be used to color the
              marker points according to a colormap.

        **kwargs
            Keyword arguments to pass on to :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        matplotlib.pyplot.scatter : Scatter plot using multiple input data
            formats.

        Examples
        --------
        Let's see how to draw a scatter plot using coordinates from the values
        in a DataFrame's columns.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
            ...                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
            ...                   columns=['length', 'width', 'species'])
            >>> ax1 = df.plot.scatter(x='length',
            ...                       y='width',
            ...                       c='DarkBlue')

        And now with the color determined by a column as well.

        .. plot::
            :context: close-figs

            >>> ax2 = df.plot.scatter(x='length',
            ...                       y='width',
            ...                       c='species',
            ...                       colormap='viridis')
        """
        return self(kind="scatter", x=x, y=y, s=s, c=c, **kwargs)

    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable | None = None,
        gridsize: int | tuple[int, int] | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Generate a hexagonal binning plot.

        Generate a hexagonal binning plot of `x` versus `y`. If `C` is `None`
        (the default), this is a histogram of the number of occurrences
        of the observations at ``(x[i], y[i])``.

        If `C` is specified, specifies values at given coordinates
        ``(x[i], y[i])``. These values are accumulated for each hexagonal
        bin and then reduced according to `reduce_C_function`,
        having as default the NumPy's mean function (:meth:`numpy.mean`).
        (If `C` is specified, it must also be a 1-D sequence
        of the same length as `x` and `y`, or a column label.)

        Parameters
        ----------
        x : int or str
            The column label or position for x points.
        y : int or str
            The column label or position for y points.
        C : int or str, optional
            The column label or position for the value of `(x, y)` point.
        reduce_C_function : callable, default `np.mean`
            Function of one argument that reduces all the values in a bin to
            a single number (e.g. `np.mean`, `np.max`, `np.sum`, `np.std`).
        gridsize : int or tuple of (int, int), default 100
            The number of hexagons in the x-direction.
            The corresponding number of hexagons in the y-direction is
            chosen in a way that the hexagons are approximately regular.
            Alternatively, gridsize can be a tuple with two elements
            specifying the number of hexagons in the x-direction and the
            y-direction.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.AxesSubplot
            The matplotlib ``Axes`` on which the hexbin is plotted.

        See Also
        --------
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.hexbin : Hexagonal binning plot using matplotlib,
            the matplotlib function that is used under the hood.

        Examples
        --------
        The following examples are generated with random data from
        a normal distribution.

        .. plot::
            :context: close-figs

            >>> n = 10000
            >>> df = pd.DataFrame({'x': np.random.randn(n),
            ...                    'y': np.random.randn(n)})
            >>> ax = df.plot.hexbin(x='x', y='y', gridsize=20)

        The next example uses `C` and `np.sum` as `reduce_C_function`.
        Note that `'observations'` values ranges from 1 to 5 but the result
        plot shows values up to more than 25. This is because of the
        `reduce_C_function`.

        .. plot::
            :context: close-figs

            >>> n = 500
            >>> df = pd.DataFrame({
            ...     'coord_x': np.random.uniform(-3, 3, size=n),
            ...     'coord_y': np.random.uniform(30, 50, size=n),
            ...     'observations': np.random.randint(1,5, size=n)
            ...     })
            >>> ax = df.plot.hexbin(x='coord_x',
            ...                     y='coord_y',
            ...                     C='observations',
            ...                     reduce_C_function=np.sum,
            ...                     gridsize=10,
            ...                     cmap="viridis")
        """
        if reduce_C_function is not None:
            kwargs["reduce_C_function"] = reduce_C_function
        if gridsize is not None:
            kwargs["gridsize"] = gridsize

        return self(kind="hexbin", x=x, y=y, C=C, **kwargs)


_backends: dict[str, types.ModuleType] = {}


def _load_backend(backend: str) -> types.ModuleType:
    """
    Load a pandas plotting backend.

    Parameters
    ----------
    backend : str
        The identifier for the backend. Either an entrypoint item registered
        with importlib.metadata, "matplotlib", or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    from importlib.metadata import entry_points

    if backend == "matplotlib":
        # Because matplotlib is an optional dependency and first-party backend,
        # we need to attempt an import here to raise an ImportError if needed.
        try:
            module = importlib.import_module("pandas.plotting._matplotlib")
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting when the "
                'default backend "matplotlib" is selected.'
            ) from None
        return module

    found_backend = False

    eps = entry_points()
    key = "pandas_plotting_backends"
    # entry_points lost dict API ~ PY 3.10
    # https://github.com/python/importlib_metadata/issues/298
    if hasattr(eps, "select"):
        entry = eps.select(group=key)
    else:
        # Argument 2 to "get" of "dict" has incompatible type "Tuple[]";
        # expected "EntryPoints"  [arg-type]
        entry = eps.get(key, ())  # type: ignore[arg-type]
    for entry_point in entry:
        found_backend = entry_point.name == backend
        if found_backend:
            module = entry_point.load()
            break

    if not found_backend:
        # Fall back to unregistered, module name approach.
        try:
            module = importlib.import_module(backend)
            found_backend = True
        except ImportError:
            # We re-raise later on.
            pass

    if found_backend:
        if hasattr(module, "plot"):
            # Validate that the interface is implemented when the option is set,
            # rather than at plot time.
            return module

    raise ValueError(
        f"Could not find plotting backend '{backend}'. Ensure that you've "
        f"installed the package providing the '{backend}' entrypoint, or that "
        "the package has a top-level `.plot` method."
    )


def _get_plot_backend(backend: str | None = None):
    """
    Return the plotting backend to use (e.g. `pandas.plotting._matplotlib`).

    The plotting system of pandas uses matplotlib by default, but the idea here
    is that it can also work with other third-party backends. This function
    returns the module which provides a top-level `.plot` method that will
    actually do the plotting. The backend is specified from a string, which
    either comes from the keyword argument `backend`, or, if not specified, from
    the option `pandas.options.plotting.backend`. All the rest of the code in
    this file uses the backend specified there for the plotting.

    The backend is imported lazily, as matplotlib is a soft dependency, and
    pandas can be used without it being installed.

    Notes
    -----
    Modifies `_backends` with imported backend as a side effect.
    """
    backend_str: str = backend or get_option("plotting.backend")

    if backend_str in _backends:
        return _backends[backend_str]

    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module
<|endoftext|>"""
Plotting public API.

Authors of third-party plotting backends should implement a module with a
public ``plot(data, kind, **kwargs)``. The parameter `data` will contain
the data structure and can be a `Series` or a `DataFrame`. For example,
for ``df.plot()`` the parameter `data` will contain the DataFrame `df`.
In some cases, the data structure is transformed before being sent to
the backend (see PlotAccessor.__call__ in pandas/plotting/_core.py for
the exact transformations).

The parameter `kind` will be one of:

- line
- bar
- barh
- box
- hist
- kde
- area
- pie
- scatter
- hexbin

See the pandas API reference for documentation on each kind of plot.

Any other keyword argument is currently assumed to be backend specific,
but some parameters may be unified and added to the signature in the
future (e.g. `title` which should be useful for any backend).

Currently, all the Matplotlib functions in pandas are accessed through
the selected backend. For example, `pandas.plotting.boxplot` (equivalent
to `DataFrame.boxplot`) is also accessed in the selected backend. This
is expected to change, and the exact API is under discussion. But with
the current version, backends are expected to implement the next functions:

- plot (describe above, used for `Series.plot` and `DataFrame.plot`)
- hist_series and hist_frame (for `Series.hist` and `DataFrame.hist`)
- boxplot (`pandas.plotting.boxplot(df)` equivalent to `DataFrame.boxplot`)
- boxplot_frame and boxplot_frame_groupby
- register and deregister (register converters for the tick formats)
- Plots not called as `Series` and `DataFrame` methods:
  - table
  - andrews_curves
  - autocorrelation_plot
  - bootstrap_plot
  - lag_plot
  - parallel_coordinates
  - radviz
  - scatter_matrix

Use the code in pandas/plotting/_matplotib.py and
https://github.com/pyviz/hvplot as a reference on how to write a backend.

For the discussion about the API see
https://github.com/pandas-dev/pandas/issues/26747.
"""
from pandas.plotting._core import (
    PlotAccessor,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
    hist_frame,
    hist_series,
)
from pandas.plotting._misc import (
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    deregister as deregister_matplotlib_converters,
    lag_plot,
    parallel_coordinates,
    plot_params,
    radviz,
    register as register_matplotlib_converters,
    scatter_matrix,
    table,
)

__all__ = [
    "PlotAccessor",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "hist_frame",
    "hist_series",
    "scatter_matrix",
    "radviz",
    "andrews_curves",
    "bootstrap_plot",
    "parallel_coordinates",
    "lag_plot",
    "autocorrelation_plot",
    "table",
    "plot_params",
    "register_matplotlib_converters",
    "deregister_matplotlib_converters",
]
<|endoftext|>from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.plotting._core import _get_plot_backend

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Mapping,
    )

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.table import Table
    import numpy as np

    from pandas import (
        DataFrame,
        Series,
    )


def table(ax: Axes, data: DataFrame | Series, **kwargs) -> Table:
    """
    Helper function to convert DataFrame and Series to matplotlib.table.

    Parameters
    ----------
    ax : Matplotlib axes object
    data : DataFrame or Series
        Data for table contents.
    **kwargs
        Keyword arguments to be passed to matplotlib.table.table.
        If `rowLabels` or `colLabels` is not specified, data index or column
        name will be used.

    Returns
    -------
    matplotlib table object

    Examples
    --------

    .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> fix, ax = plt.subplots()
            >>> ax.axis('off')
            (0.0, 1.0, 0.0, 1.0)
            >>> table = pd.plotting.table(ax, df, loc='center',
            ...                           cellLoc='center', colWidths=list([.2, .2]))
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.table(
        ax=ax, data=data, rowLabels=None, colLabels=None, **kwargs
    )


def register() -> None:
    """
    Register pandas formatters and converters with matplotlib.

    This function modifies the global ``matplotlib.units.registry``
    dictionary. pandas adds custom converters for

    * pd.Timestamp
    * pd.Period
    * np.datetime64
    * datetime.datetime
    * datetime.date
    * datetime.time

    See Also
    --------
    deregister_matplotlib_converters : Remove pandas formatters and converters.

    Examples
    --------
    .. plot::
       :context: close-figs

        The following line is done automatically by pandas so
        the plot can be rendered:

        >>> pd.plotting.register_matplotlib_converters()

        >>> df = pd.DataFrame({'ts': pd.period_range('2020', periods=2, freq='M'),
        ...                    'y': [1, 2]
        ...                    })
        >>> plot = df.plot.line(x='ts', y='y')

    Unsetting the register manually an error will be raised:

    >>> pd.set_option("plotting.matplotlib.register_converters",
    ...               False)  # doctest: +SKIP
    >>> df.plot.line(x='ts', y='y')  # doctest: +SKIP
    Traceback (most recent call last):
    TypeError: float() argument must be a string or a real number, not 'Period'
    """
    plot_backend = _get_plot_backend("matplotlib")
    plot_backend.register()


def deregister() -> None:
    """
    Remove pandas formatters and converters.

    Removes the custom converters added by :func:`register`. This
    attempts to set the state of the registry back to the state before
    pandas registered its own units. Converters for pandas' own types like
    Timestamp and Period are removed completely. Converters for types
    pandas overwrites, like ``datetime.datetime``, are restored to their
    original value.

    See Also
    --------
    register_matplotlib_converters : Register pandas formatters and converters
        with matplotlib.

    Examples
    --------
    .. plot::
       :context: close-figs

        The following line is done automatically by pandas so
        the plot can be rendered:

        >>> pd.plotting.register_matplotlib_converters()

        >>> df = pd.DataFrame({'ts': pd.period_range('2020', periods=2, freq='M'),
        ...                    'y': [1, 2]
        ...                    })
        >>> plot = df.plot.line(x='ts', y='y')

    Unsetting the register manually an error will be raised:

    >>> pd.set_option("plotting.matplotlib.register_converters",
    ...               False)  # doctest: +SKIP
    >>> df.plot.line(x='ts', y='y')  # doctest: +SKIP
    Traceback (most recent call last):
    TypeError: float() argument must be a string or a real number, not 'Period'
    """
    plot_backend = _get_plot_backend("matplotlib")
    plot_backend.deregister()


def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: Mapping[str, Any] | None = None,
    hist_kwds: Mapping[str, Any] | None = None,
    range_padding: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    Draw a matrix of scatter plots.

    Parameters
    ----------
    frame : DataFrame
    alpha : float, optional
        Amount of transparency applied.
    figsize : (float,float), optional
        A tuple (width, height) in inches.
    ax : Matplotlib axis object, optional
    grid : bool, optional
        Setting this to True will show the grid.
    diagonal : {'hist', 'kde'}
        Pick between 'kde' and 'hist' for either Kernel Density Estimation or
        Histogram plot in the diagonal.
    marker : str, optional
        Matplotlib marker type, default '.'.
    density_kwds : keywords
        Keyword arguments to be passed to kernel density estimate plot.
    hist_kwds : keywords
        Keyword arguments to be passed to hist function.
    range_padding : float, default 0.05
        Relative extension of axis range in x and y with respect to
        (x_max - x_min) or (y_max - y_min).
    **kwargs
        Keyword arguments to be passed to scatter function.

    Returns
    -------
    numpy.ndarray
        A matrix of scatter plots.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
        >>> pd.plotting.scatter_matrix(df, alpha=0.2)
        array([[<Axes: xlabel='A', ylabel='A'>, <Axes: xlabel='B', ylabel='A'>,
                <Axes: xlabel='C', ylabel='A'>, <Axes: xlabel='D', ylabel='A'>],
               [<Axes: xlabel='A', ylabel='B'>, <Axes: xlabel='B', ylabel='B'>,
                <Axes: xlabel='C', ylabel='B'>, <Axes: xlabel='D', ylabel='B'>],
               [<Axes: xlabel='A', ylabel='C'>, <Axes: xlabel='B', ylabel='C'>,
                <Axes: xlabel='C', ylabel='C'>, <Axes: xlabel='D', ylabel='C'>],
               [<Axes: xlabel='A', ylabel='D'>, <Axes: xlabel='B', ylabel='D'>,
                <Axes: xlabel='C', ylabel='D'>, <Axes: xlabel='D', ylabel='D'>]],
              dtype=object)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.scatter_matrix(
        frame=frame,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        grid=grid,
        diagonal=diagonal,
        marker=marker,
        density_kwds=density_kwds,
        hist_kwds=hist_kwds,
        range_padding=range_padding,
        **kwargs,
    )


def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = None,
    color: list[str] | tuple[str, ...] | None = None,
    colormap: Colormap | str | None = None,
    **kwds,
) -> Axes:
    """
    Plot a multidimensional dataset in 2D.

    Each Series in the DataFrame is represented as a evenly distributed
    slice on a circle. Each data point is rendered in the circle according to
    the value on each Series. Highly correlated `Series` in the `DataFrame`
    are placed closer on the unit circle.

    RadViz allow to project a N-dimensional data set into a 2D space where the
    influence of each dimension can be interpreted as a balance between the
    influence of all dimensions.

    More info available at the `original article
    <https://doi.org/10.1145/331770.331775>`_
    describing RadViz.

    Parameters
    ----------
    frame : `DataFrame`
        Object holding the data.
    class_column : str
        Column name containing the name of the data point category.
    ax : :class:`matplotlib.axes.Axes`, optional
        A plot instance to which to add the information.
    color : list[str] or tuple[str], optional
        Assign a color to each category. Example: ['blue', 'green'].
    colormap : str or :class:`matplotlib.colors.Colormap`, default None
        Colormap to select colors from. If string, load colormap with that
        name from matplotlib.
    **kwds
        Options to pass to matplotlib scatter plotting method.

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    See Also
    --------
    pandas.plotting.andrews_curves : Plot clustering visualization.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame(
        ...     {
        ...         'SepalLength': [6.5, 7.7, 5.1, 5.8, 7.6, 5.0, 5.4, 4.6, 6.7, 4.6],
        ...         'SepalWidth': [3.0, 3.8, 3.8, 2.7, 3.0, 2.3, 3.0, 3.2, 3.3, 3.6],
        ...         'PetalLength': [5.5, 6.7, 1.9, 5.1, 6.6, 3.3, 4.5, 1.4, 5.7, 1.0],
        ...         'PetalWidth': [1.8, 2.2, 0.4, 1.9, 2.1, 1.0, 1.5, 0.2, 2.1, 0.2],
        ...         'Category': [
        ...             'virginica',
        ...             'virginica',
        ...             'setosa',
        ...             'virginica',
        ...             'virginica',
        ...             'versicolor',
        ...             'versicolor',
        ...             'setosa',
        ...             'virginica',
        ...             'setosa'
        ...         ]
        ...     }
        ... )
        >>> pd.plotting.radviz(df, 'Category')  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.radviz(
        frame=frame,
        class_column=class_column,
        ax=ax,
        color=color,
        colormap=colormap,
        **kwds,
    )


def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = None,
    samples: int = 200,
    color: list[str] | tuple[str, ...] | None = None,
    colormap: Colormap | str | None = None,
    **kwargs,
) -> Axes:
    """
    Generate a matplotlib plot for visualizing clusters of multivariate data.

    Andrews curves have the functional form:

    .. math::
        f(t) = \\frac{x_1}{\\sqrt{2}} + x_2 \\sin(t) + x_3 \\cos(t) +
        x_4 \\sin(2t) + x_5 \\cos(2t) + \\cdots

    Where :math:`x` coefficients correspond to the values of each dimension
    and :math:`t` is linearly spaced between :math:`-\\pi` and :math:`+\\pi`.
    Each row of frame then corresponds to a single curve.

    Parameters
    ----------
    frame : DataFrame
        Data to be plotted, preferably normalized to (0.0, 1.0).
    class_column : label
        Name of the column containing class names.
    ax : axes object, default None
        Axes to use.
    samples : int
        Number of points to plot in each curve.
    color : str, list[str] or tuple[str], optional
        Colors to use for the different classes. Colors can be strings
        or 3-element floating point RGB values.
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If a string, load colormap with that
        name from matplotlib.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.read_csv(
        ...     'https://raw.githubusercontent.com/pandas-dev/'
        ...     'pandas/main/pandas/tests/io/data/csv/iris.csv'
        ... )
        >>> pd.plotting.andrews_curves(df, 'Name')  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.andrews_curves(
        frame=frame,
        class_column=class_column,
        ax=ax,
        samples=samples,
        color=color,
        colormap=colormap,
        **kwargs,
    )


def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds,
) -> Figure:
    """
    Bootstrap plot on mean, median and mid-range statistics.

    The bootstrap plot is used to estimate the uncertainty of a statistic
    by relying on random sampling with replacement [1]_. This function will
    generate bootstrapping plots for mean, median and mid-range statistics
    for the given number of samples of the given size.

    .. [1] "Bootstrapping (statistics)" in \
    https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    Parameters
    ----------
    series : pandas.Series
        Series from where to get the samplings for the bootstrapping.
    fig : matplotlib.figure.Figure, default None
        If given, it will use the `fig` reference for plotting instead of
        creating a new one with default parameters.
    size : int, default 50
        Number of data points to consider during each sampling. It must be
        less than or equal to the length of the `series`.
    samples : int, default 500
        Number of times the bootstrap procedure is performed.
    **kwds
        Options to pass to matplotlib plotting method.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.

    See Also
    --------
    pandas.DataFrame.plot : Basic plotting for DataFrame objects.
    pandas.Series.plot : Basic plotting for Series objects.

    Examples
    --------
    This example draws a basic bootstrap plot for a Series.

    .. plot::
        :context: close-figs

        >>> s = pd.Series(np.random.uniform(size=100))
        >>> pd.plotting.bootstrap_plot(s)  # doctest: +SKIP
        <Figure size 640x480 with 6 Axes>
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.bootstrap_plot(
        series=series, fig=fig, size=size, samples=samples, **kwds
    )


def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[str] | None = None,
    ax: Axes | None = None,
    color: list[str] | tuple[str, ...] | None = None,
    use_columns: bool = False,
    xticks: list | tuple | None = None,
    colormap: Colormap | str | None = None,
    axvlines: bool = True,
    axvlines_kwds: Mapping[str, Any] | None = None,
    sort_labels: bool = False,
    **kwargs,
) -> Axes:
    """
    Parallel coordinates plotting.

    Parameters
    ----------
    frame : DataFrame
    class_column : str
        Column name containing class names.
    cols : list, optional
        A list of column names to use.
    ax : matplotlib.axis, optional
        Matplotlib axis object.
    color : list or tuple, optional
        Colors to use for the different classes.
    use_columns : bool, optional
        If true, columns will be used as xticks.
    xticks : list or tuple, optional
        A list of values to use for xticks.
    colormap : str or matplotlib colormap, default None
        Colormap to use for line colors.
    axvlines : bool, optional
        If true, vertical lines will be added at each xtick.
    axvlines_kwds : keywords, optional
        Options to be passed to axvline method for vertical lines.
    sort_labels : bool, default False
        Sort class_column labels, useful when assigning colors.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.read_csv(
        ...     'https://raw.githubusercontent.com/pandas-dev/'
        ...     'pandas/main/pandas/tests/io/data/csv/iris.csv'
        ... )
        >>> pd.plotting.parallel_coordinates(
        ...     df, 'Name', color=('#556270', '#4ECDC4', '#C7F464')
        ... )  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.parallel_coordinates(
        frame=frame,
        class_column=class_column,
        cols=cols,
        ax=ax,
        color=color,
        use_columns=use_columns,
        xticks=xticks,
        colormap=colormap,
        axvlines=axvlines,
        axvlines_kwds=axvlines_kwds,
        sort_labels=sort_labels,
        **kwargs,
    )


def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
    """
    Lag plot for time series.

    Parameters
    ----------
    series : Series
        The time series to visualize.
    lag : int, default 1
        Lag length of the scatter plot.
    ax : Matplotlib axis object, optional
        The matplotlib axis object to use.
    **kwds
        Matplotlib scatter method keyword arguments.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    Lag plots are most commonly used to look for patterns in time series data.

    Given the following time series

    .. plot::
        :context: close-figs

        >>> np.random.seed(5)
        >>> x = np.cumsum(np.random.normal(loc=1, scale=5, size=50))
        >>> s = pd.Series(x)
        >>> s.plot()  # doctest: +SKIP

    A lag plot with ``lag=1`` returns

    .. plot::
        :context: close-figs

        >>> pd.plotting.lag_plot(s, lag=1)
        <Axes: xlabel='y(t)', ylabel='y(t + 1)'>
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.lag_plot(series=series, lag=lag, ax=ax, **kwds)


def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwargs) -> Axes:
    """
    Autocorrelation plot for time series.

    Parameters
    ----------
    series : Series
        The time series to visualize.
    ax : Matplotlib axis object, optional
        The matplotlib axis object to use.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    The horizontal lines in the plot correspond to 95% and 99% confidence bands.

    The dashed line is 99% confidence band.

    .. plot::
        :context: close-figs

        >>> spacing = np.linspace(-9 * np.pi, 9 * np.pi, num=1000)
        >>> s = pd.Series(0.7 * np.random.rand(1000) + 0.3 * np.sin(spacing))
        >>> pd.plotting.autocorrelation_plot(s)  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.autocorrelation_plot(series=series, ax=ax, **kwargs)


class _Options(dict):
    """
    Stores pandas plotting options.

    Allows for parameter aliasing so you can just use parameter names that are
    the same as the plot function parameters, but is stored in a canonical
    format that makes it easy to breakdown into groups later.

    Examples
    --------

    .. plot::
            :context: close-figs

             >>> np.random.seed(42)
             >>> df = pd.DataFrame({'A': np.random.randn(10),
             ...                   'B': np.random.randn(10)},
             ...                   index=pd.date_range("1/1/2000",
             ...                   freq='4MS', periods=10))
             >>> with pd.plotting.plot_params.use("x_compat", True):
             ...     _ = df["A"].plot(color="r")
             ...     _ = df["B"].plot(color="g")
    """

    # alias so the names are same as plotting method parameter names
    _ALIASES = {"x_compat": "xaxis.compat"}
    _DEFAULT_KEYS = ["xaxis.compat"]

    def __init__(self, deprecated: bool = False) -> None:
        self._deprecated = deprecated
        super().__setitem__("xaxis.compat", False)

    def __getitem__(self, key):
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f"{key} is not a valid pandas plotting option")
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        key = self._get_canonical_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(f"Cannot remove default parameter {key}")
        super().__delitem__(key)

    def __contains__(self, key) -> bool:
        key = self._get_canonical_key(key)
        return super().__contains__(key)

    def reset(self) -> None:
        """
        Reset the option store to its initial state

        Returns
        -------
        None
        """
        # error: Cannot access "__init__" directly
        self.__init__()  # type: ignore[misc]

    def _get_canonical_key(self, key):
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key, value) -> Generator[_Options, None, None]:
        """
        Temporarily set a parameter value using the with statement.
        Aliasing allowed.
        """
        old_value = self[key]
        try:
            self[key] = value
            yield self
        finally:
            self[key] = old_value


plot_params = _Options()
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
)
import warnings

import matplotlib as mpl
from matplotlib.artist import setp
import numpy as np

from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike

import pandas as pd
import pandas.core.common as com
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from pandas._typing import MatplotlibColor


def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None:
    """Set the tick labels of a given axis.

    Due to https://github.com/matplotlib/matplotlib/pull/17266, we need to handle the
    case of repeated ticks (due to `FixedLocator`) and thus we duplicate the number of
    labels.
    """
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    if len(ticks) != len(labels):
        i, remainder = divmod(len(ticks), len(labels))
        if Version(mpl.__version__) < Version("3.10"):
            assert remainder == 0, remainder
        labels *= i
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)


class BoxPlot(LinePlot):
    @property
    def _kind(self) -> Literal["box"]:
        return "box"

    _layout_type = "horizontal"

    _valid_return_types = (None, "axes", "dict", "both")

    class BP(NamedTuple):
        # namedtuple to hold results
        ax: Axes
        lines: dict[str, list[Line2D]]

    def __init__(self, data, return_type: str = "axes", **kwargs) -> None:
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")

        self.return_type = return_type
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called

        if self.subplots:
            # Disable label ax sharing. Otherwise, all subplots shows last
            # column label
            if self.orientation == "vertical":
                self.sharex = False
            else:
                self.sharey = False

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Axes, y: np.ndarray, column_num=None, return_type: str = "axes", **kwds
    ):
        ys: np.ndarray | list[np.ndarray]
        if y.ndim == 2:
            ys = [remove_na_arraylike(v) for v in y]
            # Boxplot fails with empty arrays, so need to add a NaN
            #   if any cols are empty
            # GH 8181
            ys = [v if v.size > 0 else np.array([np.nan]) for v in ys]
        else:
            ys = remove_na_arraylike(y)
        bp = ax.boxplot(ys, **kwds)

        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _validate_color_args(self, color, colormap):
        if color is lib.no_default:
            return None

        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used "
                "simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        if isinstance(color, dict):
            valid_keys = ["boxes", "whiskers", "medians", "caps"]
            for key in color:
                if key not in valid_keys:
                    raise ValueError(
                        f"color dict contains invalid key '{key}'. "
                        f"The key must be either {valid_keys}"
                    )
        return color

    @cache_readonly
    def _color_attrs(self):
        # get standard colors for default
        # use 2 colors by default, for box/whisker and median
        # flier colors isn't needed here
        # because it can be specified by ``sym`` kw
        return get_standard_colors(num_colors=3, colormap=self.colormap, color=None)

    @cache_readonly
    def _boxes_c(self):
        return self._color_attrs[0]

    @cache_readonly
    def _whiskers_c(self):
        return self._color_attrs[0]

    @cache_readonly
    def _medians_c(self):
        return self._color_attrs[2]

    @cache_readonly
    def _caps_c(self):
        return self._color_attrs[0]

    def _get_colors(
        self,
        num_colors=None,
        color_kwds: dict[str, MatplotlibColor]
        | MatplotlibColor
        | Collection[MatplotlibColor]
        | None = "color",
    ) -> None:
        pass

    def maybe_color_bp(self, bp) -> None:
        if isinstance(self.color, dict):
            boxes = self.color.get("boxes", self._boxes_c)
            whiskers = self.color.get("whiskers", self._whiskers_c)
            medians = self.color.get("medians", self._medians_c)
            caps = self.color.get("caps", self._caps_c)
        else:
            # Other types are forwarded to matplotlib
            # If None, use default colors
            boxes = self.color or self._boxes_c
            whiskers = self.color or self._whiskers_c
            medians = self.color or self._medians_c
            caps = self.color or self._caps_c

        color_tup = (boxes, whiskers, medians, caps)
        maybe_color_bp(bp, color_tup=color_tup, **self.kwds)

    def _make_plot(self, fig: Figure) -> None:
        if self.subplots:
            self._return_obj = pd.Series(dtype=object)

            # Re-create iterated data if `by` is assigned by users
            data = (
                create_iter_data_given_by(self.data, self._kind)
                if self.by is not None
                else self.data
            )

