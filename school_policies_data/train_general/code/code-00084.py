            md = self.read_metadata(n)
            meta = "category" if md is not None else None
            dc = GenericDataIndexableCol(
                name=n,
                pos=i,
                values=[n],
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            _indexables.append(dc)

        return _indexables

    # error: Signature of "write" incompatible with supertype "AppendableTable"
    def write(self, **kwargs) -> None:  # type: ignore[override]
        raise NotImplementedError("cannot write on an generic table")


class AppendableMultiFrameTable(AppendableFrameTable):
    """a frame with a multi-index"""

    table_type = "appendable_multiframe"
    obj_type = DataFrame
    ndim = 2
    _re_levels = re.compile(r"^level_\d+$")

    @property
    def table_type_short(self) -> str:
        return "appendable_multi"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)

        # remove names for 'level_%d'
        df.index = df.index.set_names(
            [None if self._re_levels.search(name) else name for name in df.index.names]
        )

        return df


def _reindex_axis(
    obj: DataFrame, axis: AxisInt, labels: Index, other=None
) -> DataFrame:
    ax = obj._get_axis(axis)
    labels = ensure_index(labels)

    # try not to reindex even if other is provided
    # if it equals our current index
    if other is not None:
        other = ensure_index(other)
    if (other is None or labels.equals(other)) and labels.equals(ax):
        return obj

    labels = ensure_index(labels.unique())
    if other is not None:
        labels = ensure_index(other.unique()).intersection(labels, sort=False)
    if not labels.equals(ax):
        slicer: list[slice | Index] = [slice(None, None)] * obj.ndim
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj


# tz to/from coercion


def _get_tz(tz: tzinfo) -> str | tzinfo:
    """for a tz-aware type, return an encoded zone"""
    zone = timezones.get_timezone(tz)
    return zone


@overload
def _set_tz(
    values: np.ndarray | Index, tz: str | tzinfo, coerce: bool = False
) -> DatetimeIndex:
    ...


@overload
def _set_tz(values: np.ndarray | Index, tz: None, coerce: bool = False) -> np.ndarray:
    ...


def _set_tz(
    values: np.ndarray | Index, tz: str | tzinfo | None, coerce: bool = False
) -> np.ndarray | DatetimeIndex:
    """
    coerce the values to a DatetimeIndex if tz is set
    preserve the input shape if possible

    Parameters
    ----------
    values : ndarray or Index
    tz : str or tzinfo
    coerce : if we do not have a passed timezone, coerce to M8[ns] ndarray
    """
    if isinstance(values, DatetimeIndex):
        # If values is tzaware, the tz gets dropped in the values.ravel()
        #  call below (which returns an ndarray).  So we are only non-lossy
        #  if `tz` matches `values.tz`.
        assert values.tz is None or values.tz == tz
        if values.tz is not None:
            return values

    if tz is not None:
        if isinstance(values, DatetimeIndex):
            name = values.name
        else:
            name = None
            values = values.ravel()

        tz = _ensure_decoded(tz)
        values = DatetimeIndex(values, name=name)
        values = values.tz_localize("UTC").tz_convert(tz)
    elif coerce:
        values = np.asarray(values, dtype="M8[ns]")

    # error: Incompatible return value type (got "Union[ndarray, Index]",
    # expected "Union[ndarray, DatetimeIndex]")
    return values  # type: ignore[return-value]


def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol:
    assert isinstance(name, str)

    index_name = index.name
    # error: Argument 1 to "_get_data_and_dtype_name" has incompatible type "Index";
    # expected "Union[ExtensionArray, ndarray]"
    converted, dtype_name = _get_data_and_dtype_name(index)  # type: ignore[arg-type]
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)

    if (
        lib.is_np_dtype(index.dtype, "iu")
        or needs_i8_conversion(index.dtype)
        or is_bool_dtype(index.dtype)
    ):
        # Includes Index, RangeIndex, DatetimeIndex, TimedeltaIndex, PeriodIndex,
        #  in which case "kind" is "integer", "integer", "datetime64",
        #  "timedelta64", and "integer", respectively.
        return IndexCol(
            name,
            values=converted,
            kind=kind,
            typ=atom,
            freq=getattr(index, "freq", None),
            tz=getattr(index, "tz", None),
            index_name=index_name,
        )

    if isinstance(index, MultiIndex):
        raise TypeError("MultiIndex not supported here!")

    inferred_type = lib.infer_dtype(index, skipna=False)
    # we won't get inferred_type of "datetime64" or "timedelta64" as these
    #  would go through the DatetimeIndex/TimedeltaIndex paths above

    values = np.asarray(index)

    if inferred_type == "date":
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(
            name, converted, "date", _tables().Time32Col(), index_name=index_name
        )
    elif inferred_type == "string":
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(
            name,
            converted,
            "string",
            _tables().StringCol(itemsize),
            index_name=index_name,
        )

    elif inferred_type in ["integer", "floating"]:
        return IndexCol(
            name, values=converted, kind=kind, typ=atom, index_name=index_name
        )
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == "object", kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)


def _unconvert_index(data, kind: str, encoding: str, errors: str) -> np.ndarray | Index:
    index: Index | np.ndarray

    if kind.startswith("datetime64"):
        if kind == "datetime64":
            # created before we stored resolution information
            index = DatetimeIndex(data)
        else:
            index = DatetimeIndex(data.view(kind))
    elif kind == "timedelta64":
        index = TimedeltaIndex(data)
    elif kind == "date":
        try:
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ("integer", "float", "bool"):
        index = np.asarray(data)
    elif kind in ("string"):
        index = _unconvert_string_array(
            data, nan_rep=None, encoding=encoding, errors=errors
        )
    elif kind == "object":
        index = np.asarray(data[0])
    else:  # pragma: no cover
        raise ValueError(f"unrecognized index type {kind}")
    return index


def _maybe_convert_for_string_atom(
    name: str,
    bvalues: ArrayLike,
    existing_col,
    min_itemsize,
    nan_rep,
    encoding,
    errors,
    columns: list[str],
):
    if isinstance(bvalues.dtype, StringDtype):
        # "ndarray[Any, Any]" has no attribute "to_numpy"
        bvalues = bvalues.to_numpy()  # type: ignore[union-attr]
    if bvalues.dtype != object:
        return bvalues

    bvalues = cast(np.ndarray, bvalues)

    dtype_name = bvalues.dtype.name
    inferred_type = lib.infer_dtype(bvalues, skipna=False)

    if inferred_type == "date":
        raise TypeError("[date] is not implemented as a table column")
    if inferred_type == "datetime":
        # after GH#8260
        # this only would be hit for a multi-timezone dtype which is an error
        raise TypeError(
            "too many timezones in this block, create separate data columns"
        )

    if not (inferred_type == "string" or dtype_name == "object"):
        return bvalues

    mask = isna(bvalues)
    data = bvalues.copy()
    data[mask] = nan_rep

    if existing_col and mask.any() and len(nan_rep) > existing_col.itemsize:
        raise ValueError("NaN representation is too large for existing column size")

    # see if we have a valid string type
    inferred_type = lib.infer_dtype(data, skipna=False)
    if inferred_type != "string":
        # we cannot serialize this data, so report an exception on a column
        # by column basis

        # expected behaviour:
        # search block for a non-string object column by column
        for i in range(data.shape[0]):
            col = data[i]
            inferred_type = lib.infer_dtype(col, skipna=False)
            if inferred_type != "string":
                error_column_label = columns[i] if len(columns) > i else f"No.{i}"
                raise TypeError(
                    f"Cannot serialize the column [{error_column_label}]\n"
                    f"because its data contents are not [string] but "
                    f"[{inferred_type}] object dtype"
                )

    # itemsize is the maximum length of a string (along any dimension)

    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    itemsize = data_converted.itemsize

    # specified min_itemsize?
    if isinstance(min_itemsize, dict):
        min_itemsize = int(min_itemsize.get(name) or min_itemsize.get("values") or 0)
    itemsize = max(min_itemsize or 0, itemsize)

    # check for column in the values conflicts
    if existing_col is not None:
        eci = existing_col.validate_col(itemsize)
        if eci is not None and eci > itemsize:
            itemsize = eci

    data_converted = data_converted.astype(f"|S{itemsize}", copy=False)
    return data_converted


def _convert_string_array(data: np.ndarray, encoding: str, errors: str) -> np.ndarray:
    """
    Take a string-like that is object dtype and coerce to a fixed size string type.

    Parameters
    ----------
    data : np.ndarray[object]
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[fixed-length-string]
    """
    # encode if needed
    if len(data):
        data = (
            Series(data.ravel(), copy=False, dtype="object")
            .str.encode(encoding, errors)
            ._values.reshape(data.shape)
        )

    # create the sized dtype
    ensured = ensure_object(data.ravel())
    itemsize = max(1, libwriters.max_len_string_array(ensured))

    data = np.asarray(data, dtype=f"S{itemsize}")
    return data


def _unconvert_string_array(
    data: np.ndarray, nan_rep, encoding: str, errors: str
) -> np.ndarray:
    """
    Inverse of _convert_string_array.

    Parameters
    ----------
    data : np.ndarray[fixed-length-string]
    nan_rep : the storage repr of NaN
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[object]
        Decoded data.
    """
    shape = data.shape
    data = np.asarray(data.ravel(), dtype=object)

    if len(data):
        itemsize = libwriters.max_len_string_array(ensure_object(data))
        dtype = f"U{itemsize}"

        if isinstance(data[0], bytes):
            ser = Series(data, copy=False).str.decode(
                encoding, errors=errors, dtype="object"
            )
            data = ser.to_numpy()
            data.flags.writeable = True
        else:
            data = data.astype(dtype, copy=False).astype(object, copy=False)

    if nan_rep is None:
        nan_rep = "nan"

    libwriters.string_array_replace_from_nan_rep(data, nan_rep)
    return data.reshape(shape)


def _maybe_convert(values: np.ndarray, val_kind: str, encoding: str, errors: str):
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values


def _get_converter(kind: str, encoding: str, errors: str):
    if kind == "datetime64":
        return lambda x: np.asarray(x, dtype="M8[ns]")
    elif "datetime64" in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == "string":
        return lambda x: _unconvert_string_array(
            x, nan_rep=None, encoding=encoding, errors=errors
        )
    else:  # pragma: no cover
        raise ValueError(f"invalid kind {kind}")


def _need_convert(kind: str) -> bool:
    if kind in ("datetime64", "string") or "datetime64" in kind:
        return True
    return False


def _maybe_adjust_name(name: str, version: Sequence[int]) -> str:
    """
    Prior to 0.10.1, we named values blocks like: values_block_0 an the
    name values_0, adjust the given name if necessary.

    Parameters
    ----------
    name : str
    version : Tuple[int, int, int]

    Returns
    -------
    str
    """
    if isinstance(version, str) or len(version) < 3:
        raise ValueError("Version is incorrect, expected sequence of 3 integers.")

    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r"values_block_(\d+)", name)
        if m:
            grp = m.groups()[0]
            name = f"values_{grp}"
    return name


def _dtype_to_kind(dtype_str: str) -> str:
    """
    Find the "kind" string describing the given dtype name.
    """
    dtype_str = _ensure_decoded(dtype_str)

    if dtype_str.startswith(("string", "bytes")):
        kind = "string"
    elif dtype_str.startswith("float"):
        kind = "float"
    elif dtype_str.startswith("complex"):
        kind = "complex"
    elif dtype_str.startswith(("int", "uint")):
        kind = "integer"
    elif dtype_str.startswith("datetime64"):
        kind = dtype_str
    elif dtype_str.startswith("timedelta"):
        kind = "timedelta64"
    elif dtype_str.startswith("bool"):
        kind = "bool"
    elif dtype_str.startswith("category"):
        kind = "category"
    elif dtype_str.startswith("period"):
        # We store the `freq` attr so we can restore from integers
        kind = "integer"
    elif dtype_str == "object":
        kind = "object"
    elif dtype_str == "str":
        kind = "str"
    else:
        raise ValueError(f"cannot interpret dtype of [{dtype_str}]")

    return kind


def _get_data_and_dtype_name(data: ArrayLike):
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(data, Categorical):
        data = data.codes

    if isinstance(data.dtype, DatetimeTZDtype):
        # For datetime64tz we need to drop the TZ in tests TODO: why?
        dtype_name = f"datetime64[{data.dtype.unit}]"
    else:
        dtype_name = data.dtype.name

    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
        # TODO: we used to reshape for the dt64tz case, but no longer
        #  doing that doesn't seem to break anything.  why?

    elif isinstance(data, PeriodIndex):
        data = data.asi8

    data = np.asarray(data)
    return data, dtype_name


class Selection:
    """
    Carries out a selection operation on a tables.Table object.

    Parameters
    ----------
    table : a Table object
    where : list of Terms (or convertible to)
    start, stop: indices to start and/or stop selection

    """

    def __init__(
        self,
        table: Table,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> None:
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None

        if is_list_like(where):
            # see if we have a passed coordinate like
            with suppress(ValueError):
                inferred = lib.infer_dtype(where, skipna=False)
                if inferred in ("integer", "boolean"):
                    where = np.asarray(where)
                    if where.dtype == np.bool_:
                        start, stop = self.start, self.stop
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = self.table.nrows
                        self.coordinates = np.arange(start, stop)[where]
                    elif issubclass(where.dtype.type, np.integer):
                        if (self.start is not None and (where < self.start).any()) or (
                            self.stop is not None and (where >= self.stop).any()
                        ):
                            raise ValueError(
                                "where must have index locations >= start and < stop"
                            )
                        self.coordinates = where

        if self.coordinates is None:
            self.terms = self.generate(where)

            # create the numexpr & the filter
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    def generate(self, where):
        """where can be a : dict,list,tuple,string"""
        if where is None:
            return None

        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            # raise a nice message, suggesting that the user should use
            # data_columns
            qkeys = ",".join(q.keys())
            msg = dedent(
                f"""\
                The passed where expression: {where}
                            contains an invalid variable reference
                            all of the variable references must be a reference to
                            an axis (e.g. 'index' or 'columns'), or a data_column
                            The currently defined references are: {qkeys}
                """
            )
            raise ValueError(msg) from err

    def select(self):
        """
        generate the selection
        """
        if self.condition is not None:
            return self.table.table.read_where(
                self.condition.format(), start=self.start, stop=self.stop
            )
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self):
        """
        generate the selection
        """
        start, stop = self.start, self.stop
        nrows = self.table.nrows
        if start is None:
            start = 0
        elif start < 0:
            start += nrows
        if stop is None:
            stop = nrows
        elif stop < 0:
            stop += nrows

        if self.condition is not None:
            return self.table.table.get_where_list(
                self.condition.format(), start=start, stop=stop, sort=True
            )
        elif self.coordinates is not None:
            return self.coordinates

        return np.arange(start, stop)
<|endoftext|>""" pickle compat """
from __future__ import annotations

import pickle
from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

from pandas.compat import pickle_compat as pc
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadPickleBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Series,
    )


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "filepath_or_buffer",
)
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePath | WriteBuffer[bytes],
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions | None = None,
) -> None:
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : any object
        Any python object.
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function.
        Also accepts URL. URL has to be of S3 or GCS.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible
        values for this parameter depend on the version of Python. For Python
        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.
        For Python >= 3.4, 4 is a valid value. A negative value for the
        protocol parameter is equivalent to setting its value to
        HIGHEST_PROTOCOL.

    {storage_options}

        .. [1] https://docs.python.org/3/library/pickle.html

    See Also
    --------
    read_pickle : Load pickled pandas object (or any object) from file.
    DataFrame.to_hdf : Write DataFrame to an HDF5 file.
    DataFrame.to_sql : Write DataFrame to a SQL database.
    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

    Examples
    --------
    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """  # noqa: E501
    if protocol < 0:
        protocol = pickle.HIGHEST_PROTOCOL

    with get_handle(
        filepath_or_buffer,
        "wb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # letting pickle write directly to the buffer is more memory-efficient
        pickle.dump(obj, handles.handle, protocol=protocol)


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "filepath_or_buffer",
)
def read_pickle(
    filepath_or_buffer: FilePath | ReadPickleBuffer,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
) -> DataFrame | Series:
    """
    Load pickled pandas object (or any object) from file.

    .. warning::

       Loading pickled data received from untrusted sources can be
       unsafe. See `here <https://docs.python.org/3/library/pickle.html>`__.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``readlines()`` function.
        Also accepts URL. URL is not limited to S3 and GCS.

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    Returns
    -------
    same type as object stored in file

    See Also
    --------
    DataFrame.to_pickle : Pickle (serialize) DataFrame object to file.
    Series.to_pickle : Pickle (serialize) Series object to file.
    read_hdf : Read HDF5 file into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.
    read_parquet : Load a parquet object, returning a DataFrame.

    Notes
    -----
    read_pickle is only guaranteed to be backwards compatible to pandas 0.20.3
    provided the object was serialized with to_pickle.

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {{"foo": range(5), "bar": range(5, 10)}}
    ...    )  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """
    excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
    with get_handle(
        filepath_or_buffer,
        "rb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # 1) try standard library Pickle
        # 2) try pickle_compat (older pandas version) to handle subclass changes
        # 3) try pickle_compat with latin-1 encoding upon a UnicodeDecodeError

        try:
            # TypeError for Cython complaints about object.__new__ vs Tick.__new__
            try:
                with warnings.catch_warnings(record=True):
                    # We want to silence any warnings about, e.g. moved modules.
                    warnings.simplefilter("ignore", Warning)
                    return pickle.load(handles.handle)
            except excs_to_catch:
                # e.g.
                #  "No module named 'pandas.core.sparse.series'"
                #  "Can't get attribute '__nat_unpickle' on <module 'pandas._libs.tslib"
                return pc.load(handles.handle, encoding=None)
        except UnicodeDecodeError:
            # e.g. can occur for files written in py27; see GH#28645 and GH#31988
            return pc.load(handles.handle, encoding="latin-1")
<|endoftext|>""" feather-format compat """
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend

from pandas.core.api import DataFrame
from pandas.core.shared_docs import _shared_docs

from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import get_handle

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )


@doc(storage_options=_shared_docs["storage_options"])
def to_feather(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes],
    storage_options: StorageOptions | None = None,
    **kwargs: Any,
) -> None:
    """
    Write a DataFrame to the binary Feather format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, or file-like object
    {storage_options}
    **kwargs :
        Additional keywords passed to `pyarrow.feather.write_feather`.

    """
    import_optional_dependency("pyarrow")
    from pyarrow import feather

    if not isinstance(df, DataFrame):
        raise ValueError("feather only support IO with DataFrames")

    with get_handle(
        path, "wb", storage_options=storage_options, is_text=False
    ) as handles:
        feather.write_feather(df, handles.handle, **kwargs)


@doc(storage_options=_shared_docs["storage_options"])
def read_feather(
    path: FilePath | ReadBuffer[bytes],
    columns: Sequence[Hashable] | None = None,
    use_threads: bool = True,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame:
    """
    Load a feather-format object from the file path.

    Parameters
    ----------
    path : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: ``file://localhost/path/to/table.feather``.
    columns : sequence, default None
        If not provided, all columns are read.
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.
    {storage_options}

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    type of object stored in file

    Examples
    --------
    >>> df = pd.read_feather("path/to/file.feather")  # doctest: +SKIP
    """
    import_optional_dependency("pyarrow")
    from pyarrow import feather

    # import utils to register the pyarrow extension types
    import pandas.core.arrays.arrow.extension_types  # pyright: ignore[reportUnusedImport] # noqa: F401

    check_dtype_backend(dtype_backend)

    with get_handle(
        path, "rb", storage_options=storage_options, is_text=False
    ) as handles:
        if dtype_backend is lib.no_default and not using_string_dtype():
            return feather.read_feather(
                handles.handle, columns=columns, use_threads=bool(use_threads)
            )

        pa_table = feather.read_table(
            handles.handle, columns=columns, use_threads=bool(use_threads)
        )
        return arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)
<|endoftext|>""" parquet compat """
from __future__ import annotations

import io
import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)
import warnings
from warnings import (
    catch_warnings,
    filterwarnings,
)

from pandas._config.config import _get_option

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas import (
    DataFrame,
    get_option,
)
from pandas.core.shared_docs import _shared_docs

from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import (
    IOHandles,
    get_handle,
    is_fsspec_url,
    is_url,
    stringify_path,
)

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )


def get_engine(engine: str) -> BaseImpl:
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.parquet.engine")

    if engine == "auto":
        # try engines in this order
        engine_classes = [PyArrowImpl, FastParquetImpl]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'pyarrow', 'fastparquet'.\n"
            "A suitable version of "
            "pyarrow or fastparquet is required for parquet "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "pyarrow":
        return PyArrowImpl()
    elif engine == "fastparquet":
        return FastParquetImpl()

    raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")


def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
    fs: Any,
    storage_options: StorageOptions | None = None,
    mode: str = "rb",
    is_dir: bool = False,
) -> tuple[
    FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any
]:
    """File handling for PyArrow."""
    path_or_handle = stringify_path(path)
    if fs is not None:
        pa_fs = import_optional_dependency("pyarrow.fs", errors="ignore")
        fsspec = import_optional_dependency("fsspec", errors="ignore")
        if pa_fs is not None and isinstance(fs, pa_fs.FileSystem):
            if storage_options:
                raise NotImplementedError(
                    "storage_options not supported with a pyarrow FileSystem."
                )
        elif fsspec is not None and isinstance(fs, fsspec.spec.AbstractFileSystem):
            pass
        else:
            raise ValueError(
                f"filesystem must be a pyarrow or fsspec FileSystem, "
                f"not a {type(fs).__name__}"
            )
    if is_fsspec_url(path_or_handle) and fs is None:
        if storage_options is None:
            pa = import_optional_dependency("pyarrow")
            pa_fs = import_optional_dependency("pyarrow.fs")

            try:
                fs, path_or_handle = pa_fs.FileSystem.from_uri(path)
            except (TypeError, pa.ArrowInvalid):
                pass
        if fs is None:
            fsspec = import_optional_dependency("fsspec")
            fs, path_or_handle = fsspec.core.url_to_fs(
                path_or_handle, **(storage_options or {})
            )
    elif storage_options and (not is_url(path_or_handle) or mode != "rb"):
        # can't write to a remote url
        # without making use of fsspec at the moment
        raise ValueError("storage_options passed with buffer, or non-supported URL")

    handles = None
    if (
        not fs
        and not is_dir
        and isinstance(path_or_handle, str)
        and not os.path.isdir(path_or_handle)
    ):
        # use get_handle only when we are very certain that it is not a directory
        # fsspec resources can also point to directories
        # this branch is used for example when reading from non-fsspec URLs
        handles = get_handle(
            path_or_handle, mode, is_text=False, storage_options=storage_options
        )
        fs = None
        path_or_handle = handles.handle
    return path_or_handle, handles, fs


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None:
        if not isinstance(df, DataFrame):
            raise ValueError("to_parquet only supports IO with DataFrames")

    def write(self, df: DataFrame, path, compression, **kwargs):
        raise AbstractMethodError(self)

    def read(self, path, columns=None, **kwargs) -> DataFrame:
        raise AbstractMethodError(self)


class PyArrowImpl(BaseImpl):
    def __init__(self) -> None:
        import_optional_dependency(
            "pyarrow", extra="pyarrow is required for parquet support."
        )
        import pyarrow.parquet

        # import utils to register the pyarrow extension types
        import pandas.core.arrays.arrow.extension_types  # pyright: ignore[reportUnusedImport] # noqa: F401

        self.api = pyarrow

    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: str | None = "snappy",
        index: bool | None = None,
        storage_options: StorageOptions | None = None,
        partition_cols: list[str] | None = None,
        filesystem=None,
        **kwargs,
    ) -> None:
        self.validate_dataframe(df)

        from_pandas_kwargs: dict[str, Any] = {"schema": kwargs.pop("schema", None)}
        if index is not None:
            from_pandas_kwargs["preserve_index"] = index

        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)

        if df.attrs:
            df_metadata = {"PANDAS_ATTRS": json.dumps(df.attrs)}
            existing_metadata = table.schema.metadata
            merged_metadata = {**existing_metadata, **df_metadata}
            table = table.replace_schema_metadata(merged_metadata)

        path_or_handle, handles, filesystem = _get_path_or_handle(
            path,
            filesystem,
            storage_options=storage_options,
            mode="wb",
            is_dir=partition_cols is not None,
        )
        if (
            isinstance(path_or_handle, io.BufferedWriter)
            and hasattr(path_or_handle, "name")
            and isinstance(path_or_handle.name, (str, bytes))
        ):
            if isinstance(path_or_handle.name, bytes):
                path_or_handle = path_or_handle.name.decode()
            else:
                path_or_handle = path_or_handle.name

        try:
            if partition_cols is not None:
                # writes to multiple files under the given path
                self.api.parquet.write_to_dataset(
                    table,
                    path_or_handle,
                    compression=compression,
                    partition_cols=partition_cols,
                    filesystem=filesystem,
                    **kwargs,
                )
            else:
                # write to single output file
                self.api.parquet.write_table(
                    table,
                    path_or_handle,
                    compression=compression,
                    filesystem=filesystem,
                    **kwargs,
                )
        finally:
            if handles is not None:
                handles.close()

    def read(
        self,
        path,
        columns=None,
        filters=None,
        use_nullable_dtypes: bool = False,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ) -> DataFrame:
        kwargs["use_pandas_metadata"] = True

        to_pandas_kwargs = {}

        manager = _get_option("mode.data_manager", silent=True)
        if manager == "array":
            to_pandas_kwargs["split_blocks"] = True
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path,
            filesystem,
            storage_options=storage_options,
            mode="rb",
        )
        try:
            pa_table = self.api.parquet.read_table(
                path_or_handle,
                columns=columns,
                filesystem=filesystem,
                filters=filters,
                **kwargs,
            )

            with catch_warnings():
                filterwarnings(
                    "ignore",
                    "make_block is deprecated",
                    DeprecationWarning,
                )
                result = arrow_table_to_pandas(
                    pa_table,
                    dtype_backend=dtype_backend,
                    to_pandas_kwargs=to_pandas_kwargs,
                )

            if manager == "array":
                result = result._as_manager("array", copy=False)

            if pa_table.schema.metadata:
                if b"PANDAS_ATTRS" in pa_table.schema.metadata:
                    df_metadata = pa_table.schema.metadata[b"PANDAS_ATTRS"]
                    result.attrs = json.loads(df_metadata)
            return result
        finally:
            if handles is not None:
                handles.close()


class FastParquetImpl(BaseImpl):
    def __init__(self) -> None:
        # since pandas is a dependency of fastparquet
        # we need to import on first use
        fastparquet = import_optional_dependency(
            "fastparquet", extra="fastparquet is required for parquet support."
        )
        self.api = fastparquet

    def write(
        self,
        df: DataFrame,
        path,
        compression: Literal["snappy", "gzip", "brotli"] | None = "snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ) -> None:
        self.validate_dataframe(df)

        if "partition_on" in kwargs and partition_cols is not None:
            raise ValueError(
                "Cannot use both partition_on and "
                "partition_cols. Use partition_cols for partitioning data"
            )
        if "partition_on" in kwargs:
            partition_cols = kwargs.pop("partition_on")

        if partition_cols is not None:
            kwargs["file_scheme"] = "hive"

        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )

        # cannot use get_handle as write() does not accept file buffers
        path = stringify_path(path)
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")

            # if filesystem is provided by fsspec, file must be opened in 'wb' mode.
            kwargs["open_with"] = lambda path, _: fsspec.open(
                path, "wb", **(storage_options or {})
            ).open()
        elif storage_options:
            raise ValueError(
                "storage_options passed with file object or non-fsspec file path"
            )

        with catch_warnings(record=True):
            self.api.write(
                path,
                df,
                compression=compression,
                write_index=index,
                partition_on=partition_cols,
                **kwargs,
            )

    def read(
        self,
        path,
        columns=None,
        filters=None,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ) -> DataFrame:
        parquet_kwargs: dict[str, Any] = {}
        use_nullable_dtypes = kwargs.pop("use_nullable_dtypes", False)
        dtype_backend = kwargs.pop("dtype_backend", lib.no_default)
        # We are disabling nullable dtypes for fastparquet pending discussion
        parquet_kwargs["pandas_nulls"] = False
        if use_nullable_dtypes:
            raise ValueError(
                "The 'use_nullable_dtypes' argument is not supported for the "
                "fastparquet engine"
            )
        if dtype_backend is not lib.no_default:
            raise ValueError(
                "The 'dtype_backend' argument is not supported for the "
                "fastparquet engine"
            )
        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )
        path = stringify_path(path)
        handles = None
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")

            parquet_kwargs["fs"] = fsspec.open(path, "rb", **(storage_options or {})).fs
        elif isinstance(path, str) and not os.path.isdir(path):
            # use get_handle only when we are very certain that it is not a directory
            # fsspec resources can also point to directories
            # this branch is used for example when reading from non-fsspec URLs
            handles = get_handle(
                path, "rb", is_text=False, storage_options=storage_options
            )
            path = handles.handle

        try:
            parquet_file = self.api.ParquetFile(path, **parquet_kwargs)
            return parquet_file.to_pandas(columns=columns, filters=filters, **kwargs)
        finally:
            if handles is not None:
                handles.close()


@doc(storage_options=_shared_docs["storage_options"])
def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = None,
    engine: str = "auto",
    compression: str | None = "snappy",
    index: bool | None = None,
    storage_options: StorageOptions | None = None,
    partition_cols: list[str] | None = None,
    filesystem: Any = None,
    **kwargs,
) -> bytes | None:
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, file-like object, or None, default None
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function. If None, the result is
        returned as bytes. If a string, it will be used as Root Directory path
        when writing a partitioned dataset. The engine fastparquet does not
        accept file-like objects.
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    compression : {{'snappy', 'gzip', 'brotli', 'lz4', 'zstd', None}},
        default 'snappy'. Name of the compression to use. Use ``None``
        for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    {storage_options}

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    kwargs
        Additional keyword arguments passed to the engine

    Returns
    -------
    bytes if no path argument is provided else None
    """
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]
    impl = get_engine(engine)

    path_or_buf: FilePath | WriteBuffer[bytes] = io.BytesIO() if path is None else path

    impl.write(
        df,
        path_or_buf,
        compression=compression,
        index=index,
        partition_cols=partition_cols,
        storage_options=storage_options,
        filesystem=filesystem,
        **kwargs,
    )

    if path is None:
        assert isinstance(path_or_buf, io.BytesIO)
        return path_or_buf.getvalue()
    else:
        return None


@doc(storage_options=_shared_docs["storage_options"])
def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: str = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions | None = None,
    use_nullable_dtypes: bool | lib.NoDefault = lib.no_default,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    filesystem: Any = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    **kwargs,
) -> DataFrame:
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function.
        The string could be a URL. Valid URL schemes include http, ftp, s3,
        gs, and file. For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``.
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    {storage_options}

        .. versionadded:: 1.3.0

    use_nullable_dtypes : bool, default False
        If True, use dtypes that use ``pd.NA`` as missing value indicator
        for the resulting DataFrame. (only applicable for the ``pyarrow``
        engine)
        As new dtypes are added that support ``pd.NA`` in the future, the
        output with this option will change to use those dtypes.
        Note: this is an experimental option, and behaviour (e.g. additional
        support dtypes) may change without notice.

        .. deprecated:: 2.0

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    filters : List[Tuple] or List[List[Tuple]], default None
        To filter out data.
        Filter syntax: [[(column, op, val), ...],...]
        where op is [==, =, >, >=, <, <=, !=, in, not in]
        The innermost tuples are transposed into a set of filters applied
        through an `AND` operation.
        The outer list combines these sets of filters through an `OR`
        operation.
        A single list of tuples can also be used, meaning that no `OR`
        operation between set of filters is to be conducted.

        Using this argument will NOT result in row-wise filtering of the final
        partitions unless ``engine="pyarrow"`` is also specified.  For
        other engines, filtering is only performed at the partition level, that is,
        to prevent the loading of some row-groups and/or files.

        .. versionadded:: 2.1.0

    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.to_parquet : Create a parquet object that serializes a DataFrame.

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {{"foo": range(5), "bar": range(5, 10)}}
    ...    )
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> df_parquet_bytes = original_df.to_parquet()
    >>> from io import BytesIO
    >>> restored_df = pd.read_parquet(BytesIO(df_parquet_bytes))
    >>> restored_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> restored_df.equals(original_df)
    True
    >>> restored_bar = pd.read_parquet(BytesIO(df_parquet_bytes), columns=["bar"])
    >>> restored_bar
        bar
    0    5
    1    6
    2    7
    3    8
    4    9
    >>> restored_bar.equals(original_df[['bar']])
    True

    The function uses `kwargs` that are passed directly to the engine.
    In the following example, we use the `filters` argument of the pyarrow
    engine to filter the rows of the DataFrame.

    Since `pyarrow` is the default engine, we can omit the `engine` argument.
    Note that the `filters` argument is implemented by the `pyarrow` engine,
    which can benefit from multithreading and also potentially be more
    economical in terms of memory.

    >>> sel = [("foo", ">", 2)]
    >>> restored_part = pd.read_parquet(BytesIO(df_parquet_bytes), filters=sel)
    >>> restored_part
        foo  bar
    0    3    8
    1    4    9
    """

    impl = get_engine(engine)

    if use_nullable_dtypes is not lib.no_default:
        msg = (
            "The argument 'use_nullable_dtypes' is deprecated and will be removed "
            "in a future version."
        )
        if use_nullable_dtypes is True:
            msg += (
                "Use dtype_backend='numpy_nullable' instead of use_nullable_dtype=True."
            )
        warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())
    else:
        use_nullable_dtypes = False
    check_dtype_backend(dtype_backend)

    return impl.read(
        path,
        columns=columns,
        filters=filters,
        storage_options=storage_options,
        use_nullable_dtypes=use_nullable_dtypes,
        dtype_backend=dtype_backend,
        filesystem=filesystem,
        **kwargs,
    )
<|endoftext|>""" io on the clipboard """
from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING
import warnings

from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.generic import ABCDataFrame

from pandas import (
    get_option,
    option_context,
)

if TYPE_CHECKING:
    from pandas._typing import DtypeBackend


def read_clipboard(
    sep: str = r"\s+",
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs,
):  # pragma: no cover
    r"""
    Read text from clipboard and pass to :func:`~pandas.read_csv`.

    Parses clipboard contents similar to how CSV files are parsed
    using :func:`~pandas.read_csv`.

    Parameters
    ----------
    sep : str, default '\\s+'
        A string or regex delimiter. The default of ``'\\s+'`` denotes
        one or more whitespace characters.

    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    **kwargs
        See :func:`~pandas.read_csv` for the full argument list.

    Returns
    -------
    DataFrame
        A parsed :class:`~pandas.DataFrame` object.

    See Also
    --------
    DataFrame.to_clipboard : Copy object to the system clipboard.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    >>> df.to_clipboard()  # doctest: +SKIP
    >>> pd.read_clipboard()  # doctest: +SKIP
         A  B  C
    0    1  2  3
    1    4  5  6
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # only utf-8 is valid for passed value because that's what clipboard
    # supports
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")

    check_dtype_backend(dtype_backend)

    from pandas.io.clipboard import clipboard_get
    from pandas.io.parsers import read_csv

    text = clipboard_get()

    # Try to decode (if needed, as "text" might already be a string here).
    try:
        text = text.decode(kwargs.get("encoding") or get_option("display.encoding"))
    except AttributeError:
        pass

    # Excel copies into clipboard with \t separation
    # inspect no more then the 10 first lines, if they
    # all contain an equal number (>0) of tabs, infer
    # that this came from excel and set 'sep' accordingly
    lines = text[:10000].split("\n")[:-1][:10]

    # Need to remove leading white space, since read_csv
    # accepts:
    #    a  b
    # 0  1  2
    # 1  3  4

    counts = {x.lstrip(" ").count("\t") for x in lines}
    if len(lines) > 1 and len(counts) == 1 and counts.pop() != 0:
        sep = "\t"
        # check the number of leading tabs in the first line
        # to account for index columns
        index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
        if index_length != 0:
            kwargs.setdefault("index_col", list(range(index_length)))

    # Edge case where sep is specified to be None, return to default
    if sep is None and kwargs.get("delim_whitespace") is None:
        sep = r"\s+"

    # Regex separator currently only works with python engine.
    # Default to python if separator is multi-character (regex)
    if len(sep) > 1 and kwargs.get("engine") is None:
        kwargs["engine"] = "python"
    elif len(sep) > 1 and kwargs.get("engine") == "c":
        warnings.warn(
            "read_clipboard with regex separator does not work properly with c engine.",
            stacklevel=find_stack_level(),
        )

    return read_csv(StringIO(text), sep=sep, dtype_backend=dtype_backend, **kwargs)


def to_clipboard(
    obj, excel: bool | None = True, sep: str | None = None, **kwargs
) -> None:  # pragma: no cover
    """
    Attempt to write text representation of object to the system clipboard
    The clipboard can be then pasted into Excel for example.

    Parameters
    ----------
    obj : the object to write to the clipboard
    excel : bool, defaults to True
            if True, use the provided separator, writing in a csv
            format for allowing easy pasting into excel.
            if False, write a string representation of the object
            to the clipboard
    sep : optional, defaults to tab
    other keywords are passed to to_csv

    Notes
    -----
    Requirements for your platform
      - Linux: xclip, or xsel (with PyQt4 modules)
      - Windows:
      - OS X:
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # testing if an invalid encoding is passed to clipboard
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise ValueError("clipboard only supports utf-8 encoding")

    from pandas.io.clipboard import clipboard_set

    if excel is None:
        excel = True

    if excel:
        try:
            if sep is None:
                sep = "\t"
            buf = StringIO()

            # clipboard_set (pyperclip) expects unicode
            obj.to_csv(buf, sep=sep, encoding="utf-8", **kwargs)
            text = buf.getvalue()

            clipboard_set(text)
            return
        except TypeError:
            warnings.warn(
                "to_clipboard in excel mode requires a single character separator.",
                stacklevel=find_stack_level(),
            )
    elif sep is not None:
        warnings.warn(
            "to_clipboard with excel=False ignores the sep argument.",
            stacklevel=find_stack_level(),
        )

    if isinstance(obj, ABCDataFrame):
        # str(df) has various unhelpful defaults, like truncation
        with option_context("display.max_colwidth", None):
            objstr = obj.to_string(**kwargs)
    else:
        objstr = str(obj)
    clipboard_set(objstr)
<|endoftext|>"""
:mod:``pandas.io.xml`` is a module for reading XML.
"""

from __future__ import annotations

import io
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)
import warnings

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    ParserError,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import is_list_like

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    file_exists,
    get_handle,
    infer_compression,
    is_file_like,
    is_fsspec_url,
    is_url,
    stringify_path,
)
from pandas.io.parsers import TextParser

if TYPE_CHECKING:
    from collections.abc import Sequence
    from xml.etree.ElementTree import Element

    from lxml import etree

    from pandas._typing import (
        CompressionOptions,
        ConvertersArg,
        DtypeArg,
        DtypeBackend,
        FilePath,
        ParseDatesArg,
        ReadBuffer,
        StorageOptions,
        XMLParsers,
    )

    from pandas import DataFrame


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",
)
class _XMLFrameParser:
    """
    Internal subclass to parse XML into DataFrames.

    Parameters
    ----------
    path_or_buffer : a valid JSON ``str``, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file.

    xpath : str or regex
        The ``XPath`` expression to parse required set of nodes for
        migration to :class:`~pandas.DataFrame`. ``etree`` supports limited ``XPath``.

    namespaces : dict
        The namespaces defined in XML document (``xmlns:namespace='URI'``)
        as dicts with key being namespace and value the URI.

    elems_only : bool
        Parse only the child elements at the specified ``xpath``.

    attrs_only : bool
        Parse only the attributes at the specified ``xpath``.

    names : list
        Column names for :class:`~pandas.DataFrame` of parsed XML data.

    dtype : dict
        Data type for data or columns. E.g. {{'a': np.float64,
        'b': np.int32, 'c': 'Int64'}}

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict
        Converts either index or select columns to datetimes

        .. versionadded:: 1.5.0

    encoding : str
        Encoding of xml object or document.

    stylesheet : str or file-like
        URL, file, file-like object, or a raw string containing XSLT,
        ``etree`` does not support XSLT but retained for consistency.

    iterparse : dict, optional
        Dict with row element as key and list of descendant elements
        and/or attributes as value to be retrieved in iterparsing of
        XML document.

        .. versionadded:: 1.5.0

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.xml._EtreeFrameParser
    pandas.io.xml._LxmlFrameParser

    Notes
    -----
    To subclass this class effectively you must override the following methods:`
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_iterparse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`


    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(
        self,
        path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        xpath: str,
        namespaces: dict[str, str] | None,
        elems_only: bool,
        attrs_only: bool,
        names: Sequence[str] | None,
        dtype: DtypeArg | None,
        converters: ConvertersArg | None,
        parse_dates: ParseDatesArg | None,
        encoding: str | None,
        stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
        iterparse: dict[str, list[str]] | None,
        compression: CompressionOptions,
        storage_options: StorageOptions,
    ) -> None:
        self.path_or_buffer = path_or_buffer
        self.xpath = xpath
        self.namespaces = namespaces
        self.elems_only = elems_only
        self.attrs_only = attrs_only
        self.names = names
        self.dtype = dtype
        self.converters = converters
        self.parse_dates = parse_dates
        self.encoding = encoding
        self.stylesheet = stylesheet
        self.iterparse = iterparse
        self.is_style = None
        self.compression: CompressionOptions = compression
        self.storage_options = storage_options

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, parse and return specific nodes.
        """

        raise AbstractMethodError(self)

    def _parse_nodes(self, elems: list[Any]) -> list[dict[str, str | None]]:
        """
        Parse xml nodes.

        This method will parse the children and attributes of elements
        in ``xpath``, conditionally for only elements, only attributes
        or both while optionally renaming node names.

        Raises
        ------
        ValueError
            * If only elements and only attributes are specified.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes compared to siblings
        will have optional keys filled with None values.
        """

        dicts: list[dict[str, str | None]]

        if self.elems_only and self.attrs_only:
            raise ValueError("Either element or attributes can be parsed not both.")
        if self.elems_only:
            if self.names:
                dicts = [
                    {
                        **(
                            {el.tag: el.text}
                            if el.text and not el.text.isspace()
                            else {}
                        ),
                        **{
                            nm: ch.text if ch.text else None
                            for nm, ch in zip(self.names, el.findall("*"))
                        },
                    }
                    for el in elems
                ]
            else:
                dicts = [
                    {ch.tag: ch.text if ch.text else None for ch in el.findall("*")}
                    for el in elems
                ]

        elif self.attrs_only:
            dicts = [
                {k: v if v else None for k, v in el.attrib.items()} for el in elems
            ]

        elif self.names:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),
                    **{
                        nm: ch.text if ch.text else None
                        for nm, ch in zip(self.names, el.findall("*"))
                    },
                }
                for el in elems
            ]

        else:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),
                    **{ch.tag: ch.text if ch.text else None for ch in el.findall("*")},
                }
                for el in elems
            ]

        dicts = [
            {k.split("}")[1] if "}" in k else k: v for k, v in d.items()} for d in dicts
        ]

        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]

        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]

        return dicts

    def _iterparse_nodes(self, iterparse: Callable) -> list[dict[str, str | None]]:
        """
        Iterparse xml nodes.

        This method will read in local disk, decompressed XML files for elements
        and underlying descendants using iterparse, a method to iterate through
        an XML tree without holding entire XML tree in memory.

        Raises
        ------
        TypeError
            * If ``iterparse`` is not a dict or its dict value is not list-like.
        ParserError
            * If ``path_or_buffer`` is not a physical file on disk or file-like object.
            * If no data is returned from selected items in ``iterparse``.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes in submitted list
        will have optional keys filled with None values.
        """

        dicts: list[dict[str, str | None]] = []
        row: dict[str, str | None] | None = None

        if not isinstance(self.iterparse, dict):
            raise TypeError(
                f"{type(self.iterparse).__name__} is not a valid type for iterparse"
            )

        row_node = next(iter(self.iterparse.keys())) if self.iterparse else ""
        if not is_list_like(self.iterparse[row_node]):
            raise TypeError(
                f"{type(self.iterparse[row_node])} is not a valid type "
                "for value in iterparse"
            )

        if (not hasattr(self.path_or_buffer, "read")) and (
            not isinstance(self.path_or_buffer, (str, PathLike))
            or is_url(self.path_or_buffer)
            or is_fsspec_url(self.path_or_buffer)
            or (
                isinstance(self.path_or_buffer, str)
                and self.path_or_buffer.startswith(("<?xml", "<"))
            )
            or infer_compression(self.path_or_buffer, "infer") is not None
        ):
            raise ParserError(
                "iterparse is designed for large XML files that are fully extracted on "
                "local disk and not as compressed files or online sources."
            )

        iterparse_repeats = len(self.iterparse[row_node]) != len(
            set(self.iterparse[row_node])
        )

        for event, elem in iterparse(self.path_or_buffer, events=("start", "end")):
            curr_elem = elem.tag.split("}")[1] if "}" in elem.tag else elem.tag

            if event == "start":
                if curr_elem == row_node:
                    row = {}

            if row is not None:
                if self.names and iterparse_repeats:
                    for col, nm in zip(self.iterparse[row_node], self.names):
                        if curr_elem == col:
                            elem_val = elem.text if elem.text else None
                            if elem_val not in row.values() and nm not in row:
                                row[nm] = elem_val

                        if col in elem.attrib:
                            if elem.attrib[col] not in row.values() and nm not in row:
                                row[nm] = elem.attrib[col]
                else:
                    for col in self.iterparse[row_node]:
                        if curr_elem == col:
                            row[col] = elem.text if elem.text else None
                        if col in elem.attrib:
                            row[col] = elem.attrib[col]

            if event == "end":
                if curr_elem == row_node and row is not None:
                    dicts.append(row)
                    row = None

                elem.clear()
                if hasattr(elem, "getprevious"):
                    while (
                        elem.getprevious() is not None and elem.getparent() is not None
                    ):
                        del elem.getparent()[0]

        if dicts == []:
            raise ParserError("No result from selected items in iterparse.")

        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]

        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]

        return dicts

    def _validate_path(self) -> list[Any]:
        """
        Validate ``xpath``.

        This method checks for syntax, evaluation, or empty nodes return.

        Raises
        ------
        SyntaxError
            * If xpah is not supported or issues with namespaces.

        ValueError
            * If xpah does not return any nodes.
        """

        raise AbstractMethodError(self)

    def _validate_names(self) -> None:
        """
        Validate names.

        This method will check if names is a list-like and aligns
        with length of parse nodes.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        raise AbstractMethodError(self)

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element | etree._Element:
        """
        Build tree from path_or_buffer.

        This method will parse XML object into tree
        either from string/bytes or file location.
        """
        raise AbstractMethodError(self)


class _EtreeFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with the Python
    standard library XML module: `xml.etree.ElementTree`.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        from xml.etree.ElementTree import iterparse

        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()

        self._validate_names()

        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )

        return xml_dicts

    def _validate_path(self) -> list[Any]:
        """
        Notes
        -----
        ``etree`` supports limited ``XPath``. If user attempts a more complex
        expression syntax error will raise.
        """

        msg = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )
        try:
            elems = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            children = [ch for el in elems for ch in el.findall("*")]
            attrs = {k: v for el in elems for k, v in el.attrib.items()}

            if elems is None:
                raise ValueError(msg)

            if elems is not None:
                if self.elems_only and children == []:
                    raise ValueError(msg)
                if self.attrs_only and attrs == {}:
                    raise ValueError(msg)
                if children == [] and attrs == {}:
                    raise ValueError(msg)

        except (KeyError, SyntaxError):
            raise SyntaxError(
                "You have used an incorrect or unsupported XPath "
                "expression for etree library or you used an "
                "undeclared namespace prefix."
            )

        return elems

    def _validate_names(self) -> None:
        children: list[Any]

        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                children = parent.findall("*") if parent is not None else []

            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element:
        from xml.etree.ElementTree import (
            XMLParser,
            parse,
        )

        handle_data = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            document = parse(xml_data, parser=curr_parser)

        return document.getroot()


class _LxmlFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into :class:`~pandas.DataFrame` with third-party
    full-featured XML library, ``lxml``, that supports
    ``XPath`` 1.0 and XSLT 1.0.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, optionally parse and run XSLT,
        and parse original or transformed XML and return specific nodes.
        """
        from lxml.etree import iterparse

        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)

            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()

            elems = self._validate_path()

        self._validate_names()

        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )

        return xml_dicts

    def _validate_path(self) -> list[Any]:
        msg = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )

        elems = self.xml_doc.xpath(self.xpath, namespaces=self.namespaces)
        children = [ch for el in elems for ch in el.xpath("*")]
        attrs = {k: v for el in elems for k, v in el.attrib.items()}

        if elems == []:
            raise ValueError(msg)

        if elems != []:
            if self.elems_only and children == []:
                raise ValueError(msg)
            if self.attrs_only and attrs == {}:
                raise ValueError(msg)
            if children == [] and attrs == {}:
                raise ValueError(msg)

        return elems

    def _validate_names(self) -> None:
        children: list[Any]

        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                children = self.xml_doc.xpath(
                    self.xpath + "[1]/*", namespaces=self.namespaces
                )

            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> etree._Element:
        from lxml.etree import (
            XMLParser,
            fromstring,
            parse,
        )

        handle_data = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)

            if isinstance(xml_data, io.StringIO):
                if self.encoding is None:
                    raise TypeError(
                        "Can not pass encoding None when input is StringIO."
                    )

                document = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                document = parse(xml_data, parser=curr_parser)

        return document

    def _transform_doc(self) -> etree._XSLTResultTree:
        """
        Transform original tree using stylesheet.

        This method will transform original xml using XSLT script into
        am ideally flatter xml document for easier parsing and migration
        to Data Frame.
        """
        from lxml.etree import XSLT

        transformer = XSLT(self.xsl_doc)
        new_doc = transformer(self.xml_doc)

        return new_doc


def get_data_from_filepath(
    filepath_or_buffer: FilePath | bytes | ReadBuffer[bytes] | ReadBuffer[str],
    encoding: str | None,
    compression: CompressionOptions,
    storage_options: StorageOptions,
) -> str | bytes | ReadBuffer[bytes] | ReadBuffer[str]:
    """
    Extract raw XML data.

    The method accepts three input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
        3. XML string or bytes

    This method turns (1) into (2) to simplify the rest of the processing.
    It returns input types (2) and (3) unchanged.
    """
    if not isinstance(filepath_or_buffer, bytes):
        filepath_or_buffer = stringify_path(filepath_or_buffer)

    if (
        isinstance(filepath_or_buffer, str)
        and not filepath_or_buffer.startswith(("<?xml", "<"))
    ) and (
        not isinstance(filepath_or_buffer, str)
        or is_url(filepath_or_buffer)
        or is_fsspec_url(filepath_or_buffer)
        or file_exists(filepath_or_buffer)
    ):
        with get_handle(
            filepath_or_buffer,
            "r",
            encoding=encoding,
            compression=compression,
            storage_options=storage_options,
        ) as handle_obj:
            filepath_or_buffer = (
                handle_obj.handle.read()
                if hasattr(handle_obj.handle, "read")
                else handle_obj.handle
            )

    return filepath_or_buffer


def preprocess_data(data) -> io.StringIO | io.BytesIO:
    """
    Convert extracted raw data.

    This method will return underlying data of extracted XML content.
    The data either has a `read` attribute (e.g. a file object or a
    StringIO/BytesIO) or is a string or bytes that is an XML document.
    """

    if isinstance(data, str):
        data = io.StringIO(data)

    elif isinstance(data, bytes):
        data = io.BytesIO(data)

    return data


def _data_to_frame(data, **kwargs) -> DataFrame:
    """
    Convert parsed data to Data Frame.

    This method will bind xml dictionary data of keys and values
    into named columns of Data Frame using the built-in TextParser
    class that build Data Frame and infers specific dtypes.
    """

    tags = next(iter(data))
    nodes = [list(d.values()) for d in data]

    try:
        with TextParser(nodes, names=tags, **kwargs) as tp:
            return tp.read()
    except ParserError:
        raise ParserError(
            "XML document may be too complex for import. "
            "Try to flatten document and use distinct "
            "element and attribute names."
        )


def _parse(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    xpath: str,
    namespaces: dict[str, str] | None,
    elems_only: bool,
    attrs_only: bool,
    names: Sequence[str] | None,
    dtype: DtypeArg | None,
    converters: ConvertersArg | None,
    parse_dates: ParseDatesArg | None,
    encoding: str | None,
    parser: XMLParsers,
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
    iterparse: dict[str, list[str]] | None,
    compression: CompressionOptions,
    storage_options: StorageOptions,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs,
) -> DataFrame:
    """
    Call internal parsers.

    This method will conditionally call internal parsers:
    LxmlFrameParser and/or EtreeParser.

    Raises
    ------
    ImportError
        * If lxml is not installed if selected as parser.

    ValueError
        * If parser is not lxml or etree.
    """

    p: _EtreeFrameParser | _LxmlFrameParser

    if isinstance(path_or_buffer, str) and not any(
        [
            is_file_like(path_or_buffer),
            file_exists(path_or_buffer),
            is_url(path_or_buffer),
            is_fsspec_url(path_or_buffer),
        ]
    ):
        warnings.warn(
            "Passing literal xml to 'read_xml' is deprecated and "
            "will be removed in a future version. To read from a "
            "literal string, wrap it in a 'StringIO' object.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if parser == "lxml":
        lxml = import_optional_dependency("lxml.etree", errors="ignore")

        if lxml is not None:
            p = _LxmlFrameParser(
                path_or_buffer,
                xpath,
                namespaces,
                elems_only,
                attrs_only,
                names,
                dtype,
                converters,
                parse_dates,
                encoding,
                stylesheet,
                iterparse,
                compression,
                storage_options,
            )
        else:
            raise ImportError("lxml not found, please install or use the etree parser.")

    elif parser == "etree":
        p = _EtreeFrameParser(
            path_or_buffer,
            xpath,
            namespaces,
            elems_only,
            attrs_only,
            names,
            dtype,
            converters,
            parse_dates,
            encoding,
            stylesheet,
            iterparse,
            compression,
            storage_options,
        )
    else:
        raise ValueError("Values for parser can only be lxml or etree.")

    data_dicts = p.parse_data()

    return _data_to_frame(
        data=data_dicts,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        dtype_backend=dtype_backend,
        **kwargs,
    )


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",
)
def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    xpath: str = "./*",
    namespaces: dict[str, str] | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Sequence[str] | None = None,
    dtype: DtypeArg | None = None,
    converters: ConvertersArg | None = None,
    parse_dates: ParseDatesArg | None = None,
    # encoding can not be None for lxml and StringIO input
    encoding: str | None = "utf-8",
    parser: XMLParsers = "lxml",
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame:
    r"""
    Read XML document into a :class:`~pandas.DataFrame` object.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    path_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a ``read()`` function. The string can be any valid XML
        string or a path. The string can further be a URL. Valid URL schemes
        include http, ftp, s3, and file.

        .. deprecated:: 2.1.0
            Passing xml literal strings is deprecated.
            Wrap literal xml input in ``io.StringIO`` or ``io.BytesIO`` instead.

    xpath : str, optional, default './\*'
        The ``XPath`` to parse required set of nodes for migration to
        :class:`~pandas.DataFrame`.``XPath`` should return a collection of elements
        and not a single element. Note: The ``etree`` parser supports limited ``XPath``
        expressions. For more complex ``XPath``, use ``lxml`` which requires
        installation.

    namespaces : dict, optional
        The namespaces defined in XML document as dicts with key being
        namespace prefix and value the URI. There is no need to include all
        namespaces in XML, only the ones used in ``xpath`` expression.
        Note: if XML document uses default namespace denoted as
        `xmlns='<URI>'` without a prefix, you must assign any temporary
        namespace prefix such as 'doc' to the URI in order to parse
        underlying nodes and/or attributes. For example, ::

            namespaces = {{"doc": "https://example.com"}}

    elems_only : bool, optional, default False
        Parse only the child elements at the specified ``xpath``. By default,
        all child elements and non-empty text nodes are returned.

    attrs_only :  bool, optional, default False
        Parse only the attributes at the specified ``xpath``.
        By default, all attributes are returned.

    names :  list-like, optional
        Column names for DataFrame of parsed XML data. Use this parameter to
        rename original element names and distinguish same named elements and
        attributes.

    dtype : Type name or dict of column -> type, optional
        Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32,
        'c': 'Int64'}}
        Use `str` or `object` together with suitable `na_values` settings
        to preserve and not interpret dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can either
        be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict, default False
        Identifiers to parse index or columns to datetime. The behavior is as follows:

        * boolean. If True -> try parsing the index.
        * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
          each as a separate date column.
        * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
          a single date column.
        * dict, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
          result 'foo'

        .. versionadded:: 1.5.0

    encoding : str, optional, default 'utf-8'
        Encoding of XML document.

    parser : {{'lxml','etree'}}, default 'lxml'
        Parser module to use for retrieval of data. Only 'lxml' and
        'etree' are supported. With 'lxml' more complex ``XPath`` searches
        and ability to use XSLT stylesheet are supported.

    stylesheet : str, path object or file-like object
        A URL, file-like object, or a raw string containing an XSLT script.
        This stylesheet should flatten complex, deeply nested XML documents
        for easier parsing. To use this feature you must have ``lxml`` module
        installed and specify 'lxml' as ``parser``. The ``xpath`` must
        reference nodes of transformed XML document generated after XSLT
        transformation and not the original XML document. Only XSLT 1.0
        scripts and not later versions is currently supported.

    iterparse : dict, optional
        The nodes or attributes to retrieve in iterparsing of XML document
        as a dict with key being the name of repeating element and value being
        list of elements or attribute names that are descendants of the repeated
        element. Note: If this option is used, it will replace ``xpath`` parsing
        and unlike ``xpath``, descendants do not need to relate to each other but can
        exist any where in document under the repeating element. This memory-
        efficient method should be used for very large XML files (500MB, 1GB, or 5GB+).
        For example, ::

            iterparse = {{"row_element": ["child_elem", "attr", "grandchild_elem"]}}

        .. versionadded:: 1.5.0

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    df
        A DataFrame.

    See Also
    --------
    read_json : Convert a JSON string to pandas object.
    read_html : Read HTML tables into a list of DataFrame objects.

    Notes
    -----
    This method is best designed to import shallow XML documents in
    following format which is the ideal fit for the two-dimensions of a
    ``DataFrame`` (row by column). ::

            <root>
                <row>
                  <column1>data</column1>
                  <column2>data</column2>
                  <column3>data</column3>
                  ...
               </row>
               <row>
                  ...
               </row>
               ...
            </root>

    As a file format, XML documents can be designed any way including
    layout of elements and attributes as long as it conforms to W3C
    specifications. Therefore, this method is a convenience handler for
    a specific flatter design and not all possible XML structures.

    However, for more complex XML documents, ``stylesheet`` allows you to
    temporarily redesign original document with XSLT (a special purpose
    language) for a flatter version for migration to a DataFrame.

    This function will *always* return a single :class:`DataFrame` or raise
    exceptions due to issues with XML document, ``xpath``, or other
    parameters.

    See the :ref:`read_xml documentation in the IO section of the docs
    <io.read_xml>` for more information in using this method to parse XML
    files to DataFrames.

    Examples
    --------
    >>> from io import StringIO
    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data xmlns="http://example.com">
    ...  <row>
    ...    <shape>square</shape>
    ...    <degrees>360</degrees>
    ...    <sides>4.0</sides>
    ...  </row>
    ...  <row>
    ...    <shape>circle</shape>
    ...    <degrees>360</degrees>
    ...    <sides/>
    ...  </row>
    ...  <row>
    ...    <shape>triangle</shape>
    ...    <degrees>180</degrees>
    ...    <sides>3.0</sides>
    ...  </row>
    ... </data>'''

    >>> df = pd.read_xml(StringIO(xml))
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data>
    ...   <row shape="square" degrees="360" sides="4.0"/>
    ...   <row shape="circle" degrees="360"/>
    ...   <row shape="triangle" degrees="180" sides="3.0"/>
    ... </data>'''

    >>> df = pd.read_xml(StringIO(xml), xpath=".//row")
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <doc:data xmlns:doc="https://example.com">
    ...   <doc:row>
    ...     <doc:shape>square</doc:shape>
    ...     <doc:degrees>360</doc:degrees>
    ...     <doc:sides>4.0</doc:sides>
    ...   </doc:row>
    ...   <doc:row>
    ...     <doc:shape>circle</doc:shape>
    ...     <doc:degrees>360</doc:degrees>
    ...     <doc:sides/>
    ...   </doc:row>
    ...   <doc:row>
    ...     <doc:shape>triangle</doc:shape>
    ...     <doc:degrees>180</doc:degrees>
    ...     <doc:sides>3.0</doc:sides>
    ...   </doc:row>
    ... </doc:data>'''

    >>> df = pd.read_xml(StringIO(xml),
    ...                  xpath="//doc:row",
    ...                  namespaces={{"doc": "https://example.com"}})
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml_data = '''
    ...         <data>
    ...            <row>
    ...               <index>0</index>
    ...               <a>1</a>
    ...               <b>2.5</b>
    ...               <c>True</c>
    ...               <d>a</d>
    ...               <e>2019-12-31 00:00:00</e>
    ...            </row>
    ...            <row>
    ...               <index>1</index>
    ...               <b>4.5</b>
    ...               <c>False</c>
    ...               <d>b</d>
    ...               <e>2019-12-31 00:00:00</e>
    ...            </row>
    ...         </data>
    ...         '''

    >>> df = pd.read_xml(StringIO(xml_data),
    ...                  dtype_backend="numpy_nullable",
    ...                  parse_dates=["e"])
    >>> df
       index     a    b      c  d          e
    0      0     1  2.5   True  a 2019-12-31
    1      1  <NA>  4.5  False  b 2019-12-31
    """
    check_dtype_backend(dtype_backend)

    return _parse(
        path_or_buffer=path_or_buffer,
        xpath=xpath,
        namespaces=namespaces,
        elems_only=elems_only,
        attrs_only=attrs_only,
        names=names,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        encoding=encoding,
        parser=parser,
        stylesheet=stylesheet,
        iterparse=iterparse,
        compression=compression,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )
<|endoftext|>"""
Collection of query wrappers / abstractions to both facilitate data
retrieval and to reduce dependency on DB-specific API.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from contextlib import (
    ExitStack,
    contextmanager,
)
from datetime import (
    date,
    datetime,
    time,
)
from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    DatabaseError,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas import get_option
from pandas.core.api import (
    DataFrame,
    Series,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime

from pandas.io._util import arrow_table_to_pandas

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Mapping,
    )

    from sqlalchemy import Table
    from sqlalchemy.sql.expression import (
        Select,
        TextClause,
    )

    from pandas._typing import (
        DateTimeErrorChoices,
        DtypeArg,
        DtypeBackend,
        IndexLabel,
        Self,
    )

    from pandas import Index

# -----------------------------------------------------------------------------
# -- Helper functions


def _process_parse_dates_argument(parse_dates):
    """Process parse_dates argument for read_sql functions"""
    # handle non-list entries for parse_dates gracefully
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []

    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    return parse_dates


def _handle_date_column(
    col, utc: bool = False, format: str | dict[str, Any] | None = None
):
    if isinstance(format, dict):
        # GH35185 Allow custom error values in parse_dates argument of
        # read_sql like functions.
        # Format can take on custom to_datetime argument values such as
        # {"errors": "coerce"} or {"dayfirst": True}
        error: DateTimeErrorChoices = format.pop("errors", None) or "ignore"
        if error == "ignore":
            try:
                return to_datetime(col, **format)
            except (TypeError, ValueError):
                # TODO: not reached 2023-10-27; needed?
                return col
        return to_datetime(col, errors=error, **format)
    else:
        # Allow passing of formatting string for integers
        # GH17855
        if format is None and (
            issubclass(col.dtype.type, np.floating)
            or issubclass(col.dtype.type, np.integer)
        ):
            format = "s"
        if format in ["D", "d", "h", "m", "s", "ms", "us", "ns"]:
            return to_datetime(col, errors="coerce", unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            # coerce to UTC timezone
            # GH11216
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors="coerce", format=format, utc=utc)


def _parse_date_columns(data_frame, parse_dates):
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    parse_dates = _process_parse_dates_argument(parse_dates)

    # we want to coerce datetime64_tz dtypes for now to UTC
    # we could in theory do a 'nice' conversion from a FixedOffset tz
    # GH11216
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))

    return data_frame


def _convert_arrays_to_dataframe(
    data,
    columns,
    coerce_float: bool = True,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    content = lib.to_object_array_tuples(data)
    arrays = convert_object_array(
        list(content.T),
        dtype=None,
        coerce_float=coerce_float,
        dtype_backend=dtype_backend,
    )
    if dtype_backend == "pyarrow":
        pa = import_optional_dependency("pyarrow")

        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                # TODO: Arrow still infers strings arrays as regular strings instead
                # of large_string, which is what we preserver everywhere else for
                # dtype_backend="pyarrow". We may want to reconsider this
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays  # type: ignore[assignment]
    if arrays:
        df = DataFrame(dict(zip(list(range(len(columns))), arrays)))
        df.columns = columns
        return df
    else:
        return DataFrame(columns=columns)


def _wrap_result(
    data,
    columns,
    index_col=None,
    coerce_float: bool = True,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
):
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)

    if dtype:
        frame = frame.astype(dtype)

    frame = _parse_date_columns(frame, parse_dates)

    if index_col is not None:
        frame = frame.set_index(index_col)

    return frame


def _wrap_result_adbc(
    df: DataFrame,
    *,
    index_col=None,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)

    df = _parse_date_columns(df, parse_dates)

    if index_col is not None:
        df = df.set_index(index_col)

    return df


def execute(sql, con, params=None):
    """
    Execute the given SQL query using the provided connection object.

    Parameters
    ----------
    sql : string
        SQL query to be executed.
    con : SQLAlchemy connection or sqlite3 connection
        If a DBAPI2 object, only sqlite3 is supported.
    params : list or tuple, optional, default: None
        List of parameters to pass to execute method.

    Returns
    -------
    Results Iterable
    """
    warnings.warn(
        "`pandas.io.sql.execute` is deprecated and "
        "will be removed in the future version.",
        FutureWarning,
        stacklevel=find_stack_level(),
    )  # GH50185
    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")

    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Engine)):
        raise TypeError("pandas.io.sql.execute requires a connection")  # GH50185
    with pandasSQL_builder(con, need_transaction=True) as pandas_sql:
        return pandas_sql.execute(sql, params)


# -----------------------------------------------------------------------------
# -- Read and write to DataFrames


@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    columns: list[str] | None = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    columns: list[str] | None = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]:
    ...


def read_sql_table(
    table_name: str,
    con,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list[str] | dict[str, str] | None = None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table('table_name', 'postgres:///db_name')  # doctest:+SKIP
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f"Table {table_name} not found")

        table = pandas_sql.read_table(
            table_name,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    if table is not None:
        return table
    else:
        raise ValueError(f"Table {table_name} not found", con)


@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    chunksize: int = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]:
    ...


def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list[Any] | Mapping[str, Any] | None = None,
    parse_dates: list[str] | dict[str, str] | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or mapping, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.

        .. versionadded:: 1.3.0
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.

    Examples
    --------
    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("sqlite:///database.db")  # doctest: +SKIP
    >>> with engine.connect() as conn, conn.begin():  # doctest: +SKIP
    ...     data = pd.read_sql_table("data", conn)  # doctest: +SKIP
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        return pandas_sql.read_query(
            sql,
            index_col=index_col,
            params=params,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            chunksize=chunksize,
            dtype=dtype,
            dtype_backend=dtype_backend,
        )


@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> DataFrame:
    ...


@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> Iterator[DataFrame]:
    ...


def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params=None,
    parse_dates=None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    dtype: DtypeArg | None = None,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the ADBC connection and
        SQLAlchemy connectable; str connections are closed automatically. See
        `here <https://docs.sqlalchemy.org/en/20/core/connections.html>`_.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.
        The argument is ignored if a table is passed instead of a query.

        .. versionadded:: 2.0.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Examples
    --------
    Read data from SQL via either a SQL query or a SQL tablename.
    When using a SQLite database only SQL queries are accepted,
    providing only the SQL tablename will result in an error.

    >>> from sqlite3 import connect
    >>> conn = connect(':memory:')
    >>> df = pd.DataFrame(data=[[0, '10/11/12'], [1, '12/11/10']],
    ...                   columns=['int_column', 'date_column'])
    >>> df.to_sql(name='test_data', con=conn)
    2

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data', conn)
       int_column date_column
    0           0    10/11/12
    1           1    12/11/10

    >>> pd.read_sql('test_data', 'postgres:///db_name')  # doctest:+SKIP

    Apply date parsing to columns through the ``parse_dates`` argument
    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.
    Custom argument values for applying ``pd.to_datetime`` on a column are specified
    via a dictionary format:

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data',
    ...             conn,
    ...             parse_dates={"date_column": {"format": "%d/%m/%y"}})
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12

    .. versionadded:: 2.2.0

       pandas now supports reading via ADBC drivers

    >>> from adbc_driver_postgresql import dbapi  # doctest:+SKIP
    >>> with dbapi.connect('postgres:///db_name') as conn:  # doctest:+SKIP
    ...     pd.read_sql('SELECT int_column FROM test_data', conn)
       int_column
    0           0
    1           1
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        if isinstance(pandas_sql, SQLiteDatabase):
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )

        try:
            _is_table_name = pandas_sql.has_table(sql)
        except Exception:
            # using generic exception to catch errors from sql drivers (GH24988)
            _is_table_name = False

        if _is_table_name:
            return pandas_sql.read_table(
                sql,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
            )
        else:
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )


def to_sql(
    frame,
    name: str,
    con,
    schema: str | None = None,
    if_exists: Literal["fail", "replace", "append"] = "fail",
    index: bool = True,
    index_label: IndexLabel | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    method: Literal["multi"] | Callable | None = None,
    engine: str = "auto",
    **engine_kwargs,
) -> int | None:
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    frame : DataFrame, Series
    name : str
        Name of SQL table.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        or sqlite3 DBAPI2 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : str, optional
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
    index : bool, default True
        Write DataFrame index as a column.
    index_label : str or sequence, optional
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.
    chunksize : int, optional
        Specify the number of rows in each batch to be written at a time.
        By default, all rows will be written at once.
    dtype : dict or scalar, optional
        Specifying the datatype for columns. If a dictionary is used, the
        keys should be the column names and the values should be the
        SQLAlchemy types or strings for the sqlite3 fallback mode. If a
        scalar is provided, it will be applied to all columns.
    method : {None, 'multi', callable}, optional
        Controls the SQL insertion clause used:

        - None : Uses standard SQL ``INSERT`` clause (one per row).
        - ``'multi'``: Pass multiple values in a single ``INSERT`` clause.
        - callable with signature ``(pd_table, conn, keys, data_iter) -> int | None``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.
    engine : {'auto', 'sqlalchemy'}, default 'auto'
        SQL engine library to use. If 'auto', then the option
        ``io.sql.engine`` is used. The default ``io.sql.engine``
        behavior is 'sqlalchemy'

        .. versionadded:: 1.3.0

    **engine_kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    None or int
        Number of rows affected by to_sql. None is returned if the callable
        passed into ``method`` does not return an integer number of rows.

        .. versionadded:: 1.4.0

    Notes
    -----
    The returned rows affected is the sum of the ``rowcount`` attribute of ``sqlite3.Cursor``
    or SQLAlchemy connectable. If using ADBC the returned rows are the result
    of ``Cursor.adbc_ingest``. The returned value may not reflect the exact number of written
    rows as stipulated in the
    `sqlite3 <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.rowcount>`__ or
    `SQLAlchemy <https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.BaseCursorResult.rowcount>`__
    """  # noqa: E501
    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError(
            "'frame' argument should be either a Series or a DataFrame"
        )

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        return pandas_sql.to_sql(
            frame,
            name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            engine=engine,
            **engine_kwargs,
        )


def has_table(table_name: str, con, schema: str | None = None) -> bool:
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        return pandas_sql.has_table(table_name)


table_exists = has_table


def pandasSQL_builder(
    con,
    schema: str | None = None,
    need_transaction: bool = False,
) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.
    """
    import sqlite3

    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)

    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")

    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError("Using URI string without sqlalchemy installed.")

    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)

    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if adbc and isinstance(con, adbc.Connection):
        return ADBCDatabase(con)

    warnings.warn(
        "pandas only supports SQLAlchemy connectable (engine/connection) or "
        "database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 "
        "objects are not tested. Please consider using SQLAlchemy.",
        UserWarning,
        stacklevel=find_stack_level(),
    )
    return SQLiteDatabase(con)


class SQLTable(PandasObject):
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """

    # TODO: support for multiIndex

    def __init__(
        self,
        name: str,
        pandas_sql_engine,
        frame=None,
        index: bool | str | list[str] | None = True,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        prefix: str = "pandas",
        index_label=None,
        schema=None,
        keys=None,
        dtype: DtypeArg | None = None,
    ) -> None:
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype

        if frame is not None:
            # We want to initialize based on a dataframe
            self.table = self._create_table_setup()
        else:
            # no data provided, read-only mode
            self.table = self.pd_sql.get_table(self.name, self.schema)

        if self.table is None:
            raise ValueError(f"Could not init table '{name}'")

        if not len(self.name):
            raise ValueError("Empty table name specified")

    def exists(self):
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self) -> str:
        from sqlalchemy.schema import CreateTable

        return str(CreateTable(self.table).compile(self.pd_sql.con))

    def _execute_create(self) -> None:
        # Inserting table into database, add to MetaData object
        self.table = self.table.to_metadata(self.pd_sql.meta)
        with self.pd_sql.run_transaction():
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        if self.exists():
            if self.if_exists == "fail":
                raise ValueError(f"Table '{self.name}' already exists.")
            if self.if_exists == "replace":
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == "append":
                pass
            else:
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            self._execute_create()

    def _execute_insert(self, conn, keys: list[str], data_iter) -> int:
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
           Column names
        data_iter : generator of list
           Each item contains a list of values to be inserted
        """
        data = [dict(zip(keys, row)) for row in data_iter]
        result = conn.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn, keys: list[str], data_iter) -> int:
        """
        Alternative to _execute_insert for DBs support multi-value INSERT.

        Note: multi-value insert is usually faster for analytics DBs
        and tables containing a few columns
        but performance degrades quickly with increase of columns.

        """

        from sqlalchemy import insert

        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(self.table).values(data)
        result = conn.execute(stmt)
        return result.rowcount

    def insert_data(self) -> tuple[list[str], list[np.ndarray]]:
        if self.index is not None:
            temp = self.frame.copy()
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f"duplicate name in index/columns: {err}") from err
        else:
            temp = self.frame

        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        # this just pre-allocates the list: None's will be replaced with ndarrays
        # error: List item 0 has incompatible type "None"; expected "ndarray"
        data_list: list[np.ndarray] = [None] * ncols  # type: ignore[list-item]

        for i, (_, ser) in enumerate(temp.items()):
            if ser.dtype.kind == "M":
                if isinstance(ser._values, ArrowExtensionArray):
                    import pyarrow as pa

                    if pa.types.is_date(ser.dtype.pyarrow_dtype):
                        # GH#53854 to_pydatetime not supported for pyarrow date dtypes
                        d = ser._values.to_numpy(dtype=object)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            # GH#52459 to_pydatetime will return Index[object]
                            d = np.asarray(ser.dt.to_pydatetime(), dtype=object)
                else:
                    d = ser._values.to_pydatetime()
            elif ser.dtype.kind == "m":
                vals = ser._values
                if isinstance(vals, ArrowExtensionArray):
                    vals = vals.to_numpy(dtype=np.dtype("m8[ns]"))
                # store as integers, see GH#6921, GH#7076
                d = vals.view("i8").astype(object)
            else:
                d = ser._values.astype(object)

            assert isinstance(d, np.ndarray), type(d)

            if ser._can_hold_na:
                # Note: this will miss timedeltas since they are converted to int
                mask = isna(d)
                d[mask] = None

            data_list[i] = d

        return column_names, data_list

    def insert(
        self,
        chunksize: int | None = None,
        method: Literal["multi"] | Callable | None = None,
    ) -> int | None:
        # set insert method
        if method is None:
            exec_insert = self._execute_insert
        elif method == "multi":
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f"Invalid parameter `method`: {method}")

        keys, data_list = self.insert_data()

        nrows = len(self.frame)

        if nrows == 0:
            return 0

        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError("chunksize argument should be non-zero")

        chunks = (nrows // chunksize) + 1
        total_inserted = None
        with self.pd_sql.run_transaction() as conn:
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break

                chunk_iter = zip(*(arr[start_i:end_i] for arr in data_list))
                num_inserted = exec_insert(conn, keys, chunk_iter)
                # GH 46891
                if num_inserted is not None:
                    if total_inserted is None:
                        total_inserted = num_inserted
                    else:
                        total_inserted += num_inserted
        return total_inserted

    def _query_iterator(
        self,
        result,
        exit_stack: ExitStack,
        chunksize: int | None,
        columns,
        coerce_float: bool = True,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Return generator through chunked result set."""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield DataFrame.from_records(
                            [], columns=columns, coerce_float=coerce_float
                        )
                    break

                has_read_data = True
                self.frame = _convert_arrays_to_dataframe(
                    data, columns, coerce_float, dtype_backend
                )

                self._harmonize_columns(
                    parse_dates=parse_dates, dtype_backend=dtype_backend
                )

                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)

                yield self.frame

    def read(
        self,
        exit_stack: ExitStack,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        from sqlalchemy import select

        if columns is not None and len(columns) > 0:
            cols = [self.table.c[n] for n in columns]
            if self.index is not None:
                for idx in self.index[::-1]:
                    cols.insert(0, self.table.c[idx])
            sql_select = select(*cols)
        else:
            sql_select = select(self.table)
        result = self.pd_sql.execute(sql_select)
        column_names = result.keys()

        if chunksize is not None:
            return self._query_iterator(
                result,
                exit_stack,
                chunksize,
                column_names,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            self.frame = _convert_arrays_to_dataframe(
                data, column_names, coerce_float, dtype_backend
            )

            self._harmonize_columns(
                parse_dates=parse_dates, dtype_backend=dtype_backend
            )

            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)

            return self.frame

    def _index_name(self, index, index_label):
        # for writing: index=True to include index in sql table
        if index is True:
            nlevels = self.frame.index.nlevels
            # if index_label is specified, set this as index name(s)
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError(
                        "Length of 'index_label' should match number of "
                        f"levels, which is {nlevels}"
                    )
                return index_label
            # return the used column labels for the index columns
            if (
                nlevels == 1
                and "index" not in self.frame.columns
                and self.frame.index.name is None
            ):
                return ["index"]
            else:
                return com.fill_missing_names(self.frame.index.names)

        # for reading: index=(list of) string to specify column to set as index
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(self, dtype_mapper):
        column_names_and_types = []
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
                column_names_and_types.append((str(idx_label), idx_type, True))

        column_names_and_types += [
            (str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False)
            for i in range(len(self.frame.columns))
        ]

        return column_names_and_types

    def _create_table_setup(self):
        from sqlalchemy import (
            Column,
            PrimaryKeyConstraint,
            Table,
        )
        from sqlalchemy.schema import MetaData

        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)

        columns: list[Any] = [
            Column(name, typ, index=is_index)
            for name, typ, is_index in column_names_and_types
        ]

        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + "_pk")
            columns.append(pkc)

        schema = self.schema or self.pd_sql.meta.schema

        # At this point, attach to new metadata, only attach to self.meta
        # once table is created.
        meta = MetaData()
        return Table(self.name, meta, *columns, schema=schema)

    def _harmonize_columns(
        self,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> None:
        """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
        parse_dates = _process_parse_dates_argument(parse_dates)

        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]

                # Handle date parsing upfront; don't try to convert columns
                # twice
                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue

                # the type the dataframe column should have
                col_type = self._get_dtype(sql_col.type)

                if (
                    col_type is datetime
                    or col_type is date
                    or col_type is DatetimeTZDtype
                ):
                    # Convert tz-aware Datetime SQL columns to UTC
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif dtype_backend == "numpy" and col_type is float:
                    # floats support NA, can always convert!
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
                elif (
                    using_string_dtype()
                    and is_string_dtype(col_type)
                    and is_object_dtype(self.frame[col_name])
                ):
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
                elif dtype_backend == "numpy" and len(df_col) == df_col.count():
                    # No NA values, can convert ints and bools
                    if col_type is np.dtype("int64") or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type, copy=False)
            except KeyError:
                pass  # this column not in results

    def _sqlalchemy_type(self, col: Index | Series):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        from sqlalchemy.types import (
            TIMESTAMP,
            BigInteger,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            SmallInteger,
            Text,
            Time,
        )

        if col_type in ("datetime64", "datetime"):
            # GH 9086: TIMESTAMP is the suggested type if the column contains
            # timezone information
            try:
                # error: Item "Index" of "Union[Index, Series]" has no attribute "dt"
                if col.dt.tz is not None:  # type: ignore[union-attr]
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                # The column is actually a DatetimeIndex
                # GH 26761 or an Index with date-like data e.g. 9999-01-01
                if getattr(col, "tz", None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime
        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return BigInteger
        elif col_type == "floating":
            if col.dtype == "float32":
                return Float(precision=23)
            else:
                return Float(precision=53)
        elif col_type == "integer":
            # GH35076 Map pandas integer to optimal SQLAlchemy integer type
            if col.dtype.name.lower() in ("int8", "uint8", "int16"):
                return SmallInteger
            elif col.dtype.name.lower() in ("uint16", "int32"):
                return Integer
            elif col.dtype.name.lower() == "uint64":
                raise ValueError("Unsigned 64 bit integer datatype is not supported")
            else:
                return BigInteger
        elif col_type == "boolean":
            return Boolean
        elif col_type == "date":
            return Date
        elif col_type == "time":
            return Time
        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        return Text

    def _get_dtype(self, sqltype):
        from sqlalchemy.types import (
            TIMESTAMP,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            String,
        )

        if isinstance(sqltype, Float):
            return float
        elif isinstance(sqltype, Integer):
            # TODO: Refine integer size.
            return np.dtype("int64")
        elif isinstance(sqltype, TIMESTAMP):
            # we have a timezone capable type
            if not sqltype.timezone:
                return datetime
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            # Caution: np.datetime64 is also a subclass of np.number.
            return datetime
        elif isinstance(sqltype, Date):
            return date
        elif isinstance(sqltype, Boolean):
            return bool
        elif isinstance(sqltype, String):
            if using_string_dtype():
                return StringDtype(na_value=np.nan)

        return object


class PandasSQL(PandasObject, ABC):
    """
    Subclasses Should define read_query and to_sql.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        pass

    @abstractmethod
    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        pass

    @abstractmethod
    def execute(self, sql: str | Select | TextClause, params=None):
        pass

    @abstractmethod
    def has_table(self, name: str, schema: str | None = None) -> bool:
        pass

    @abstractmethod
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        pass


class BaseEngine:
    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        """
        Inserts data into already-prepared table
        """
        raise AbstractMethodError(self)


class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None:
        import_optional_dependency(
            "sqlalchemy", extra="sqlalchemy is required for SQL support."
        )

    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        from sqlalchemy import exc

        try:
            return table.insert(chunksize=chunksize, method=method)
        except exc.StatementError as err:
            # GH34431
            # https://stackoverflow.com/a/67358288/6067848
            msg = r"""(\(1054, "Unknown column 'inf(e0)?' in 'field list'"\))(?#
            )|inf can not be used with MySQL"""
            err_text = str(err.orig)
            if re.search(msg, err_text):
                raise ValueError("inf cannot be used with MySQL") from err
            raise err


def get_engine(engine: str) -> BaseEngine:
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.sql.engine")

    if engine == "auto":
        # try engines in this order
        engine_classes = [SQLAlchemyEngine]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'sqlalchemy'.\n"
            "A suitable version of "
            "sqlalchemy is required for sql I/O "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "sqlalchemy":
        return SQLAlchemyEngine()

    raise ValueError("engine must be one of 'auto', 'sqlalchemy'")


class SQLDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.

    """

    def __init__(
        self, con, schema: str | None = None, need_transaction: bool = False
    ) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.schema import MetaData

        # self.exit_stack cleans up the Engine and Connection and commits the
        # transaction if any of those objects was created below.
        # Cleanup happens either in self.__exit__ or at the end of the iterator
        # returned by read_sql when chunksize is not None.
        self.exit_stack = ExitStack()
        if isinstance(con, str):
            con = create_engine(con)
            self.exit_stack.callback(con.dispose)
        if isinstance(con, Engine):
            con = self.exit_stack.enter_context(con.connect())
        if need_transaction and not con.in_transaction():
            self.exit_stack.enter_context(con.begin())
        self.con = con
        self.meta = MetaData(schema=schema)
        self.returns_generator = False

    def __exit__(self, *args) -> None:
        if not self.returns_generator:
            self.exit_stack.close()

    @contextmanager
    def run_transaction(self):
        if not self.con.in_transaction():
            with self.con.begin():
                yield self.con
        else:
            yield self.con

    def execute(self, sql: str | Select | TextClause, params=None):
        """Simple passthrough to SQLAlchemy connectable"""
        args = [] if params is None else [params]
        if isinstance(sql, str):
            return self.con.exec_driver_sql(sql, *args)
        return self.con.execute(sql, *args)

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        index_col : string, optional, default: None
            Column to set as index.
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
              (default).
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
              DataFrame.

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        self.meta.reflect(bind=self.con, only=[table_name], views=True)
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        if chunksize is not None:
            self.returns_generator = True
        return table.read(
            self.exit_stack,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    @staticmethod
    def _query_iterator(
        result,
        exit_stack: ExitStack,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Return generator through chunked result set"""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield _wrap_result(
                            [],
                            columns,
                            index_col=index_col,
                            coerce_float=coerce_float,
                            parse_dates=parse_dates,
                            dtype=dtype,
                            dtype_backend=dtype_backend,
                        )
                    break

                has_read_data = True
                yield _wrap_result(
                    data,
                    columns,
                    index_col=index_col,
                    coerce_float=coerce_float,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    dtype_backend=dtype_backend,
                )

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        result = self.execute(sql, params)
        columns = result.keys()

        if chunksize is not None:
            self.returns_generator = True
            return self._query_iterator(
                result,
                self.exit_stack,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    read_sql = read_query

    def prep_table(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool | str | list[str] | None = True,
        index_label=None,
        schema=None,
        dtype: DtypeArg | None = None,
    ) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)

            from sqlalchemy.types import TypeEngine

            for col, my_type in dtype.items():
                if isinstance(my_type, type) and issubclass(my_type, TypeEngine):
                    pass
                elif isinstance(my_type, TypeEngine):
                    pass
                else:
                    raise ValueError(f"The type of {col} is not a SQLAlchemy type")

        table = SQLTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )
        table.create()
        return table

    def check_case_sensitive(
        self,
        name: str,
        schema: str | None,
    ) -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
        if not name.isdigit() and not name.islower():
            # check for potentially case sensitivity issues (GH7815)
            # Only check when name is not a number and name is not lower case
            from sqlalchemy import inspect as sqlalchemy_inspect

            insp = sqlalchemy_inspect(self.con)
            table_names = insp.get_table_names(schema=schema or self.meta.schema)
            if name not in table_names:
                msg = (
                    f"The provided table name '{name}' is not found exactly as "
                    "such in the database after writing the table, possibly "
                    "due to case sensitivity issues. Consider using lower "
                    "case table names."
                )
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.
        method : {None', 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            SQL engine library to use. If 'auto', then the option
            ``io.sql.engine`` is used. The default ``io.sql.engine``
            behavior is 'sqlalchemy'

            .. versionadded:: 1.3.0

        **engine_kwargs
            Any additional kwargs are passed to the engine.
        """
        sql_engine = get_engine(engine)

        table = self.prep_table(
            frame=frame,
            name=name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )

        total_inserted = sql_engine.insert_records(
            table=table,
            con=self.con,
            frame=frame,
            name=name,
            index=index,
            schema=schema,
            chunksize=chunksize,
            method=method,
            **engine_kwargs,
        )

        self.check_case_sensitive(name=name, schema=schema)
        return total_inserted

    @property
    def tables(self):
        return self.meta.tables

    def has_table(self, name: str, schema: str | None = None) -> bool:
        from sqlalchemy import inspect as sqlalchemy_inspect

        insp = sqlalchemy_inspect(self.con)
        return insp.has_table(name, schema or self.meta.schema)

    def get_table(self, table_name: str, schema: str | None = None) -> Table:
        from sqlalchemy import (
            Numeric,
            Table,
        )

        schema = schema or self.meta.schema
        tbl = Table(table_name, self.meta, autoload_with=self.con, schema=schema)
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    def drop_table(self, table_name: str, schema: str | None = None) -> None:
        schema = schema or self.meta.schema
        if self.has_table(table_name, schema):
            self.meta.reflect(
                bind=self.con, only=[table_name], schema=schema, views=True
            )
            with self.run_transaction():
                self.get_table(table_name, schema).drop(bind=self.con)
            self.meta.clear()

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        table = SQLTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


# ---- SQL without SQLAlchemy ---


class ADBCDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using ADBC to handle DataBase abstraction.

    Parameters
    ----------
    con : adbc_driver_manager.dbapi.Connection
    """

    def __init__(self, con) -> None:
        self.con = con

    @contextmanager
    def run_transaction(self):
        with self.con.cursor() as cur:
            try:
                yield cur
            except Exception:
                self.con.rollback()
                raise
            self.con.commit()

    def execute(self, sql: str | Select | TextClause, params=None):
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Exception as exc:
            try:
                self.con.rollback()
            except Exception as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        coerce_float : bool, default True
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
              (default).
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
              DataFrame.

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")

        if columns:
            if index_col:
                index_select = maybe_make_list(index_col)
            else:
                index_select = []
            to_select = index_select + columns
            select_list = ", ".join(f'"{x}"' for x in to_select)
        else:
            select_list = "*"
        if schema:
            stmt = f"SELECT {select_list} FROM {schema}.{table_name}"
        else:
            stmt = f"SELECT {select_list} FROM {table_name}"

        with self.con.cursor() as cur:
            cur.execute(stmt)
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)

        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
        )

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Raises NotImplementedError
        params : list, tuple or dict, optional, default: None
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if params:
            raise NotImplementedError("'params' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")

        with self.con.cursor() as cur:
            cur.execute(sql)
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)

        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
            dtype=dtype,
        )

    read_sql = read_query

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Raises NotImplementedError
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : single type or dict of column name to SQL type, default None
            Raises NotImplementedError
        method : {None', 'multi', callable}, default None
            Raises NotImplementedError
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            Raises NotImplementedError if not set to 'auto'
        """
        if index_label:
            raise NotImplementedError(
                "'index_label' is not implemented for ADBC drivers"
            )
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if dtype:
            raise NotImplementedError("'dtype' is not implemented for ADBC drivers")
        if method:
            raise NotImplementedError("'method' is not implemented for ADBC drivers")
        if engine != "auto":
            raise NotImplementedError(
                "engine != 'auto' not implemented for ADBC drivers"
            )

        if schema:
            table_name = f"{schema}.{name}"
        else:
            table_name = name

        # pandas if_exists="append" will still create the
        # table if it does not exist; ADBC is more explicit with append/create
        # as applicable modes, so the semantics get blurred across
        # the libraries
        mode = "create"
        if self.has_table(name, schema):
            if if_exists == "fail":
                raise ValueError(f"Table '{table_name}' already exists.")
            elif if_exists == "replace":
                with self.con.cursor() as cur:
                    cur.execute(f"DROP TABLE {table_name}")
            elif if_exists == "append":
                mode = "append"

        import pyarrow as pa

        try:
            tbl = pa.Table.from_pandas(frame, preserve_index=index)
        except pa.ArrowNotImplementedError as exc:
            raise ValueError("datatypes not supported") from exc

        with self.con.cursor() as cur:
            total_inserted = cur.adbc_ingest(
                table_name=name, data=tbl, mode=mode, db_schema_name=schema
            )

        self.con.commit()
        return total_inserted

    def has_table(self, name: str, schema: str | None = None) -> bool:
        meta = self.con.adbc_get_objects(
            db_schema_filter=schema, table_name_filter=name
        ).read_all()

        for catalog_schema in meta["catalog_db_schemas"].to_pylist():
            if not catalog_schema:
                continue
            for schema_record in catalog_schema:
                if not schema_record:
                    continue

                for table_record in schema_record["db_schema_tables"]:
                    if table_record["table_name"] == name:
                        return True

        return False

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        raise NotImplementedError("not implemented for adbc")


# sqlite-specific sql strings and handler class
# dictionary used for readability purposes
_SQL_TYPES = {
    "string": "TEXT",
    "floating": "REAL",
    "integer": "INTEGER",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "INTEGER",
}


def _get_unicode_name(name: object):
    try:
        uname = str(name).encode("utf-8", "strict").decode("utf-8")
    except UnicodeError as err:
        raise ValueError(f"Cannot convert identifier to UTF-8: '{name}'") from err
    return uname


def _get_valid_sqlite_name(name: object):
    # See https://stackoverflow.com/questions/6514274/how-do-you-escape-strings\
    # -for-sqlite-table-column-names-in-python
    # Ensure the string can be encoded as UTF-8.
    # Ensure the string does not include any NUL characters.
    # Replace all " with "".
    # Wrap the entire thing in double quotes.

    uname = _get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        raise ValueError("SQLite identifier cannot contain NULs")
    return '"' + uname.replace('"', '""') + '"'


class SQLiteTable(SQLTable):
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._register_date_adapters()

    def _register_date_adapters(self) -> None:
        # GH 8341
        # register an adapter callable for datetime.time object
        import sqlite3

        # this will transform time(12,34,56,789) into '12:34:56.000789'
        # (this is what sqlalchemy does)
        def _adapt_time(t) -> str:
            # This is faster than strftime
            return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"

        # Also register adapters for date/datetime and co
        # xref https://docs.python.org/3.12/library/sqlite3.html#adapter-and-converter-recipes
        # Python 3.12+ doesn't auto-register adapters for us anymore

        adapt_date_iso = lambda val: val.isoformat()
        adapt_datetime_iso = lambda val: val.isoformat(" ")

        sqlite3.register_adapter(time, _adapt_time)

        sqlite3.register_adapter(date, adapt_date_iso)
        sqlite3.register_adapter(datetime, adapt_datetime_iso)

        convert_date = lambda val: date.fromisoformat(val.decode())
        convert_timestamp = lambda val: datetime.fromisoformat(val.decode())

        sqlite3.register_converter("date", convert_date)
        sqlite3.register_converter("timestamp", convert_timestamp)

    def sql_schema(self) -> str:
        return str(";\n".join(self.table))

    def _execute_create(self) -> None:
        with self.pd_sql.run_transaction() as conn:
            for stmt in self.table:
                conn.execute(stmt)

    def insert_statement(self, *, num_rows: int) -> str:
        names = list(map(str, self.frame.columns))
        wld = "?"  # wildcard char
        escape = _get_valid_sqlite_name

        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)

        bracketed_names = [escape(column) for column in names]
        col_names = ",".join(bracketed_names)

        row_wildcards = ",".join([wld] * len(names))
        wildcards = ",".join([f"({row_wildcards})" for _ in range(num_rows)])
        insert_statement = (
            f"INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}"
        )
        return insert_statement

    def _execute_insert(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        conn.executemany(self.insert_statement(num_rows=1), data_list)
        return conn.rowcount

    def _execute_insert_multi(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)
        return conn.rowcount

    def _create_table_setup(self):
        """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = _get_valid_sqlite_name

        create_tbl_stmts = [
            escape(cname) + " " + ctype for cname, ctype, _ in column_names_and_types
        ]

        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ", ".join([escape(c) for c in keys])
            create_tbl_stmts.append(
                f"CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})"
            )
        if self.schema:
            schema_name = self.schema + "."
        else:
            schema_name = ""
        create_stmts = [
            "CREATE TABLE "
            + schema_name
            + escape(self.name)
            + " (\n"
            + ",\n  ".join(create_tbl_stmts)
            + "\n)"
        ]

        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if len(ix_cols):
            cnames = "_".join(ix_cols)
            cnames_br = ",".join([escape(c) for c in ix_cols])
            create_stmts.append(
                "CREATE INDEX "
                + escape("ix_" + self.name + "_" + cnames)
                + "ON "
                + escape(self.name)
                + " ("
                + cnames_br
                + ")"
            )

        return create_stmts

    def _sql_type_name(self, col):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            col_type = "integer"

        elif col_type == "datetime64":
            col_type = "datetime"

        elif col_type == "empty":
            col_type = "string"

        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        if col_type not in _SQL_TYPES:
            col_type = "string"

        return _SQL_TYPES[col_type]


class SQLiteDatabase(PandasSQL):
    """
    Version of SQLDatabase to support SQLite connections (fallback without
    SQLAlchemy). This should only be used internally.

    Parameters
    ----------
    con : sqlite connection object

    """

    def __init__(self, con) -> None:
        self.con = con

    @contextmanager
    def run_transaction(self):
        cur = self.con.cursor()
        try:
            yield cur
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str | Select | TextClause, params=None):
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Exception as exc:
            try:
                self.con.rollback()
            except Exception as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    @staticmethod
    def _query_iterator(
        cursor,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Return generator through chunked result set"""
        has_read_data = False
        while True:
            data = cursor.fetchmany(chunksize)
            if type(data) == tuple:
                data = list(data)
            if not data:
                cursor.close()
                if not has_read_data:
                    result = DataFrame.from_records(
                        [], columns=columns, coerce_float=coerce_float
                    )
                    if dtype:
                        result = result.astype(dtype)
                    yield result
                break

            has_read_data = True
            yield _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

    def read_query(
        self,
        sql,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        cursor = self.execute(sql, params)
        columns = [col_desc[0] for col_desc in cursor.description]

        if chunksize is not None:
            return self._query_iterator(
                cursor,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = self._fetchall_as_list(cursor)
            cursor.close()

            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    def _fetchall_as_list(self, cur):
        result = cur.fetchall()
        if not isinstance(result, list):
            result = list(result)
        return result

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: str = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame: DataFrame
        name: string
            Name of SQL table.
        if_exists: {'fail', 'replace', 'append'}, default 'fail'
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if it does not exist.
        index : bool, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Ignored parameter included for compatibility with SQLAlchemy
            version of ``to_sql``.
        chunksize : int, default None
            If not None, then rows will be written in batches of this
            size at a time. If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a string. If all columns are of the same type, one single value
            can be used.
        method : {None, 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)

            for col, my_type in dtype.items():
                if not isinstance(my_type, str):
                    raise ValueError(f"{col} ({my_type}) not a string")

        table = SQLiteTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            dtype=dtype,
        )
        table.create()
        return table.insert(chunksize, method)

    def has_table(self, name: str, schema: str | None = None) -> bool:
        wld = "?"
        query = f"""
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type IN ('table', 'view')
            AND name={wld};
        """

        return len(self.execute(query, [name]).fetchall()) > 0

    def get_table(self, table_name: str, schema: str | None = None) -> None:
        return None  # not supported in fallback mode

    def drop_table(self, name: str, schema: str | None = None) -> None:
        drop_sql = f"DROP TABLE {_get_valid_sqlite_name(name)}"
        self.execute(drop_sql)

    def _create_sql_schema(
        self,
        frame,
        table_name: str,
        keys=None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        table = SQLiteTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


def get_schema(
    frame,
    name: str,
    keys=None,
    con=None,
    dtype: DtypeArg | None = None,
    schema: str | None = None,
) -> str:
    """
    Get the SQL db table schema for the given frame.

    Parameters
    ----------
    frame : DataFrame
    name : str
        name of SQL table
    keys : string or sequence, default: None
        columns to use a primary key
    con: ADBC Connection, SQLAlchemy connectable, sqlite3 connection, default: None
