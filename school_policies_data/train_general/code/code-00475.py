            num_shards=(num_shards or {}).get(split),
            create_pr=False,
            embed_external_files=embed_external_files,
            num_proc=num_proc,
        )
        additions += split_additions
        new_parquet_paths += split_new_parquet_paths
        splits_info.append(split_info)
        uploaded_sizes.append(uploaded_size)

    # Don't delete the new files
    new_parquet_paths = set(new_parquet_paths)
    delete = [file_to_delete for file_to_delete in files_to_delete if file_to_delete not in new_parquet_paths]

    # Update the dataset card
    new_dataset_card, new_legacy_dataset_infos = _get_updated_dataset_card(
        fs=dirfs,
        config_name=config_name,
        splits_info=splits_info,
        features=features,
        data_dir=data_dir,
        set_default=set_default,
        uploaded_sizes=uploaded_sizes,
        deleted_sizes=[],
        remove_other_splits=True,
    )
    path_prefix = (path + "/") if path else ""
    add = [(str(new_dataset_card).encode(), path_prefix + config.REPOCARD_FILENAME)]
    if new_legacy_dataset_infos:
        add.append(
            (json.dumps(new_legacy_dataset_infos).encode("utf-8"), path_prefix + config.DATASETDICT_INFOS_FILENAME)
        )

    # Upload dataset card and delete old files
    api.batch_bucket_files(
        bucket_id=bucket_id,
        add=add,
        delete=delete,
    )
<|endoftext|>import copy
import os
from collections.abc import Iterator
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .utils.logging import get_logger


if TYPE_CHECKING:
    from .features.features import Features, FeatureType


logger = get_logger(__name__)


def inject_arrow_table_documentation(arrow_table_method):
    def wrapper(fn):
        fn.__doc__ = arrow_table_method.__doc__ + (fn.__doc__ if fn.__doc__ is not None else "")
        fn.__doc__ = fn.__doc__.replace("pyarrow.Table", "Table")
        if hasattr(arrow_table_method, "__annotations__"):
            fn.__annotations__ = arrow_table_method.__annotations__
        return fn

    return wrapper


def _in_memory_arrow_table_from_file(filename: str) -> pa.Table:
    in_memory_stream = pa.input_stream(filename)
    opened_stream = pa.ipc.open_stream(in_memory_stream)
    pa_table = opened_stream.read_all()
    return pa_table


def _in_memory_arrow_table_from_buffer(buffer: pa.Buffer) -> pa.Table:
    stream = pa.BufferReader(buffer)
    opened_stream = pa.ipc.open_stream(stream)
    table = opened_stream.read_all()
    return table


def _memory_mapped_record_batch_reader_from_file(filename: str) -> pa.RecordBatchStreamReader:
    memory_mapped_stream = pa.memory_map(filename)
    return pa.ipc.open_stream(memory_mapped_stream)


def read_schema_from_file(filename: str) -> pa.Schema:
    """
    Infer arrow table schema from file without loading whole file into memory.
    Useful especially while having very big files.
    """
    with pa.memory_map(filename) as memory_mapped_stream:
        schema = pa.ipc.open_stream(memory_mapped_stream).schema
    return schema


def _batch_arrow_table(table: pa.Table) -> pa.Table:
    batched_columns = []
    for column_name in table.column_names:
        column = table[column_name].combine_chunks()
        offsets = pa.array([0, len(column)], type=pa.int32())
        batched_columns.append(pa.ListArray.from_arrays(offsets, column))
    return pa.Table.from_arrays(batched_columns, names=table.column_names)


def _batch_accumulate_arrow_table_by_columns(
    table: pa.Table, indices: list[int], columns: tuple[str], tables_accumulator: list[pa.Table], length: Optional[int]
) -> pa.Table:
    accumulate_last_batch = length is None or indices[-1] + 1 < length
    # keep accumulating if key is the same, otherwise include the accumulated tables
    if tables_accumulator and accumulate_last_batch:
        for column in columns:
            if any(pc.not_equal(table[column], tables_accumulator[0][column][0]).to_pylist()):
                break
        else:
            tables_accumulator.append(table)
            empty_batched_table = pa.Table.from_arrays(
                [
                    pa.ListArray.from_arrays(pa.array([0], type=pa.int32()), table[column].chunk(0))
                    for column in table.column_names
                ],
                names=table.column_names,
            )
            return empty_batched_table
    table = pa.concat_tables(tables_accumulator + [table])
    tables_accumulator[:] = []
    if len(table) == 0:
        return table
    # cut the table per key, i.e. when the columns change value
    if len(table) > 1:
        cut_array = pc.not_equal(table[columns[0]][1:], table[columns[0]][:-1])
        for column in columns[1:]:
            cut_array = pc.or_(cut_array, pc.not_equal(table[column][1:], table[column][:-1]))
    else:
        cut_array = pa.array([], type=pa.uint64())
    offsets = pc.indices_nonzero(cut_array)
    # make the batched table
    offsets = pc.add(1, offsets)
    offsets = pa.concat_arrays([pa.array([0], type=pa.int32()), offsets.cast(pa.int32())])
    if not accumulate_last_batch:
        offsets = pa.concat_arrays([offsets, pa.array([len(table)], type=pa.int32())])
    batched_columns = []
    for column_name in table.column_names:
        column = table[column_name].combine_chunks()
        batched_columns.append(pa.ListArray.from_arrays(offsets, column))
    batched_table = pa.Table.from_arrays(batched_columns, names=table.column_names)
    # add the last batch to the accumulator since it might not be full yet
    if accumulate_last_batch:
        last_offset = offsets[-1].as_py()
        tables_accumulator.append(table.slice(last_offset, len(table) - last_offset))
    return batched_table


def _memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    opened_stream = _memory_mapped_record_batch_reader_from_file(filename)
    pa_table = opened_stream.read_all()
    return pa_table


def _deepcopy(x, memo: dict):
    """deepcopy a regular class instance"""
    cls = x.__class__
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result


def _interpolation_search(arr: list[int], x: int) -> int:
    """
    Return the position i of a sorted array so that arr[i] <= x < arr[i+1]

    Args:
        arr (`List[int]`): non-empty sorted list of integers
        x (`int`): query

    Returns:
        `int`: the position i so that arr[i] <= x < arr[i+1]

    Raises:
        `IndexError`: if the array is empty or if the query is outside the array values
    """
    i, j = 0, len(arr) - 1
    while i < j and arr[i] <= x < arr[j]:
        k = i + ((j - i) * (x - arr[i]) // (arr[j] - arr[i]))
        if arr[k] <= x < arr[k + 1]:
            return k
        elif arr[k] < x:
            i, j = k + 1, j
        else:
            i, j = i, k
    raise IndexError(f"Invalid query '{x}' for size {arr[-1] if len(arr) else 'none'}.")


class IndexedTableMixin:
    def __init__(self, table: pa.Table):
        self._schema: pa.Schema = table.schema
        self._batches: list[pa.RecordBatch] = [
            recordbatch for recordbatch in table.to_batches() if len(recordbatch) > 0
        ]
        self._offsets: np.ndarray = np.cumsum([0] + [len(b) for b in self._batches], dtype=np.int64)

    def fast_gather(self, indices: Union[list[int], np.ndarray]) -> pa.Table:
        """
        Create a pa.Table by gathering the records at the records at the specified indices. Should be faster
        than pa.concat_tables(table.fast_slice(int(i) % table.num_rows, 1) for i in indices) since NumPy can compute
        the binary searches in parallel, highly optimized C
        """
        if not len(indices):
            raise ValueError("Indices must be non-empty")
        batch_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        return pa.Table.from_batches(
            [
                self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1)
                for batch_idx, i in zip(batch_indices, indices)
            ],
            schema=self._schema,
        )

    def fast_slice(self, offset=0, length=None) -> pa.Table:
        """
        Slice the Table using interpolation search.
        The behavior is the same as `pyarrow.Table.slice` but it's significantly faster.

        Interpolation search is used to find the start and end indexes of the batches we want to keep.
        The batches to keep are then concatenated to form the sliced Table.
        """
        if offset < 0:
            raise IndexError("Offset must be non-negative")
        elif offset >= self._offsets[-1] or (length is not None and length <= 0):
            return pa.Table.from_batches([], schema=self._schema)
        i = _interpolation_search(self._offsets, offset)
        if length is None or length + offset >= self._offsets[-1]:
            batches = self._batches[i:]
            batches[0] = batches[0].slice(offset - self._offsets[i])
        else:
            j = _interpolation_search(self._offsets, offset + length - 1)
            batches = self._batches[i : j + 1]
            batches[-1] = batches[-1].slice(0, offset + length - self._offsets[j])
            batches[0] = batches[0].slice(offset - self._offsets[i])
        return pa.Table.from_batches(batches, schema=self._schema)


class Table(IndexedTableMixin):
    """
    Wraps a pyarrow Table by using composition.
    This is the base class for `InMemoryTable`, `MemoryMappedTable` and `ConcatenationTable`.

    It implements all the basic attributes/methods of the pyarrow Table class except
    the Table transforms: `slice, filter, flatten, combine_chunks, cast, add_column,
    append_column, remove_column, set_column, rename_columns` and `drop`.

    The implementation of these methods differs for the subclasses.
    """

    def __init__(self, table: pa.Table):
        super().__init__(table)
        self.table = table

    def __deepcopy__(self, memo: dict):
        # arrow tables are immutable, so there's no need to copy self.table
        # moreover calling deepcopy on a pyarrow table seems to make pa.total_allocated_bytes() decrease for some reason
        # by adding it to the memo, self.table won't be copied
        memo[id(self.table)] = self.table
        # same for the recordbatches used by the index
        memo[id(self._batches)] = list(self._batches)
        return _deepcopy(self, memo)

    def validate(self, *args, **kwargs):
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially `O(n)`).

        Args:
            full (`bool`, defaults to `False`):
                If `True`, run expensive checks, otherwise cheap checks only.

        Raises:
            `pa.lib.ArrowInvalid`: if validation fails
        """
        return self.table.validate(*args, **kwargs)

    def equals(self, *args, **kwargs):
        """
        Check if contents of two tables are equal.

        Args:
            other ([`~datasets.table.Table`]):
                Table to compare against.
            check_metadata `bool`, defaults to `False`):
                Whether schema metadata equality should be checked as well.

        Returns:
            `bool`
        """
        args = tuple(arg.table if isinstance(arg, Table) else arg for arg in args)
        kwargs = {k: v.table if isinstance(v, Table) else v for k, v in kwargs}
        return self.table.equals(*args, **kwargs)

    def to_batches(self, *args, **kwargs):
        """
        Convert Table to list of (contiguous) `RecordBatch` objects.

        Args:
            max_chunksize (`int`, defaults to `None`):
                Maximum size for `RecordBatch` chunks. Individual chunks may be
                smaller depending on the chunk layout of individual columns.

        Returns:
            `List[pyarrow.RecordBatch]`
        """
        return self.table.to_batches(*args, **kwargs)

    def to_pydict(self, *args, **kwargs):
        """
        Convert the Table to a `dict` or `OrderedDict`.

        Returns:
            `dict`
        """
        return self.table.to_pydict(*args, **kwargs)

    def to_pylist(self, *args, **kwargs):
        """
        Convert the Table to a list

        Returns:
            `list`
        """
        return self.table.to_pylist(*args, **kwargs)

    def to_pandas(self, *args, **kwargs):
        """
        Convert to a pandas-compatible NumPy array or DataFrame, as appropriate.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                Arrow MemoryPool to use for allocations. Uses the default memory
                pool is not passed.
            strings_to_categorical (`bool`, defaults to `False`):
                Encode string (UTF8) and binary types to `pandas.Categorical`.
            categories (`list`, defaults to `empty`):
                List of fields that should be returned as `pandas.Categorical`. Only
                applies to table-like data structures.
            zero_copy_only (`bool`, defaults to `False`):
                Raise an `ArrowException` if this function call would require copying
                the underlying data.
            integer_object_nulls (`bool`, defaults to `False`):
                Cast integers with nulls to objects.
            date_as_object (`bool`, defaults to `True`):
                Cast dates to objects. If `False`, convert to `datetime64[ns]` dtype.
            timestamp_as_object (`bool`, defaults to `False`):
                Cast non-nanosecond timestamps (`np.datetime64`) to objects. This is
                useful if you have timestamps that don't fit in the normal date
                range of nanosecond timestamps (1678 CE-2262 CE).
                If `False`, all timestamps are converted to `datetime64[ns]` dtype.
            use_threads (`bool`, defaults to `True`):
                Whether to parallelize the conversion using multiple threads.
            deduplicate_objects (`bool`, defaults to `False`):
                Do not create multiple copies Python objects when created, to save
                on memory use. Conversion will be slower.
            ignore_metadata (`bool`, defaults to `False`):
                If `True`, do not use the 'pandas' metadata to reconstruct the
                DataFrame index, if present.
            safe (`bool`, defaults to `True`):
                For certain data types, a cast is needed in order to store the
                data in a pandas DataFrame or Series (e.g. timestamps are always
                stored as nanoseconds in pandas). This option controls whether it
                is a safe cast or not.
            split_blocks (`bool`, defaults to `False`):
                If `True`, generate one internal "block" for each column when
                creating a pandas.DataFrame from a `RecordBatch` or `Table`. While this
                can temporarily reduce memory note that various pandas operations
                can trigger "consolidation" which may balloon memory use.
            self_destruct (`bool`, defaults to `False`):
                EXPERIMENTAL: If `True`, attempt to deallocate the originating Arrow
                memory while converting the Arrow object to pandas. If you use the
                object after calling `to_pandas` with this option it will crash your
                program.
            types_mapper (`function`, defaults to `None`):
                A function mapping a pyarrow DataType to a pandas `ExtensionDtype`.
                This can be used to override the default pandas type for conversion
                of built-in pyarrow types or in absence of `pandas_metadata` in the
                Table schema. The function receives a pyarrow DataType and is
                expected to return a pandas `ExtensionDtype` or `None` if the
                default conversion should be used for that type. If you have
                a dictionary mapping, you can pass `dict.get` as function.

        Returns:
            `pandas.Series` or `pandas.DataFrame`: `pandas.Series` or `pandas.DataFrame` depending on type of object
        """
        return self.table.to_pandas(*args, **kwargs)

    def to_string(self, *args, **kwargs):
        return self.table.to_string(*args, **kwargs)

    def to_reader(self, max_chunksize: Optional[int] = None):
        """
        Convert the Table to a RecordBatchReader.

        Note that this method is zero-copy, it merely exposes the same data under a different API.

        Args:
            max_chunksize (`int`, defaults to `None`)
                Maximum size for RecordBatch chunks. Individual chunks may be smaller depending
                on the chunk layout of individual columns.

        Returns:
            `pyarrow.RecordBatchReader`
        """
        return self.table.to_reader(max_chunksize=max_chunksize)

    def field(self, *args, **kwargs):
        """
        Select a schema field by its column name or numeric index.

        Args:
            i (`Union[int, str]`):
                The index or name of the field to retrieve.

        Returns:
            `pyarrow.Field`
        """
        return self.table.field(*args, **kwargs)

    def column(self, *args, **kwargs):
        """
        Select a column by its column name, or numeric index.

        Args:
            i (`Union[int, str]`):
                The index or name of the column to retrieve.

        Returns:
            `pyarrow.ChunkedArray`
        """
        return self.table.column(*args, **kwargs)

    def itercolumns(self, *args, **kwargs):
        """
        Iterator over all columns in their numerical order.

        Yields:
            `pyarrow.ChunkedArray`
        """
        return self.table.itercolumns(*args, **kwargs)

    @property
    def schema(self):
        """
        Schema of the table and its columns.

        Returns:
            `pyarrow.Schema`
        """
        return self.table.schema

    @property
    def columns(self):
        """
        List of all columns in numerical order.

        Returns:
            `List[pa.ChunkedArray]`
        """
        return self.table.columns

    @property
    def num_columns(self):
        """
        Number of columns in this table.

        Returns:
            int
        """
        return self.table.num_columns

    @property
    def num_rows(self):
        """
        Number of rows in this table.

        Due to the definition of a table, all columns have the same number of
        rows.

        Returns:
            int
        """
        return self.table.num_rows

    @property
    def shape(self):
        """
        Dimensions of the table: (#rows, #columns).

        Returns:
            `(int, int)`: Number of rows and number of columns.
        """
        return self.table.shape

    @property
    def nbytes(self):
        """
        Total number of bytes consumed by the elements of the table.
        """
        return self.table.nbytes

    @property
    def column_names(self):
        """
        Names of the table's columns.
        """
        return self.table.column_names

    def __eq__(self, other):
        return self.equals(other)

    def __getitem__(self, i):
        return self.table[i]

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return self.table.__repr__().replace("pyarrow.Table", self.__class__.__name__)

    def __str__(self):
        return self.table.__str__().replace("pyarrow.Table", self.__class__.__name__)

    def slice(self, *args, **kwargs):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        raise NotImplementedError()

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be None,
        which deletes any existing metadata

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        raise NotImplementedError()

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        raise NotImplementedError()

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:  New table with the passed column added.
        """
        raise NotImplementedError()

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`: New table without the column.
        """
        raise NotImplementedError()

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column set.
        """
        raise NotImplementedError()

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        raise NotImplementedError()

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`: New table without the columns.
        """
        raise NotImplementedError()

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            `datasets.table.Table`: table with only a subset of the columns
        """
        raise NotImplementedError()


class TableBlock(Table):
    """
    `TableBlock` is the allowed class inside a `ConcanetationTable`.
    Only `MemoryMappedTable` and `InMemoryTable` are `TableBlock`.
    This is because we don't want a `ConcanetationTable` made out of other `ConcanetationTables`.
    """

    pass


class InMemoryTable(TableBlock):
    """
    The table is said in-memory when it is loaded into the user's RAM.

    Pickling it does copy all the data using memory.
    Its implementation is simple and uses the underlying pyarrow Table methods directly.

    This is different from the `MemoryMapped` table, for which pickling doesn't copy all the
    data in memory. For a `MemoryMapped`, unpickling instead reloads the table from the disk.

    `InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    def __getstate__(self):
        # Serialize only the underlying table: the `_batches`/`_offsets` index is
        # derived and rebuilt in `__init__`, and pickling it makes the fingerprint
        # cost scale with the number of chunks (see #8327).
        return {"table": self.table}

    def __setstate__(self, state):
        InMemoryTable.__init__(self, state["table"])

    @classmethod
    def from_file(cls, filename: str):
        table = _in_memory_arrow_table_from_file(filename)
        return cls(table)

    @classmethod
    def from_buffer(cls, buffer: pa.Buffer):
        table = _in_memory_arrow_table_from_buffer(buffer)
        return cls(table)

    @classmethod
    def from_pandas(cls, *args, **kwargs):
        """
        Convert pandas.DataFrame to an Arrow Table.

        The column types in the resulting Arrow Table are inferred from the
        dtypes of the pandas.Series in the DataFrame. In the case of non-object
        Series, the NumPy dtype is translated to its Arrow equivalent. In the
        case of `object`, we need to guess the datatype by looking at the
        Python objects in this Series.

        Be aware that Series of the `object` dtype don't carry enough
        information to always lead to a meaningful Arrow type. In the case that
        we cannot infer a type, e.g. because the DataFrame is of length 0 or
        the Series only contains `None/nan` objects, the type is set to
        null. This behavior can be avoided by constructing an explicit schema
        and passing it to this function.

        Args:
            df (`pandas.DataFrame`):
            schema (`pyarrow.Schema`, *optional*):
                The expected schema of the Arrow Table. This can be used to
                indicate the type of columns if we cannot infer it automatically.
                If passed, the output will have exactly this schema. Columns
                specified in the schema that are not found in the DataFrame columns
                or its index will raise an error. Additional columns or index
                levels in the DataFrame which are not specified in the schema will
                be ignored.
            preserve_index (`bool`, *optional*):
                Whether to store the index as an additional column in the resulting
                `Table`. The default of None will store the index as a column,
                except for RangeIndex which is stored as metadata only. Use
                `preserve_index=True` to force it to be stored as a column.
            nthreads (`int`, defaults to `None` (may use up to system CPU count threads))
                If greater than 1, convert columns to Arrow in parallel using
                indicated number of threads.
            columns (`List[str]`, *optional*):
               List of column to be converted. If `None`, use all columns.
            safe (`bool`, defaults to `True`):
               Check for overflows or other unsafe conversions,

        Returns:
            `datasets.table.Table`:

        Examples:
        ```python
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df = pd.DataFrame({
            ...     'int': [1, 2],
            ...     'str': ['a', 'b']
            ... })
        >>> pa.Table.from_pandas(df)
        <pyarrow.lib.Table object at 0x7f05d1fb1b40>
        ```
        """
        return cls(pa.Table.from_pandas(*args, **kwargs))

    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct a Table from Arrow arrays.

        Args:
            arrays (`List[Union[pyarrow.Array, pyarrow.ChunkedArray]]`):
                Equal-length arrays that should form the table.
            names (`List[str]`, *optional*):
                Names for the table columns. If not passed, schema must be passed.
            schema (`Schema`, defaults to `None`):
                Schema for the created table. If not passed, names must be passed.
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_arrays(*args, **kwargs))

    @classmethod
    def from_pydict(cls, *args, **kwargs):
        """
        Construct a Table from Arrow arrays or columns.

        Args:
            mapping (`Union[dict, Mapping]`):
                A mapping of strings to Arrays or Python lists.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the Mapping values
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_pydict(*args, **kwargs))

    @classmethod
    def from_pylist(cls, mapping, *args, **kwargs):
        """
        Construct a Table from list of rows / dictionaries.

        Args:
            mapping (`List[dict]`):
                A mapping of strings to row values.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the Mapping values
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_pylist(mapping, *args, **kwargs))

    @classmethod
    def from_batches(cls, *args, **kwargs):
        """
        Construct a Table from a sequence or iterator of Arrow `RecordBatches`.

        Args:
            batches (`Union[Sequence[pyarrow.RecordBatch], Iterator[pyarrow.RecordBatch]]`):
                Sequence of `RecordBatch` to be converted, all schemas must be equal.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the first `RecordBatch`.

        Returns:
            `datasets.table.Table`:
        """
        return cls(pa.Table.from_batches(*args, **kwargs))

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        # Use fast slicing here
        return InMemoryTable(self.fast_slice(offset=offset, length=length))

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        return InMemoryTable(self.table.filter(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(table_flatten(self.table, *args, **kwargs))

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(self.table.combine_chunks(*args, **kwargs))

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(table_cast(self.table, *args, **kwargs))

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be `None`,
        which deletes any existing metadata).

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        return InMemoryTable(self.table.replace_schema_metadata(*args, **kwargs))

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        return InMemoryTable(self.table.add_column(*args, **kwargs))

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        return InMemoryTable(self.table.append_column(*args, **kwargs))

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        return InMemoryTable(self.table.remove_column(*args, **kwargs))

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        return InMemoryTable(self.table.set_column(*args, **kwargs))

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        return InMemoryTable(self.table.rename_columns(*args, **kwargs))

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        return InMemoryTable(self.table.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        return InMemoryTable(self.table.select(*args, **kwargs))


# The MemoryMappedTable needs replays to properly reload tables from the disk
Replay = tuple[str, tuple, dict]


class MemoryMappedTable(TableBlock):
    """
    The table is said memory mapped when it doesn't use the user's RAM but loads the data
    from the disk instead.

    Pickling it doesn't copy the data into memory.
    Instead, only the path to the memory mapped arrow file is pickled, as well as the list
    of transforms to "replay" when reloading the table from the disk.

    Its implementation requires to store an history of all the transforms that were applied
    to the underlying pyarrow Table, so that they can be "replayed" when reloading the Table
    from the disk.

    This is different from the `InMemoryTable` table, for which pickling does copy all the
    data in memory.

    `InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    def __init__(self, table: pa.Table, path: str, replays: Optional[list[Replay]] = None):
        super().__init__(table)
        self.path = os.path.abspath(path)
        self.replays: list[Replay] = replays if replays is not None else []

    @classmethod
    def from_file(cls, filename: str, replays=None):
        table = _memory_mapped_arrow_table_from_file(filename)
        table = cls._apply_replays(table, replays)
        return cls(table, filename, replays)

    def __getstate__(self):
        return {"path": self.path, "replays": self.replays}

    def __setstate__(self, state):
        path = state["path"]
        replays = state["replays"]
        table = _memory_mapped_arrow_table_from_file(path)
        table = self._apply_replays(table, replays)
        MemoryMappedTable.__init__(self, table, path=path, replays=replays)

    @staticmethod
    def _apply_replays(table: pa.Table, replays: Optional[list[Replay]] = None) -> pa.Table:
        if replays is not None:
            for name, args, kwargs in replays:
                if name == "cast":
                    table = table_cast(table, *args, **kwargs)
                elif name == "flatten":
                    table = table_flatten(table, *args, **kwargs)
                else:
                    table = getattr(table, name)(*args, **kwargs)
        return table

    def _append_replay(self, replay: Replay) -> list[Replay]:
        replays = copy.deepcopy(self.replays)
        replays.append(replay)
        return replays

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        replay = ("slice", (offset, length), {})
        replays = self._append_replay(replay)
        # Use fast slicing here
        return MemoryMappedTable(self.fast_slice(offset=offset, length=length), self.path, replays)

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        replay = ("filter", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.filter(*args, **kwargs), self.path, replays)

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        replay = ("flatten", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_flatten(self.table, *args, **kwargs), self.path, replays)

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the ChunkedArray of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        replay = ("combine_chunks", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.combine_chunks(*args, **kwargs), self.path, replays)

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        replay = ("cast", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_cast(self.table, *args, **kwargs), self.path, replays)

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be None,
        which deletes any existing metadata.

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        replay = ("replace_schema_metadata", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.replace_schema_metadata(*args, **kwargs), self.path, replays)

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        replay = ("add_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.add_column(*args, **kwargs), self.path, replays)

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        replay = ("append_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.append_column(*args, **kwargs), self.path, replays)

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        replay = ("remove_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.remove_column(*args, **kwargs), self.path, replays)

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        replay = ("set_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.set_column(*args, **kwargs), self.path, replays)

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        replay = ("rename_columns", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.rename_columns(*args, **kwargs), self.path, replays)

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        replay = ("drop", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.drop(*args, **kwargs), self.path, replays)

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        replay = ("select", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.select(*args, **kwargs), self.path, replays)


# A ConcatenationTable is the concatenation of several tables.
# The ``blocks`` attributes stores a list of list of blocks.
# The first axis concatenates the tables along the axis 0 (it appends rows),
# while the second axis concatenates tables along the axis 1 (it appends columns).
TableBlockContainer = TypeVar("TableBlockContainer", TableBlock, list[TableBlock], list[list[TableBlock]])


class ConcatenationTable(Table):
    """
    The table comes from the concatenation of several tables called blocks.
    It enables concatenation on both axis 0 (append rows) and axis 1 (append columns).

    The underlying tables are called "blocks" and can be either `InMemoryTable`
    or `MemoryMappedTable` objects.
    This allows to combine tables that come from memory or that are memory mapped.
    When a `ConcatenationTable` is pickled, then each block is pickled:
    - the `InMemoryTable` objects are pickled by copying all the data in memory.
    - the MemoryMappedTable objects are pickled without copying the data into memory.
    Instead, only the path to the memory mapped arrow file is pickled, as well as the list
    of transforms to "replays" when reloading the table from the disk.

    Its implementation requires to store each block separately.
    The `blocks` attributes stores a list of list of blocks.
    The first axis concatenates the tables along the axis 0 (it appends rows),
    while the second axis concatenates tables along the axis 1 (it appends columns).

    If some columns are missing when concatenating on axis 0, they are filled with null values.
    This is done using `pyarrow.concat_tables(tables, promote=True)`.

    You can access the fully combined table by accessing the `ConcatenationTable.table` attribute,
    and the blocks by accessing the `ConcatenationTable.blocks` attribute.
    """

    def __init__(self, table: pa.Table, blocks: list[list[TableBlock]]):
        super().__init__(table)
        self.blocks = blocks
        # Check that all the blocks have the right type.
        # Only InMemoryTable and MemoryMappedTable are allowed.
        for subtables in blocks:
            for subtable in subtables:
                if not isinstance(subtable, TableBlock):
                    raise TypeError(
                        "The blocks of a ConcatenationTable must be InMemoryTable or MemoryMappedTable objects"
                        f", but got {_short_str(subtable)}."
                    )

    def __getstate__(self):
        return {"blocks": self.blocks, "schema": self.table.schema}

    def __setstate__(self, state):
        blocks = state["blocks"]
        schema = state["schema"]
        table = self._concat_blocks_horizontally_and_vertically(blocks)
        if schema is not None and table.schema != schema:
            # We fix the columns by concatenating with an empty table with the right columns
            empty_table = pa.Table.from_batches([], schema=schema)
            # We set promote_options="default" to fill missing columns with null values
            table = pa.concat_tables([table, empty_table], promote_options="default")
        ConcatenationTable.__init__(self, table, blocks=blocks)

    @staticmethod
    def _concat_blocks(blocks: list[Union[TableBlock, pa.Table]], axis: int = 0) -> pa.Table:
        pa_tables = [table.table if hasattr(table, "table") else table for table in blocks]
        if axis == 0:
            # We set promote_options="default" to fill missing columns with null values
            return pa.concat_tables(pa_tables, promote_options="default")
        elif axis == 1:
            for i, table in enumerate(pa_tables):
                if i == 0:
                    pa_table = table
                else:
                    for name, col in zip(table.column_names, table.columns):
                        pa_table = pa_table.append_column(name, col)
            return pa_table
        else:
            raise ValueError("'axis' must be either 0 or 1")

    @classmethod
    def _concat_blocks_horizontally_and_vertically(cls, blocks: list[list[TableBlock]]) -> pa.Table:
        pa_tables_to_concat_vertically = []
        for i, tables in enumerate(blocks):
            if not tables:
                continue
            pa_table_horizontally_concatenated = cls._concat_blocks(tables, axis=1)
            pa_tables_to_concat_vertically.append(pa_table_horizontally_concatenated)
        return cls._concat_blocks(pa_tables_to_concat_vertically, axis=0)

    @classmethod
    def _merge_blocks(cls, blocks: TableBlockContainer, axis: Optional[int] = None) -> TableBlockContainer:
        if axis is not None:
            merged_blocks = []
            for is_in_memory, block_group in groupby(blocks, key=lambda x: isinstance(x, InMemoryTable)):
                if is_in_memory:
                    block_group = [InMemoryTable(cls._concat_blocks(list(block_group), axis=axis))]
                merged_blocks += list(block_group)
        else:  # both
            merged_blocks = [cls._merge_blocks(row_block, axis=1) for row_block in blocks]
            if all(len(row_block) == 1 for row_block in merged_blocks):
                merged_blocks = cls._merge_blocks(
                    [block for row_block in merged_blocks for block in row_block], axis=0
                )
        return merged_blocks

    @classmethod
    def _consolidate_blocks(cls, blocks: TableBlockContainer) -> TableBlockContainer:
        if isinstance(blocks, TableBlock):
            return blocks
        elif isinstance(blocks[0], TableBlock):
            return cls._merge_blocks(blocks, axis=0)
        else:
            return cls._merge_blocks(blocks)

    @classmethod
    def from_blocks(cls, blocks: TableBlockContainer) -> "ConcatenationTable":
        blocks = cls._consolidate_blocks(blocks)
        if isinstance(blocks, TableBlock):
            table = blocks
            return cls(table.table, [[table]])
        elif isinstance(blocks[0], TableBlock):
            table = cls._concat_blocks(blocks, axis=0)
            blocks = [[t] for t in blocks]
            return cls(table, blocks)
        else:
            table = cls._concat_blocks_horizontally_and_vertically(blocks)
            return cls(table, blocks)

    @classmethod
    def from_tables(cls, tables: list[Union[pa.Table, Table]], axis: int = 0) -> "ConcatenationTable":
        """Create `ConcatenationTable` from list of tables.

        Args:
            tables (list of `Table` or list of `pyarrow.Table`):
                List of tables.
            axis (`{0, 1}`, defaults to `0`, meaning over rows):
                Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
                (horizontally).

                <Added version="1.6.0"/>
        """

        def to_blocks(table: Union[pa.Table, Table]) -> list[list[TableBlock]]:
            if isinstance(table, pa.Table):
                return [[InMemoryTable(table)]]
            elif isinstance(table, ConcatenationTable):
                return copy.deepcopy(table.blocks)
            else:
                return [[table]]

        def _slice_row_block(row_block: list[TableBlock], length: int) -> tuple[list[TableBlock], list[TableBlock]]:
            sliced = [table.slice(0, length) for table in row_block]
            remainder = [table.slice(length, len(row_block[0]) - length) for table in row_block]
            return sliced, remainder

        def _split_both_like(
            result: list[list[TableBlock]], blocks: list[list[TableBlock]]
        ) -> tuple[list[list[TableBlock]], list[list[TableBlock]]]:
            """
            Make sure each row_block contain the same num_rows to be able to concatenate them on axis=1.

            To do so, we modify both blocks sets to have the same row_blocks boundaries.
            For example, if `result` has 2 row_blocks of 3 rows and `blocks` has 3 row_blocks of 2 rows,
            we modify both to have 4 row_blocks of size 2, 1, 1 and 2:

                    [ x   x   x | x   x   x ]
                +   [ y   y | y   y | y   y ]
                -----------------------------
                =   [ x   x | x | x | x   x ]
                    [ y   y | y | y | y   y ]

            """
            result, blocks = list(result), list(blocks)
            new_result, new_blocks = [], []
            while result and blocks:
                # we slice the longest row block to save two row blocks of same length
                # and we replace the long row block by its remainder if necessary
                if len(result[0][0]) > len(blocks[0][0]):
                    new_blocks.append(blocks[0])
                    sliced, result[0] = _slice_row_block(result[0], len(blocks.pop(0)[0]))
                    new_result.append(sliced)
                elif len(result[0][0]) < len(blocks[0][0]):
                    new_result.append(result[0])
                    sliced, blocks[0] = _slice_row_block(blocks[0], len(result.pop(0)[0]))
                    new_blocks.append(sliced)
                else:
                    new_result.append(result.pop(0))
                    new_blocks.append(blocks.pop(0))
            if result or blocks:
                raise ValueError("Failed to concatenate on axis=1 because tables don't have the same number of rows")
            return new_result, new_blocks

        def _extend_blocks(
            result: list[list[TableBlock]], blocks: list[list[TableBlock]], axis: int = 0
        ) -> list[list[TableBlock]]:
            if axis == 0:
                result.extend(blocks)
            elif axis == 1:
                # We make sure each row_block have the same num_rows
                result, blocks = _split_both_like(result, blocks)
                for i, row_block in enumerate(blocks):
                    result[i].extend(row_block)
            return result

        blocks = to_blocks(tables[0])
        for table in tables[1:]:
            table_blocks = to_blocks(table)
            blocks = _extend_blocks(blocks, table_blocks, axis=axis)
        return cls.from_blocks(blocks)

    @property
    def _slices(self):
        offset = 0
        for tables in self.blocks:
            length = len(tables[0])
            yield (offset, length)
            offset += length

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        table = self.table.slice(offset, length=length)
        length = length if length is not None else self.num_rows - offset
        blocks = []
        for tables in self.blocks:
            n_rows = len(tables[0])
            if length == 0:
                break
            elif n_rows <= offset:
                offset = offset - n_rows
            elif n_rows <= offset + length:
                blocks.append([t.slice(offset) for t in tables])
                length, offset = length + offset - n_rows, 0
            else:
                blocks.append([t.slice(offset, length) for t in tables])
                length, offset = 0, 0
        return ConcatenationTable(table, blocks)

    def filter(self, mask, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        table = self.table.filter(mask, *args, **kwargs)
        blocks = []
        for (offset, length), tables in zip(self._slices, self.blocks):
            submask = mask.slice(offset, length)
            blocks.append([t.filter(submask, *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        table = table_flatten(self.table, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.flatten(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        table = self.table.combine_chunks(*args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.combine_chunks(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def cast(self, target_schema, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        from .features import Features

        table = table_cast(self.table, target_schema, *args, **kwargs)
        target_features = Features.from_arrow_schema(target_schema)
        blocks = []
        for subtables in self.blocks:
            new_tables = []
            fields = list(target_schema)
            for subtable in subtables:
                subfields = []
                for name in subtable.column_names:
                    subfields.append(fields.pop(next(i for i, field in enumerate(fields) if field.name == name)))
                subfeatures = Features({subfield.name: target_features[subfield.name] for subfield in subfields})
                subschema = subfeatures.arrow_schema
                new_tables.append(subtable.cast(subschema, *args, **kwargs))
            blocks.append(new_tables)
        return ConcatenationTable(table, blocks)

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be `None`,
        which deletes any existing metadata).

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        table = self.table.replace_schema_metadata(*args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.replace_schema_metadata(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, self.blocks)

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        raise NotImplementedError()

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        raise NotImplementedError()

    def remove_column(self, i, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        table = self.table.remove_column(i, *args, **kwargs)
        name = self.table.column_names[i]
        blocks = []
        for tables in self.blocks:
            blocks.append(
                [
                    t.remove_column(t.column_names.index(name), *args, **kwargs) if name in t.column_names else t
                    for t in tables
                ]
            )
        return ConcatenationTable(table, blocks)

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        raise NotImplementedError()

    def rename_columns(self, names, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        table = self.table.rename_columns(names, *args, **kwargs)
        names = dict(zip(self.table.column_names, names))
        blocks = []
        for tables in self.blocks:
            blocks.append(
                [t.rename_columns([names[name] for name in t.column_names], *args, **kwargs) for t in tables]
            )
        return ConcatenationTable(table, blocks)

    def drop(self, columns, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        table = self.table.drop(columns, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.drop([c for c in columns if c in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def select(self, columns, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        table = self.table.select(columns, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.select([c for c in columns if c in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)


def concat_tables(tables: list[Table], axis: int = 0) -> Table:
    """
    Concatenate tables.

    Args:
        tables (list of `Table`):
            List of tables to be concatenated.
        axis (`{0, 1}`, defaults to `0`, meaning over rows):
            Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
            (horizontally).

            <Added version="1.6.0"/>
    Returns:
        `datasets.table.Table`:
            If the number of input tables is > 1, then the returned table is a `datasets.table.ConcatenationTable`.
            Otherwise if there's only one table, it is returned as is.
    """
    tables = list(tables)
    if len(tables) == 1:
        return tables[0]
    return ConcatenationTable.from_tables(tables, axis=axis)


def list_table_cache_files(table: Table) -> list[str]:
    """
    Get the cache files that are loaded by the table.
    Cache file are used when parts of the table come from the disk via memory mapping.

    Returns:
        `List[str]`:
            A list of paths to the cache files loaded by the table.
    """
    if isinstance(table, ConcatenationTable):
        cache_files = []
        for subtables in table.blocks:
            for subtable in subtables:
                cache_files += list_table_cache_files(subtable)
        return cache_files
    elif isinstance(table, MemoryMappedTable):
        return [table.path]
    else:
        return []


def _wrap_for_chunked_arrays(func):
    """Apply the function on each chunk of a `pyarrow.ChunkedArray`, or on the array directly"""

    def wrapper(array, *args, **kwargs):
        if isinstance(array, pa.ChunkedArray):
            return pa.chunked_array([func(chunk, *args, **kwargs) for chunk in array.chunks])
        else:
            return func(array, *args, **kwargs)

    return wrapper


def _are_list_values_of_length(array: pa.ListArray, length: int) -> bool:
    """Check if all the sub-lists of a `pa.ListArray` have the specified length."""
    return pc.all(pc.equal(array.value_lengths(), length)).as_py() or array.null_count == len(array)


def _combine_list_array_offsets_with_mask(array: pa.ListArray) -> pa.Array:
    """Add the null bitmap to the offsets of a `pa.ListArray`."""
    offsets = array.offsets
    if array.null_count > 0:
        offsets = pa.concat_arrays(
            [
                pc.replace_with_mask(offsets[:-1], array.is_null(), pa.nulls(len(array), offsets.type)),
                offsets[-1:],
            ]
        )
    return offsets


def _storage_type(type: pa.DataType) -> pa.DataType:
    """Convert a (possibly nested) `pa.ExtensionType` to its storage type."""
    if isinstance(type, pa.ExtensionType):
        return _storage_type(type.storage_type)
    elif isinstance(type, pa.StructType):
        return pa.struct([pa.field(field.name, _storage_type(field.type)) for field in type])
    elif isinstance(type, pa.ListType):
        return pa.list_(_storage_type(type.value_type))
    elif isinstance(type, pa.FixedSizeListType):
        return pa.list_(_storage_type(type.value_type), type.list_size)
    return type


def _short_str(value: Any) -> str:
    out = str(value)
    if len(out) > 3000:
        out = out[:1500] + "\n...\n" + out[-1500:]
    return out


@_wrap_for_chunked_arrays
def array_cast(
    array: pa.Array, pa_type: pa.DataType, allow_primitive_to_str: bool = True, allow_decimal_to_str: bool = True
) -> Union[pa.Array, pa.FixedSizeListArray, pa.ListArray, pa.StructArray, pa.ExtensionArray]:
    """Improved version of `pa.Array.cast`

    It supports casting `pa.StructArray` objects to re-order the fields.
    It also let you control certain aspects of the casting, e.g. whether
    to disable casting primitives (`booleans`, `floats` or `ints`) or
    disable casting decimals to strings.

    Args:
        array (`pa.Array`):
            PyArrow array to cast
        pa_type (`pa.DataType`):
            Target PyArrow type
        allow_primitive_to_str (`bool`, defaults to `True`):
            Whether to allow casting primitives to strings.
            Defaults to `True`.
        allow_decimal_to_str (`bool`, defaults to `True`):
            Whether to allow casting decimals to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from primitives to strings and `allow_primitive_to_str` is `False`
            - if casting from decimals to strings and `allow_decimal_to_str` is `False`

    Returns:
        `List[pyarrow.Array]`: the casted array
    """
    _c = partial(array_cast, allow_primitive_to_str=allow_primitive_to_str, allow_decimal_to_str=allow_decimal_to_str)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if isinstance(pa_type, pa.ExtensionType):
        return pa_type.wrap_array(_c(array, pa_type.storage_type))
    elif array.type == pa_type:
        return array
    elif pa.types.is_struct(array.type):
        if pa.types.is_struct(pa_type) and ({field.name for field in pa_type} == {field.name for field in array.type}):
            if array.type.num_fields == 0:
                return array
            arrays = [_c(array.field(field.name), field.type) for field in pa_type]
            return pa.StructArray.from_arrays(arrays, fields=list(pa_type), mask=array.is_null())
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if _are_list_values_of_length(array, pa_type.list_size):
                if array.null_count > 0:
                    # Ensure each null value in the array translates to [null] * pa_type.list_size in the array's values array
                    array_type = array.type
                    storage_type = _storage_type(array_type)
                    if array_type != storage_type:
                        # Temporarily convert to the storage type to support extension types in the slice operation
                        array = _c(array, storage_type)
                        array = pc.list_slice(array, 0, pa_type.list_size, return_fixed_size_list=True)
                        array = _c(array, array_type)
                    else:
                        array = pc.list_slice(array, 0, pa_type.list_size, return_fixed_size_list=True)
                    array_values = array.values
                    return pa.FixedSizeListArray.from_arrays(
                        _c(array_values, pa_type.value_type), pa_type.list_size, mask=array.is_null()
                    )
                else:
                    array_values = array.values[
                        array.offset * pa_type.list_size : (array.offset + len(array)) * pa_type.list_size
                    ]
                    return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size)
        elif pa.types.is_list(pa_type):
            # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
            array_offsets = _combine_list_array_offsets_with_mask(array)
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type))
        elif pa.types.is_large_list(pa_type):
            # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
            array_offsets = _combine_list_array_offsets_with_mask(array)
            return pa.LargeListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type))
    elif pa.types.is_fixed_size_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if pa_type.list_size == array.type.list_size:
                array_values = array.values[
                    array.offset * array.type.list_size : (array.offset + len(array)) * array.type.list_size
                ]
                return pa.FixedSizeListArray.from_arrays(
                    _c(array_values, pa_type.value_type), pa_type.list_size, mask=array.is_null()
                )
        elif pa.types.is_list(pa_type):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type), mask=array.is_null())
        elif pa.types.is_large_list(pa_type):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.LargeListArray.from_arrays(
                array_offsets, _c(array.values, pa_type.value_type), mask=array.is_null()
            )
    else:
        if pa.types.is_string(pa_type):
            if not allow_primitive_to_str and pa.types.is_primitive(array.type):
                raise TypeError(
                    f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)} "
                    f"since allow_primitive_to_str is set to {allow_primitive_to_str} "
                )
            if not allow_decimal_to_str and pa.types.is_decimal(array.type):
                raise TypeError(
                    f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)} "
                    f"and allow_decimal_to_str is set to {allow_decimal_to_str}"
                )
        if pa.types.is_null(pa_type) and not pa.types.is_null(array.type):
            raise TypeError(f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)}")
        return array.cast(pa_type)
    raise TypeError(f"Couldn't cast array of type {_short_str(array.type)} to {_short_str(pa_type)}")


@_wrap_for_chunked_arrays
def cast_array_to_feature(
    array: pa.Array, feature: "FeatureType", allow_primitive_to_str: bool = True, allow_decimal_to_str: bool = True
) -> pa.Array:
    """Cast an array to the arrow type that corresponds to the requested feature type.
    For custom features like [`Audio`] or [`Image`], it takes into account the "cast_storage" methods
    they defined to enable casting from other arrow types.

    Args:
        array (`pa.Array`):
            The PyArrow array to cast.
        feature (`datasets.features.FeatureType`):
            The target feature type.
        allow_primitive_to_str (`bool`, defaults to `True`):
            Whether to allow casting primitives to strings.
            Defaults to `True`.
        allow_decimal_to_str (`bool`, defaults to `True`):
            Whether to allow casting decimals to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from primitives and `allow_primitive_to_str` is `False`
            - if casting from decimals and `allow_decimal_to_str` is `False`

    Returns:
        array (`pyarrow.Array`): the casted array
    """
    from .features.features import LargeList, List, get_nested_type

    _c = partial(
        cast_array_to_feature,
        allow_primitive_to_str=allow_primitive_to_str,
        allow_decimal_to_str=allow_decimal_to_str,
    )

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "cast_storage"):
        return feature.cast_storage(array)

    if pa.types.is_struct(array.type):
        # feature must be a dict
        if isinstance(feature, dict) and (array_fields := {field.name for field in array.type}) <= set(feature):
            null_array = pa.array([None] * len(array))
            # flatten() nulls the children under null struct rows, whose values are otherwise placeholders
            # (e.g. "") that a child's cast_storage could read as data
            children = dict(zip([field.name for field in array.type], array.flatten()))
            arrays = [
                _c(children[name] if name in array_fields else null_array, subfeature)
                for name, subfeature in feature.items()
            ]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        # feature must be either List(subfeature) or LargeList(subfeature)
        if isinstance(feature, LargeList):
            casted_array_values = _c(array.values, feature.feature)
            if pa.types.is_large_list(array.type) and casted_array_values.type == array.values.type:
                # Both array and feature have equal large_list type and values (within the list) type
                return array
            else:
                # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                array_offsets = _combine_list_array_offsets_with_mask(array)
                return pa.LargeListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(feature, List):
            if feature.length > -1:
                if _are_list_values_of_length(array, feature.length):
                    if array.null_count > 0:
                        # Ensure each null value in the array translates to [null] * pa_type.list_size in the array's values array
                        array_type = array.type
                        storage_type = _storage_type(array_type)
                        if array_type != storage_type:
                            # Temporarily convert to the storage type to support extension types in the slice operation
                            array = array_cast(
                                array,
                                storage_type,
                                allow_primitive_to_str=allow_primitive_to_str,
                                allow_decimal_to_str=allow_decimal_to_str,
                            )
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                            array = array_cast(
                                array,
                                array_type,
                                allow_primitive_to_str=allow_primitive_to_str,
                                allow_decimal_to_str=allow_decimal_to_str,
                            )
                        else:
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                        array_values = array.values
                        casted_array_values = _c(array_values, feature.feature)
                        return pa.FixedSizeListArray.from_arrays(
                            casted_array_values, feature.length, mask=array.is_null()
                        )
                    else:
                        array_values = array.values[
                            array.offset * feature.length : (array.offset + len(array)) * feature.length
                        ]
                        return pa.FixedSizeListArray.from_arrays(_c(array_values, feature.feature), feature.length)
            else:
                casted_array_values = _c(array.values, feature.feature)
                if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                    # Both array and feature have equal list type and values (within the list) type
                    return array
                else:
                    # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                    array_offsets = _combine_list_array_offsets_with_mask(array)
                    return pa.ListArray.from_arrays(array_offsets, casted_array_values)
    elif pa.types.is_fixed_size_list(array.type):
        # feature must be List(subfeature)
        if isinstance(feature, LargeList):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.LargeListArray.from_arrays(
                array_offsets, _c(array.values, feature.feature), mask=array.is_null()
            )
        elif isinstance(feature, List):
            if feature.length > -1:
                if feature.length == array.type.list_size:
                    array_values = array.values[
                        array.offset * array.type.list_size : (array.offset + len(array)) * array.type.list_size
                    ]
                    casted_array_values = _c(array_values, feature.feature)
                    return pa.FixedSizeListArray.from_arrays(casted_array_values, feature.length, mask=array.is_null())
            else:
                array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
                return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature.feature), mask=array.is_null())
    if pa.types.is_null(array.type):
        return array_cast(
            array,
            get_nested_type(feature),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    elif not isinstance(feature, (List, LargeList, dict)):
        return array_cast(
            array,
            feature(),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    raise TypeError(f"Couldn't cast array of type\n{_short_str(array.type)}\nto\n{_short_str(feature)}")


@_wrap_for_chunked_arrays
def embed_array_storage(
    array: pa.Array,
    feature: "FeatureType",
    token_per_repo_id=None,
    local_files: bool = True,
    remote_files: bool = True,
):
    """Embed data into an arrays's storage.
    For custom features like Audio or Image, it takes into account the "embed_storage" methods
    they define to embed external data (e.g. an image file) into an array.

    <Added version="2.4.0"/>

    Args:
        array (`pa.Array`):
            The PyArrow array in which to embed data.
        feature (`datasets.features.FeatureType`):
            Array features.
        local_files (`bool`, defaults to `True`)
            Whether to embed local files data in the array

            <Added version="4.8.5"/>
        remote_files (`bool`, defaults to `True`)
            Whether to embed remote files data in the array.
            E.g. files with paths that start with hf:// or https://

            <Added version="4.8.5"/>

    Raises:
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing

    Returns:
         array (`pyarrow.Array`): the casted array
    """
    if not local_files and not remote_files:
        return array

    from .features import LargeList, List

    _e = partial(
        embed_array_storage, token_per_repo_id=token_per_repo_id, local_files=local_files, remote_files=remote_files
    )

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "embed_storage"):
        return feature.embed_storage(
            array, token_per_repo_id=token_per_repo_id, local_files=local_files, remote_files=remote_files
        )
    elif pa.types.is_struct(array.type):
        # feature must be a dict
        if isinstance(feature, dict):
            arrays = [_e(array.field(name), subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type):
        # feature must be either List(subfeature)
        # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
        array_offsets = _combine_list_array_offsets_with_mask(array)
        if isinstance(feature, List) and feature.length == -1:
            return pa.ListArray.from_arrays(array_offsets, _e(array.values, feature.feature))
    elif pa.types.is_large_list(array.type):
        # feature must be LargeList(subfeature)
        # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
        array_offsets = _combine_list_array_offsets_with_mask(array)
        return pa.LargeListArray.from_arrays(array_offsets, _e(array.values, feature.feature))
    elif pa.types.is_fixed_size_list(array.type):
        # feature must be List(subfeature)
        if isinstance(feature, List) and feature.length > -1:
            array_values = array.values[
                array.offset * array.type.list_size : (array.offset + len(array)) * array.type.list_size
            ]
            embedded_array_values = _e(array_values, feature.feature)
            return pa.FixedSizeListArray.from_arrays(embedded_array_values, feature.length, mask=array.is_null())
    if not isinstance(feature, (List, LargeList, dict)):
        return array
    raise TypeError(f"Couldn't embed array of type\n{_short_str(array.type)}\nwith\n{_short_str(feature)}")


class CastError(ValueError):
    """When it's not possible to cast an Arrow table to a specific schema or set of features"""

    def __init__(self, *args, table_column_names: list[str], requested_column_names: list[str]) -> None:
        super().__init__(*args)
        self.table_column_names = table_column_names
        self.requested_column_names = requested_column_names

    def __reduce__(self):
        # Fix unpickling: TypeError: __init__() missing 2 required keyword-only arguments: 'table_column_names' and 'requested_column_names'
        return partial(
            CastError, table_column_names=self.table_column_names, requested_column_names=self.requested_column_names
        ), ()

    def details(self):
        new_columns = set(self.table_column_names) - set(self.requested_column_names)
        missing_columns = set(self.requested_column_names) - set(self.table_column_names)
        if new_columns and missing_columns:
            return f"there are {len(new_columns)} new columns ({_short_str(new_columns)}) and {len(missing_columns)} missing columns ({_short_str(missing_columns)})."
        elif new_columns:
            return f"there are {len(new_columns)} new columns ({_short_str(new_columns)})"
        else:
            return f"there are {len(missing_columns)} missing columns ({_short_str(missing_columns)})"


def cast_table_to_features(table: pa.Table, features: "Features"):
    """Cast a table to the arrow schema that corresponds to the requested features.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        features ([`Features`]):
            Target features.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if sorted(table.column_names) != sorted(features):
        raise CastError(
            f"Couldn't cast\n{_short_str(table.schema)}\nto\n{_short_str(features)}\nbecause column names don't match",
            table_column_names=table.column_names,
            requested_column_names=list(features),
        )
    arrays = [cast_array_to_feature(table[name], feature) for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)


def cast_table_to_schema(table: pa.Table, schema: pa.Schema):
    """Cast a table to the arrow schema. Different from `cast_table_to_features`, this method can preserve nullability.

    Args:
        table (`pa.Table`):
            PyArrow table to cast.
        features ([`Features`]):
            Target features.

    Returns:
        `pa.Table`: the casted table
    """
    from .features import Features

    features = Features.from_arrow_schema(schema)
    table_column_names = set(table.column_names)
    if not table_column_names <= set(schema.names):
        raise CastError(
            f"Couldn't cast\n{_short_str(table.schema)}\nto\n{_short_str(features)}\nbecause column names don't match",
            table_column_names=table.column_names,
            requested_column_names=list(features),
        )
    arrays = [
        cast_array_to_feature(
            table[name] if name in table_column_names else pa.array([None] * len(table), type=schema.field(name).type),
            feature,
        )
        for name, feature in features.items()
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def embed_table_storage(table: pa.Table, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True):
    """Embed external data into a table's storage.

    <Added version="2.4.0"/>

    Args:
        table (`pyarrow.Table`):
            PyArrow table in which to embed data.
        local_files (`bool`, defaults to `True`)
            Whether to embed local files data in the table

            <Added version="4.8.5"/>
        remote_files (`bool`, defaults to `True`)
            Whether to embed remote files data in the table.
            E.g. files with paths that start with hf:// or https://

            <Added version="4.8.5"/>

    Returns:
        table (`pyarrow.Table`): the table with embedded data
    """
    if not local_files and not remote_files:
        return table

    from .features.features import Features, require_storage_embed

    features = Features.from_arrow_schema(table.schema)
    arrays = [
        embed_array_storage(
            table[name],
            feature,
            token_per_repo_id=token_per_repo_id,
            local_files=local_files,
            remote_files=remote_files,
        )
        if require_storage_embed(feature)
        else table[name]
        for name, feature in features.items()
    ]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)


def table_cast(table: pa.Table, schema: pa.Schema):
    """Improved version of `pa.Table.cast`.

    It supports casting to feature types stored in the schema metadata.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        schema (`pyarrow.Schema`):
            Target PyArrow schema.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if table.schema != schema:
        return cast_table_to_schema(table, schema)
    elif table.schema.metadata != schema.metadata:
        return table.replace_schema_metadata(schema.metadata)
    else:
        return table


def table_flatten(table: pa.Table):
    """Improved version of `pa.Table.flatten`.

    It behaves as `pa.Table.flatten` in a sense it does 1-step flatten of the columns with a struct type into one column per struct field,
    but updates the metadata and skips decodable features unless the `decode` attribute of these features is set to False.

    Args:
        table (`pa.Table`):
            PyArrow table to flatten.

    Returns:
        `Table`: the flattened table
    """
    from .features import Features

    features = Features.from_arrow_schema(table.schema)
    if any(hasattr(subfeature, "flatten") and subfeature.flatten() == subfeature for subfeature in features.values()):
        flat_arrays = []
        flat_column_names = []
        for field in table.schema:
            array = table.column(field.name)
            subfeature = features[field.name]
            if pa.types.is_struct(field.type) and (
                not hasattr(subfeature, "flatten") or subfeature.flatten() != subfeature
            ):
                flat_arrays.extend(array.flatten())
                flat_column_names.extend([f"{field.name}.{subfield.name}" for subfield in field.type])
            else:
                flat_arrays.append(array)
                flat_column_names.append(field.name)
        flat_table = pa.Table.from_arrays(
            flat_arrays,
            names=flat_column_names,
        )
    else:
        flat_table = table.flatten()
    # Preserve complex types in the metadata
    flat_features = features.flatten(max_depth=2)
    flat_features = Features({column_name: flat_features[column_name] for column_name in flat_table.column_names})
    return flat_table.replace_schema_metadata(flat_features.arrow_schema.metadata)


def table_visitor(table: pa.Table, function: Callable[[pa.Array], None]):
    """Visit all arrays in a table and apply a function to them.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to visit.
        function (`Callable[[pa.Array], None]`):
            Function to apply to each array.
    """
    from .features import Features, LargeList, List

    features = Features.from_arrow_schema(table.schema)

    def _visit(array, feature):
        if isinstance(array, pa.ChunkedArray):
            for chunk in array.chunks:
                _visit(chunk, feature)
        else:
            if isinstance(array, pa.ExtensionArray):
                array = array.storage
            function(array, feature)
            if pa.types.is_struct(array.type) and not hasattr(feature, "cast_storage"):
                for name, subfeature in feature.items():
                    _visit(array.field(name), subfeature)
            elif pa.types.is_list(array.type):
                if isinstance(feature, (LargeList, List)):
                    _visit(array.values, feature.feature)

    for name, feature in features.items():
        _visit(table[name], feature)


def table_iter(table: Table, batch_size: int, drop_last_batch=False) -> Iterator[pa.Table]:
    """Iterate over sub-tables of size `batch_size`.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to iterate over.
        batch_size (`int`):
            Size of each sub-table to yield.
        drop_last_batch (`bool`, defaults to `False`):
            Drop the last batch if it is smaller than `batch_size`.
    """
    chunks_buffer = []
    chunks_buffer_size = 0
    for chunk in table.to_reader(max_chunksize=batch_size):
        if len(chunk) == 0:
            continue
        elif chunks_buffer_size + len(chunk) < batch_size:
            chunks_buffer.append(chunk)
            chunks_buffer_size += len(chunk)
            continue
        elif chunks_buffer_size + len(chunk) == batch_size:
            chunks_buffer.append(chunk)
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = []
            chunks_buffer_size = 0
        else:
            cropped_chunk_length = batch_size - chunks_buffer_size
            chunks_buffer.append(chunk.slice(0, cropped_chunk_length))
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = [chunk.slice(cropped_chunk_length, len(chunk) - cropped_chunk_length)]
            chunks_buffer_size = len(chunk) - cropped_chunk_length
    if not drop_last_batch and chunks_buffer:
        yield pa.Table.from_batches(chunks_buffer)
<|endoftext|>import importlib
import importlib.metadata
import logging
import os
import platform
from pathlib import Path
from typing import Optional

from huggingface_hub import constants
from packaging import version


logger = logging.getLogger(__name__.split(".", 1)[0])  # to avoid circular import from .utils.logging

# Datasets
S3_DATASETS_BUCKET_PREFIX = "https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets"
CLOUDFRONT_DATASETS_DISTRIB_PREFIX = "https://cdn-datasets.huggingface.co/datasets/datasets"
REPO_DATASETS_URL = "https://raw.githubusercontent.com/huggingface/datasets/{revision}/datasets/{path}/{name}"

# Hub
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HUB_DATASETS_URL = HF_ENDPOINT + "/datasets/{repo_id}/resolve/{revision}/{path}"
HUB_DATASETS_HFFS_URL = "hf://datasets/{repo_id}@{revision}/{path}"
HUB_DEFAULT_VERSION = "main"

PY_VERSION = version.parse(platform.python_version())

# General environment variables accepted values for booleans
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_FALSE_VALUES = {"0", "OFF", "NO", "FALSE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})
ENV_VARS_FALSE_AND_AUTO_VALUES = ENV_VARS_FALSE_VALUES.union({"AUTO"})


# Imports
DILL_VERSION = version.parse(importlib.metadata.version("dill"))
FSSPEC_VERSION = version.parse(importlib.metadata.version("fsspec"))
PANDAS_VERSION = version.parse(importlib.metadata.version("pandas"))
PYARROW_VERSION = version.parse(importlib.metadata.version("pyarrow"))
HF_HUB_VERSION = version.parse(importlib.metadata.version("huggingface_hub"))

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()

TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
    if TORCH_AVAILABLE:
        try:
            TORCH_VERSION = version.parse(importlib.metadata.version("torch"))
            logger.debug(f"PyTorch version {TORCH_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling PyTorch because USE_TF is set")

POLARS_VERSION = "N/A"
POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None

if POLARS_AVAILABLE:
    try:
        POLARS_VERSION = version.parse(importlib.metadata.version("polars"))
        logger.debug(f"Polars version {POLARS_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass


DUCKDB_VERSION = "N/A"
DUCKDB_AVAILABLE = importlib.util.find_spec("duckdb") is not None

if DUCKDB_AVAILABLE:
    try:
        DUCKDB_VERSION = version.parse(importlib.metadata.version("duckdb"))
        logger.debug(f"Duckdb version {DUCKDB_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

TF_VERSION = "N/A"
TF_AVAILABLE = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
    if TF_AVAILABLE:
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for package in [
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        ]:
            try:
                TF_VERSION = version.parse(importlib.metadata.version(package))
            except importlib.metadata.PackageNotFoundError:
                continue
            else:
                break
        else:
            TF_AVAILABLE = False
    if TF_AVAILABLE:
        if TF_VERSION.major < 2:
            logger.info(f"TensorFlow found but with version {TF_VERSION}. `datasets` requires version 2 minimum.")
            TF_AVAILABLE = False
        else:
            logger.info(f"TensorFlow version {TF_VERSION} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")


JAX_VERSION = "N/A"
JAX_AVAILABLE = False

if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    JAX_AVAILABLE = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
    if JAX_AVAILABLE:
        try:
            JAX_VERSION = version.parse(importlib.metadata.version("jax"))
            logger.info(f"JAX version {JAX_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling JAX because USE_JAX is set to False")


# Optional tools for data loading
SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None

# Optional tools for file parsing and feature decoding
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
IS_OPUS_SUPPORTED = True
IS_MP3_SUPPORTED = True
TORCHCODEC_AVAILABLE = importlib.util.find_spec("torchcodec") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
PDFPLUMBER_AVAILABLE = importlib.util.find_spec("pdfplumber") is not None
# biopython is imported as ``Bio`` (the PyPI package name is ``biopython``).
BIOPYTHON_AVAILABLE = importlib.util.find_spec("Bio") is not None
NIBABEL_AVAILABLE = importlib.util.find_spec("nibabel") is not None
TRIMESH_AVAILABLE = importlib.util.find_spec("trimesh") is not None
TEICH_AVAILABLE = importlib.util.find_spec("teich") is not None

# Optional compression tools
RARFILE_AVAILABLE = importlib.util.find_spec("rarfile") is not None
ZSTANDARD_AVAILABLE = importlib.util.find_spec("zstandard") is not None
LZ4_AVAILABLE = importlib.util.find_spec("lz4") is not None
PY7ZR_AVAILABLE = importlib.util.find_spec("py7zr") is not None

# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "huggingface")
HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", DEFAULT_HF_CACHE_HOME))

DEFAULT_HF_DATASETS_CACHE = os.path.join(HF_CACHE_HOME, "datasets")
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE))

DEFAULT_HF_MODULES_CACHE = os.path.join(HF_CACHE_HOME, "modules")
HF_MODULES_CACHE = Path(os.getenv("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE))

DOWNLOADED_DATASETS_DIR = "downloads"
DEFAULT_DOWNLOADED_DATASETS_PATH = os.path.join(HF_DATASETS_CACHE, DOWNLOADED_DATASETS_DIR)
DOWNLOADED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_DOWNLOADED_DATASETS_PATH", DEFAULT_DOWNLOADED_DATASETS_PATH))

EXTRACTED_DATASETS_DIR = "extracted"
DEFAULT_EXTRACTED_DATASETS_PATH = os.path.join(DEFAULT_DOWNLOADED_DATASETS_PATH, EXTRACTED_DATASETS_DIR)
EXTRACTED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_EXTRACTED_DATASETS_PATH", DEFAULT_EXTRACTED_DATASETS_PATH))

# Cached dataset info options
SAVE_ORIGINAL_SHARD_LENGTHS = False

# Download count for the website
HF_UPDATE_DOWNLOAD_COUNTS = (
    os.environ.get("HF_UPDATE_DOWNLOAD_COUNTS", "AUTO").upper() in ENV_VARS_TRUE_AND_AUTO_VALUES
)

# For downloads and to check remote files metadata
HF_DATASETS_MULTITHREADING_MAX_WORKERS = 16

# Dataset viewer API
USE_PARQUET_EXPORT = True

# Batch size constants. For more info, see:
# https://github.com/apache/arrow/blob/master/docs/source/cpp/arrays.rst#size-limitations-and-recommendations)
DEFAULT_MAX_BATCH_SIZE = 1000

DEFAULT_CDC_OPTIONS = {"min_chunk_size": 256 * 1024, "max_chunk_size": 1024 * 1024, "norm_level": 0}

# Size of the preloaded record batch in `Dataset.__iter__`
ARROW_READER_BATCH_SIZE_IN_DATASET_ITER = 10

# Max uncompressed shard size in bytes (e.g. to shard parquet datasets in push_to_hub or download_and_prepare)
MAX_SHARD_SIZE = "500MB"

# Max uncompressed row group size in bytes (e.g. for parquet files in push_to_hub or download_and_prepare)
MAX_ROW_GROUP_SIZE = "100MB"

# Parquet configuration
PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS = None
PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS = None
PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS = None
PARQUET_ROW_GROUP_SIZE_FOR_VIDEO_DATASETS = None

# Arrow configuration
ARROW_RECORD_BATCH_SIZE_FOR_AUDIO_DATASETS = 100
ARROW_RECORD_BATCH_SIZE_FOR_IMAGE_DATASETS = 100
ARROW_RECORD_BATCH_SIZE_FOR_BINARY_DATASETS = 100
ARROW_RECORD_BATCH_SIZE_FOR_VIDEO_DATASETS = 10

# Offline mode
_offline = os.environ.get("HF_DATASETS_OFFLINE")
HF_HUB_OFFLINE = constants.HF_HUB_OFFLINE if _offline is None else _offline.upper() in ENV_VARS_TRUE_VALUES
HF_DATASETS_OFFLINE = HF_HUB_OFFLINE  # kept for backward-compatibility

# Here, `True` will disable progress bars globally without possibility of enabling it
# programmatically. `False` will enable them without possibility of disabling them.
# If environment variable is not set (None), then the user is free to enable/disable
# them programmatically.
# TL;DR: env variable has priority over code
__HF_DATASETS_DISABLE_PROGRESS_BARS = os.environ.get("HF_DATASETS_DISABLE_PROGRESS_BARS")
HF_DATASETS_DISABLE_PROGRESS_BARS: Optional[bool] = (
    __HF_DATASETS_DISABLE_PROGRESS_BARS.upper() in ENV_VARS_TRUE_VALUES
    if __HF_DATASETS_DISABLE_PROGRESS_BARS is not None
    else None
)

# In-memory
DEFAULT_IN_MEMORY_MAX_SIZE = 0  # Disabled
IN_MEMORY_MAX_SIZE = float(os.environ.get("HF_DATASETS_IN_MEMORY_MAX_SIZE", DEFAULT_IN_MEMORY_MAX_SIZE))

# File names
DATASET_ARROW_FILENAME = "dataset.arrow"
DATASET_INDICES_FILENAME = "indices.arrow"
DATASET_STATE_JSON_FILENAME = "state.json"
DATASET_INFO_FILENAME = "dataset_info.json"
DATASETDICT_INFOS_FILENAME = "dataset_infos.json"
LICENSE_FILENAME = "LICENSE"
DATASETDICT_JSON_FILENAME = "dataset_dict.json"
METADATA_CONFIGS_FIELD = "configs"
REPOCARD_FILENAME = "README.md"
REPOYAML_FILENAME = ".huggingface.yaml"

MODULE_NAME_FOR_DYNAMIC_MODULES = "datasets_modules"

MAX_DATASET_CONFIG_ID_READABLE_LENGTH = 255

# Temporary cache directory prefix
TEMP_CACHE_DIR_PREFIX = "hf_datasets-"

# Streaming
STREAMING_READ_MAX_RETRIES = 20
STREAMING_READ_RETRY_INTERVAL = 5
STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL = 20
STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL = 60
STREAMING_OPEN_MAX_RETRIES = 20
STREAMING_OPEN_RETRY_INTERVAL = 5

# Datasets repositories exploration
ARCHIVES_MAX_NUMBER_FOR_MODULE_INFERENCE = 10

# Async map functions
MAX_NUM_RUNNING_ASYNC_MAP_FUNCTIONS_IN_PARALLEL = 1000

# Progress bars
PBAR_REFRESH_TIME_INTERVAL = 0.05  # 20 progress updates per sec

# Maximum number of uploaded files per commit
UPLOADS_MAX_NUMBER_PER_COMMIT = 50

# Backward compatibility
MAX_TABLE_NBYTES_FOR_PICKLING = 4 << 30

# Time to let PyArrow close threads gracefully
SLEEP_TIME_ON_THREADS_SHUTDOWN = 5
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "5.1.0"

from .arrow_dataset import Column, Dataset
from .arrow_reader import ReadInstruction
from .builder import ArrowBasedBuilder, BuilderConfig, DatasetBuilder, GeneratorBasedBuilder
from .combine import concatenate_datasets, interleave_datasets
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download import *
from .features import *
from .fingerprint import disable_caching, enable_caching, is_caching_enabled
from .info import DatasetInfo
from .inspect import (
    get_dataset_config_info,
    get_dataset_config_names,
    get_dataset_default_config_name,
    get_dataset_infos,
    get_dataset_split_names,
)
from .iterable_dataset import IterableColumn, IterableDataset
from .load import load_dataset, load_dataset_builder, load_from_disk
from .splits import (
    NamedSplit,
    NamedSplitAll,
    Split,
    SplitBase,
    SplitDict,
    SplitGenerator,
    SplitInfo,
    SubSplitInfo,
    percent,
)
from .utils import *
from .utils import logging
<|endoftext|>from typing import Literal, TypeVar

from .arrow_dataset import Dataset, _split_by_node_map_style_dataset
from .iterable_dataset import IterableDataset, _split_by_node_iterable_dataset


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


def split_dataset_by_node(
    dataset: DatasetType, rank: int, world_size: int, strategy: Literal["auto", "shards", "examples"] = "auto"
) -> DatasetType:
    """
    Split a dataset for the node at rank `rank` in a pool of nodes of size `world_size`.

    For map-style datasets:

    Each node is assigned a chunk of data, e.g. rank 0 is given the first chunk of the dataset.
    To maximize data loading throughput, chunks are made of contiguous data on disk if possible.

    For iterable datasets:

    The splitting `strategy` can be `"auto"`, `"shards"`, or `"examples"`:

    * `"shards"`: shards are evenly assigned across the nodes
    * `"examples"`: each node keeps 1 example out of `world_size`, skipping the other examples
    * `"auto"` (default): uses `"shards"` if the dataset has a number of shards that is a factor of `world_size`
    (i.e. if `dataset.num_shards % world_size == 0`), which is the most optimized. Otherwise uses `"examples"`.

    > [!WARNING]
    > If you shuffle your iterable dataset in a distributed setup, make sure to set a fixed `seed` in [`IterableDataset.shuffle`]
    so the same shuffled list of shards is used on every node to know which shards the node should skip.

    Args:
        dataset ([`Dataset`] or [`IterableDataset`]):
            The dataset to split by node.
        rank (`int`):
            Rank of the current node.
        world_size (`int`):
            Total number of nodes.
        strategy (`str`, defaults to `"auto"`):
            How to split an iterable dataset. Must be one of `"auto"`, `"shards"`, or `"examples"`.
            Map-style datasets only support `"auto"`.

    Returns:
        [`Dataset`] or [`IterableDataset`]: The dataset to be used on the node at rank `rank`.
    """
    if strategy not in {"auto", "shards", "examples"}:
        raise ValueError(f"Invalid strategy {strategy!r}. Expected one of 'auto', 'shards', or 'examples'.")
    if isinstance(dataset, Dataset):
        if strategy != "auto":
            raise ValueError(f"Map-style datasets only support strategy='auto', but got strategy={strategy!r}.")
        return _split_by_node_map_style_dataset(dataset, rank=rank, world_size=world_size)
    else:
        return _split_by_node_iterable_dataset(dataset, rank=rank, world_size=world_size, strategy=strategy)
<|endoftext|>from typing import Optional, TypeVar

from .arrow_dataset import Dataset, _concatenate_map_style_datasets, _interleave_map_style_datasets
from .dataset_dict import DatasetDict, IterableDatasetDict
from .info import DatasetInfo
from .iterable_dataset import IterableDataset, _concatenate_iterable_datasets, _interleave_iterable_datasets
from .splits import NamedSplit
from .utils import logging
from .utils.py_utils import Literal


logger = logging.get_logger(__name__)


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


def interleave_datasets(
    datasets: list[DatasetType],
    probabilities: Optional[list[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = "first_exhausted",
) -> DatasetType:
    """
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.

    You can use this function on a list of [`Dataset`] objects, or on a list of [`IterableDataset`] objects.

        - If `probabilities` is `None` (default) the new dataset is constructed by cycling between each source to get the examples.
        - If `probabilities` is not `None`, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    The resulting dataset ends when one of the source datasets runs out of examples except when `oversampling` is `True`,
    in which case, the resulting dataset ends when all datasets have ran out of examples at least one time.

    Note for iterable datasets:

    * The resulting dataset's `num_shards` is the minimum of each dataset's `num_shards` to ensure good parallelism.
      If some of your datasets have a very low number of shards, you may use [`IterableDataset.reshard`].
    * In a distributed setup or in PyTorch DataLoader workers, the stopping strategy is applied per process.
      Therefore the "first_exhausted" strategy on an sharded iterable dataset can generate less samples in total (up to 1 missing sample per subdataset per worker).

    Args:
        datasets (`List[Dataset]` or `List[IterableDataset]`):
            List of datasets to interleave.
        probabilities (`List[float]`, *optional*, defaults to `None`):
            If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        seed (`int`, *optional*, defaults to `None`):
            The random seed used to choose a source for each example.
        info ([`DatasetInfo`], *optional*):
            Dataset information, like description, citation, etc.
            <Added version="2.4.0"/>
        split ([`NamedSplit`], *optional*):
            Name of the dataset split.
            <Added version="2.4.0"/>
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Three strategies are proposed right now, `first_exhausted`, `all_exhausted` and `all_exhausted_without_replacement`.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            When strategy is `all_exhausted_without_replacement` we make sure that each sample in each dataset is sampled only once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have `max_length_datasets*nb_dataset` samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
    Returns:
        [`Dataset`] or [`IterableDataset`]: Return type depends on the input `datasets`
        parameter. `Dataset` if the input is a list of `Dataset`, `IterableDataset` if the input is a list of
        `IterableDataset`.

    Example:

        For regular datasets (map-style):

        ```python
        >>> from datasets import Dataset, interleave_datasets
        >>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
        >>> d2 = Dataset.from_dict({"a": [10, 11, 12]})
        >>> d3 = Dataset.from_dict({"a": [20, 21, 22]})
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [10, 0, 11, 1, 2, 20, 12, 10, 0, 1, 2, 21, 0, 11, 1, 2, 0, 1, 12, 2, 10, 0, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42)
        >>> dataset["a"]
        [10, 0, 11, 1, 2]
        >>> dataset = interleave_datasets([d1, d2, d3])
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
        >>> d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
        >>> d3 = Dataset.from_dict({"a": [20, 21, 22, 23, 24]})
        >>> dataset = interleave_datasets([d1, d2, d3])
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22, 0, 13, 23, 1, 10, 24]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42)
        >>> dataset["a"]
        [10, 0, 11, 1, 2]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [10, 0, 11, 1, 2, 20, 12, 13, ..., 0, 1, 2, 0, 24]
        For datasets in streaming mode (iterable):

        >>> from datasets import interleave_datasets
        >>> d1 = load_dataset('allenai/c4', 'es', split='train', streaming=True)
        >>> d2 = load_dataset('allenai/c4', 'fr', split='train', streaming=True)
        >>> dataset = interleave_datasets([d1, d2])
        >>> iterator = iter(dataset)
        >>> next(iterator)
        {'text': 'Comprar Zapatillas para niña en chancla con goma por...'}
        >>> next(iterator)
        {'text': 'Le sacre de philippe ier, 23 mai 1059 - Compte Rendu...'
        ```
    """
    from .arrow_dataset import Dataset
    from .iterable_dataset import IterableDataset

    if not datasets:
        raise ValueError("Unable to interleave an empty list of datasets.")
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if stopping_strategy not in ["first_exhausted", "all_exhausted", "all_exhausted_without_replacement"]:
        raise ValueError(f"{stopping_strategy} is not supported. Please enter a valid stopping_strategy.")
    if dataset_type is Dataset:
        return _interleave_map_style_datasets(
            datasets, probabilities, seed, info=info, split=split, stopping_strategy=stopping_strategy
        )
    else:
        return _interleave_iterable_datasets(
            datasets,
            probabilities,
            seed,
            info=info,
            split=split,
            stopping_strategy=stopping_strategy,
        )


def concatenate_datasets(
    dsets: list[DatasetType],
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    axis: int = 0,
) -> DatasetType:
    """
    Concatenate several datasets (sources) into a single dataset.

    Use axis=0 to concatenate vertically (default), or axis=1 to concatenate horizontally.

    Note for iterable datasets:

    * if axis=0, the resulting dataset's `num_shards` is the sum of each dataset's `num_shards`.
    * if axis=1, the resulting dataset has one (1) shard to not misalign data.

    Args:
        dsets (`List[datasets.Dataset]` or `List[datasets.IterableDataset]`):
            List of Datasets to concatenate.
        info (`DatasetInfo`, *optional*):
            Dataset information, like description, citation, etc.
        split (`NamedSplit`, *optional*):
            Name of the dataset split.
        axis (`{0, 1}`, defaults to `0`):
            Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
            (horizontally).

            <Added version="1.6.0"/>

    Example:

    ```py
    >>> ds3 = concatenate_datasets([ds1, ds2])
    ```
    """

    if not dsets:
        raise ValueError("Unable to concatenate an empty list of datasets.")
    for i, dataset in enumerate(dsets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if dataset_type is Dataset:
        return _concatenate_map_style_datasets(dsets, info=info, split=split, axis=axis)
    else:
        return _concatenate_iterable_datasets(dsets, info=info, split=split, axis=axis)
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Splits related API."""

import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Optional, Union

from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict


@dataclass
class SplitInfo:
    name: str = dataclasses.field(default="", metadata={"include_in_asdict_even_if_is_default": True})
    num_bytes: int = dataclasses.field(default=0, metadata={"include_in_asdict_even_if_is_default": True})
    num_examples: int = dataclasses.field(default=0, metadata={"include_in_asdict_even_if_is_default": True})
    shard_lengths: Optional[list[int]] = None
    original_shard_lengths: Optional[list[int]] = None

    # Deprecated
    # For backward compatibility, this field needs to always be included in files like
    # dataset_infos.json and dataset_info.json files
    # To do so, we always include it in the output of datasets.utils.py_utils.asdict(split_info)
    dataset_name: Optional[str] = dataclasses.field(
        default=None, metadata={"include_in_asdict_even_if_is_default": True}
    )

    @property
    def file_instructions(self):
        """Returns the list of dict(filename, take, skip)."""
        # `self.dataset_name` is assigned in `SplitDict.add()`.
        instructions = make_file_instructions(
            name=self.dataset_name,
            split_infos=[self],
            instruction=str(self.name),
        )
        return instructions.file_instructions

    def __repr__(self):
        return (
            self.__class__.__qualname__
            + "("
            + ", ".join(
                [f"{f.name}={repr(getattr(self, f.name))}" for f in dataclasses.fields(self) if getattr(self, f.name)]
            )
            + ")"
        )


@dataclass
class SubSplitInfo:
    """Wrapper around a sub split info.
    This class exposes info on the subsplit:
    ```
    ds, info = datasets.load_dataset(..., split='train[75%:]', with_info=True)
    info.splits['train[75%:]'].num_examples
    ```
    """

    instructions: FileInstructions

    @property
    def num_examples(self):
        """Returns the number of examples in the subsplit."""
        return self.instructions.num_examples

    @property
    def file_instructions(self):
        """Returns the list of dict(filename, take, skip)."""
        return self.instructions.file_instructions


class SplitBase(metaclass=abc.ABCMeta):
    # pylint: disable=line-too-long
    """Abstract base class for Split compositionality.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.

    There are three parts to the composition:
        1) The splits are composed (defined, merged, split,...) together before
             calling the `.as_dataset()` function. This is done with the `__add__`,
             `__getitem__`, which return a tree of `SplitBase` (whose leaf
             are the `NamedSplit` objects)

        ```
        split = datasets.Split.TRAIN + datasets.Split.TEST.subsplit(datasets.percent[:50])
        ```

        2) The `SplitBase` is forwarded to the `.as_dataset()` function
             to be resolved into actual read instruction. This is done by the
             `.get_read_instruction()` method which takes the real dataset splits
             (name, number of shards,...) and parse the tree to return a
             `SplitReadInstruction()` object

        ```
        read_instruction = split.get_read_instruction(self.info.splits)
        ```

        3) The `SplitReadInstruction` is then used in the `tf.data.Dataset` pipeline
             to define which files to read and how to skip examples within file.

    """

    # pylint: enable=line-too-long

    @abc.abstractmethod
    def get_read_instruction(self, split_dict):
        """Parse the descriptor tree and compile all read instructions together.

        Args:
            split_dict: `dict`, The `dict[split_name, SplitInfo]` of the dataset

        Returns:
            split_read_instruction: `SplitReadInstruction`
        """
        raise NotImplementedError("Abstract method")

    def __eq__(self, other):
        """Equality: datasets.Split.TRAIN == 'train'."""
        if isinstance(other, (NamedSplit, str)):
            return False
        raise NotImplementedError("Equality is not implemented between merged/sub splits.")

    def __ne__(self, other):
        """InEquality: datasets.Split.TRAIN != 'test'."""
        return not self.__eq__(other)

    def __add__(self, other):
        """Merging: datasets.Split.TRAIN + datasets.Split.TEST."""
        return _SplitMerged(self, other)

    def subsplit(self, arg=None, k=None, percent=None, weighted=None):  # pylint: disable=redefined-outer-name
        """Divides this split into subsplits.

        There are 3 ways to define subsplits, which correspond to the 3
        arguments `k` (get `k` even subsplits), `percent` (get a slice of the
        dataset with `datasets.percent`), and `weighted` (get subsplits with proportions
        specified by `weighted`).

        Example::

        ```
        # 50% train, 50% test
        train, test = split.subsplit(k=2)
        # 50% train, 25% test, 25% validation
        train, test, validation = split.subsplit(weighted=[2, 1, 1])
        # Extract last 20%
        subsplit = split.subsplit(datasets.percent[-20:])
        ```

        Warning: k and weighted will be converted into percent which mean that
        values below the percent will be rounded up or down. The final split may be
        bigger to deal with remainders. For instance:

        ```
        train, test, valid = split.subsplit(k=3)  # 33%, 33%, 34%
        s1, s2, s3, s4 = split.subsplit(weighted=[2, 2, 1, 1])  # 33%, 33%, 16%, 18%
        ```

        Args:
            arg: If no kwargs are given, `arg` will be interpreted as one of
                `k`, `percent`, or `weighted` depending on the type.
                For example:
                ```
                split.subsplit(10)  # Equivalent to split.subsplit(k=10)
                split.subsplit(datasets.percent[:-20])  # percent=datasets.percent[:-20]
                split.subsplit([1, 1, 2])  # weighted=[1, 1, 2]
                ```
            k: `int` If set, subdivide the split into `k` equal parts.
            percent: `datasets.percent slice`, return a single subsplit corresponding to
                a slice of the original split. For example:
                `split.subsplit(datasets.percent[-20:])  # Last 20% of the dataset`.
            weighted: `list[int]`, return a list of subsplits whose proportions match
                the normalized sum of the list. For example:
                `split.subsplit(weighted=[1, 1, 2])  # 25%, 25%, 50%`.

        Returns:
            A subsplit or list of subsplits extracted from this split object.
        """
        # Note that the percent kwargs redefine the outer name datasets.percent. This
        # is done for consistency (.subsplit(percent=datasets.percent[:40]))
        if sum(bool(x) for x in (arg, k, percent, weighted)) != 1:
            raise ValueError("Only one argument of subsplit should be set.")

        # Auto deduce k
        if isinstance(arg, int):
            k = arg
        elif isinstance(arg, slice):
            percent = arg
        elif isinstance(arg, list):
            weighted = arg

        if not (k or percent or weighted):
            raise ValueError(
                f"Invalid split argument {arg}. Only list, slice and int supported. "
                "One of k, weighted or percent should be set to a non empty value."
            )

        def assert_slices_coverage(slices):
            # Ensure that the expended slices cover all percents.
            assert sum((list(range(*s.indices(100))) for s in slices), []) == list(range(100))

        if k:
            if not 0 < k <= 100:
                raise ValueError(f"Subsplit k should be between 0 and 100, got {k}")
            shift = 100 // k
            slices = [slice(i * shift, (i + 1) * shift) for i in range(k)]
            # Round up last element to ensure all elements are taken
            slices[-1] = slice(slices[-1].start, 100)
            # Internal check to ensure full coverage
            assert_slices_coverage(slices)
            return tuple(_SubSplit(self, s) for s in slices)
        elif percent:
            return _SubSplit(self, percent)
        elif weighted:
            # Normalize the weighted sum
            total = sum(weighted)
            weighted = [100 * x // total for x in weighted]
            # Create the slice for each of the elements
            start = 0
            stop = 0
            slices = []
            for v in weighted:
                stop += v
                slices.append(slice(start, stop))
                start = stop
            # Round up last element to ensure all elements are taken
            slices[-1] = slice(slices[-1].start, 100)
            # Internal check to ensure full coverage
            assert_slices_coverage(slices)
            return tuple(_SubSplit(self, s) for s in slices)
        else:
            # Should not be possible
            raise ValueError("Could not determine the split")


# 2 requirements:
# 1. datasets.percent be sliceable
# 2. datasets.percent be documented
#
# Instances are not documented, so we want datasets.percent to be a class, but to
# have it be sliceable, we need this metaclass.
class PercentSliceMeta(type):
    def __getitem__(cls, slice_value):
        if not isinstance(slice_value, slice):
            raise ValueError(f"datasets.percent should only be called with slice, not {slice_value}")
        return slice_value


class PercentSlice(metaclass=PercentSliceMeta):
    # pylint: disable=line-too-long
    """Syntactic sugar for defining slice subsplits: `datasets.percent[75:-5]`.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.
    """

    # pylint: enable=line-too-long
    pass


percent = PercentSlice  # pylint: disable=invalid-name


class _SplitMerged(SplitBase):
    """Represent two split descriptors merged together."""

    def __init__(self, split1, split2):
        self._split1 = split1
        self._split2 = split2

    def get_read_instruction(self, split_dict):
        read_instruction1 = self._split1.get_read_instruction(split_dict)
        read_instruction2 = self._split2.get_read_instruction(split_dict)
        return read_instruction1 + read_instruction2

    def __repr__(self):
        return f"({repr(self._split1)} + {repr(self._split2)})"


class _SubSplit(SplitBase):
    """Represent a sub split of a split descriptor."""

    def __init__(self, split, slice_value):
        self._split = split
        self._slice_value = slice_value

    def get_read_instruction(self, split_dict):
        return self._split.get_read_instruction(split_dict)[self._slice_value]

    def __repr__(self):
        slice_str = "{start}:{stop}"
        if self._slice_value.step is not None:
            slice_str += ":{step}"
        slice_str = slice_str.format(
            start="" if self._slice_value.start is None else self._slice_value.start,
            stop="" if self._slice_value.stop is None else self._slice_value.stop,
            step=self._slice_value.step,
        )
        return f"{repr(self._split)}(datasets.percent[{slice_str}])"


class NamedSplit(SplitBase):
    """Descriptor corresponding to a named split (train, test, ...).

    Example:
        Each descriptor can be composed with other using addition or slice:

            ```py
            split = datasets.Split.TRAIN.subsplit(datasets.percent[0:25]) + datasets.Split.TEST
            ```

        The resulting split will correspond to 25% of the train split merged with
        100% of the test split.

        A split cannot be added twice, so the following will fail:

            ```py
            split = (
                    datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
                    datasets.Split.TRAIN.subsplit(datasets.percent[75:])
            )  # Error
            split = datasets.Split.TEST + datasets.Split.ALL  # Error
            ```

        The slices can be applied only one time. So the following are valid:

            ```py
            split = (
                    datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
                    datasets.Split.TEST.subsplit(datasets.percent[:50])
            )
            split = (datasets.Split.TRAIN + datasets.Split.TEST).subsplit(datasets.percent[:50])
            ```

        But this is not valid:

            ```py
            train = datasets.Split.TRAIN
            test = datasets.Split.TEST
            split = train.subsplit(datasets.percent[:25]).subsplit(datasets.percent[:25])
            split = (train.subsplit(datasets.percent[:25]) + test).subsplit(datasets.percent[:50])
            ```
    """

    def __init__(self, name: str):
        self._name = name
        split_names_from_instruction = [split_instruction.split("[")[0] for split_instruction in name.split("+")]
        for split_name in split_names_from_instruction:
            if not re.match(_split_re, split_name):
                raise ValueError(f"Split name should match '{_split_re}' but got '{split_name}'.")

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"NamedSplit({self._name!r})"

    def __eq__(self, other):
        """Equality: datasets.Split.TRAIN == 'train'."""
        if isinstance(other, NamedSplit):
            return self._name == other._name  # pylint: disable=protected-access
        elif isinstance(other, SplitBase):
            return False
        elif isinstance(other, str):  # Other should be string
            return self._name == other
        else:
            return False

    def __lt__(self, other):
        return self._name < other._name  # pylint: disable=protected-access

    def __hash__(self):
        return hash(self._name)

    def get_read_instruction(self, split_dict):
        return SplitReadInstruction(split_dict[self._name])


class NamedSplitAll(NamedSplit):
    """Split corresponding to the union of all defined dataset splits."""

    def __init__(self):
        super().__init__("all")

    def __repr__(self):
        return "NamedSplitAll()"

    def get_read_instruction(self, split_dict):
        # Merge all dataset splits together
        read_instructions = [SplitReadInstruction(s) for s in split_dict.values()]
        return sum(read_instructions, SplitReadInstruction())


class Split:
    # pylint: disable=line-too-long
    """`Enum` for dataset splits.

    Datasets are typically split into different subsets to be used at various
    stages of training and evaluation.

    - `TRAIN`: the training data.
    - `VALIDATION`: the validation data. If present, this is typically used as
      evaluation data while iterating on a model (e.g. changing hyperparameters,
      model architecture, etc.).
    - `TEST`: the testing data. This is the data to report metrics on. Typically
      you do not want to use this during model iteration as you may overfit to it.
    - `ALL`: the union of all defined dataset splits.

    All splits, including compositions inherit from `datasets.SplitBase`.

    See the [guide](../load_hub#splits) on splits for more information.

    Example:

    ```py
    >>> datasets.SplitGenerator(
    ...     name=datasets.Split.TRAIN,
    ...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and extract(url)},
    ... ),
    ... datasets.SplitGenerator(
    ...     name=datasets.Split.VALIDATION,
    ...     gen_kwargs={"split_key": "validation", "files": dl_manager.download_and extract(url)},
    ... ),
    ... datasets.SplitGenerator(
    ...     name=datasets.Split.TEST,
    ...     gen_kwargs={"split_key": "test", "files": dl_manager.download_and extract(url)},
    ... )
    ```
    """

    # pylint: enable=line-too-long
    TRAIN = NamedSplit("train")
    TEST = NamedSplit("test")
    VALIDATION = NamedSplit("validation")
    ALL = NamedSplitAll()

    def __new__(cls, name):
        """Create a custom split with datasets.Split('custom_name')."""
        return NamedSplitAll() if name == "all" else NamedSplit(name)


# Similar to SplitInfo, but contain an additional slice info
SlicedSplitInfo = collections.namedtuple(
    "SlicedSplitInfo",
    [
        "split_info",
        "slice_value",
    ],
)  # noqa: E231


class SplitReadInstruction:
    """Object containing the reading instruction for the dataset.

    Similarly to `SplitDescriptor` nodes, this object can be composed with itself,
    but the resolution happens instantaneously, instead of keeping track of the
    tree, such as all instructions are compiled and flattened in a single
    SplitReadInstruction object containing the list of files and slice to use.

    Once resolved, the instructions can be accessed with:

    ```
    read_instructions.get_list_sliced_split_info()  # List of splits to use
    ```

    """

    def __init__(self, split_info=None):
        self._splits = NonMutableDict(error_msg="Overlap between splits. Split {key} has been added with itself.")

        if split_info:
            self.add(SlicedSplitInfo(split_info=split_info, slice_value=None))

    def add(self, sliced_split):
        """Add a SlicedSplitInfo the read instructions."""
        # TODO(epot): Check that the number of examples per shard % 100 == 0
        # Otherwise the slices value may be unbalanced and not exactly reflect the
        # requested slice.
        self._splits[sliced_split.split_info.name] = sliced_split

    def __add__(self, other):
        """Merging split together."""
        # Will raise error if a split has already be added (NonMutableDict)
        # TODO(epot): If a split is already added but there is no overlap between
        # the slices, should merge the slices (ex: [:10] + [80:])
        split_instruction = SplitReadInstruction()
        split_instruction._splits.update(self._splits)  # pylint: disable=protected-access
        split_instruction._splits.update(other._splits)  # pylint: disable=protected-access
        return split_instruction

    def __getitem__(self, slice_value):
        """Sub-splits."""
        # Will raise an error if a split has already been sliced
        split_instruction = SplitReadInstruction()
        for v in self._splits.values():
            if v.slice_value is not None:
                raise ValueError(f"Trying to slice Split {v.split_info.name} which has already been sliced")
            v = v._asdict()
            v["slice_value"] = slice_value
            split_instruction.add(SlicedSplitInfo(**v))
        return split_instruction

    def get_list_sliced_split_info(self):
        return list(self._splits.values())


class SplitDict(dict[str, SplitInfo]):
    """Split info object."""

    def __init__(self, *args, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def __getitem__(self, key: Union[SplitBase, str]):
        # 1st case: The key exists: `info.splits['train']`
        if str(key) in self:
            return super().__getitem__(str(key))
        # 2nd case: Uses instructions: `info.splits['train[50%]']`
        else:
            instructions = make_file_instructions(
                name=self.dataset_name,
                split_infos=self.values(),
                instruction=key,
            )
            return SubSplitInfo(instructions)

    def __setitem__(self, key: Union[SplitBase, str], value: SplitInfo):
        if key != value.name:
            raise ValueError(f"Cannot add elem. (key mismatch: '{key}' != '{value.name}')")
        super().__setitem__(key, value)

    def add(self, split_info: SplitInfo):
        """Add the split info."""
        if split_info.name in self:
            raise ValueError(f"Split {split_info.name} already present")
        split_info.dataset_name = self.dataset_name
        super().__setitem__(split_info.name, split_info)

    @property
    def total_num_examples(self):
        """Return the total number of examples."""
        return sum(s.num_examples for s in self.values())

    @classmethod
    def from_split_dict(cls, split_infos: Union[list, dict], dataset_name: Optional[str] = None):
        """Returns a new SplitDict initialized from a Dict or List of `split_infos`."""
        if isinstance(split_infos, dict):
            split_infos = list(split_infos.values())

        if dataset_name is None:
            dataset_name = split_infos[0].get("dataset_name") if split_infos else None

        split_dict = cls(dataset_name=dataset_name)

        for split_info in split_infos:
            if isinstance(split_info, dict):
                split_info = SplitInfo(**split_info)
            split_dict.add(split_info)

        return split_dict

    def to_split_dict(self):
        """Returns a list of SplitInfo protos that we have."""
        out = []
        for split_name, split_info in self.items():
            split_info = copy.deepcopy(split_info)
            split_info.name = split_name
            out.append(split_info)
        return out

    def copy(self):
        return SplitDict.from_split_dict(self.to_split_dict(), self.dataset_name)

    def _to_yaml_list(self) -> list:
        out = [asdict(s) for s in self.to_split_dict()]
        # we don't need the shard lengths in YAML
        for split_info_dict in out:
            split_info_dict.pop("shard_lengths", None)
            split_info_dict.pop("original_shard_lengths", None)
        # we don't need the dataset_name attribute that is deprecated
        for split_info_dict in out:
            split_info_dict.pop("dataset_name", None)
        return out

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "SplitDict":
        if not all(
            isinstance(split_info, dict)
            and "name" in split_info
            and "num_examples" in split_info
            and "num_bytes" in split_info
            for split_info in yaml_data
        ):
            raise ValueError(
                "The `splits` field in YAML is malformed or missing the name/num_examples/num_bytes fields"
            )
        return cls.from_split_dict(yaml_data)


@dataclass
class SplitGenerator:
    """Defines the split information for the generator.

    This should be used as returned value of
    `GeneratorBasedBuilder._split_generators`.
    See `GeneratorBasedBuilder._split_generators` for more info and example
    of usage.

    Args:
        name (`str`):
            Name of the `Split` for which the generator will
            create the examples.
        **gen_kwargs (additional keyword arguments):
            Keyword arguments to forward to the `DatasetBuilder._generate_examples` method
            of the builder.

    Example:

    ```py
    >>> datasets.SplitGenerator(
    ...     name=datasets.Split.TRAIN,
    ...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and_extract(url)},
    ... )
    ```
    """

    name: str
    gen_kwargs: dict = dataclasses.field(default_factory=dict)
    split_info: SplitInfo = dataclasses.field(init=False)

    def __post_init__(self):
        self.name = str(self.name)  # Make sure we convert NamedSplits in strings
        NamedSplit(self.name)  # check that it's a valid split name
        self.split_info = SplitInfo(name=self.name)
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utilities for file names."""

import itertools
import os
import re


_uppercase_uppercase_re = re.compile(r"([A-Z]+)([A-Z][a-z])")
_lowercase_uppercase_re = re.compile(r"([a-z\d])([A-Z])")

_single_underscore_re = re.compile(r"(?<!_)_(?!_)")
_multiple_underscores_re = re.compile(r"(_{2,})")

_split_re = r"^\w+(\.\w+)*$"

INVALID_WINDOWS_CHARACTERS_IN_PATH = r"<>:/\|?*"


def camelcase_to_snakecase(name):
    """Convert camel-case string to snake-case."""
    name = _uppercase_uppercase_re.sub(r"\1_\2", name)
    name = _lowercase_uppercase_re.sub(r"\1_\2", name)
    return name.lower()


def snakecase_to_camelcase(name):
    """Convert snake-case string to camel-case string."""
    name = _single_underscore_re.split(name)
    name = [_multiple_underscores_re.split(n) for n in name]
    return "".join(n.capitalize() for n in itertools.chain.from_iterable(name) if n != "")


def filename_prefix_for_name(name):
    if os.path.basename(name) != name:
        raise ValueError(f"Should be a dataset name, not a path: {name}")
    return camelcase_to_snakecase(name)


def filename_prefix_for_split(name, split):
    if os.path.basename(name) != name:
        raise ValueError(f"Should be a dataset name, not a path: {name}")
    if not re.match(_split_re, split):
        raise ValueError(f"Split name should match '{_split_re}'' but got '{split}'.")
    return f"{filename_prefix_for_name(name)}-{split}"


def filepattern_for_dataset_split(dataset_name, split, data_dir, filetype_suffix=None):
    prefix = filename_prefix_for_split(dataset_name, split)
    if filetype_suffix:
        prefix += f".{filetype_suffix}"
    filepath = os.path.join(data_dir, prefix)
    return f"{filepath}*"


def filenames_for_dataset_split(path, dataset_name, split, filetype_suffix=None, shard_lengths=None):
    prefix = filename_prefix_for_split(dataset_name, split)
    prefix = os.path.join(path, prefix)

    if shard_lengths and len(shard_lengths) > 1:
        num_shards = len(shard_lengths)
        filenames = [f"{prefix}-{shard_id:05d}-of-{num_shards:05d}" for shard_id in range(num_shards)]
        if filetype_suffix:
            filenames = [filename + f".{filetype_suffix}" for filename in filenames]
        return filenames
    else:
        filename = prefix
        if filetype_suffix:
            filename += f".{filetype_suffix}"
        return [filename]
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Access datasets."""

import glob
import importlib
import inspect
import json
import os
import posixpath
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import fsspec
import yaml
from fsspec.core import url_to_fs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from huggingface_hub.errors import BucketNotFoundError
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    get_session,
    httpx,
)

from . import __version__, config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
    DataFilesDict,
    DataFilesList,
    DataFilesPatternsDict,
    EmptyDatasetError,
    get_data_patterns,
    sanitize_patterns,
)
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .features.features import _fix_for_backward_compatible_features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
    _ALL_ALLOWED_EXTENSIONS,
    _ALL_METADATA_FILENAMES,
    _EXTENSION_TO_MODULE,
    _MODULE_TO_EXTENSIONS,
    _MODULE_TO_METADATA_EXTENSIONS,
    _MODULE_TO_METADATA_FILE_NAMES,
    _PACKAGED_DATASETS_MODULES,
)
from .splits import Split
from .utils import _dataset_viewer
from .utils.file_utils import (
    _raise_if_offline_mode_is_enabled,
    cached_path,
    get_datasets_user_agent,
    is_relative_path,
    relative_to_absolute_path,
)
from .utils.hub import hf_dataset_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.typing import PathLike
from .utils.version import Version


logger = get_logger(__name__)


class _InitializeConfiguredDatasetBuilder:
    """
    From https://stackoverflow.com/questions/4647566/pickle-a-dynamically-parameterized-sub-class
    See also ConfiguredDatasetBuilder.__reduce__
    When called with the param value as the only argument, returns an
    un-initialized instance of the parameterized class. Subsequent __setstate__
    will be called by pickle.
    """

    def __call__(self, builder_cls, metadata_configs, default_config_name, name):
        # make a simple object which has no complex __init__ (this one will do)
        obj = _InitializeConfiguredDatasetBuilder()
        obj.__class__ = configure_builder_class(
            builder_cls, metadata_configs, default_config_name=default_config_name, dataset_name=name
        )
        return obj


def configure_builder_class(
    builder_cls: type[DatasetBuilder],
    builder_configs: list[BuilderConfig],
    default_config_name: Optional[str],
    dataset_name: str,
) -> type[DatasetBuilder]:
    """
    Dynamically create a builder class with custom builder configs parsed from README.md file,
    i.e. set BUILDER_CONFIGS class variable of a builder class to custom configs list.
    """

    class ConfiguredDatasetBuilder(builder_cls):
        BUILDER_CONFIGS = builder_configs
        DEFAULT_CONFIG_NAME = default_config_name

        __module__ = builder_cls.__module__  # so that the actual packaged builder can be imported

        def __reduce__(self):  # to make dynamically created class pickable, see _InitializeParameterizedDatasetBuilder
            parent_builder_cls = self.__class__.__mro__[1]
            return (
                _InitializeConfiguredDatasetBuilder(),
                (
                    parent_builder_cls,
                    self.BUILDER_CONFIGS,
                    self.DEFAULT_CONFIG_NAME,
                    self.dataset_name,
                ),
                self.__dict__.copy(),
            )

    ConfiguredDatasetBuilder.__name__ = (
        f"{builder_cls.__name__.lower().capitalize()}{snakecase_to_camelcase(dataset_name)}"
    )
    ConfiguredDatasetBuilder.__qualname__ = (
        f"{builder_cls.__name__.lower().capitalize()}{snakecase_to_camelcase(dataset_name)}"
    )

    return ConfiguredDatasetBuilder


def import_main_class(module_path) -> Optional[type[DatasetBuilder]]:
    """Import a module at module_path and return its main class: a DatasetBuilder"""
    module = importlib.import_module(module_path)
    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, DatasetBuilder):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            obj_module = inspect.getmodule(obj)
            if obj_module is not None and module == obj_module:
                break

    return module_main_cls


def get_dataset_builder_class(
    dataset_module: "DatasetModule", dataset_name: Optional[str] = None
) -> type[DatasetBuilder]:
    builder_cls = import_main_class(dataset_module.module_path)
    if dataset_module.builder_configs_parameters.builder_configs:
        dataset_name = dataset_name or dataset_module.builder_kwargs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("dataset_name should be specified but got None")
        builder_cls = configure_builder_class(
            builder_cls,
            builder_configs=dataset_module.builder_configs_parameters.builder_configs,
            default_config_name=dataset_module.builder_configs_parameters.default_config_name,
            dataset_name=dataset_name,
        )
    return builder_cls


def increase_load_count(name: str):
    """Update the download count of a dataset."""
    if not config.HF_HUB_OFFLINE and config.HF_UPDATE_DOWNLOAD_COUNTS:
        try:
            get_session().head(
                "/".join((config.S3_DATASETS_BUCKET_PREFIX, name, name + ".py")),
                headers={"User-Agent": get_datasets_user_agent()},
                timeout=3,
            )
        except Exception:
            pass


def infer_module_for_data_files_list(
    data_files_list: DataFilesList, download_config: Optional[DownloadConfig] = None
) -> tuple[Optional[str], dict]:
    """Infer module (and builder kwargs) from list of data files.

    It picks the module based on the most common file extension.
    In case of a draw ".parquet" is the favorite, and then alphabetical order.

    Args:
        data_files_list (DataFilesList): List of data files.
        download_config (bool or str, optional): Mainly use `token` or `storage_options` to support different platforms and auth types.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - dict of builder kwargs
    """
    extensions_counter = Counter(
        ("." + suffix.lower(), xbasename(filepath) in _ALL_METADATA_FILENAMES)
        for filepath in data_files_list
        for suffix in xbasename(filepath).split(".")[1:]
    )
    if extensions_counter:

        def sort_key(ext_count: tuple[tuple[str, bool], int]) -> tuple[int, bool]:
            """Sort by count and set ".parquet" as the favorite in case of a draw, and ignore metadata files"""
            (ext, is_metadata), count = ext_count
            return (
                not is_metadata,
                count,
                ext == ".parquet",
                ext == ".lance",
                ext == ".vortex",
                ext == ".arrow",
                ext == ".jsonl",
                ext == ".json",
                ext == ".csv",
                ext,
            )

        for (ext, _), _ in sorted(extensions_counter.items(), key=sort_key, reverse=True):
            if ext in _EXTENSION_TO_MODULE:
                return _EXTENSION_TO_MODULE[ext]
            elif ext == ".zip":
                return infer_module_for_data_files_list_in_archives(data_files_list, download_config=download_config)
    return None, {}


def infer_module_for_data_files_list_in_archives(
    data_files_list: DataFilesList, download_config: Optional[DownloadConfig] = None
) -> tuple[Optional[str], dict]:
    """Infer module (and builder kwargs) from list of archive data files.

    Args:
        data_files_list (DataFilesList): List of data files.
        download_config (bool or str, optional): Mainly use `token` or `storage_options` to support different platforms and auth types.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - dict of builder kwargs
    """
    archived_files = []
    archive_files_counter = 0
    for filepath in data_files_list:
        if str(filepath).endswith(".zip"):
            archive_files_counter += 1
            if archive_files_counter > config.ARCHIVES_MAX_NUMBER_FOR_MODULE_INFERENCE:
                break
            extracted = xjoin(StreamingDownloadManager().extract(filepath), "**")
            archived_files += [
                f.split("::")[0] for f in xglob(extracted, recursive=True, download_config=download_config)
            ]
    extensions_counter = Counter(
        "." + suffix.lower() for filepath in archived_files for suffix in xbasename(filepath).split(".")[1:]
    )
    if extensions_counter:
        most_common = extensions_counter.most_common(1)[0][0]
        if most_common in _EXTENSION_TO_MODULE:
            return _EXTENSION_TO_MODULE[most_common]
    return None, {}


def infer_module_for_data_files(
    data_files: DataFilesDict, path: Optional[str] = None, download_config: Optional[DownloadConfig] = None
) -> tuple[Optional[str], dict[str, Any]]:
    """Infer module (and builder kwargs) from data files. Raise if module names for different splits don't match.

    Args:
        data_files ([`DataFilesDict`]): Dict of list of data files.
        path (str, *optional*): Dataset name or path.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters to authenticate on the Hugging Face Hub for private remote files.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - builder kwargs
    """
    split_modules = {
        split: infer_module_for_data_files_list(data_files_list, download_config=download_config)
        for split, data_files_list in data_files.items()
    }
    module_name, default_builder_kwargs = next(iter(split_modules.values()))
    if any((module_name, default_builder_kwargs) != split_module for split_module in split_modules.values()):
        raise ValueError(f"Couldn't infer the same data file format for all splits. Got {split_modules}")
    if not module_name:
        raise DataFilesNotFoundError("No (supported) data files found" + (f" in {path}" if path else ""))
    return module_name, default_builder_kwargs


def create_builder_configs_from_metadata_configs(
    module_path: str,
    metadata_configs: MetadataConfigs,
    base_path: Optional[str] = None,
    default_builder_kwargs: dict[str, Any] = None,
    download_config: Optional[DownloadConfig] = None,
) -> tuple[list[BuilderConfig], str]:
    builder_cls = import_main_class(module_path)
    builder_config_cls = builder_cls.BUILDER_CONFIG_CLASS
    default_config_name = metadata_configs.get_default_config_name()
    builder_configs = []
    default_builder_kwargs = {} if default_builder_kwargs is None else default_builder_kwargs

    base_path = base_path if base_path is not None else ""
    for config_name, config_params in metadata_configs.items():
        config_data_files = config_params.get("data_files")
        config_data_dir = config_params.get("data_dir")
        config_base_path = xjoin(base_path, config_data_dir) if config_data_dir else base_path
        try:
            config_patterns = (
                sanitize_patterns(config_data_files)
                if config_data_files is not None
                else get_data_patterns(config_base_path, download_config=download_config)
            )
            config_data_files_dict = DataFilesPatternsDict.from_patterns(
                config_patterns,
                allowed_extensions=_ALL_ALLOWED_EXTENSIONS,
            )
        except EmptyDatasetError as e:
            raise EmptyDatasetError(
                f"Dataset at '{base_path}' doesn't contain data files matching the patterns for config '{config_name}',"
                f" check `data_files` and `data_fir` parameters in the `configs` YAML field in README.md. "
            ) from e
        ignored_params = [
            param for param in config_params if not hasattr(builder_config_cls, param) and param != "default"
        ]
        if ignored_params:
            logger.warning(
                f"Some datasets params were ignored: {ignored_params}. "
                "Make sure to use only valid params for the dataset builder and to have "
                "a up-to-date version of the `datasets` library."
            )
        builder_configs.append(
            builder_config_cls(
                name=config_name,
                data_files=config_data_files_dict,
                data_dir=config_data_dir,
                **{
                    param: value
                    for param, value in {**default_builder_kwargs, **config_params}.items()
                    if hasattr(builder_config_cls, param) and param not in ("default", "data_files", "data_dir")
                },
            )
        )
    return builder_configs, default_config_name


@dataclass
class BuilderConfigsParameters:
    """Dataclass containing objects related to creation of builder configurations from yaml's metadata content.

    Attributes:
        metadata_configs (`MetadataConfigs`, *optional*):
            Configs parsed from yaml's metadata.
        builder_configs (`list[BuilderConfig]`, *optional*):
            List of BuilderConfig objects created from metadata_configs above.
        default_config_name (`str`):
            Name of default config taken from yaml's metadata.
    """

    metadata_configs: Optional[MetadataConfigs] = None
    builder_configs: Optional[list[BuilderConfig]] = None
    default_config_name: Optional[str] = None


@dataclass
class DatasetModule:
    module_path: str
    hash: str
    builder_kwargs: dict
    builder_configs_parameters: BuilderConfigsParameters = field(default_factory=BuilderConfigsParameters)
    dataset_infos: Optional[DatasetInfosDict] = None


class _DatasetModuleFactory:
    def get_module(self) -> DatasetModule:
        raise NotImplementedError


class LocalDatasetModuleFactory(_DatasetModuleFactory):
    """Get the module of a dataset loaded from the user's data files. The dataset builder module to use is inferred
    from the data files extensions."""

    def __init__(
        self,
        path: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        if data_dir and os.path.isabs(data_dir):
            raise ValueError(f"`data_dir` must be relative to a dataset directory's root: {path}")

        self.path = Path(path).as_posix()
        self.name = Path(path).stem
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_mode = download_mode

    def get_module(self) -> DatasetModule:
        readme_path = os.path.join(self.path, config.REPOCARD_FILENAME)
        standalone_yaml_path = os.path.join(self.path, config.REPOYAML_FILENAME)
        dataset_card_data = DatasetCard.load(readme_path).data if os.path.isfile(readme_path) else DatasetCardData()
        if os.path.exists(standalone_yaml_path):
            with open(standalone_yaml_path, encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        # we need a set of data files to find which dataset builder to use
        # because we need to infer module name by files extensions
        base_path = Path(self.path, self.data_dir or "").expanduser().resolve().as_posix()
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path)
        data_files = DataFilesDict.from_patterns(
            patterns,
            base_path=base_path,
            allowed_extensions=_ALL_ALLOWED_EXTENSIONS,
        )
        module_name, default_builder_kwargs = infer_module_for_data_files(
            data_files=data_files,
            path=self.path,
        )
        data_files = data_files.filter(
            extensions=_MODULE_TO_EXTENSIONS[module_name] + _MODULE_TO_METADATA_EXTENSIONS[module_name],
            file_names=_MODULE_TO_METADATA_FILE_NAMES[module_name],
        )
        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                default_builder_kwargs=default_builder_kwargs,
            )
        else:
            builder_configs: list[BuilderConfig] = [
                import_main_class(module_path).BUILDER_CONFIG_CLASS(
                    data_files=data_files,
                    **default_builder_kwargs,
                )
            ]
            default_config_name = None
        builder_kwargs = {
            "base_path": self.path,
            "dataset_name": camelcase_to_snakecase(Path(self.path).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        # this file is deprecated and was created automatically in old versions of push_to_hub
        if os.path.isfile(os.path.join(self.path, config.DATASETDICT_INFOS_FILENAME)):
            with open(os.path.join(self.path, config.DATASETDICT_INFOS_FILENAME), encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
                if len(legacy_dataset_infos) == 1:
                    # old config e.g. named "username--dataset_name"
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        hash = Hasher.hash({"dataset_infos": dataset_infos, "builder_configs": builder_configs})
        return DatasetModule(
            module_path,
            hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


class PackagedDatasetModuleFactory(_DatasetModuleFactory):
    """Get the dataset builder module from the ones that are packaged with the library: csv, json, etc."""

    def __init__(
        self,
        name: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        self.name = name
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config
        self.download_mode = download_mode
        increase_load_count(name)

    def get_module(self) -> DatasetModule:
        base_path = Path(self.data_dir or "").expanduser().resolve().as_posix()
        patterns = (
            sanitize_patterns(self.data_files)
            if self.data_files is not None
            else get_data_patterns(base_path, download_config=self.download_config)
        )
        data_files = DataFilesDict.from_patterns(
            patterns,
            download_config=self.download_config,
            base_path=base_path,
        )

        module_path, hash = _PACKAGED_DATASETS_MODULES[self.name]

        builder_kwargs = {
            "data_files": data_files,
            "dataset_name": self.name,
        }

        return DatasetModule(module_path, hash, builder_kwargs)


class HubDatasetModuleFactory(_DatasetModuleFactory):
    """
    Get the module of a dataset loaded from data files of a dataset repository.
    The dataset builder module to use is inferred from the data files extensions.
    """

    def __init__(
        self,
        name: str,
        commit_hash: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        use_exported_dataset_infos: bool = False,
    ):
        self.name = name
        self.commit_hash = commit_hash
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.use_exported_dataset_infos = use_exported_dataset_infos
        increase_load_count(name)

    def get_module(self) -> DatasetModule:
        # Get the Dataset Card and fix the revision in case there are new commits in the meantime
        api = HfApi(
            endpoint=config.HF_ENDPOINT,
            token=self.download_config.token,
            library_name="datasets",
            library_version=__version__,
            user_agent=get_datasets_user_agent(self.download_config.user_agent),
        )
        try:
            dataset_readme_path = api.hf_hub_download(
                repo_id=self.name,
                filename=config.REPOCARD_FILENAME,
                repo_type="dataset",
                revision=self.commit_hash,
            )
            dataset_card_data = DatasetCard.load(dataset_readme_path).data
        except EntryNotFoundError:
            dataset_card_data = DatasetCardData()
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading standalone yaml"
        try:
            standalone_yaml_path = cached_path(
                hf_dataset_url(self.name, config.REPOYAML_FILENAME, revision=self.commit_hash),
                download_config=download_config,
            )
            with open(standalone_yaml_path, encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        except FileNotFoundError:
            pass
        base_path = f"hf://datasets/{self.name}@{self.commit_hash}/{self.data_dir or ''}".rstrip("/")
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        if config.USE_PARQUET_EXPORT and self.use_exported_dataset_infos:
            try:
                exported_dataset_infos = _dataset_viewer.get_exported_dataset_infos(
                    dataset=self.name, commit_hash=self.commit_hash, token=self.download_config.token
                )
                exported_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name])
                        for config_name in exported_dataset_infos
                    }
                )
            except _dataset_viewer.DatasetViewerError:
                exported_dataset_infos = None
        else:
            exported_dataset_infos = None
        if exported_dataset_infos:
            exported_dataset_infos.update(dataset_infos)
            dataset_infos = exported_dataset_infos
        # we need a set of data files to find which dataset builder to use
        # because we need to infer module name by files extensions
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path, download_config=self.download_config)
        data_files = DataFilesDict.from_patterns(
            patterns,
            base_path=base_path,
            allowed_extensions=_ALL_ALLOWED_EXTENSIONS,
            download_config=self.download_config,
        )
        module_name, default_builder_kwargs = infer_module_for_data_files(
            data_files=data_files,
            path=self.name,
            download_config=self.download_config,
        )
        data_files = data_files.filter(
            extensions=_MODULE_TO_EXTENSIONS[module_name] + _MODULE_TO_METADATA_EXTENSIONS[module_name],
            file_names=_MODULE_TO_METADATA_FILE_NAMES[module_name],
        )
        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                default_builder_kwargs=default_builder_kwargs,
                download_config=self.download_config,
            )
        else:
            builder_configs: list[BuilderConfig] = [
                import_main_class(module_path).BUILDER_CONFIG_CLASS(
                    data_files=data_files,
                    **default_builder_kwargs,
                )
            ]
            default_config_name = None
        builder_kwargs = {
            "base_path": base_path,
            "repo_id": self.name,
            "dataset_name": camelcase_to_snakecase(Path(self.name).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading metadata"
        try:
            # this file is deprecated and was created automatically in old versions of push_to_hub
            dataset_infos_path = cached_path(
                hf_dataset_url(self.name, config.DATASETDICT_INFOS_FILENAME, revision=self.commit_hash),
                download_config=download_config,
            )
            with open(dataset_infos_path, encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
                if len(legacy_dataset_infos) == 1:
                    # old config e.g. named "username--dataset_name"
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        except FileNotFoundError:
            pass
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        return DatasetModule(
            module_path,
            self.commit_hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


class HubDatasetModuleFactoryWithParquetExport(_DatasetModuleFactory):
    """
    Get the module of a dataset loaded from parquet files of a dataset repository parquet export.
    """

    def __init__(
        self,
        name: str,
        commit_hash: str,
        download_config: Optional[DownloadConfig] = None,
    ):
        self.name = name
        self.commit_hash = commit_hash
        self.download_config = download_config or DownloadConfig()
        increase_load_count(name)

    def get_module(self) -> DatasetModule:
        exported_parquet_files = _dataset_viewer.get_exported_parquet_files(
            dataset=self.name, commit_hash=self.commit_hash, token=self.download_config.token
        )
        exported_dataset_infos = _dataset_viewer.get_exported_dataset_infos(
            dataset=self.name, commit_hash=self.commit_hash, token=self.download_config.token
        )
        dataset_infos = DatasetInfosDict(
            {
                config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name])
                for config_name in exported_dataset_infos
            }
        )
        parquet_commit_hash = (
            HfApi(
                endpoint=config.HF_ENDPOINT,
                token=self.download_config.token,
                library_name="datasets",
                library_version=__version__,
                user_agent=get_datasets_user_agent(self.download_config.user_agent),
            )
            .dataset_info(
                self.name,
                revision="refs/convert/parquet",
                token=self.download_config.token,
                timeout=100.0,
            )
            .sha
        )  # fix the revision in case there are new commits in the meantime
        metadata_configs = MetadataConfigs._from_exported_parquet_files_and_dataset_infos(
            parquet_commit_hash=parquet_commit_hash,
            exported_parquet_files=exported_parquet_files,
            dataset_infos=dataset_infos,
        )
        module_path, _ = _PACKAGED_DATASETS_MODULES["parquet"]
        builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
            module_path,
            metadata_configs,
            download_config=self.download_config,
        )
        builder_kwargs = {
            "repo_id": self.name,
            "dataset_name": camelcase_to_snakecase(Path(self.name).name),
        }

        return DatasetModule(
            module_path,
            self.commit_hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


class CachedDatasetModuleFactory(_DatasetModuleFactory):
    """
    Get the module of a dataset that has been loaded once already and cached.
    """

    def __init__(
        self,
        name: str,
        cache_dir: Optional[str] = None,
    ):
        self.name = name
        self.cache_dir = cache_dir
        assert self.name.count("/") <= 1

    def get_module(self) -> DatasetModule:
        cache_dir = os.path.expanduser(str(self.cache_dir or config.HF_DATASETS_CACHE))
        namespace_and_dataset_name = self.name.split("/")
        namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
        cached_relative_path = "___".join(namespace_and_dataset_name)
        cached_datasets_directory_path_root = os.path.join(cache_dir, cached_relative_path)
        cached_directory_paths = [
            cached_directory_path
            for cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", "*", "*"))
            if os.path.isdir(cached_directory_path)
        ]
        if cached_directory_paths:
            builder_kwargs = {
                "repo_id": self.name,
                "dataset_name": self.name.split("/")[-1],
            }
            warning_msg = f"Using the latest cached version of the dataset since {self.name} couldn't be found on the Hugging Face Hub"
            if config.HF_HUB_OFFLINE:
                warning_msg += " (offline mode is enabled)."
            logger.warning(warning_msg)
            return DatasetModule(
                "datasets.packaged_modules.cache.cache",
                "auto",
                {**builder_kwargs, "version": "auto"},
            )
        raise FileNotFoundError(f"Dataset {self.name} is not cached in {self.cache_dir}")


class HubBucketDatasetModuleFactory(_DatasetModuleFactory):
    """
    Get the module of a dataset loaded from data files of a a Storage Bucket.
    The dataset builder module to use is inferred from the data files extensions.
    """

    def __init__(
        self,
        path: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        self.path = Path(path).as_posix()
        self.name = Path(path).stem
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config
        self.download_mode = download_mode

    def get_module(self) -> DatasetModule:
        hffs = HfFileSystem(
            endpoint=config.HF_ENDPOINT,
            token=self.download_config.token,
        )
        readme_path = posixpath.join(self.path, config.REPOCARD_FILENAME)
        standalone_yaml_path = posixpath.join(self.path, config.REPOYAML_FILENAME)
        try:
            dataset_card_data = DatasetCard(hffs.read_text(readme_path, newline="", encoding="utf-8")).data
        except FileNotFoundError:
            dataset_card_data = DatasetCardData()
        try:
            standalone_yaml_data = yaml.safe_load(hffs.read_text(standalone_yaml_path, newline="", encoding="utf-8"))
        except FileNotFoundError:
            standalone_yaml_data = None
        if hffs.exists(standalone_yaml_path):
            with hffs.open(standalone_yaml_path, "r", encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        # we need a set of data files to find which dataset builder to use
        # because we need to infer module name by files extensions
        base_path = "hf://" + Path(self.path, self.data_dir or "").as_posix()
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path, download_config=self.download_config)
        data_files = DataFilesDict.from_patterns(
            patterns,
            base_path=base_path,
            allowed_extensions=_ALL_ALLOWED_EXTENSIONS,
        )
        module_name, default_builder_kwargs = infer_module_for_data_files(
            data_files=data_files,
            path=self.path,
        )
        data_files = data_files.filter(
            extensions=_MODULE_TO_EXTENSIONS[module_name] + _MODULE_TO_METADATA_EXTENSIONS[module_name],
            file_names=_MODULE_TO_METADATA_FILE_NAMES[module_name],
        )
        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                default_builder_kwargs=default_builder_kwargs,
            )
        else:
            builder_configs: list[BuilderConfig] = [
                import_main_class(module_path).BUILDER_CONFIG_CLASS(
                    data_files=data_files,
                    **default_builder_kwargs,
                )
            ]
            default_config_name = None
        builder_kwargs = {
            "base_path": base_path,
            "dataset_name": camelcase_to_snakecase(Path(self.path).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        # this file is deprecated and was created automatically in old versions of push_to_hub
        if hffs.isfile(xjoin(self.path, config.DATASETDICT_INFOS_FILENAME)):
            with hffs.open(xjoin(self.path, config.DATASETDICT_INFOS_FILENAME), "r", encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
                if len(legacy_dataset_infos) == 1:
                    # old config e.g. named "username--dataset_name"
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        hash = Hasher.hash({"dataset_infos": dataset_infos, "builder_configs": builder_configs})
        return DatasetModule(
            module_path,
            hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


def dataset_module_factory(
    path: str,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[dict, list, str, DataFilesDict]] = None,
    cache_dir: Optional[str] = None,
    **download_kwargs,
) -> DatasetModule:
    """
    Download/extract/cache a dataset module.

    Dataset codes are cached inside the dynamic modules cache to allow easy import (avoid ugly sys.path tweaks).

    Args:

        path (str): Path or name of the dataset.
            Depending on ``path``, the dataset builder that is used comes from one of the generic dataset builders (JSON, CSV, Parquet, text etc.).

            For local datasets:

            - if ``path`` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. ``'./path/to/directory/with/my/csv/data'``.

            For datasets on the Hugging Face Hub (list all available datasets with ``huggingface_hub.list_datasets()``)

            - if ``path`` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. ``'username/dataset_name'``, a dataset repository on the HF hub containing your data files.

            For datasets in Storage Buckets on the Hugging Face Hub (list all available buckets with ``huggingface_hub.list_buckets()``)

            - if `path` is a directory within a Storage Bucket on the HF Hub (containing data files only)
              -> load the dataset from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'buckets/username/bucket_name/my_dataset'`.

        revision (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        download_config (:class:`DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        data_dir (:obj:`str`, optional): Directory with the data files. Used only if `data_files` is not specified,
            in which case it's equal to pass `os.path.join(data_dir, "**")` as `data_files`.
        data_files (:obj:`Union[Dict, List, str]`, optional): Defining the data_files of the dataset configuration.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.

            <Added version="2.16.0"/>

        **download_kwargs (additional keyword arguments): optional attributes for DownloadConfig() which will override
            the attributes in download_config if supplied.

    Returns:
        DatasetModule
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    download_config.force_download = download_mode == DownloadMode.FORCE_REDOWNLOAD

    filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
    if not filename.endswith(".py"):
        filename = filename + ".py"
    combined_path = os.path.join(path, filename)

    # We have several ways to get a dataset builder:
    #
    # - if path is the name of a packaged dataset module
    #   -> use the packaged module (json, csv, etc.)
    #
    # - if path is a local directory (but no python file)
    #   -> use a packaged module (csv, text etc.) based on content of the directory
    #
    # - if path has one "/" and is dataset repository on the HF hub
    #   -> use a packaged module (csv, text etc.) based on content of the repository
    #
    # - if path starts with "buckets/" and points to a Storage Bucket on the HF hub
    #   -> use a packaged module (csv, text etc.) based on content of the directory in the bucket

    if path.startswith("hf://datasets/"):
        path = path[len("hf://datasets/") :]
        remote_only = True
    elif path.startswith("hf://buckets/"):
        path = path[len("hf://") :]
        remote_only = True
    else:
        remote_only = False

    # Try packaged
    if path in _PACKAGED_DATASETS_MODULES:
        return PackagedDatasetModuleFactory(
            path,
            data_dir=data_dir,
            data_files=data_files,
            download_config=download_config,
            download_mode=download_mode,
        ).get_module()
    # Try locally
    elif path.endswith(filename):
        raise RuntimeError(f"Dataset scripts are no longer supported, but found {filename}")
    elif os.path.isfile(combined_path):
        raise RuntimeError(f"Dataset scripts are no longer supported, but found {filename}")
    elif os.path.isdir(path) and not remote_only:
        return LocalDatasetModuleFactory(
            path, data_dir=data_dir, data_files=data_files, download_mode=download_mode
        ).get_module()
    # Try remotely
    elif path.startswith("buckets/"):
