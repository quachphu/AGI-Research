        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_fragments',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_fragments/')

        List the fragments:

        >>> dataset.fragments
        [<pyarrow.dataset.ParquetFileFragment path=dataset_v2_fragments/...
        """
        return list(self._dataset.get_fragments())

    @property
    def files(self):
        """
        A list of absolute Parquet file paths in the Dataset source.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_files',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_files/')

        List the files:

        >>> dataset.files
        ['dataset_v2_files/year=2019/...-0.parquet', ...
        """
        return self._dataset.files

    @property
    def filesystem(self):
        """
        The filesystem type of the Dataset source.
        """
        return self._dataset.filesystem

    @property
    def partitioning(self):
        """
        The partitioning of the Dataset source, if discovered.
        """
        return self._dataset.partitioning


_read_table_docstring = """
{0}

Parameters
----------
source : str, list of str, pyarrow.NativeFile, or file-like object
    If a string is passed, can be a single file name or directory name. If a
    list of strings is passed, should be file names. For file-like objects,
    only read a single file. Use pyarrow.BufferReader to read a file contained
    in a bytes or buffer-like object.
columns : list
    If not None, only these columns will be read from the file. A column
    name may be a prefix of a nested field, e.g. 'a' will select 'a.b',
    'a.c', and 'a.d.e'. If empty, no columns will be read. Note
    that the table will still have the correct num_rows set despite having
    no columns.
use_threads : bool, default True
    Perform multi-threaded column reads.
schema : Schema, optional
    Optionally provide the Schema for the parquet dataset, in which case it
    will not be inferred from the source.
{1}
filesystem : FileSystem, default None
    If nothing passed, will be inferred based on path.
    Path will try to be found in the local on-disk filesystem otherwise
    it will be parsed as an URI to determine the filesystem.
filters : pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]], default None
    Rows which do not match the filter predicate will be removed from scanned
    data. Partition keys embedded in a nested directory structure will be
    exploited to avoid loading files at all if they contain no matching rows.
    Within-file level filtering and different partitioning schemes are supported.

    {3}
ignore_prefixes : list, optional
    Files matching any of these prefixes will be ignored by the
    discovery process.
    This is matched to the basename of a path.
    By default this is ['.', '_'].
    Note that discovery happens only if a directory is passed as source.
pre_buffer : bool, default True
    Coalesce and issue file reads in parallel to improve performance on
    high-latency filesystems (e.g. S3). If True, Arrow will use a
    background I/O thread pool. If using a filesystem layer that itself
    performs readahead (e.g. fsspec's S3FS), disable readahead for best
    results.
coerce_int96_timestamp_unit : str, default None
    Cast timestamps that are stored in INT96 format to a particular
    resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
    and therefore INT96 timestamps will be inferred as timestamps
    in nanoseconds.
decryption_properties : FileDecryptionProperties or None
    File-level decryption properties.
    The decryption properties can be created using
    ``CryptoFactory.file_decryption_properties()``.
thrift_string_size_limit : int, default None
    If not None, override the maximum total string size allocated
    when decoding Thrift structures. The default limit should be
    sufficient for most Parquet files.
thrift_container_size_limit : int, default None
    If not None, override the maximum total size of containers allocated
    when decoding Thrift structures. The default limit should be
    sufficient for most Parquet files.
page_checksum_verification : bool, default False
    If True, verify the checksum for each page read from the file.
arrow_extensions_enabled : bool, default True
    If True, read Parquet logical types as Arrow extension types where possible,
    (e.g., read JSON as the canonical `arrow.json` extension type or UUID as
    the canonical `arrow.uuid` extension type).

Returns
-------
{2}

{4}
"""

_read_table_example = """\

Examples
--------

Generate an example PyArrow Table and write it to a partitioned dataset:

>>> import pyarrow as pa
>>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
...                   'n_legs': [2, 2, 4, 4, 5, 100],
...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
...                              "Brittle stars", "Centipede"]})
>>> import pyarrow.parquet as pq
>>> pq.write_to_dataset(table, root_path='dataset_name_2',
...                     partition_cols=['year'])

Read the data:

>>> pq.read_table('dataset_name_2').to_pandas()
   n_legs         animal  year
0       5  Brittle stars  2019
1       2       Flamingo  2020
2       4            Dog  2021
3     100      Centipede  2021
4       2         Parrot  2022
5       4          Horse  2022


Read only a subset of columns:

>>> pq.read_table('dataset_name_2', columns=["n_legs", "animal"])
pyarrow.Table
n_legs: int64
animal: string
----
n_legs: [[5],[2],[4,100],[2,4]]
animal: [["Brittle stars"],["Flamingo"],["Dog","Centipede"],["Parrot","Horse"]]

Read a subset of columns and read one column as DictionaryArray:

>>> pq.read_table('dataset_name_2', columns=["n_legs", "animal"],
...               read_dictionary=["animal"])
pyarrow.Table
n_legs: int64
animal: dictionary<values=string, indices=int32, ordered=0>
----
n_legs: [[5],[2],[4,100],[2,4]]
animal: [  -- dictionary:
["Brittle stars"]  -- indices:
[0],  -- dictionary:
["Flamingo"]  -- indices:
[0],  -- dictionary:
["Dog","Centipede"]  -- indices:
[0,1],  -- dictionary:
["Parrot","Horse"]  -- indices:
[0,1]]

Read the table with filter:

>>> pq.read_table('dataset_name_2', columns=["n_legs", "animal"],
...               filters=[('n_legs','<',4)]).to_pandas()
   n_legs    animal
0       2  Flamingo
1       2    Parrot

Read data from a single Parquet file:

>>> pq.write_table(table, 'example.parquet')
>>> pq.read_table('dataset_name_2').to_pandas()
   n_legs         animal  year
0       5  Brittle stars  2019
1       2       Flamingo  2020
2       4            Dog  2021
3     100      Centipede  2021
4       2         Parrot  2022
5       4          Horse  2022
"""


def read_table(source, *, columns=None, use_threads=True,
               schema=None, use_pandas_metadata=False, read_dictionary=None,
               binary_type=None, list_type=None, memory_map=False, buffer_size=0,
               partitioning="hive", filesystem=None, filters=None,
               ignore_prefixes=None, pre_buffer=True,
               coerce_int96_timestamp_unit=None,
               decryption_properties=None, thrift_string_size_limit=None,
               thrift_container_size_limit=None,
               page_checksum_verification=False,
               arrow_extensions_enabled=True):

    try:
        dataset = ParquetDataset(
            source,
            schema=schema,
            filesystem=filesystem,
            partitioning=partitioning,
            memory_map=memory_map,
            read_dictionary=read_dictionary,
            binary_type=binary_type,
            list_type=list_type,
            buffer_size=buffer_size,
            filters=filters,
            ignore_prefixes=ignore_prefixes,
            pre_buffer=pre_buffer,
            coerce_int96_timestamp_unit=coerce_int96_timestamp_unit,
            decryption_properties=decryption_properties,
            thrift_string_size_limit=thrift_string_size_limit,
            thrift_container_size_limit=thrift_container_size_limit,
            page_checksum_verification=page_checksum_verification,
            arrow_extensions_enabled=arrow_extensions_enabled,
        )
    except ImportError:
        # fall back on ParquetFile for simple cases when pyarrow.dataset
        # module is not available
        if filters is not None:
            raise ValueError(
                "the 'filters' keyword is not supported when the "
                "pyarrow.dataset module is not available"
            )
        if partitioning != "hive":
            raise ValueError(
                "the 'partitioning' keyword is not supported when the "
                "pyarrow.dataset module is not available"
            )
        if schema is not None:
            raise ValueError(
                "the 'schema' argument is not supported when the "
                "pyarrow.dataset module is not available"
            )
        if isinstance(source, list):
            raise ValueError(
                "the 'source' argument cannot be a list of files "
                "when the pyarrow.dataset module is not available"
            )

        filesystem, path = _resolve_filesystem_and_path(source, filesystem)
        if filesystem is not None:
            if not filesystem.get_file_info(path).is_file:
                raise ValueError(
                    "the 'source' argument should be "
                    "an existing parquet file and not a directory "
                    "when the pyarrow.dataset module is not available"
                )

            source = filesystem.open_input_file(path)

        dataset = ParquetFile(
            source, read_dictionary=read_dictionary,
            binary_type=binary_type,
            list_type=list_type,
            memory_map=memory_map, buffer_size=buffer_size,
            pre_buffer=pre_buffer,
            coerce_int96_timestamp_unit=coerce_int96_timestamp_unit,
            decryption_properties=decryption_properties,
            thrift_string_size_limit=thrift_string_size_limit,
            thrift_container_size_limit=thrift_container_size_limit,
            page_checksum_verification=page_checksum_verification,
        )

    return dataset.read(columns=columns, use_threads=use_threads,
                        use_pandas_metadata=use_pandas_metadata)


read_table.__doc__ = _read_table_docstring.format(
    """Read a Table from Parquet format""",
    "\n".join(("""use_pandas_metadata : bool, default False
    If True and file has custom pandas schema metadata, ensure that
    index columns are also loaded.""", _read_docstring_common)),
    """pyarrow.Table
    Content of the file as a table (of columns)""",
    _DNF_filter_doc, _read_table_example)


def read_pandas(source, columns=None, **kwargs):
    return read_table(
        source, columns=columns, use_pandas_metadata=True, **kwargs
    )


read_pandas.__doc__ = _read_table_docstring.format(
    'Read a Table from Parquet format, also reading DataFrame\n'
    'index values if known in the file metadata',
    "\n".join((_read_docstring_common,
               """**kwargs
    additional options for :func:`read_table`""")),
    """pyarrow.Table
    Content of the file as a Table of Columns, including DataFrame
    indexes as columns""",
    _DNF_filter_doc, "")


def write_table(table, where, row_group_size=None, version='2.6',
                use_dictionary=True, compression='snappy',
                write_statistics=True,
                use_deprecated_int96_timestamps=None,
                coerce_timestamps=None,
                allow_truncated_timestamps=False,
                data_page_size=None, flavor=None,
                filesystem=None,
                compression_level=None,
                use_byte_stream_split=False,
                column_encoding=None,
                data_page_version='1.0',
                use_compliant_nested_type=True,
                encryption_properties=None,
                write_batch_size=None,
                dictionary_pagesize_limit=None,
                store_schema=True,
                write_page_index=False,
                write_page_checksum=False,
                sorting_columns=None,
                store_decimal_as_integer=False,
                write_time_adjusted_to_utc=False,
                max_rows_per_page=None,
                bloom_filter_options=None,
                **kwargs):
    # Implementor's note: when adding keywords here / updating defaults, also
    # update it in write_to_dataset and _dataset_parquet.pyx ParquetFileWriteOptions
    row_group_size = kwargs.pop('chunk_size', row_group_size)
    use_int96 = use_deprecated_int96_timestamps
    try:
        with ParquetWriter(
                where, table.schema,
                filesystem=filesystem,
                version=version,
                flavor=flavor,
                use_dictionary=use_dictionary,
                write_statistics=write_statistics,
                coerce_timestamps=coerce_timestamps,
                data_page_size=data_page_size,
                allow_truncated_timestamps=allow_truncated_timestamps,
                compression=compression,
                use_deprecated_int96_timestamps=use_int96,
                compression_level=compression_level,
                use_byte_stream_split=use_byte_stream_split,
                column_encoding=column_encoding,
                data_page_version=data_page_version,
                use_compliant_nested_type=use_compliant_nested_type,
                encryption_properties=encryption_properties,
                write_batch_size=write_batch_size,
                dictionary_pagesize_limit=dictionary_pagesize_limit,
                store_schema=store_schema,
                write_page_index=write_page_index,
                write_page_checksum=write_page_checksum,
                sorting_columns=sorting_columns,
                store_decimal_as_integer=store_decimal_as_integer,
                write_time_adjusted_to_utc=write_time_adjusted_to_utc,
                max_rows_per_page=max_rows_per_page,
                bloom_filter_options=bloom_filter_options,
                **kwargs) as writer:
            writer.write_table(table, row_group_size=row_group_size)
    except Exception:
        if _is_path_like(where):
            try:
                os.remove(_stringify_path(where))
            except os.error:
                pass
        raise


_write_table_example = """\
Generate an example PyArrow Table:

>>> import pyarrow as pa
>>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
...                              "Brittle stars", "Centipede"]})

and write the Table into Parquet file:

>>> import pyarrow.parquet as pq
>>> pq.write_table(table, 'example.parquet')

Defining row group size for the Parquet file:

>>> pq.write_table(table, 'example.parquet', row_group_size=3)

Defining row group compression (default is Snappy):

>>> pq.write_table(table, 'example.parquet', compression='none')

Defining row group compression and encoding per-column:

>>> pq.write_table(table, 'example.parquet',
...                compression={'n_legs': 'snappy', 'animal': 'gzip'},
...                use_dictionary=['n_legs', 'animal'])

Defining column encoding per-column:

>>> pq.write_table(table, 'example.parquet',
...                column_encoding={'animal':'PLAIN'},
...                use_dictionary=False)
"""

write_table.__doc__ = f"""
Write a Table to Parquet format.

Parameters
----------
table : pyarrow.Table
where : string or pyarrow.NativeFile
row_group_size : int, default None
    Maximum number of rows in each written row group. If None, the
    row group size will be the minimum of the Table size (in rows)
    and 1024 * 1024. If set larger than 64 * 1024 * 1024 then
    64 * 1024 * 1024 will be used instead.
{_parquet_writer_arg_docs}
**kwargs : optional
    Additional options for ParquetWriter

Examples
--------
{_write_table_example}
"""


def write_to_dataset(table, root_path, partition_cols=None,
                     filesystem=None, schema=None, partitioning=None,
                     basename_template=None, use_threads=None,
                     file_visitor=None, existing_data_behavior=None,
                     **kwargs):
    """Wrapper around dataset.write_dataset for writing a Table to
    Parquet format by partitions.
    For each combination of partition columns and values,
    a subdirectories are created in the following
    manner:

    root_dir/
      group1=value1
        group2=value1
          <uuid>.parquet
        group2=value2
          <uuid>.parquet
      group1=valueN
        group2=value1
          <uuid>.parquet
        group2=valueN
          <uuid>.parquet

    Parameters
    ----------
    table : pyarrow.Table
    root_path : str, pathlib.Path
        The root directory of the dataset.
    partition_cols : list,
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    schema : Schema, optional
        This Schema of the dataset.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the
        ``pyarrow.dataset.partitioning()`` function or a list of field names.
        When providing a list of field names, you can use
        ``partitioning_flavor`` to drive which partitioning type should be
        used.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to "guid-{i}.parquet".
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores.
    file_visitor : function
        If set, this function will be called with a WrittenFile instance
        for each file created during the call.  This object will have both
        a path attribute and a metadata attribute.

        The path attribute will be a string containing the path to
        the created file.

        The metadata attribute will be the parquet metadata of the file.
        This metadata will have the file path attribute set and can be used
        to build a _metadata file.  The metadata attribute will be None if
        the format is not parquet.

        Example visitor which simple collects the filenames created::

            visited_paths = []

            def file_visitor(written_file):
                visited_paths.append(written_file.path)

    existing_data_behavior : 'overwrite_or_ignore' | 'error' | \
'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination. The default behaviour is 'overwrite_or_ignore'.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'error' will raise an error if any data exists in the destination.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    **kwargs : dict,
        Used as additional kwargs for :func:`pyarrow.dataset.write_dataset`
        function for matching kwargs, and remainder to
        :func:`pyarrow.dataset.ParquetFileFormat.make_write_options`.
        See the docstring of :func:`write_table` and
        :func:`pyarrow.dataset.write_dataset` for the available options.
        Using `metadata_collector` in kwargs allows one to collect the
        file metadata instances of dataset pieces. The file paths in the
        ColumnChunkMetaData will be set relative to `root_path`.

    Examples
    --------
    Generate an example PyArrow Table:

    >>> import pyarrow as pa
    >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
    ...                   'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    and write it to a partitioned dataset:

    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, root_path='dataset_name_3',
    ...                     partition_cols=['year'])
    >>> pq.ParquetDataset('dataset_name_3').files
    ['dataset_name_3/year=2019/...-0.parquet', ...

    Write a single Parquet file into the root folder:

    >>> pq.write_to_dataset(table, root_path='dataset_name_4')
    >>> pq.ParquetDataset('dataset_name_4/').files
    ['dataset_name_4/...-0.parquet']
    """
    metadata_collector = kwargs.pop('metadata_collector', None)

    # Check for conflicting keywords
    msg_confl = (
        "The '{1}' argument is not supported. "
        "Use only '{0}' instead."
    )
    if partition_cols is not None and partitioning is not None:
        raise ValueError(msg_confl.format("partitioning",
                                          "partition_cols"))

    if metadata_collector is not None and file_visitor is not None:
        raise ValueError(msg_confl.format("file_visitor",
                                          "metadata_collector"))

    import pyarrow.dataset as ds

    # extract write_dataset specific options
    # reset assumed to go to make_write_options
    write_dataset_kwargs = dict()
    for key in inspect.signature(ds.write_dataset).parameters:
        if key in kwargs:
            write_dataset_kwargs[key] = kwargs.pop(key)
    write_dataset_kwargs['max_rows_per_group'] = kwargs.pop(
        'row_group_size', kwargs.pop("chunk_size", None)
    )

    if metadata_collector is not None:
        def file_visitor(written_file):
            metadata_collector.append(written_file.metadata)

    # map format arguments
    parquet_format = ds.ParquetFileFormat()
    write_options = parquet_format.make_write_options(**kwargs)

    # map old filesystems to new one
    if filesystem is not None:
        filesystem = _ensure_filesystem(filesystem)

    if partition_cols:
        part_schema = table.select(partition_cols).schema
        partitioning = ds.partitioning(part_schema, flavor="hive")

    if basename_template is None:
        basename_template = guid() + '-{i}.parquet'

    if existing_data_behavior is None:
        existing_data_behavior = 'overwrite_or_ignore'

    ds.write_dataset(
        table, root_path, filesystem=filesystem,
        format=parquet_format, file_options=write_options, schema=schema,
        partitioning=partitioning, use_threads=use_threads,
        file_visitor=file_visitor,
        basename_template=basename_template,
        existing_data_behavior=existing_data_behavior,
        **write_dataset_kwargs)
    return


def write_metadata(schema, where, metadata_collector=None, filesystem=None,
                   **kwargs):
    """
    Write metadata-only Parquet file from schema. This can be used with
    `write_to_dataset` to generate `_common_metadata` and `_metadata` sidecar
    files.

    Parameters
    ----------
    schema : pyarrow.Schema
    where : string or pyarrow.NativeFile
    metadata_collector : list
        where to collect metadata information.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred from `where` if path-like, else
        `where` is already a file-like object so no filesystem is needed.
    **kwargs : dict,
        Additional kwargs for ParquetWriter class. See docstring for
        `ParquetWriter` for more information.

    Examples
    --------
    Generate example data:

    >>> import pyarrow as pa
    >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    Write a dataset and collect metadata information.

    >>> metadata_collector = []
    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(
    ...     table, 'dataset_metadata',
    ...      metadata_collector=metadata_collector)

    Write the `_common_metadata` parquet file without row groups statistics.

    >>> pq.write_metadata(
    ...     table.schema, 'dataset_metadata/_common_metadata')

    Write the `_metadata` parquet file with row groups statistics.

    >>> pq.write_metadata(
    ...     table.schema, 'dataset_metadata/_metadata',
    ...     metadata_collector=metadata_collector)
    """
    filesystem, where = _resolve_filesystem_and_path(where, filesystem)

    if hasattr(where, "seek"):  # file-like
        cursor_position = where.tell()

    writer = ParquetWriter(where, schema, filesystem, **kwargs)
    writer.close()

    if metadata_collector is not None:
        # ParquetWriter doesn't expose the metadata until it's written. Write
        # it and read it again.
        metadata = read_metadata(where, filesystem=filesystem)
        if hasattr(where, "seek"):
            where.seek(cursor_position)  # file-like, set cursor back.

        for m in metadata_collector:
            metadata.append_row_groups(m)
        if filesystem is not None:
            with filesystem.open_output_stream(where) as f:
                metadata.write_metadata_file(f)
        else:
            metadata.write_metadata_file(where)


def read_metadata(where, memory_map=False, decryption_properties=None,
                  filesystem=None, arrow_extensions_enabled=True):
    """
    Read FileMetaData from footer of a single Parquet file.

    Parameters
    ----------
    where : str (file path) or file-like object
    memory_map : bool, default False
        Create memory map when the source is a file path.
    decryption_properties : FileDecryptionProperties, default None
        Decryption properties for reading encrypted Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    arrow_extensions_enabled : bool, default True
        If True, read Parquet logical types as Arrow extension types where
        possible (e.g. UUID as the canonical `arrow.uuid` extension type).
        If False, use the underlying storage types instead.

    Returns
    -------
    metadata : FileMetaData
        The metadata of the Parquet file

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({'n_legs': [4, 5, 100],
    ...                   'animal': ["Dog", "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, 'example.parquet')

    >>> pq.read_metadata('example.parquet')
    <pyarrow._parquet.FileMetaData object at ...>
      created_by: parquet-cpp-arrow version ...
      num_columns: 2
      num_rows: 3
      num_row_groups: 1
      format_version: 2.6
      serialized_size: ...
    """
    filesystem, where = _resolve_filesystem_and_path(where, filesystem)
    file_ctx = nullcontext()
    if filesystem is not None:
        file_ctx = where = filesystem.open_input_file(where)

    with file_ctx:
        file = ParquetFile(
            where,
            memory_map=memory_map,
            decryption_properties=decryption_properties,
            arrow_extensions_enabled=arrow_extensions_enabled,
        )
        return file.metadata


def read_schema(where, memory_map=False, decryption_properties=None,
                filesystem=None, arrow_extensions_enabled=True):
    """
    Read effective Arrow schema from Parquet file metadata.

    Parameters
    ----------
    where : str (file path) or file-like object
    memory_map : bool, default False
        Create memory map when the source is a file path.
    decryption_properties : FileDecryptionProperties, default None
        Decryption properties for reading encrypted Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    arrow_extensions_enabled : bool, default True
        If True, read Parquet logical types as Arrow extension types where
        possible (e.g., UUID as the canonical `arrow.uuid` extension type).

    Returns
    -------
    schema : pyarrow.Schema
        The schema of the Parquet file

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({'n_legs': [4, 5, 100],
    ...                   'animal': ["Dog", "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, 'example.parquet')

    >>> pq.read_schema('example.parquet')
    n_legs: int64
    animal: string
    """
    filesystem, where = _resolve_filesystem_and_path(where, filesystem)
    file_ctx = nullcontext()
    if filesystem is not None:
        file_ctx = where = filesystem.open_input_file(where)

    with file_ctx:
        file = ParquetFile(
            where,
            memory_map=memory_map,
            decryption_properties=decryption_properties,
            arrow_extensions_enabled=arrow_extensions_enabled,
        )
        return file.schema_arrow


__all__ = (
    "ColumnChunkMetaData",
    "ColumnSchema",
    "FileDecryptionProperties",
    "FileEncryptionProperties",
    "FileMetaData",
    "ParquetDataset",
    "ParquetFile",
    "ParquetLogicalType",
    "ParquetReader",
    "ParquetSchema",
    "ParquetWriter",
    "RowGroupMetaData",
    "SortingColumn",
    "Statistics",
    "read_metadata",
    "read_pandas",
    "read_schema",
    "read_table",
    "write_metadata",
    "write_table",
    "write_to_dataset",
    "_filters_to_expression",
    "filters_to_expression",
)
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import enum
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Tuple,
)

import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.interchange.buffer import _PyArrowBuffer


class DtypeKind(enum.IntEnum):
    """
    Integer enum for data types.

    Attributes
    ----------
    INT : int
        Matches to signed integer data type.
    UINT : int
        Matches to unsigned integer data type.
    FLOAT : int
        Matches to floating point data type.
    BOOL : int
        Matches to boolean data type.
    STRING : int
        Matches to string data type (UTF-8 encoded).
    DATETIME : int
        Matches to datetime data type.
    CATEGORICAL : int
        Matches to categorical data type.
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


Dtype = Tuple[DtypeKind, int, str, str]  # see Column.dtype


_PYARROW_KINDS = {
    pa.int8(): (DtypeKind.INT, "c"),
    pa.int16(): (DtypeKind.INT, "s"),
    pa.int32(): (DtypeKind.INT, "i"),
    pa.int64(): (DtypeKind.INT, "l"),
    pa.uint8(): (DtypeKind.UINT, "C"),
    pa.uint16(): (DtypeKind.UINT, "S"),
    pa.uint32(): (DtypeKind.UINT, "I"),
    pa.uint64(): (DtypeKind.UINT, "L"),
    pa.float16(): (DtypeKind.FLOAT, "e"),
    pa.float32(): (DtypeKind.FLOAT, "f"),
    pa.float64(): (DtypeKind.FLOAT, "g"),
    pa.bool_(): (DtypeKind.BOOL, "b"),
    pa.string(): (DtypeKind.STRING, "u"),
    pa.large_string(): (DtypeKind.STRING, "U"),
}


class ColumnNullType(enum.IntEnum):
    """
    Integer enum for null type representation.

    Attributes
    ----------
    NON_NULLABLE : int
        Non-nullable column.
    USE_NAN : int
        Use explicit float NaN value.
    USE_SENTINEL : int
        Sentinel value besides NaN.
    USE_BITMASK : int
        The bit is set/unset representing a null on a certain position.
    USE_BYTEMASK : int
        The byte is set/unset representing a null on a certain position.
    """

    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4


class ColumnBuffers(TypedDict):
    # first element is a buffer containing the column data;
    # second element is the data buffer's associated dtype
    data: Tuple[_PyArrowBuffer, Dtype]

    # first element is a buffer containing mask values indicating missing data;
    # second element is the mask value buffer's associated dtype.
    # None if the null representation is not a bit or byte mask
    validity: Optional[Tuple[_PyArrowBuffer, Dtype]]

    # first element is a buffer containing the offset values for
    # variable-size binary data (e.g., variable-length strings);
    # second element is the offsets buffer's associated dtype.
    # None if the data buffer does not have an associated offsets buffer
    offsets: Optional[Tuple[_PyArrowBuffer, Dtype]]


class CategoricalDescription(TypedDict):
    # whether the ordering of dictionary indices is semantically meaningful
    is_ordered: bool
    # whether a dictionary-style mapping of categorical values to other objects
    # exists
    is_dictionary: bool
    # Python-level only (e.g. ``{int: str}``).
    # None if not a dictionary-style categorical.
    categories: Optional[_PyArrowColumn]


class Endianness:
    """Enum indicating the byte-order of a data-type."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"


class NoBufferPresent(Exception):
    """Exception to signal that there is no requested buffer."""


class _PyArrowColumn:
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.

         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.

         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.

    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.

         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(
        self, column: pa.Array | pa.ChunkedArray, allow_copy: bool = True
    ) -> None:
        """
        Handles PyArrow Arrays and ChunkedArrays.
        """
        # Store the column as a private attribute
        if isinstance(column, pa.ChunkedArray):
            if column.num_chunks == 1:
                column = column.chunk(0)
            else:
                if not allow_copy:
                    raise RuntimeError(
                        "Chunks will be combined and a copy is required which "
                        "is forbidden by allow_copy=False"
                    )
                column = column.combine_chunks()

        self._allow_copy = allow_copy

        if pa.types.is_boolean(column.type):
            if not allow_copy:
                raise RuntimeError(
                    "Boolean column will be casted to uint8 and a copy "
                    "is required which is forbidden by allow_copy=False"
                )
            self._dtype = self._dtype_from_arrowdtype(column.type, 8)
            self._col = pc.cast(column, pa.uint8())
        else:
            self._col = column
            dtype = self._col.type
            try:
                bit_width = dtype.bit_width
            except ValueError:
                # in case of a variable-length strings, considered as array
                # of bytes (8 bits)
                bit_width = 8
            self._dtype = self._dtype_from_arrowdtype(dtype, bit_width)

    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.

        Is a method rather than a property because it may cause a (potentially
        expensive) computation for some dataframe implementations.
        """
        return len(self._col)

    @property
    def offset(self) -> int:
        """
        Offset of first element.

        May be > 0 if using chunks; for example for a column with N chunks of
        equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.
        """
        return self._col.offset

    @property
    def dtype(self) -> Tuple[DtypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes:
            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for
              bit masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the
              future we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary,
              decimal, and nested (list, struct, map, union) dtypes.
        """
        return self._dtype

    def _dtype_from_arrowdtype(
        self, dtype: pa.DataType, bit_width: int
    ) -> Tuple[DtypeKind, int, str, str]:
        """
        See `self.dtype` for details.
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void)
        #       not handled datetime and timedelta both map to datetime
        #       (is timedelta handled?)

        if pa.types.is_timestamp(dtype):
            kind = DtypeKind.DATETIME
            ts = dtype.unit[0]
            tz = dtype.tz if dtype.tz else ""
            f_string = f"ts{ts}:{tz}"
            return kind, bit_width, f_string, Endianness.NATIVE
        elif pa.types.is_dictionary(dtype):
            kind = DtypeKind.CATEGORICAL
            arr = self._col
            indices_dtype = arr.indices.type
            _, f_string = _PYARROW_KINDS.get(indices_dtype)
            return kind, bit_width, f_string, Endianness.NATIVE
        else:
            kind, f_string = _PYARROW_KINDS.get(dtype, (None, None))
            if kind is None:
                raise ValueError(
                    f"Data type {dtype} not supported by interchange protocol")

            return kind, bit_width, f_string, Endianness.NATIVE

    @property
    def describe_categorical(self) -> CategoricalDescription:
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding categorical
          values.

        Raises TypeError if the dtype is not categorical

        Returns the dictionary with description on how to interpret the
        data buffer:
            - "is_ordered" : bool, whether the ordering of dictionary indices
                             is semantically meaningful.
            - "is_dictionary" : bool, whether a mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of
                             indices to category values (e.g. an array of
                             cat1, cat2, ...). None if not a dictionary-style
                             categorical.

        TBD: are there any other in-memory representations that are needed?
        """
        arr = self._col
        if not pa.types.is_dictionary(arr.type):
            raise TypeError(
                "describe_categorical only works on a column with "
                "categorical dtype!"
            )

        return {
            "is_ordered": self._col.type.ordered,
            "is_dictionary": True,
            "categories": _PyArrowColumn(arr.dictionary),
        }

    @property
    def describe_null(self) -> Tuple[ColumnNullType, Any]:
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Value : if kind is "sentinel value", the actual value. If kind is a bit
        mask or a byte mask, the value (0 or 1) indicating a missing value.
        None otherwise.
        """
        # In case of no missing values, we need to set ColumnNullType to
        # non nullable as in the current __dataframe__ protocol bit/byte masks
        # cannot be None
        if self.null_count == 0:
            return ColumnNullType.NON_NULLABLE, None
        else:
            return ColumnNullType.USE_BITMASK, 0

    @property
    def null_count(self) -> int:
        """
        Number of null elements, if known.

        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        arrow_null_count = self._col.null_count
        n = arrow_null_count if arrow_null_count != -1 else None
        return n

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        The metadata for the column. See `DataFrame.metadata` for more details.
        """
        pass

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        return 1

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable[_PyArrowColumn]:
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        if n_chunks and n_chunks > 1:
            chunk_size = self.size() // n_chunks
            if self.size() % n_chunks != 0:
                chunk_size += 1

            array = self._col
            i = 0
            for start in range(0, chunk_size * n_chunks, chunk_size):
                yield _PyArrowColumn(
                    array.slice(start, chunk_size), self._allow_copy
                )
                i += 1
        else:
            yield self

    def get_buffers(self) -> ColumnBuffers:
        """
        Return a dictionary containing the underlying buffers.

        The returned dictionary has the following contents:

            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """
        buffers: ColumnBuffers = {
            "data": self._get_data_buffer(),
            "validity": None,
            "offsets": None,
        }

        try:
            buffers["validity"] = self._get_validity_buffer()
        except NoBufferPresent:
            pass

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except NoBufferPresent:
            pass

        return buffers

    def _get_data_buffer(
        self,
    ) -> Tuple[_PyArrowBuffer, Any]:  # Any is for self.dtype tuple
        """
        Return the buffer containing the data and the buffer's
        associated dtype.
        """
        array = self._col
        dtype = self.dtype

        # In case of dictionary arrays, use indices
        # to define a buffer, codes are transferred through
        # describe_categorical()
        if pa.types.is_dictionary(array.type):
            array = array.indices
            dtype = _PyArrowColumn(array).dtype

        n = len(array.buffers())
        if n == 2:
            return _PyArrowBuffer(array.buffers()[1]), dtype
        elif n == 3:
            return _PyArrowBuffer(array.buffers()[2]), dtype

    def _get_validity_buffer(self) -> Tuple[_PyArrowBuffer, Any]:
        """
        Return the buffer containing the mask values indicating missing data
        and the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte
        mask.
        """
        # Define the dtype of the returned buffer
        dtype = (DtypeKind.BOOL, 1, "b", Endianness.NATIVE)
        array = self._col
        buff = array.buffers()[0]
        if buff:
            return _PyArrowBuffer(buff), dtype
        else:
            raise NoBufferPresent(
                "There are no missing values so "
                "does not have a separate mask")

    def _get_offsets_buffer(self) -> Tuple[_PyArrowBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
        array = self._col
        n = len(array.buffers())
        if n == 2:
            raise NoBufferPresent(
                "This column has a fixed-length dtype so "
                "it does not have an offsets buffer"
            )
        elif n == 3:
            # Define the dtype of the returned buffer
            dtype = self._col.type
            if pa.types.is_large_string(dtype):
                dtype = (DtypeKind.INT, 64, "l", Endianness.NATIVE)
            else:
                dtype = (DtypeKind.INT, 32, "i", Endianness.NATIVE)
            return _PyArrowBuffer(array.buffers()[1]), dtype
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

from typing import (
    Any,
    Tuple,
)

from pyarrow.interchange.column import (
    DtypeKind,
    ColumnBuffers,
    ColumnNullType,
)

import pyarrow as pa
import re

import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype


# A typing protocol could be added later to let Mypy validate code using
# `from_dataframe` better.
DataFrameObject = Any
ColumnObject = Any
BufferObject = Any


_PYARROW_DTYPES: dict[DtypeKind, dict[int, Any]] = {
    DtypeKind.INT: {8: pa.int8(),
                    16: pa.int16(),
                    32: pa.int32(),
                    64: pa.int64()},
    DtypeKind.UINT: {8: pa.uint8(),
                     16: pa.uint16(),
                     32: pa.uint32(),
                     64: pa.uint64()},
    DtypeKind.FLOAT: {16: pa.float16(),
                      32: pa.float32(),
                      64: pa.float64()},
    DtypeKind.BOOL: {1: pa.bool_(),
                     8: pa.uint8()},
    DtypeKind.STRING: {8: pa.string()},
}


def from_dataframe(df: DataFrameObject, allow_copy=True) -> pa.Table:
    """
    Build a ``pa.Table`` from any DataFrame supporting the interchange protocol.

    Parameters
    ----------
    df : DataFrameObject
        Object supporting the interchange protocol, i.e. `__dataframe__`
        method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Table
    """
    if isinstance(df, pa.Table):
        return df
    elif isinstance(df, pa.RecordBatch):
        return pa.Table.from_batches([df])

    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    return _from_dataframe(df.__dataframe__(allow_copy=allow_copy),
                           allow_copy=allow_copy)


def _from_dataframe(df: DataFrameObject, allow_copy=True):
    """
    Build a ``pa.Table`` from the DataFrame interchange object.

    Parameters
    ----------
    df : DataFrameObject
        Object supporting the interchange protocol, i.e. `__dataframe__`
        method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Table
    """
    batches = []
    for chunk in df.get_chunks():
        batch = protocol_df_chunk_to_pyarrow(chunk, allow_copy)
        batches.append(batch)

    if not batches:
        batch = protocol_df_chunk_to_pyarrow(df)
        batches.append(batch)

    return pa.Table.from_batches(batches)


def protocol_df_chunk_to_pyarrow(
    df: DataFrameObject,
    allow_copy: bool = True
) -> pa.RecordBatch:
    """
    Convert interchange protocol chunk to ``pa.RecordBatch``.

    Parameters
    ----------
    df : DataFrameObject
        Object supporting the interchange protocol, i.e. `__dataframe__`
        method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.RecordBatch
    """
    # We need a dict of columns here, with each column being a pa.Array
    columns: dict[str, pa.Array] = {}
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f"Column {name} is not a string")
        if name in columns:
            raise ValueError(f"Column {name} is not unique")
        col = df.get_column_by_name(name)
        dtype = col.dtype[0]
        if dtype in (
            DtypeKind.INT,
            DtypeKind.UINT,
            DtypeKind.FLOAT,
            DtypeKind.STRING,
            DtypeKind.DATETIME,
        ):
            columns[name] = column_to_array(col, allow_copy)
        elif dtype == DtypeKind.BOOL:
            columns[name] = bool_column_to_array(col, allow_copy)
        elif dtype == DtypeKind.CATEGORICAL:
            columns[name] = categorical_column_to_dictionary(col, allow_copy)
        else:
            raise NotImplementedError(f"Data type {dtype} not handled yet")

    return pa.RecordBatch.from_pydict(columns)


def column_to_array(
    col: ColumnObject,
    allow_copy: bool = True,
) -> pa.Array:
    """
    Convert a column holding one of the primitive dtypes to a PyArrow array.
    A primitive type is one of: int, uint, float, bool (1 bit).

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array
    """
    buffers = col.get_buffers()
    data_type = col.dtype
    data = buffers_to_array(buffers, data_type,
                            col.size(),
                            col.describe_null,
                            col.offset,
                            allow_copy)
    return data


def bool_column_to_array(
    col: ColumnObject,
    allow_copy: bool = True,
) -> pa.Array:
    """
    Convert a column holding boolean dtype to a PyArrow array.

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array
    """
    buffers = col.get_buffers()
    size = buffers["data"][1][1]

    # If booleans are byte-packed a copy to bit-packed will be made
    if size == 8 and not allow_copy:
        raise RuntimeError(
            "Boolean column will be casted from uint8 and a copy "
            "is required which is forbidden by allow_copy=False"
        )

    data_type = col.dtype
    data = buffers_to_array(buffers, data_type,
                            col.size(),
                            col.describe_null,
                            col.offset)
    if size == 8:
        data = pc.cast(data, pa.bool_())

    return data


def categorical_column_to_dictionary(
    col: ColumnObject,
    allow_copy: bool = True,
) -> pa.DictionaryArray:
    """
    Convert a column holding categorical data to a pa.DictionaryArray.

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.DictionaryArray
    """
    if not allow_copy:
        raise RuntimeError(
            "Categorical column will be casted from uint8 and a copy "
            "is required which is forbidden by allow_copy=False"
        )

    categorical = col.describe_categorical

    if not categorical["is_dictionary"]:
        raise NotImplementedError(
            "Non-dictionary categoricals not supported yet")

    # We need to first convert the dictionary column
    cat_column = categorical["categories"]
    dictionary = column_to_array(cat_column)
    # Then we need to convert the indices
    # Here we need to use the buffer data type!
    buffers = col.get_buffers()
    _, data_type = buffers["data"]
    indices = buffers_to_array(buffers, data_type,
                               col.size(),
                               col.describe_null,
                               col.offset)

    # Constructing a pa.DictionaryArray
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)

    return dict_array


def parse_datetime_format_str(format_str):
    """Parse datetime `format_str` to interpret the `data`."""

    # timestamp 'ts{unit}:tz'
    timestamp_meta = re.match(r"ts([smun]):(.*)", format_str)
    if timestamp_meta:
        unit, tz = timestamp_meta.group(1), timestamp_meta.group(2)
        if unit != "s":
            # the format string describes only a first letter of the unit, so
            # add one extra letter to convert the unit to numpy-style:
            # 'm' -> 'ms', 'u' -> 'us', 'n' -> 'ns'
            unit += "s"

        return unit, tz

    raise NotImplementedError(f"DateTime kind is not supported: {format_str}")


def map_date_type(data_type):
    """Map column date type to pyarrow date type. """
    kind, bit_width, f_string, _ = data_type

    if kind == DtypeKind.DATETIME:
        unit, tz = parse_datetime_format_str(f_string)
        return pa.timestamp(unit, tz=tz)
    else:
        pa_dtype = _PYARROW_DTYPES.get(kind, {}).get(bit_width, None)

        # Error if dtype is not supported
        if pa_dtype:
            return pa_dtype
        else:
            raise NotImplementedError(
                f"Conversion for {data_type} is not yet supported.")


def buffers_to_array(
    buffers: ColumnBuffers,
    data_type: Tuple[DtypeKind, int, str, str],
    length: int,
    describe_null: ColumnNullType,
    offset: int = 0,
    allow_copy: bool = True,
) -> pa.Array:
    """
    Build a PyArrow array from the passed buffer.

    Parameters
    ----------
    buffer : ColumnBuffers
        Dictionary containing tuples of underlying buffers and
        their associated dtype.
    data_type : Tuple[DtypeKind, int, str, str],
        Dtype description of the column as a tuple ``(kind, bit-width, format string,
        endianness)``.
    length : int
        The number of values in the array.
    describe_null: ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array

    Notes
    -----
    The returned array doesn't own the memory. The caller of this function
    is responsible for keeping the memory owner object alive as long as
    the returned PyArrow array is being used.
    """
    data_buff, _ = buffers["data"]
    try:
        validity_buff, validity_dtype = buffers["validity"]
    except TypeError:
        validity_buff = None
    try:
        offset_buff, offset_dtype = buffers["offsets"]
    except TypeError:
        offset_buff = None

    # Construct a pyarrow Buffer
    data_pa_buffer = pa.foreign_buffer(data_buff.ptr, data_buff.bufsize,
                                       base=data_buff)

    # Construct a validity pyarrow Buffer, if applicable
    if validity_buff:
        validity_pa_buff = validity_buffer_from_mask(validity_buff,
                                                     validity_dtype,
                                                     describe_null,
                                                     length,
                                                     offset,
                                                     allow_copy)
    else:
        validity_pa_buff = validity_buffer_nan_sentinel(data_pa_buffer,
                                                        data_type,
                                                        describe_null,
                                                        length,
                                                        offset,
                                                        allow_copy)

    # Construct a pyarrow Array from buffers
    data_dtype = map_date_type(data_type)

    if offset_buff:
        _, offset_bit_width, _, _ = offset_dtype
        # If an offset buffer exists, construct an offset pyarrow Buffer
        # and add it to the construction of an array
        offset_pa_buffer = pa.foreign_buffer(offset_buff.ptr,
                                             offset_buff.bufsize,
                                             base=offset_buff)

        if data_type[2] == 'U':
            string_type = pa.large_string()
        else:
            if offset_bit_width == 64:
                string_type = pa.large_string()
            else:
                string_type = pa.string()
        array = pa.Array.from_buffers(
            string_type,
            length,
            [validity_pa_buff, offset_pa_buffer, data_pa_buffer],
            offset=offset,
        )
    else:
        array = pa.Array.from_buffers(
            data_dtype,
            length,
            [validity_pa_buff, data_pa_buffer],
            offset=offset,
        )

    return array


def validity_buffer_from_mask(
    validity_buff: BufferObject,
    validity_dtype: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> pa.Buffer:
    """
    Build a PyArrow buffer from the passed mask buffer.

    Parameters
    ----------
    validity_buff : BufferObject
        Tuple of underlying validity buffer and associated dtype.
    validity_dtype : Dtype
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.
    describe_null : ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    length : int
        The number of values in the array.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Buffer
    """
    null_kind, sentinel_val = describe_null
    validity_kind, _, _, _ = validity_dtype
    assert validity_kind == DtypeKind.BOOL

    if null_kind == ColumnNullType.NON_NULLABLE:
        # Sliced array can have a NON_NULLABLE ColumnNullType due
        # to no missing values in that slice of an array though the bitmask
        # exists and validity_buff must be set to None in this case
        return None

    elif null_kind == ColumnNullType.USE_BYTEMASK or (
        null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 1
    ):
        buff = pa.foreign_buffer(validity_buff.ptr,
                                 validity_buff.bufsize,
                                 base=validity_buff)

        if null_kind == ColumnNullType.USE_BYTEMASK:
            if not allow_copy:
                raise RuntimeError(
                    "To create a bitmask a copy of the data is "
                    "required which is forbidden by allow_copy=False"
                )
            mask = pa.Array.from_buffers(pa.int8(), length,
                                         [None, buff],
                                         offset=offset)
            mask_bool = pc.cast(mask, pa.bool_())
        else:
            mask_bool = pa.Array.from_buffers(pa.bool_(), length,
                                              [None, buff],
                                              offset=offset)

        if sentinel_val == 1:
            mask_bool = pc.invert(mask_bool)

        return mask_bool.buffers()[1]

    elif null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 0:
        return pa.foreign_buffer(validity_buff.ptr,
                                 validity_buff.bufsize,
                                 base=validity_buff)
    else:
        raise NotImplementedError(
            f"{describe_null} null representation is not yet supported.")


def validity_buffer_nan_sentinel(
    data_pa_buffer: BufferObject,
    data_type: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> pa.Buffer:
    """
    Build a PyArrow buffer from NaN or sentinel values.

    Parameters
    ----------
    data_pa_buffer : pa.Buffer
        PyArrow buffer for the column data.
    data_type : Dtype
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.
    describe_null : ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    length : int
        The number of values in the array.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Buffer
    """
    kind, bit_width, _, _ = data_type
    data_dtype = map_date_type(data_type)
    null_kind, sentinel_val = describe_null

    # Check for float NaN values
    if null_kind == ColumnNullType.USE_NAN:
        if not allow_copy:
            raise RuntimeError(
                "To create a bitmask a copy of the data is "
                "required which is forbidden by allow_copy=False"
            )

        if kind == DtypeKind.FLOAT and bit_width == 16:
            # 'pyarrow.compute.is_nan' kernel not yet implemented
            # for float16
            raise NotImplementedError(
                f"{data_type} with {null_kind} is not yet supported.")
        else:
            pyarrow_data = pa.Array.from_buffers(
                data_dtype,
                length,
                [None, data_pa_buffer],
                offset=offset,
            )
            mask = pc.is_nan(pyarrow_data)
            mask = pc.invert(mask)
            return mask.buffers()[1]

    # Check for sentinel values
    elif null_kind == ColumnNullType.USE_SENTINEL:
        if not allow_copy:
            raise RuntimeError(
                "To create a bitmask a copy of the data is "
                "required which is forbidden by allow_copy=False"
            )

        if kind == DtypeKind.DATETIME:
            sentinel_dtype = pa.int64()
        else:
            sentinel_dtype = data_dtype
        pyarrow_data = pa.Array.from_buffers(sentinel_dtype,
                                             length,
                                             [None, data_pa_buffer],
                                             offset=offset)
        sentinel_arr = pc.equal(pyarrow_data, sentinel_val)
        mask_bool = pc.invert(sentinel_arr)
        return mask_bool.buffers()[1]

    elif null_kind == ColumnNullType.NON_NULLABLE:
        pass
    else:
        raise NotImplementedError(
            f"{describe_null} null representation is not yet supported.")
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations
from typing import (
    Any,
    Iterable,
    Optional,
    Sequence,
)

import pyarrow as pa

from pyarrow.interchange.column import _PyArrowColumn


class _PyArrowDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string.
    Columns may be accessed by name or by position.

    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    def __init__(
        self, df: pa.Table | pa.RecordBatch,
        nan_as_null: bool = False,
        allow_copy: bool = True
    ) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `pa.Table.__dataframe__` or `pa.RecordBatch.__dataframe__`.
        """
        self._df = df
        # ``nan_as_null`` is a keyword intended for the consumer to tell the
        # producer to overwrite null values in the data with ``NaN`` (or
        # ``NaT``).
        if nan_as_null is True:
            raise RuntimeError(
                "nan_as_null=True currently has no effect, "
                "use the default nan_as_null=False"
            )
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame:
        """
        Construct a new exchange object, potentially changing the parameters.
        ``nan_as_null`` is a keyword intended for the consumer to tell the
        producer to overwrite null values in the data with ``NaN``.
        It is intended for cases where the consumer does not support the bit
        mask or byte mask that is the producer's native representation.
        ``allow_copy`` is a keyword that defines whether or not the library is
        allowed to make a copy of the data. For example, copying data would be
        necessary if a library supports strided buffers, given that this
        protocol specifies contiguous buffers.
        """
        return _PyArrowDataFrame(self._df, nan_as_null, allow_copy)

    @property
    def metadata(self) -> dict[str, Any]:
        """
        The metadata for the data frame, as a dictionary with string keys. The
        contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.
        """
        # The metadata for the data frame, as a dictionary with string keys.
        # Add schema metadata here (pandas metadata or custom metadata)
        if self._df.schema.metadata:
            schema_metadata = {"pyarrow." + k.decode('utf8'): v.decode('utf8')
                               for k, v in self._df.schema.metadata.items()}
            return schema_metadata
        else:
            return {}

    def num_columns(self) -> int:
        """
        Return the number of columns in the DataFrame.
        """
        return self._df.num_columns

    def num_rows(self) -> int:
        """
        Return the number of rows in the DataFrame, if available.
        """
        return self._df.num_rows

    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.
        """
        if isinstance(self._df, pa.RecordBatch):
            return 1
        else:
            # pyarrow.Table can have columns with different number
            # of chunks so we take the number of chunks that
            # .to_batches() returns as it takes the min chunk size
            # of all the columns (to_batches is a zero copy method)
            batches = self._df.to_batches()
            return len(batches)

    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.
        """
        return self._df.schema.names

    def get_column(self, i: int) -> _PyArrowColumn:
        """
        Return the column at the indicated position.
        """
        return _PyArrowColumn(self._df.column(i),
                              allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _PyArrowColumn:
        """
        Return the column whose name is the indicated name.
        """
        return _PyArrowColumn(self._df.column(name),
                              allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_PyArrowColumn]:
        """
        Return an iterator yielding the columns.
        """
        return [
            _PyArrowColumn(col, allow_copy=self._allow_copy)
            for col in self._df.columns
        ]

    def select_columns(self, indices: Sequence[int]) -> _PyArrowDataFrame:
        """
        Create a new DataFrame by selecting a subset of columns by index.
        """
        return _PyArrowDataFrame(
            self._df.select(list(indices)), self._nan_as_null, self._allow_copy
        )

    def select_columns_by_name(
        self, names: Sequence[str]
    ) -> _PyArrowDataFrame:
        """
        Create a new DataFrame by selecting a subset of columns by name.
        """
        return _PyArrowDataFrame(
            self._df.select(list(names)), self._nan_as_null, self._allow_copy
        )

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable[_PyArrowDataFrame]:
        """
        Return an iterator yielding the chunks.

        By default (None), yields the chunks that the data is stored as by the
        producer. If given, ``n_chunks`` must be a multiple of
        ``self.num_chunks()``, meaning the producer must subdivide each chunk
        before yielding it.

        Note that the producer must ensure that all columns are chunked the
        same way.
        """
        # Subdivide chunks
        if n_chunks and n_chunks > 1:
            chunk_size = self.num_rows() // n_chunks
            if self.num_rows() % n_chunks != 0:
                chunk_size += 1
            if isinstance(self._df, pa.Table):
                batches = self._df.to_batches(max_chunksize=chunk_size)
            else:
                batches = []
                for start in range(0, chunk_size * n_chunks, chunk_size):
                    batches.append(self._df.slice(start, chunk_size))
            # In case when the size of the chunk is such that the resulting
            # list is one less chunk then n_chunks -> append an empty chunk
            if len(batches) == n_chunks - 1:
                batches.append(pa.record_batch([[]], schema=self._df.schema))
        # yields the chunks that the data is stored as
        else:
            if isinstance(self._df, pa.Table):
                batches = self._df.to_batches()
            else:
                batches = [self._df]

        # Create an iterator of RecordBatches
        iterator = [_PyArrowDataFrame(batch,
                                      self._nan_as_null,
                                      self._allow_copy)
                    for batch in batches]
        return iterator
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations
import enum

import pyarrow as pa


class DlpackDeviceType(enum.IntEnum):
    """Integer enum for device type codes matching DLPack."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


class _PyArrowBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.
    """

    def __init__(self, x: pa.Buffer, allow_copy: bool = True) -> None:
        """
        Handle PyArrow Buffers.
        """
        self._x = x

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
        return self._x.size

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._x.address

    def __dlpack__(self):
        """
        Produce DLPack capsule (see array API standard).

        Raises:
            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented

        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        Uses device type codes matching DLPack.
        Note: must be implemented even if ``__dlpack__`` is not.
        """
        if self._x.is_cpu:
            return (DlpackDeviceType.CPU, None)
        else:
            raise NotImplementedError("__dlpack_device__")

    def __repr__(self) -> str:
        return (
            "PyArrowBuffer(" +
            str(
                {
                    "bufsize": self.bufsize,
                    "ptr": self.ptr,
                    "device": self.__dlpack_device__()[0].name,
                }
            ) +
            ")"
        )
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import string
import typing as t

try:
    from ._speedups import _escape_inner
except ImportError:
    from ._native import _escape_inner

if t.TYPE_CHECKING:
    import typing_extensions as te


class _HasHTML(t.Protocol):
    def __html__(self, /) -> str: ...


class _TPEscape(t.Protocol):
    def __call__(self, s: t.Any, /) -> Markup: ...


def escape(s: t.Any, /) -> Markup:
    """Replace the characters ``&``, ``<``, ``>``, ``'``, and ``"`` in
    the string with HTML-safe sequences. Use this if you need to display
    text that might contain such characters in HTML.

    If the object has an ``__html__`` method, it is called and the
    return value is assumed to already be safe for HTML.

    :param s: An object to be converted to a string and escaped.
    :return: A :class:`Markup` string with the escaped text.
    """
    # If the object is already a plain string, skip __html__ check and string
    # conversion. This is the most common use case.
    # Use type(s) instead of s.__class__ because a proxy object may be reporting
    # the __class__ of the proxied value.
    if type(s) is str:
        return Markup(_escape_inner(s))

    if hasattr(s, "__html__"):
        return Markup(s.__html__())

    return Markup(_escape_inner(str(s)))


def escape_silent(s: t.Any | None, /) -> Markup:
    """Like :func:`escape` but treats ``None`` as the empty string.
    Useful with optional values, as otherwise you get the string
    ``'None'`` when the value is ``None``.

    >>> escape(None)
    Markup('None')
    >>> escape_silent(None)
    Markup('')
    """
    if s is None:
        return Markup()

    return escape(s)


def soft_str(s: t.Any, /) -> str:
    """Convert an object to a string if it isn't already. This preserves
    a :class:`Markup` string rather than converting it back to a basic
    string, so it will still be marked as safe and won't be escaped
    again.

    >>> value = escape("<User 1>")
    >>> value
    Markup('&lt;User 1&gt;')
    >>> escape(str(value))
    Markup('&amp;lt;User 1&amp;gt;')
    >>> escape(soft_str(value))
    Markup('&lt;User 1&gt;')
    """
    if not isinstance(s, str):
        return str(s)

    return s


class Markup(str):
    """A string that is ready to be safely inserted into an HTML or XML
    document, either because it was escaped or because it was marked
    safe.

    Passing an object to the constructor converts it to text and wraps
    it to mark it safe without escaping. To escape the text, use the
    :meth:`escape` class method instead.

    >>> Markup("Hello, <em>World</em>!")
    Markup('Hello, <em>World</em>!')
    >>> Markup(42)
    Markup('42')
    >>> Markup.escape("Hello, <em>World</em>!")
    Markup('Hello &lt;em&gt;World&lt;/em&gt;!')

    This implements the ``__html__()`` interface that some frameworks
    use. Passing an object that implements ``__html__()`` will wrap the
    output of that method, marking it safe.

    >>> class Foo:
    ...     def __html__(self):
    ...         return '<a href="/foo">foo</a>'
    ...
    >>> Markup(Foo())
    Markup('<a href="/foo">foo</a>')

    This is a subclass of :class:`str`. It has the same methods, but
    escapes their arguments and returns a ``Markup`` instance.

    >>> Markup("<em>%s</em>") % ("foo & bar",)
    Markup('<em>foo &amp; bar</em>')
    >>> Markup("<em>Hello</em> ") + "<foo>"
    Markup('<em>Hello</em> &lt;foo&gt;')
    """

    __slots__ = ()

    def __new__(
        cls, object: t.Any = "", encoding: str | None = None, errors: str = "strict"
    ) -> te.Self:
        if hasattr(object, "__html__"):
            object = object.__html__()

        if encoding is None:
            return super().__new__(cls, object)

        return super().__new__(cls, object, encoding, errors)

    def __html__(self, /) -> te.Self:
        return self

    def __add__(self, value: str | _HasHTML, /) -> te.Self:
        if isinstance(value, str) or hasattr(value, "__html__"):
            return self.__class__(super().__add__(self.escape(value)))

        return NotImplemented

    def __radd__(self, value: str | _HasHTML, /) -> te.Self:
        if isinstance(value, str) or hasattr(value, "__html__"):
            return self.escape(value).__add__(self)

        return NotImplemented

    def __mul__(self, value: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().__mul__(value))

    def __rmul__(self, value: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().__mul__(value))

    def __mod__(self, value: t.Any, /) -> te.Self:
        if isinstance(value, tuple):
            # a tuple of arguments, each wrapped
            value = tuple(_MarkupEscapeHelper(x, self.escape) for x in value)
        elif hasattr(type(value), "__getitem__") and not isinstance(value, str):
            # a mapping of arguments, wrapped
            value = _MarkupEscapeHelper(value, self.escape)
        else:
            # a single argument, wrapped with the helper and a tuple
            value = (_MarkupEscapeHelper(value, self.escape),)

        return self.__class__(super().__mod__(value))

    def __repr__(self, /) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    def join(self, iterable: cabc.Iterable[str | _HasHTML], /) -> te.Self:
        return self.__class__(super().join(map(self.escape, iterable)))

    def split(  # type: ignore[override]
        self, /, sep: str | None = None, maxsplit: t.SupportsIndex = -1
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().split(sep, maxsplit)]

    def rsplit(  # type: ignore[override]
        self, /, sep: str | None = None, maxsplit: t.SupportsIndex = -1
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().rsplit(sep, maxsplit)]

    def splitlines(  # type: ignore[override]
        self, /, keepends: bool = False
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().splitlines(keepends)]

    def unescape(self, /) -> str:
        """Convert escaped markup back into a text string. This replaces
        HTML entities with the characters they represent.

        >>> Markup("Main &raquo; <em>About</em>").unescape()
        'Main » <em>About</em>'
        """
        from html import unescape

        return unescape(str(self))

    def striptags(self, /) -> str:
        """:meth:`unescape` the markup, remove tags, and normalize
        whitespace to single spaces.

        >>> Markup("Main &raquo;\t<em>About</em>").striptags()
        'Main » About'
        """
        value = str(self)
        parts = []
        pos = 0

        while (start := value.find("<", pos)) != -1:
            if value.startswith("<!--", start):
                # comment
                if (end := value.find("-->", start + 4)) == -1:
                    # unclosed
                    break

                end += 3
            else:
                # tag
                if (end := value.find(">", start)) == -1:
                    # unclosed
                    break

                end += 1

            parts.append(value[pos:start])
            pos = end

        # found at least one tag, combine parts
        if pos > 0:
            # any trailing data after the last closed tag
            parts.append(value[pos:])
            value = "".join(parts)

        # collapse spaces
        value = " ".join(value.split())
        # unescape the processed value using the current class
        return self.__class__(value).unescape()

    @classmethod
    def escape(cls, s: t.Any, /) -> te.Self:
        """Escape a string. Calls :func:`escape` and ensures that for
        subclasses the correct type is returned.
        """
        rv = escape(s)

        if rv.__class__ is not cls:
            return cls(rv)

        return rv

    def __getitem__(self, key: t.SupportsIndex | slice, /) -> te.Self:
        return self.__class__(super().__getitem__(key))

    def capitalize(self, /) -> te.Self:
        return self.__class__(super().capitalize())

    def title(self, /) -> te.Self:
        return self.__class__(super().title())

    def lower(self, /) -> te.Self:
        return self.__class__(super().lower())

    def upper(self, /) -> te.Self:
        return self.__class__(super().upper())

    def replace(self, old: str, new: str, count: t.SupportsIndex = -1, /) -> te.Self:
        return self.__class__(super().replace(old, self.escape(new), count))

    def ljust(self, width: t.SupportsIndex, fillchar: str = " ", /) -> te.Self:
        return self.__class__(super().ljust(width, self.escape(fillchar)))

    def rjust(self, width: t.SupportsIndex, fillchar: str = " ", /) -> te.Self:
        return self.__class__(super().rjust(width, self.escape(fillchar)))

    def lstrip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().lstrip(chars))

    def rstrip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().rstrip(chars))

    def center(self, width: t.SupportsIndex, fillchar: str = " ", /) -> te.Self:
        return self.__class__(super().center(width, self.escape(fillchar)))

    def strip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().strip(chars))

    def translate(
        self,
        table: cabc.Mapping[int, str | int | None],  # type: ignore[override]
        /,
    ) -> str:
        return self.__class__(super().translate(table))

    def expandtabs(self, /, tabsize: t.SupportsIndex = 8) -> te.Self:
        return self.__class__(super().expandtabs(tabsize))

    def swapcase(self, /) -> te.Self:
        return self.__class__(super().swapcase())

    def zfill(self, width: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().zfill(width))

    def casefold(self, /) -> te.Self:
        return self.__class__(super().casefold())

    def removeprefix(self, prefix: str, /) -> te.Self:
        return self.__class__(super().removeprefix(prefix))

    def removesuffix(self, suffix: str) -> te.Self:
        return self.__class__(super().removesuffix(suffix))

    def partition(self, sep: str, /) -> tuple[te.Self, te.Self, te.Self]:
        left, sep, right = super().partition(sep)
        cls = self.__class__
        return cls(left), cls(sep), cls(right)

    def rpartition(self, sep: str, /) -> tuple[te.Self, te.Self, te.Self]:
        left, sep, right = super().rpartition(sep)
        cls = self.__class__
        return cls(left), cls(sep), cls(right)

    def format(self, *args: t.Any, **kwargs: t.Any) -> te.Self:
        formatter = EscapeFormatter(self.escape)
        return self.__class__(formatter.vformat(self, args, kwargs))

    def format_map(
        self,
        mapping: cabc.Mapping[str, t.Any],  # type: ignore[override]
        /,
    ) -> te.Self:
        formatter = EscapeFormatter(self.escape)
        return self.__class__(formatter.vformat(self, (), mapping))

    def __html_format__(self, format_spec: str, /) -> te.Self:
        if format_spec:
            raise ValueError("Unsupported format specification for Markup.")

        return self


class EscapeFormatter(string.Formatter):
    __slots__ = ("escape",)

    def __init__(self, escape: _TPEscape) -> None:
        self.escape: _TPEscape = escape
        super().__init__()

    def format_field(self, value: t.Any, format_spec: str) -> str:
        if hasattr(value, "__html_format__"):
            rv = value.__html_format__(format_spec)
        elif hasattr(value, "__html__"):
            if format_spec:
                raise ValueError(
                    f"Format specifier {format_spec} given, but {type(value)} does not"
                    " define __html_format__. A class that defines __html__ must define"
                    " __html_format__ to work with format specifiers."
                )
            rv = value.__html__()
        else:
            # We need to make sure the format spec is str here as
            # otherwise the wrong callback methods are invoked.
            rv = super().format_field(value, str(format_spec))
        return str(self.escape(rv))


class _MarkupEscapeHelper:
    """Helper for :meth:`Markup.__mod__`."""

    __slots__ = ("obj", "escape")

    def __init__(self, obj: t.Any, escape: _TPEscape) -> None:
        self.obj: t.Any = obj
        self.escape: _TPEscape = escape

    def __getitem__(self, key: t.Any, /) -> te.Self:
        return self.__class__(self.obj[key], self.escape)

    def __str__(self, /) -> str:
        return str(self.escape(self.obj))

    def __repr__(self, /) -> str:
        return str(self.escape(repr(self.obj)))

    def __int__(self, /) -> int:
        return int(self.obj)

    def __float__(self, /) -> float:
        return float(self.obj)


def __getattr__(name: str) -> t.Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " MarkupSafe 3.1. Use feature detection, or"
            ' `importlib.metadata.version("markupsafe")`, instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.metadata.version("markupsafe")

    raise AttributeError(name)
<|endoftext|>from __future__ import annotations

import typing as t
from datetime import datetime


class BadData(Exception):
    """Raised if bad data of any sort was encountered. This is the base
    for all exceptions that ItsDangerous defines.

    .. versionadded:: 0.15
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class BadSignature(BadData):
    """Raised if a signature does not match."""

    def __init__(self, message: str, payload: t.Any | None = None):
        super().__init__(message)

        #: The payload that failed the signature test. In some
        #: situations you might still want to inspect this, even if
        #: you know it was tampered with.
        #:
        #: .. versionadded:: 0.14
        self.payload: t.Any | None = payload


class BadTimeSignature(BadSignature):
    """Raised if a time-based signature is invalid. This is a subclass
    of :class:`BadSignature`.
    """

    def __init__(
        self,
        message: str,
        payload: t.Any | None = None,
        date_signed: datetime | None = None,
    ):
        super().__init__(message, payload)

        #: If the signature expired this exposes the date of when the
        #: signature was created. This can be helpful in order to
        #: tell the user how long a link has been gone stale.
        #:
        #: .. versionchanged:: 2.0
        #:     The datetime value is timezone-aware rather than naive.
        #:
        #: .. versionadded:: 0.14
        self.date_signed = date_signed


class SignatureExpired(BadTimeSignature):
    """Raised if a signature timestamp is older than ``max_age``. This
    is a subclass of :exc:`BadTimeSignature`.
    """


class BadHeader(BadSignature):
    """Raised if a signed header is invalid in some form. This only
    happens for serializers that have a header that goes with the
    signature.

    .. versionadded:: 0.24
    """

    def __init__(
        self,
        message: str,
        payload: t.Any | None = None,
        header: t.Any | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, payload)

        #: If the header is actually available but just malformed it
        #: might be stored here.
        self.header: t.Any | None = header

        #: If available, the error that indicates why the payload was
        #: not valid. This might be ``None``.
        self.original_error: Exception | None = original_error


class BadPayload(BadData):
    """Raised if a payload is invalid. This could happen if the payload
    is loaded despite an invalid signature, or if there is a mismatch
    between the serializer and deserializer. The original exception
    that occurred during loading is stored on as :attr:`original_error`.

    .. versionadded:: 0.15
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)

        #: If available, the error that indicates why the payload was
        #: not valid. This might be ``None``.
        self.original_error: Exception | None = original_error
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import hashlib
import hmac
import typing as t

from .encoding import _base64_alphabet
from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import want_bytes
from .exc import BadSignature


class SigningAlgorithm:
    """Subclasses must implement :meth:`get_signature` to provide
    signature generation functionality.
    """

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        """Returns the signature for the given key and value."""
        raise NotImplementedError()

    def verify_signature(self, key: bytes, value: bytes, sig: bytes) -> bool:
        """Verifies the given signature matches the expected
        signature.
        """
        return hmac.compare_digest(sig, self.get_signature(key, value))


class NoneAlgorithm(SigningAlgorithm):
    """Provides an algorithm that does not perform any signing and
    returns an empty signature.
    """

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        return b""


def _lazy_sha1(string: bytes = b"") -> t.Any:
    """Don't access ``hashlib.sha1`` until runtime. FIPS builds may not include
    SHA-1, in which case the import and use as a default would fail before the
    developer can configure something else.
    """
    return hashlib.sha1(string)


class HMACAlgorithm(SigningAlgorithm):
    """Provides signature generation using HMACs."""

    #: The digest method to use with the MAC algorithm. This defaults to
    #: SHA1, but can be changed to any other function in the hashlib
    #: module.
    default_digest_method: t.Any = staticmethod(_lazy_sha1)

    def __init__(self, digest_method: t.Any = None):
        if digest_method is None:
            digest_method = self.default_digest_method

        self.digest_method: t.Any = digest_method

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        mac = hmac.new(key, msg=value, digestmod=self.digest_method)
        return mac.digest()


def _make_keys_list(
    secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
) -> list[bytes]:
    if isinstance(secret_key, (str, bytes)):
        return [want_bytes(secret_key)]

    return [want_bytes(s) for s in secret_key]  # pyright: ignore


class Signer:
    """A signer securely signs bytes, then unsigns them to verify that
    the value hasn't been changed.

    The secret key should be a random string of ``bytes`` and should not
    be saved to code or version control. Different salts should be used
    to distinguish signing in different contexts. See :doc:`/concepts`
    for information about the security of the secret key and salt.

    :param secret_key: The secret key to sign and verify with. Can be a
        list of keys, oldest to newest, to support key rotation.
    :param salt: Extra key to combine with ``secret_key`` to distinguish
        signatures in different contexts.
    :param sep: Separator between the signature and value.
    :param key_derivation: How to derive the signing key from the secret
        key and salt. Possible values are ``concat``, ``django-concat``,
        or ``hmac``. Defaults to :attr:`default_key_derivation`, which
        defaults to ``django-concat``.
    :param digest_method: Hash function to use when generating the HMAC
        signature. Defaults to :attr:`default_digest_method`, which
        defaults to :func:`hashlib.sha1`. Note that the security of the
        hash alone doesn't apply when used intermediately in HMAC.
    :param algorithm: A :class:`SigningAlgorithm` instance to use
        instead of building a default :class:`HMACAlgorithm` with the
        ``digest_method``.

    .. versionchanged:: 2.0
        Added support for key rotation by passing a list to
        ``secret_key``.

    .. versionchanged:: 0.18
        ``algorithm`` was added as an argument to the class constructor.

    .. versionchanged:: 0.14
        ``key_derivation`` and ``digest_method`` were added as arguments
        to the class constructor.
    """

    #: The default digest method to use for the signer. The default is
    #: :func:`hashlib.sha1`, but can be changed to any :mod:`hashlib` or
    #: compatible object. Note that the security of the hash alone
    #: doesn't apply when used intermediately in HMAC.
    #:
    #: .. versionadded:: 0.14
    default_digest_method: t.Any = staticmethod(_lazy_sha1)

    #: The default scheme to use to derive the signing key from the
    #: secret key and salt. The default is ``django-concat``. Possible
    #: values are ``concat``, ``django-concat``, and ``hmac``.
    #:
    #: .. versionadded:: 0.14
    default_key_derivation: str = "django-concat"

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous.Signer",
        sep: str | bytes = b".",
        key_derivation: str | None = None,
        digest_method: t.Any | None = None,
        algorithm: SigningAlgorithm | None = None,
    ):
        #: The list of secret keys to try for verifying signatures, from
        #: oldest to newest. The newest (last) key is used for signing.
        #:
        #: This allows a key rotation system to keep a list of allowed
        #: keys and remove expired ones.
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)
        self.sep: bytes = want_bytes(sep)

        if self.sep in _base64_alphabet:
            raise ValueError(
                "The given separator cannot be used because it may be"
                " contained in the signature itself. ASCII letters,"
                " digits, and '-_=' must not be used."
            )

        if salt is not None:
            salt = want_bytes(salt)
        else:
            salt = b"itsdangerous.Signer"

        self.salt = salt

        if key_derivation is None:
            key_derivation = self.default_key_derivation

        self.key_derivation: str = key_derivation

        if digest_method is None:
            digest_method = self.default_digest_method

        self.digest_method: t.Any = digest_method

        if algorithm is None:
            algorithm = HMACAlgorithm(self.digest_method)

        self.algorithm: SigningAlgorithm = algorithm

    @property
    def secret_key(self) -> bytes:
        """The newest (last) entry in the :attr:`secret_keys` list. This
        is for compatibility from before key rotation support was added.
        """
        return self.secret_keys[-1]

    def derive_key(self, secret_key: str | bytes | None = None) -> bytes:
        """This method is called to derive the key. The default key
        derivation choices can be overridden here. Key derivation is not
        intended to be used as a security method to make a complex key
        out of a short password. Instead you should use large random
        secret keys.

        :param secret_key: A specific secret key to derive from.
            Defaults to the last item in :attr:`secret_keys`.

        .. versionchanged:: 2.0
            Added the ``secret_key`` parameter.
        """
        if secret_key is None:
            secret_key = self.secret_keys[-1]
        else:
            secret_key = want_bytes(secret_key)

        if self.key_derivation == "concat":
            return t.cast(bytes, self.digest_method(self.salt + secret_key).digest())
        elif self.key_derivation == "django-concat":
            return t.cast(
                bytes, self.digest_method(self.salt + b"signer" + secret_key).digest()
            )
        elif self.key_derivation == "hmac":
            mac = hmac.new(secret_key, digestmod=self.digest_method)
            mac.update(self.salt)
            return mac.digest()
        elif self.key_derivation == "none":
            return secret_key
        else:
            raise TypeError("Unknown key derivation method")

    def get_signature(self, value: str | bytes) -> bytes:
        """Returns the signature for the given value."""
        value = want_bytes(value)
        key = self.derive_key()
        sig = self.algorithm.get_signature(key, value)
        return base64_encode(sig)

    def sign(self, value: str | bytes) -> bytes:
        """Signs the given string."""
        value = want_bytes(value)
        return value + self.sep + self.get_signature(value)

    def verify_signature(self, value: str | bytes, sig: str | bytes) -> bool:
        """Verifies the signature for the given value."""
        try:
            sig = base64_decode(sig)
        except Exception:
            return False

        value = want_bytes(value)

        for secret_key in reversed(self.secret_keys):
            key = self.derive_key(secret_key)

            if self.algorithm.verify_signature(key, value, sig):
                return True

        return False

    def unsign(self, signed_value: str | bytes) -> bytes:
        """Unsigns the given string."""
        signed_value = want_bytes(signed_value)

        if self.sep not in signed_value:
            raise BadSignature(f"No {self.sep!r} found in value")

        value, sig = signed_value.rsplit(self.sep, 1)

        if self.verify_signature(value, sig):
            return value

        raise BadSignature(f"Signature {sig!r} does not match", payload=value)

    def validate(self, signed_value: str | bytes) -> bool:
        """Only validates the given signed value. Returns ``True`` if
        the signature exists and is valid.
        """
        try:
            self.unsign(signed_value)
            return True
        except BadSignature:
            return False
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import time
import typing as t
from datetime import datetime
from datetime import timezone

from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import bytes_to_int
from .encoding import int_to_bytes
from .encoding import want_bytes
from .exc import BadSignature
from .exc import BadTimeSignature
from .exc import SignatureExpired
from .serializer import _TSerialized
from .serializer import Serializer
from .signer import Signer


class TimestampSigner(Signer):
    """Works like the regular :class:`.Signer` but also records the time
    of the signing and can be used to expire signatures. The
    :meth:`unsign` method can raise :exc:`.SignatureExpired` if the
    unsigning failed because the signature is expired.
    """

    def get_timestamp(self) -> int:
        """Returns the current timestamp. The function must return an
        integer.
        """
        return int(time.time())

    def timestamp_to_datetime(self, ts: int) -> datetime:
        """Convert the timestamp from :meth:`get_timestamp` into an
        aware :class`datetime.datetime` in UTC.

        .. versionchanged:: 2.0
            The timestamp is returned as a timezone-aware ``datetime``
            in UTC rather than a naive ``datetime`` assumed to be UTC.
        """
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def sign(self, value: str | bytes) -> bytes:
        """Signs the given string and also attaches time information."""
        value = want_bytes(value)
        timestamp = base64_encode(int_to_bytes(self.get_timestamp()))
        sep = want_bytes(self.sep)
        value = value + sep + timestamp
        return value + sep + self.get_signature(value)

    # Ignore overlapping signatures check, return_timestamp is the only
    # parameter that affects the return type.

    @t.overload
    def unsign(  # type: ignore[overload-overlap]
        self,
        signed_value: str | bytes,
        max_age: int | None = None,
        return_timestamp: t.Literal[False] = False,
    ) -> bytes: ...

    @t.overload
    def unsign(
        self,
        signed_value: str | bytes,
        max_age: int | None = None,
        return_timestamp: t.Literal[True] = True,
    ) -> tuple[bytes, datetime]: ...

    def unsign(
        self,
        signed_value: str | bytes,
        max_age: int | None = None,
        return_timestamp: bool = False,
    ) -> tuple[bytes, datetime] | bytes:
        """Works like the regular :meth:`.Signer.unsign` but can also
        validate the time. See the base docstring of the class for
        the general behavior. If ``return_timestamp`` is ``True`` the
        timestamp of the signature will be returned as an aware
        :class:`datetime.datetime` object in UTC.

        .. versionchanged:: 2.0
            The timestamp is returned as a timezone-aware ``datetime``
            in UTC rather than a naive ``datetime`` assumed to be UTC.
        """
        try:
            result = super().unsign(signed_value)
            sig_error = None
        except BadSignature as e:
            sig_error = e
            result = e.payload or b""

        sep = want_bytes(self.sep)

        # If there is no timestamp in the result there is something
        # seriously wrong. In case there was a signature error, we raise
        # that one directly, otherwise we have a weird situation in
        # which we shouldn't have come except someone uses a time-based
        # serializer on non-timestamp data, so catch that.
        if sep not in result:
            if sig_error:
                raise sig_error

            raise BadTimeSignature("timestamp missing", payload=result)

        value, ts_bytes = result.rsplit(sep, 1)
        ts_int: int | None = None
        ts_dt: datetime | None = None

        try:
            ts_int = bytes_to_int(base64_decode(ts_bytes))
        except Exception:
            pass

        # Signature is *not* okay. Raise a proper error now that we have
        # split the value and the timestamp.
        if sig_error is not None:
            if ts_int is not None:
                try:
                    ts_dt = self.timestamp_to_datetime(ts_int)
                except (ValueError, OSError, OverflowError) as exc:
                    # Windows raises OSError
                    # 32-bit raises OverflowError
                    raise BadTimeSignature(
                        "Malformed timestamp", payload=value
                    ) from exc

            raise BadTimeSignature(str(sig_error), payload=value, date_signed=ts_dt)

        # Signature was okay but the timestamp is actually not there or
        # malformed. Should not happen, but we handle it anyway.
        if ts_int is None:
            raise BadTimeSignature("Malformed timestamp", payload=value)

        # Check timestamp is not older than max_age
        if max_age is not None:
            age = self.get_timestamp() - ts_int

            if age > max_age:
                raise SignatureExpired(
                    f"Signature age {age} > {max_age} seconds",
                    payload=value,
                    date_signed=self.timestamp_to_datetime(ts_int),
                )

            if age < 0:
                raise SignatureExpired(
                    f"Signature age {age} < 0 seconds",
                    payload=value,
                    date_signed=self.timestamp_to_datetime(ts_int),
                )

        if return_timestamp:
            return value, self.timestamp_to_datetime(ts_int)

        return value

    def validate(self, signed_value: str | bytes, max_age: int | None = None) -> bool:
        """Only validates the given signed value. Returns ``True`` if
        the signature exists and is valid."""
        try:
            self.unsign(signed_value, max_age=max_age)
            return True
        except BadSignature:
            return False


class TimedSerializer(Serializer[_TSerialized]):
    """Uses :class:`TimestampSigner` instead of the default
    :class:`.Signer`.
    """

    default_signer: type[TimestampSigner] = TimestampSigner

    def iter_unsigners(
        self, salt: str | bytes | None = None
    ) -> cabc.Iterator[TimestampSigner]:
        return t.cast("cabc.Iterator[TimestampSigner]", super().iter_unsigners(salt))

    # TODO: Signature is incompatible because parameters were added
    #  before salt.

    def loads(  # type: ignore[override]
        self,
        s: str | bytes,
        max_age: int | None = None,
        return_timestamp: bool = False,
        salt: str | bytes | None = None,
    ) -> t.Any:
        """Reverse of :meth:`dumps`, raises :exc:`.BadSignature` if the
        signature validation fails. If a ``max_age`` is provided it will
        ensure the signature is not older than that time in seconds. In
        case the signature is outdated, :exc:`.SignatureExpired` is
        raised. All arguments are forwarded to the signer's
        :meth:`~TimestampSigner.unsign` method.
        """
        s = want_bytes(s)
        last_exception = None

        for signer in self.iter_unsigners(salt):
            try:
                base64d, timestamp = signer.unsign(
                    s, max_age=max_age, return_timestamp=True
                )
                payload = self.load_payload(base64d)

                if return_timestamp:
                    return payload, timestamp

                return payload
            except SignatureExpired:
                # The signature was unsigned successfully but was
                # expired. Do not try the next signer.
                raise
            except BadSignature as err:
                last_exception = err

        raise t.cast(BadSignature, last_exception)

    def loads_unsafe(  # type: ignore[override]
        self,
        s: str | bytes,
        max_age: int | None = None,
        salt: str | bytes | None = None,
    ) -> tuple[bool, t.Any]:
        return self._loads_unsafe_impl(s, salt, load_kwargs={"max_age": max_age})
<|endoftext|>from __future__ import annotations

import base64
import string
import struct
import typing as t

from .exc import BadData


def want_bytes(
    s: str | bytes, encoding: str = "utf-8", errors: str = "strict"
) -> bytes:
    if isinstance(s, str):
        s = s.encode(encoding, errors)

    return s


def base64_encode(string: str | bytes) -> bytes:
    """Base64 encode a string of bytes or text. The resulting bytes are
    safe to use in URLs.
    """
    string = want_bytes(string)
    return base64.urlsafe_b64encode(string).rstrip(b"=")


def base64_decode(string: str | bytes) -> bytes:
    """Base64 decode a URL-safe string of bytes or text. The result is
    bytes.
    """
    string = want_bytes(string, encoding="ascii", errors="ignore")
    string += b"=" * (-len(string) % 4)

    try:
        return base64.urlsafe_b64decode(string)
    except (TypeError, ValueError) as e:
        raise BadData("Invalid base64-encoded data") from e


# The alphabet used by base64.urlsafe_*
_base64_alphabet = f"{string.ascii_letters}{string.digits}-_=".encode("ascii")

_int64_struct = struct.Struct(">Q")
_int_to_bytes = _int64_struct.pack
_bytes_to_int = t.cast("t.Callable[[bytes], tuple[int]]", _int64_struct.unpack)


def int_to_bytes(num: int) -> bytes:
    return _int_to_bytes(num).lstrip(b"\x00")


def bytes_to_int(bytestr: bytes) -> int:
    return _bytes_to_int(bytestr.rjust(8, b"\x00"))[0]
<|endoftext|>from __future__ import annotations

import typing as t
import zlib

from ._json import _CompactJSON
from .encoding import base64_decode
from .encoding import base64_encode
from .exc import BadPayload
from .serializer import _PDataSerializer
from .serializer import Serializer
from .timed import TimedSerializer


class URLSafeSerializerMixin(Serializer[str]):
    """Mixed in with a regular serializer it will attempt to zlib
    compress the string to make it shorter if necessary. It will also
    base64 encode the string so that it can safely be placed in a URL.
    """

    default_serializer: _PDataSerializer[str] = _CompactJSON

    def load_payload(
        self,
        payload: bytes,
        *args: t.Any,
        serializer: t.Any | None = None,
        **kwargs: t.Any,
    ) -> t.Any:
        decompress = False

        if payload.startswith(b"."):
            payload = payload[1:]
            decompress = True

        try:
            json = base64_decode(payload)
        except Exception as e:
            raise BadPayload(
                "Could not base64 decode the payload because of an exception",
                original_error=e,
            ) from e

        if decompress:
            try:
                json = zlib.decompress(json)
            except Exception as e:
                raise BadPayload(
                    "Could not zlib decompress the payload before decoding the payload",
                    original_error=e,
                ) from e

        return super().load_payload(json, *args, **kwargs)

    def dump_payload(self, obj: t.Any) -> bytes:
        json = super().dump_payload(obj)
        is_compressed = False
        compressed = zlib.compress(json)

        if len(compressed) < (len(json) - 1):
            json = compressed
            is_compressed = True

        base64d = base64_encode(json)

        if is_compressed:
            base64d = b"." + base64d

        return base64d


class URLSafeSerializer(URLSafeSerializerMixin, Serializer[str]):
    """Works like :class:`.Serializer` but dumps and loads into a URL
    safe string consisting of the upper and lowercase character of the
    alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """


class URLSafeTimedSerializer(URLSafeSerializerMixin, TimedSerializer[str]):
    """Works like :class:`.TimedSerializer` but dumps and loads into a
    URL safe string consisting of the upper and lowercase character of
    the alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """
<|endoftext|>from __future__ import annotations

import typing as t

from .encoding import base64_decode as base64_decode
from .encoding import base64_encode as base64_encode
from .encoding import want_bytes as want_bytes
from .exc import BadData as BadData
from .exc import BadHeader as BadHeader
from .exc import BadPayload as BadPayload
from .exc import BadSignature as BadSignature
from .exc import BadTimeSignature as BadTimeSignature
from .exc import SignatureExpired as SignatureExpired
from .serializer import Serializer as Serializer
from .signer import HMACAlgorithm as HMACAlgorithm
from .signer import NoneAlgorithm as NoneAlgorithm
from .signer import Signer as Signer
from .timed import TimedSerializer as TimedSerializer
from .timed import TimestampSigner as TimestampSigner
from .url_safe import URLSafeSerializer as URLSafeSerializer
from .url_safe import URLSafeTimedSerializer as URLSafeTimedSerializer


def __getattr__(name: str) -> t.Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " ItsDangerous 2.3. Use feature detection or"
            " 'importlib.metadata.version(\"itsdangerous\")' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.metadata.version("itsdangerous")

    raise AttributeError(name)
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import json
import typing as t

from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer

if t.TYPE_CHECKING:
    import typing_extensions as te

    # This should be either be str or bytes. To avoid having to specify the
    # bound type, it falls back to a union if structural matching fails.
    _TSerialized = te.TypeVar(
        "_TSerialized", bound=t.Union[str, bytes], default=t.Union[str, bytes]
    )
else:
    # Still available at runtime on Python < 3.13, but without the default.
    _TSerialized = t.TypeVar("_TSerialized", bound=t.Union[str, bytes])


class _PDataSerializer(t.Protocol[_TSerialized]):
    def loads(self, payload: _TSerialized, /) -> t.Any: ...
    # A signature with additional arguments is not handled correctly by type
    # checkers right now, so an overload is used below for serializers that
    # don't match this strict protocol.
    def dumps(self, obj: t.Any, /) -> _TSerialized: ...


# Use TypeIs once it's available in typing_extensions or 3.13.
def is_text_serializer(
    serializer: _PDataSerializer[t.Any],
) -> te.TypeGuard[_PDataSerializer[str]]:
    """Checks whether a serializer generates text or binary."""
    return isinstance(serializer.dumps({}), str)


class Serializer(t.Generic[_TSerialized]):
    """A serializer wraps a :class:`~itsdangerous.signer.Signer` to
    enable serializing and securely signing data other than bytes. It
    can unsign to verify that the data hasn't been changed.

    The serializer provides :meth:`dumps` and :meth:`loads`, similar to
    :mod:`json`, and by default uses :mod:`json` internally to serialize
    the data to bytes.

    The secret key should be a random string of ``bytes`` and should not
    be saved to code or version control. Different salts should be used
    to distinguish signing in different contexts. See :doc:`/concepts`
    for information about the security of the secret key and salt.

    :param secret_key: The secret key to sign and verify with. Can be a
        list of keys, oldest to newest, to support key rotation.
    :param salt: Extra key to combine with ``secret_key`` to distinguish
        signatures in different contexts.
    :param serializer: An object that provides ``dumps`` and ``loads``
        methods for serializing data to a string. Defaults to
        :attr:`default_serializer`, which defaults to :mod:`json`.
    :param serializer_kwargs: Keyword arguments to pass when calling
        ``serializer.dumps``.
    :param signer: A ``Signer`` class to instantiate when signing data.
        Defaults to :attr:`default_signer`, which defaults to
        :class:`~itsdangerous.signer.Signer`.
    :param signer_kwargs: Keyword arguments to pass when instantiating
        the ``Signer`` class.
    :param fallback_signers: List of signer parameters to try when
        unsigning with the default signer fails. Each item can be a dict
        of ``signer_kwargs``, a ``Signer`` class, or a tuple of
        ``(signer, signer_kwargs)``. Defaults to
        :attr:`default_fallback_signers`.

    .. versionchanged:: 2.0
        Added support for key rotation by passing a list to
        ``secret_key``.

    .. versionchanged:: 2.0
        Removed the default SHA-512 fallback signer from
        ``default_fallback_signers``.

    .. versionchanged:: 1.1
        Added support for ``fallback_signers`` and configured a default
        SHA-512 fallback. This fallback is for users who used the yanked
        1.0.0 release which defaulted to SHA-512.

    .. versionchanged:: 0.14
        The ``signer`` and ``signer_kwargs`` parameters were added to
        the constructor.
    """

    #: The default serialization module to use to serialize data to a
    #: string internally. The default is :mod:`json`, but can be changed
    #: to any object that provides ``dumps`` and ``loads`` methods.
    default_serializer: _PDataSerializer[t.Any] = json

    #: The default ``Signer`` class to instantiate when signing data.
    #: The default is :class:`itsdangerous.signer.Signer`.
    default_signer: type[Signer] = Signer

    #: The default fallback signers to try when unsigning fails.
    default_fallback_signers: list[
        dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
    ] = []

    # Serializer[str] if no data serializer is provided, or if it returns str.
    @t.overload
    def __init__(
        self: Serializer[str],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: None | _PDataSerializer[str] = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Serializer[bytes] with a bytes data serializer positional argument.
    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Serializer[bytes] with a bytes data serializer keyword argument.
    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Fall back with a positional argument. If the strict signature of
    # _PDataSerializer doesn't match, fall back to a union, requiring the user
    # to specify the type.
    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Fall back with a keyword argument.
    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: t.Any | None = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ):
        #: The list of secret keys to try for verifying signatures, from
        #: oldest to newest. The newest (last) key is used for signing.
        #:
        #: This allows a key rotation system to keep a list of allowed
        #: keys and remove expired ones.
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)

        if salt is not None:
            salt = want_bytes(salt)
            # if salt is None then the signer's default is used

        self.salt = salt

        if serializer is None:
            serializer = self.default_serializer

        self.serializer: _PDataSerializer[_TSerialized] = serializer
        self.is_text_serializer: bool = is_text_serializer(serializer)

        if signer is None:
            signer = self.default_signer

        self.signer: type[Signer] = signer
        self.signer_kwargs: dict[str, t.Any] = signer_kwargs or {}

        if fallback_signers is None:
            fallback_signers = list(self.default_fallback_signers)

        self.fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ] = fallback_signers
        self.serializer_kwargs: dict[str, t.Any] = serializer_kwargs or {}

    @property
    def secret_key(self) -> bytes:
        """The newest (last) entry in the :attr:`secret_keys` list. This
        is for compatibility from before key rotation support was added.
        """
        return self.secret_keys[-1]

    def load_payload(
        self, payload: bytes, serializer: _PDataSerializer[t.Any] | None = None
    ) -> t.Any:
        """Loads the encoded object. This function raises
        :class:`.BadPayload` if the payload is not valid. The
        ``serializer`` parameter can be used to override the serializer
        stored on the class. The encoded ``payload`` should always be
        bytes.
        """
        if serializer is None:
            use_serializer = self.serializer
            is_text = self.is_text_serializer
        else:
            use_serializer = serializer
            is_text = is_text_serializer(serializer)

        try:
            if is_text:
                return use_serializer.loads(payload.decode("utf-8"))  # type: ignore[arg-type]

            return use_serializer.loads(payload)  # type: ignore[arg-type]
        except Exception as e:
            raise BadPayload(
                "Could not load the payload because an exception"
                " occurred on unserializing the data.",
                original_error=e,
            ) from e

    def dump_payload(self, obj: t.Any) -> bytes:
        """Dumps the encoded object. The return value is always bytes.
        If the internal serializer returns text, the value will be
        encoded as UTF-8.
        """
        return want_bytes(self.serializer.dumps(obj, **self.serializer_kwargs))

    def make_signer(self, salt: str | bytes | None = None) -> Signer:
        """Creates a new instance of the signer to be used. The default
        implementation uses the :class:`.Signer` base class.
        """
        if salt is None:
            salt = self.salt

        return self.signer(self.secret_keys, salt=salt, **self.signer_kwargs)

    def iter_unsigners(self, salt: str | bytes | None = None) -> cabc.Iterator[Signer]:
        """Iterates over all signers to be tried for unsigning. Starts
        with the configured signer, then constructs each signer
        specified in ``fallback_signers``.
        """
        if salt is None:
            salt = self.salt

        yield self.make_signer(salt)

        for fallback in self.fallback_signers:
            if isinstance(fallback, dict):
                kwargs = fallback
                fallback = self.signer
            elif isinstance(fallback, tuple):
                fallback, kwargs = fallback
            else:
                kwargs = self.signer_kwargs

            for secret_key in self.secret_keys:
                yield fallback(secret_key, salt=salt, **kwargs)

    def dumps(self, obj: t.Any, salt: str | bytes | None = None) -> _TSerialized:
        """Returns a signed string serialized with the internal
        serializer. The return value can be either a byte or unicode
        string depending on the format of the internal serializer.
        """
        payload = want_bytes(self.dump_payload(obj))
        rv = self.make_signer(salt).sign(payload)

        if self.is_text_serializer:
            return rv.decode("utf-8")  # type: ignore[return-value]

        return rv  # type: ignore[return-value]

    def dump(self, obj: t.Any, f: t.IO[t.Any], salt: str | bytes | None = None) -> None:
        """Like :meth:`dumps` but dumps into a file. The file handle has
        to be compatible with what the internal serializer expects.
        """
        f.write(self.dumps(obj, salt))

    def loads(
        self, s: str | bytes, salt: str | bytes | None = None, **kwargs: t.Any
    ) -> t.Any:
        """Reverse of :meth:`dumps`. Raises :exc:`.BadSignature` if the
        signature validation fails.
        """
        s = want_bytes(s)
        last_exception = None

        for signer in self.iter_unsigners(salt):
            try:
                return self.load_payload(signer.unsign(s))
            except BadSignature as err:
                last_exception = err

        raise t.cast(BadSignature, last_exception)

    def load(self, f: t.IO[t.Any], salt: str | bytes | None = None) -> t.Any:
        """Like :meth:`loads` but loads from a file."""
        return self.loads(f.read(), salt)

    def loads_unsafe(
        self, s: str | bytes, salt: str | bytes | None = None
    ) -> tuple[bool, t.Any]:
        """Like :meth:`loads` but without verifying the signature. This
        is potentially very dangerous to use depending on how your
        serializer works. The return value is ``(signature_valid,
        payload)`` instead of just the payload. The first item will be a
        boolean that indicates if the signature is valid. This function
        never fails.

        Use it for debugging only and if you know that your serializer
        module is not exploitable (for example, do not use it with a
        pickle serializer).

        .. versionadded:: 0.15
        """
        return self._loads_unsafe_impl(s, salt)

    def _loads_unsafe_impl(
        self,
        s: str | bytes,
        salt: str | bytes | None,
        load_kwargs: dict[str, t.Any] | None = None,
        load_payload_kwargs: dict[str, t.Any] | None = None,
    ) -> tuple[bool, t.Any]:
        """Low level helper function to implement :meth:`loads_unsafe`
        in serializer subclasses.
        """
        if load_kwargs is None:
            load_kwargs = {}

        try:
            return True, self.loads(s, salt=salt, **load_kwargs)
        except BadSignature as e:
            if e.payload is None:
                return False, None

            if load_payload_kwargs is None:
                load_payload_kwargs = {}

            try:
                return (
                    False,
                    self.load_payload(e.payload, **load_payload_kwargs),
                )
            except BadPayload:
                return False, None

    def load_unsafe(
        self, f: t.IO[t.Any], salt: str | bytes | None = None
    ) -> tuple[bool, t.Any]:
        """Like :meth:`loads_unsafe` but loads from a file.

        .. versionadded:: 0.15
        """
        return self.loads_unsafe(f.read(), salt=salt)
<|endoftext|>from __future__ import annotations

import json as _json
import typing as t


class _CompactJSON:
    """Wrapper around json module that strips whitespace."""

    @staticmethod
    def loads(payload: str | bytes) -> t.Any:
        return _json.loads(payload)

    @staticmethod
    def dumps(obj: t.Any, **kwargs: t.Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("separators", (",", ":"))
        return _json.dumps(obj, **kwargs)
<|endoftext|>import functools
import math
import os
import statistics
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from . import runtime


def nvsmi(attrs):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


# pure Python implementation of np.quantile/torch.quantile
# to avoid unnecessary runtime dependency on numpy/torch


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


@contextmanager
def _proton_bench_session():
    import triton.profiler as proton

    with tempfile.TemporaryDirectory(prefix=f"triton-bench-proton-{uuid.uuid4().hex}") as tmpdir:
        session = proton.start(os.path.join(tmpdir, "profile"), context="shadow", data="tree")
        try:
            yield proton, session
        finally:
            if session is not None:
                proton.finalize(session)


def _collect_proton_scope_times(database, prefix):
    scope_times = []

    def kernel_time_ms(node):
        children = node.get("children", [])
        if len(children) == 0:
            return node.get("metrics", {}).get("time (ns)", 0) / 1e6
        return sum(kernel_time_ms(child) for child in children)

    def visit(node):
        name = node.get("frame", {}).get("name", "")
        if name.startswith(prefix):
            time_ms = kernel_time_ms(node)
            if time_ms > 0:
                scope_times.append((name, time_ms))
            return
        for child in node.get("children", []):
            visit(child)

    for node in database:
        visit(node)
    return [time for _, time in sorted(scope_times)]


def do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    import torch
    assert return_mode in ["min", "max", "mean", "median", "all"]

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        # step 1 - we estimate the amount of time the kernel call takes
        # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
        #       but it is probably good enough
        # NOTE: we don't use a graph to estimate the runtime because creating a graph is expensive,
        #       ~300ms on A100, so we default to the same method used in `do_bench` (minus the L2
        #       cache flush).
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        # Rewrite to avoid possible division by 0 issues with fast benchmarks
        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))
        # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
        # host overhead
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                fn()
        torch.cuda.synchronize()
        # measure time and return
        ret = []
        n_retries = 10
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            ret += [start_event.elapsed_time(end_event) / n_repeat]
        return _summarize_statistics(ret, quantiles, return_mode)


def do_bench_cudagraph_proton(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of kernels invoked by the provided function using the Proton profiler and CUDA graphs.
    This function is similar to `do_bench_cudagraph` that avoids CPU overhead by replaying a CUDA graph with multiple iterations of the provided function,
    but it uses the Proton profiler to measure the runtime of each kernel in the graph instead of using CUDA events to measure the total runtime of the graph.
    This allows us to get more fine-grained measurements of the kernel runtimes and to exclude cache flushes from the measurement.
    Note that this function has several constraints compared to `do_bench_cudagraph`:
    - It does not measure GPU operations other than kernels (e.g., memory copies, synchronization, etc.).
    - It supports only the NVIDIA GPU. AMD GPU is a TODO.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    target = runtime.driver.active.get_current_target()
    if target.backend != "cuda":
        raise RuntimeError("do_bench_cudagraph_proton requires the NVIDIA backend because Proton does not reliably "
                           "attribute CUDA graph replay launches to scopes on HIP.")

    import torch

    with torch.cuda.stream(torch.cuda.Stream()):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        with _proton_bench_session() as (proton, session):
            if session is None:
                raise RuntimeError(
                    "Proton profiler session could not be created. Make sure you are running on a supported GPU and "
                    "that the Proton profiler is properly installed.")
            cache = runtime.driver.active.get_empty_cache_for_benchmark()
            g = torch.cuda.CUDAGraph()
            scope_prefix = f"proton.{uuid.uuid4().hex}."
            with torch.cuda.graph(g):
                for i in range(n_repeat):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    runtime.driver.active.clear_cache(cache)
                    with proton.scope(f"{scope_prefix}{i:08d}"):
                        fn()
            torch.cuda.synchronize()
            n_retries = 10
            try:
                for i in range(n_retries):
                    g.replay()
                torch.cuda.synchronize()
            finally:
                proton.deactivate(session, flushing=True)

            times = [t / n_retries for t in _collect_proton_scope_times(proton.data.get(session), scope_prefix)]

        return _summarize_statistics(times, quantiles, return_mode)


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, returns the mean runtime of :code:`fn` as a single float.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms). Controls how long the function is run before timing begins.
    :type warmup: int
    :param rep: Repetition time (in ms). Controls the total duration of the timed runs.
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensors to None before each run,
        to avoid gradient accumulation affecting timing.
    :type grad_to_none: list[torch.Tensor], optional
    :param quantiles: If provided, return these quantiles of the runtime distribution instead
        of a summary statistic. For example, ``[0.2, 0.5, 0.8]`` returns the 20th, 50th, and
        80th percentile runtimes in ms. When set, ``return_mode`` is ignored.
    :type quantiles: list[float], optional
    :param return_mode: The summary statistic to return when ``quantiles`` is not set.
        ``"mean"`` and ``"median"`` return a single float. ``"min"`` and ``"max"`` return
        the fastest and slowest run respectively. ``"all"`` returns the raw list of
        per-run timings in ms. Default is ``"mean"``.
    :type return_mode: str
    :return: A single float by default, a list of floats if ``quantiles`` is provided,
        or a list of all per-run timings if ``return_mode="all"``.
    :rtype: float | list[float]
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_proton(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of kernels invoked by the provided function using the Proton profiler.

    The measured runtime is generally more accurate than `do_bench` for short kernels that are affected by CPU overhead.
    Note that this function has several constraints compared to `do_bench`:
    - It does not measure GPU operations other than kernels (e.g., memory copies, synchronization, etc.).
    - It supports only AMD and NVIDIA GPUs.

    :param fn: Function to benchmark.
    :type fn: Callable
    :param warmup: Warmup time (in ms).
    :type warmup: int
    :param rep: Repetition time (in ms).
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor(s) to `None`.
    :type grad_to_none: torch.Tensor, optional
    :param quantiles: Performance percentiles to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    if estimate_ms == 0:
        n_warmup = 1000
        n_repeat = 1000
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))

    for _ in range(n_warmup):
        fn()
    di.synchronize()

    scope_prefix = f"proton.{uuid.uuid4().hex}."
    with _proton_bench_session() as (proton, session):
        if session is None:
            raise RuntimeError(
                "Proton profiler session could not be created. Make sure you are running on a supported GPU and "
                "that the Proton profiler is properly installed.")
        try:
            for i in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                runtime.driver.active.clear_cache(cache)
                with proton.scope(f"{scope_prefix}{i:08d}"):
                    fn()
            di.synchronize()
        finally:
            proton.deactivate(session, flushing=True)

        times = _collect_proton_scope_times(proton.data.get(session), scope_prefix)

    return _summarize_statistics(times, quantiles, return_mode)


def assert_close(x, y, atol=None, rtol=None, err_msg=''):
    """
    Asserts that two inputs are close within a certain tolerance.

    :param x: The first input.
    :type x: scala, list, numpy.ndarray, or torch.Tensor
    :param y: The second input.
    :type y: scala, list, numpy.ndarray, or torch.Tensor
    :param atol: The absolute tolerance. Default value is 1e-2.
    :type atol: float, optional
    :param rtol: The relative tolerance. Default value is 0.
    :type rtol: float, optional
    :param err_msg: The error message to use if the assertion fails.
    :type err_msg: str
    """
    import numpy as np
    import torch

    # canonicalize arguments to be tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    # absolute tolerance
    if atol is None:
        atol = 1e-2
    atol = atol(x.dtype) if callable(atol) else atol
    # relative tolerance hook
    if rtol is None:
        rtol = 0.
    rtol = rtol(x.dtype) if callable(rtol) else rtol
    # we use numpy instead of pytorch
    # as it seems more memory efficient
    # pytorch tends to oom on large tensors
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    # we handle size==1 case separately as we can
    # provide better error message there
    if x.size > 1 or y.size > 1:
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, equal_nan=True, err_msg=err_msg)
        return
    if not np.allclose(x, y, atol=atol, rtol=rtol):
        raise AssertionError(f'{err_msg} {x} is not close to {y} (atol={atol}, rtol={rtol})')


class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = '',
        ylabel: str = '',
        x_log: bool = False,
        y_log: bool = False,
        styles=None,
    ):
        """
        Constructor.
        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        :param x_names: Name of the arguments that should appear on the x axis of the plot.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[Any]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
        :type args: Dict[str, Any]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
        :param styles: A list of tuples, where each tuple contains two elements: a color and a linestyle.
        :type styles: list[tuple[str, str]]
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        # plot info
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwargs):
        import os

        import matplotlib.pyplot as plt
        import pandas as pd
        y_mean_labels = [f'{x} ({bench.ylabel})' for x in bench.line_names]
        y_min_labels = [f'{x}-min ({bench.ylabel})' for x in bench.line_names]
        y_max_labels = [f'{x}-max ({bench.ylabel})' for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean_labels + y_min_labels + y_max_labels)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwargs)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max

        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for i, (mean_label, min_label, max_label) in enumerate(zip(y_mean_labels, y_min_labels, y_max_labels)):
                y_min, y_max = df[min_label], df[max_label]
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[first_x], df[mean_label], label=mean_label, color=col, ls=sty)
                if not y_min.isnull().all() and not y_max.isnull().all():
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        df = df[x_names + y_mean_labels]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ':')
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path='', return_df=False, **kwargs):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []
        try:
            for bench in benchmarks:
                result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
        finally:
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, "results.html"), "w") as html:
                    html.write("<html><body>\n")
                    for bench in benchmarks[:len(result_dfs)]:
                        html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
                    html.write("</body></html>\n")
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


def get_dram_gbps(device=None):
    ''' return DRAM bandwidth in GB/s '''

    from .runtime import driver
    if device is None:
        device = driver.active.get_device_interface().current_device()
    mem_clock_khz = driver.active.utils.get_device_properties(device)["mem_clock_rate"]  # in kHz
    bus_width = driver.active.utils.get_device_properties(device)["mem_bus_width"]
    bw_gbps = mem_clock_khz * bus_width * 2 / 1e6 / 8  # In GB/s
    return bw_gbps


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512
        elif dtype in [torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


# create decorator that wraps test function into
# a cuda-memcheck system call


def cuda_memcheck(**target_kwargs):

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            import psutil
            ppid_name = psutil.Process(os.getppid()).name()
            run_cuda_memcheck = target_kwargs.items() <= kwargs.items()
            if run_cuda_memcheck and ppid_name != "cuda-memcheck":
                path = os.path.realpath(test_fn.__globals__["__file__"])
                # get path of current file
                env = {"PATH": os.environ["PATH"], "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
                assert 'request' in kwargs, "memcheck'ed test must have a (possibly unused) `request` fixture"
                test_id = kwargs['request'].node.callspec.id
                cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                out = subprocess.run(["cuda-memcheck", "pytest", "-vs", cmd], capture_output=True, env=env)
                assert out.returncode == 0, "cuda-memcheck returned an error: bounds checking failed"
                assert "ERROR SUMMARY: 0 errors" in str(out.stdout)
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215):
    try:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "1"])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}",
        ])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}",
        ])
        cur_sm_clock = nvsmi(["clocks.current.sm"])[0]
        cur_mem_clock = nvsmi(["clocks.current.memory"])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f"GPU SMs must run at {ref_sm_clock} MHz"
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f"GPU SMs must run at {ref_mem_clock} MHz"
        tflops = 1e-6 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 1e-3
        yield tflops, gbps
    finally:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "0"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rgc"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rmc"])


def get_max_simd_tflops(dtype, clock_rate, device=None):
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        if dtype == torch.float32:
            ops_per_sub_core = 32  # 2*16
        elif dtype == torch.float16:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    else:
        if dtype == torch.float32:
            ops_per_sub_core = 32
        elif dtype in [torch.float16, torch.bfloat16]:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops
<|endoftext|>import multiprocessing
import os
import queue
import re
import tempfile
import numpy as np
import sys
import torch
import triton
import triton.language as tl
from triton import knobs
from typing import Optional, Set, Union
from dataclasses import dataclass
import pytest

from numpy.random import RandomState
from triton.runtime.jit import TensorWrapper, reinterpret, type_canonicalisation_dict

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
integral_dtypes = int_dtypes + uint_dtypes
float_dtypes = ['float16', 'float32', 'float64']
float_dtypes_with_bfloat16 = float_dtypes + ['bfloat16']
dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_float8_dtypes = ['float8_e4m3fn', 'float8_e5m2']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']
tma_dtypes = sorted(set(dtypes_with_bfloat16) - {"int64", "uint64", "float64"})


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def get_current_target():
    if is_interpreter():
        return None
    return triton.runtime.driver.active.get_current_target()


def is_cuda():
    target = get_current_target()
    return False if target is None else target.backend == "cuda"


def is_ampere_or_newer():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 8


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]


def is_blackwell_ultra():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 3)


def is_rubin():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 7)


def is_hopper_or_newer():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


def is_sm12x():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 12


def is_hip():
    target = get_current_target()
    return False if target is None else target.backend == "hip"


def is_hip_cdna2():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx90a'


def is_hip_cdna3():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx942'


def is_hip_cdna4():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def is_hip_rdna3():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and 'gfx11' in target.arch


def is_hip_rdna4():
    target = get_current_target()
    # check for gfx120 instead of gfx12, to avoid matching gfx1250
    return target is not None and target.backend == 'hip' and 'gfx120' in target.arch


def is_hip_gfx1250():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and 'gfx1250' in target.arch


def is_hip_cdna3_or_newer():
    return is_hip_cdna3() or is_hip_cdna4()


def is_hip_cdna():
    return is_hip_cdna2() or is_hip_cdna3() or is_hip_cdna4()


def is_hip_rdna():
    return is_hip_rdna3() or is_hip_rdna4()


def get_hip_lds_size():
    return 163840 if is_hip_cdna4() else 65536


def is_xpu():
    target = get_current_target()
    return False if target is None else target.backend == "xpu"


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Workaround. Never return zero so tests of division don't error out.
        return x
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32') & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device, dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def str_to_triton_dtype(x: str) -> tl.dtype:
    return tl.str_to_ty(type_canonicalisation_dict[x], None)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def supports_tma(byval_only=False):
    if is_interpreter():
        return True
    if not is_cuda():
        return False
    cuda_version = knobs.nvidia.ptxas.version
    min_cuda_version = (12, 0) if byval_only else (12, 3)
    cuda_version_tuple = tuple(map(int, cuda_version.split(".")))
    assert len(cuda_version_tuple) == 2, cuda_version_tuple
    return torch.cuda.get_device_capability()[0] >= 9 and cuda_version_tuple >= min_cuda_version


def supports_ws():
    if is_interpreter():
        return True
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def tma_skip_msg(byval_only=False):
    if byval_only:
        return "Requires __grid_constant__ TMA support (NVIDIA Hopper or higher, CUDA 12.0 or higher)"
    else:
        return "Requires advanced TMA support (NVIDIA Hopper or higher, CUDA 12.3 or higher)"


requires_tma = pytest.mark.skipif(not supports_tma(), reason=tma_skip_msg())


def default_alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def unwrap_tensor(t: Union[torch.Tensor, triton.runtime.jit.TensorWrapper]) -> torch.Tensor:
    if isinstance(t, triton.runtime.jit.TensorWrapper):
        return t.base
    return t


@dataclass
class ProcessResult:
    exc: None | BaseException
    driver_stderr_output: str


def _run_in_process_worker(client_fn, q, args, kwargs, env, stderr_file):
    if env is not None:
        os.environ.update(env)

    # Capture driver/runtime writes to stderr that bypass Python's file objects.
    with open(stderr_file, "w+b") as tmp_stderr:
        saved_stderr_fd = os.dup(2)
        os.dup2(tmp_stderr.fileno(), 2)
        exc = None

        try:
            client_fn(*args, **kwargs)
            # Raise any CUDA errors
            torch.cuda.synchronize()
        except Exception as e:
            exc = e
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
            q.put(exc)


def run_in_process(client_fn, args=(), kwargs=None, env=None):
    if kwargs is None:
        kwargs = {}

    ctx = multiprocessing.get_context("forkserver")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmpdir:
        stderr_file = os.path.join(tmpdir, "err.log")
        process = ctx.Process(target=_run_in_process_worker, args=(client_fn, q, args, kwargs, env, stderr_file))
        process.start()
        process.join()
        with open(stderr_file, "r") as f:
            stderr = f.read()
    exc = None
    try:
        exc = q.get(timeout=1)
    except queue.Empty:
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"child process exited with code {process.exitcode} without returning a result") from None
    return ProcessResult(exc, stderr)


def _fresh_knobs_impl(skipped_attr: Optional[Set[str]] = None):
    from triton import knobs

    if skipped_attr is None:
        skipped_attr = set()

    monkeypatch = pytest.MonkeyPatch()

    knobs_map = {
        name: knobset
        for name, knobset in knobs.__dict__.items()
        if isinstance(knobset, knobs.base_knobs) and knobset != knobs.base_knobs and name not in skipped_attr
    }

    # We store which variables we need to unset below in finally because
    # monkeypatch doesn't appear to reset variables that were never set
    # before the monkeypatch.delenv call below.
    env_to_unset = []
    prev_propagate_env = knobs.propagate_env

    def fresh_function():
        nonlocal env_to_unset
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset.copy().reset())
            for knob in knobset.knob_descriptors.values():
                if knob.key in os.environ:
                    monkeypatch.delenv(knob.key, raising=False)
                else:
                    env_to_unset.append(knob.key)
        knobs.propagate_env = True
        return knobs

    def reset_function():
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset)
        # `undo` should be placed before `del os.environ`
        # Otherwise, it may restore environment variables that monkeypatch deleted
        monkeypatch.undo()
        for k in env_to_unset:
            if k in os.environ:
                del os.environ[k]
        knobs.propagate_env = prev_propagate_env

    return fresh_function, reset_function
<|endoftext|>import functools
import os
import inspect
import subprocess
import tempfile

import triton
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from triton.runtime.jit import create_function_from_signature
from triton._C.libtriton import ir

# ===-----------------------------------------------------------------------===#
# filecheck_test
# ===-----------------------------------------------------------------------===#

# Stub target for testing the frontend.
stub_target = GPUTarget("cuda", 100, 32)

triton_dir = os.path.dirname(__file__)
filecheck_path = os.path.join(triton_dir, "FileCheck")


class MatchError(ValueError):

    def __init__(self, message, module_str):
        super().__init__(message)
        self.module_str = module_str

    def __str__(self):
        return f"{super().__str__()}\n{self.module_str}"


def run_filecheck(name, module_str, check_template):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_module = os.path.join(tempdir, "module")
        with open(temp_module, "w") as temp:
            temp.write(module_str)

        temp_expected = os.path.join(tempdir, "expected")
        with open(temp_expected, "w") as temp:
            temp.write(check_template)

        try:
            subprocess.check_output(
                [filecheck_path, temp_expected, "--input-file", temp_module, "--dump-input-context=50"],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            decoded = error.output.decode('unicode_escape')
            raise ValueError(decoded)


def run_parser(kernel_fn, args=(), kwargs={}, target=stub_target):
    if "sanitize_overflow" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["sanitize_overflow"] = False
    backend = make_backend(target)
    binder = create_function_from_signature(
        kernel_fn.signature,
        kernel_fn.params,
        backend,
    )

    bound_args, specialization, options = binder(*args, **kwargs)
    options, signature, constexprs, attrs = kernel_fn._pack_args(backend, kwargs, bound_args, specialization, options)
    source_cls = GluonASTSource if kernel_fn.is_gluon() else ASTSource
    src = source_cls(kernel_fn, signature, constexprs, attrs)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    return module


def run_filecheck_test(kernel_fn):
    assert isinstance(kernel_fn, triton.runtime.JITFunction)
    check_template = inspect.getsource(kernel_fn.fn)
    if check_template is None:
        raise ValueError("kernel function must have a docstring with FileCheck template")
    mlir_module = run_parser(kernel_fn)

    run_filecheck("placeholder", mlir_module.str_nodebug(), check_template)


def filecheck_test(fn):

    @functools.wraps(fn)
    def test_fn():
        run_filecheck_test(fn)

    return test_fn
<|endoftext|>from __future__ import annotations

import functools
import importlib
import os
import re
import subprocess
import sysconfig
import pathlib

from dataclasses import dataclass
from contextlib import contextmanager
from typing import cast, Any, Callable, Generator, Generic, Optional, Protocol, Type, TypeVar, TypedDict, TYPE_CHECKING, Union

from triton._C.libtriton import getenv, getenv_bool  # type: ignore

if TYPE_CHECKING:
    from .runtime.cache import CacheManager, RemoteCacheBackend
    from .runtime.jit import JitFunctionInfo, KernelParam, JITFunction
    from .runtime.autotuner import Config
    from .compiler.compiler import ASTSource, LazyDict, IRSource


class Env:
    pass


env = Env()

propagate_env: bool = True


def setenv(key: str, value: Optional[str]) -> None:
    if not propagate_env:
        return

    if value is not None:
        os.environ[key] = value
    elif key in os.environ:
        del os.environ[key]


def toenv(val: Any) -> Union[None, tuple[Optional[str]]]:
    if val is None:
        return (None, )

    t = type(val)
    if t is bool:
        return ("1" if val else "0", )

    if t is str:
        return (val, )

    if t is int:
        return (str(val), )

    return None


# There's an asymmetry here so that e.g. env_nvidia_tool can be specified with a
# a string but return an NvidiaTool.
SetType = TypeVar("SetType")
GetType = TypeVar("GetType")

_NOTHING = object()


class env_base(Generic[SetType, GetType]):

    def __init__(self, key: str) -> None:
        self.key = key

    def __set_name__(self, objclass: Type[object], name: str) -> None:
        self.name = name

    def __get__(self, obj: Optional[object], objclass: Optional[Type[object]]) -> GetType:
        py_val = obj.__dict__.get(self.name, _NOTHING)
        if py_val is _NOTHING:
            return self.get()
        return self.transform(py_val)

    def get(self) -> GetType:
        raise NotImplementedError()

    def __set__(self, obj: object, value: Union[SetType, Env]) -> None:
        if isinstance(value, Env):
            obj.__dict__.pop(self.name, None)
        else:
            obj.__dict__[self.name] = value
            if env_val := toenv(value):
                setenv(self.key, env_val[0])

    def __delete__(self, obj: object) -> None:
        obj.__dict__.pop(self.name, None)

    def transform(self, val: SetType) -> GetType:
        # See comment about GetType/SetType in their definition above. Only needed
        # if GetType != SetType.
        return cast(GetType, val)


class env_str(env_base[str, str]):

    def __init__(self, key: str, default: str):
        super().__init__(key)
        self.default = default

    def get(self) -> str:
        return getenv(self.key, self.default)


class env_str_callable_default(env_base[str, str]):

    def __init__(self, key: str, default_factory: Callable[[], str]):
        super().__init__(key)
        self.default_factory = default_factory

    def get(self) -> str:
        env_val = getenv(self.key)
        if env_val is None:
            return self.default_factory()
        return env_val


class env_bool(env_base[bool, bool]):

    def __init__(self, key: str, default: bool = False) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> bool:
        return getenv_bool(self.key, self.default)


class env_int(env_base[int, int]):

    def __init__(self, key: str, default: int = 0) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> int:
        val = getenv(self.key)
        if val is None:
            return self.default
        try:
            return int(val)
        except ValueError as exc:
            raise RuntimeError(f"Unable to use {self.key}={val}: expected int") from exc


ClassType = TypeVar("ClassType")


class env_class(Generic[ClassType], env_base[Optional[Type[ClassType]], Optional[Type[ClassType]]]):

    def __init__(self, key: str, type: str) -> None:
        super().__init__(key)
        # We can't pass the type directly to avoid import cycles
        self.type = type

    def get(self) -> Optional[Type[ClassType]]:
        val = getenv(self.key)
        if val is None:
            return None
        comps = val.split(":", 1)
        if len(comps) != 2:
            raise RuntimeError(f"Unable to read {self.key}: '{val}' isn't of the form MODULE:CLASS")
        cls = getattr(importlib.import_module(comps[0]), comps[1])

        if not any((c.__name__ == self.type for c in cls.mro())):
            raise RuntimeError(f"Unable to use '{val}' from {self.key}: not of type '{self.type}'")

        return cast(Type[ClassType], cls)


@dataclass
class NvidiaTool:
    path: str
    version: str

    @staticmethod
    @functools.lru_cache
    def from_path(path: str) -> Optional[NvidiaTool]:
        try:
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
            if version is None:
                return None
            return NvidiaTool(path, version.group(1))
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


class env_nvidia_tool(env_base[str, NvidiaTool]):

    def __init__(self, binary: str) -> None:
        binary += sysconfig.get_config_var("EXE")
        self.binary = binary
        self.default_path = os.path.join(os.path.dirname(__file__), "backends", "nvidia", "bin", binary)
        # Convert ptxas-blackwell to PTXAS_BLACKWELL, not PTXAS-BLACKWELL
        super().__init__(f"TRITON_{binary.upper().replace('-', '_')}_PATH")

    def get(self) -> NvidiaTool:
        return self.transform(getenv(self.key))

    def transform(self, path: str) -> NvidiaTool:
        # We still add default as fallback in case the pointed binary isn't
        # accessible.
        if path is not None:
            paths = [path, self.default_path]
        else:
            paths = [self.default_path]

        for path in paths:
            if tool := NvidiaTool.from_path(path):
                return tool

        raise RuntimeError(f"Cannot find {self.binary}")


# Separate classes so that types are correct
class env_opt_str(env_base[Optional[str], Optional[str]]):

    def get(self) -> Optional[str]:
        return getenv(self.key)


class env_opt_bool(env_base):

    def get(self) -> Optional[str]:
        return getenv_bool(self.key, None)


@dataclass(frozen=True)
class CompileTimes:
    """
    Model holding timing information for an invocation of the compiler.

    All times in microseconds.
    """

    # Duration of make_ir
    ir_initialization: int

    # Ordered mapping from lowering stage to duration spent in that stage.
    # Keyed by stage extension, e.g. ttir, ttgir
    lowering_stages: list[tuple[str, int]]

    # Duration of saving artifacts/metadata to cache
    store_results: int

    @property
    def total_lowering(self) -> int:
        return sum((stage[1] for stage in self.lowering_stages))

    @property
    def total(self) -> int:
        return self.ir_initialization + self.total_lowering + self.store_results


class CompilationListener(Protocol):

    def __call__(self, *, src: Union[ASTSource, IRSource], metadata: dict[str, Any], metadata_group: dict[str, str],
                 times: CompileTimes, cache_hit: bool) -> None:
        ...


knobs_type = TypeVar("knobs_type", bound='base_knobs')


class base_knobs:

    @property
    def knob_descriptors(self) -> dict[str, env_base]:
        return {
            k: v
            # data descriptors live on the class object
            for k, v in type(self).__dict__.items()
            if isinstance(v, env_base)
        }

    @property
    def knobs(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.knob_descriptors.keys()}

    def copy(self: knobs_type) -> knobs_type:
        res = type(self)()
        res.__dict__.update(self.__dict__)
        return res

    def reset(self: knobs_type) -> knobs_type:
        for knob in self.knob_descriptors.keys():
            delattr(self, knob)
        return self

    @contextmanager
    def scope(self) -> Generator[None, None, None]:
        try:
            initial_env = {knob.key: getenv(knob.key) for knob in self.knob_descriptors.values()}
            orig = dict(self.__dict__)
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(orig)

            for k, v in initial_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]


class BuildImpl(Protocol):

    def __call__(self, name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str],
                 libraries: list[str], /) -> str:
        ...


class build_knobs(base_knobs):
    """Configuration controlling how the native compiler is invoked"""
    cc: env_opt_str = env_opt_str("CC")

    cudacrt_path: env_opt_str = env_opt_str("TRITON_CUDACRT_PATH")
    cudart_path: env_opt_str = env_opt_str("TRITON_CUDART_PATH")

    impl: Optional[BuildImpl] = None

    @property
    def backend_dirs(self) -> set[str]:
        return {path for path in (self.cudacrt_path, self.cudart_path) if path is not None}


class redis_knobs(base_knobs):
    key_format: env_str = env_str("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: env_str = env_str("TRITON_REDIS_HOST", "localhost")
    port: env_int = env_int("TRITON_REDIS_PORT", 6379)


cache: cache_knobs


class cache_knobs(base_knobs):
    home_dir: env_str = env_str("TRITON_HOME", os.path.expanduser("~/"))

    dump_dir = env_str_callable_default("TRITON_DUMP_DIR", lambda: cache.get_triton_dir("dump"))
    override_dir = env_str_callable_default("TRITON_OVERRIDE_DIR", lambda: cache.get_triton_dir("override"))
    dir = env_str_callable_default("TRITON_CACHE_DIR", lambda: cache.get_triton_dir("cache"))

    manager_class: env_class[CacheManager] = env_class("TRITON_CACHE_MANAGER", "CacheManager")
    remote_manager_class: env_class[RemoteCacheBackend] = env_class("TRITON_REMOTE_CACHE_BACKEND", "RemoteCacheBackend")

    def get_triton_dir(self, dirname: str) -> str:
        return os.path.join(self.home_dir, ".triton", dirname)


class compilation_knobs(base_knobs):
    override: env_bool = env_bool("TRITON_KERNEL_OVERRIDE")
    dump_ir: env_bool = env_bool("TRITON_KERNEL_DUMP")
    dump_ir_extract_di_local_variables: env_bool = env_bool("LLVM_EXTRACT_DI_LOCAL_VARIABLES")
    store_binary_only: env_bool = env_bool("TRITON_STORE_BINARY_ONLY")
    always_compile: env_bool = env_bool("TRITON_ALWAYS_COMPILE")
    # TODO: Use enum to constrain / 'typecheck' the values
    use_ir_loc: env_opt_str = env_opt_str("USE_IR_LOC")
    enable_asan: env_bool = env_bool("TRITON_ENABLE_ASAN")
    disable_line_info: env_bool = env_bool("TRITON_DISABLE_LINE_INFO")
    front_end_debugging: env_bool = env_bool("TRITON_FRONT_END_DEBUGGING")
    allow_non_constexpr_globals: env_bool = env_bool("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS")
    # Instrumentation mode is checked on every run, which is expensive.
    # We cache the value here to avoid the expensive check on every run.
    instrumentation_mode: str = env_str("TRITON_INSTRUMENTATION_MODE", "").get()
    listener: Union[CompilationListener, None] = None


class AutotuneListener(Protocol):

    def __call__(self, *, fn: JITFunction, key: tuple, best_config: Config, configs_timings: dict[Config, list[float]],
                 duration: Optional[float], cache_hit: bool) -> None:
        ...


class autotuning_knobs(base_knobs):
    cache: env_bool = env_bool("TRITON_CACHE_AUTOTUNING")
    print: env_bool = env_bool("TRITON_PRINT_AUTOTUNING")
    listener: Union[AutotuneListener, None] = None


class LaunchHook(Protocol):
    """Hook invoked before and after kernel launching
    """

    def __call__(self, metadata: LazyDict) -> None:
        ...


class InitHandleHook(Protocol):
    """Hook invoked around kernel binary/module loading.
    module/function can be None for the *start* hook (before loading).
    """

    def __call__(
        self,
        module: Optional[object],
        function: Optional[Callable],
        name: str,
        metadata_group: dict[str, str],
        hash: str,
    ) -> None:
        ...


F = TypeVar("F", bound=Callable)


class HookChain(Generic[F]):
    """A chain of hooks of the same type F to be called in order.
    """

    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed

    def add(self, func: F) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func: F) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)


# This is of the form [attr_name, attr_val]
# TODO: Use tuple instead of list for better typing.
KernelAttr = list[Union[str, int]]


class JITHookCompileInfo(TypedDict):
    key: str
    signature: dict[KernelParam, str]
    device: int
    constants: None
    num_warps: int
    num_ctas: int
    num_stages: int
    enable_fp_fusion: bool
    launch_cooperative_grid: bool
    extern_libs: tuple[tuple[str, str], ...]
    configs: list[dict[tuple[int, ...], list[KernelAttr]]]
    specialization_data: str
    is_warmup: bool


class JITHook(Protocol):

    def __call__(self, *, key: str, repr: str, fn: JitFunctionInfo, compile: JITHookCompileInfo, is_manual_warmup: bool,
                 already_compiled: bool) -> Optional[bool]:
        ...


class PipelineStagesHook(Protocol):

    def __call__(self, stages, options, language, capability):
        ...


class runtime_knobs(base_knobs):
    interpret: env_bool = env_bool("TRITON_INTERPRET")
    # debug is on critical path for kernel launches
    # avoid repeated reads from env-var by calling get directly
    debug: bool = env_bool("TRITON_DEBUG").get()
    override_arch: env_opt_str = env_opt_str("TRITON_OVERRIDE_ARCH")

    launch_enter_hook: HookChain[LaunchHook] = HookChain()
    launch_exit_hook: HookChain[LaunchHook] = HookChain(reversed=True)
    kernel_load_start_hook: HookChain[InitHandleHook] = HookChain()
    kernel_load_end_hook: HookChain[InitHandleHook] = HookChain(reversed=True)

    # hook to unload module when kernel is freed
    kernel_unload_hook: HookChain[InitHandleHook] = HookChain()

    # Hook for inspecting compiled functions and modules
    jit_cache_hook: Optional[JITHook] = None
    # Hook to signal that a kernel is done compiling and inspect compiled function.
    # jit_cache_hook will always be called before compilation and jit_post_compile_hook after.
    jit_post_compile_hook: Optional[JITHook] = None

    # Hook for inspecting compiler pipeline stages
    add_stages_inspection_hook: Optional[PipelineStagesHook] = None


class language_knobs(base_knobs):
    fp32_default: env_opt_str = env_opt_str("TRITON_F32_DEFAULT")
    default_fp_fusion: env_bool = env_bool("TRITON_DEFAULT_FP_FUSION", True)


class nvidia_knobs(base_knobs):
    cuobjdump: env_nvidia_tool = env_nvidia_tool("cuobjdump")
    nvdisasm: env_nvidia_tool = env_nvidia_tool("nvdisasm")
    ptxas: env_nvidia_tool = env_nvidia_tool("ptxas")
    ptxas_blackwell: env_nvidia_tool = env_nvidia_tool("ptxas-blackwell")

    dump_nvptx: env_bool = env_bool("NVPTX_ENABLE_DUMP")
    disable_ptxas_opt: env_bool = env_bool("DISABLE_PTXAS_OPT")
    ptxas_options: env_opt_str = env_opt_str("PTXAS_OPTIONS")
    mock_ptx_version: env_opt_str = env_opt_str("TRITON_MOCK_PTX_VERSION")
    dump_ptxas_log: env_bool = env_bool("TRITON_DUMP_PTXAS_LOG")

    libdevice_path: env_opt_str = env_opt_str("TRITON_LIBDEVICE_PATH")
    libcuda_path: env_opt_str = env_opt_str("TRITON_LIBCUDA_PATH")


class amd_knobs(base_knobs):
    use_buffer_ops: env_bool = env_bool("AMDGCN_USE_BUFFER_OPS", True)
    # Note: This requires use_buffer_ops be true to have any effect
    use_buffer_atomics: env_bool = env_bool("AMDGCN_USE_BUFFER_ATOMICS", True)
    # Note: This requires use_buffer_ops be true to have any effect
    buffer_ops_analyze_small_tensor_range: env_bool = env_bool("AMDGCN_ANALYZE_SMALL_TENSOR_RANGE", False)
    dump_amdgcn: env_bool = env_bool("AMDGCN_ENABLE_DUMP")
    libhip_path: env_opt_str = env_opt_str("TRITON_LIBHIP_PATH")

    # We use strs so that we can have a default value based on other runtime info
    use_block_pingpong: env_opt_bool = env_opt_bool("TRITON_HIP_USE_BLOCK_PINGPONG")
    use_in_thread_transpose: env_opt_bool = env_opt_bool("TRITON_HIP_USE_IN_THREAD_TRANSPOSE")
    use_async_copy: env_opt_bool = env_opt_bool("TRITON_HIP_USE_ASYNC_COPY")
    use_coexec_scheduler: env_opt_bool = env_opt_bool("TRITON_HIP_USE_COEXEC_SCHEDULER")
    use_expert_scheduling: env_opt_bool = env_opt_bool("TRITON_HIP_USE_EXPERT_SCHEDULING")

    scalarize_packed_fops: env_bool = env_bool("AMDGCN_SCALARIZE_PACKED_FOPS")

    # Path to dump MIR files for debugging/analysis
    dump_mir: env_opt_str = env_opt_str("TRITON_DUMP_MIR")
    # Path to externally-provided MIR files to use instead of generated ones
    swap_mir: env_opt_str = env_opt_str("TRITON_SWAP_MIR")
    # Enable machine instruction scheduler in MIR swap mode
    swap_mir_enable_misched: env_bool = env_bool("TRITON_SWAP_MIR_ENABLE_MISCHED", False)


class proton_knobs(base_knobs):
    disable: env_bool = env_bool("TRITON_PROTON_DISABLE", False)
    cupti_lib_dir: env_str = env_str(
        "TRITON_CUPTI_LIB_PATH",
        str(pathlib.Path(__file__).parent.absolute() / "backends" / "nvidia" / "lib" / "cupti"))
    cupti_lib_blackwell_dir: env_str = env_str(
        "TRITON_CUPTI_LIB_BLACKWELL_PATH",
        str(pathlib.Path(__file__).parent.absolute() / "backends" / "nvidia" / "lib" / "cupti-blackwell"))
    rocprofiler_sdk_include_path: env_opt_str = env_opt_str("TRITON_ROCPROFILER_SDK_INCLUDE_PATH")
    rocprofiler_sdk_lib_path: env_opt_str = env_opt_str("TRITON_ROCPROFILER_SDK_LIB_PATH")
    profile_buffer_size: env_int = env_int("TRITON_PROFILE_BUFFER_SIZE", 64 * 1024 * 1024)
    profile_metric_buffer_size: env_int = env_int("TRITON_PROFILE_METRIC_BUFFER_SIZE", 64 * 1024 * 1024)
    enable_nvtx: env_bool = env_bool("TRITON_ENABLE_NVTX", True)
    # This knob is effective only on Blackwell+ GPUs.
    #
    # When enabled, the profiling session must start after CUDA driver
    # initialization but before the CUDA context is created.
    #
    # You can ensure this in one of the following ways:
    #
    # 1) Use the `proton` CLI tool to launch the Python script, e.g.:
    #    `TRITON_ENABLE_HW_TRACE=1 proton python my_script.py`
    #
    # 2) Call `proton.start()` immediately after importing Proton, e.g.:
    #    ```python
    #    import triton
    #    import triton.profiler as proton
    #    triton.knobs.proton.enable_hw_trace = True
    #    proton.start(hook="triton")
    #    ```
    enable_hw_trace: env_bool = env_bool("TRITON_ENABLE_HW_TRACE", False)


build = build_knobs()
redis = redis_knobs()
cache = cache_knobs()
compilation = compilation_knobs()
autotuning = autotuning_knobs()
runtime = runtime_knobs()
language = language_knobs()
nvidia = nvidia_knobs()
amd = amd_knobs()
proton = proton_knobs()


def refresh_knobs():
    runtime.debug = env_bool("TRITON_DEBUG").get()
    compilation.instrumentation_mode = env_str("TRITON_INSTRUMENTATION_MODE", "").get()
<|endoftext|>"""isort:skip_file"""
__version__ = '3.8.0'

# ---------------------------------------
