                self.blocksize, self._fetch_range, self.size, **cache_options
            )
        else:
            self.buffer = io.BytesIO()
            self.offset = None
            self.forced = False
            self.location = None

    @property
    def details(self):
        if self._details is None:
            self._details = self.fs.info(self.path)
        return self._details

    @details.setter
    def details(self, value):
        self._details = value
        self.size = value["size"]

    @property
    def full_name(self):
        return _unstrip_protocol(self.path, self.fs)

    @property
    def closed(self):
        # get around this attr being read-only in IOBase
        # use getattr here, since this can be called during del
        return getattr(self, "_closed", True)

    @closed.setter
    def closed(self, c):
        self._closed = c

    def __hash__(self):
        if "w" in self.mode:
            return id(self)
        else:
            return int(tokenize(self.details), 16)

    def __eq__(self, other):
        """Files are equal if they have the same checksum, only in read mode"""
        if self is other:
            return True
        return (
            isinstance(other, type(self))
            and self.mode == "rb"
            and other.mode == "rb"
            and hash(self) == hash(other)
        )

    def commit(self):
        """Move from temp to final destination"""

    def discard(self):
        """Throw away temporary file"""

    def info(self):
        """File information about this path"""
        if self.readable():
            return self.details
        else:
            raise ValueError("Info not available while writing")

    def tell(self):
        """Current file location"""
        return self.loc

    def seek(self, loc, whence=0):
        """Set current file location

        Parameters
        ----------
        loc: int
            byte location
        whence: {0, 1, 2}
            from start of file, current location or end of file, resp.
        """
        loc = int(loc)
        if not self.mode == "rb":
            raise OSError(ESPIPE, "Seek only available in read mode")
        if whence == 0:
            nloc = loc
        elif whence == 1:
            nloc = self.loc + loc
        elif whence == 2:
            nloc = self.size + loc
        else:
            raise ValueError(f"invalid whence ({whence}, should be 0, 1 or 2)")
        if nloc < 0:
            raise ValueError("Seek before start of file")
        self.loc = nloc
        return self.loc

    def write(self, data):
        """
        Write data to buffer.

        Buffer only sent on flush() or if buffer is greater than
        or equal to blocksize.

        Parameters
        ----------
        data: bytes
            Set of bytes to be written.
        """
        if not self.writable():
            raise ValueError("File not in write mode")
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self.forced:
            raise ValueError("This file has been force-flushed, can only close")
        out = self.buffer.write(data)
        self.loc += out
        if self.buffer.tell() >= self.blocksize:
            self.flush()
        return out

    def flush(self, force=False):
        """
        Write buffered data to backend store.

        Writes the current buffer, if it is larger than the block-size, or if
        the file is being closed.

        Parameters
        ----------
        force: bool
            When closing, write the last block even if it is smaller than
            blocks are allowed to be. Disallows further writing to this file.
        """

        if self.closed:
            raise ValueError("Flush on closed file")
        if force and self.forced:
            raise ValueError("Force flush cannot be called more than once")
        if force:
            self.forced = True

        if self.readable():
            # no-op to flush on read-mode
            return

        if not force and self.buffer.tell() < self.blocksize:
            # Defer write on small block
            return

        if self.offset is None:
            # Initialize a multipart upload
            self.offset = 0
            try:
                self._initiate_upload()
            except Exception:
                self.closed = True
                raise

        if self._upload_chunk(final=force) is not False:
            self.offset += self.buffer.seek(0, 2)
            self.buffer = io.BytesIO()

    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """
        # may not yet have been initialized, may need to call _initialize_upload

    def _initiate_upload(self):
        """Create remote file/upload"""
        pass

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        return self.fs.cat_file(self.path, start=start, end=end)

    def read(self, length=-1):
        """
        Return data from cache, or fetch pieces as necessary

        Parameters
        ----------
        length: int (-1)
            Number of bytes to read; if <0, all remaining bytes.
        """
        length = -1 if length is None else int(length)
        if self.mode != "rb":
            raise ValueError("File not in read mode")
        if length < 0:
            length = self.size - self.loc
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if length == 0:
            # don't even bother calling fetch
            return b""
        out = self.cache._fetch(self.loc, self.loc + length)

        logger.debug(
            "%s read: %i - %i %s",
            self,
            self.loc,
            self.loc + length,
            self.cache._log_stats(),
        )
        self.loc += len(out)
        return out

    def readinto(self, b):
        """mirrors builtin file's readinto method

        https://docs.python.org/3/library/io.html#io.RawIOBase.readinto
        """
        out = memoryview(b).cast("B")
        data = self.read(out.nbytes)
        out[: len(data)] = data
        return len(data)

    def readuntil(self, char=b"\n", blocks=None):
        """Return data between current position and first occurrence of char

        char is included in the output, except if the end of the tile is
        encountered first.

        Parameters
        ----------
        char: bytes
            Thing to find
        blocks: None or int
            How much to read in each go. Defaults to file blocksize - which may
            mean a new read on every call.
        """
        out = []
        while True:
            start = self.tell()
            part = self.read(blocks or self.blocksize)
            if len(part) == 0:
                break
            found = part.find(char)
            if found > -1:
                out.append(part[: found + len(char)])
                self.seek(start + found + len(char))
                break
            out.append(part)
        return b"".join(out)

    def readline(self):
        """Read until and including the first occurrence of newline character

        Note that, because of character encoding, this is not necessarily a
        true line ending.
        """
        return self.readuntil(b"\n")

    def __next__(self):
        out = self.readline()
        if out:
            return out
        raise StopIteration

    def __iter__(self):
        return self

    def readlines(self):
        """Return all data, split by the newline character, including the newline character"""
        data = self.read()
        lines = data.split(b"\n")
        out = [l + b"\n" for l in lines[:-1]]
        if data.endswith(b"\n"):
            return out
        else:
            return out + [lines[-1]]
        # return list(self)  ???

    def readinto1(self, b):
        return self.readinto(b)

    def close(self):
        """Close file

        Finalizes writes, discards cache
        """
        if getattr(self, "_unclosable", False):
            return
        if self.closed:
            return
        try:
            if self.mode == "rb":
                cache = getattr(self, "cache", None)
                if cache is not None:
                    close = getattr(cache, "close", None)
                    if callable(close):
                        close()
                self.cache = None
            else:
                if not getattr(self, "forced", True):
                    self.flush(force=True)

                if self.fs is not None:
                    self.fs.invalidate_cache(self.path)
                    self.fs.invalidate_cache(self.fs._parent(self.path))
        finally:
            self.closed = True

    def readable(self):
        """Whether opened for reading"""
        return "r" in self.mode and not self.closed

    def seekable(self):
        """Whether is seekable (only in read mode)"""
        return self.readable()

    def writable(self):
        """Whether opened for writing"""
        return self.mode in {"wb", "ab", "xb"} and not self.closed

    def __reduce__(self):
        if self.mode != "rb":
            raise RuntimeError("Pickling a writeable file is not supported")

        return reopen, (
            self.fs,
            self.path,
            self.mode,
            self.blocksize,
            self.loc,
            self.size,
            self.autocommit,
            self.cache.name if self.cache else "none",
            self.kwargs,
        )

    def __del__(self):
        if not self.closed:
            self.close()

    def __str__(self):
        return f"<File-like object {type(self.fs).__name__}, {self.path}>"

    __repr__ = __str__

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def reopen(fs, path, mode, blocksize, loc, size, autocommit, cache_type, kwargs):
    file = fs.open(
        path,
        mode=mode,
        block_size=blocksize,
        autocommit=autocommit,
        cache_type=cache_type,
        size=size,
        **kwargs,
    )
    if loc > 0:
        file.seek(loc)
    return file
<|endoftext|>import io
import json
import warnings

import fsspec

from .core import url_to_fs
from .spec import AbstractBufferedFile
from .utils import merge_offset_ranges

# Parquet-Specific Utilities for fsspec
#
# Most of the functions defined in this module are NOT
# intended for public consumption. The only exception
# to this is `open_parquet_file`, which should be used
# place of `fs.open()` to open parquet-formatted files
# on remote file systems.


class AlreadyBufferedFile(AbstractBufferedFile):
    def _fetch_range(self, start, end):
        raise NotImplementedError


def open_parquet_files(
    path: list[str],
    fs: None | fsspec.AbstractFileSystem = None,
    metadata=None,
    columns: None | list[str] = None,
    row_groups: None | list[int] = None,
    storage_options: None | dict = None,
    engine: str = "auto",
    max_gap: int = 64_000,
    max_block: int = 256_000_000,
    footer_sample_size: int = 1_000_000,
    filters: None | list[list[list[str]]] = None,
    **kwargs,
):
    """
    Return a file-like object for a single Parquet file.

    The specified parquet `engine` will be used to parse the
    footer metadata, and determine the required byte ranges
    from the file. The target path will then be opened with
    the "parts" (`KnownPartsOfAFile`) caching strategy.

    Note that this method is intended for usage with remote
    file systems, and is unlikely to improve parquet-read
    performance on local file systems.

    Parameters
    ----------
    path: str
        Target file path.
    metadata: Any, optional
        Parquet metadata object. Object type must be supported
        by the backend parquet engine. For now, only the "fastparquet"
        engine supports an explicit `ParquetFile` metadata object.
        If a metadata object is supplied, the remote footer metadata
        will not need to be transferred into local memory.
    fs: AbstractFileSystem, optional
        Filesystem object to use for opening the file. If nothing is
        specified, an `AbstractFileSystem` object will be inferred.
    engine : str, default "auto"
        Parquet engine to use for metadata parsing. Allowed options
        include "fastparquet", "pyarrow", and "auto". The specified
        engine must be installed in the current environment. If
        "auto" is specified, and both engines are installed,
        "fastparquet" will take precedence over "pyarrow".
    columns: list, optional
        List of all column names that may be read from the file.
    row_groups : list, optional
        List of all row-groups that may be read from the file. This
        may be a list of row-group indices (integers), or it may be
        a list of `RowGroup` metadata objects (if the "fastparquet"
        engine is used).
    storage_options : dict, optional
        Used to generate an `AbstractFileSystem` object if `fs` was
        not specified.
    max_gap : int, optional
        Neighboring byte ranges will only be merged when their
        inter-range gap is <= `max_gap`. Default is 64KB.
    max_block : int, optional
        Neighboring byte ranges will only be merged when the size of
        the aggregated range is <= `max_block`. Default is 256MB.
    footer_sample_size : int, optional
        Number of bytes to read from the end of the path to look
        for the footer metadata. If the sampled bytes do not contain
        the footer, a second read request will be required, and
        performance will suffer. Default is 1MB.
    filters : list[list], optional
        List of filters to apply to prevent reading row groups, of the
        same format as accepted by the loading engines. Ignored if
        ``row_groups`` is specified.
    **kwargs :
        Optional key-word arguments to pass to `fs.open`
    """

    # Make sure we have an `AbstractFileSystem` object
    # to work with
    if fs is None:
        path0 = path
        if isinstance(path, (list, tuple)):
            path = path[0]
        fs, path = url_to_fs(path, **(storage_options or {}))
    else:
        path0 = path

    # For now, `columns == []` not supported, is the same
    # as all columns
    if columns is not None and len(columns) == 0:
        columns = None

    # Set the engine
    engine = _set_engine(engine)

    if isinstance(path0, (list, tuple)):
        paths = path0
    elif "*" in path:
        paths = fs.glob(path)
    elif path0.endswith("/"):  # or fs.isdir(path):
        paths = [
            _
            for _ in fs.find(path, withdirs=False, detail=False)
            if _.endswith((".parquet", ".parq"))
        ]
    else:
        paths = [path]

    data = _get_parquet_byte_ranges(
        paths,
        fs,
        metadata=metadata,
        columns=columns,
        row_groups=row_groups,
        engine=engine,
        max_gap=max_gap,
        max_block=max_block,
        footer_sample_size=footer_sample_size,
        filters=filters,
    )

    # Call self.open with "parts" caching
    options = kwargs.pop("cache_options", {}).copy()
    return [
        AlreadyBufferedFile(
            fs=None,
            path=fn,
            mode="rb",
            cache_type="parts",
            cache_options={
                **options,
                "data": ranges,
            },
            size=max(_[1] for _ in ranges),
            **kwargs,
        )
        for fn, ranges in data.items()
    ]


def open_parquet_file(*args, **kwargs):
    """Create files tailed to reading specific parts of parquet files

    Please see ``open_parquet_files`` for details of the arguments. The
    difference is, this function always returns a single ``AlreadyBufferedFile``,
    whereas `open_parquet_files`` always returns a list of files, even if
    there are one or zero matching parquet files.
    """
    return open_parquet_files(*args, **kwargs)[0]


def _get_parquet_byte_ranges(
    paths,
    fs,
    metadata=None,
    columns=None,
    row_groups=None,
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
    engine="auto",
    filters=None,
):
    """Get a dictionary of the known byte ranges needed
    to read a specific column/row-group selection from a
    Parquet dataset. Each value in the output dictionary
    is intended for use as the `data` argument for the
    `KnownPartsOfAFile` caching strategy of a single path.
    """

    # Set engine if necessary
    if isinstance(engine, str):
        engine = _set_engine(engine)

    # Pass to a specialized function if metadata is defined
    if metadata is not None:
        # Use the provided parquet metadata object
        # to avoid transferring/parsing footer metadata
        return _get_parquet_byte_ranges_from_metadata(
            metadata,
            fs,
            engine,
            columns=columns,
            row_groups=row_groups,
            max_gap=max_gap,
            max_block=max_block,
            filters=filters,
        )

    # Populate global paths, starts, & ends
    if columns is None and row_groups is None and filters is None:
        # We are NOT selecting specific columns or row-groups.
        #
        # We can avoid sampling the footers, and just transfer
        # all file data with cat_ranges
        result = {path: {(0, len(data)): data} for path, data in fs.cat(paths).items()}
    else:
        # We ARE selecting specific columns or row-groups.
        #
        # Get file sizes asynchronously
        file_sizes = fs.sizes(paths)
        data_paths = []
        data_starts = []
        data_ends = []
        # Gather file footers.
        # We just take the last `footer_sample_size` bytes of each
        # file (or the entire file if it is smaller than that)
        footer_starts = [
            max(0, file_size - footer_sample_size) for file_size in file_sizes
        ]
        footer_samples = fs.cat_ranges(paths, footer_starts, file_sizes)

        # Check our footer samples and re-sample if necessary.
        large_footer = []
        for i, path in enumerate(paths):
            footer_size = int.from_bytes(footer_samples[i][-8:-4], "little")
            real_footer_start = file_sizes[i] - (footer_size + 8)
            if real_footer_start < footer_starts[i]:
                large_footer.append((i, real_footer_start))
        if large_footer:
            warnings.warn(
                f"Not enough data was used to sample the parquet footer. "
                f"Try setting footer_sample_size >= {large_footer}."
            )
            path0 = [paths[i] for i, _ in large_footer]
            starts = [_[1] for _ in large_footer]
            ends = [file_sizes[i] - footer_sample_size for i, _ in large_footer]
            data = fs.cat_ranges(path0, starts, ends)
            for i, (path, start, block) in enumerate(zip(path0, starts, data)):
                footer_samples[i] = block + footer_samples[i]
                footer_starts[i] = start
        result = {
            path: {(start, size): data}
            for path, start, size, data in zip(
                paths, footer_starts, file_sizes, footer_samples
            )
        }

        # Calculate required byte ranges for each path
        for i, path in enumerate(paths):
            # Use "engine" to collect data byte ranges
            path_data_starts, path_data_ends = engine._parquet_byte_ranges(
                columns,
                row_groups=row_groups,
                footer=footer_samples[i],
                footer_start=footer_starts[i],
                filters=filters,
            )

            data_paths += [path] * len(path_data_starts)
            data_starts += path_data_starts
            data_ends += path_data_ends

        # Merge adjacent offset ranges
        data_paths, data_starts, data_ends = merge_offset_ranges(
            data_paths,
            data_starts,
            data_ends,
            max_gap=max_gap,
            max_block=max_block,
            sort=True,
        )

        # Transfer the data byte-ranges into local memory
        _transfer_ranges(fs, result, data_paths, data_starts, data_ends)

    # Add b"PAR1" to headers
    _add_header_magic(result)

    return result


def _get_parquet_byte_ranges_from_metadata(
    metadata,
    fs,
    engine,
    columns=None,
    row_groups=None,
    max_gap=64_000,
    max_block=256_000_000,
    filters=None,
):
    """Simplified version of `_get_parquet_byte_ranges` for
    the case that an engine-specific `metadata` object is
    provided, and the remote footer metadata does not need to
    be transferred before calculating the required byte ranges.
    """

    # Use "engine" to collect data byte ranges
    data_paths, data_starts, data_ends = engine._parquet_byte_ranges(
        columns, row_groups=row_groups, metadata=metadata, filters=filters
    )

    # Merge adjacent offset ranges
    data_paths, data_starts, data_ends = merge_offset_ranges(
        data_paths,
        data_starts,
        data_ends,
        max_gap=max_gap,
        max_block=max_block,
        sort=False,  # Should be sorted
    )

    # Transfer the data byte-ranges into local memory
    result = {fn: {} for fn in list(set(data_paths))}
    _transfer_ranges(fs, result, data_paths, data_starts, data_ends)

    # Add b"PAR1" to header
    _add_header_magic(result)

    return result


def _transfer_ranges(fs, blocks, paths, starts, ends):
    # Use cat_ranges to gather the data byte_ranges
    ranges = (paths, starts, ends)
    for path, start, stop, data in zip(*ranges, fs.cat_ranges(*ranges)):
        blocks[path][(start, stop)] = data


def _add_header_magic(data):
    # Add b"PAR1" to file headers
    for path in list(data):
        add_magic = True
        for k in data[path]:
            if k[0] == 0 and k[1] >= 4:
                add_magic = False
                break
        if add_magic:
            data[path][(0, 4)] = b"PAR1"


def _set_engine(engine_str):
    # Define a list of parquet engines to try
    if engine_str == "auto":
        try_engines = ("fastparquet", "pyarrow")
    elif not isinstance(engine_str, str):
        raise ValueError(
            "Failed to set parquet engine! "
            "Please pass 'fastparquet', 'pyarrow', or 'auto'"
        )
    elif engine_str not in ("fastparquet", "pyarrow"):
        raise ValueError(f"{engine_str} engine not supported by `fsspec.parquet`")
    else:
        try_engines = [engine_str]

    # Try importing the engines in `try_engines`,
    # and choose the first one that succeeds
    for engine in try_engines:
        try:
            if engine == "fastparquet":
                return FastparquetEngine()
            elif engine == "pyarrow":
                return PyarrowEngine()
        except ImportError:
            pass

    # Raise an error if a supported parquet engine
    # was not found
    raise ImportError(
        f"The following parquet engines are not installed "
        f"in your python environment: {try_engines}."
        f"Please install 'fastparquert' or 'pyarrow' to "
        f"utilize the `fsspec.parquet` module."
    )


class FastparquetEngine:
    # The purpose of the FastparquetEngine class is
    # to check if fastparquet can be imported (on initialization)
    # and to define a `_parquet_byte_ranges` method. In the
    # future, this class may also be used to define other
    # methods/logic that are specific to fastparquet.

    def __init__(self):
        import fastparquet as fp

        self.fp = fp

    def _parquet_byte_ranges(
        self,
        columns,
        row_groups=None,
        metadata=None,
        footer=None,
        footer_start=None,
        filters=None,
    ):
        # Initialize offset ranges and define ParqetFile metadata
        pf = metadata
        data_paths, data_starts, data_ends = [], [], []
        if filters and row_groups:
            raise ValueError("filters and row_groups cannot be used together")
        if pf is None:
            pf = self.fp.ParquetFile(io.BytesIO(footer))

        # Convert columns to a set and add any index columns
        # specified in the pandas metadata (just in case)
        column_set = None if columns is None else {c.split(".", 1)[0] for c in columns}
        if column_set is not None and hasattr(pf, "pandas_metadata"):
            md_index = [
                ind
                for ind in pf.pandas_metadata.get("index_columns", [])
                # Ignore RangeIndex information
                if not isinstance(ind, dict)
            ]
            column_set |= set(md_index)

        # Check if row_groups is a list of integers
        # or a list of row-group metadata
        if filters:
            from fastparquet.api import filter_row_groups

            row_group_indices = None
            row_groups = filter_row_groups(pf, filters)
        elif row_groups and not isinstance(row_groups[0], int):
            # Input row_groups contains row-group metadata
            row_group_indices = None
        else:
            # Input row_groups contains row-group indices
            row_group_indices = row_groups
            row_groups = pf.row_groups
        if column_set is not None:
            column_set = [
                _ if isinstance(_, list) else _.split(".") for _ in column_set
            ]

        # Loop through column chunks to add required byte ranges
        for r, row_group in enumerate(row_groups):
            # Skip this row-group if we are targeting
            # specific row-groups
            if row_group_indices is None or r in row_group_indices:
                # Find the target parquet-file path for `row_group`
                fn = pf.row_group_filename(row_group)

                for column in row_group.columns:
                    name = column.meta_data.path_in_schema
                    # Skip this column if we are targeting specific columns
                    if column_set is None or _cmp(name, column_set):
                        file_offset0 = column.meta_data.dictionary_page_offset
                        if file_offset0 is None:
                            file_offset0 = column.meta_data.data_page_offset
                        num_bytes = column.meta_data.total_compressed_size
                        if footer_start is None or file_offset0 < footer_start:
                            data_paths.append(fn)
                            data_starts.append(file_offset0)
                            data_ends.append(
                                min(
                                    file_offset0 + num_bytes,
                                    footer_start or (file_offset0 + num_bytes),
                                )
                            )

        if metadata:
            # The metadata in this call may map to multiple
            # file paths. Need to include `data_paths`
            return data_paths, data_starts, data_ends
        return data_starts, data_ends


class PyarrowEngine:
    # The purpose of the PyarrowEngine class is
    # to check if pyarrow can be imported (on initialization)
    # and to define a `_parquet_byte_ranges` method. In the
    # future, this class may also be used to define other
    # methods/logic that are specific to pyarrow.

    def __init__(self):
        import pyarrow.parquet as pq

        self.pq = pq

    def _parquet_byte_ranges(
        self,
        columns,
        row_groups=None,
        metadata=None,
        footer=None,
        footer_start=None,
        filters=None,
    ):
        if metadata is not None:
            raise ValueError("metadata input not supported for PyarrowEngine")
        if filters:
            # there must be a way!
            raise NotImplementedError

        data_starts, data_ends = [], []
        md = self.pq.ParquetFile(io.BytesIO(footer)).metadata

        # Convert columns to a set and add any index columns
        # specified in the pandas metadata (just in case)
        column_set = None if columns is None else set(columns)
        if column_set is not None:
            schema = md.schema.to_arrow_schema()
            has_pandas_metadata = (
                schema.metadata is not None and b"pandas" in schema.metadata
            )
            if has_pandas_metadata:
                md_index = [
                    ind
                    for ind in json.loads(
                        schema.metadata[b"pandas"].decode("utf8")
                    ).get("index_columns", [])
                    # Ignore RangeIndex information
                    if not isinstance(ind, dict)
                ]
                column_set |= set(md_index)
        if column_set is not None:
            column_set = [
                _[:1] if isinstance(_, list) else _.split(".")[:1] for _ in column_set
            ]

        # Loop through column chunks to add required byte ranges
        for r in range(md.num_row_groups):
            # Skip this row-group if we are targeting
            # specific row-groups
            if row_groups is None or r in row_groups:
                row_group = md.row_group(r)
                for c in range(row_group.num_columns):
                    column = row_group.column(c)
                    name = column.path_in_schema.split(".")
                    # Skip this column if we are targeting specific columns
                    if column_set is None or _cmp(name, column_set):
                        meta = column.to_dict()
                        # Any offset could be the first one
                        file_offset0 = min(
                            _
                            for _ in [
                                meta.get("dictionary_page_offset"),
                                meta.get("data_page_offset"),
                                meta.get("index_page_offset"),
                            ]
                            if _ is not None
                        )
                        if file_offset0 < footer_start:
                            data_starts.append(file_offset0)
                            data_ends.append(
                                min(
                                    meta["total_compressed_size"] + file_offset0,
                                    footer_start,
                                )
                            )

        data_starts.append(footer_start)
        data_ends.append(footer_start + len(footer))
        return data_starts, data_ends


def _cmp(name, column_set):
    return any(all(a == b for a, b in zip(name, _)) for _ in column_set)
<|endoftext|>import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property

from fsspec.core import url_to_fs

logger = logging.getLogger("fsspec.mapping")


class FSMap(MutableMapping):
    """Wrap a FileSystem instance as a mutable wrapping.

    The keys of the mapping become files under the given root, and the
    values (which must be bytes) the contents of those files.

    Parameters
    ----------
    root: string
        prefix for all the files
    fs: FileSystem instance
    check: bool (=True)
        performs a touch at the location, to check for write access.

    Examples
    --------
    >>> fs = FileSystem(**parameters)  # doctest: +SKIP
    >>> d = FSMap('my-data/path/', fs)  # doctest: +SKIP
    or, more likely
    >>> d = fs.get_mapper('my-data/path/')

    >>> d['loc1'] = b'Hello World'  # doctest: +SKIP
    >>> list(d.keys())  # doctest: +SKIP
    ['loc1']
    >>> d['loc1']  # doctest: +SKIP
    b'Hello World'
    """

    def __init__(self, root, fs, check=False, create=False, missing_exceptions=None):
        self.fs = fs
        self.root = fs._strip_protocol(root)
        self._root_key_to_str = fs._strip_protocol(posixpath.join(root, "x"))[:-1]
        if missing_exceptions is None:
            missing_exceptions = (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
            )
        self.missing_exceptions = missing_exceptions
        self.check = check
        self.create = create
        if create:
            if not self.fs.exists(root):
                self.fs.mkdir(root)
        if check:
            if not self.fs.exists(root):
                raise ValueError(
                    f"Path {root} does not exist. Create "
                    f" with the ``create=True`` keyword"
                )
            self.fs.touch(root + "/a")
            self.fs.rm(root + "/a")

    @cached_property
    def dirfs(self):
        """dirfs instance that can be used with the same keys as the mapper"""
        from .implementations.dirfs import DirFileSystem

        return DirFileSystem(path=self._root_key_to_str, fs=self.fs)

    def clear(self):
        """Remove all keys below root - empties out mapping"""
        logger.info("Clear mapping at %s", self.root)
        try:
            self.fs.rm(self.root, True)
            self.fs.mkdir(self.root)
        except:  # noqa: E722
            pass

    def getitems(self, keys, on_error="raise"):
        """Fetch multiple items from the store

        If the backend is async-able, this might proceed concurrently

        Parameters
        ----------
        keys: list(str)
            They keys to be fetched
        on_error : "raise", "omit", "return"
            If raise, an underlying exception will be raised (converted to KeyError
            if the type is in self.missing_exceptions); if omit, keys with exception
            will simply not be included in the output; if "return", all keys are
            included in the output, but the value will be bytes or an exception
            instance.

        Returns
        -------
        dict(key, bytes|exception)
        """
        keys2 = [self._key_to_str(k) for k in keys]
        oe = on_error if on_error == "raise" else "return"
        try:
            out = self.fs.cat(keys2, on_error=oe)
            if isinstance(out, bytes):
                out = {keys2[0]: out}
        except self.missing_exceptions as e:
            raise KeyError from e
        out = {
            k: (KeyError() if isinstance(v, self.missing_exceptions) else v)
            for k, v in out.items()
        }
        return {
            key: out[k2] if on_error == "raise" else out.get(k2, KeyError(k2))
            for key, k2 in zip(keys, keys2)
            if on_error == "return" or not isinstance(out[k2], BaseException)
        }

    def setitems(self, values_dict):
        """Set the values of multiple items in the store

        Parameters
        ----------
        values_dict: dict(str, bytes)
        """
        values = {self._key_to_str(k): maybe_convert(v) for k, v in values_dict.items()}
        self.fs.pipe(values)

    def delitems(self, keys):
        """Remove multiple keys from the store"""
        self.fs.rm([self._key_to_str(k) for k in keys])

    def _key_to_str(self, key):
        """Generate full path for the key"""
        if not isinstance(key, str):
            # raise TypeError("key must be of type `str`, got `{type(key).__name__}`"
            warnings.warn(
                "from fsspec 2023.5 onward FSMap non-str keys will raise TypeError",
                DeprecationWarning,
            )
            if isinstance(key, list):
                key = tuple(key)
            key = str(key)
        return f"{self._root_key_to_str}{key}".rstrip("/")

    def _str_to_key(self, s):
        """Strip path of to leave key name"""
        return s[len(self.root) :].lstrip("/")

    def __getitem__(self, key, default=None):
        """Retrieve data"""
        k = self._key_to_str(key)
        try:
            result = self.fs.cat(k)
        except self.missing_exceptions as exc:
            if default is not None:
                return default
            raise KeyError(key) from exc
        return result

    def pop(self, key, default=None):
        """Pop data"""
        result = self.__getitem__(key, default)
        try:
            del self[key]
        except KeyError:
            pass
        return result

    def __setitem__(self, key, value):
        """Store value in key"""
        key = self._key_to_str(key)
        self.fs.mkdirs(self.fs._parent(key), exist_ok=True)
        self.fs.pipe_file(key, maybe_convert(value))

    def __iter__(self):
        return (self._str_to_key(x) for x in self.fs.find(self.root))

    def __len__(self):
        return len(self.fs.find(self.root))

    def __delitem__(self, key):
        """Remove key"""
        try:
            self.fs.rm(self._key_to_str(key))
        except Exception as exc:
            raise KeyError from exc

    def __contains__(self, key):
        """Does key exist in mapping?"""
        path = self._key_to_str(key)
        return self.fs.isfile(path)

    def __reduce__(self):
        return FSMap, (self.root, self.fs, False, False, self.missing_exceptions)


def maybe_convert(value):
    if isinstance(value, array.array) or hasattr(value, "__array__"):
        # bytes-like things
        if hasattr(value, "dtype") and value.dtype.kind in "Mm":
            # The buffer interface doesn't support datetime64/timdelta64 numpy
            # arrays
            value = value.view("int64")
        value = bytes(memoryview(value))
    return value


def get_mapper(
    url="",
    check=False,
    create=False,
    missing_exceptions=None,
    alternate_root=None,
    **kwargs,
):
    """Create key-value interface for given URL and options

    The URL will be of the form "protocol://location" and point to the root
    of the mapper required. All keys will be file-names below this location,
    and their values the contents of each key.

    Also accepts compound URLs like zip::s3://bucket/file.zip , see ``fsspec.open``.

    Parameters
    ----------
    url: str
        Root URL of mapping
    check: bool
        Whether to attempt to read from the location before instantiation, to
        check that the mapping does exist
    create: bool
        Whether to make the directory corresponding to the root before
        instantiating
    missing_exceptions: None or tuple
        If given, these exception types will be regarded as missing keys and
        return KeyError when trying to read data. By default, you get
        (FileNotFoundError, IsADirectoryError, NotADirectoryError)
    alternate_root: None or str
        In cases of complex URLs, the parser may fail to pick the correct part
        for the mapper root, so this arg can override

    Returns
    -------
    ``FSMap`` instance, the dict-like key-value store.
    """
    # Removing protocol here - could defer to each open() on the backend
    fs, urlpath = url_to_fs(url, **kwargs)
    root = alternate_root if alternate_root is not None else urlpath
    return FSMap(root, fs, check, create, missing_exceptions=missing_exceptions)
<|endoftext|>import time
from collections.abc import MutableMapping
from functools import lru_cache


class DirCache(MutableMapping):
    """
    Caching of directory listings, in a structure like::

        {"path0": [
            {"name": "path0/file0",
             "size": 123,
             "type": "file",
             ...
            },
            {"name": "path0/file1",
            },
            ...
            ],
         "path1": [...]
        }

    Parameters to this class control listing expiry or indeed turn
    caching off
    """

    def __init__(
        self,
        use_listings_cache=True,
        listings_expiry_time=None,
        max_paths=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        use_listings_cache: bool
            If False, this cache never returns items, but always reports KeyError,
            and setting items has no effect
        listings_expiry_time: int or float (optional)
            Time in seconds that a listing is considered valid. If None,
            listings do not expire.
        max_paths: int (optional)
            The number of most recent listings that are considered valid; 'recent'
            refers to when the entry was set.
        """
        self._cache = {}
        self._times = {}
        if max_paths:
            self._q = lru_cache(max_paths + 1)(lambda key: self._cache.pop(key, None))
        self.use_listings_cache = use_listings_cache
        self.listings_expiry_time = listings_expiry_time
        self.max_paths = max_paths

    def __getitem__(self, item):
        if self.listings_expiry_time is not None:
            if self._times.get(item, 0) - time.time() < -self.listings_expiry_time:
                del self._cache[item]
        if self.max_paths:
            self._q(item)
        return self._cache[item]  # maybe raises KeyError

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False

    def __setitem__(self, key, value):
        if not self.use_listings_cache:
            return
        if self.max_paths:
            self._q(key)
        self._cache[key] = value
        if self.listings_expiry_time is not None:
            self._times[key] = time.time()

    def __delitem__(self, key):
        del self._cache[key]

    def __iter__(self):
        entries = list(self._cache)

        return (k for k in entries if k in self)

    def __reduce__(self):
        return (
            DirCache,
            (self.use_listings_cache, self.listings_expiry_time, self.max_paths),
        )
<|endoftext|>import ast
import contextlib
import logging
import os
import re
from collections.abc import Sequence
from typing import ClassVar

import panel as pn

from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations

pn.extension()
logger = logging.getLogger("fsspec.gui")


class SigSlot:
    """Signal-slot mixin, for Panel event passing

    Include this class in a widget manager's superclasses to be able to
    register events and callbacks on Panel widgets managed by that class.

    The method ``_register`` should be called as widgets are added, and external
    code should call ``connect`` to associate callbacks.

    By default, all signals emit a DEBUG logging statement.
    """

    # names of signals that this class may emit each of which must be
    # set by _register for any new instance
    signals: ClassVar[Sequence[str]] = []
    # names of actions that this class may respond to
    slots: ClassVar[Sequence[str]] = []

    # each of which must be a method name

    def __init__(self):
        self._ignoring_events = False
        self._sigs = {}
        self._map = {}
        self._setup()

    def _setup(self):
        """Create GUI elements and register signals"""
        self.panel = pn.pane.PaneBase()
        # no signals to set up in the base class

    def _register(
        self, widget, name, thing="value", log_level=logging.DEBUG, auto=False
    ):
        """Watch the given attribute of a widget and assign it a named event

        This is normally called at the time a widget is instantiated, in the
        class which owns it.

        Parameters
        ----------
        widget : pn.layout.Panel or None
            Widget to watch. If None, an anonymous signal not associated with
            any widget.
        name : str
            Name of this event
        thing : str
            Attribute of the given widget to watch
        log_level : int
            When the signal is triggered, a logging event of the given level
            will be fired in the dfviz logger.
        auto : bool
            If True, automatically connects with a method in this class of the
            same name.
        """
        if name not in self.signals:
            raise ValueError(f"Attempt to assign an undeclared signal: {name}")
        self._sigs[name] = {
            "widget": widget,
            "callbacks": [],
            "thing": thing,
            "log": log_level,
        }
        wn = "-".join(
            [
                getattr(widget, "name", str(widget)) if widget is not None else "none",
                thing,
            ]
        )
        self._map[wn] = name
        if widget is not None:
            widget.param.watch(self._signal, thing, onlychanged=True)
        if auto and hasattr(self, name):
            self.connect(name, getattr(self, name))

    def _repr_mimebundle_(self, *args, **kwargs):
        """Display in a notebook or a server"""
        try:
            return self.panel._repr_mimebundle_(*args, **kwargs)
        except (ValueError, AttributeError) as exc:
            raise NotImplementedError(
                "Panel does not seem to be set up properly"
            ) from exc

    def connect(self, signal, slot):
        """Associate call back with given event

        The callback must be a function which takes the "new" value of the
        watched attribute as the only parameter. If the callback return False,
        this cancels any further processing of the given event.

        Alternatively, the callback can be a string, in which case it means
        emitting the correspondingly-named event (i.e., connect to self)
        """
        self._sigs[signal]["callbacks"].append(slot)

    def _signal(self, event):
        """This is called by a an action on a widget

        Within an self.ignore_events context, nothing happens.

        Tests can execute this method by directly changing the values of
        widget components.
        """
        if not self._ignoring_events:
            wn = "-".join([event.obj.name, event.name])
            if wn in self._map and self._map[wn] in self._sigs:
                self._emit(self._map[wn], event.new)

    @contextlib.contextmanager
    def ignore_events(self):
        """Temporarily turn off events processing in this instance

        (does not propagate to children)
        """
        self._ignoring_events = True
        try:
            yield
        finally:
            self._ignoring_events = False

    def _emit(self, sig, value=None):
        """An event happened, call its callbacks

        This method can be used in tests to simulate message passing without
        directly changing visual elements.

        Calling of callbacks will halt whenever one returns False.
        """
        logger.log(self._sigs[sig]["log"], f"{sig}: {value}")
        for callback in self._sigs[sig]["callbacks"]:
            if isinstance(callback, str):
                self._emit(callback)
            else:
                try:
                    # running callbacks should not break the interface
                    ret = callback(value)
                    if ret is False:
                        break
                except Exception as e:
                    logger.exception(
                        "Exception (%s) while executing callback for signal: %s",
                        e,
                        sig,
                    )

    def show(self, threads=False):
        """Open a new browser tab and display this instance's interface"""
        self.panel.show(threads=threads, verbose=False)
        return self


class SingleSelect(SigSlot):
    """A multiselect which only allows you to select one item for an event"""

    signals = ["_selected", "selected"]  # the first is internal
    slots = ["set_options", "set_selection", "add", "clear", "select"]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def _setup(self):
        self.panel = pn.widgets.MultiSelect(**self.kwargs)
        self._register(self.panel, "_selected", "value")
        self._register(None, "selected")
        self.connect("_selected", self.select_one)

    def _signal(self, *args, **kwargs):
        super()._signal(*args, **kwargs)

    def select_one(self, *_):
        with self.ignore_events():
            val = [self.panel.value[-1]] if self.panel.value else []
            self.panel.value = val
        self._emit("selected", self.panel.value)

    def set_options(self, options):
        self.panel.options = options

    def clear(self):
        self.panel.options = []

    @property
    def value(self):
        return self.panel.value

    def set_selection(self, selection):
        self.panel.value = [selection]


class FileSelector(SigSlot):
    """Panel-based graphical file selector widget

    Instances of this widget are interactive and can be displayed in jupyter by having
    them as the output of a cell,  or in a separate browser tab using ``.show()``.
    """

    signals = [
        "protocol_changed",
        "selection_changed",
        "directory_entered",
        "home_clicked",
        "up_clicked",
        "go_clicked",
        "filters_changed",
    ]
    slots = ["set_filters", "go_home"]

    def __init__(self, url=None, filters=None, ignore=None, kwargs=None):
        """

        Parameters
        ----------
        url : str (optional)
            Initial value of the URL to populate the dialog; should include protocol
        filters : list(str) (optional)
            File endings to include in the listings. If not included, all files are
            allowed. Does not affect directories.
            If given, the endings will appear as checkboxes in the interface
        ignore : list(str) (optional)
            Regex(s) of file basename patterns to ignore, e.g., "\\." for typical
            hidden files on posix
        kwargs : dict (optional)
            To pass to file system instance
        """
        if url:
            self.init_protocol, url = split_protocol(url)
        else:
            self.init_protocol, url = "file", os.getcwd()
        self.init_url = url
        self.init_kwargs = (kwargs if isinstance(kwargs, str) else str(kwargs)) or "{}"
        self.filters = filters
        self.ignore = [re.compile(i) for i in ignore or []]
        self._fs = None
        super().__init__()

    def _setup(self):
        self.url = pn.widgets.TextInput(
            name="url",
            value=self.init_url,
            align="end",
            sizing_mode="stretch_width",
            width_policy="max",
        )
        self.protocol = pn.widgets.Select(
            options=sorted(known_implementations),
            value=self.init_protocol,
            name="protocol",
            align="center",
        )
        self.kwargs = pn.widgets.TextInput(
            name="kwargs", value=self.init_kwargs, align="center"
        )
        self.go = pn.widgets.Button(name="⇨", align="end", width=45)
        self.main = SingleSelect(size=10)
        self.home = pn.widgets.Button(name="🏠", width=40, height=30, align="end")
        self.up = pn.widgets.Button(name="‹", width=30, height=30, align="end")

        self._register(self.protocol, "protocol_changed", auto=True)
        self._register(self.go, "go_clicked", "clicks", auto=True)
        self._register(self.up, "up_clicked", "clicks", auto=True)
        self._register(self.home, "home_clicked", "clicks", auto=True)
        self._register(None, "selection_changed")
        self.main.connect("selected", self.selection_changed)
        self._register(None, "directory_entered")
        self.prev_protocol = self.protocol.value
        self.prev_kwargs = self.storage_options

        self.filter_sel = pn.widgets.CheckBoxGroup(
            value=[], options=[], inline=False, align="end", width_policy="min"
        )
        self._register(self.filter_sel, "filters_changed", auto=True)

        self.panel = pn.Column(
            pn.Row(self.protocol, self.kwargs),
            pn.Row(self.home, self.up, self.url, self.go, self.filter_sel),
            self.main.panel,
        )
        self.set_filters(self.filters)
        self.go_clicked()

    def set_filters(self, filters=None):
        self.filters = filters
        if filters:
            self.filter_sel.options = filters
            self.filter_sel.value = filters
        else:
            self.filter_sel.options = []
            self.filter_sel.value = []

    @property
    def storage_options(self):
        """Value of the kwargs box as a dictionary"""
        return ast.literal_eval(self.kwargs.value) or {}

    @property
    def fs(self):
        """Current filesystem instance"""
        if self._fs is None:
            cls = get_filesystem_class(self.protocol.value)
            self._fs = cls(**self.storage_options)
        return self._fs

    @property
    def urlpath(self):
        """URL of currently selected item"""
        return (
            (f"{self.protocol.value}://{self.main.value[0]}")
            if self.main.value
            else None
        )

    def open_file(self, mode="rb", compression=None, encoding=None):
        """Create OpenFile instance for the currently selected item

        For example, in a notebook you might do something like

        .. code-block::

            [ ]: sel = FileSelector(); sel

            # user selects their file

            [ ]: with sel.open_file('rb') as f:
            ...      out = f.read()

        Parameters
        ----------
        mode: str (optional)
            Open mode for the file.
        compression: str (optional)
            The interact with the file as compressed. Set to 'infer' to guess
            compression from the file ending
        encoding: str (optional)
            If using text mode, use this encoding; defaults to UTF8.
        """
        if self.urlpath is None:
            raise ValueError("No file selected")
        return OpenFile(self.fs, self.urlpath, mode, compression, encoding)

    def filters_changed(self, values):
        self.filters = values
        self.go_clicked()

    def selection_changed(self, *_):
        if self.urlpath is None:
            return
        if self.fs.isdir(self.urlpath):
            self.url.value = self.fs._strip_protocol(self.urlpath)
        self.go_clicked()

    def go_clicked(self, *_):
        if (
            self.prev_protocol != self.protocol.value
            or self.prev_kwargs != self.storage_options
        ):
            self._fs = None  # causes fs to be recreated
            self.prev_protocol = self.protocol.value
            self.prev_kwargs = self.storage_options
        listing = sorted(
            self.fs.ls(self.url.value, detail=True), key=lambda x: x["name"]
        )
        listing = [
            l
            for l in listing
            if not any(i.match(l["name"].rsplit("/", 1)[-1]) for i in self.ignore)
        ]
        folders = {
            "📁 " + o["name"].rsplit("/", 1)[-1]: o["name"]
            for o in listing
            if o["type"] == "directory"
        }
        files = {
            "📄 " + o["name"].rsplit("/", 1)[-1]: o["name"]
            for o in listing
            if o["type"] == "file"
        }
        if self.filters:
            files = {
                k: v
                for k, v in files.items()
                if any(v.endswith(ext) for ext in self.filters)
            }
        self.main.set_options(dict(**folders, **files))

    def protocol_changed(self, *_):
        self._fs = None
        self.main.options = []
        self.url.value = ""

    def home_clicked(self, *_):
        self.protocol.value = self.init_protocol
        self.kwargs.value = self.init_kwargs
        self.url.value = self.init_url
        self.go_clicked()

    def up_clicked(self, *_):
        self.url.value = self.fs._parent(self.url.value)
        self.go_clicked()
<|endoftext|>from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from hashlib import md5
from importlib.metadata import version
from typing import IO, TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import pathlib
    from typing import TypeGuard

    from fsspec.spec import AbstractFileSystem


DEFAULT_BLOCK_SIZE = 5 * 2**20

T = TypeVar("T")


def infer_storage_options(
    urlpath: str, inherit_storage_options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Infer storage options from URL path and merge it with existing storage
    options.

    Parameters
    ----------
    urlpath: str or unicode
        Either local absolute file path or URL (hdfs://namenode:8020/file.csv)
    inherit_storage_options: dict (optional)
        Its contents will get merged with the inferred information from the
        given path

    Returns
    -------
    Storage options dict.

    Examples
    --------
    >>> infer_storage_options('/mnt/datasets/test.csv')  # doctest: +SKIP
    {"protocol": "file", "path", "/mnt/datasets/test.csv"}
    >>> infer_storage_options(
    ...     'hdfs://username:pwd@node:123/mnt/datasets/test.csv?q=1',
    ...     inherit_storage_options={'extra': 'value'},
    ... )  # doctest: +SKIP
    {"protocol": "hdfs", "username": "username", "password": "pwd",
    "host": "node", "port": 123, "path": "/mnt/datasets/test.csv",
    "url_query": "q=1", "extra": "value"}
    """

    # Discover Windows paths including disk name in this special case.
    is_filesystem = re.match(r"^[a-zA-Z]:[\\/]", urlpath)

    # Discover URI according to RFC 3986: Scheme names consist of a
    # sequence of characters beginning with a letter and followed by
    # any combination of letters, digits, plus ("+"), period ("."),
    # or hyphen ("-").
    # https://datatracker.ietf.org/doc/html/rfc3986#section-3.1
    is_uri = re.match(r"^[a-zA-Z0-9+.-]+://", urlpath)

    if is_filesystem or is_uri is None:
        return {"protocol": "file", "path": urlpath}

    parsed_path = urlsplit(urlpath)
    protocol = parsed_path.scheme or "file"
    if parsed_path.fragment:
        path = "#".join([parsed_path.path, parsed_path.fragment])
    else:
        path = parsed_path.path
    if protocol == "file":
        # Special case parsing file protocol URL on Windows according to:
        # https://msdn.microsoft.com/en-us/library/jj710207.aspx
        windows_path = re.match(r"^/([a-zA-Z])[:|]([\\/].*)$", path)
        if windows_path:
            drive, path = windows_path.groups()
            path = f"{drive}:{path}"

    if protocol in ["http", "https"]:
        # for HTTP, we don't want to parse, as requests will anyway
        return {"protocol": protocol, "path": urlpath}

    options: dict[str, Any] = {"protocol": protocol, "path": path}

    if parsed_path.netloc:
        # Parse `hostname` from netloc manually because `parsed_path.hostname`
        # lowercases the hostname which is not always desirable (e.g. in S3):
        # https://github.com/dask/dask/issues/1417
        options["host"] = parsed_path.netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0]

        if protocol in ("s3", "s3a", "gcs", "gs"):
            options["path"] = options["host"] + options["path"]
        else:
            options["host"] = options["host"]
        if parsed_path.port:
            options["port"] = parsed_path.port
        if parsed_path.username:
            options["username"] = parsed_path.username
        if parsed_path.password:
            options["password"] = parsed_path.password

    if parsed_path.query:
        options["url_query"] = parsed_path.query
    if parsed_path.fragment:
        options["url_fragment"] = parsed_path.fragment

    if inherit_storage_options:
        update_storage_options(options, inherit_storage_options)

    return options


def update_storage_options(
    options: dict[str, Any], inherited: dict[str, Any] | None = None
) -> None:
    if not inherited:
        inherited = {}
    collisions = set(options) & set(inherited)
    if collisions:
        for collision in collisions:
            if options.get(collision) != inherited.get(collision):
                raise KeyError(
                    f"Collision between inferred and specified storage "
                    f"option:\n{collision}"
                )
    options.update(inherited)


# Compression extensions registered via fsspec.compression.register_compression
compressions: dict[str, str] = {}


def infer_compression(filename: str) -> str | None:
    """Infer compression, if available, from filename.

    Infer a named compression type, if registered and available, from filename
    extension. This includes builtin (gz, bz2, zip) compressions, as well as
    optional compressions. See fsspec.compression.register_compression.
    """
    extension = os.path.splitext(filename)[-1].strip(".").lower()
    if extension in compressions:
        return compressions[extension]
    return None


def build_name_function(max_int: float) -> Callable[[int], str]:
    """Returns a function that receives a single integer
    and returns it as a string padded by enough zero characters
    to align with maximum possible integer

    >>> name_f = build_name_function(57)

    >>> name_f(7)
    '07'
    >>> name_f(31)
    '31'
    >>> build_name_function(1000)(42)
    '0042'
    >>> build_name_function(999)(42)
    '042'
    >>> build_name_function(0)(0)
    '0'
    """
    # handle corner cases max_int is 0 or exact power of 10
    max_int += 1e-8

    pad_length = int(math.ceil(math.log10(max_int)))

    def name_function(i: int) -> str:
        return str(i).zfill(pad_length)

    return name_function


def seek_delimiter(file: IO[bytes], delimiter: bytes, blocksize: int) -> bool:
    r"""Seek current file to file start, file end, or byte after delimiter seq.

    Seeks file to next chunk delimiter, where chunks are defined on file start,
    a delimiting sequence, and file end. Use file.tell() to see location afterwards.
    Note that file start is a valid split, so must be at offset > 0 to seek for
    delimiter.

    Parameters
    ----------
    file: a file
    delimiter: bytes
        a delimiter like ``b'\n'`` or message sentinel, matching file .read() type
    blocksize: int
        Number of bytes to read from the file at once.


    Returns
    -------
    Returns True if a delimiter was found, False if at file start or end.

    """

    if file.tell() == 0:
        # beginning-of-file, return without seek
        return False

    # Interface is for binary IO, with delimiter as bytes, but initialize last
    # with result of file.read to preserve compatibility with text IO.
    last: bytes | None = None
    while True:
        current = file.read(blocksize)
        if not current:
            # end-of-file without delimiter
            return False
        full = last + current if last else current
        try:
            if delimiter in full:
                i = full.index(delimiter)
                file.seek(file.tell() - (len(full) - i) + len(delimiter))
                return True
            elif len(current) < blocksize:
                # end-of-file without delimiter
                return False
        except (OSError, ValueError):
            pass
        last = full[-len(delimiter) :]


def read_block(
    f: IO[bytes],
    offset: int,
    length: int | None,
    delimiter: bytes | None = None,
    split_before: bool = False,
) -> bytes:
    """Read a block of bytes from a file

    Parameters
    ----------
    f: File
        Open file
    offset: int
        Byte offset to start read
    length: int
        Number of bytes to read, read through end of file if None
    delimiter: bytes (optional)
        Ensure reading starts and stops at delimiter bytestring
    split_before: bool (optional)
        Start/stop read *before* delimiter bytestring.


    If using the ``delimiter=`` keyword argument we ensure that the read
    starts and stops at delimiter boundaries that follow the locations
    ``offset`` and ``offset + length``.  If ``offset`` is zero then we
    start at zero, regardless of delimiter.  The bytestring returned WILL
    include the terminating delimiter string.

    Examples
    --------

    >>> from io import BytesIO  # doctest: +SKIP
    >>> f = BytesIO(b'Alice, 100\\nBob, 200\\nCharlie, 300')  # doctest: +SKIP
    >>> read_block(f, 0, 13)  # doctest: +SKIP
    b'Alice, 100\\nBo'

    >>> read_block(f, 0, 13, delimiter=b'\\n')  # doctest: +SKIP
    b'Alice, 100\\nBob, 200\\n'

    >>> read_block(f, 10, 10, delimiter=b'\\n')  # doctest: +SKIP
    b'Bob, 200\\nCharlie, 300'
    """
    if delimiter:
        f.seek(offset)
        found_start_delim = seek_delimiter(f, delimiter, 2**16)
        if length is None:
            return f.read()
        start = f.tell()
        length -= start - offset

        f.seek(start + length)
        found_end_delim = seek_delimiter(f, delimiter, 2**16)
        end = f.tell()

        # Adjust split location to before delimiter if seek found the
        # delimiter sequence, not start or end of file.
        if found_start_delim and split_before:
            start -= len(delimiter)

        if found_end_delim and split_before:
            end -= len(delimiter)

        offset = start
        length = end - start

    f.seek(offset)

    # TODO: allow length to be None and read to the end of the file?
    assert length is not None
    b = f.read(length)
    return b


def tokenize(*args: Any, **kwargs: Any) -> str:
    """Deterministic token

    (modified from dask.base)

    >>> tokenize([1, 2, '3'])
    '9d71491b50023b06fc76928e6eddb952'

    >>> tokenize('Hello') == tokenize('Hello')
    True
    """
    if kwargs:
        args += (kwargs,)
    try:
        h = md5(str(args).encode())
    except ValueError:
        # FIPS systems: https://github.com/fsspec/filesystem_spec/issues/380
        h = md5(str(args).encode(), usedforsecurity=False)
    return h.hexdigest()


def stringify_path(filepath: str | os.PathLike[str] | pathlib.Path) -> str:
    """Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath: object to be converted

    Returns
    -------
    filepath_str: maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced according to its
    __fspath__ method.

    For backwards compatibility with older Python version, pathlib.Path
    objects are specially coerced.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if isinstance(filepath, str):
        return filepath
    elif hasattr(filepath, "__fspath__"):
        return filepath.__fspath__()
    elif hasattr(filepath, "path"):
        return filepath.path
    else:
        return filepath  # type: ignore[return-value]


def make_instance(
    cls: Callable[..., T], args: Sequence[Any], kwargs: dict[str, Any]
) -> T:
    inst = cls(*args, **kwargs)
    inst._determine_worker()  # type: ignore[attr-defined]
    return inst


def common_prefix(paths: Iterable[str]) -> str:
    """For a list of paths, find the shortest prefix common to all"""
    parts = [p.split("/") for p in paths]
    lmax = min(len(p) for p in parts)
    end = 0
    for i in range(lmax):
        end = all(p[i] == parts[0][i] for p in parts)
        if not end:
            break
    i += end
    return "/".join(parts[0][:i])


def other_paths(
    paths: list[str],
    path2: str | list[str],
    exists: bool = False,
    flatten: bool = False,
) -> list[str]:
    """In bulk file operations, construct a new file tree from a list of files

    Parameters
    ----------
    paths: list of str
        The input file tree
    path2: str or list of str
        Root to construct the new list in. If this is already a list of str, we just
        assert it has the right number of elements.
    exists: bool (optional)
        For a str destination, it is already exists (and is a dir), files should
        end up inside.
    flatten: bool (optional)
        Whether to flatten the input directory tree structure so that the output files
        are in the same directory.

    Returns
    -------
    list of str
    """

    if isinstance(path2, str):
        path2 = path2.rstrip("/")

        if flatten:
            path2 = ["/".join((path2, p.split("/")[-1])) for p in paths]
        else:
            cp = common_prefix(paths)
            if exists:
                cp = cp.rsplit("/", 1)[0]
            if not cp and all(not s.startswith("/") for s in paths):
                path2 = ["/".join([path2, p]) for p in paths]
            else:
                path2 = [p.replace(cp, path2, 1) for p in paths]
    else:
        assert len(paths) == len(path2)
    return path2


def is_exception(obj: Any) -> bool:
    return isinstance(obj, BaseException)


def isfilelike(f: Any) -> TypeGuard[IO[bytes]]:
    return all(hasattr(f, attr) for attr in ["read", "close", "tell"])


def get_protocol(url: str) -> str:
    url = stringify_path(url)
    parts = re.split(r"(\:\:|\://)", url, maxsplit=1)
    if len(parts) > 1:
        return parts[0]
    return "file"


def get_file_extension(url: str) -> str:
    url = stringify_path(url)
    # Only consider the final path component: a "." in a parent directory name
    # (e.g. "/path/to.dir/file") is not the file's extension.
    ext_parts = url.rsplit("/", 1)[-1].rsplit(".", 1)
    if len(ext_parts) > 1:
        return ext_parts[-1]
    return ""


def can_be_local(path: str) -> bool:
    """Can the given URL be used with open_local?"""
    from fsspec import get_filesystem_class

    try:
        return getattr(get_filesystem_class(get_protocol(path)), "local_file", False)
    except (ValueError, ImportError):
        # not in registry or import failed
        return False


def get_package_version_without_import(name: str) -> str | None:
    """For given package name, try to find the version without importing it

    Import and package.__version__ is still the backup here, so an import
    *might* happen.

    Returns either the version string, or None if the package
    or the version was not readily  found.
    """
    if name in sys.modules:
        mod = sys.modules[name]
        if hasattr(mod, "__version__"):
            return mod.__version__
    try:
        return version(name)
    except:  # noqa: E722
        pass
    try:
        import importlib

        mod = importlib.import_module(name)
        return mod.__version__
    except (ImportError, AttributeError):
        return None


def setup_logging(
    logger: logging.Logger | None = None,
    logger_name: str | None = None,
    level: str = "DEBUG",
    clear: bool = True,
) -> logging.Logger:
    if logger is None and logger_name is None:
        raise ValueError("Provide either logger object or logger name")
    logger = logger or logging.getLogger(logger_name)
    handle = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s -- %(message)s"
    )
    handle.setFormatter(formatter)
    if clear:
        logger.handlers.clear()
    logger.addHandler(handle)
    logger.setLevel(level)
    return logger


def _unstrip_protocol(name: str, fs: AbstractFileSystem) -> str:
    return fs.unstrip_protocol(name)


def mirror_from(
    origin_name: str, methods: Iterable[str]
) -> Callable[[type[T]], type[T]]:
    """Mirror attributes and methods from the given
    origin_name attribute of the instance to the
    decorated class"""

    def origin_getter(method: str, self: Any) -> Any:
        origin = getattr(self, origin_name)
        return getattr(origin, method)

    def wrapper(cls: type[T]) -> type[T]:
        for method in methods:
            wrapped_method = partial(origin_getter, method)
            setattr(cls, method, property(wrapped_method))
        return cls

    return wrapper


@contextlib.contextmanager
def nullcontext(obj: T) -> Iterator[T]:
    yield obj


def merge_offset_ranges(
    paths: list[str],
    starts: list[int] | int,
    ends: list[int] | int,
    max_gap: int = 0,
    max_block: int | None = None,
    sort: bool = True,
) -> tuple[list[str], list[int], list[int]]:
    """Merge adjacent byte-offset ranges when the inter-range
    gap is <= `max_gap`, and when the merged byte range does not
    exceed `max_block` (if specified). By default, this function
    will re-order the input paths and byte ranges to ensure sorted
    order. If the user can guarantee that the inputs are already
    sorted, passing `sort=False` will skip the re-ordering.
    """
    # Check input
    if not isinstance(paths, list):
        raise TypeError
    if not isinstance(starts, list):
        starts = [starts] * len(paths)
    if not isinstance(ends, list):
        ends = [ends] * len(paths)
    if len(starts) != len(paths) or len(ends) != len(paths):
        raise ValueError

    # Early Return
    if len(starts) <= 1:
        return paths, starts, ends

    starts = [s or 0 for s in starts]
    # Sort by paths and then ranges if `sort=True`
    if sort:
        paths, starts, ends = (
            list(v)
            for v in zip(
                *sorted(
                    zip(paths, starts, ends),
                )
            )
        )
    remove = []
    for i, (path, start, end) in enumerate(zip(paths, starts, ends)):
        if any(
            e is not None and p == path and start >= s and end <= e and i != i2
            for i2, (p, s, e) in enumerate(zip(paths, starts, ends))
        ):
            remove.append(i)
    paths = [p for i, p in enumerate(paths) if i not in remove]
    starts = [s for i, s in enumerate(starts) if i not in remove]
    ends = [e for i, e in enumerate(ends) if i not in remove]

    if paths:
        # Loop through the coupled `paths`, `starts`, and
        # `ends`, and merge adjacent blocks when appropriate
        new_paths = paths[:1]
        new_starts = starts[:1]
        new_ends = ends[:1]
        for i in range(1, len(paths)):
            if paths[i] == paths[i - 1] and new_ends[-1] is None:
                continue
            elif (
                paths[i] != paths[i - 1]
                or ((starts[i] - new_ends[-1]) > max_gap)
                or (max_block is not None and (ends[i] - new_starts[-1]) > max_block)
            ):
                # Cannot merge with previous block.
                # Add new `paths`, `starts`, and `ends` elements
                new_paths.append(paths[i])
                new_starts.append(starts[i])
                new_ends.append(ends[i])
            else:
                # Merge with the previous block by updating the
                # last element of `ends`
                new_ends[-1] = ends[i]
        return new_paths, new_starts, new_ends

    # `paths` is empty. Just return input lists
    return paths, starts, ends


def file_size(filelike: IO[bytes]) -> int:
    """Find length of any open read-mode file-like"""
    pos = filelike.tell()
    try:
        return filelike.seek(0, 2)
    finally:
        filelike.seek(pos)


@contextlib.contextmanager
def atomic_write(path: str, mode: str = "wb"):
    """
    A context manager that opens a temporary file next to `path` and, on exit,
    replaces `path` with the temporary file, thereby updating `path`
    atomically.
    """
    fd, fn = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=os.path.basename(path) + "-"
    )
    try:
        with open(fd, mode) as fp:
            yield fp
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(fn)
        raise
    else:
        os.replace(fn, path)


def _translate(pat, STAR, QUESTION_MARK):
    # Copied from: https://github.com/python/cpython/pull/106703.
    res: list[str] = []
    add = res.append
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        i = i + 1
        if c == "*":
            # compress consecutive `*` into one
            if (not res) or res[-1] is not STAR:
                add(STAR)
        elif c == "?":
            add(QUESTION_MARK)
        elif c == "[":
            j = i
            if j < n and pat[j] == "!":
                j = j + 1
            if j < n and pat[j] == "]":
                j = j + 1
            while j < n and pat[j] != "]":
                j = j + 1
            if j >= n:
                add("\\[")
            else:
                stuff = pat[i:j]
                if "-" not in stuff:
                    stuff = stuff.replace("\\", r"\\")
                else:
                    chunks = []
                    k = i + 2 if pat[i] == "!" else i + 1
                    while True:
                        k = pat.find("-", k, j)
                        if k < 0:
                            break
                        chunks.append(pat[i:k])
                        i = k + 1
                        k = k + 3
                    chunk = pat[i:j]
                    if chunk:
                        chunks.append(chunk)
                    else:
                        chunks[-1] += "-"
                    # Remove empty ranges -- invalid in RE.
                    for k in range(len(chunks) - 1, 0, -1):
                        if chunks[k - 1][-1] > chunks[k][0]:
                            chunks[k - 1] = chunks[k - 1][:-1] + chunks[k][1:]
                            del chunks[k]
                    # Escape backslashes and hyphens for set difference (--).
                    # Hyphens that create ranges shouldn't be escaped.
                    stuff = "-".join(
                        s.replace("\\", r"\\").replace("-", r"\-") for s in chunks
                    )
                # Escape set operations (&&, ~~ and ||).
                stuff = re.sub(r"([&~|])", r"\\\1", stuff)
                i = j + 1
                if not stuff:
                    # Empty range: never match.
                    add("(?!)")
                elif stuff == "!":
                    # Negated empty range: match any character.
                    add(".")
                else:
                    if stuff[0] == "!":
                        stuff = "^" + stuff[1:]
                    elif stuff[0] in ("^", "["):
                        stuff = "\\" + stuff
                    add(f"[{stuff}]")
        else:
            add(re.escape(c))
    assert i == n
    return res


def glob_translate(pat):
    # Copied from: https://github.com/python/cpython/pull/106703.
    # The keyword parameters' values are fixed to:
    # recursive=True, include_hidden=True, seps=None
    """Translate a pathname with shell wildcards to a regular expression."""
    if os.path.altsep:
        seps = os.path.sep + os.path.altsep
    else:
        seps = os.path.sep
    escaped_seps = "".join(map(re.escape, seps))
    any_sep = f"[{escaped_seps}]" if len(seps) > 1 else escaped_seps
    not_sep = f"[^{escaped_seps}]"
    one_last_segment = f"{not_sep}+"
    one_segment = f"{one_last_segment}{any_sep}"
    any_segments = f"(?:.+{any_sep})?"
    any_last_segments = ".*"
    results = []
    parts = re.split(any_sep, pat)
    last_part_idx = len(parts) - 1
    for idx, part in enumerate(parts):
        if part == "*":
            results.append(one_segment if idx < last_part_idx else one_last_segment)
            continue
        if part == "**":
            results.append(any_segments if idx < last_part_idx else any_last_segments)
            continue
        elif "**" in part:
            raise ValueError(
                "Invalid pattern: '**' can only be an entire path component"
            )
        if part:
            results.extend(_translate(part, f"{not_sep}*", not_sep))
        if idx < last_part_idx:
            results.append(any_sep)
    res = "".join(results)
    return rf"(?s:{res})\Z"
<|endoftext|>from __future__ import annotations

import configparser
import json
import os
import warnings
from typing import Any

conf: dict[str, dict[str, Any]] = {}
default_conf_dir = os.path.join(os.path.expanduser("~"), ".config/fsspec")
conf_dir = os.environ.get("FSSPEC_CONFIG_DIR", default_conf_dir)


def set_conf_env(conf_dict, envdict=os.environ):
    """Set config values from environment variables

    Looks for variables of the form ``FSSPEC_<protocol>`` and
    ``FSSPEC_<protocol>_<kwarg>``. For ``FSSPEC_<protocol>`` the value is parsed
    as a json dictionary and used to ``update`` the config of the
    corresponding protocol. For ``FSSPEC_<protocol>_<kwarg>`` there is no
    attempt to convert the string value, but the kwarg keys will be lower-cased.

    The ``FSSPEC_<protocol>_<kwarg>`` variables are applied after the
    ``FSSPEC_<protocol>`` ones.

    Parameters
    ----------
    conf_dict : dict(str, dict)
        This dict will be mutated
    envdict : dict-like(str, str)
        Source for the values - usually the real environment
    """
    envdict = dict(envdict)
    kwarg_keys = []
    for key in envdict:
        if key.startswith("FSSPEC_") and len(key) > 7 and key[7] != "_":
            try:
                value = json.loads(envdict[key])
                envdict[key] = value
            except json.decoder.JSONDecodeError:
                value = envdict[key]
            if key.count("_") > 1:
                kwarg_keys.append(key)
                continue
            else:
                if isinstance(value, dict):
                    _, proto = key.split("_", 1)
                    conf_dict.setdefault(proto.lower(), {}).update(value)
                else:
                    warnings.warn(
                        f"Ignoring environment variable {key} due to not being a dict:"
                        f" {type(value)}"
                    )
        elif key.startswith("FSSPEC"):
            warnings.warn(
                f"Ignoring environment variable {key} due to having an unexpected name"
            )

    for key in kwarg_keys:
        _, proto, kwarg = key.split("_", 2)
        conf_dict.setdefault(proto.lower(), {})[kwarg.lower()] = envdict[key]


def set_conf_files(cdir, conf_dict):
    """Set config values from files

    Scans for INI and JSON files in the given dictionary, and uses their
    contents to set the config. In case of repeated values, later values
    win.

    In the case of INI files, all values are strings, and these will not
    be converted.

    Parameters
    ----------
    cdir : str
        Directory to search
    conf_dict : dict(str, dict)
        This dict will be mutated
    """
    if not os.path.isdir(cdir):
        return
    allfiles = sorted(os.listdir(cdir))
    for fn in allfiles:
        if fn.endswith(".ini"):
            ini = configparser.ConfigParser()
            ini.read(os.path.join(cdir, fn))
            for key in ini:
                if key == "DEFAULT":
                    continue
                conf_dict.setdefault(key, {}).update(dict(ini[key]))
        if fn.endswith(".json"):
            with open(os.path.join(cdir, fn)) as f:
                js = json.load(f)
            for key in js:
                conf_dict.setdefault(key, {}).update(dict(js[key]))


def apply_config(cls, kwargs, conf_dict=None):
    """Supply default values for kwargs when instantiating class

    Augments the passed kwargs, by finding entries in the config dict
    which match the classes ``.protocol`` attribute (one or more str)

    Parameters
    ----------
    cls : file system implementation
    kwargs : dict
    conf_dict : dict of dict
        Typically this is the global configuration

    Returns
    -------
    dict : the modified set of kwargs
    """
    if conf_dict is None:
        conf_dict = conf
    protos = cls.protocol if isinstance(cls.protocol, (tuple, list)) else [cls.protocol]
    kw = {}
    for proto in protos:
        # default kwargs from the current state of the config
        if proto in conf_dict:
            kw.update(conf_dict[proto])
    # explicit kwargs always win
    kw.update(**kwargs)
    kwargs = kw
    return kwargs


set_conf_files(conf_dir, conf)
set_conf_env(conf)
<|endoftext|>from . import caching
from .callbacks import Callback
from .compression import available_compressions
from .core import get_fs_token_paths, open, open_files, open_local, url_to_fs
from .exceptions import FSTimeoutError
from .mapping import FSMap, get_mapper
from .registry import (
    available_protocols,
    filesystem,
    get_filesystem_class,
    register_implementation,
    registry,
)
from .spec import AbstractFileSystem

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"

__all__ = [
    "AbstractFileSystem",
    "FSTimeoutError",
    "FSMap",
    "filesystem",
    "register_implementation",
    "get_filesystem_class",
    "get_fs_token_paths",
    "get_mapper",
    "open",
    "open_files",
    "open_local",
    "registry",
    "caching",
    "Callback",
    "available_protocols",
    "available_compressions",
    "url_to_fs",
]


def process_entries():
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    if entry_points is not None:
        try:
            eps = entry_points()
        except TypeError:
            pass  # importlib-metadata < 0.8
        else:
            if hasattr(eps, "select"):  # Python 3.10+ / importlib_metadata >= 3.9.0
                specs = eps.select(group="fsspec.specs")
            else:
                specs = eps.get("fsspec.specs", [])
            registered_names = {}
            for spec in specs:
                err_msg = f"Unable to load filesystem from {spec}"
                name = spec.name
                if name in registered_names:
                    continue
                registered_names[name] = True
                register_implementation(
                    name,
                    spec.value.replace(":", "."),
                    errtxt=err_msg,
                    # We take our implementations as the ones to overload with if
                    # for some reason we encounter some, may be the same, already
                    # registered
                    clobber=True,
                )


process_entries()
<|endoftext|>from __future__ import annotations

import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path

# for backwards compat, we export cache things from here too
from fsspec.caching import (  # noqa: F401
    BaseCache,
    BlockCache,
    BytesCache,
    MMapCache,
    ReadAheadCache,
    caches,
)
from fsspec.compression import compr
from fsspec.config import conf
from fsspec.registry import available_protocols, filesystem, get_filesystem_class
from fsspec.utils import (
    _unstrip_protocol,
    build_name_function,
    infer_compression,
    stringify_path,
)

logger = logging.getLogger("fsspec")


class OpenFile:
    """
    File-like object to be used in a context

    Can layer (buffered) text-mode and compression over any file-system, which
    are typically binary-only.

    These instances are safe to serialize, as the low-level file object
    is not created until invoked using ``with``.

    Parameters
    ----------
    fs: FileSystem
        The file system to use for opening the file. Should be a subclass or duck-type
        with ``fsspec.spec.AbstractFileSystem``
    path: str
        Location to open
    mode: str like 'rb', optional
        Mode of the opened file
    compression: str or None, optional
        Compression to apply
    encoding: str or None, optional
        The encoding to use if opened in text mode.
    errors: str or None, optional
        How to handle encoding errors if opened in text mode.
    newline: None or str
        Passed to TextIOWrapper in text mode, how to handle line endings.
    autoopen: bool
        If True, calls open() immediately. Mostly used by pickle
    pos: int
        If given and autoopen is True, seek to this location immediately
    """

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        compression=None,
        encoding=None,
        errors=None,
        newline=None,
    ):
        self.fs = fs
        self.path = path
        self.mode = mode
        self.compression = get_compression(path, compression)
        self.encoding = encoding
        self.errors = errors
        self.newline = newline
        self.fobjects = []

    def __reduce__(self):
        return (
            OpenFile,
            (
                self.fs,
                self.path,
                self.mode,
                self.compression,
                self.encoding,
                self.errors,
                self.newline,
            ),
        )

    def __repr__(self):
        return f"<OpenFile '{self.path}'>"

    def __enter__(self):
        mode = self.mode.replace("t", "").replace("b", "") + "b"

        try:
            f = self.fs.open(self.path, mode=mode)
        except FileNotFoundError as e:
            if has_magic(self.path):
                raise FileNotFoundError(
                    "%s not found. The URL contains glob characters: you maybe needed\n"
                    "to pass expand=True in fsspec.open() or the storage_options of \n"
                    "your library. You can also set the config value 'open_expand'\n"
                    "before import, or fsspec.core.DEFAULT_EXPAND at runtime, to True.",
                    self.path,
                ) from e
            raise

        self.fobjects = [f]

        if self.compression is not None:
            compress = compr[self.compression]
            f = compress(f, mode=mode[0])
            self.fobjects.append(f)

        if "b" not in self.mode:
            # assume, for example, that 'r' is equivalent to 'rt' as in builtin
            f = PickleableTextIOWrapper(
                f, encoding=self.encoding, errors=self.errors, newline=self.newline
            )
            self.fobjects.append(f)

        return self.fobjects[-1]

    def __exit__(self, *args):
        self.close()

    @property
    def full_name(self):
        return _unstrip_protocol(self.path, self.fs)

    def open(self):
        """Materialise this as a real open file without context

        The OpenFile object should be explicitly closed to avoid enclosed file
        instances persisting. You must, therefore, keep a reference to the OpenFile
        during the life of the file-like it generates.
        """
        return self.__enter__()

    def close(self):
        """Close all encapsulated file objects"""
        for f in reversed(self.fobjects):
            if "r" not in self.mode and not f.closed:
                f.flush()
            f.close()
        self.fobjects.clear()


class OpenFiles(list):
    """List of OpenFile instances

    Can be used in a single context, which opens and closes all of the
    contained files. Normal list access to get the elements works as
    normal.

    A special case is made for caching filesystems - the files will
    be down/uploaded together at the start or end of the context, and
    this may happen concurrently, if the target filesystem supports it.
    """

    def __init__(self, *args, mode="rb", fs=None):
        self.mode = mode
        self.fs = fs
        self.files = []
        super().__init__(*args)

    def __enter__(self):
        if self.fs is None:
            raise ValueError("Context has already been used")

        fs = self.fs
        while True:
            if hasattr(fs, "open_many"):
                # check for concurrent cache download; or set up for upload
                self.files = fs.open_many(self)
                return self.files
            if hasattr(fs, "fs") and fs.fs is not None:
                fs = fs.fs
            else:
                break
        return [s.__enter__() for s in self]

    def __exit__(self, *args):
        fs = self.fs
        [s.__exit__(*args) for s in self]
        if "r" not in self.mode:
            while True:
                if hasattr(fs, "open_many"):
                    # check for concurrent cache upload
                    fs.commit_many(self.files)
                    return
                if hasattr(fs, "fs") and fs.fs is not None:
                    fs = fs.fs
                else:
                    break

    def __getitem__(self, item):
        out = super().__getitem__(item)
        if isinstance(item, slice):
            return OpenFiles(out, mode=self.mode, fs=self.fs)
        return out

    def __repr__(self):
        return f"<List of {len(self)} OpenFile instances>"


def open_files(
    urlpath,
    mode="rb",
    compression=None,
    encoding="utf8",
    errors=None,
    name_function=None,
    num=1,
    protocol=None,
    newline=None,
    auto_mkdir=True,
    expand=True,
    **kwargs,
):
    """Given a path or paths, return a list of ``OpenFile`` objects.

    For writing, a str path must contain the "*" character, which will be filled
    in by increasing numbers, e.g., "part*" ->  "part1", "part2" if num=2.

    For either reading or writing, can instead provide explicit list of paths.

    Parameters
    ----------
    urlpath: string or list
        Absolute or relative filepath(s). Prefix with a protocol like ``s3://``
        to read from alternative filesystems. To read from multiple files you
        can pass a globstring or a list of paths, with the caveat that they
        must all have the same protocol.
    mode: 'rb', 'wt', etc.
    compression: string or None
        If given, open file using compression codec. Can either be a compression
        name (a key in ``fsspec.compression.compr``) or "infer" to guess the
        compression from the filename suffix.
    encoding: str
        For text mode only
    errors: None or str
        Passed to TextIOWrapper in text mode
    name_function: function or None
        if opening a set of files for writing, those files do not yet exist,
        so we need to generate their names by formatting the urlpath for
        each sequence number
    num: int [1]
        if writing mode, number of files we expect to create (passed to
        name+function)
    protocol: str or None
        If given, overrides the protocol found in the URL.
    newline: bytes or None
        Used for line terminator in text mode. If None, uses system default;
        if blank, uses no translation.
    auto_mkdir: bool (True)
        If in write mode, this will ensure the target directory exists before
        writing, by calling ``fs.mkdirs(exist_ok=True)``.
    expand: bool
    **kwargs: dict
        Extra options that make sense to a particular storage connection, e.g.
        host, port, username, password, etc.

    Examples
    --------
    >>> files = open_files('2015-*-*.csv')  # doctest: +SKIP
    >>> files = open_files(
    ...     's3://bucket/2015-*-*.csv.gz', compression='gzip'
    ... )  # doctest: +SKIP

    Returns
    -------
    An ``OpenFiles`` instance, which is a list of ``OpenFile`` objects that can
    be used as a single context

    Notes
    -----
    For a full list of the available protocols and the implementations that
    they map across to see the latest online documentation:

    - For implementations built into ``fsspec`` see
      https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations
    - For implementations in separate packages see
      https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations
    """
    fs, fs_token, paths = get_fs_token_paths(
        urlpath,
        mode,
        num=num,
        name_function=name_function,
        storage_options=kwargs,
        protocol=protocol,
        expand=expand,
    )
    if fs.protocol == "file":
        fs.auto_mkdir = auto_mkdir
    elif "r" not in mode and auto_mkdir:
        parents = {fs._parent(path) for path in paths}
        for parent in parents:
            try:
                fs.makedirs(parent, exist_ok=True)
            except PermissionError:
                pass
    return OpenFiles(
        [
            OpenFile(
                fs,
                path,
                mode=mode,
                compression=compression,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
            for path in paths
        ],
        mode=mode,
        fs=fs,
    )


def _un_chain(path, kwargs):
    # Avoid a circular import
    from fsspec.implementations.chained import ChainedFileSystem

    if "::" in path:
        x = re.compile(".*[^a-z]+.*")  # test for non protocol-like single word
        known_protocols = set(available_protocols())
        bits = []

        # split on '::', then ensure each bit has a protocol
        for p in path.split("::"):
            if p in known_protocols:
                bits.append(p + "://")
            elif "://" in p or x.match(p):
                bits.append(p)
            else:
                bits.append(p + "://")
    else:
        bits = [path]

    # [[url, protocol, kwargs], ...]
    out = []
    previous_bit = None
    kwargs = kwargs.copy()

    for bit in reversed(bits):
        protocol = kwargs.pop("protocol", None) or split_protocol(bit)[0] or "file"
        cls = get_filesystem_class(protocol)
        extra_kwargs = cls._get_kwargs_from_urls(bit)
        kws = kwargs.pop(protocol, {})

        if bit is bits[0]:
            kws.update(kwargs)

        kw = dict(
            **{k: v for k, v in extra_kwargs.items() if k not in kws or v != kws[k]},
            **kws,
        )
        bit = cls._strip_protocol(bit)

        if (
            "target_protocol" not in kw
            and issubclass(cls, ChainedFileSystem)
            and not bit
        ):
            # replace bit if we are chaining and no path given
            bit = previous_bit

        out.append((bit, protocol, kw))
        previous_bit = bit

    out.reverse()
    return out


def url_to_fs(url, **kwargs):
    """
    Turn fully-qualified and potentially chained URL into filesystem instance

    Parameters
    ----------
    url : str
        The fsspec-compatible URL
    **kwargs: dict
        Extra options that make sense to a particular storage connection, e.g.
        host, port, username, password, etc.

    Returns
    -------
    filesystem : FileSystem
        The new filesystem discovered from ``url`` and created with
        ``**kwargs``.
    urlpath : str
        The file-systems-specific URL for ``url``.
    """
    url = stringify_path(url)
    # non-FS arguments that appear in fsspec.open()
    # inspect could keep this in sync with open()'s signature
    known_kwargs = {
        "compression",
        "encoding",
        "errors",
        "expand",
        "mode",
        "name_function",
        "newline",
        "num",
    }
    kwargs = {k: v for k, v in kwargs.items() if k not in known_kwargs}
    chain = _un_chain(url, kwargs)
    inkwargs = {}
    # Reverse iterate the chain, creating a nested target_* structure
    for i, ch in enumerate(reversed(chain)):
        urls, protocol, kw = ch
        if i == len(chain) - 1:
            inkwargs = dict(**kw, **inkwargs)
            continue
        inkwargs["target_options"] = dict(**kw, **inkwargs)
        inkwargs["target_protocol"] = protocol
        inkwargs["fo"] = urls
    urlpath, protocol, _ = chain[0]
    fs = filesystem(protocol, **inkwargs)
    return fs, urlpath


DEFAULT_EXPAND = conf.get("open_expand", False)


def open(
    urlpath,
    mode="rb",
    compression=None,
    encoding="utf8",
    errors=None,
    protocol=None,
    newline=None,
    expand=None,
    **kwargs,
):
    """Given a path or paths, return one ``OpenFile`` object.

    Parameters
    ----------
    urlpath: string or list
        Absolute or relative filepath. Prefix with a protocol like ``s3://``
        to read from alternative filesystems. Should not include glob
        character(s).
    mode: 'rb', 'wt', etc.
    compression: string or None
        If given, open file using compression codec. Can either be a compression
        name (a key in ``fsspec.compression.compr``) or "infer" to guess the
        compression from the filename suffix.
    encoding: str
        For text mode only
    errors: None or str
        Passed to TextIOWrapper in text mode
    protocol: str or None
        If given, overrides the protocol found in the URL.
    newline: bytes or None
        Used for line terminator in text mode. If None, uses system default;
        if blank, uses no translation.
    expand: bool or None
        Whether to regard file paths containing special glob characters as needing
        expansion (finding the first match) or absolute. Setting False allows using
        paths which do embed such characters. If None (default), this argument
        takes its value from the DEFAULT_EXPAND module variable, which takes
        its initial value from the "open_expand" config value at startup, which will
        be False if not set.
    **kwargs: dict
        Extra options that make sense to a particular storage connection, e.g.
        host, port, username, password, etc.

    Examples
    --------
    >>> openfile = open('2015-01-01.csv')  # doctest: +SKIP
    >>> openfile = open(
    ...     's3://bucket/2015-01-01.csv.gz', compression='gzip'
    ... )  # doctest: +SKIP
    >>> with openfile as f:
    ...     df = pd.read_csv(f)  # doctest: +SKIP
    ...

    Returns
    -------
    ``OpenFile`` object.

    Notes
    -----
    For a full list of the available protocols and the implementations that
    they map across to see the latest online documentation:

    - For implementations built into ``fsspec`` see
      https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations
    - For implementations in separate packages see
      https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations
    """
    expand = DEFAULT_EXPAND if expand is None else expand
    out = open_files(
        urlpath=[urlpath],
        mode=mode,
        compression=compression,
        encoding=encoding,
        errors=errors,
        protocol=protocol,
        newline=newline,
        expand=expand,
        **kwargs,
    )
    if not out:
        raise FileNotFoundError(urlpath)
    return out[0]


def open_local(
    url: str | list[str] | Path | list[Path],
    mode: str = "rb",
    **storage_options: dict,
) -> str | list[str]:
    """Open file(s) which can be resolved to local

    For files which either are local, or get downloaded upon open
    (e.g., by file caching)

    Parameters
    ----------
    url: str or list(str)
    mode: str
        Must be read mode
    storage_options:
        passed on to FS for or used by open_files (e.g., compression)
    """
    if "r" not in mode:
        raise ValueError("Can only ensure local files when reading")
    of = open_files(url, mode=mode, **storage_options)
    if not getattr(of[0].fs, "local_file", False):
        raise ValueError(
            "open_local can only be used on a filesystem which"
            " has attribute local_file=True"
        )
    with of as files:
        paths = [f.name for f in files]
    if (isinstance(url, str) and not has_magic(url)) or isinstance(url, Path):
        return paths[0]
    return paths


def get_compression(urlpath, compression):
    if compression == "infer":
        compression = infer_compression(urlpath)
    if compression is not None and compression not in compr:
        raise ValueError(f"Compression type {compression} not supported")
    return compression


def split_protocol(urlpath):
    """Return protocol, path pair"""
    urlpath = stringify_path(urlpath)
    if "://" in urlpath:
        protocol, path = urlpath.split("://", 1)
        if len(protocol) > 1:
            # excludes Windows paths
            return protocol, path
    if urlpath.startswith("data:"):
        return urlpath.split(":", 1)
    return None, urlpath


def strip_protocol(urlpath):
    """Return only path part of full URL, according to appropriate backend"""
    protocol, _ = split_protocol(urlpath)
    cls = get_filesystem_class(protocol)
    return cls._strip_protocol(urlpath)


def expand_paths_if_needed(paths, mode, num, fs, name_function):
    """Expand paths if they have a ``*`` in them (write mode) or any of ``*?[]``
    in them (read mode).

    :param paths: list of paths
    mode: str
        Mode in which to open files.
    num: int
        If opening in writing mode, number of files we expect to create.
    fs: filesystem object
    name_function: callable
        If opening in writing mode, this callable is used to generate path
        names. Names are generated for each partition by
        ``urlpath.replace('*', name_function(partition_index))``.
    :return: list of paths
    """
    expanded_paths = []
    paths = list(paths)

    if "w" in mode or "x" in mode:  # write mode
        if sum(1 for p in paths if "*" in p) > 1:
            raise ValueError(
                "When writing data, only one filename mask can be specified."
            )
        num = max(num, len(paths))

        for curr_path in paths:
            if "*" in curr_path:
                # expand using name_function
                expanded_paths.extend(_expand_paths(curr_path, name_function, num))
            else:
                expanded_paths.append(curr_path)
        # if we generated more paths that asked for, trim the list
        if len(expanded_paths) > num:
            expanded_paths = expanded_paths[:num]

    else:  # read mode
        for curr_path in paths:
            if has_magic(curr_path):
                # expand using glob
                expanded_paths.extend(fs.glob(curr_path))
            else:
                expanded_paths.append(curr_path)

    return expanded_paths


def get_fs_token_paths(
    urlpath,
    mode="rb",
    num=1,
    name_function=None,
    storage_options=None,
    protocol=None,
    expand=True,
):
    """Filesystem, deterministic token, and paths from a urlpath and options.

    Parameters
    ----------
    urlpath: string or iterable
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data.
    mode: str, optional
        Mode in which to open files.
    num: int, optional
        If opening in writing mode, number of files we expect to create.
    name_function: callable, optional
        If opening in writing mode, this callable is used to generate path
        names. Names are generated for each partition by
        ``urlpath.replace('*', name_function(partition_index))``.
    storage_options: dict, optional
        Additional keywords to pass to the filesystem class.
    protocol: str or None
        To override the protocol specifier in the URL
    expand: bool
        Expand string paths for writing, assuming the path is a directory
    """
    if isinstance(urlpath, (list, tuple, set)):
        if not urlpath:
            raise ValueError("empty urlpath sequence")
        urlpath0 = stringify_path(next(iter(urlpath)))
    else:
        urlpath0 = stringify_path(urlpath)
    storage_options = storage_options or {}
    if protocol:
        storage_options["protocol"] = protocol
    chain = _un_chain(urlpath0, storage_options or {})
    inkwargs = {}
    # Reverse iterate the chain, creating a nested target_* structure
    for i, ch in enumerate(reversed(chain)):
        urls, nested_protocol, kw = ch
        if i == len(chain) - 1:
            inkwargs = dict(**kw, **inkwargs)
            continue
        inkwargs["target_options"] = dict(**kw, **inkwargs)
        inkwargs["target_protocol"] = nested_protocol
        inkwargs["fo"] = urls
    paths, protocol, _ = chain[0]
    fs = filesystem(protocol, **inkwargs)
    if isinstance(urlpath, (list, tuple, set)):
        pchains = [
            _un_chain(stringify_path(u), storage_options or {})[0] for u in urlpath
        ]
        if len({pc[1] for pc in pchains}) > 1:
            raise ValueError("Protocol mismatch getting fs from %s", urlpath)
        paths = [pc[0] for pc in pchains]
    else:
        paths = fs._strip_protocol(paths)
    if isinstance(paths, (list, tuple, set)):
        if expand:
            paths = expand_paths_if_needed(paths, mode, num, fs, name_function)
        elif not isinstance(paths, list):
            paths = list(paths)
    else:
        if ("w" in mode or "x" in mode) and expand:
            paths = _expand_paths(paths, name_function, num)
        elif "*" in paths:
            paths = [f for f in sorted(fs.glob(paths)) if not fs.isdir(f)]
        else:
            paths = [paths]

    return fs, fs._fs_token, paths


def _expand_paths(path, name_function, num):
    if isinstance(path, str):
        if path.count("*") > 1:
            raise ValueError("Output path spec must contain exactly one '*'.")
        elif "*" not in path:
            path = os.path.join(path, "*.part")

        if name_function is None:
            name_function = build_name_function(num - 1)

        paths = [path.replace("*", name_function(i)) for i in range(num)]
        if paths != sorted(paths):
            logger.warning(
                "In order to preserve order between partitions"
                " paths created with ``name_function`` should "
                "sort to partition order"
            )
    elif isinstance(path, (tuple, list)):
        assert len(path) == num
        paths = list(path)
    else:
        raise ValueError(
            "Path should be either\n"
            "1. A list of paths: ['foo.json', 'bar.json', ...]\n"
            "2. A directory: 'foo/\n"
            "3. A path with a '*' in it: 'foo.*.json'"
        )
    return paths


class PickleableTextIOWrapper(io.TextIOWrapper):
    """TextIOWrapper cannot be pickled. This solves it.

    Requires that ``buffer`` be pickleable, which all instances of
    AbstractBufferedFile are.
    """

    def __init__(
        self,
        buffer,
        encoding=None,
        errors=None,
        newline=None,
        line_buffering=False,
        write_through=False,
    ):
        self.args = buffer, encoding, errors, newline, line_buffering, write_through
        super().__init__(*self.args)

    def __reduce__(self):
        return PickleableTextIOWrapper, self.args
<|endoftext|>from collections import deque


class Transaction:
    """Filesystem transaction write context

    Gathers files for deferred commit or discard, so that several write
    operations can be finalized semi-atomically. This works by having this
    instance as the ``.transaction`` attribute of the given filesystem
    """

    def __init__(self, fs, **kwargs):
        """
        Parameters
        ----------
        fs: FileSystem instance
        """
        self.fs = fs
        self.files = deque()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction and commit, if exit is not due to exception"""
        # only commit if there was no exception
        self.complete(commit=exc_type is None)
        if self.fs:
            self.fs._intrans = False
            self.fs._transaction = None
            self.fs = None

    def start(self):
        """Start a transaction on this FileSystem"""
        self.files = deque()  # clean up after previous failed completions
        self.fs._intrans = True

    def complete(self, commit=True):
        """Finish transaction: commit or discard all deferred files"""
        while self.files:
            f = self.files.popleft()
            if commit:
                f.commit()
            else:
                f.discard()
        self.fs._intrans = False
        self.fs._transaction = None
        self.fs = None


class FileActor:
    def __init__(self):
        self.files = []

    def commit(self):
        for f in self.files:
            f.commit()
        self.files.clear()

    def discard(self):
        for f in self.files:
            f.discard()
        self.files.clear()

    def append(self, f):
        self.files.append(f)


class DaskTransaction(Transaction):
    def __init__(self, fs):
        """
        Parameters
        ----------
        fs: FileSystem instance
        """
        import distributed

        super().__init__(fs)
        client = distributed.default_client()
        self.files = client.submit(FileActor, actor=True).result()

    def complete(self, commit=True):
        """Finish transaction: commit or discard all deferred files"""
        if commit:
            self.files.commit().result()
        else:
            self.files.discard().result()
        self.fs._intrans = False
        self.fs = None
<|endoftext|>"""Helper functions for a standard streaming compression API"""

import sys
from zipfile import ZipFile

import fsspec.utils
from fsspec.spec import AbstractBufferedFile


def noop_file(file, mode, **kwargs):
    return file


# TODO: files should also be available as contexts
# should be functions of the form func(infile, mode=, **kwargs) -> file-like
compr = {None: noop_file}


def register_compression(name, callback, extensions, force=False):
    """Register an "inferable" file compression type.

    Registers transparent file compression type for use with fsspec.open.
    Compression can be specified by name in open, or "infer"-ed for any files
    ending with the given extensions.

    Args:
        name: (str) The compression type name. Eg. "gzip".
        callback: A callable of form (infile, mode, **kwargs) -> file-like.
            Accepts an input file-like object, the target mode and kwargs.
            Returns a wrapped file-like object.
        extensions: (str, Iterable[str]) A file extension, or list of file
            extensions for which to infer this compression scheme. Eg. "gz".
        force: (bool) Force re-registration of compression type or extensions.

    Raises:
        ValueError: If name or extensions already registered, and not force.

    """
    if isinstance(extensions, str):
        extensions = [extensions]

    # Validate registration
    if name in compr and not force:
        raise ValueError(f"Duplicate compression registration: {name}")

    for ext in extensions:
        if ext in fsspec.utils.compressions and not force:
            raise ValueError(f"Duplicate compression file extension: {ext} ({name})")

    compr[name] = callback

    for ext in extensions:
        fsspec.utils.compressions[ext] = name


def unzip(infile, mode="rb", filename=None, **kwargs):
    if "r" not in mode:
        filename = filename or "file"
        z = ZipFile(infile, mode="w", **kwargs)
        fo = z.open(filename, mode="w")
        fo.close = lambda closer=fo.close: closer() or z.close()
        return fo
    z = ZipFile(infile)
    if filename is None:
        filename = z.namelist()[0]
    return z.open(filename, mode="r", **kwargs)


register_compression("zip", unzip, "zip")

try:
    from bz2 import BZ2File
except ImportError:
    pass
else:
    register_compression("bz2", BZ2File, "bz2")

try:  # pragma: no cover
    from isal import igzip

    def isal(infile, mode="rb", **kwargs):
        return igzip.IGzipFile(fileobj=infile, mode=mode, **kwargs)

    register_compression("gzip", isal, "gz")
except ImportError:
    from gzip import GzipFile

    register_compression(
        "gzip", lambda f, **kwargs: GzipFile(fileobj=f, **kwargs), "gz"
    )

try:
    from lzma import LZMAFile

    register_compression("lzma", LZMAFile, "lzma")
    register_compression("xz", LZMAFile, "xz")
except ImportError:
    pass

try:
    import lzmaffi

    register_compression("lzma", lzmaffi.LZMAFile, "lzma", force=True)
    register_compression("xz", lzmaffi.LZMAFile, "xz", force=True)
except ImportError:
    pass


class SnappyFile(AbstractBufferedFile):
    def __init__(self, infile, mode, **kwargs):
        import snappy

        super().__init__(
            fs=None, path="snappy", mode=mode.strip("b") + "b", size=999999999, **kwargs
        )
        self.infile = infile
        if "r" in mode:
            self.codec = snappy.StreamDecompressor()
        else:
            self.codec = snappy.StreamCompressor()

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        out = self.codec.add_chunk(self.buffer.read())
        self.infile.write(out)
        return True

    def seek(self, loc, whence=0):
        raise NotImplementedError("SnappyFile is not seekable")

    def seekable(self):
        return False

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        data = self.infile.read(end - start)
        return self.codec.decompress(data)


try:
    import snappy

    snappy.compress(b"")
    # Snappy may use the .sz file extension, but this is not part of the
    # standard implementation.
    register_compression("snappy", SnappyFile, [])

except (ImportError, NameError, AttributeError):
    pass

try:
    import lz4.frame

    register_compression("lz4", lz4.frame.open, "lz4")
except ImportError:
    pass

try:
    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        from backports import zstd

    register_compression("zstd", zstd.ZstdFile, "zst")
except ImportError:
    try:
        import zstandard as zstd

        def zstandard_file(infile, mode="rb"):
            if "r" in mode:
                cctx = zstd.ZstdDecompressor()
                return cctx.stream_reader(infile)
            else:
                cctx = zstd.ZstdCompressor(level=10)
                return cctx.stream_writer(infile)

        register_compression("zstd", zstandard_file, "zst")
    except ImportError:
        pass
    pass


def available_compressions():
    """Return a list of the implemented compressions."""
    return list(compr)
<|endoftext|>import asyncio
import asyncio.events
import functools
import inspect
import io
import numbers
import os
import re
import threading
from collections.abc import Iterable
from glob import has_magic
from typing import TYPE_CHECKING

from .callbacks import DEFAULT_CALLBACK
from .exceptions import FSTimeoutError
from .implementations.local import LocalFileSystem, make_path_posix, trailing_sep
from .spec import AbstractBufferedFile, AbstractFileSystem
from .utils import glob_translate, is_exception, other_paths

private = re.compile("_[^_]")
iothread = [None]  # dedicated fsspec IO thread
loop = [None]  # global event loop for any non-async instance
_lock = None  # global lock placeholder
get_running_loop = asyncio.get_running_loop


def get_lock():
    """Allocate or return a threading lock.

    The lock is allocated on first use to allow setting one lock per forked process.
    """
    global _lock
    if not _lock:
        _lock = threading.Lock()
    return _lock


def reset_lock():
    """Reset the global lock.

    This should be called only on the init of a forked process to reset the lock to
    None, enabling the new forked process to get a new lock.
    """
    global _lock

    iothread[0] = None
    loop[0] = None
    _lock = None


async def _runner(event, coro, result, timeout=None):
    timeout = timeout if timeout else None  # convert 0 or 0.0 to None
    if timeout is not None:
        coro = asyncio.wait_for(coro, timeout=timeout)
    try:
        result[0] = await coro
    except Exception as ex:
        result[0] = ex
    finally:
        event.set()


def sync(loop, func, *args, timeout=None, **kwargs):
    """
    Make loop run coroutine until it returns. Runs in other thread

    Examples
    --------
    >>> fsspec.asyn.sync(fsspec.asyn.get_loop(), func, *args,
                         timeout=timeout, **kwargs)
    """
    timeout = timeout if timeout else None  # convert 0 or 0.0 to None
    # NB: if the loop is not running *yet*, it is OK to submit work
    # and we will wait for it
    if loop is None or loop.is_closed():
        raise RuntimeError("Loop is not running")
    try:
        loop0 = asyncio.events.get_running_loop()
        if loop0 is loop:
            raise NotImplementedError("Calling sync() from within a running loop")
    except NotImplementedError:
        raise
    except RuntimeError:
        pass
    coro = func(*args, **kwargs)
    result = [None]
    event = threading.Event()
    asyncio.run_coroutine_threadsafe(_runner(event, coro, result, timeout), loop)
    while True:
        # this loops allows thread to get interrupted
        if event.wait(1):
            break
        if timeout is not None:
            timeout -= 1
            if timeout < 0:
                raise FSTimeoutError

    return_result = result[0]
    if isinstance(return_result, asyncio.TimeoutError):
        # suppress asyncio.TimeoutError, raise FSTimeoutError
        raise FSTimeoutError from return_result
    elif isinstance(return_result, BaseException):
        raise return_result
    else:
        return return_result


def sync_wrapper(func, obj=None):
    """Given a function, make so can be called in blocking contexts

    Leave obj=None if defining within a class. Pass the instance if attaching
    as an attribute of the instance.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = obj or args[0]
        return sync(self.loop, func, *args, **kwargs)

    return wrapper


def async_gen_wrapper(func, obj=None):
    """Given a async generator, make so can be called in blocking contexts"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = obj or args[0]
        gen = func(*args, **kwargs)
        while True:
            try:
                yield sync(self.loop, gen.__anext__)
            except StopAsyncIteration:
                break

    return wrapper


def get_loop():
    """Create or return the default fsspec IO loop

    The loop will be running on a separate thread.
    """
    if loop[0] is None:
        with get_lock():
            # repeat the check just in case the loop got filled between the
            # previous two calls from another thread
            if loop[0] is None:
                loop[0] = asyncio.new_event_loop()
                th = threading.Thread(target=loop[0].run_forever, name="fsspecIO")
                th.daemon = True
                th.start()
                iothread[0] = th
    return loop[0]


def reset_after_fork():
    global lock
    loop[0] = None
    iothread[0] = None
    lock = None


if hasattr(os, "register_at_fork"):
    # should be posix; this will do nothing for spawn or forkserver subprocesses
    os.register_at_fork(after_in_child=reset_after_fork)


if TYPE_CHECKING:
    import resource

    ResourceError = resource.error
else:
    try:
        import resource
    except ImportError:
        resource = None
        ResourceError = OSError
    else:
        ResourceError = getattr(resource, "error", OSError)

_DEFAULT_BATCH_SIZE = 128
_NOFILES_DEFAULT_BATCH_SIZE = 1280


def _get_batch_size(nofiles=False):
    from fsspec.config import conf

    if nofiles:
        if "nofiles_gather_batch_size" in conf:
            return conf["nofiles_gather_batch_size"]
    else:
        if "gather_batch_size" in conf:
            return conf["gather_batch_size"]
    if nofiles:
        return _NOFILES_DEFAULT_BATCH_SIZE
    if resource is None:
        return _DEFAULT_BATCH_SIZE

    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, ValueError, ResourceError):
        return _DEFAULT_BATCH_SIZE

    if soft_limit == resource.RLIM_INFINITY:
        return -1
    else:
        return soft_limit // 8


def running_async() -> bool:
    """Being executed by an event loop?"""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


async def _run_coros_in_chunks(
    coros,
    batch_size=None,
    callback=DEFAULT_CALLBACK,
    timeout=None,
    return_exceptions=False,
    nofiles=False,
):
    """Run the given coroutines in  chunks.

    Parameters
    ----------
    coros: list of coroutines to run
    batch_size: int or None
        Number of coroutines to submit/wait on simultaneously.
        If -1, then it will not be any throttling. If
        None, it will be inferred from _get_batch_size()
    callback: fsspec.callbacks.Callback instance
        Gets a relative_update when each coroutine completes
    timeout: number or None
        If given, each coroutine times out after this time. Note that, since
        there are multiple batches, the total run time of this function will in
        general be longer
    return_exceptions: bool
        Same meaning as in asyncio.gather
    nofiles: bool
        If inferring the batch_size, does this operation involve local files?
        If yes, you normally expect smaller batches.
    """

    if batch_size is None:
        batch_size = _get_batch_size(nofiles=nofiles)

    if batch_size == -1:
        batch_size = len(coros)
    elif batch_size <= 0:
        raise ValueError

    async def _run_coro(coro, i):
        try:
            return await asyncio.wait_for(coro, timeout=timeout), i
        except Exception as e:
            if not return_exceptions:
                raise
            return e, i
        finally:
            callback.relative_update(1)

    i = 0
    n = len(coros)
    results = [None] * n
    pending = set()

    while pending or i < n:
        while len(pending) < batch_size and i < n:
            pending.add(asyncio.ensure_future(_run_coro(coros[i], i)))
            i += 1

        if not pending:
            break

        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        first_exc = None
        while done:
            task = done.pop()
            try:
                result, k = await task
                results[k] = result
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc

        if first_exc is not None:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise first_exc

    return results


# these methods should be implemented as async by any async-able backend
async_methods = [
    "_ls",
    "_cat_file",
    "_get_file",
    "_put_file",
    "_rm_file",
    "_cp_file",
    "_pipe_file",
    "_expand_path",
    "_info",
    "_isfile",
    "_isdir",
    "_exists",
    "_walk",
    "_glob",
    "_find",
    "_du",
    "_size",
    "_mkdir",
    "_makedirs",
]


class AsyncFileSystem(AbstractFileSystem):
    """Async file operations, default implementations

    Passes bulk operations to asyncio.gather for concurrent operation.

    Implementations that have concurrent batch operations and/or async methods
    should inherit from this class instead of AbstractFileSystem. Docstrings are
    copied from the un-underscored method in AbstractFileSystem, if not given.
    """

    # note that methods do not have docstring here; they will be copied
    # for _* methods and inferred for overridden methods.

    async_impl = True
    mirror_sync_methods = True
    disable_throttling = False

    def __init__(self, *args, asynchronous=False, loop=None, batch_size=None, **kwargs):
        self.asynchronous = asynchronous
        self._pid = os.getpid()
        if not asynchronous:
            self._loop = loop or get_loop()
        else:
            self._loop = None
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    @property
    def loop(self):
        if self._pid != os.getpid():
            raise RuntimeError("This class is not fork-safe")
        return self._loop

    async def _rm_file(self, path, **kwargs):
        if (
            inspect.iscoroutinefunction(self._rm)
            and type(self)._rm is not AsyncFileSystem._rm
        ):
            return await self._rm(path, recursive=False, batch_size=1, **kwargs)
        raise NotImplementedError

    async def _rm(self, path, recursive=False, batch_size=None, **kwargs):
        # TODO: implement on_error
        batch_size = batch_size or self.batch_size
        path = await self._expand_path(path, recursive=recursive)
        return await _run_coros_in_chunks(
            [self._rm_file(p, **kwargs) for p in reversed(path)],
            batch_size=batch_size,
            nofiles=True,
        )

    async def _cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError

    async def _mv_file(self, path1, path2):
        await self._cp_file(path1, path2)
        await self._rm_file(path1)

    async def _copy(
        self,
        path1,
        path2,
        recursive=False,
        on_error=None,
        maxdepth=None,
        batch_size=None,
        **kwargs,
    ):
        if on_error is None and recursive:
            on_error = "ignore"
        elif on_error is None:
            on_error = "raise"

        if isinstance(path1, list) and isinstance(path2, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            paths1 = path1
            paths2 = path2
        else:
            source_is_str = isinstance(path1, str)
            paths1 = await self._expand_path(
                path1, maxdepth=maxdepth, recursive=recursive
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                paths1 = [
                    p for p in paths1 if not (trailing_sep(p) or await self._isdir(p))
                ]
                if not paths1:
                    return

            source_is_file = len(paths1) == 1
            dest_is_dir = isinstance(path2, str) and (
                trailing_sep(path2) or await self._isdir(path2)
            )

            exists = source_is_str and (
                (has_magic(path1) and source_is_file)
                or (not has_magic(path1) and dest_is_dir and not trailing_sep(path1))
            )
            paths2 = other_paths(
                paths1,
                path2,
                exists=exists,
                flatten=not source_is_str,
            )

        batch_size = batch_size or self.batch_size
        coros = [self._cp_file(p1, p2, **kwargs) for p1, p2 in zip(paths1, paths2)]
        result = await _run_coros_in_chunks(
            coros, batch_size=batch_size, return_exceptions=True, nofiles=True
        )

        for ex in filter(is_exception, result):
            if on_error == "ignore" and isinstance(ex, FileNotFoundError):
                continue
            raise ex

    async def _pipe_file(self, path, value, mode="overwrite", **kwargs):
        raise NotImplementedError

    async def _pipe(self, path, value=None, batch_size=None, **kwargs):
        if isinstance(path, str):
            path = {path: value}
        batch_size = batch_size or self.batch_size
        return await _run_coros_in_chunks(
            [self._pipe_file(k, v, **kwargs) for k, v in path.items()],
            batch_size=batch_size,
            nofiles=True,
        )

    async def _process_limits(self, url, start, end):
        """Helper for "Range"-based _cat_file"""
        size = None
        suff = False
        if start is not None and start < 0:
            # if start is negative and end None, end is the "suffix length"
            if end is None:
                end = -start
                start = ""
                suff = True
            else:
                size = size or (await self._info(url))["size"]
                start = size + start
        elif start is None:
            start = 0
        if not suff:
            if end is not None and end < 0:
                if start is not None:
                    size = size or (await self._info(url))["size"]
                    end = size + end
            elif end is None:
                end = ""
            if isinstance(end, numbers.Integral):
                end -= 1  # bytes range is inclusive
        return f"bytes={start}-{end}"

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        raise NotImplementedError

    async def _cat(
        self, path, recursive=False, on_error="raise", batch_size=None, **kwargs
    ):
        paths = await self._expand_path(path, recursive=recursive)
        coros = [self._cat_file(path, **kwargs) for path in paths]
        batch_size = batch_size or self.batch_size
        out = await _run_coros_in_chunks(
            coros, batch_size=batch_size, nofiles=True, return_exceptions=True
        )
        if on_error == "raise":
            ex = next(filter(is_exception, out), False)
            if ex:
                raise ex
        if (
            len(paths) > 1
            or isinstance(path, list)
            or paths[0] != self._strip_protocol(path)
        ):
            return {
                k: v
                for k, v in zip(paths, out)
                if on_error != "omit" or not is_exception(v)
            }
        else:
            return out[0]

    async def _cat_ranges(
        self,
        paths,
        starts,
        ends,
        max_gap=None,
        batch_size=None,
        on_error="return",
        **kwargs,
    ):
        """Get the contents of byte ranges from one or more files

        Parameters
        ----------
        paths: list
            A list of of filepaths on this filesystems
        starts, ends: int or list
            Bytes limits of the read. If using a single int, the same value will be
            used to read all the specified files.
        on_error: "return" or "raise"
            If "return" (default), any per-range exception is placed in the output
            list at the corresponding position. Otherwise the first such exception
            is raised. Matches ``AbstractFileSystem.cat_ranges``.
        """
        if max_gap is not None:
            # use utils.merge_offset_ranges
            raise NotImplementedError
        if not isinstance(paths, list):
            raise TypeError
        if not isinstance(starts, Iterable):
            starts = [starts] * len(paths)
        if not isinstance(ends, Iterable):
            ends = [ends] * len(paths)
        if len(starts) != len(paths) or len(ends) != len(paths):
            raise ValueError
        coros = [
            self._cat_file(p, start=s, end=e, **kwargs)
            for p, s, e in zip(paths, starts, ends)
        ]
        batch_size = batch_size or self.batch_size
        out = await _run_coros_in_chunks(
            coros, batch_size=batch_size, nofiles=True, return_exceptions=True
        )
        if on_error != "return":
            ex = next(filter(is_exception, out), None)
            if ex is not None:
                raise ex
        return out

    async def _put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        raise NotImplementedError

    async def _put(
        self,
        lpath,
        rpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        batch_size=None,
        maxdepth=None,
        **kwargs,
    ):
        """Copy file(s) from local.

        Copies a specific file or tree of files (if recursive=True). If rpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within.

        The put_file method will be called concurrently on a batch of files. The
        batch_size option can configure the amount of futures that can be executed
        at the same time. If it is -1, then all the files will be uploaded concurrently.
        The default can be set for this instance by passing "batch_size" in the
        constructor, or for all instances by setting the "gather_batch_size" key
        in ``fsspec.config.conf``, falling back to 1/8th of the system limit .
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            rpaths = rpath
            lpaths = lpath
        else:
            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(lpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                lpaths = [p for p in lpaths if not (trailing_sep(p) or fs.isdir(p))]
                if not lpaths:
                    return

            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (
                trailing_sep(rpath) or await self._isdir(rpath)
            )

            rpath = self._strip_protocol(rpath)
            exists = source_is_str and (
                (has_magic(lpath) and source_is_file)
                or (not has_magic(lpath) and dest_is_dir and not trailing_sep(lpath))
            )
            rpaths = other_paths(
                lpaths,
                rpath,
                exists=exists,
                flatten=not source_is_str,
            )

        is_dir = {l: os.path.isdir(l) for l in lpaths}
        rdirs = [r for l, r in zip(lpaths, rpaths) if is_dir[l]]
        file_pairs = [(l, r) for l, r in zip(lpaths, rpaths) if not is_dir[l]]

        await asyncio.gather(*[self._makedirs(d, exist_ok=True) for d in rdirs])
        batch_size = batch_size or self.batch_size

        coros = []
        callback.set_size(len(file_pairs))
        for lfile, rfile in file_pairs:
            put_file = callback.branch_coro(self._put_file)
            coros.append(put_file(lfile, rfile, **kwargs))

        return await _run_coros_in_chunks(
            coros, batch_size=batch_size, callback=callback
        )

    async def _get_file(self, rpath, lpath, **kwargs):
        raise NotImplementedError

    async def _get(
        self,
        rpath,
        lpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        maxdepth=None,
        **kwargs,
    ):
        """Copy file(s) to local.

        Copies a specific file or tree of files (if recursive=True). If lpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within. Can submit a list of paths, which may be glob-patterns
        and will be expanded.

        The get_file method will be called concurrently on a batch of files. The
        batch_size option can configure the amount of futures that can be executed
        at the same time. If it is -1, then all the files will be uploaded concurrently.
        The default can be set for this instance by passing "batch_size" in the
        constructor, or for all instances by setting the "gather_batch_size" key
        in ``fsspec.config.conf``, falling back to 1/8th of the system limit .
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            rpaths = rpath
            lpaths = lpath
        else:
            source_is_str = isinstance(rpath, str)
            # First check for rpath trailing slash as _strip_protocol removes it.
            source_not_trailing_sep = source_is_str and not trailing_sep(rpath)
            rpath = self._strip_protocol(rpath)
            rpaths = await self._expand_path(
                rpath, recursive=recursive, maxdepth=maxdepth
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                rpaths = [
                    p for p in rpaths if not (trailing_sep(p) or await self._isdir(p))
                ]
                if not rpaths:
                    return

            lpath = make_path_posix(lpath)
            source_is_file = len(rpaths) == 1
            dest_is_dir = isinstance(lpath, str) and (
                trailing_sep(lpath) or LocalFileSystem().isdir(lpath)
            )

            exists = source_is_str and (
                (has_magic(rpath) and source_is_file)
                or (not has_magic(rpath) and dest_is_dir and source_not_trailing_sep)
            )
            lpaths = other_paths(
                rpaths,
                lpath,
                exists=exists,
                flatten=not source_is_str,
            )

        [os.makedirs(os.path.dirname(lp), exist_ok=True) for lp in lpaths]
        batch_size = kwargs.pop("batch_size", self.batch_size)

        coros = []
        callback.set_size(len(lpaths))
        for lpath, rpath in zip(lpaths, rpaths):
            get_file = callback.branch_coro(self._get_file)
            coros.append(get_file(rpath, lpath, **kwargs))
        return await _run_coros_in_chunks(
            coros, batch_size=batch_size, callback=callback
        )

    async def _isfile(self, path):
        try:
            return (await self._info(path))["type"] == "file"
        except:  # noqa: E722
            return False

    async def _isdir(self, path):
        try:
            return (await self._info(path))["type"] == "directory"
        except OSError:
            return False

    async def _size(self, path):
        return (await self._info(path)).get("size", None)

    async def _sizes(self, paths, batch_size=None):
        batch_size = batch_size or self.batch_size
        return await _run_coros_in_chunks(
            [self._size(p) for p in paths], batch_size=batch_size
        )

    async def _exists(self, path, **kwargs):
        try:
            await self._info(path, **kwargs)
            return True
        except FileNotFoundError:
            return False

    async def _info(self, path, **kwargs):
        raise NotImplementedError

    async def _ls(self, path, detail=True, **kwargs):
        raise NotImplementedError

    async def _walk(self, path, maxdepth=None, topdown=True, on_error="omit", **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        path = self._strip_protocol(path)
        full_dirs = {}
        dirs = {}
        files = {}

        detail = kwargs.pop("detail", False)
        try:
            listing = await self._ls(path, detail=True, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if on_error == "raise":
                raise
            elif callable(on_error):
                on_error(e)
            if detail:
                yield path, {}, {}
            else:
                yield path, [], []
            return

        for info in listing:
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            pathname = info["name"].rstrip("/")
            name = pathname.rsplit("/", 1)[-1]
            if info["type"] == "directory" and pathname != path:
                # do not include "self" path
                full_dirs[name] = pathname
                dirs[name] = info
            elif pathname == path:
                # file-like with same name as give path
                files[""] = info
            else:
                files[name] = info

        if not detail:
            dirs = list(dirs)
            files = list(files)

        if topdown:
            # Yield before recursion if walking top down
            yield path, dirs, files

        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                if not topdown:
                    yield path, dirs, files
                return

        for d in dirs:
            async for _ in self._walk(
                full_dirs[d],
                maxdepth=maxdepth,
                detail=detail,
                topdown=topdown,
                **kwargs,
            ):
                yield _

        if not topdown:
            # Yield after recursion if walking bottom up
            yield path, dirs, files

    async def _glob(self, path, maxdepth=None, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        import re

        seps = (os.path.sep, os.path.altsep) if os.path.altsep else (os.path.sep,)
        ends_with_sep = path.endswith(seps)  # _strip_protocol strips trailing slash
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_sep or path.endswith(
            tuple(sep + "**" for sep in seps)
        )
        idx_star = path.find("*") if path.find("*") >= 0 else len(path)
        idx_qmark = path.find("?") if path.find("?") >= 0 else len(path)
        idx_brace = path.find("[") if path.find("[") >= 0 else len(path)

        min_idx = min(idx_star, idx_qmark, idx_brace)

        detail = kwargs.pop("detail", False)
        withdirs = kwargs.pop("withdirs", True)

        if not has_magic(path):
            if await self._exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: await self._info(path, **kwargs)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:min_idx]:
            first_wildcard_idx = min_idx
            min_idx = path[:min_idx].rindex("/")
            root = path[
                : min_idx + 1
            ]  # everything up to the last / before the first wildcard
            prefix = path[
                min_idx + 1 : first_wildcard_idx
            ]  # stem between last "/" and first wildcard
            depth = path[min_idx + 1 :].count("/") + 1
        else:
            root = ""
            prefix = path[:min_idx]  # stem up to the first wildcard
            depth = path[min_idx + 1 :].count("/") + 1

        if "**" in path:
            if maxdepth is not None:
                idx_double_stars = path.find("**")
                depth_double_stars = path[idx_double_stars:].count("/") + 1
                depth = depth - depth_double_stars + maxdepth
            else:
                depth = None

        # Pass the filename stem as prefix= so backends that support it such as
        # gcsfs, s3fs and adlfs can filter server-side up to the first wildcard.
        if prefix:
            kwargs["prefix"] = prefix
        allpaths = await self._find(
            root, maxdepth=depth, withdirs=withdirs, detail=True, **kwargs
        )

        pattern = glob_translate(path + ("/" if ends_with_sep else ""))
        pattern = re.compile(pattern)

        out = {
            p: info
            for p, info in sorted(allpaths.items())
            if pattern.match(
                p + "/"
                if append_slash_to_dirname and info["type"] == "directory"
                else p
            )
        }

        if detail:
            return out
        else:
            return list(out)

    async def _du(self, path, total=True, maxdepth=None, **kwargs):
        sizes = {}
        # async for?
        for f in await self._find(path, maxdepth=maxdepth, **kwargs):
            info = await self._info(f)
            sizes[info["name"]] = info["size"]
        if total:
            return sum(sizes.values())
        else:
            return sizes

    async def _find(self, path, maxdepth=None, withdirs=False, **kwargs):
        path = self._strip_protocol(path)
        out = {}
        detail = kwargs.pop("detail", False)

        # Add the root directory if withdirs is requested
        # This is needed for posix glob compliance
        if withdirs and path != "" and await self._isdir(path):
            out[path] = await self._info(path)

        # async for?
        async for _, dirs, files in self._walk(path, maxdepth, detail=True, **kwargs):
            if withdirs:
                files.update(dirs)
            out.update({info["name"]: info for name, info in files.items()})
        if not out and (await self._isfile(path)):
            # walk works on directories, but find should also return [path]
            # when path happens to be a file
            out[path] = {}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    async def _expand_path(
        self, path, recursive=False, maxdepth=None, assume_literal=False
    ):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        if isinstance(path, str):
            out = await self._expand_path([path], recursive, maxdepth)
        else:
            out = set()
            path = [self._strip_protocol(p) for p in path]
            for p in path:  # can gather here
                if not assume_literal and has_magic(p):
                    bit = set(await self._glob(p, maxdepth=maxdepth))
                    out |= bit
                    if recursive:
                        # glob call above expanded one depth so if maxdepth is defined
                        # then decrement it in expand_path call below. If it is zero
                        # after decrementing then avoid expand_path call.
                        if maxdepth is not None and maxdepth <= 1:
                            continue
                        out |= set(
                            await self._expand_path(
                                list(bit),
                                recursive=recursive,
                                maxdepth=maxdepth - 1 if maxdepth is not None else None,
                                assume_literal=True,
                            )
                        )
                    continue
                elif recursive:
                    rec = set(await self._find(p, maxdepth=maxdepth, withdirs=True))
                    out |= rec
                if p not in out and (recursive is False or (await self._exists(p))):
                    # should only check once, for the root
                    out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return sorted(out)

    async def _mkdir(self, path, create_parents=True, **kwargs):
        pass  # not necessary to implement, may not have directories

    async def _makedirs(self, path, exist_ok=False):
        pass  # not necessary to implement, may not have directories

    async def open_async(self, path, mode="rb", **kwargs):
        if "b" not in mode or kwargs.get("compression"):
            raise ValueError
        raise NotImplementedError


def mirror_sync_methods(obj):
    """Populate sync and async methods for obj

    For each method will create a sync version if the name refers to an async method
    (coroutine) and there is no override in the child class; will create an async
    method for the corresponding sync method if there is no implementation.

    Uses the methods specified in
    - async_methods: the set that an implementation is expected to provide
    - default_async_methods: that can be derived from their sync version in
      AbstractFileSystem
    - AsyncFileSystem: async-specific default coroutines
    """
    from fsspec import AbstractFileSystem

    for method in set(async_methods + dir(AsyncFileSystem)):
        if not method.startswith("_"):
            continue
        smethod = method[1:]
        if private.match(method):
            isco = inspect.iscoroutinefunction(getattr(obj, method, None))
            unsync = getattr(getattr(obj, smethod, False), "__func__", None)
            is_default = unsync is getattr(AbstractFileSystem, smethod, "")
            if isco and is_default:
                mth = sync_wrapper(getattr(obj, method), obj=obj)
            elif inspect.isasyncgenfunction(getattr(obj, method, None)) and is_default:
                mth = async_gen_wrapper(getattr(obj, method), obj=obj)
            else:
                continue
            setattr(obj, smethod, mth)
            if not mth.__doc__:
                mth.__doc__ = getattr(
                    getattr(AbstractFileSystem, smethod, None), "__doc__", ""
                )


class FSSpecCoroutineCancel(Exception):
    pass


def _dump_running_tasks(
    printout=True, cancel=True, exc=FSSpecCoroutineCancel, with_task=False
):
    import traceback

    tasks = [t for t in asyncio.tasks.all_tasks(loop[0]) if not t.done()]
    if printout:
        [task.print_stack() for task in tasks]
    out = [
        {
            "locals": task._coro.cr_frame.f_locals,
            "file": task._coro.cr_frame.f_code.co_filename,
            "firstline": task._coro.cr_frame.f_code.co_firstlineno,
            "linelo": task._coro.cr_frame.f_lineno,
            "stack": traceback.format_stack(task._coro.cr_frame),
            "task": task if with_task else None,
        }
        for task in tasks
    ]
    if cancel:
        for t in tasks:
            cbs = t._callbacks
            t.cancel()
            asyncio.futures.Future.set_exception(t, exc)
            asyncio.futures.Future.cancel(t)
            [cb[0](t) for cb in cbs]  # cancels any dependent concurrent.futures
            try:
                t._coro.throw(exc)  # exits coro, unless explicitly handled
            except exc:
                pass
    return out


class AbstractAsyncStreamedFile(AbstractBufferedFile):
    # no read buffering, and always auto-commit
    # TODO: readahead might still be useful here, but needs async version

    async def read(self, length=-1):
        """
        Return data from cache, or fetch pieces as necessary

        Parameters
        ----------
        length: int (-1)
            Number of bytes to read; if <0, all remaining bytes.
        """
        length = -1 if length is None else int(length)
        if self.mode != "rb":
            raise ValueError("File not in read mode")
        if length < 0:
            length = self.size - self.loc
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if length == 0:
            # don't even bother calling fetch
            return b""
        out = await self._fetch_range(self.loc, self.loc + length)
        self.loc += len(out)
        return out

    async def write(self, data):
        """
        Write data to buffer.

        Buffer only sent on flush() or if buffer is greater than
        or equal to blocksize.

        Parameters
        ----------
        data: bytes
            Set of bytes to be written.
        """
        if self.mode not in {"wb", "ab"}:
            raise ValueError("File not in write mode")
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self.forced:
            raise ValueError("This file has been force-flushed, can only close")
        out = self.buffer.write(data)
        self.loc += out
        if self.buffer.tell() >= self.blocksize:
            await self.flush()
        return out

    async def close(self):
        """Close file

        Finalizes writes, discards cache
        """
        if getattr(self, "_unclosable", False):
            return
        if self.closed:
            return
        if self.mode == "rb":
            self.cache = None
        else:
            if not self.forced:
                await self.flush(force=True)

            if self.fs is not None:
                self.fs.invalidate_cache(self.path)
                self.fs.invalidate_cache(self.fs._parent(self.path))

        self.closed = True

    async def flush(self, force=False):
        if self.closed:
            raise ValueError("Flush on closed file")
        if force and self.forced:
            raise ValueError("Force flush cannot be called more than once")
        if force:
            self.forced = True

        if self.mode not in {"wb", "ab"}:
            # no-op to flush on read-mode
            return

        if not force and self.buffer.tell() < self.blocksize:
            # Defer write on small block
            return

        if self.offset is None:
            # Initialize a multipart upload
            self.offset = 0
            try:
                await self._initiate_upload()
            except:
                self.closed = True
                raise

        if await self._upload_chunk(final=force) is not False:
            self.offset += self.buffer.seek(0, 2)
            self.buffer = io.BytesIO()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _fetch_range(self, start, end):
        raise NotImplementedError

    async def _initiate_upload(self):
        pass

    async def _upload_chunk(self, final=False):
        raise NotImplementedError
<|endoftext|>from functools import wraps


class Callback:
    """
    Base class and interface for callback mechanism

    This class can be used directly for monitoring file transfers by
    providing ``callback=Callback(hooks=...)`` (see the ``hooks`` argument,
    below), or subclassed for more specialised behaviour.

    Parameters
    ----------
    size: int (optional)
        Nominal quantity for the value that corresponds to a complete
        transfer, e.g., total number of tiles or total number of
        bytes
    value: int (0)
        Starting internal counter value
    hooks: dict or None
        A dict of named functions to be called on each update. The signature
        of these must be ``f(size, value, **kwargs)``
    """

    def __init__(self, size=None, value=0, hooks=None, **kwargs):
        self.size = size
        self.value = value
        self.hooks = hooks or {}
        self.kw = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        """Close callback."""

    def branched(self, path_1, path_2, **kwargs):
        """
        Return callback for child transfers

        If this callback is operating at a higher level, e.g., put, which may
        trigger transfers that can also be monitored. The function returns a callback
        that has to be passed to the child method, e.g., put_file,
        as `callback=` argument.

        The implementation uses `callback.branch` for compatibility.
        When implementing callbacks, it is recommended to override this function instead
        of `branch` and avoid calling `super().branched(...)`.

        Prefer using this function over `branch`.

        Parameters
        ----------
        path_1: str
            Child's source path
        path_2: str
            Child's destination path
        **kwargs:
            Arbitrary keyword arguments

        Returns
        -------
        callback: Callback
            A callback instance to be passed to the child method
        """
        self.branch(path_1, path_2, kwargs)
        # mutate kwargs so that we can force the caller to pass "callback=" explicitly
        return kwargs.pop("callback", DEFAULT_CALLBACK)

    def branch_coro(self, fn):
        """
        Wraps a coroutine, and pass a new child callback to it.
        """

        @wraps(fn)
        async def func(path1, path2: str, **kwargs):
            with self.branched(path1, path2, **kwargs) as child:
                return await fn(path1, path2, callback=child, **kwargs)

        return func

    def set_size(self, size):
        """
        Set the internal maximum size attribute

        Usually called if not initially set at instantiation. Note that this
        triggers a ``call()``.

        Parameters
        ----------
        size: int
        """
        self.size = size
        self.call()

    def absolute_update(self, value):
        """
        Set the internal value state

        Triggers ``call()``

        Parameters
        ----------
        value: int
        """
        self.value = value
        self.call()

    def relative_update(self, inc=1):
        """
        Delta increment the internal counter

        Triggers ``call()``

        Parameters
        ----------
        inc: int
        """
        self.value += inc
        self.call()

    def call(self, hook_name=None, **kwargs):
        """
        Execute hook(s) with current state

        Each function is passed the internal size and current value

        Parameters
        ----------
        hook_name: str or None
            If given, execute on this hook
        kwargs: passed on to (all) hook(s)
        """
        if not self.hooks:
            return
        kw = self.kw.copy()
        kw.update(kwargs)
        if hook_name:
            if hook_name not in self.hooks:
                return
            return self.hooks[hook_name](self.size, self.value, **kw)
        for hook in self.hooks.values() or []:
            hook(self.size, self.value, **kw)

    def wrap(self, iterable):
        """
        Wrap an iterable to call ``relative_update`` on each iterations

        Parameters
        ----------
        iterable: Iterable
            The iterable that is being wrapped
        """
        for item in iterable:
            self.relative_update()
            yield item

    def branch(self, path_1, path_2, kwargs):
        """
        Set callbacks for child transfers

        If this callback is operating at a higher level, e.g., put, which may
        trigger transfers that can also be monitored. The passed kwargs are
        to be *mutated* to add ``callback=``, if this class supports branching
        to children.

        Parameters
        ----------
        path_1: str
            Child's source path
        path_2: str
            Child's destination path
        kwargs: dict
            arguments passed to child method, e.g., put_file.

        Returns
        -------

        """
        return None

    def no_op(self, *_, **__):
        pass

    def __getattr__(self, item):
        """
        If undefined methods are called on this class, nothing happens
        """
        return self.no_op

    @classmethod
    def as_callback(cls, maybe_callback=None):
        """Transform callback=... into Callback instance

        For the special value of ``None``, return the global instance of
        ``NoOpCallback``. This is an alternative to including
        ``callback=DEFAULT_CALLBACK`` directly in a method signature.
        """
        if maybe_callback is None:
            return DEFAULT_CALLBACK
        return maybe_callback


class NoOpCallback(Callback):
    """
    This implementation of Callback does exactly nothing
    """

    def call(self, *args, **kwargs):
        return None


class DotPrinterCallback(Callback):
    """
    Simple example Callback implementation

    Almost identical to Callback with a hook that prints a char; here we
    demonstrate how the outer layer may print "#" and the inner layer "."
    """

    def __init__(self, chr_to_print="#", **kwargs):
        self.chr = chr_to_print
        super().__init__(**kwargs)

    def branch(self, path_1, path_2, kwargs):
        """Mutate kwargs to add new instance with different print char"""
        kwargs["callback"] = DotPrinterCallback(".")

    def call(self, **kwargs):
        """Just outputs a character"""
        print(self.chr, end="")


class TqdmCallback(Callback):
    """
    A callback to display a progress bar using tqdm

    Parameters
    ----------
    tqdm_kwargs : dict, (optional)
        Any argument accepted by the tqdm constructor.
        See the `tqdm doc <https://tqdm.github.io/docs/tqdm/#__init__>`_.
        Will be forwarded to `tqdm_cls`.
    tqdm_cls: (optional)
        subclass of `tqdm.tqdm`. If not passed, it will default to `tqdm.tqdm`.

    Examples
    --------
    >>> import fsspec
    >>> from fsspec.callbacks import TqdmCallback
    >>> fs = fsspec.filesystem("memory")
    >>> path2distant_data = "/your-path"
    >>> fs.upload(
            ".",
            path2distant_data,
            recursive=True,
            callback=TqdmCallback(),
        )

    You can forward args to tqdm using the ``tqdm_kwargs`` parameter.

    >>> fs.upload(
            ".",
            path2distant_data,
            recursive=True,
            callback=TqdmCallback(tqdm_kwargs={"desc": "Your tqdm description"}),
        )

    You can also customize the progress bar by passing a subclass of `tqdm`.

    .. code-block:: python

        class TqdmFormat(tqdm):
            '''Provides a `total_time` format parameter'''
            @property
            def format_dict(self):
                d = super().format_dict
                total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
                d.update(total_time=self.format_interval(total_time) + " in total")
                return d

    >>> with TqdmCallback(
            tqdm_kwargs={
                "desc": "desc",
                "bar_format": "{total_time}: {percentage:.0f}%|{bar}{r_bar}",
            },
            tqdm_cls=TqdmFormat,
        ) as callback:
            fs.upload(".", path2distant_data, recursive=True, callback=callback)
    """

    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        try:
            from tqdm import tqdm

        except ImportError as exce:
            raise ImportError(
                "Using TqdmCallback requires tqdm to be installed"
            ) from exce

        self._tqdm_cls = kwargs.pop("tqdm_cls", tqdm)
        self._tqdm_kwargs = tqdm_kwargs or {}
        self.tqdm = None
        super().__init__(*args, **kwargs)

    def call(self, *args, **kwargs):
        if self.tqdm is None:
            self.tqdm = self._tqdm_cls(total=self.size, **self._tqdm_kwargs)
        self.tqdm.total = self.size
        self.tqdm.update(self.value - self.tqdm.n)

    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None

    def __del__(self):
        return self.close()


DEFAULT_CALLBACK = _DEFAULT_CALLBACK = NoOpCallback()
<|endoftext|>from __future__ import annotations

import collections
import functools
import logging
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING, Any, ClassVar, Generic, NamedTuple, TypeVar

if TYPE_CHECKING:
    import mmap

    from typing_extensions import ParamSpec

    P = ParamSpec("P")
else:
    P = TypeVar("P")

T = TypeVar("T")


logger = logging.getLogger("fsspec.caching")

Fetcher = Callable[[int, int], bytes]  # Maps (start, end) to bytes
MultiFetcher = Callable[[list[int, int]], bytes]  # Maps [(start, end)] to bytes


class BaseCache:
    """Pass-though cache: doesn't keep anything, calls every time

    Acts as base class for other cachers

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    """

    name: ClassVar[str] = "none"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        self.blocksize = blocksize
        self.nblocks = 0
        self.fetcher = fetcher
        self.size = size
        self.hit_count = 0
        self.miss_count = 0
        # the bytes that we actually requested
        self.total_requested_bytes = 0

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        if start >= self.size or start >= stop:
            return b""
        return self.fetcher(start, stop)

    def _reset_stats(self) -> None:
        """Reset hit and miss counts for a more ganular report e.g. by file."""
        self.hit_count = 0
        self.miss_count = 0
        self.total_requested_bytes = 0

    def _log_stats(self) -> str:
        """Return a formatted string of the cache statistics."""
        if self.hit_count == 0 and self.miss_count == 0:
            # a cache that does nothing, this is for logs only
            return ""
        return f" , {self.name}: {self.hit_count} hits, {self.miss_count} misses, {self.total_requested_bytes} total requested bytes"

    def __repr__(self) -> str:
        # TODO: use rich for better formatting
        return f"""
        <{self.__class__.__name__}:
            block size  :   {self.blocksize}
            block count :   {self.nblocks}
            file size   :   {self.size}
            cache hits  :   {self.hit_count}
            cache misses:   {self.miss_count}
            total requested bytes: {self.total_requested_bytes}>
        """


class MMapCache(BaseCache):
    """memory-mapped sparse file cache

    Opens temporary file, which is filled blocks-wise when data is requested.
    Ensure there is enough disc space in the temporary location.

    This cache method might only work on posix

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: Fetcher
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    location: str
        Where to create the temporary file. If None, a temporary file is
        created using tempfile.TemporaryFile().
    blocks: set[int]
        Set of block numbers that have already been fetched. If None, an empty
        set is created.
    multi_fetcher: MultiFetcher
        Function of the form f([(start, end)]) which gets bytes from remote
        as specified. This function is used to fetch multiple blocks at once.
        If not specified, the fetcher function is used instead.
    """

    name = "mmap"

    def __init__(
        self,
        blocksize: int,
        fetcher: Fetcher,
        size: int,
        location: str | None = None,
        blocks: set[int] | None = None,
        multi_fetcher: MultiFetcher | None = None,
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.blocks = set() if blocks is None else blocks
        self.location = location
        self.multi_fetcher = multi_fetcher
        self.cache = self._makefile()

    def _makefile(self) -> mmap.mmap | bytearray:
        import mmap
        import tempfile

        if self.size == 0:
            return bytearray()

        # posix version
        if self.location is None or not os.path.exists(self.location):
            if self.location is None:
                fd = tempfile.TemporaryFile()
                self.blocks = set()
            else:
                fd = open(self.location, "wb+")
            fd.seek(self.size - 1)
            fd.write(b"1")
            fd.flush()
        else:
            fd = open(self.location, "r+b")

        return mmap.mmap(fd.fileno(), self.size)

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        logger.debug(f"MMap cache fetching {start}-{end}")
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        start_block = start // self.blocksize
        end_block = end // self.blocksize
        block_range = range(start_block, end_block + 1)
        # Determine which blocks need to be fetched. This sequence is sorted by construction.
        need = (i for i in block_range if i not in self.blocks)
        # Count the number of blocks already cached
        self.hit_count += sum(1 for i in block_range if i in self.blocks)

        ranges = []

        # Consolidate needed blocks.
        # Algorithm adapted from Python 2.x itertools documentation.
        # We are grouping an enumerated sequence of blocks. By comparing when the difference
        # between an ascending range (provided by enumerate) and the needed block numbers
        # we can detect when the block number skips values. The key computes this difference.
        # Whenever the difference changes, we know that we have previously cached block(s),
        # and a new group is started. In other words, this algorithm neatly groups
        # runs of consecutive block numbers so they can be fetched together.
        for _, _blocks in groupby(enumerate(need), key=lambda x: x[0] - x[1]):
            # Extract the blocks from the enumerated sequence
            _blocks = tuple(map(itemgetter(1), _blocks))
            # Compute start of first block
            sstart = _blocks[0] * self.blocksize
            # Compute the end of the last block. Last block may not be full size.
            send = min(_blocks[-1] * self.blocksize + self.blocksize, self.size)

            # Fetch bytes (could be multiple consecutive blocks)
            self.total_requested_bytes += send - sstart
            logger.debug(
                f"MMap get blocks {_blocks[0]}-{_blocks[-1]} ({sstart}-{send})"
            )
            ranges.append((sstart, send))

            # Update set of cached blocks
            self.blocks.update(_blocks)
            # Update cache statistics with number of blocks we had to cache
            self.miss_count += len(_blocks)

        if not ranges:
            return self.cache[start:end]

        if self.multi_fetcher:
            logger.debug(f"MMap get blocks {ranges}")
            for idx, r in enumerate(self.multi_fetcher(ranges)):
                sstart, send = ranges[idx]
                logger.debug(f"MMap copy block ({sstart}-{send}")
                self.cache[sstart:send] = r
        else:
            for sstart, send in ranges:
                logger.debug(f"MMap get block ({sstart}-{send}")
                self.cache[sstart:send] = self.fetcher(sstart, send)

        return self.cache[start:end]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["cache"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore instance attributes
        self.__dict__.update(state)
        self.cache = self._makefile()


class ReadAheadCache(BaseCache):
    """Cache which reads only when we get beyond a block of data

    This is a much simpler version of BytesCache, and does not attempt to
    fill holes in the cache or keep fragments alive. It is best suited to
    many small reads in a sequential order (e.g., reading lines from a file).
    """

    name = "readahead"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache = b""
        self.start = 0
        self.end = 0

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        l = end - start
        if start >= self.start and end <= self.end:
            # cache hit
            self.hit_count += 1
            return self.cache[start - self.start : end - self.start]
        elif self.start <= start < self.end:
            # partial hit
            self.miss_count += 1
            part = self.cache[start - self.start :]
            l -= len(part)
            start = self.end
        else:
            # miss
            self.miss_count += 1
            part = b""
        end = min(self.size, end + self.blocksize)
        self.total_requested_bytes += end - start
        self.cache = self.fetcher(start, end)  # new block replaces old
        self.start = start
        self.end = self.start + len(self.cache)
        return part + self.cache[:l]


class FirstChunkCache(BaseCache):
    """Caches the first block of a file only

    This may be useful for file types where the metadata is stored in the header,
    but is randomly accessed.
    """

    name = "first"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        if blocksize > size:
            # this will buffer the whole thing
            blocksize = size
        super().__init__(blocksize, fetcher, size)
        self.cache: bytes | None = None

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        start = start or 0
        if start > self.size:
            logger.debug("FirstChunkCache: requested start > file size")
            return b""

        if end is None:
            end = self.size
        end = min(end, self.size)

        if start < self.blocksize:
            if self.cache is None:
                self.miss_count += 1
                if end > self.blocksize:
                    self.total_requested_bytes += end
                    data = self.fetcher(0, end)
                    self.cache = data[: self.blocksize]
                    return data[start:]
                self.cache = self.fetcher(0, self.blocksize)
                self.total_requested_bytes += self.blocksize
            part = self.cache[start:end]
            if end > self.blocksize:
                self.total_requested_bytes += end - self.blocksize
                part += self.fetcher(self.blocksize, end)
            self.hit_count += 1
            return part
        else:
            self.miss_count += 1
            self.total_requested_bytes += end - start
            return self.fetcher(start, end)


class BlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """

    name = "blockcache"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, maxblocks: int = 32
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = functools.lru_cache(maxblocks)(self._fetch_block)

    def cache_info(self):
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        del state["_fetch_block_cached"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fetch_block_cached = functools.lru_cache(state["maxblocks"])(
            self._fetch_block
        )

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size or start >= end:
            return b""

        return self._read_cache(
            start, end, start // self.blocksize, (end - 1) // self.blocksize
        )

    def _fetch_block(self, block_number: int) -> bytes:
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(
                f"'block_number={block_number}' is greater than "
                f"the number of blocks ({self.nblocks})"
            )

        start = block_number * self.blocksize
        end = start + self.blocksize
        self.total_requested_bytes += end - start
        self.miss_count += 1
        logger.info("BlockCache fetching block %d", block_number)
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(
        self, start: int, end: int, start_block_number: int, end_block_number: int
    ) -> bytes:
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize
        if end_pos == 0:
            end_pos = self.blocksize

        self.hit_count += 1
        if start_block_number == end_block_number:
            block: bytes = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]

        else:
            # read from the initial
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]

            # intermediate blocks
            # Note: it'd be nice to combine these into one big request. However
            # that doesn't play nicely with our LRU cache.
            out.extend(
                map(
                    self._fetch_block_cached,
                    range(start_block_number + 1, end_block_number),
                )
            )

            # final block
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])

            return b"".join(out)


class BytesCache(BaseCache):
    """Cache which holds data in a in-memory bytes object

    Implements read-ahead by the block size, for semi-random reads progressing
    through the file.

    Parameters
    ----------
    trim: bool
        As we read more data, whether to discard the start of the buffer when
        we are more than a blocksize ahead of it.
    """

    name: ClassVar[str] = "bytes"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, trim: bool = True
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache = b""
        self.start: int | None = None
        self.end: int | None = None
        self.trim = trim

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        # TODO: only set start/end after fetch, in case it fails?
        # is this where retry logic might go?
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        if (
            self.start is not None
            and start >= self.start
            and self.end is not None
            and end < self.end
        ):
            # cache hit: we have all the required data
            offset = start - self.start
            self.hit_count += 1
            return self.cache[offset : offset + end - start]

        if self.blocksize:
            bend = min(self.size, end + self.blocksize)
        else:
            bend = end

        if bend == start or start > self.size:
            return b""

        if (self.start is None or start < self.start) and (
            self.end is None or end > self.end
        ):
            # First read, or extending both before and after
            self.total_requested_bytes += bend - start
            self.miss_count += 1
            self.cache = self.fetcher(start, bend)
            self.start = start
        else:
            assert self.start is not None
            assert self.end is not None
            self.miss_count += 1

            if start < self.start:
                if self.end is None or self.end - end > self.blocksize:
                    self.total_requested_bytes += bend - start
                    self.cache = self.fetcher(start, bend)
                    self.start = start
                else:
                    self.total_requested_bytes += self.start - start
                    new = self.fetcher(start, self.start)
                    self.start = start
                    self.cache = new + self.cache
            elif self.end is not None and bend > self.end:
                if self.end > self.size:
                    pass
                elif end - self.end > self.blocksize:
                    self.total_requested_bytes += bend - start
                    self.cache = self.fetcher(start, bend)
                    self.start = start
                else:
                    self.total_requested_bytes += bend - self.end
                    new = self.fetcher(self.end, bend)
                    self.cache = self.cache + new

        self.end = self.start + len(self.cache)
        offset = start - self.start
        out = self.cache[offset : offset + end - start]
        if self.trim:
            num = (self.end - self.start) // (self.blocksize + 1)
            if num > 1:
                self.start += self.blocksize * num
                self.cache = self.cache[self.blocksize * num :]
        return out

    def __len__(self) -> int:
        return len(self.cache)


class AllBytes(BaseCache):
    """Cache entire contents of the file"""

    name: ClassVar[str] = "all"

    def __init__(
        self,
        blocksize: int | None = None,
        fetcher: Fetcher | None = None,
        size: int | None = None,
        data: bytes | None = None,
    ) -> None:
        super().__init__(blocksize, fetcher, size)  # type: ignore[arg-type]
        if data is None:
            self.miss_count += 1
            self.total_requested_bytes += self.size
            data = self.fetcher(0, self.size)
        self.data = data

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        self.hit_count += 1
        return self.data[start:stop]


class KnownPartsOfAFile(BaseCache):
    """
    Cache holding known file parts.

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    data: dict
        A dictionary mapping explicit `(start, stop)` file-offset tuples
        with known bytes.
    strict: bool, default True
        Whether to fetch reads that go beyond a known byte-range boundary.
        If `False`, any read that ends outside a known part will be zero
        padded. Note that zero padding will not be used for reads that
        begin outside a known byte-range.
    """

    name: ClassVar[str] = "parts"

    def __init__(
        self,
        blocksize: int,
        fetcher: Fetcher,
        size: int,
        data: dict[tuple[int, int], bytes] | None = None,
        strict: bool = False,
        **_: Any,
    ):
        super().__init__(blocksize, fetcher, size)
        self.strict = strict

        # simple consolidation of contiguous blocks
        if data:
            old_offsets = sorted(data.keys())
            offsets = [old_offsets[0]]
            blocks = [data.pop(old_offsets[0])]
            for start, stop in old_offsets[1:]:
                start0, stop0 = offsets[-1]
                if start == stop0:
                    offsets[-1] = (start0, stop)
                    blocks[-1] += data.pop((start, stop))
                else:
                    offsets.append((start, stop))
                    blocks.append(data.pop((start, stop)))

            self.data = dict(zip(offsets, blocks))
        else:
            self.data = {}

    @property
    def size(self):
        return sum(_[1] - _[0] for _ in self.data)

    @size.setter
    def size(self, value):
        pass

    @property
    def nblocks(self):
        return len(self.data)

    @nblocks.setter
    def nblocks(self, value):
        pass

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        logger.debug("Known parts request %s %s", start, stop)
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        self.total_requested_bytes += stop - start
        out = b""
        started = False
        loc_old = 0
        for loc0, loc1 in sorted(self.data):
            if (loc0 <= start < loc1) and (loc0 <= stop <= loc1):
                # entirely within the block
                off = start - loc0
                self.hit_count += 1
                return self.data[(loc0, loc1)][off : off + stop - start]
            if stop <= loc0:
                break
            if started and loc0 > loc_old:
                # a gap where we need data
                self.miss_count += 1
                if self.strict:
                    raise ValueError
                out += b"\x00" * (loc0 - loc_old)
            if loc0 <= start < loc1:
                # found the start
                self.hit_count += 1
                off = start - loc0
                out = self.data[(loc0, loc1)][off : off + stop - start]
                started = True
            elif start < loc0 and stop > loc1:
                # the whole block
                self.hit_count += 1
                out += self.data[(loc0, loc1)]
            elif loc0 <= stop <= loc1:
                # end block
                self.hit_count += 1
                out = out + self.data[(loc0, loc1)][: stop - loc0]
                return out
            loc_old = loc1
        self.miss_count += 1
        if started and not self.strict:
            out = out + b"\x00" * (stop - loc_old)
            return out
        raise ValueError


class UpdatableLRU(Generic[P, T]):
    """
    Custom implementation of LRU cache that allows updating keys

    Used by BackgroundBlockCache
    """

    class CacheInfo(NamedTuple):
        hits: int
        misses: int
        maxsize: int
        currsize: int

    def __init__(self, func: Callable[P, T], max_size: int = 128) -> None:
        self._cache: OrderedDict[Any, T] = collections.OrderedDict()
        self._func = func
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if kwargs:
            raise TypeError(f"Got unexpected keyword argument {kwargs.keys()}")
        with self._lock:
            if args in self._cache:
                self._cache.move_to_end(args)
                self._hits += 1
                return self._cache[args]

        result = self._func(*args, **kwargs)

        with self._lock:
            self._cache[args] = result
            self._misses += 1
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return result

    def is_key_cached(self, *args: Any) -> bool:
        with self._lock:
            return args in self._cache

    def add_key(self, result: T, *args: Any) -> None:
        with self._lock:
            self._cache[args] = result
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def cache_info(self) -> UpdatableLRU.CacheInfo:
        with self._lock:
            return self.CacheInfo(
                maxsize=self._max_size,
                currsize=len(self._cache),
                hits=self._hits,
                misses=self._misses,
            )


class BackgroundBlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks with pre-loading of
    the next block in the background.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored. If the
    next block is not in cache, it is loaded in a separate thread
    in non-blocking way.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """

    name: ClassVar[str] = "background"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, maxblocks: int = 32
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, maxblocks)

        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number: int | None = None
        self._fetch_future: Future[bytes] | None = None
        self._fetch_future_lock = threading.Lock()
        self._closed = False

    def cache_info(self) -> UpdatableLRU.CacheInfo:
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def close(self) -> None:
        """Cancel pending work and shut down the background worker."""
        with self._fetch_future_lock:
            if self._closed:
                return
            self._closed = True
            future = self._fetch_future
            self._fetch_future = None
            self._fetch_future_block_number = None

        if future is not None:
            future.cancel()
        self._thread_executor.shutdown(wait=True, cancel_futures=True)

        # UpdatableLRU stores a bound method and otherwise forms a reference cycle.
        del self._fetch_block_cached

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        del state["_fetch_block_cached"]
        del state["_thread_executor"]
        del state["_fetch_future_block_number"]
        del state["_fetch_future"]
        del state["_fetch_future_lock"]
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, state["maxblocks"])
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number = None
        self._fetch_future = None
        self._fetch_future_lock = threading.Lock()
        self._closed = False

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size or start >= end:
            return b""

        # byte position -> block numbers
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize

        fetch_future_block_number = None
        fetch_future = None
        with self._fetch_future_lock:
            # Background thread is running. Check we we can or must join it.
            if self._fetch_future is not None:
                assert self._fetch_future_block_number is not None
                if self._fetch_future.done():
                    logger.info("BlockCache joined background fetch without waiting.")
                    self._fetch_block_cached.add_key(
                        self._fetch_future.result(), self._fetch_future_block_number
                    )
                    # Cleanup the fetch variables. Done with fetching the block.
                    self._fetch_future_block_number = None
                    self._fetch_future = None
                else:
                    # Must join if we need the block for the current fetch
                    must_join = bool(
                        start_block_number
                        <= self._fetch_future_block_number
                        <= end_block_number
                    )
                    if must_join:
                        # Copy to the local variables to release lock
                        # before waiting for result
                        fetch_future_block_number = self._fetch_future_block_number
                        fetch_future = self._fetch_future

                        # Cleanup the fetch variables. Have a local copy.
                        self._fetch_future_block_number = None
                        self._fetch_future = None

        # Need to wait for the future for the current read
        if fetch_future is not None:
            logger.info("BlockCache waiting for background fetch.")
            # Wait until result and put it in cache
            self._fetch_block_cached.add_key(
                fetch_future.result(), fetch_future_block_number
            )

        # these are cached, so safe to do multiple calls for the same start and end.
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)

        # fetch next block in the background if nothing is running in the background,
        # the block is within file and it is not already cached
        end_block_plus_1 = end_block_number + 1
        with self._fetch_future_lock:
            if (
                self._fetch_future is None
                and end_block_plus_1 <= self.nblocks
                and not self._fetch_block_cached.is_key_cached(end_block_plus_1)
            ):
                self._fetch_future_block_number = end_block_plus_1
                self._fetch_future = self._thread_executor.submit(
                    self._fetch_block, end_block_plus_1, "async"
                )

        return self._read_cache(
            start,
            end,
            start_block_number=start_block_number,
            end_block_number=end_block_number,
        )

    def _fetch_block(self, block_number: int, log_info: str = "sync") -> bytes:
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(
                f"'block_number={block_number}' is greater than "
                f"the number of blocks ({self.nblocks})"
            )

        start = block_number * self.blocksize
        end = start + self.blocksize
        logger.info("BlockCache fetching block (%s) %d", log_info, block_number)
        self.total_requested_bytes += end - start
        self.miss_count += 1
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(
        self, start: int, end: int, start_block_number: int, end_block_number: int
    ) -> bytes:
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize

        # kind of pointless to count this as a hit, but it is
        self.hit_count += 1

        if start_block_number == end_block_number:
            block = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]

        else:
