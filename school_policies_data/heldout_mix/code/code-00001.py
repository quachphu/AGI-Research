    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
    compression_method: str | None
    if isinstance(compression, Mapping):
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop("method")
        except KeyError as err:
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        compression_args = {}
        compression_method = compression
    return compression_method, compression_args


@doc(compression_options=_shared_docs["compression_options"] % "filepath_or_buffer")
def infer_compression(
    filepath_or_buffer: FilePath | BaseBuffer, compression: str | None
) -> str | None:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    Returns
    -------
    string or None

    Raises
    ------
    ValueError on invalid compression specified.
    """
    if compression is None:
        return None

    # Infer compression
    if compression == "infer":
        # Convert all path types (e.g. pathlib.Path) to strings
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        if not isinstance(filepath_or_buffer, str):
            # Cannot infer compression of a buffer, assume no compression
            return None

        # Infer compression from the filename/URL extension
        for extension, compression in extension_to_compression.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _supported_compressions:
        return compression

    valid = ["infer", None] + sorted(_supported_compressions)
    msg = (
        f"Unrecognized compression type: {compression}\n"
        f"Valid compression types are {valid}"
    )
    raise ValueError(msg)


def check_parent_directory(path: Path | str) -> None:
    """
    Check if parent directory of a file exists, raise OSError if it does not

    Parameters
    ----------
    path: Path or str
        Path to check parent directory of
    """
    parent = Path(path).parent
    if not parent.is_dir():
        raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[False],
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[bytes]:
    ...


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[True] = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str]:
    ...


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str] | IOHandles[bytes]:
    ...


@doc(compression_options=_shared_docs["compression_options"] % "path_or_buf")
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = None,
    compression: CompressionOptions | None = None,
    memory_map: bool = False,
    is_text: bool = True,
    errors: str | None = None,
    storage_options: StorageOptions | None = None,
) -> IOHandles[str] | IOHandles[bytes]:
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : str or file handle
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str or None
        Encoding to use.
    {compression_options}

           May be a dict with key 'method' as compression mode
           and other keys as compression options if compression
           mode is 'zip'.

           Passing compression options as keys in dict is
           supported for compression modes 'gzip', 'bz2', 'zstd' and 'zip'.

        .. versionchanged:: 1.4.0 Zstandard support.

    memory_map : bool, default False
        See parsers._parser_params for more information. Only used by read_csv.
    is_text : bool, default True
        Whether the type of the content passed to the file/buffer is string or
        bytes. This is not the same as `"b" not in mode`. If a string content is
        passed to a binary file/buffer, a wrapper is inserted.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    storage_options: StorageOptions = None
        Passed to _get_filepath_or_buffer

    Returns the dataclass IOHandles
    """
    # Windows does not default to utf-8. Set to utf-8 for a consistent behavior
    encoding = encoding or "utf-8"

    errors = errors or "strict"

    # read_csv does not know whether the buffer is opened in binary/text mode
    if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
        mode += "b"

    # validate encoding and errors
    codecs.lookup(encoding)
    if isinstance(errors, str):
        codecs.lookup_error(errors)

    # open URLs
    ioargs = _get_filepath_or_buffer(
        path_or_buf,
        encoding=encoding,
        compression=compression,
        mode=mode,
        storage_options=storage_options,
    )

    handle = ioargs.filepath_or_buffer
    handles: list[BaseBuffer]

    # memory mapping needs to be the first step
    # only used for read_csv
    handle, memory_map, handles = _maybe_memory_map(handle, memory_map)

    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop("method")

    # Only for write methods
    if "r" not in mode and is_path:
        check_parent_directory(str(handle))

    if compression:
        if compression != "zstd":
            # compression libraries do not like an explicit text-mode
            ioargs.mode = ioargs.mode.replace("t", "")
        elif compression == "zstd" and "b" not in ioargs.mode:
            # python-zstandard defaults to text mode, but we always expect
            # compression libraries to use binary mode.
            ioargs.mode += "b"

        # GZ Compression
        if compression == "gzip":
            if isinstance(handle, str):
                # error: Incompatible types in assignment (expression has type
                # "GzipFile", variable has type "Union[str, BaseBuffer]")
                handle = gzip.GzipFile(  # type: ignore[assignment]
                    filename=handle,
                    mode=ioargs.mode,
                    **compression_args,
                )
            else:
                handle = gzip.GzipFile(
                    # No overload variant of "GzipFile" matches argument types
                    # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                    fileobj=handle,  # type: ignore[call-overload]
                    mode=ioargs.mode,
                    **compression_args,
                )

        # BZ Compression
        elif compression == "bz2":
            # Overload of "BZ2File" to handle pickle protocol 5
            # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
            handle = get_bz2_file()(  # type: ignore[call-overload]
                handle,
                mode=ioargs.mode,
                **compression_args,
            )

        # ZIP Compression
        elif compression == "zip":
            # error: Argument 1 to "_BytesZipFile" has incompatible type
            # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
            # ReadBuffer[bytes], WriteBuffer[bytes]]"
            handle = _BytesZipFile(
                handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
            )
            if handle.buffer.mode == "r":
                handles.append(handle)
                zip_names = handle.buffer.namelist()
                if len(zip_names) == 1:
                    handle = handle.buffer.open(zip_names.pop())
                elif not zip_names:
                    raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in ZIP file. "
                        f"Only one file per ZIP: {zip_names}"
                    )

        # TAR Encoding
        elif compression == "tar":
            compression_args.setdefault("mode", ioargs.mode)
            if isinstance(handle, str):
                handle = _BytesTarFile(name=handle, **compression_args)
            else:
                # error: Argument "fileobj" to "_BytesTarFile" has incompatible
                # type "BaseBuffer"; expected "Union[ReadBuffer[bytes],
                # WriteBuffer[bytes], None]"
                handle = _BytesTarFile(
                    fileobj=handle, **compression_args  # type: ignore[arg-type]
                )
            assert isinstance(handle, _BytesTarFile)
            if "r" in handle.buffer.mode:
                handles.append(handle)
                files = handle.buffer.getnames()
                if len(files) == 1:
                    file = handle.buffer.extractfile(files[0])
                    assert file is not None
                    handle = file
                elif not files:
                    raise ValueError(f"Zero files found in TAR archive {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in TAR archive. "
                        f"Only one file per TAR archive: {files}"
                    )

        # XZ Compression
        elif compression == "xz":
            # error: Argument 1 to "LZMAFile" has incompatible type "Union[str,
            # BaseBuffer]"; expected "Optional[Union[Union[str, bytes, PathLike[str],
            # PathLike[bytes]], IO[bytes]], None]"
            handle = get_lzma_file()(
                handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
            )

        # Zstd Compression
        elif compression == "zstd":
            zstd = import_optional_dependency("zstandard")
            if "r" in ioargs.mode:
                open_args = {"dctx": zstd.ZstdDecompressor(**compression_args)}
            else:
                open_args = {"cctx": zstd.ZstdCompressor(**compression_args)}
            handle = zstd.open(
                handle,
                mode=ioargs.mode,
                **open_args,
            )

        # Unrecognized Compression
        else:
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        assert not isinstance(handle, str)
        handles.append(handle)

    elif isinstance(handle, str):
        # Check whether the filename is to be opened in binary mode.
        # Binary mode does not support 'encoding' and 'newline'.
        if ioargs.encoding and "b" not in ioargs.mode:
            # Encoding
            handle = open(
                handle,
                ioargs.mode,
                encoding=ioargs.encoding,
                errors=errors,
                newline="",
            )
        else:
            # Binary mode
            handle = open(handle, ioargs.mode)
        handles.append(handle)

    # Convert BytesIO or file objects passed with an encoding
    is_wrapped = False
    if not is_text and ioargs.mode == "rb" and isinstance(handle, TextIOBase):
        # not added to handles as it does not open/buffer resources
        handle = _BytesIOWrapper(
            handle,
            encoding=ioargs.encoding,
        )
    elif is_text and (
        compression or memory_map or _is_binary_mode(handle, ioargs.mode)
    ):
        if (
            not hasattr(handle, "readable")
            or not hasattr(handle, "writable")
            or not hasattr(handle, "seekable")
        ):
            handle = _IOWrapper(handle)
        # error: Argument 1 to "TextIOWrapper" has incompatible type
        # "_IOWrapper"; expected "IO[bytes]"
        handle = TextIOWrapper(
            handle,  # type: ignore[arg-type]
            encoding=ioargs.encoding,
            errors=errors,
            newline="",
        )
        handles.append(handle)
        # only marked as wrapped when the caller provided a handle
        is_wrapped = not (
            isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close
        )

    if "r" in ioargs.mode and not hasattr(handle, "read"):
        raise TypeError(
            "Expected file path name or file-like object, "
            f"got {type(ioargs.filepath_or_buffer)} type"
        )

    handles.reverse()  # close the most recently added buffer first
    if ioargs.should_close:
        assert not isinstance(ioargs.filepath_or_buffer, str)
        handles.append(ioargs.filepath_or_buffer)

    return IOHandles(
        # error: Argument "handle" to "IOHandles" has incompatible type
        # "Union[TextIOWrapper, GzipFile, BaseBuffer, typing.IO[bytes],
        # typing.IO[Any]]"; expected "pandas._typing.IO[Any]"
        handle=handle,  # type: ignore[arg-type]
        # error: Argument "created_handles" to "IOHandles" has incompatible type
        # "List[BaseBuffer]"; expected "List[Union[IO[bytes], IO[str]]]"
        created_handles=handles,  # type: ignore[arg-type]
        is_wrapped=is_wrapped,
        compression=ioargs.compression,
    )


# error: Definition of "__enter__" in base class "IOBase" is incompatible
# with definition in base class "BinaryIO"
class _BufferedWriter(BytesIO, ABC):  # type: ignore[misc]
    """
    Some objects do not support multiple .write() calls (TarFile and ZipFile).
    This wrapper writes to the underlying buffer on close.
    """

    buffer = BytesIO()

    @abstractmethod
    def write_to_buffer(self) -> None:
        ...

    def close(self) -> None:
        if self.closed:
            # already closed
            return
        if self.getbuffer().nbytes:
            # write to buffer
            self.seek(0)
            with self.buffer:
                self.write_to_buffer()
        else:
            self.buffer.close()
        super().close()


class _BytesTarFile(_BufferedWriter):
    def __init__(
        self,
        name: str | None = None,
        mode: Literal["r", "a", "w", "x"] = "r",
        fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None = None,
        archive_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.archive_name = archive_name
        self.name = name
        # error: Incompatible types in assignment (expression has type "TarFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: tarfile.TarFile = tarfile.TarFile.open(  # type: ignore[assignment]
            name=name,
            mode=self.extend_mode(mode),
            fileobj=fileobj,
            **kwargs,
        )

    def extend_mode(self, mode: str) -> str:
        mode = mode.replace("b", "")
        if mode != "w":
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in (".gz", ".xz", ".bz2"):
                mode = f"{mode}:{suffix[1:]}"
        return mode

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
        if self.name is None:
            return None

        filename = Path(self.name)
        if filename.suffix == ".tar":
            return filename.with_suffix("").name
        elif filename.suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            return filename.with_suffix("").with_suffix("").name
        return filename.name

    def write_to_buffer(self) -> None:
        # TarFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "tar"
        tarinfo = tarfile.TarInfo(name=archive_name)
        tarinfo.size = len(self.getvalue())
        self.buffer.addfile(tarinfo, self)


class _BytesZipFile(_BufferedWriter):
    def __init__(
        self,
        file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
        mode: str,
        archive_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        mode = mode.replace("b", "")
        self.archive_name = archive_name

        kwargs.setdefault("compression", zipfile.ZIP_DEFLATED)
        # error: Incompatible types in assignment (expression has type "ZipFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
            file, mode, **kwargs
        )

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.zip, because that causes confusion (GH39465).
        """
        if isinstance(self.buffer.filename, (os.PathLike, str)):
            filename = Path(self.buffer.filename)
            if filename.suffix == ".zip":
                return filename.with_suffix("").name
            return filename.name
        return None

    def write_to_buffer(self) -> None:
        # ZipFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "zip"
        self.buffer.writestr(archive_name, self.getvalue())


class _IOWrapper:
    # TextIOWrapper is overly strict: it request that the buffer has seekable, readable,
    # and writable. If we have a read-only buffer, we shouldn't need writable and vice
    # versa. Some buffers, are seek/read/writ-able but they do not have the "-able"
    # methods, e.g., tempfile.SpooledTemporaryFile.
    # If a buffer does not have the above "-able" methods, we simple assume they are
    # seek/read/writ-able.
    def __init__(self, buffer: BaseBuffer) -> None:
        self.buffer = buffer

    def __getattr__(self, name: str):
        return getattr(self.buffer, name)

    def readable(self) -> bool:
        if hasattr(self.buffer, "readable"):
            return self.buffer.readable()
        return True

    def seekable(self) -> bool:
        if hasattr(self.buffer, "seekable"):
            return self.buffer.seekable()
        return True

    def writable(self) -> bool:
        if hasattr(self.buffer, "writable"):
            return self.buffer.writable()
        return True


class _BytesIOWrapper:
    # Wrapper that wraps a StringIO buffer and reads bytes from it
    # Created for compat with pyarrow read_csv
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = "utf-8") -> None:
        self.buffer = buffer
        self.encoding = encoding
        # Because a character can be represented by more than 1 byte,
        # it is possible that reading will produce more bytes than n
        # We store the extra bytes in this overflow variable, and append the
        # overflow to the front of the bytestring the next time reading is performed
        self.overflow = b""

    def __getattr__(self, attr: str):
        return getattr(self.buffer, attr)

    def read(self, n: int | None = -1) -> bytes:
        assert self.buffer is not None
        bytestring = self.buffer.read(n).encode(self.encoding)
        # When n=-1/n greater than remaining bytes: Read entire file/rest of file
        combined_bytestring = self.overflow + bytestring
        if n is None or n < 0 or n >= len(combined_bytestring):
            self.overflow = b""
            return combined_bytestring
        else:
            to_return = combined_bytestring[:n]
            self.overflow = combined_bytestring[n:]
            return to_return


def _maybe_memory_map(
    handle: str | BaseBuffer, memory_map: bool
) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
    """Try to memory map file/buffer."""
    handles: list[BaseBuffer] = []
    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
    if not memory_map:
        return handle, memory_map, handles

    # mmap used by only read_csv
    handle = cast(ReadCsvBuffer, handle)

    # need to open the file first
    if isinstance(handle, str):
        handle = open(handle, "rb")
        handles.append(handle)

    try:
        # open mmap and adds *-able
        # error: Argument 1 to "_IOWrapper" has incompatible type "mmap";
        # expected "BaseBuffer"
        wrapped = _IOWrapper(
            mmap.mmap(
                handle.fileno(), 0, access=mmap.ACCESS_READ  # type: ignore[arg-type]
            )
        )
    finally:
        for handle in reversed(handles):
            # error: "BaseBuffer" has no attribute "close"
            handle.close()  # type: ignore[attr-defined]

    return wrapped, memory_map, [wrapped]


def file_exists(filepath_or_buffer: FilePath | BaseBuffer) -> bool:
    """Test whether file exists."""
    exists = False
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if not isinstance(filepath_or_buffer, str):
        return exists
    try:
        exists = os.path.exists(filepath_or_buffer)
        # gh-5874: if the filepath is too long will raise here
    except (TypeError, ValueError):
        pass
    return exists


def _is_binary_mode(handle: FilePath | BaseBuffer, mode: str) -> bool:
    """Whether the handle is opened in binary mode"""
    # specified by user
    if "t" in mode or "b" in mode:
        return "b" in mode

    # exceptions
    text_classes = (
        # classes that expect string but have 'b' in mode
        codecs.StreamWriter,
        codecs.StreamReader,
        codecs.StreamReaderWriter,
    )
    if issubclass(type(handle), text_classes):
        return False

    return isinstance(handle, _get_binary_io_classes()) or "b" in getattr(
        handle, "mode", mode
    )


@functools.lru_cache
def _get_binary_io_classes() -> tuple[type, ...]:
    """IO classes that that expect bytes"""
    binary_classes: tuple[type, ...] = (BufferedIOBase, RawIOBase)

    # python-zstandard doesn't use any of the builtin base classes; instead we
    # have to use the `zstd.ZstdDecompressionReader` class for isinstance checks.
    # Unfortunately `zstd.ZstdDecompressionReader` isn't exposed by python-zstandard
    # so we have to get it from a `zstd.ZstdDecompressor` instance.
    # See also https://github.com/indygreg/python-zstandard/pull/165.
    zstd = import_optional_dependency("zstandard", errors="ignore")
    if zstd is not None:
        with zstd.ZstdDecompressor().stream_reader(b"") as reader:
            binary_classes += (type(reader),)

    return binary_classes


def is_potential_multi_index(
    columns: Sequence[Hashable] | MultiIndex,
    index_col: bool | Sequence[int] | None = None,
) -> bool:
    """
    Check whether or not the `columns` parameter
    could be converted into a MultiIndex.

    Parameters
    ----------
    columns : array-like
        Object which may or may not be convertible into a MultiIndex
    index_col : None, bool or list, optional
        Column or columns to use as the (possibly hierarchical) index

    Returns
    -------
    bool : Whether or not columns could become a MultiIndex
    """
    if index_col is None or isinstance(index_col, bool):
        index_col = []

    return bool(
        len(columns)
        and not isinstance(columns, ABCMultiIndex)
        and all(isinstance(c, tuple) for c in columns if c not in list(index_col))
    )


def dedup_names(
    names: Sequence[Hashable], is_potential_multiindex: bool
) -> Sequence[Hashable]:
    """
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    but a custom pattern may be supported in the future.

    Examples
    --------
    >>> dedup_names(["x", "y", "x", "x"], is_potential_multiindex=False)
    ['x', 'y', 'x.1', 'x.2']
    """
    names = list(names)  # so we can index
    counts: DefaultDict[Hashable, int] = defaultdict(int)

    for i, col in enumerate(names):
        cur_count = counts[col]

        while cur_count > 0:
            counts[col] = cur_count + 1

            if is_potential_multiindex:
                # for mypy
                assert isinstance(col, tuple)
                col = col[:-1] + (f"{col[-1]}.{cur_count}",)
            else:
                col = f"{col}.{cur_count}"
            cur_count = counts[col]

        names[i] = col
        counts[col] = cur_count + 1

    return names
<|endoftext|># -*- coding: utf-8 -*-
"""
This module offers a parser for ISO-8601 strings

It is intended to support all valid date, time and datetime formats per the
ISO-8601 specification.

..versionadded:: 2.7.0
"""
from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz

from functools import wraps

import re
import six

__all__ = ["isoparse", "isoparser"]


def _takes_ascii(f):
    @wraps(f)
    def func(self, str_in, *args, **kwargs):
        # If it's a stream, read the whole thing
        str_in = getattr(str_in, 'read', lambda: str_in)()

        # If it's unicode, turn it into bytes, since ISO-8601 only covers ASCII
        if isinstance(str_in, six.text_type):
            # ASCII is the same in UTF-8
            try:
                str_in = str_in.encode('ascii')
            except UnicodeEncodeError as e:
                msg = 'ISO-8601 strings should contain only ASCII characters'
                six.raise_from(ValueError(msg), e)

        return f(self, str_in, *args, **kwargs)

    return func


class isoparser(object):
    def __init__(self, sep=None):
        """
        :param sep:
            A single character that separates date and time portions. If
            ``None``, the parser will accept any single character.
            For strict ISO-8601 adherence, pass ``'T'``.
        """
        if sep is not None:
            if (len(sep) != 1 or ord(sep) >= 128 or sep in '0123456789'):
                raise ValueError('Separator must be a single, non-numeric ' +
                                 'ASCII character')

            sep = sep.encode('ascii')

        self._sep = sep

    @_takes_ascii
    def isoparse(self, dt_str):
        """
        Parse an ISO-8601 datetime string into a :class:`datetime.datetime`.

        An ISO-8601 datetime string consists of a date portion, followed
        optionally by a time portion - the date and time portions are separated
        by a single character separator, which is ``T`` in the official
        standard. Incomplete date formats (such as ``YYYY-MM``) may *not* be
        combined with a time portion.

        Supported date formats are:

        Common:

        - ``YYYY``
        - ``YYYY-MM``
        - ``YYYY-MM-DD`` or ``YYYYMMDD``

        Uncommon:

        - ``YYYY-Www`` or ``YYYYWww`` - ISO week (day defaults to 0)
        - ``YYYY-Www-D`` or ``YYYYWwwD`` - ISO week and day

        The ISO week and day numbering follows the same logic as
        :func:`datetime.date.isocalendar`.

        Supported time formats are:

        - ``hh``
        - ``hh:mm`` or ``hhmm``
        - ``hh:mm:ss`` or ``hhmmss``
        - ``hh:mm:ss.ssssss`` (Up to 6 sub-second digits)

        Midnight is a special case for `hh`, as the standard supports both
        00:00 and 24:00 as a representation. The decimal separator can be
        either a dot or a comma.


        .. caution::

            Support for fractional components other than seconds is part of the
            ISO-8601 standard, but is not currently implemented in this parser.

        Supported time zone offset formats are:

        - `Z` (UTC)
        - `±HH:MM`
        - `±HHMM`
        - `±HH`

        Offsets will be represented as :class:`dateutil.tz.tzoffset` objects,
        with the exception of UTC, which will be represented as
        :class:`dateutil.tz.tzutc`. Time zone offsets equivalent to UTC (such
        as `+00:00`) will also be represented as :class:`dateutil.tz.tzutc`.

        :param dt_str:
            A string or stream containing only an ISO-8601 datetime string

        :return:
            Returns a :class:`datetime.datetime` representing the string.
            Unspecified components default to their lowest value.

        .. warning::

            As of version 2.7.0, the strictness of the parser should not be
            considered a stable part of the contract. Any valid ISO-8601 string
            that parses correctly with the default settings will continue to
            parse correctly in future versions, but invalid strings that
            currently fail (e.g. ``2017-01-01T00:00+00:00:00``) are not
            guaranteed to continue failing in future versions if they encode
            a valid date.

        .. versionadded:: 2.7.0
        """
        components, pos = self._parse_isodate(dt_str)

        if len(dt_str) > pos:
            if self._sep is None or dt_str[pos:pos + 1] == self._sep:
                components += self._parse_isotime(dt_str[pos + 1:])
            else:
                raise ValueError('String contains unknown ISO components')

        if len(components) > 3 and components[3] == 24:
            components[3] = 0
            return datetime(*components) + timedelta(days=1)

        return datetime(*components)

    @_takes_ascii
    def parse_isodate(self, datestr):
        """
        Parse the date portion of an ISO string.

        :param datestr:
            The string portion of an ISO string, without a separator

        :return:
            Returns a :class:`datetime.date` object
        """
        components, pos = self._parse_isodate(datestr)
        if pos < len(datestr):
            raise ValueError('String contains unknown ISO ' +
                             'components: {!r}'.format(datestr.decode('ascii')))
        return date(*components)

    @_takes_ascii
    def parse_isotime(self, timestr):
        """
        Parse the time portion of an ISO string.

        :param timestr:
            The time portion of an ISO string, without a separator

        :return:
            Returns a :class:`datetime.time` object
        """
        components = self._parse_isotime(timestr)
        if components[0] == 24:
            components[0] = 0
        return time(*components)

    @_takes_ascii
    def parse_tzstr(self, tzstr, zero_as_utc=True):
        """
        Parse a valid ISO time zone string.

        See :func:`isoparser.isoparse` for details on supported formats.

        :param tzstr:
            A string representing an ISO time zone offset

        :param zero_as_utc:
            Whether to return :class:`dateutil.tz.tzutc` for zero-offset zones

        :return:
            Returns :class:`dateutil.tz.tzoffset` for offsets and
            :class:`dateutil.tz.tzutc` for ``Z`` and (if ``zero_as_utc`` is
            specified) offsets equivalent to UTC.
        """
        return self._parse_tzstr(tzstr, zero_as_utc=zero_as_utc)

    # Constants
    _DATE_SEP = b'-'
    _TIME_SEP = b':'
    _FRACTION_REGEX = re.compile(b'[\\.,]([0-9]+)')

    def _parse_isodate(self, dt_str):
        try:
            return self._parse_isodate_common(dt_str)
        except ValueError:
            return self._parse_isodate_uncommon(dt_str)

    def _parse_isodate_common(self, dt_str):
        len_str = len(dt_str)
        components = [1, 1, 1]

        if len_str < 4:
            raise ValueError('ISO string too short')

        # Year
        components[0] = int(dt_str[0:4])
        pos = 4
        if pos >= len_str:
            return components, pos

        has_sep = dt_str[pos:pos + 1] == self._DATE_SEP
        if has_sep:
            pos += 1

        # Month
        if len_str - pos < 2:
            raise ValueError('Invalid common month')

        components[1] = int(dt_str[pos:pos + 2])
        pos += 2

        if pos >= len_str:
            if has_sep:
                return components, pos
            else:
                raise ValueError('Invalid ISO format')

        if has_sep:
            if dt_str[pos:pos + 1] != self._DATE_SEP:
                raise ValueError('Invalid separator in ISO string')
            pos += 1

        # Day
        if len_str - pos < 2:
            raise ValueError('Invalid common day')
        components[2] = int(dt_str[pos:pos + 2])
        return components, pos + 2

    def _parse_isodate_uncommon(self, dt_str):
        if len(dt_str) < 4:
            raise ValueError('ISO string too short')

        # All ISO formats start with the year
        year = int(dt_str[0:4])

        has_sep = dt_str[4:5] == self._DATE_SEP

        pos = 4 + has_sep       # Skip '-' if it's there
        if dt_str[pos:pos + 1] == b'W':
            # YYYY-?Www-?D?
            pos += 1
            weekno = int(dt_str[pos:pos + 2])
            pos += 2

            dayno = 1
            if len(dt_str) > pos:
                if (dt_str[pos:pos + 1] == self._DATE_SEP) != has_sep:
                    raise ValueError('Inconsistent use of dash separator')

                pos += has_sep

                dayno = int(dt_str[pos:pos + 1])
                pos += 1

            base_date = self._calculate_weekdate(year, weekno, dayno)
        else:
            # YYYYDDD or YYYY-DDD
            if len(dt_str) - pos < 3:
                raise ValueError('Invalid ordinal day')

            ordinal_day = int(dt_str[pos:pos + 3])
            pos += 3

            if ordinal_day < 1 or ordinal_day > (365 + calendar.isleap(year)):
                raise ValueError('Invalid ordinal day' +
                                 ' {} for year {}'.format(ordinal_day, year))

            base_date = date(year, 1, 1) + timedelta(days=ordinal_day - 1)

        components = [base_date.year, base_date.month, base_date.day]
        return components, pos

    def _calculate_weekdate(self, year, week, day):
        """
        Calculate the day of corresponding to the ISO year-week-day calendar.

        This function is effectively the inverse of
        :func:`datetime.date.isocalendar`.

        :param year:
            The year in the ISO calendar

        :param week:
            The week in the ISO calendar - range is [1, 53]

        :param day:
            The day in the ISO calendar - range is [1 (MON), 7 (SUN)]

        :return:
            Returns a :class:`datetime.date`
        """
        if not 0 < week < 54:
            raise ValueError('Invalid week: {}'.format(week))

        if not 0 < day < 8:     # Range is 1-7
            raise ValueError('Invalid weekday: {}'.format(day))

        # Get week 1 for the specific year:
        jan_4 = date(year, 1, 4)   # Week 1 always has January 4th in it
        week_1 = jan_4 - timedelta(days=jan_4.isocalendar()[2] - 1)

        # Now add the specific number of weeks and days to get what we want
        week_offset = (week - 1) * 7 + (day - 1)
        return week_1 + timedelta(days=week_offset)

    def _parse_isotime(self, timestr):
        len_str = len(timestr)
        components = [0, 0, 0, 0, None]
        pos = 0
        comp = -1

        if len_str < 2:
            raise ValueError('ISO time too short')

        has_sep = False

        while pos < len_str and comp < 5:
            comp += 1

            if timestr[pos:pos + 1] in b'-+Zz':
                # Detect time zone boundary
                components[-1] = self._parse_tzstr(timestr[pos:])
                pos = len_str
                break

            if comp == 1 and timestr[pos:pos+1] == self._TIME_SEP:
                has_sep = True
                pos += 1
            elif comp == 2 and has_sep:
                if timestr[pos:pos+1] != self._TIME_SEP:
                    raise ValueError('Inconsistent use of colon separator')
                pos += 1

            if comp < 3:
                # Hour, minute, second
                components[comp] = int(timestr[pos:pos + 2])
                pos += 2

            if comp == 3:
                # Fraction of a second
                frac = self._FRACTION_REGEX.match(timestr[pos:])
                if not frac:
                    continue

                us_str = frac.group(1)[:6]  # Truncate to microseconds
                components[comp] = int(us_str) * 10**(6 - len(us_str))
                pos += len(frac.group())

        if pos < len_str:
            raise ValueError('Unused components in ISO string')

        if components[0] == 24:
            # Standard supports 00:00 and 24:00 as representations of midnight
            if any(component != 0 for component in components[1:4]):
                raise ValueError('Hour may only be 24 at 24:00:00.000')

        return components

    def _parse_tzstr(self, tzstr, zero_as_utc=True):
        if tzstr == b'Z' or tzstr == b'z':
            return tz.UTC

        if len(tzstr) not in {3, 5, 6}:
            raise ValueError('Time zone offset must be 1, 3, 5 or 6 characters')

        if tzstr[0:1] == b'-':
            mult = -1
        elif tzstr[0:1] == b'+':
            mult = 1
        else:
            raise ValueError('Time zone offset requires sign')

        hours = int(tzstr[1:3])
        if len(tzstr) == 3:
            minutes = 0
        else:
            minutes = int(tzstr[(4 if tzstr[3:4] == self._TIME_SEP else 3):])

        if zero_as_utc and hours == 0 and minutes == 0:
            return tz.UTC
        else:
            if minutes > 59:
                raise ValueError('Invalid minutes in time zone offset')

            if hours > 23:
                raise ValueError('Invalid hours in time zone offset')

            return tz.tzoffset(None, mult * (hours * 60 + minutes) * 60)


DEFAULT_ISOPARSER = isoparser()
isoparse = DEFAULT_ISOPARSER.isoparse
<|endoftext|>"""Flow based node and edge disjoint paths."""

import networkx as nx

# Define the default maximum flow function to use for the underlying
# maximum flow computations
from networkx.algorithms.flow import (
    edmonds_karp,
    preflow_push,
    shortest_augmenting_path,
)
from networkx.exception import NetworkXNoPath

default_flow_func = edmonds_karp
from itertools import filterfalse as _filterfalse

# Functions to build auxiliary data structures.
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity

__all__ = ["edge_disjoint_paths", "node_disjoint_paths"]


@nx._dispatchable(
    graphs={"G": 0, "auxiliary?": 5},
    preserve_edge_attrs={"auxiliary": {"capacity": float("inf")}},
)
def edge_disjoint_paths(
    G, s, t, flow_func=None, cutoff=None, auxiliary=None, residual=None
):
    """Returns the edges disjoint paths between source and target.

    Edge disjoint paths are paths that do not share any edge. The
    number of edge disjoint paths between source and target is equal
    to their edge connectivity.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. The choice of the default function
        may change from version to version and should not be relied on.
        Default value: None.

    cutoff : integer or None (default: None)
        Maximum number of paths to yield. If specified, the maximum flow
        algorithm will terminate when the flow value reaches or exceeds the
        cutoff. This only works for flows that support the cutoff parameter
        (most do) and is ignored otherwise.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based edge connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    paths : generator
        A generator of edge independent paths.

    Raises
    ------
    NetworkXNoPath
        If there is no path between source and target.

    NetworkXError
        If source or target are not in the graph G.

    See also
    --------
    :meth:`node_disjoint_paths`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Examples
    --------
    We use in this example the platonic icosahedral graph, which has node
    edge connectivity 5, thus there are 5 edge disjoint paths between any
    pair of nodes.

    >>> G = nx.icosahedral_graph()
    >>> len(list(nx.edge_disjoint_paths(G, 0, 6)))
    5


    If you need to compute edge disjoint paths on several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for edge connectivity, and the residual
    network for the underlying maximum flow computation.

    Example of how to compute edge disjoint paths among all pairs of
    nodes of the platonic icosahedral graph reusing the data
    structures.

    >>> import itertools
    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
    >>> H = build_auxiliary_edge_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> result = {n: {} for n in G}
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as arguments
    >>> for u, v in itertools.combinations(G, 2):
    ...     k = len(list(nx.edge_disjoint_paths(G, u, v, auxiliary=H, residual=R)))
    ...     result[u][v] = k
    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))
    True

    You can also use alternative flow algorithms for computing edge disjoint
    paths. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(list(nx.edge_disjoint_paths(G, 0, 6, flow_func=shortest_augmenting_path)))
    5

    Notes
    -----
    This is a flow based implementation of edge disjoint paths. We compute
    the maximum flow between source and target on an auxiliary directed
    network. The saturated edges in the residual network after running the
    maximum flow algorithm correspond to edge disjoint paths between source
    and target in the original network. This function handles both directed
    and undirected graphs, and can use all flow algorithms from NetworkX flow
    package.

    """
    if s not in G:
        raise nx.NetworkXError(f"node {s} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {t} not in graph")

    if flow_func is None:
        flow_func = default_flow_func

    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary

    # Maximum possible edge disjoint paths
    possible = min(H.out_degree(s), H.in_degree(t))
    if not possible:
        raise NetworkXNoPath

    if cutoff is None:
        cutoff = possible
    else:
        cutoff = min(cutoff, possible)

    # Compute maximum flow between source and target. Flow functions in
    # NetworkX return a residual network.
    kwargs = {
        "capacity": "capacity",
        "residual": residual,
        "cutoff": cutoff,
        "value_only": True,
    }
    if flow_func is preflow_push:
        del kwargs["cutoff"]
    if flow_func is shortest_augmenting_path:
        kwargs["two_phase"] = True
    R = flow_func(H, s, t, **kwargs)

    if R.graph["flow_value"] == 0:
        raise NetworkXNoPath

    # Saturated edges in the residual network form the edge disjoint paths
    # between source and target
    cutset = [
        (u, v)
        for u, v, d in R.edges(data=True)
        if d["capacity"] == d["flow"] and d["flow"] > 0
    ]
    # This is equivalent of what flow.utils.build_flow_dict returns, but
    # only for the nodes with saturated edges and without reporting 0 flows.
    flow_dict = {n: {} for edge in cutset for n in edge}
    for u, v in cutset:
        flow_dict[u][v] = 1

    # Rebuild the edge disjoint paths from the flow dictionary.
    paths_found = 0
    for v in list(flow_dict[s]):
        if paths_found >= cutoff:
            # preflow_push does not support cutoff: we have to
            # keep track of the paths founds and stop at cutoff.
            break
        path = [s]
        if v == t:
            path.append(v)
            yield path
            continue
        u = v
        while u != t:
            path.append(u)
            try:
                u, _ = flow_dict[u].popitem()
            except KeyError:
                break
        else:
            path.append(t)
            yield path
            paths_found += 1


@nx._dispatchable(
    graphs={"G": 0, "auxiliary?": 5},
    preserve_node_attrs={"auxiliary": {"id": None}},
    preserve_graph_attrs={"auxiliary"},
)
def node_disjoint_paths(
    G, s, t, flow_func=None, cutoff=None, auxiliary=None, residual=None
):
    r"""Computes node disjoint paths between source and target.

    Node disjoint paths are paths that only share their first and last
    nodes. The number of node independent paths between two nodes is
    equal to their local node connectivity.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node.

    t : node
        Target node.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The choice
        of the default function may change from version to version and
        should not be relied on. Default value: None.

    cutoff : integer or None (default: None)
        Maximum number of paths to yield. If specified, the maximum flow
        algorithm will terminate when the flow value reaches or exceeds the
        cutoff. This only works for flows that support the cutoff parameter
        (most do) and is ignored otherwise.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    paths : generator
        Generator of node disjoint paths.

    Raises
    ------
    NetworkXNoPath
        If there is no path between source and target.

    NetworkXError
        If source or target are not in the graph G.

    Examples
    --------
    We use in this example the platonic icosahedral graph, which has node
    connectivity 5, thus there are 5 node disjoint paths between any pair
    of non neighbor nodes.

    >>> G = nx.icosahedral_graph()
    >>> len(list(nx.node_disjoint_paths(G, 0, 6)))
    5

    If you need to compute node disjoint paths between several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for node connectivity and node cuts, and the
    residual network for the underlying maximum flow computation.

    Example of how to compute node disjoint paths reusing the data
    structures:

    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
    >>> H = build_auxiliary_node_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as arguments
    >>> len(list(nx.node_disjoint_paths(G, 0, 6, auxiliary=H, residual=R)))
    5

    You can also use alternative flow algorithms for computing node disjoint
    paths. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(list(nx.node_disjoint_paths(G, 0, 6, flow_func=shortest_augmenting_path)))
    5

    Notes
    -----
    This is a flow based implementation of node disjoint paths. We compute
    the maximum flow between source and target on an auxiliary directed
    network. The saturated edges in the residual network after running the
    maximum flow algorithm correspond to node disjoint paths between source
    and target in the original network. This function handles both directed
    and undirected graphs, and can use all flow algorithms from NetworkX flow
    package.

    See also
    --------
    :meth:`edge_disjoint_paths`
    :meth:`node_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    """
    if s not in G:
        raise nx.NetworkXError(f"node {s} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {t} not in graph")

    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary

    mapping = H.graph.get("mapping", None)
    if mapping is None:
        raise nx.NetworkXError("Invalid auxiliary digraph.")

    # Maximum possible edge disjoint paths
    possible = min(H.out_degree(f"{mapping[s]}B"), H.in_degree(f"{mapping[t]}A"))
    if not possible:
        raise NetworkXNoPath

    if cutoff is None:
        cutoff = possible
    else:
        cutoff = min(cutoff, possible)

    kwargs = {
        "flow_func": flow_func,
        "residual": residual,
        "auxiliary": H,
        "cutoff": cutoff,
    }

    # The edge disjoint paths in the auxiliary digraph correspond to the node
    # disjoint paths in the original graph.
    paths_edges = edge_disjoint_paths(H, f"{mapping[s]}B", f"{mapping[t]}A", **kwargs)
    for path in paths_edges:
        # Each node in the original graph maps to two nodes in auxiliary graph
        yield list(_unique_everseen(H.nodes[node]["id"] for node in path))


def _unique_everseen(iterable):
    # Adapted from https://docs.python.org/3/library/itertools.html examples
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    seen = set()
    seen_add = seen.add
    for element in _filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element
<|endoftext|>"""
Kanevsky all minimum node k cutsets algorithm.
"""

import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter

import networkx as nx
from networkx.algorithms.flow import (
    build_residual_network,
    edmonds_karp,
    shortest_augmenting_path,
)

from .utils import build_auxiliary_node_connectivity

default_flow_func = edmonds_karp


__all__ = ["all_node_cuts"]


@nx._dispatchable
def all_node_cuts(G, k=None, flow_func=None):
    r"""Returns all minimum k cutsets of an undirected graph G.

    This implementation is based on Kanevsky's algorithm [1]_ for finding all
    minimum-size node cut-sets of an undirected graph G; ie the set (or sets)
    of nodes of cardinality equal to the node connectivity of G. Thus if
    removed, would break G into two or more connected components.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    k : Integer
        Node connectivity of the input graph. If k is None, then it is
        computed. Default value: None.

    flow_func : function
        Function to perform the underlying flow computations. Default value is
        :func:`~networkx.algorithms.flow.edmonds_karp`. This function performs
        better in sparse graphs with right tailed degree distributions.
        :func:`~networkx.algorithms.flow.shortest_augmenting_path` will
        perform better in denser graphs.


    Returns
    -------
    cuts : a generator of node cutsets
        Each node cutset has cardinality equal to the node connectivity of
        the input graph.

    Examples
    --------
    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2
    >>> G = nx.grid_2d_graph(5, 5)
    >>> cutsets = list(nx.all_node_cuts(G))
    >>> len(cutsets)
    4
    >>> all(2 == len(cutset) for cutset in cutsets)
    True
    >>> nx.node_connectivity(G)
    2

    Notes
    -----
    This implementation is based on the sequential algorithm for finding all
    minimum-size separating vertex sets in a graph [1]_. The main idea is to
    compute minimum cuts using local maximum flow computations among a set
    of nodes of highest degree and all other non-adjacent nodes in the Graph.
    Once we find a minimum cut, we add an edge between the high degree
    node and the target node of the local maximum flow computation to make
    sure that we will not find that minimum cut again.

    See also
    --------
    node_connectivity
    edmonds_karp
    shortest_augmenting_path

    References
    ----------
    .. [1]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Input graph is disconnected.")

    # Address some corner cases first.
    # For complete Graphs

    if nx.density(G) == 1:
        yield from ()
        return

    # Initialize data structures.
    # Keep track of the cuts already computed so we do not repeat them.
    seen = []
    # Even-Tarjan reduction is what we call auxiliary digraph
    # for node connectivity.
    H = build_auxiliary_node_connectivity(G)
    H_nodes = H.nodes  # for speed
    mapping = H.graph["mapping"]
    # Keep a copy of original predecessors, H will be modified later.
    # Shallow copy is enough.
    original_H_pred = copy.copy(H._pred)
    R = build_residual_network(H, "capacity")
    kwargs = {"capacity": "capacity", "residual": R}
    # Define default flow function
    if flow_func is None:
        flow_func = default_flow_func
    if flow_func is shortest_augmenting_path:
        kwargs["two_phase"] = True
    # Begin the actual algorithm
    # step 1: Find node connectivity k of G
    if k is None:
        k = nx.node_connectivity(G, flow_func=flow_func)
    # step 2:
    # Find k nodes with top degree, call it X:
    X = {n for n, d in sorted(G.degree(), key=itemgetter(1), reverse=True)[:k]}
    # Check if X is a k-node-cutset
    if _is_separating_set(G, X):
        seen.append(X)
        yield X

    for x in X:
        # step 3: Compute local connectivity flow of x with all other
        # non adjacent nodes in G
        non_adjacent = set(G) - {x} - set(G[x])
        for v in non_adjacent:
            # step 4: compute maximum flow in an Even-Tarjan reduction H of G
            # and step 5: build the associated residual network R
            R = flow_func(H, f"{mapping[x]}B", f"{mapping[v]}A", **kwargs)
            flow_value = R.graph["flow_value"]

            if flow_value == k:
                # Find the nodes incident to the flow.
                E1 = flowed_edges = [
                    (u, w) for (u, w, d) in R.edges(data=True) if d["flow"] != 0
                ]
                VE1 = incident_nodes = {n for edge in E1 for n in edge}
                # Remove saturated edges form the residual network.
                # Note that reversed edges are introduced with capacity 0
                # in the residual graph and they need to be removed too.
                saturated_edges = [
                    (u, w, d)
                    for (u, w, d) in R.edges(data=True)
                    if d["capacity"] == d["flow"] or d["capacity"] == 0
                ]
                R.remove_edges_from(saturated_edges)
                R_closure = nx.transitive_closure(R)
                # step 6: shrink the strongly connected components of
                # residual flow network R and call it L.
                L = nx.condensation(R)
                cmap = L.graph["mapping"]
                inv_cmap = defaultdict(list)
                for n, scc in cmap.items():
                    inv_cmap[scc].append(n)
                # Find the incident nodes in the condensed graph.
                VE1 = {cmap[n] for n in VE1}
                # step 7: Compute all antichains of L;
                # they map to closed sets in H.
                # Any edge in H that links a closed set is part of a cutset.
                for antichain in nx.antichains(L):
                    # Only antichains that are subsets of incident nodes counts.
                    # Lemma 8 in reference.
                    if not set(antichain).issubset(VE1):
                        continue
                    # Nodes in an antichain of the condensation graph of
                    # the residual network map to a closed set of nodes that
                    # define a node partition of the auxiliary digraph H
                    # through taking all of antichain's predecessors in the
                    # transitive closure.
                    S = set()
                    for scc in antichain:
                        S.update(inv_cmap[scc])
                    S_ancestors = set()
                    for n in S:
                        S_ancestors.update(R_closure._pred[n])
                    S.update(S_ancestors)
                    if f"{mapping[x]}B" not in S or f"{mapping[v]}A" in S:
                        continue
                    # Find the cutset that links the node partition (S,~S) in H
                    cutset = set()
                    for u in S:
                        cutset.update((u, w) for w in original_H_pred[u] if w not in S)
                    # The edges in H that form the cutset are internal edges
                    # (ie edges that represent a node of the original graph G)
                    if any(H_nodes[u]["id"] != H_nodes[w]["id"] for u, w in cutset):
                        continue
                    node_cut = {H_nodes[u]["id"] for u, _ in cutset}

                    if len(node_cut) == k:
                        # The cut is invalid if it includes internal edges of
                        # end nodes. The other half of Lemma 8 in ref.
                        if x in node_cut or v in node_cut:
                            continue
                        if node_cut not in seen:
                            yield node_cut
                            seen.append(node_cut)

                # Add an edge (x, v) to make sure that we do not
                # find this cutset again. This is equivalent
                # of adding the edge in the input graph
                # G.add_edge(x, v) and then regenerate H and R:
                # Add edges to the auxiliary digraph.
                # See build_residual_network for convention we used
                # in residual graphs.
                H.add_edge(f"{mapping[x]}B", f"{mapping[v]}A", capacity=1)
                H.add_edge(f"{mapping[v]}B", f"{mapping[x]}A", capacity=1)
                # Add edges to the residual network.
                R.add_edge(f"{mapping[x]}B", f"{mapping[v]}A", capacity=1)
                R.add_edge(f"{mapping[v]}A", f"{mapping[x]}B", capacity=0)
                R.add_edge(f"{mapping[v]}B", f"{mapping[x]}A", capacity=1)
                R.add_edge(f"{mapping[x]}A", f"{mapping[v]}B", capacity=0)

                # Add again the saturated edges to reuse the residual network
                R.add_edges_from(saturated_edges)


def _is_separating_set(G, cut):
    """Assumes that the input graph is connected"""
    if len(cut) == len(G) - 1:
        return True

    H = nx.restricted_view(G, cut, [])
    if nx.is_connected(H):
        return False
    return True
<|endoftext|>"""Laplacian matrix of graphs.

All calculations here are done using the out-degree. For Laplacians using
in-degree, use `G.reverse(copy=False)` instead of `G` and take the transpose.

The `laplacian_matrix` function provides an unnormalized matrix,
while `normalized_laplacian_matrix`, `directed_laplacian_matrix`,
and `directed_combinatorial_laplacian_matrix` are all normalized.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "laplacian_matrix",
    "normalized_laplacian_matrix",
    "total_spanning_tree_weight",
    "directed_laplacian_matrix",
    "directed_combinatorial_laplacian_matrix",
]


@nx._dispatchable(edge_attrs="weight")
def laplacian_matrix(G, nodelist=None, weight="weight"):
    """Returns the Laplacian matrix of G.

    The graph Laplacian is the matrix L = D - A, where
    A is the adjacency matrix and D is the diagonal matrix of node degrees.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    L : SciPy sparse array
      The Laplacian matrix of G.

    Notes
    -----
    For MultiGraph, the edges weights are summed.

    This returns an unnormalized matrix. For a normalized output,
    use `normalized_laplacian_matrix`, `directed_laplacian_matrix`,
    or `directed_combinatorial_laplacian_matrix`.

    This calculation uses the out-degree of the graph `G`. To use the
    in-degree for calculations instead, use `G.reverse(copy=False)` and
    take the transpose.

    See Also
    --------
    :func:`~networkx.convert_matrix.to_numpy_array`
    normalized_laplacian_matrix
    directed_laplacian_matrix
    directed_combinatorial_laplacian_matrix
    :func:`~networkx.linalg.spectrum.laplacian_spectrum`

    Examples
    --------
    For graphs with multiple connected components, L is permutation-similar
    to a block diagonal matrix where each block is the respective Laplacian
    matrix for each component.

    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> print(nx.laplacian_matrix(G).toarray())
    [[ 1 -1  0  0  0]
     [-1  2 -1  0  0]
     [ 0 -1  1  0  0]
     [ 0  0  0  1 -1]
     [ 0  0  0 -1  1]]

    >>> edges = [
    ...     (1, 2),
    ...     (2, 1),
    ...     (2, 4),
    ...     (4, 3),
    ...     (3, 4),
    ... ]
    >>> DiG = nx.DiGraph(edges)
    >>> print(nx.laplacian_matrix(DiG).toarray())
    [[ 1 -1  0  0]
     [-1  2 -1  0]
     [ 0  0  1 -1]
     [ 0  0 -1  1]]

    Notice that node 4 is represented by the third column and row. This is because
    by default the row/column order is the order of `G.nodes` (i.e. the node added
    order -- in the edgelist, 4 first appears in (2, 4), before node 3 in edge (4, 3).)
    To control the node order of the matrix, use the `nodelist` argument.

    >>> print(nx.laplacian_matrix(DiG, nodelist=[1, 2, 3, 4]).toarray())
    [[ 1 -1  0  0]
     [-1  2  0 -1]
     [ 0  0  1 -1]
     [ 0  0 -1  1]]

    This calculation uses the out-degree of the graph `G`. To use the
    in-degree for calculations instead, use `G.reverse(copy=False)` and
    take the transpose.

    >>> print(nx.laplacian_matrix(DiG.reverse(copy=False)).toarray().T)
    [[ 1 -1  0  0]
     [-1  1 -1  0]
     [ 0  0  2 -1]
     [ 0  0 -1  1]]

    References
    ----------
    .. [1] Langville, Amy N., and Carl D. Meyer. Google’s PageRank and Beyond:
       The Science of Search Engine Rankings. Princeton University Press, 2006.

    """
    import scipy as sp

    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format="csr")
    n, m = A.shape
    # TODO: rm csr_array wrapper when spdiags can produce arrays
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format="csr"))
    return D - A


@nx._dispatchable(edge_attrs="weight")
def normalized_laplacian_matrix(G, nodelist=None, weight="weight"):
    r"""Returns the normalized Laplacian matrix of G.

    The normalized graph Laplacian is the matrix

    .. math::

        N = D^{-1/2} L D^{-1/2}

    where `L` is the graph Laplacian and `D` is the diagonal matrix of
    node degrees [1]_.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    N : SciPy sparse array
      The normalized Laplacian matrix of G.

    Notes
    -----
    For MultiGraph, the edges weights are summed.
    See :func:`to_numpy_array` for other options.

    If the Graph contains selfloops, D is defined as ``diag(sum(A, 1))``, where A is
    the adjacency matrix [2]_.

    This calculation uses the out-degree of the graph `G`. To use the
    in-degree for calculations instead, use `G.reverse(copy=False)` and
    take the transpose.

    For an unnormalized output, use `laplacian_matrix`.

    Examples
    --------

    >>> import numpy as np
    >>> edges = [
    ...     (1, 2),
    ...     (2, 1),
    ...     (2, 4),
    ...     (4, 3),
    ...     (3, 4),
    ... ]
    >>> DiG = nx.DiGraph(edges)
    >>> print(nx.normalized_laplacian_matrix(DiG).toarray())
    [[ 1.         -0.70710678  0.          0.        ]
     [-0.70710678  1.         -0.70710678  0.        ]
     [ 0.          0.          1.         -1.        ]
     [ 0.          0.         -1.          1.        ]]

    Notice that node 4 is represented by the third column and row. This is because
    by default the row/column order is the order of `G.nodes` (i.e. the node added
    order -- in the edgelist, 4 first appears in (2, 4), before node 3 in edge (4, 3).)
    To control the node order of the matrix, use the `nodelist` argument.

    >>> print(nx.normalized_laplacian_matrix(DiG, nodelist=[1, 2, 3, 4]).toarray())
    [[ 1.         -0.70710678  0.          0.        ]
     [-0.70710678  1.          0.         -0.70710678]
     [ 0.          0.          1.         -1.        ]
     [ 0.          0.         -1.          1.        ]]
    >>> G = nx.Graph(edges)
    >>> print(nx.normalized_laplacian_matrix(G).toarray())
    [[ 1.         -0.70710678  0.          0.        ]
     [-0.70710678  1.         -0.5         0.        ]
     [ 0.         -0.5         1.         -0.70710678]
     [ 0.          0.         -0.70710678  1.        ]]

    See Also
    --------
    laplacian_matrix
    normalized_laplacian_spectrum
    directed_laplacian_matrix
    directed_combinatorial_laplacian_matrix

    References
    ----------
    .. [1] Fan Chung-Graham, Spectral Graph Theory,
       CBMS Regional Conference Series in Mathematics, Number 92, 1997.
    .. [2] Steve Butler, Interlacing For Weighted Graphs Using The Normalized
       Laplacian, Electronic Journal of Linear Algebra, Volume 16, pp. 90-98,
       March 2007.
    .. [3] Langville, Amy N., and Carl D. Meyer. Google’s PageRank and Beyond:
       The Science of Search Engine Rankings. Princeton University Press, 2006.
    """
    import numpy as np
    import scipy as sp

    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format="csr")
    n, _ = A.shape
    diags = A.sum(axis=1)
    # TODO: rm csr_array wrapper when spdiags can produce arrays
    D = sp.sparse.csr_array(sp.sparse.spdiags(diags, 0, n, n, format="csr"))
    L = D - A
    with np.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    # TODO: rm csr_array wrapper when spdiags can produce arrays
    DH = sp.sparse.csr_array(sp.sparse.spdiags(diags_sqrt, 0, n, n, format="csr"))
    return DH @ (L @ DH)


@nx._dispatchable(edge_attrs="weight")
def total_spanning_tree_weight(G, weight=None, root=None):
    """
    Returns the total weight of all spanning trees of `G`.

    Kirchoff's Tree Matrix Theorem [1]_, [2]_ states that the determinant of any
    cofactor of the Laplacian matrix of a graph is the number of spanning trees
    in the graph. For a weighted Laplacian matrix, it is the sum across all
    spanning trees of the multiplicative weight of each tree. That is, the
    weight of each tree is the product of its edge weights.

    For unweighted graphs, the total weight equals the number of spanning trees in `G`.

    For directed graphs, the total weight follows by summing over all directed
    spanning trees in `G` that start in the `root` node [3]_.

    .. deprecated:: 3.3

       ``total_spanning_tree_weight`` is deprecated and will be removed in v3.5.
       Use ``nx.number_of_spanning_trees(G)`` instead.

    Parameters
    ----------
    G : NetworkX Graph

    weight : string or None, optional (default=None)
        The key for the edge attribute holding the edge weight.
        If None, then each edge has weight 1.

    root : node (only required for directed graphs)
       A node in the directed graph `G`.

    Returns
    -------
    total_weight : float
        Undirected graphs:
            The sum of the total multiplicative weights for all spanning trees in `G`.
        Directed graphs:
            The sum of the total multiplicative weights for all spanning trees of `G`,
            rooted at node `root`.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` does not contain any nodes.

    NetworkXError
        If the graph `G` is not (weakly) connected,
        or if `G` is directed and the root node is not specified or not in G.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> round(nx.total_spanning_tree_weight(G))
    125

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=2)
    >>> G.add_edge(1, 3, weight=1)
    >>> G.add_edge(2, 3, weight=1)
    >>> round(nx.total_spanning_tree_weight(G, "weight"))
    5

    Notes
    -----
    Self-loops are excluded. Multi-edges are contracted in one edge
    equal to the sum of the weights.

    References
    ----------
    .. [1] Wikipedia
       "Kirchhoff's theorem."
       https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
    .. [2] Kirchhoff, G. R.
        Über die Auflösung der Gleichungen, auf welche man
        bei der Untersuchung der linearen Vertheilung
        Galvanischer Ströme geführt wird
        Annalen der Physik und Chemie, vol. 72, pp. 497-508, 1847.
    .. [3] Margoliash, J.
        "Matrix-Tree Theorem for Directed Graphs"
        https://www.math.uchicago.edu/~may/VIGRE/VIGRE2010/REUPapers/Margoliash.pdf
    """
    import warnings

    warnings.warn(
        (
            "\n\ntotal_spanning_tree_weight is deprecated and will be removed in v3.5.\n"
            "Use `nx.number_of_spanning_trees(G)` instead."
        ),
        category=DeprecationWarning,
        stacklevel=3,
    )

    return nx.number_of_spanning_trees(G, weight=weight, root=root)


###############################################################################
# Code based on work from https://github.com/bjedwards


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def directed_laplacian_matrix(
    G, nodelist=None, weight="weight", walk_type=None, alpha=0.95
):
    r"""Returns the directed Laplacian matrix of G.

    The graph directed Laplacian is the matrix

    .. math::

        L = I - \frac{1}{2} \left (\Phi^{1/2} P \Phi^{-1/2} + \Phi^{-1/2} P^T \Phi^{1/2} \right )

    where `I` is the identity matrix, `P` is the transition matrix of the
    graph, and `\Phi` a matrix with the Perron vector of `P` in the diagonal and
    zeros elsewhere [1]_.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
       (the default), then a value is selected according to the properties of `G`:
       - ``walk_type="random"`` if `G` is strongly connected and aperiodic
       - ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
       - ``walk_type="pagerank"`` for all other cases.

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy matrix
      Normalized Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    The result is always a symmetric matrix.

    This calculation uses the out-degree of the graph `G`. To use the
    in-degree for calculations instead, use `G.reverse(copy=False)` and
    take the transpose.

    See Also
    --------
    laplacian_matrix
    normalized_laplacian_matrix
    directed_combinatorial_laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    import numpy as np
    import scipy as sp

    # NOTE: P has type ndarray if walk_type=="pagerank", else csr_array
    P = _transition_matrix(
        G, nodelist=nodelist, weight=weight, walk_type=walk_type, alpha=alpha
    )

    n, m = P.shape

    evals, evecs = sp.sparse.linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    # p>=0 by Perron-Frobenius Thm. Use abs() to fix roundoff across zero gh-6865
    sqrtp = np.sqrt(np.abs(p))
    Q = (
        # TODO: rm csr_array wrapper when spdiags creates arrays
        sp.sparse.csr_array(sp.sparse.spdiags(sqrtp, 0, n, n))
        @ P
        # TODO: rm csr_array wrapper when spdiags creates arrays
        @ sp.sparse.csr_array(sp.sparse.spdiags(1.0 / sqrtp, 0, n, n))
    )
    # NOTE: This could be sparsified for the non-pagerank cases
    I = np.identity(len(G))

    return I - (Q + Q.T) / 2.0


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def directed_combinatorial_laplacian_matrix(
    G, nodelist=None, weight="weight", walk_type=None, alpha=0.95
):
    r"""Return the directed combinatorial Laplacian matrix of G.

    The graph directed combinatorial Laplacian is the matrix

    .. math::

        L = \Phi - \frac{1}{2} \left (\Phi P + P^T \Phi \right)

    where `P` is the transition matrix of the graph and `\Phi` a matrix
    with the Perron vector of `P` in the diagonal and zeros elsewhere [1]_.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
        One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
        (the default), then a value is selected according to the properties of `G`:
        - ``walk_type="random"`` if `G` is strongly connected and aperiodic
        - ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
        - ``walk_type="pagerank"`` for all other cases.

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy matrix
      Combinatorial Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    The result is always a symmetric matrix.

    This calculation uses the out-degree of the graph `G`. To use the
    in-degree for calculations instead, use `G.reverse(copy=False)` and
    take the transpose.

    See Also
    --------
    laplacian_matrix
    normalized_laplacian_matrix
    directed_laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    import scipy as sp

    P = _transition_matrix(
        G, nodelist=nodelist, weight=weight, walk_type=walk_type, alpha=alpha
    )

    n, m = P.shape

    evals, evecs = sp.sparse.linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    # NOTE: could be improved by not densifying
    # TODO: Rm csr_array wrapper when spdiags array creation becomes available
    Phi = sp.sparse.csr_array(sp.sparse.spdiags(p, 0, n, n)).toarray()

    return Phi - (Phi @ P + P.T @ Phi) / 2.0


def _transition_matrix(G, nodelist=None, weight="weight", walk_type=None, alpha=0.95):
    """Returns the transition matrix of G.

    This is a row stochastic giving the transition probabilities while
    performing a random walk on the graph. Depending on the value of walk_type,
    P can be the transition matrix induced by a random walk, a lazy random walk,
    or a random walk with teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
       (the default), then a value is selected according to the properties of `G`:
        - ``walk_type="random"`` if `G` is strongly connected and aperiodic
        - ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
        - ``walk_type="pagerank"`` for all other cases.

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    P : numpy.ndarray
      transition matrix of G.

    Raises
    ------
    NetworkXError
        If walk_type not specified or alpha not in valid range
    """
    import numpy as np
    import scipy as sp

    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = "random"
            else:
                walk_type = "lazy"
        else:
            walk_type = "pagerank"

    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    n, m = A.shape
    if walk_type in ["random", "lazy"]:
        # TODO: Rm csr_array wrapper when spdiags array creation becomes available
        DI = sp.sparse.csr_array(sp.sparse.spdiags(1.0 / A.sum(axis=1), 0, n, n))
        if walk_type == "random":
            P = DI @ A
        else:
            # TODO: Rm csr_array wrapper when identity array creation becomes available
            I = sp.sparse.csr_array(sp.sparse.identity(n))
            P = (I + DI @ A) / 2.0

    elif walk_type == "pagerank":
        if not (0 < alpha < 1):
            raise nx.NetworkXError("alpha must be between 0 and 1")
        # this is using a dense representation. NOTE: This should be sparsified!
        A = A.toarray()
        # add constant to dangling nodes' row
        A[A.sum(axis=1) == 0, :] = 1 / n
        # normalize
        A = A / A.sum(axis=1)[np.newaxis, :].T
        P = alpha * A + (1 - alpha) / n
    else:
        raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")

    return P
<|endoftext|>"""
Implementation of Harwell-Boeing read/write.

At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

"""
# TODO:
#   - Add more support (symmetric/complex matrices, non-assembled matrices ?)

# XXX: reading is reasonably efficient (>= 85 % is in numpy.fromstring), but
# takes a lot of memory. Being faster would require compiled code.
# write is not efficient. Although not a terribly exciting task,
# having reusable facilities to efficiently read/write fortran-formatted files
# would be useful outside this module.

import warnings

import numpy as np
from scipy.sparse import csc_array, csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat

__all__ = ["hb_read", "hb_write"]


class MalformedHeader(Exception):
    pass


class LineOverflow(Warning):
    pass


def _nbytes_full(fmt, nlines):
    """Return the number of bytes to read to get every full lines for the
    given parsed fortran format."""
    return (fmt.repeat * fmt.width + 1) * (nlines - 1)


class HBInfo:
    @classmethod
    def from_data(cls, m, title="Default title", key="0", mxtype=None, fmt=None):
        """Create a HBInfo instance from an existing sparse matrix.

        Parameters
        ----------
        m : sparse array or matrix
            the HBInfo instance will derive its parameters from m
        title : str
            Title to put in the HB header
        key : str
            Key
        mxtype : HBMatrixType
            type of the input matrix
        fmt : dict
            not implemented

        Returns
        -------
        hb_info : HBInfo instance
        """
        m = m.tocsc(copy=False)

        pointer = m.indptr
        indices = m.indices
        values = m.data

        nrows, ncols = m.shape
        nnon_zeros = m.nnz

        if fmt is None:
            # +1 because HB use one-based indexing (Fortran), and we will write
            # the indices /pointer as such
            pointer_fmt = IntFormat.from_number(np.max(pointer+1))
            indices_fmt = IntFormat.from_number(np.max(indices+1))

            if values.dtype.kind in np.typecodes["AllFloat"]:
                values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
            elif values.dtype.kind in np.typecodes["AllInteger"]:
                values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
            else:
                message = f"type {values.dtype.kind} not implemented yet"
                raise NotImplementedError(message)
        else:
            raise NotImplementedError("fmt argument not supported yet.")

        if mxtype is None:
            if not np.isrealobj(values):
                raise ValueError("Complex values not supported yet")
            if values.dtype.kind in np.typecodes["AllInteger"]:
                tp = "integer"
            elif values.dtype.kind in np.typecodes["AllFloat"]:
                tp = "real"
            else:
                raise NotImplementedError(
                    f"type {values.dtype} for values not implemented")
            mxtype = HBMatrixType(tp, "unsymmetric", "assembled")
        else:
            raise ValueError("mxtype argument not handled yet.")

        def _nlines(fmt, size):
            nlines = size // fmt.repeat
            if nlines * fmt.repeat != size:
                nlines += 1
            return nlines

        pointer_nlines = _nlines(pointer_fmt, pointer.size)
        indices_nlines = _nlines(indices_fmt, indices.size)
        values_nlines = _nlines(values_fmt, values.size)

        total_nlines = pointer_nlines + indices_nlines + values_nlines

        return cls(title, key,
            total_nlines, pointer_nlines, indices_nlines, values_nlines,
            mxtype, nrows, ncols, nnon_zeros,
            pointer_fmt.fortran_format, indices_fmt.fortran_format,
            values_fmt.fortran_format)

    @classmethod
    def from_file(cls, fid):
        """Create a HBInfo instance from a file object containing a matrix in the
        HB format.

        Parameters
        ----------
        fid : file-like matrix
            File or file-like object containing a matrix in the HB format.

        Returns
        -------
        hb_info : HBInfo instance
        """
        # First line
        line = fid.readline().strip("\n")
        if not len(line) > 72:
            raise ValueError("Expected at least 72 characters for first line, "
                             f"got: \n{line}")
        title = line[:72]
        key = line[72:]

        # Second line
        line = fid.readline().strip("\n")
        if not len(line.rstrip()) >= 56:
            raise ValueError("Expected at least 56 characters for second line, "
                             f"got: \n{line}")
        total_nlines = _expect_int(line[:14])
        pointer_nlines = _expect_int(line[14:28])
        indices_nlines = _expect_int(line[28:42])
        values_nlines = _expect_int(line[42:56])

        rhs_nlines = line[56:72].strip()
        if rhs_nlines == '':
            rhs_nlines = 0
        else:
            rhs_nlines = _expect_int(rhs_nlines)
        if not rhs_nlines == 0:
            raise ValueError("Only files without right hand side supported for "
                             "now.")

        # Third line
        line = fid.readline().strip("\n")
        if not len(line) >= 70:
            raise ValueError(f"Expected at least 72 character for third line, "
                             f"got:\n{line}")

        mxtype_s = line[:3].upper()
        if not len(mxtype_s) == 3:
            raise ValueError("mxtype expected to be 3 characters long")

        mxtype = HBMatrixType.from_fortran(mxtype_s)
        if mxtype.value_type not in ["real", "integer"]:
            raise ValueError("Only real or integer matrices supported for "
                             f"now (detected {mxtype})")
        if not mxtype.structure == "unsymmetric":
            raise ValueError("Only unsymmetric matrices supported for "
                             f"now (detected {mxtype})")
        if not mxtype.storage == "assembled":
            raise ValueError("Only assembled matrices supported for now")

        if not line[3:14] == " " * 11:
            raise ValueError(f"Malformed data for third line: {line}")

        nrows = _expect_int(line[14:28])
        ncols = _expect_int(line[28:42])
        nnon_zeros = _expect_int(line[42:56])
        nelementals = _expect_int(line[56:70])
        if not nelementals == 0:
            raise ValueError("Unexpected value %d for nltvl (last entry of line 3)"
                             % nelementals)

        # Fourth line
        line = fid.readline().strip("\n")

        ct = line.split()
        if not len(ct) == 3:
            raise ValueError(f"Expected 3 formats, got {ct}")

        return cls(title, key,
                   total_nlines, pointer_nlines, indices_nlines, values_nlines,
                   mxtype, nrows, ncols, nnon_zeros,
                   ct[0], ct[1], ct[2],
                   rhs_nlines, nelementals)

    def __init__(self, title, key,
            total_nlines, pointer_nlines, indices_nlines, values_nlines,
            mxtype, nrows, ncols, nnon_zeros,
            pointer_format_str, indices_format_str, values_format_str,
            right_hand_sides_nlines=0, nelementals=0):
        """Do not use this directly, but the class ctrs (from_* functions)."""
        if title is None:
            title = "No Title"
        if len(title) > 72:
            raise ValueError("title cannot be > 72 characters")

        if key is None:
            key = "|No Key"
        if len(key) > 8:
            warnings.warn(f"key is > 8 characters (key is {key})",
                          LineOverflow, stacklevel=3)
        self.title = title
        self.key = key

        self.total_nlines = total_nlines
        self.pointer_nlines = pointer_nlines
        self.indices_nlines = indices_nlines
        self.values_nlines = values_nlines

        parser = FortranFormatParser()
        pointer_format = parser.parse(pointer_format_str)
        if not isinstance(pointer_format, IntFormat):
            raise ValueError("Expected int format for pointer format, got "
                             f"{pointer_format}")

        indices_format = parser.parse(indices_format_str)
        if not isinstance(indices_format, IntFormat):
            raise ValueError("Expected int format for indices format, got "
                             f"{indices_format}")

        values_format = parser.parse(values_format_str)
        if isinstance(values_format, ExpFormat):
            if mxtype.value_type not in ["real", "complex"]:
                raise ValueError(f"Inconsistency between matrix type {mxtype} and "
                                 f"value type {values_format}")
            values_dtype = np.float64
        elif isinstance(values_format, IntFormat):
            if mxtype.value_type not in ["integer"]:
                raise ValueError(f"Inconsistency between matrix type {mxtype} and "
                                 f"value type {values_format}")
            # XXX: fortran int -> dtype association ?
            values_dtype = int
        else:
            raise ValueError(f"Unsupported format for values {values_format!r}")

        self.pointer_format = pointer_format
        self.indices_format = indices_format
        self.values_format = values_format

        self.pointer_dtype = np.int32
        self.indices_dtype = np.int32
        self.values_dtype = values_dtype

        self.pointer_nlines = pointer_nlines
        self.pointer_nbytes_full = _nbytes_full(pointer_format, pointer_nlines)

        self.indices_nlines = indices_nlines
        self.indices_nbytes_full = _nbytes_full(indices_format, indices_nlines)

        self.values_nlines = values_nlines
        self.values_nbytes_full = _nbytes_full(values_format, values_nlines)

        self.nrows = nrows
        self.ncols = ncols
        self.nnon_zeros = nnon_zeros
        self.nelementals = nelementals
        self.mxtype = mxtype

    def dump(self):
        """Gives the header corresponding to this instance as a string."""
        header = [self.title.ljust(72) + self.key.ljust(8)]

        header.append("%14d%14d%14d%14d" %
                      (self.total_nlines, self.pointer_nlines,
                       self.indices_nlines, self.values_nlines))
        header.append("%14s%14d%14d%14d%14d" %
                      (self.mxtype.fortran_format.ljust(14), self.nrows,
                       self.ncols, self.nnon_zeros, 0))

        pffmt = self.pointer_format.fortran_format
        iffmt = self.indices_format.fortran_format
        vffmt = self.values_format.fortran_format
        header.append("%16s%16s%20s" %
                      (pffmt.ljust(16), iffmt.ljust(16), vffmt.ljust(20)))
        return "\n".join(header)


def _expect_int(value, msg=None):
    try:
        return int(value)
    except ValueError as e:
        if msg is None:
            msg = "Expected an int, got %s"
        raise ValueError(msg % value) from e


def _read_hb_data(content, header):
    # XXX: look at a way to reduce memory here (big string creation)
    ptr_string = "".join([content.read(header.pointer_nbytes_full),
                           content.readline()])
    ptr = np.fromstring(ptr_string,
            dtype=int, sep=' ')

    ind_string = "".join([content.read(header.indices_nbytes_full),
                       content.readline()])
    ind = np.fromstring(ind_string,
            dtype=int, sep=' ')

    val_string = "".join([content.read(header.values_nbytes_full),
                          content.readline()])
    val = np.fromstring(val_string,
            dtype=header.values_dtype, sep=' ')

    return csc_array((val, ind-1, ptr-1), shape=(header.nrows, header.ncols))


def _write_data(m, fid, header):
    m = m.tocsc(copy=False)

    def write_array(f, ar, nlines, fmt):
        # ar_nlines is the number of full lines, n is the number of items per
        # line, ffmt the fortran format
        pyfmt = fmt.python_format
        pyfmt_full = pyfmt * fmt.repeat

        # for each array to write, we first write the full lines, and special
        # case for partial line
        full = ar[:(nlines - 1) * fmt.repeat]
        for row in full.reshape((nlines-1, fmt.repeat)):
            f.write(pyfmt_full % tuple(row) + "\n")
        nremain = ar.size - full.size
        if nremain > 0:
            f.write((pyfmt * nremain) % tuple(ar[ar.size - nremain:]) + "\n")

    fid.write(header.dump())
    fid.write("\n")
    # +1 is for Fortran one-based indexing
    write_array(fid, m.indptr+1, header.pointer_nlines,
                header.pointer_format)
    write_array(fid, m.indices+1, header.indices_nlines,
                header.indices_format)
    write_array(fid, m.data, header.values_nlines,
                header.values_format)


class HBMatrixType:
    """Class to hold the matrix type."""
    # q2f* translates qualified names to Fortran character
    _q2f_type = {
        "real": "R",
        "complex": "C",
        "pattern": "P",
        "integer": "I",
    }
    _q2f_structure = {
            "symmetric": "S",
            "unsymmetric": "U",
            "hermitian": "H",
            "skewsymmetric": "Z",
            "rectangular": "R"
    }
    _q2f_storage = {
        "assembled": "A",
        "elemental": "E",
    }

    _f2q_type = {j: i for i, j in _q2f_type.items()}
    _f2q_structure = {j: i for i, j in _q2f_structure.items()}
    _f2q_storage = {j: i for i, j in _q2f_storage.items()}

    @classmethod
    def from_fortran(cls, fmt):
        if not len(fmt) == 3:
            raise ValueError("Fortran format for matrix type should be 3 "
                             "characters long")
        try:
            value_type = cls._f2q_type[fmt[0]]
            structure = cls._f2q_structure[fmt[1]]
            storage = cls._f2q_storage[fmt[2]]
            return cls(value_type, structure, storage)
        except KeyError as e:
            raise ValueError(f"Unrecognized format {fmt}") from e

    def __init__(self, value_type, structure, storage="assembled"):
        self.value_type = value_type
        self.structure = structure
        self.storage = storage

        if value_type not in self._q2f_type:
            raise ValueError(f"Unrecognized type {value_type}")
        if structure not in self._q2f_structure:
            raise ValueError(f"Unrecognized structure {structure}")
        if storage not in self._q2f_storage:
            raise ValueError(f"Unrecognized storage {storage}")

    @property
    def fortran_format(self):
        return self._q2f_type[self.value_type] + \
               self._q2f_structure[self.structure] + \
               self._q2f_storage[self.storage]

    def __repr__(self):
        return f"HBMatrixType({self.value_type}, {self.structure}, {self.storage})"


class HBFile:
    def __init__(self, file, hb_info=None):
        """Create a HBFile instance.

        Parameters
        ----------
        file : file-object
            StringIO work as well
        hb_info : HBInfo, optional
            Should be given as an argument for writing, in which case the file
            should be writable.
        """
        self._fid = file
        if hb_info is None:
            self._hb_info = HBInfo.from_file(file)
        else:
            #raise OSError("file %s is not writable, and hb_info "
            #              "was given." % file)
            self._hb_info = hb_info

    @property
    def title(self):
        return self._hb_info.title

    @property
    def key(self):
        return self._hb_info.key

    @property
    def type(self):
        return self._hb_info.mxtype.value_type

    @property
    def structure(self):
        return self._hb_info.mxtype.structure

    @property
    def storage(self):
        return self._hb_info.mxtype.storage

    def read_matrix(self):
        return _read_hb_data(self._fid, self._hb_info)

    def write_matrix(self, m):
        return _write_data(m, self._fid, self._hb_info)


def hb_read(path_or_open_file, *, spmatrix=True):
    """Read HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before reading.
    spmatrix : bool, optional (default: True)
        If ``True``, return sparse ``coo_matrix``. Otherwise return ``coo_array``.

    Returns
    -------
    data : csc_array or csc_matrix
        The data read from the HB file as a sparse array.

    Notes
    -----
    At the moment not the full Harwell-Boeing format is supported. Supported
    features are:

        - assembled, non-symmetric, real matrices
        - integer for pointer/indices
        - exponential format for float values, and int format

    Examples
    --------
    We can read and write a harwell-boeing format file:

    >>> from scipy.io import hb_read, hb_write
    >>> from scipy.sparse import csr_array, eye
    >>> data = csr_array(eye(3))  # create a sparse array
    >>> hb_write("data.hb", data)  # write a hb file
    >>> print(hb_read("data.hb", spmatrix=False))  # read a hb file
    <Compressed Sparse Column sparse array of dtype 'float64'
        with 3 stored elements and shape (3, 3)>
        Coords	Values
        (0, 0)	1.0
        (1, 1)	1.0
        (2, 2)	1.0
    """
    def _get_matrix(fid):
        hb = HBFile(fid)
        return hb.read_matrix()

    if hasattr(path_or_open_file, 'read'):
        data = _get_matrix(path_or_open_file)
    else:
        with open(path_or_open_file) as f:
            data = _get_matrix(f)
    if spmatrix:
        return csc_matrix(data)
    return data


def hb_write(path_or_open_file, m, hb_info=None):
    """Write HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before writing.
    m : sparse array or matrix
        the sparse array to write
    hb_info : HBInfo
        contains the meta-data for write

    Returns
    -------
    None

    Notes
    -----
    At the moment not the full Harwell-Boeing format is supported. Supported
    features are:

        - assembled, non-symmetric, real matrices
        - integer for pointer/indices
        - exponential format for float values, and int format

    Examples
    --------
    We can read and write a harwell-boeing format file:

    >>> from scipy.io import hb_read, hb_write
    >>> from scipy.sparse import csr_array, eye
    >>> data = csr_array(eye(3))  # create a sparse array
    >>> hb_write("data.hb", data)  # write a hb file
    >>> print(hb_read("data.hb", spmatrix=False))  # read a hb file
    <Compressed Sparse Column sparse array of dtype 'float64'
        with 3 stored elements and shape (3, 3)>
        Coords	Values
        (0, 0)	1.0
        (1, 1)	1.0
        (2, 2)	1.0
    """
    m = m.tocsc(copy=False)

    if hb_info is None:
        hb_info = HBInfo.from_data(m)

    def _set_matrix(fid):
        hb = HBFile(fid, hb_info)
        return hb.write_matrix(m)

    if hasattr(path_or_open_file, 'write'):
        return _set_matrix(path_or_open_file)
    else:
        with open(path_or_open_file, 'w') as f:
            return _set_matrix(f)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'find_repeats', 'gmean', 'hmean', 'pmean', 'mode', 'tmean', 'tvar',
    'tmin', 'tmax', 'tstd', 'tsem', 'moment',
    'skew', 'kurtosis', 'describe', 'skewtest', 'kurtosistest',
    'normaltest', 'jarque_bera',
    'scoreatpercentile', 'percentileofscore',
    'cumfreq', 'relfreq', 'obrientransform',
    'sem', 'zmap', 'zscore', 'gzscore', 'iqr', 'gstd',
    'median_abs_deviation',
    'sigmaclip', 'trimboth', 'trim1', 'trim_mean',
    'f_oneway',
    'pearsonr', 'fisher_exact',
    'spearmanr', 'pointbiserialr',
    'kendalltau', 'weightedtau', 'multiscale_graphcorr',
    'linregress', 'siegelslopes', 'theilslopes', 'ttest_1samp',
    'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel',
    'kstest', 'ks_1samp', 'ks_2samp',
    'chisquare', 'power_divergence',
    'tiecorrect', 'ranksums', 'kruskal', 'friedmanchisquare',
    'rankdata',
    'combine_pvalues', 'wasserstein_distance', 'energy_distance',
    'brunnermunzel', 'alexandergovern', 'distributions',
    'mstats_basic',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="stats",
                                   private_modules=["_stats_py", "_mgc"], all=__all__,
                                   attribute=name)
<|endoftext|>#__docformat__ = "restructuredtext en"
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

# A collection of optimization algorithms. Version 0.5
# CHANGES
#  Added fminbound (July 2001)
#  Added brute (Aug. 2002)
#  Finished line search satisfying strong Wolfe conditions (Mar. 2004)
#  Updated strong Wolfe conditions line search to use
#  cubic-interpolation (Mar. 2004)


# Minimization routines

__all__ = ['fmin', 'fmin_powell', 'fmin_bfgs', 'fmin_ncg', 'fmin_cg',
           'fminbound', 'brent', 'golden', 'bracket', 'rosen', 'rosen_der',
           'rosen_hess', 'rosen_hess_prod', 'brute', 'approx_fprime',
           'line_search', 'check_grad', 'OptimizeResult', 'show_options',
           'OptimizeWarning']

__docformat__ = "restructuredtext en"

import math
import warnings
import sys
import inspect
from numpy import eye, argmin, zeros, shape, asarray, sqrt
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
                          line_search_wolfe2 as line_search,
                          LineSearchWarning)
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import (MapWrapper, check_random_state, _RichResult,
                              _call_callback_maybe_halt, _transition_to_rng)
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy._lib._array_api import array_namespace
from scipy._lib import array_api_extra as xpx


# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


class MemoizeJac:
    """Decorator that caches the return values of a function returning ``(fun, grad)``
    each time it is called."""

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]

    def __call__(self, x, *args):
        """ returns the function value """
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac


def _wrap_callback(callback, method=None):
    """Wrap a user-provided callback so that attributes can be attached."""
    if callback is None or method in {'tnc', 'slsqp', 'cobyla', 'cobyqa'}:
        return callback  # don't wrap

    sig = inspect.signature(callback)

    if set(sig.parameters) == {'intermediate_result'}:
        def wrapped_callback(res):
            return callback(intermediate_result=res)
    elif method == 'trust-constr':
        def wrapped_callback(res):
            return callback(np.copy(res.x), res)
    elif method == 'differential_evolution':
        def wrapped_callback(res):
            return callback(np.copy(res.x), res.convergence)
    else:
        def wrapped_callback(res):
            return callback(np.copy(res.x))

    wrapped_callback.stop_iteration = False
    return wrapped_callback


class OptimizeResult(_RichResult):
    """
    Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun : float
        Value of objective function at `x`.
    jac, hess : ndarray
        Values of objective function's Jacobian and its Hessian at `x` (if
        available). The Hessian may be an approximation, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    Depending on the specific solver being used, `OptimizeResult` may
    not have all attributes listed here, and they may have additional
    attributes not listed here. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.

    """
    pass


class OptimizeWarning(UserWarning):
    pass

def _check_positive_definite(Hk):
    def is_pos_def(A):
        if issymmetric(A):
            try:
                cholesky(A)
                return True
            except LinAlgError:
                return False
        else:
            return False
    if Hk is not None:
        if not is_pos_def(Hk):
            raise ValueError("'hess_inv0' matrix isn't positive definite.")


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn(f"Unknown solver options: {msg}", OptimizeWarning, stacklevel=4)


def is_finite_scalar(x):
    """Test whether `x` is either a finite scalar or a finite array scalar.

    """
    return np.size(x) == 1 and np.isfinite(x)


_epsilon = sqrt(np.finfo(float).eps)


def vecnorm(x, ord=2):
    if ord == np.inf:
        return np.amax(np.abs(x))
    elif ord == -np.inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**ord, axis=0)**(1.0 / ord)


def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If ``jac is None`` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf


def _clip_x_for_func(func, bounds):
    # ensures that x values sent to func are clipped to bounds

    # this is used as a mitigation for gh11403, slsqp/tnc sometimes
    # suggest a move that is outside the limits by 1 or 2 ULP. This
    # unclean fix makes sure x is strictly within bounds.
    def eval(x):
        x = _check_clip_x(x, bounds)
        return func(x)

    return eval


def _check_clip_x(x, bounds):
    if (x < bounds[0]).any() or (x > bounds[1]).any():
        warnings.warn("Values in x were outside bounds during a "
                      "minimize step, clipping to bounds",
                      RuntimeWarning, stacklevel=3)
        x = np.clip(x, bounds[0], bounds[1])
        return x

    return x


def rosen(x):
    """
    The Rosenbrock function.

    The function computed is::

        sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    f : float
        The value of the Rosenbrock function.

    See Also
    --------
    rosen_der, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen
    >>> X = 0.1 * np.arange(10)
    >>> rosen(X)
    76.56

    For higher-dimensional input ``rosen`` broadcasts.
    In the following example, we use this to plot a 2D landscape.
    Note that ``rosen_hess`` does not broadcast in this manner.

    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> x = np.linspace(-1, 1, 50)
    >>> X, Y = np.meshgrid(x, x)
    >>> ax = plt.subplot(111, projection='3d')
    >>> ax.plot_surface(X, Y, rosen([X, Y]))
    >>> plt.show()
    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    if xp.isdtype(x.dtype, 'integral'):
        x = xp.astype(x, xp.asarray(1.).dtype)
    r = xp.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
               axis=0, dtype=x.dtype)
    return r


def rosen_der(x):
    """
    The derivative (i.e. gradient) of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the derivative is to be computed.

    Returns
    -------
    rosen_der : (N,) ndarray
        The gradient of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_der
    >>> X = 0.1 * np.arange(9)
    >>> rosen_der(X)
    array([ -2. ,  10.6,  15.6,  13.4,   6.4,  -3. , -12.4, -19.4,  62. ])

    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    if xp.isdtype(x.dtype, 'integral'):
        x = xp.astype(x, xp.asarray(1.).dtype)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = xp.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der


def rosen_hess(x):
    """
    The Hessian matrix of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.

    Returns
    -------
    rosen_hess : ndarray
        The Hessian matrix of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_hess
    >>> X = 0.1 * np.arange(4)
    >>> rosen_hess(X)
    array([[-38.,   0.,   0.,   0.],
           [  0., 134., -40.,   0.],
           [  0., -40., 130., -80.],
           [  0.,   0., -80., 200.]])

    """
    xp = array_namespace(x)
    x = xpx.atleast_nd(x, ndim=1, xp=xp)
    if xp.isdtype(x.dtype, 'integral'):
        x = xp.astype(x, xp.asarray(1.).dtype)
    H = (xpx.create_diagonal(-400 * x[:-1], offset=1, xp=xp) 
         - xpx.create_diagonal(400 * x[:-1], offset=-1, xp=xp))
    diagonal = xp.zeros(x.shape[0], dtype=x.dtype)
    diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
    return H + xpx.create_diagonal(diagonal, xp=xp)


def rosen_hess_prod(x, p):
    """
    Product of the Hessian matrix of the Rosenbrock function with a vector.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.
    p : array_like
        1-D array, the vector to be multiplied by the Hessian matrix.

    Returns
    -------
    rosen_hess_prod : ndarray
        The Hessian matrix of the Rosenbrock function at `x` multiplied
        by the vector `p`.

    See Also
    --------
    rosen, rosen_der, rosen_hess

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_hess_prod
    >>> X = 0.1 * np.arange(9)
    >>> p = 0.5 * np.arange(9)
    >>> rosen_hess_prod(X, p)
    array([  -0.,   27.,  -10.,  -95., -192., -265., -278., -195., -180.])

    """
    xp = array_namespace(x, p)
    x = xpx.atleast_nd(x, ndim=1, xp=xp)
    if xp.isdtype(x.dtype, 'integral'):
        x = xp.astype(x, xp.asarray(1.).dtype)
    p = xp.asarray(p, dtype=x.dtype)
    Hp = xp.zeros(x.shape[0], dtype=x.dtype)
    Hp[0] = (1200 * x[0]**2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
    Hp[1:-1] = (-400 * x[:-2] * p[:-2] +
                (202 + 1200 * x[1:-1]**2 - 400 * x[2:]) * p[1:-1] -
                400 * x[1:-1] * p[2:])
    Hp[-1] = -400 * x[-2] * p[-2] + 200*p[-1]
    return Hp


def _wrap_scalar_function(function, args):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]), np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper


class _MaxFuncCallError(RuntimeError):
    pass


def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]),
        # np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper


def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None):
    """
    Minimize a function using the downhill simplex algorithm.

    This algorithm only uses function values, not derivatives or second
    derivatives.

    Parameters
    ----------
    func : callable func(x,*args)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func, i.e., ``f(x,*args)``.
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    ftol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : number, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.
    disp : bool, optional
        Set to True to print convergence messages.
    retall : bool, optional
        Set to True to return list of solutions at each iteration.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    initial_simplex : array_like of shape (N + 1, N), optional
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.

    Returns
    -------
    xopt : ndarray
        Parameter that minimizes function.
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
    iter : int
        Number of iterations performed.
    funcalls : int
        Number of function calls made.
    warnflag : int
        1 : Maximum number of function evaluations made.
        2 : Maximum number of iterations reached.
    allvecs : list
        Solution at each iteration.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Nelder-Mead' `method` in particular.

    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.

    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice, it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does. Both the ftol and
    xtol criteria must be met for convergence.

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin(f, 1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 17
             Function evaluations: 34
    >>> minimum[0]
    -8.8817841970012523e-16

    References
    ----------
    .. [1] Nelder, J.A. and Mead, R. (1965), "A simplex method for function
           minimization", The Computer Journal, 7, pp. 308-313

    .. [2] Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
           Respectable", in Numerical Analysis 1995, Proceedings of the
           1995 Dundee Biennial Conference in Numerical Analysis, D.F.
           Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
           Harlow, UK, pp. 191-208.

    """
    opts = {'xatol': xtol,
            'fatol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'return_all': retall,
            'initial_simplex': initial_simplex}

    callback = _wrap_callback(callback)
    res = _minimize_neldermead(func, x0, args, callback=callback, **opts)
    if full_output:
        retlist = res['x'], res['fun'], res['nit'], res['nfev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=1e-4, fatol=1e-4, adaptive=False, bounds=None,
                         **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`. None
           is used to specify no bound.

        Note that this just clips all vertices in simplex based on
        the bounds.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277

    """
    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all

    x0 = np.atleast_1d(x0).flatten()
    dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
    x0 = np.asarray(x0, dtype=dtype)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    if bounds is not None:
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError("Nelder Mead - one of the lower bounds "
                             "is greater than an upper bound.")
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, stacklevel=3)

    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)

    if initial_simplex is None:
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(initial_simplex).copy()
        dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
        sim = np.asarray(sim, dtype=dtype)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    if bounds is not None:
        # The default simplex construction may make all entries (for a given
        # parameter) greater than an upper bound if x0 is very close to the
        # upper bound. If one simply clips the simplex to the bounds this could
        # make the simplex entries degenerate. If that occurs reflect into the
        # interior.
        msk = sim > upper_bound
        # reflect into the interior
        sim = np.where(msk, 2*upper_bound - sim, sim)
        # but make sure the reflection is no less than the lower_bound
        sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)

    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    try:
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        try:
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                break

            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)
            fxr = func(xr)
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                fxe = func(xe)

                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        if bounds is not None:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        fxc = func(xc)

                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        if bounds is not None:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        fxcc = func(xcc)

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if bounds is not None:
                                sim[j] = np.clip(
                                    sim[j], lower_bound, upper_bound)
                            fsim[j] = func(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        finally:
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)
            if retall:
                allvecs.append(sim[0])
            intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])
            if _call_callback_maybe_halt(callback, intermediate_result):
                break

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print(f"         Current function value: {fval:f}")
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result


def approx_fprime(xk, f, epsilon=_epsilon, *args):
    """Finite difference approximation of the derivatives of a
    scalar or vector-valued function.

    If a function maps from :math:`R^n` to :math:`R^m`, its derivatives form
    an m-by-n matrix
    called the Jacobian, where an element :math:`(i, j)` is a partial
    derivative of f[i] with respect to ``xk[j]``.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        Function of which to estimate the derivatives of. Has the signature
        ``f(xk, *args)`` where `xk` is the argument in the form of a 1-D array
        and `args` is a tuple of any additional fixed parameters needed to
        completely specify the function. The argument `xk` passed to this
        function is an ndarray of shape (n,) (never a scalar even if n=1).
        It must return a 1-D array_like of shape (m,) or a scalar.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.

        .. versionchanged:: 1.9.0
            `f` is now able to return a 1-D array-like, with the :math:`(m, n)`
            Jacobian being estimated.

    epsilon : {float, array_like}, optional
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives. If an array, should contain one value per element of
        `xk`. Defaults to ``sqrt(np.finfo(float).eps)``, which is approximately
        1.49e-08.
    \\*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    jac : ndarray
        The partial derivatives of `f` to `xk`.

    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import optimize
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(float).eps)
    >>> optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004208])

    """
    xk = np.asarray(xk, float)
    f0 = f(xk, *args)

    return approx_derivative(f, xk, method='2-point', abs_step=epsilon,
                             args=args, f0=f0)


@_transition_to_rng("seed", position_num=6)
def check_grad(func, grad, x0, *args, epsilon=_epsilon,
                direction='all', rng=None):
    r"""Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func : callable ``func(x0, *args)``
        Function whose derivative is to be checked.
    grad : callable ``grad(x0, *args)``
        Jacobian of `func`.
    x0 : ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args : \\*args, optional
        Extra arguments passed to `func` and `grad`.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        ``sqrt(np.finfo(float).eps)``, which is approximately 1.49e-08.
    direction : str, optional
        If set to ``'random'``, then gradients along a random vector
        are used to check `grad` against forward difference approximation
        using `func`. By default it is ``'all'``, in which case, all
        the one hot direction vectors are considered to check `grad`.
        If `func` is a vector valued function then only ``'all'`` can be used.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        The random numbers generated affect the random vector along which gradients
        are computed to check ``grad``. Note that `rng` is only used when `direction`
        argument is set to `'random'`.

    Returns
    -------
    err : float
        The square root of the sum of squares (i.e., the 2-norm) of the
        difference between ``grad(x0, *args)`` and the finite difference
        approximation of `grad` using func at the points `x0`.

    See Also
    --------
    approx_fprime

    Examples
    --------
    >>> import numpy as np
    >>> def func(x):
    ...     return x[0]**2 - 0.5 * x[1]**3
    >>> def grad(x):
    ...     return [2 * x[0], -1.5 * x[1]**2]
    >>> from scipy.optimize import check_grad
    >>> check_grad(func, grad, [1.5, -1.5])
    2.9802322387695312e-08  # may vary
    >>> rng = np.random.default_rng()
    >>> check_grad(func, grad, [1.5, -1.5],
    ...             direction='random', seed=rng)
    2.9802322387695312e-08

    """
    step = epsilon
    x0 = np.asarray(x0)

    def g(w, func, x0, v, *args):
        return func(x0 + w*v, *args)

    if direction == 'random':
        _grad = np.asanyarray(grad(x0, *args))
        if _grad.ndim > 1:
            raise ValueError("'random' can only be used with scalar valued"
                             " func")
        rng_gen = check_random_state(rng)
        v = rng_gen.standard_normal(size=(x0.shape))
        _args = (func, x0, v) + args
        _func = g
        vars = np.zeros((1,))
        analytical_grad = np.dot(_grad, v)
    elif direction == 'all':
        _args = args
        _func = func
        vars = x0
        analytical_grad = grad(x0, *args)
    else:
        raise ValueError(f"{direction} is not a valid string for "
                         "``direction`` argument")

    return np.sqrt(np.sum(np.abs(
        (analytical_grad - approx_fprime(vars, _func, step, *_args))**2
    )))


def approx_fhess_p(x0, p, fprime, epsilon, *args):
    # calculate fprime(x0) first, as this may be cached by ScalarFunction
    f1 = fprime(*((x0,) + args))
    f2 = fprime(*((x0 + epsilon*p,) + args))
    return (f2 - f1) / epsilon


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None, xrtol=0, c1=1e-4, c2=0.9,
              hess_inv0=None):
    """
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable ``f(x,*args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess, shape (n,)
    fprime : callable ``f'(x,*args)``, optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Terminate successfully if gradient norm is less than `gtol`
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If `fprime` is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration. Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return ``fopt``, ``func_calls``, ``grad_calls``, and
        ``warnflag`` in addition to ``xopt``.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step
        size is less than ``xk * xrtol`` where ``xk`` is the current
        parameter vector.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.
    hess_inv0 : None or ndarray, optional``
        Initial inverse hessian estimate, shape (n, n). If None (default) then
        the identity matrix is used.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e., the inverse Hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    allvecs : list
        The value of `xopt` at each iteration. Only returned if `retall` is
        True.

    Notes
    -----
    Optimize the function, `f`, whose gradient is given by `fprime`
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS).

    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    See Also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See ``method='BFGS'`` in particular.

    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, p. 198.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import fmin_bfgs
    >>> def quadratic_cost(x, Q):
    ...     return x @ Q @ x
    ...
    >>> x0 = np.array([-3, -4])
    >>> cost_weight =  np.diag([1., 10.])
    >>> # Note that a trailing comma is necessary for a tuple with single element
    >>> fmin_bfgs(quadratic_cost, x0, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7                   # may vary
            Function evaluations: 24        # may vary
            Gradient evaluations: 8         # may vary
    array([ 2.85169950e-06, -4.61820139e-07])

    >>> def quadratic_cost_grad(x, Q):
    ...     return 2 * Q @ x
    ...
    >>> fmin_bfgs(quadratic_cost, x0, quadratic_cost_grad, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7
            Function evaluations: 8
            Gradient evaluations: 8
    array([ 2.85916637e-06, -4.54371951e-07])

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall,
            'xrtol': xrtol,
            'c1': c1,
            'c2': c2,
            'hess_inv0': hess_inv0}

    callback = _wrap_callback(callback)
    res = _minimize_bfgs(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=np.inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   xrtol=0, c1=1e-4, c2=0.9,
                   hess_inv0=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.
    hess_inv0 : None or ndarray, optional
        Initial inverse hessian estimate, shape (n, n). If None (default) then
        the identity matrix is used.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    If minimization doesn't complete successfully, with an error message of
    ``Desired error not necessarily achieved due to precision loss``, then
    consider setting `gtol` to a higher value. This precision loss typically
    occurs when the (finite difference) numerical differentiation cannot provide
    sufficient precision to satisfy the `gtol` termination criterion.
    This can happen when working in single precision and a callable jac is not
    provided. For single precision problems a `gtol` of 1e-3 seems to work.
    """
    _check_unknown_options(unknown_options)
    _check_positive_definite(hess_inv0)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I if hess_inv0 is None else hess_inv0

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100,
                                          amax=1e100, c1=c1, c2=c2)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        sk = alpha_k * pk
        xkp1 = xk + sk

        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        #  See Chapter 5 in  P.E. Frandsen, K. Jonasson, H.B. Nielsen,
        #  O. Tingleff: "Unconstrained Optimization", IMM, DTU.  1999.
        #  These notes are available here:
        #  http://www2.imm.dtu.dk/documents/ftp/publlec.html
        if (alpha_k*vecnorm(pk) <= xrtol*(xrtol + vecnorm(xk))):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remains for more safety
        # Cryptic comment above is preserved for posterity. Future reader:
        # consider change to condition below proposed in gh-1261/gh-17345.
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                msg = "Divide-by-zero encountered: rhok assumed large"
                _print_success_message_or_warn(True, msg)
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])

    fval = old_fval

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print(f"         Current function value: {fval:f}")
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _print_success_message_or_warn(warnflag, message, warntype=None):
    if not warnflag:
        print(message)
    else:
        warnings.warn(message, warntype or OptimizeWarning, stacklevel=3)


def fmin_cg(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
            epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
            callback=None, c1=1e-4, c2=0.4):
    """
    Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    f : callable, ``f(x, *args)``
        Objective function to be minimized. Here `x` must be a 1-D array of
        the variables that are to be changed in the search for a minimum, and
        `args` are the other (fixed) parameters of `f`.
    x0 : ndarray
        A user-supplied initial estimate of `xopt`, the optimal value of `x`.
        It must be a 1-D array of values.
    fprime : callable, ``fprime(x, *args)``, optional
        A function that returns the gradient of `f` at `x`. Here `x` and `args`
        are as described above for `f`. The returned value must be a 1-D array.
        Defaults to None, in which case the gradient is approximated
        numerically (see `epsilon`, below).
    args : tuple, optional
        Parameter values passed to `f` and `fprime`. Must be supplied whenever
        additional fixed parameters are needed to completely specify the
        functions `f` and `fprime`.
    gtol : float, optional
        Stop when the norm of the gradient is less than `gtol`.
    norm : float, optional
        Order to use for the norm of the gradient
        (``-np.inf`` is min, ``np.inf`` is max).
    epsilon : float or ndarray, optional
        Step size(s) to use when `fprime` is approximated numerically. Can be a
        scalar or a 1-D array. Defaults to ``sqrt(eps)``, with eps the
        floating point machine precision.  Usually ``sqrt(eps)`` is about
        1.5e-8.
    maxiter : int, optional
        Maximum number of iterations to perform. Default is ``200 * len(x0)``.
    full_output : bool, optional
        If True, return `fopt`, `func_calls`, `grad_calls`, and `warnflag` in
        addition to `xopt`.  See the Returns section below for additional
        information on optional return values.
    disp : bool, optional
        If True, return a convergence message, followed by `xopt`.
    retall : bool, optional
        If True, add to the returned values the results of each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each iteration.
        Called as ``callback(xk)``, where ``xk`` is the current value of `x0`.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.4
        Parameter for curvature condition rule.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float, optional
        Minimum value found, f(xopt). Only returned if `full_output` is True.
    func_calls : int, optional
        The number of function_calls made. Only returned if `full_output`
        is True.
    grad_calls : int, optional
        The number of gradient calls made. Only returned if `full_output` is
        True.
    warnflag : int, optional
        Integer value with warning status, only returned if `full_output` is
        True.

        0 : Success.

        1 : The maximum number of iterations was exceeded.

        2 : Gradient and/or function calls were not changing. May indicate
            that precision was lost, i.e., the routine did not converge.

        3 : NaN result encountered.

    allvecs : list of ndarray, optional
        List of arrays, containing the results at each iteration.
        Only returned if `retall` is True.

    See Also
    --------
    minimize : common interface to all `scipy.optimize` algorithms for
               unconstrained and constrained minimization of multivariate
               functions. It provides an alternative way to call
               ``fmin_cg``, by specifying ``method='CG'``.

    Notes
    -----
    This conjugate gradient algorithm is based on that of Polak and Ribiere
    [1]_.

    Conjugate gradient methods tend to work better when:

    1. `f` has a unique global minimizing point, and no local minima or
       other stationary points,
    2. `f` is, at least locally, reasonably well approximated by a
       quadratic function of the variables,
    3. `f` is continuous and has a continuous gradient,
    4. `fprime` is not too large, e.g., has a norm less than 1000,
    5. The initial guess, `x0`, is reasonably close to `f` 's global
       minimizing point, `xopt`.

    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    References
    ----------
    .. [1] Wright & Nocedal, "Numerical Optimization", 1999, pp. 120-122.

    Examples
    --------
    Example 1: seek the minimum value of the expression
    ``a*u**2 + b*u*v + c*v**2 + d*u + e*v + f`` for given values
    of the parameters and an initial guess ``(u, v) = (0, 0)``.

    >>> import numpy as np
    >>> args = (2, 3, 7, 8, 9, 10)  # parameter values
    >>> def f(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    >>> def gradf(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     gu = 2*a*u + b*v + d     # u-component of the gradient
    ...     gv = b*u + 2*c*v + e     # v-component of the gradient
    ...     return np.asarray((gu, gv))
    >>> x0 = np.asarray((0, 0))  # Initial guess.
    >>> from scipy import optimize
    >>> res1 = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
    Optimization terminated successfully.
             Current function value: 1.617021
             Iterations: 4
             Function evaluations: 8
             Gradient evaluations: 8
    >>> res1
    array([-1.80851064, -0.25531915])

    Example 2: solve the same problem using the `minimize` function.
    (This `myopts` dictionary shows all of the available options,
    although in practice only non-default values would be needed.
    The returned value will be a dictionary.)

    >>> opts = {'maxiter' : None,    # default value.
    ...         'disp' : True,    # non-default value.
    ...         'gtol' : 1e-5,    # default value.
    ...         'norm' : np.inf,  # default value.
    ...         'eps' : 1.4901161193847656e-08}  # default value.
    >>> res2 = optimize.minimize(f, x0, jac=gradf, args=args,
    ...                          method='CG', options=opts)
    Optimization terminated successfully.
            Current function value: 1.617021
            Iterations: 4
            Function evaluations: 8
            Gradient evaluations: 8
    >>> res2.x  # minimum found
    array([-1.80851064, -0.25531915])

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    callback = _wrap_callback(callback)
    res = _minimize_cg(f, x0, args, fprime, callback=callback, c1=c1, c2=c2,
                       **opts)

    if full_output:
        retlist = res['x'], res['fun'], res['nfev'], res['njev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_cg(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, norm=np.inf, eps=_epsilon, maxiter=None,
                 disp=False, return_all=False, finite_diff_rel_step=None,
                 c1=1e-4, c2=0.4, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.4
        Parameter for curvature condition rule.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)

    retall = return_all

    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    xk = x0
    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, ord=norm)

    sigma_3 = 0.01

    while (gnorm > gtol) and (k < maxiter):
        deltak = np.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            #
            # See Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step

            # Accept step if it leads to convergence.
            if gnorm <= gtol:
                return True

            # Accept step if sufficient descent condition applies.
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
                                          old_old_fval, c1=c1, c2=c2, amin=1e-100,
                                          amax=1e100, extra_condition=descent_condition)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)

        if retall:
            allvecs.append(xk)
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print(f"         Current function value: {fval:f}")
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def fmin_ncg(f, x0, fprime, fhess_p=None, fhess=None, args=(), avextol=1e-5,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
             callback=None, c1=1e-4, c2=0.9):
    """
    Unconstrained minimization of a function using the Newton-CG method.

    Parameters
    ----------
    f : callable ``f(x, *args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x, *args)``
        Gradient of f.
    fhess_p : callable ``fhess_p(x, p, *args)``, optional
        Function which computes the Hessian of f times an
        arbitrary vector, p.
    fhess : callable ``fhess(x, *args)``, optional
        Function to compute the Hessian matrix of f.
    args : tuple, optional
        Extra arguments passed to f, fprime, fhess_p, and fhess
        (the same set of extra arguments is supplied to all of
        these functions).
    epsilon : float or ndarray, optional
        If fhess is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function which is called after
        each iteration. Called as callback(xk), where xk is the
        current parameter vector.
    avextol : float, optional
        Convergence is assumed when the average relative error in
        the minimizer falls below this amount.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return the optional outputs.
    disp : bool, optional
        If True, print convergence message.
    retall : bool, optional
        If True, return a list of results at each iteration.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Value of the function at xopt, i.e., ``fopt = f(xopt)``.
    fcalls : int
        Number of function calls made.
    gcalls : int
        Number of gradient calls made.
    hcalls : int
        Number of Hessian calls made.
    warnflag : int
        Warnings generated by the algorithm.
        1 : Maximum number of iterations exceeded.
        2 : Line search failure (precision loss).
        3 : NaN result encountered.
    allvecs : list
        The result at each iteration, if retall is True (see below).

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Newton-CG' `method` in particular.

    Notes
    -----
    Only one of `fhess_p` or `fhess` need to be given.  If `fhess`
    is provided, then `fhess_p` will be ignored. If neither `fhess`
    nor `fhess_p` is provided, then the hessian product will be
    approximated using finite differences on `fprime`. `fhess_p`
    must compute the hessian times an arbitrary vector. If it is not
    given, finite-differences on `fprime` are used to compute
    it.

    Newton-CG methods are also called truncated Newton methods. This
    function differs from scipy.optimize.fmin_tnc because

    1. scipy.optimize.fmin_ncg is written purely in Python using NumPy
        and scipy while scipy.optimize.fmin_tnc calls a C function.
    2. scipy.optimize.fmin_ncg is only for unconstrained minimization
        while scipy.optimize.fmin_tnc is for unconstrained minimization
        or box constrained minimization. (Box constraints give
        lower and upper bounds for each variable separately.)

    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    References
    ----------
    Wright & Nocedal, 'Numerical Optimization', 1999, p. 140.

    """
    opts = {'xtol': avextol,
            'eps': epsilon,
            'maxiter': maxiter,
            'disp': disp,
            'return_all': retall}

    callback = _wrap_callback(callback)
    res = _minimize_newtoncg(f, x0, args, fprime, fhess, fhess_p,
                             callback=callback, c1=c1, c2=c2, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['nfev'], res['njev'],
                   res['nhev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                       disp=False, return_all=False, c1=1e-4, c2=0.9,
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    # TODO: add hessp (callable or FD) to ScalarFunction?
    sf = _prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )
    f = sf.fun
    fprime = sf.grad
    _h = sf.hess(x0)

    # Logic for hess/hessp
    # - If a callable(hess) is provided, then use that
    # - If hess is a FD_METHOD, or the output from hess(x) is a LinearOperator
    #   then create a hessp function using those.
    # - If hess is None but you have callable(hessp) then use the hessp.
    # - If hess and hessp are None then approximate hessp using the grad/jac.

    if (hess in FD_METHODS or isinstance(_h, LinearOperator)):
        fhess = None

        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)

        fhess_p = _hessp

    def terminate(warnflag, msg):
        if disp:
            _print_success_message_or_warn(warnflag, msg)
            print(f"         Current function value: {old_fval:f}")
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    # Make sure we enter the while loop.
    update_l1norm = np.finfo(float).max
    xk = np.copy(x0)
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
    while update_l1norm > xtol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.linalg.norm(b, ord=1)
        eta = min(0.5, math.sqrt(maggrad))
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = sf.hess(xk)
            hcalls += 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls += 1
            else:
                # hess was supplied as a callable or hessian update strategy, so
                # A is a dense numpy array or sparse matrix
                Ap = A.dot(psupi)
            # check curvature
            Ap = asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi += alphai * psupi
            ri += alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i += 1
            dri0 = dri1          # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval, c1=c1, c2=c2)
        except _LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk += update        # upcast if necessary
        if retall:
            allvecs.append(xk)
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            return terminate(5, "")
        update_l1norm = np.linalg.norm(update, ord=1)

    else:
        if np.isnan(old_fval) or np.isnan(update_l1norm):
            return terminate(3, _status_message['nan'])

        msg = _status_message['success']
        return terminate(0, msg)


def fminbound(func, x1, x2, args=(), xtol=1e-5, maxfun=500,
              full_output=0, disp=1):
    """Bounded minimization for scalar functions.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized (must accept and return scalars).
    x1, x2 : float or array scalar
        Finite optimization bounds.
    args : tuple, optional
        Extra arguments passed to function.
    xtol : float, optional
        The convergence tolerance.
    maxfun : int, optional
        Maximum number of function evaluations allowed.
    full_output : bool, optional
        If True, return optional outputs.
    disp: int, optional
        If non-zero, print messages.

        ``0`` : no message printing.

        ``1`` : non-convergence notification messages only.

        ``2`` : print a message on convergence too.

        ``3`` : print iteration results.

    Returns
    -------
    xopt : ndarray
        Parameters (over given interval) which minimize the
        objective function.
    fval : number
        (Optional output) The function value evaluated at the minimizer.
    ierr : int
        (Optional output) An error flag (0 if converged, 1 if maximum number of
        function calls reached).
    numfunc : int
        (Optional output) The number of function calls made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Bounded' `method` in particular.

    Notes
    -----
    Finds a local minimizer of the scalar function `func` in the
    interval x1 < xopt < x2 using Brent's method. (See `brent`
    for auto-bracketing.)

    References
    ----------
    .. [1] Forsythe, G.E., M. A. Malcolm, and C. B. Moler. "Computer Methods
           for Mathematical Computations." Prentice-Hall Series in Automatic
           Computation 259 (1977).
    .. [2] Brent, Richard P. Algorithms for Minimization Without Derivatives.
           Courier Corporation, 2013.

    Examples
    --------
    `fminbound` finds the minimizer of the function in the given range.
    The following examples illustrate this.

    >>> from scipy import optimize
    >>> def f(x):
    ...     return (x-1)**2
    >>> minimizer = optimize.fminbound(f, -4, 4)
    >>> minimizer
    1.0
    >>> minimum = f(minimizer)
    >>> minimum
    0.0
    >>> res = optimize.fminbound(f, 3, 4, full_output=True)
    >>> minimizer, fval, ierr, numfunc = res
    >>> minimizer
    3.000005960860986
    >>> minimum = f(minimizer)
    >>> minimum, fval
    (4.000023843479476, 4.000023843479476)
    """
    options = {'xatol': xtol,
               'maxiter': maxfun,
               'disp': disp}

    res = _minimize_scalar_bounded(func, (x1, x2), args, **options)
    if full_output:
        return res['x'], res['fun'], res['status'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.

        ``0`` : no message printing.

        ``1`` : non-convergence notification messages only.

        ``2`` : print a message on convergence too.

        ``3`` : print iteration results.

    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if not (is_finite_scalar(x1) and is_finite_scalar(x2)):
        raise ValueError("Optimization bounds must be finite scalars.")

    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.',
                                     2: _status_message['nan']}.get(flag, ''),
                            x=xf, nfev=num, nit=num)

    return result


class Brent:
    #need to rethink design of __init__
    def __init__(self, func, args=(), tol=1.48e-8, maxiter=500,
                 full_output=0, disp=0):
        self.func = func
        self.args = args
        self.tol = tol
        self.maxiter = maxiter
        self._mintol = 1.0e-11
        self._cg = 0.3819660
        self.xmin = None
        self.fval = None
        self.iter = 0
        self.funcalls = 0
        self.disp = disp

    # need to rethink design of set_bracket (new options, etc.)
    def set_bracket(self, brack=None):
        self.brack = brack

    def get_bracket_info(self):
        #set up
        func = self.func
        args = self.args
        brack = self.brack
        ### BEGIN core bracket_info code ###
        ### carefully DOCUMENT any CHANGES in core ##
        if brack is None:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
        elif len(brack) == 2:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                       xb=brack[1], args=args)
        elif len(brack) == 3:
            xa, xb, xc = brack
            if (xa > xc):  # swap so xa < xc can be assumed
                xc, xa = xa, xc
            if not ((xa < xb) and (xb < xc)):
                raise ValueError(
                    "Bracketing values (xa, xb, xc) do not"
                    " fulfill this requirement: (xa < xb) and (xb < xc)"
                )
            fa = func(*((xa,) + args))
            fb = func(*((xb,) + args))
            fc = func(*((xc,) + args))
            if not ((fb < fa) and (fb < fc)):
                raise ValueError(
                    "Bracketing values (xa, xb, xc) do not fulfill"
                    " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
                )

            funcalls = 3
        else:
            raise ValueError("Bracketing interval must be "
                             "length 2 or 3 sequence.")
        ### END core bracket_info code ###

        return xa, xb, xc, fa, fb, fc, funcalls

    def optimize(self):
        # set up for optimization
        func = self.func
        xa, xb, xc, fa, fb, fc, funcalls = self.get_bracket_info()
        _mintol = self._mintol
        _cg = self._cg
        #################################
        #BEGIN CORE ALGORITHM
        #################################
        x = w = v = xb
        fw = fv = fx = fb
        if (xa < xc):
            a = xa
            b = xc
        else:
            a = xc
            b = xa
        deltax = 0.0
        iter = 0

        if self.disp > 2:
            print(" ")
            print(f"{'Func-count':^12} {'x':^12} {'f(x)': ^12}")
            print(f"{funcalls:^12g} {x:^12.6g} {fx:^12.6g}")

        while (iter < self.maxiter):
            tol1 = self.tol * np.abs(x) + _mintol
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a + b)
            # check for convergence
            if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
                break
            # XXX In the first iteration, rat is only bound in the true case
            # of this conditional. This used to cause an UnboundLocalError
            # (gh-4140). It should be set before the if (but to what?).
            if (np.abs(deltax) <= tol1):
                if (x >= xmid):
                    deltax = a - x       # do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax
            else:                              # do a parabolic step
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = 2.0 * (tmp2 - tmp1)
                if (tmp2 > 0.0):
                    p = -p
                tmp2 = np.abs(tmp2)
                dx_temp = deltax
                deltax = rat
                # check parabolic fit
                if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                        (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                    rat = p * 1.0 / tmp2        # if parabolic step is useful.
                    u = x + rat
                    if ((u - a) < tol2 or (b - u) < tol2):
                        if xmid - x >= 0:
                            rat = tol1
                        else:
                            rat = -tol1
                else:
                    if (x >= xmid):
                        deltax = a - x  # if it's not do a golden section step
                    else:
                        deltax = b - x
                    rat = _cg * deltax

            if (np.abs(rat) < tol1):            # update by at least tol1
                if rat >= 0:
                    u = x + tol1
                else:
                    u = x - tol1
            else:
                u = x + rat
            fu = func(*((u,) + self.args))      # calculate new output value
            funcalls += 1

            if (fu > fx):                 # if it's bigger than current
                if (u < x):
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v = u
                    fv = fu
            else:
                if (u >= x):
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu

            if self.disp > 2:
                print(f"{funcalls:^12g} {x:^12.6g} {fx:^12.6g}")

            iter += 1
        #################################
        #END CORE ALGORITHM
        #################################

        self.xmin = x
        self.fval = fx
        self.iter = iter
        self.funcalls = funcalls

    def get_result(self, full_output=False):
        if full_output:
            return self.xmin, self.fval, self.iter, self.funcalls
        else:
            return self.xmin


def brent(func, args=(), brack=None, tol=1.48e-8, full_output=0, maxiter=500):
    """
    Given a function of one variable and a possible bracket, return
    a local minimizer of the function isolated to a fractional precision
    of tol.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function.
    args : tuple, optional
        Additional arguments (if present).
    brack : tuple, optional
        Either a triple ``(xa, xb, xc)`` satisfying ``xa < xb < xc`` and
        ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair
        ``(xa, xb)`` to be used as initial points for a downhill bracket search
        (see `scipy.optimize.bracket`).
        The minimizer ``x`` will not necessarily satisfy ``xa <= x <= xb``.
    tol : float, optional
        Relative error in solution `xopt` acceptable for convergence.
    full_output : bool, optional
        If True, return all output args (xmin, fval, iter,
        funcalls).
    maxiter : int, optional
        Maximum number of iterations in solution.

    Returns
    -------
    xmin : ndarray
        Optimum point.
    fval : float
        (Optional output) Optimum function value.
    iter : int
        (Optional output) Number of iterations.
    funcalls : int
        (Optional output) Number of objective function evaluations made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Brent' `method` in particular.

    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    Does not ensure that the minimum lies in the range specified by
    `brack`. See `scipy.optimize.fminbound`.

    Examples
    --------
    We illustrate the behaviour of the function when `brack` is of
    size 2 and 3 respectively. In the case where `brack` is of the
    form ``(xa, xb)``, we can see for the given values, the output does
    not necessarily lie in the range ``(xa, xb)``.

    >>> def f(x):
    ...     return (x-1)**2

    >>> from scipy import optimize

    >>> minimizer = optimize.brent(f, brack=(1, 2))
    >>> minimizer
    1
    >>> res = optimize.brent(f, brack=(-1, 0.5, 2), full_output=True)
    >>> xmin, fval, iter, funcalls = res
    >>> f(xmin), fval
    (0.0, 0.0)

    """
    options = {'xtol': tol,
               'maxiter': maxiter}
    res = _minimize_scalar_brent(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nit'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_brent(func, brack=None, args=(), xtol=1.48e-8,
                           maxiter=500, disp=0,
                           **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    disp : int, optional
        If non-zero, print messages.

        ``0`` : no message printing.

        ``1`` : non-convergence notification messages only.

        ``2`` : print a message on convergence too.

        ``3`` : print iteration results.

    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    """
    _check_unknown_options(unknown_options)
    tol = xtol
    if tol < 0:
        raise ValueError(f'tolerance should be >= 0, got {tol!r}')

    brent = Brent(func=func, args=args, tol=tol,
                  full_output=True, maxiter=maxiter, disp=disp)
    brent.set_bracket(brack)
    brent.optimize()
    x, fval, nit, nfev = brent.get_result(full_output=True)

    success = nit < maxiter and not (np.isnan(x) or np.isnan(fval))

    if success:
        message = ("\nOptimization terminated successfully;\n"
                   "The returned value satisfies the termination criteria\n"
                   f"(using xtol = {xtol} )")
    else:
        if nit >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        if np.isnan(x) or np.isnan(fval):
            message = f"{_status_message['nan']}"

    if disp:
        _print_success_message_or_warn(not success, message)

    return OptimizeResult(fun=fval, x=x, nit=nit, nfev=nfev,
                          success=success, message=message)


def golden(func, args=(), brack=None, tol=_epsilon,
           full_output=0, maxiter=5000):
    """
    Return the minimizer of a function of one variable using the golden section
    method.

    Given a function of one variable and a possible bracketing interval,
    return a minimizer of the function isolated to a fractional precision of
    tol.

    Parameters
    ----------
    func : callable func(x,*args)
        Objective function to minimize.
    args : tuple, optional
        Additional arguments (if present), passed to func.
    brack : tuple, optional
        Either a triple ``(xa, xb, xc)`` where ``xa < xb < xc`` and
        ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair (xa, xb)
        to be used as initial points for a downhill bracket search (see
        `scipy.optimize.bracket`).
        The minimizer ``x`` will not necessarily satisfy ``xa <= x <= xb``.
    tol : float, optional
        x tolerance stop criterion
    full_output : bool, optional
        If True, return optional outputs.
    maxiter : int
        Maximum number of iterations to perform.

    Returns
    -------
    xmin : ndarray
        Optimum point.
    fval : float
