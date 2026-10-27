        out_buffer = ffi.new("ZSTD_outBuffer *")

        in_buffer.src = ffi.NULL
        in_buffer.size = 0
        in_buffer.pos = 0

        dst_buffer = ffi.new("char[]", write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = write_size
        out_buffer.pos = 0

        while True:
            # We should never have output data sitting around after a previous
            # iteration.
            assert out_buffer.pos == 0

            # Collect input data.
            if have_read:
                read_result = reader.read(read_size)
            else:
                remaining = len(reader) - buffer_offset
                slice_size = min(remaining, read_size)
                read_result = reader[buffer_offset : buffer_offset + slice_size]
                buffer_offset += slice_size

            # No new input data. Break out of the read loop.
            if not read_result:
                break

            # Feed all read data into the compressor and emit output until
            # exhausted.
            read_buffer = ffi.from_buffer(read_result)
            in_buffer.src = read_buffer
            in_buffer.size = len(read_buffer)
            in_buffer.pos = 0

            while in_buffer.pos < in_buffer.size:
                zresult = lib.ZSTD_compressStream2(
                    self._cctx, out_buffer, in_buffer, lib.ZSTD_e_continue
                )
                if lib.ZSTD_isError(zresult):
                    raise ZstdError(
                        "zstd compress error: %s" % _zstd_error(zresult)
                    )

                if out_buffer.pos:
                    data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                    out_buffer.pos = 0
                    yield data

            assert out_buffer.pos == 0

            # And repeat the loop to collect more data.
            continue

        # If we get here, input is exhausted. End the stream and emit what
        # remains.
        while True:
            assert out_buffer.pos == 0
            zresult = lib.ZSTD_compressStream2(
                self._cctx, out_buffer, in_buffer, lib.ZSTD_e_end
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "error ending compression stream: %s" % _zstd_error(zresult)
                )

            if out_buffer.pos:
                data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                out_buffer.pos = 0
                yield data

            if zresult == 0:
                break

    def multi_compress_to_buffer(self, data, threads=-1):
        """
        Compress multiple pieces of data as a single function call.

        (Experimental. Not yet supported by CFFI backend.)

        This function is optimized to perform multiple compression operations
        as as possible with as little overhead as possible.

        Data to be compressed can be passed as a ``BufferWithSegmentsCollection``,
        a ``BufferWithSegments``, or a list containing byte like objects. Each
        element of the container will be compressed individually using the
        configured parameters on the ``ZstdCompressor`` instance.

        The ``threads`` argument controls how many threads to use for
        compression. The default is ``0`` which means to use a single thread.
        Negative values use the number of logical CPUs in the machine.

        The function returns a ``BufferWithSegmentsCollection``. This type
        represents N discrete memory allocations, each holding 1 or more
        compressed frames.

        Output data is written to shared memory buffers. This means that unlike
        regular Python objects, a reference to *any* object within the collection
        keeps the shared buffer and therefore memory backing it alive. This can
        have undesirable effects on process memory usage.

        The API and behavior of this function is experimental and will likely
        change. Known deficiencies include:

        * If asked to use multiple threads, it will always spawn that many
          threads, even if the input is too small to use them. It should
          automatically lower the thread count when the extra threads would
          just add overhead.
        * The buffer allocation strategy is fixed. There is room to make it
          dynamic, perhaps even to allow one output buffer per input,
          facilitating a variation of the API to return a list without the
          adverse effects of shared memory buffers.

        :param data:
           Source to read discrete pieces of data to compress.

           Can be a ``BufferWithSegmentsCollection``, a ``BufferWithSegments``,
           or a ``list[bytes]``.
        :return:
           BufferWithSegmentsCollection holding compressed data.
        """
        raise NotImplementedError()

    def frame_progression(self):
        """
        Return information on how much work the compressor has done.

        Returns a 3-tuple of (ingested, consumed, produced).

        >>> cctx = zstandard.ZstdCompressor()
        >>> (ingested, consumed, produced) = cctx.frame_progression()
        """
        progression = lib.ZSTD_getFrameProgression(self._cctx)

        return progression.ingested, progression.consumed, progression.produced


class FrameParameters(object):
    """Information about a zstd frame.

    Instances have the following attributes:

    ``content_size``
       Integer size of original, uncompressed content. This will be ``0`` if the
       original content size isn't written to the frame (controlled with the
       ``write_content_size`` argument to ``ZstdCompressor``) or if the input
       content size was ``0``.

    ``window_size``
       Integer size of maximum back-reference distance in compressed data.

    ``dict_id``
       Integer of dictionary ID used for compression. ``0`` if no dictionary
       ID was used or if the dictionary ID was ``0``.

    ``has_checksum``
       Bool indicating whether a 4 byte content checksum is stored at the end
       of the frame.
    """

    def __init__(self, fparams):
        self.content_size = fparams.frameContentSize
        self.window_size = fparams.windowSize
        self.dict_id = fparams.dictID
        self.has_checksum = bool(fparams.checksumFlag)


def frame_content_size(data):
    """Obtain the decompressed size of a frame.

    The returned value is usually accurate. But strictly speaking it should
    not be trusted.

    :return:
       ``-1`` if size unknown and a non-negative integer otherwise.
    """
    data_buffer = ffi.from_buffer(data)

    size = lib.ZSTD_getFrameContentSize(data_buffer, len(data_buffer))

    if size == lib.ZSTD_CONTENTSIZE_ERROR:
        raise ZstdError("error when determining content size")
    elif size == lib.ZSTD_CONTENTSIZE_UNKNOWN:
        return -1
    else:
        return size


def frame_header_size(data):
    """Obtain the size of a frame header.

    :return:
       Integer size in bytes.
    """
    data_buffer = ffi.from_buffer(data)

    zresult = lib.ZSTD_frameHeaderSize(data_buffer, len(data_buffer))
    if lib.ZSTD_isError(zresult):
        raise ZstdError(
            "could not determine frame header size: %s" % _zstd_error(zresult)
        )

    return zresult


def get_frame_parameters(data, format=FORMAT_ZSTD1):
    """
    Parse a zstd frame header into frame parameters.

    Depending on which fields are present in the frame and their values, the
    length of the frame parameters varies. If insufficient bytes are passed
    in to fully parse the frame parameters, ``ZstdError`` is raised. To ensure
    frame parameters can be parsed, pass in at least 18 bytes.

    :param data:
       Data from which to read frame parameters.
    :param format:
       Set the format of data for the decoder.
    :return:
       :py:class:`FrameParameters`
    """
    params = ffi.new("ZSTD_FrameHeader *")

    data_buffer = ffi.from_buffer(data)
    zresult = lib.ZSTD_getFrameHeader_advanced(
        params, data_buffer, len(data_buffer), format
    )
    if lib.ZSTD_isError(zresult):
        raise ZstdError(
            "cannot get frame parameters: %s" % _zstd_error(zresult)
        )

    if zresult:
        raise ZstdError(
            "not enough data for frame parameters; need %d bytes" % zresult
        )

    return FrameParameters(params[0])


class ZstdCompressionDict(object):
    """Represents a computed compression dictionary.

    Instances are obtained by calling :py:func:`train_dictionary` or by
    passing bytes obtained from another source into the constructor.

    Instances can be constructed from bytes:

    >>> dict_data = zstandard.ZstdCompressionDict(data)

    It is possible to construct a dictionary from *any* data. If the data
    doesn't begin with a magic header, it will be treated as a *prefix*
    dictionary. *Prefix* dictionaries allow compression operations to
    reference raw data within the dictionary.

    It is possible to force the use of *prefix* dictionaries or to require
    a dictionary header:

    >>> dict_data = zstandard.ZstdCompressionDict(data, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    >>> dict_data = zstandard.ZstdCompressionDict(data, dict_type=zstandard.DICT_TYPE_FULLDICT)

    You can see how many bytes are in the dictionary by calling ``len()``:

    >>> dict_data = zstandard.train_dictionary(size, samples)
    >>> dict_size = len(dict_data)  # will not be larger than ``size``

    Once you have a dictionary, you can pass it to the objects performing
    compression and decompression:

    >>> dict_data = zstandard.train_dictionary(131072, samples)
    >>> cctx = zstandard.ZstdCompressor(dict_data=dict_data)
    >>> for source_data in input_data:
    ...     compressed = cctx.compress(source_data)
    ...     # Do something with compressed data.
    ...
    >>> dctx = zstandard.ZstdDecompressor(dict_data=dict_data)
    >>> for compressed_data in input_data:
    ...     buffer = io.BytesIO()
    ...     with dctx.stream_writer(buffer) as decompressor:
    ...         decompressor.write(compressed_data)
    ...         # Do something with raw data in ``buffer``.

    Dictionaries have unique integer IDs. You can retrieve this ID via:

    >>> dict_id = dict_data.dict_id()

    You can obtain the raw data in the dict (useful for persisting and constructing
    a ``ZstdCompressionDict`` later) via ``as_bytes()``:

    >>> dict_data = zstandard.train_dictionary(size, samples)
    >>> raw_data = dict_data.as_bytes()

    By default, when a ``ZstdCompressionDict`` is *attached* to a
    ``ZstdCompressor``, each ``ZstdCompressor`` performs work to prepare the
    dictionary for use. This is fine if only 1 compression operation is being
    performed or if the ``ZstdCompressor`` is being reused for multiple operations.
    But if multiple ``ZstdCompressor`` instances are being used with the dictionary,
    this can add overhead.

    It is possible to *precompute* the dictionary so it can readily be consumed
    by multiple ``ZstdCompressor`` instances:

    >>> d = zstandard.ZstdCompressionDict(data)
    >>> # Precompute for compression level 3.
    >>> d.precompute_compress(level=3)
    >>> # Precompute with specific compression parameters.
    >>> params = zstandard.ZstdCompressionParameters(...)
    >>> d.precompute_compress(compression_params=params)

    .. note::

       When a dictionary is precomputed, the compression parameters used to
       precompute the dictionary overwrite some of the compression parameters
       specified to ``ZstdCompressor``.

    :param data:
       Dictionary data.
    :param dict_type:
       Type of dictionary. One of the ``DICT_TYPE_*`` constants.
    """

    def __init__(self, data, dict_type=DICT_TYPE_AUTO, k=0, d=0):
        assert isinstance(data, bytes)
        self._data = data
        self.k = k
        self.d = d

        if dict_type not in (
            DICT_TYPE_AUTO,
            DICT_TYPE_RAWCONTENT,
            DICT_TYPE_FULLDICT,
        ):
            raise ValueError(
                "invalid dictionary load mode: %d; must use "
                "DICT_TYPE_* constants"
            )

        self._dict_type = dict_type
        self._cdict = None

    def __len__(self):
        return len(self._data)

    def dict_id(self):
        """Obtain the integer ID of the dictionary."""
        return int(lib.ZDICT_getDictID(self._data, len(self._data)))

    def as_bytes(self):
        """Obtain the ``bytes`` representation of the dictionary."""
        return self._data

    def precompute_compress(self, level=0, compression_params=None):
        """Precompute a dictionary os it can be used by multiple compressors.

        Calling this method on an instance that will be used by multiple
        :py:class:`ZstdCompressor` instances will improve performance.
        """
        if level and compression_params:
            raise ValueError(
                "must only specify one of level or compression_params"
            )

        if not level and not compression_params:
            raise ValueError("must specify one of level or compression_params")

        if level:
            cparams = lib.ZSTD_getCParams(level, 0, len(self._data))
        else:
            cparams = ffi.new("ZSTD_compressionParameters")
            cparams.chainLog = compression_params.chain_log
            cparams.hashLog = compression_params.hash_log
            cparams.minMatch = compression_params.min_match
            cparams.searchLog = compression_params.search_log
            cparams.strategy = compression_params.strategy
            cparams.targetLength = compression_params.target_length
            cparams.windowLog = compression_params.window_log

        cdict = lib.ZSTD_createCDict_advanced(
            self._data,
            len(self._data),
            lib.ZSTD_dlm_byRef,
            self._dict_type,
            cparams,
            lib.ZSTD_defaultCMem,
        )
        if cdict == ffi.NULL:
            raise ZstdError("unable to precompute dictionary")

        self._cdict = ffi.gc(
            cdict, lib.ZSTD_freeCDict, size=lib.ZSTD_sizeof_CDict(cdict)
        )

    @property
    def _ddict(self):
        ddict = lib.ZSTD_createDDict_advanced(
            self._data,
            len(self._data),
            lib.ZSTD_dlm_byRef,
            self._dict_type,
            lib.ZSTD_defaultCMem,
        )

        if ddict == ffi.NULL:
            raise ZstdError("could not create decompression dict")

        ddict = ffi.gc(
            ddict, lib.ZSTD_freeDDict, size=lib.ZSTD_sizeof_DDict(ddict)
        )
        self.__dict__["_ddict"] = ddict

        return ddict


def train_dictionary(
    dict_size,
    samples,
    k=0,
    d=0,
    f=0,
    split_point=0.0,
    accel=0,
    notifications=0,
    dict_id=0,
    level=0,
    steps=0,
    threads=0,
):
    """Train a dictionary from sample data using the COVER algorithm.

    A compression dictionary of size ``dict_size`` will be created from the
    iterable of ``samples``. The raw dictionary bytes will be returned.

    The dictionary training mechanism is known as *cover*. More details about it
    are available in the paper *Effective Construction of Relative Lempel-Ziv
    Dictionaries* (authors: Liao, Petri, Moffat, Wirth).

    The cover algorithm takes parameters ``k`` and ``d``. These are the
    *segment size* and *dmer size*, respectively. The returned dictionary
    instance created by this function has ``k`` and ``d`` attributes
    containing the values for these parameters. If a ``ZstdCompressionDict``
    is constructed from raw bytes data (a content-only dictionary), the
    ``k`` and ``d`` attributes will be ``0``.

    The segment and dmer size parameters to the cover algorithm can either be
    specified manually or ``train_dictionary()`` can try multiple values
    and pick the best one, where *best* means the smallest compressed data size.
    This later mode is called *optimization* mode.

    Under the hood, this function always calls
    ``ZDICT_optimizeTrainFromBuffer_fastCover()``. See the corresponding C library
    documentation for more.

    If neither ``steps`` nor ``threads`` is defined, defaults for ``d``, ``steps``,
    and ``level`` will be used that are equivalent with what
    ``ZDICT_trainFromBuffer()`` would use.


    :param dict_size:
       Target size in bytes of the dictionary to generate.
    :param samples:
       A list of bytes holding samples the dictionary will be trained from.
    :param k:
       Segment size : constraint: 0 < k : Reasonable range [16, 2048+]
    :param d:
       dmer size : constraint: 0 < d <= k : Reasonable range [6, 16]
    :param f:
       log of size of frequency array : constraint: 0 < f <= 31 : 1 means
       default(20)
    :param split_point:
       Percentage of samples used for training: Only used for optimization.
       The first # samples * ``split_point`` samples will be used to training.
       The last # samples * (1 - split_point) samples will be used for testing.
       0 means default (0.75), 1.0 when all samples are used for both training
       and testing.
    :param accel:
       Acceleration level: constraint: 0 < accel <= 10. Higher means faster
       and less accurate, 0 means default(1).
    :param dict_id:
       Integer dictionary ID for the produced dictionary. Default is 0, which uses
       a random value.
    :param steps:
       Number of steps through ``k`` values to perform when trying parameter
       variations.
    :param threads:
       Number of threads to use when trying parameter variations. Default is 0,
       which means to use a single thread. A negative value can be specified to
       use as many threads as there are detected logical CPUs.
    :param level:
       Integer target compression level when trying parameter variations.
    :param notifications:
       Controls writing of informational messages to ``stderr``. ``0`` (the
       default) means to write nothing. ``1`` writes errors. ``2`` writes
       progression info. ``3`` writes more details. And ``4`` writes all info.
    """

    if not isinstance(samples, list):
        raise TypeError("samples must be a list")

    if threads < 0:
        threads = _cpu_count()

    if not steps and not threads:
        d = d or 8
        steps = steps or 4
        level = level or 3

    total_size = sum(map(len, samples))

    samples_buffer = new_nonzero("char[]", total_size)
    sample_sizes = new_nonzero("size_t[]", len(samples))

    offset = 0
    for i, sample in enumerate(samples):
        if not isinstance(sample, bytes):
            raise ValueError("samples must be bytes")

        sample_len = len(sample)
        ffi.memmove(samples_buffer + offset, sample, sample_len)
        offset += sample_len
        sample_sizes[i] = sample_len

    dict_data = new_nonzero("char[]", dict_size)

    dparams = ffi.new("ZDICT_fastCover_params_t *")[0]
    dparams.k = k
    dparams.d = d
    dparams.f = f
    dparams.steps = steps
    dparams.nbThreads = threads
    dparams.splitPoint = split_point
    dparams.accel = accel
    dparams.zParams.notificationLevel = notifications
    dparams.zParams.dictID = dict_id
    dparams.zParams.compressionLevel = level

    zresult = lib.ZDICT_optimizeTrainFromBuffer_fastCover(
        ffi.addressof(dict_data),
        dict_size,
        ffi.addressof(samples_buffer),
        ffi.addressof(sample_sizes, 0),
        len(samples),
        ffi.addressof(dparams),
    )

    if lib.ZDICT_isError(zresult):
        msg = ffi.string(lib.ZDICT_getErrorName(zresult)).decode("utf-8")
        raise ZstdError("cannot train dict: %s" % msg)

    return ZstdCompressionDict(
        ffi.buffer(dict_data, zresult)[:],
        dict_type=DICT_TYPE_FULLDICT,
        k=dparams.k,
        d=dparams.d,
    )


class ZstdDecompressionObj(object):
    """A standard library API compatible decompressor.

    This type implements a compressor that conforms to the API by other
    decompressors in Python's standard library. e.g. ``zlib.decompressobj``
    or ``bz2.BZ2Decompressor``. This allows callers to use zstd compression
    while conforming to a similar API.

    Compressed data chunks are fed into ``decompress(data)`` and
    uncompressed output (or an empty bytes) is returned. Output from
    subsequent calls needs to be concatenated to reassemble the full
    decompressed byte sequence.

    If ``read_across_frames=False``, each instance is single use: once an
    input frame is decoded, ``decompress()`` will raise an exception. If
    ``read_across_frames=True``, instances can decode multiple frames.

    >>> dctx = zstandard.ZstdDecompressor()
    >>> dobj = dctx.decompressobj()
    >>> data = dobj.decompress(compressed_chunk_0)
    >>> data = dobj.decompress(compressed_chunk_1)

    By default, calls to ``decompress()`` write output data in chunks of size
    ``DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE``. These chunks are concatenated
    before being returned to the caller. It is possible to define the size of
    these temporary chunks by passing ``write_size`` to ``decompressobj()``:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> dobj = dctx.decompressobj(write_size=1048576)

    .. note::

       Because calls to ``decompress()`` may need to perform multiple
       memory (re)allocations, this streaming decompression API isn't as
       efficient as other APIs.
    """

    def __init__(self, decompressor, write_size, read_across_frames):
        self._decompressor = decompressor
        self._write_size = write_size
        self._finished = False
        self._read_across_frames = read_across_frames
        self._unused_input = b""

    def decompress(self, data):
        """Send compressed data to the decompressor and obtain decompressed data.

        :param data:
           Data to feed into the decompressor.
        :return:
           Decompressed bytes.
        """
        if self._finished:
            raise ZstdError("cannot use a decompressobj multiple times")

        in_buffer = ffi.new("ZSTD_inBuffer *")
        out_buffer = ffi.new("ZSTD_outBuffer *")

        data_buffer = ffi.from_buffer(data)

        if len(data_buffer) == 0:
            return b""

        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0

        dst_buffer = ffi.new("char[]", self._write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = len(dst_buffer)
        out_buffer.pos = 0

        chunks = []

        while True:
            zresult = lib.ZSTD_decompressStream(
                self._decompressor._dctx, out_buffer, in_buffer
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd decompressor error: %s" % _zstd_error(zresult)
                )

            # Always record any output from decompressor.
            if out_buffer.pos:
                chunks.append(ffi.buffer(out_buffer.dst, out_buffer.pos)[:])

            # 0 is only seen when a frame is fully decoded *and* fully flushed.
            # Behavior depends on whether we're in single or multiple frame
            # mode.
            if zresult == 0 and not self._read_across_frames:
                # Mark the instance as done and make any unconsumed input available
                # for retrieval.
                self._finished = True
                self._decompressor = None
                self._unused_input = data[in_buffer.pos : in_buffer.size]
                break
            elif zresult == 0 and self._read_across_frames:
                # We're at the end of a fully flushed frame and we can read more.
                # Try to read more if there's any more input.
                if in_buffer.pos == in_buffer.size:
                    break
                else:
                    out_buffer.pos = 0

            # We're not at the end of the frame *or* we're not fully flushed.

            # The decompressor will write out all the bytes it can to the output
            # buffer. So if the output buffer is partially filled and the input
            # is exhausted, there's nothing more to write. So we've done all we
            # can.
            elif (
                in_buffer.pos == in_buffer.size
                and out_buffer.pos < out_buffer.size
            ):
                break
            else:
                out_buffer.pos = 0

        return b"".join(chunks)

    def flush(self, length=0):
        """Effectively a no-op.

        Implemented for compatibility with the standard library APIs.

        Safe to call at any time.

        :return:
           Empty bytes.
        """
        return b""

    @property
    def unused_data(self):
        """Bytes past the end of compressed data.

        If ``decompress()`` is fed additional data beyond the end of a zstd
        frame, this value will be non-empty once ``decompress()`` fully decodes
        the input frame.
        """
        return self._unused_input

    @property
    def unconsumed_tail(self):
        """Data that has not yet been fed into the decompressor."""
        return b""

    @property
    def eof(self):
        """Whether the end of the compressed data stream has been reached."""
        return self._finished


class ZstdDecompressionReader(object):
    """Read only decompressor that pull uncompressed data from another stream.

    This type provides a read-only stream interface for performing transparent
    decompression from another stream or data source. It conforms to the
    ``io.RawIOBase`` interface. Only methods relevant to reading are
    implemented.

    >>> with open(path, 'rb') as fh:
    >>> dctx = zstandard.ZstdDecompressor()
    >>> reader = dctx.stream_reader(fh)
    >>> while True:
    ...     chunk = reader.read(16384)
    ...     if not chunk:
    ...         break
    ...     # Do something with decompressed chunk.

    The stream can also be used as a context manager:

    >>> with open(path, 'rb') as fh:
    ...     dctx = zstandard.ZstdDecompressor()
    ...     with dctx.stream_reader(fh) as reader:
    ...         ...

    When used as a context manager, the stream is closed and the underlying
    resources are released when the context manager exits. Future operations
    against the stream will fail.

    The ``source`` argument to ``stream_reader()`` can be any object with a
    ``read(size)`` method or any object implementing the *buffer protocol*.

    If the ``source`` is a stream, you can specify how large ``read()`` requests
    to that stream should be via the ``read_size`` argument. It defaults to
    ``zstandard.DECOMPRESSION_RECOMMENDED_INPUT_SIZE``.:

    >>> with open(path, 'rb') as fh:
    ...     dctx = zstandard.ZstdDecompressor()
    ...     # Will perform fh.read(8192) when obtaining data for the decompressor.
    ...     with dctx.stream_reader(fh, read_size=8192) as reader:
    ...         ...

    Instances are *partially* seekable. Absolute and relative positions
    (``SEEK_SET`` and ``SEEK_CUR``) forward of the current position are
    allowed. Offsets behind the current read position and offsets relative
    to the end of stream are not allowed and will raise ``ValueError``
    if attempted.

    ``tell()`` returns the number of decompressed bytes read so far.

    Not all I/O methods are implemented. Notably missing is support for
    ``readline()``, ``readlines()``, and linewise iteration support. This is
    because streams operate on binary data - not text data. If you want to
    convert decompressed output to text, you can chain an ``io.TextIOWrapper``
    to the stream:

    >>> with open(path, 'rb') as fh:
    ...     dctx = zstandard.ZstdDecompressor()
    ...     stream_reader = dctx.stream_reader(fh)
    ...     text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
    ...     for line in text_stream:
    ...         ...
    """

    def __init__(
        self,
        decompressor,
        source,
        read_size,
        read_across_frames,
        closefd=True,
    ):
        self._decompressor = decompressor
        self._source = source
        self._read_size = read_size
        self._read_across_frames = bool(read_across_frames)
        self._closefd = bool(closefd)
        self._entered = False
        self._closed = False
        self._bytes_decompressed = 0
        self._finished_input = False
        self._finished_output = False
        self._in_buffer = ffi.new("ZSTD_inBuffer *")
        # Holds a ref to self._in_buffer.src.
        self._source_buffer = None

    def __enter__(self):
        if self._entered:
            raise ValueError("cannot __enter__ multiple times")

        if self._closed:
            raise ValueError("stream is closed")

        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self._decompressor = None
        self.close()
        self._source = None

        return False

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def readline(self, size=-1):
        raise io.UnsupportedOperation()

    def readlines(self, hint=-1):
        raise io.UnsupportedOperation()

    def write(self, data):
        raise io.UnsupportedOperation()

    def writelines(self, lines):
        raise io.UnsupportedOperation()

    def isatty(self):
        return False

    def flush(self):
        return None

    def close(self):
        if self._closed:
            return None

        self._closed = True

        f = getattr(self._source, "close", None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def tell(self):
        return self._bytes_decompressed

    def readall(self):
        chunks = []

        while True:
            chunk = self.read(1048576)
            if not chunk:
                break

            chunks.append(chunk)

        return b"".join(chunks)

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()

    next = __next__

    def _read_input(self):
        # We have data left over in the input buffer. Use it.
        if self._in_buffer.pos < self._in_buffer.size:
            return

        # All input data exhausted. Nothing to do.
        if self._finished_input:
            return

        # Else populate the input buffer from our source.
        if hasattr(self._source, "read"):
            data = self._source.read(self._read_size)

            if not data:
                self._finished_input = True
                return

            self._source_buffer = ffi.from_buffer(data)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0
        else:
            self._source_buffer = ffi.from_buffer(self._source)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0

    def _decompress_into_buffer(self, out_buffer):
        """Decompress available input into an output buffer.

        Returns True if data in output buffer should be emitted.
        """
        zresult = lib.ZSTD_decompressStream(
            self._decompressor._dctx, out_buffer, self._in_buffer
        )

        if self._in_buffer.pos == self._in_buffer.size:
            self._in_buffer.src = ffi.NULL
            self._in_buffer.pos = 0
            self._in_buffer.size = 0
            self._source_buffer = None

            if not hasattr(self._source, "read"):
                self._finished_input = True

        if lib.ZSTD_isError(zresult):
            raise ZstdError("zstd decompress error: %s" % _zstd_error(zresult))

        # Emit data if there is data AND either:
        # a) output buffer is full (read amount is satisfied)
        # b) we're at end of a frame and not in frame spanning mode
        return out_buffer.pos and (
            out_buffer.pos == out_buffer.size
            or zresult == 0
            and not self._read_across_frames
        )

    def read(self, size=-1):
        if self._closed:
            raise ValueError("stream is closed")

        if size < -1:
            raise ValueError("cannot read negative amounts less than -1")

        if size == -1:
            # This is recursive. But it gets the job done.
            return self.readall()

        if self._finished_output or size == 0:
            return b""

        # We /could/ call into readinto() here. But that introduces more
        # overhead.
        dst_buffer = ffi.new("char[]", size)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0

        self._read_input()
        if self._decompress_into_buffer(out_buffer):
            self._bytes_decompressed += out_buffer.pos
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        while not self._finished_input:
            self._read_input()
            if self._decompress_into_buffer(out_buffer):
                self._bytes_decompressed += out_buffer.pos
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        self._bytes_decompressed += out_buffer.pos
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def readinto(self, b):
        if self._closed:
            raise ValueError("stream is closed")

        if self._finished_output:
            return 0

        # TODO use writable=True once we require CFFI >= 1.12.
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b"", 0)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0

        self._read_input()
        if self._decompress_into_buffer(out_buffer):
            self._bytes_decompressed += out_buffer.pos
            return out_buffer.pos

        while not self._finished_input:
            self._read_input()
            if self._decompress_into_buffer(out_buffer):
                self._bytes_decompressed += out_buffer.pos
                return out_buffer.pos

        self._bytes_decompressed += out_buffer.pos
        return out_buffer.pos

    def read1(self, size=-1):
        if self._closed:
            raise ValueError("stream is closed")

        if size < -1:
            raise ValueError("cannot read negative amounts less than -1")

        if self._finished_output or size == 0:
            return b""

        # -1 returns arbitrary number of bytes.
        if size == -1:
            size = DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE

        dst_buffer = ffi.new("char[]", size)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0

        # read1() dictates that we can perform at most 1 call to underlying
        # stream to get input. However, we can't satisfy this restriction with
        # decompression because not all input generates output. So we allow
        # multiple read(). But unlike read(), we stop once we have any output.
        while not self._finished_input:
            self._read_input()
            self._decompress_into_buffer(out_buffer)

            if out_buffer.pos:
                break

        self._bytes_decompressed += out_buffer.pos
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def readinto1(self, b):
        if self._closed:
            raise ValueError("stream is closed")

        if self._finished_output:
            return 0

        # TODO use writable=True once we require CFFI >= 1.12.
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b"", 0)

        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0

        while not self._finished_input and not self._finished_output:
            self._read_input()
            self._decompress_into_buffer(out_buffer)

            if out_buffer.pos:
                break

        self._bytes_decompressed += out_buffer.pos
        return out_buffer.pos

    def seek(self, pos, whence=os.SEEK_SET):
        if self._closed:
            raise ValueError("stream is closed")

        read_amount = 0

        if whence == os.SEEK_SET:
            if pos < 0:
                raise OSError("cannot seek to negative position with SEEK_SET")

            if pos < self._bytes_decompressed:
                raise OSError("cannot seek zstd decompression stream backwards")

            read_amount = pos - self._bytes_decompressed

        elif whence == os.SEEK_CUR:
            if pos < 0:
                raise OSError("cannot seek zstd decompression stream backwards")

            read_amount = pos
        elif whence == os.SEEK_END:
            raise OSError(
                "zstd decompression streams cannot be seeked with SEEK_END"
            )

        while read_amount:
            result = self.read(
                min(read_amount, DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE)
            )

            if not result:
                break

            read_amount -= len(result)

        return self._bytes_decompressed


class ZstdDecompressionWriter(object):
    """
    Write-only stream wrapper that performs decompression.

    This type provides a writable stream that performs decompression and writes
    decompressed data to another stream.

    This type implements the ``io.RawIOBase`` interface. Only methods that
    involve writing will do useful things.

    Behavior is similar to :py:meth:`ZstdCompressor.stream_writer`: compressed
    data is sent to the decompressor by calling ``write(data)`` and decompressed
    output is written to the inner stream by calling its ``write(data)``
    method:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> decompressor = dctx.stream_writer(fh)
    >>> # Will call fh.write() with uncompressed data.
    >>> decompressor.write(compressed_data)

    Instances can be used as context managers. However, context managers add no
    extra special behavior other than automatically calling ``close()`` when
    they exit.

    Calling ``close()`` will mark the stream as closed and subsequent I/O
    operations will raise ``ValueError`` (per the documented behavior of
    ``io.RawIOBase``). ``close()`` will also call ``close()`` on the
    underlying stream if such a method exists and the instance was created with
    ``closefd=True``.

    The size of chunks to ``write()`` to the destination can be specified:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> with dctx.stream_writer(fh, write_size=16384) as decompressor:
    >>>    pass

    You can see how much memory is being used by the decompressor:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> with dctx.stream_writer(fh) as decompressor:
    >>>    byte_size = decompressor.memory_size()

    ``stream_writer()`` accepts a ``write_return_read`` boolean argument to control
    the return value of ``write()``. When ``True`` (the default)``, ``write()``
    returns the number of bytes that were read from the input. When ``False``,
    ``write()`` returns the number of bytes that were ``write()`` to the inner
    stream.
    """

    def __init__(
        self,
        decompressor,
        writer,
        write_size,
        write_return_read,
        closefd=True,
    ):
        decompressor._ensure_dctx()

        self._decompressor = decompressor
        self._writer = writer
        self._write_size = write_size
        self._write_return_read = bool(write_return_read)
        self._closefd = bool(closefd)
        self._entered = False
        self._closing = False
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise ValueError("stream is closed")

        if self._entered:
            raise ZstdError("cannot __enter__ multiple times")

        self._entered = True

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self.close()

        return False

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()

    def memory_size(self):
        return lib.ZSTD_sizeof_DCtx(self._decompressor._dctx)

    def close(self):
        if self._closed:
            return

        try:
            self._closing = True
            self.flush()
        finally:
            self._closing = False
            self._closed = True

        f = getattr(self._writer, "close", None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def fileno(self):
        f = getattr(self._writer, "fileno", None)
        if f:
            return f()
        else:
            raise OSError("fileno not available on underlying writer")

    def flush(self):
        if self._closed:
            raise ValueError("stream is closed")

        f = getattr(self._writer, "flush", None)
        if f and not self._closing:
            return f()

    def isatty(self):
        return False

    def readable(self):
        return False

    def readline(self, size=-1):
        raise io.UnsupportedOperation()

    def readlines(self, hint=-1):
        raise io.UnsupportedOperation()

    def seek(self, offset, whence=None):
        raise io.UnsupportedOperation()

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation()

    def truncate(self, size=None):
        raise io.UnsupportedOperation()

    def writable(self):
        return True

    def writelines(self, lines):
        raise io.UnsupportedOperation()

    def read(self, size=-1):
        raise io.UnsupportedOperation()

    def readall(self):
        raise io.UnsupportedOperation()

    def readinto(self, b):
        raise io.UnsupportedOperation()

    def write(self, data):
        if self._closed:
            raise ValueError("stream is closed")

        total_write = 0

        in_buffer = ffi.new("ZSTD_inBuffer *")
        out_buffer = ffi.new("ZSTD_outBuffer *")

        data_buffer = ffi.from_buffer(data)
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0

        dst_buffer = ffi.new("char[]", self._write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = len(dst_buffer)
        out_buffer.pos = 0

        dctx = self._decompressor._dctx

        while in_buffer.pos < in_buffer.size:
            zresult = lib.ZSTD_decompressStream(dctx, out_buffer, in_buffer)
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd decompress error: %s" % _zstd_error(zresult)
                )

            if out_buffer.pos:
                self._writer.write(
                    ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                )
                total_write += out_buffer.pos
                out_buffer.pos = 0

        if self._write_return_read:
            return in_buffer.pos
        else:
            return total_write


class ZstdDecompressor(object):
    """
    Context for performing zstandard decompression.

    Each instance is essentially a wrapper around a ``ZSTD_DCtx`` from zstd's
    C API.

    An instance can compress data various ways. Instances can be used multiple
    times.

    The interface of this class is very similar to
    :py:class:`zstandard.ZstdCompressor` (by design).

    Assume that each ``ZstdDecompressor`` instance can only handle a single
    logical compression operation at the same time. i.e. if you call a method
    like ``decompressobj()`` to obtain multiple objects derived from the same
    ``ZstdDecompressor`` instance and attempt to use them simultaneously, errors
    will likely occur.

    If you need to perform multiple logical decompression operations and you
    can't guarantee those operations are temporally non-overlapping, you need
    to obtain multiple ``ZstdDecompressor`` instances.

    Unless specified otherwise, assume that no two methods of
    ``ZstdDecompressor`` instances can be called from multiple Python
    threads simultaneously. In other words, assume instances are not thread safe
    unless stated otherwise.

    :param dict_data:
       Compression dictionary to use.
    :param max_window_size:
       Sets an upper limit on the window size for decompression operations in
       kibibytes. This setting can be used to prevent large memory allocations
       for inputs using large compression windows.
    :param format:
       Set the format of data for the decoder.

       By default this is ``zstandard.FORMAT_ZSTD1``. It can be set to
       ``zstandard.FORMAT_ZSTD1_MAGICLESS`` to allow decoding frames without
       the 4 byte magic header. Not all decompression APIs support this mode.
    """

    def __init__(self, dict_data=None, max_window_size=0, format=FORMAT_ZSTD1):
        self._dict_data = dict_data
        self._max_window_size = max_window_size
        self._format = format

        dctx = lib.ZSTD_createDCtx()
        if dctx == ffi.NULL:
            raise MemoryError()

        self._dctx = dctx

        # Defer setting up garbage collection until full state is loaded so
        # the memory size is more accurate.
        try:
            self._ensure_dctx()
        finally:
            self._dctx = ffi.gc(
                dctx, lib.ZSTD_freeDCtx, size=lib.ZSTD_sizeof_DCtx(dctx)
            )

    def memory_size(self):
        """Size of decompression context, in bytes.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> size = dctx.memory_size()
        """
        return lib.ZSTD_sizeof_DCtx(self._dctx)

    def decompress(
        self,
        data,
        max_output_size=0,
        read_across_frames=False,
        allow_extra_data=True,
    ):
        """
        Decompress data in a single operation.

        This method will decompress the input data in a single operation and
        return the decompressed data.

        The input bytes are expected to contain at least 1 full Zstandard frame
        (something compressed with :py:meth:`ZstdCompressor.compress` or
        similar). If the input does not contain a full frame, an exception will
        be raised.

        ``read_across_frames`` controls whether to read multiple zstandard
        frames in the input. When False, decompression stops after reading the
        first frame. This feature is not yet implemented but the argument is
        provided for forward API compatibility when the default is changed to
        True in a future release. For now, if you need to decompress multiple
        frames, use an API like :py:meth:`ZstdCompressor.stream_reader` with
        ``read_across_frames=True``.

        ``allow_extra_data`` controls how to handle extra input data after a
        fully decoded frame. If False, any extra data (which could be a valid
        zstd frame) will result in ``ZstdError`` being raised. If True, extra
        data is silently ignored. The default will likely change to False in a
        future release when ``read_across_frames`` defaults to True.

        If the input contains extra data after a full frame, that extra input
        data is silently ignored. This behavior is undesirable in many scenarios
        and will likely be changed or controllable in a future release (see
        #181).

        If the frame header of the compressed data does not contain the content
        size, ``max_output_size`` must be specified or ``ZstdError`` will be
        raised. An allocation of size ``max_output_size`` will be performed and an
        attempt will be made to perform decompression into that buffer. If the
        buffer is too small or cannot be allocated, ``ZstdError`` will be
        raised. The buffer will be resized if it is too large.

        Uncompressed data could be much larger than compressed data. As a result,
        calling this function could result in a very large memory allocation
        being performed to hold the uncompressed data. This could potentially
        result in ``MemoryError`` or system memory swapping. If you don't need
        the full output data in a single contiguous array in memory, consider
        using streaming decompression for more resilient memory behavior.

        Usage:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> decompressed = dctx.decompress(data)

        If the compressed data doesn't have its content size embedded within it,
        decompression can be attempted by specifying the ``max_output_size``
        argument:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> uncompressed = dctx.decompress(data, max_output_size=1048576)

        Ideally, ``max_output_size`` will be identical to the decompressed
        output size.

        .. important::

           If the exact size of decompressed data is unknown (not passed in
           explicitly and not stored in the zstd frame), for performance
           reasons it is encouraged to use a streaming API.

        :param data:
           Compressed data to decompress.
        :param max_output_size:
           Integer max size of response.

           If ``0``, there is no limit and we can attempt to allocate an output
           buffer of infinite size.
        :return:
           ``bytes`` representing decompressed output.
        """

        if read_across_frames:
            raise ZstdError(
                "ZstdDecompressor.read_across_frames=True is not yet implemented"
            )

        self._ensure_dctx()

        data_buffer = ffi.from_buffer(data)

        params = ffi.new("ZSTD_FrameHeader *")
        zresult = lib.ZSTD_getFrameHeader_advanced(
            params, data_buffer, len(data_buffer), self._format
        )
        if zresult != 0:
            raise ZstdError("error determining content size from frame header")
        output_size = params.frameContentSize
        if output_size == 0:
            return b""
        elif output_size == lib.ZSTD_CONTENTSIZE_UNKNOWN:
            if not max_output_size:
                raise ZstdError(
                    "could not determine content size in frame header"
                )

            result_buffer = ffi.new("char[]", max_output_size)
            result_size = max_output_size
            output_size = 0
        else:
            result_buffer = ffi.new("char[]", output_size)
            result_size = output_size

        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = result_buffer
        out_buffer.size = result_size
        out_buffer.pos = 0

        in_buffer = ffi.new("ZSTD_inBuffer *")
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0

        zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
        if lib.ZSTD_isError(zresult):
            raise ZstdError("decompression error: %s" % _zstd_error(zresult))
        elif zresult:
            raise ZstdError(
                "decompression error: did not decompress full frame"
            )
        elif output_size and out_buffer.pos != output_size:
            raise ZstdError(
                "decompression error: decompressed %d bytes; expected %d"
                % (zresult, output_size)
            )
        elif not allow_extra_data and in_buffer.pos < in_buffer.size:
            count = in_buffer.size - in_buffer.pos

            raise ZstdError(
                "compressed input contains %d bytes of unused data, which is disallowed"
                % count
            )

        return ffi.buffer(result_buffer, out_buffer.pos)[:]

    def stream_reader(
        self,
        source,
        read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE,
        read_across_frames=False,
        closefd=True,
    ):
        """
        Read-only stream wrapper that performs decompression.

        This method obtains an object that conforms to the ``io.RawIOBase``
        interface and performs transparent decompression via ``read()``
        operations. Source data is obtained by calling ``read()`` on a
        source stream or object implementing the buffer protocol.

        See :py:class:`zstandard.ZstdDecompressionReader` for more documentation
        and usage examples.

        :param source:
           Source of compressed data to decompress. Can be any object
           with a ``read(size)`` method or that conforms to the buffer protocol.
        :param read_size:
           Integer number of bytes to read from the source and feed into the
           compressor at a time.
        :param read_across_frames:
           Whether to read data across multiple zstd frames. If False,
           decompression is stopped at frame boundaries.
        :param closefd:
           Whether to close the source stream when this instance is closed.
        :return:
           :py:class:`zstandard.ZstdDecompressionReader`.
        """
        self._ensure_dctx()
        return ZstdDecompressionReader(
            self, source, read_size, read_across_frames, closefd=closefd
        )

    def decompressobj(
        self,
        write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE,
        read_across_frames=False,
    ):
        """Obtain a standard library compatible incremental decompressor.

        See :py:class:`ZstdDecompressionObj` for more documentation
        and usage examples.

        :param write_size: size of internal output buffer to collect decompressed
          chunks in.
        :param read_across_frames: whether to read across multiple zstd frames.
          If False, reading stops after 1 frame and subsequent decompress
          attempts will raise an exception.
        :return:
           :py:class:`zstandard.ZstdDecompressionObj`
        """
        if write_size < 1:
            raise ValueError("write_size must be positive")

        self._ensure_dctx()
        return ZstdDecompressionObj(
            self, write_size=write_size, read_across_frames=read_across_frames
        )

    def read_to_iter(
        self,
        reader,
        read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE,
        write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE,
        skip_bytes=0,
    ):
        """Read compressed data to an iterator of uncompressed chunks.

        This method will read data from ``reader``, feed it to a decompressor,
        and emit ``bytes`` chunks representing the decompressed result.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh):
        ...     # Do something with original data.

        ``read_to_iter()`` accepts an object with a ``read(size)`` method that
        will return compressed bytes or an object conforming to the buffer
        protocol.

        ``read_to_iter()`` returns an iterator whose elements are chunks of the
        decompressed data.

        The size of requested ``read()`` from the source can be specified:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh, read_size=16384):
        ...    pass

        It is also possible to skip leading bytes in the input data:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh, skip_bytes=1):
        ...    pass

        .. tip::

           Skipping leading bytes is useful if the source data contains extra
           *header* data. Traditionally, you would need to create a slice or
           ``memoryview`` of the data you want to decompress. This would create
           overhead. It is more efficient to pass the offset into this API.

        Similarly to :py:meth:`ZstdCompressor.read_to_iter`, the consumer of the
        iterator controls when data is decompressed. If the iterator isn't consumed,
        decompression is put on hold.

        When ``read_to_iter()`` is passed an object conforming to the buffer protocol,
        the behavior may seem similar to what occurs when the simple decompression
        API is used. However, this API works when the decompressed size is unknown.
        Furthermore, if feeding large inputs, the decompressor will work in chunks
        instead of performing a single operation.

        :param reader:
           Source of compressed data. Can be any object with a
           ``read(size)`` method or any object conforming to the buffer
           protocol.
        :param read_size:
           Integer size of data chunks to read from ``reader`` and feed into
           the decompressor.
        :param write_size:
           Integer size of data chunks to emit from iterator.
        :param skip_bytes:
           Integer number of bytes to skip over before sending data into
           the decompressor.
        :return:
           Iterator of ``bytes`` representing uncompressed data.
        """

        if skip_bytes >= read_size:
            raise ValueError("skip_bytes must be smaller than read_size")

        if hasattr(reader, "read"):
            have_read = True
        elif hasattr(reader, "__getitem__"):
            have_read = False
            buffer_offset = 0
            size = len(reader)
        else:
            raise ValueError(
                "must pass an object with a read() method or "
                "conforms to buffer protocol"
            )

        if skip_bytes:
            if have_read:
                reader.read(skip_bytes)
            else:
                if skip_bytes > size:
                    raise ValueError("skip_bytes larger than first input chunk")

                buffer_offset = skip_bytes

        self._ensure_dctx()

        in_buffer = ffi.new("ZSTD_inBuffer *")
        out_buffer = ffi.new("ZSTD_outBuffer *")

        dst_buffer = ffi.new("char[]", write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = len(dst_buffer)
        out_buffer.pos = 0

        while True:
            assert out_buffer.pos == 0

            if have_read:
                read_result = reader.read(read_size)
            else:
                remaining = size - buffer_offset
                slice_size = min(remaining, read_size)
                read_result = reader[buffer_offset : buffer_offset + slice_size]
                buffer_offset += slice_size

            # No new input. Break out of read loop.
            if not read_result:
                break

            # Feed all read data into decompressor and emit output until
            # exhausted.
            read_buffer = ffi.from_buffer(read_result)
            in_buffer.src = read_buffer
            in_buffer.size = len(read_buffer)
            in_buffer.pos = 0

            while in_buffer.pos < in_buffer.size:
                assert out_buffer.pos == 0

                zresult = lib.ZSTD_decompressStream(
                    self._dctx, out_buffer, in_buffer
                )
                if lib.ZSTD_isError(zresult):
                    raise ZstdError(
                        "zstd decompress error: %s" % _zstd_error(zresult)
                    )

                if out_buffer.pos:
                    data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                    out_buffer.pos = 0
                    yield data

                if zresult == 0:
                    return

            # Repeat loop to collect more input data.
            continue

        # If we get here, input is exhausted.

    def stream_writer(
        self,
        writer,
        write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE,
        write_return_read=True,
        closefd=True,
    ):
        """
        Push-based stream wrapper that performs decompression.

        This method constructs a stream wrapper that conforms to the
        ``io.RawIOBase`` interface and performs transparent decompression
        when writing to a wrapper stream.

        See :py:class:`zstandard.ZstdDecompressionWriter` for more documentation
        and usage examples.

        :param writer:
           Destination for decompressed output. Can be any object with a
           ``write(data)``.
        :param write_size:
           Integer size of chunks to ``write()`` to ``writer``.
        :param write_return_read:
           Whether ``write()`` should return the number of bytes of input
           consumed. If False, ``write()`` returns the number of bytes sent
           to the inner stream.
        :param closefd:
           Whether to ``close()`` the inner stream when this stream is closed.
        :return:
           :py:class:`zstandard.ZstdDecompressionWriter`
        """
        if not hasattr(writer, "write"):
            raise ValueError("must pass an object with a write() method")

        return ZstdDecompressionWriter(
            self,
            writer,
            write_size,
            write_return_read,
            closefd=closefd,
        )

    def copy_stream(
        self,
        ifh,
        ofh,
        read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE,
        write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE,
    ):
        """
        Copy data between streams, decompressing in the process.

        Compressed data will be read from ``ifh``, decompressed, and written
        to ``ofh``.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> dctx.copy_stream(ifh, ofh)

        e.g. to decompress a file to another file:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> with open(input_path, 'rb') as ifh, open(output_path, 'wb') as ofh:
        ...     dctx.copy_stream(ifh, ofh)

        The size of chunks being ``read()`` and ``write()`` from and to the
        streams can be specified:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> dctx.copy_stream(ifh, ofh, read_size=8192, write_size=16384)

        :param ifh:
           Source stream to read compressed data from.

           Must have a ``read()`` method.
        :param ofh:
           Destination stream to write uncompressed data to.

           Must have a ``write()`` method.
        :param read_size:
           The number of bytes to ``read()`` from the source in a single
           operation.
        :param write_size:
           The number of bytes to ``write()`` to the destination in a single
           operation.
        :return:
           2-tuple of integers representing the number of bytes read and
           written, respectively.
        """

        if not hasattr(ifh, "read"):
            raise ValueError("first argument must have a read() method")
        if not hasattr(ofh, "write"):
            raise ValueError("second argument must have a write() method")

        self._ensure_dctx()

        in_buffer = ffi.new("ZSTD_inBuffer *")
        out_buffer = ffi.new("ZSTD_outBuffer *")

        dst_buffer = ffi.new("char[]", write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = write_size
        out_buffer.pos = 0

        total_read, total_write = 0, 0

        # Read all available input.
        while True:
            data = ifh.read(read_size)
            if not data:
                break

            data_buffer = ffi.from_buffer(data)
            total_read += len(data_buffer)
            in_buffer.src = data_buffer
            in_buffer.size = len(data_buffer)
            in_buffer.pos = 0

            # Flush all read data to output.
            while in_buffer.pos < in_buffer.size:
                zresult = lib.ZSTD_decompressStream(
                    self._dctx, out_buffer, in_buffer
                )
                if lib.ZSTD_isError(zresult):
                    raise ZstdError(
                        "zstd decompressor error: %s" % _zstd_error(zresult)
                    )

                if out_buffer.pos:
                    ofh.write(ffi.buffer(out_buffer.dst, out_buffer.pos))
                    total_write += out_buffer.pos
                    out_buffer.pos = 0

            # Continue loop to keep reading.

        return total_read, total_write

    def decompress_content_dict_chain(self, frames):
        """
        Decompress a series of frames using the content dictionary chaining technique.

        Such a list of frames is produced by compressing discrete inputs where
        each non-initial input is compressed with a *prefix* dictionary consisting
        of the content of the previous input.

        For example, say you have the following inputs:

        >>> inputs = [b"input 1", b"input 2", b"input 3"]

        The zstd frame chain consists of:

        1. ``b"input 1"`` compressed in standalone/discrete mode
        2. ``b"input 2"`` compressed using ``b"input 1"`` as a *prefix* dictionary
        3. ``b"input 3"`` compressed using ``b"input 2"`` as a *prefix* dictionary

        Each zstd frame **must** have the content size written.

        The following Python code can be used to produce a *prefix dictionary chain*:

        >>> def make_chain(inputs):
        ...    frames = []
        ...
        ...    # First frame is compressed in standalone/discrete mode.
        ...    zctx = zstandard.ZstdCompressor()
        ...    frames.append(zctx.compress(inputs[0]))
        ...
        ...    # Subsequent frames use the previous fulltext as a prefix dictionary
        ...    for i, raw in enumerate(inputs[1:]):
        ...        dict_data = zstandard.ZstdCompressionDict(
        ...            inputs[i], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        ...        zctx = zstandard.ZstdCompressor(dict_data=dict_data)
        ...        frames.append(zctx.compress(raw))
        ...
        ...    return frames

        ``decompress_content_dict_chain()`` returns the uncompressed data of the last
        element in the input chain.

        .. note::

           It is possible to implement *prefix dictionary chain* decompression
           on top of other APIs. However, this function will likely be faster -
           especially for long input chains - as it avoids the overhead of
           instantiating and passing around intermediate objects between
           multiple functions.

        :param frames:
           List of ``bytes`` holding compressed zstd frames.
        :return:
        """
        if not isinstance(frames, list):
            raise TypeError("argument must be a list")

        if not frames:
            raise ValueError("empty input chain")

        # First chunk should not be using a dictionary. We handle it specially.
        chunk = frames[0]
        if not isinstance(chunk, bytes):
            raise ValueError("chunk 0 must be bytes")

        # All chunks should be zstd frames and should have content size set.
        chunk_buffer = ffi.from_buffer(chunk)
        params = ffi.new("ZSTD_FrameHeader *")
        zresult = lib.ZSTD_getFrameHeader(
            params, chunk_buffer, len(chunk_buffer)
        )
        if lib.ZSTD_isError(zresult):
            raise ValueError("chunk 0 is not a valid zstd frame")
        elif zresult:
            raise ValueError("chunk 0 is too small to contain a zstd frame")

        if params.frameContentSize == lib.ZSTD_CONTENTSIZE_UNKNOWN:
            raise ValueError("chunk 0 missing content size in frame")

        self._ensure_dctx(load_dict=False)

        last_buffer = ffi.new("char[]", params.frameContentSize)

        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = last_buffer
        out_buffer.size = len(last_buffer)
        out_buffer.pos = 0

        in_buffer = ffi.new("ZSTD_inBuffer *")
        in_buffer.src = chunk_buffer
        in_buffer.size = len(chunk_buffer)
        in_buffer.pos = 0

        zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "could not decompress chunk 0: %s" % _zstd_error(zresult)
            )
        elif zresult:
            raise ZstdError("chunk 0 did not decompress full frame")

        # Special case of chain length of 1
        if len(frames) == 1:
            return ffi.buffer(last_buffer, len(last_buffer))[:]

        i = 1
        while i < len(frames):
            chunk = frames[i]
            if not isinstance(chunk, bytes):
                raise ValueError("chunk %d must be bytes" % i)

            chunk_buffer = ffi.from_buffer(chunk)
            zresult = lib.ZSTD_getFrameHeader(
                params, chunk_buffer, len(chunk_buffer)
            )
            if lib.ZSTD_isError(zresult):
                raise ValueError("chunk %d is not a valid zstd frame" % i)
            elif zresult:
                raise ValueError(
                    "chunk %d is too small to contain a zstd frame" % i
                )

            if params.frameContentSize == lib.ZSTD_CONTENTSIZE_UNKNOWN:
                raise ValueError("chunk %d missing content size in frame" % i)

            dest_buffer = ffi.new("char[]", params.frameContentSize)

            out_buffer.dst = dest_buffer
            out_buffer.size = len(dest_buffer)
            out_buffer.pos = 0

            in_buffer.src = chunk_buffer
            in_buffer.size = len(chunk_buffer)
            in_buffer.pos = 0

            zresult = lib.ZSTD_decompressStream(
                self._dctx, out_buffer, in_buffer
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "could not decompress chunk %d: %s" % _zstd_error(zresult)
                )
            elif zresult:
                raise ZstdError("chunk %d did not decompress full frame" % i)

            last_buffer = dest_buffer
            i += 1

        return ffi.buffer(last_buffer, len(last_buffer))[:]

    def multi_decompress_to_buffer(
        self, frames, decompressed_sizes=None, threads=0
    ):
        """
        Decompress multiple zstd frames to output buffers as a single operation.

        (Experimental. Not available in CFFI backend.)

        Compressed frames can be passed to the function as a
        ``BufferWithSegments``, a ``BufferWithSegmentsCollection``, or as a
        list containing objects that conform to the buffer protocol. For best
        performance, pass a ``BufferWithSegmentsCollection`` or a
        ``BufferWithSegments``, as minimal input validation will be done for
        that type. If calling from Python (as opposed to C), constructing one
        of these instances may add overhead cancelling out the performance
        overhead of validation for list inputs.

        Returns a ``BufferWithSegmentsCollection`` containing the decompressed
        data. All decompressed data is allocated in a single memory buffer. The
        ``BufferWithSegments`` instance tracks which objects are at which offsets
        and their respective lengths.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> results = dctx.multi_decompress_to_buffer([b'...', b'...'])

        The decompressed size of each frame MUST be discoverable. It can either be
        embedded within the zstd frame or passed in via the ``decompressed_sizes``
        argument.

        The ``decompressed_sizes`` argument is an object conforming to the buffer
        protocol which holds an array of 64-bit unsigned integers in the machine's
        native format defining the decompressed sizes of each frame. If this argument
        is passed, it avoids having to scan each frame for its decompressed size.
        This frame scanning can add noticeable overhead in some scenarios.

        >>> frames = [...]
        >>> sizes = struct.pack('=QQQQ', len0, len1, len2, len3)
        >>>
        >>> dctx = zstandard.ZstdDecompressor()
        >>> results = dctx.multi_decompress_to_buffer(frames, decompressed_sizes=sizes)

        .. note::

           It is possible to pass a ``mmap.mmap()`` instance into this function by
           wrapping it with a ``BufferWithSegments`` instance (which will define the
           offsets of frames within the memory mapped region).

        This function is logically equivalent to performing
        :py:meth:`ZstdCompressor.decompress` on each input frame and returning the
        result.

        This function exists to perform decompression on multiple frames as fast
        as possible by having as little overhead as possible. Since decompression is
        performed as a single operation and since the decompressed output is stored in
        a single buffer, extra memory allocations, Python objects, and Python function
        calls are avoided. This is ideal for scenarios where callers know up front that
        they need to access data for multiple frames, such as when  *delta chains* are
        being used.

        Currently, the implementation always spawns multiple threads when requested,
        even if the amount of work to do is small. In the future, it will be smarter
        about avoiding threads and their associated overhead when the amount of
        work to do is small.

        :param frames:
           Source defining zstd frames to decompress.
        :param decompressed_sizes:
           Array of integers representing sizes of decompressed zstd frames.
        :param threads:
           How many threads to use for decompression operations.

           Negative values will use the same number of threads as logical CPUs
           on the machine. Values ``0`` or ``1`` use a single thread.
        :return:
           ``BufferWithSegmentsCollection``
        """
        raise NotImplementedError()

    def _ensure_dctx(self, load_dict=True):
        lib.ZSTD_DCtx_reset(self._dctx, lib.ZSTD_reset_session_only)

        if self._max_window_size:
            zresult = lib.ZSTD_DCtx_setMaxWindowSize(
                self._dctx, self._max_window_size
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "unable to set max window size: %s" % _zstd_error(zresult)
                )

        zresult = lib.ZSTD_DCtx_setParameter(
            self._dctx, lib.ZSTD_d_format, self._format
        )
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "unable to set decoding format: %s" % _zstd_error(zresult)
            )

        if self._dict_data and load_dict:
            zresult = lib.ZSTD_DCtx_refDDict(self._dctx, self._dict_data._ddict)
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "unable to reference prepared dictionary: %s"
                    % _zstd_error(zresult)
                )
<|endoftext|>"""
Type definitions and utilities for the `create_commit` API
"""

import base64
import io
import json
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, NamedTuple, Union

from . import constants
from .errors import EntryNotFoundError
from .file_download import hf_hub_url
from .lfs import UploadInfo, lfs_upload, post_lfs_batch_info
from .utils import (
    FORBIDDEN_FOLDERS,
    chunk_iterable,
    get_session,
    hf_raise_for_status,
    http_backoff,
    logging,
    sha,
    validate_hf_hub_args,
)
from .utils._runtime import is_xet_available


if TYPE_CHECKING:
    from .hf_api import CommitInfo, RepoFile


logger = logging.get_logger(__name__)


UploadMode = Literal["lfs", "regular"]

# Max is 1,000 per request on the Hub for HfApi.get_paths_info
# Otherwise we get:
# HfHubHTTPError: 413 Client Error: Payload Too Large for url: https://huggingface.co/api/datasets/xxx (Request ID: xxx)\n\ntoo many parameters
# See https://github.com/huggingface/huggingface_hub/issues/1503
FETCH_LFS_BATCH_SIZE = 500
DUPLICATE_LFS_BATCH_SIZE = 500

UPLOAD_BATCH_MAX_NUM_FILES = 256


@dataclass
class CommitOperationDelete:
    """
    Data structure holding necessary info to delete a file or a folder from a repository
    on the Hub.

    Args:
        path_in_repo (`str`):
            Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
            for a file or `"checkpoints/1fec34a/"` for a folder.
        is_folder (`bool` or `Literal["auto"]`, *optional*):
            Whether the Delete Operation applies to a folder or not. If "auto", the path
            type (file or folder) is guessed automatically by looking if path ends with
            a "/" (folder) or not (file). To explicitly set the path type, you can set
            `is_folder=True` or `is_folder=False`.
    """

    path_in_repo: str
    is_folder: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        self.path_in_repo = _validate_path_in_repo(self.path_in_repo)

        if self.is_folder == "auto":
            self.is_folder = self.path_in_repo.endswith("/")
        if not isinstance(self.is_folder, bool):
            raise ValueError(
                f"Wrong value for `is_folder`. Must be one of [`True`, `False`, `'auto'`]. Got '{self.is_folder}'."
            )


@dataclass
class CommitOperationCopy:
    """
    Data structure holding necessary info to copy a file in a repository on the Hub.

    Both LFS files and regular files are supported. LFS files are copied server-side while regular files are
    downloaded and re-uploaded as part of the commit.

    Cross-repository copies are supported by setting `src_repo_id` and `src_repo_type`. For cross-repo LFS copies,
    the LFS objects are duplicated to the destination repository before the commit is created. This is handled
    automatically by [`create_commit`]. Note that cross-repository copies only work within the same
    [storage region](https://huggingface.co/docs/hub/storage-regions); copying across regions is not supported.

    Note: you can combine a [`CommitOperationCopy`] and a [`CommitOperationDelete`] to rename an LFS file on the Hub.

    Args:
        src_path_in_repo (`str`):
            Relative filepath in the repo of the file to be copied, e.g. `"checkpoints/1fec34a/weights.bin"`.
        path_in_repo (`str`):
            Relative filepath in the repo where to copy the file, e.g. `"checkpoints/1fec34a/weights_copy.bin"`.
        src_revision (`str`, *optional*):
            The git revision of the file to be copied. Can be any valid git revision.
            Default to the target commit revision.
        src_repo_id (`str`, *optional*):
            The source repository to copy from (e.g. `"username/source-model"`).
            Default to the destination repository (intra-repo copy).
        src_repo_type (`str`, *optional*):
            The type of the source repository (`"model"`, `"dataset"` or `"space"`).
            Required when `src_repo_id` is set.
    """

    src_path_in_repo: str
    path_in_repo: str
    src_revision: str | None = None
    src_repo_id: str | None = None
    src_repo_type: str | None = None
    # set to the OID of the file to be copied if it has already been uploaded
    # useful to determine if a commit will be empty or not.
    _src_oid: str | None = None
    # set to the OID of the file to copy to if it has already been uploaded
    # useful to determine if a commit will be empty or not.
    _dest_oid: str | None = None
    # set to True once cross-repo LFS files have been duplicated to the destination repo
    _is_duplicated: bool = False

    def __post_init__(self):
        self.src_path_in_repo = _validate_path_in_repo(self.src_path_in_repo)
        self.path_in_repo = _validate_path_in_repo(self.path_in_repo)
        if self.src_repo_id is not None and self.src_repo_type is None:
            raise ValueError("`src_repo_type` is required when `src_repo_id` is set.")
        if self.src_repo_type is not None and self.src_repo_id is None:
            raise ValueError("`src_repo_id` is required when `src_repo_type` is set.")


@dataclass
class CommitOperationAdd:
    """
    Data structure holding necessary info to upload a file to a repository on the Hub.

    Args:
        path_in_repo (`str`):
            Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
        path_or_fileobj (`str`, `Path`, `bytes`, or `BinaryIO`):
            Either:
            - a path to a local file (as `str` or `pathlib.Path`) to upload
            - a buffer of bytes (`bytes`) holding the content of the file to upload
            - a "file object" (subclass of `io.BufferedIOBase`), typically obtained
                with `open(path, "rb")`. It must support `seek()` and `tell()` methods.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `path_or_fileobj` is not one of `str`, `Path`, `bytes` or `io.BufferedIOBase`.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `path_or_fileobj` is a `str` or `Path` but not a path to an existing file.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `path_or_fileobj` is a `io.BufferedIOBase` but it doesn't support both
            `seek()` and `tell()`.
    """

    path_in_repo: str
    path_or_fileobj: str | Path | bytes | BinaryIO
    upload_info: UploadInfo = field(init=False, repr=False)

    # Internal attributes

    # set to "lfs" or "regular" once known
    _upload_mode: UploadMode | None = field(init=False, repr=False, default=None)

    # set to True if .gitignore rules prevent the file from being uploaded as LFS
    # (server-side check)
    _should_ignore: bool | None = field(init=False, repr=False, default=None)

    # set to the remote OID of the file if it has already been uploaded
    # useful to determine if a commit will be empty or not
    _remote_oid: str | None = field(init=False, repr=False, default=None)

    # set to True once the file has been uploaded as LFS
    _is_uploaded: bool = field(init=False, repr=False, default=False)

    # set to True once the file has been committed
    _is_committed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Validates `path_or_fileobj` and compute `upload_info`."""
        self.path_in_repo = _validate_path_in_repo(self.path_in_repo)

        # Validate `path_or_fileobj` value
        if isinstance(self.path_or_fileobj, Path):
            self.path_or_fileobj = str(self.path_or_fileobj)
        if isinstance(self.path_or_fileobj, str):
            self.path_or_fileobj = os.path.normpath(os.path.expanduser(self.path_or_fileobj))
            if not os.path.isfile(self.path_or_fileobj):
                raise ValueError(f"Provided path: '{self.path_or_fileobj}' is not a file on the local file system")
        elif not isinstance(self.path_or_fileobj, (io.BufferedIOBase, bytes)):
            # ^^ Inspired from: https://stackoverflow.com/questions/44584829/how-to-determine-if-file-is-opened-in-binary-or-text-mode
            raise ValueError(
                "path_or_fileobj must be either an instance of str, bytes or"
                " io.BufferedIOBase. If you passed a file-like object, make sure it is"
                " in binary mode."
            )
        if isinstance(self.path_or_fileobj, io.BufferedIOBase):
            try:
                self.path_or_fileobj.tell()
                self.path_or_fileobj.seek(0, os.SEEK_CUR)
            except (OSError, AttributeError) as exc:
                raise ValueError(
                    "path_or_fileobj is a file-like object but does not implement seek() and tell()"
                ) from exc

        # Compute "upload_info" attribute
        if isinstance(self.path_or_fileobj, str):
            self.upload_info = UploadInfo.from_path(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, bytes):
            self.upload_info = UploadInfo.from_bytes(self.path_or_fileobj)
        else:
            self.upload_info = UploadInfo.from_fileobj(self.path_or_fileobj)

    @contextmanager
    def as_file(self, with_tqdm: bool = False) -> Iterator[BinaryIO]:
        """
        A context manager that yields a file-like object allowing to read the underlying
        data behind `path_or_fileobj`.

        Args:
            with_tqdm (`bool`, *optional*, defaults to `False`):
                If True, iterating over the file object will display a progress bar. Only
                works if the file-like object is a path to a file. Pure bytes and buffers
                are not supported.

        Example:

        ```python
        >>> operation = CommitOperationAdd(
        ...        path_in_repo="remote/dir/weights.h5",
        ...        path_or_fileobj="./local/weights.h5",
        ... )
        CommitOperationAdd(path_in_repo='remote/dir/weights.h5', path_or_fileobj='./local/weights.h5')

        >>> with operation.as_file() as file:
        ...     content = file.read()

        >>> with operation.as_file(with_tqdm=True) as file:
        ...     while True:
        ...         data = file.read(1024)
        ...         if not data:
        ...              break
        config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]

        >>> with operation.as_file(with_tqdm=True) as file:
        ...     httpx2.put(..., data=file)
        config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]
        ```
        """
        if isinstance(self.path_or_fileobj, str) or isinstance(self.path_or_fileobj, Path):
            if with_tqdm:
                from .utils import tqdm_stream_file

                with tqdm_stream_file(self.path_or_fileobj) as file:
                    yield file
            else:
                with open(self.path_or_fileobj, "rb") as file:
                    yield file
        elif isinstance(self.path_or_fileobj, bytes):
            yield io.BytesIO(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, io.BufferedIOBase):
            prev_pos = self.path_or_fileobj.tell()
            yield self.path_or_fileobj
            self.path_or_fileobj.seek(prev_pos, io.SEEK_SET)

    def b64content(self) -> bytes:
        """
        The base64-encoded content of `path_or_fileobj`

        Returns: `bytes`
        """
        with self.as_file() as file:
            return base64.b64encode(file.read())

    @property
    def _local_oid(self) -> str | None:
        """Return the OID of the local file.

        This OID is then compared to `self._remote_oid` to check if the file has changed compared to the remote one.
        If the file did not change, we won't upload it again to prevent empty commits.

        For LFS files, the OID corresponds to the SHA256 of the file content (used a LFS ref).
        For regular files, the OID corresponds to the SHA1 of the file content.
        Note: this is slightly different to git OID computation since the oid of an LFS file is usually the git-SHA1 of the
              pointer file content (not the actual file content). However, using the SHA256 is enough to detect changes
              and more convenient client-side.
        """
        if self._upload_mode is None:
            return None
        elif self._upload_mode == "lfs":
            return self.upload_info.sha256.hex()
        else:
            # Regular file => compute sha1
            # => no need to read by chunk since the file is guaranteed to be <=5MB.
            with self.as_file() as file:
                return sha.git_hash(file.read())


def _validate_path_in_repo(path_in_repo: str) -> str:
    # Validate `path_in_repo` value to prevent a server-side issue
    if path_in_repo.startswith("/"):
        path_in_repo = path_in_repo[1:]
    # Reject any ".." segment, wherever it sits in the path: a segment in the middle
    # (e.g. "a/../../etc/passwd" or "a/../file.txt") is not caught by a prefix check
    # and can still escape the repo root once resolved.
    if path_in_repo == "." or any(part == ".." for part in path_in_repo.split("/")):
        raise ValueError(f"Invalid `path_in_repo` in CommitOperation: '{path_in_repo}'")
    if path_in_repo.startswith("./"):
        path_in_repo = path_in_repo[2:]
    for forbidden in FORBIDDEN_FOLDERS:
        if any(part == forbidden for part in path_in_repo.split("/")):
            raise ValueError(
                f"Invalid `path_in_repo` in CommitOperation: cannot update files under a '{forbidden}/' folder (path:"
                f" '{path_in_repo}')."
            )
    return path_in_repo


CommitOperation = Union[CommitOperationAdd, CommitOperationCopy, CommitOperationDelete]


def _warn_on_overwriting_operations(operations: list[CommitOperation]) -> None:
    """
    Warn user when a list of operations is expected to overwrite itself in a single
    commit.

    Rules:
    - If a filepath is updated by multiple `CommitOperationAdd` operations, a warning
      message is triggered.
    - If a filepath is updated at least once by a `CommitOperationAdd` and then deleted
      by a `CommitOperationDelete`, a warning is triggered.
    - If a `CommitOperationDelete` deletes a filepath that is then updated by a
      `CommitOperationAdd`, no warning is triggered. This is usually useless (no need to
      delete before upload) but can happen if a user deletes an entire folder and then
      add new files to it.
    """
    nb_additions_per_path: dict[str, int] = defaultdict(int)
    for operation in operations:
        path_in_repo = operation.path_in_repo
        if isinstance(operation, CommitOperationAdd):
            if nb_additions_per_path[path_in_repo] > 0:
                warnings.warn(
                    "About to update multiple times the same file in the same commit:"
                    f" '{path_in_repo}'. This can cause undesired inconsistencies in"
                    " your repo."
                )
            nb_additions_per_path[path_in_repo] += 1
            for parent in PurePosixPath(path_in_repo).parents:
                # Also keep track of number of updated files per folder
                # => warns if deleting a folder overwrite some contained files
                nb_additions_per_path[str(parent)] += 1
        if isinstance(operation, CommitOperationDelete):
            if nb_additions_per_path[str(PurePosixPath(path_in_repo))] > 0:
                if operation.is_folder:
                    warnings.warn(
                        "About to delete a folder containing files that have just been"
                        f" updated within the same commit: '{path_in_repo}'. This can"
                        " cause undesired inconsistencies in your repo."
                    )
                else:
                    warnings.warn(
                        "About to delete a file that have just been updated within the"
                        f" same commit: '{path_in_repo}'. This can cause undesired"
                        " inconsistencies in your repo."
                    )


@validate_hf_hub_args
def _upload_files(
    *,
    additions: list[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    headers: dict[str, str],
    endpoint: str | None = None,
    num_threads: int = 5,
    revision: str | None = None,
    create_pr: bool | None = None,
):
    """
    Uploads the files through the Xet protocol if possible, otherwise through the legacy LFS protocol.

    The Xet path does not require any Python-side sha256 computation: hashing happens inside `hf_xet`
    while chunking the files (single read pass) and is backfilled on the operations afterwards.
    """
    has_buffered_io_data = any(isinstance(op.path_or_fileobj, io.BufferedIOBase) for op in additions)
    if is_xet_available():
        if not has_buffered_io_data:
            _upload_xet_files(
                additions=additions,
                repo_type=repo_type,
                repo_id=repo_id,
                headers=headers,
                endpoint=endpoint,
                revision=revision,
                create_pr=create_pr,
            )
            return
        logger.warning(
            "Uploading files as a binary IO buffer is not supported by Xet Storage. Falling back to HTTP upload."
        )

    # Legacy LFS path: sha256 is required by the LFS batch endpoint => compute missing ones (in parallel).
    _compute_missing_sha256s(additions, num_threads=num_threads)

    lfs_actions: list[dict[str, Any]] = []
    lfs_oid2addop: dict[str, CommitOperationAdd] = {}
    for chunk in chunk_iterable(additions, chunk_size=UPLOAD_BATCH_MAX_NUM_FILES):
        chunk_list = [op for op in chunk]
        actions_chunk, errors_chunk, _ = post_lfs_batch_info(
            upload_infos=[op.upload_info for op in chunk_list],
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
            headers=headers,
            token=None,  # already passed in 'headers'
            transfers=["basic", "multipart"],
        )
        if errors_chunk:
            message = "\n".join(
                [
                    f"Encountered error for file with OID {err.get('oid')}: `{err.get('error', {}).get('message')}"
                    for err in errors_chunk
                ]
            )
            raise ValueError(f"LFS batch API returned errors:\n{message}")
        lfs_actions.extend(actions_chunk)
        for op in chunk_list:
            lfs_oid2addop[op.upload_info.sha256.hex()] = op

    if len(lfs_actions) > 0:
        _upload_lfs_files(
            actions=lfs_actions,
            oid2addop=lfs_oid2addop,
            headers=headers,
            endpoint=endpoint,
            num_threads=num_threads,
        )


def _compute_missing_sha256s(additions: list[CommitOperationAdd], num_threads: int) -> None:
    """Compute the sha256 of the operations that don't have one yet, in parallel."""
    not_hashed = [op for op in additions if not op.upload_info.is_hashed]
    if len(not_hashed) == 0:
        return
    logger.info(f"Computing sha256 for {len(not_hashed)} files.")
    if len(not_hashed) == 1:
        _ = not_hashed[0].upload_info.sha256
        return
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(lambda op: op.upload_info.sha256, not_hashed))


@validate_hf_hub_args
def _upload_lfs_files(
    *,
    actions: list[dict[str, Any]],
    oid2addop: dict[str, CommitOperationAdd],
    headers: dict[str, str],
    endpoint: str | None = None,
    num_threads: int = 5,
):
    """
    Uploads the content of `additions` to the Hub using the large file storage protocol.

    Relevant external documentation:
        - LFS Batch API: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        actions (`list[dict[str, Any]]`):
            LFS batch actions returned by the server.
        oid2addop (`dict[str, CommitOperationAdd]`):
            A dictionary mapping the OID of the file to the corresponding `CommitOperationAdd` object.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        endpoint (`str`, *optional*):
            The endpoint to use for the request. Defaults to `constants.ENDPOINT`.
        num_threads (`int`, *optional*):
            The number of concurrent threads to use when uploading. Defaults to 5.

    Raises:
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If an upload failed for any reason
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        num_threads (`int`, *optional*):
            The number of concurrent threads to use when uploading. Defaults to 5.
        revision (`str`, *optional*):
            The git revision to upload to.

    Raises:
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If an upload failed for any reason
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the server returns malformed responses
        [`HfHubHTTPError`]
            If the LFS batch endpoint returned an HTTP error.
    """
    # Filter out files already present upstream
    filtered_actions = []
    for action in actions:
        if action.get("actions") is None:
            logger.debug(
                f"Content of file {oid2addop[action['oid']].path_in_repo} is already present upstream - skipping upload."
            )
        else:
            filtered_actions.append(action)

    # Upload according to server-provided actions
    def _wrapped_lfs_upload(batch_action) -> None:
        try:
            operation = oid2addop[batch_action["oid"]]
            lfs_upload(operation=operation, lfs_batch_action=batch_action, headers=headers, endpoint=endpoint)
        except Exception as exc:
            raise RuntimeError(f"Error while uploading '{operation.path_in_repo}' to the Hub.") from exc

    if len(filtered_actions) == 1:
        logger.debug("Uploading 1 LFS file to the Hub")
        _wrapped_lfs_upload(filtered_actions[0])
    else:
        logger.debug(
            f"Uploading {len(filtered_actions)} LFS files to the Hub using up to {num_threads} threads concurrently"
        )
        from .utils import hf_thread_map
        from .utils import tqdm as hf_tqdm

        hf_thread_map(
            _wrapped_lfs_upload,
            filtered_actions,
            desc=f"Upload {len(filtered_actions)} LFS files",
            max_workers=num_threads,
            tqdm_class=hf_tqdm,
        )


@validate_hf_hub_args
def _upload_xet_files(
    *,
    additions: list[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    headers: dict[str, str],
    endpoint: str | None = None,
    revision: str | None = None,
    create_pr: bool | None = None,
):
    """
    Uploads the content of `additions` to the Hub using the xet storage protocol.
    This chunks the files and deduplicates the chunks before uploading them to xetcas storage.

    Args:
        additions (`` of `CommitOperationAdd`):
            The files to be uploaded.
        repo_type (`str`):
            Type of the repo (e.g. `"model"`, `"dataset"`, `"space"`).
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        endpoint: (`str`, *optional*):
            The endpoint to use for the xetcas service. Defaults to `constants.ENDPOINT`.
        revision (`str`, *optional*):
            The git revision to upload to.
        create_pr (`bool`, *optional*):
            Whether or not to create a Pull Request with that commit.

    Raises:
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If an upload failed for any reason.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the server returns malformed responses or if the user is unauthorized to upload to xet storage.
        [`HfHubHTTPError`]
            If the LFS batch endpoint returned an HTTP error.

    **How it works:**
        The file upload system uses Xet storage, which is a content-addressable storage system that breaks files into chunks
        for efficient storage and transfer.

        ``session.new_upload_commit()`` manages uploading files by:
            - Registering upload tasks and starting upload immediately in the background
            - Breaking files into smaller chunks for efficient storage
            - Avoiding duplicate storage by recognizing identical chunks across files
            - Connecting to a storage server (CAS server) that manages these chunks

        Authentication works transparently: the upload commit accepts a ``token_refresh_url``
        that is used to refresh the short-lived xet write token as needed.

        The upload process works like this:
        1. Upload tasks run in parallel:
            1.1. Read the file content from a path, bytes array, or a stream.
            1.2. Split the file content into smaller chunks based on content patterns: each chunk gets a unique ID based on what's in it.
            1.3. For each chunk:
                - Check if it already exists in storage.
                - Skip uploading chunks that already exist.
            1.4. Group chunks into larger blocks for efficient transfer.
            1.5. Upload these blocks to the storage server.
            1.6. Assemble the file manifest locally and send it to the server for validation.
        2. Return reference files that contain information about the uploaded files, which can be used later to download them.
    """
    if len(additions) == 0:
        return

    # at this point, we know that hf_xet is installed
    from .utils._xet import (
        XetTokenType,
        abort_xet_session,
        get_xet_session,
        xet_connection_info_refresh_url,
        xet_headers_without_auth,
    )
    from .utils._xet_progress_reporting import XetUploadProgressReporter

    refresh_url = xet_connection_info_refresh_url(
        token_type=XetTokenType.WRITE,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
    )
    if create_pr:
        refresh_url += "?create_pr=1"

    xet_headers = xet_headers_without_auth(headers)

    import hf_xet

    session = get_xet_session()
    progress = None

    def _sha256_arg(op: CommitOperationAdd):
        # If the sha256 is already known, pass it to avoid recomputation. Otherwise let hf_xet
        # compute it while chunking the file (single read pass) and backfill it afterwards.
        return op.upload_info.sha256.hex() if op.upload_info.is_hashed else hf_xet.COMPUTE_SHA256

    try:
        from .utils import are_progress_bars_disabled

        if not are_progress_bars_disabled():
            progress = XetUploadProgressReporter()
            progress_callback = progress.update_progress
        else:
            progress_callback = None

        all_bytes_ops = [op for op in additions if isinstance(op.path_or_fileobj, bytes)]
        all_paths_ops = [op for op in additions if isinstance(op.path_or_fileobj, (str, Path))]

        handles: list[tuple[CommitOperationAdd, Any]] = []
        with session.new_upload_commit(
            token_refresh_url=refresh_url,
            token_refresh_headers=headers,
            custom_headers=xet_headers,
            progress_callback=progress_callback,
        ) as commit:
            for op in all_paths_ops:
                handles.append((op, commit.start_upload_file(str(op.path_or_fileobj), sha256=_sha256_arg(op))))
            for op in all_bytes_ops:
                handles.append((op, commit.start_upload_bytes(op.path_or_fileobj, sha256=_sha256_arg(op))))

        # Backfill sha256 computed by hf_xet (needed later for the commit payload).
        for op, handle in handles:
            if not op.upload_info.is_hashed:
                op.upload_info.sha256 = bytes.fromhex(handle.result().xet_info.sha256)
    except KeyboardInterrupt:
        abort_xet_session()
        raise
    finally:
        if progress is not None:
            progress.close()


def _validate_preupload_info(preupload_info: dict):
    files = preupload_info.get("files")
    if not isinstance(files, list):
        raise ValueError("preupload_info is improperly formatted")
    for file_info in files:
        if not (
            isinstance(file_info, dict)
            and isinstance(file_info.get("path"), str)
            and isinstance(file_info.get("uploadMode"), str)
            and (file_info["uploadMode"] in ("lfs", "regular"))
        ):
            raise ValueError("preupload_info is improperly formatted:")
    return preupload_info


@validate_hf_hub_args
def _fetch_upload_modes(
    additions: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    headers: dict[str, str],
    revision: str,
    endpoint: str | None = None,
    create_pr: bool = False,
    gitignore_content: str | None = None,
) -> None:
    """
    Requests the Hub "preupload" endpoint to determine whether each input file should be uploaded as a regular git blob,
    as a git LFS blob, or as a XET file. Input `additions` are mutated in-place with the upload mode.

    Args:
        additions (`Iterable` of :class:`CommitOperationAdd`):
            Iterable of :class:`CommitOperationAdd` describing the files to
            upload to the Hub.
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.
        gitignore_content (`str`, *optional*):
            The content of the `.gitignore` file to know which files should be ignored. The order of priority
            is to first check if `gitignore_content` is passed, then check if the `.gitignore` file is present
            in the list of files to commit and finally default to the `.gitignore` file already hosted on the Hub
            (if any).
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT

    # Fetch upload mode (LFS or regular) chunk by chunk.
    upload_modes: dict[str, UploadMode] = {}
    should_ignore_info: dict[str, bool] = {}
    oid_info: dict[str, str | None] = {}

    for chunk in chunk_iterable(additions, 256):
        payload: dict = {
            "files": [
                {
                    "path": op.path_in_repo,
                    "sample": base64.b64encode(op.upload_info.sample).decode("ascii"),
                    "size": op.upload_info.size,
                }
                for op in chunk
            ]
        }
        if gitignore_content is not None:
            payload["gitIgnore"] = gitignore_content

        resp = http_backoff(
            "POST",
            f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}",
            json=payload,
            headers=headers,
            params={"create_pr": "1"} if create_pr else None,
        )
        hf_raise_for_status(resp)
        preupload_info = _validate_preupload_info(resp.json())
        upload_modes.update(**{file["path"]: file["uploadMode"] for file in preupload_info["files"]})
        should_ignore_info.update(**{file["path"]: file["shouldIgnore"] for file in preupload_info["files"]})
        oid_info.update(**{file["path"]: file.get("oid") for file in preupload_info["files"]})

    # Set upload mode for each addition operation
    for addition in additions:
        addition._upload_mode = upload_modes[addition.path_in_repo]
        addition._should_ignore = should_ignore_info[addition.path_in_repo]
        addition._remote_oid = oid_info[addition.path_in_repo]

    # Empty files cannot be uploaded as LFS (S3 would fail with a 501 Not Implemented)
    # => empty files are uploaded as "regular" to still allow users to commit them.
    for addition in additions:
        if addition.upload_info.size == 0:
            addition._upload_mode = "regular"


class _CopySource(NamedTuple):
    src_repo_id: str | None
    src_repo_type: str | None
    path: str
    revision: str | None


@validate_hf_hub_args
def _fetch_files_to_copy(
    copies: Iterable[CommitOperationCopy],
    repo_type: str,
    repo_id: str,
    headers: dict[str, str],
    revision: str,
    endpoint: str | None = None,
) -> dict[_CopySource, Union["RepoFile", bytes]]:
    """
    Fetch information about the files to copy.

    For LFS files, we only need their metadata (file size and sha256) while for regular files
    we need to download the raw content from the Hub.

    Args:
        copies (`Iterable` of :class:`CommitOperationCopy`):
            Iterable of :class:`CommitOperationCopy` describing the files to
            copy on the Hub.
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.

    Returns: `dict[_CopySource, Union[RepoFile, bytes]]]`
        Key is `(src_repo_id, src_repo_type, path, revision)`. For intra-repo copies,
        `src_repo_id` and `src_repo_type` are `None`.
        Value is the raw content as bytes (for regular files) or the file information as a RepoFile (for LFS files).

    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    from .hf_api import HfApi, RepoFolder

    hf_api = HfApi(endpoint=endpoint, headers=headers)
    copies = list(copies)
    files_to_copy: dict[_CopySource, Union["RepoFile", bytes]] = {}
    oid_info: dict[tuple[str, str | None], str | None] = {}

    # 1. Fetch OIDs for destination paths in batches.
    dest_paths = [op.path_in_repo for op in copies]
    for batch in chunk_iterable(dest_paths, FETCH_LFS_BATCH_SIZE):
        dest_repo_files = hf_api.get_paths_info(
            repo_id=repo_id,
            paths=list(batch),
            revision=revision,
            repo_type=repo_type,
        )
        for file in dest_repo_files:
            if not isinstance(file, RepoFolder):
                oid_info[(file.path, revision)] = file.blob_id

    # 2. Fetch source file info, grouped by (src_repo_id, src_repo_type, src_revision).
    copies.sort(key=lambda op: (op.src_repo_id or "", op.src_repo_type or "", op.src_revision or ""))
    for (src_repo_id_key, src_repo_type_key, src_revision_key), group in groupby(
        copies, key=lambda op: (op.src_repo_id, op.src_repo_type, op.src_revision)
    ):
        operations = list(group)
        is_cross_repo = src_repo_id_key is not None
        eff_repo_id = src_repo_id_key or repo_id
        eff_repo_type = src_repo_type_key or repo_type
        eff_revision = src_revision_key or ("main" if is_cross_repo else revision)

        src_paths = [op.src_path_in_repo for op in operations]
        for paths_batch in chunk_iterable(src_paths, FETCH_LFS_BATCH_SIZE):
            src_repo_files = hf_api.get_paths_info(
                repo_id=eff_repo_id,
                paths=list(paths_batch),
                revision=eff_revision,
                repo_type=eff_repo_type,
            )
            for src_repo_file in src_repo_files:
                if isinstance(src_repo_file, RepoFolder):
                    raise NotImplementedError("Copying a folder is not implemented.")
                source = _CopySource(src_repo_id_key, src_repo_type_key, src_repo_file.path, src_revision_key)
                if src_repo_file.lfs:
                    files_to_copy[source] = src_repo_file
                else:
                    url = hf_hub_url(
                        endpoint=endpoint,
                        repo_type=eff_repo_type,
                        repo_id=eff_repo_id,
                        revision=eff_revision,
                        filename=src_repo_file.path,
                    )
                    response = get_session().get(url, headers=headers)
                    hf_raise_for_status(response)
                    files_to_copy[source] = response.content
                if not is_cross_repo:
                    oid_info[(src_repo_file.path, src_revision_key)] = src_repo_file.blob_id

        for operation in operations:
            key = _CopySource(src_repo_id_key, src_repo_type_key, operation.src_path_in_repo, src_revision_key)
            if key not in files_to_copy:
                source_desc = f" from {eff_repo_type}s/{eff_repo_id}" if is_cross_repo else ""
                raise EntryNotFoundError(
                    f"Cannot copy {operation.src_path_in_repo} at revision "
                    f"{eff_revision}{source_desc}: file is missing on repo."
                )
            operation._dest_oid = oid_info.get((operation.path_in_repo, revision))
            if not is_cross_repo:
                operation._src_oid = oid_info.get((operation.src_path_in_repo, operation.src_revision))

    return files_to_copy


def _prepare_commit_payload(
    operations: Iterable[CommitOperation],
    files_to_copy: dict[_CopySource, Union["RepoFile", bytes]],
    commit_message: str,
    commit_description: str | None = None,
    parent_commit: str | None = None,
) -> Iterable[dict[str, Any]]:
    """
    Builds the payload to POST to the `/commit` API of the Hub.

    Payload is returned as an iterator so that it can be streamed as a ndjson in the
    POST request.

    For more information, see:
        - https://github.com/huggingface/huggingface_hub/issues/1085#issuecomment-1265208073
        - http://ndjson.org/
    """
    commit_description = commit_description if commit_description is not None else ""

    # 1. Send a header item with the commit metadata
    header_value = {"summary": commit_message, "description": commit_description}
    if parent_commit is not None:
        header_value["parentCommit"] = parent_commit
    yield {"key": "header", "value": header_value}

    nb_ignored_files = 0

    # 2. Send operations, one per line
    for operation in operations:
        # Skip ignored files
        if isinstance(operation, CommitOperationAdd) and operation._should_ignore:
            logger.debug(f"Skipping file '{operation.path_in_repo}' in commit (ignored by gitignore file).")
            nb_ignored_files += 1
            continue

        # 2.a. Case adding a regular file
        if isinstance(operation, CommitOperationAdd) and operation._upload_mode == "regular":
            yield {
                "key": "file",
                "value": {
                    "content": operation.b64content().decode(),
                    "path": operation.path_in_repo,
                    "encoding": "base64",
                },
            }
        # 2.b. Case adding an LFS file
        elif isinstance(operation, CommitOperationAdd) and operation._upload_mode == "lfs":
            yield {
                "key": "lfsFile",
                "value": {
                    "path": operation.path_in_repo,
                    "algo": "sha256",
                    "oid": operation.upload_info.sha256.hex(),
                    "size": operation.upload_info.size,
                },
            }
        # 2.c. Case deleting a file or folder
        elif isinstance(operation, CommitOperationDelete):
            yield {
                "key": "deletedFolder" if operation.is_folder else "deletedFile",
                "value": {"path": operation.path_in_repo},
            }
        # 2.d. Case copying a file or folder
        elif isinstance(operation, CommitOperationCopy):
            source = _CopySource(
                operation.src_repo_id,
                operation.src_repo_type,
                operation.src_path_in_repo,
                operation.src_revision,
            )
            file_to_copy = files_to_copy[source]
            if isinstance(file_to_copy, bytes):
                yield {
                    "key": "file",
                    "value": {
                        "content": base64.b64encode(file_to_copy).decode(),
                        "path": operation.path_in_repo,
                        "encoding": "base64",
                    },
                }
            elif file_to_copy.lfs:
                yield {
                    "key": "lfsFile",
                    "value": {
                        "path": operation.path_in_repo,
                        "algo": "sha256",
                        "oid": file_to_copy.lfs.sha256,
                    },
                }
            else:
                raise ValueError(
                    "Malformed files_to_copy (should be raw file content as bytes or RepoFile objects with LFS info."
                )
        # 2.e. Never expected to happen
        else:
            raise ValueError(
                f"Unknown operation to commit. Operation: {operation}. Upload mode:"
                f" {getattr(operation, '_upload_mode', None)}"
            )

    if nb_ignored_files > 0:
        logger.info(f"Skipped {nb_ignored_files} file(s) in commit (ignored by gitignore file).")


def _send_commit(
    *,
    operations: Iterable[CommitOperation],
    files_to_copy: dict[_CopySource, Union["RepoFile", bytes]],
    commit_message: str,
    commit_description: str,
    repo_type: str,
    repo_id: str,
    headers: dict[str, str],
    revision: str,
    endpoint: str | None = None,
    parent_commit: str | None = None,
    create_pr: bool = False,
    hot_reload: bool = False,
    retry_on_error: bool = False,
) -> "CommitInfo":
    """
    Build the ndjson payload for `operations` and POST it to the `/commit` endpoint of the Hub.

    `revision` must already be URL-quoted. Operations must be ready to commit: upload modes fetched
    and LFS/Xet content uploaded.

    If `retry_on_error` is `True`, the POST is retried with exponential backoff on transient errors.
    Caution: if a commit succeeds but its response is lost, the retry will create a duplicate commit
    (and a duplicate pull request if `create_pr` is set) — only enable it when this is acceptable.

    Returns a [`CommitInfo`] built from the server response.

    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
    """
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    payload = _prepare_commit_payload(
        operations=operations,
        files_to_copy=files_to_copy,
        commit_message=commit_message,
        commit_description=commit_description,
        parent_commit=parent_commit,
    )
    data = b"".join(json.dumps(item).encode() + b"\n" for item in payload)
    # See https://github.com/huggingface/huggingface_hub/issues/1085#issuecomment-1265208073
    headers = {"Content-Type": "application/x-ndjson", **headers}
    url = f"{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"

    params: dict[str, str] = {}
    if create_pr:
        params["create_pr"] = "1"
    if hot_reload:
        params["hot_reload"] = "1"

    if retry_on_error:
        response = http_backoff("POST", url, headers=headers, content=data, params=params)
    else:
        response = get_session().post(url, headers=headers, content=data, params=params)
    hf_raise_for_status(response, endpoint_name="commit")

    commit_data = response.json()
    from .hf_api import CommitInfo

    return CommitInfo(
        commit_url=commit_data["commitUrl"],
        commit_message=commit_message,
        commit_description=commit_description,
        oid=commit_data["commitOid"],
        pr_url=commit_data["pullRequestUrl"] if create_pr else None,
        _endpoint=endpoint,
    )
<|endoftext|>import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from huggingface_hub.utils import logging, yaml_dump


logger = logging.get_logger(__name__)


@dataclass
class EvalResult:
    """
    Flattened representation of individual evaluation results found in model-index of Model Cards.

    For more information on the model-index spec, see https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1.

    Args:
        task_type (`str`):
            The task identifier. Example: "image-classification".
        dataset_type (`str`):
            The dataset identifier. Example: "common_voice". Use dataset id from https://hf.co/datasets.
        dataset_name (`str`):
            A pretty name for the dataset. Example: "Common Voice (French)".
        metric_type (`str`):
            The metric identifier. Example: "wer". Use metric id from https://hf.co/metrics.
        metric_value (`Any`):
            The metric value. Example: 0.9 or "20.0 ± 1.2".
        task_name (`str`, *optional*):
            A pretty name for the task. Example: "Speech Recognition".
        dataset_config (`str`, *optional*):
            The name of the dataset configuration used in `load_dataset()`.
            Example: fr in `load_dataset("common_voice", "fr")`. See the `datasets` docs for more info:
            https://hf.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name
        dataset_split (`str`, *optional*):
            The split used in `load_dataset()`. Example: "test".
        dataset_revision (`str`, *optional*):
            The revision (AKA Git Sha) of the dataset used in `load_dataset()`.
            Example: 5503434ddd753f426f4b38109466949a1217c2bb
        dataset_args (`dict[str, Any]`, *optional*):
            The arguments passed during `Metric.compute()`. Example for `bleu`: `{"max_order": 4}`
        metric_name (`str`, *optional*):
            A pretty name for the metric. Example: "Test WER".
        metric_config (`str`, *optional*):
            The name of the metric configuration used in `load_metric()`.
            Example: bleurt-large-512 in `load_metric("bleurt", "bleurt-large-512")`.
            See the `datasets` docs for more info: https://huggingface.co/docs/datasets/v2.1.0/en/loading#load-configurations
        metric_args (`dict[str, Any]`, *optional*):
            The arguments passed during `Metric.compute()`. Example for `bleu`: max_order: 4
        verified (`bool`, *optional*):
            Indicates whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not. Automatically computed by Hugging Face, do not set.
        verify_token (`str`, *optional*):
            A JSON Web Token that is used to verify whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not.
        source_name (`str`, *optional*):
            The name of the source of the evaluation result. Example: "Open LLM Leaderboard".
        source_url (`str`, *optional*):
            The URL of the source of the evaluation result. Example: "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard".
    """

    # Required

    # The task identifier
    # Example: automatic-speech-recognition
    task_type: str

    # The dataset identifier
    # Example: common_voice. Use dataset id from https://hf.co/datasets
    dataset_type: str

    # A pretty name for the dataset.
    # Example: Common Voice (French)
    dataset_name: str

    # The metric identifier
    # Example: wer. Use metric id from https://hf.co/metrics
    metric_type: str

    # Value of the metric.
    # Example: 20.0 or "20.0 ± 1.2"
    metric_value: Any

    # Optional

    # A pretty name for the task.
    # Example: Speech Recognition
    task_name: str | None = None

    # The name of the dataset configuration used in `load_dataset()`.
    # Example: fr in `load_dataset("common_voice", "fr")`.
    # See the `datasets` docs for more info:
    # https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name
    dataset_config: str | None = None

    # The split used in `load_dataset()`.
    # Example: test
    dataset_split: str | None = None

    # The revision (AKA Git Sha) of the dataset used in `load_dataset()`.
    # Example: 5503434ddd753f426f4b38109466949a1217c2bb
    dataset_revision: str | None = None

    # The arguments passed during `Metric.compute()`.
    # Example for `bleu`: max_order: 4
    dataset_args: dict[str, Any] | None = None

    # A pretty name for the metric.
    # Example: Test WER
    metric_name: str | None = None

    # The name of the metric configuration used in `load_metric()`.
    # Example: bleurt-large-512 in `load_metric("bleurt", "bleurt-large-512")`.
    # See the `datasets` docs for more info: https://huggingface.co/docs/datasets/v2.1.0/en/loading#load-configurations
    metric_config: str | None = None

    # The arguments passed during `Metric.compute()`.
    # Example for `bleu`: max_order: 4
    metric_args: dict[str, Any] | None = None

    # Indicates whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not. Automatically computed by Hugging Face, do not set.
    verified: bool | None = None

    # A JSON Web Token that is used to verify whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not.
    verify_token: str | None = None

    # The name of the source of the evaluation result.
    # Example: Open LLM Leaderboard
    source_name: str | None = None

    # The URL of the source of the evaluation result.
    # Example: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
    source_url: str | None = None

    @property
    def unique_identifier(self) -> tuple:
        """Returns a tuple that uniquely identifies this evaluation."""
        return (
            self.task_type,
            self.dataset_type,
            self.dataset_config,
            self.dataset_split,
            self.dataset_revision,
        )

    def is_equal_except_value(self, other: "EvalResult") -> bool:
        """
        Return True if `self` and `other` describe exactly the same metric but with a
        different value.
        """
        for key, _ in self.__dict__.items():
            if key == "metric_value":
                continue
            # For metrics computed by Hugging Face's evaluation service, `verify_token` is derived from `metric_value`,
            # so we exclude it here in the comparison.
            if key != "verify_token" and getattr(self, key) != getattr(other, key):
                return False
        return True

    def __post_init__(self) -> None:
        if self.source_name is not None and self.source_url is None:
            raise ValueError("If `source_name` is provided, `source_url` must also be provided.")


@dataclass
class CardData:
    """Structure containing metadata from a RepoCard.

    [`CardData`] is the parent class of [`ModelCardData`] and [`DatasetCardData`].

    Metadata can be exported as a dictionary or YAML. Export can be customized to alter the representation of the data
    (example: flatten evaluation results). `CardData` behaves as a dictionary (can get, pop, set values) but do not
    inherit from `dict` to allow this export step.
    """

    def __init__(self, ignore_metadata_errors: bool = False, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        """Converts CardData to a dict.

        Returns:
            `dict`: CardData represented as a dictionary ready to be dumped to a YAML
            block for inclusion in a README.md file.
        """

        data_dict = copy.deepcopy(self.__dict__)
        self._to_dict(data_dict)
        return {key: value for key, value in data_dict.items() if value is not None}

    def _to_dict(self, data_dict):
        """Use this method in child classes to alter the dict representation of the data. Alter the dict in-place.

        Args:
            data_dict (`dict`): The raw dict representation of the card data.
        """
        pass

    def to_yaml(self, line_break=None, original_order: list[str] | None = None) -> str:
        """Dumps CardData to a YAML block for inclusion in a README.md file.

        Args:
            line_break (str, *optional*):
                The line break to use when dumping to yaml.
            original_order (`list[str]`, *optional*):
                If provided, reorder the metadata fields to match this list before dumping.
                Any keys not in `original_order` are appended after the listed keys, preserving
                their existing relative order. Useful for round-tripping a YAML block without
                shuffling its keys.

        Returns:
            `str`: CardData represented as a YAML block.
        """
        if original_order:
            original_order_set = set(original_order)
            self.__dict__ = {
                k: self.__dict__[k]
                for k in original_order + [k for k in self.__dict__ if k not in original_order_set]
                if k in self.__dict__
            }
        return yaml_dump(self.to_dict(), sort_keys=False, line_break=line_break).strip()

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return self.to_yaml()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for a given metadata key."""
        value = self.__dict__.get(key)
        return default if value is None else value

    def pop(self, key: str, default: Any = None) -> Any:
        """Pop value for a given metadata key."""
        return self.__dict__.pop(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get value for a given metadata key."""
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value for a given metadata key."""
        self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a given metadata key is set."""
        return key in self.__dict__

    def __len__(self) -> int:
        """Return the number of metadata keys set."""
        return len(self.__dict__)


def _validate_eval_results(
    eval_results: EvalResult | list[EvalResult] | None,
    model_name: str | None,
) -> list[EvalResult]:
    if eval_results is None:
        return []
    if isinstance(eval_results, EvalResult):
        eval_results = [eval_results]
    if not isinstance(eval_results, list) or not all(isinstance(r, EvalResult) for r in eval_results):
        raise ValueError(
            f"`eval_results` should be of type `EvalResult` or a list of `EvalResult`, got {type(eval_results)}."
        )
    if model_name is None:
        raise ValueError("Passing `eval_results` requires `model_name` to be set.")
    return eval_results


class ModelCardData(CardData):
    """Model Card Metadata that is used by Hugging Face Hub when included at the top of your README.md

    Args:
        base_model (`str` or `list[str]`, *optional*):
            The identifier of the base model from which the model derives. This is applicable for example if your model is a
            fine-tune or adapter of an existing model. The value must be the ID of a model on the Hub (or a list of IDs
            if your model derives from multiple models). Defaults to None.
        datasets (`Union[str, list[str]]`, *optional*):
            Dataset or list of datasets that were used to train this model. Should be a dataset ID
            found on https://hf.co/datasets. Defaults to None.
        eval_results (`Union[list[EvalResult], EvalResult]`, *optional*):
            List of `huggingface_hub.EvalResult` that define evaluation results of the model. If provided,
            `model_name` is used to as a name on PapersWithCode's leaderboards. Defaults to `None`.
        language (`Union[str, list[str]]`, *optional*):
            Language of model's training data or metadata. It must be an ISO 639-1, 639-2 or
            639-3 code (two/three letters), or a special value like "code", "multilingual". Defaults to `None`.
        library_name (`str`, *optional*):
            Name of library used by this model. Example: keras or any library from
            https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts.
            Defaults to None.
        license (`str`, *optional*):
            License of this model. Example: apache-2.0 or any license from
            https://huggingface.co/docs/hub/repositories-licenses. Defaults to None.
        license_name (`str`, *optional*):
            Name of the license of this model. Defaults to None. To be used in conjunction with `license_link`.
            Common licenses (Apache-2.0, MIT, CC-BY-SA-4.0) do not need a name. In that case, use `license` instead.
        license_link (`str`, *optional*):
            Link to the license of this model. Defaults to None. To be used in conjunction with `license_name`.
            Common licenses (Apache-2.0, MIT, CC-BY-SA-4.0) do not need a link. In that case, use `license` instead.
        metrics (`list[str]`, *optional*):
            List of metrics used to evaluate this model. Should be a metric name that can be found
            at https://hf.co/metrics. Example: 'accuracy'. Defaults to None.
        model_name (`str`, *optional*):
            A name for this model. It is used along with
            `eval_results` to construct the `model-index` within the card's metadata. The name
            you supply here is what will be used on PapersWithCode's leaderboards. If None is provided
            then the repo name is used as a default. Defaults to None.
        pipeline_tag (`str`, *optional*):
            The pipeline tag associated with the model. Example: "text-classification".
        tags (`list[str]`, *optional*):
            List of tags to add to your model that can be used when filtering on the Hugging
            Face Hub. Defaults to None.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
        kwargs (`dict`, *optional*):
            Additional metadata that will be added to the model card. Defaults to None.

    Example:
        ```python
        >>> from huggingface_hub import ModelCardData
        >>> card_data = ModelCardData(
        ...     language="en",
        ...     license="mit",
        ...     library_name="timm",
        ...     tags=['image-classification', 'resnet'],
        ... )
        >>> card_data.to_dict()
        {'language': 'en', 'license': 'mit', 'library_name': 'timm', 'tags': ['image-classification', 'resnet']}

        ```
    """

    def __init__(
        self,
        *,
        base_model: str | list[str] | None = None,
        datasets: str | list[str] | None = None,
        eval_results: list[EvalResult] | None = None,
        language: str | list[str] | None = None,
        library_name: str | None = None,
        license: str | None = None,
        license_name: str | None = None,
        license_link: str | None = None,
        metrics: list[str] | None = None,
        model_name: str | None = None,
        pipeline_tag: str | None = None,
        tags: list[str] | None = None,
        ignore_metadata_errors: bool = False,
        **kwargs,
    ):
        self.base_model = base_model
        self.datasets = datasets
        self.eval_results = eval_results
        self.language = language
        self.library_name = library_name
        self.license = license
        self.license_name = license_name
        self.license_link = license_link
        self.metrics = metrics
        self.model_name = model_name
        self.pipeline_tag = pipeline_tag
        self.tags = _to_unique_list(tags)

        model_index = kwargs.pop("model-index", None)
        if model_index:
            try:
                model_name, eval_results = model_index_to_eval_results(model_index)
                self.model_name = model_name
                self.eval_results = eval_results
            except (KeyError, TypeError) as error:
                if ignore_metadata_errors:
                    logger.warning("Invalid model-index. Not loading eval results into CardData.")
                else:
                    raise ValueError(
                        f"Invalid `model_index` in metadata cannot be parsed: {error.__class__} {error}. Pass"
                        " `ignore_metadata_errors=True` to ignore this error while loading a Model Card. Warning:"
                        " some information will be lost. Use it at your own risk."
                    )

        super().__init__(**kwargs)

        if self.eval_results:
            try:
                self.eval_results = _validate_eval_results(self.eval_results, self.model_name)
            except Exception as e:
                if ignore_metadata_errors:
                    logger.warning(f"Failed to validate eval_results: {e}. Not loading eval results into CardData.")
                else:
                    raise ValueError(f"Failed to validate eval_results: {e}") from e

    def _to_dict(self, data_dict):
        """Format the internal data dict. In this case, we convert eval results to a valid model index"""
        if self.eval_results is not None:
            data_dict["model-index"] = eval_results_to_model_index(self.model_name, self.eval_results)  # type: ignore
            del data_dict["eval_results"], data_dict["model_name"]


class DatasetCardData(CardData):
    """Dataset Card Metadata that is used by Hugging Face Hub when included at the top of your README.md

    Args:
        language (`list[str]`, *optional*):
            Language of dataset's data or metadata. It must be an ISO 639-1, 639-2 or
            639-3 code (two/three letters), or a special value like "code", "multilingual".
        license (`Union[str, list[str]]`, *optional*):
            License(s) of this dataset. Example: apache-2.0 or any license from
            https://huggingface.co/docs/hub/repositories-licenses.
        annotations_creators (`Union[str, list[str]]`, *optional*):
            How the annotations for the dataset were created.
            Options are: 'found', 'crowdsourced', 'expert-generated', 'machine-generated', 'no-annotation', 'other'.
        language_creators (`Union[str, list[str]]`, *optional*):
            How the text-based data in the dataset was created.
            Options are: 'found', 'crowdsourced', 'expert-generated', 'machine-generated', 'other'
        multilinguality (`Union[str, list[str]]`, *optional*):
            Whether the dataset is multilingual.
            Options are: 'monolingual', 'multilingual', 'translation', 'other'.
        size_categories (`Union[str, list[str]]`, *optional*):
            The number of examples in the dataset. Options are: 'n<1K', '1K<n<10K', '10K<n<100K',
            '100K<n<1M', '1M<n<10M', '10M<n<100M', '100M<n<1B', '1B<n<10B', '10B<n<100B', '100B<n<1T', 'n>1T', and 'other'.
        source_datasets (`list[str]]`, *optional*):
            Indicates whether the dataset is an original dataset or extended from another existing dataset.
            Options are: 'original' and 'extended'.
        task_categories (`Union[str, list[str]]`, *optional*):
            What categories of task does the dataset support?
        task_ids (`Union[str, list[str]]`, *optional*):
            What specific tasks does the dataset support?
        paperswithcode_id (`str`, *optional*):
            ID of the dataset on PapersWithCode.
        pretty_name (`str`, *optional*):
            A more human-readable name for the dataset. (ex. "Cats vs. Dogs")
        train_eval_index (`dict`, *optional*):
            A dictionary that describes the necessary spec for doing evaluation on the Hub.
            If not provided, it will be gathered from the 'train-eval-index' key of the kwargs.
        config_names (`Union[str, list[str]]`, *optional*):
            A list of the available dataset configs for the dataset.
    """

    def __init__(
        self,
        *,
        language: str | list[str] | None = None,
        license: str | list[str] | None = None,
        annotations_creators: str | list[str] | None = None,
        language_creators: str | list[str] | None = None,
        multilinguality: str | list[str] | None = None,
        size_categories: str | list[str] | None = None,
        source_datasets: list[str] | None = None,
        task_categories: str | list[str] | None = None,
        task_ids: str | list[str] | None = None,
        paperswithcode_id: str | None = None,
        pretty_name: str | None = None,
        train_eval_index: dict | None = None,
        config_names: str | list[str] | None = None,
        ignore_metadata_errors: bool = False,
        **kwargs,
    ):
        self.annotations_creators = annotations_creators
        self.language_creators = language_creators
        self.language = language
        self.license = license
        self.multilinguality = multilinguality
        self.size_categories = size_categories
        self.source_datasets = source_datasets
        self.task_categories = task_categories
        self.task_ids = task_ids
        self.paperswithcode_id = paperswithcode_id
        self.pretty_name = pretty_name
        self.config_names = config_names

        # TODO - maybe handle this similarly to EvalResult?
        self.train_eval_index = train_eval_index or kwargs.pop("train-eval-index", None)
        super().__init__(**kwargs)

    def _to_dict(self, data_dict):
        data_dict["train-eval-index"] = data_dict.pop("train_eval_index")


class SpaceCardData(CardData):
    """Space Card Metadata that is used by Hugging Face Hub when included at the top of your README.md

    To get an exhaustive reference of Spaces configuration, please visit https://huggingface.co/docs/hub/spaces-config-reference#spaces-configuration-reference.

    Args:
        title (`str`, *optional*):
            Title of the Space.
        sdk (`str`, *optional*):
            SDK of the Space (one of `gradio`, `streamlit`, `docker`, or `static`).
        sdk_version (`str`, *optional*):
            Version of the used SDK (if Gradio/Streamlit sdk).
        python_version (`str`, *optional*):
            Python version used in the Space (if Gradio/Streamlit sdk).
        app_file (`str`, *optional*):
            Path to your main application file (which contains either gradio or streamlit Python code, or static html code).
            Path is relative to the root of the repository.
        app_port (`str`, *optional*):
            Port on which your application is running. Used only if sdk is `docker`.
        license (`str`, *optional*):
            License of this model. Example: apache-2.0 or any license from
            https://huggingface.co/docs/hub/repositories-licenses.
        duplicated_from (`str`, *optional*):
            ID of the original Space if this is a duplicated Space.
        models (`list[str]`, *optional*):
            List of models related to this Space. Should be a dataset ID found on https://hf.co/models.
        datasets (`list[str]`, *optional*):
            List of datasets related to this Space. Should be a dataset ID found on https://hf.co/datasets.
        tags (`list[str]`, *optional*):
            List of tags to add to your Space that can be used when filtering on the Hub.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
        kwargs (`dict`, *optional*):
            Additional metadata that will be added to the space card.

    Example:
        ```python
        >>> from huggingface_hub import SpaceCardData
        >>> card_data = SpaceCardData(
        ...     title="Dreambooth Training",
        ...     license="mit",
        ...     sdk="gradio",
        ...     duplicated_from="multimodalart/dreambooth-training"
        ... )
        >>> card_data.to_dict()
        {'title': 'Dreambooth Training', 'sdk': 'gradio', 'license': 'mit', 'duplicated_from': 'multimodalart/dreambooth-training'}
        ```
    """

    def __init__(
        self,
        *,
        title: str | None = None,
        sdk: str | None = None,
        sdk_version: str | None = None,
        python_version: str | None = None,
        app_file: str | None = None,
        app_port: int | None = None,
        license: str | None = None,
        duplicated_from: str | None = None,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
        tags: list[str] | None = None,
        ignore_metadata_errors: bool = False,
        **kwargs,
    ):
        self.title = title
        self.sdk = sdk
        self.sdk_version = sdk_version
        self.python_version = python_version
        self.app_file = app_file
        self.app_port = app_port
        self.license = license
        self.duplicated_from = duplicated_from
        self.models = models
        self.datasets = datasets
        self.tags = _to_unique_list(tags)
        super().__init__(**kwargs)


def model_index_to_eval_results(model_index: list[dict[str, Any]]) -> tuple[str, list[EvalResult]]:
    """Takes in a model index and returns the model name and a list of `huggingface_hub.EvalResult` objects.

    A detailed spec of the model index can be found here:
    https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1

    Args:
        model_index (`list[dict[str, Any]]`):
            A model index data structure, likely coming from a README.md file on the
            Hugging Face Hub.

    Returns:
        model_name (`str`):
            The name of the model as found in the model index. This is used as the
            identifier for the model on leaderboards like PapersWithCode.
        eval_results (`list[EvalResult]`):
            A list of `huggingface_hub.EvalResult` objects containing the metrics
            reported in the provided model_index.

    Example:
        ```python
        >>> from huggingface_hub.repocard_data import model_index_to_eval_results
        >>> # Define a minimal model index
        >>> model_index = [
        ...     {
        ...         "name": "my-cool-model",
        ...         "results": [
        ...             {
        ...                 "task": {
        ...                     "type": "image-classification"
        ...                 },
        ...                 "dataset": {
        ...                     "type": "beans",
        ...                     "name": "Beans"
        ...                 },
        ...                 "metrics": [
        ...                     {
        ...                         "type": "accuracy",
        ...                         "value": 0.9
        ...                     }
        ...                 ]
        ...             }
        ...         ]
        ...     }
        ... ]
        >>> model_name, eval_results = model_index_to_eval_results(model_index)
        >>> model_name
        'my-cool-model'
        >>> eval_results[0].task_type
        'image-classification'
        >>> eval_results[0].metric_type
        'accuracy'

        ```
    """

    if not model_index:
        raise ValueError("`model_index` must be a non-empty list.")

    eval_results = []
    for elem in model_index:
        name = elem["name"]
        results = elem["results"]
        for result in results:
            task_type = result["task"]["type"]
            task_name = result["task"].get("name")
            dataset_type = result["dataset"]["type"]
            dataset_name = result["dataset"]["name"]
            dataset_config = result["dataset"].get("config")
            dataset_split = result["dataset"].get("split")
            dataset_revision = result["dataset"].get("revision")
            dataset_args = result["dataset"].get("args")
            source = result.get("source") or {}
            source_name = source.get("name")
            source_url = source.get("url")

            for metric in result["metrics"]:
                metric_type = metric["type"]
                metric_value = metric["value"]
                metric_name = metric.get("name")
                metric_args = metric.get("args")
                metric_config = metric.get("config")
                verified = metric.get("verified")
                verify_token = metric.get("verifyToken")

                eval_result = EvalResult(
                    task_type=task_type,  # Required
                    dataset_type=dataset_type,  # Required
                    dataset_name=dataset_name,  # Required
                    metric_type=metric_type,  # Required
                    metric_value=metric_value,  # Required
                    task_name=task_name,
                    dataset_config=dataset_config,
                    dataset_split=dataset_split,
                    dataset_revision=dataset_revision,
                    dataset_args=dataset_args,
                    metric_name=metric_name,
                    metric_args=metric_args,
                    metric_config=metric_config,
                    verified=verified,
                    verify_token=verify_token,
                    source_name=source_name,
                    source_url=source_url,
                )
                eval_results.append(eval_result)
    return name, eval_results


def _remove_none(obj):
    """
    Recursively remove `None` values from a dict. Borrowed from: https://stackoverflow.com/a/20558778
    """
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_remove_none(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return type(obj)((_remove_none(k), _remove_none(v)) for k, v in obj.items() if k is not None and v is not None)
    else:
        return obj


def eval_results_to_model_index(model_name: str, eval_results: list[EvalResult]) -> list[dict[str, Any]]:
    """Takes in given model name and list of `huggingface_hub.EvalResult` and returns a
    valid model-index that will be compatible with the format expected by the
    Hugging Face Hub.

    Args:
        model_name (`str`):
            Name of the model (ex. "my-cool-model"). This is used as the identifier
            for the model on leaderboards like PapersWithCode.
        eval_results (`list[EvalResult]`):
            List of `huggingface_hub.EvalResult` objects containing the metrics to be
            reported in the model-index.

    Returns:
        model_index (`list[dict[str, Any]]`): The eval_results converted to a model-index.

    Example:
        ```python
        >>> from huggingface_hub.repocard_data import eval_results_to_model_index, EvalResult
        >>> # Define minimal eval_results
        >>> eval_results = [
        ...     EvalResult(
        ...         task_type="image-classification",  # Required
        ...         dataset_type="beans",  # Required
        ...         dataset_name="Beans",  # Required
        ...         metric_type="accuracy",  # Required
        ...         metric_value=0.9,  # Required
        ...     )
        ... ]
        >>> eval_results_to_model_index("my-cool-model", eval_results)
        [{'name': 'my-cool-model', 'results': [{'task': {'type': 'image-classification'}, 'dataset': {'name': 'Beans', 'type': 'beans'}, 'metrics': [{'type': 'accuracy', 'value': 0.9}]}]}]

        ```
    """

    # Metrics are reported on a unique task-and-dataset basis.
    # Here, we make a map of those pairs and the associated EvalResults.
    task_and_ds_types_map: dict[Any, list[EvalResult]] = defaultdict(list)
    for eval_result in eval_results:
        task_and_ds_types_map[eval_result.unique_identifier].append(eval_result)

    # Use the map from above to generate the model index data.
    model_index_data: list[dict[str, Any]] = []
    for results in task_and_ds_types_map.values():
        # All items from `results` share same metadata
        sample_result = results[0]
        data: dict[str, Any] = {
            "task": {
                "type": sample_result.task_type,
                "name": sample_result.task_name,
            },
            "dataset": {
                "name": sample_result.dataset_name,
                "type": sample_result.dataset_type,
                "config": sample_result.dataset_config,
                "split": sample_result.dataset_split,
                "revision": sample_result.dataset_revision,
                "args": sample_result.dataset_args,
            },
            "metrics": [
                {
                    "type": result.metric_type,
                    "value": result.metric_value,
                    "name": result.metric_name,
                    "config": result.metric_config,
                    "args": result.metric_args,
                    "verified": result.verified,
                    "verifyToken": result.verify_token,
                }
                for result in results
            ],
        }
        if sample_result.source_url is not None:
            source: dict[str, str] = {
                "url": sample_result.source_url,
            }
            if sample_result.source_name is not None:
                source["name"] = sample_result.source_name
            data["source"] = source
        model_index_data.append(data)

    # TODO - Check if there cases where this list is longer than one?
    # Finally, the model index itself is list of dicts.
    model_index = [
        {
            "name": model_name,
            "results": model_index_data,
        }
    ]
    return _remove_none(model_index)


def _to_unique_list(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return tags
    if isinstance(tags, str):
        tags = [tags]
    unique_tags = []  # make tags unique + keep order explicitly
    for tag in tags:
        if tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags
<|endoftext|># Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Keyless CI/CD authentication via OIDC token exchange ("Trusted Publishers").

A CI job proves its identity to the Hub with a short-lived OIDC id token minted by its CI
provider (e.g. GitHub Actions), then exchanges it at ``POST {ENDPOINT}/oauth/token`` (RFC 8693)
for a short-lived Hugging Face token — no long-lived ``HF_TOKEN`` secret to store.

This module is self-contained: it only handles minting the provider id token and the exchange.
It deliberately does not register a public API or a CLI verb; the integration point is the token
resolution in ``utils/_auth.py`` (see ``_get_token_from_oidc``).

Docs: https://huggingface.co/docs/hub/trusted-publishers
"""

import os
from enum import Enum

from . import constants
from .errors import OIDCError
from .utils import get_session, hf_raise_for_status


# RFC 8693 token-exchange grant + id-token subject type (see trusted-publishers docs).
_TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
_ID_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"


class Provider(str, Enum):
    """CI providers that can mint an OIDC id token natively. GitHub Actions only for now."""

    GITHUB = "github"


def detect_provider() -> Provider | None:
    """Detect the CI provider able to mint an OIDC id token, or `None` if not in a supported CI."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return Provider.GITHUB
    return None


def _get_github_oidc_token(audience: str) -> str:
    """Mint an OIDC id token from the GitHub Actions runtime.

    Relies on the `ACTIONS_ID_TOKEN_REQUEST_URL` / `ACTIONS_ID_TOKEN_REQUEST_TOKEN` env vars,
    which GitHub only injects when the job declares `permissions: id-token: write`.
    """
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not request_url or not request_token:
        raise OIDCError(
            "Cannot request an OIDC id token from GitHub Actions. Make sure the workflow job sets "
            "`permissions: id-token: write`. See "
            "https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect"
        )
    response = get_session().get(
        request_url,
        params={"audience": audience},
        headers={"Authorization": f"Bearer {request_token}"},
    )
    hf_raise_for_status(response)
    return response.json()["value"]


def get_oidc_token(*, provider: Provider | str | None = None, audience: str | None = None) -> str:
    """Mint a raw OIDC id token (JWT) from the current CI provider.

    Args:
        provider (`str`, *optional*):
            CI provider to use. Auto-detected from the environment when omitted.
        audience (`str`, *optional*):
            The `aud` claim to request. Defaults to `constants.ENDPOINT` so it matches the endpoint
            that validates it (respects `HF_ENDPOINT`/staging).

    Returns:
        `str`: The raw id token (JWT) to pass to [`exchange_oidc_token`].
    """
    audience = audience or constants.ENDPOINT
    provider = provider or detect_provider()
    supported = ", ".join(p.value for p in Provider)
    if provider is None:
        raise OIDCError(f"No supported CI OIDC provider detected. Trusted Publishers currently supports: {supported}.")
    if provider == Provider.GITHUB:
        return _get_github_oidc_token(audience)
    raise NotImplementedError(f"OIDC provider '{provider}' is not supported yet. Supported: {supported}.")


def exchange_oidc_token(*, subject_token: str, resource: str, endpoint: str | None = None) -> dict:
    """Exchange a CI OIDC id token for a short-lived Hugging Face token (RFC 8693).

    Args:
        subject_token (`str`):
            The raw OIDC id token (JWT) from the CI provider. Its `aud` claim must be the Hub URL.
        resource (`str`):
            What to scope the token to: a Hub repo (`namespace/name`, `datasets/namespace/name`,
            `spaces/namespace/name`, `kernels/namespace/name`) for a write token, or a bare Hub
            username for a read-only `gated-repos` token.
        endpoint (`str`, *optional*):
            Hub endpoint. Defaults to `constants.ENDPOINT` (respects `HF_ENDPOINT`/staging).

    Returns:
        `dict`: The token-exchange response, e.g.
        `{"access_token": "hf_jwt_…", "token_type": "bearer", "expires_in": 3600, ...}`.
    """
    response = get_session().post(
        f"{endpoint or constants.ENDPOINT}/oauth/token",
        json={
            "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
            "subject_token_type": _ID_TOKEN_TYPE,
            "subject_token": subject_token,
            "resource": resource,
        },
    )
    hf_raise_for_status(response)
    return response.json()


def oidc_login(
    *,
    resource: str,
    subject_token: str | None = None,
    provider: Provider | str | None = None,
    audience: str | None = None,
    endpoint: str | None = None,
) -> dict:
    """Mint a CI OIDC id token and exchange it for a Hugging Face token.

    Convenience wrapper around [`get_oidc_token`] + [`exchange_oidc_token`]. Returns the raw
    exchange response (it does not persist anything — the caller decides what to do with the token).

    Args:
        resource (`str`):
            Repo or username to scope the token to. See [`exchange_oidc_token`].
        subject_token (`str`, *optional*):
            A pre-minted OIDC id token to exchange directly. Use this for CI providers not yet
            supported natively (e.g. GitLab): mint the id token in your job and pass it here. When
            omitted, the token is minted from the detected `provider`.
        provider (`str`, *optional*):
            CI provider. Auto-detected when omitted. Ignored when `subject_token` is provided.
        audience (`str`, *optional*):
            The `aud` claim to request. Defaults to the resolved `endpoint`, so it matches the
            endpoint that validates it.
        endpoint (`str`, *optional*):
            Hub endpoint. Defaults to `constants.ENDPOINT`.

    Returns:
        `dict`: The token-exchange response (`access_token`, `token_type`, `expires_in`, ...).
    """
    endpoint = endpoint or constants.ENDPOINT
    if subject_token is None:
        subject_token = get_oidc_token(provider=provider, audience=audience or endpoint)
    return exchange_oidc_token(subject_token=subject_token, resource=resource, endpoint=endpoint)
<|endoftext|>"""Contains all custom errors."""

from enum import Enum
from pathlib import Path

from httpx2 import HTTPError, Response


# CACHE ERRORS


class CacheNotFound(Exception):
    """Exception thrown when the Huggingface cache is not found."""

    cache_dir: str | Path

    def __init__(self, msg: str, cache_dir: str | Path, *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.cache_dir = cache_dir


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the Huggingface cache-system."""


class CachedRepoTreeNotFoundError(Exception):
    """Raised by [`get_cached_repo_tree`] when no tree listing is cached for the requested revision.

    The tree listing is populated as a side effect of [`snapshot_download`].
    """


# HEADERS ERRORS


class LocalTokenNotFoundError(EnvironmentError):
    """Raised if local token is required but not found."""


# OIDC ERRORS


class OIDCError(Exception):
    """Raised when keyless CI/CD auth via OIDC token exchange ("Trusted Publishers") cannot proceed.

    Typically because `HF_OIDC_RESOURCE` is set but no id token is available: not running in a
    supported CI provider and `HF_OIDC_ID_TOKEN` is unset.

    See https://huggingface.co/docs/hub/trusted-publishers.
    """


# DEVICE CODE OAUTH ERRORS


class OAuthErrorCode(str, Enum):
    """Known OAuth `error` codes returned by the Hub's token endpoint (RFC 6749 / RFC 8628)."""

    AUTHORIZATION_PENDING = "authorization_pending"
    SLOW_DOWN = "slow_down"
    EXPIRED_TOKEN = "expired_token"
    ACCESS_DENIED = "access_denied"
    INVALID_GRANT = "invalid_grant"


class DeviceCodeError(Exception):
    """Raised when the Device Code OAuth login flow (RFC 8628) or an OAuth token refresh fails.

    Covers failures at any step: requesting the device code, polling for the token,
    authorization denied/expired, or unexpected server responses.

    Attributes:
        error_code (`str`, *optional*):
            The OAuth `error` code returned by the server, if any. Known values are listed in
            [`OAuthErrorCode`] but the server may return other codes.
    """

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


# HTTP ERRORS


class OfflineModeIsEnabled(ConnectionError):
    """Raised when a request is made but `HF_HUB_OFFLINE=1` is set as environment variable."""


class HfHubHTTPError(HTTPError, OSError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in HF Hub.

    Any HTTPError is converted at least into a `HfHubHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request ID sourced from headers in order of precedence: "X-Request-Id", "X-Amzn-Trace-Id", "X-Amz-Cf-Id".
    - Server error message from the header "X-Error-Message".
    - Server error message if we can found one in the response body.

    Example:
    ```py
        import httpx2
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("\n`create_commit` expects the repository to exist.")
            raise
    ```
    """

    def __init__(
        self,
        message: str,
        *,
        response: Response,
        server_message: str | None = None,
    ):
        self.request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("X-Amzn-Trace-Id")
            or response.headers.get("x-amz-cf-id")
        )
        self.server_message = server_message
        self.response = response
        self.request = response.request
        super().__init__(message)

    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]

    @classmethod
    def _reconstruct_hf_hub_http_error(
        cls, message: str, response: Response, server_message: str | None
    ) -> "HfHubHTTPError":
        return cls(message, response=response, server_message=server_message)

    def __reduce_ex__(self, protocol):
        """Fix pickling of Exception subclass with kwargs. We need to override __reduce_ex__ of the parent class"""
        return (self.__class__._reconstruct_hf_hub_http_error, (str(self), self.response, self.server_message))


# INFERENCE CLIENT ERRORS


class InferenceTimeoutError(HTTPError, TimeoutError):
    """Error raised when a model is unavailable or the request times out."""


# INFERENCE ENDPOINT ERRORS


class InferenceEndpointError(Exception):
    """Generic exception when dealing with Inference Endpoints."""


class InferenceEndpointTimeoutError(InferenceEndpointError, TimeoutError):
    """Exception for timeouts while waiting for Inference Endpoint."""


# SAFETENSORS ERRORS


class SafetensorsParsingError(Exception):
    """Raised when failing to parse a safetensors file metadata.

    This can be the case if the file is not a safetensors file or does not respect the specification.
    """


class NotASafetensorsRepoError(Exception):
    """Raised when a repo is not a Safetensors repo i.e. doesn't have either a `model.safetensors` or a
    `model.safetensors.index.json` file.
    """


# TEXT GENERATION ERRORS


class TextGenerationError(HTTPError):
    """Generic error raised if text-generation went wrong."""


# Text Generation Inference Errors
class ValidationError(TextGenerationError):
    """Server-side validation error."""


class GenerationError(TextGenerationError):
    pass


class OverloadedError(TextGenerationError):
    pass


class IncompleteGenerationError(TextGenerationError):
    pass


class UnknownError(TextGenerationError):
    pass


# VALIDATION ERRORS


class HFValidationError(ValueError):
    """Generic exception thrown by `huggingface_hub` validators.

    Inherits from [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError).
    """


class HfUriError(ValueError):
    """Raised when an `hf://...` URI is malformed.

    See [`parse_hf_uri`] and the
    [HF URIs reference](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_uris)
    for the canonical syntax.

    Inherits from [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError).
    """

    def __init__(self, uri: str, msg: str):
        self.uri = uri
        self.msg = msg
        full_msg = f"Invalid HF URI '{uri}'. {msg}" if uri else f"Invalid HF URI. {msg}"
        super().__init__(full_msg)


# FILE METADATA ERRORS


class DryRunError(OSError):
    """Error triggered when a dry run is requested but cannot be performed (e.g. invalid repo)."""


class FileMetadataError(OSError):
    """Error triggered when the metadata of a file on the Hub cannot be retrieved (missing ETag or commit_hash).

    Inherits from `OSError` for backward compatibility.
    """


# BUCKET ERRORS


class BucketNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a bucket that does not exist.

    Attributes:
        bucket_id (`str` or `None`):
            The bucket id (namespace/name) that was not found, if it could be determined from the request URL.

    Example:

    ```py
    >>> from huggingface_hub import bucket_info
    >>> bucket_info("<non_existent_bucket>")
    (...)
    huggingface_hub.errors.BucketNotFoundError: 404 Client Error. (Request ID: XXX)

    Bucket Not Found for url: https://huggingface.co/api/buckets/namespace/name.
    Please make sure you specified the correct bucket id (namespace/name).
    If the bucket is private, make sure you are authenticated and your token has the required permissions.
    ```
    """

    bucket_id: str | None = None


class BucketBatchError(HfHubHTTPError):
    """
    Raised when the server reports that some operations of a bucket batch failed.

    Attributes:
        failures (`list[dict[str, str]]`):
            The failed operations as listed by the server, e.g. `[{"path": "config.json", "error": "..."}]`.
    """

    failures: list[dict[str, str]] = []


# JOB ERRORS


class JobNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a Job that does not exist.

    Attributes:
        job_id (`str`):
            The job id that was not found.
    """

    job_id: str


# REPOSITORY ERRORS


class RepositoryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.

    Attributes:
        repo_id (`str` or `None`):
            The repo id that was not found, if it could be determined from the request URL.
        repo_type (`str` or `None`):
            The repo type ("model", "dataset", or "space"), if it could be determined from the request URL.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existent_repository>")
    (...)
    huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. (Request ID: PvMw_VjBMjVdMz53WKIzP)

    Repository Not Found for url: https://huggingface.co/api/models/%3Cnon_existent_repository%3E.
    Please make sure you specified the correct `repo_id` and `repo_type`.
    If the repo is private, make sure you are authenticated and your token has the required permissions.
    Invalid username or password.
    ```
    """

    repo_id: str | None = None
    repo_type: str | None = None


class GatedRepoError(RepositoryNotFoundError):
    """
    Raised when trying to access a gated repository for which the user is not on the
    authorized list.

    Note: derives from `RepositoryNotFoundError` to ensure backward compatibility.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<gated_repository>")
    (...)
    huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: ViT1Bf7O_026LGSQuVqfa)

    Cannot access gated repo for url https://huggingface.co/api/models/ardent-figment/gated-model.
    Access to model ardent-figment/gated-model is restricted and you are not in the authorized list.
    Visit https://huggingface.co/ardent-figment/gated-model to ask for access.
    ```
    """


class DisabledRepoError(HfHubHTTPError):
    """
    Raised when trying to access a repository that has been disabled by its author.

    Example:

    ```py
    >>> from huggingface_hub import dataset_info
    >>> dataset_info("laion/laion-art")
    (...)
    huggingface_hub.errors.DisabledRepoError: 403 Client Error. (Request ID: Root=1-659fc3fa-3031673e0f92c71a2260dbe2;bc6f4dfb-b30a-4862-af0a-5cfe827610d8)

    Cannot access repository for url https://huggingface.co/api/datasets/laion/laion-art.
    Access to this resource is disabled.
    ```
    """


# REVISION ERROR


class RevisionNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository but an invalid
    revision.

    Attributes:
        repo_id (`str` or `None`):
            The repo id, if it could be determined from the request URL.
        repo_type (`str` or `None`):
            The repo type ("model", "dataset", or "space"), if it could be determined from the request URL.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', 'config.json', revision='<non-existent-revision>')
    (...)
    huggingface_hub.errors.RevisionNotFoundError: 404 Client Error. (Request ID: Mwhe_c3Kt650GcdKEFomX)

    Revision Not Found for url: https://huggingface.co/bert-base-cased/resolve/%3Cnon-existent-revision%3E/config.json.
    ```
    """

    repo_id: str | None = None
    repo_type: str | None = None


class RevisionResolutionError(Exception):
    """
    Raised by [`HfApi.resolve_revision`] when a revision cannot be resolved to a commit hash: the Hub could not be
    reached (offline mode, connection error, timeout, Hub downtime, ...) and no matching entry was found in the
    local cache.
    """


# ENTRY ERRORS
class EntryNotFoundError(Exception):
    """
    Raised when entry not found, either locally or remotely.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-existent-file>')
    (...)
    huggingface_hub.errors.RemoteEntryNotFoundError (...)
    >>> hf_hub_download('bert-base-cased', '<non-existent-file>', local_files_only=True)
    (...)
    huggingface_hub.utils.errors.LocalEntryNotFoundError (...)
    ```
    """


class RemoteEntryNotFoundError(HfHubHTTPError, EntryNotFoundError):
    """
    Raised when trying to access a hf.co URL with a valid repository and revision
    but an invalid filename.

    Attributes:
        repo_id (`str` or `None`):
            The repo id, if it could be determined from the request URL.
        repo_type (`str` or `None`):
            The repo type ("model", "dataset", or "space"), if it could be determined from the request URL.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-existent-file>')
    (...)
    huggingface_hub.errors.EntryNotFoundError: 404 Client Error. (Request ID: 53pNl6M0MxsnG5Sw8JA6x)

    Entry Not Found for url: https://huggingface.co/bert-base-cased/resolve/main/%3Cnon-existent-file%3E.
    ```
    """

    repo_id: str | None = None
    repo_type: str | None = None


class LocalEntryNotFoundError(FileNotFoundError, EntryNotFoundError):
    """
    Raised when trying to access a file or snapshot that is not on the disk when network is
    disabled or unavailable (connection issue). The entry may exist on the Hub.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-cached-file>',  local_files_only=True)
    (...)
    huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.
    ```
    """

    def __init__(self, message: str):
        super().__init__(message)


class IncompleteSnapshotError(LocalEntryNotFoundError):
    """
    Raised by [`snapshot_download`] when the Hub cannot be reached (offline, connection issue, or
    `local_files_only=True`) and the cached snapshot is known to be incomplete: some files listed in
    the repository's cached tree listing are missing from the local snapshot.

    This is a subclass of [`LocalEntryNotFoundError`] for backward compatibility.

    The `snapshot_path` attribute holds the path to the incomplete local snapshot, so a downstream library can locate
    the latest cached files even though they are known to be incomplete.
    """

    def __init__(self, message: str, snapshot_path: str):
        super().__init__(message)
        self.snapshot_path = snapshot_path


# REQUEST ERROR
class BadRequestError(HfHubHTTPError, ValueError):
    """
    Raised by `hf_raise_for_status` when the server returns a HTTP 400 error.

    Example:

    ```py
    >>> resp = httpx2.post("hf.co/api/check", ...)
    >>> hf_raise_for_status(resp, endpoint_name="check")
    huggingface_hub.errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
    ```
    """


# DDUF file format ERROR


class DDUFError(Exception):
    """Base exception for errors related to the DDUF format."""


class DDUFCorruptedFileError(DDUFError):
    """Exception thrown when the DDUF file is corrupted."""


class DDUFExportError(DDUFError):
    """Base exception for errors during DDUF export."""


class DDUFInvalidEntryNameError(DDUFExportError):
    """Exception thrown when the entry name is invalid."""


# STRICT DATACLASSES ERRORS


class StrictDataclassError(Exception):
    """Base exception for strict dataclasses."""


class StrictDataclassDefinitionError(StrictDataclassError):
    """Exception thrown when a strict dataclass is defined incorrectly."""


class StrictDataclassFieldValidationError(StrictDataclassError):
    """Exception thrown when a strict dataclass fails validation for a given field."""

    def __init__(self, field: str, cause: Exception):
        error_message = f"Validation error for field '{field}':"
        error_message += f"\n    {cause.__class__.__name__}: {cause}"
        super().__init__(error_message)


class StrictDataclassClassValidationError(StrictDataclassError):
    """Exception thrown when a strict dataclass fails validation on a class validator."""

    def __init__(self, validator: str, cause: Exception):
        error_message = f"Class validation error for validator '{validator}':"
        error_message += f"\n    {cause.__class__.__name__}: {cause}"
        super().__init__(error_message)


# XET ERRORS


class XetDownloadError(Exception):
    """Exception thrown when the download from Xet Storage fails."""


# LFS ERRORS


class FileDuplicationError(Exception):
    """Raised when duplicating files across repos fails."""


# CLI ERRORS


class CLIError(Exception):
    """CLI error with clean message (no traceback by default)."""


class ConfirmationError(CLIError):
    """Raised when a confirmation prompt is declined (non-interactive mode)."""


class CLIExtensionInstallError(CLIError):
    """Error during CLI extension installation."""


# SANDBOX ERRORS


class SandboxError(Exception):
    """Base exception for sandbox operations (see `huggingface_hub.Sandbox`).

    Attributes:
        status_code: The HTTP status returned by the in-sandbox server, if the error
            originated from an API response (e.g. `404` for a missing file). `None` otherwise.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SandboxCommandError(SandboxError):
    """Raised when a command run in a sandbox exits with a non-zero code.

    Attributes:
        cmd: The command that failed.
        result: The full `SandboxCommandResult` (exit_code, stdout, stderr, ...).
    """

    def __init__(self, cmd, result) -> None:
        self.cmd = cmd
        self.result = result
        stderr_tail = result.stderr[-1000:] if result.stderr else "<empty>"
        if result.timed_out:
            reason = "timed out"
        elif result.signal is not None:
            reason = f"was killed by signal {result.signal}"
        else:
            reason = f"exited with code {result.exit_code}"
        super().__init__(f"Command {cmd!r} {reason}. stderr:\n{stderr_tail}")
<|endoftext|># Copyright 2026-present, the HuggingFace Inc. team.
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
"""Shared logic for bucket operations.

This module contains the core buckets logic used by both the CLI and the Python API.
"""

import fnmatch
import json
import mimetypes
import os
import stat
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from . import constants, logging
from ._local_folder import _validate_relative_filename
from .errors import BucketNotFoundError
from .utils import (
    HfUri,
    StatusLine,
    XetFileData,
    disable_progress_bars,
    enable_progress_bars,
    parse_datetime,
    parse_hf_uri,
)
from .utils._hf_uris import _looks_like_hf_url
from .utils._paths import _normalize_separators


if TYPE_CHECKING:
    from .hf_api import HfApi


logger = logging.get_logger(__name__)


BUCKET_PREFIX = "hf://buckets/"
_SYNC_TIME_WINDOW_MS = 1000  # 1s safety-window for file modification time comparisons


# =============================================================================
# Bucket data structures
# =============================================================================


@dataclass
class BucketInfo:
    """
    Contains information about a bucket on the Hub. This object is returned by [`bucket_info`] and [`list_buckets`].

    Attributes:
        id (`str`):
            ID of the bucket.
        private (`bool`):
            Is the bucket private.
        created_at (`datetime`):
            Date of creation of the bucket on the Hub.
        size (`int`):
            Size of the bucket in bytes.
        total_files (`int`):
            Total number of files in the bucket.
    """

    id: str
    private: bool
    created_at: datetime
    size: int
    total_files: int

    def __init__(self, **kwargs):
        self.id = kwargs.pop("id")
        self.private = kwargs.pop("private")
        self.created_at = parse_datetime(kwargs.pop("createdAt"))
        self.size = kwargs.pop("size")
        self.total_files = kwargs.pop("totalFiles")
        self.__dict__.update(**kwargs)


@dataclass
class _BucketAddFile:
    source: str | Path | bytes
    destination: str

    xet_hash: str | None = field(default=None)
    size: int | None = field(default=None)
    mtime: int = field(init=False)
    content_type: str | None = field(init=False)

    def __post_init__(self) -> None:
        self.content_type = None
        if isinstance(self.source, (str, Path)):  # guess content type from source path
            self.content_type = mimetypes.guess_type(self.source)[0]
        if self.content_type is None:  # or default to destination path content type
            self.content_type = mimetypes.guess_type(self.destination)[0]

        self.mtime = int(
            os.path.getmtime(self.source) * 1000 if not isinstance(self.source, bytes) else time.time() * 1000
        )


@dataclass
class _BucketCopyFile:
    destination: str
    xet_hash: str
    source_repo_type: str  # "model", "dataset", "space", "bucket"
    source_repo_id: str
    size: int | None = field(default=None)
    mtime: int = field(init=False)
    content_type: str | None = field(init=False)

    def __post_init__(self) -> None:
        self.content_type = mimetypes.guess_type(self.destination)[0]
        self.mtime = int(time.time() * 1000)


@dataclass
class _BucketDeleteFile:
    path: str


@dataclass(frozen=True)
class BucketFileMetadata:
    """Data structure containing information about a file in a bucket.

    Returned by [`get_bucket_file_metadata`].

    Args:
        size (`int`):
            Size of the file in bytes.
        xet_file_data (`XetFileData`):
            Xet information for the file (hash and refresh route).
    """

    size: int
    xet_file_data: XetFileData


@dataclass
class BucketUrl:
    """Describes a bucket URL on the Hub.

    `BucketUrl` is returned by [`create_bucket`]. At initialization, the URL is parsed to populate properties:
    - endpoint (`str`)
    - namespace (`str`)
    - bucket_id (`str`)
    - url (`str`)
    - uri (`HfUri`)

    Args:
        url (`str`):
            String value of the bucket url.
        endpoint (`str`, *optional*):
            Endpoint of the Hub. Defaults to <https://huggingface.co>.
    """

    url: str
    endpoint: str = ""
    namespace: str = field(init=False)
    bucket_id: str = field(init=False)
    uri: HfUri = field(init=False)

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint or constants.ENDPOINT

        # Parse URL: expected format is `{endpoint}/buckets/{namespace}/{bucket_name}`
        url_path = self.url.replace(self.endpoint, "").strip("/")
        # Remove leading "buckets/" prefix
        if url_path.startswith("buckets/"):
            url_path = url_path[len("buckets/") :]
        parsed = _parse_bucket_uri(url_path)
        if parsed.path_in_repo:
            raise ValueError(f"Unable to parse bucket URL: {self.url}")
        self.namespace = parsed.id.split("/")[0]
        self.bucket_id = parsed.id
        self.uri = parsed


@dataclass
class BucketFile:
    """
    Contains information about a file in a bucket on the Hub. This object is returned by [`list_bucket_tree`].

    Similar to [`RepoFile`] but for files in buckets.
    """

    type: Literal["file"]
    path: str
    size: int
    xet_hash: str
    mtime: datetime | None
    uploaded_at: datetime | None

    def __init__(self, **kwargs):
        self.type = kwargs.pop("type")
        self.path = kwargs.pop("path")
        self.size = kwargs.pop("size")
        self.xet_hash = kwargs.pop("xetHash")
        mtime = kwargs.pop("mtime", None)
        self.mtime = parse_datetime(mtime) if mtime else None
        uploaded_at = kwargs.pop("uploadedAt", None)
        self.uploaded_at = parse_datetime(uploaded_at) if uploaded_at else None


@dataclass
class BucketFolder:
    """
    Contains information about a directory in a bucket on the Hub. This object is returned by [`list_bucket_tree`].

    Similar to [`RepoFolder`] but for directories in buckets.
    """

    type: Literal["directory"]
    path: str
    uploaded_at: datetime | None

    def __init__(self, **kwargs):
        self.type = kwargs.pop("type")
        self.path = kwargs.pop("path")
        uploaded_at = kwargs.pop("uploadedAt", None) or kwargs.pop("uploaded_at", None)
        self.uploaded_at = (
            (uploaded_at if isinstance(uploaded_at, datetime) else parse_datetime(uploaded_at))
            if uploaded_at
            else None
        )


# =============================================================================
# Bucket path parsing
# =============================================================================


def _parse_bucket_uri(path: str) -> HfUri:
    """Parse a bucket path into a HfUri.

    Accepts:
    - `hf://buckets/namespace/name(/path/in/repo)` URIs,
    - Hugging Face web URLs such as `https://huggingface.co/buckets/namespace/name(/tree/path)`,
    - plain `namespace/name(/path/in/repo)` paths.
    """
    if path.startswith(constants.HF_PROTOCOL) or _looks_like_hf_url(path):
        # Don't use 'if is_hf_uri(...)' here as we prefer 'parse_hf_uri(...)' to raise the exact error message.
        parsed = parse_hf_uri(path)
        if not parsed.is_bucket:
            raise ValueError(f"Invalid bucket path: {path}. Must be a bucket URI (hf://buckets/...).")
        return parsed
    parts = path.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid bucket path: '{path}'. Expected format: namespace/bucket_name")
    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = "/".join(parts[2:])
    return HfUri(type="bucket", id=bucket_id, path_in_repo=prefix)


def _is_bucket_path(path: str) -> bool:
    """Check if a path is a bucket path.

    Do not raise if the path is not a hf:// URI.
    Raise if the path is a hf:// URI but with an incorrect format.
    """
    if not path.startswith(constants.HF_PROTOCOL):
        return False
    return parse_hf_uri(path).is_bucket


# =============================================================================
# Sync data structures
# =============================================================================


@dataclass
class SyncOperation:
    """Represents a sync operation to be performed."""

    action: Literal["upload", "download", "delete", "skip"]
    path: str
    size: int | None = None
    reason: str = ""
    local_mtime: str | None = None
    remote_mtime: str | None = None
    bucket_file: BucketFile | None = None  # BucketFile when available (not serialized to plan file)


@dataclass
class SyncPlan:
    """Represents a complete sync plan."""

    source: str
    dest: str
    timestamp: str
    operations: list[SyncOperation] = field(default_factory=list)

    def summary(self) -> dict[str, int | str]:
        uploads = sum(1 for op in self.operations if op.action == "upload")
        downloads = sum(1 for op in self.operations if op.action == "download")
        deletes = sum(1 for op in self.operations if op.action == "delete")
        skips = sum(1 for op in self.operations if op.action == "skip")
        total_size = sum(op.size or 0 for op in self.operations if op.action in ("upload", "download"))
        return {
            "uploads": uploads,
            "downloads": downloads,
            "deletes": deletes,
            "skips": skips,
            "total_size": total_size,
        }


# =============================================================================
# Filter matching
# =============================================================================


class FilterMatcher:
    """Matches file paths against include/exclude patterns.

    Matching is case-sensitive on every platform, like bucket paths. Backslashes in patterns are treated as path
    separators, since bucket paths and local relative paths always use `/`.
    """

    def __init__(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        filter_rules: list[tuple[str, str]] | None = None,
    ):
        """Initialize the filter matcher.

        Args:
            include_patterns: Patterns to include (from --include)
            exclude_patterns: Patterns to exclude (from --exclude)
            filter_rules: Rules from filter file as list of ("+"/"-", pattern) tuples
        """
        self.include_patterns = [_normalize_separators(p) for p in include_patterns or []]
        self.exclude_patterns = [_normalize_separators(p) for p in exclude_patterns or []]
        self.filter_rules = [(sign, _normalize_separators(p)) for sign, p in filter_rules or []]

    def matches(self, path: str) -> bool:
        """Check if a path should be included based on the filter rules.

        Filtering rules:
        - Filters are evaluated in order, first matching rule decides
        - If no rules match, include by default (unless include patterns are specified)
        """
        # First check filter rules from file (in order)
        for sign, pattern in self.filter_rules:
            if fnmatch.fnmatchcase(path, pattern):
                return sign == "+"

        # Then check CLI patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatchcase(path, pattern):
                return False

        for pattern in self.include_patterns:
            if fnmatch.fnmatchcase(path, pattern):
                return True

        # If include patterns were specified but none matched, exclude
        if self.include_patterns:
            return False

        # Default: include
        return True


def _parse_filter_file(filter_file: str) -> list[tuple[str, str]]:
    """Parse a filter file and return a list of (sign, pattern) tuples.

    Filter file format:
    - Lines starting with "+" are include patterns
    - Lines starting with "-" are exclude patterns
    - Empty lines and lines starting with "#" are ignored
    """
    rules = []
    with open(filter_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("+"):
                rules.append(("+", line[1:].strip()))
            elif line.startswith("-"):
                rules.append(("-", line[1:].strip()))
            else:
                # Default to include if no prefix
                rules.append(("+", line))
    return rules


# =============================================================================
# File listing
# =============================================================================


def _stat_local(path: str) -> tuple[int, float] | None:
    """Stat a local file and return (size, mtime_ms).

    Returns None if the path is missing or is a directory. Uses a single
    ``os.stat`` call so callers don't pay for multiple syscalls per file.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if stat.S_ISDIR(st.st_mode):
        return None
    return st.st_size, st.st_mtime * 1000


def _list_local_files(local_path: str) -> Iterator[tuple[str, int, float]]:
    """List all files in a local directory.

    Yields:
        tuple: (relative_path, size, mtime_ms) for each file
    """
    local_path = os.path.abspath(local_path)
    if not os.path.isdir(local_path):
        raise ValueError(f"Local path must be a directory: {local_path}")

    for root, _, files in os.walk(local_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            stat_info = _stat_local(full_path)
            if stat_info is None:
                continue
            rel_path = os.path.relpath(full_path, local_path)
            # Normalize to forward slashes for consistency
            rel_path = rel_path.replace(os.sep, "/")
            yield rel_path, stat_info[0], stat_info[1]


def _list_remote_files(api: "HfApi", bucket_id: str, prefix: str) -> Iterator[tuple[str, int, float, Any]]:
    """List all files in a bucket with a given prefix.

    Yields:
        tuple: (relative_path, size, mtime_ms, bucket_file) for each file.
            bucket_file is the BucketFile object from list_bucket_tree.
    """
    for item in api.list_bucket_tree(bucket_id, prefix=prefix or None, recursive=True):
        if isinstance(item, BucketFolder):
            continue
        path = item.path
        # Remove prefix from path to get relative path
        # Only strip prefix if it's followed by "/" (directory boundary) or is exact match
        if prefix:
            if path.startswith(prefix + "/"):
                rel_path = path[len(prefix) + 1 :]
            elif path == prefix:
                # Exact match: the file IS the prefix (e.g., single file download)
                rel_path = path.rsplit("/", 1)[-1] if "/" in path else path
            else:
                # Path doesn't match prefix pattern (e.g., "submarine.txt" for prefix "sub")
                # Skip this file - it was returned by the API but doesn't belong to this prefix
                continue
        else:
            rel_path = path
        # Reject server keys that would escape the local dir when joined onto it (see PR #4540).
        _validate_relative_filename(rel_path)
        mtime_ms = item.mtime.timestamp() * 1000 if item.mtime else 0
        yield rel_path, item.size, mtime_ms, item


# =============================================================================
# Sync plan computation
# =============================================================================


def _mtime_to_iso(mtime_ms: float) -> str:
    """Convert mtime in milliseconds to ISO format string."""
    return datetime.fromtimestamp(mtime_ms / 1000, tz=timezone.utc).isoformat()


def _compare_files_for_sync(
    *,
    path: str,
    action: Literal["upload", "download"],
    source_size: int,
    source_mtime: float,
    dest_size: int,
    dest_mtime: float,
    source_newer_label: str,
    dest_newer_label: str,
    ignore_sizes: bool,
    ignore_times: bool,
    ignore_existing: bool,
    bucket_file: Any | None = None,
) -> SyncOperation:
    """Compare source and dest files and return the appropriate sync operation.

    This is a unified helper for both upload and download directions.

    Args:
        path: Relative file path
        action: "upload" or "download"
        source_size: Size of the source file (bytes)
        source_mtime: Mtime of the source file (milliseconds)
        dest_size: Size of the destination file (bytes)
        dest_mtime: Mtime of the destination file (milliseconds)
        source_newer_label: Label when source is newer (e.g., "local newer" or "remote newer")
        dest_newer_label: Label when dest is newer (e.g., "remote newer" or "local newer")
        ignore_sizes: Only compare mtime
        ignore_times: Only compare size
        ignore_existing: Skip files that exist on receiver
        bucket_file: BucketFile object (for downloads only)

    Returns:
        SyncOperation describing the action to take
    """
    local_mtime_iso = _mtime_to_iso(source_mtime if action == "upload" else dest_mtime)
    remote_mtime_iso = _mtime_to_iso(dest_mtime if action == "upload" else source_mtime)

    base_kwargs: dict[str, Any] = {
        "path": path,
        "size": source_size,
        "local_mtime": local_mtime_iso,
        "remote_mtime": remote_mtime_iso,
    }

    if ignore_existing:
        return SyncOperation(action="skip", reason="exists on receiver (--ignore-existing)", **base_kwargs)

    size_differs = source_size != dest_size
    source_newer = (source_mtime - dest_mtime) > _SYNC_TIME_WINDOW_MS

    if ignore_sizes:
        if source_newer:
            return SyncOperation(action=action, reason=source_newer_label, bucket_file=bucket_file, **base_kwargs)
        else:
            dest_newer = (dest_mtime - source_mtime) > _SYNC_TIME_WINDOW_MS
            skip_reason = dest_newer_label if dest_newer else "same mtime"
