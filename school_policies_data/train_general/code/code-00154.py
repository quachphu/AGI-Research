        _raise_bad_format(format_tag)

    # move file pointer to next chunk
    if size > bytes_read:
        fid.read(size - bytes_read)

    # fmt should always be 16, 18 or 40, but handle it just in case
    _handle_pad_byte(fid, size)

    if format_tag == WAVE_FORMAT.PCM:
        if bytes_per_second != fs * block_align:
            raise ValueError("WAV header is invalid: nAvgBytesPerSec must"
                             " equal product of nSamplesPerSec and"
                             " nBlockAlign, but file has nSamplesPerSec ="
                             f" {fs}, nBlockAlign = {block_align}, and"
                             f" nAvgBytesPerSec = {bytes_per_second}")

    return (size, format_tag, channels, fs, bytes_per_second, block_align,
            bit_depth)


def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian, is_rf64,
                     block_align, mmap=False):
    """
    Notes
    -----
    Assumes file pointer is immediately after the 'data' id

    It's possible to not use all available bits in a container, or to store
    samples in a container bigger than necessary, so bytes_per_sample uses
    the actual reported container size (nBlockAlign / nChannels).  Real-world
    examples:

    Adobe Audition's "24-bit packed int (type 1, 20-bit)"

        nChannels = 2, nBlockAlign = 6, wBitsPerSample = 20

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/AFsp/M1F1-int12-AFsp.wav
    is:

        nChannels = 2, nBlockAlign = 4, wBitsPerSample = 12

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/multichaudP.pdf
    gives an example of:

        nChannels = 2, nBlockAlign = 8, wBitsPerSample = 20
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    # Size of the data subchunk in bytes
    if not is_rf64:
        size = struct.unpack(fmt+'I', fid.read(4))[0]
    else:
        pos = fid.tell()
        # chunk size is stored in global file header for RF64
        fid.seek(28)
        size = struct.unpack('<Q', fid.read(8))[0]
        fid.seek(pos)
        # skip data chunk size as it is 0xFFFFFFF
        fid.read(4)

    # Number of bytes per sample (sample container size)
    bytes_per_sample = block_align // channels
    n_samples = size // bytes_per_sample

    if format_tag == WAVE_FORMAT.PCM:
        if 1 <= bit_depth <= 8:
            dtype = 'u1'  # WAV of 8-bit integer or less are unsigned
        elif bytes_per_sample in {3, 5, 6, 7}:
            # No compatible dtype.  Load as raw bytes for reshaping later.
            dtype = 'V1'
        elif bit_depth <= 64:
            # Remaining bit depths can map directly to signed numpy dtypes
            dtype = f'{fmt}i{bytes_per_sample}'
        else:
            raise ValueError("Unsupported bit depth: the WAV file "
                             f"has {bit_depth}-bit integer data.")
    elif format_tag == WAVE_FORMAT.IEEE_FLOAT:
        if bit_depth in {32, 64}:
            dtype = f'{fmt}f{bytes_per_sample}'
        else:
            raise ValueError("Unsupported bit depth: the WAV file "
                             f"has {bit_depth}-bit floating-point data.")
    else:
        _raise_bad_format(format_tag)

    start = fid.tell()
    if not mmap:
        try:
            count = size if dtype == 'V1' else n_samples
            data = np.fromfile(fid, dtype=dtype, count=count)
        except io.UnsupportedOperation:  # not a C-like file
            fid.seek(start, 0)  # just in case it seeked, though it shouldn't
            data = np.frombuffer(fid.read(size), dtype=dtype)

        if dtype == 'V1':
            # Rearrange raw bytes into smallest compatible numpy dtype
            dt = f'{fmt}i4' if bytes_per_sample == 3 else f'{fmt}i8'
            a = np.zeros((len(data) // bytes_per_sample, np.dtype(dt).itemsize),
                            dtype='V1')
            if is_big_endian:
                a[:, :bytes_per_sample] = data.reshape((-1, bytes_per_sample))
            else:
                a[:, -bytes_per_sample:] = data.reshape((-1, bytes_per_sample))
            data = a.view(dt).reshape(a.shape[:-1])
    else:
        if bytes_per_sample in {1, 2, 4, 8}:
            start = fid.tell()
            data = np.memmap(fid, dtype=dtype, mode='c', offset=start,
                                shape=(n_samples,))
            fid.seek(start + size)
        else:
            raise ValueError("mmap=True not compatible with "
                             f"{bytes_per_sample}-byte container size.")

    _handle_pad_byte(fid, size)

    if channels > 1:
        data = data.reshape(-1, channels)
    return data


def _skip_unknown_chunk(fid, is_big_endian):
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'

    data = fid.read(4)
    # call unpack() and seek() only if we have really read data from file
    # otherwise empty read at the end of the file would trigger
    # unnecessary exception at unpack() call
    # in case data equals somehow to 0, there is no need for seek() anyway
    if data:
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)
        _handle_pad_byte(fid, size)


def _read_riff_chunk(fid):
    str1 = fid.read(4)  # File signature
    if str1 == b'RIFF':
        is_rf64 = False
        is_big_endian = False
        fmt = '<I'
    elif str1 == b'RIFX':
        is_rf64 = False
        is_big_endian = True
        fmt = '>I'
    elif str1 == b'RF64':
        is_rf64 = True
        is_big_endian = False
        fmt = '<Q'
    else:
        # There are also .wav files with "FFIR" or "XFIR" signatures?
        raise ValueError(f"File format {repr(str1)} not understood. Only "
                         "'RIFF', 'RIFX', and 'RF64' supported.")
    # Size of entire file
    if not is_rf64:
        file_size = struct.unpack(fmt, fid.read(4))[0] + 8
        str2 = fid.read(4)
    else:
        # Skip 0xFFFFFFFF (-1) bytes
        fid.read(4)
        str2 = fid.read(4)
        str3 = fid.read(4)
        if str3 != b'ds64':
            raise ValueError("Invalid RF64 file: ds64 chunk not found.")
        ds64_size = struct.unpack("<I", fid.read(4))[0]
        file_size = struct.unpack(fmt, fid.read(8))[0] + 8
        # Ignore additional attributes of ds64 chunk like sample count, tables, etc.
        # and just skip to the next chunk
        fid.seek(ds64_size - 8, 1)

    if str2 != b'WAVE':
        raise ValueError(f"Not a WAV file. RIFF form type is {repr(str2)}.")

    return file_size, is_big_endian, is_rf64


def _handle_pad_byte(fid, size):
    # "If the chunk size is an odd number of bytes, a pad byte with value zero
    # is written after ckData." So we need to seek past this after each chunk.
    if size % 2:
        fid.seek(1, 1)


def read(filename, mmap=False):
    """
    Open a WAV file.

    Return the sample rate (in samples/sec) and data from an LPCM WAV file.

    Parameters
    ----------
    filename : string or open file handle
        Input WAV file.
    mmap : bool, optional
        Whether to read data as memory-mapped (default: False).  Not compatible
        with some bit depths; see Notes.  Only to be used on real files.

        .. versionadded:: 0.12.0

    Returns
    -------
    rate : int
        Sample rate of WAV file.
    data : numpy array
        Data read from WAV file. Data-type is determined from the file;
        see Notes.  Data is 1-D for 1-channel WAV, or 2-D of shape
        (Nsamples, Nchannels) otherwise. If a file-like input without a
        C-like file descriptor (e.g., :class:`python:io.BytesIO`) is
        passed, this will not be writeable.

    Notes
    -----
    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit integer PCM     -2147483648  +2147483647  int32
    24-bit integer PCM     -2147483648  +2147483392  int32
    16-bit integer PCM     -32768       +32767       int16
    8-bit integer PCM      0            255          uint8
    =====================  ===========  ===========  =============

    WAV files can specify arbitrary bit depth, and this function supports
    reading any integer PCM depth from 1 to 64 bits.  Data is returned in the
    smallest compatible numpy int type, in left-justified format.  8-bit and
    lower is unsigned, while 9-bit and higher is signed.

    For example, 24-bit data will be stored as int32, with the MSB of the
    24-bit data stored at the MSB of the int32, and typically the least
    significant byte is 0x00.  (However, if a file actually contains data past
    its specified bit depth, those bits will be read and output, too. [2]_)

    This bit justification and sign matches WAV's native internal format, which
    allows memory mapping of WAV files that use 1, 2, 4, or 8 bytes per sample
    (so 24-bit files cannot be memory-mapped, but 32-bit can).

    IEEE float PCM in 32- or 64-bit format is supported, with or without mmap.
    Values exceeding [-1, +1] are not clipped.

    Non-linear PCM (mu-law, A-law) is not supported.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html
    .. [2] Adobe Systems Incorporated, "Adobe Audition 3 User Guide", section
       "Audio file formats: 24-bit Packed Int (type 1, 20-bit)", 2007

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> from scipy.io import wavfile
    >>> import scipy.io

    Get the filename for an example .wav file from the tests/data directory.

    >>> data_dir = pjoin(dirname(scipy.io.__file__), 'tests', 'data')
    >>> wav_fname = pjoin(data_dir, 'test-44100Hz-2ch-32bit-float-be.wav')

    Load the .wav file contents.

    >>> samplerate, data = wavfile.read(wav_fname)
    >>> print(f"number of channels = {data.shape[1]}")
    number of channels = 2
    >>> length = data.shape[0] / samplerate
    >>> print(f"length = {length}s")
    length = 0.01s

    Plot the waveform.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time = np.linspace(0., length, data.shape[0])
    >>> plt.plot(time, data[:, 0], label="Left channel")
    >>> plt.plot(time, data[:, 1], label="Right channel")
    >>> plt.legend()
    >>> plt.xlabel("Time [s]")
    >>> plt.ylabel("Amplitude")
    >>> plt.show()

    """
    if hasattr(filename, 'read'):
        fid = filename
        mmap = False
    else:
        fid = open(filename, 'rb')

    try:
        file_size, is_big_endian, is_rf64 = _read_riff_chunk(fid)
        fmt_chunk_received = False
        data_chunk_received = False
        while fid.tell() < file_size:
            # read the next chunk
            chunk_id = fid.read(4)

            if not chunk_id:
                if data_chunk_received:
                    # End of file but data successfully read
                    warnings.warn(
                        f"Reached EOF prematurely; finished at {fid.tell():d} bytes, "
                        f"expected {file_size:d} bytes from header.",
                        WavFileWarning, stacklevel=2)
                    break
                else:
                    raise ValueError("Unexpected end of file.")
            elif len(chunk_id) < 4:
                msg = f"Incomplete chunk ID: {repr(chunk_id)}"
                # If we have the data, ignore the broken chunk
                if fmt_chunk_received and data_chunk_received:
                    warnings.warn(msg + ", ignoring it.", WavFileWarning,
                                  stacklevel=2)
                else:
                    raise ValueError(msg)

            if chunk_id == b'fmt ':
                fmt_chunk_received = True
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                format_tag, channels, fs = fmt_chunk[1:4]
                bit_depth = fmt_chunk[6]
                block_align = fmt_chunk[5]
            elif chunk_id == b'fact':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id == b'data':
                data_chunk_received = True
                if not fmt_chunk_received:
                    raise ValueError("No fmt chunk before data")
                data = _read_data_chunk(fid, format_tag, channels, bit_depth,
                                        is_big_endian, is_rf64, block_align, mmap)
            elif chunk_id == b'LIST':
                # Someday this could be handled properly but for now skip it
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id in {b'JUNK', b'Fake'}:
                # Skip alignment chunks without warning
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                warnings.warn("Chunk (non-data) not understood, skipping it.",
                              WavFileWarning, stacklevel=2)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:
        if not hasattr(filename, 'read'):
            fid.close()
        else:
            fid.seek(0)

    return fs, data


def write(filename, rate, data):
    """
    Write a NumPy array as a WAV file.

    Parameters
    ----------
    filename : string or open file handle
        Output wav file.
    rate : int
        The sample rate (in samples/sec).
    data : ndarray
        A 1-D or 2-D NumPy array of either integer or float data-type.

    Notes
    -----
    * Writes a simple uncompressed WAV file.
    * To write multiple-channels, use a 2-D array of shape
      (Nsamples, Nchannels).
    * The bits-per-sample and PCM/float will be determined by the data-type.

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    Examples
    --------
    Create a 100Hz sine wave, sampled at 44100Hz.
    Write to 16-bit PCM, Mono.

    >>> from scipy.io.wavfile import write
    >>> import numpy as np
    >>> samplerate = 44100; fs = 100
    >>> t = np.linspace(0., 1., samplerate)
    >>> amplitude = np.iinfo(np.int16).max
    >>> data = amplitude * np.sin(2. * np.pi * fs * t)
    >>> write("example.wav", samplerate, data.astype(np.int16))

    """
    if hasattr(filename, 'write'):
        fid = filename
    else:
        fid = open(filename, 'wb')

    fs = rate

    try:
        dkind = data.dtype.kind
        allowed_dtypes = ['float32', 'float64',
                          'uint8', 'int16', 'int32', 'int64']
        if data.dtype.name not in allowed_dtypes:
            raise ValueError(f"Unsupported data type '{data.dtype}'")

        header_data = b''

        header_data += b'RIFF'
        header_data += b'\x00\x00\x00\x00'
        header_data += b'WAVE'

        # fmt chunk
        header_data += b'fmt '
        if dkind == 'f':
            format_tag = WAVE_FORMAT.IEEE_FLOAT
        else:
            format_tag = WAVE_FORMAT.PCM
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]
        bit_depth = data.dtype.itemsize * 8
        bytes_per_second = fs*(bit_depth // 8)*channels
        block_align = channels * (bit_depth // 8)

        fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs,
                                     bytes_per_second, block_align, bit_depth)
        if not (dkind == 'i' or dkind == 'u'):
            # add cbSize field for non-PCM files
            fmt_chunk_data += b'\x00\x00'

        header_data += struct.pack('<I', len(fmt_chunk_data))
        header_data += fmt_chunk_data

        # check data size (needs to be immediately before the data chunk)
        # if too large for standard RIFF, use RF64 instead
        resulting_file_size = len(header_data) + 4 + 4 + data.nbytes
        is_rf64 = (resulting_file_size - 8) > 0xFFFFFFFF
        if is_rf64:
            header_data = b''
            header_data += b'RF64'
            header_data += b'\xFF\xFF\xFF\xFF'
            header_data += b'WAVE'
            header_data += b'ds64'
            # size of ds64 chunk
            header_data += struct.pack('<I', 28)
            # will be filled later with real file size
            header_data += struct.pack('<Q', 0)
            header_data += struct.pack('<Q', data.nbytes)
            header_data += struct.pack('<Q', data.shape[0])
            # ignore 'table' field for now
            header_data += struct.pack('<I', 0)
            header_data += b'fmt '
            header_data += struct.pack('<I', len(fmt_chunk_data))
            header_data += fmt_chunk_data

        # fact chunk (non-PCM files)
        if not (dkind == 'i' or dkind == 'u'):
            header_data += b'fact'
            header_data += struct.pack('<II', 4, data.shape[0])

        fid.write(header_data)

        # data chunk
        fid.write(b'data')
        # write data chunk size, unless its too big in which case 0xFFFFFFFF is written
        fid.write(struct.pack('<I', min(data.nbytes, 4294967295)))

        if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and
                                           sys.byteorder == 'big'):
            data = data.byteswap()
        _array_tofile(fid, data)

        # Determine file size and place it in correct
        # position at start of the file or the data chunk.
        size = fid.tell()
        if not is_rf64:
            fid.seek(4)
            fid.write(struct.pack('<I', size-8))
        else:
            fid.seek(20)
            fid.write(struct.pack('<Q', size-8))

    finally:
        if not hasattr(filename, 'write'):
            fid.close()
        else:
            fid.seek(0)


def _array_tofile(fid, data):
    # ravel gives a c-contiguous buffer
    fid.write(data.ravel().view('b').data)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ["readsav"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="idl",
                                   private_modules=["_idl"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
==================================
Input and output (:mod:`scipy.io`)
==================================

.. currentmodule:: scipy.io

SciPy has many modules, classes, and functions available to read data
from and write data to a variety of file formats.

.. seealso:: `NumPy IO routines <https://www.numpy.org/devdocs/reference/routines.io.html>`__

MATLAB® files
=============

.. autosummary::
   :toctree: generated/

   loadmat - Read a MATLAB style mat file (version 4 through 7.1)
   savemat - Write a MATLAB style mat file (version 4 through 7.1)
   whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)

For low-level MATLAB reading and writing utilities, see `scipy.io.matlab`.

IDL® files
==========

.. autosummary::
   :toctree: generated/

   readsav - Read an IDL 'save' file

Matrix Market files
===================

.. autosummary::
   :toctree: generated/

   mminfo - Query matrix info from Matrix Market formatted file
   mmread - Read matrix from Matrix Market formatted file
   mmwrite - Write matrix to Matrix Market formatted file

Unformatted Fortran files
===============================

.. autosummary::
   :toctree: generated/

   FortranFile - A file object for unformatted sequential Fortran files
   FortranEOFError - Exception indicating the end of a well-formed file
   FortranFormattingError - Exception indicating an inappropriate end

Netcdf
======

.. autosummary::
   :toctree: generated/

   netcdf_file - A file object for NetCDF data
   netcdf_variable - A data object for the netcdf module

Harwell-Boeing files
====================

.. autosummary::
   :toctree: generated/

   hb_read   -- read H-B file
   hb_write  -- write H-B file

Wav sound files (:mod:`scipy.io.wavfile`)
=========================================

.. module:: scipy.io.wavfile

.. autosummary::
   :toctree: generated/

   read
   write
   WavFileWarning

Arff files (:mod:`scipy.io.arff`)
=================================

.. module:: scipy.io.arff

.. autosummary::
   :toctree: generated/

   loadarff
   MetaData
   ArffError
   ParseArffError
"""
# matfile read and write
from .matlab import loadmat, savemat, whosmat

# netCDF file support
from ._netcdf import netcdf_file, netcdf_variable

# Fortran file support
from ._fortran import FortranFile, FortranEOFError, FortranFormattingError

from ._fast_matrix_market import mminfo, mmread, mmwrite
from ._idl import readsav
from ._harwell_boeing import hb_read, hb_write

# Deprecated namespaces, to be removed in v2.0.0
from . import arff, harwell_boeing, idl, mmio, netcdf, wavfile

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""
  Matrix Market I/O in Python.
  See http://math.nist.gov/MatrixMarket/formats.html
  for information about the Matrix Market format.
"""
#
# Author: Pearu Peterson <pearu@cens.ioc.ee>
# Created: October, 2004
#
# References:
#  http://math.nist.gov/MatrixMarket/
#
import os

import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
                   ones, can_cast)

from scipy.sparse import coo_array, issparse, coo_matrix

__all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile']


# -----------------------------------------------------------------------------
def asstr(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def mminfo(source):
    """
    Return size and storage parameters from Matrix Market file-like 'source'.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object

    Returns
    -------
    rows : int
        Number of matrix rows.
    cols : int
        Number of matrix columns.
    entries : int
        Number of non-zero entries of a sparse matrix
        or rows*cols for a dense matrix.
    format : str
        Either 'coordinate' or 'array'.
    field : str
        Either 'real', 'complex', 'pattern', or 'integer'.
    symmetry : str
        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mminfo

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''


    ``mminfo(source)`` returns the number of rows, number of columns,
    format, field type and symmetry attribute of the source file.

    >>> mminfo(StringIO(text))
    (5, 5, 7, 'coordinate', 'real', 'general')
    """
    return MMFile.info(source)

# -----------------------------------------------------------------------------


def mmread(source, *, spmatrix=True):
    """
    Reads the contents of a Matrix Market file-like 'source' into a matrix.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extensions .mtx, .mtz.gz)
        or open file-like object.
    spmatrix : bool, optional (default: True)
        If ``True``, return sparse ``coo_matrix``. Otherwise return ``coo_array``.

    Returns
    -------
    a : ndarray or coo_array or coo_matrix
        Dense or sparse array depending on the matrix format in the
        Matrix Market file.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mmread

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''

    ``mmread(source)`` returns the data as sparse matrix in COO format.

    >>> m = mmread(StringIO(text), spmatrix=False)
    >>> m
    <COOrdinate sparse array of dtype 'float64'
         with 7 stored elements and shape (5, 5)>
    >>> m.toarray()
    array([[0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 2., 3.],
           [4., 5., 6., 7., 0.],
           [0., 0., 0., 0., 0.]])
    """
    return MMFile().read(source, spmatrix=spmatrix)

# -----------------------------------------------------------------------------


def mmwrite(target, a, comment='', field=None, precision=None, symmetry=None):
    r"""
    Writes the sparse or dense array `a` to Matrix Market file-like `target`.

    Parameters
    ----------
    target : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object.
    a : array like
        Sparse or dense 2-D array.
    comment : str, optional
        Comments to be prepended to the Matrix Market file.
    field : None or str, optional
        Either 'real', 'complex', 'pattern', or 'integer'.
    precision : None or int, optional
        Number of digits to display for real or complex values.
    symmetry : None or str, optional
        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        If symmetry is None the symmetry type of 'a' is determined by its
        values.

    Returns
    -------
    None

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from scipy.io import mmwrite

    Write a small NumPy array to a matrix market file.  The file will be
    written in the ``'array'`` format.

    >>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])
    >>> target = BytesIO()
    >>> mmwrite(target, a)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    2 4
    1
    0
    0
    2.5
    0
    0
    0
    6.25

    Add a comment to the output file, and set the precision to 3.

    >>> target = BytesIO()
    >>> mmwrite(target, a, comment='\n Some test data.\n', precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    % Some test data.
    %
    2 4
    1.00e+00
    0.00e+00
    0.00e+00
    2.50e+00
    0.00e+00
    0.00e+00
    0.00e+00
    6.25e+00

    Convert to a sparse matrix before calling ``mmwrite``.  This will
    result in the output format being ``'coordinate'`` rather than
    ``'array'``.

    >>> target = BytesIO()
    >>> mmwrite(target, coo_array(a), precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix coordinate real general
    %
    2 4 3
    1 1 1.00e+00
    2 2 2.50e+00
    2 4 6.25e+00

    Write a complex Hermitian array to a matrix market file.  Note that
    only six values are actually written to the file; the other values
    are implied by the symmetry.

    >>> z = np.array([[3, 1+2j, 4-3j], [1-2j, 1, -5j], [4+3j, 5j, 2.5]])
    >>> z
    array([[ 3. +0.j,  1. +2.j,  4. -3.j],
           [ 1. -2.j,  1. +0.j, -0. -5.j],
           [ 4. +3.j,  0. +5.j,  2.5+0.j]])

    >>> target = BytesIO()
    >>> mmwrite(target, z, precision=2)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array complex hermitian
    %
    3 3
    3.0e+00 0.0e+00
    1.0e+00 -2.0e+00
    4.0e+00 3.0e+00
    1.0e+00 0.0e+00
    0.0e+00 5.0e+00
    2.5e+00 0.0e+00

    """
    MMFile().write(target, a, comment, field, precision, symmetry)


###############################################################################
class MMFile:
    __slots__ = ('_rows',
                 '_cols',
                 '_entries',
                 '_format',
                 '_field',
                 '_symmetry')

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def entries(self):
        return self._entries

    @property
    def format(self):
        return self._format

    @property
    def field(self):
        return self._field

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def has_symmetry(self):
        return self._symmetry in (self.SYMMETRY_SYMMETRIC,
                                  self.SYMMETRY_SKEW_SYMMETRIC,
                                  self.SYMMETRY_HERMITIAN)

    # format values
    FORMAT_COORDINATE = 'coordinate'
    FORMAT_ARRAY = 'array'
    FORMAT_VALUES = (FORMAT_COORDINATE, FORMAT_ARRAY)

    @classmethod
    def _validate_format(self, format):
        if format not in self.FORMAT_VALUES:
            msg = f'unknown format type {format}, must be one of {self.FORMAT_VALUES}'
            raise ValueError(msg)

    # field values
    FIELD_INTEGER = 'integer'
    FIELD_UNSIGNED = 'unsigned-integer'
    FIELD_REAL = 'real'
    FIELD_COMPLEX = 'complex'
    FIELD_PATTERN = 'pattern'
    FIELD_VALUES = (FIELD_INTEGER, FIELD_UNSIGNED, FIELD_REAL, FIELD_COMPLEX,
                    FIELD_PATTERN)

    @classmethod
    def _validate_field(self, field):
        if field not in self.FIELD_VALUES:
            msg = f'unknown field type {field}, must be one of {self.FIELD_VALUES}'
            raise ValueError(msg)

    # symmetry values
    SYMMETRY_GENERAL = 'general'
    SYMMETRY_SYMMETRIC = 'symmetric'
    SYMMETRY_SKEW_SYMMETRIC = 'skew-symmetric'
    SYMMETRY_HERMITIAN = 'hermitian'
    SYMMETRY_VALUES = (SYMMETRY_GENERAL, SYMMETRY_SYMMETRIC,
                       SYMMETRY_SKEW_SYMMETRIC, SYMMETRY_HERMITIAN)

    @classmethod
    def _validate_symmetry(self, symmetry):
        if symmetry not in self.SYMMETRY_VALUES:
            raise ValueError(f'unknown symmetry type {symmetry}, '
                             f'must be one of {self.SYMMETRY_VALUES}')

    DTYPES_BY_FIELD = {FIELD_INTEGER: 'intp',
                       FIELD_UNSIGNED: 'uint64',
                       FIELD_REAL: 'd',
                       FIELD_COMPLEX: 'D',
                       FIELD_PATTERN: 'd'}

    # -------------------------------------------------------------------------
    @staticmethod
    def reader():
        pass

    # -------------------------------------------------------------------------
    @staticmethod
    def writer():
        pass

    # -------------------------------------------------------------------------
    @classmethod
    def info(self, source):
        """
        Return size, storage parameters from Matrix Market file-like 'source'.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extension .mtx) or open file-like object

        Returns
        -------
        rows : int
            Number of matrix rows.
        cols : int
            Number of matrix columns.
        entries : int
            Number of non-zero entries of a sparse matrix
            or rows*cols for a dense matrix.
        format : str
            Either 'coordinate' or 'array'.
        field : str
            Either 'real', 'complex', 'pattern', or 'integer'.
        symmetry : str
            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        """

        stream, close_it = self._open(source)

        try:

            # read and validate header line
            line = stream.readline()
            mmid, matrix, format, field, symmetry = \
                (asstr(part.strip()) for part in line.split())
            if not mmid.startswith('%%MatrixMarket'):
                raise ValueError('source is not in Matrix Market format')
            if not matrix.lower() == 'matrix':
                raise ValueError("Problem reading file header: " + line)

            # http://math.nist.gov/MatrixMarket/formats.html
            if format.lower() == 'array':
                format = self.FORMAT_ARRAY
            elif format.lower() == 'coordinate':
                format = self.FORMAT_COORDINATE

            # skip comments
            # line.startswith('%')
            while line:
                if line.lstrip() and line.lstrip()[0] in ['%', 37]:
                    line = stream.readline()
                else:
                    break

            # skip empty lines
            while not line.strip():
                line = stream.readline()

            split_line = line.split()
            if format == self.FORMAT_ARRAY:
                if not len(split_line) == 2:
                    raise ValueError("Header line not of length 2: " +
                                     line.decode('ascii'))
                rows, cols = map(int, split_line)
                entries = rows * cols
            else:
                if not len(split_line) == 3:
                    raise ValueError("Header line not of length 3: " +
                                     line.decode('ascii'))
                rows, cols, entries = map(int, split_line)

            return (rows, cols, entries, format, field.lower(),
                    symmetry.lower())

        finally:
            if close_it:
                stream.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def _open(filespec, mode='rb'):
        """ Return an open file stream for reading based on source.

        If source is a file name, open it (after trying to find it with mtx and
        gzipped mtx extensions). Otherwise, just return source.

        Parameters
        ----------
        filespec : str or file-like
            String giving file name or file-like object
        mode : str, optional
            Mode with which to open file, if `filespec` is a file name.

        Returns
        -------
        fobj : file-like
            Open file-like object.
        close_it : bool
            True if the calling function should close this file when done,
            false otherwise.
        """
        # If 'filespec' is path-like (str, pathlib.Path, os.DirEntry, other class
        # implementing a '__fspath__' method), try to convert it to str. If this
        # fails by throwing a 'TypeError', assume it's an open file handle and
        # return it as-is.
        try:
            filespec = os.fspath(filespec)
        except TypeError:
            return filespec, False

        # 'filespec' is definitely a str now

        # open for reading
        if mode[0] == 'r':

            # determine filename plus extension
            if not os.path.isfile(filespec):
                if os.path.isfile(filespec+'.mtx'):
                    filespec = filespec + '.mtx'
                elif os.path.isfile(filespec+'.mtx.gz'):
                    filespec = filespec + '.mtx.gz'
                elif os.path.isfile(filespec+'.mtx.bz2'):
                    filespec = filespec + '.mtx.bz2'
            # open filename
            if filespec.endswith('.gz'):
                import gzip
                stream = gzip.open(filespec, mode)
            elif filespec.endswith('.bz2'):
                import bz2
                stream = bz2.BZ2File(filespec, 'rb')
            else:
                stream = open(filespec, mode)

        # open for writing
        else:
            if filespec[-4:] != '.mtx':
                filespec = filespec + '.mtx'
            stream = open(filespec, mode)

        return stream, True

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_symmetry(a):
        m, n = a.shape
        if m != n:
            return MMFile.SYMMETRY_GENERAL
        issymm = True
        isskew = True
        isherm = a.dtype.char in 'FD'

        # sparse input
        if issparse(a):
            # check if number of nonzero entries of lower and upper triangle
            # matrix are equal
            a = a.tocoo()
            (row, col) = a.nonzero()
            if (row < col).sum() != (row > col).sum():
                return MMFile.SYMMETRY_GENERAL

            # define iterator over symmetric pair entries
            a = a.todok()

            def symm_iterator():
                for ((i, j), aij) in a.items():
                    if i > j:
                        aji = a[j, i]
                        yield (aij, aji, False)
                    elif i == j:
                        yield (aij, aij, True)

        # non-sparse input
        else:
            # define iterator over symmetric pair entries
            def symm_iterator():
                for j in range(n):
                    for i in range(j, n):
                        aij, aji = a[i][j], a[j][i]
                        yield (aij, aji, i == j)

        # check for symmetry
        # yields aij, aji, is_diagonal
        for (aij, aji, is_diagonal) in symm_iterator():
            if isskew and is_diagonal and aij != 0:
                isskew = False
            else:
                if issymm and aij != aji:
                    issymm = False
                with np.errstate(over="ignore"):
                    # This can give a warning for uint dtypes, so silence that
                    if isskew and aij != -aji:
                        isskew = False
                if isherm and aij != conj(aji):
                    isherm = False
            if not (issymm or isskew or isherm):
                break

        # return symmetry value
        if issymm:
            return MMFile.SYMMETRY_SYMMETRIC
        if isskew:
            return MMFile.SYMMETRY_SKEW_SYMMETRIC
        if isherm:
            return MMFile.SYMMETRY_HERMITIAN
        return MMFile.SYMMETRY_GENERAL

    # -------------------------------------------------------------------------
    @staticmethod
    def _field_template(field, precision):
        return {MMFile.FIELD_REAL: '%%.%ie\n' % precision,
                MMFile.FIELD_INTEGER: '%i\n',
                MMFile.FIELD_UNSIGNED: '%u\n',
                MMFile.FIELD_COMPLEX: '%%.%ie %%.%ie\n' %
                    (precision, precision)
                }.get(field, None)

    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        self._init_attrs(**kwargs)

    # -------------------------------------------------------------------------
    def read(self, source, *, spmatrix=True):
        """
        Reads the contents of a Matrix Market file-like 'source' into a matrix.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extensions .mtx, .mtz.gz)
            or open file object.
        spmatrix : bool, optional (default: True)
            If ``True``, return sparse ``coo_matrix``. Otherwise return ``coo_array``.

        Returns
        -------
        a : ndarray or coo_array or coo_matrix
            Dense or sparse array depending on the matrix format in the
            Matrix Market file.
        """
        stream, close_it = self._open(source)

        try:
            self._parse_header(stream)
            data = self._parse_body(stream)

        finally:
            if close_it:
                stream.close()
        if spmatrix and isinstance(data, coo_array):
            data = coo_matrix(data)
        return data


    # -------------------------------------------------------------------------
    def write(self, target, a, comment='', field=None, precision=None,
              symmetry=None):
        """
        Writes sparse or dense array `a` to Matrix Market file-like `target`.

        Parameters
        ----------
        target : str or file-like
            Matrix Market filename (extension .mtx) or open file-like object.
        a : array like
            Sparse or dense 2-D array.
        comment : str, optional
            Comments to be prepended to the Matrix Market file.
        field : None or str, optional
            Either 'real', 'complex', 'pattern', or 'integer'.
        precision : None or int, optional
            Number of digits to display for real or complex values.
        symmetry : None or str, optional
            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
            If symmetry is None the symmetry type of 'a' is determined by its
            values.
        """

        stream, close_it = self._open(target, 'wb')

        try:
            self._write(stream, a, comment, field, precision, symmetry)

        finally:
            if close_it:
                stream.close()
            else:
                stream.flush()

    # -------------------------------------------------------------------------
    def _init_attrs(self, **kwargs):
        """
        Initialize each attributes with the corresponding keyword arg value
        or a default of None
        """

        attrs = self.__class__.__slots__
        public_attrs = [attr[1:] for attr in attrs]
        invalid_keys = set(kwargs.keys()) - set(public_attrs)

        if invalid_keys:
            raise ValueError(f"found {tuple(invalid_keys)} invalid keyword "
                             f"arguments, please only use {public_attrs}")

        for attr in attrs:
            setattr(self, attr, kwargs.get(attr[1:], None))

    # -------------------------------------------------------------------------
    def _parse_header(self, stream):
        rows, cols, entries, format, field, symmetry = \
            self.__class__.info(stream)
        self._init_attrs(rows=rows, cols=cols, entries=entries, format=format,
                         field=field, symmetry=symmetry)

    # -------------------------------------------------------------------------
    def _parse_body(self, stream):
        rows, cols, entries, format, field, symm = (self.rows, self.cols,
                                                    self.entries, self.format,
                                                    self.field, self.symmetry)

        dtype = self.DTYPES_BY_FIELD.get(field, None)

        has_symmetry = self.has_symmetry
        is_integer = field == self.FIELD_INTEGER
        is_unsigned_integer = field == self.FIELD_UNSIGNED
        is_complex = field == self.FIELD_COMPLEX
        is_skew = symm == self.SYMMETRY_SKEW_SYMMETRIC
        is_herm = symm == self.SYMMETRY_HERMITIAN
        is_pattern = field == self.FIELD_PATTERN

        if format == self.FORMAT_ARRAY:
            a = zeros((rows, cols), dtype=dtype)
            line = 1
            i, j = 0, 0
            if is_skew:
                a[i, j] = 0
                if i < rows - 1:
                    i += 1
            while line:
                line = stream.readline()
                # line.startswith('%')
                if not line or line[0] in ['%', 37] or not line.strip():
                    continue
                if is_integer:
                    aij = int(line)
                elif is_unsigned_integer:
                    aij = int(line)
                elif is_complex:
                    aij = complex(*map(float, line.split()))
                else:
                    aij = float(line)
                a[i, j] = aij
                if has_symmetry and i != j:
                    if is_skew:
                        a[j, i] = -aij
                    elif is_herm:
                        a[j, i] = conj(aij)
                    else:
                        a[j, i] = aij
                if i < rows-1:
                    i = i + 1
                else:
                    j = j + 1
                    if not has_symmetry:
                        i = 0
                    else:
                        i = j
                        if is_skew:
                            a[i, j] = 0
                            if i < rows-1:
                                i += 1

            if is_skew:
                if not (i in [0, j] and j == cols - 1):
                    raise ValueError("Parse error, did not read all lines.")
            else:
                if not (i in [0, j] and j == cols):
                    raise ValueError("Parse error, did not read all lines.")

        elif format == self.FORMAT_COORDINATE:
            # Read sparse COOrdinate format

            if entries == 0:
                # empty matrix
                return coo_array((rows, cols), dtype=dtype)

            I = zeros(entries, dtype='intc')
            J = zeros(entries, dtype='intc')
            if is_pattern:
                V = ones(entries, dtype='int8')
            elif is_integer:
                V = zeros(entries, dtype='intp')
            elif is_unsigned_integer:
                V = zeros(entries, dtype='uint64')
            elif is_complex:
                V = zeros(entries, dtype='complex')
            else:
                V = zeros(entries, dtype='float')

            entry_number = 0
            for line in stream:
                # line.startswith('%')
                if not line or line[0] in ['%', 37] or not line.strip():
                    continue

                if entry_number+1 > entries:
                    raise ValueError("'entries' in header is smaller than "
                                     "number of entries")
                l = line.split()
                I[entry_number], J[entry_number] = map(int, l[:2])

                if not is_pattern:
                    if is_integer:
                        V[entry_number] = int(l[2])
                    elif is_unsigned_integer:
                        V[entry_number] = int(l[2])
                    elif is_complex:
                        V[entry_number] = complex(*map(float, l[2:]))
                    else:
                        V[entry_number] = float(l[2])
                entry_number += 1
            if entry_number < entries:
                raise ValueError("'entries' in header is larger than "
                                 "number of entries")

            I -= 1  # adjust indices (base 1 -> base 0)
            J -= 1

            if has_symmetry:
                mask = (I != J)       # off diagonal mask
                od_I = I[mask]
                od_J = J[mask]
                od_V = V[mask]

                I = concatenate((I, od_J))
                J = concatenate((J, od_I))

                if is_skew:
                    od_V *= -1
                elif is_herm:
                    od_V = od_V.conjugate()

                V = concatenate((V, od_V))

            a = coo_array((V, (I, J)), shape=(rows, cols), dtype=dtype)
        else:
            raise NotImplementedError(format)

        return a

    #  ------------------------------------------------------------------------
    def _write(self, stream, a, comment='', field=None, precision=None,
               symmetry=None):
        if isinstance(a, list) or isinstance(a, ndarray) or \
           isinstance(a, tuple) or hasattr(a, '__array__'):
            rep = self.FORMAT_ARRAY
            a = asarray(a)
            if len(a.shape) != 2:
                raise ValueError('Expected 2 dimensional array')
            rows, cols = a.shape

            if field is not None:

                if field == self.FIELD_INTEGER:
                    if not can_cast(a.dtype, 'intp'):
                        raise OverflowError("mmwrite does not support integer "
                                            "dtypes larger than native 'intp'.")
                    a = a.astype('intp')
                elif field == self.FIELD_REAL:
                    if a.dtype.char not in 'fd':
                        a = a.astype('d')
                elif field == self.FIELD_COMPLEX:
                    if a.dtype.char not in 'FD':
                        a = a.astype('D')

        else:
            if not issparse(a):
                raise ValueError(f'unknown matrix type: {type(a)}')

            rep = 'coordinate'
            rows, cols = a.shape

        typecode = a.dtype.char

        if precision is None:
            if typecode in 'fF':
                precision = 8
            else:
                precision = 16
        if field is None:
            kind = a.dtype.kind
            if kind == 'i':
                if not can_cast(a.dtype, 'intp'):
                    raise OverflowError("mmwrite does not support integer "
                                        "dtypes larger than native 'intp'.")
                field = 'integer'
            elif kind == 'f':
                field = 'real'
            elif kind == 'c':
                field = 'complex'
            elif kind == 'u':
                field = 'unsigned-integer'
            else:
                raise TypeError('unexpected dtype kind ' + kind)

        if symmetry is None:
            symmetry = self._get_symmetry(a)

        # validate rep, field, and symmetry
        self.__class__._validate_format(rep)
        self.__class__._validate_field(field)
        self.__class__._validate_symmetry(symmetry)

        # write initial header line
        data = f'%%MatrixMarket matrix {rep} {field} {symmetry}\n'
        stream.write(data.encode('latin1'))

        # write comments
        for line in comment.split('\n'):
            data = f'%{line}\n'
            stream.write(data.encode('latin1'))

        template = self._field_template(field, precision)
        # write dense format
        if rep == self.FORMAT_ARRAY:
            # write shape spec
            data = '%i %i\n' % (rows, cols)
            stream.write(data.encode('latin1'))

            if field in (self.FIELD_INTEGER, self.FIELD_REAL,
                         self.FIELD_UNSIGNED):
                if symmetry == self.SYMMETRY_GENERAL:
                    for j in range(cols):
                        for i in range(rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))

                elif symmetry == self.SYMMETRY_SKEW_SYMMETRIC:
                    for j in range(cols):
                        for i in range(j + 1, rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))

                else:
                    for j in range(cols):
                        for i in range(j, rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))

            elif field == self.FIELD_COMPLEX:

                if symmetry == self.SYMMETRY_GENERAL:
                    for j in range(cols):
                        for i in range(rows):
                            aij = a[i, j]
                            data = template % (real(aij), imag(aij))
                            stream.write(data.encode('latin1'))
                else:
                    for j in range(cols):
                        for i in range(j, rows):
                            aij = a[i, j]
                            data = template % (real(aij), imag(aij))
                            stream.write(data.encode('latin1'))

            elif field == self.FIELD_PATTERN:
                raise ValueError('pattern type inconsisted with dense format')

            else:
                raise TypeError(f'Unknown field type {field}')

        # write sparse format
        else:
            coo = a.tocoo()  # convert to COOrdinate format

            # if symmetry format used, remove values above main diagonal
            if symmetry != self.SYMMETRY_GENERAL:
                lower_triangle_mask = coo.row >= coo.col
                coo = coo_array((coo.data[lower_triangle_mask],
                                (coo.row[lower_triangle_mask],
                                 coo.col[lower_triangle_mask])),
                                shape=coo.shape)

            # write shape spec
            data = '%i %i %i\n' % (rows, cols, coo.nnz)
            stream.write(data.encode('latin1'))

            template = self._field_template(field, precision-1)

            if field == self.FIELD_PATTERN:
                for r, c in zip(coo.row+1, coo.col+1):
                    data = "%i %i\n" % (r, c)
                    stream.write(data.encode('latin1'))
            elif field in (self.FIELD_INTEGER, self.FIELD_REAL,
                           self.FIELD_UNSIGNED):
                for r, c, d in zip(coo.row+1, coo.col+1, coo.data):
                    data = ("%i %i " % (r, c)) + (template % d)
                    stream.write(data.encode('latin1'))
            elif field == self.FIELD_COMPLEX:
                for r, c, d in zip(coo.row+1, coo.col+1, coo.data):
                    data = ("%i %i " % (r, c)) + (template % (d.real, d.imag))
                    stream.write(data.encode('latin1'))
            else:
                raise TypeError(f'Unknown field type {field}')


def _is_fromfile_compatible(stream):
    """
    Check whether `stream` is compatible with numpy.fromfile.

    Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with
    Python 3.
    """

    bad_cls = []
    try:
        import gzip
        bad_cls.append(gzip.GzipFile)
    except ImportError:
        pass
    try:
        import bz2
        bad_cls.append(bz2.BZ2File)
    except ImportError:
        pass

    bad_cls = tuple(bad_cls)
    return not isinstance(stream, bad_cls)
<|endoftext|>"""
Module to read / write Fortran unformatted sequential files.

This is in the spirit of code written by Neil Martinsen-Burrell and Joe Zuntz.

"""
import warnings
import numpy as np

__all__ = ['FortranFile', 'FortranEOFError', 'FortranFormattingError']


class FortranEOFError(TypeError, OSError):
    """Indicates that the file ended properly.

    This error descends from TypeError because the code used to raise
    TypeError (and this was the only way to know that the file had
    ended) so users might have ``except TypeError:``.

    """
    pass


class FortranFormattingError(TypeError, OSError):
    """Indicates that the file ended mid-record.

    Descends from TypeError for backward compatibility.

    """
    pass


class FortranFile:
    """
    A file object for unformatted sequential files from Fortran code.

    Parameters
    ----------
    filename : file or str
        Open file object or filename.
    mode : {'r', 'w'}, optional
        Read-write mode, default is 'r'.
    header_dtype : dtype, optional
        Data type of the header. Size and endianness must match the input/output file.

    Notes
    -----
    These files are broken up into records of unspecified types. The size of
    each record is given at the start (although the size of this header is not
    standard) and the data is written onto disk without any formatting. Fortran
    compilers supporting the BACKSPACE statement will write a second copy of
    the size to facilitate backwards seeking.

    This class only supports files written with both sizes for the record.
    It also does not support the subrecords used in Intel and gfortran compilers
    for records which are greater than 2GB with a 4-byte header.

    An example of an unformatted sequential file in Fortran would be written as::

        OPEN(1, FILE=myfilename, FORM='unformatted')

        WRITE(1) myvariable

    Since this is a non-standard file format, whose contents depend on the
    compiler and the endianness of the machine, caution is advised. Files from
    gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.

    Consider using Fortran direct-access files or files from the newer Stream
    I/O, which can be easily read by `numpy.fromfile`.

    Examples
    --------
    To create an unformatted sequential Fortran file:

    >>> from scipy.io import FortranFile
    >>> import numpy as np
    >>> f = FortranFile('test.unf', 'w')
    >>> f.write_record(np.array([1,2,3,4,5], dtype=np.int32))
    >>> f.write_record(np.linspace(0,1,20).reshape((5,4)).T)
    >>> f.close()

    To read this file:

    >>> f = FortranFile('test.unf', 'r')
    >>> print(f.read_ints(np.int32))
    [1 2 3 4 5]
    >>> print(f.read_reals(float).reshape((5,4), order="F"))
    [[0.         0.05263158 0.10526316 0.15789474]
     [0.21052632 0.26315789 0.31578947 0.36842105]
     [0.42105263 0.47368421 0.52631579 0.57894737]
     [0.63157895 0.68421053 0.73684211 0.78947368]
     [0.84210526 0.89473684 0.94736842 1.        ]]
    >>> f.close()

    Or, in Fortran::

        integer :: a(5), i
        double precision :: b(5,4)
        open(1, file='test.unf', form='unformatted')
        read(1) a
        read(1) b
        close(1)
        write(*,*) a
        do i = 1, 5
            write(*,*) b(i,:)
        end do

    """
    def __init__(self, filename, mode='r', header_dtype=np.uint32):
        if header_dtype is None:
            raise ValueError('Must specify dtype')

        header_dtype = np.dtype(header_dtype)
        if header_dtype.kind != 'u':
            warnings.warn("Given a dtype which is not unsigned.", stacklevel=2)

        if mode not in 'rw' or len(mode) != 1:
            raise ValueError('mode must be either r or w')

        if hasattr(filename, 'seek'):
            self._fp = filename
        else:
            self._fp = open(filename, f'{mode}b')

        self._header_dtype = header_dtype

    def _read_size(self, eof_ok=False):
        n = self._header_dtype.itemsize
        b = self._fp.read(n)
        if (not b) and eof_ok:
            raise FortranEOFError("End of file occurred at end of record")
        elif len(b) < n:
            raise FortranFormattingError(
                "End of file in the middle of the record size")
        return int(np.frombuffer(b, dtype=self._header_dtype, count=1)[0])

    def write_record(self, *items):
        """
        Write a record (including sizes) to the file.

        Parameters
        ----------
        *items : array_like
            The data arrays to write.

        Notes
        -----
        Writes data items to a file::

            write_record(a.T, b.T, c.T, ...)

            write(1) a, b, c, ...

        Note that data in multidimensional arrays is written in
        row-major order --- to make them read correctly by Fortran
        programs, you need to transpose the arrays yourself when
        writing them.

        """
        items = tuple(np.asarray(item) for item in items)
        total_size = sum(item.nbytes for item in items)

        nb = np.array([total_size], dtype=self._header_dtype)

        nb.tofile(self._fp)
        for item in items:
            item.tofile(self._fp)
        nb.tofile(self._fp)

    def read_record(self, *dtypes, **kwargs):
        """
        Reads a record of a given type from the file.

        Parameters
        ----------
        *dtypes : dtypes, optional
            Data type(s) specifying the size and endianness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        Raises
        ------
        FortranEOFError
            To signal that no further records are available
        FortranFormattingError
            To signal that the end of the file was encountered
            part-way through a record

        Notes
        -----
        If the record contains a multidimensional array, you can specify
        the size in the dtype. For example::

            INTEGER var(5,4)

        can be read with::

            read_record('(4,5)i4').T

        Note that this function does **not** assume the file data is in Fortran
        column major order, so you need to (i) swap the order of dimensions
        when reading and (ii) transpose the resulting array.

        Alternatively, you can read the data as a 1-D array and handle the
        ordering yourself. For example::

            read_record('i4').reshape(5, 4, order='F')

        For records that contain several variables or mixed types (as opposed
        to single scalar or array types), give them as separate arguments::

            double precision :: a
            integer :: b
            write(1) a, b

            record = f.read_record('<f4', '<i4')
            a = record[0]  # first number
            b = record[1]  # second number

        and if any of the variables are arrays, the shape can be specified as
        the third item in the relevant dtype::

            double precision :: a
            integer :: b(3,4)
            write(1) a, b

            record = f.read_record('<f4', np.dtype(('<i4', (4, 3))))
            a = record[0]
            b = record[1].T

        NumPy also supports a short syntax for this kind of type::

            record = f.read_record('<f4', '(3,3)<i4')

        See Also
        --------
        read_reals
        read_ints

        """
        dtype = kwargs.pop('dtype', None)
        if kwargs:
            raise ValueError(f"Unknown keyword arguments {tuple(kwargs.keys())}")

        if dtype is not None:
            dtypes = dtypes + (dtype,)
        elif not dtypes:
            raise ValueError('Must specify at least one dtype')

        first_size = self._read_size(eof_ok=True)

        dtypes = tuple(np.dtype(dtype) for dtype in dtypes)
        block_size = sum(dtype.itemsize for dtype in dtypes)

        num_blocks, remainder = divmod(first_size, block_size)
        if remainder != 0:
            raise ValueError(f'Size obtained ({first_size}) is not a multiple of the '
                             f'dtypes given ({block_size}).')

        if len(dtypes) != 1 and first_size != block_size:
            # Fortran does not write mixed type array items in interleaved order,
            # and it's not possible to guess the sizes of the arrays that were written.
            # The user must specify the exact sizes of each of the arrays.
            raise ValueError(f'Size obtained ({first_size}) does not match with the '
                             f'expected size ({block_size}) of multi-item record')

        data = []
        for dtype in dtypes:
            r = np.fromfile(self._fp, dtype=dtype, count=num_blocks)
            if len(r) != num_blocks:
                raise FortranFormattingError(
                    "End of file in the middle of a record")
            if dtype.shape != ():
                # Squeeze outmost block dimension for array items
                if num_blocks == 1:
                    assert r.shape == (1,) + dtype.shape
                    r = r[0]

            data.append(r)

        second_size = self._read_size()
        if first_size != second_size:
            raise ValueError('Sizes do not agree in the header and footer for '
                             'this record - check header dtype')

        # Unpack result
        if len(dtypes) == 1:
            return data[0]
        else:
            return tuple(data)

    def read_ints(self, dtype='i4'):
        """
        Reads a record of a given type from the file, defaulting to an integer
        type (``INTEGER*4`` in Fortran).

        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endianness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        See Also
        --------
        read_reals
        read_record

        """
        return self.read_record(dtype)

    def read_reals(self, dtype='f8'):
        """
        Reads a record of a given type from the file, defaulting to a floating
        point number (``real*8`` in Fortran).

        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endianness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        See Also
        --------
        read_ints
        read_record

        """
        return self.read_record(dtype)

    def close(self):
        """
        Closes the file. It is unsupported to call any other methods off this
        object after closing it. Note that this class supports the 'with'
        statement in modern versions of Python, to call this automatically

        """
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
<|endoftext|># IDLSave - a python module to read IDL 'save' files
# Copyright (c) 2010 Thomas P. Robitaille

# Many thanks to Craig Markwardt for publishing the Unofficial Format
# Specification for IDL .sav files, without which this Python module would not
# exist (http://cow.physics.wisc.edu/~craigm/idl/savefmt).

# This code was developed by with permission from ITT Visual Information
# Systems. IDL(r) is a registered trademark of ITT Visual Information Systems,
# Inc. for their Interactive Data Language software.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

__all__ = ['readsav']

import struct
import numpy as np
import tempfile
import zlib
import warnings

# Define the different data types that can be found in an IDL save file
DTYPE_DICT = {1: '>u1',
              2: '>i2',
              3: '>i4',
              4: '>f4',
              5: '>f8',
              6: '>c8',
              7: '|O',
              8: '|O',
              9: '>c16',
              10: '|O',
              11: '|O',
              12: '>u2',
              13: '>u4',
              14: '>i8',
              15: '>u8'}

# Define the different record types that can be found in an IDL save file
RECTYPE_DICT = {0: "START_MARKER",
                1: "COMMON_VARIABLE",
                2: "VARIABLE",
                3: "SYSTEM_VARIABLE",
                6: "END_MARKER",
                10: "TIMESTAMP",
                12: "COMPILED",
                13: "IDENTIFICATION",
                14: "VERSION",
                15: "HEAP_HEADER",
                16: "HEAP_DATA",
                17: "PROMOTE64",
                19: "NOTICE",
                20: "DESCRIPTION"}

# Define a dictionary to contain structure definitions
STRUCT_DICT = {}


def _align_32(f):
    '''Align to the next 32-bit position in a file'''

    pos = f.tell()
    if pos % 4 != 0:
        f.seek(pos + 4 - pos % 4)
    return


def _skip_bytes(f, n):
    '''Skip `n` bytes'''
    f.read(n)
    return


def _read_bytes(f, n):
    '''Read the next `n` bytes'''
    return f.read(n)


def _read_byte(f):
    '''Read a single byte'''
    return np.uint8(struct.unpack('>B', f.read(4)[:1])[0])


def _read_long(f):
    '''Read a signed 32-bit integer'''
    return np.int32(struct.unpack('>l', f.read(4))[0])


def _read_int16(f):
    '''Read a signed 16-bit integer'''
    return np.int16(struct.unpack('>h', f.read(4)[2:4])[0])


def _read_int32(f):
    '''Read a signed 32-bit integer'''
    return np.int32(struct.unpack('>i', f.read(4))[0])


def _read_int64(f):
    '''Read a signed 64-bit integer'''
    return np.int64(struct.unpack('>q', f.read(8))[0])


def _read_uint16(f):
    '''Read an unsigned 16-bit integer'''
    return np.uint16(struct.unpack('>H', f.read(4)[2:4])[0])


def _read_uint32(f):
    '''Read an unsigned 32-bit integer'''
    return np.uint32(struct.unpack('>I', f.read(4))[0])


def _read_uint64(f):
    '''Read an unsigned 64-bit integer'''
    return np.uint64(struct.unpack('>Q', f.read(8))[0])


def _read_float32(f):
    '''Read a 32-bit float'''
    return np.float32(struct.unpack('>f', f.read(4))[0])


def _read_float64(f):
    '''Read a 64-bit float'''
    return np.float64(struct.unpack('>d', f.read(8))[0])


class Pointer:
    '''Class used to define pointers'''

    def __init__(self, index):
        self.index = index
        return


class ObjectPointer(Pointer):
    '''Class used to define object pointers'''
    pass


def _read_string(f):
    '''Read a string'''
    length = _read_long(f)
    if length > 0:
        chars = _read_bytes(f, length).decode('latin1')
        _align_32(f)
    else:
        chars = ''
    return chars


def _read_string_data(f):
    '''Read a data string (length is specified twice)'''
    length = _read_long(f)
    if length > 0:
        length = _read_long(f)
        string_data = _read_bytes(f, length)
        _align_32(f)
    else:
        string_data = ''
    return string_data


def _read_data(f, dtype):
    '''Read a variable with a specified data type'''
    if dtype == 1:
        if _read_int32(f) != 1:
            raise Exception("Error occurred while reading byte variable")
        return _read_byte(f)
    elif dtype == 2:
        return _read_int16(f)
    elif dtype == 3:
        return _read_int32(f)
    elif dtype == 4:
        return _read_float32(f)
    elif dtype == 5:
        return _read_float64(f)
    elif dtype == 6:
        real = _read_float32(f)
        imag = _read_float32(f)
        return np.complex64(real + imag * 1j)
    elif dtype == 7:
        return _read_string_data(f)
    elif dtype == 8:
        raise Exception("Should not be here - please report this")
    elif dtype == 9:
        real = _read_float64(f)
        imag = _read_float64(f)
        return np.complex128(real + imag * 1j)
    elif dtype == 10:
        return Pointer(_read_int32(f))
    elif dtype == 11:
        return ObjectPointer(_read_int32(f))
    elif dtype == 12:
        return _read_uint16(f)
    elif dtype == 13:
        return _read_uint32(f)
    elif dtype == 14:
        return _read_int64(f)
    elif dtype == 15:
        return _read_uint64(f)
    else:
        raise Exception("Unknown IDL type: %i - please report this" % dtype)


def _read_structure(f, array_desc, struct_desc):
    '''
    Read a structure, with the array and structure descriptors given as
    `array_desc` and `structure_desc` respectively.
    '''

    nrows = array_desc['nelements']
    columns = struct_desc['tagtable']

    dtype = []
    for col in columns:
        if col['structure'] or col['array']:
            dtype.append(((col['name'].lower(), col['name']), np.object_))
        else:
            if col['typecode'] in DTYPE_DICT:
                dtype.append(((col['name'].lower(), col['name']),
                                    DTYPE_DICT[col['typecode']]))
            else:
                raise Exception("Variable type %i not implemented" %
                                                            col['typecode'])

    structure = np.rec.recarray((nrows, ), dtype=dtype)

    for i in range(nrows):
        for col in columns:
            dtype = col['typecode']
            if col['structure']:
                structure[col['name']][i] = _read_structure(f,
                                      struct_desc['arrtable'][col['name']],
                                      struct_desc['structtable'][col['name']])
            elif col['array']:
                structure[col['name']][i] = _read_array(f, dtype,
                                      struct_desc['arrtable'][col['name']])
            else:
                structure[col['name']][i] = _read_data(f, dtype)

    # Reshape structure if needed
    if array_desc['ndims'] > 1:
        dims = array_desc['dims'][:int(array_desc['ndims'])]
        dims.reverse()
        structure = structure.reshape(dims)

    return structure


def _read_array(f, typecode, array_desc):
    '''
    Read an array of type `typecode`, with the array descriptor given as
    `array_desc`.
    '''

    if typecode in [1, 3, 4, 5, 6, 9, 13, 14, 15]:

        if typecode == 1:
            nbytes = _read_int32(f)
            if nbytes != array_desc['nbytes']:
                warnings.warn("Not able to verify number of bytes from header",
                              stacklevel=3)

        # Read bytes as numpy array
        array = np.frombuffer(f.read(array_desc['nbytes']),
                              dtype=DTYPE_DICT[typecode])

    elif typecode in [2, 12]:

        # These are 2 byte types, need to skip every two as they are not packed

        array = np.frombuffer(f.read(array_desc['nbytes']*2),
                              dtype=DTYPE_DICT[typecode])[1::2]

    else:

        # Read bytes into list
        array = []
        for i in range(array_desc['nelements']):
            dtype = typecode
            data = _read_data(f, dtype)
            array.append(data)

        array = np.array(array, dtype=np.object_)

    # Reshape array if needed
    if array_desc['ndims'] > 1:
        dims = array_desc['dims'][:int(array_desc['ndims'])]
        dims.reverse()
        array = array.reshape(dims)

    # Go to next alignment position
    _align_32(f)

    return array


def _read_record(f):
    '''Function to read in a full record'''

    record = {'rectype': _read_long(f)}

    nextrec = _read_uint32(f)
    nextrec += _read_uint32(f).astype(np.int64) * 2**32

    _skip_bytes(f, 4)

    if record['rectype'] not in RECTYPE_DICT:
        raise Exception("Unknown RECTYPE: %i" % record['rectype'])

    record['rectype'] = RECTYPE_DICT[record['rectype']]

    if record['rectype'] in ["VARIABLE", "HEAP_DATA"]:

        if record['rectype'] == "VARIABLE":
            record['varname'] = _read_string(f)
        else:
            record['heap_index'] = _read_long(f)
            _skip_bytes(f, 4)

        rectypedesc = _read_typedesc(f)

        if rectypedesc['typecode'] == 0:

            if nextrec == f.tell():
                record['data'] = None  # Indicates NULL value
            else:
                raise ValueError("Unexpected type code: 0")

        else:

            varstart = _read_long(f)
            if varstart != 7:
                raise Exception("VARSTART is not 7")

            if rectypedesc['structure']:
                record['data'] = _read_structure(f, rectypedesc['array_desc'],
                                                    rectypedesc['struct_desc'])
            elif rectypedesc['array']:
                record['data'] = _read_array(f, rectypedesc['typecode'],
                                                rectypedesc['array_desc'])
            else:
                dtype = rectypedesc['typecode']
                record['data'] = _read_data(f, dtype)

    elif record['rectype'] == "TIMESTAMP":

        _skip_bytes(f, 4*256)
        record['date'] = _read_string(f)
        record['user'] = _read_string(f)
        record['host'] = _read_string(f)

    elif record['rectype'] == "VERSION":

        record['format'] = _read_long(f)
        record['arch'] = _read_string(f)
        record['os'] = _read_string(f)
        record['release'] = _read_string(f)

    elif record['rectype'] == "IDENTIFICATON":

        record['author'] = _read_string(f)
        record['title'] = _read_string(f)
        record['idcode'] = _read_string(f)

    elif record['rectype'] == "NOTICE":

        record['notice'] = _read_string(f)

    elif record['rectype'] == "DESCRIPTION":

        record['description'] = _read_string_data(f)

    elif record['rectype'] == "HEAP_HEADER":

        record['nvalues'] = _read_long(f)
        record['indices'] = [_read_long(f) for _ in range(record['nvalues'])]

    elif record['rectype'] == "COMMONBLOCK":

        record['nvars'] = _read_long(f)
        record['name'] = _read_string(f)
        record['varnames'] = [_read_string(f) for _ in range(record['nvars'])]

    elif record['rectype'] == "END_MARKER":

        record['end'] = True

    elif record['rectype'] == "UNKNOWN":

        warnings.warn("Skipping UNKNOWN record", stacklevel=3)

    elif record['rectype'] == "SYSTEM_VARIABLE":

        warnings.warn("Skipping SYSTEM_VARIABLE record", stacklevel=3)

    else:

        raise Exception(f"record['rectype']={record['rectype']} not implemented")

    f.seek(nextrec)

    return record


def _read_typedesc(f):
    '''Function to read in a type descriptor'''

    typedesc = {'typecode': _read_long(f), 'varflags': _read_long(f)}

    if typedesc['varflags'] & 2 == 2:
        raise Exception("System variables not implemented")

    typedesc['array'] = typedesc['varflags'] & 4 == 4
    typedesc['structure'] = typedesc['varflags'] & 32 == 32

    if typedesc['structure']:
        typedesc['array_desc'] = _read_arraydesc(f)
        typedesc['struct_desc'] = _read_structdesc(f)
    elif typedesc['array']:
        typedesc['array_desc'] = _read_arraydesc(f)

    return typedesc


def _read_arraydesc(f):
    '''Function to read in an array descriptor'''

    arraydesc = {'arrstart': _read_long(f)}

    if arraydesc['arrstart'] == 8:

        _skip_bytes(f, 4)

        arraydesc['nbytes'] = _read_long(f)
        arraydesc['nelements'] = _read_long(f)
        arraydesc['ndims'] = _read_long(f)

        _skip_bytes(f, 8)

        arraydesc['nmax'] = _read_long(f)

        arraydesc['dims'] = [_read_long(f) for _ in range(arraydesc['nmax'])]

    elif arraydesc['arrstart'] == 18:

        warnings.warn("Using experimental 64-bit array read", stacklevel=3)

        _skip_bytes(f, 8)

        arraydesc['nbytes'] = _read_uint64(f)
        arraydesc['nelements'] = _read_uint64(f)
        arraydesc['ndims'] = _read_long(f)

        _skip_bytes(f, 8)

        arraydesc['nmax'] = 8

        arraydesc['dims'] = []
        for d in range(arraydesc['nmax']):
            v = _read_long(f)
            if v != 0:
                raise Exception("Expected a zero in ARRAY_DESC")
            arraydesc['dims'].append(_read_long(f))

    else:

        raise Exception("Unknown ARRSTART: %i" % arraydesc['arrstart'])

    return arraydesc


def _read_structdesc(f):
    '''Function to read in a structure descriptor'''

    structdesc = {}

    structstart = _read_long(f)
    if structstart != 9:
        raise Exception("STRUCTSTART should be 9")

    structdesc['name'] = _read_string(f)
    predef = _read_long(f)
    structdesc['ntags'] = _read_long(f)
    structdesc['nbytes'] = _read_long(f)

    structdesc['predef'] = predef & 1
    structdesc['inherits'] = predef & 2
    structdesc['is_super'] = predef & 4

    if not structdesc['predef']:

        structdesc['tagtable'] = [_read_tagdesc(f)
                                  for _ in range(structdesc['ntags'])]

        for tag in structdesc['tagtable']:
            tag['name'] = _read_string(f)

        structdesc['arrtable'] = {tag['name']: _read_arraydesc(f)
                                  for tag in structdesc['tagtable']
                                  if tag['array']}

        structdesc['structtable'] = {tag['name']: _read_structdesc(f)
                                     for tag in structdesc['tagtable']
                                     if tag['structure']}

        if structdesc['inherits'] or structdesc['is_super']:
            structdesc['classname'] = _read_string(f)
            structdesc['nsupclasses'] = _read_long(f)
            structdesc['supclassnames'] = [
                _read_string(f) for _ in range(structdesc['nsupclasses'])]
            structdesc['supclasstable'] = [
                _read_structdesc(f) for _ in range(structdesc['nsupclasses'])]

        STRUCT_DICT[structdesc['name']] = structdesc

    else:

        if structdesc['name'] not in STRUCT_DICT:
            raise Exception("PREDEF=1 but can't find definition")

        structdesc = STRUCT_DICT[structdesc['name']]

    return structdesc


def _read_tagdesc(f):
    '''Function to read in a tag descriptor'''

    tagdesc = {'offset': _read_long(f)}

    if tagdesc['offset'] == -1:
        tagdesc['offset'] = _read_uint64(f)

    tagdesc['typecode'] = _read_long(f)
    tagflags = _read_long(f)

    tagdesc['array'] = tagflags & 4 == 4
    tagdesc['structure'] = tagflags & 32 == 32
    tagdesc['scalar'] = tagdesc['typecode'] in DTYPE_DICT
    # Assume '10'x is scalar

    return tagdesc


def _replace_heap(variable, heap):

    if isinstance(variable, Pointer):

        while isinstance(variable, Pointer):

            if variable.index == 0:
                variable = None
            else:
                if variable.index in heap:
                    variable = heap[variable.index]
                else:
                    warnings.warn("Variable referenced by pointer not found "
                                  "in heap: variable will be set to None",
                                  stacklevel=3)
                    variable = None

        replace, new = _replace_heap(variable, heap)

        if replace:
            variable = new

        return True, variable

    elif isinstance(variable, np.rec.recarray):

        # Loop over records
        for ir, record in enumerate(variable):

            replace, new = _replace_heap(record, heap)

            if replace:
                variable[ir] = new

        return False, variable

    elif isinstance(variable, np.record):

        # Loop over values
        for iv, value in enumerate(variable):

            replace, new = _replace_heap(value, heap)

            if replace:
                variable[iv] = new

        return False, variable

    elif isinstance(variable, np.ndarray):

        # Loop over values if type is np.object_
        if variable.dtype.type is np.object_:

            for iv in range(variable.size):

                replace, new = _replace_heap(variable.item(iv), heap)

                if replace:
                    variable.reshape(-1)[iv] = new

        return False, variable

    else:

        return False, variable


class AttrDict(dict):
    '''
    A case-insensitive dictionary with access via item, attribute, and call
    notations:

        >>> from scipy.io._idl import AttrDict
        >>> d = AttrDict()
        >>> d['Variable'] = 123
        >>> d['Variable']
        123
        >>> d.Variable
        123
        >>> d.variable
        123
        >>> d('VARIABLE')
        123
        >>> d['missing']
        Traceback (most recent error last):
        ...
        KeyError: 'missing'
        >>> d.missing
        Traceback (most recent error last):
        ...
        AttributeError: 'AttrDict' object has no attribute 'missing'
    '''

    def __init__(self, init=None):
        if init is None:
            init = {}
        dict.__init__(self, init)

    def __getitem__(self, name):
        return super().__getitem__(name.lower())

    def __setitem__(self, key, value):
        return super().__setitem__(key.lower(), value)

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(
                f"'{type(self)}' object has no attribute '{name}'") from None

    __setattr__ = __setitem__
    __call__ = __getitem__


def readsav(file_name, idict=None, python_dict=False,
            uncompressed_file_name=None, verbose=False):
    """
    Read an IDL .sav file.

    Parameters
    ----------
    file_name : str
        Name of the IDL save file.
    idict : dict, optional
        Dictionary in which to insert .sav file variables.
    python_dict : bool, optional
        By default, the object return is not a Python dictionary, but a
        case-insensitive dictionary with item, attribute, and call access
        to variables. To get a standard Python dictionary, set this option
        to True.
    uncompressed_file_name : str, optional
        This option only has an effect for .sav files written with the
        /compress option. If a file name is specified, compressed .sav
        files are uncompressed to this file. Otherwise, readsav will use
        the `tempfile` module to determine a temporary filename
        automatically, and will remove the temporary file upon successfully
        reading it in.
    verbose : bool, optional
        Whether to print out information about the save file, including
        the records read, and available variables.

    Returns
    -------
    idl_dict : AttrDict or dict
        If `python_dict` is set to False (default), this function returns a
        case-insensitive dictionary with item, attribute, and call access
        to variables. If `python_dict` is set to True, this function
        returns a Python dictionary with all variable names in lowercase.
        If `idict` was specified, then variables are written to the
        dictionary specified, and the updated dictionary is returned.

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio
    >>> from scipy.io import readsav

    Get the filename for an example .sav file from the tests/data directory.

    >>> data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
    >>> sav_fname = pjoin(data_dir, 'array_float32_1d.sav')

    Load the .sav file contents.

    >>> sav_data = readsav(sav_fname)

    Get keys of the .sav file contents.

    >>> print(sav_data.keys())
    dict_keys(['array1d'])

    Access a content with a key.

    >>> print(sav_data['array1d'])
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0.]

    """

    # Initialize record and variable holders
    records = []
    if python_dict or idict:
        variables = {}
    else:
        variables = AttrDict()

    # Open the IDL file
    f = open(file_name, 'rb')

    # Read the signature, which should be 'SR'
    signature = _read_bytes(f, 2)
    if signature != b'SR':
        raise Exception(f"Invalid SIGNATURE: {signature}")

    # Next, the record format, which is '\x00\x04' for normal .sav
    # files, and '\x00\x06' for compressed .sav files.
    recfmt = _read_bytes(f, 2)

    if recfmt == b'\x00\x04':
        pass

    elif recfmt == b'\x00\x06':

        if verbose:
            print("IDL Save file is compressed")

        if uncompressed_file_name:
            fout = open(uncompressed_file_name, 'w+b')
        else:
            fout = tempfile.NamedTemporaryFile(suffix='.sav')

        if verbose:
            print(f" -> expanding to {fout.name}")

        # Write header
        fout.write(b'SR\x00\x04')

        # Cycle through records
        while True:

            # Read record type
            rectype = _read_long(f)
            fout.write(struct.pack('>l', int(rectype)))

            # Read position of next record and return as int
            nextrec = _read_uint32(f)
            nextrec += _read_uint32(f).astype(np.int64) * 2**32

            # Read the unknown 4 bytes
            unknown = f.read(4)

            # Check if the end of the file has been reached
            if RECTYPE_DICT[rectype] == 'END_MARKER':
                modval = np.int64(2**32)
                fout.write(struct.pack('>I', int(nextrec) % modval))
                fout.write(
                    struct.pack('>I', int((nextrec - (nextrec % modval)) / modval))
                )
                fout.write(unknown)
                break

            # Find current position
            pos = f.tell()

            # Decompress record
            rec_string = zlib.decompress(f.read(nextrec-pos))

            # Find new position of next record
            nextrec = fout.tell() + len(rec_string) + 12

            # Write out record
            fout.write(struct.pack('>I', int(nextrec % 2**32)))
            fout.write(struct.pack('>I', int((nextrec - (nextrec % 2**32)) / 2**32)))
            fout.write(unknown)
            fout.write(rec_string)

        # Close the original compressed file
        f.close()

        # Set f to be the decompressed file, and skip the first four bytes
        f = fout
        f.seek(4)

    else:
        raise Exception(f"Invalid RECFMT: {recfmt}")

    # Loop through records, and add them to the list
    while True:
        r = _read_record(f)
        records.append(r)
        if 'end' in r:
            if r['end']:
                break

    # Close the file
    f.close()

    # Find heap data variables
    heap = {}
    for r in records:
        if r['rectype'] == "HEAP_DATA":
            heap[r['heap_index']] = r['data']

    # Find all variables
    for r in records:
        if r['rectype'] == "VARIABLE":
            replace, new = _replace_heap(r['data'], heap)
            if replace:
                r['data'] = new
            variables[r['varname'].lower()] = r['data']

    if verbose:

        # Print out timestamp info about the file
        for record in records:
            if record['rectype'] == "TIMESTAMP":
                print("-"*50)
                print(f"Date: {record['date']}")
                print(f"User: {record['user']}")
                print(f"Host: {record['host']}")
                break

        # Print out version info about the file
        for record in records:
            if record['rectype'] == "VERSION":
                print("-"*50)
                print(f"Format: {record['format']}")
                print(f"Architecture: {record['arch']}")
                print(f"Operating System: {record['os']}")
                print(f"IDL Version: {record['release']}")
                break

        # Print out identification info about the file
        for record in records:
            if record['rectype'] == "IDENTIFICATON":
                print("-"*50)
                print(f"Author: {record['author']}")
                print(f"Title: {record['title']}")
                print(f"ID Code: {record['idcode']}")
                break

        # Print out descriptions saved with the file
        for record in records:
            if record['rectype'] == "DESCRIPTION":
                print("-"*50)
                print(f"Description: {record['description']}")
                break

        print("-"*50)
        print(f"Successfully read {len(records)} records of which:")

        # Create convenience list of record types
        rectypes = [r['rectype'] for r in records]

        for rt in set(rectypes):
            if rt != 'END_MARKER':
                print(" - %i are of type %s" % (rectypes.count(rt), rt))
        print("-"*50)

        if 'VARIABLE' in rectypes:
            print("Available variables:")
            for var in variables:
                print(f" - {var} [{type(variables[var])}]")
            print("-"*50)

    if idict:
        for var in variables:
            idict[var] = variables[var]
        return idict
    else:
        return variables
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ["mminfo", "mmread", "mmwrite"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="mmio",
                                   private_modules=["_mmio"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ["netcdf_file", "netcdf_variable"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="netcdf",
                                   private_modules=["_netcdf"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ["hb_read", "hb_write"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="harwell_boeing",
                                   private_modules=["_harwell_boeing"], all=__all__,
                                   attribute=name)
<|endoftext|>''' Byteorder utilities for system - numpy byteorder encoding

Converts a variety of string codes for little endian, big endian,
native byte order and swapped byte order to explicit NumPy endian
codes - one of '<' (little endian) or '>' (big endian)

'''
import sys

__all__ = [
    'aliases', 'native_code', 'swapped_code',
    'sys_is_le', 'to_numpy_code'
]

sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'

aliases = {'little': ('little', '<', 'l', 'le'),
           'big': ('big', '>', 'b', 'be'),
           'native': ('native', '='),
           'swapped': ('swapped', 'S')}


def to_numpy_code(code):
    """
    Convert various order codings to NumPy format.

    Parameters
    ----------
    code : str
        The code to convert. It is converted to lower case before parsing.
        Legal values are:
        'little', 'big', 'l', 'b', 'le', 'be', '<', '>', 'native', '=',
        'swapped', 's'.

    Returns
    -------
    out_code : {'<', '>'}
        Here '<' is the numpy dtype code for little endian,
        and '>' is the code for big endian.

    Examples
    --------
    >>> import sys
    >>> from scipy.io.matlab._byteordercodes import to_numpy_code
    >>> sys_is_le = (sys.byteorder == 'little')
    >>> sys_is_le
    True
    >>> to_numpy_code('big')
    '>'
    >>> to_numpy_code('little')
    '<'
    >>> nc = to_numpy_code('native')
    >>> nc == '<' if sys_is_le else nc == '>'
    True
    >>> sc = to_numpy_code('swapped')
    >>> sc == '>' if sys_is_le else sc == '<'
    True

    """
    code = code.lower()
    if code is None:
        return native_code
    if code in aliases['little']:
        return '<'
    elif code in aliases['big']:
        return '>'
    elif code in aliases['native']:
        return native_code
    elif code in aliases['swapped']:
        return swapped_code
    else:
        raise ValueError(
            f'We cannot handle byte order {code}')
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'MatWriteError', 'MatReadError', 'MatReadWarning', 'MatlabObject',
    'MatlabFunction', 'mat_struct', 'varmats_from_mat',
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5",
                                   private_modules=["_mio5"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio_utils",
                                   private_modules=["_mio_utils"], all=__all__,
                                   attribute=name)
<|endoftext|># Authors: Travis Oliphant, Matthew Brett

"""
Base classes for MATLAB file stream reading.

MATLAB is a registered trademark of the Mathworks inc.
"""

from typing import Final

import numpy as np
from scipy._lib import doccer

from . import _byteordercodes as boc

__all__ = [
    'MatReadError', 'MatReadWarning', 'MatWriteError',
]

class MatReadError(Exception):
    """Exception indicating a read issue."""


class MatWriteError(Exception):
    """Exception indicating a write issue."""


class MatReadWarning(UserWarning):
    """Warning class for read issues."""


doc_dict = \
    {'file_arg':
         '''file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.''',
     'append_arg':
         '''appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present. Default is True.''',
     'load_args':
         '''byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).''',
     'struct_arg':
         '''struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.''',
     'matstream_arg':
         '''mat_stream : file-like
   Object with file API, open for reading.''',
     'long_fields':
         '''long_field_names : bool, optional
   * False - maximum field name length in a structure is 31 characters
     which is the documented maximum length. This is the default.
   * True - maximum field name length in a structure is 63 characters
     which works for MATLAB 7.6''',
     'do_compression':
         '''do_compression : bool, optional
   Whether to compress matrices on write. Default is False.''',
     'oned_as':
         '''oned_as : {'row', 'column'}, optional
   If 'column', write 1-D NumPy arrays as column vectors.
   If 'row', write 1D NumPy arrays as row vectors.''',
     'unicode_strings':
         '''unicode_strings : bool, optional
   If True, write strings as Unicode, else MATLAB usual encoding.'''}

docfiller: Final = doccer.filldoc(doc_dict)

'''

 Note on architecture
======================

There are three sets of parameters relevant for reading files. The
first are *file read parameters* - containing options that are common
for reading the whole file, and therefore every variable within that
file. At the moment these are:

* mat_stream
* dtypes (derived from byte code)
* byte_order
* chars_as_strings
* squeeze_me
* struct_as_record (MATLAB 5 files)
* class_dtypes (derived from order code, MATLAB 5 files)
* codecs (MATLAB 5 files)
* uint16_codec (MATLAB 5 files)

Another set of parameters are those that apply only to the current
variable being read - the *header*:

* header related variables (different for v4 and v5 mat files)
* is_complex
* mclass
* var_stream

With the header, we need ``next_position`` to tell us where the next
variable in the stream is.

Then, for each element in a matrix, there can be *element read
parameters*. An element is, for example, one element in a MATLAB cell
array. At the moment, these are:

* mat_dtype

The file-reading object contains the *file read parameters*. The
*header* is passed around as a data object, or may be read and discarded
in a single function. The *element read parameters* - the mat_dtype in
this instance, is passed into a general post-processing function - see
``mio_utils`` for details.
'''


def convert_dtypes(dtype_template, order_code):
    ''' Convert dtypes in mapping to given order

    Parameters
    ----------
    dtype_template : mapping
       mapping with values returning numpy dtype from ``np.dtype(val)``
    order_code : str
       an order code suitable for using in ``dtype.newbyteorder()``

    Returns
    -------
    dtypes : mapping
       mapping where values have been replaced by
       ``np.dtype(val).newbyteorder(order_code)``

    '''
    dtypes = dtype_template.copy()
    for k in dtypes:
        dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
    return dtypes


def read_dtype(mat_stream, a_dtype):
    """
    Generic get of byte stream data of known type

    Parameters
    ----------
    mat_stream : file_like object
        MATLAB (tm) mat file stream
    a_dtype : dtype
        dtype of array to read. `a_dtype` is assumed to be correct
        endianness.

    Returns
    -------
    arr : ndarray
        Array of dtype `a_dtype` read from stream.

    """
    num_bytes = a_dtype.itemsize
    arr = np.ndarray(shape=(),
                     dtype=a_dtype,
                     buffer=mat_stream.read(num_bytes),
                     order='F')
    return arr


def matfile_version(file_name, *, appendmat=True):
    """
    Return major, minor tuple depending on apparent mat file type

    Where:

     #. 0,x -> version 4 format mat files
     #. 1,x -> version 5 format mat files
     #. 2,x -> version 7.3 format mat files (HDF format)

    Parameters
    ----------
    file_name : str
       Name of the mat file (do not need .mat extension if
       appendmat==True). Can also pass open file-like object.
    appendmat : bool, optional
       True to append the .mat extension to the end of the given
       filename, if not already present. Default is True.

    Returns
    -------
    major_version : {0, 1, 2}
        major MATLAB File format version
    minor_version : int
        minor MATLAB file format version

    Raises
    ------
    MatReadError
        If the file is empty.
    ValueError
        The matfile version is unknown.

    Notes
    -----
    Has the side effect of setting the file read pointer to 0
    """
    from ._mio import _open_file_context
    with _open_file_context(file_name, appendmat=appendmat) as fileobj:
        return _get_matfile_version(fileobj)


get_matfile_version = matfile_version


_HDR_N_BYTES = 20


def _get_matfile_version(fileobj):
    # Mat4 files have a zero somewhere in first 4 bytes
    fileobj.seek(0)
    hdr_bytes = fileobj.read(_HDR_N_BYTES)
    if len(hdr_bytes) < _HDR_N_BYTES:
        raise MatReadError("Mat file appears to be truncated")
    if hdr_bytes.count(0) == _HDR_N_BYTES:
        raise MatReadError("Mat file appears to be corrupt "
                           f"(first {_HDR_N_BYTES} bytes == 0)")
    mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=hdr_bytes[:4])
    if 0 in mopt_ints:
        fileobj.seek(0)
        return (0,0)
    # For 5 format or 7.3 format we need to read an integer in the
    # header. Bytes 124 through 128 contain a version integer and an
    # endian test string
    fileobj.seek(124)
    tst_str = fileobj.read(4)
    fileobj.seek(0)
    maj_ind = int(tst_str[2] == b'I'[0])
    maj_val = int(tst_str[maj_ind])
    min_val = int(tst_str[1 - maj_ind])
    ret = (maj_val, min_val)
    if maj_val in (1, 2):
        return ret
    raise ValueError('Unknown mat file type, version {}, {}'.format(*ret))


def matdims(arr, oned_as='column'):
    """
    Determine equivalent MATLAB dimensions for given array

    Parameters
    ----------
    arr : ndarray
        Input array
    oned_as : {'column', 'row'}, optional
        Whether 1-D arrays are returned as MATLAB row or column matrices.
        Default is 'column'.

    Returns
    -------
    dims : tuple
        Shape tuple, in the form MATLAB expects it.

    Notes
    -----
    We had to decide what shape a 1 dimensional array would be by
    default. ``np.atleast_2d`` thinks it is a row vector. The
    default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

    Versions of scipy up to and including 0.11 resulted (accidentally)
    in 1-D arrays being read as column vectors. For the moment, we
    maintain the same tradition here.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.io.matlab._miobase import matdims
    >>> matdims(np.array(1)) # NumPy scalar
    (1, 1)
    >>> matdims(np.array([1])) # 1-D array, 1 element
    (1, 1)
    >>> matdims(np.array([1,2])) # 1-D array, 2 elements
    (2, 1)
    >>> matdims(np.array([[2],[3]])) # 2-D array, column vector
    (2, 1)
    >>> matdims(np.array([[2,3]])) # 2-D array, row vector
    (1, 2)
    >>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
    (1, 1, 2)
    >>> matdims(np.array([])) # empty 1-D array
    (0, 0)
    >>> matdims(np.array([[]])) # empty 2-D array
    (0, 0)
    >>> matdims(np.array([[[]]])) # empty 3-D array
    (0, 0, 0)

    Optional argument flips 1-D shape behavior.

    >>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
    (1, 2)

    The argument has to make sense though

    >>> matdims(np.array([1,2]), 'bizarre')
    Traceback (most recent call last):
       ...
    ValueError: 1-D option "bizarre" is strange

    """
    shape = arr.shape
    if shape == ():  # scalar
        return (1, 1)
    if len(shape) == 1:  # 1D
        if shape[0] == 0:
            return (0, 0)
        elif oned_as == 'column':
            return shape + (1,)
        elif oned_as == 'row':
            return (1,) + shape
        else:
            raise ValueError(f'1-D option "{oned_as}" is strange')
    return shape


class MatVarReader:
    ''' Abstract class defining required interface for var readers'''
    def __init__(self, file_reader):
        pass

    def read_header(self):
        ''' Returns header '''
        pass

    def array_from_header(self, header):
        ''' Reads array given header '''
        pass


class MatFileReader:
    """ Base object for reading mat files

    To make this class functional, you will need to override the
    following methods:

    matrix_getter_factory   - gives object to fetch next matrix from stream
    guess_byte_order        - guesses file byte order from file
    """

    @docfiller
    def __init__(self, mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 simplify_cells=False):
        '''
        Initializer for mat file reader

        mat_stream : file-like
            object with file API, open for reading
    %(load_args)s
        '''
        # Initialize stream
        self.mat_stream = mat_stream
        self.dtypes = {}
        if not byte_order:
            byte_order = self.guess_byte_order()
        else:
            byte_order = boc.to_numpy_code(byte_order)
        self.byte_order = byte_order
        self.struct_as_record = struct_as_record
        if matlab_compatible:
            self.set_matlab_compatible()
        else:
            self.squeeze_me = squeeze_me
            self.chars_as_strings = chars_as_strings
            self.mat_dtype = mat_dtype
        self.verify_compressed_data_integrity = verify_compressed_data_integrity
        self.simplify_cells = simplify_cells
        if simplify_cells:
            self.squeeze_me = True
            self.struct_as_record = False

    def set_matlab_compatible(self):
        ''' Sets options to return arrays as MATLAB loads them '''
        self.mat_dtype = True
        self.squeeze_me = False
        self.chars_as_strings = False

    def guess_byte_order(self):
        ''' As we do not know what file type we have, assume native '''
        return boc.native_code

    def end_of_stream(self):
        b = self.mat_stream.read(1)
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos-1)
        return len(b) == 0


def arr_dtype_number(arr, num):
    ''' Return dtype for given number of items per element'''
    return np.dtype(arr.dtype.str[:2] + str(num))


def arr_to_chars(arr):
    ''' Convert string array to char array '''
    dims = list(arr.shape)
    if not dims:
        dims = [1]
    dims.append(int(arr.dtype.str[2:]))
    arr = np.ndarray(shape=dims,
                     dtype=arr_dtype_number(arr, 1),
                     buffer=arr)
    empties = [arr == np.array('', dtype=arr.dtype)]
    if not np.any(empties):
        return arr
    arr = arr.copy()
    arr[tuple(empties)] = ' '
    return arr
<|endoftext|>''' Classes for read / write of matlab (TM) 5 files

The matfile specification last found here:

https://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf

(as of December 5 2008)

=================================
 Note on functions and mat files
=================================

The document above does not give any hints as to the storage of matlab
function handles, or anonymous function handles. I had, therefore, to
guess the format of matlab arrays of ``mxFUNCTION_CLASS`` and
``mxOPAQUE_CLASS`` by looking at example mat files.

``mxFUNCTION_CLASS`` stores all types of matlab functions. It seems to
contain a struct matrix with a set pattern of fields. For anonymous
functions, a sub-fields of one of these fields seems to contain the
well-named ``mxOPAQUE_CLASS``. This seems to contain:

* array flags as for any matlab matrix
* 3 int8 strings
* a matrix

It seems that whenever the mat file contains a ``mxOPAQUE_CLASS``
instance, there is also an un-named matrix (name == '') at the end of
the mat file. I'll call this the ``__function_workspace__`` matrix.

When I saved two anonymous functions in a mat file, or appended another
anonymous function to the mat file, there was still only one
``__function_workspace__`` un-named matrix at the end, but larger than
that for a mat file with a single anonymous function, suggesting that
the workspaces for the two functions had been merged.

The ``__function_workspace__`` matrix appears to be of double class
(``mxCLASS_DOUBLE``), but stored as uint8, the memory for which is in
the format of a mini .mat file, without the first 124 bytes of the file
header (the description and the subsystem_offset), but with the version
U2 bytes, and the S2 endian test bytes. There follow 4 zero bytes,
presumably for 8 byte padding, and then a series of ``miMATRIX``
entries, as in a standard mat file. The ``miMATRIX`` entries appear to
be series of un-named (name == '') matrices, and may also contain arrays
of this same mini-mat format.

I guess that:

* saving an anonymous function back to a mat file will need the
  associated ``__function_workspace__`` matrix saved as well for the
  anonymous function to work correctly.
* appending to a mat file that has a ``__function_workspace__`` would
  involve first pulling off this workspace, appending, checking whether
  there were any more anonymous functions appended, and then somehow
  merging the relevant workspaces, and saving at the end of the mat
  file.

The mat files I was playing with are in ``tests/data``:

* sqr.mat
* parabola.mat
* some_functions.mat

See ``tests/test_mio.py:test_mio_funcs.py`` for the debugging
script I was working with.

Small fragments of current code adapted from matfile.py by Heiko
Henkelmann; parts of the code for simplify_cells=True adapted from
http://blog.nephics.com/2019/08/28/better-loadmat-for-scipy/.
'''

import math
import os
import time
import sys
import zlib

from io import BytesIO

import warnings

import numpy as np

import scipy.sparse

from ._byteordercodes import native_code, swapped_code

from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
                      arr_to_chars, arr_dtype_number, MatWriteError,
                      MatReadError, MatReadWarning)

# Reader object for matlab 5 format variables
from ._mio5_utils import VarReader5

# Constants and helper objects
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
                          NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8,
                          miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS,
                          mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS,
                          mxDOUBLE_CLASS, mclass_info, mat_struct)

from ._streams import ZlibInputStream


def _has_struct(elem):
    """Determine if elem is an array and if first array item is a struct."""
    return (isinstance(elem, np.ndarray) and (elem.size > 0) and (elem.ndim > 0) and
            isinstance(elem[0], mat_struct))


def _inspect_cell_array(ndarray):
    """Construct lists from cell arrays (loaded as numpy ndarrays), recursing
    into items if they contain mat_struct objects."""
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, mat_struct):
            elem_list.append(_matstruct_to_dict(sub_elem))
        elif _has_struct(sub_elem):
            elem_list.append(_inspect_cell_array(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def _matstruct_to_dict(matobj):
    """Construct nested dicts from mat_struct objects."""
    d = {}
    for f in matobj._fieldnames:
        elem = matobj.__dict__[f]
        if isinstance(elem, mat_struct):
            d[f] = _matstruct_to_dict(elem)
        elif _has_struct(elem):
            d[f] = _inspect_cell_array(elem)
        else:
            d[f] = elem
    return d


def _simplify_cells(d):
    """Convert mat objects in dict to nested dicts."""
    for key in d:
        if isinstance(d[key], mat_struct):
            d[key] = _matstruct_to_dict(d[key])
        elif _has_struct(d[key]):
            d[key] = _inspect_cell_array(d[key])
    return d


class MatFile5Reader(MatFileReader):
    ''' Reader for Mat 5 mat files
    Adds the following attribute to base class

    uint16_codec - char codec to use for uint16 char arrays
        (defaults to system default codec)

    Uses variable reader that has the following standard interface (see
    abstract class in ``miobase``::

       __init__(self, file_reader)
       read_header(self)
       array_from_header(self)

    and added interface::

       set_stream(self, stream)
       read_full_tag(self)

    '''
    @docfiller
    def __init__(self,
                 mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 uint16_codec=None,
                 simplify_cells=False):
        '''Initializer for matlab 5 file format reader

    %(matstream_arg)s
    %(load_args)s
    %(struct_arg)s
    uint16_codec : {None, string}
        Set codec to use for uint16 char arrays (e.g., 'utf-8').
        Use system default codec if None
        '''
        super().__init__(
            mat_stream,
            byte_order,
            mat_dtype,
            squeeze_me,
            chars_as_strings,
            matlab_compatible,
            struct_as_record,
            verify_compressed_data_integrity,
            simplify_cells)
        # Set uint16 codec
        if not uint16_codec:
            uint16_codec = sys.getdefaultencoding()
        self.uint16_codec = uint16_codec
        # placeholders for readers - see initialize_read method
        self._file_reader = None
        self._matrix_reader = None

    def guess_byte_order(self):
        ''' Guess byte order.
        Sets stream pointer to 0'''
        self.mat_stream.seek(126)
        mi = self.mat_stream.read(2)
        self.mat_stream.seek(0)
        return mi == b'IM' and '<' or '>'

    def read_file_header(self):
        ''' Read in mat 5 file header '''
        hdict = {}
        hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
        hdr = read_dtype(self.mat_stream, hdr_dtype)
        hdict['__header__'] = hdr['description'].item().strip(b' \t\n\000')
        v_major = hdr['version'] >> 8
        v_minor = hdr['version'] & 0xFF
        hdict['__version__'] = '%d.%d' % (v_major, v_minor)
        return hdict

    def initialize_read(self):
        ''' Run when beginning read of variables

        Sets up readers from parameters in `self`
        '''
        # reader for top level stream. We need this extra top-level
        # reader because we use the matrix_reader object to contain
        # compressed matrices (so they have their own stream)
        self._file_reader = VarReader5(self)
        # reader for matrix streams
        self._matrix_reader = VarReader5(self)

    def read_var_header(self):
        ''' Read header, return header, next position

        Header has to define at least .name and .is_global

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes .name and .is_global
        next_position : int
           position in stream of next variable
        '''
        mdtype, byte_count = self._file_reader.read_full_tag()
        if not byte_count > 0:
            raise ValueError("Did not read any bytes")
        next_pos = self.mat_stream.tell() + byte_count
        if mdtype == miCOMPRESSED:
            # Make new stream from compressed data
            stream = ZlibInputStream(self.mat_stream, byte_count)
            self._matrix_reader.set_stream(stream)
            check_stream_limit = self.verify_compressed_data_integrity
            mdtype, byte_count = self._matrix_reader.read_full_tag()
        else:
            check_stream_limit = False
            self._matrix_reader.set_stream(self.mat_stream)
        if not mdtype == miMATRIX:
            raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
        header = self._matrix_reader.read_header(check_stream_limit)
        return header, next_pos

    def read_var_array(self, header, process=True):
        ''' Read array, given `header`

        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False} bool, optional
           If True, apply recursive post-processing during loading of
           array.

        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        '''
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        ''' get variables from stream as dictionary

        variable_names   - optional list of variable names to get

        If variable_names is None, then get all variables in file
        '''
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)

        self.mat_stream.seek(0)
        # Here we pass all the parameters in self to the reading objects
        self.initialize_read()
        mdict = self.read_file_header()
        mdict['__globals__'] = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if name in mdict:
                msg = (
                    f'Duplicate variable name "{name}" in stream'
                    " - replacing previous with new\nConsider"
                    "scipy.io.matlab.varmats_from_mat to split "
                    "file into single variable files"
                )
                warnings.warn(msg, MatReadWarning, stacklevel=2)
            if name == '':
                # can only be a matlab 7 function workspace
                name = '__function_workspace__'
                # We want to keep this raw because mat_dtype processing
                # will break the format (uint8 as mxDOUBLE_CLASS)
                process = False
            else:
                process = True
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            try:
                res = self.read_var_array(hdr, process)
            except MatReadError as err:
                warnings.warn(
                    f'Unreadable variable "{name}", because "{err}"',
                    Warning, stacklevel=2)
                res = f"Read error: {err}"
            self.mat_stream.seek(next_position)
            mdict[name] = res
            if hdr.is_global:
                mdict['__globals__'].append(name)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        if self.simplify_cells:
            return _simplify_cells(mdict)
        else:
            return mdict

    def list_variables(self):
        ''' list variables from stream '''
        self.mat_stream.seek(0)
        # Here we pass all the parameters in self to the reading objects
        self.initialize_read()
        self.read_file_header()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if name == '':
                # can only be a matlab 7 function workspace
                name = '__function_workspace__'

            shape = self._matrix_reader.shape_from_header(hdr)
            if hdr.is_logical:
                info = 'logical'
            else:
                info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))

            self.mat_stream.seek(next_position)
        return vars


def varmats_from_mat(file_obj):
    """ Pull variables out of mat 5 file as a sequence of mat file objects

    This can be useful with a difficult mat file, containing unreadable
    variables. This routine pulls the variables out in raw form and puts them,
    unread, back into a file stream for saving or reading. Another use is the
    pathological case where there is more than one variable of the same name in
    the file; this routine returns the duplicates, whereas the standard reader
    will overwrite duplicates in the returned dictionary.

    The file pointer in `file_obj` will be undefined. File pointers for the
    returned file-like objects are set at 0.

    Parameters
    ----------
    file_obj : file-like
        file object containing mat file

    Returns
    -------
    named_mats : list
        list contains tuples of (name, BytesIO) where BytesIO is a file-like
        object containing mat file contents as for a single variable. The
        BytesIO contains a string with the original header and a single var. If
        ``var_file_obj`` is an individual BytesIO instance, then save as a mat
        file with something like ``open('test.mat',
        'wb').write(var_file_obj.read())``

    Examples
    --------
    >>> import scipy.io
    >>> import numpy as np
    >>> from io import BytesIO
    >>> from scipy.io.matlab._mio5 import varmats_from_mat
    >>> mat_fileobj = BytesIO()
    >>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})
    >>> varmats = varmats_from_mat(mat_fileobj)
    >>> sorted([name for name, str_obj in varmats])
    ['a', 'b']
    """
    rdr = MatFile5Reader(file_obj)
    file_obj.seek(0)
    # Raw read of top-level file header
    hdr_len = MDTYPES[native_code]['dtypes']['file_header'].itemsize
    raw_hdr = file_obj.read(hdr_len)
    # Initialize variable reading
    file_obj.seek(0)
    rdr.initialize_read()
    rdr.read_file_header()
    next_position = file_obj.tell()
    named_mats = []
    while not rdr.end_of_stream():
        start_position = next_position
        hdr, next_position = rdr.read_var_header()
        name = 'None' if hdr.name is None else hdr.name.decode('latin1')
        # Read raw variable string
        file_obj.seek(start_position)
        byte_count = next_position - start_position
        var_str = file_obj.read(byte_count)
        # write to stringio object
        out_obj = BytesIO()
        out_obj.write(raw_hdr)
        out_obj.write(var_str)
        out_obj.seek(0)
        named_mats.append((name, out_obj))
    return named_mats


class EmptyStructMarker:
    """ Class to indicate presence of empty matlab struct on output """


def to_writeable(source):
    ''' Convert input object ``source`` to something we can write

    Parameters
    ----------
    source : object

    Returns
    -------
    arr : None or ndarray or EmptyStructMarker
        If `source` cannot be converted to something we can write to a matfile,
        return None.  If `source` is equivalent to an empty dictionary, return
        ``EmptyStructMarker``.  Otherwise return `source` converted to an
        ndarray with contents for writing to matfile.
    '''
    if isinstance(source, np.ndarray):
        return source
    if source is None:
        return None
    if hasattr(source, "__array__"):
        return np.asarray(source)
    # Objects that implement mappings
    is_mapping = (hasattr(source, 'keys') and hasattr(source, 'values') and
                  hasattr(source, 'items'))
    # Objects that don't implement mappings, but do have dicts
    if isinstance(source, np.generic):
        # NumPy scalars are never mappings (PyPy issue workaround)
        pass
    elif not is_mapping and hasattr(source, '__dict__'):
        source = {key: value for key, value in source.__dict__.items()
                      if not key.startswith('_')}
        is_mapping = True
    if is_mapping:
        dtype = []
        values = []
        for field, value in source.items():
            if (isinstance(field, str) and
                    field[0] not in '_0123456789'):
                dtype.append((str(field), object))
                values.append(value)
        if dtype:
            return np.array([tuple(values)], dtype)
        else:
            return EmptyStructMarker
    # Next try and convert to an array
    try:
        narr = np.asanyarray(source)
    except ValueError:
        narr = np.asanyarray(source, dtype=object)
    if narr.dtype.type in (object, np.object_) and \
       narr.shape == () and narr == source:
        # No interesting conversion possible
        return None
    return narr


# Native byte ordered dtypes for convenience for writers
NDT_FILE_HDR = MDTYPES[native_code]['dtypes']['file_header']
NDT_TAG_FULL = MDTYPES[native_code]['dtypes']['tag_full']
NDT_TAG_SMALL = MDTYPES[native_code]['dtypes']['tag_smalldata']
NDT_ARRAY_FLAGS = MDTYPES[native_code]['dtypes']['array_flags']


class VarWriter5:
    ''' Generic matlab matrix writing class '''
    mat_tag = np.zeros((), NDT_TAG_FULL)
    mat_tag['mdtype'] = miMATRIX  # type: ignore[call-overload]

    def __init__(self, file_writer):
        self.file_stream = file_writer.file_stream
        self.unicode_strings = file_writer.unicode_strings
        self.long_field_names = file_writer.long_field_names
        self.oned_as = file_writer.oned_as
        # These are used for top level writes, and unset after
        self._var_name = None
        self._var_is_global = False

    def write_bytes(self, arr):
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        self.file_stream.write(s)

    def write_element(self, arr, mdtype=None):
        ''' write tag and data '''
        if mdtype is None:
            mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
        # Array needs to be in native byte order
        if arr.dtype.byteorder == swapped_code:
            arr = arr.byteswap().view(arr.dtype.newbyteorder())
        byte_count = arr.size*arr.itemsize
        if byte_count <= 4:
            self.write_smalldata_element(arr, mdtype, byte_count)
        else:
            self.write_regular_element(arr, mdtype, byte_count)

    def write_smalldata_element(self, arr, mdtype, byte_count):
        # write tag with embedded data
        tag = np.zeros((), NDT_TAG_SMALL)
        tag['byte_count_mdtype'] = (byte_count << 16) + mdtype
        # if arr.tobytes is < 4, the element will be zero-padded as needed.
        tag['data'] = arr.tobytes(order='F')
        self.write_bytes(tag)

    def write_regular_element(self, arr, mdtype, byte_count):
        # write tag, data
        tag = np.zeros((), NDT_TAG_FULL)
        tag['mdtype'] = mdtype
        tag['byte_count'] = byte_count
        self.write_bytes(tag)
        self.write_bytes(arr)
        # pad to next 64-bit boundary
        bc_mod_8 = byte_count % 8
        if bc_mod_8:
            self.file_stream.write(b'\x00' * (8-bc_mod_8))

    def write_header(self,
                     shape,
                     mclass,
                     is_complex=False,
                     is_logical=False,
                     nzmax=0):
        ''' Write header for given data options
        shape : sequence
           array shape
        mclass      - mat5 matrix class
        is_complex  - True if matrix is complex
        is_logical  - True if matrix is logical
        nzmax        - max non zero elements for sparse arrays

        We get the name and the global flag from the object, and reset
        them to defaults after we've used them
        '''
        # get name and is_global from one-shot object store
        name = self._var_name
        is_global = self._var_is_global
        # initialize the top-level matrix tag, store position
        self._mat_tag_pos = self.file_stream.tell()
        self.write_bytes(self.mat_tag)
        # write array flags (complex, global, logical, class, nzmax)
        af = np.zeros((), NDT_ARRAY_FLAGS)
        af['data_type'] = miUINT32
        af['byte_count'] = 8
        flags = is_complex << 3 | is_global << 2 | is_logical << 1
        af['flags_class'] = mclass | flags << 8
        af['nzmax'] = nzmax
        self.write_bytes(af)
        # shape
        self.write_element(np.array(shape, dtype='i4'))
        # write name
        name = np.asarray(name)
        if name == '':  # empty string zero-terminated
            self.write_smalldata_element(name, miINT8, 0)
        else:
            self.write_element(name, miINT8)
        # reset the one-shot store to defaults
        self._var_name = ''
        self._var_is_global = False

    def update_matrix_tag(self, start_pos):
        curr_pos = self.file_stream.tell()
        self.file_stream.seek(start_pos)
        byte_count = curr_pos - start_pos - 8
        if byte_count >= 2**32:
            raise MatWriteError("Matrix too large to save with Matlab "
                                "5 format")
        self.mat_tag['byte_count'] = byte_count
        self.write_bytes(self.mat_tag)
        self.file_stream.seek(curr_pos)

    def write_top(self, arr, name, is_global):
        """ Write variable at top level of mat file

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        name : str, optional
            name as it will appear in matlab workspace
            default is empty string
        is_global : {False, True}, optional
            whether variable will be global on load into matlab
        """
        # these are set before the top-level header write, and unset at
        # the end of the same write, because they do not apply for lower levels
        self._var_is_global = is_global
        self._var_name = name
        # write the header and data
        self.write(arr)

    def write(self, arr):
        ''' Write `arr` to stream at top and sub levels

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        '''
        # store position, so we can update the matrix tag
        mat_tag_pos = self.file_stream.tell()
        # First check if these are sparse
        if scipy.sparse.issparse(arr):
            self.write_sparse(arr)
            self.update_matrix_tag(mat_tag_pos)
            return
        # Try to convert things that aren't arrays
        narr = to_writeable(arr)
        if narr is None:
            raise TypeError(f'Could not convert {arr} (type {type(arr)}) to array')
        if isinstance(narr, MatlabObject):
            self.write_object(narr)
        elif isinstance(narr, MatlabFunction):
            raise MatWriteError('Cannot write matlab functions')
        elif narr is EmptyStructMarker:  # empty struct array
            self.write_empty_struct()
        elif narr.dtype.fields:  # struct array
            self.write_struct(narr)
        elif narr.dtype.hasobject:  # cell array
            self.write_cells(narr)
        elif narr.dtype.kind in ('U', 'S'):
            if self.unicode_strings:
                codec = 'UTF8'
            else:
                codec = 'ascii'
            self.write_char(narr, codec)
        else:
            self.write_numeric(narr)
        self.update_matrix_tag(mat_tag_pos)

    def write_numeric(self, arr):
        imagf = arr.dtype.kind == 'c'
        logif = arr.dtype.kind == 'b'
        try:
            mclass = NP_TO_MXTYPES[arr.dtype.str[1:]]
        except KeyError:
            # No matching matlab type, probably complex256 / float128 / float96
            # Cast data to complex128 / float64.
            if imagf:
                arr = arr.astype('c128')
            elif logif:
                arr = arr.astype('i1')  # Should only contain 0/1
            else:
                arr = arr.astype('f8')
            mclass = mxDOUBLE_CLASS
        self.write_header(matdims(arr, self.oned_as),
                          mclass,
                          is_complex=imagf,
                          is_logical=logif)
        if imagf:
            self.write_element(arr.real)
            self.write_element(arr.imag)
        else:
            self.write_element(arr)

    def write_char(self, arr, codec='ascii'):
        ''' Write string array `arr` with given `codec`
        '''
        if arr.size == 0 or np.all(arr == ''):
            # This an empty string array or a string array containing
            # only empty strings. Matlab cannot distinguish between a
            # string array that is empty, and a string array containing
            # only empty strings, because it stores strings as arrays of
            # char. There is no way of having an array of char that is
            # not empty, but contains an empty string. We have to
            # special-case the array-with-empty-strings because even
            # empty strings have zero padding, which would otherwise
            # appear in matlab as a string with a space.
            shape = (0,) * np.max([arr.ndim, 2])
            self.write_header(shape, mxCHAR_CLASS)
            self.write_smalldata_element(arr, miUTF8, 0)
            return
        # non-empty string.
        #
        # Convert to char array
        arr = arr_to_chars(arr)
        # We have to write the shape directly, because we are going
        # recode the characters, and the resulting stream of chars
        # may have a different length
        shape = arr.shape
        self.write_header(shape, mxCHAR_CLASS)
        if arr.dtype.kind == 'U' and arr.size:
            # Make one long string from all the characters. We need to
            # transpose here, because we're flattening the array, before
            # we write the bytes. The bytes have to be written in
            # Fortran order.
            n_chars = math.prod(shape)
            st_arr = np.ndarray(shape=(),
                                dtype=arr_dtype_number(arr, n_chars),
                                buffer=arr.T.copy())  # Fortran order
            # Recode with codec to give byte string
            st = st_arr.item().encode(codec)
            # Reconstruct as 1-D byte array
            arr = np.ndarray(shape=(len(st),),
                             dtype='S1',
                             buffer=st)
        self.write_element(arr, mdtype=miUTF8)

    def write_sparse(self, arr):
        ''' Sparse matrices are 2D
        '''
        A = arr.tocsc()  # convert to sparse CSC format
        A.sort_indices()     # MATLAB expects sorted row indices
        is_complex = (A.dtype.kind == 'c')
        is_logical = (A.dtype.kind == 'b')
        nz = A.nnz
        self.write_header(matdims(arr, self.oned_as),
                          mxSPARSE_CLASS,
                          is_complex=is_complex,
                          is_logical=is_logical,
                          # matlab won't load file with 0 nzmax
                          nzmax=1 if nz == 0 else nz)
        self.write_element(A.indices.astype('i4'))
        self.write_element(A.indptr.astype('i4'))
        self.write_element(A.data.real)
        if is_complex:
            self.write_element(A.data.imag)

    def write_cells(self, arr):
        self.write_header(matdims(arr, self.oned_as),
                          mxCELL_CLASS)
        # loop over data, column major
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            self.write(el)

    def write_empty_struct(self):
        self.write_header((1, 1), mxSTRUCT_CLASS)
        # max field name length set to 1 in an example matlab struct
        self.write_element(np.array(1, dtype=np.int32))
        # Field names element is empty
        self.write_element(np.array([], dtype=np.int8))

    def write_struct(self, arr):
        self.write_header(matdims(arr, self.oned_as),
                          mxSTRUCT_CLASS)
        self._write_items(arr)

    def _write_items(self, arr):
        # write fieldnames
        fieldnames = [f[0] for f in arr.dtype.descr]
        length = max([len(fieldname) for fieldname in fieldnames])+1
        max_length = (self.long_field_names and 64) or 32
        if length > max_length:
            raise ValueError("Field names are restricted to %d characters" %
                             (max_length-1))
        self.write_element(np.array([length], dtype='i4'))
        self.write_element(
            np.array(fieldnames, dtype='S%d' % (length)),
            mdtype=miINT8)
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            for f in fieldnames:
                self.write(el[f])

    def write_object(self, arr):
        '''Same as writing structs, except different mx class, and extra
        classname element after header
        '''
        self.write_header(matdims(arr, self.oned_as),
                          mxOBJECT_CLASS)
        self.write_element(np.array(arr.classname, dtype='S'),
                           mdtype=miINT8)
        self._write_items(arr)


class MatFile5Writer:
    ''' Class for writing mat5 files '''

    @docfiller
    def __init__(self, file_stream,
                 do_compression=False,
                 unicode_strings=False,
                 global_vars=None,
                 long_field_names=False,
                 oned_as='row'):
        ''' Initialize writer for matlab 5 format files

        Parameters
        ----------
        %(do_compression)s
        %(unicode_strings)s
        global_vars : None or sequence of strings, optional
            Names of variables to be marked as global for matlab
        %(long_fields)s
        %(oned_as)s
        '''
        self.file_stream = file_stream
        self.do_compression = do_compression
        self.unicode_strings = unicode_strings
        if global_vars:
            self.global_vars = global_vars
        else:
            self.global_vars = []
        self.long_field_names = long_field_names
        self.oned_as = oned_as
        self._matrix_writer = None

    def write_file_header(self):
        # write header
        hdr = np.zeros((), NDT_FILE_HDR)
        hdr['description'] = (f'MATLAB 5.0 MAT-file Platform: {os.name}, '
                              f'Created on: {time.asctime()}')
        hdr['version'] = 0x0100
        hdr['endian_test'] = np.ndarray(shape=(),
                                      dtype='S2',
                                      buffer=np.uint16(0x4d49))
        self.file_stream.write(hdr.tobytes())

    def put_variables(self, mdict, write_header=None):
        ''' Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` returns name, contents pairs where
           ``name`` which will appear in the matlab workspace in file load, and
           ``contents`` is something writeable to a matlab file, such as a NumPy
           array.
        write_header : {None, True, False}, optional
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        '''
        # write header if requested, or None and start of file
        if write_header is None:
            write_header = self.file_stream.tell() == 0
        if write_header:
            self.write_file_header()
        self._matrix_writer = VarWriter5(self)
        for name, var in mdict.items():
            if name[0] == '_':
                continue
            is_global = name in self.global_vars
            if self.do_compression:
                stream = BytesIO()
                self._matrix_writer.file_stream = stream
                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
                out_str = zlib.compress(stream.getvalue())
                tag = np.empty((), NDT_TAG_FULL)
                tag['mdtype'] = miCOMPRESSED
                tag['byte_count'] = len(out_str)
                self.file_stream.write(tag.tobytes())
                self.file_stream.write(out_str)
            else:  # not compressing
                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="byteordercodes",
                                   private_modules=["_byteordercodes"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'MatlabFunction', 'MatlabObject', 'MatlabOpaque', 'mat_struct',
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5_params",
                                   private_modules=["_mio5_params"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ["loadmat", "savemat", "whosmat"]  # noqa: F822

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio",
                                   private_modules=["_mio"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
MATLAB® file utilities (:mod:`scipy.io.matlab`)
===============================================

.. currentmodule:: scipy.io.matlab

This submodule is meant to provide lower-level file utilities related to reading
and writing MATLAB files.

.. autosummary::
   :toctree: generated/

   matfile_version - Get the MATLAB file version
   MatReadError - Exception indicating a read issue
   MatReadWarning - Warning class for read issues
   MatWriteError - Exception indicating a write issue
   mat_struct - Class used when ``struct_as_record=False``
   varmats_from_mat - Pull variables out of mat 5 file as a sequence of mat file objects

.. autosummary::
   :toctree: generated/
   :template: autosummary/ndarray_subclass.rst
   :nosignatures:

   MatlabObject - Class for a MATLAB object
   MatlabOpaque - Class for a MATLAB opaque matrix
   MatlabFunction - Class for a MATLAB function object

The following utilities that live in the :mod:`scipy.io`
namespace also exist in this namespace:

.. autosummary::
   :toctree: generated/

   loadmat - Read a MATLAB style mat file (version 4 through 7.1)
   savemat - Write a MATLAB style mat file (version 4 through 7.1)
   whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)

Notes
-----
MATLAB(R) is a registered trademark of The MathWorks, Inc., 3 Apple Hill
Drive, Natick, MA 01760-2098, USA.

"""
# Matlab file read and write utilities
from ._mio import loadmat, savemat, whosmat
from ._mio5 import MatlabFunction, varmats_from_mat
from ._mio5_params import MatlabObject, MatlabOpaque, mat_struct
from ._miobase import (matfile_version, MatReadError, MatReadWarning,
                      MatWriteError)

# Deprecated namespaces, to be removed in v2.0.0
from .import (mio, mio5, mio5_params, mio4, byteordercodes,
            miobase, mio_utils, streams, mio5_utils)

__all__ = [
    'loadmat', 'savemat', 'whosmat', 'MatlabObject',
    'matfile_version', 'MatReadError', 'MatReadWarning',
    'MatWriteError', 'mat_struct', 'MatlabOpaque', 'MatlabFunction', 'varmats_from_mat',
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ["MatReadError", "MatReadWarning", "MatWriteError"]  # noqa: F822

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="miobase",
                                   private_modules=["_miobase"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="streams",
                                   private_modules=["_streams"], all=__all__,
                                   attribute=name)
<|endoftext|>''' Classes for read / write of matlab (TM) 4 files
'''
import sys
import warnings
import math
from operator import mul

import numpy as np

import scipy.sparse

from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
                      convert_dtypes, arr_to_chars, arr_dtype_number)

from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce


__all__ = [
    'MatFile4Reader', 'MatFile4Writer', 'SYS_LITTLE_ENDIAN',
    'VarHeader4', 'VarReader4', 'VarWriter4', 'arr_to_2d', 'mclass_info',
    'mdtypes_template', 'miDOUBLE', 'miINT16', 'miINT32', 'miSINGLE',
    'miUINT16', 'miUINT8', 'mxCHAR_CLASS', 'mxFULL_CLASS', 'mxSPARSE_CLASS',
    'np_to_mtypes', 'order_codes'
]


SYS_LITTLE_ENDIAN = sys.byteorder == 'little'

miDOUBLE = 0
miSINGLE = 1
miINT32 = 2
miINT16 = 3
miUINT16 = 4
miUINT8 = 5

mdtypes_template = {
    miDOUBLE: 'f8',
    miSINGLE: 'f4',
    miINT32: 'i4',
    miINT16: 'i2',
    miUINT16: 'u2',
    miUINT8: 'u1',
    'header': [('mopt', 'i4'),
               ('mrows', 'i4'),
               ('ncols', 'i4'),
               ('imagf', 'i4'),
               ('namlen', 'i4')],
    'U1': 'U1',
    }

np_to_mtypes = {
    'f8': miDOUBLE,
    'c32': miDOUBLE,
    'c24': miDOUBLE,
    'c16': miDOUBLE,
    'f4': miSINGLE,
    'c8': miSINGLE,
    'i4': miINT32,
    'i2': miINT16,
    'u2': miUINT16,
    'u1': miUINT8,
    'S1': miUINT8,
    }

# matrix classes
mxFULL_CLASS = 0
mxCHAR_CLASS = 1
mxSPARSE_CLASS = 2

order_codes = {
    0: '<',
    1: '>',
    2: 'VAX D-float',  # !
    3: 'VAX G-float',
    4: 'Cray',  # !!
    }

mclass_info = {
    mxFULL_CLASS: 'double',
    mxCHAR_CLASS: 'char',
    mxSPARSE_CLASS: 'sparse',
    }


_MAX_INTP = np.iinfo(np.intp).max


class VarHeader4:
    # Mat4 variables never logical or global
    is_logical = False
    is_global = False

    def __init__(self,
                 name,
                 dtype,
                 mclass,
                 dims,
                 is_complex):
        self.name = name
        self.dtype = dtype
        self.mclass = mclass
        self.dims = dims
        self.is_complex = is_complex


class VarReader4:
    ''' Class to read matlab 4 variables '''

    def __init__(self, file_reader):
        self.file_reader = file_reader
        self.mat_stream = file_reader.mat_stream
        self.dtypes = file_reader.dtypes
        self.chars_as_strings = file_reader.chars_as_strings
        self.squeeze_me = file_reader.squeeze_me

    def read_header(self):
        ''' Read and return header for variable '''
        data = read_dtype(self.mat_stream, self.dtypes['header'])
        name = self.mat_stream.read(int(data['namlen'])).strip(b'\x00')
        if data['mopt'] < 0 or data['mopt'] > 5000:
            raise ValueError('Mat 4 mopt wrong format, byteswapping problem?')
        M, rest = divmod(data['mopt'], 1000)  # order code
        if M not in (0, 1):
            warnings.warn(f"We do not support byte ordering '{order_codes[M]}';"
                          " returned data may be corrupt",
                          UserWarning, stacklevel=3)
        O, rest = divmod(rest, 100)  # unused, should be 0
        if O != 0:
            raise ValueError('O in MOPT integer should be 0, wrong format?')
        P, rest = divmod(rest, 10)  # data type code e.g miDOUBLE (see above)
        T = rest  # matrix type code e.g., mxFULL_CLASS (see above)
        dims = (data['mrows'], data['ncols'])
        is_complex = data['imagf'] == 1
        dtype = self.dtypes[P]
        return VarHeader4(
            name,
            dtype,
            T,
            dims,
            is_complex)

    def array_from_header(self, hdr, process=True):
        mclass = hdr.mclass
        if mclass == mxFULL_CLASS:
            arr = self.read_full_array(hdr)
        elif mclass == mxCHAR_CLASS:
            arr = self.read_char_array(hdr)
            if process and self.chars_as_strings:
                arr = chars_to_strings(arr)
        elif mclass == mxSPARSE_CLASS:
            # no current processing (below) makes sense for sparse
            return self.read_sparse_array(hdr)
        else:
            raise TypeError(f'No reader for class code {mclass}')
        if process and self.squeeze_me:
            return squeeze_element(arr)
        return arr

    def read_sub_array(self, hdr, copy=True):
        ''' Mat4 read using header `hdr` dtype and dims

        Parameters
        ----------
        hdr : object
           object with attributes ``dtype``, ``dims``. dtype is assumed to be
           the correct endianness
        copy : bool, optional
           copies array before return if True (default True)
           (buffer is usually read only)

        Returns
        -------
        arr : ndarray
            of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``
        '''
        dt = hdr.dtype
        # Fast product for large (>2GB) arrays.
        num_bytes = reduce(mul, hdr.dims, np.int64(dt.itemsize))
        if num_bytes > _MAX_INTP:
            raise ValueError(
                f"Variable '{hdr.name.decode('latin1')}' has byte length "
                f"longer than largest possible NumPy array on this platform.")
        buffer = self.mat_stream.read(num_bytes)
        if len(buffer) != num_bytes:
            raise ValueError(
                f"Not enough bytes to read matrix "
                f"'{hdr.name.decode('latin1')}'; is this a badly-formed file? "
                f"Consider listing matrices with `whosmat` and loading named "
                f"matrices with `variable_names` kwarg to `loadmat`")
        arr = np.ndarray(shape=hdr.dims,
                         dtype=dt,
                         buffer=buffer,
                         order='F')
        if copy:
            arr = arr.copy()
        return arr

    def read_full_array(self, hdr):
        ''' Full (rather than sparse) matrix getter

        Read matrix (array) can be real or complex

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            complex array if ``hdr.is_complex`` is True, otherwise a real
            numeric array
        '''
        if hdr.is_complex:
            # avoid array copy to save memory
            res = self.read_sub_array(hdr, copy=False)
            res_j = self.read_sub_array(hdr, copy=False)
            return res + (res_j * 1j)
        return self.read_sub_array(hdr)

    def read_char_array(self, hdr):
        ''' latin-1 text matrix (char matrix) reader

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            with dtype 'U1', shape given by `hdr` ``dims``
        '''
        arr = self.read_sub_array(hdr).astype(np.uint8)
        S = arr.tobytes().decode('latin-1')
        return np.ndarray(shape=hdr.dims,
                          dtype=np.dtype('U1'),
                          buffer=np.array(S)).copy()

    def read_sparse_array(self, hdr):
        ''' Read and return sparse matrix type

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : coo_array
            with dtype ``float`` and shape read from the sparse array data

        Notes
        -----
        MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
        N is the number of non-zero values. Column 1 values [0:N] are the
        (1-based) row indices of the each non-zero value, column 2 [0:N] are the
        column indices, column 3 [0:N] are the (real) values. The last values
        [-1,0:2] of the rows, column indices are shape[0] and shape[1]
        respectively of the output matrix. The last value for the values column
        is a padding 0. mrows and ncols values from the header give the shape of
        the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column
        matrix, where the fourth column contains the imaginary component; the
        last value is again 0. Complex sparse data do *not* have the header
        ``imagf`` field set to True; the fact that the data are complex is only
        detectable because there are 4 storage columns.
        '''
        res = self.read_sub_array(hdr)
        tmp = res[:-1,:]
        # All numbers are float64 in Matlab, but SciPy sparse expects int shape
        dims = (int(res[-1,0]), int(res[-1,1]))
        I = np.ascontiguousarray(tmp[:,0],dtype='intc')  # fixes byte order also
        J = np.ascontiguousarray(tmp[:,1],dtype='intc')
        I -= 1  # for 1-based indexing
        J -= 1
        if res.shape[1] == 3:
            V = np.ascontiguousarray(tmp[:,2],dtype='float')
        else:
            V = np.ascontiguousarray(tmp[:,2],dtype='complex')
            V.imag = tmp[:,3]
        return scipy.sparse.coo_array((V,(I,J)), dims)

    def shape_from_header(self, hdr):
        '''Read the shape of the array described by the header.
        The file position after this call is unspecified.
        '''
        mclass = hdr.mclass
        if mclass == mxFULL_CLASS:
            shape = tuple(map(int, hdr.dims))
        elif mclass == mxCHAR_CLASS:
            shape = tuple(map(int, hdr.dims))
            if self.chars_as_strings:
                shape = shape[:-1]
        elif mclass == mxSPARSE_CLASS:
            dt = hdr.dtype
            dims = hdr.dims

            if not (len(dims) == 2 and dims[0] >= 1 and dims[1] >= 1):
                return ()

            # Read only the row and column counts
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            rows = np.ndarray(shape=(), dtype=dt,
                              buffer=self.mat_stream.read(dt.itemsize))
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            cols = np.ndarray(shape=(), dtype=dt,
                              buffer=self.mat_stream.read(dt.itemsize))

            shape = (int(rows), int(cols))
        else:
            raise TypeError(f'No reader for class code {mclass}')

        if self.squeeze_me:
            shape = tuple([x for x in shape if x != 1])
        return shape


class MatFile4Reader(MatFileReader):
    ''' Reader for Mat4 files '''
    @docfiller
    def __init__(self, mat_stream, *args, **kwargs):
        ''' Initialize matlab 4 file reader

    %(matstream_arg)s
    %(load_args)s
        '''
        super().__init__(mat_stream, *args, **kwargs)
        self._matrix_reader = None

    def guess_byte_order(self):
        self.mat_stream.seek(0)
        mopt = read_dtype(self.mat_stream, np.dtype('i4'))
        self.mat_stream.seek(0)
        if mopt == 0:
            return '<'
        if mopt < 0 or mopt > 5000:
            # Number must have been byteswapped
            return SYS_LITTLE_ENDIAN and '>' or '<'
        # Not byteswapped
        return SYS_LITTLE_ENDIAN and '<' or '>'

    def initialize_read(self):
        ''' Run when beginning read of variables

        Sets up readers from parameters in `self`
        '''
        self.dtypes = convert_dtypes(mdtypes_template, self.byte_order)
        self._matrix_reader = VarReader4(self)

    def read_var_header(self):
        ''' Read and return header, next position

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes ``name`` and ``is_global``
        next_position : int
           position in stream of next variable
        '''
        hdr = self._matrix_reader.read_header()
        # Fast product for large (>2GB) arrays.
        remaining_bytes = reduce(mul, hdr.dims, np.int64(hdr.dtype.itemsize))
        if hdr.is_complex and not hdr.mclass == mxSPARSE_CLASS:
            remaining_bytes *= 2
        next_position = self.mat_stream.tell() + remaining_bytes
        return hdr, next_position

    def read_var_array(self, header, process=True):
        ''' Read array, given `header`

        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False}, optional
           If True, apply recursive post-processing during loading of array.

        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        '''
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        ''' get variables from stream as dictionary

        Parameters
        ----------
        variable_names : None or str or sequence of str, optional
            variable name, or sequence of variable names to get from Mat file /
            file stream. If None, then get all variables in file.
        '''
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)
        self.mat_stream.seek(0)
        # set up variable reader
        self.initialize_read()
        mdict = {}
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            mdict[name] = self.read_var_array(hdr)
            self.mat_stream.seek(next_position)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        return mdict

    def list_variables(self):
        ''' list variables from stream '''
        self.mat_stream.seek(0)
        # set up variable reader
        self.initialize_read()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            shape = self._matrix_reader.shape_from_header(hdr)
            info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))

            self.mat_stream.seek(next_position)
        return vars


def arr_to_2d(arr, oned_as='row'):
    ''' Make ``arr`` exactly two dimensional

    If `arr` has more than 2 dimensions, raise a ValueError

    Parameters
    ----------
    arr : array
    oned_as : {'row', 'column'}, optional
       Whether to reshape 1-D vectors as row vectors or column vectors.
       See documentation for ``matdims`` for more detail

    Returns
    -------
    arr2d : array
       2-D version of the array
    '''
    dims = matdims(arr, oned_as)
    if len(dims) > 2:
        raise ValueError('Matlab 4 files cannot save arrays with more than '
                         '2 dimensions')
    return arr.reshape(dims)


class VarWriter4:
    def __init__(self, file_writer):
        self.file_stream = file_writer.file_stream
        self.oned_as = file_writer.oned_as

    def write_bytes(self, arr):
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        self.file_stream.write(s)

    def write_header(self, name, shape, P=miDOUBLE, T=mxFULL_CLASS, imagf=0):
        ''' Write header for given data options

        Parameters
        ----------
        name : str
            name of variable
        shape : sequence
            Shape of array as it will be read in matlab
        P : int, optional
            code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,
            miINT16, miUINT16, miUINT8``
        T : int, optional
            code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,
            mxSPARSE_CLASS``
        imagf : int, optional
            flag indicating complex
        '''
        header = np.empty((), mdtypes_template['header'])
        M = not SYS_LITTLE_ENDIAN
        O = 0
        header['mopt'] = (M * 1000 +
                          O * 100 +
                          P * 10 +
                          T)
        header['mrows'] = shape[0]
        header['ncols'] = shape[1]
        header['imagf'] = imagf
        header['namlen'] = len(name) + 1
        self.write_bytes(header)
        data = name + '\0'
        self.write_string(data.encode('latin1'))

    def write(self, arr, name):
        ''' Write matrix `arr`, with name `name`

        Parameters
        ----------
        arr : array_like
           array to write
        name : str
           name in matlab workspace
        '''
        # we need to catch sparse first, because np.asarray returns an
        # an object array for scipy.sparse
        if scipy.sparse.issparse(arr):
            self.write_sparse(arr, name)
            return
        arr = np.asarray(arr)
        dt = arr.dtype
        if not dt.isnative:
            arr = arr.astype(dt.newbyteorder('='))
        dtt = dt.type
        if dtt is np.object_:
            raise TypeError('Cannot save object arrays in Mat4')
        elif dtt is np.void:
            raise TypeError('Cannot save void type arrays')
        elif dtt in (np.str_, np.bytes_):
            self.write_char(arr, name)
            return
        self.write_numeric(arr, name)

    def write_numeric(self, arr, name):
        arr = arr_to_2d(arr, self.oned_as)
        imagf = arr.dtype.kind == 'c'
        try:
            P = np_to_mtypes[arr.dtype.str[1:]]
        except KeyError:
            if imagf:
                arr = arr.astype('c128')
            else:
                arr = arr.astype('f8')
            P = miDOUBLE
        self.write_header(name,
                          arr.shape,
                          P=P,
                          T=mxFULL_CLASS,
                          imagf=imagf)
        if imagf:
            self.write_bytes(arr.real)
            self.write_bytes(arr.imag)
        else:
            self.write_bytes(arr)

    def write_char(self, arr, name):
        if arr.dtype.type == np.str_ and arr.dtype.itemsize != np.dtype('U1').itemsize:
            arr = arr_to_chars(arr)
        arr = arr_to_2d(arr, self.oned_as)
        dims = arr.shape
        self.write_header(
            name,
            dims,
            P=miUINT8,
            T=mxCHAR_CLASS)
        if arr.dtype.kind == 'U':
            # Recode unicode to latin1
            n_chars = math.prod(dims)
            st_arr = np.ndarray(shape=(),
                                dtype=arr_dtype_number(arr, n_chars),
                                buffer=arr)
            st = st_arr.item().encode('latin-1')
            arr = np.ndarray(shape=dims, dtype='S1', buffer=st)
        self.write_bytes(arr)

    def write_sparse(self, arr, name):
        ''' Sparse matrices are 2-D

        See docstring for VarReader4.read_sparse_array
        '''
        A = arr.tocoo()  # convert to sparse COO format (ijv)
        imagf = A.dtype.kind == 'c'
        ijv = np.zeros((A.nnz + 1, 3+imagf), dtype='f8')
        ijv[:-1,0] = A.row
        ijv[:-1,1] = A.col
        ijv[:-1,0:2] += 1  # 1 based indexing
        if imagf:
            ijv[:-1,2] = A.data.real
            ijv[:-1,3] = A.data.imag
        else:
            ijv[:-1,2] = A.data
        ijv[-1,0:2] = A.shape
        self.write_header(
            name,
            ijv.shape,
            P=miDOUBLE,
            T=mxSPARSE_CLASS)
        self.write_bytes(ijv)


class MatFile4Writer:
    ''' Class for writing matlab 4 format files '''
    def __init__(self, file_stream, oned_as=None):
        self.file_stream = file_stream
        if oned_as is None:
            oned_as = 'row'
        self.oned_as = oned_as
        self._matrix_writer = None

    def put_variables(self, mdict, write_header=None):
        ''' Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` return name, contents pairs
           where ``name`` which will appeak in the matlab workspace in
           file load, and ``contents`` is something writeable to a
           matlab file, such as a NumPy array.
        write_header : {None, True, False}
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        '''
        # there is no header for a matlab 4 mat file, so we ignore the
        # ``write_header`` input argument. It's there for compatibility
        # with the matlab 5 version of this method
        self._matrix_writer = VarWriter4(self)
        for name, var in mdict.items():
            self._matrix_writer.write(var, name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio4",
                                   private_modules=["_mio4"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Module for reading and writing matlab (TM) .mat files
"""
# Authors: Travis Oliphant, Matthew Brett

from contextlib import contextmanager

from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer

__all__ = ['loadmat', 'savemat', 'whosmat']


@contextmanager
def _open_file_context(file_like, appendmat, mode='rb'):
    f, opened = _open_file(file_like, appendmat, mode)
    try:
        yield f
    finally:
        if opened:
            f.close()


def _open_file(file_like, appendmat, mode='rb'):
    """
    Open `file_like` and return as file-like object. First, check if object is
    already file-like; if so, return it as-is. Otherwise, try to pass it
    to open(). If that fails, and `file_like` is a string, and `appendmat` is true,
    append '.mat' and try again.
    """
    reqs = {'read'} if set(mode) & set('r+') else set()
    if set(mode) & set('wax+'):
        reqs.add('write')
    if reqs.issubset(dir(file_like)):
        return file_like, False

    try:
        return open(file_like, mode), True
    except OSError as e:
        # Probably "not found"
        if isinstance(file_like, str):
            if appendmat and not file_like.endswith('.mat'):
                file_like += '.mat'
            return open(file_like, mode), True
        else:
            raise OSError(
                'Reader needs file name or open file-like object'
            ) from e


@docfiller
def mat_reader_factory(file_name, appendmat=True, **kwargs):
    """
    Create reader for matlab .mat format files.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------
    matreader : MatFileReader object
       Initialized instance of MatFileReader class matching the mat file
       type detected in `filename`.
    file_opened : bool
       Whether the file was opened by this routine.

    """
    byte_stream, file_opened = _open_file(file_name, appendmat)
    mjv, mnv = _get_matfile_version(byte_stream)
    if mjv == 0:
        return MatFile4Reader(byte_stream, **kwargs), file_opened
    elif mjv == 1:
        return MatFile5Reader(byte_stream, **kwargs), file_opened
    elif mjv == 2:
        raise NotImplementedError('Please use HDF reader for matlab v7.3 '
                                  'files, e.g. h5py')
    else:
        raise TypeError(f'Did not recognize version {mjv}')


@docfiller
def loadmat(file_name, mdict=None, appendmat=True, *, spmatrix=True, **kwargs):
    """
    Load MATLAB file.

    Parameters
    ----------
    file_name : str
       Name of the mat file (do not need .mat extension if
       appendmat==True). Can also pass open file-like object.
    mdict : dict, optional
        Dictionary in which to insert matfile variables.
    appendmat : bool, optional
       True to append the .mat extension to the end of the given
       filename, if not already present. Default is True.
    spmatrix : bool, optional (default: True)
        If ``True``, return sparse ``coo_matrix``. Otherwise return ``coo_array``.
        Only relevant for sparse variables.
    byte_order : str or None, optional
       None by default, implying byte order guessed from mat
       file. Otherwise can be one of ('native', '=', 'little', '<',
       'BIG', '>').
    mat_dtype : bool, optional
       If True, return arrays in same dtype as would be loaded into
       MATLAB (instead of the dtype with which they are saved).
    squeeze_me : bool, optional
       Whether to squeeze unit matrix dimensions or not.
    chars_as_strings : bool, optional
       Whether to convert char arrays to string arrays.
    matlab_compatible : bool, optional
       Returns matrices as would be loaded by MATLAB (implies
       squeeze_me=False, chars_as_strings=False, mat_dtype=True,
       struct_as_record=True).
    struct_as_record : bool, optional
       Whether to load MATLAB structs as NumPy record arrays, or as
       old-style NumPy arrays with dtype=object. Setting this flag to
       False replicates the behavior of scipy version 0.7.x (returning
       NumPy object arrays). The default setting is True, because it
       allows easier round-trip load and save of MATLAB files.
    verify_compressed_data_integrity : bool, optional
        Whether the length of compressed sequences in the MATLAB file
        should be checked, to ensure that they are not longer than we expect.
        It is advisable to enable this (the default) because overlong
        compressed sequences in MATLAB files generally indicate that the
        files have experienced some sort of corruption.
    variable_names : None or sequence
        If None (the default) - read all variables in file. Otherwise,
        `variable_names` should be a sequence of strings, giving names of the
        MATLAB variables to read from the file. The reader will skip any
        variable with a name not in this sequence, possibly saving some read
        processing.
    simplify_cells : False, optional
        If True, return a simplified dict structure (which is useful if the mat
        file contains cell arrays). Note that this only affects the structure
        of the result and not its contents (which is identical for both output
        structures). If True, this automatically sets `struct_as_record` to
        False and `squeeze_me` to True, which is required to simplify cells.
    uint16_codec : str, optional
        The codec to use for decoding characters, which are stored as uint16
        values. The default uses the system encoding, but this can be manually
        set to other values such as 'ascii', 'latin1', and 'utf-8'. This
        parameter is relevant only for files stored as v6 and above, and not
        for files stored as v4.

    Returns
    -------
    mat_dict : dict
       dictionary with variable names as keys, and loaded matrices as values.

    Notes
    -----
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 Python library to read MATLAB 7.3 format mat
    files. Because SciPy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio

    Get the filename for an example .mat file from the tests/data directory.

    >>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    >>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

    Load the .mat file contents.

    >>> mat_contents = sio.loadmat(mat_fname, spmatrix=False)

    The result is a dictionary, one key/value pair for each variable:

    >>> sorted(mat_contents.keys())
    ['__globals__', '__header__', '__version__', 'testdouble']
    >>> mat_contents['testdouble']
    array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
            3.92699082, 4.71238898, 5.49778714, 6.28318531]])

    By default SciPy reads MATLAB structs as structured NumPy arrays where the
    dtype fields are of type `object` and the names correspond to the MATLAB
    struct field names. This can be disabled by setting the optional argument
    `struct_as_record=False`.

    Get the filename for an example .mat file that contains a MATLAB struct
    called `teststruct` and load the contents.

    >>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
    >>> matstruct_contents = sio.loadmat(matstruct_fname)
    >>> teststruct = matstruct_contents['teststruct']
    >>> teststruct.dtype
    dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])

    The size of the structured array is the size of the MATLAB struct, not the
    number of elements in any particular field. The shape defaults to 2-D
    unless the optional argument `squeeze_me=True`, in which case all length 1
    dimensions are removed.

    >>> teststruct.size
    1
    >>> teststruct.shape
    (1, 1)

    Get the 'stringfield' of the first element in the MATLAB struct.

    >>> teststruct[0, 0]['stringfield']
    array(['Rats live on no evil star.'],
      dtype='<U26')

    Get the first element of the 'doublefield'.

    >>> teststruct['doublefield'][0, 0]
    array([[ 1.41421356,  2.71828183,  3.14159265]])

    Load the MATLAB struct, squeezing out length 1 dimensions, and get the item
    from the 'complexfield'.

    >>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
    >>> matstruct_squeezed['teststruct'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].item()
    array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
        3.14159265+3.14159265j])
    """
    variable_names = kwargs.pop('variable_names', None)
    with _open_file_context(file_name, appendmat) as f:
        MR, _ = mat_reader_factory(f, **kwargs)
        matfile_dict = MR.get_variables(variable_names)
    if spmatrix:
        from scipy.sparse import issparse, coo_matrix
        for name, var in list(matfile_dict.items()):
            if issparse(var):
                matfile_dict[name] = coo_matrix(var)

    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict

    return mdict


@docfiller
def savemat(file_name, mdict,
            appendmat=True,
            format='5',
            long_field_names=False,
            do_compression=False,
            oned_as='row'):
    """
    Save a dictionary of names and arrays into a MATLAB-style .mat file.

    This saves the array objects in the given dictionary to a MATLAB-
    style .mat file.

    Parameters
    ----------
    file_name : str or file-like object
        Name of the .mat file (.mat extension not needed if ``appendmat ==
        True``).
        Can also pass open file_like object.
    mdict : dict
        Dictionary from which to save matfile variables.
    appendmat : bool, optional
        True (the default) to append the .mat extension to the end of the
        given filename, if not already present.
    format : {'5', '4'}, string, optional
        '5' (the default) for MATLAB 5 and up (to 7.2),
        '4' for MATLAB 4 .mat files.
    long_field_names : bool, optional
        False (the default) - maximum field name length in a structure is
        31 characters which is the documented maximum length.
        True - maximum field name length in a structure is 63 characters
        which works for MATLAB 7.6+.
    do_compression : bool, optional
        Whether or not to compress matrices on write. Default is False.
    oned_as : {'row', 'column'}, optional
        If 'column', write 1-D NumPy arrays as column vectors.
        If 'row', write 1-D NumPy arrays as row vectors.

    Examples
    --------
    >>> from scipy.io import savemat
    >>> import numpy as np
    >>> a = np.arange(20)
    >>> mdic = {"a": a, "label": "experiment"}
    >>> mdic
    {'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19]),
    'label': 'experiment'}
    >>> savemat("matlab_matrix.mat", mdic)
    """
    with _open_file_context(file_name, appendmat, 'wb') as file_stream:
        if format == '4':
            if long_field_names:
                message = "Long field names are not available for version 4 files"
                raise ValueError(message)
            MW = MatFile4Writer(file_stream, oned_as)
        elif format == '5':
            MW = MatFile5Writer(file_stream,
                                do_compression=do_compression,
                                unicode_strings=True,
                                long_field_names=long_field_names,
                                oned_as=oned_as)
        else:
            raise ValueError("Format should be '4' or '5'")
        MW.put_variables(mdict)


@docfiller
def whosmat(file_name, appendmat=True, **kwargs):
    """
    List variables inside a MATLAB file.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------
    variables : list of tuples
        A list of tuples, where each tuple holds the matrix name (a string),
        its shape (tuple of ints), and its data class (a string).
        Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
        int64, uint64, single, double, cell, struct, object, char, sparse,
        function, opaque, logical, unknown.

    Notes
    -----
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 python library to read matlab 7.3 format mat
    files (e.g. h5py). Because SciPy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    .. versionadded:: 0.12.0

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.io import savemat, whosmat

    Create some arrays, and use `savemat` to write them to a ``BytesIO``
    instance.

    >>> a = np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32)
    >>> b = np.geomspace(1, 10, 5)
    >>> f = BytesIO()
    >>> savemat(f, {'a': a, 'b': b})

    Use `whosmat` to inspect ``f``.  Each tuple in the output list gives
    the name, shape and data type of the array in ``f``.

    >>> whosmat(f)
    [('a', (2, 3), 'int32'), ('b', (1, 5), 'double')]

    """
    with _open_file_context(file_name, appendmat) as f:
        ML, file_opened = mat_reader_factory(f, **kwargs)
        variables = ML.list_variables()
    return variables
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5_utils",
                                   private_modules=["_mio5_utils"], all=__all__,
                                   attribute=name)
<|endoftext|>''' Constants and classes for matlab 5 read and write

See also mio5_utils.pyx where these same constants arise as c enums.

If you make changes in this file, don't forget to change mio5_utils.pyx
'''
import numpy as np

from ._miobase import convert_dtypes


__all__ = [
    'MDTYPES', 'MatlabFunction', 'MatlabObject', 'MatlabOpaque',
    'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'OPAQUE_DTYPE', 'codecs_template',
    'mat_struct', 'mclass_dtypes_template', 'mclass_info', 'mdtypes_template',
    'miCOMPRESSED', 'miDOUBLE', 'miINT16', 'miINT32', 'miINT64', 'miINT8',
    'miMATRIX', 'miSINGLE', 'miUINT16', 'miUINT32', 'miUINT64', 'miUINT8',
    'miUTF16', 'miUTF32', 'miUTF8', 'mxCELL_CLASS', 'mxCHAR_CLASS',
    'mxDOUBLE_CLASS', 'mxFUNCTION_CLASS', 'mxINT16_CLASS', 'mxINT32_CLASS',
    'mxINT64_CLASS', 'mxINT8_CLASS', 'mxOBJECT_CLASS',
    'mxOBJECT_CLASS_FROM_MATRIX_H', 'mxOPAQUE_CLASS', 'mxSINGLE_CLASS',
    'mxSPARSE_CLASS', 'mxSTRUCT_CLASS', 'mxUINT16_CLASS', 'mxUINT32_CLASS',
    'mxUINT64_CLASS', 'mxUINT8_CLASS'
]
miINT8 = 1
miUINT8 = 2
miINT16 = 3
miUINT16 = 4
miINT32 = 5
miUINT32 = 6
miSINGLE = 7
miDOUBLE = 9
miINT64 = 12
miUINT64 = 13
miMATRIX = 14
miCOMPRESSED = 15
miUTF8 = 16
miUTF16 = 17
miUTF32 = 18

mxCELL_CLASS = 1
mxSTRUCT_CLASS = 2
# The March 2008 edition of "Matlab 7 MAT-File Format" says that
# mxOBJECT_CLASS = 3, whereas matrix.h says that mxLOGICAL = 3.
# Matlab 2008a appears to save logicals as type 9, so we assume that
# the document is correct. See type 18, below.
mxOBJECT_CLASS = 3
mxCHAR_CLASS = 4
mxSPARSE_CLASS = 5
mxDOUBLE_CLASS = 6
mxSINGLE_CLASS = 7
mxINT8_CLASS = 8
mxUINT8_CLASS = 9
mxINT16_CLASS = 10
mxUINT16_CLASS = 11
mxINT32_CLASS = 12
mxUINT32_CLASS = 13
# The following are not in the March 2008 edition of "Matlab 7
# MAT-File Format," but were guessed from matrix.h.
mxINT64_CLASS = 14
mxUINT64_CLASS = 15
mxFUNCTION_CLASS = 16
# Not doing anything with these at the moment.
mxOPAQUE_CLASS = 17  # This appears to be a function workspace
# Thread 'saving/loading symbol table of annymous functions',
# octave-maintainers, April-May 2007
# https://lists.gnu.org/archive/html/octave-maintainers/2007-04/msg00031.html
# https://lists.gnu.org/archive/html/octave-maintainers/2007-05/msg00032.html
# (Was/Deprecated: https://www-old.cae.wisc.edu/pipermail/octave-maintainers/2007-May/002824.html)
mxOBJECT_CLASS_FROM_MATRIX_H = 18

mdtypes_template = {
    miINT8: 'i1',
    miUINT8: 'u1',
    miINT16: 'i2',
    miUINT16: 'u2',
    miINT32: 'i4',
    miUINT32: 'u4',
    miSINGLE: 'f4',
    miDOUBLE: 'f8',
    miINT64: 'i8',
    miUINT64: 'u8',
    miUTF8: 'u1',
    miUTF16: 'u2',
    miUTF32: 'u4',
    'file_header': [('description', 'S116'),
                    ('subsystem_offset', 'i8'),
                    ('version', 'u2'),
                    ('endian_test', 'S2')],
    'tag_full': [('mdtype', 'u4'), ('byte_count', 'u4')],
    'tag_smalldata':[('byte_count_mdtype', 'u4'), ('data', 'S4')],
    'array_flags': [('data_type', 'u4'),
                    ('byte_count', 'u4'),
                    ('flags_class','u4'),
                    ('nzmax', 'u4')],
    'U1': 'U1',
    }

mclass_dtypes_template = {
    mxINT8_CLASS: 'i1',
    mxUINT8_CLASS: 'u1',
    mxINT16_CLASS: 'i2',
    mxUINT16_CLASS: 'u2',
    mxINT32_CLASS: 'i4',
    mxUINT32_CLASS: 'u4',
    mxINT64_CLASS: 'i8',
    mxUINT64_CLASS: 'u8',
    mxSINGLE_CLASS: 'f4',
    mxDOUBLE_CLASS: 'f8',
    }

mclass_info = {
    mxINT8_CLASS: 'int8',
    mxUINT8_CLASS: 'uint8',
    mxINT16_CLASS: 'int16',
    mxUINT16_CLASS: 'uint16',
    mxINT32_CLASS: 'int32',
    mxUINT32_CLASS: 'uint32',
    mxINT64_CLASS: 'int64',
    mxUINT64_CLASS: 'uint64',
    mxSINGLE_CLASS: 'single',
    mxDOUBLE_CLASS: 'double',
    mxCELL_CLASS: 'cell',
    mxSTRUCT_CLASS: 'struct',
    mxOBJECT_CLASS: 'object',
    mxCHAR_CLASS: 'char',
    mxSPARSE_CLASS: 'sparse',
    mxFUNCTION_CLASS: 'function',
    mxOPAQUE_CLASS: 'opaque',
    }

NP_TO_MTYPES = {
    'f8': miDOUBLE,
    'c32': miDOUBLE,
    'c24': miDOUBLE,
    'c16': miDOUBLE,
    'f4': miSINGLE,
    'c8': miSINGLE,
    'i8': miINT64,
    'i4': miINT32,
    'i2': miINT16,
    'i1': miINT8,
    'u8': miUINT64,
    'u4': miUINT32,
    'u2': miUINT16,
    'u1': miUINT8,
    'S1': miUINT8,
    'U1': miUTF16,
    'b1': miUINT8,  # not standard but seems MATLAB uses this (gh-4022)
    }


NP_TO_MXTYPES = {
    'f8': mxDOUBLE_CLASS,
    'c32': mxDOUBLE_CLASS,
    'c24': mxDOUBLE_CLASS,
    'c16': mxDOUBLE_CLASS,
    'f4': mxSINGLE_CLASS,
    'c8': mxSINGLE_CLASS,
    'i8': mxINT64_CLASS,
    'i4': mxINT32_CLASS,
    'i2': mxINT16_CLASS,
    'i1': mxINT8_CLASS,
    'u8': mxUINT64_CLASS,
    'u4': mxUINT32_CLASS,
    'u2': mxUINT16_CLASS,
    'u1': mxUINT8_CLASS,
    'S1': mxUINT8_CLASS,
    'b1': mxUINT8_CLASS,  # not standard but seems MATLAB uses this
    }

''' Before release v7.1 (release 14) matlab (TM) used the system
default character encoding scheme padded out to 16-bits. Release 14
and later use Unicode. When saving character data, R14 checks if it
can be encoded in 7-bit ascii, and saves in that format if so.'''

codecs_template = {
    miUTF8: {'codec': 'utf_8', 'width': 1},
    miUTF16: {'codec': 'utf_16', 'width': 2},
    miUTF32: {'codec': 'utf_32','width': 4},
    }


def _convert_codecs(template, byte_order):
    ''' Convert codec template mapping to byte order

    Set codecs not on this system to None

    Parameters
    ----------
    template : mapping
       key, value are respectively codec name, and root name for codec
       (without byte order suffix)
    byte_order : {'<', '>'}
       code for little or big endian

    Returns
    -------
    codecs : dict
       key, value are name, codec (as in .encode(codec))
    '''
    codecs = {}
    postfix = byte_order == '<' and '_le' or '_be'
    for k, v in template.items():
        codec = v['codec']
        try:
            " ".encode(codec)
        except LookupError:
            codecs[k] = None
            continue
        if v['width'] > 1:
            codec += postfix
        codecs[k] = codec
    return codecs.copy()


MDTYPES = {}
for _bytecode in '<>':
    _def = {'dtypes': convert_dtypes(mdtypes_template, _bytecode),
            'classes': convert_dtypes(mclass_dtypes_template, _bytecode),
            'codecs': _convert_codecs(codecs_template, _bytecode)}
    MDTYPES[_bytecode] = _def


class mat_struct:
    """Placeholder for holding read data from structs.

    We use instances of this class when the user passes False as a value to the
    ``struct_as_record`` parameter of the :func:`scipy.io.loadmat` function.
    """
    pass


class MatlabObject(np.ndarray):
    """Subclass of ndarray to signal this is a matlab object.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be instantiated directly.
    """

    def __new__(cls, input_array, classname=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.classname = classname
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self,obj):
        # reset the attribute from passed original object
        self.classname = getattr(obj, 'classname', None)
        # We do not need to return anything


class MatlabFunction(np.ndarray):
    """Subclass for a MATLAB function.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


class MatlabOpaque(np.ndarray):
    """Subclass for a MATLAB opaque matrix.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


OPAQUE_DTYPE = np.dtype(
    [('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])
<|endoftext|># Copyright (C) 2022-2023 Adam Lugowski. All rights reserved.
# Use of this source code is governed by the BSD 2-clause license found in
# the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause
"""
Matrix Market I/O with a C++ backend.
See http://math.nist.gov/MatrixMarket/formats.html
for information about the Matrix Market format.

.. versionadded:: 1.12.0
"""
import io
import os

import numpy as np
from scipy.sparse import coo_array, issparse, coo_matrix
from scipy.io import _mmio

__all__ = ['mminfo', 'mmread', 'mmwrite']

PARALLELISM = 0
"""
Number of threads that `mmread()` and `mmwrite()` use.
0 means number of CPUs in the system.
Use `threadpoolctl` to set this value.
"""

ALWAYS_FIND_SYMMETRY = False
"""
Whether mmwrite() with symmetry='AUTO' will always search for symmetry
inside the matrix. This is scipy.io._mmio.mmwrite()'s default behavior,
but has a significant performance cost on large matrices.
"""

_field_to_dtype = {
    "integer": "int64",
    "unsigned-integer": "uint64",
    "real": "float64",
    "complex": "complex",
    "pattern": "float64",
}


def _fmm_version():
    from . import _fmm_core
    return _fmm_core.__version__


# Register with threadpoolctl, if available
try:
    import threadpoolctl

    class _FMMThreadPoolCtlController(threadpoolctl.LibController):
        user_api = "scipy"
        internal_api = "scipy_mmio"

        filename_prefixes = ("_fmm_core",)

        def get_num_threads(self):
            global PARALLELISM
            return PARALLELISM

        def set_num_threads(self, num_threads):
            global PARALLELISM
            PARALLELISM = num_threads

        def get_version(self):
            return _fmm_version

        def set_additional_attributes(self):
            pass

    threadpoolctl.register(_FMMThreadPoolCtlController)
except (ImportError, AttributeError):
    # threadpoolctl not installed or version too old
    pass


class _TextToBytesWrapper(io.BufferedReader):
    """
    Convert a TextIOBase string stream to a byte stream.
    """

    def __init__(self, text_io_buffer, encoding=None, errors=None, **kwargs):
        super().__init__(text_io_buffer, **kwargs)
        self.encoding = encoding or text_io_buffer.encoding or 'utf-8'
        self.errors = errors or text_io_buffer.errors or 'strict'

    def __del__(self):
        # do not close the wrapped stream
        self.detach()

    def _encoding_call(self, method_name, *args, **kwargs):
        raw_method = getattr(self.raw, method_name)
        val = raw_method(*args, **kwargs)
        return val.encode(self.encoding, errors=self.errors)

    def read(self, size=-1):
        return self._encoding_call('read', size)

    def read1(self, size=-1):
        return self._encoding_call('read1', size)

    def peek(self, size=-1):
        return self._encoding_call('peek', size)

    def seek(self, offset, whence=0):
        # Random seeks are not allowed because of non-trivial conversion
        # between byte and character offsets,
        # with the possibility of a byte offset landing within a character.
        if offset == 0 and whence == 0 or \
           offset == 0 and whence == 2:
            # seek to start or end is ok
            super().seek(offset, whence)
        else:
            # Drop any other seek
            # In this application this may happen when pystreambuf seeks during sync(),
            # which can happen when closing a partially-read stream.
            # Ex. when mminfo() only reads the header then exits.
            pass


def _read_body_array(cursor):
    """
    Read MatrixMarket array body
    """
    from . import _fmm_core

    vals = np.zeros(cursor.header.shape, dtype=_field_to_dtype.get(cursor.header.field))
    _fmm_core.read_body_array(cursor, vals)
    return vals


def _read_body_coo(cursor, generalize_symmetry=True):
    """
    Read MatrixMarket coordinate body
    """
    from . import _fmm_core

    index_dtype = "int32"
    if cursor.header.nrows >= 2**31 or cursor.header.ncols >= 2**31:
        # Dimensions are too large to fit in int32
        index_dtype = "int64"

    i = np.zeros(cursor.header.nnz, dtype=index_dtype)
    j = np.zeros(cursor.header.nnz, dtype=index_dtype)
    data = np.zeros(cursor.header.nnz, dtype=_field_to_dtype.get(cursor.header.field))

    _fmm_core.read_body_coo(cursor, i, j, data)

    if generalize_symmetry and cursor.header.symmetry != "general":
        off_diagonal_mask = (i != j)
        off_diagonal_rows = i[off_diagonal_mask]
        off_diagonal_cols = j[off_diagonal_mask]
        off_diagonal_data = data[off_diagonal_mask]

        if cursor.header.symmetry == "skew-symmetric":
            off_diagonal_data *= -1
        elif cursor.header.symmetry == "hermitian":
            off_diagonal_data = off_diagonal_data.conjugate()

        i = np.concatenate((i, off_diagonal_cols))
        j = np.concatenate((j, off_diagonal_rows))
        data = np.concatenate((data, off_diagonal_data))

    return (data, (i, j)), cursor.header.shape


def _get_read_cursor(source, parallelism=None):
    """
    Open file for reading.
    """
    from . import _fmm_core

    ret_stream_to_close = None
    if parallelism is None:
        parallelism = PARALLELISM

    try:
        source = os.fspath(source)
        # It's a file path
        is_path = True
    except TypeError:
        is_path = False

    if is_path:
        path = str(source)
        if path.endswith('.gz'):
            import gzip
            source = gzip.GzipFile(path, 'r')
            ret_stream_to_close = source
        elif path.endswith('.bz2'):
            import bz2
            source = bz2.BZ2File(path, 'rb')
            ret_stream_to_close = source
        else:
            return _fmm_core.open_read_file(path, parallelism), ret_stream_to_close

    # Stream object.
    if hasattr(source, "read"):
        if isinstance(source, io.TextIOBase):
            source = _TextToBytesWrapper(source)
        return _fmm_core.open_read_stream(source, parallelism), ret_stream_to_close
    else:
        raise TypeError("Unknown source type")


def _get_write_cursor(target, h=None, comment=None, parallelism=None,
                      symmetry="general", precision=None):
    """
    Open file for writing.
    """
    from . import _fmm_core

    if parallelism is None:
        parallelism = PARALLELISM
    if comment is None:
        comment = ''
    if symmetry is None:
        symmetry = "general"
    if precision is None:
        precision = -1

    if not h:
        h = _fmm_core.header(comment=comment, symmetry=symmetry)

    try:
        target = os.fspath(target)
        # It's a file path
        if target[-4:] != '.mtx':
            target += '.mtx'
        return _fmm_core.open_write_file(str(target), h, parallelism, precision)
    except TypeError:
        pass

    if hasattr(target, "write"):
        # Stream object.
        if isinstance(target, io.TextIOBase):
            raise TypeError("target stream must be open in binary mode.")
        return _fmm_core.open_write_stream(target, h, parallelism, precision)
    else:
        raise TypeError("Unknown source object")


def _apply_field(data, field, no_pattern=False):
    """
    Ensure that ``data.dtype`` is compatible with the specified MatrixMarket field type.

    Parameters
    ----------
    data : ndarray
        Input array.

    field : str
        Matrix Market field, such as 'real', 'complex', 'integer', 'pattern'.

    no_pattern : bool, optional
