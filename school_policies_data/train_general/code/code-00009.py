from collections.abc import Sequence
import os
import warnings

from pyarrow.pandas_compat import _pandas_api  # noqa
from pyarrow.lib import (Codec, Table,  # noqa
                         concat_tables, schema)
import pyarrow.lib as ext
from pyarrow import _feather
from pyarrow._feather import FeatherError  # noqa: F401


class FeatherDataset:
    """
    Encapsulates details of reading a list of Feather files.

    Parameters
    ----------
    path_or_paths : List[str]
        A list of file names
    validate_schema : bool, default True
        Check that individual file schemas are all the same / compatible
    """

    def __init__(self, path_or_paths, validate_schema=True):
        self.paths = path_or_paths
        self.validate_schema = validate_schema

    def read_table(self, columns=None):
        """
        Read multiple feather files as a single pyarrow.Table

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the file

        Returns
        -------
        pyarrow.Table
            Content of the file as a table (of columns)
        """
        _fil = _read_table_internal(self.paths[0], columns=columns)
        self._tables = [_fil]
        self.schema = _fil.schema

        for path in self.paths[1:]:
            table = _read_table_internal(path, columns=columns)
            if self.validate_schema:
                self.validate_schemas(path, table)
            self._tables.append(table)
        return concat_tables(self._tables)

    def validate_schemas(self, piece, table):
        if not self.schema.equals(table.schema):
            raise ValueError(f'Schema in {piece} was different. \n'
                             f'{self.schema}\n\nvs\n\n{table.schema}')

    def read_pandas(self, columns=None, use_threads=True):
        """
        Read multiple Parquet files as a single pandas DataFrame

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the file
        use_threads : bool, default True
            Use multiple threads when converting to pandas

        Returns
        -------
        pandas.DataFrame
            Content of the file as a pandas DataFrame (of columns)
        """
        return self.read_table(columns=columns).to_pandas(
            use_threads=use_threads)


def check_chunked_overflow(name, col):
    if col.num_chunks == 1:
        return

    if col.type in (ext.binary(), ext.string()):
        raise ValueError(f"Column '{name}' exceeds 2GB maximum capacity of "
                         "a Feather binary column. This restriction may be "
                         "lifted in the future")
    else:
        # TODO(wesm): Not sure when else this might be reached
        raise ValueError(
            f"Column '{name}' of type {col.type} was chunked on conversion to Arrow "
            "and cannot be currently written to Feather format"
        )


_FEATHER_SUPPORTED_CODECS = {'lz4', 'zstd', 'uncompressed'}


def write_feather(df, dest, compression=None, compression_level=None,
                  chunksize=None, version=2):
    """
    Write a pandas.DataFrame to Feather format.

    Parameters
    ----------
    df : pandas.DataFrame or pyarrow.Table
        Data to write out as Feather format.
    dest : str
        Local destination path.
    compression : string, default None
        Can be one of {"zstd", "lz4", "uncompressed"}. The default of None uses
        LZ4 for V2 files if it is available, otherwise uncompressed.
    compression_level : int, default None
        Use a compression level particular to the chosen compressor. If None
        use the default compression level
    chunksize : int, default None
        For V2 files, the internal maximum size of Arrow RecordBatch chunks
        when writing the Arrow IPC file format. None means use the default,
        which is currently 64K
    version : int, default 2
        Feather file version. Version 2 is the current. Version 1 is the more
        limited legacy format.

        .. deprecated:: 25.0.0
           Writing Feather V1 files is deprecated. Use the default
           ``version=2`` to write Arrow IPC files instead.
    """
    if version == 1:
        warnings.warn(
            "Feather V1 files are deprecated as of 25.0.0 and support will "
            "be removed in a future version. Use the default version=2 to "
            "write Arrow IPC files instead.",
            DeprecationWarning,
            stacklevel=2
        )
    if _pandas_api.have_pandas:
        if (_pandas_api.has_sparse and
                isinstance(df, _pandas_api.pd.SparseDataFrame)):
            df = df.to_dense()

    if _pandas_api.is_data_frame(df):
        # Feather v1 creates a new column in the resultant Table to
        # store index information if index type is not RangeIndex

        if version == 1:
            preserve_index = False
        elif version == 2:
            preserve_index = None
        else:
            raise ValueError("Version value should either be 1 or 2")

        table = Table.from_pandas(df, preserve_index=preserve_index)

        if version == 1:
            # Version 1 does not chunking
            for i, name in enumerate(table.schema.names):
                col = table[i]
                check_chunked_overflow(name, col)
    else:
        table = df

    if version == 1:
        if len(table.column_names) > len(set(table.column_names)):
            raise ValueError("cannot serialize duplicate column names")

        if compression is not None:
            raise ValueError("Feather V1 files do not support compression "
                             "option")

        if chunksize is not None:
            raise ValueError("Feather V1 files do not support chunksize "
                             "option")
    else:
        if compression is None and Codec.is_available('lz4_frame'):
            compression = 'lz4'
        elif (compression is not None and
              compression not in _FEATHER_SUPPORTED_CODECS):
            raise ValueError(f'compression="{compression}" not supported, must be '
                             f'one of {_FEATHER_SUPPORTED_CODECS}')

    try:
        _feather.write_feather(table, dest, compression=compression,
                               compression_level=compression_level,
                               chunksize=chunksize, version=version)
    except Exception:
        if isinstance(dest, str):
            try:
                os.remove(dest)
            except os.error:
                pass
        raise


def read_feather(source, columns=None, use_threads=True,
                 memory_map=False, **kwargs):
    """
    Read a pandas.DataFrame from Feather format. To read as pyarrow.Table use
    feather.read_table.

    Parameters
    ----------
    source : str file path, or file-like object
        You can use MemoryMappedFile as source, for explicitly use memory map.
    columns : sequence, optional
        Only read a specific set of columns. If not provided, all columns are
        read.
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads. If false the
        restriction is used in the conversion to Pandas as well as in the
        reading from Feather format.
    memory_map : boolean, default False
        Use memory mapping when opening file on disk, when source is a str.
    **kwargs
        Additional keyword arguments passed on to `pyarrow.Table.to_pandas`.

    Returns
    -------
    df : pandas.DataFrame
        The contents of the Feather file as a pandas.DataFrame
    """
    return (_read_table_internal(
        source, columns=columns, memory_map=memory_map,
        use_threads=use_threads).to_pandas(use_threads=use_threads, **kwargs))


def _read_table_internal(source, columns=None, memory_map=False,
                         use_threads=True):
    """
    Internal implementation for reading a Feather file as a pyarrow.Table.
    Emits a deprecation warning if the file is a legacy Feather V1 file.
    """
    reader = _feather.FeatherReader(
        source, use_memory_map=memory_map, use_threads=use_threads)

    if reader.version < 3:
        warnings.warn(
            "Feather V1 files are deprecated as of 25.0.0 and support will "
            "be removed in a future version. Consider rewriting this file "
            "in the Arrow IPC file format (Feather V2).",
            DeprecationWarning,
            stacklevel=3
        )

    if columns is None:
        return reader.read()

    if not isinstance(columns, Sequence):
        raise TypeError("Columns must be a sequence but, got {}"
                        .format(type(columns).__name__))

    column_types = [type(column) for column in columns]
    if all(map(lambda t: t == int, column_types)):
        table = reader.read_indices(columns)
    elif all(map(lambda t: t == str, column_types)):
        table = reader.read_names(columns)
    else:
        column_type_names = [t.__name__ for t in column_types]
        raise TypeError("Columns must be indices or names. "
                        f"Got columns {columns} of types {column_type_names}")

    # Feather v1 already respects the column selection
    if reader.version < 3:
        return table
    # Feather v2 reads with sorted / deduplicated selection
    elif sorted(set(columns)) == columns:
        return table
    else:
        # follow exact order / selection of names
        return table.select(columns)


def read_table(source, columns=None, memory_map=False, use_threads=True):
    """
    Read a pyarrow.Table from Feather format

    Parameters
    ----------
    source : str file path, or file-like object
        You can use MemoryMappedFile as source, for explicitly use memory map.
    columns : sequence, optional
        Only read a specific set of columns. If not provided, all columns are
        read.
    memory_map : boolean, default False
        Use memory mapping when opening file on disk, when source is a str
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.

    Returns
    -------
    table : pyarrow.Table
        The contents of the Feather file as a pyarrow.Table
    """
    return _read_table_internal(source, columns=columns,
                                memory_map=memory_map,
                                use_threads=use_threads)
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

# flake8: noqa

"""
PyArrow is the python implementation of Apache Arrow.

Apache Arrow is a cross-language development platform for in-memory data.
It specifies a standardized language-independent columnar memory format for
flat and hierarchical data, organized for efficient analytic operations on
modern hardware. It also provides computational libraries and zero-copy
streaming messaging and interprocess communication.

For more information see the official page at https://arrow.apache.org
"""

import importlib as _importlib
import os as _os
import platform as _platform
import sys as _sys

try:
    from ._generated_version import version as __version__
except ImportError:
    # Package is not installed, parse git tag at runtime
    try:
        import setuptools_scm
        # Code duplicated from setup.py to avoid a dependency on each other

        def parse_git(root, **kwargs):
            """
            Parse function for setuptools_scm that ignores tags for non-C++
            subprojects, e.g. apache-arrow-js-XXX tags.
            """
            from setuptools_scm.git import parse
            kwargs['describe_command'] = \
                "git describe --dirty --tags --long --match 'apache-arrow-[0-9]*.*'"
            return parse(root, **kwargs)
        __version__ = setuptools_scm.get_version('../',
                                                 parse=parse_git)
    except ImportError:
        __version__ = None

from pyarrow.lib import (BuildInfo, CppBuildInfo, RuntimeInfo, set_timezone_db_path,
                         MonthDayNano, VersionInfo, build_info, cpp_build_info,
                         cpp_version, cpp_version_info, runtime_info,
                         cpu_count, set_cpu_count, enable_signal_handlers,
                         io_thread_count, is_opentelemetry_enabled, set_io_thread_count)


def show_versions():
    """
    Print various version information, to help with error reporting.
    """
    def print_entry(label, value):
        print(f"{label: <26}: {value: <8}")

    print("pyarrow version info\n--------------------")
    print_entry("Package kind", build_info.cpp_build_info.package_kind
                if len(build_info.cpp_build_info.package_kind) > 0
                else "not indicated")
    print_entry("Arrow C++ library version", build_info.cpp_build_info.version)
    print_entry("Arrow C++ compiler",
                (f"{build_info.cpp_build_info.compiler_id} "
                 f"{build_info.cpp_build_info.compiler_version}"))
    print_entry("Arrow C++ compiler flags", build_info.cpp_build_info.compiler_flags)
    print_entry("Arrow C++ git revision", build_info.cpp_build_info.git_id)
    print_entry("Arrow C++ git description", build_info.cpp_build_info.git_description)
    print_entry("Arrow C++ build type", build_info.cpp_build_info.build_type)
    print_entry("PyArrow build type", build_info.build_type)


def _module_is_available(module):
    try:
        _importlib.import_module(f'pyarrow.{module}')
    except ImportError:
        return False
    else:
        return True


def _filesystem_is_available(fs):
    try:
        import pyarrow.fs
    except ImportError:
        return False

    try:
        getattr(pyarrow.fs, fs)
    except (ImportError, AttributeError):
        return False
    else:
        return True


def show_info():
    """
    Print detailed version and platform information, for error reporting
    """
    show_versions()

    def print_entry(label, value):
        print(f"  {label: <20}: {value: <8}")

    print("\nPlatform:")
    print_entry("OS / Arch", f"{_platform.system()} {_platform.machine()}")
    print_entry("SIMD Level", runtime_info().simd_level)
    print_entry("Detected SIMD Level", runtime_info().detected_simd_level)

    pool = default_memory_pool()
    print("\nMemory:")
    print_entry("Default backend", pool.backend_name)
    print_entry("Bytes allocated", f"{pool.bytes_allocated()} bytes")
    print_entry("Max memory", f"{pool.max_memory()} bytes")
    print_entry("Supported Backends", ', '.join(supported_memory_backends()))

    print("\nOptional modules:")
    modules = ["csv", "cuda", "dataset", "feather", "flight", "fs", "gandiva", "json",
               "orc", "parquet"]
    for module in modules:
        status = "Enabled" if _module_is_available(module) else "-"
        print(f"  {module: <20}: {status: <8}")
    print(f"  {'opentelemetry': <20}: {'Enabled' if is_opentelemetry_enabled() else '-': <8}")

    print("\nFilesystems:")
    filesystems = ["AzureFileSystem", "GcsFileSystem",
                   "HadoopFileSystem", "S3FileSystem"]
    for fs in filesystems:
        status = "Enabled" if _filesystem_is_available(fs) else "-"
        print(f"  {fs: <20}: {status: <8}")

    print("\nCompression Codecs:")
    codecs = ["brotli", "bz2", "gzip", "lz4_frame", "lz4", "snappy", "zstd"]
    for codec in codecs:
        status = "Enabled" if Codec.is_available(codec) else "-"
        print(f"  {codec: <20}: {status: <8}")


from pyarrow.lib import (null, bool_,
                         int8, int16, int32, int64,
                         uint8, uint16, uint32, uint64,
                         time32, time64, timestamp, date32, date64, duration,
                         month_day_nano_interval,
                         float16, float32, float64,
                         binary, string, utf8, binary_view, string_view,
                         large_binary, large_string, large_utf8,
                         decimal32, decimal64, decimal128, decimal256,
                         list_, large_list, list_view, large_list_view,
                         map_, struct,
                         union, sparse_union, dense_union,
                         dictionary,
                         run_end_encoded,
                         bool8, fixed_shape_tensor, json_, opaque, uuid,
                         field,
                         type_for_alias,
                         DataType, DictionaryType, StructType,
                         ListType, LargeListType, FixedSizeListType,
                         ListViewType, LargeListViewType,
                         MapType, UnionType, SparseUnionType, DenseUnionType,
                         TimestampType, Time32Type, Time64Type, DurationType,
                         FixedSizeBinaryType,
                         Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type,
                         BaseExtensionType, ExtensionType,
                         RunEndEncodedType, Bool8Type, FixedShapeTensorType,
                         JsonType, OpaqueType, UuidType,
                         UnknownExtensionType,
                         register_extension_type, unregister_extension_type,
                         DictionaryMemo,
                         KeyValueMetadata,
                         Field,
                         Schema,
                         schema,
                         unify_schemas,
                         Array, Tensor,
                         array, chunked_array, record_batch, nulls, repeat,
                         SparseCOOTensor, SparseCSRMatrix, SparseCSCMatrix,
                         SparseCSFTensor,
                         infer_type, from_numpy_dtype,
                         arange,
                         NullArray,
                         NumericArray, IntegerArray, FloatingPointArray,
                         BooleanArray,
                         Int8Array, UInt8Array,
                         Int16Array, UInt16Array,
                         Int32Array, UInt32Array,
                         Int64Array, UInt64Array,
                         HalfFloatArray, FloatArray, DoubleArray,
                         ListArray, LargeListArray, FixedSizeListArray,
                         ListViewArray, LargeListViewArray,
                         MapArray, UnionArray,
                         BinaryArray, StringArray,
                         LargeBinaryArray, LargeStringArray,
                         BinaryViewArray, StringViewArray,
                         FixedSizeBinaryArray,
                         DictionaryArray,
                         Date32Array, Date64Array, TimestampArray,
                         Time32Array, Time64Array, DurationArray,
                         MonthDayNanoIntervalArray,
                         Decimal32Array, Decimal64Array, Decimal128Array, Decimal256Array,
                         StructArray, ExtensionArray,
                         RunEndEncodedArray, Bool8Array, FixedShapeTensorArray,
                         JsonArray, OpaqueArray, UuidArray,
                         scalar, NA, _NULL as NULL, Scalar,
                         NullScalar, BooleanScalar,
                         Int8Scalar, Int16Scalar, Int32Scalar, Int64Scalar,
                         UInt8Scalar, UInt16Scalar, UInt32Scalar, UInt64Scalar,
                         HalfFloatScalar, FloatScalar, DoubleScalar,
                         Decimal32Scalar, Decimal64Scalar, Decimal128Scalar, Decimal256Scalar,
                         ListScalar, LargeListScalar, FixedSizeListScalar,
                         ListViewScalar, LargeListViewScalar,
                         Date32Scalar, Date64Scalar,
                         Time32Scalar, Time64Scalar,
                         TimestampScalar, DurationScalar,
                         MonthDayNanoIntervalScalar,
                         BinaryScalar, LargeBinaryScalar, BinaryViewScalar,
                         StringScalar, LargeStringScalar, StringViewScalar,
                         FixedSizeBinaryScalar, DictionaryScalar,
                         MapScalar, StructScalar, UnionScalar,
                         RunEndEncodedScalar, Bool8Scalar, ExtensionScalar,
                         FixedShapeTensorScalar, JsonScalar, OpaqueScalar, UuidScalar)

# Buffers, allocation
from pyarrow.lib import (DeviceAllocationType, Device, MemoryManager,
                         default_cpu_memory_manager)

from pyarrow.lib import (Buffer, ResizableBuffer, foreign_buffer, py_buffer,
                         Codec, compress, decompress, allocate_buffer)

from pyarrow.lib import (MemoryPool, LoggingMemoryPool, ProxyMemoryPool,
                         total_allocated_bytes, set_memory_pool,
                         default_memory_pool, system_memory_pool,
                         jemalloc_memory_pool, mimalloc_memory_pool,
                         logging_memory_pool, proxy_memory_pool,
                         log_memory_allocations, jemalloc_set_decay_ms,
                         supported_memory_backends)

# I/O
from pyarrow.lib import (NativeFile, PythonFile,
                         BufferedInputStream, BufferedOutputStream, CacheOptions,
                         CompressedInputStream, CompressedOutputStream,
                         TransformInputStream, transcoding_input_stream,
                         FixedSizeBufferWriter,
                         BufferReader, BufferOutputStream,
                         OSFile, MemoryMappedFile, memory_map,
                         create_memory_map, MockOutputStream,
                         input_stream, output_stream,
                         have_libhdfs)

from pyarrow.lib import (ChunkedArray, RecordBatch, Table, table,
                         concat_arrays, concat_tables, TableGroupBy,
                         RecordBatchReader, concat_batches)

# Exceptions
from pyarrow.lib import (ArrowCancelled,
                         ArrowCapacityError,
                         ArrowException,
                         ArrowKeyError,
                         ArrowIndexError,
                         ArrowInvalid,
                         ArrowIOError,
                         ArrowMemoryError,
                         ArrowNotImplementedError,
                         ArrowTypeError,
                         ArrowSerializationError)

from pyarrow.ipc import serialize_pandas, deserialize_pandas
import pyarrow.ipc as ipc

import pyarrow.types as types


# ----------------------------------------------------------------------
# Deprecations

from pyarrow.util import _deprecate_api, _deprecate_class


# TODO: Deprecate these somehow in the pyarrow namespace
from pyarrow.ipc import (Message, MessageReader, MetadataVersion,
                         RecordBatchFileReader, RecordBatchFileWriter,
                         RecordBatchStreamReader, RecordBatchStreamWriter)

# ----------------------------------------------------------------------
# Returning absolute path to the pyarrow include directory (if bundled, e.g. in
# wheels)


def get_include():
    """
    Return absolute path to directory containing Arrow C++ include
    headers. Similar to numpy.get_include
    """
    # Use pyarrow.lib location instead of just __file__. That works
    # for both editable and non-editable builds as it points
    # to the actual location of the compiled C++ extension.
    from pyarrow import lib as _lib
    return _os.path.join(_os.path.dirname(_lib.__file__), 'include')


def _get_pkg_config_executable():
    return _os.environ.get('PKG_CONFIG', 'pkg-config')


def _has_pkg_config(pkgname):
    import subprocess
    try:
        return subprocess.call([_get_pkg_config_executable(),
                                '--exists', pkgname]) == 0
    except FileNotFoundError:
        return False


def _read_pkg_config_variable(pkgname, cli_args):
    import subprocess
    cmd = [_get_pkg_config_executable(), pkgname] + cli_args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("pkg-config failed: " + err.decode('utf8'))
    return out.rstrip().decode('utf8')


def get_libraries():
    """
    Return list of library names to include in the `libraries` argument for C
    or Cython extensions using pyarrow
    """
    return ['arrow_python', 'arrow']


def create_library_symlinks():
    """
    With Linux and macOS wheels, the bundled shared libraries have an embedded
    ABI version like libarrow.so.17 or libarrow.17.dylib and so linking to them
    with -larrow won't work unless we create symlinks at locations like
    site-packages/pyarrow/libarrow.so. This unfortunate workaround addresses
    prior problems we had with shipping two copies of the shared libraries to
    permit third party projects like turbodbc to build their C++ extensions
    against the pyarrow wheels.

    This function must only be invoked once and only when the shared libraries
    are bundled with the Python package, which should only apply to wheel-based
    installs. It requires write access to the site-packages/pyarrow directory
    and so depending on your system may need to be run with root.
    """
    import glob
    if _sys.platform == 'win32':
        return
    package_cwd = _os.path.dirname(__file__)

    if _sys.platform == 'linux':
        bundled_libs = glob.glob(_os.path.join(package_cwd, '*.so.*'))

        def get_symlink_path(hard_path):
            return hard_path.rsplit('.', 1)[0]
    else:
        bundled_libs = glob.glob(_os.path.join(package_cwd, '*.*.dylib'))

        def get_symlink_path(hard_path):
            return '.'.join((hard_path.rsplit('.', 2)[0], 'dylib'))

    for lib_hard_path in bundled_libs:
        symlink_path = get_symlink_path(lib_hard_path)
        if _os.path.exists(symlink_path):
            continue
        try:
            _os.symlink(lib_hard_path, symlink_path)
        except PermissionError:
            print("Tried creating symlink {}. If you need to link to "
                  "bundled shared libraries, run "
                  "pyarrow.create_library_symlinks() as root")


def get_library_dirs():
    """
    Return lists of directories likely to contain Arrow C++ libraries for
    linking C or Cython extensions using pyarrow
    """
    # Use pyarrow.lib location instead of just __file__. That works
    # for both editable and non-editable builds as it points
    # to the actual location of the compiled C++ extension.
    from pyarrow import lib as _lib
    package_cwd = _os.path.dirname(_lib.__file__)
    library_dirs = [package_cwd]

    def append_library_dir(library_dir):
        if library_dir not in library_dirs:
            library_dirs.append(library_dir)

    # Search library paths via pkg-config. This is necessary if the user
    # installed libarrow and the other shared libraries manually and they
    # are not shipped inside the pyarrow package (see also ARROW-2976).
    pkg_config_executable = _os.environ.get('PKG_CONFIG') or 'pkg-config'
    for pkgname in ["arrow", "arrow_python"]:
        if _has_pkg_config(pkgname):
            library_dir = _read_pkg_config_variable(pkgname,
                                                    ["--libs-only-L"])
            # pkg-config output could be empty if Arrow is installed
            # as a system package.
            if library_dir:
                if not library_dir.startswith("-L"):
                    raise ValueError(
                        "pkg-config --libs-only-L returned unexpected "
                        f"value {library_dir!r}")
                append_library_dir(library_dir[2:])

    if _sys.platform == 'win32':
        # TODO(wesm): Is this necessary, or does setuptools within a conda
        # installation add Library\lib to the linker path for MSVC?
        python_base_install = _os.path.dirname(_sys.executable)
        library_dir = _os.path.join(python_base_install, 'Library', 'lib')

        if _os.path.exists(_os.path.join(library_dir, 'arrow.lib')):
            append_library_dir(library_dir)

        # GH-45530: Add pyarrow.libs dir containing delvewheel-mangled
        # msvcp140.dll
        pyarrow_libs_dir = _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), _os.pardir, "pyarrow.libs")
        )
        if _os.path.exists(pyarrow_libs_dir):
            append_library_dir(pyarrow_libs_dir)

    # ARROW-4074: Allow for ARROW_HOME to be set to some other directory
    if _os.environ.get('ARROW_HOME'):
        append_library_dir(_os.path.join(_os.environ['ARROW_HOME'], 'lib'))
    else:
        # Python wheels bundle the Arrow libraries in the pyarrow directory.
        append_library_dir(_os.path.dirname(_os.path.abspath(__file__)))

    return library_dirs
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

# ---------------------------------------------------------------------
# Implement Internal ExecPlan bindings

# cython: profile=False
# distutils: language = c++
# cython: language_level = 3

from pyarrow.lib import Table, RecordBatch, array
from pyarrow.compute import Expression, field

try:
    from pyarrow._acero import (  # noqa
        Declaration,
        ExecNodeOptions,
        TableSourceNodeOptions,
        FilterNodeOptions,
        ProjectNodeOptions,
        AggregateNodeOptions,
        OrderByNodeOptions,
        HashJoinNodeOptions,
        AsofJoinNodeOptions,
    )
except ImportError as exc:
    raise ImportError(
        f"The pyarrow installation is not built with support for 'acero' ({str(exc)})"
    ) from None


try:
    import pyarrow.dataset as ds
    from pyarrow._dataset import ScanNodeOptions
except ImportError:
    class DatasetModuleStub:
        class Dataset:
            pass

        class InMemoryDataset:
            pass
    ds = DatasetModuleStub


def _dataset_to_decl(dataset, use_threads=True, implicit_ordering=False):
    decl = Declaration("scan", ScanNodeOptions(
        dataset, use_threads=use_threads,
        implicit_ordering=implicit_ordering))

    # Get rid of special dataset columns
    # "__fragment_index", "__batch_index", "__last_in_fragment", "__filename"
    projections = [field(f) for f in dataset.schema.names]
    decl = Declaration.from_sequence(
        [decl, Declaration("project", ProjectNodeOptions(projections))]
    )

    filter_expr = dataset._scan_options.get("filter")
    if filter_expr is not None:
        # Filters applied in CScanNodeOptions are "best effort" for the scan node itself
        # so we always need to inject an additional Filter node to apply them for real.
        decl = Declaration.from_sequence(
            [decl, Declaration("filter", FilterNodeOptions(filter_expr))]
        )

    return decl


def _perform_join(join_type, left_operand, left_keys,
                  right_operand, right_keys,
                  left_suffix=None, right_suffix=None,
                  use_threads=True, coalesce_keys=False,
                  output_type=Table, filter_expression=None):
    """
    Perform join of two tables or datasets.

    The result will be an output table with the result of the join operation

    Parameters
    ----------
    join_type : str
        One of supported join types.
    left_operand : Table or Dataset
        The left operand for the join operation.
    left_keys : str or list[str]
        The left key (or keys) on which the join operation should be performed.
    right_operand : Table or Dataset
        The right operand for the join operation.
    right_keys : str or list[str]
        The right key (or keys) on which the join operation should be performed.
    left_suffix : str, default None
        Which suffix to add to left column names. This prevents confusion
        when the columns in left and right operands have colliding names.
    right_suffix : str, default None
        Which suffix to add to the right column names. This prevents confusion
        when the columns in left and right operands have colliding names.
    use_threads : bool, default True
        Whether to use multithreading or not.
    coalesce_keys : bool, default False
        If the duplicated keys should be omitted from one of the sides
        in the join result.
    output_type: Table or InMemoryDataset
        The output type for the exec plan result.
    filter_expression : pyarrow.compute.Expression
        Residual filter which is applied to matching row.

    Returns
    -------
    result_table : Table or InMemoryDataset
    """
    if not isinstance(left_operand, (Table, ds.Dataset)):
        raise TypeError(f"Expected Table or Dataset, got {type(left_operand)}")
    if not isinstance(right_operand, (Table, ds.Dataset)):
        raise TypeError(f"Expected Table or Dataset, got {type(right_operand)}")

    # Prepare left and right tables Keys to send them to the C++ function
    left_keys_order = {}
    if not isinstance(left_keys, (tuple, list)):
        left_keys = [left_keys]
    for idx, key in enumerate(left_keys):
        left_keys_order[key] = idx

    right_keys_order = {}
    if not isinstance(right_keys, (list, tuple)):
        right_keys = [right_keys]
    for idx, key in enumerate(right_keys):
        right_keys_order[key] = idx

    # By default expose all columns on both left and right table
    left_columns = left_operand.schema.names
    right_columns = right_operand.schema.names

    # Pick the join type
    if join_type == "left semi" or join_type == "left anti":
        right_columns = []
    elif join_type == "right semi" or join_type == "right anti":
        left_columns = []
    elif join_type == "inner" or join_type == "left outer":
        right_columns = [
            col for col in right_columns if col not in right_keys_order
        ]
    elif join_type == "right outer":
        left_columns = [
            col for col in left_columns if col not in left_keys_order
        ]

    # Turn the columns to vectors of FieldRefs
    # and set aside indices of keys.
    left_column_keys_indices = {}
    for idx, colname in enumerate(left_columns):
        if colname in left_keys:
            left_column_keys_indices[colname] = idx
    right_column_keys_indices = {}
    for idx, colname in enumerate(right_columns):
        if colname in right_keys:
            right_column_keys_indices[colname] = idx

    # Add the join node to the execplan
    if isinstance(left_operand, ds.Dataset):
        left_source = _dataset_to_decl(left_operand, use_threads=use_threads)
    else:
        left_source = Declaration("table_source", TableSourceNodeOptions(left_operand))
    if isinstance(right_operand, ds.Dataset):
        right_source = _dataset_to_decl(right_operand, use_threads=use_threads)
    else:
        right_source = Declaration(
            "table_source", TableSourceNodeOptions(right_operand)
        )

    if coalesce_keys:
        join_opts = HashJoinNodeOptions(
            join_type, left_keys, right_keys, left_columns, right_columns,
            output_suffix_for_left=left_suffix or "",
            output_suffix_for_right=right_suffix or "",
            filter_expression=filter_expression,
        )
    else:
        join_opts = HashJoinNodeOptions(
            join_type, left_keys, right_keys,
            output_suffix_for_left=left_suffix or "",
            output_suffix_for_right=right_suffix or "",
            filter_expression=filter_expression,
        )
    decl = Declaration(
        "hashjoin", options=join_opts, inputs=[left_source, right_source]
    )

    if coalesce_keys and join_type == "full outer":
        # In case of full outer joins, the join operation will output all columns
        # so that we can coalesce the keys and exclude duplicates in a subsequent
        # projection.
        left_columns_set = set(left_columns)
        right_columns_set = set(right_columns)
        # Where the right table columns start.
        right_operand_index = len(left_columns)
        projected_col_names = []
        projections = []
        for idx, col in enumerate(left_columns + right_columns):
            if idx < len(left_columns) and col in left_column_keys_indices:
                # Include keys only once and coalesce left+right table keys.
                projected_col_names.append(col)
                # Get the index of the right key that is being paired
                # with this left key. We do so by retrieving the name
                # of the right key that is in the same position in the provided keys
                # and then looking up the index for that name in the right table.
                right_key_index = right_column_keys_indices[
                    right_keys[left_keys_order[col]]]
                projections.append(
                    Expression._call("coalesce", [
                        Expression._field(idx), Expression._field(
                            right_operand_index+right_key_index)
                    ])
                )
            elif idx >= right_operand_index and col in right_column_keys_indices:
                # Do not include right table keys. As they would lead to duplicated keys
                continue
            else:
                # For all the other columns include them as they are.
                # Just recompute the suffixes that the join produced as the projection
                # would lose them otherwise.
                if (
                    left_suffix and idx < right_operand_index
                    and col in right_columns_set
                ):
                    col += left_suffix
                if (
                    right_suffix and idx >= right_operand_index
                    and col in left_columns_set
                ):
                    col += right_suffix
                projected_col_names.append(col)
                projections.append(
                    Expression._field(idx)
                )
        projection = Declaration(
            "project", ProjectNodeOptions(projections, projected_col_names)
        )
        decl = Declaration.from_sequence([decl, projection])

    result_table = decl.to_table(use_threads=use_threads)

    if output_type == Table:
        return result_table
    elif output_type == ds.InMemoryDataset:
        return ds.InMemoryDataset(result_table)
    else:
        raise TypeError("Unsupported output type")


def _perform_join_asof(left_operand, left_on, left_by,
                       right_operand, right_on, right_by,
                       tolerance, use_threads=True,
                       output_type=Table):
    """
    Perform asof join of two tables or datasets.

    The result will be an output table with the result of the join operation

    Parameters
    ----------
    left_operand : Table or Dataset
        The left operand for the join operation.
    left_on : str
        The left key (or keys) on which the join operation should be performed.
    left_by: str or list[str]
        The left key (or keys) on which the join operation should be performed.
    right_operand : Table or Dataset
        The right operand for the join operation.
    right_on : str or list[str]
        The right key (or keys) on which the join operation should be performed.
    right_by: str or list[str]
        The right key (or keys) on which the join operation should be performed.
    tolerance : int
        The tolerance to use for the asof join. The tolerance is interpreted in
        the same units as the "on" key.
    output_type: Table or InMemoryDataset
        The output type for the exec plan result.

    Returns
    -------
    result_table : Table or InMemoryDataset
    """
    if not isinstance(left_operand, (Table, ds.Dataset)):
        raise TypeError(f"Expected Table or Dataset, got {type(left_operand)}")
    if not isinstance(right_operand, (Table, ds.Dataset)):
        raise TypeError(f"Expected Table or Dataset, got {type(right_operand)}")

    if not isinstance(left_by, (tuple, list)):
        left_by = [left_by]
    if not isinstance(right_by, (tuple, list)):
        right_by = [right_by]

    # AsofJoin does not return on or by columns for right_operand.
    right_columns = [
        col for col in right_operand.schema.names
        if col not in [right_on] + right_by
    ]
    columns_collisions = set(left_operand.schema.names) & set(right_columns)
    if columns_collisions:
        raise ValueError(
            f"Columns {columns_collisions} present in both tables. "
            "AsofJoin does not support column collisions."
        )

    # Add the join node to the execplan
    if isinstance(left_operand, ds.Dataset):
        left_source = _dataset_to_decl(
            left_operand,
            use_threads=use_threads,
            implicit_ordering=True)
    else:
        left_source = Declaration(
            "table_source", TableSourceNodeOptions(left_operand),
        )
    if isinstance(right_operand, ds.Dataset):
        right_source = _dataset_to_decl(
            right_operand, use_threads=use_threads,
            implicit_ordering=True)
    else:
        right_source = Declaration(
            "table_source", TableSourceNodeOptions(right_operand)
        )

    join_opts = AsofJoinNodeOptions(
        left_on, left_by, right_on, right_by, tolerance
    )
    decl = Declaration(
        "asofjoin", options=join_opts, inputs=[left_source, right_source]
    )

    result_table = decl.to_table(use_threads=use_threads)

    if output_type == Table:
        return result_table
    elif output_type == ds.InMemoryDataset:
        return ds.InMemoryDataset(result_table)
    else:
        raise TypeError("Unsupported output type")


def _filter_table(table, expression):
    """Filter rows of a table based on the provided expression.

    The result will be an output table with only the rows matching
    the provided expression.

    Parameters
    ----------
    table : Table or RecordBatch
        Table that should be filtered.
    expression : Expression
        The expression on which rows should be filtered.

    Returns
    -------
    Table or RecordBatch
    """
    is_batch = False
    if isinstance(table, RecordBatch):
        table = Table.from_batches([table])
        is_batch = True

    decl = Declaration.from_sequence([
        Declaration("table_source", options=TableSourceNodeOptions(table)),
        Declaration("filter", options=FilterNodeOptions(expression))
    ])
    result = decl.to_table(use_threads=True)
    if is_batch:
        if result.num_rows > 0:
            result = result.combine_chunks().to_batches()[0]
        else:
            arrays = [array([], type=field.type) for field in result.schema]
            result = RecordBatch.from_arrays(arrays, schema=result.schema)
    return result


def _sort_source(table_or_dataset, sort_keys, output_type=Table, **kwargs):

    if isinstance(table_or_dataset, ds.Dataset):
        data_source = _dataset_to_decl(table_or_dataset, use_threads=True)
    else:
        data_source = Declaration(
            "table_source", TableSourceNodeOptions(table_or_dataset)
        )

    order_by = Declaration("order_by", OrderByNodeOptions(sort_keys, **kwargs))

    decl = Declaration.from_sequence([data_source, order_by])
    result_table = decl.to_table(use_threads=True)

    if output_type == Table:
        return result_table
    elif output_type == ds.InMemoryDataset:
        return ds.InMemoryDataset(result_table)
    else:
        raise TypeError("Unsupported output type")


def _group_by(table, aggregates, keys, use_threads=True):

    decl = Declaration.from_sequence([
        Declaration("table_source", TableSourceNodeOptions(table)),
        Declaration("aggregate", AggregateNodeOptions(aggregates, keys=keys))
    ])
    return decl.to_table(use_threads=use_threads)
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
"""
Functions to interact with Arrow memory allocated by Arrow Java.

These functions convert the objects holding the metadata, the actual
data is not copied at all.

This will only work with a JVM running in the same process such as provided
through jpype. Modules that talk to a remote JVM like py4j will not work as the
memory addresses reported by them are not reachable in the python process.
"""

import pyarrow as pa


class _JvmBufferNanny:
    """
    An object that keeps a org.apache.arrow.memory.ArrowBuf's underlying
    memory alive.
    """
    ref_manager = None

    def __init__(self, jvm_buf):
        ref_manager = jvm_buf.getReferenceManager()
        # Will raise a java.lang.IllegalArgumentException if the buffer
        # is already freed.  It seems that exception cannot easily be
        # caught...
        ref_manager.retain()
        self.ref_manager = ref_manager

    def __del__(self):
        if self.ref_manager is not None:
            self.ref_manager.release()


def jvm_buffer(jvm_buf):
    """
    Construct an Arrow buffer from org.apache.arrow.memory.ArrowBuf

    Parameters
    ----------

    jvm_buf: org.apache.arrow.memory.ArrowBuf
        Arrow Buffer representation on the JVM.

    Returns
    -------
    pyarrow.Buffer
        Python Buffer that references the JVM memory.
    """
    nanny = _JvmBufferNanny(jvm_buf)
    address = jvm_buf.memoryAddress()
    size = jvm_buf.capacity()
    return pa.foreign_buffer(address, size, base=nanny)


def _from_jvm_int_type(jvm_type):
    """
    Convert a JVM int type to its Python equivalent.

    Parameters
    ----------
    jvm_type : org.apache.arrow.vector.types.pojo.ArrowType$Int

    Returns
    -------
    typ : pyarrow.DataType
    """

    bit_width = jvm_type.getBitWidth()
    if jvm_type.getIsSigned():
        if bit_width == 8:
            return pa.int8()
        elif bit_width == 16:
            return pa.int16()
        elif bit_width == 32:
            return pa.int32()
        elif bit_width == 64:
            return pa.int64()
    else:
        if bit_width == 8:
            return pa.uint8()
        elif bit_width == 16:
            return pa.uint16()
        elif bit_width == 32:
            return pa.uint32()
        elif bit_width == 64:
            return pa.uint64()


def _from_jvm_float_type(jvm_type):
    """
    Convert a JVM float type to its Python equivalent.

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$FloatingPoint

    Returns
    -------
    typ: pyarrow.DataType
    """
    precision = jvm_type.getPrecision().toString()
    if precision == 'HALF':
        return pa.float16()
    elif precision == 'SINGLE':
        return pa.float32()
    elif precision == 'DOUBLE':
        return pa.float64()


def _from_jvm_time_type(jvm_type):
    """
    Convert a JVM time type to its Python equivalent.

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$Time

    Returns
    -------
    typ: pyarrow.DataType
    """
    time_unit = jvm_type.getUnit().toString()
    if time_unit == 'SECOND':
        assert jvm_type.getBitWidth() == 32
        return pa.time32('s')
    elif time_unit == 'MILLISECOND':
        assert jvm_type.getBitWidth() == 32
        return pa.time32('ms')
    elif time_unit == 'MICROSECOND':
        assert jvm_type.getBitWidth() == 64
        return pa.time64('us')
    elif time_unit == 'NANOSECOND':
        assert jvm_type.getBitWidth() == 64
        return pa.time64('ns')


def _from_jvm_timestamp_type(jvm_type):
    """
    Convert a JVM timestamp type to its Python equivalent.

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$Timestamp

    Returns
    -------
    typ: pyarrow.DataType
    """
    time_unit = jvm_type.getUnit().toString()
    timezone = jvm_type.getTimezone()
    if timezone is not None:
        timezone = str(timezone)
    if time_unit == 'SECOND':
        return pa.timestamp('s', tz=timezone)
    elif time_unit == 'MILLISECOND':
        return pa.timestamp('ms', tz=timezone)
    elif time_unit == 'MICROSECOND':
        return pa.timestamp('us', tz=timezone)
    elif time_unit == 'NANOSECOND':
        return pa.timestamp('ns', tz=timezone)


def _from_jvm_date_type(jvm_type):
    """
    Convert a JVM date type to its Python equivalent

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$Date

    Returns
    -------
    typ: pyarrow.DataType
    """
    day_unit = jvm_type.getUnit().toString()
    if day_unit == 'DAY':
        return pa.date32()
    elif day_unit == 'MILLISECOND':
        return pa.date64()


def field(jvm_field):
    """
    Construct a Field from a org.apache.arrow.vector.types.pojo.Field
    instance.

    Parameters
    ----------
    jvm_field: org.apache.arrow.vector.types.pojo.Field

    Returns
    -------
    pyarrow.Field
    """
    name = str(jvm_field.getName())
    jvm_type = jvm_field.getType()

    typ = None
    if not jvm_type.isComplex():
        type_str = jvm_type.getTypeID().toString()
        if type_str == 'Null':
            typ = pa.null()
        elif type_str == 'Int':
            typ = _from_jvm_int_type(jvm_type)
        elif type_str == 'FloatingPoint':
            typ = _from_jvm_float_type(jvm_type)
        elif type_str == 'Utf8':
            typ = pa.string()
        elif type_str == 'Binary':
            typ = pa.binary()
        elif type_str == 'FixedSizeBinary':
            typ = pa.binary(jvm_type.getByteWidth())
        elif type_str == 'Bool':
            typ = pa.bool_()
        elif type_str == 'Time':
            typ = _from_jvm_time_type(jvm_type)
        elif type_str == 'Timestamp':
            typ = _from_jvm_timestamp_type(jvm_type)
        elif type_str == 'Date':
            typ = _from_jvm_date_type(jvm_type)
        elif type_str == 'Decimal':
            typ = pa.decimal128(jvm_type.getPrecision(), jvm_type.getScale())
        else:
            raise NotImplementedError(
                f"Unsupported JVM type: {type_str}")
    else:
        # TODO: The following JVM types are not implemented:
        #       Struct, List, FixedSizeList, Union, Dictionary
        raise NotImplementedError(
            "JVM field conversion only implemented for primitive types.")

    nullable = jvm_field.isNullable()
    jvm_metadata = jvm_field.getMetadata()
    if jvm_metadata.isEmpty():
        metadata = None
    else:
        metadata = {str(entry.getKey()): str(entry.getValue())
                    for entry in jvm_metadata.entrySet()}
    return pa.field(name, typ, nullable, metadata)


def schema(jvm_schema):
    """
    Construct a Schema from a org.apache.arrow.vector.types.pojo.Schema
    instance.

    Parameters
    ----------
    jvm_schema: org.apache.arrow.vector.types.pojo.Schema

    Returns
    -------
    pyarrow.Schema
    """
    fields = jvm_schema.getFields()
    fields = [field(f) for f in fields]
    jvm_metadata = jvm_schema.getCustomMetadata()
    if jvm_metadata.isEmpty():
        metadata = None
    else:
        metadata = {str(entry.getKey()): str(entry.getValue())
                    for entry in jvm_metadata.entrySet()}
    return pa.schema(fields, metadata)


def array(jvm_array):
    """
    Construct an (Python) Array from its JVM equivalent.

    Parameters
    ----------
    jvm_array : org.apache.arrow.vector.ValueVector

    Returns
    -------
    array : Array
    """
    if jvm_array.getField().getType().isComplex():
        minor_type_str = jvm_array.getMinorType().toString()
        raise NotImplementedError(
            f"Cannot convert JVM Arrow array of type {minor_type_str}, "
            "complex types not yet implemented.")
    dtype = field(jvm_array.getField()).type
    buffers = [jvm_buffer(buf)
               for buf in list(jvm_array.getBuffers(False))]

    # If JVM has an empty Vector, buffer list will be empty so create manually
    if len(buffers) == 0:
        return pa.array([], type=dtype)

    length = jvm_array.getValueCount()
    null_count = jvm_array.getNullCount()
    return pa.Array.from_buffers(dtype, length, buffers, null_count)


def record_batch(jvm_vector_schema_root):
    """
    Construct a (Python) RecordBatch from a JVM VectorSchemaRoot

    Parameters
    ----------
    jvm_vector_schema_root : org.apache.arrow.vector.VectorSchemaRoot

    Returns
    -------
    record_batch: pyarrow.RecordBatch
    """
    pa_schema = schema(jvm_vector_schema_root.getSchema())

    arrays = []
    for name in pa_schema.names:
        arrays.append(array(jvm_vector_schema_root.getVector(name)))

    return pa.RecordBatch.from_arrays(
        arrays,
        pa_schema.names,
        metadata=pa_schema.metadata
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

"""Dataset is currently unstable. APIs subject to change without notice."""

import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like

try:
    from pyarrow._dataset import (  # noqa
        CsvFileFormat,
        CsvFragmentScanOptions,
        JsonFileFormat,
        JsonFragmentScanOptions,
        Dataset,
        DatasetFactory,
        DirectoryPartitioning,
        FeatherFileFormat,
        FilenamePartitioning,
        FileFormat,
        FileFragment,
        FileSystemDataset,
        FileSystemDatasetFactory,
        FileSystemFactoryOptions,
        FileWriteOptions,
        Fragment,
        FragmentScanOptions,
        HivePartitioning,
        IpcFileFormat,
        IpcFileWriteOptions,
        InMemoryDataset,
        Partitioning,
        PartitioningFactory,
        Scanner,
        TaggedRecordBatch,
        UnionDataset,
        UnionDatasetFactory,
        WrittenFile,
        get_partition_keys,
        get_partition_keys as _get_partition_keys,  # keep for backwards compatibility
        _filesystemdataset_write,
    )
except ImportError as exc:
    raise ImportError(
        f"The pyarrow installation is not built with support for 'dataset' ({str(exc)})"
    ) from None

# keep Expression functionality exposed here for backwards compatibility
from pyarrow.compute import Expression, scalar, field  # noqa


_orc_available = False
_orc_msg = (
    "The pyarrow installation is not built with support for the ORC file "
    "format."
)

try:
    from pyarrow._dataset_orc import OrcFileFormat
    _orc_available = True
except ImportError:
    pass

_parquet_available = False
_parquet_msg = (
    "The pyarrow installation is not built with support for the Parquet file "
    "format."
)

try:
    from pyarrow._dataset_parquet import (  # noqa
        ParquetDatasetFactory,
        ParquetFactoryOptions,
        ParquetFileFormat,
        ParquetFileFragment,
        ParquetFileWriteOptions,
        ParquetFragmentScanOptions,
        ParquetReadOptions,
        RowGroupInfo,
    )
    _parquet_available = True
except ImportError:
    pass


try:
    from pyarrow._dataset_parquet_encryption import (  # noqa
        ParquetDecryptionConfig,
        ParquetEncryptionConfig,
    )
except ImportError:
    pass


def __getattr__(name):
    if name == "OrcFileFormat" and not _orc_available:
        raise ImportError(_orc_msg)

    if name == "ParquetFileFormat" and not _parquet_available:
        raise ImportError(_parquet_msg)

    raise AttributeError(
        f"module 'pyarrow.dataset' has no attribute '{name}'"
    )


def partitioning(schema=None, field_names=None, flavor=None,
                 dictionaries=None):
    """
    Specify a partitioning scheme.

    The supported schemes include:

    - "DirectoryPartitioning": this scheme expects one segment in the file path
      for each field in the specified schema (all fields are required to be
      present). For example given schema<year:int16, month:int8> the path
      "/2009/11" would be parsed to ("year"_ == 2009 and "month"_ == 11).
    - "HivePartitioning": a scheme for "/$key=$value/" nested directories as
      found in Apache Hive. This is a multi-level, directory based partitioning
      scheme. Data is partitioned by static values of a particular column in
      the schema. Partition keys are represented in the form $key=$value in
      directory names. Field order is ignored, as are missing or unrecognized
      field names.
      For example, given schema<year:int16, month:int8, day:int8>, a possible
      path would be "/year=2009/month=11/day=15" (but the field order does not
      need to match).
    - "FilenamePartitioning": this scheme expects the partitions will have
      filenames containing the field values separated by "_".
      For example, given schema<year:int16, month:int8, day:int8>, a possible
      partition filename "2009_11_part-0.parquet" would be parsed
      to ("year"_ == 2009 and "month"_ == 11).

    Parameters
    ----------
    schema : pyarrow.Schema, default None
        The schema that describes the partitions present in the file path.
        If not specified, and `field_names` and/or `flavor` are specified,
        the schema will be inferred from the file path (and a
        PartitioningFactory is returned).
    field_names :  list of str, default None
        A list of strings (field names). If specified, the schema's types are
        inferred from the file paths (only valid for DirectoryPartitioning).
    flavor : str, default None
        The default is DirectoryPartitioning. Specify ``flavor="hive"`` for
        a HivePartitioning, and ``flavor="filename"`` for a
        FilenamePartitioning.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing. Alternatively, pass `infer` to have
        Arrow discover the dictionary values, in which case a
        PartitioningFactory is returned.

    Returns
    -------
    Partitioning or PartitioningFactory
        The partitioning scheme

    Examples
    --------

    Specify the Schema for paths like "/2009/June":

    >>> import pyarrow as pa
    >>> import pyarrow.dataset as ds
    >>> part = ds.partitioning(pa.schema([("year", pa.int16()),
    ...                                   ("month", pa.string())]))

    or let the types be inferred by only specifying the field names:

    >>> part =  ds.partitioning(field_names=["year", "month"])

    For paths like "/2009/June", the year will be inferred as int32 while month
    will be inferred as string.

    Specify a Schema with dictionary encoding, providing dictionary values:

    >>> part = ds.partitioning(
    ...     pa.schema([
    ...         ("year", pa.int16()),
    ...         ("month", pa.dictionary(pa.int8(), pa.string()))
    ...     ]),
    ...     dictionaries={
    ...         "month": pa.array(["January", "February", "March"]),
    ...     })

    Alternatively, specify a Schema with dictionary encoding, but have Arrow
    infer the dictionary values:

    >>> part = ds.partitioning(
    ...     pa.schema([
    ...         ("year", pa.int16()),
    ...         ("month", pa.dictionary(pa.int8(), pa.string()))
    ...     ]),
    ...     dictionaries="infer")

    Create a Hive scheme for a path like "/year=2009/month=11":

    >>> part = ds.partitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())]),
    ...     flavor="hive")

    A Hive scheme can also be discovered from the directory structure (and
    types will be inferred):

    >>> part = ds.partitioning(flavor="hive")
    """
    if flavor is None:
        # default flavor
        if schema is not None:
            if field_names is not None:
                raise ValueError(
                    "Cannot specify both 'schema' and 'field_names'")
            if dictionaries == 'infer':
                return DirectoryPartitioning.discover(schema=schema)
            return DirectoryPartitioning(schema, dictionaries)
        elif field_names is not None:
            if isinstance(field_names, list):
                return DirectoryPartitioning.discover(field_names)
            else:
                raise ValueError(
                    f"Expected list of field names, got {type(field_names)}")
        else:
            raise ValueError(
                "For the default directory flavor, need to specify "
                "a Schema or a list of field names")
    if flavor == "filename":
        if schema is not None:
            if field_names is not None:
                raise ValueError(
                    "Cannot specify both 'schema' and 'field_names'")
            if dictionaries == 'infer':
                return FilenamePartitioning.discover(schema=schema)
            return FilenamePartitioning(schema, dictionaries)
        elif field_names is not None:
            if isinstance(field_names, list):
                return FilenamePartitioning.discover(field_names)
            else:
                raise ValueError(
                    f"Expected list of field names, got {type(field_names)}")
        else:
            raise ValueError(
                "For the filename flavor, need to specify "
                "a Schema or a list of field names")
    elif flavor == 'hive':
        if field_names is not None:
            raise ValueError("Cannot specify 'field_names' for flavor 'hive'")
        elif schema is not None:
            if isinstance(schema, pa.Schema):
                if dictionaries == 'infer':
                    return HivePartitioning.discover(schema=schema)
                return HivePartitioning(schema, dictionaries)
            else:
                raise ValueError(
                    f"Expected Schema for 'schema', got {type(schema)}")
        else:
            return HivePartitioning.discover()
    else:
        raise ValueError("Unsupported flavor")


def _ensure_partitioning(scheme):
    """
    Validate input and return a Partitioning(Factory).

    It passes None through if no partitioning scheme is defined.
    """
    if scheme is None:
        pass
    elif isinstance(scheme, str):
        scheme = partitioning(flavor=scheme)
    elif isinstance(scheme, list):
        scheme = partitioning(field_names=scheme)
    elif isinstance(scheme, (Partitioning, PartitioningFactory)):
        pass
    else:
        raise ValueError(
            f"Expected Partitioning or PartitioningFactory, got {type(scheme)}")
    return scheme


def _ensure_format(obj):
    if isinstance(obj, FileFormat):
        return obj
    elif obj == "parquet":
        if not _parquet_available:
            raise ValueError(_parquet_msg)
        return ParquetFileFormat()
    elif obj in {"ipc", "arrow"}:
        return IpcFileFormat()
    elif obj == "feather":
        return FeatherFileFormat()
    elif obj == "csv":
        return CsvFileFormat()
    elif obj == "orc":
        if not _orc_available:
            raise ValueError(_orc_msg)
        return OrcFileFormat()
    elif obj == "json":
        return JsonFileFormat()
    else:
        raise ValueError(f"format '{obj}' is not supported")


def _ensure_multiple_sources(paths, filesystem=None):
    """
    Treat a list of paths as files belonging to a single file system

    If the file system is local then also validates that all paths
    are referencing existing *files* otherwise any non-file paths will be
    silently skipped (for example on a remote filesystem).

    Parameters
    ----------
    paths : list of path-like
        Note that URIs are not allowed.
    filesystem : FileSystem or str, optional
        If an URI is passed, then its path component will act as a prefix for
        the file paths.

    Returns
    -------
    (FileSystem, list of str)
        File system object and a list of normalized paths.

    Raises
    ------
    TypeError
        If the passed filesystem has wrong type.
    IOError
        If the file system is local and a referenced path is not available or
        not a file.
    """
    from pyarrow.fs import (
        LocalFileSystem, SubTreeFileSystem, _MockFileSystem, FileType,
        _ensure_filesystem
    )

    if filesystem is None:
        # fall back to local file system as the default
        filesystem = LocalFileSystem()
    else:
        # construct a filesystem if it is a valid URI
        filesystem = _ensure_filesystem(filesystem)

    is_local = (
        isinstance(filesystem, (LocalFileSystem, _MockFileSystem)) or
        (isinstance(filesystem, SubTreeFileSystem) and
         isinstance(filesystem.base_fs, LocalFileSystem))
    )

    # allow normalizing irregular paths such as Windows local paths
    paths = [filesystem.normalize_path(_stringify_path(p)) for p in paths]

    # validate that all of the paths are pointing to existing *files*
    # possible improvement is to group the file_infos by type and raise for
    # multiple paths per error category
    if is_local:
        for info in filesystem.get_file_info(paths):
            file_type = info.type
            if file_type == FileType.File:
                continue
            elif file_type == FileType.NotFound:
                raise FileNotFoundError(info.path)
            elif file_type == FileType.Directory:
                raise IsADirectoryError(
                    f'Path {info.path} points to a directory, but only file paths are '
                    'supported. To construct a nested or union dataset pass '
                    'a list of dataset objects instead.'
                )
            else:
                raise IOError(
                    f'Path {info.path} exists but its type is unknown (could be a '
                    'special file such as a Unix socket or character device, '
                    'or Windows NUL / CON / ...)')

    return filesystem, paths


def _ensure_single_source(path, filesystem=None):
    """
    Treat path as either a recursively traversable directory or a single file.

    Parameters
    ----------
    path : path-like
    filesystem : FileSystem or str, optional
        If an URI is passed, then its path component will act as a prefix for
        the file paths.

    Returns
    -------
    (FileSystem, list of str or fs.Selector)
        File system object and either a single item list pointing to a file or
        an fs.Selector object pointing to a directory.

    Raises
    ------
    TypeError
        If the passed filesystem has wrong type.
    FileNotFoundError
        If the referenced file or directory doesn't exist.
    """
    from pyarrow.fs import FileType, FileSelector, _resolve_filesystem_and_path

    # at this point we already checked that `path` is a path-like
    filesystem, path = _resolve_filesystem_and_path(path, filesystem)

    # ensure that the path is normalized before passing to dataset discovery
    path = filesystem.normalize_path(path)

    # retrieve the file descriptor
    file_info = filesystem.get_file_info(path)

    # depending on the path type either return with a recursive
    # directory selector or as a list containing a single file
    if file_info.type == FileType.Directory:
        paths_or_selector = FileSelector(path, recursive=True)
    elif file_info.type == FileType.File:
        paths_or_selector = [path]
    else:
        raise FileNotFoundError(path)

    return filesystem, paths_or_selector


def _filesystem_dataset(source, schema=None, filesystem=None,
                        partitioning=None, format=None,
                        partition_base_dir=None, exclude_invalid_files=None,
                        selector_ignore_prefixes=None):
    """
    Create a FileSystemDataset which can be used to build a Dataset.

    Parameters are documented in the dataset function.

    Returns
    -------
    FileSystemDataset
    """
    from pyarrow.fs import LocalFileSystem, _ensure_filesystem, FileInfo

    format = _ensure_format(format or 'parquet')
    partitioning = _ensure_partitioning(partitioning)

    if isinstance(source, (list, tuple)):
        if source and isinstance(source[0], FileInfo):
            if filesystem is None:
                # fall back to local file system as the default
                fs = LocalFileSystem()
            else:
                # construct a filesystem if it is a valid URI
                fs = _ensure_filesystem(filesystem)
            paths_or_selector = source
        else:
            fs, paths_or_selector = _ensure_multiple_sources(source, filesystem)
    else:
        fs, paths_or_selector = _ensure_single_source(source, filesystem)

    options = FileSystemFactoryOptions(
        partitioning=partitioning,
        partition_base_dir=partition_base_dir,
        exclude_invalid_files=exclude_invalid_files,
        selector_ignore_prefixes=selector_ignore_prefixes
    )
    factory = FileSystemDatasetFactory(fs, paths_or_selector, format, options)

    return factory.finish(schema)


def _in_memory_dataset(source, schema=None, **kwargs):
    if any(v is not None for v in kwargs.values()):
        raise ValueError(
            "For in-memory datasets, you cannot pass any additional arguments")
    return InMemoryDataset(source, schema)


def _union_dataset(children, schema=None, **kwargs):
    if any(v is not None for v in kwargs.values()):
        raise ValueError(
            "When passing a list of Datasets, you cannot pass any additional "
            "arguments"
        )

    if schema is None:
        # unify the children datasets' schemas
        schema = pa.unify_schemas([child.schema for child in children])

    for child in children:
        if getattr(child, "_scan_options", None):
            raise ValueError(
                "Creating an UnionDataset from filtered or projected Datasets "
                "is currently not supported. Union the unfiltered datasets "
                "and apply the filter to the resulting union."
            )

    # create datasets with the requested schema
    children = [child.replace_schema(schema) for child in children]

    return UnionDataset(schema, children)


def parquet_dataset(metadata_path, schema=None, filesystem=None, format=None,
                    partitioning=None, partition_base_dir=None):
    """
    Create a FileSystemDataset from a `_metadata` file created via
    `pyarrow.parquet.write_metadata`.

    Parameters
    ----------
    metadata_path : path,
        Path pointing to a single file parquet metadata file
    schema : Schema, optional
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    filesystem : FileSystem or URI string, default None
        If a single path is given as source and filesystem is None, then the
        filesystem will be inferred from the path.
        If an URI string is passed, then a filesystem object is constructed
        using the URI's optional path component as a directory prefix. See the
        examples below.
        Note that the URIs on Windows must follow 'file:///C:...' or
        'file:/C:...' patterns.
    format : ParquetFileFormat
        An instance of a ParquetFileFormat if special options needs to be
        passed.
    partitioning : Partitioning, PartitioningFactory, str, list of str
        The partitioning scheme specified with the ``partitioning()``
        function. A flavor string can be used as shortcut, and with a list of
        field names a DirectoryPartitioning will be inferred.
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.

    Returns
    -------
    FileSystemDataset
        The dataset corresponding to the given metadata
    """
    from pyarrow.fs import LocalFileSystem, _ensure_filesystem

    if format is None:
        format = ParquetFileFormat()
    elif not isinstance(format, ParquetFileFormat):
        raise ValueError("format argument must be a ParquetFileFormat")

    if filesystem is None:
        filesystem = LocalFileSystem()
    else:
        filesystem = _ensure_filesystem(filesystem)

    metadata_path = filesystem.normalize_path(_stringify_path(metadata_path))
    options = ParquetFactoryOptions(
        partition_base_dir=partition_base_dir,
        partitioning=_ensure_partitioning(partitioning)
    )

    factory = ParquetDatasetFactory(
        metadata_path, filesystem, format, options=options)
    return factory.finish(schema)


def dataset(source, schema=None, format=None, filesystem=None,
            partitioning=None, partition_base_dir=None,
            exclude_invalid_files=None, ignore_prefixes=None):
    """
    Open a dataset.

    Datasets provides functionality to efficiently work with tabular,
    potentially larger than memory and multi-file dataset.

    - A unified interface for different sources, like Parquet and Feather
    - Discovery of sources (crawling directories, handle directory-based
      partitioned datasets, basic schema normalization)
    - Optimized reading with predicate pushdown (filtering rows), projection
      (selecting columns), parallel reading or fine-grained managing of tasks.

    Note that this is the high-level API, to have more control over the dataset
    construction use the low-level API classes (FileSystemDataset,
    FilesystemDatasetFactory, etc.)

    Parameters
    ----------
    source : path, list of paths, dataset, list of datasets, (list of) \
RecordBatch or Table, iterable of RecordBatch, RecordBatchReader, or URI
        Path pointing to a single file:
            Open a FileSystemDataset from a single file.
        Path pointing to a directory:
            The directory gets discovered recursively according to a
            partitioning scheme if given.
        List of file paths:
            Create a FileSystemDataset from explicitly given files. The files
            must be located on the same filesystem given by the filesystem
            parameter.
            Note that in contrary of construction from a single file, passing
            URIs as paths is not allowed.
        List of datasets:
            A nested UnionDataset gets constructed, it allows arbitrary
            composition of other datasets.
            Note that additional keyword arguments are not allowed.
        (List of) batches or tables, iterable of batches, or RecordBatchReader:
            Create an InMemoryDataset. If an iterable or empty list is given,
            a schema must also be given. If an iterable or RecordBatchReader
            is given, the resulting dataset can only be scanned once; further
            attempts will raise an error.
    schema : Schema, optional
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    format : FileFormat or str
        Currently "parquet", "ipc"/"arrow"/"feather", "csv", "json", and "orc" are
        supported. For Feather, only version 2 files are supported.
    filesystem : FileSystem or URI string, default None
        If a single path is given as source and filesystem is None, then the
        filesystem will be inferred from the path.
        If an URI string is passed, then a filesystem object is constructed
        using the URI's optional path component as a directory prefix. See the
        examples below.
        Note that the URIs on Windows must follow 'file:///C:...' or
        'file:/C:...' patterns.
    partitioning : Partitioning, PartitioningFactory, str, list of str
        The partitioning scheme specified with the ``partitioning()``
        function. A flavor string can be used as shortcut, and with a list of
        field names a DirectoryPartitioning will be inferred.
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    exclude_invalid_files : bool, optional (default False)
        If True, invalid files will be excluded (file format specific check).
        This will incur IO for each files in a serial and single threaded
        fashion. Disabling this feature will skip the IO, but unsupported
        files may be present in the Dataset (resulting in an error at scan
        time).
    ignore_prefixes : list, optional
        Files matching any of these prefixes will be ignored by the
        discovery process. This is matched to the basename of a path.
        By default this is ['.', '_'].
        Note that discovery happens only if a directory is passed as source.

    Returns
    -------
    dataset : Dataset
        Either a FileSystemDataset or a UnionDataset depending on the source
        parameter.

    Examples
    --------
    Creating an example Table:

    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
    ...                   'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, "file.parquet")

    Opening a single file:

    >>> import pyarrow.dataset as ds
    >>> dataset = ds.dataset("file.parquet", format="parquet")
    >>> dataset.to_table()
    pyarrow.Table
    year: int64
    n_legs: int64
    animal: string
    ----
    year: [[2020,2022,2021,2022,2019,2021]]
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Opening a single file with an explicit schema:

    >>> myschema = pa.schema([
    ...     ('n_legs', pa.int64()),
    ...     ('animal', pa.string())])
    >>> dataset = ds.dataset("file.parquet", schema=myschema, format="parquet")
    >>> dataset.to_table()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Opening a dataset for a single directory:

    >>> ds.write_dataset(table, "partitioned_dataset", format="parquet",
    ...                  partitioning=['year'])
    >>> dataset = ds.dataset("partitioned_dataset", format="parquet")
    >>> dataset.to_table()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[5],[2],[4,100],[2,4]]
    animal: [["Brittle stars"],["Flamingo"],...["Parrot","Horse"]]

    For a single directory from a S3 bucket:

    >>> ds.dataset("s3://mybucket/nyc-taxi/",
    ...            format="parquet") # doctest: +SKIP

    Opening a dataset from a list of relatives local paths:

    >>> dataset = ds.dataset([
    ...     "partitioned_dataset/2019/part-0.parquet",
    ...     "partitioned_dataset/2020/part-0.parquet",
    ...     "partitioned_dataset/2021/part-0.parquet",
    ... ], format='parquet')
    >>> dataset.to_table()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[5],[2],[4,100]]
    animal: [["Brittle stars"],["Flamingo"],["Dog","Centipede"]]

    With filesystem provided:

    >>> paths = [
    ...     'part0/data.parquet',
    ...     'part1/data.parquet',
    ...     'part3/data.parquet',
    ... ]
    >>> ds.dataset(paths, filesystem='file:///directory/prefix,
    ...            format='parquet') # doctest: +SKIP

    Which is equivalent with:

    >>> fs = SubTreeFileSystem("/directory/prefix",
    ...                        LocalFileSystem()) # doctest: +SKIP
    >>> ds.dataset(paths, filesystem=fs, format='parquet') # doctest: +SKIP

    With a remote filesystem URI:

    >>> paths = [
    ...     'nested/directory/part0/data.parquet',
    ...     'nested/directory/part1/data.parquet',
    ...     'nested/directory/part3/data.parquet',
    ... ]
    >>> ds.dataset(paths, filesystem='s3://bucket/',
    ...            format='parquet') # doctest: +SKIP

    Similarly to the local example, the directory prefix may be included in the
    filesystem URI:

    >>> ds.dataset(paths, filesystem='s3://bucket/nested/directory',
    ...         format='parquet') # doctest: +SKIP

    Construction of a nested dataset:

    >>> ds.dataset([
    ...     dataset("s3://old-taxi-data", format="parquet"),
    ...     dataset("local/path/to/data", format="ipc")
    ... ]) # doctest: +SKIP
    """
    from pyarrow.fs import FileInfo
    # collect the keyword arguments for later reuse
    kwargs = dict(
        schema=schema,
        filesystem=filesystem,
        partitioning=partitioning,
        format=format,
        partition_base_dir=partition_base_dir,
        exclude_invalid_files=exclude_invalid_files,
        selector_ignore_prefixes=ignore_prefixes
    )

    if _is_path_like(source):
        return _filesystem_dataset(source, **kwargs)
    elif isinstance(source, (tuple, list)):
        if all(_is_path_like(elem) or isinstance(elem, FileInfo) for elem in source):
            return _filesystem_dataset(source, **kwargs)
        elif all(isinstance(elem, Dataset) for elem in source):
            return _union_dataset(source, **kwargs)
        elif all(isinstance(elem, (pa.RecordBatch, pa.Table))
                 for elem in source):
            return _in_memory_dataset(source, **kwargs)
        else:
            unique_types = set(type(elem).__name__ for elem in source)
            type_names = ', '.join(f'{t}' for t in unique_types)
            raise TypeError(
                'Expected a list of path-like or dataset objects, or a list '
                'of batches or tables. The given list contains the following '
                f'types: {type_names}'
            )
    elif isinstance(source, (pa.RecordBatch, pa.Table, pa.RecordBatchReader)):
        return _in_memory_dataset(source, **kwargs)
    else:
        raise TypeError(
            'Expected a path-like, list of path-likes or a list of Datasets '
            f'instead of the given type: {type(source).__name__}'
        )


def _ensure_write_partitioning(part, schema, flavor):
    if isinstance(part, PartitioningFactory):
        raise ValueError("A PartitioningFactory cannot be used. "
                         "Did you call the partitioning function "
                         "without supplying a schema?")

    if isinstance(part, Partitioning) and flavor:
        raise ValueError(
            "Providing a partitioning_flavor with "
            "a Partitioning object is not supported"
        )
    elif isinstance(part, (tuple, list)):
        # Name of fields were provided instead of a partitioning object.
        # Create a partitioning factory with those field names.
        part = partitioning(
            schema=pa.schema([schema.field(f) for f in part]),
            flavor=flavor
        )
    elif part is None:
        part = partitioning(pa.schema([]), flavor=flavor)

    if not isinstance(part, Partitioning):
        raise ValueError(
            "partitioning must be a Partitioning object or "
            "a list of column names"
        )

    return part


def write_dataset(data, base_dir, *, basename_template=None, format=None,
                  partitioning=None, partitioning_flavor=None,
                  schema=None, filesystem=None, file_options=None, use_threads=True,
                  preserve_order=False, max_partitions=None, max_open_files=None,
                  max_rows_per_file=None, min_rows_per_group=None,
                  max_rows_per_group=None, file_visitor=None,
                  existing_data_behavior='error', create_dir=True):
    """
    Write a dataset to a given format and partitioning.

    Parameters
    ----------
    data : Dataset, Table/RecordBatch, RecordBatchReader, list of \
Table/RecordBatch, or iterable of RecordBatch
        The data to write. This can be a Dataset instance or
        in-memory Arrow data. If an iterable is given, the schema must
        also be given.
    base_dir : str
        The root directory where to write the dataset.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to
        "part-{i}." + format.default_extname
    format : FileFormat or str
        The format in which to write the dataset. Currently supported:
        "parquet", "ipc"/"arrow"/"feather", and "csv". If a FileSystemDataset
        is being written and `format` is not specified, it defaults to the
        same format as the specified FileSystemDataset. When writing a
        Table or RecordBatch, this keyword is required.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the ``partitioning()``
        function or a list of field names. When providing a list of
        field names, you can use ``partitioning_flavor`` to drive which
        partitioning type should be used.
    partitioning_flavor : str, optional
        One of the partitioning flavors supported by
        ``pyarrow.dataset.partitioning``. If omitted will use the
        default of ``partitioning()`` which is directory partitioning.
    schema : Schema, optional
    filesystem : FileSystem, optional
    file_options : pyarrow.dataset.FileWriteOptions, optional
        FileFormat specific write options, created using the
        ``FileFormat.make_write_options()`` function.
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores. Using multiple
        threads may change the order of rows in the written dataset if
        preserve_order is set to False.
    preserve_order : bool, default False
        Preserve the order of rows. If enabled, order of rows in the dataset are
        guaranteed to be preserved even if use_threads is set to True. This may
        cause notable performance degradation.
    max_partitions : int, default 1024
        Maximum number of partitions any batch may be written into.
    max_open_files : int, default 1024
        If greater than 0 then this will limit the maximum number of
        files that can be left open. If an attempt is made to open
        too many files then the least recently used file will be closed.
        If this setting is set too low you may end up fragmenting your
        data into many small files.
    max_rows_per_file : int, default 0
        Maximum number of rows per file. If greater than 0 then this will
        limit how many rows are placed in any single file. Otherwise there
        will be no limit and one file will be created in each output
        directory unless files need to be closed to respect max_open_files
    min_rows_per_group : int, default 0
        Minimum number of rows per group. When the value is greater than 0,
        the dataset writer will batch incoming data and only write the row
        groups to the disk when sufficient rows have accumulated.
    max_rows_per_group : int, default 1024 * 1024
        Maximum number of rows per group. If the value is greater than 0,
        then the dataset writer may split up large incoming batches into
        multiple row groups.  If this value is set, then min_rows_per_group
        should also be set. Otherwise it could end up with very small row
        groups.
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
    existing_data_behavior : 'error' | 'overwrite_or_ignore' | \
'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination.  The default behavior ('error') is to raise an error
        if any data exists in the destination.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    create_dir : bool, default True
        If False, directories will not be created.  This can be useful for
        filesystems that do not require directories.
    """
    from pyarrow.fs import _resolve_filesystem_and_path

    if isinstance(data, (list, tuple)):
        schema = schema or data[0].schema
        data = InMemoryDataset(data, schema=schema)
    elif isinstance(data, (pa.RecordBatch, pa.Table)):
        schema = schema or data.schema
        data = InMemoryDataset(data, schema=schema)
    elif (
        isinstance(data, pa.ipc.RecordBatchReader)
        or hasattr(data, "__arrow_c_stream__")
        or _is_iterable(data)
    ):
        data = Scanner.from_batches(data, schema=schema)
        schema = None
    elif not isinstance(data, (Dataset, Scanner)):
        raise ValueError(
            "Only Dataset, Scanner, Table/RecordBatch, RecordBatchReader, "
            "a list of Tables/RecordBatches, or iterable of batches are "
            "supported."
        )

    if format is None and isinstance(data, FileSystemDataset):
        format = data.format
    else:
        format = _ensure_format(format)

    if file_options is None:
        file_options = format.make_write_options()

    if format != file_options.format:
        raise TypeError(f"Supplied FileWriteOptions have format {format}, "
                        f"which doesn't match supplied FileFormat {file_options}")

    if basename_template is None:
        basename_template = "part-{i}." + format.default_extname

    if max_partitions is None:
        max_partitions = 1024

    if max_open_files is None:
        max_open_files = 1024

    if max_rows_per_file is None:
        max_rows_per_file = 0

    if max_rows_per_group is None:
        max_rows_per_group = 1 << 20

    if min_rows_per_group is None:
        min_rows_per_group = 0

    # at this point data is a Scanner or a Dataset, anything else
    # was converted to one of those two. So we can grab the schema
    # to build the partitioning object from Dataset.
    if isinstance(data, Scanner):
        partitioning_schema = data.projected_schema
    else:
        partitioning_schema = data.schema
    partitioning = _ensure_write_partitioning(partitioning,
                                              schema=partitioning_schema,
                                              flavor=partitioning_flavor)

    filesystem, base_dir = _resolve_filesystem_and_path(base_dir, filesystem)

    if isinstance(data, Dataset):
        scanner = data.scanner(use_threads=use_threads)
    else:
        # scanner was passed directly by the user, in which case a schema
        # cannot be passed
        if schema is not None:
            raise ValueError("Cannot specify a schema when writing a Scanner")
        scanner = data

    _filesystemdataset_write(
        scanner, base_dir, basename_template, filesystem, partitioning,
        preserve_order, file_options, max_partitions, file_visitor,
        existing_data_behavior, max_open_files, max_rows_per_file,
        min_rows_per_group, max_rows_per_group, create_dir
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


from numbers import Integral
import warnings

from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path


class ORCFile:
    """
    Reader interface for a single ORC file

    Parameters
    ----------
    source : str or pyarrow.NativeFile
        Readable source. For passing Python file objects or byte buffers,
        see pyarrow.io.PythonFileInterface or pyarrow.io.BufferReader.
    """

    def __init__(self, source):
        self.reader = _orc.ORCReader()
        self.reader.open(source)

    @property
    def metadata(self):
        """The file metadata, as an arrow KeyValueMetadata"""
        return self.reader.metadata()

    @property
    def schema(self):
        """The file schema, as an arrow schema"""
        return self.reader.schema()

    @property
    def nrows(self):
        """The number of rows in the file"""
        return self.reader.nrows()

    @property
    def nstripes(self):
        """The number of stripes in the file"""
        return self.reader.nstripes()

    @property
    def file_version(self):
        """Format version of the ORC file, must be 0.11 or 0.12"""
        return self.reader.file_version()

    @property
    def software_version(self):
        """Software instance and version that wrote this file"""
        return self.reader.software_version()

    @property
    def compression(self):
        """Compression codec of the file"""
        return self.reader.compression()

    @property
    def compression_size(self):
        """Number of bytes to buffer for the compression codec in the file"""
        return self.reader.compression_size()

    @property
    def writer(self):
        """Name of the writer that wrote this file.
        If the writer is unknown then its Writer ID
        (a number) is returned"""
        return self.reader.writer()

    @property
    def writer_version(self):
        """Version of the writer"""
        return self.reader.writer_version()

    @property
    def row_index_stride(self):
        """Number of rows per an entry in the row index or 0
        if there is no row index"""
        return self.reader.row_index_stride()

    @property
    def nstripe_statistics(self):
        """Number of stripe statistics"""
        return self.reader.nstripe_statistics()

    @property
    def content_length(self):
        """Length of the data stripes in the file in bytes"""
        return self.reader.content_length()

    @property
    def stripe_statistics_length(self):
        """The number of compressed bytes in the file stripe statistics"""
        return self.reader.stripe_statistics_length()

    @property
    def file_footer_length(self):
        """The number of compressed bytes in the file footer"""
        return self.reader.file_footer_length()

    @property
    def file_postscript_length(self):
        """The number of bytes in the file postscript"""
        return self.reader.file_postscript_length()

    @property
    def file_length(self):
        """The number of bytes in the file"""
        return self.reader.file_length()

    def _select_names(self, columns=None):
        if columns is None:
            return None

        schema = self.schema
        names = []
        for col in columns:
            if isinstance(col, Integral):
                col = int(col)
                if 0 <= col < len(schema):
                    col = schema[col].name
                    names.append(col)
                else:
                    raise ValueError("Column indices must be in 0 <= ind < %d,"
                                     " got %d" % (len(schema), col))
            else:
                return columns

        return names

    def read_stripe(self, n, columns=None):
        """Read a single stripe from the file.

        Parameters
        ----------
        n : int
            The stripe index
        columns : list
            If not None, only these columns will be read from the stripe. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'

        Returns
        -------
        pyarrow.RecordBatch
            Content of the stripe as a RecordBatch.
        """
        columns = self._select_names(columns)
        return self.reader.read_stripe(n, columns=columns)

    def read(self, columns=None):
        """Read the whole file.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'. Output always follows the
            ordering of the file and not the `columns` list.

        Returns
        -------
        pyarrow.Table
            Content of the file as a Table.
        """
        columns = self._select_names(columns)
        return self.reader.read(columns=columns)


_orc_writer_args_docs = """file_version : {"0.11", "0.12"}, default "0.12"
    Determine which ORC file version to use.
    `Hive 0.11 / ORC v0 <https://orc.apache.org/specification/ORCv0/>`_
    is the older version
    while `Hive 0.12 / ORC v1 <https://orc.apache.org/specification/ORCv1/>`_
    is the newer one.
batch_size : int, default 1024
    Number of rows the ORC writer writes at a time.
stripe_size : int, default 64 * 1024 * 1024
    Size of each ORC stripe in bytes.
compression : string, default 'uncompressed'
    The compression codec.
    Valid values: {'UNCOMPRESSED', 'SNAPPY', 'ZLIB', 'LZ4', 'ZSTD'}
    Note that LZ0 is currently not supported.
compression_block_size : int, default 64 * 1024
    Size of each compression block in bytes.
compression_strategy : string, default 'speed'
    The compression strategy i.e. speed vs size reduction.
    Valid values: {'SPEED', 'COMPRESSION'}
row_index_stride : int, default 10000
    The row index stride i.e. the number of rows per
    an entry in the row index.
padding_tolerance : double, default 0.0
    The padding tolerance.
dictionary_key_size_threshold : double, default 0.0
    The dictionary key size threshold. 0 to disable dictionary encoding.
    1 to always enable dictionary encoding.
bloom_filter_columns : None, set-like or list-like, default None
    Columns that use the bloom filter.
bloom_filter_fpp : double, default 0.05
    Upper limit of the false-positive rate of the bloom filter.
"""


class ORCWriter:
    __doc__ = f"""
Writer interface for a single ORC file

Parameters
----------
where : str or pyarrow.io.NativeFile
    Writable target. For passing Python file objects or byte buffers,
    see pyarrow.io.PythonFileInterface, pyarrow.io.BufferOutputStream
    or pyarrow.io.FixedSizeBufferWriter.
{_orc_writer_args_docs}
"""

    is_open = False

    def __init__(self, where, *,
                 file_version='0.12',
                 batch_size=1024,
                 stripe_size=64 * 1024 * 1024,
                 compression='uncompressed',
                 compression_block_size=65536,
                 compression_strategy='speed',
                 row_index_stride=10000,
                 padding_tolerance=0.0,
                 dictionary_key_size_threshold=0.0,
                 bloom_filter_columns=None,
                 bloom_filter_fpp=0.05,
                 ):
        self.writer = _orc.ORCWriter()
        self.writer.open(
            where,
            file_version=file_version,
            batch_size=batch_size,
            stripe_size=stripe_size,
            compression=compression,
            compression_block_size=compression_block_size,
            compression_strategy=compression_strategy,
            row_index_stride=row_index_stride,
            padding_tolerance=padding_tolerance,
            dictionary_key_size_threshold=dictionary_key_size_threshold,
            bloom_filter_columns=bloom_filter_columns,
            bloom_filter_fpp=bloom_filter_fpp
        )
        self.is_open = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def write(self, table):
        """
        Write the table into an ORC file. The schema of the table must
        be equal to the schema used when opening the ORC file.

        Parameters
        ----------
        table : pyarrow.Table
            The table to be written into the ORC file
        """
        assert self.is_open
        self.writer.write(table)

    def close(self):
        """
        Close the ORC file
        """
        if self.is_open:
            self.writer.close()
            self.is_open = False


def read_table(source, columns=None, filesystem=None):
    filesystem, path = _resolve_filesystem_and_path(source, filesystem)
    if filesystem is not None:
        source = filesystem.open_input_file(path)

    if columns is not None and len(columns) == 0:
        result = ORCFile(source).read().select(columns)
    else:
        result = ORCFile(source).read(columns=columns)

    return result


read_table.__doc__ = """
Read a Table from an ORC file.

Parameters
----------
source : str, pyarrow.NativeFile, or file-like object
    If a string passed, can be a single file name. For file-like objects,
    only read a single file. Use pyarrow.BufferReader to read a file
    contained in a bytes or buffer-like object.
columns : list
    If not None, only these columns will be read from the file. A column
    name may be a prefix of a nested field, e.g. 'a' will select 'a.b',
    'a.c', and 'a.d.e'. Output always follows the ordering of the file and
    not the `columns` list. If empty, no columns will be read. Note
    that the table will still have the correct num_rows set despite having
    no columns.
filesystem : FileSystem, default None
    If nothing passed, will be inferred based on path.
    Path will try to be found in the local on-disk filesystem otherwise
    it will be parsed as an URI to determine the filesystem.
"""


def write_table(table, where, *,
                file_version='0.12',
                batch_size=1024,
                stripe_size=64 * 1024 * 1024,
                compression='uncompressed',
                compression_block_size=65536,
                compression_strategy='speed',
                row_index_stride=10000,
                padding_tolerance=0.0,
                dictionary_key_size_threshold=0.0,
                bloom_filter_columns=None,
                bloom_filter_fpp=0.05):
    if isinstance(where, Table):
        warnings.warn(
            "The order of the arguments has changed. Pass as "
            "'write_table(table, where)' instead. The old order will raise "
            "an error in the future.", FutureWarning, stacklevel=2
        )
        table, where = where, table
    with ORCWriter(
        where,
        file_version=file_version,
        batch_size=batch_size,
        stripe_size=stripe_size,
        compression=compression,
        compression_block_size=compression_block_size,
        compression_strategy=compression_strategy,
        row_index_stride=row_index_stride,
        padding_tolerance=padding_tolerance,
        dictionary_key_size_threshold=dictionary_key_size_threshold,
        bloom_filter_columns=bloom_filter_columns,
        bloom_filter_fpp=bloom_filter_fpp
    ) as writer:
        writer.write(table)


write_table.__doc__ = f"""
Write a table into an ORC file.

Parameters
----------
table : pyarrow.lib.Table
    The table to be written into the ORC file
where : str or pyarrow.io.NativeFile
    Writable target. For passing Python file objects or byte buffers,
    see pyarrow.io.PythonFileInterface, pyarrow.io.BufferOutputStream
    or pyarrow.io.FixedSizeBufferWriter.
{_orc_writer_args_docs}
"""
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

from pyarrow._compute import (  # noqa
    Function,
    FunctionOptions,
    FunctionRegistry,
    HashAggregateFunction,
    HashAggregateKernel,
    Kernel,
    ScalarAggregateFunction,
    ScalarAggregateKernel,
    ScalarFunction,
    ScalarKernel,
    VectorFunction,
    VectorKernel,
    # Option classes
    ArraySortOptions,
    AssumeTimezoneOptions,
    CastOptions,
    CountOptions,
    CumulativeOptions,
    CumulativeSumOptions,
    DayOfWeekOptions,
    DictionaryEncodeOptions,
    RunEndEncodeOptions,
    ElementWiseAggregateOptions,
    ExtractRegexOptions,
    ExtractRegexSpanOptions,
    FilterOptions,
    IndexOptions,
    InversePermutationOptions,
    JoinOptions,
    ListSliceOptions,
    ListFlattenOptions,
    MakeStructOptions,
    MapLookupOptions,
    MatchSubstringOptions,
    ModeOptions,
    NullOptions,
    PadOptions,
    PairwiseOptions,
    PartitionNthOptions,
    PivotWiderOptions,
    QuantileOptions,
    RandomOptions,
    RankOptions,
    RankQuantileOptions,
    ReplaceSliceOptions,
    ReplaceSubstringOptions,
    RoundBinaryOptions,
    RoundOptions,
    RoundTemporalOptions,
    RoundToMultipleOptions,
    ScalarAggregateOptions,
    ScatterOptions,
    SelectKOptions,
    SetLookupOptions,
    SkewOptions,
    SliceOptions,
    SortOptions,
    SplitOptions,
    SplitPatternOptions,
    StrftimeOptions,
    StrptimeOptions,
    StructFieldOptions,
    TakeOptions,
    TDigestOptions,
    TrimOptions,
    Utf8NormalizeOptions,
    VarianceOptions,
    WeekOptions,
    WinsorizeOptions,
    ZeroFillOptions,
    # Functions
    call_function,
    function_registry,
    get_function,
    list_functions,
    # Udf
    call_tabular_function,
    register_scalar_function,
    register_tabular_function,
    register_aggregate_function,
    register_vector_function,
    UdfContext,
    # Expressions
    Expression,
)

from collections import namedtuple
import inspect
from textwrap import dedent
import warnings

import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape


def _get_arg_names(func):
    return func._doc.arg_names


_OptionsClassDoc = namedtuple('_OptionsClassDoc', ('params',))


def _scrape_options_class_doc(options_class):
    if not options_class.__doc__:
        return None
    doc = docscrape.NumpyDocString(options_class.__doc__)
    return _OptionsClassDoc(doc['Parameters'])


def _decorate_compute_function(wrapper, exposed_name, func, options_class):
    # Decorate the given compute function wrapper with useful metadata
    # and documentation.
    cpp_doc = func._doc

    wrapper.__arrow_compute_function__ = dict(
        name=func.name,
        arity=func.arity,
        options_class=cpp_doc.options_class,
        options_required=cpp_doc.options_required)
    wrapper.__name__ = exposed_name
    wrapper.__qualname__ = exposed_name

    doc_pieces = []

    # 1. One-line summary
    summary = cpp_doc.summary
    if not summary:
        arg_str = "arguments" if func.arity > 1 else "argument"
        summary = f"Call compute function {func.name!r} with the given {arg_str}"

    doc_pieces.append(f"{summary}.\n\n")

    # 2. Multi-line description
    description = cpp_doc.description
    if description:
        doc_pieces.append(f"{description}\n\n")

    doc_addition = _compute_docstrings.function_doc_additions.get(func.name)

    # 3. Parameter description
    doc_pieces.append(dedent("""\
        Parameters
        ----------
        """))

    # 3a. Compute function parameters
    arg_names = _get_arg_names(func)
    for arg_name in arg_names:
        if func.kind in ('vector', 'scalar_aggregate'):
            arg_type = 'Array-like'
        else:
            arg_type = 'Array-like or scalar-like'
        doc_pieces.append(f"{arg_name} : {arg_type}\n")
        doc_pieces.append("    Argument to compute function.\n")

    # 3b. Compute function option values
    if options_class is not None:
        options_class_doc = _scrape_options_class_doc(options_class)
        if options_class_doc:
            for p in options_class_doc.params:
                doc_pieces.append(f"{p.name} : {p.type}\n")
                for s in p.desc:
                    doc_pieces.append(f"    {s}\n")
        else:
            warnings.warn(f"Options class {options_class.__name__} "
                          f"does not have a docstring", RuntimeWarning)
            options_sig = inspect.signature(options_class)
            for p in options_sig.parameters.values():
                doc_pieces.append(dedent(f"""\
                {p.name} : optional
                    Parameter for {options_class.__name__} constructor. Either `options`
                    or `{p.name}` can be passed, but not both at the same time.
                """))
        doc_pieces.append(dedent(f"""\
            options : pyarrow.compute.{options_class.__name__}, optional
                Alternative way of passing options.
            """))

    doc_pieces.append(dedent("""\
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the default memory pool.
        """))

    # 4. Custom addition (e.g. examples)
    if doc_addition is not None:
        stripped = dedent(doc_addition).strip('\n')
        doc_pieces.append(f"\n{stripped}\n")

    wrapper.__doc__ = "".join(doc_pieces)
    return wrapper


def _get_options_class(func):
    class_name = func._doc.options_class
    if not class_name:
        return None
    try:
        return globals()[class_name]
    except KeyError:
        warnings.warn(f"Python binding for {class_name} not exposed",
                      RuntimeWarning)
        return None


def _handle_options(name, options_class, options, args, kwargs):
    if args or kwargs:
        if options is not None:
            raise TypeError(
                f"Function {name!r} called with both an 'options' argument "
                f"and additional arguments")
        return options_class(*args, **kwargs)

    if options is not None:
        if isinstance(options, dict):
            return options_class(**options)
        elif isinstance(options, options_class):
            return options
        raise TypeError(
            f"Function {name!r} expected a {options_class} parameter, "
            f"got {type(options)}")

    return None


def _make_generic_wrapper(func_name, func, options_class, arity):
    if options_class is None:
        def wrapper(*args, memory_pool=None):
            if arity is not Ellipsis and len(args) != arity:
                raise TypeError(
                    f"{func_name} takes {arity} positional argument(s), "
                    f"but {len(args)} were given"
                )
            if args and isinstance(args[0], Expression):
                return Expression._call(func_name, list(args))
            return func.call(args, None, memory_pool)
    else:
        def wrapper(*args, memory_pool=None, options=None, **kwargs):
            if arity is not Ellipsis:
                if len(args) < arity:
                    raise TypeError(
                        f"{func_name} takes {arity} positional argument(s), "
                        f"but {len(args)} were given"
                    )
                option_args = args[arity:]
                args = args[:arity]
            else:
                option_args = ()
            options = _handle_options(func_name, options_class, options,
                                      option_args, kwargs)
            if args and isinstance(args[0], Expression):
                return Expression._call(func_name, list(args), options)
            return func.call(args, options, memory_pool)
    return wrapper


def _make_signature(arg_names, var_arg_names, options_class):
    from inspect import Parameter
    params = []
    for name in arg_names:
        params.append(Parameter(name, Parameter.POSITIONAL_ONLY))
    for name in var_arg_names:
        params.append(Parameter(name, Parameter.VAR_POSITIONAL))
    if options_class is not None:
        options_sig = inspect.signature(options_class)
        for p in options_sig.parameters.values():
            assert p.kind in (Parameter.POSITIONAL_OR_KEYWORD,
                              Parameter.KEYWORD_ONLY)
            if var_arg_names:
                # Cannot have a positional argument after a *args
                p = p.replace(kind=Parameter.KEYWORD_ONLY)
            params.append(p)
        params.append(Parameter("options", Parameter.KEYWORD_ONLY,
                                default=None))
    params.append(Parameter("memory_pool", Parameter.KEYWORD_ONLY,
                            default=None))
    return inspect.Signature(params)


def _wrap_function(name, func):
    options_class = _get_options_class(func)
    arg_names = _get_arg_names(func)
    has_vararg = arg_names and arg_names[-1].startswith('*')
    if has_vararg:
        var_arg_names = [arg_names.pop().lstrip('*')]
    else:
        var_arg_names = []

    wrapper = _make_generic_wrapper(
        name, func, options_class, arity=func.arity)
    wrapper.__signature__ = _make_signature(arg_names, var_arg_names,
                                            options_class)
    return _decorate_compute_function(wrapper, name, func, options_class)


def _make_global_functions():
    """
    Make global functions wrapping each compute function.

    Note that some of the automatically-generated wrappers may be overridden
    by custom versions below.
    """
    g = globals()
    reg = function_registry()

    # Avoid clashes with Python keywords
    rewrites = {'and': 'and_',
                'or': 'or_'}

    for cpp_name in reg.list_functions():
        name = rewrites.get(cpp_name, cpp_name)
        func = reg.get_function(cpp_name)
        if func.kind == "hash_aggregate":
            # Hash aggregate functions are not callable,
            # so let's not expose them at module level.
            continue
        if func.kind == "scalar_aggregate" and func.arity == 0:
            # Nullary scalar aggregate functions are not callable
            # directly so let's not expose them at module level.
            continue
        assert name not in g, name
        g[cpp_name] = g[name] = _wrap_function(name, func)


_make_global_functions()
# Alias for consistency; globals() is needed to avoid Python lint errors
utf8_zfill = utf8_zero_fill = globals()["utf8_zero_fill"]


def cast(arr, target_type=None, safe=None, options=None, memory_pool=None):
    """
    Cast array values to another data type. Can also be invoked as an array
    instance method.

    Parameters
    ----------
    arr : Array-like
    target_type : DataType or str
        Type to cast to
    safe : bool, default True
        Check for overflows or other unsafe conversions
    options : CastOptions, default None
        Additional checks pass by CastOptions
    memory_pool : MemoryPool, optional
        memory pool to use for allocations during function execution.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pyarrow as pa
    >>> arr = pa.array([datetime(2010, 1, 1), datetime(2015, 1, 1)])
    >>> arr.type
    TimestampType(timestamp[us])

    You can use ``pyarrow.DataType`` objects to specify the target type:

    >>> cast(arr, pa.timestamp('ms'))
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]

    >>> cast(arr, pa.timestamp('ms')).type
    TimestampType(timestamp[ms])

    Alternatively, it is also supported to use the string aliases for these
    types:

    >>> arr.cast('timestamp[ms]')
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]
    >>> arr.cast('timestamp[ms]').type
    TimestampType(timestamp[ms])

    Returns
    -------
    casted : Array
        The cast result as a new Array
    """
    safe_vars_passed = (safe is not None) or (target_type is not None)

    if safe_vars_passed and (options is not None):
        raise ValueError("Must either pass values for 'target_type' and 'safe'"
                         " or pass a value for 'options'")

    if options is None:
        target_type = pa.types.lib.ensure_type(target_type)
        if safe is False:
            options = CastOptions.unsafe(target_type)
        else:
            options = CastOptions.safe(target_type)
    return call_function("cast", [arr], options, memory_pool)


def index(data, value, start=None, end=None, *, memory_pool=None):
    """
    Find the index of the first occurrence of a given value.

    Parameters
    ----------
    data : Array-like
    value : Scalar-like object
        The value to search for.
    start : int, optional
    end : int, optional
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    index : int
        the index, or -1 if not found

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["Lorem", "ipsum", "dolor", "sit", "Lorem", "ipsum"])
    >>> pc.index(arr, "ipsum")
    <pyarrow.Int64Scalar: 1>
    >>> pc.index(arr, "ipsum", start=2)
    <pyarrow.Int64Scalar: 5>
    >>> pc.index(arr, "amet")
    <pyarrow.Int64Scalar: -1>
    """
    if start is not None:
        if end is not None:
            data = data.slice(start, end - start)
        else:
            data = data.slice(start)
    elif end is not None:
        data = data.slice(0, end)

    if not isinstance(value, pa.Scalar):
        value = pa.scalar(value, type=data.type)
    elif data.type != value.type:
        value = pa.scalar(value.as_py(), type=data.type)
    options = IndexOptions(value=value)
    result = call_function('index', [data], options, memory_pool)
    if start is not None and result.as_py() >= 0:
        result = pa.scalar(result.as_py() + start, type=pa.int64())
    return result


def take(data, indices, *, boundscheck=True, memory_pool=None):
    """
    Select values (or records) from array- or table-like data given integer
    selection indices.

    The result will be of the same type(s) as the input, with elements taken
    from the input array (or record batch / table fields) at the given
    indices. If an index is null then the corresponding value in the output
    will be null.

    Parameters
    ----------
    data : Array, ChunkedArray, RecordBatch, or Table
    indices : Array, ChunkedArray
        Must be of integer type
    boundscheck : boolean, default True
        Whether to boundscheck the indices. If False and there is an out of
        bounds index, will likely cause the process to crash.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : depends on inputs
        Selected values for the given indices

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> indices = pa.array([0, None, 4, 3])
    >>> arr.take(indices)
    <pyarrow.lib.StringArray object at ...>
    [
      "a",
      null,
      "e",
      null
    ]
    """
    options = TakeOptions(boundscheck=boundscheck)
    return call_function('take', [data, indices], options, memory_pool)


def fill_null(values, fill_value):
    """Replace each null element in values with a corresponding
    element from fill_value.

    If fill_value is scalar-like, then every null element in values
    will be replaced with fill_value. If fill_value is array-like,
    then the i-th element in values will be replaced with the i-th
    element in fill_value.

    The fill_value's type must be the same as that of values, or it
    must be able to be implicitly casted to the array's type.

    This is an alias for :func:`coalesce`.

    Parameters
    ----------
    values : Array, ChunkedArray, or Scalar-like object
        Each null element is replaced with the corresponding value
        from fill_value.
    fill_value : Array, ChunkedArray, or Scalar-like object
        If not same type as values, will attempt to cast.

    Returns
    -------
    result : depends on inputs
        Values with all null elements replaced

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array([1, 2, None, 3], type=pa.int8())
    >>> fill_value = pa.scalar(5, type=pa.int8())
    >>> arr.fill_null(fill_value)
    <pyarrow.lib.Int8Array object at ...>
    [
      1,
      2,
      5,
      3
    ]
    >>> arr = pa.array([1, 2, None, 4, None])
    >>> arr.fill_null(pa.array([10, 20, 30, 40, 50]))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      2,
      30,
      4,
      50
    ]
    """
    if not isinstance(fill_value, (pa.Array, pa.ChunkedArray, pa.Scalar)):
        fill_value = pa.scalar(fill_value, type=values.type)
    elif values.type != fill_value.type:
        fill_value = pa.scalar(fill_value.as_py(), type=values.type)

    return call_function("coalesce", [values, fill_value])


def top_k_unstable(
        values, k, sort_keys=None, null_placements=None, *, memory_pool=None):
    """
    Select the indices of the top-k ordered elements from array- or table-like
    data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get top indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    null_placements : A list of "at_start" or "at_end"
        Whether nulls and NaNs are placed at the start or at the end.
        Accepted values are "at_end", "at_start".
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array
        Indices of the top-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.top_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      5,
      4,
      2
    ]
    """
    if sort_keys is None:
        sort_keys = []
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        sort_keys.append(("dummy", "descending", "at_end"))
    else:
        sort_keys = [(sort_key, "descending", null_placement)
                     for sort_key, null_placement in zip(sort_keys, null_placements)]
    options = SelectKOptions(k, sort_keys)
    return call_function("select_k_unstable", [values], options, memory_pool)


def bottom_k_unstable(
        values, k, sort_keys=None, null_placements=None, *, memory_pool=None):
    """
    Select the indices of the bottom-k ordered elements from
    array- or table-like data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get bottom indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    null_placements : A list of "at_start" or "at_end"
        Whether nulls and NaNs are placed at the start or at the end.
        Accepted values are "at_end", "at_start".
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array of indices
        Indices of the bottom-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.bottom_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      0,
      1,
      2
    ]
    """
    if sort_keys is None:
        sort_keys = []
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        sort_keys.append(("dummy", "ascending", "at_end"))
    else:
        sort_keys = [(sort_key, "ascending", null_placement)
                     for sort_key, null_placement in zip(sort_keys, null_placements)]

    options = SelectKOptions(k, sort_keys)
    return call_function("select_k_unstable", [values], options, memory_pool)


def random(n, *, initializer='system', options=None, memory_pool=None):
    """
    Generate numbers in the range [0, 1).

    Generated values are uniformly-distributed, double-precision
    in range [0, 1). Algorithm and seed can be changed via RandomOptions.

    Parameters
    ----------
    n : int
        Number of values to generate, must be greater than or equal to 0
    initializer : int or str
        How to initialize the underlying random generator.
        If an integer is given, it is used as a seed.
        If "system" is given, the random generator is initialized with
        a system-specific source of (hopefully true) randomness.
        Other values are invalid.
    options : pyarrow.compute.RandomOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """
    options = RandomOptions(initializer=initializer)
    return call_function("random", [], options, memory_pool, length=n)


def field(*name_or_index):
    """Reference a column of the dataset.

    Stores only the field's name. Type and other information is known only when
    the expression is bound to a dataset having an explicit scheme.

    Nested references are allowed by passing multiple names or a tuple of
    names. For example ``('foo', 'bar')`` references the field named "bar"
    inside the field named "foo".

    Parameters
    ----------
    *name_or_index : string, multiple strings, tuple or int
        The name or index of the (possibly nested) field the expression
        references to.

    Returns
    -------
    field_expr : Expression
        Reference to the given field

    Examples
    --------
    >>> import pyarrow.compute as pc
    >>> pc.field("a")
    <pyarrow.compute.Expression a>
    >>> pc.field(1)
    <pyarrow.compute.Expression FieldPath(1)>
    >>> pc.field(("a", "b"))
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    >>> pc.field("a", "b")
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    """
    n = len(name_or_index)
    if n == 1:
        if isinstance(name_or_index[0], (str, int)):
            return Expression._field(name_or_index[0])
        elif isinstance(name_or_index[0], tuple):
            return Expression._nested_field(name_or_index[0])
        else:
            raise TypeError(
                "field reference should be str, multiple str, tuple or "
                f"integer, got {type(name_or_index[0])}"
            )
    # In case of multiple strings not supplied in a tuple
    else:
        return Expression._nested_field(name_or_index)


def scalar(value):
    """Expression representing a scalar value.

    Creates an Expression object representing a scalar value that can be used
    in compute expressions and predicates.

    Parameters
    ----------
    value : bool, int, float or string
        Python value of the scalar. This function accepts any value that can be
        converted to a ``pyarrow.Scalar`` using ``pa.scalar()``.

    Notes
    -----
    This function differs from ``pyarrow.scalar()`` in the following way:

    * ``pyarrow.scalar()`` creates a ``pyarrow.Scalar`` object that represents
      a single value in Arrow's memory model.
    * ``pyarrow.compute.scalar()`` creates an ``Expression`` object representing
      a scalar value that can be used in compute expressions, predicates, and
      dataset filtering operations.

    Returns
    -------
    scalar_expr : Expression
        An Expression representing the scalar value
    """
    return Expression._scalar(value)
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


from collections import defaultdict
from contextlib import nullcontext
from functools import reduce

import inspect
import json
import os
import re
import operator

import pyarrow as pa

try:
    import pyarrow._parquet as _parquet
except ImportError as exc:
    raise ImportError(
        "The pyarrow installation is not built with support "
        f"for the Parquet file format ({str(exc)})"
    ) from None

from pyarrow._parquet import (ParquetReader, Statistics,  # noqa
                              FileMetaData, RowGroupMetaData,
                              ColumnChunkMetaData,
                              ParquetSchema, ColumnSchema,
                              ParquetLogicalType,
                              FileEncryptionProperties,
                              FileDecryptionProperties,
                              SortingColumn)
from pyarrow.fs import (LocalFileSystem, FileType, _resolve_filesystem_and_path,
                        _ensure_filesystem)
from pyarrow.util import guid, _is_path_like, _stringify_path, _deprecate_api


def _check_contains_null(val):
    if isinstance(val, bytes):
        for byte in val:
            if isinstance(byte, bytes):
                compare_to = chr(0)
            else:
                compare_to = 0
            if byte == compare_to:
                return True
    elif isinstance(val, str):
        return '\x00' in val
    return False


def _check_filters(filters, check_null_strings=True):
    """
    Check if filters are well-formed.
    """
    if filters is not None:
        if len(filters) == 0 or any(len(f) == 0 for f in filters):
            raise ValueError("Malformed filters")
        if isinstance(filters[0][0], str):
            # We have encountered the situation where we have one nesting level
            # too few:
            #   We have [(,,), ..] instead of [[(,,), ..]]
            filters = [filters]
        if check_null_strings:
            for conjunction in filters:
                for col, op, val in conjunction:
                    if (
                        isinstance(val, list) and
                        all(_check_contains_null(v) for v in val) or
                        _check_contains_null(val)
                    ):
                        raise NotImplementedError(
                            "Null-terminated binary strings are not supported "
                            "as filter values."
                        )
    return filters


_DNF_filter_doc = """Predicates are expressed using an ``Expression`` or using
    the disjunctive normal form (DNF), like ``[[('x', '=', 0), ...], ...]``.
    DNF allows arbitrary boolean logical combinations of single column predicates.
    The innermost tuples each describe a single column predicate. The list of inner
    predicates is interpreted as a conjunction (AND), forming a more selective and
    multiple column predicate. Finally, the most outer list combines these filters
    as a disjunction (OR).

    Predicates may also be passed as List[Tuple]. This form is interpreted
    as a single conjunction. To express OR in predicates, one must
    use the (preferred) List[List[Tuple]] notation.

    Each tuple has format: (``key``, ``op``, ``value``) and compares the
    ``key`` with the ``value``.
    The supported ``op`` are:  ``=`` or ``==``, ``!=``, ``<``, ``>``, ``<=``,
    ``>=``, ``in`` and ``not in``. If the ``op`` is ``in`` or ``not in``, the
    ``value`` must be a collection such as a ``list``, a ``set`` or a
    ``tuple``.

    Examples:

    Using the ``Expression`` API:

    .. code-block:: python

        import pyarrow.compute as pc
        pc.field('x') = 0
        pc.field('y').isin(['a', 'b', 'c'])
        ~pc.field('y').isin({'a', 'b'})

    Using the DNF format:

    .. code-block:: python

        ('x', '=', 0)
        ('y', 'in', ['a', 'b', 'c'])
        ('z', 'not in', {'a','b'})

    """


def filters_to_expression(filters):
    """
    Check if filters are well-formed and convert to an ``Expression``.

    Parameters
    ----------
    filters : List[Tuple] or List[List[Tuple]]

    Notes
    -----
    See internal ``pyarrow._DNF_filter_doc`` attribute for more details.

    Examples
    --------

    >>> filters_to_expression([('foo', '==', 'bar')])
    <pyarrow.compute.Expression (foo == "bar")>

    Returns
    -------
    pyarrow.compute.Expression
        An Expression representing the filters
    """
    import pyarrow.dataset as ds

    if isinstance(filters, ds.Expression):
        return filters

    filters = _check_filters(filters, check_null_strings=False)

    def convert_single_predicate(col, op, val):
        field = ds.field(col)

        if op == "=" or op == "==":
            return field == val
        elif op == "!=":
            return field != val
        elif op == '<':
            return field < val
        elif op == '>':
            return field > val
        elif op == '<=':
            return field <= val
        elif op == '>=':
            return field >= val
        elif op == 'in':
            return field.isin(val)
        elif op == 'not in':
            return ~field.isin(val)
        else:
            raise ValueError(f'"{col}" is not a valid operator in predicates.')

    disjunction_members = []

    for conjunction in filters:
        conjunction_members = [
            convert_single_predicate(col, op, val)
            for col, op, val in conjunction
        ]

        disjunction_members.append(reduce(operator.and_, conjunction_members))

    return reduce(operator.or_, disjunction_members)


_filters_to_expression = _deprecate_api(
    "_filters_to_expression", "filters_to_expression",
    filters_to_expression, "10.0.0", DeprecationWarning)


# ----------------------------------------------------------------------
# Reading a single Parquet file


class ParquetFile:
    """
    Reader interface for a single Parquet file.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
        Readable source. For passing bytes or buffer-like file containing a
        Parquet file, use pyarrow.BufferReader.
    metadata : FileMetaData, default None
        Use existing metadata object, rather than reading from file.
    common_metadata : FileMetaData, default None
        Will be used in reads for pandas schema metadata if not found in the
        main file's metadata, no other uses at the moment.
    read_dictionary : list
        List of column names to read directly as DictionaryArray.
    binary_type : pyarrow.DataType, default None
        If given, Parquet binary columns will be read as this datatype.
        This setting is ignored if a serialized Arrow schema is found in
        the Parquet metadata.
    list_type : subclass of pyarrow.DataType, default None
        If given, non-MAP repeated columns will be read as an instance of
        this datatype (either pyarrow.ListType or pyarrow.LargeListType).
        This setting is ignored if a serialized Arrow schema is found in
        the Parquet metadata.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    pre_buffer : bool, default True
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3, GCS). If True, Arrow will use a
        background I/O thread pool. If using a filesystem layer that itself
        performs readahead (e.g. fsspec's S3FS), disable readahead for best
        results. Set to False if you want to prioritize minimal memory usage
        over maximum speed.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds.
    decryption_properties : FileDecryptionProperties, default None
        File decryption properties for Parquet Modular Encryption.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    page_checksum_verification : bool, default False
        If True, verify the checksum for each page read from the file.
    arrow_extensions_enabled : bool, default True
        If True, read Parquet logical types as Arrow extension types where
        possible (e.g., read JSON as the canonical `arrow.json` extension type
        or UUID as the canonical `arrow.uuid` extension type).

    Examples
    --------

    Generate an example PyArrow Table and write it to Parquet file:

    >>> import pyarrow as pa
    >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    >>> import pyarrow.parquet as pq
    >>> pq.write_table(table, 'example.parquet')

    Create a ``ParquetFile`` object from the Parquet file:

    >>> parquet_file = pq.ParquetFile('example.parquet')

    Read the data:

    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Create a ParquetFile object with "animal" column as DictionaryArray:

    >>> parquet_file = pq.ParquetFile('example.parquet',
    ...                               read_dictionary=["animal"])
    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: dictionary<values=string, indices=int32, ordered=0>
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [  -- dictionary:
    ["Flamingo","Parrot",...,"Brittle stars","Centipede"]  -- indices:
    [0,1,2,3,4,5]]
    """

    def __init__(self, source, *, metadata=None, common_metadata=None,
                 read_dictionary=None, binary_type=None, list_type=None,
                 memory_map=False, buffer_size=0, pre_buffer=True,
                 coerce_int96_timestamp_unit=None,
                 decryption_properties=None, thrift_string_size_limit=None,
                 thrift_container_size_limit=None, filesystem=None,
                 page_checksum_verification=False, arrow_extensions_enabled=True):

        self._close_source = getattr(source, 'closed', True)

        filesystem, source = _resolve_filesystem_and_path(
            source, filesystem, memory_map=memory_map)
        if filesystem is not None:
            source = filesystem.open_input_file(source)
            self._close_source = True  # We opened it here, ensure we close it.

        self.reader = ParquetReader()
        self.reader.open(
            source, use_memory_map=memory_map,
            buffer_size=buffer_size, pre_buffer=pre_buffer,
            read_dictionary=read_dictionary, metadata=metadata,
            binary_type=binary_type, list_type=list_type,
            coerce_int96_timestamp_unit=coerce_int96_timestamp_unit,
            decryption_properties=decryption_properties,
            thrift_string_size_limit=thrift_string_size_limit,
            thrift_container_size_limit=thrift_container_size_limit,
            page_checksum_verification=page_checksum_verification,
            arrow_extensions_enabled=arrow_extensions_enabled,
        )
        self.common_metadata = common_metadata
        self._nested_paths_by_prefix = self._build_nested_paths()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _build_nested_paths(self):
        paths = self.reader.column_paths

        result = defaultdict(list)

        for i, path in enumerate(paths):
            key = path[0]
            rest = path[1:]
            while True:
                result[key].append(i)

                if not rest:
                    break

                key = '.'.join((key, rest[0]))
                rest = rest[1:]

        return result

    @property
    def metadata(self):
        """
        Return the Parquet metadata.
        """
        return self.reader.metadata

    @property
    def schema(self):
        """
        Return the Parquet schema, unconverted to Arrow types
        """
        return self.metadata.schema

    @property
    def schema_arrow(self):
        """
        Return the inferred Arrow schema, converted from the whole Parquet
        file's schema

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read the Arrow schema:

        >>> parquet_file.schema_arrow
        n_legs: int64
        animal: string
        """
        return self.reader.schema_arrow

    @property
    def num_row_groups(self):
        """
        Return the number of row groups of the Parquet file.

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.num_row_groups
        1
        """
        return self.reader.num_row_groups

    def close(self, force: bool = False):
        if self._close_source or force:
            self.reader.close()

    @property
    def closed(self) -> bool:
        return self.reader.closed

    def read_row_group(self, i, columns=None, use_threads=True,
                       use_pandas_metadata=False):
        """
        Read a single row group from a Parquet file.

        Parameters
        ----------
        i : int
            Index of the individual row group that we want to read.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row group as a table (of columns)

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_group(0)
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,100]]
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
        column_indices = self._get_column_indices(
            columns, use_pandas_metadata=use_pandas_metadata)
        return self.reader.read_row_group(i, column_indices=column_indices,
                                          use_threads=use_threads)

    def read_row_groups(self, row_groups, columns=None, use_threads=True,
                        use_pandas_metadata=False):
        """
        Read a multiple row groups from a Parquet file.

        Parameters
        ----------
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row groups as a table (of columns).

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_groups([0,0])
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,...,2,4,4,5,100]]
        animal: [["Flamingo","Parrot","Dog",...,"Brittle stars","Centipede"]]
        """
        column_indices = self._get_column_indices(
            columns, use_pandas_metadata=use_pandas_metadata)
        return self.reader.read_row_groups(row_groups,
                                           column_indices=column_indices,
                                           use_threads=use_threads)

    def iter_batches(self, batch_size=65536, row_groups=None, columns=None,
                     use_threads=True, use_pandas_metadata=False):
        """
        Read streaming batches from a Parquet file.

        Parameters
        ----------
        batch_size : int, default 64K
            Maximum number of records to yield per batch. Batches may be
            smaller if there aren't enough rows in the file.
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : boolean, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : boolean, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Yields
        ------
        pyarrow.RecordBatch
            Contents of each batch as a record batch

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')
        >>> for i in parquet_file.iter_batches():
        ...     print("RecordBatch")
        ...     print(i.to_pandas())
        ...
        RecordBatch
           n_legs         animal
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if row_groups is None:
            row_groups = range(0, self.metadata.num_row_groups)
        column_indices = self._get_column_indices(
            columns, use_pandas_metadata=use_pandas_metadata)

        batches = self.reader.iter_batches(batch_size,
                                           row_groups=row_groups,
                                           column_indices=column_indices,
                                           use_threads=use_threads)
        return batches

    def read(self, columns=None, use_threads=True, use_pandas_metadata=False):
        """
        Read a Table from Parquet format.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read a Table:

        >>> parquet_file.read(columns=["animal"])
        pyarrow.Table
        animal: string
        ----
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
        column_indices = self._get_column_indices(
            columns, use_pandas_metadata=use_pandas_metadata)
        return self.reader.read_all(column_indices=column_indices,
                                    use_threads=use_threads)

    def scan_contents(self, columns=None, batch_size=65536):
        """
        Read contents of file for the given columns and batch size.

        Notes
        -----
        This function's primary purpose is benchmarking.
        The scan is executed on a single thread.

        Parameters
        ----------
        columns : list of integers, default None
            Select columns to read, if None scan all columns.
        batch_size : int, default 64K
            Number of rows to read at a time internally.

        Returns
        -------
        num_rows : int
            Number of rows in file

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.scan_contents()
        6
        """
        column_indices = self._get_column_indices(columns)
        return self.reader.scan_contents(column_indices,
                                         batch_size=batch_size)

    def _get_column_indices(self, column_names, use_pandas_metadata=False):
        if column_names is None:
            return None

        indices = []

        for name in column_names:
            if name in self._nested_paths_by_prefix:
                indices.extend(self._nested_paths_by_prefix[name])

        if use_pandas_metadata:
            file_keyvalues = self.metadata.metadata
            common_keyvalues = (self.common_metadata.metadata
                                if self.common_metadata is not None
                                else None)

            if file_keyvalues and b'pandas' in file_keyvalues:
                index_columns = _get_pandas_index_columns(file_keyvalues)
            elif common_keyvalues and b'pandas' in common_keyvalues:
                index_columns = _get_pandas_index_columns(common_keyvalues)
            else:
                index_columns = []

            if indices is not None and index_columns:
                indices += [self.reader.column_name_idx(descr)
                            for descr in index_columns
                            if not isinstance(descr, dict)]

        return indices


_SPARK_DISALLOWED_CHARS = re.compile('[ ,;{}()\n\t=]')


def _sanitized_spark_field_name(name):
    return _SPARK_DISALLOWED_CHARS.sub('_', name)


def _sanitize_schema(schema, flavor):
    if 'spark' in flavor:
        sanitized_fields = []

        schema_changed = False

        for field in schema:
            name = field.name
            sanitized_name = _sanitized_spark_field_name(name)

            if sanitized_name != name:
                schema_changed = True
                sanitized_field = pa.field(sanitized_name, field.type,
                                           field.nullable, field.metadata)
                sanitized_fields.append(sanitized_field)
            else:
                sanitized_fields.append(field)

        new_schema = pa.schema(sanitized_fields, metadata=schema.metadata)
        return new_schema, schema_changed
    else:
        return schema, False


def _sanitize_table(table, new_schema, flavor):
    # TODO: This will not handle prohibited characters in nested field names
    if 'spark' in flavor:
        column_data = [table[i] for i in range(table.num_columns)]
        return pa.Table.from_arrays(column_data, schema=new_schema)
    else:
        return table


_parquet_writer_arg_docs = """version : {"1.0", "2.4", "2.6"}, default "2.6"
    Determine which Parquet logical types are available for use, whether the
    reduced set from the Parquet 1.x.x format or the expanded logical types
    added in later format versions.
    Files written with version='2.4' or '2.6' may not be readable in all
    Parquet implementations, so version='1.0' is likely the choice that
    maximizes file compatibility.
    UINT32 and some logical types are only available with version '2.4'.
    Nanosecond timestamps are only available with version '2.6'.
    Other features such as compression algorithms or the new serialized
    data page format must be enabled separately (see 'compression' and
    'data_page_version').
use_dictionary : bool or list, default True
    Specify if we should use dictionary encoding in general or only for
    some columns.
    When encoding the column, if the dictionary size is too large, the
    column will fallback to ``PLAIN`` encoding. Specially, ``BOOLEAN`` type
    doesn't support dictionary encoding.
compression : str or dict, default 'snappy'
    Specify the compression codec, either on a general basis or per-column.
    Valid values: {'NONE', 'SNAPPY', 'GZIP', 'BROTLI', 'LZ4', 'LZ4_RAW', 'ZSTD'}.
    'LZ4_RAW' is accepted as an alias for 'LZ4' (both use the LZ4_RAW
    codec as defined in the Parquet specification).
write_statistics : bool or list, default True
    Specify if we should write statistics in general (default is True) or only
    for some columns.
use_deprecated_int96_timestamps : bool, default None
    Write timestamps to INT96 Parquet format. Defaults to False unless enabled
    by flavor argument. This take priority over the coerce_timestamps option.
coerce_timestamps : str, default None
    Cast timestamps to a particular resolution. If omitted, defaults are chosen
    depending on `version`. For ``version='1.0'`` and ``version='2.4'``,
    nanoseconds are cast to microseconds ('us'), while for
    ``version='2.6'`` (the default), they are written natively without loss
    of resolution.  Seconds are always cast to milliseconds ('ms') by default,
    as Parquet does not have any temporal type with seconds resolution.
    If the casting results in loss of data, it will raise an exception
    unless ``allow_truncated_timestamps=True`` is given.
    Valid values: {None, 'ms', 'us'}
allow_truncated_timestamps : bool, default False
    Allow loss of data when coercing timestamps to a particular
    resolution. E.g. if microsecond or nanosecond data is lost when coercing to
    'ms', do not raise an exception. Passing ``allow_truncated_timestamp=True``
    will NOT result in the truncation exception being ignored unless
    ``coerce_timestamps`` is not None.
data_page_size : int, default None
    Set a target threshold for the approximate encoded size of data
    pages within a column chunk (in bytes). If None, use the default data page
    size of 1MByte.
max_rows_per_page : int, default None
    Maximum number of rows per page within a column chunk.
    If None, use the default of 20000.
    Smaller values reduce memory usage during reads but increase metadata overhead.
flavor : {'spark'}, default None
    Sanitize schema or set other compatibility options to work with
    various target systems.
filesystem : FileSystem, default None
    If nothing passed, will be inferred from `where` if path-like, else
    `where` is already a file-like object so no filesystem is needed.
compression_level : int or dict, default None
    Specify the compression level for a codec, either on a general basis or
    per-column. If None is passed, arrow selects the compression level for
    the compression codec in use. The compression level has a different
    meaning for each codec, so you have to read the documentation of the
    codec you are using.
    An exception is thrown if the compression codec does not allow specifying
    a compression level.
use_byte_stream_split : bool or list, default False
    Specify if the byte_stream_split encoding should be used in general or
    only for some columns. If both dictionary and byte_stream_stream are
    enabled, then dictionary is preferred.
    The byte_stream_split encoding is valid for integer, floating-point
    and fixed-size binary data types (including decimals); it should be
    combined with a compression codec so as to achieve size reduction.
column_encoding : string or dict, default None
    Specify the encoding scheme on a per column basis.
    Can only be used when ``use_dictionary`` is set to False, and
    cannot be used in combination with ``use_byte_stream_split``.
    Currently supported values: {'PLAIN', 'BYTE_STREAM_SPLIT',
    'DELTA_BINARY_PACKED', 'DELTA_LENGTH_BYTE_ARRAY', 'DELTA_BYTE_ARRAY'}.
    Certain encodings are only compatible with certain data types.
    Please refer to the encodings section of `Reading and writing Parquet
    files <https://arrow.apache.org/docs/cpp/parquet.html#encodings>`_.
data_page_version : {"1.0", "2.0"}, default "1.0"
    The serialized Parquet data page format version to write, defaults to
    1.0. This does not impact the file schema logical types and Arrow to
    Parquet type casting behavior; for that use the "version" option.
use_compliant_nested_type : bool, default True
    Whether to write compliant Parquet nested type (lists) as defined
    `here <https://github.com/apache/parquet-format/blob/master/
    LogicalTypes.md#nested-types>`_, defaults to ``True``.
    For ``use_compliant_nested_type=True``, this will write into a list
    with 3-level structure where the middle level, named ``list``,
    is a repeated group with a single field named ``element``::

        <list-repetition> group <name> (LIST) {
            repeated group list {
                  <element-repetition> <element-type> element;
            }
        }

    For ``use_compliant_nested_type=False``, this will also write into a list
    with 3-level structure, where the name of the single field of the middle
    level ``list`` is taken from the element name for nested columns in Arrow,
    which defaults to ``item``::

        <list-repetition> group <name> (LIST) {
            repeated group list {
                <element-repetition> <element-type> item;
            }
        }

encryption_properties : FileEncryptionProperties, default None
    File encryption properties for Parquet Modular Encryption.
    If None, no encryption will be done.
    The encryption properties can be created using:
    ``CryptoFactory.file_encryption_properties()``.
write_batch_size : int, default None
    Number of values to write to a page at a time. If None, use the default of
    1024. ``write_batch_size`` is complementary to ``data_page_size``. If pages
    are exceeding the ``data_page_size`` due to large column values, lowering
    the batch size can help keep page sizes closer to the intended size.
dictionary_pagesize_limit : int, default None
    Specify the dictionary page size limit per row group. If None, use the
    default 1MB.
store_schema : bool, default True
    By default, the Arrow schema is serialized and stored in the Parquet
    file metadata (in the "ARROW:schema" key). When reading the file,
    if this key is available, it will be used to more faithfully recreate
    the original Arrow data. For example, for tz-aware timestamp columns
    it will restore the timezone (Parquet only stores the UTC values without
    timezone), or columns with duration type will be restored from the int64
    Parquet column.
write_page_index : bool, default False
    Whether to write a page index in general for all columns.
    Writing statistics to the page index disables the old method of writing
    statistics to each data page header. The page index makes statistics-based
    filtering more efficient than the page header, as it gathers all the
    statistics for a Parquet file in a single place, avoiding scattered I/O.
    Note that the page index is not yet used on the read size by PyArrow.
write_page_checksum : bool, default False
    Whether to write page checksums in general for all columns.
    Page checksums enable detection of data corruption, which might occur during
    transmission or in the storage.
sorting_columns : Sequence of SortingColumn, default None
    Specify the sort order of the data being written. The writer does not sort
    the data nor does it verify that the data is sorted. The sort order is
    written to the row group metadata, which can then be used by readers.
store_decimal_as_integer : bool, default False
    Allow decimals with 1 <= precision <= 18 to be stored as integers.
    In Parquet, DECIMAL can be stored in any of the following physical types:

    - int32: for 1 <= precision <= 9.
    - int64: for 10 <= precision <= 18.
    - fixed_len_byte_array: precision is limited by the array size.
      Length n can store <= floor(log_10(2^(8*n - 1) - 1)) base-10 digits.
    - binary: precision is unlimited. The minimum number of bytes to store the
      unscaled value is used.

    By default, this is DISABLED and all decimal types annotate fixed_len_byte_array.
    When enabled, the writer will use the following physical types to store decimals:

    - int32: for 1 <= precision <= 9.
    - int64: for 10 <= precision <= 18.
    - fixed_len_byte_array: for precision > 18.

    As a consequence, decimal columns stored in integer types are more compact.
use_content_defined_chunking : bool or dict, default False
    Optimize parquet files for content addressable storage (CAS) systems by writing
    data pages according to content-defined chunk boundaries. This allows for more
    efficient deduplication of data across files, hence more efficient network
    transfers and storage. The chunking is based on a rolling hash algorithm that
    identifies chunk boundaries based on the actual content of the data.

    Note that it is an experimental feature and the API may change in the future.

    If set to ``True``, a default configuration is used with `min_chunk_size=256 KiB`
    and `max_chunk_size=1024 KiB`. The chunk size distribution approximates a normal
    distribution between `min_chunk_size` and `max_chunk_size` (sizes are accounted
    before any Parquet encodings).

    A `dict` can be passed to adjust the chunker parameters with the following keys:

    - `min_chunk_size`: minimum chunk size in bytes, default 256 KiB
      The rolling hash will not be updated until this size is reached for each chunk.
      Note that all data sent through the hash function is counted towards the chunk
      size, including definition and repetition levels if present.
    - `max_chunk_size`: maximum chunk size in bytes, default is 1024 KiB
      The chunker will create a new chunk whenever the chunk size exceeds this value.
      Note that the parquet writer has a related `data_pagesize` property that controls
      the maximum size of a parquet data page after encoding. While setting
      `data_page_size` to a smaller value than `max_chunk_size` doesn't affect the
      chunking effectiveness, it results in more small parquet data pages.
    - `norm_level`: normalization level to center the chunk size around the average
      size more aggressively, default 0
      Increasing the normalization level increases the probability of finding a chunk,
      improving the deduplication ratio, but also increasing the number of small chunks
      resulting in many small parquet data pages. The default value provides a good
      balance between deduplication ratio and fragmentation. Use norm_level=1 or
      norm_level=2 to reach a higher deduplication ratio at the expense of
      fragmentation.

write_time_adjusted_to_utc : bool, default False
    Set the value of isAdjustedTOUTC when writing a TIME column.
    If True, this tells the Parquet reader that the TIME columns
    are expressed in reference to midnight in the UTC timezone.
    If False (the default), the TIME columns are assumed to be expressed
    in reference to midnight in an unknown, presumably local, timezone.
bloom_filter_options : dict, default None
    Create Bloom filters for the columns specified by the provided `dict`.

    Bloom filters can be configured with two parameters: number of distinct values
    (NDV), and false-positive probability (FPP).

    Bloom filters are most effective for high-cardinality columns. A good default
    is to set NDV equal to the number of rows. Lower values reduce disk usage but
    may not be worthwhile for very small NDVs. Increasing NDV (without increasing FPP)
    increases disk and memory usage.

    Lower FPP values require more disk and memory space. For a fixed NDV, the
    space requirement grows roughly proportional to log(1/FPP). Recommended
    values are 0.1, 0.05, or 0.01. Very small values are counterproductive as
    the bitset may exceed the size of the actual data. Set NDV appropriately
    to minimize space usage.

    The keys of the `dict` are column paths. For each path, the value can be either:

    - A dictionary, with keys `ndv` and `fpp`. The value for `ndv` must be a positive
      integer. If the 'ndv' key is not present, the default value of `1048576` will be
      used. The value for `fpp` must be a float between 0.0 and 1.0. If the `fpp` key
      is not present, the default value of `0.05` will be used.
    - A boolean, with ``True`` indicating that a Bloom filter should be produced with
      the above mentioned default values of `ndv=1048576` and `fpp=0.05`. This is
      equivalent to passing an empty dict.
"""

_parquet_writer_example_doc = """\
Generate an example PyArrow Table and RecordBatch:

>>> import pyarrow as pa
>>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
...                              "Brittle stars", "Centipede"]})
>>> batch = pa.record_batch([[2, 2, 4, 4, 5, 100],
...                         ["Flamingo", "Parrot", "Dog", "Horse",
...                          "Brittle stars", "Centipede"]],
...                         names=['n_legs', 'animal'])

create a ParquetWriter object:

>>> import pyarrow.parquet as pq
>>> writer = pq.ParquetWriter('example.parquet', table.schema)

and write the Table into the Parquet file:

>>> writer.write_table(table)
>>> writer.close()

>>> pq.read_table('example.parquet').to_pandas()
   n_legs         animal
0       2       Flamingo
1       2         Parrot
2       4            Dog
3       4          Horse
4       5  Brittle stars
5     100      Centipede

create a ParquetWriter object for the RecordBatch:

>>> writer2 = pq.ParquetWriter('example2.parquet', batch.schema)

and write the RecordBatch into the Parquet file:

>>> writer2.write_batch(batch)
>>> writer2.close()

>>> pq.read_table('example2.parquet').to_pandas()
   n_legs         animal
0       2       Flamingo
1       2         Parrot
2       4            Dog
3       4          Horse
4       5  Brittle stars
5     100      Centipede
"""


class ParquetWriter:

    __doc__ = f"""
Class for incrementally building a Parquet file for Arrow tables.

Parameters
----------
where : path or file-like object
schema : pyarrow.Schema
{_parquet_writer_arg_docs}
writer_engine_version : unused
**options : dict
    If options contains a key `metadata_collector` then the
    corresponding value is assumed to be a list (or any object with
    `.append` method) that will be filled with the file metadata instance
    of the written file.

Examples
--------
{_parquet_writer_example_doc}
"""

    def __init__(self, where, schema, filesystem=None,
                 flavor=None,
                 version='2.6',
                 use_dictionary=True,
                 compression='snappy',
                 write_statistics=True,
                 use_deprecated_int96_timestamps=None,
                 compression_level=None,
                 use_byte_stream_split=False,
                 column_encoding=None,
                 writer_engine_version=None,
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
                 **options):
        if use_deprecated_int96_timestamps is None:
            # Use int96 timestamps for Spark
            if flavor is not None and 'spark' in flavor:
                use_deprecated_int96_timestamps = True
            else:
                use_deprecated_int96_timestamps = False

        self.flavor = flavor
        if flavor is not None:
            schema, self.schema_changed = _sanitize_schema(schema, flavor)
        else:
            self.schema_changed = False

        self.schema = schema
        self.where = where

        # If we open a file using a filesystem, store file handle so we can be
        # sure to close it when `self.close` is called.
        self.file_handle = None

        filesystem, path = _resolve_filesystem_and_path(where, filesystem)
        if filesystem is not None:
            # ARROW-10480: do not auto-detect compression.  While
            # a filename like foo.parquet.gz is nonconforming, it
            # shouldn't implicitly apply compression.
            sink = self.file_handle = filesystem.open_output_stream(
                path, compression=None)
        else:
            sink = where
        self._metadata_collector = options.pop('metadata_collector', None)
        engine_version = 'V2'
        self.writer = _parquet.ParquetWriter(
            sink, schema,
            version=version,
            compression=compression,
            use_dictionary=use_dictionary,
            write_statistics=write_statistics,
            use_deprecated_int96_timestamps=use_deprecated_int96_timestamps,
            compression_level=compression_level,
            use_byte_stream_split=use_byte_stream_split,
            column_encoding=column_encoding,
            writer_engine_version=engine_version,
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
            **options)
        self.is_open = True

    def __del__(self):
        if getattr(self, 'is_open', False):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
        # return false since we want to propagate exceptions
        return False

    def write(self, table_or_batch, row_group_size=None):
        """
        Write RecordBatch or Table to the Parquet file.

        Parameters
        ----------
        table_or_batch : {RecordBatch, Table}
        row_group_size : int, default None
            Maximum number of rows in each written row group. If None, the row
            group size will be the minimum of the number of rows in the
            Table/RecordBatch and 1024 * 1024.
        """
        if isinstance(table_or_batch, pa.RecordBatch):
            self.write_batch(table_or_batch, row_group_size)
        elif isinstance(table_or_batch, pa.Table):
            self.write_table(table_or_batch, row_group_size)
        else:
            raise TypeError(type(table_or_batch))

    def write_batch(self, batch, row_group_size=None):
        """
        Write RecordBatch to the Parquet file.

        Parameters
        ----------
        batch : RecordBatch
        row_group_size : int, default None
            Maximum number of rows in written row group. If None, the
            row group size will be the minimum of the RecordBatch
            size (in rows) and 1024 * 1024. If set larger than 64 * 1024 * 1024
            then 64 * 1024 * 1024 will be used instead.
        """
        table = pa.Table.from_batches([batch], batch.schema)
        self.write_table(table, row_group_size)

    def write_table(self, table, row_group_size=None):
        """
        Write Table to the Parquet file.

        Parameters
        ----------
        table : Table
        row_group_size : int, default None
            Maximum number of rows in each written row group. If None,
            the row group size will be the minimum of the Table size (in rows)
            and 1024 * 1024. If set larger than 64 * 1024 * 1024 then
            64 * 1024 * 1024 will be used instead.

        """
        if self.schema_changed:
            table = _sanitize_table(table, self.schema, self.flavor)
        assert self.is_open

        if not table.schema.equals(self.schema, check_metadata=False):
            msg = (
                "Table schema does not match schema used to create file: \n"
                f"table:\n{table.schema!s} vs. \nfile:\n{self.schema!s}"
            )
            raise ValueError(msg)

        self.writer.write_table(table, row_group_size=row_group_size)

    def close(self):
        """
        Close the connection to the Parquet file.
        """
        if self.is_open:
            self.writer.close()
            self.is_open = False
            if self._metadata_collector is not None:
                self._metadata_collector.append(self.writer.metadata)
        if self.file_handle is not None:
            self.file_handle.close()

    def add_key_value_metadata(self, key_value_metadata):
        """
        Add key-value metadata to the file.
        This will overwrite any existing metadata with the same key.

        Parameters
        ----------
        key_value_metadata : dict
            Keys and values must be string-like / coercible to bytes.
        """
        assert self.is_open
        self.writer.add_key_value_metadata(key_value_metadata)


def _get_pandas_index_columns(keyvalues):
    return (json.loads(keyvalues[b'pandas'].decode('utf8'))
            ['index_columns'])


EXCLUDED_PARQUET_PATHS = {'_SUCCESS'}


_read_docstring_common = """\
read_dictionary : list, default None
    List of names or column paths (for nested types) to read directly
    as DictionaryArray. Only supported for BYTE_ARRAY storage. To read
    a flat column as dictionary-encoded pass the column name. For
    nested types, you must pass the full column "path", which could be
    something like level1.level2.list.item. Refer to the Parquet
    file's schema to obtain the paths.
binary_type : pyarrow.DataType, default None
    If given, Parquet binary columns will be read as this datatype.
    This setting is ignored if a serialized Arrow schema is found in
    the Parquet metadata.
list_type : subclass of pyarrow.DataType, default None
    If given, non-MAP repeated columns will be read as an instance of
    this datatype (either pyarrow.ListType or pyarrow.LargeListType).
    This setting is ignored if a serialized Arrow schema is found in
    the Parquet metadata.
memory_map : bool, default False
    If the source is a file path, use a memory map to read file, which can
    improve performance in some environments.
buffer_size : int, default 0
    If positive, perform read buffering when deserializing individual
    column chunks. Otherwise IO calls are unbuffered.
partitioning : pyarrow.dataset.Partitioning or str or list of str, \
default "hive"
    The partitioning scheme for a partitioned dataset. The default of "hive"
    assumes directory names with key=value pairs like "/year=2009/month=11".
    In addition, a scheme like "/2009/11" is also supported, in which case
    you need to specify the field names or a full schema. See the
    ``pyarrow.dataset.partitioning()`` function for more details."""


_parquet_dataset_example = """\
Generate an example PyArrow Table and write it to a partitioned dataset:

>>> import pyarrow as pa
>>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
...                   'n_legs': [2, 2, 4, 4, 5, 100],
...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
...                              "Brittle stars", "Centipede"]})
>>> import pyarrow.parquet as pq
>>> pq.write_to_dataset(table, root_path='dataset_v2',
...                     partition_cols=['year'])

create a ParquetDataset object from the dataset source:

>>> dataset = pq.ParquetDataset('dataset_v2/')

and read the data:

>>> dataset.read().to_pandas()
   n_legs         animal  year
0       5  Brittle stars  2019
1       2       Flamingo  2020
2       4            Dog  2021
3     100      Centipede  2021
4       2         Parrot  2022
5       4          Horse  2022

create a ParquetDataset object with filter:

>>> dataset = pq.ParquetDataset('dataset_v2/',
...                             filters=[('n_legs','=',4)])
>>> dataset.read().to_pandas()
   n_legs animal  year
0       4    Dog  2021
1       4  Horse  2022
"""


class ParquetDataset:
    __doc__ = f"""
Encapsulates details of reading a complete Parquet dataset possibly
consisting of multiple files and partitions in subdirectories.

Parameters
----------
path_or_paths : str or List[str]
    A directory name, single file name, or list of file names.
filesystem : FileSystem, default None
    If nothing passed, will be inferred based on path.
    Path will try to be found in the local on-disk filesystem otherwise
    it will be parsed as an URI to determine the filesystem.
schema : pyarrow.parquet.Schema
    Optionally provide the Schema for the Dataset, in which case it will
    not be inferred from the source.
filters : pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]], default None
    Rows which do not match the filter predicate will be removed from scanned
    data. Partition keys embedded in a nested directory structure will be
    exploited to avoid loading files at all if they contain no matching rows.
    Within-file level filtering and different partitioning schemes are supported.

    {_DNF_filter_doc}
{_read_docstring_common}
ignore_prefixes : list, optional
    Files matching any of these prefixes will be ignored by the
    discovery process.
    This is matched to the basename of a path.
    By default this is ['.', '_'].
    Note that discovery happens only if a directory is passed as source.
pre_buffer : bool, default True
    Coalesce and issue file reads in parallel to improve performance on
    high-latency filesystems (e.g. S3, GCS). If True, Arrow will use a
    background I/O thread pool. If using a filesystem layer that itself
    performs readahead (e.g. fsspec's S3FS), disable readahead for best
    results. Set to False if you want to prioritize minimal memory usage
    over maximum speed.
coerce_int96_timestamp_unit : str, default None
    Cast timestamps that are stored in INT96 format to a particular resolution
    (e.g. 'ms'). Setting to None is equivalent to 'ns' and therefore INT96
    timestamps will be inferred as timestamps in nanoseconds.
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
    If True, verify the page checksum for each page read from the file.
arrow_extensions_enabled : bool, default True
    If True, read Parquet logical types as Arrow extension types where possible,
    (e.g., read JSON as the canonical `arrow.json` extension type or UUID as
    the canonical `arrow.uuid` extension type).

Examples
--------
{_parquet_dataset_example}
"""

    def __init__(self, path_or_paths, filesystem=None, schema=None, *, filters=None,
                 read_dictionary=None, binary_type=None, list_type=None,
                 memory_map=False, buffer_size=None, partitioning="hive",
                 ignore_prefixes=None,
                 pre_buffer=True, coerce_int96_timestamp_unit=None,
                 decryption_properties=None, thrift_string_size_limit=None,
                 thrift_container_size_limit=None,
                 page_checksum_verification=False,
                 arrow_extensions_enabled=True):
        import pyarrow.dataset as ds

        # map format arguments
        read_options = {
            "pre_buffer": pre_buffer,
            "coerce_int96_timestamp_unit": coerce_int96_timestamp_unit,
            "thrift_string_size_limit": thrift_string_size_limit,
            "thrift_container_size_limit": thrift_container_size_limit,
            "page_checksum_verification": page_checksum_verification,
            "arrow_extensions_enabled": arrow_extensions_enabled,
            "binary_type": binary_type,
            "list_type": list_type,
        }
        if buffer_size:
            read_options.update(use_buffered_stream=True,
                                buffer_size=buffer_size)
        if read_dictionary is not None:
            read_options.update(dictionary_columns=read_dictionary)

        if decryption_properties is not None:
            read_options.update(decryption_properties=decryption_properties)

        self._filter_expression = None
        if filters is not None:
            self._filter_expression = filters_to_expression(filters)

        # map old filesystems to new one
        if filesystem is not None:
            filesystem = _ensure_filesystem(
                filesystem, use_mmap=memory_map)
        elif filesystem is None and memory_map:
            # if memory_map is specified, assume local file system (string
            # path can in principle be URI for any filesystem)
            filesystem = LocalFileSystem(use_mmap=memory_map)

        # This needs to be checked after _ensure_filesystem, because that
        # handles the case of an fsspec LocalFileSystem
        if (
            hasattr(path_or_paths, "__fspath__") and
            filesystem is not None and
            not isinstance(filesystem, LocalFileSystem)
        ):
            raise TypeError(
                "Path-like objects with __fspath__ must only be used with "
                f"local file systems, not {type(filesystem)}"
            )

        # check for single fragment dataset or dataset directory
        single_file = None
        self._base_dir = None
        if not isinstance(path_or_paths, list):
            if _is_path_like(path_or_paths):
                filesystem, path_or_paths = _resolve_filesystem_and_path(
                    path_or_paths, filesystem, memory_map=memory_map
                )
                finfo = filesystem.get_file_info(path_or_paths)
                if finfo.is_file:
                    single_file = path_or_paths
                if finfo.type == FileType.Directory:
                    self._base_dir = path_or_paths
            else:
                single_file = path_or_paths

        parquet_format = ds.ParquetFileFormat(**read_options)

        if single_file is not None:
            fragment = parquet_format.make_fragment(single_file, filesystem)

            self._dataset = ds.FileSystemDataset(
                [fragment], schema=schema or fragment.physical_schema,
                format=parquet_format,
                filesystem=fragment.filesystem
            )
            return

        # check partitioning to enable dictionary encoding
        if partitioning == "hive":
            partitioning = ds.HivePartitioning.discover(
                infer_dictionary=True)

        self._dataset = ds.dataset(path_or_paths, filesystem=filesystem,
                                   schema=schema, format=parquet_format,
                                   partitioning=partitioning,
                                   ignore_prefixes=ignore_prefixes)

    def equals(self, other):
        if not isinstance(other, ParquetDataset):
            raise TypeError('`other` must be an instance of ParquetDataset')

        return (self.schema == other.schema and
                self._dataset.format == other._dataset.format and
                self.filesystem == other.filesystem and
                # self.fragments == other.fragments and
                self.files == other.files)

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    @property
    def schema(self):
        """
        Schema of the Dataset.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_schema',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_schema/')

        Read the schema:

        >>> dataset.schema
        n_legs: int64
        animal: string
        year: dictionary<values=int32, indices=int32, ordered=0>
        """
        return self._dataset.schema

    def read(self, columns=None, use_threads=True, use_pandas_metadata=False):
        """
        Read (multiple) Parquet files as a single pyarrow.Table.

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the dataset. The partition fields
            are not automatically included.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_read',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_read/')

        Read the dataset:

        >>> dataset.read(columns=["n_legs"])
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[5],[2],[4,100],[2,4]]
        """
        # if use_pandas_metadata, we need to include index columns in the
        # column selection, to be able to restore those in the pandas DataFrame
        metadata = self.schema.metadata or {}

        if use_pandas_metadata:
            # if the dataset schema metadata itself doesn't have pandas
            # then try to get this from common file (for backwards compat)
            if b"pandas" not in metadata:
                common_metadata = self._get_common_pandas_metadata()
                if common_metadata:
                    metadata = common_metadata

        if columns is not None and use_pandas_metadata:
            if metadata and b'pandas' in metadata:
                # RangeIndex can be represented as dict instead of column name
                index_columns = [
                    col for col in _get_pandas_index_columns(metadata)
                    if not isinstance(col, dict)
                ]
                columns = (
                    list(columns) + list(set(index_columns) - set(columns))
                )

        table = self._dataset.to_table(
            columns=columns, filter=self._filter_expression,
            use_threads=use_threads
        )

        # if use_pandas_metadata, restore the pandas metadata (which gets
        # lost if doing a specific `columns` selection in to_table)
        if use_pandas_metadata:
            if metadata and b"pandas" in metadata:
                new_metadata = table.schema.metadata or {}
                new_metadata.update({b"pandas": metadata[b"pandas"]})
                table = table.replace_schema_metadata(new_metadata)

        return table

    def _get_common_pandas_metadata(self):

        if not self._base_dir:
            return None

        metadata = None
        for name in ["_common_metadata", "_metadata"]:
            metadata_path = os.path.join(str(self._base_dir), name)
            finfo = self.filesystem.get_file_info(metadata_path)
            if finfo.is_file:
                pq_meta = read_metadata(
                    metadata_path, filesystem=self.filesystem)
                metadata = pq_meta.metadata
                if metadata and b'pandas' in metadata:
                    break

        return metadata

    def read_pandas(self, **kwargs):
        """
        Read dataset including pandas metadata, if any. Other arguments passed
        through to :func:`read`, see docstring for further details.

        Parameters
        ----------
        **kwargs : optional
            Additional options for :func:`read`

        Examples
        --------
        Generate an example parquet file:

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                    'n_legs': [2, 2, 4, 4, 5, 100],
        ...                    'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                    "Brittle stars", "Centipede"]})
        >>> table = pa.Table.from_pandas(df)
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'table_V2.parquet')
        >>> dataset = pq.ParquetDataset('table_V2.parquet')

        Read the dataset with pandas metadata:

        >>> dataset.read_pandas(columns=["n_legs"])
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[2,2,4,4,5,100]]

        >>> dataset.read_pandas(columns=["n_legs"]).schema.pandas_metadata
        {'index_columns': [{'kind': 'range', 'name': None, 'start': 0, ...}
        """
        return self.read(use_pandas_metadata=True, **kwargs)

    @property
    def fragments(self):
        """
        A list of the Dataset source fragments or pieces with absolute
        file paths.

        Examples
        --------
