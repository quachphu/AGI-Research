from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames


__all__ = [
    "FileOpenerIterDataPipe",
]


@functional_datapipe("open_files")
class FileOpenerIterDataPipe(IterDataPipe[tuple[str, IOBase]]):
    r"""
    Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``r``, other options are
            ``b`` for reading in binary mode and ``t`` for text mode.
        encoding: An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python's GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import (
        ...     FileLister,
        ...     FileOpener,
        ...     StreamReader,
        ... )
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith(".txt"))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [('./abc.txt', 'abc')]
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "r",
        encoding: str | None = None,
        length: int = -1,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.mode: str = mode
        self.encoding: str | None = encoding

        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError(f"Invalid mode {mode}")
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken

        if "b" in mode and encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")

        self.length: int = length

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self) -> Iterator[tuple[str, IOBase]]:
        yield from get_file_binaries_from_pathnames(
            self.datapipe, self.mode, self.encoding
        )

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
<|endoftext|>from collections.abc import Iterator
from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = ["StreamReaderIterDataPipe"]


@functional_datapipe("read_from_stream")
class StreamReaderIterDataPipe(IterDataPipe[tuple[str, bytes]]):
    r"""
    Given IO streams and their label names, yield bytes with label name as tuple.

    (functional name: ``read_from_stream``).

    Args:
        datapipe: Iterable DataPipe provides label/URL and byte stream
        chunk: Number of bytes to be read from stream per iteration.
            If ``None``, all bytes will be read until the EOF.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """

    def __init__(
        self, datapipe: IterDataPipe[tuple[str, IOBase]], chunk: int | None = None
    ) -> None:
        self.datapipe = datapipe
        self.chunk = chunk

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        for furl, stream in self.datapipe:
            while True:
                d = stream.read(self.chunk)
                if not d:
                    stream.close()
                    break
                yield (furl, d)
<|endoftext|>from collections.abc import Iterator, Sequence

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root


__all__ = ["FileListerIterDataPipe"]


@functional_datapipe("list_files")
class FileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.

    Multiple root directories can be provided (functional name: ``list_files``).

    Args:
        root: Root directory or a sequence of root directories
        masks: Unix style filter string or string list for filtering file name(s)
        recursive: Whether to return pathname from nested directories or not
        abspath: Whether to return relative pathname or absolute pathname
        non_deterministic: Whether to return pathname in sorted order or not.
            If ``False``, the results yielded from each root directory will be sorted
        length: Nominal length of the datapipe

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import FileLister
        >>> dp = FileLister(root=".", recursive=True)
        >>> list(dp)
        ['example.py', './data/data.tar']
    """

    def __init__(
        self,
        root: str | Sequence[str] | IterDataPipe = ".",
        masks: str | list[str] = "",
        *,
        recursive: bool = False,
        abspath: bool = False,
        non_deterministic: bool = False,
        length: int = -1,
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = [root]
        if not isinstance(root, IterDataPipe):
            root = IterableWrapperIterDataPipe(root)
        self.datapipe: IterDataPipe = root
        self.masks: str | list[str] = masks
        self.recursive: bool = recursive
        self.abspath: bool = abspath
        self.non_deterministic: bool = non_deterministic
        self.length: int = length

    def __iter__(self) -> Iterator[str]:
        for path in self.datapipe:
            yield from get_file_pathnames_from_root(
                path, self.masks, self.recursive, self.abspath, self.non_deterministic
            )

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections import namedtuple
from collections.abc import Callable, Iterator, Sized
from typing import Any, TypeVar

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    validate_input_col,
)


__all__ = [
    "CollatorIterDataPipe",
    "MapperIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Applies a function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source Iterable DataPipe
        fn: Function being applied over each item
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

        output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``

            - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
              multiple indices, the left-most one is used, and other indices will be removed.
            - Integer is used for list/tuple. ``-1`` represents to append result at the end.
            - Key is used for dict. New key is acceptable.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = IterableWrapper(range(10))
        >>> # Invocation via functional form is preferred
        ... map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
        >>> # Use `functools.partial` or explicitly define the function instead
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
    ) -> None:
        torch._C._log_api_usage_once("python.data_pipes.map")
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col
        validate_input_col(fn, input_col)

    def _apply_fn(self, data):
        if self.input_col is None and self.output_col is None:
            return self.fn(data)

        if self.input_col is None:
            res = self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            res = self.fn(*args)
        else:
            res = self.fn(data[self.input_col])

        # Copy tuple to list and run in-place modification because tuple is immutable.
        if isinstance(data, tuple):
            t_flag = True
            data = list(data)
        else:
            t_flag = False

        if self.output_col is None:
            if isinstance(self.input_col, (list, tuple)):
                data[self.input_col[0]] = res
                for idx in sorted(self.input_col[1:], reverse=True):
                    del data[idx]
            else:
                data[self.input_col] = res
        else:
            if self.output_col == -1:
                data.append(res)
            else:
                data[self.output_col] = res

        # Convert list back to tuple
        return tuple(data) if t_flag else data

    def __iter__(self) -> Iterator[_T_co]:
        for data in self.datapipe:
            yield self._apply_fn(data)

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


def _collate_helper(conversion, item):
    # TODO(VitalyFedyunin): Verify that item is any sort of batch
    if len(item.items) > 1:
        # TODO(VitalyFedyunin): Compact all batch dataframes into one
        raise RuntimeError("Only supports one DataFrame per batch")
    df = item[0]
    columns_name = df_wrapper.get_columns(df)
    tuple_names: list = []
    tuple_values: list = []

    for name in conversion:
        if name not in columns_name:
            raise RuntimeError("Conversion keys mismatch")

    for name in columns_name:
        if name in conversion:
            if not callable(conversion[name]):
                raise RuntimeError(
                    "Collate (DF)DataPipe requires callable as dict values"
                )
            collation_fn = conversion[name]
        else:
            # TODO(VitalyFedyunin): Add default collation into df_wrapper
            try:
                import torcharrow.pytorch as tap  # type: ignore[import]

                collation_fn = tap.rec.Default()
            except Exception as e:
                raise RuntimeError(
                    "unable to import default collation function from the TorchArrow"
                ) from e

        tuple_names.append(str(name))
        value = collation_fn(df[name])
        tuple_values.append(value)

    # TODO(VitalyFedyunin): We can dynamically extract types from the tuple_values here
    # TODO(VitalyFedyunin): Instead of ignoring mypy error, make sure tuple_names is not empty
    tpl_cls = namedtuple("CollateResult", tuple_names)  # type: ignore[misc]
    tuple = tpl_cls(*tuple_values)
    return tuple


@functional_datapipe("collate")
class CollatorIterDataPipe(MapperIterDataPipe):
    r"""
    Collates samples from DataPipe to Tensor(s) by a custom collate function (functional name: ``collate``).

    By default, it uses :func:`torch.utils.data.default_collate`.

    .. note::
        While writing a custom collate function, you can import :func:`torch.utils.data.default_collate` for the
        default behavior and `functools.partial` to specify any additional arguments.

    Args:
        datapipe: Iterable DataPipe being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
            Default function collates to Tensor(s) based on data type.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Convert integer data to float Tensor
        >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
        ...     def __init__(self, start, end):
        ...         super(MyIterDataPipe).__init__()
        ...         assert end > start, "this example only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        ...     def __len__(self):
        ...         return self.end - self.start
        >>> ds = MyIterDataPipe(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]
        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        conversion: Callable[..., Any]
        | dict[str | Any, Callable | Any]
        | None = default_collate,
        collate_fn: Callable | None = None,
    ) -> None:
        # TODO(VitalyFedyunin): Replace `Callable[..., Any]` with `Callable[[IColumn], Any]`
        # TODO(VitalyFedyunin): Replace with `Dict[Union[str, IColumn], Union[Callable, Enum]]`
        if collate_fn is not None:
            super().__init__(datapipe, fn=collate_fn)
        else:
            if callable(conversion):
                super().__init__(datapipe, fn=conversion)
            else:
                # TODO(VitalyFedyunin): Validate passed dictionary
                collate_fn = functools.partial(_collate_helper, conversion)
                super().__init__(datapipe, fn=collate_fn)
<|endoftext|># mypy: allow-untyped-defs
import random
from collections.abc import Iterator, Sized
from typing import TypeVar

import torch
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.sampler import Sampler, SequentialSampler


__all__ = [
    "SamplerIterDataPipe",
    "ShufflerIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


class SamplerIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

    Args:
        datapipe: IterDataPipe to sample from
        sampler: Sampler class to generate sample elements from input DataPipe.
            Default is :class:`SequentialSampler` for IterDataPipe
    """

    datapipe: IterDataPipe
    sampler: Sampler

    def __init__(
        self,
        datapipe: IterDataPipe,
        sampler: type[Sampler] = SequentialSampler,
        sampler_args: tuple | None = None,
        sampler_kwargs: dict | None = None,
    ) -> None:
        # pyrefly: ignore [unsafe-overlap]
        if not isinstance(datapipe, Sized):
            raise AssertionError(
                "Sampler class requires input datapipe implemented `__len__`"
            )
        super().__init__()

        self.datapipe = datapipe
        self.sampler_args = () if sampler_args is None else sampler_args
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        self.sampler_kwargs["data_source"] = self.datapipe
        self.sampler = sampler(*self.sampler_args, **self.sampler_kwargs)

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized):
            return len(self.sampler)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("shuffle")
class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Shuffle the input DataPipe with a buffer (functional name: ``shuffle``).

    The buffer with ``buffer_size`` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.

    ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    ``buffer_size`` is required to be greater than or equal to the size of datapipe.

    When it is used with :class:`torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: The IterDataPipe being shuffled
        buffer_size: The buffer size for shuffling (default to ``10000``)
        unbatch_level: Specifies if it is necessary to unbatch source data before
            applying the shuffle

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> shuffle_dp = dp.shuffle()
        >>> list(shuffle_dp)
        [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
    """

    datapipe: IterDataPipe[_T_co]
    buffer_size: int
    _buffer: list[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        *,
        buffer_size: int = 10000,
        unbatch_level: int = 0,
    ) -> None:
        super().__init__()
        # TODO: Performance optimization
        #       buffer can be a fixed size and remove expensive `append()` and `len()` operations
        self._buffer: list[_T_co] = []
        if buffer_size <= 0:
            raise AssertionError("buffer_size should be larger than 0")
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.buffer_size = buffer_size
        self._enabled = True
        self._seed = None
        self._rng = random.Random()

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[_T_co]:
        if not self._enabled:
            yield from self.datapipe
        else:
            for x in self.datapipe:
                if len(self._buffer) == self.buffer_size:
                    idx = self._rng.randint(0, len(self._buffer) - 1)
                    val, self._buffer[idx] = self._buffer[idx], x
                    yield val
                else:
                    self._buffer.append(x)
            while self._buffer:
                idx = self._rng.randint(0, len(self._buffer) - 1)
                yield self._buffer.pop(idx)

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def reset(self) -> None:
        self._buffer = []
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._rng.seed(self._seed)
            self._seed = None

    def __getstate__(self):
        state = (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            self._rng.getstate(),
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    def __del__(self) -> None:
        self._buffer.clear()
<|endoftext|># mypy: allow-untyped-defs
r"""Signal handling for multiprocessing data loading.

NOTE [ Signal handling in multiprocessing data loading ]

In cases like DataLoader, if a worker process dies due to bus error/segfault
or just hang, the main process will hang waiting for data. This is difficult
to avoid on PyTorch side as it can be caused by limited shm, or other
libraries users call in the workers. In this file and `DataLoader.cpp`, we make
our best effort to provide some error message to users when such unfortunate
events happen.

When a _BaseDataLoaderIter starts worker processes, their pids are registered in a
defined in `DataLoader.cpp`: id(_BaseDataLoaderIter) => Collection[ Worker pids ]
via `_set_worker_pids`.

When an error happens in a worker process, the main process receives a SIGCHLD,
and Python will eventually call the handler registered below
(in `_set_SIGCHLD_handler`). In the handler, the `_error_if_any_worker_fails`
call checks all registered worker pids and raises a proper error message to
prevent main process from hanging waiting for data from worker.

Additionally, at the beginning of each worker's `_utils.worker._worker_loop`,
`_set_worker_signal_handlers` is called to register critical signal handlers
(e.g., for SIGSEGV, SIGBUS, SIGFPE, SIGTERM) in C, which just prints an error
message to stderr before triggering the default handler. So a message will also
be printed from the worker process when it is killed by such signals.

See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for the reasoning of
this signal handling design and other mechanism we implement to make our
multiprocessing data loading robust to errors.
"""

import signal
import threading

# Some of the following imported functions are not used in this file, but are to
# be used `_utils.signal_handling.XXXXX`.
from torch._C import (  # noqa: F401
    _error_if_any_worker_fails,
    _remove_worker_pids,
    _set_worker_pids,
    _set_worker_signal_handlers,
)

from . import IS_WINDOWS


_SIGCHLD_handler_set = False
r"""Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process."""


def _set_SIGCHLD_handler() -> None:
    # Windows doesn't support SIGCHLD handler
    if IS_WINDOWS:
        return
    # can't set signal in child threads
    if not isinstance(threading.current_thread(), threading._MainThread):  # type: ignore[attr-defined]
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        # This doesn't catch default handler, but SIGCHLD default handler is a
        # no-op.
        previous_handler = None

    def handler(signum, frame) -> None:
        # This following call uses `waitid` with WNOHANG from C side. Therefore,
        # Python can still get and update the process status successfully.
        _error_if_any_worker_fails()
        if previous_handler is not None:
            if not callable(previous_handler):
                raise AssertionError("previous_handler is not callable")
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True
<|endoftext|># mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **need** to be in global scope since Py2 doesn't support serializing
static methods.
"""

from __future__ import annotations

import os
import queue
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch._utils import ExceptionWrapper

from . import HAS_NUMPY, IS_WINDOWS, MP_STATUS_CHECK_INTERVAL, signal_handling


if TYPE_CHECKING:
    from torch.utils.data import Dataset

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import BOOL, DWORD, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self) -> None:
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(
                SYNCHRONIZE, 0, self.manager_pid
            )

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self) -> bool:
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = (
                    self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
                )
            return not self.manager_dead

else:

    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self) -> bool:
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


_worker_info: WorkerInfo | None = None


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Information about the current DataLoader worker process or thread.

    Attributes:
        id: The current worker id (0 to num_workers - 1)
        num_workers: Total number of workers
        seed: Random seed set for this worker
        dataset: Copy of the dataset object in this worker
        rng: Optional RNG state container. Defaults to None.
        worker_method: Optional worker method ("multiprocessing" or "thread"). Defaults to "multiprocessing".
    """

    id: int
    num_workers: int
    seed: int
    dataset: Dataset
    rng: _RNG | None = None
    worker_method: str | None = "multiprocessing"


def get_worker_info() -> WorkerInfo | None:
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    return _worker_info


r"""Dummy class used to signal the end of an IterableDataset"""


@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int


r"""Dummy class used to resume the fetching when worker reuse is enabled"""


@dataclass(frozen=True)
class _ResumeIteration:
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class _RNG:
    """Container for thread-local random number generator state.

    Used by thread workers to maintain separate RNG state per worker thread
    to avoid race conditions.

    Attributes:
        random_generator: Python random.Random generator for this thread
        torch_generator: PyTorch Generator for this thread
        numpy_generator: NumPy Generator for this thread (None if numpy not available)
    """

    random_generator: random.Random
    torch_generator: torch.Generator
    numpy_generator: object | None = None


# The function `_generate_state` is adapted from `numpy.random.SeedSequence`
# from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx
# It's MIT licensed, here is the copyright:

# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This function generates an array of int32 as the seed for
# `numpy.random`, in order to prevent state collision due to same
# seed and algorithm for `numpy.random` and `random` modules.
# TODO: Implement `SeedSequence` like object for `torch.random`
def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def _worker_loop(
    dataset_kind,
    dataset_holder,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
) -> None:
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    dataset = dataset_holder.pop()

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.multiprocessing._set_thread_name("pt_data_worker")

        torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np

            np.random.seed(np_seed)

        from torch.utils.data import IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        shared_rng = torch.Generator()
        if isinstance(dataset, IterDataPipe):
            if shared_seed is None:
                raise AssertionError(
                    "shared_seed must be provided for IterDataPipe workers"
                )
            shared_rng.manual_seed(shared_seed)
            dataset = apply_random_seed(dataset, shared_rng)

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset
        )

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collation, collate_fn, drop_last
            )
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader worker process {worker_id}"
            )

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    if r.seed is None:
                        raise AssertionError(
                            "resume iteration seed is None for IterDataPipe"
                        )
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last
                )
                continue
            elif r is None:
                # Received the final signal
                if not done_event.is_set() and not iteration_end:
                    raise AssertionError(
                        "Received final signal but neither done_event nor iteration_end is set"
                    )
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, index = r
            data: _IterableDatasetStopIteration | ExceptionWrapper
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
                except Exception as e:
                    if (
                        isinstance(e, StopIteration)
                        and dataset_kind == _DatasetKind.Iterable
                    ):
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where=f"in DataLoader worker process {worker_id}"
                        )
            data_queue.put((idx, data))
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    finally:
        _worker_info = None
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
<|endoftext|># mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

from typing import NoReturn


class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index) -> NoReturn:
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
<|endoftext|>r"""Utility classes & functions for data loading. Code in this folder is mostly used by ../dataloader.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""

import atexit
import sys

# old private location of the ExceptionWrapper that some users rely on:
from torch._utils import ExceptionWrapper


IS_WINDOWS = sys.platform == "win32"


MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""


python_exit_status = False
r"""Whether Python is shutting down. This flag is guaranteed to be set before
the Python core library resources are freed, but Python may already be exiting
for some time when this is set.

Hook to set this flag is `_set_python_exit_flag`, and is inspired by a similar
hook in Python 3.7 multiprocessing library:
https://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327
"""


try:
    import numpy

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


def _set_python_exit_flag() -> None:
    global python_exit_status
    python_exit_status = True


atexit.register(_set_python_exit_flag)


from . import collate, fetch, pin_memory, signal_handling, worker
<|endoftext|># mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **need** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import copy
import queue

import torch
from torch._utils import ExceptionWrapper

from . import MP_STATUS_CHECK_INTERVAL


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device) -> None:
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.multiprocessing._set_thread_name("pt_data_pin")
    torch.accelerator.set_device_index(device_id)

    def do_one_step() -> None:
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(
                    where=f"in pin memory thread for device {device_id}"
                )
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()

    if hasattr(data, "pin_memory"):
        return data.pin_memory()

    if isinstance(data, (str, bytes)):
        return data

    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                # pyrefly: ignore [bad-instantiation]
                return type(data)(
                    # pyrefly: ignore [bad-argument-count]
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}

    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*(pin_memory(sample, device) for sample in data))
        return type(data)(pin_memory(sample, device) for sample in data)

    if isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]

    return data
<|endoftext|># mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These methods are used to collate samples fetched from dataset into Tensor(s).
These **need** to be in global scope since Py2 doesn't support serializing
static methods.

`default_collate` and `default_convert` are exposed to users via 'dataloader.py'.
"""

import collections
import contextlib
import copy
import re
from collections.abc import Callable

import torch


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def default_convert(data):
    r"""
    Convert each NumPy array element into a :class:`torch.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
    If the input is not a NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~torch.utils.data.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~torch.utils.data.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with `int`
        >>> default_convert(0)
        0
        >>> # Example with NumPy array
        >>> default_convert(np.array([0, 1]))
        tensor([0, 1])
        >>> # Example with NamedTuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert(Point(np.array(0), np.array(0)))
        Point(x=tensor(0), y=tensor(0))
        >>> # Example with List
        >>> default_convert([np.array([0, 1]), np.array([2, 3])])
        [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        # array of string classes and object
        if (
            elem_type.__name__ == "ndarray"
            and np_str_obj_array_pattern.search(data.dtype.str) is not None
        ):
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            else:
                return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, d in enumerate(data):
                    clone[i] = default_convert(d)
                return clone
            else:
                return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def collate(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    r"""
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. `default_collate_fn_map`
    provides default collate functions for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        ...     # Extend this function to handle batch of tensors
        ...     return torch.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {torch.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch, strict=False)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))

        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(
            zip(*batch, strict=False)
        )  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_tensor_fn(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError(
            "Batches of nested tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if elem.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        raise RuntimeError(
            "Batches of sparse tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def collate_numpy_array_fn(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    return torch.as_tensor(batch)


def collate_float_fn(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    return torch.tensor(batch)


def collate_str_fn(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    return batch


default_collate_fn_map: dict[type | tuple[type, ...], Callable] = {
    torch.Tensor: collate_tensor_fn
}
with contextlib.suppress(ImportError):
    import numpy as np

    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn


def default_collate(batch):
    r"""
    Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.

    The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(["a", "b", "c"])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{"A": 0, "B": 1}, {"A": 100, "B": 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Two options to extend `default_collate` to handle specific type
        >>> # Option 1: Write custom collate function and invoke `default_collate`
        >>> def custom_collate(batch):
        ...     elem = batch[0]
        ...     if isinstance(elem, CustomType):  # Some custom condition
        ...         return ...
        ...     else:  # Fall back to `default_collate`
        ...         return default_collate(batch)
        >>> # Option 2: In-place modify `default_collate_fn_map`
        >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
        ...     return ...
        >>> default_collate_fn_map.update(CustomType, collate_customtype_fn)
        >>> default_collate(batch)  # Handle `CustomType` automatically
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs
"""
model_dump: a one-stop shop for TorchScript model inspection.

The goal of this tool is to provide a simple way to extract lots of
useful information from a TorchScript model and make it easy for humans
to consume.  It (mostly) replaces zipinfo, common uses of show_pickle,
and various ad-hoc analysis notebooks.

The tool extracts information from the model and serializes it as JSON.
That JSON can then be rendered by an HTML+JS page, either by
loading the JSON over HTTP or producing a fully self-contained page
with all of the code and data burned-in.
"""

# Maintainer notes follow.
"""
The implementation strategy has tension between 3 goals:
- Small file size.
- Fully self-contained.
- Easy, modern JS environment.
Using Preact and HTM achieves 1 and 2 with a decent result for 3.
However, the models I tested with result in ~1MB JSON output,
so even using something heavier like full React might be tolerable
if the build process can be worked out.

One principle I have followed that I think is very beneficial
is to keep the JSON data as close as possible to the model
and do most of the rendering logic on the client.
This makes for easier development (just refresh, usually),
allows for more laziness and dynamism, and lets us add more
views of the same data without bloating the HTML file.

Currently, this code doesn't actually load the model or even
depend on any part of PyTorch.  I don't know if that's an important
feature to maintain, but it's probably worth preserving the ability
to run at least basic analysis on models that cannot be loaded.

I think the easiest way to develop this code is to cd into model_dump and
run "python -m http.server", then load http://localhost:8000/skeleton.html
in the browser.  In another terminal, run
"python -m torch.utils.model_dump --style=json FILE > \
    torch/utils/model_dump/model_info.json"
every time you update the Python code or model.
When you update JS, just refresh.

Possible improvements:
    - Fix various TODO comments in this file and the JS.
    - Make the HTML much less janky, especially the auxiliary data panel.
    - Make the auxiliary data panel start small, expand when
      data is available, and have a button to clear/contract.
    - Clean up the JS.  There's a lot of copypasta because
      I don't really know how to use Preact.
    - Make the HTML render and work nicely inside a Jupyter notebook.
    - Add the ability for JS to choose the URL to load the JSON based
      on the page URL (query or hash).  That way we could publish the
      inlined skeleton once and have it load various JSON blobs.
    - Add a button to expand all expandable sections so ctrl-F works well.
    - Add hyperlinking from data to code, and code to code.
    - Add hyperlinking from debug info to Diffusion.
    - Make small tensor contents available.
    - Do something nice for quantized models
      (they probably don't work at all right now).
"""

import argparse
import io
import itertools
import json
import os
import pickle
import pprint
import re
import sys
import urllib.parse
import zipfile
from pathlib import Path
import warnings

import torch.utils.show_pickle


DEFAULT_EXTRA_FILE_SIZE_LIMIT = 16 * 1024

__all__ = ['get_storage_info', 'hierarchical_pickle', 'get_model_info', 'get_inline_skeleton',
           'burn_in_info', 'get_info_and_burn_skeleton']

def get_storage_info(storage):
    if not isinstance(storage, torch.utils.show_pickle.FakeObject):
        raise AssertionError(f"storage is not FakeObject: {type(storage)}")
    if storage.module != "pers":
        raise AssertionError(f"storage.module is not 'pers': {storage.module!r}")
    if storage.name != "obj":
        raise AssertionError(f"storage.name is not 'obj': {storage.name!r}")
    if storage.state is not None:
        raise AssertionError(f"storage.state is not None: {storage.state!r}")
    if not isinstance(storage.args, tuple):
        raise AssertionError(f"storage.args is not a tuple: {type(storage.args)}")
    if len(storage.args) != 1:
        raise AssertionError(f"len(storage.args) is not 1: {len(storage.args)}")
    sa = storage.args[0]
    if not isinstance(sa, tuple):
        raise AssertionError(f"sa is not a tuple: {type(sa)}")
    if len(sa) != 5:
        raise AssertionError(f"len(sa) is not 5: {len(sa)}")
    if sa[0] != "storage":
        raise AssertionError(f"sa[0] is not 'storage': {sa[0]!r}")
    if not isinstance(sa[1], torch.utils.show_pickle.FakeClass):
        raise AssertionError(f"sa[1] is not FakeClass: {type(sa[1])}")
    if sa[1].module != "torch":
        raise AssertionError(f"sa[1].module is not 'torch': {sa[1].module!r}")
    if not sa[1].name.endswith("Storage"):
        raise AssertionError(f"sa[1].name does not end with 'Storage': {sa[1].name!r}")
    storage_info = [sa[1].name.replace("Storage", "")] + list(sa[2:])
    return storage_info


def hierarchical_pickle(data):
    if isinstance(data, (bool, int, float, str, type(None))):
        return data
    if isinstance(data, list):
        return [hierarchical_pickle(d) for d in data]
    if isinstance(data, tuple):
        return {
            "__tuple_values__": hierarchical_pickle(list(data)),
        }
    if isinstance(data, dict):
        return {
            "__is_dict__": True,
            "keys": hierarchical_pickle(list(data.keys())),
            "values": hierarchical_pickle(list(data.values())),
        }
    if isinstance(data, torch.utils.show_pickle.FakeObject):
        typename = f"{data.module}.{data.name}"
        if (
            typename.startswith(('__torch__.', 'torch.jit.LoweredWrapper.', 'torch.jit.LoweredModule.'))
        ):
            if data.args != ():
                raise AssertionError("data.args is not ()")
            return {
                "__module_type__": typename,
                "state": hierarchical_pickle(data.state),
            }
        if typename == "torch._utils._rebuild_tensor_v2":
            if data.state is not None:
                raise AssertionError("data.state is not None")
            storage, offset, size, stride, requires_grad, *_ = data.args
            storage_info = get_storage_info(storage)
            return {"__tensor_v2__": [storage_info, offset, size, stride, requires_grad]}
        if typename == "torch._utils._rebuild_qtensor":
            if data.state is not None:
                raise AssertionError("data.state is not None")
            storage, offset, size, stride, quantizer, requires_grad, *_ = data.args
            storage_info = get_storage_info(storage)
            if not isinstance(quantizer, tuple):
                raise AssertionError("quantizer is not a tuple")
            if not isinstance(quantizer[0], torch.utils.show_pickle.FakeClass):
                raise AssertionError("quantizer[0] is not a FakeClass")
            if quantizer[0].module != "torch":
                raise AssertionError("quantizer[0].module is not torch")
            if quantizer[0].name == "per_tensor_affine":
                if len(quantizer) != 3:
                    raise AssertionError("len(quantizer) is not 3")
                if not isinstance(quantizer[1], float):
                    raise AssertionError("quantizer[1] is not a float")
                if not isinstance(quantizer[2], int):
                    raise AssertionError("quantizer[2] is not an int")
                quantizer_extra = list(quantizer[1:3])
            else:
                quantizer_extra = []
            quantizer_json = [quantizer[0].name] + quantizer_extra
            return {"__qtensor__": [storage_info, offset, size, stride, quantizer_json, requires_grad]}
        if typename == "torch.jit._pickle.restore_type_tag":
            if data.state is not None:
                raise AssertionError("data.state is not None")
            obj, typ = data.args
            if not isinstance(typ, str):
                raise AssertionError("typ is not a string")
            return hierarchical_pickle(obj)
        if re.fullmatch(r"torch\.jit\._pickle\.build_[a-z]+list", typename):
            if data.state is not None:
                raise AssertionError("data.state is not None")
            ls, = data.args
            if not isinstance(ls, list):
                raise AssertionError("ls is not a list")
            return hierarchical_pickle(ls)
        if typename == "torch.device":
            if data.state is not None:
                raise AssertionError("data.state is not None")
            name, = data.args
            if not isinstance(name, str):
                raise AssertionError("name is not a string")
            # Just forget that it was a device and return the name.
            return name
        if typename == "builtin.UnicodeDecodeError":
            if data.state is not None:
                raise AssertionError("data.state is not None")
            msg, = data.args
            if not isinstance(msg, str):
                raise AssertionError("msg is not a string")
            # Hack: Pretend this is a module so we don't need custom serialization.
            # Hack: Wrap the message in a tuple so it looks like a nice state object.
            # TODO: Undo at least that second hack.  We should support string states.
            return {
                "__module_type__": typename,
                "state": hierarchical_pickle((msg,)),
            }
        raise Exception(f"Can't prepare fake object of type for JS: {typename}")  # noqa: TRY002
    raise Exception(f"Can't prepare data of type for JS: {type(data)}")  # noqa: TRY002


def get_model_info(
        path_or_file,
        title=None,
        extra_file_size_limit=DEFAULT_EXTRA_FILE_SIZE_LIMIT):
    """Get JSON-friendly information about a model.

    The result is suitable for being saved as model_info.json,
    or passed to burn_in_info.
    """

    if isinstance(path_or_file, os.PathLike):
        default_title = os.fspath(path_or_file)
        file_size = path_or_file.stat().st_size  # type: ignore[attr-defined]
    elif isinstance(path_or_file, str):
        default_title = path_or_file
        file_size = Path(path_or_file).stat().st_size
    else:
        default_title = "buffer"
        path_or_file.seek(0, io.SEEK_END)
        file_size = path_or_file.tell()
        path_or_file.seek(0)

    title = title or default_title

    with zipfile.ZipFile(path_or_file) as zf:
        path_prefix = None
        zip_files = []
        # pyrefly: ignore [bad-assignment]
        for zi in zf.infolist():
            prefix = re.sub("/.*", "", zi.filename)
            if path_prefix is None:
                path_prefix = prefix
            elif prefix != path_prefix:
                raise Exception(f"Mismatched prefixes: {path_prefix} != {prefix}")  # noqa: TRY002
            zip_files.append(
                {
                    "filename": zi.filename,
                    "compression": zi.compress_type,
                    "compressed_size": zi.compress_size,
                    "file_size": zi.file_size,
                }
            )
        if path_prefix is None:
            raise AssertionError("path_prefix is None")
        version = zf.read(path_prefix + "/version").decode("utf-8").strip()

        def get_pickle(name):
            if path_prefix is None:
                raise AssertionError("path_prefix is None")
            with zf.open(path_prefix + f"/{name}.pkl") as handle:
                raw = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
                return hierarchical_pickle(raw)

        model_data = get_pickle("data")
        constants = get_pickle("constants")

        # Intern strings that are likely to be reused.
        # Pickle automatically detects shared structure,
        # so reused strings are stored efficiently.
        # However, JSON has no way of representing this,
        # so we have to do it manually.
        interned_strings : dict[str, int] = {}

        def intern(s):
            if s not in interned_strings:
                interned_strings[s] = len(interned_strings)
            return interned_strings[s]

        code_files = {}
        for zi in zf.infolist():
            if not zi.filename.endswith(".py"):
                continue
            with zf.open(zi) as handle:
                raw_code = handle.read()
            with zf.open(zi.filename + ".debug_pkl") as handle:
                raw_debug = handle.read()

            # Parse debug info and add begin/end markers if not present
            # to ensure that we cover the entire source code.
            debug_info_t = pickle.loads(raw_debug)
            text_table = None

            if (len(debug_info_t) == 3 and
                    isinstance(debug_info_t[0], str) and
                    debug_info_t[0] == 'FORMAT_WITH_STRING_TABLE'):
                _, text_table, content = debug_info_t

                def parse_new_format(line):
                    # (0, (('', '', 0), 0, 0))
                    num, ((text_indexes, fname_idx, offset), start, end), tag = line
                    text = ''.join(text_table[x] for x in text_indexes)  # type: ignore[index]
                    fname = text_table[fname_idx]  # type: ignore[index]
                    return num, ((text, fname, offset), start, end), tag

                debug_info_t = map(parse_new_format, content)

            debug_info = list(debug_info_t)
            if not debug_info:
                debug_info.append((0, (('', '', 0), 0, 0)))
            if debug_info[-1][0] != len(raw_code):
                debug_info.append((len(raw_code), (('', '', 0), 0, 0)))

            code_parts = []
            for di, di_next in itertools.pairwise(debug_info):
                start, source_range, *_ = di
                end = di_next[0]
                if end <= start:
                    raise AssertionError("end is not greater than start")
                source, s_start, s_end = source_range
                s_text, s_file, s_line = source
                # TODO: Handle this case better.  TorchScript ranges are in bytes,
                # but JS doesn't really handle byte strings.
                # if bytes and chars are not equivalent for this string,
                # zero out the ranges so we don't highlight the wrong thing.
                if len(s_text) != len(s_text.encode("utf-8")):
                    s_start = 0
                    s_end = 0
                text = raw_code[start:end]
                code_parts.append([text.decode("utf-8"), intern(s_file), s_line, intern(s_text), s_start, s_end])
            code_files[zi.filename] = code_parts

        extra_files_json_pattern = re.compile(re.escape(path_prefix) + "/extra/.*\\.json")
        extra_files_jsons = {}
        for zi in zf.infolist():
            if not extra_files_json_pattern.fullmatch(zi.filename):
                continue
            if zi.file_size > extra_file_size_limit:
                continue
            with zf.open(zi) as handle:
                try:
                    json_content = json.load(handle)
                    extra_files_jsons[zi.filename] = json_content
                except json.JSONDecodeError:
                    extra_files_jsons[zi.filename] = "INVALID JSON"

        always_render_pickles = {
            "bytecode.pkl",
        }
        extra_pickles = {}
        for zi in zf.infolist():
            if not zi.filename.endswith(".pkl"):
                continue
            with zf.open(zi) as handle:
                # TODO: handle errors here and just ignore the file?
                # NOTE: For a lot of these files (like bytecode),
                # we could get away with just unpickling, but this should be safer.
                obj = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
            buf = io.StringIO()
            pprint.pprint(obj, buf)
            contents = buf.getvalue()
            # Checked the rendered length instead of the file size
            # because pickles with shared structure can explode in size during rendering.
            if os.path.basename(zi.filename) not in always_render_pickles and \
                    len(contents) > extra_file_size_limit:
                continue
            extra_pickles[zi.filename] = contents

    return {
        "model": {
            "title": title,
            "file_size": file_size,
            "version": version,
            "zip_files": zip_files,
            "interned_strings": list(interned_strings),
            "code_files": code_files,
            "model_data": model_data,
            "constants": constants,
            "extra_files_jsons": extra_files_jsons,
            "extra_pickles": extra_pickles,
        }
    }


def get_inline_skeleton():
    """Get a fully-inlined skeleton of the frontend.

    The returned HTML page has no external network dependencies for code.
    It can load model_info.json over HTTP, or be passed to burn_in_info.
    """

    import importlib.resources

    # pyrefly: ignore [bad-argument-type]
    skeleton = importlib.resources.read_text(__package__, "skeleton.html")
    # pyrefly: ignore [bad-argument-type]
    js_code = importlib.resources.read_text(__package__, "code.js")
    for js_module in ["preact", "htm"]:
        # pyrefly: ignore [bad-argument-type]
        js_lib = importlib.resources.read_binary(__package__, f"{js_module}.mjs")
        js_url = "data:application/javascript," + urllib.parse.quote(js_lib)
        js_code = js_code.replace(f"https://unpkg.com/{js_module}?module", js_url)
    skeleton = skeleton.replace(' src="./code.js">', ">\n" + js_code)
    return skeleton


def burn_in_info(skeleton, info):
    """Burn model info into the HTML skeleton.

    The result will render the hard-coded model info and
    have no external network dependencies for code or data.
    """

    # Note that Python's json serializer does not escape slashes in strings.
    # Since we're inlining this JSON directly into a script tag, a string
    # containing "</script>" would end the script prematurely and
    # mess up our page.  Unconditionally escape fixes that.
    return skeleton.replace(
        "BURNED_IN_MODEL_INFO = null",
        "BURNED_IN_MODEL_INFO = " + json.dumps(info, sort_keys=True).replace("/", "\\/"))


def get_info_and_burn_skeleton(path_or_bytesio, **kwargs):
    model_info = get_model_info(path_or_bytesio, **kwargs)
    skeleton = get_inline_skeleton()
    page = burn_in_info(skeleton, model_info)
    return page


def main(argv, *, stdout=None) -> None:
    warnings.warn("torch.utils.model_dump is deprecated and will be removed in a future PyTorch release.", stacklevel=2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["json", "html"])
    parser.add_argument("--title")
    parser.add_argument("model")
    args = parser.parse_args(argv[1:])

    info = get_model_info(args.model, title=args.title)

    output = stdout or sys.stdout

    if args.style == "json":
        output.write(json.dumps(info, sort_keys=True) + "\n")
    elif args.style == "html":
        skeleton = get_inline_skeleton()
        page = burn_in_info(skeleton, info)
        output.write(page)
    else:
        raise Exception("Invalid style")  # noqa: TRY002
<|endoftext|>import sys
from typing import Optional as _Optional, TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from torch.serialization import LoadEndianness as _LoadEndianess

from torch.utils._config_module import install_config_module as _install_config_module


class load:
    mmap: bool = False
    endianness: _Optional["_LoadEndianess"] = None
    # MAP_PRIVATE = 2
    mmap_flags: int | None = None if sys.platform == "win32" else 2
    calculate_storage_offsets: bool = False


class save:
    compute_crc32: bool = True
    use_pinned_memory_for_d2h: bool = False
    storage_alignment: int = 64


_install_config_module(sys.modules[__name__])
<|endoftext|>"""
This is a simple interpreter for Sympy expressions that dispatches to
classes following the torch._inductor.virtualized calling convention.
For directness, the interpreter takes the handler directly rather than
consulting the TLS.  It does not use most of the methods on the full
handler; only those with corresponding Sympy expressions.  To see an example
of a full handler, see torch.utils._sympy.value_ranges.ValueRangeAnalysis.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

import sympy
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom

import torch

from .functions import (
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    CeilToInt,
    CleanDiv,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Min,
    Mod,
    ModularIndexing,
    OpaqueUnaryFn_log2,
    PowByNatural,
    PythonMod,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToFloat,
    TruncToInt,
    Where,
)


log = logging.getLogger(__name__)


# TODO: Dedupe this with SYMPY_INTERP


@functools.cache
def handlers() -> dict[type[sympy.Basic], str]:
    # TODO add CeilDiv (it doesn't appear in the index_expr)

    # TODO default to some decompositions if the interpreter doesn't have them
    # like decomposing ModularIndexing or implementing Le(a,b) as Ge(b, a)

    HANDLERS = {
        sympy.Or: "or_",
        sympy.And: "and_",
        sympy.Eq: "eq",
        sympy.Ne: "ne",
        sympy.Lt: "lt",
        sympy.Gt: "gt",
        sympy.Le: "le",
        sympy.Ge: "ge",
        sympy.Not: "not_",
        IntTrueDiv: "int_truediv",
        FloatTrueDiv: "truediv",
        FloorDiv: "floordiv",
        CleanDiv: "floordiv",  # TODO: hmm?
        TruncToFloat: "trunc",
        Where: "where",
        sympy.Add: "add",
        sympy.Mul: "mul",
        FloatPow: "pow",
        PowByNatural: "pow_by_natural",
        # sympy simplifies x * x into Pow(x, 2), so we need to handle this.
        # Do NOT use builtin Pow for floats
        # TODO: There is a hazard here, if we have float * float it will
        # also get turned into Pow(float, 2) but we don't want this because
        # pow_by_natural is assumed to only be integers.  Probably the fix is
        # to add a FloatMul to impede this optimization
        sympy.Pow: "pow_by_natural",
        Mod: "mod",
        PythonMod: "python_mod",
        # TODO: Inductor can generate these, but it's ill-specified which
        # semantics were intended here.  Needs to be cleaned up along with
        # FloorDiv in a bigger cleanup
        sympy.Mod: "mod",
        sympy.Abs: "abs",
        sympy.log: "log",
        sympy.exp: "exp",
        sympy.Min: "minimum",
        sympy.Max: "maximum",
        Min: "minimum",
        Max: "maximum",
        ModularIndexing: "modular_indexing",
        sympy.functions.elementary.piecewise.ExprCondPair: "expr_cond_pair",
        sympy.Piecewise: "piecewise",
        Identity: "identity",
        IsNonOverlappingAndDenseIndicator: "is_non_overlapping_and_dense_indicator",
        RoundDecimal: "round_decimal",
        # TODO: do the rest of the opaque unary functions...
        OpaqueUnaryFn_log2: "log2",
        BitwiseFn_bitwise_and: "bitwise_and",
        BitwiseFn_bitwise_or: "bitwise_or",
        BitwiseFn_bitwise_xor: "bitwise_xor",
    }
    # TODO: This is kind of pointless, we shouldn't be generating sympy.sin
    # for these functions, they should be Opaque instead
    for name in ["cos", "sin", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan"]:
        HANDLERS[getattr(sympy, name)] = name

    return HANDLERS


ASSOCIATIVE_OPS = {"minimum", "maximum", "mul", "add", "and_", "or_"}


def _run_sympy_handler(
    analysis: Any,
    args: list[Any],
    expr: sympy.Basic,
    index_dtype: torch.dtype = torch.int64,
) -> Any:
    # Special cases
    if isinstance(expr, sympy.Pow) and isinstance(
        expr.args[1], sympy.core.numbers.Half
    ):
        return analysis.sqrt(args[0])
    if isinstance(expr, ToFloat):
        return analysis.to_dtype(args[0], torch.float64)

    # These handlers are special because they take an extra dtype argument
    # specifying what they should convert to, and we need to appropriately set
    # this up when we convert from Sympy.  A reasonable default when you
    # are translating is to conservatively do int64, and then narrow these
    # arguments later when you discover you can narrow the index range.  But
    # if you already know that 32-bit indexing is OK, you can directly do the
    # sympy translation with index_dtype=torch.int32
    INDEX_DTYPE_HANDLERS = {
        TruncToInt: "trunc_to_int",
        sympy.floor: "floor_to_int",
        sympy.ceiling: "ceil_to_int",
        FloorToInt: "floor_to_int",
        CeilToInt: "ceil_to_int",
        RoundToInt: "round_to_int",
    }
    if (handler_name := INDEX_DTYPE_HANDLERS.get(expr.func)) is not None:
        return getattr(analysis, handler_name)(*args, index_dtype)

    # Fastpath for n-ary integral addition
    if expr.func is sympy.Add and expr.is_integer and hasattr(analysis, "sym_sum"):
        r = analysis.sym_sum(args)
        log.debug("sym_sum(%s) -> %s", args, r)
        return r

    if expr.func is sympy.Pow:
        exp = expr.args[1]
        if exp.is_integer and exp.is_nonnegative:
            handler_name = "pow_by_natural"
        else:
            handler_name = "pow"
    elif hasattr(expr.func, "_torch_handler_name"):
        handler_name = expr.func._torch_handler_name
    else:
        handler_name = handlers()[expr.func]
    handler = getattr(analysis, handler_name)
    try:
        if handler_name in ASSOCIATIVE_OPS:
            if len(args) <= 1:
                raise AssertionError("associative op needs >1 args")
            acc = handler(args[0], args[1])
            for i in range(2, len(args)):
                acc = handler(acc, args[i])
            log.debug("%s(%s) -> %s", handler_name, args, acc)
            return acc
        else:
            r = handler(*args)
            log.debug("%s(%s) -> %s", handler_name, args, r)
            return r
    except NotImplementedError:
        raise
    except Exception:
        log.warning("failed while executing %s(%s)", handler_name, args)
        raise


_nil = object()


def sympy_interp(
    analysis: Any,
    env: dict[sympy.Symbol, Any],
    expr: sympy.Expr | SympyBoolean,
    *,
    index_dtype: torch.dtype = torch.int64,
    missing_handler: Callable[[sympy.Symbol], object] | None = None,
) -> Any:
    # Handle base cases
    dtype = None
    if isinstance(expr, BooleanAtom):
        dtype = torch.bool
    elif isinstance(expr, sympy.Integer):
        dtype = torch.int64
    elif isinstance(expr, sympy.Number):
        dtype = torch.double

    if dtype is not None:
        return analysis.constant(expr, dtype)
    elif isinstance(expr, sympy.Symbol):
        if (r := env.get(expr, _nil)) is not _nil:
            return r
        elif missing_handler:
            return missing_handler(expr)
        else:
            raise KeyError(expr)

    # Recursive case
    return _run_sympy_handler(
        analysis,
        [
            sympy_interp(
                analysis,
                env,
                arg,
                index_dtype=index_dtype,
                missing_handler=missing_handler,
            )
            for arg in expr.args
        ],
        expr,
        index_dtype=index_dtype,
    )
<|endoftext|>from typing import Any

import sympy
from sympy.multipledispatch import dispatch


__all__ = ["SingletonInt"]


class SingletonInt(sympy.AtomicExpr):
    # This is probably not super important unless we are in multiple dispatch
    # situations with other more exotic Expr types.
    _op_priority = 99999
    _val: int
    _coeff: int

    def __new__(
        cls, *args: Any, coeff: int | None = None, **kwargs: Any
    ) -> "SingletonInt":
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    # The semantics of this class should match that of NestedIntSymNodeImpl in
    # c10/core/NestedIntSymNodeImpl.h
    def __init__(self, val: int, *, coeff: int = 1) -> None:
        self._val = val
        self._coeff = coeff
        super().__init__()

    # See NOTE [ Inequalities with nested int ]
    def _eval_Eq(self, other: sympy.Basic) -> sympy.logic.boolalg.BooleanAtom:
        if (
            isinstance(other, SingletonInt)
            and other._val == self._val
            and self._coeff == other._coeff
        ):
            return sympy.true
        else:
            return sympy.false

    # This is necessary so that calling expr.free_symbols on exprs that contain
    # this Singleton does not error
    @property
    def free_symbols(self) -> set[sympy.Symbol]:
        return set()

    def __mul__(self, other: int) -> "SingletonInt":
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other: int) -> "SingletonInt":
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    # Make sure we promptly raise an error instead of falling back to building
    # an expression tree. There are probably more ops, how can we be exhaustive?
    def __add__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __sub__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __truediv__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __floordiv__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __mod__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")


# See NOTE [ Inequalities with nested int ]
@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a: sympy.Integer, b: SingletonInt) -> sympy.logic.boolalg.BooleanAtom:
    if a < 2:
        return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, sympy.Integer)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: sympy.Integer) -> sympy.logic.boolalg.BooleanAtom:
    if b <= 2:
        return sympy.true
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, SingletonInt)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: SingletonInt) -> sympy.logic.boolalg.BooleanAtom:
    if a._val == b._val:
        if a._coeff >= b._coeff:
            return sympy.true
        else:
            return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")
<|endoftext|># mypy: allow-untyped-defs
import math
import operator
from typing import NoReturn

import sympy

import torch
from torch.utils._sympy.functions import (
    _keep_float,
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_log2,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToInt,
)


# The sympy interpretation of operators.  It will also sometimes work with
# plain int/float, but if you do certain operations you will get out a
# sympy.Basic in the end.  If you want the Python/FX traceable interpretation,
# check PythonReferenceAnalysis.
# NB: For magic methods this needs to use normal magic methods
# so that test_magic_methods works
class ReferenceAnalysis:
    @staticmethod
    def constant(c, dtype):
        return sympy.sympify(c)

    @staticmethod
    def or_(a, b):
        return a | b

    @staticmethod
    def and_(a, b):
        return a & b

    @staticmethod
    def eq(a, b):
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        return a < b

    @staticmethod
    def gt(a, b):
        return a > b

    @staticmethod
    def le(a, b):
        return a <= b

    @staticmethod
    def ge(a, b):
        return a >= b

    @staticmethod
    def not_(a):
        if isinstance(a, bool):
            raise AssertionError("not_ needs sympy expr")
        return ~a

    @staticmethod
    def reciprocal(x):
        return FloatTrueDiv(1.0, x)

    @staticmethod
    def square(x):
        return PowByNatural(x, 2)

    @staticmethod
    def trunc_to_int(x, dtype):
        return TruncToInt(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        return sympy.ceiling(x)

    @staticmethod
    def floor_to_int(x, dtype):
        return sympy.floor(x)

    @staticmethod
    def floor(x):
        return _keep_float(sympy.floor)(x)

    @staticmethod
    def ceil(x):
        return _keep_float(sympy.ceiling)(x)

    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return ToFloat(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def mod(x, y):
        return Mod(x, y)

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def truediv(a, b):
        return FloatTrueDiv(a, b)

    @staticmethod
    def int_truediv(a, b):
        return IntTrueDiv(a, b)

    @staticmethod
    def floordiv(a, b):
        return FloorDiv(a, b)

    @staticmethod
    def truncdiv(a, b) -> NoReturn:
        raise NotImplementedError("TODO: truncdiv")

    @staticmethod
    def add(a, b):
        return _keep_float(operator.add)(a, b)

    @classmethod
    def sym_sum(cls, args):
        return sympy.Add(*args)

    @staticmethod
    def mul(a, b):
        return _keep_float(operator.mul)(a, b)

    @staticmethod
    def sub(a, b):
        return _keep_float(operator.sub)(a, b)

    @staticmethod
    def exp(x):
        return OpaqueUnaryFn_exp(x)

    @staticmethod
    def log(x):
        return OpaqueUnaryFn_log(x)

    @staticmethod
    def log2(x):
        return OpaqueUnaryFn_log2(x)

    @staticmethod
    def sqrt(x):
        return OpaqueUnaryFn_sqrt(x)

    @staticmethod
    def pow(a, b):
        # pyrefly: ignore [bad-argument-count, bad-argument-type]
        return _keep_float(FloatPow)(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        return PowByNatural(a, b)

    @staticmethod
    def minimum(a, b):
        return Min(a, b)

    @staticmethod
    def maximum(a, b):
        return Max(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        return RoundToInt(a)

    @staticmethod
    def round_decimal(a, b):
        return RoundDecimal(a, b)

    @staticmethod
    def bitwise_and(a, b):
        return BitwiseFn_bitwise_and(a, b)

    @staticmethod
    def bitwise_or(a, b):
        return BitwiseFn_bitwise_or(a, b)

    @staticmethod
    def bitwise_xor(a, b):
        return BitwiseFn_bitwise_xor(a, b)


# Unlike ReferenceAnalysis, does NOT sympyify, instead, works with plain
# Python types and is FX traceable.  Inheritance here is purely for code
# sharing (TODO: considering splitting out a BaseReferenceAnalysis).
class PythonReferenceAnalysis(ReferenceAnalysis):
    @staticmethod
    def constant(c, dtype):
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")

    @staticmethod
    def not_(a):
        return torch.sym_not(a)

    @classmethod
    def sym_sum(cls, args):
        if len(args) == 0:
            return 0
        if len(args) == 1:
            return args[0]
        acc = cls.add(args[0], args[1])
        for i in range(2, len(args)):
            acc = cls.add(acc, args[i])
        return acc

    @staticmethod
    def floordiv(a, b):
        return a // b

    @staticmethod
    def mod(x, y):
        return x % y

    @staticmethod
    def python_mod(x, y):
        return x % y

    @staticmethod
    def truncdiv(a, b):
        return a / b

    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return torch.sym_float(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def exp(x) -> NoReturn:
        raise AssertionError("exp is not valid shape sympy expr")

    @staticmethod
    def log(x) -> NoReturn:
        raise AssertionError("log is not valid shape sympy expr")

    @staticmethod
    def log2(x):
        return torch._sym_log2(x)  # type: ignore[attr-defined]

    @staticmethod
    def sqrt(x):
        return torch._sym_sqrt(x)  # type: ignore[attr-defined]

    @staticmethod
    def minimum(a, b):
        return torch.sym_min(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.sym_max(a, b)

    @staticmethod
    def floor_to_int(x, dtype):
        return math.floor(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        return math.ceil(x)

    @staticmethod
    def floor(x):
        return float(math.floor(x))

    @staticmethod
    def ceil(x):
        return float(math.ceil(x))

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def pow_by_natural(a, b):
        # Pray that safe_pow is not needed here lol.  In particular, this
        # never participates in VR low/high ranges, so overflow should be
        # unlikely
        return a**b

    @staticmethod
    def round_to_int(a, dtype):
        return round(a)

    @staticmethod
    def round_decimal(a, b):
        return round(a, ndigits=b)

    @staticmethod
    def bitwise_and(a, b):
        return a & b

    @staticmethod
    def bitwise_or(a, b):
        return a | b

    @staticmethod
    def bitwise_xor(a, b):
        return a ^ b

    @staticmethod
    def expr_cond_pair(expr, cond):
        return (expr, cond)

    @staticmethod
    def piecewise(*pairs):
        # Build nested sym_ite from right to left.
        # Piecewise((e1, c1), (e2, c2), ..., (en, True)) becomes
        # sym_ite(c1, e1, sym_ite(c2, e2, ... en))
        result = pairs[-1][0]
        for expr, cond in reversed(pairs[:-1]):
            result = torch.sym_ite(cond, expr, result)
        return result


# Like PythonReferenceAnalysis, but some export-unfriendly choices of
# operators to make things faster
class OptimizedPythonReferenceAnalysis(PythonReferenceAnalysis):
    @staticmethod
    def sym_sum(args):
        return torch.sym_sum(args)

    @staticmethod
    def floor_to_int(x, dtype):
        if isinstance(x, (int, float)):
            return math.floor(x)
        return x

    @staticmethod
    def ceil_to_int(x, dtype):
        if isinstance(x, (int, float)):
            return math.ceil(x)
        return x

    @staticmethod
    def trunc_to_int(x, dtype):
        if isinstance(x, (int, float)):
            return math.trunc(x)
        return x


def _to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.ops.prims.convert_element_type.default(x, dtype)


# Suppose we have some int/float arguments.  This diagram commutes:
#
#   int/float  -- PythonReferenceAnalysis.op -->  int/float
#       |                                           |
#       |                                           |
#      torch.tensor(..., dtype=torch.int64/torch.float64)
#       |                                           |
#       V                                           V
#    Tensor    -- TensorReferenceAnalysis.op -->  Tensor
#
# NB: int before and after must be representable in int64 (we will
# insert guards accordingly.)
#
# This is guaranteed to be FX traceable with OpOverloads only.
class TensorReferenceAnalysis:
    # NB: This is actually dead, because with Proxy tracing the factory
    # function isn't traced correctly.  Here for completeness.
    @staticmethod
    def constant(c, dtype):
        d: int | float | bool
        if dtype is torch.int64:
            d = int(c)
        elif dtype is torch.double:
            d = float(c)
        elif dtype is torch.bool:
            d = bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")
        return torch.ops.aten.scalar_tensor.default(d, dtype=dtype)

    @staticmethod
    def or_(a, b):
        return torch.ops.aten.logical_or.default(a, b)

    @staticmethod
    def and_(a, b):
        return torch.ops.aten.logical_and.default(a, b)

    @staticmethod
    def bitwise_and(a, b):
        return torch.ops.aten.bitwise_and(a, b)

    @staticmethod
    def bitwise_or(a, b):
        return torch.ops.aten.bitwise_or(a, b)

    @staticmethod
    def bitwise_xor(a, b):
        return torch.ops.aten.bitwise_xor(a, b)

    @staticmethod
    def eq(a, b):
        return torch.ops.aten.eq.Tensor(a, b)

    @classmethod
    def ne(cls, a, b):
        return torch.ops.aten.ne.Tensor(a, b)

    @staticmethod
    def lt(a, b):
        return torch.ops.aten.lt.Tensor(a, b)

    @staticmethod
    def gt(a, b):
        return torch.ops.aten.gt.Tensor(a, b)

    @staticmethod
    def le(a, b):
        return torch.ops.aten.le.Tensor(a, b)

    @staticmethod
    def ge(a, b):
        return torch.ops.aten.ge.Tensor(a, b)

    @staticmethod
    def not_(a):
        return torch.ops.aten.logical_not.default(a)

    @staticmethod
    def reciprocal(x):
        return torch.ops.aten.reciprocal.default(x)

    @staticmethod
    def square(x):
        # TODO: maybe composite implicit autograd doesn't work here?
        return torch.ops.aten.square.default(x)

    @staticmethod
    def trunc_to_int(x, dtype):
        return _to_dtype(torch.ops.aten.trunc.default(x), dtype)

    @staticmethod
    def ceil_to_int(x, dtype):
        return _to_dtype(torch.ops.aten.ceil.default(x), dtype)

    @staticmethod
    def floor_to_int(x, dtype):
        return _to_dtype(torch.ops.aten.floor.default(x), dtype)

    @staticmethod
    def floor(x):
        return torch.ops.aten.floor.default(x)

    @staticmethod
    def ceil(x):
        return torch.ops.aten.ceil.default(x)

    @staticmethod
    def to_dtype(x, dtype):
        return _to_dtype(x, dtype)

    @staticmethod
    def mod(x, y) -> NoReturn:
        # TODO: https://github.com/pytorch/pytorch/pull/133654
        raise NotImplementedError(
            "no C-style modulus operation available from frontend atm"
        )

    @staticmethod
    def abs(x):
        return torch.ops.aten.abs.default(x)

    @staticmethod
    def neg(x):
        return torch.ops.aten.neg.default(x)

    @staticmethod
    def truediv(a, b):
        return torch.ops.aten.true_divide.Tensor(a, b)

    @staticmethod
    def int_truediv(a, b):
        raise NotImplementedError(
            "Python int truediv difficult to implement in PyTorch atm"
        )

        # TODO: This is wrong, CPython has a custom implementation of true
        # division that results in higher precision when the floats are
        # sufficiently large.  Short term fix: add a guard here
        # pyrefly: ignore [unreachable]
        return torch.ops.aten.true_divide.default(
            _to_dtype(a, torch.float64), _to_dtype(b, torch.float64)
        )

    @staticmethod
    def floordiv(a, b):
        return torch.ops.aten.div.Tensor_mode(a, b, rounding_mode="floor")

    @staticmethod
    def truncdiv(a, b) -> NoReturn:
        raise NotImplementedError(
            "no C-style truncdiv operation available from frontend atm"
        )

    @staticmethod
    def add(a, b):
        return torch.ops.aten.add.Tensor(a, b)

    @staticmethod
    def mul(a, b):
        return torch.ops.aten.mul.Tensor(a, b)

    @staticmethod
    def sub(a, b):
        return torch.ops.aten.sub.Tensor(a, b)

    @staticmethod
    def exp(x):
        return torch.ops.aten.exp.default(x)

    @staticmethod
    def log(x):
        return torch.ops.aten.log.default(x)

    @staticmethod
    def log2(x):
        return torch.ops.aten.log2.default(x)

    @staticmethod
    def sqrt(x):
        return torch.ops.aten.sqrt.default(x)

    @staticmethod
    def sin(x):
        return torch.ops.aten.sin.default(x)

    @staticmethod
    def cos(x):
        return torch.ops.aten.cos.default(x)

    @staticmethod
    def tanh(x):
        return torch.ops.aten.tanh.default(x)

    @staticmethod
    def sinh(x):
        return torch.ops.aten.sinh.default(x)

    @staticmethod
    def cosh(x):
        return torch.ops.aten.cosh.default(x)

    @staticmethod
    def tan(x):
        return torch.ops.aten.tan.default(x)

    @staticmethod
    def acos(x):
        return torch.ops.aten.acos.default(x)

    @staticmethod
    def atan(x):
        return torch.ops.aten.atan.default(x)

    @staticmethod
    def asin(x):
        return torch.ops.aten.asin.default(x)

    @staticmethod
    def pow(a, b):
        return torch.ops.aten.pow.Tensor_Tensor(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        # NB: pow handles int x int fine
        return torch.ops.aten.pow.Tensor_Tensor(a, b)

    @staticmethod
    def minimum(a, b):
        return torch.ops.aten.minimum.default(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.ops.aten.maximum.default(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        return torch.ops.aten.round.default(a)

    @staticmethod
    def round_decimal(a, b) -> NoReturn:
        raise NotImplementedError(
            "round decimal doesn't support Tensor second argument atm"
        )

        # return torch.ops.aten.round.decimals(a, b)
<|endoftext|>import logging

import sympy

from torch.utils._sympy.functions import FloorDiv


log = logging.getLogger(__name__)

_MIRROR_REL_OP: dict[type[sympy.Basic], type[sympy.Rel]] = {
    sympy.Eq: sympy.Eq,
    sympy.Ne: sympy.Ne,
    sympy.Ge: sympy.Le,
    sympy.Gt: sympy.Lt,
    sympy.Le: sympy.Ge,
    sympy.Lt: sympy.Gt,
}

INEQUALITY_TYPES = (sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le)


def mirror_rel_op(type: type) -> type[sympy.Rel] | None:
    return _MIRROR_REL_OP.get(type)


# Tries to simplify 'expr', so as to leave only 'thing' in the left-hand side.
#
# Returns a tuple of:
#   1. The simplified expression
#   2. The expression on the right-hand side
#
# Returns 'None' if it can't reach a state where the only thing in the left
# hand side is 'thing'.
#
# 'trials': number of times 'try_solve' will try to isolate 'thing' to the
# left-hand side.
#
# 'floordiv_inequality': flag to enable conversion of 'FloorDiv' into
# inequalities.
def try_solve(
    expr: sympy.Basic,
    thing: sympy.Basic,
    trials: int = 5,
    floordiv_inequality: bool = True,
) -> tuple[sympy.Rel, sympy.Expr] | None:
    mirror = mirror_rel_op(type(expr))

    # Ignore unsupported expressions:
    #   - Those that are not relational operations
    #   - Those that don't have a mirror (just avoiding unexpected classes)
    if not isinstance(expr, sympy.Rel) or mirror is None:
        log.debug("expression with unsupported type: %s", type(expr))
        return None

    lhs_has_thing = expr.lhs.has(thing)
    rhs_has_thing = expr.rhs.has(thing)

    # Give up when 'thing' appears on both sides of the relational expression.
    # That is because, as is, we assume the thing we are trying to isolate is
    # only on the right-hand side.
    if lhs_has_thing and rhs_has_thing:
        log.debug("thing (%s) found in both sides of expression: %s", thing, expr)
        return None

    # Try considering both LHS and RHS by mirroring the original expression:
    # a < b ==> b > a
    expressions = []

    # Add each version of 'expr' if 'thing' is in its left-hand side.
    if lhs_has_thing:
        expressions.append(expr)
    if rhs_has_thing:
        expressions.append(mirror(expr.rhs, expr.lhs))

    for e in expressions:
        if e is None:
            continue

        if not isinstance(e, sympy.Rel):
            raise AssertionError("expected sympy.Rel")

        for _ in range(trials):
            trial = _try_isolate_lhs(e, thing, floordiv_inequality=floordiv_inequality)
            # Stop if there was no change in this trial.
            if trial == e:
                break
            e = trial  # type: ignore[assignment]

        # Return if we were able to isolate 'thing' on the left-hand side.
        if isinstance(e, sympy.Rel) and e.lhs == thing:
            log.debug("solved: %s ---> %s", expr, e)
            return e, e.rhs

    return None


def _try_isolate_lhs(
    e: sympy.Basic, thing: sympy.Basic, floordiv_inequality: bool
) -> sympy.Basic:
    op = type(e)

    if isinstance(e, sympy.Rel):
        # Move any constants in the left-hand side to the right-hand side.
        lhs_not_thing = (
            sum(a for a in e.lhs.args if not a.has(thing))
            if isinstance(e.lhs, sympy.Add)
            else 0
        )
        e = op(e.lhs - lhs_not_thing, e.rhs - lhs_not_thing)  # type: ignore[attr-defined]

    # Divide both sides by the factors that don't contain thing.
    if isinstance(e, sympy.Rel) and isinstance(e.lhs, sympy.Mul):
        lhs, rhs = e.args
        other = sympy.Mul(*[a for a in lhs.args if not a.has(thing)])

        # If we can't tell whether 'other' is negative or positive, we do nothing.
        # That is because we don't know whether we have mirror the operation or not.
        # We also divide only when we know 'rhs' is not zero.
        if not (isinstance(e, INEQUALITY_TYPES) and other.is_negative is None) and not (
            not isinstance(e, INEQUALITY_TYPES) and rhs.is_zero
        ):
            # Divide both sides by 'other'.
            lhs = lhs / other
            rhs = rhs / other

            # If 'e' is an inequality and 'other' is negative, we have to
            # mirror the expression.
            if isinstance(e, INEQUALITY_TYPES) and other.is_negative:
                op = mirror_rel_op(op)  # type: ignore[assignment]

            if op is None:
                raise AssertionError("expected op to be not None")
            e = op(lhs, rhs)

    ################################################################################
    # left-hand side is FloorDiv
    ################################################################################
    #
    # Given the expression: a // b op c
    # where 'op' is a relational operation, these rules only work if:
    #   - b > 0
    #   - c is an integer
    if (
        floordiv_inequality
        and isinstance(e, sympy.Rel)
        and isinstance(e.lhs, FloorDiv)
        and e.lhs.divisor.is_positive
        and e.rhs.is_integer
    ):
        # a // b == expr
        # => a >= (b * expr) and a < (b * (expr + 1))
        if isinstance(e, sympy.Eq):
            numerator, denominator = e.lhs.args
            return sympy.And(
                sympy.Ge(numerator, (e.rhs * denominator)),
                sympy.Lt(numerator, ((e.rhs + 1) * denominator)),
            )
        # a // b != expr
        # => a < (b * expr) or a >= (b * (expr + 1))
        if isinstance(e, sympy.Ne):
            numerator, denominator = e.lhs.args
            return sympy.Or(
                sympy.Lt(numerator, (e.rhs * denominator)),
                sympy.Ge(numerator, ((e.rhs + 1) * denominator)),
            )
        # The transformations below only work if b is positive.
        # Note: we only have this information for constants.
        # a // b > expr  => a >= b * (expr + 1)
        # a // b >= expr => a >= b * expr
        if isinstance(e, (sympy.Gt, sympy.Ge)):
            quotient = e.rhs if isinstance(e, sympy.Ge) else (e.rhs + 1)
            return sympy.Ge(e.lhs.args[0], (quotient * e.lhs.args[1]))
        # a // b < expr  => a < b * expr
        # a // b <= expr => a < b * (expr + 1)
        if isinstance(e, (sympy.Lt, sympy.Le)):
            quotient = e.rhs if isinstance(e, sympy.Lt) else (e.rhs + 1)
            return sympy.Lt(e.lhs.args[0], (quotient * e.lhs.args[1]))

    return e
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
import operator
from collections.abc import Callable
from typing import Generic, overload, SupportsFloat, TYPE_CHECKING, TypeGuard, TypeVar
from typing_extensions import TypeIs

import sympy
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom

import torch
from torch._logging import LazyString
from torch._prims_common import dtype_to_type

from .functions import (
    _keep_float,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    Mod,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_log2,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    PythonMod,
    RoundDecimal,
    RoundToInt,
    safe_pow,
    ToFloat,
    TruncToFloat,
    TruncToInt,
)
from .interp import sympy_interp
from .numbers import int_oo, IntInfinity, NegativeIntInfinity


log = logging.getLogger(__name__)

__all__ = ["ValueRanges", "bound_sympy"]

_T = TypeVar("_T", sympy.Expr, SympyBoolean)


class ValueRangeError(RuntimeError):
    pass


# Like sympify, but supports less stuff, and also ensures that direct
# sympy expressions don't have free variables
def simple_sympify(e):
    if isinstance(e, bool):
        return sympy.true if e else sympy.false
    elif isinstance(e, int):
        return sympy.Integer(e)
    elif isinstance(e, float):
        # infinity is special; we use it to bracket integers as well
        if math.isinf(e):
            return sympy.oo if e > 0 else -sympy.oo
        return sympy.Float(e)
    elif isinstance(e, sympy.Expr):
        if not getattr(e, "is_number", False):
            raise AssertionError(e)
        # NaNs can occur when doing things like 0 * sympy.oo, but it is better
        # if the operator notices this and takes care of it, because sometimes
        # the NaN is inappropriate (for example, for ints, the [-oo, oo] range
        # should go to zero when multiplied with [0, 0])
        if e == sympy.nan:
            raise AssertionError("sympy expression is NaN")
        return e
    elif isinstance(e, BooleanAtom):
        return e
    else:
        raise AssertionError(f"not simple sympy type {type(e)}: {e}")


# Sympy atomics only. Unlike <=, it also works on Sympy bools.
def sympy_generic_le(lower, upper):
    if isinstance(lower, sympy.Expr):
        if not isinstance(upper, sympy.Expr):
            raise AssertionError(
                "upper must be a sympy.Expr when lower is a sympy.Expr"
            )
        # instead of lower <= upper, we do upper >= lower since upper is mostly int_oo
        # and we have better code paths there.
        return upper >= lower
    else:
        # only negative condition is True > False
        if not isinstance(lower, SympyBoolean) or not isinstance(upper, SympyBoolean):
            raise AssertionError((lower, upper))
        return not (lower and not upper)


def vr_is_bool(vr: ValueRanges[_T]) -> TypeGuard[ValueRanges[SympyBoolean]]:
    return vr.is_bool


def vr_is_expr(vr: ValueRanges[_T]) -> TypeGuard[ValueRanges[sympy.Expr]]:
    return not vr.is_bool


def is_sympy_integer(value) -> TypeIs[sympy.Integer]:
    return isinstance(value, sympy.Integer)


ExprIn = int | float | sympy.Expr
BoolIn = bool | SympyBoolean
AllIn = ExprIn | BoolIn
ExprFn = Callable[[sympy.Expr], sympy.Expr]
ExprFn2 = Callable[[sympy.Expr, sympy.Expr], sympy.Expr]
BoolFn = Callable[[SympyBoolean], SympyBoolean]
BoolFn2 = Callable[[SympyBoolean, SympyBoolean], SympyBoolean]
AllFn = ExprFn | BoolFn
AllFn2 = ExprFn2 | BoolFn2


@dataclasses.dataclass(frozen=True)
class ValueRanges(Generic[_T]):
    if TYPE_CHECKING:
        # ruff doesn't understand circular references but mypy does
        # pyrefly: ignore [unbound-name]
        ExprVR = ValueRanges[sympy.Expr]  # noqa: F821
        # pyrefly: ignore [unbound-name]
        BoolVR = ValueRanges[SympyBoolean]  # noqa: F821
        AllVR = ExprVR | BoolVR

    # Although the type signature here suggests you can pass any
    # sympy expression, in practice the analysis here only works
    # with constant sympy expressions
    lower: _T
    upper: _T
    is_bool: bool
    is_int: bool
    is_float: bool

    def __repr__(self) -> str:
        return f"VR[{self.lower}, {self.upper}]"

    @overload
    def __init__(
        self: ValueRanges[sympy.Expr],
        lower: ExprIn,
        upper: ExprIn,
    ) -> None: ...

    @overload
    def __init__(  # type: ignore[misc]
        self: ValueRanges[SympyBoolean],
        lower: BoolIn,
        upper: BoolIn,
    ) -> None: ...

    def __init__(self, lower: AllIn, upper: AllIn) -> None:
        lower = simple_sympify(lower)
        upper = simple_sympify(upper)
        # TODO: when the bounds have free variables, this may be
        # nontrivial to actually verify
        try:
            if not sympy_generic_le(lower, upper):
                raise ValueRangeError(f"Invalid ranges [{lower}:{upper}]")
        except TypeError as e:
            raise TypeError(f"Could not compare {lower} <= {upper}") from e

        is_bool_lower = isinstance(lower, SympyBoolean)
        is_bool_upper = isinstance(upper, SympyBoolean)
        if is_bool_lower != is_bool_upper:
            raise AssertionError((lower, upper))

        # Warning: is_int/is_float is best effort.  We do pretty well in
        # Dynamo, but in Inductor these attributes are often wrong because we
        # are not very rigorous in dtype analysis.  This is also why we need
        # the flexible analysis for is_int: sometimes a sympy.oo pops in for
        # an integer bound. I would /like/ for us not to do this, but it's
        # too hard to push the invariant through right now.
        if isinstance(lower, sympy.Integer) and upper == sympy.oo:
            upper = int_oo
        if isinstance(upper, sympy.Integer) and lower == -sympy.oo:
            lower = -int_oo
        # NB: [-int_oo, -int_oo] and [int_oo, int_oo] are allowed
        integer_types = (sympy.Integer, NegativeIntInfinity, IntInfinity)
        is_int_lower = isinstance(lower, integer_types)
        is_int_upper = isinstance(upper, integer_types)

        # Because this is a frozen class
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        # Unlike bool/int in Python, we don't report bools are ints
        #
        # NB: is_bool_lower == is_bool_upper, so we only need to check one
        object.__setattr__(self, "is_bool", is_bool_lower)
        object.__setattr__(
            self,
            "is_int",
            not self.is_bool and is_int_lower and is_int_upper,
        )
        """
        # This assert is just impossible right now, too many sympy bugs
        if self.is_int:
            # NB: sympy will sometimes randomly lose the float-ness of zero,
            # so we also need to account for that in the assertion here.
            # See also https://github.com/sympy/sympy/issues/26620
            assert isinstance(lower, sympy.Integer) or lower in [-sympy.oo, 0], (
                lower,
                upper,
            )
            assert isinstance(upper, sympy.Integer) or upper in [sympy.oo, 0], (lower, upper)
        """
        # NB: [-oo, oo] always advertises as float!
        object.__setattr__(self, "is_float", not self.is_bool and not self.is_int)
        if not self.is_bool and not self.is_int and not self.is_float:
            raise AssertionError((lower, upper))

    def boolify(self) -> ValueRanges[SympyBoolean]:
        if vr_is_bool(self):
            return self
        elif self == ValueRanges.unknown():
            return ValueRanges.unknown_bool()
        else:
            raise AssertionError(f"not bool like {self}")

    def __contains__(self, x: AllIn) -> bool:
        return ValueRanges.wrap(x).issubset(self)

    def issubset(self, other):
        if other is self.unknown_int():
            return True
        return sympy_generic_le(other.lower, self.lower) and sympy_generic_le(
            self.upper, other.upper
        )

    def tighten(self, other) -> ValueRanges:
        """Given two ValueRanges, returns their intersection"""
        return self & other

    # Intersection
    @overload
    def __and__(
        self: ValueRanges[sympy.Expr],
        other: ValueRanges[sympy.Expr],
    ) -> ValueRanges[sympy.Expr]: ...

    @overload
    def __and__(  # type: ignore[misc]
        self: ValueRanges[SympyBoolean],
        other: ValueRanges[SympyBoolean],
    ) -> ValueRanges[SympyBoolean]: ...

    def __and__(self: AllVR, other: AllVR) -> AllVR:
        if other in (ValueRanges.unknown(), ValueRanges.unknown_int()):
            return self
        if self in (ValueRanges.unknown(), ValueRanges.unknown_int()):
            return other
        if self.is_bool != other.is_bool:
            raise AssertionError((self, other))
        if self.is_int != other.is_int:
            raise AssertionError((self, other))
        if self.is_float != other.is_float:
            raise AssertionError((self, other))
        if self.is_bool:
            return ValueRanges(
                sympy.Or(self.lower, other.lower), sympy.And(self.upper, other.upper)
            )
        else:
            return ValueRanges(
                sympy.Max(self.lower, other.lower), sympy.Min(self.upper, other.upper)
            )

    # Union
    @overload
    def __or__(
        self: ValueRanges[sympy.Expr],
        other: ValueRanges[sympy.Expr],
    ) -> ValueRanges[sympy.Expr]: ...

    @overload
    def __or__(  # type: ignore[misc]
        self: ValueRanges[SympyBoolean],
        other: ValueRanges[SympyBoolean],
    ) -> ValueRanges[SympyBoolean]: ...

    def __or__(self: AllVR, other: AllVR) -> AllVR:
        if ValueRanges.unknown() in (self, other):
            return ValueRanges.unknown()
        if self.is_bool != other.is_bool:
            raise AssertionError((self, other))
        if self.is_int != other.is_int:
            raise AssertionError((self, other))
        if self.is_float != other.is_float:
            raise AssertionError((self, other))
        if self.is_bool:
            return ValueRanges(
                sympy.And(self.lower, other.lower), sympy.Or(self.upper, other.upper)
            )
        else:
            return ValueRanges(
                sympy.Min(self.lower, other.lower), sympy.Max(self.upper, other.upper)
            )

    def is_singleton(self) -> bool:
        return self.lower == self.upper

    @staticmethod
    @functools.cache
    def unknown() -> ValueRanges[sympy.Expr]:
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    @functools.cache
    def unknown_int() -> ValueRanges[sympy.Expr]:
        return ValueRanges(-int_oo, int_oo)

    @staticmethod
    @functools.cache
    def unknown_bool() -> ValueRanges[SympyBoolean]:
        return ValueRanges(sympy.false, sympy.true)

    @overload
    @staticmethod
    # work around the fact that bool and int overlap
    def wrap(arg: ExprIn | ExprVR) -> ExprVR:  # type: ignore[overload-overlap]
        ...

    @overload
    @staticmethod
    def wrap(arg: BoolIn | BoolVR) -> BoolVR:  # type: ignore[misc]
        ...

    @staticmethod
    def wrap(arg: AllIn | AllVR) -> AllVR:
        if isinstance(arg, ValueRanges):
            return arg
        if isinstance(arg, float) and math.isnan(arg):
            return ValueRanges.unknown()
        # arg is either ExprIn or BoolIn, but we don't know it here
        return ValueRanges(arg, arg)  # type: ignore[arg-type]

    @staticmethod
    def increasing_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
        """Increasing: x <= y => f(x) <= f(y)."""
        x = ValueRanges.wrap(x)
        return ValueRanges(fn(x.lower), fn(x.upper))

    @overload
    @staticmethod
    def decreasing_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR: ...

    @overload
    @staticmethod
    def decreasing_map(x: BoolIn | BoolVR, fn: BoolFn) -> BoolVR:  # type: ignore[misc]
        ...

    @staticmethod
    def decreasing_map(x: AllIn | AllVR, fn: AllFn) -> AllVR:
        """Decreasing: x <= y => f(x) >= f(y)."""
        x = ValueRanges.wrap(x)
        # consistently either Expr or Bool, but we don't know it here
        return ValueRanges(fn(x.upper), fn(x.lower))  # type: ignore[arg-type]

    @staticmethod
    def monotone_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
        """It's increasing or decreasing."""
        x = ValueRanges.wrap(x)
        l = fn(x.lower)
        u = fn(x.upper)
        return ValueRanges(min(l, u), max(l, u))

    @staticmethod
    def convex_min_zero_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
        """Fn is convex and has a minimum at 0."""
        x = ValueRanges.wrap(x)
        if 0 in x:
            upper = max(fn(x.lower), fn(x.upper))
            upper = simple_sympify(upper)
            if isinstance(upper, sympy.Float) or upper == sympy.oo:
                return ValueRanges(0.0, upper)
            return ValueRanges(0, upper)
        return ValueRanges.monotone_map(x, fn)

    @overload
    @staticmethod
    def coordinatewise_increasing_map(
        x: ExprIn | ExprVR,
        y: ExprIn | ExprVR,
        fn: ExprFn2,
    ) -> ExprVR: ...

    @overload
    @staticmethod
    def coordinatewise_increasing_map(  # type: ignore[misc]
        x: BoolIn | BoolVR,
        y: BoolIn | BoolVR,
        fn: BoolFn2,
    ) -> BoolVR: ...

    @staticmethod
    def coordinatewise_increasing_map(
        x: AllIn | AllVR,
        y: AllIn | AllVR,
        fn: AllFn2,
    ) -> AllVR:
        """
        It's increasing on each coordinate.

        Mathematically:
        For every 1 <= i <= n and x_i <= y_i we have that
        f(x1, .., xn) <= f(x1, , yi, ..., xn)
        """
        x, y = ValueRanges.wrap(x), ValueRanges.wrap(y)
        return ValueRanges(
            fn(x.lower, y.lower),  # type: ignore[arg-type]
            fn(x.upper, y.upper),  # type: ignore[arg-type]
        )

    @classmethod
    def coordinatewise_monotone_map(cls, x, y, fn):
        """It's increasing or decreasing on each coordinate."""
        x, y = cls.wrap(x), cls.wrap(y)
        products = [
            fn(a, b)
            for a, b in itertools.product([x.lower, x.upper], [y.lower, y.upper])
        ]
        return ValueRanges(min(products), max(products))


class SymPyValueRangeAnalysis:
    """
    It gives bounds on a SymPy operator given bounds on its arguments
    See the function `bound_sympy` for a function that applies this logic to a full SymPy expression
    """

    @staticmethod
    def constant(value, dtype):
        if isinstance(value, ValueRanges):
            if not value.is_singleton():
                raise AssertionError("ValueRanges must be a singleton for constant()")
            value = value.lower
        # NB: value is NOT a sympy expression, it's a constant!
        is_python = isinstance(value, (int, float, bool))
        if not is_python and not isinstance(
            value, (BooleanAtom, sympy.Integer, sympy.Number)
        ):
            raise AssertionError(f"not a supported constant type: {type(value)}")

        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if isinstance(value, SupportsFloat) and math.isnan(value):
            if dtype == torch.bool:
                return ValueRanges.unknown_bool()
            elif dtype.is_floating_point:
                return ValueRanges.unknown()
            else:
                return ValueRanges.unknown_int()

        if is_python:
            type_ = dtype_to_type(dtype)
            value = type_(value)
        else:
            # We do a type check on a best-effort basis
            # We don't want to force a cast to sympy.Float if the value is Rational to avoid losing precision
            if dtype == torch.bool:
                if not isinstance(value, BooleanAtom):
                    raise AssertionError("expected BooleanAtom for bool dtype")
            elif dtype.is_floating_point:
                # pyrefly: ignore [missing-attribute]
                if value.is_finite and not value.is_real:
                    raise AssertionError(
                        "expected float-like sympy value for float dtype"
                    )
            else:
                # dtype is intXX
                if not getattr(value, "is_integer", False):
                    raise AssertionError("expected integer sympy value for int dtype")

        r = ValueRanges.wrap(value)
        return r

    @staticmethod
    def to_dtype(a, dtype, src_dtype=None):
        if dtype == torch.float64:
            # pyrefly: ignore [bad-argument-type]
            return ValueRanges.increasing_map(a, ToFloat)
        elif dtype == torch.bool:
            return ValueRanges.unknown_bool()
        elif not dtype.is_floating_point:
            return ValueRanges.unknown_int()
        return ValueRanges.unknown()

    @staticmethod
    def trunc_to_int(a, dtype):
        # pyrefly: ignore [bad-argument-type]
        return ValueRanges.increasing_map(a, TruncToInt)

    @staticmethod
    def not_(a):
        a = ValueRanges.wrap(a)
        a = a.boolify()
        if not a.is_bool:
            raise AssertionError("not_ expects a boolean ValueRanges")
        return ValueRanges.decreasing_map(a, sympy.Not)

    @staticmethod
    def or_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.Or)

    @staticmethod
    def and_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.And)

    @staticmethod
    def _bool_to_int(x):
        if x.is_singleton():
            return ValueRanges.wrap(sympy.Integer(1 if x.lower else 0))
        else:
            return ValueRanges(sympy.Integer(0), sympy.Integer(1))

    @classmethod
    def bitwise_and(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool and b.is_bool:
            return cls.and_(a, b)
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        lower = min(a.lower, b.lower)
        if lower < 0 and lower != -sympy.oo and lower != -int_oo:
            # If both lower bounds are negative, then bits start like
            # 1...10..., so the smallest possible value is 1...101...1.
            # Thus, we need to find the next smallest power of 2 (inclusive).
            try:
                lower = -(1 << int(-lower - 1).bit_length())
            except Exception:
                lower = -int_oo
        else:
            lower = 0
        return ValueRanges(lower, max(a.upper, b.upper))

    @classmethod
    def bitwise_or(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool and b.is_bool:
            return cls.or_(a, b)
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        upper = max(a.upper, b.upper)
        if upper == 0:
            upper = 0
        elif upper > 0 and upper != sympy.oo and upper != int_oo:
            # If both upper bounds are positive, then the largest
            # possible value is 01...1, so we need to find
            # next largest power of 2 (exclusive), minus 1
            try:
                upper = (1 << int(upper).bit_length()) - 1
            except Exception:
                upper = int_oo
        elif upper < 0:
            upper = -1
        return ValueRanges(min(a.lower, b.lower), upper)

    @classmethod
    def bitwise_xor(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool and b.is_bool:
            bounds = {
                a.lower ^ b.lower,
                a.lower ^ b.upper,
                a.upper ^ b.lower,
                a.upper ^ b.upper,
            }

            has_false = any(bound == sympy.false for bound in bounds)
            has_true = any(bound == sympy.true for bound in bounds)

            if has_false and has_true:
                lower, upper = sympy.false, sympy.true
            elif has_true:
                lower = upper = sympy.true
            elif has_false:
                lower = upper = sympy.false
            else:
                raise AssertionError(f"Non-boolean xor result: {bounds}")

            return ValueRanges(lower, upper)
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        if (
            a.lower == a.upper
            and b.lower == b.upper
            and is_sympy_integer(a.lower)
            and is_sympy_integer(b.lower)
        ):
            value_range = a.lower ^ b.lower
            return ValueRanges(value_range, value_range)
        return ValueRanges(-int_oo, int_oo)

    @classmethod
    def eq(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and a.lower == b.lower:
            return ValueRanges.wrap(sympy.true)
        # sympy booleans do not support ordered comparison (bool >/< raises), so
        # map them to {0, 1} before the disjoint-range test below.
        if a.is_bool:
            a = cls._bool_to_int(a)
        if b.is_bool:
            b = cls._bool_to_int(b)
        if a.lower > b.upper or b.lower > a.upper:  # ranges disjoint
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @classmethod
    def identity(cls, a):
        return ValueRanges.wrap(a)

    @classmethod
    def lt(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.and_(cls.not_(a), b)
        else:
            if a.upper < b.lower:
                return ValueRanges.wrap(sympy.true)
            elif a.lower >= b.upper:
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def gt(cls, a, b):
        return cls.lt(b, a)

    @classmethod
    def le(cls, a, b):
        return cls.not_(cls.gt(a, b))

    @classmethod
    def ge(cls, a, b):
        return cls.not_(cls.lt(a, b))

    @staticmethod
    def add(a, b):
        return ValueRanges.coordinatewise_increasing_map(
            a, b, _keep_float(operator.add)
        )

    @classmethod
    def mul(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.and_(a, b)

        def safe_mul(a, b):
            # Make unknown() * wrap(0.0) == wrap(0.0)
            if a == 0.0 or a == 0:
                return a
            elif b == 0.0 or b == 0:
                return b
            else:
                return a * b

        return ValueRanges.coordinatewise_monotone_map(a, b, _keep_float(safe_mul))

    @staticmethod
    def int_truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-int_oo in a or int_oo in a) and (-int_oo in b or int_oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(
                a,
                b,
                _keep_float(IntTrueDiv),
            )

    @staticmethod
    def truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or (
            (-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)
        ):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(
                a,
                b,
                _keep_float(FloatTrueDiv),
            )

    @staticmethod
    def floordiv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        # TODO We shall assume division is always valid probably.
        if 0 in b:
            if b.lower >= 0 and a.lower >= 0:
                return ValueRanges(0, int_oo)
            if b.upper <= 0 and a.upper <= 0:
                return ValueRanges(0, int_oo)
            if b.upper <= 0 and a.lower >= 0:
                return ValueRanges(-int_oo, 0)
            if b.lower >= 0 and a.upper <= 0:
                return ValueRanges(-int_oo, 0)
            return ValueRanges.unknown_int()
        products = []
        for x, y in itertools.product([a.lower, a.upper], [b.lower, b.upper]):
            r = FloorDiv(x, y)
            if r is sympy.nan:
                products.append((sympy.sign(x) * sympy.sign(y)) * int_oo)
            else:
                products.append(r)

        return ValueRanges(min(products), max(products))

    @classmethod
    def mod(cls, x, y):
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        # nb. We implement C semantics

        def c_mod(a, b):
            ret = abs(a) % abs(b)
            if a < 0:
                ret *= -1
            return ret

        def c_div(a, b):
            x = a / b
            return sympy.Integer(x) if x.is_finite and x not in (int_oo, -int_oo) else x

        if 0 in y:
            return ValueRanges.unknown_int()
        elif y.is_singleton():
            y_val = abs(y.lower)
            # If it wraps, we need to take the whole interval

            # The function is locally linear if they are in the same class
            if c_div(x.lower, y_val) == c_div(x.upper, y_val):
                return ValueRanges.increasing_map(x, lambda u: c_mod(u, y_val))
            if x.upper < 0:
                # Negative case
                return ValueRanges(-y_val + 1, 0)
            elif x.lower > 0:
                # Positive case
                return ValueRanges(0, y_val - 1)
            else:
                # Mixed case
                lower = max(-y_val + 1, x.lower)
                upper = min(y_val - 1, x.upper)
                return ValueRanges(lower, upper)
        else:
            # Too difficult, we bail out
            upper = cls.abs(y).upper - 1
            return ValueRanges(-upper, upper)

    @classmethod
    def python_mod(cls, x, y):
        """Python-style modulo: result has same sign as divisor.

        Assumes valid input where y is never 0.
        - When y > 0: result is in [0, y - 1]
        - When y < 0: result is in [y + 1, 0]
        """

        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        if x.lower >= 0 and y.lower >= 0:
            return SymPyValueRangeAnalysis.mod(x, y)
        lower = y.lower + 1 if y.lower < 0 else 0
        upper = y.upper - 1 if y.upper > 0 else 0
        return ValueRanges(lower, upper)

    @classmethod
    def modular_indexing(cls, a, b, c):
        return cls.mod(cls.floordiv(a, b), c)

    @classmethod
    def is_non_overlapping_and_dense_indicator(cls, *args):
        return ValueRanges.unknown_int()

    @classmethod
    def pow_by_natural(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton():
            return ValueRanges.wrap(safe_pow(a.lower, b.lower))
        # NB: Exclude zero, because zero is special
        elif a.lower >= 1:
            # We should know that b >= 0 but we may have forgotten this fact due
            # to replacements, so don't assert it, but DO clamp it to prevent
            # degenerate problems
            # pyrefly: ignore [no-matching-overload]
            return ValueRanges.coordinatewise_increasing_map(
                a, b & ValueRanges(0, int_oo), PowByNatural
            )
        elif b.is_singleton():
            if b.lower % 2 == 0:
                # x^n where n is even
                return ValueRanges.convex_min_zero_map(
                    a, lambda x: safe_pow(x, b.lower)
                )
            else:
                # x^n where n is odd
                return ValueRanges.increasing_map(a, lambda x: safe_pow(x, b.lower))
        else:
            # a is potentially negative, and we don't know if the exponent is
            # even or odd.  So just conservatively set the upper and lower
            # bound based on what the maximum absolute value could be, in both
            # directions
            max_base = max(a.upper, -a.lower)
            return ValueRanges(
                -(safe_pow(max_base, b.upper)), safe_pow(max_base, b.upper)
            )

    @classmethod
    def pow(cls, a, b):
        return ValueRanges.unknown()

        # We could implement all this, but for floating point pow, is there
        # really a point?
        """
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        # Not implemented yet. It's a bit tricky
        # If you want to implement it, compute the partial derivatives of a ** b
        # and check the ranges where the function is increasing / decreasing
        # Another non-tight way of doing this is defaulting to doing noting that for a > 0,  a ** b == exp(b * log(a))
        # If this second option is implemented, be careful about the types and possible infinities here and there.
        if not b.is_singleton():
            return ValueRanges.unknown()

        b = b.lower
        if a.is_singleton():
            a = a.lower
            r = a**b
            if not r.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)

        if b == 0:
            if not a.lower.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(1.0)

        if b < 0:
            a = cls.reciprocal(a)
            b = -b

        if a == ValueRanges.unknown():
            return ValueRanges.unknown()

        # If the base is positive, then we're good, otherwise nothing's defined
        if a.lower >= 0:
            return ValueRanges.increasing_map(a, lambda x: x**b)
        else:
            return ValueRanges.unknown()
        """

    @staticmethod
    def reciprocal(x):
        """Needed as it's used in pow, but it won't appear on a SymPy expression"""
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
        else:
            return ValueRanges.decreasing_map(x, lambda y: FloatTrueDiv(1.0, y))  # type: ignore[operator]

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_exp)

    @staticmethod
    def log(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_log)

    @staticmethod
    def log2(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_log2)

    @classmethod
    def minimum(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.and_(a, b)
        return cls.min_or_max(a, b, sympy.Min)

    @classmethod
    def maximum(cls, a, b):
        a, b = ValueRanges.wrap(a), ValueRanges.wrap(b)
        if a.is_bool != b.is_bool:
            raise AssertionError(
                "operands must both be boolean ValueRanges or both non-boolean"
            )
        if a.is_bool:
            return cls.or_(a, b)
        return cls.min_or_max(a, b, sympy.Max)

    @staticmethod
    def min_or_max(a, b, fn):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        return ValueRanges.coordinatewise_increasing_map(a, b, fn)

    @classmethod
    def floor_to_int(cls, x, dtype):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.floor)

    @classmethod
    def ceil_to_int(cls, x, dtype):
        return ValueRanges.increasing_map(
            x, sympy.functions.elementary.integers.ceiling
        )

    # I think these implementations are sound.  The hazard here is that sympy
    # will carry out the floor/ceil at too high precision and then something
    # bad will happen when we convert it to float.
    #
    # For truncation, the implementation is clearly sound, because the desired
    # target float is always exactly representable, since you're just chopping
    # off bits the mantissa.  But what about ceil/floor?
    #
    # The important constraint here is that we're not defining floor on
    # arbitrary real numbers, only representable float numbers.  So we can
    # take advantage of the fact that before we reach the first
    # unrepresentable integer in floating point space, we have the range of
    # numbers corresponding to exponent zero: all integers, with no fractional
    # amounts.  floor/ceil is an identity operation in this case.  In the
    # range below here, representable floating point numbers are spaced
    # exactly 1/2 apart, and notably, both the floor/ceil are defined floating
    # point numbers.  There is no "gap" as you step up to the next exponent.

    @classmethod
    def floor(cls, x):
        return ValueRanges.increasing_map(
            x, _keep_float(sympy.functions.elementary.integers.floor)
        )

    @classmethod
    def ceil(cls, x):
        return ValueRanges.increasing_map(
            x, _keep_float(sympy.functions.elementary.integers.ceiling)
        )

    @classmethod
    def round_decimal(cls, number, ndigits):
        if not ndigits.is_singleton():
            return ValueRanges.unknown()

        ndigits = ndigits.lower
        # We can't use functools.partial here since sympy doesn't support keyword arguments, but we have to bind
        # the second parameter.
        fn = lambda number: RoundDecimal(number, ndigits)  # type: ignore[misc, assignment]  # noqa: E731

        return ValueRanges.increasing_map(number, fn)

    @classmethod
    def round_to_int(cls, number, dtype):
        # pyrefly: ignore [bad-argument-type]
        return ValueRanges.increasing_map(number, RoundToInt)

    # It's used in some models on symints
    @staticmethod
    def sqrt(x):
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, OpaqueUnaryFn_sqrt)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        a = a.boolify()
        # We sometimes write unknown without specifying the type correctly
        # In particular, we do that when initialising the bounds for loads in bounds.py
        if b.is_bool != c.is_bool and ValueRanges.unknown() not in (b, c):
            raise AssertionError(
                "where() requires b and c to have the same boolean-ness or allow unknown()"
            )
        if b.is_bool:
            return ValueRanges(sympy.And(b.lower, c.lower), sympy.Or(b.upper, c.upper))
        else:
            return ValueRanges(sympy.Min(b.lower, c.lower), sympy.Max(b.upper, c.upper))

    # expr_cond_pair is used to represent a single (expr, condition) pair in piecewise.
    # We just return the value range of the expression and its corresponding condition as a tuple
    # and defer the analysis to piecewise
    @staticmethod
    def expr_cond_pair(a, b):
        b = b.boolify()
        return (a, b)

    # piecewise function can be used to convert a SymBool to SymInt:
    # int_expr = Piecewise((1, bool_expr), (0, True)), it evaluates to 1 when sym_bool is True and 0 otherwise.
    #
    # ranges is a sequence of (expr_range, condition_range) pairs. The range pair is constructed in expr_cond_pair.
    # The ValueRange of Piecewise is just the union of all expr ranges whose condition expr can be True.
    @staticmethod
    def piecewise(*ranges):
        init_range = None
        for expr_range, cond_range in ranges:
            if sympy.true in cond_range:
                if init_range is None:
                    init_range = expr_range
                else:
                    init_range = init_range | expr_range
        return init_range

    @staticmethod
    def cos(x):
        # TODO: We should tighten value ranges
        # If input range span is pi + 2*pi*k, then output range is (-1, 1)
        # otherwise the minimum of the value of the function on the extremes
        return ValueRanges(-1.0, 1.0)

    @staticmethod
    def cosh(x):
        return ValueRanges(0.0, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if x.lower > 0:
            return ValueRanges.increasing_map(x, OpaqueUnaryFn_cosh)
        elif x.upper < 0:
            return ValueRanges.decreasing_map(x, OpaqueUnaryFn_cosh)
        return ValueRanges(0.0, sympy.oo)
        """

    @staticmethod
    def sin(x):
        # TODO: We should tighten value ranges
        # See details on cos
        return ValueRanges(-1.0, 1.0)

    @staticmethod
    def sinh(x):
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_sinh)
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def tan(x):
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def tanh(x):
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_tanh)
        return ValueRanges(-sympy.oo, sympy.oo)

    @staticmethod
    def asin(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if -1 <= x.lower and x.upper <= 1:
            return ValueRanges.increasing_map(x, OpaqueUnaryFn_asinh)
        return ValueRanges.unknown()
        """

    @staticmethod
    def acos(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        """
        x = ValueRanges.wrap(x)
        if -1 <= x.lower and x.upper <= 1:
            return ValueRanges.decreasing_map(x, OpaqueUnaryFn_acos)
        return ValueRanges.unknown()
        """

    @staticmethod
    def atan(x):
        return ValueRanges(-sympy.oo, sympy.oo)
        # return ValueRanges.increasing_map(x, OpaqueUnaryFn_atan)

    @staticmethod
    def trunc(x):
        # pyrefly: ignore [bad-argument-type]
        return ValueRanges.increasing_map(x, TruncToFloat)


def _default_symbol_range(s: sympy.Symbol) -> ValueRanges:
    if s.is_integer:
        if s.is_positive:
            return ValueRanges(1, int_oo)
        if s.is_nonnegative:
            return ValueRanges(0, int_oo)
        return ValueRanges.unknown_int()
    return ValueRanges.unknown()


def _bound_sympy_for_rewrite_guard(
    expr: sympy.Expr, ranges: dict[sympy.Symbol, ValueRanges]
) -> ValueRanges | None:
    try:
        return sympy_interp(
            SymPyValueRangeAnalysis,
            ranges,
            expr,
            missing_handler=_default_symbol_range,
        )
    except (AttributeError, KeyError, NotImplementedError):
        return None


def _definitely_ge_value(value: sympy.Expr, lower: int) -> bool:
    try:
        return bool(value >= lower)
    except TypeError:
        return False


def _definitely_ge(
    expr: sympy.Expr, lower: int, ranges: dict[sympy.Symbol, ValueRanges]
) -> bool:
    if lower == 0 and expr.is_nonnegative:
        return True
    if lower == 1 and expr.is_positive:
        return True

    if isinstance(expr, sympy.Symbol):
        vr = ranges.get(expr)
        if isinstance(vr, ValueRanges):
            return bool(vr.lower >= lower)

    vr = _bound_sympy_for_rewrite_guard(expr, ranges)
    return vr is not None and _definitely_ge_value(vr.lower, lower)


def _mod_rewrite_is_valid(
    mod: type[sympy.Function],
    base: sympy.Expr,
    divisor: sympy.Expr,
    ranges: dict[sympy.Symbol, ValueRanges],
) -> bool:
    if not _definitely_ge(divisor, 1, ranges):
        return False
    return mod is PythonMod or _definitely_ge(base, 0, ranges)


def _rewrite_mod_subtraction(
    base: sympy.Expr,
    divisor: sympy.Expr,
    coeff: sympy.Expr,
) -> sympy.Expr:
    return coeff * FloorDiv(base, divisor) * divisor


def _terms_of_add(expr: sympy.Expr) -> dict[sympy.Expr, sympy.Expr]:
    terms: dict[sympy.Expr, sympy.Expr] = {}
    for term in sympy.Add.make_args(expr):
        coeff, factor = term.as_coeff_Mul()
        terms[factor] = terms.get(factor, sympy.S.Zero) + coeff
    return terms


def _rewrite_mod_subtractions_in_add(
    expr: sympy.Add, ranges: dict[sympy.Symbol, ValueRanges]
) -> sympy.Expr:
    terms = _terms_of_add(expr)

    replacements = []
    for factor, mod_coeff in tuple(terms.items()):
        if mod_coeff == 0 or mod_coeff.is_integer is not True:
            continue

        mod = factor.func
        if mod not in (PythonMod, Mod, sympy.Mod):
            continue

        base, divisor = factor.args
        if not _mod_rewrite_is_valid(mod, base, divisor, ranges):
            continue

        matched_terms = []
        for base_factor, base_coeff in _terms_of_add(base).items():
            if base_coeff.is_integer is not True:
                break

            term_coeff = terms.get(base_factor, sympy.S.Zero)
            needed_coeff = -mod_coeff * base_coeff
            if needed_coeff == 0:
                continue
            if term_coeff == 0 or term_coeff * needed_coeff <= 0:
                break
            if needed_coeff > 0 and term_coeff < needed_coeff:
                break
            if needed_coeff < 0 and term_coeff > needed_coeff:
                break

            matched_terms.append((base_factor, needed_coeff))
        else:
            for base_factor, needed_coeff in matched_terms:
                terms[base_factor] -= needed_coeff
            terms[factor] = sympy.S.Zero
            replacements.append(_rewrite_mod_subtraction(base, divisor, -mod_coeff))

    if not replacements:
        return expr

    new_terms = []
    for factor, coeff in terms.items():
        if coeff == 0:
            continue
        if factor == 1:
            new_terms.append(coeff)
        elif coeff == 1:
            new_terms.append(factor)
        else:
            new_terms.append(coeff * factor)

    return sympy.Add(*new_terms, *replacements)


def _rewrite_for_value_range_analysis(
    expr: sympy.Basic, ranges: dict[sympy.Symbol, ValueRanges]
) -> sympy.Basic:
    """Preserve simple dependencies that interval arithmetic would lose."""
    if not expr.args:
        return expr

    args = tuple(_rewrite_for_value_range_analysis(arg, ranges) for arg in expr.args)
    if args != expr.args:
        expr = expr.func(*args)

    if isinstance(expr, sympy.Add):
        return _rewrite_mod_subtractions_in_add(expr, ranges)

    return expr


def bound_sympy(
    expr: sympy.Expr, ranges: dict[sympy.Symbol, ValueRanges] | None = None
) -> ValueRanges:
    log.debug(
        "bound_sympy(%s)%s",
        expr,
        LazyString(
            lambda: (
                "\n"
                + "\n".join(
                    f"  {k}: {r}" for k, r in ranges.items() if k in expr.free_symbols
                )
                if ranges
                else ""
            )
        ),
    )
    if isinstance(expr, sympy.Number):
        return ValueRanges.wrap(expr)

    ranges = ranges or {}

    # If there's a tracing context, augment available constrained ranges.
    context = torch._guards.TracingContext.try_get()
    if context and context.fake_mode and context.fake_mode.shape_env:
        if ranges:
            ranges = {**context.fake_mode.shape_env.var_to_range, **ranges}
        else:
            ranges = context.fake_mode.shape_env.var_to_range

    if expr.has(PythonMod, Mod, sympy.Mod):
        expr = _rewrite_for_value_range_analysis(expr, ranges)

    return sympy_interp(
        SymPyValueRangeAnalysis, ranges, expr, missing_handler=_default_symbol_range
    )
<|endoftext|>"""
This file contains canonical definitions for our symbol naming conventions,
across torch.fx.experimental.symbolic_shapes and torch._inductor.  The
intention is:

1. To make it easily greppable where all the sites we use a prefix are
2. Make it possible to easily tell if we can introduce a new prefix without
   introducing a conflict

You can occasionally test if prefixes have been hardcoded by renaming prefixes
in this file and seeing what breaks.
"""

from collections.abc import Iterable
from enum import auto, Enum

import sympy


class SymT(Enum):
    SIZE = auto()
    FLOAT = auto()
    UNBACKED_INT = auto()
    UNBACKED_FLOAT = auto()
    # Inductor: The intermediates in inner_fn tmp0, one generated per ops call.
    # If one of these shows up in an indexing expression, that means an
    # indirect load is happening.
    TMP = auto()
    # Inductor: Placeholder variable that is later replaced with TMP
    INDIRECT = auto()
    # Inductor: Some size expressions are replaced with a precomputed size ps0
    # which is computed host side, and then directly reused in the kernel, so
    # we don't repeatedly recompute it on device.
    PRECOMPUTED_SIZE = auto()
    # Inductor: An indexing variable i0 in loops IR which ranges over non-reduced
    # dim in the loop
    INDEX = auto()
    # Inductor: A reduction indexing (r0, r1) variables in loops IR which ranges over
    # reduced dim(s) in the loop
    R0_INDEX = auto()
    R1_INDEX = auto()
    # Inductor: In templated kernels torch._inductor.kernel, we have a hook to
    # store the final output and append epilogue fusions.  To do this, we must
    # know what the indexes the outputs range over.  NB: These will also
    # advertise as INDEX, this is... probably OK?
    TEMPLATE_INDEX = auto()
    # Inductor: iteration domain for blockIdx.x/blockIdx.y
    XBLOCK = auto()
    YBLOCK = auto()
    ZBLOCK = auto()
    # Inductor: this is used solely for dynamic_reshape_indexer
    VIEW = auto()
    # Alternate (non-modular) indexing used in halide kernels
    HALIDE = auto()


# Invariant: there must not be a prefix which is a prefix of another string,
# as this introduces ambiguity
prefix_str = {
    SymT.SIZE: "s",  # integer
    SymT.UNBACKED_INT: "u",  # integer
    # Prefix z here is chosen to avoid false aliasing in symbol_is_type test
    # DO NOT add a "z" type.  You also need to avoid conflicts on these
    # prefixes but this is somewhat easier to manage
    SymT.FLOAT: "zf",
    SymT.UNBACKED_FLOAT: "zuf",
    SymT.TMP: "tmp",
    SymT.PRECOMPUTED_SIZE: "ps",
    SymT.INDEX: "i",
    SymT.R0_INDEX: "r0_",
    SymT.R1_INDEX: "r1_",
    SymT.TEMPLATE_INDEX: "idx",
    SymT.XBLOCK: "x",
    SymT.YBLOCK: "y",
    SymT.ZBLOCK: "z",
    SymT.INDIRECT: "indirect",  # false aliasing?
    SymT.VIEW: "view",
    SymT.HALIDE: "h",
}


def make_symbol(prefix: SymT, idx: int, **kwargs: bool | None) -> sympy.Symbol:
    # TODO: maybe put the assumptions here directly
    return sympy.Symbol(f"{prefix_str[prefix]}{idx}", **kwargs)


# This type is a little wider than it should be, because free_symbols says
# that it contains Basic, rather than Symbol
def symbol_is_type(sym: sympy.Basic, prefix: SymT | Iterable[SymT]) -> bool:
    if not isinstance(sym, sympy.Symbol):
        raise AssertionError("expected sympy.Symbol")
    name_str = sym.name.lower()  # Match capitalized names like XBLOCK, RBLOCK
    if isinstance(prefix, SymT):
        return name_str.startswith(prefix_str[prefix])
    else:
        return name_str.startswith(tuple(prefix_str[p] for p in prefix))


def free_symbol_is_type(e: sympy.Expr, prefix: SymT | Iterable[SymT]) -> bool:
    return any(symbol_is_type(v, prefix) for v in e.free_symbols)
<|endoftext|>import sys
from typing import Literal

import sympy
from sympy.printing.precedence import PRECEDENCE, precedence
from sympy.printing.str import StrPrinter


INDEX_TYPE = "int64_t"
INDEX_TYPE_MAX = (1 << 63) - 1
INDEX_TYPE_MIN = -1 << 63


# This printer contains rules that are supposed to be generic for both C/C++ and
# Python
class ExprPrinter(StrPrinter):
    # override this so that _print_FloorDiv is used
    printmethod = "_torch_sympystr"

    def _print_Mul(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, "*", precedence(expr))

    def _print_Not(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        return f"not ({self._print(expr.args[0])})"

    def _print_Add(self, expr: sympy.Expr, order: str | None = None) -> str:
        return self.stringify(expr.args, " + ", precedence(expr))

    def _print_Relational(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, f" {expr.rel_op} ", precedence(expr))

    def _print_BitwiseFn_bitwise_and(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " & ", PRECEDENCE["BitwiseAnd"])

    def _print_BitwiseFn_bitwise_or(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    def _print_BitwiseFn_bitwise_xor(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])

    # NB: this is OK to put here, because Mod is only defined for positive
    # numbers, and so across C/Python its behavior is consistent
    def _print_Mod(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)

    def _print_FloatTrueDiv(self, expr: sympy.Expr) -> str:
        s = self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)
        return f"({s})"

    def _print_CleanDiv(self, expr: sympy.Expr) -> str:
        return self._print_FloorDiv(expr)

    def _print_Identity(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        return self._print(expr.args[0])

    def _print_Float(self, expr: sympy.Expr) -> str:
        if expr._prec == 53:
            # IEEE-754 double precision have 53 bits. SymPy prints them with
            # 15 digits, but we need 17 for round-trip correctness
            return str(sympy.Float(expr, dps=17))
        else:
            # We don't use other precisions in pytorch
            return str(expr)

    # This must be implemented because sympy will collect x * x into Pow(x, 2), without
    # any explicit intervention.  We print it just like x * x, notably, we
    # never generate sympy.Pow with floats.
    #
    # NB: this pow by natural, you should never have used builtin sympy.pow
    # for FloatPow, and a symbolic exponent should be PowByNatural.  These
    # means exp is guaranteed to be integer.
    def _print_Pow(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        if exp != int(exp):
            raise AssertionError(exp)
        exp = int(exp)
        if exp < 0:
            raise AssertionError(f"exponent must be non-negative, got {exp}")
        if exp > 0:
            return self.stringify([base] * exp, "*", PRECEDENCE["Mul"])
        return "1"

    # Explicit NotImplemented functions are to prevent default sympy printing
    # behavior, which will just barf out ToFloat(...) to your IR.  The error
    # message is better here because it tells you which printer class it needs
    # to go in.

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_ToFloat not implemented for {type(self)}")

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_Infinity not implemented for {type(self)}")

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_NegativeInfinity not implemented for {type(self)}"
        )

    def _print_NaN(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_NaN not implemented for {type(self)}")

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_FloorDiv not implemented for {type(self)}")

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_PythonMod not implemented for {type(self)}")

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_IntTrueDiv not implemented for {type(self)}")

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_FloatPow not implemented for {type(self)}")

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_TruncToInt not implemented for {type(self)}")

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_RoundToInt not implemented for {type(self)}")

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_RoundDecimal not implemented for {type(self)}"
        )

    # NB: Some float operations are INTENTIONALLY not implemented for
    # printers.  You can implement them as a quick unblock, but it is better
    # to ask yourself why we haven't done this computation in the Tensor
    # universe instead

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_TruncToFloat not implemented for {type(self)}"
        )


class PythonPrinter(ExprPrinter):
    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        # NB: We use sym_float here because the printer is used for cache
        # serialization, and cache guards get evaluated with SymInt to
        # propagate guards to the parent ShapeEnv.  However, this comes at a
        # runtime cost for guards involving float.  If this is unacceptable
        # overhead, what you want to do is have two separate printers for
        # SymInt, one for when the inputs are guaranteed to be int, and
        # another for when they could be SymInt.
        #
        # NB: sym_min/sym_max also have this problem, but I chose not to fix
        # those.
        #
        # See https://github.com/pytorch/pytorch/issues/142507 for more
        # context.
        # pyrefly: ignore [missing-attribute]
        return f"torch.sym_float({self._print(expr.args[0])})"

    def _print_And(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " and ", precedence(expr))

    def _print_Or(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " or ", precedence(expr))

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args
        )
        if div != "1":
            x = f"({x} // {div})"
        return f"({x} % {mod})"

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        return "math.inf"

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        return "-math.inf"

    def _print_NaN(self, expr: sympy.Expr) -> str:
        return "math.nan"

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = (self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args)
        return f"{x} // {div}"

    # WARNING: this is dangerous for Triton, when lhs, rhs > 2**53, Python
    # does a special algorithm
    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)

    def _helper_sqrt(self, expr: sympy.Expr) -> str:
        # NB: We use torch._sym_sqrt here instead of math.sqrt because the
        # guard expression may be evaluated with SymInt/SymFloat inputs (e.g.
        # during cache hit re-evaluation in evaluate_guards_expression).
        # math.sqrt on a SymFloat triggers evaluate_expr which forces
        # concretization/specialization of the symbol, creating spurious
        # guards that didn't exist in the original program.
        # torch._sym_sqrt properly propagates through the symbolic system
        # without forcing specialization.
        # See https://github.com/pytorch/pytorch/issues/152435
        # pyrefly: ignore [missing-attribute]
        return f"torch._sym_sqrt({self._print(expr)})"

    def _print_OpaqueUnaryFn_sqrt(self, expr: sympy.Expr) -> str:
        return self._helper_sqrt(expr.args[0])

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " ** ", PRECEDENCE["Pow"])

    # TODO: Not sure this works with Triton, even when base/exp are integral
    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " ** ", PRECEDENCE["Pow"])

    def _print_floor(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("floor expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.floor({self._print(expr.args[0])})"

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("FloorToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.floor({self._print(expr.args[0])})"

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("TruncToInt expects exactly one argument")
        # This also could have been int(), they'll do the same thing for float
        # pyrefly: ignore [missing-attribute]
        return f"math.trunc({self._print(expr.args[0])})"

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ceiling expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("CeilToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("Abs expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"abs({self._print(expr.args[0])})"

    # NB: It's expected that we've made explicit any promotion in the sympy
    # expression, so it doesn't matter that Python max/min doesn't perform
    # promotion
    def _print_Max(self, expr: sympy.Expr) -> str:
        if len(expr.args) < 2:
            raise AssertionError("Max expects at least two arguments")
        # pyrefly: ignore [missing-attribute]
        return f"max({', '.join(map(self._print, expr.args))})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        if len(expr.args) < 2:
            raise AssertionError("Min expects at least two arguments")
        # pyrefly: ignore [missing-attribute]
        return f"min({', '.join(map(self._print, expr.args))})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("cos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("cosh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("acos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("sin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("sinh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("asin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("tan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("tanh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("atan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("log2 expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.log2({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("RoundToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"round({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise AssertionError("RoundDecimal expects exactly two arguments")
        number, ndigits = expr.args
        if not isinstance(ndigits, sympy.Integer):
            raise TypeError("ndigits must be an instance of sympy.Integer")
        # pyrefly: ignore [missing-attribute]
        return f"round({self._print(number)}, {ndigits})"

    def _print_Piecewise(self, expr: sympy.Expr) -> str:
        # Convert Piecewise(expr_cond_pairs) to nested ternary expressions
        # Piecewise((e1, c1), (e2, c2), ..., (eN, cN))
        # becomes: e1 if c1 else (e2 if c2 else (... else eN))
        result: str | None = None
        for expr_i, cond_i in reversed(expr.args):
            # pyrefly: ignore [missing-attribute]
            expr_str = self._print(expr_i)
            if cond_i == True:  # noqa: E712
                # This is the default case
                result = expr_str
            else:
                # pyrefly: ignore [missing-attribute]
                cond_str = self._print(cond_i)
                if result is None:
                    result = expr_str
                else:
                    result = f"({expr_str} if {cond_str} else {result})"
        return result if result else "0"


class CppPrinter(ExprPrinter):
    def _print_Integer(self, expr: sympy.Expr) -> str:
        suffix = "LL" if sys.platform in ["darwin", "win32"] else "L"
        i = int(expr)
        if i > INDEX_TYPE_MAX or i < INDEX_TYPE_MIN:
            raise OverflowError(f"{i} too big to convert to {INDEX_TYPE}")
        elif i == INDEX_TYPE_MIN:
            if i != (-1) << 63:
                raise AssertionError("unexpected minimum index type value")
            # Writing -9223372036854775808L makes the value overflow
            # as it is parsed as -(9223372036854775808L) by the C/C++ compiler
            return f"(-1{suffix} << 63)"
        return f"{i}{suffix}"

    def _print_Where(self, expr: sympy.Expr) -> str:
        c, p, q = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args
        )
        return f"{c} ? {p} : {q}"

    def _print_Or(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " || ", precedence(expr))

    def _print_Piecewise(self, expr: sympy.Expr) -> str:
        # Convert Piecewise(expr_cond_pairs) to nested ternary operators
        # Piecewise((e1, c1), (e2, c2), ..., (eN, cN))
        # becomes: c1 ? e1 : (c2 ? e2 : (... : eN))
        result: str | None = None
        for expr_i, cond_i in reversed(expr.args):
            expr_str = self.parenthesize(expr_i, PRECEDENCE["Atom"] - 0.5)
            if cond_i == True:  # noqa: E712
                # This is the default case
                result = expr_str
            else:
                cond_str = self.parenthesize(cond_i, PRECEDENCE["Atom"] - 0.5)
                if result is None:
                    result = expr_str
                else:
                    result = f"{cond_str} ? {expr_str} : {result}"
        return f"({result})" if result else "0"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            if expr.is_integer:
                x = f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
            else:
                x = f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"
        mod = self.doprint(mod)
        return f"(static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod}))"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
        return f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"

    def _print_floor(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("floor expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("FloorToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("TruncToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::trunc({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})"

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("TruncToFloat expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::trunc({self._print(expr.args[0])})"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<double>({self._print(expr.args[0])})"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        return f"c10::div_mod({x}, {div})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**53
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<double>({self._print(lhs)}) / static_cast<double>({self._print(rhs)})"

    # TODO: PowByNatural: we need to implement our own int-int pow.  Do NOT
    # use std::pow, that operates on floats
    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        # Implement the special-case of 2**x for now
        base, exp = expr.args
        if base == 2:
            # pyrefly: ignore [missing-attribute]
            return f"(1 << ({self._print(exp)}))"
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        # pyrefly: ignore [missing-attribute]
        return f"std::pow({self._print(base)}, {self._print(exp)})"

    def _print_Pow(self, expr: sympy.Expr) -> str:
        # Uses float constants to perform FP div
        base, exp = expr.args

        if exp == 0.5 or exp == -0.5:
            # pyrefly: ignore [missing-attribute]
            base = self._print(base)
            return f"std::sqrt({base})" if exp == 0.5 else f"1.0/std::sqrt({base})"
        if exp.is_integer:
            exp = int(exp)
            if exp > 0:
                r = self.stringify([base] * exp, "*", PRECEDENCE["Mul"])
            elif exp < -1:
                r = (
                    "1.0/("
                    + self.stringify([base] * abs(exp), "*", PRECEDENCE["Mul"])
                    + ")"
                )
            elif exp == -1:
                # pyrefly: ignore [missing-attribute]
                r = "1.0/" + self._print(base)
            else:  # exp == 0
                r = "1.0"

            return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r
        else:
            # TODO: float vs double
            return f"std::pow({base}, {float(exp)})"

    def _print_Rational(self, expr: sympy.Expr) -> str:
        # Uses float constants to perform FP div
        if expr.q == 1:
            r = f"{expr.p}"
        else:
            r = f"{expr.p}.0/{expr.q}.0"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ceiling expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("CeilToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_min_max(self, expr: sympy.Expr, name: Literal["min", "max"]) -> str:
        # pyrefly: ignore [missing-attribute]
        args = [self._print(a) for a in expr.args]
        if not expr.is_integer:
            args = [f"static_cast<double>({arg})" for arg in args]
            if len(args) == 2:
                return f"std::{name}({args[0]}, {args[1]})"
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::{name}<double>({il})"

        if len(args) == 2:
            return f"std::{name}(static_cast<{INDEX_TYPE}>({args[0]}), static_cast<{INDEX_TYPE}>({args[1]}))"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::{name}<{INDEX_TYPE}>({il})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "min")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "max")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("Abs expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("cos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("cosh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("acos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("sin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("sinh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("asin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("tan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("tanh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("atan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sqrt(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        return f"std::sqrt({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        return f"std::log2({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("RoundToInt expects exactly one argument")
        # TODO: dispatch to llrint depending on index type
        # pyrefly: ignore [missing-attribute]
        return f"std::lrint({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise AssertionError("RoundDecimal expects exactly two arguments")
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            if ndigits >= 0:
                raise AssertionError("ndigits must be negative for integer inputs")
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"static_cast<double>(std::nearbyint(1e{ndigits} * {number_str}) * 1e{-ndigits})"

    def _print_BooleanTrue(self, expr: sympy.Expr) -> str:
        return "true"

    def _print_BooleanFalse(self, expr: sympy.Expr) -> str:
        return "false"

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        return "std::numeric_limits<double>::infinity()"

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        return f"-{self._print_Infinity(expr)}"

    def _print_NaN(self, expr: sympy.Expr) -> str:
        return "std::numeric_limits<double>::quiet_NaN()"
<|endoftext|>from typing import Any

import mpmath.libmp as mlib  # type: ignore[import-untyped]
import sympy
from sympy import Expr
from sympy.core.decorators import _sympifyit
from sympy.core.expr import AtomicExpr
from sympy.core.numbers import Number
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S, Singleton


# pyrefly: ignore [invalid-inheritance]
class IntInfinity(Number, metaclass=Singleton):
    r"""Positive integer infinite quantity.

    Integer infinity is a value in an extended integers which
    is greater than all other integers.  We distinguish it from
    sympy's existing notion of infinity in that it reports that
    it is_integer.

    Infinity is a singleton, and can be accessed by ``S.IntInfinity``,
    or can be imported as ``int_oo``.
    """

    # NB: We can't actually mark this as infinite, as integer and infinite are
    # inconsistent assumptions in sympy.  We also report that we are complex,
    # different from sympy.oo

    is_integer = True
    is_commutative = True
    is_number = True
    is_extended_real = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    # Ensure we get dispatched to before plain numbers
    _op_priority = 100.0

    __slots__: tuple[str, ...] = ()

    def __new__(cls) -> "IntInfinity":
        return AtomicExpr.__new__(cls)

    def _sympystr(self, printer: Any) -> str:
        return "int_oo"

    def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> sympy.Basic | None:
        if self == old:
            return new
        return None

    # We could do these, not sure about it
    """
    def _eval_evalf(self, prec=None):
        return Float('inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    @_sympifyit("other", NotImplemented)
    def __add__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NegativeInfinity):
                return other
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    __radd__ = __add__

    @_sympifyit("other", NotImplemented)
    def __sub__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.NegativeInfinity
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other: sympy.Expr) -> sympy.Expr:
        return (-self).__add__(other)

    @_sympifyit("other", NotImplemented)
    def __mul__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.NegativeIntInfinity
        return Number.__mul__(self, other)

    __rmul__ = __mul__

    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other: sympy.Expr) -> sympy.Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return S.Infinity  # truediv produces float
            return S.NegativeInfinity  # truediv produces float
        return Number.__truediv__(self, other)

    def __abs__(self) -> sympy.Expr:
        return S.IntInfinity

    def __neg__(self) -> sympy.Expr:
        return S.NegativeIntInfinity

    def _eval_power(self, expt: sympy.Expr) -> sympy.Expr | None:
        if expt.is_extended_positive:
            return S.IntInfinity
        if expt.is_extended_negative:
            return S.Zero
        if expt is S.NaN:
            return S.NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re

            expt_real = re(expt)
            if expt_real.is_positive:
                return S.ComplexInfinity
            if expt_real.is_negative:
                return S.Zero
            if expt_real.is_zero:
                return S.NaN

            return self ** expt.evalf()
        return None

    def _as_mpf_val(self, prec: int) -> Any:
        return mlib.finf

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        return other is S.IntInfinity

    def __ne__(self, other: object) -> bool:
        return other is not S.IntInfinity

    def __gt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.true

    def __ge__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.true

    def __lt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        if other is S.Infinity:
