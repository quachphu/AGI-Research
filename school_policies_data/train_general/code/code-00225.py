    # 3. Run the script with the `send` option in the second shell:
    # (shell2) ./test_socket.py sock_tmp 1017 send

    def _get_data(self):
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError(
                    f"DataLoader timed out after {self._timeout} seconds"
                )
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError("Pin memory thread exited unexpectedly")
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info.get(self._rcvd_idx, None)
                if info:
                    worker_id = info[0]
                    if (
                        len(info) == 2 or self._workers_status[worker_id]
                    ):  # has data or is still active
                        break
                    del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                worker_id, data = self._task_info.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                return self._process_data(data, worker_id)

            if self._shutdown or self._tasks_outstanding <= 0:
                raise AssertionError(
                    "Invalid iterator state: shutdown or no outstanding tasks when fetching next data"
                )
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                if not self._in_order:
                    # don't store it for later, process now
                    # delete from self._task_info immediately
                    # this keeps the object size manageable
                    worker_id = self._task_info.pop(idx)[0]
                    return self._process_data(data, worker_id)
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                worker_id = self._task_info.pop(idx)[0]
                self._rcvd_idx += 1
                return self._process_data(data, worker_id)

    def _try_put_index(self) -> None:
        max_tasks = self._prefetch_factor * self._num_workers
        if self._tasks_outstanding >= max_tasks:
            raise AssertionError(
                "Number of outstanding tasks exceeded maximum allowed tasks"
            )

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                if self._in_order:
                    break
                elif self._workers_num_tasks[worker_queue_idx] < max_tasks // sum(
                    self._workers_status
                ):
                    # when self._in_order is False, distribute work to a worker if it has capacity
                    # _workers_status is updated only in this thread, so the sum is guaranteed > 0
                    break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))  # type: ignore[possibly-undefined]
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._workers_num_tasks[worker_queue_idx] += 1
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data, worker_idx):
        self._workers_num_tasks[worker_idx] -= 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False) -> None:
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        if (
            not self._workers_status[worker_id]
            and not self._persistent_workers
            and not shutdown
        ):
            raise AssertionError(
                "Worker status inconsistent when marking worker as unavailable"
            )

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joining is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        if self._workers_done_event.is_set() != shutdown:
            raise AssertionError(
                "_workers_done_event state does not match shutdown flag"
            )

    def _shutdown_workers(self) -> None:
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if (
            _utils is None
            # pyrefly: ignore [unnecessary-comparison]
            or _utils.python_exit_status is True
            # pyrefly: ignore [unnecessary-comparison]
            or _utils.python_exit_status is None
        ):
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, "_pin_memory_thread"):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()
                        w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)

    @staticmethod
    def _clean_up_persistent_workers_atexit() -> None:
        # This callback is registered after `_utils._set_python_exit_flag`,
        # so it runs while `_shutdown_workers()` can still use multiprocessing.
        with _persistent_workers_atexit_lock:
            iterators = tuple(_persistent_workers_atexit)
        for iterator in iterators:
            iterator._shutdown_workers()

    def _register_persistent_workers_atexit(self) -> None:
        import atexit

        global _persistent_workers_atexit_registered
        with _persistent_workers_atexit_lock:
            _persistent_workers_atexit.add(self)
            if _persistent_workers_atexit_registered:
                return
            atexit.register(
                _MultiProcessingDataLoaderIter._clean_up_persistent_workers_atexit
            )
            _persistent_workers_atexit_registered = True

    def __del__(self) -> None:
        self._shutdown_workers()
<|endoftext|># mypy: allow-untyped-defs
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, get_type_hints

from torch.utils.data.datapipes._typing import _DataPipeMeta
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


######################################################
# Functional API
######################################################
class functional_datapipe:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        """
        Define a functional datapipe.

        Args:
            enable_df_api_tracing - if set, any returned DataPipe would accept
            DataFrames API in tracing mode.
        """
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        if issubclass(cls, IterDataPipe):
            if isinstance(cls, type):  # type: ignore[arg-type]
                if not isinstance(cls, _DataPipeMeta):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            # with non_deterministic decorator
            else:
                if not isinstance(cls, non_deterministic) and not (
                    hasattr(cls, "__self__")
                    and isinstance(cls.__self__, non_deterministic)
                ):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            IterDataPipe.register_datapipe_as_function(
                self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing
            )
        elif issubclass(cls, MapDataPipe):
            MapDataPipe.register_datapipe_as_function(self.name, cls)

        return cls


######################################################
# Determinism
######################################################
_determinism: bool = False


class guaranteed_datapipes_determinism:
    prev: bool

    def __init__(self) -> None:
        global _determinism
        self.prev = _determinism
        _determinism = True

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _determinism
        _determinism = self.prev


class non_deterministic:
    cls: type[IterDataPipe] | None = None
    # TODO: Lambda for picking
    deterministic_fn: Callable[..., bool]

    def __init__(self, arg: type[IterDataPipe] | Callable[..., bool]) -> None:
        # 1. Decorator doesn't have any argument
        if isinstance(arg, type):  # type: ignore[arg-type]
            if not issubclass(arg, IterDataPipe):  # type: ignore[arg-type]
                raise TypeError(
                    "Only `IterDataPipe` can be decorated with `non_deterministic`"
                    f", but {arg.__name__} is found"
                )
            self.cls = arg  # type: ignore[assignment]
        # 2. Decorator has an argument of a function
        #    This class should behave differently given different inputs. Use this
        #    function to verify the determinism for each instance.
        #    When the function returns True, the instance is non-deterministic. Otherwise,
        #    the instance is a deterministic DataPipe.
        elif isinstance(arg, Callable):  # type:ignore[arg-type]
            self.deterministic_fn = arg
        else:
            raise TypeError(f"{arg} can not be decorated by non_deterministic")

    def __call__(self, *args, **kwargs):
        global _determinism
        #  Decorate IterDataPipe
        if self.cls is not None:
            if _determinism:
                raise TypeError(
                    f"{self.cls.__name__} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. "
                    "You can turn off determinism for this DataPipe if that is acceptable "
                    "for your application"
                )
            return self.cls(*args, **kwargs)  # type: ignore[call-arg]

        # Decorate with a functional argument
        if not (
            isinstance(args[0], type) and issubclass(args[0], IterDataPipe)  # type: ignore[arg-type]
        ):
            raise TypeError(
                f"Only `IterDataPipe` can be decorated, but {args[0].__name__} is found"
            )
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        res = self.deterministic_fn(*args, **kwargs)
        if not isinstance(res, bool):
            raise TypeError(
                "deterministic_fn of `non_deterministic` decorator is required "
                f"to return a boolean value, but {type(res)} is found"
            )
        global _determinism
        if _determinism and res:
            raise TypeError(
                f"{self.cls.__name__} is non-deterministic with the inputs, but you set "  # type: ignore[union-attr]
                "'guaranteed_datapipes_determinism'. You can turn off determinism "
                "for this DataPipe if that is acceptable for your application"
            )
        return self.cls(*args, **kwargs)  # type: ignore[call-arg, misc]


######################################################
# Type validation
######################################################
# Validate each argument of DataPipe with hint as a subtype of the hint.
def argument_validation(f):
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for argument_name, value in bound.arguments.items():
            if argument_name in hints and isinstance(
                hints[argument_name], _DataPipeMeta
            ):
                hint = hints[argument_name]
                if not isinstance(value, IterDataPipe):
                    raise TypeError(
                        f"Expected argument '{argument_name}' as an IterDataPipe, but found {type(value)}"
                    )
                if not value.type.issubtype(hint.type):
                    raise TypeError(
                        f"Expected type of argument '{argument_name}' as a subtype of "
                        f"hint {hint.type}, but found {value.type}"
                    )

        return f(*args, **kwargs)

    return wrapper


# Default value is True
_runtime_validation_enabled: bool = True


class runtime_validation_disabled:
    prev: bool

    def __init__(self) -> None:
        global _runtime_validation_enabled
        self.prev = _runtime_validation_enabled
        _runtime_validation_enabled = False

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _runtime_validation_enabled
        _runtime_validation_enabled = self.prev


# Runtime checking
# Validate output data is subtype of return hint
def runtime_validation(f):
    # TODO:
    # Can be extended to validate '__getitem__' and nonblocking
    if f.__name__ != "__iter__":
        raise TypeError(
            f"Can not decorate function {f.__name__} with 'runtime_validation'"
        )

    @wraps(f)
    def wrapper(self):
        global _runtime_validation_enabled
        if not _runtime_validation_enabled:
            yield from f(self)
        else:
            it = f(self)
            for d in it:
                if not self.type.issubtype_of_instance(d):
                    raise RuntimeError(
                        f"Expected an instance as subtype of {self.type}, but found {d}({type(d)})"
                    )
                yield d

    return wrapper
<|endoftext|># mypy: allow-untyped-defs
# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import functools
import numbers
import sys

# Please check [Note: TypeMeta and TypeAlias]
# In case of metaclass conflict due to ABCMeta or _ProtocolMeta
# For Python 3.9, only Protocol in typing uses metaclass
from abc import ABCMeta
from collections.abc import Iterator

# TODO: Use TypeAlias when Python 3.6 is deprecated
from typing import (
    _eval_type,  # pyrefly: ignore [missing-module-attribute]
    _GenericAlias,  # pyrefly: ignore [missing-module-attribute]
    _tp_cache,  # pyrefly: ignore [missing-module-attribute]
    _type_check,  # pyrefly: ignore [missing-module-attribute]
    _type_repr,
    Any,
    ForwardRef,
    Generic,
    get_args,
    get_type_hints,
    TypeVar,
    Union,
)

from torch.utils.data.datapipes._hook_iterator import _SnapshotState, hook_iterator


class GenericMeta(ABCMeta):  # type: ignore[no-redef]
    pass


class Integer(numbers.Integral):
    pass


class Boolean(numbers.Integral):
    pass


# Python 'type' object is not subscriptable
# Tuple[int, List, dict] -> valid
# tuple[int, list, dict] -> invalid
# Map Python 'type' to abstract base class
TYPE2ABC = {
    bool: Boolean,
    int: Integer,
    float: numbers.Real,
    complex: numbers.Complex,
    dict: dict,
    list: list,
    set: set,
    tuple: tuple,
    None: type(None),
}


def issubtype(left, right, recursive=True):
    r"""
    Check if the left-side type is a subtype of the right-side type.

    If any of type is a composite type like `Union` and `TypeVar` with
    bounds, it would be expanded into a list of types and check all
    of left-side types are subtypes of either one from right-side types.
    """
    left = TYPE2ABC.get(left, left)
    right = TYPE2ABC.get(right, right)

    if right is Any or left == right:
        return True

    if isinstance(right, _GenericAlias):
        if getattr(right, "__origin__", None) is Generic:
            return True

    if right is type(None):
        return False

    # Right-side type
    constraints = _decompose_type(right)

    if len(constraints) == 0 or Any in constraints:
        return True

    if left is Any:
        return False

    # Left-side type
    variants = _decompose_type(left)

    # all() will return True for empty variants
    if len(variants) == 0:
        return False

    return all(
        _issubtype_with_constraints(variant, constraints, recursive)
        for variant in variants
    )


def _decompose_type(t, to_list=True):
    if isinstance(t, TypeVar):
        if t.__bound__ is not None:
            ts = [t.__bound__]
        else:
            # For T_co, __constraints__ is ()
            ts = list(t.__constraints__)
    elif hasattr(t, "__origin__") and t.__origin__ == Union:
        ts = get_args(t)
    else:
        if not to_list:
            return None
        ts = [t]
    # Ignored: Generator has incompatible item type "object"; expected "Type[Any]"
    ts = [TYPE2ABC.get(_t, _t) for _t in ts]  # type: ignore[misc]
    return ts


def _issubtype_with_constraints(variant, constraints, recursive=True):
    r"""
    Check if the variant is a subtype of either one from constraints.

    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
    if variant in constraints:
        return True

    # [Note: Subtype for Union and TypeVar]
    # Python typing is able to flatten Union[Union[...]] or Union[TypeVar].
    # But it couldn't flatten the following scenarios:
    #   - Union[int, TypeVar[Union[...]]]
    #   - TypeVar[TypeVar[...]]
    # So, variant and each constraint may be a TypeVar or a Union.
    # In these cases, all of inner types from the variant are required to be
    # extracted and verified as a subtype of any constraint. And, all of
    # inner types from any constraint being a TypeVar or a Union are
    # also required to be extracted and verified if the variant belongs to
    # any of them.

    # Variant
    vs = _decompose_type(variant, to_list=False)

    # Variant is TypeVar or Union
    if vs is not None:
        return all(_issubtype_with_constraints(v, constraints, recursive) for v in vs)

    # Variant is not TypeVar or Union
    if hasattr(variant, "__origin__") and variant.__origin__ is not None:
        v_origin = variant.__origin__
        v_args = get_args(variant) or None
    else:
        v_origin = variant
        v_args = None

    # Constraints
    for constraint in constraints:
        cs = _decompose_type(constraint, to_list=False)

        # Constraint is TypeVar or Union
        if cs is not None:
            if _issubtype_with_constraints(variant, cs, recursive):
                return True
        # Constraint is not TypeVar or Union
        else:
            # __origin__ can be None for plain list, tuple, ... in Python 3.6
            if hasattr(constraint, "__origin__") and constraint.__origin__ is not None:
                c_origin = constraint.__origin__
                if v_origin == c_origin:
                    if not recursive:
                        return True
                    c_args = get_args(constraint) or None
                    if c_args is None or len(c_args) == 0:
                        return True
                    if (
                        v_args is not None
                        and len(v_args) == len(c_args)
                        and all(
                            issubtype(v_arg, c_arg)
                            for v_arg, c_arg in zip(v_args, c_args, strict=True)
                        )
                    ):
                        return True
            # Tuple[int] -> Tuple
            else:
                if v_origin == constraint:
                    return True

    return False


def issubinstance(data, data_type):
    if not issubtype(type(data), data_type, recursive=False):
        return False

    dt_args = get_args(data_type) or None
    if isinstance(data, tuple):
        if dt_args is None or len(dt_args) == 0:
            return True
        if len(dt_args) != len(data):
            return False
        return all(issubinstance(d, t) for d, t in zip(data, dt_args, strict=True))
    elif isinstance(data, (list, set)):
        if dt_args is None or len(dt_args) == 0:
            return True
        t = dt_args[0]
        return all(issubinstance(d, t) for d in data)
    elif isinstance(data, dict):
        if dt_args is None or len(dt_args) == 0:
            return True
        kt, vt = dt_args
        return all(
            issubinstance(k, kt) and issubinstance(v, vt) for k, v in data.items()
        )

    return True


# [Note: TypeMeta and TypeAlias]
# In order to keep compatibility for Python 3.6, use Meta for the typing.
# TODO: When PyTorch drops the support for Python 3.6, it can be converted
# into the Alias system and using `__class_getitem__` for DataPipe. The
# typing system will gain benefit of performance and resolving metaclass
# conflicts as elaborated in https://www.python.org/dev/peps/pep-0560/


class _DataPipeType:
    r"""Save type annotation in `param`."""

    def __init__(self, param) -> None:
        self.param = param

    def __repr__(self) -> str:
        return _type_repr(self.param)

    def __eq__(self, other):
        if isinstance(other, _DataPipeType):
            return self.param == other.param
        return NotImplemented

    def __hash__(self):
        return hash(self.param)

    def issubtype(self, other):
        if isinstance(other.param, _GenericAlias):
            if getattr(other.param, "__origin__", None) is Generic:
                return True
        if isinstance(other, _DataPipeType):
            return issubtype(self.param, other.param)
        if isinstance(other, type):
            return issubtype(self.param, other)
        raise TypeError(f"Expected '_DataPipeType' or 'type', but found {type(other)}")

    def issubtype_of_instance(self, other):
        return issubinstance(other, self.param)


# Default type for DataPipe without annotation
_T_co = TypeVar("_T_co", covariant=True)
# pyrefly: ignore [invalid-annotation]
_DEFAULT_TYPE = _DataPipeType(Generic[_T_co])


class _DataPipeMeta(GenericMeta):
    r"""
    Metaclass for `DataPipe`.

    Add `type` attribute and `__init_subclass__` based on the type, and validate the return hint of `__iter__`.

    Note that there is subclass `_IterDataPipeMeta` specifically for `IterDataPipe`.
    """

    type: _DataPipeType

    def __new__(cls, name, bases, namespace, **kwargs):
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        # TODO: the statements below are not reachable by design as there is a bug and typing is low priority for now.
        cls.__origin__ = None
        if "type" in namespace:
            return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        namespace["__type_class__"] = False
        #  For plain derived class without annotation
        for base in bases:
            if isinstance(base, _DataPipeMeta):
                return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        namespace.update(
            {"type": _DEFAULT_TYPE, "__init_subclass__": _dp_init_subclass}
        )
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

    def __init__(self, name, bases, namespace, **kwargs) -> None:
        super().__init__(name, bases, namespace, **kwargs)  # type: ignore[call-overload]

    # TODO: Fix isinstance bug
    @_tp_cache
    def _getitem_(self, params):
        if params is None:
            raise TypeError(f"{self.__name__}[t]: t can not be None")
        if isinstance(params, str):
            params = ForwardRef(params)
        if not isinstance(params, tuple):
            params = (params,)

        msg = f"{self.__name__}[t]: t must be a type"
        params = tuple(_type_check(p, msg) for p in params)

        if isinstance(self.type.param, _GenericAlias):
            orig = getattr(self.type.param, "__origin__", None)
            if isinstance(orig, type) and orig is not Generic:
                p = self.type.param[params]  # type: ignore[index]
                t = _DataPipeType(p)
                l = len(str(self.type)) + 2
                name = self.__name__[:-l]
                name = name + "[" + str(t) + "]"
                bases = (self,) + self.__bases__
                return self.__class__(
                    name,
                    bases,
                    {
                        "__init_subclass__": _dp_init_subclass,
                        "type": t,
                        "__type_class__": True,
                    },
                )

        if len(params) > 1:
            raise TypeError(
                f"Too many parameters for {self} actual {len(params)}, expected 1"
            )

        t = _DataPipeType(params[0])

        if not t.issubtype(self.type):
            raise TypeError(
                f"Can not subclass a DataPipe[{t}] from DataPipe[{self.type}]"
            )

        # Types are equal, fast path for inheritance
        if self.type == t:
            return self

        name = self.__name__ + "[" + str(t) + "]"
        bases = (self,) + self.__bases__

        return self.__class__(
            name,
            bases,
            {"__init_subclass__": _dp_init_subclass, "__type_class__": True, "type": t},
        )

    # TODO: Fix isinstance bug
    def _eq_(self, other):
        if not isinstance(other, _DataPipeMeta):
            return NotImplemented
        if self.__origin__ is None or other.__origin__ is None:  # type: ignore[has-type]
            return self is other
        return (
            self.__origin__ == other.__origin__  # type: ignore[has-type]
            and self.type == other.type
        )

    # TODO: Fix isinstance bug
    def _hash_(self):
        return hash((self.__name__, self.type))


class _IterDataPipeMeta(_DataPipeMeta):
    r"""
    Metaclass for `IterDataPipe` and inherits from `_DataPipeMeta`.

    Add various functions for behaviors specific to `IterDataPipe`.
    """

    def __new__(cls, name, bases, namespace, **kwargs):
        if "reset" in namespace:
            reset_func = namespace["reset"]

            @functools.wraps(reset_func)
            def conditional_reset(*args, **kwargs) -> None:
                r"""
                Only execute DataPipe's `reset()` method if `_SnapshotState` is `Iterating` or `NotStarted`.

                This allows recently restored DataPipe to preserve its restored state during the initial `__iter__` call.
                """
                datapipe = args[0]
                if datapipe._snapshot_state in (
                    _SnapshotState.Iterating,
                    _SnapshotState.NotStarted,
                ):
                    # Reset `NotStarted` is necessary because the `source_datapipe` of a DataPipe might have
                    # already begun iterating.
                    datapipe._number_of_samples_yielded = 0
                    datapipe._fast_forward_iterator = None
                    reset_func(*args, **kwargs)
                datapipe._snapshot_state = _SnapshotState.Iterating

            namespace["reset"] = conditional_reset

        if "__iter__" in namespace:
            hook_iterator(namespace)
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]


def _dp_init_subclass(sub_cls, *args, **kwargs) -> None:
    # Add function for datapipe instance to reinforce the type
    sub_cls.reinforce_type = reinforce_type

    # TODO:
    # - add global switch for type checking at compile-time

    # Ignore internal type class
    if getattr(sub_cls, "__type_class__", False):
        return

    # Check if the string type is valid
    if isinstance(sub_cls.type.param, ForwardRef):
        base_globals = sys.modules[sub_cls.__module__].__dict__
        try:
            param = _eval_type(sub_cls.type.param, base_globals, locals())
            sub_cls.type.param = param
        except TypeError as e:
            raise TypeError(
                f"{sub_cls.type.param.__forward_arg__} is not supported by Python typing"
            ) from e

    if "__iter__" in sub_cls.__dict__:
        iter_fn = sub_cls.__dict__["__iter__"]
        hints = get_type_hints(iter_fn)
        if "return" in hints:
            return_hint = hints["return"]
            # Plain Return Hint for Python 3.6
            if return_hint == Iterator:
                return
            if not (
                hasattr(return_hint, "__origin__")
                and (
                    return_hint.__origin__ == Iterator
                    or return_hint.__origin__ == collections.abc.Iterator
                )
            ):
                raise TypeError(
                    "Expected 'Iterator' as the return annotation for `__iter__` of {}"
                    ", but found {}".format(
                        sub_cls.__name__, _type_repr(hints["return"])
                    )
                )
            data_type = get_args(return_hint)[0]
            if not issubtype(data_type, sub_cls.type.param):
                raise TypeError(
                    f"Expected return type of '__iter__' as a subtype of {sub_cls.type},"
                    f" but found {_type_repr(data_type)} for {sub_cls.__name__}"
                )


def reinforce_type(self, expected_type):
    r"""
    Reinforce the type for DataPipe instance.

    And the 'expected_type' is required to be a subtype of the original type
    hint to restrict the type requirement of DataPipe instance.
    """
    if isinstance(expected_type, tuple):
        expected_type = tuple[expected_type]  # type: ignore[valid-type]
    _type_check(expected_type, msg="'expected_type' must be a type")

    if not issubtype(expected_type, self.type.param):
        raise TypeError(
            f"Expected 'expected_type' as subtype of {self.type}, but found {_type_repr(expected_type)}"
        )

    self.type = _DataPipeType(expected_type)
    return self
<|endoftext|># mypy: allow-untyped-defs
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing_extensions import deprecated


try:
    from torchgen.api.python import format_function_signature
    from torchgen.utils import FileManager
except ImportError:
    import sys

    REPO_ROOT = Path(__file__).absolute().parents[4]
    sys.path.insert(0, str(REPO_ROOT))

    from torchgen.api.python import format_function_signature
    from torchgen.utils import FileManager

    if len(sys.path) > 0 and sys.path[0] == str(REPO_ROOT):
        del sys.path[0]


__all__: list[str] = []  # not intended to expose any symbols


def __dir__() -> list[str]:
    return []  # appease public API test


@deprecated(
    "`torch.utils.data.datapipes.gen_pyi.materialize_lines` is deprecated and will be removed in the future.",
    category=FutureWarning,
)
def materialize_lines(lines: list[str], indentation: int) -> str:
    output = ""
    new_line_with_indent = "\n" + " " * indentation
    for i, line in enumerate(lines):
        if i != 0:
            output += new_line_with_indent
        output += line.replace("\n", new_line_with_indent)
    return output


@deprecated(
    "`torch.utils.data.datapipes.gen_pyi.gen_from_template` is deprecated and will be removed in the future.",
    category=FutureWarning,
)
def gen_from_template(
    dir: str,
    template_name: str,
    output_name: str,
    replacements: list[tuple[str, Any, int]],
) -> None:
    template_path = os.path.join(dir, template_name)
    output_path = os.path.join(dir, output_name)

    with open(template_path, encoding="utf-8") as f:
        content = f.read()
    for placeholder, lines, indentation in replacements:
        with open(output_path, "w", encoding="utf-8") as f:
            content = content.replace(
                placeholder, materialize_lines(lines, indentation)
            )
            f.write(content)


def find_file_paths(dir_paths: list[str], files_to_exclude: set[str]) -> set[str]:
    """
    When given a path to a directory, returns the paths to the relevant files within it.

    This function does NOT recursive traverse to subdirectories.
    """
    paths: set[str] = set()
    for dir_path in dir_paths:
        all_files = os.listdir(dir_path)
        python_files = {fname for fname in all_files if ".py" == fname[-3:]}
        filter_files = {
            fname for fname in python_files if fname not in files_to_exclude
        }
        paths.update({os.path.join(dir_path, fname) for fname in filter_files})
    return paths


def extract_method_name(line: str) -> str:
    """Extract method name from decorator in the form of "@functional_datapipe({method_name})"."""
    if '("' in line:
        start_token, end_token = '("', '")'
    elif "('" in line:
        start_token, end_token = "('", "')"
    else:
        raise RuntimeError(
            f"Unable to find appropriate method name within line:\n{line}"
        )
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def extract_class_name(line: str) -> str:
    """Extract class name from class definition in the form of "class {CLASS_NAME}({Type}):"."""
    start_token = "class "
    end_token = "("
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def parse_datapipe_file(
    file_path: str,
) -> tuple[dict[str, list[str]], dict[str, str], set[str], dict[str, list[str]]]:
    """Given a path to file, parses the file and returns a dictionary of method names to function signatures."""
    method_to_signature, method_to_class_name, special_output_type = {}, {}, set()
    doc_string_dict = defaultdict(list)
    with open(file_path, encoding="utf-8") as f:
        open_paren_count = 0
        method_name, class_name, signature = "", "", ""
        skip = False
        for line in f:
            if line.count('"""') % 2 == 1:
                skip = not skip
            if skip or '"""' in line:  # Saving docstrings
                doc_string_dict[method_name].append(line)
                continue
            if "@functional_datapipe" in line:
                method_name = extract_method_name(line)
                doc_string_dict[method_name] = []
                continue
            if method_name and "class " in line:
                class_name = extract_class_name(line)
                continue
            if method_name and ("def __init__(" in line or "def __new__(" in line):
                if "def __new__(" in line:
                    special_output_type.add(method_name)
                open_paren_count += 1
                start = line.find("(") + len("(")
                line = line[start:]
            if open_paren_count > 0:
                open_paren_count += line.count("(")
                open_paren_count -= line.count(")")
                if open_paren_count == 0:
                    end = line.rfind(")")
                    signature += line[:end]
                    method_to_signature[method_name] = process_signature(signature)
                    method_to_class_name[method_name] = class_name
                    method_name, class_name, signature = "", "", ""
                elif open_paren_count < 0:
                    raise RuntimeError(
                        "open parenthesis count < 0. This shouldn't be possible."
                    )
                else:
                    signature += line.strip()
    return (
        method_to_signature,
        method_to_class_name,
        special_output_type,
        doc_string_dict,
    )


def parse_datapipe_files(
    file_paths: set[str],
) -> tuple[dict[str, list[str]], dict[str, str], set[str], dict[str, list[str]]]:
    methods_and_signatures = {}
    methods_and_class_names = {}
    methods_with_special_output_types = set()
    methods_and_doc_strings = {}
    for path in file_paths:
        (
            method_to_signature,
            method_to_class_name,
            methods_needing_special_output_types,
            doc_string_dict,
        ) = parse_datapipe_file(path)
        methods_and_signatures.update(method_to_signature)
        methods_and_class_names.update(method_to_class_name)
        methods_with_special_output_types.update(methods_needing_special_output_types)
        methods_and_doc_strings.update(doc_string_dict)
    return (
        methods_and_signatures,
        methods_and_class_names,
        methods_with_special_output_types,
        methods_and_doc_strings,
    )


def split_outside_bracket(line: str, delimiter: str = ",") -> list[str]:
    """Given a line of text, split it on comma unless the comma is within a bracket '[]'."""
    bracket_count = 0
    curr_token = ""
    res = []
    for char in line:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
        elif char == delimiter and bracket_count == 0:
            res.append(curr_token)
            curr_token = ""
            continue
        curr_token += char
    res.append(curr_token)
    return res


def process_signature(line: str) -> list[str]:
    """
    Clean up a given raw function signature.

    This includes removing the self-referential datapipe argument, default
    arguments of input functions, newlines, and spaces.
    """
    tokens: list[str] = split_outside_bracket(line)
    for i, token in enumerate(tokens):
        tokens[i] = token.strip(" ")
        if token == "cls":
            tokens[i] = "self"
        elif i > 0 and ("self" == tokens[i - 1]) and (tokens[i][0] != "*"):
            # Remove the datapipe after 'self' or 'cls' unless it has '*'
            tokens[i] = ""
        elif "Callable =" in token:  # Remove default argument if it is a function
            head = token.rpartition("=")[0]
            tokens[i] = head.strip(" ") + " = ..."
    tokens = [t for t in tokens if t != ""]
    return tokens


def get_method_definitions(
    file_path: str | list[str],
    files_to_exclude: set[str],
    deprecated_files: set[str],
    default_output_type: str,
    method_to_special_output_type: dict[str, str],
    root: str = "",
) -> list[str]:
    """
    #.pyi generation for functional DataPipes Process.

    # 1. Find files that we want to process (exclude the ones who don't)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    """
    if root == "":
        root = str(Path(__file__).parent.resolve())
    file_path = [file_path] if isinstance(file_path, str) else file_path
    file_path = [os.path.join(root, path) for path in file_path]
    file_paths = find_file_paths(
        file_path, files_to_exclude=files_to_exclude.union(deprecated_files)
    )
    (
        methods_and_signatures,
        methods_and_class_names,
        methods_w_special_output_types,
        methods_and_doc_strings,
    ) = parse_datapipe_files(file_paths)

    for fn_name in method_to_special_output_type:
        if fn_name not in methods_w_special_output_types:
            methods_w_special_output_types.add(fn_name)

    method_definitions = []
    for method_name, arguments in methods_and_signatures.items():
        class_name = methods_and_class_names[method_name]
        if method_name in methods_w_special_output_types:
            output_type = method_to_special_output_type[method_name]
        else:
            output_type = default_output_type
        doc_string = "".join(methods_and_doc_strings[method_name])
        if doc_string == "":
            doc_string = " ..."
        else:
            doc_string = "\n" + doc_string
        definition = format_function_signature(method_name, arguments, output_type)
        method_definitions.append(
            f"# Functional form of '{class_name}'\n"
            + definition.removesuffix("...").rstrip()  # remove "..."
            + doc_string,
        )
    method_definitions.sort(
        key=lambda s: s.split("\n")[1]
    )  # sorting based on method_name

    return method_definitions


# Defined outside of main() so they can be imported by TorchData
iterDP_file_path: str = "iter"
iterDP_files_to_exclude: set[str] = {"__init__.py", "utils.py"}
iterDP_deprecated_files: set[str] = set()
iterDP_method_to_special_output_type: dict[str, str] = {
    "demux": "list[IterDataPipe]",
    "fork": "list[IterDataPipe]",
}

mapDP_file_path: str = "map"
mapDP_files_to_exclude: set[str] = {"__init__.py", "utils.py"}
mapDP_deprecated_files: set[str] = set()
mapDP_method_to_special_output_type: dict[str, str] = {"shuffle": "IterDataPipe"}


def main() -> None:
    """
    # Inject file into template datapipe.pyi.in.

    TODO: The current implementation of this script only generates interfaces for built-in methods. To generate
          interface for user-defined DataPipes, consider changing `IterDataPipe.register_datapipe_as_function`.
    """
    iter_method_definitions = get_method_definitions(
        iterDP_file_path,
        iterDP_files_to_exclude,
        iterDP_deprecated_files,
        "IterDataPipe",
        iterDP_method_to_special_output_type,
    )

    map_method_definitions = get_method_definitions(
        mapDP_file_path,
        mapDP_files_to_exclude,
        mapDP_deprecated_files,
        "MapDataPipe",
        mapDP_method_to_special_output_type,
    )

    path = Path(__file__).absolute().parent
    fm = FileManager(install_dir=path, template_dir=path, dry_run=False)
    fm.write_with_template(
        "datapipe.pyi",
        "datapipe.pyi.in",
        lambda: {
            "IterDataPipeMethods": iter_method_definitions,
            "MapDataPipeMethods": map_method_definitions,
        },
    )


if __name__ == "__main__":
    main()
<|endoftext|>import functools
import pickle
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from torch.utils._import_utils import import_dill
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes.utils.common import (
    _deprecation_warning,
    _iter_deprecated_functional_names,
    _map_deprecated_functional_names,
)
from torch.utils.data.dataset import Dataset, IterableDataset


dill = import_dill()
HAS_DILL = dill is not None

__all__ = [
    "DataChunk",
    "DFIterDataPipe",
    "IterDataPipe",
    "MapDataPipe",
]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

UNTRACABLE_DATAFRAME_PIPES = [
    "batch",  # As it returns DataChunks
    "groupby",  # As it returns DataChunks
    "_dataframes_as_tuples",  # As it unpacks DF
    "trace_as_dataframe",  # As it used to mark DF for tracing
]


class DataChunk(list[_T]):
    def __init__(self, items: Iterable[_T]) -> None:
        items = list(items)
        super().__init__(items)
        self.items = items

    def as_str(self, indent: str = "") -> str:
        return indent + "[" + ", ".join(str(i) for i in iter(self)) + "]"

    def __iter__(self) -> Iterator[_T]:
        yield from super().__iter__()

    def raw_iterator(self) -> Iterator[_T]:
        yield from self.items


class IterDataPipe(IterableDataset[_T_co], metaclass=_IterDataPipeMeta):
    r"""
    Iterable-style DataPipe.

    All DataPipes that represent an iterable of data samples should subclass this.
    This style of DataPipes is particularly useful when data come from a stream, or
    when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
    elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
    method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
    override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
    and various state variables within the custom ``IterDataPipe``.

    Note:
        Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
        and the creation a second iterator will invalidate the first one. This constraint is necessary because
        some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
        The code example below presents details on how this constraint looks in practice.
        If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
    You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
    operations in succession.

    .. _GitHub IterDataPipe Single Iterator Issue:
        https://github.com/pytorch/data/issues/45

    Note:
        When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
        item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
        iterator. When :attr:`num_workers > 0`, each worker process will have a
        different copy of the DataPipe object, so it is often desired to configure
        each copy independently to avoid having duplicate data returned from the
        workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
        process, returns information about the worker. It can be used in either the
        dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
        :attr:`worker_init_fn` option to modify each copy's behavior.

    Examples:
        General Usage:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
            >>> map_dp_2 = dp.map(
            ...     lambda x: x + 1
            ... )  # Using functional form (recommended)
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
            >>> list(filter_dp)
            [2, 4, 6, 8, 10]
        Single Iterator Constraint Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> source_dp = IterableWrapper(range(10))
            >>> it1 = iter(source_dp)
            >>> list(it1)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> it1 = iter(source_dp)
            >>> it2 = iter(
            ...     source_dp
            ... )  # The creation of a new iterator invalidates `it1`
            >>> next(it2)
            0
            >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`
    """

    functions: dict[str, Callable] = {}
    reduce_ex_hook: Callable | None = None
    getstate_hook: Callable | None = None
    str_hook: Callable | None = None
    repr_hook: Callable | None = None
    _valid_iterator_id: int | None = None
    _number_of_samples_yielded: int = 0
    _snapshot_state: _SnapshotState = _SnapshotState.NotStarted
    _fast_forward_iterator: Iterator | None = None

    def __iter__(self) -> Iterator[_T_co]:
        # pyrefly: ignore [bad-return]
        return self

    def __getattr__(self, attribute_name):
        if attribute_name in IterDataPipe.functions:
            if attribute_name in _iter_deprecated_functional_names:
                kwargs = _iter_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            f = IterDataPipe.functions[attribute_name]
            function = functools.partial(f, self)
            functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
            return function
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attribute_name}"
            )

    @classmethod
    def register_function(cls, function_name, function) -> None:
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(
        cls, function_name, cls_to_register, enable_df_api_tracing=False
    ) -> None:
        if function_name in cls.functions:
            raise Exception(  # noqa: TRY002
                f"Unable to add DataPipe function name {function_name} as it is already taken"
            )

        def class_function(cls, enable_df_api_tracing, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            if isinstance(result_pipe, IterDataPipe):
                if enable_df_api_tracing or isinstance(source_dp, DFIterDataPipe):
                    if function_name not in UNTRACABLE_DATAFRAME_PIPES:
                        result_pipe = result_pipe.trace_as_dataframe()

            return result_pipe

        function = functools.partial(
            class_function, cls_to_register, enable_df_api_tracing
        )
        functools.update_wrapper(
            wrapper=function, wrapped=cls_to_register, assigned=("__doc__",)
        )
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None:
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing getstate_hook")
        IterDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None:
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing reduce_ex_hook")
        IterDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self) -> str:
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperIterDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __str__(self) -> str:
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperIterDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __dir__(self):
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return list(super().__dir__()) + list(self.functions.keys())

    def reset(self) -> None:
        r"""
        Reset the `IterDataPipe` to the initial state.

        By default, no-op. For subclasses of `IterDataPipe`, depending on their functionalities,
        they may want to override this method with implementations that
        may clear the buffers and reset pointers of the DataPipe.
        The `reset` method is always called when `__iter__` is called as part of `hook_iterator`.
        """


class DFIterDataPipe(IterDataPipe):
    def _is_dfpipe(self) -> bool:
        return True


class MapDataPipe(Dataset[_T_co], metaclass=_DataPipeMeta):
    r"""
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """

    functions: dict[str, Callable] = {}
    reduce_ex_hook: Callable | None = None
    getstate_hook: Callable | None = None
    str_hook: Callable | None = None
    repr_hook: Callable | None = None

    def __getattr__(self, attribute_name):
        if attribute_name in MapDataPipe.functions:
            if attribute_name in _map_deprecated_functional_names:
                kwargs = _map_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            f = MapDataPipe.functions[attribute_name]
            function = functools.partial(f, self)
            functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
            return function
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attribute_name}"
            )

    @classmethod
    def register_function(cls, function_name, function) -> None:
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register) -> None:
        if function_name in cls.functions:
            raise Exception(  # noqa: TRY002
                f"Unable to add DataPipe function name {function_name} as it is already taken"
            )

        def class_function(cls, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        functools.update_wrapper(
            wrapper=function, wrapped=cls_to_register, assigned=("__doc__",)
        )
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        state = self.__dict__
        if MapDataPipe.getstate_hook is not None:
            return MapDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        if MapDataPipe.reduce_ex_hook is not None:
            try:
                return MapDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None:
        if MapDataPipe.getstate_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing getstate_hook")
        MapDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None:
        if MapDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing reduce_ex_hook")
        MapDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self) -> str:
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __str__(self) -> str:
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __dir__(self):
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return list(super().__dir__()) + list(self.functions.keys())


class _DataPipeSerializationWrapper:
    def __init__(self, datapipe) -> None:
        self._datapipe = datapipe

    def __getstate__(self):
        use_dill = False
        try:
            value = pickle.dumps(self._datapipe)
        except Exception:
            if HAS_DILL:
                # pyrefly: ignore [missing-attribute]
                value = dill.dumps(self._datapipe)
                use_dill = True
            else:
                raise
        return (value, use_dill)

    def __setstate__(self, state):
        value, use_dill = state
        if use_dill:
            # pyrefly: ignore [missing-attribute]
            self._datapipe = dill.loads(value)
        else:
            self._datapipe = pickle.loads(value)

    def __len__(self) -> int:
        try:
            return len(self._datapipe)
        except Exception as e:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length"
            ) from e


class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[_T_co]) -> None:
        super().__init__(datapipe)
        # pyrefly: ignore [invalid-type-var]
        self._datapipe_iter: Iterator[_T_co] | None = None

    def __iter__(self) -> "_IterDataPipeSerializationWrapper":
        self._datapipe_iter = iter(self._datapipe)
        return self

    def __next__(self) -> _T_co:  # type: ignore[type-var]
        if self._datapipe_iter is None:
            raise AssertionError(
                "Iterator has not been initialized; call __iter__() before __next__()"
            )
        return next(self._datapipe_iter)


class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):
    def __getitem__(self, idx):
        return self._datapipe[idx]
<|endoftext|># mypy: allow-untyped-defs
import functools
import inspect
from enum import Enum

import torch


class _SnapshotState(Enum):
    r"""
    These are the snapshotting-related states that IterDataPipes can be in.

    `NotStarted` - allows you to restore a snapshot and create an iterator with reset
    `Restored` - cannot restore again, allows you to create an iterator without resetting the DataPipe
    `Iterating` - can restore, will reset if you create a new iterator
    """

    NotStarted = 0
    Restored = 1
    Iterating = 2


def _simplify_obj_name(obj) -> str:
    """Simplify the display strings of objects for the purpose of rendering within DataPipe error messages."""
    if inspect.isfunction(obj):
        return obj.__name__
    else:
        return repr(obj)


def _strip_datapipe_from_name(name: str) -> str:
    return name.replace("IterDataPipe", "").replace("MapDataPipe", "")


def _generate_input_args_string(obj):
    """Generate a string for the input arguments of an object."""
    signature = inspect.signature(obj.__class__)
    input_param_names = set(signature.parameters.keys())
    result = []
    for name, value in inspect.getmembers(obj):
        if name in input_param_names:
            result.append((name, _simplify_obj_name(value)))
    return ", ".join([f"{name}={value}" for name, value in result])


def _generate_iterdatapipe_msg(datapipe, simplify_dp_name: bool = False):
    output_string = (
        f"{datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})"
    )
    if simplify_dp_name:
        output_string = _strip_datapipe_from_name(output_string)
    return output_string


def _gen_invalid_iterdatapipe_msg(datapipe) -> str:
    return (
        "This iterator has been invalidated because another iterator has been created "
        f"from the same IterDataPipe: {_generate_iterdatapipe_msg(datapipe)}\n"
        "This may be caused by multiple references to the same IterDataPipe. We recommend "
        "using `.fork()` if that is necessary."
    )


_feedback_msg = (
    "\nFor feedback regarding this single iterator per IterDataPipe constraint, feel free "
    "to comment on this issue: https://github.com/pytorch/data/issues/45."
)


def _check_iterator_valid(datapipe, iterator_id, next_method_exists=False) -> None:
    r"""
    Given an instance of a DataPipe and an iterator ID, check if the IDs match, and if not, raises an exception.

    In the case of ChildDataPipe, the ID gets compared to the one stored in `main_datapipe` as well.
    """
    if next_method_exists:
        # This is the case where `IterDataPipe` has both `__iter__` and `__next__`.
        # The `_valid_iterator_id` should either be never set (`None`), or set by at most one
        # iterator (`0`). Otherwise, it means there are multiple iterators.
        if datapipe._valid_iterator_id is not None and datapipe._valid_iterator_id != 0:
            extra_msg = "\nNote that this exception is raised inside your IterDataPipe's `__next__` method"
            raise RuntimeError(
                _gen_invalid_iterdatapipe_msg(datapipe) + extra_msg + _feedback_msg
            )
    elif (
        hasattr(datapipe, "_is_child_datapipe") and datapipe._is_child_datapipe is True
    ):
        if hasattr(datapipe, "_check_valid_iterator_id"):
            if not datapipe._check_valid_iterator_id(iterator_id):
                raise RuntimeError(
                    "This iterator has been invalidated, because a new iterator has been created "
                    f"from one of the ChildDataPipes of "
                    f"{_generate_iterdatapipe_msg(datapipe.main_datapipe)}."
                    + _feedback_msg
                )
        else:
            raise RuntimeError(
                "ChildDataPipe must have method `_check_valid_iterator_id`."
            )
    elif datapipe._valid_iterator_id != iterator_id:
        raise RuntimeError(_gen_invalid_iterdatapipe_msg(datapipe) + _feedback_msg)


def _set_datapipe_valid_iterator_id(datapipe):
    """Given a DataPipe, updates its valid iterator ID and reset the DataPipe."""
    if hasattr(datapipe, "_is_child_datapipe") and datapipe._is_child_datapipe is True:
        if hasattr(datapipe, "_set_main_datapipe_valid_iterator_id"):
            datapipe._set_main_datapipe_valid_iterator_id()  # reset() is called within this method when appropriate
        else:
            raise RuntimeError(
                "ChildDataPipe must have method `_set_main_datapipe_valid_iterator_id`."
            )
    else:
        if datapipe._valid_iterator_id is None:
            datapipe._valid_iterator_id = 0
        else:
            datapipe._valid_iterator_id += 1
        datapipe.reset()
    return datapipe._valid_iterator_id


def hook_iterator(namespace) -> None:
    r"""
    Define a hook that is applied to all `__iter__` of metaclass `_DataPipeMeta`.

    This is done for the purpose of profiling and checking if an iterator is still valid.
    """

    def profiler_record_fn_context(datapipe):
        if not hasattr(datapipe, "_profile_name"):
            datapipe._profile_name = _generate_iterdatapipe_msg(
                datapipe, simplify_dp_name=True
            )
        return torch.autograd.profiler.record_function(datapipe._profile_name)

    class IteratorDecorator:
        r"""
        Wrap the iterator and modifying its `__next__` method.

        This decorator is applied to DataPipes of which `__iter__` method is NOT a generator function.
        Those `__iter__` method commonly returns `self` but not necessarily.
        """

        def __init__(self, iterator, datapipe, iterator_id, has_next_method) -> None:
            self.iterator = iterator
            self.datapipe = datapipe
            self.iterator_id = iterator_id
            self._profiler_enabled = torch.autograd._profiler_enabled()
            # Check if `__iter__` returns `self` and `DataPipe` has `__next__`
            self.self_and_has_next_method = (
                self.iterator is self.datapipe and has_next_method
            )

        def __iter__(self):
            return self

        def _get_next(self):
            """Return next with logic related to iterator validity, profiler, and incrementation of samples yielded."""
            _check_iterator_valid(self.datapipe, self.iterator_id)
            result = next(self.iterator)
            if not self.self_and_has_next_method:
                self.datapipe._number_of_samples_yielded += 1
            return result

        def __next__(self):
            # TODO: Add try-except to in-place reduce traceback from the Exception
            # See: https://github.com/pytorch/data/issues/284
            if self._profiler_enabled:
                with profiler_record_fn_context(self.datapipe):
                    return self._get_next()
            else:  # Decided against using `contextlib.nullcontext` for performance reasons
                return self._get_next()

        def __getattr__(self, name):
            return getattr(self.iterator, name)

    func = namespace["__iter__"]

    # ``__iter__`` of IterDataPipe is a generator function
    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def wrap_generator(*args, **kwargs):
            gen = func(*args, **kwargs)
            datapipe = args[0]
            if datapipe._fast_forward_iterator:
                it = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                datapipe._snapshot_state = _SnapshotState.Iterating
                while True:
                    try:
                        yield next(it)
                    except StopIteration:
                        return
            iterator_id = _set_datapipe_valid_iterator_id(
                datapipe
            )  # This ID is tied to each created iterator
            _profiler_enabled = torch.autograd._profiler_enabled()
            try:
                if _profiler_enabled:
                    with profiler_record_fn_context(datapipe):
                        response = gen.send(None)
                else:
                    response = gen.send(None)

                while True:
                    datapipe._number_of_samples_yielded += 1
                    request = yield response
                    # Pass through here every time `__next__` is called
                    if _profiler_enabled:
                        with profiler_record_fn_context(datapipe):
                            _check_iterator_valid(datapipe, iterator_id)
                            response = gen.send(request)
                    else:  # Decided against using `contextlib.nullcontext` for performance reasons
                        _check_iterator_valid(datapipe, iterator_id)
                        response = gen.send(request)
            except StopIteration:
                return
            except Exception as e:
                # TODO: Simplify the traceback message to skip over `response = gen.send(None)`
                #       Part of https://github.com/pytorch/data/issues/284
                datapipe = args[0]
                msg = "thrown by __iter__ of"
                single_iterator_msg = "single iterator per IterDataPipe constraint"
                if hasattr(e.args, "__len__"):
                    full_msg = f"{msg} {datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})"
                    if len(e.args) == 0 or not isinstance(
                        e.args[0], str
                    ):  # If an exception message doesn't exist
                        e.args = (f"\nThis exception is {full_msg}",)
                    elif msg not in e.args[0] and single_iterator_msg not in e.args[0]:
                        e.args = (
                            e.args[0] + f"\nThis exception is {full_msg}",
                        ) + e.args[1:]
                raise

        namespace["__iter__"] = wrap_generator
    else:  # ``__iter__`` of IterDataPipe is NOT a generator function
        # IterDataPipe is an iterator with both ``__iter__`` and ``__next__``
        # And ``__iter__`` may or may not return `self`
        if "__next__" in namespace:  # If `__next__` exists, put a wrapper around it
            next_func = namespace["__next__"]

            @functools.wraps(next_func)
            def wrap_next(*args, **kwargs):
                datapipe = args[0]
                if torch.autograd._profiler_enabled():
                    with profiler_record_fn_context(datapipe):
                        result = next_func(*args, **kwargs)
                else:
                    result = next_func(*args, **kwargs)
                datapipe._number_of_samples_yielded += 1
                return result

            namespace["__next__"] = wrap_next

            # Note that if the `__next__` and `__iter__` do something completely unrelated. It may cause issue but
            # the user will be violating the iterator protocol. Potential issue:
            # 1. Valid iterator ID may not update or checked properly
            # 2. The number of samples yielded will be miscounted

        # Regardless if `__next__` exists or not, `__iter__` needs a wrapper to track the number of valid iterators
        @functools.wraps(func)
        def wrap_iter(*args, **kwargs):
            iter_ret = func(*args, **kwargs)
            datapipe = args[0]
            datapipe._snapshot_state = _SnapshotState.Iterating
            if datapipe._fast_forward_iterator:
                iter_ret = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                return iter_ret
            iterator_id = _set_datapipe_valid_iterator_id(
                datapipe
            )  # This ID is tied to each created iterator
            return IteratorDecorator(
                iter_ret, datapipe, iterator_id, "__next__" in namespace
            )

        namespace["__iter__"] = wrap_iter
<|endoftext|># mypy: allow-untyped-defs
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.graph_settings import apply_random_seed


# TODO: Caveats
#   1. Caller (either the ReadingService or DataLoader) must pass in the initial RNG
#   2. `in_batch_shuffle` and `bucketbatch` are not compatible with this because they currently
#      lack the option to `set_seed`.
def _simple_graph_snapshot_restoration(
    datapipe: IterDataPipe, n_iterations: int, rng=None
) -> None:
    r"""
    Fast-forward the given DataPipe and its parents by ``n_iterations``, re-doing computations to restore a snapshot.

    For instance, applying this function to the final DataPipe of a graph will restore the snapshot
    (via fast-forward) every DataPipe within the graph.

    After you deserialize a DataPipe, you can use its `_number_of_samples_yielded` attribute as the input
    to this function to forward the DataPipe.

    A DataPipe cannot be restored twice in a row unless there is an iteration started between the restoration
    attempts.

    Note:
        This is the simplest but least efficient way to fast-forward a DataPipe. Usage of other fast-forwarding
        methods (custom ones if necessary) are recommended.

    Args:
        datapipe: IterDataPipe to be fast-forwarded
        n_iterations: number of iterations to fast-forward
        rng: ``Optional[torch.Generator]``. If not ``None``, this RNG will be used for shuffling. The generator
            should be in its `initial` state as it was first passed into ``DataLoader`` or ``ReadingService``.
    """
    if datapipe._snapshot_state == _SnapshotState.Restored:
        raise RuntimeError(
            "Snapshot restoration cannot be applied. You can only restore simple snapshot to the graph "
            "if your graph has not been restored."
        )

    # For this snapshot restoration function, we want the DataPipe to be at its initial state prior to
    # simple fast-forwarding. Therefore, we need to call `reset` twice, because if `SnapshotState` is `Restored`,
    # the first reset will not actually reset.
    datapipe.reset()  # This ensures `SnapshotState` is `Iterating` by this point, even if it was `Restored`.
    # pyrefly: ignore [bad-argument-type]
    apply_random_seed(datapipe, rng)

    remainder = n_iterations
    it = iter(datapipe)  # This always reset the DataPipe if it hasn't already.
    while remainder > 0:
        try:
            next(it)
            remainder -= 1
        except StopIteration as e:
            raise RuntimeError(
                f"Fast-forward {datapipe} by {n_iterations} iterations "
                "exceeds the number of samples available."
            ) from e
    datapipe._fast_forward_iterator = it
    # While the DataPipe has `_fast_forward_iterator`, `next()` will get result from there instead of elsewhere.

    # This will prevent the DataPipe from resetting in the `iter()` call
    # If another DataPipe is consuming it, it won't have to start over again
    datapipe._snapshot_state = _SnapshotState.Restored
<|endoftext|># mypy: allow-untyped-defs
import fnmatch
import functools
import inspect
import os
import warnings
from collections.abc import Callable, Iterable
from io import IOBase
from typing import Any, NoReturn

from torch.utils._import_utils import dill_available


__all__ = [
    "validate_input_col",
    "StreamWrapper",
    "get_file_binaries_from_pathnames",
    "get_file_pathnames_from_root",
    "match_masks",
    "validate_pathname_binary_tuple",
]


# BC for torchdata
DILL_AVAILABLE = dill_available()


def validate_input_col(fn: Callable, input_col: int | tuple | list | None) -> None:
    """
    Check that function used in a callable datapipe works with the input column.

    This simply ensures that the number of positional arguments matches the size
    of the input column. The function must not contain any non-default
    keyword-only arguments.

    Examples:
        >>> # xdoctest: +SKIP("Failing on some CI machines")
        >>> def f(a, b, *, c=1):
        >>>     return a + b + c
        >>> def f_def(a, b=1, *, c=1):
        >>>     return a + b + c
        >>> assert validate_input_col(f, [1, 2])
        >>> assert validate_input_col(f_def, 1)
        >>> assert validate_input_col(f_def, [1, 2])

    Notes:
        If the function contains variable positional (``inspect.VAR_POSITIONAL``) arguments,
        for example, ``f(a, *args)``, the validator will accept any size of input column
        greater than or equal to the number of positional arguments.
        (in this case, 1).

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Raises:
        ValueError: If the function is not compatible with the input column.
    """
    try:
        sig = inspect.signature(fn)
    except (
        ValueError
    ):  # Signature cannot be inspected, likely it is a built-in fn or written in C
        return
    if isinstance(input_col, (list, tuple)):
        input_col_size = len(input_col)
    else:
        input_col_size = 1

    pos = []
    var_positional = False
    non_default_kw_only = []

    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos.append(p)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            if p.default is p.empty:
                non_default_kw_only.append(p)
        else:
            continue

    if isinstance(fn, functools.partial):
        fn_name = getattr(fn.func, "__name__", repr(fn.func))
    else:
        fn_name = getattr(fn, "__name__", repr(fn))

    if len(non_default_kw_only) > 0:
        raise ValueError(
            f"The function {fn_name} takes {len(non_default_kw_only)} "
            f"non-default keyword-only parameters, which is not allowed."
        )

    if len(sig.parameters) < input_col_size:
        if not var_positional:
            raise ValueError(
                f"The function {fn_name} takes {len(sig.parameters)} "
                f"parameters, but {input_col_size} are required."
            )
    else:
        if len(pos) > input_col_size:
            if any(p.default is p.empty for p in pos[input_col_size:]):
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )
        elif len(pos) < input_col_size:
            if not var_positional:
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )


def _is_local_fn(fn):
    # Functions or Methods
    if hasattr(fn, "__code__"):
        return fn.__code__.co_flags & inspect.CO_NESTED
    # Callable Objects
    else:
        if hasattr(fn, "__qualname__"):
            return "<locals>" in fn.__qualname__
        fn_type = type(fn)
        if hasattr(fn_type, "__qualname__"):
            return "<locals>" in fn_type.__qualname__
    return False


def _check_unpickable_fn(fn: Callable) -> None:
    """
    Check function is pickable or not.

    If it is a lambda or local function, a UserWarning will be raised. If it's not a callable function, a TypeError will be raised.
    """
    if not callable(fn):
        raise TypeError(f"A callable function is expected, but {type(fn)} is provided.")

    # Extract function from partial object
    # Nested partial function is automatically expanded as a single partial object
    if isinstance(fn, functools.partial):
        fn = fn.func

    # Local function
    if _is_local_fn(fn) and not dill_available():
        warnings.warn(
            "Local function is not supported by pickle, please use "
            "regular python function or functools.partial instead.",
            stacklevel=2,
        )
        return

    # Lambda function
    if hasattr(fn, "__name__") and fn.__name__ == "<lambda>" and not dill_available():
        warnings.warn(
            "Lambda function is not supported by pickle, please use "
            "regular python function or functools.partial instead.",
            stacklevel=2,
        )
        return


def match_masks(name: str, masks: str | list[str]) -> bool:
    # empty mask matches any input name
    if not masks:
        return True

    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False


def get_file_pathnames_from_root(
    root: str,
    masks: str | list[str],
    recursive: bool = False,
    abspath: bool = False,
    non_deterministic: bool = False,
) -> Iterable[str]:
    # print out an error message and raise the error out
    def onerror(err: OSError) -> NoReturn:
        warnings.warn(err.filename + " : " + err.strerror, stacklevel=2)
        raise err

    if os.path.isfile(root):
        path = root
        if abspath:
            path = os.path.abspath(path)
        fname = os.path.basename(path)
        if match_masks(fname, masks):
            yield path
    else:
        for path, dirs, files in os.walk(root, onerror=onerror):
            if abspath:
                path = os.path.abspath(path)
            if not non_deterministic:
                files.sort()
            for f in files:
                if match_masks(f, masks):
                    yield os.path.join(path, f)
            if not recursive:
                break
            if not non_deterministic:
                # Note that this is in-place modifying the internal list from `os.walk`
                # This only works because `os.walk` doesn't shallow copy before turn
                # https://github.com/python/cpython/blob/f4c03484da59049eb62a9bf7777b963e2267d187/Lib/os.py#L407
                dirs.sort()


def get_file_binaries_from_pathnames(
    pathnames: Iterable, mode: str, encoding: str | None = None
):
    if not isinstance(pathnames, Iterable):
        pathnames = [
            pathnames,
        ]

    if mode in ("b", "t"):
        mode = "r" + mode

    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError(
                f"Expected string type for pathname, but got {type(pathname)}"
            )
        yield pathname, StreamWrapper(open(pathname, mode, encoding=encoding))  # noqa:SIM115


def validate_pathname_binary_tuple(data: tuple[str, IOBase]) -> None:
    if not isinstance(data, tuple):
        raise TypeError(
            f"pathname binary data should be tuple type, but it is type {type(data)}"
        )
    if len(data) != 2:
        raise TypeError(
            f"pathname binary stream tuple length should be 2, but got {len(data)}"
        )
    if not isinstance(data[0], str):
        raise TypeError(
            f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}"
        )
    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )


# Deprecated function names and its corresponding DataPipe type and kwargs for the `_deprecation_warning` function
_iter_deprecated_functional_names: dict[str, dict] = {}
_map_deprecated_functional_names: dict[str, dict] = {}


def _deprecation_warning(
    old_class_name: str,
    *,
    deprecation_version: str,
    removal_version: str,
    old_functional_name: str = "",
    old_argument_name: str = "",
    new_class_name: str = "",
    new_functional_name: str = "",
    new_argument_name: str = "",
    deprecate_functional_name_only: bool = False,
) -> None:
    if new_functional_name and not old_functional_name:
        raise ValueError(
            "Old functional API needs to be specified for the deprecation warning."
        )
    if new_argument_name and not old_argument_name:
        raise ValueError(
            "Old argument name needs to be specified for the deprecation warning."
        )

    if old_functional_name and old_argument_name:
        raise ValueError(
            "Deprecating warning for functional API and argument should be separated."
        )

    msg = f"`{old_class_name}()`"
    if deprecate_functional_name_only and old_functional_name:
        msg = f"{msg}'s functional API `.{old_functional_name}()` is"
    elif old_functional_name:
        msg = f"{msg} and its functional API `.{old_functional_name}()` are"
    elif old_argument_name:
        msg = f"The argument `{old_argument_name}` of {msg} is"
    else:
        msg = f"{msg} is"
    msg = (
        f"{msg} deprecated since {deprecation_version} and will be removed in {removal_version}."
        f"\nSee https://github.com/pytorch/data/issues/163 for details."
    )

    if new_class_name or new_functional_name:
        msg = f"{msg}\nPlease use"
        if new_class_name:
            msg = f"{msg} `{new_class_name}()`"
        if new_class_name and new_functional_name:
            msg = f"{msg} or"
        if new_functional_name:
            msg = f"{msg} `.{new_functional_name}()`"
        msg = f"{msg} instead."

    if new_argument_name:
        msg = f"{msg}\nPlease use `{old_class_name}({new_argument_name}=)` instead."

    warnings.warn(msg, FutureWarning, stacklevel=2)


class StreamWrapper:
    """
    StreamWrapper is introduced to wrap file handler generated by DataPipe operation like `FileOpener`.

    StreamWrapper would guarantee the wrapped file handler is closed when it's out of scope.
    """

    session_streams: dict[Any, int] = {}
    debug_unclosed_streams: bool = False

    def __init__(self, file_obj, parent_stream=None, name=None) -> None:
        self.file_obj = file_obj
        self.child_counter = 0
        self.parent_stream = parent_stream
        self.close_on_last_child = False
        self.name = name
        self.closed = False
        if parent_stream is not None:
            if not isinstance(parent_stream, StreamWrapper):
                raise RuntimeError(
                    f"Parent stream should be StreamWrapper, {type(parent_stream)} was given"
                )
            parent_stream.child_counter += 1
            self.parent_stream = parent_stream
        if StreamWrapper.debug_unclosed_streams:
            StreamWrapper.session_streams[self] = 1

    @classmethod
    def close_streams(cls, v, depth=0) -> None:
        """Traverse structure and attempts to close all found StreamWrappers on best effort basis."""
        if depth > 10:
            return
        if isinstance(v, StreamWrapper):
            v.close()
        else:
            # Traverse only simple structures
            if isinstance(v, dict):
                for vv in v.values():
                    cls.close_streams(vv, depth=depth + 1)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    cls.close_streams(vv, depth=depth + 1)

    def __getattr__(self, name):
        file_obj = self.__dict__["file_obj"]
        return getattr(file_obj, name)

    def close(self, *args, **kwargs) -> None:
        if self.closed:
            return
        if StreamWrapper.debug_unclosed_streams:
            del StreamWrapper.session_streams[self]
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if (
                not self.parent_stream.child_counter
                and self.parent_stream.close_on_last_child
            ):
                self.parent_stream.close()
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        self.closed = True

    def autoclose(self) -> None:
        """Automatically close stream when all child streams are closed or if there are none."""
        self.close_on_last_child = True
        if self.child_counter == 0:
            self.close()

    def __dir__(self):
        attrs = list(self.__dict__.keys()) + list(StreamWrapper.__dict__.keys())
        attrs += dir(self.file_obj)
        return list(set(attrs))

    def __del__(self) -> None:
        if not self.closed:
            self.close()

    def __iter__(self):
        yield from self.file_obj

    def __next__(self):
        return next(self.file_obj)

    def __repr__(self) -> str:
        if self.name is None:
            return f"StreamWrapper<{self.file_obj!r}>"
        else:
            return f"StreamWrapper<{self.name},{self.file_obj!r}>"

    def __getstate__(self):
        return self.file_obj

    def __setstate__(self, obj):
        self.file_obj = obj
<|endoftext|># mypy: allow-untyped-defs
# This file takes partial of the implementation from NVIDIA's webdataset at here:
# https://github.com/tmbdev/webdataset/blob/master/webdataset/autodecode.py

import io
import json
import os.path
import pickle
import tempfile

import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper


__all__ = [
    "Decoder",
    "ImageHandler",
    "MatHandler",
    "audiohandler",
    "basichandlers",
    "extension_extract_fn",
    "handle_extension",
    "imagehandler",
    "mathandler",
    "videohandler",
]


################################################################
# handle basic datatypes
################################################################
def basichandlers(extension: str, data):
    """Transforms raw data (byte stream) into python objects.

    Looks at the extension and loads the data into a python object supporting
    the corresponding extension.

    Args:
        extension (str): The file extension
        data (byte stream): Data to load into a python object.

    Returns:
        object: The data loaded into a corresponding python object
            supporting the extension.

    Example:
        >>> import pickle
        >>> data = pickle.dumps("some data")
        >>> new_data = basichandlers("pickle", data)
        >>> new_data
        some data

    The transformation of data for extensions are:
        - txt, text, transcript: utf-8 decoded data of str format
        - cls, cls2, class, count, index, inx, id: int
        - json, jsn: json loaded data
        - pickle, pyd: pickle loaded data
        - pt: torch loaded data
    """

    if extension in "txt text transcript":
        return data.decode("utf-8")

    if extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        try:
            return int(data)
        except ValueError:
            return None

    if extension in "json jsn":
        return json.loads(data)

    if extension in ["pyd", "pickle"]:
        return pickle.loads(data)

    if extension == "pt":
        stream = io.BytesIO(data)
        return torch.load(stream)

    # if extension in "ten tb".split():
    #     from . import tenbin
    #     return tenbin.decode_buffer(data)

    # if extension in "mp msgpack msg".split():
    #     import msgpack
    #     return msgpack.unpackb(data)

    return None


################################################################
# handle images
################################################################
imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    "torch": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


def handle_extension(extensions, f):
    """
    Return a decoder handler function for the list of extensions.

    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.
    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """
    extensions = extensions.lower().split()

    def g(key, data):
        extension = key.lower().split(".")

        for target in extensions:
            target = target.split(".")
            if len(target) > len(extension):
                continue

            if extension[-len(target) :] == target:
                return f(data)
            return None

    return g


class ImageHandler:
    """
    Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """

    def __init__(self, imagespec) -> None:
        if imagespec not in list(imagespecs.keys()):
            raise AssertionError(f"unknown image specification: {imagespec}")
        self.imagespec = imagespec.lower()

    def __call__(self, extension, data):
        if extension.lower() not in ["jpg", "jpeg", "png", "ppm", "pgm", "pbm", "pnm"]:
            return None

        try:
            import numpy as np
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Package `numpy` is required to be installed for default image decoder."
                "Please use `pip install numpy` to install the package"
            ) from e

        try:
            import PIL.Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Package `PIL` is required to be installed for default image decoder."
                "Please use `pip install Pillow` to install the package"
            ) from e

        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]

        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
            if atype == "pil":
                return img
            elif atype == "numpy":
                result = np.asarray(img)
                if result.dtype != np.uint8:
                    raise AssertionError(
                        f"numpy image array should be type uint8, but got {result.dtype}"
                    )
                if etype == "uint8":
                    return result
                else:
                    return result.astype("f") / 255.0
            elif atype == "torch":
                result = np.asarray(img)
                if result.dtype != np.uint8:
                    raise AssertionError(
                        f"numpy image array should be type uint8, but got {result.dtype}"
                    )

                if etype == "uint8":
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result)
                else:
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result) / 255.0
            return None


def imagehandler(imagespec):
    return ImageHandler(imagespec)


################################################################
# torch video
################################################################
def videohandler(extension, data):
    if extension not in [
        "mp4",
        "ogv",
        "mjpeg",
        "avi",
        "mov",
        "h264",
        "mpg",
        "webm",
        "wmv",
    ]:
        return None

    try:
        import torchvision.io
    except ImportError as e:
        raise ModuleNotFoundError(
            "Package `torchvision` is required to be installed for default video file loader."
            "Please use `pip install torchvision`"
            "to install the package"
        ) from e

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return torchvision.io.read_video(fname)


################################################################
# torchaudio
################################################################
def audiohandler(extension, data):
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    try:
        import torchaudio  # type: ignore[import]
    except ImportError as e:
        raise ModuleNotFoundError(
            "Package `torchaudio` is required to be installed for default audio file loader."
            "Please use `pip install torchaudio`"
            "to install the package"
        ) from e

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return torchaudio.load(fname)


################################################################
# mat
################################################################
class MatHandler:
    def __init__(self, **loadmat_kwargs) -> None:
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ModuleNotFoundError(
                "Package `scipy` is required to be installed for mat file."
                "Please use `pip install scipy`"
                "to install the package"
            ) from e
        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def __call__(self, extension, data):
        if extension != "mat":
            return None
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)


def mathandler(**loadmat_kwargs):
    return MatHandler(**loadmat_kwargs)


################################################################
# a sample decoder
################################################################
# Extract extension from pathname
def extension_extract_fn(pathname):
    ext = os.path.splitext(pathname)[1]
    # Remove dot
    if ext:
        ext = ext[1:]
    return ext


class Decoder:
    """
    Decode key/data sets using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, *handler, key_fn=extension_extract_fn) -> None:
        self.handlers = list(handler) if handler else []
        self.key_fn = key_fn

    # Insert new handler from the beginning of handlers list to make sure the new
    # handler having the highest priority
    def add_handler(self, *handler) -> None:
        if not handler:
            return
        self.handlers = list(handler) + self.handlers

    @staticmethod
    def _is_stream_handle(data):
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, (io.BufferedIOBase, io.RawIOBase))

    def decode1(self, key, data):
        if not data:
            return data

        # if data is a stream handle, we need to read all the content before decoding
        if Decoder._is_stream_handle(data):
            ds = data
            # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
            data = b"".join(data)
            ds.close()

        for f in self.handlers:
            result = f(key, data)
            if result is not None:
                return result
        return data

    def decode(self, data):
        result = {}
        # single data tuple(pathname, data stream)
        if isinstance(data, tuple):
            data = [data]

        if data is not None:
            for k, v in data:
                # TODO: xinyu, figure out why Nvidia do this?
                if k[0] == "_":
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                result[k] = self.decode1(self.key_fn(k), v)
        return result

    def __call__(self, data):
        return self.decode(data)
<|endoftext|># mypy: allow-untyped-defs
import random
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


__all__ = [
    "ConcatDataFramesPipe",
    "DataFramesAsTuplesPipe",
    "ExampleAggregateAsDataFrames",
    "FilterDataFramesPipe",
    "PerRowDataFramesPipe",
    "ShuffleDataFramesPipe",
]


@functional_datapipe("_dataframes_as_tuples")
class DataFramesAsTuplesPipe(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # for record in df.to_records(index=False):
            yield from df_wrapper.iterate(df)


@functional_datapipe("_dataframes_per_row", enable_df_api_tracing=True)
class PerRowDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # TODO(VitalyFedyunin): Replacing with TorchArrow only API, as we are dropping pandas as followup
            for i in range(len(df)):
                yield df[i : i + 1]


@functional_datapipe("_dataframes_concat", enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, batch=3) -> None:
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                yield df_wrapper.concat(buffer)
                buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_shuffle", enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        size = None
        all_buffer: list[Any] = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            all_buffer.extend(
                df_wrapper.get_item(df, i) for i in range(df_wrapper.get_len(df))
            )
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield df_wrapper.concat(buffer)
                buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_filter", enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, filter_fn) -> None:
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn

    def __iter__(self):
        size = None
        all_buffer = []
        filter_res = []
        # pyrefly: ignore [bad-assignment]
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i : i + 1])
                filter_res.append(self.filter_fn(df.iloc[i]))

        buffer = []
        for df, res in zip(all_buffer, filter_res, strict=True):
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_to_dataframes_pipe", enable_df_api_tracing=True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):
    def __init__(self, source_datapipe, dataframe_size=10, columns=None) -> None:
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        try:
            return list(item)
        except (
            Exception
        ):  # TODO(VitalyFedyunin): Replace with better iterable exception
            return [item]

    def __iter__(self):
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
<|endoftext|>from collections.abc import Iterator
from typing import Any

from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DataChunk


__all__ = ["DataChunkDF"]


class DataChunkDF(DataChunk):
    """DataChunkDF iterating over individual items inside of DataFrame containers, to access DataFrames user `raw_iterator`."""

    def __iter__(self) -> Iterator[Any]:
        for df in self.items:
            yield from df_wrapper.iterate(df)

    def __len__(self) -> int:
        total_len = 0
        for df in self.items:
            total_len += df_wrapper.get_len(df)
        return total_len
<|endoftext|>from torch.utils.data.datapipes.dataframe.dataframes import (
    CaptureDataFrame,
    DFIterDataPipe,
)
from torch.utils.data.datapipes.dataframe.datapipes import DataFramesAsTuplesPipe


__all__ = ["CaptureDataFrame", "DFIterDataPipe", "DataFramesAsTuplesPipe"]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
<|endoftext|># mypy: allow-untyped-defs
from typing import Any, NoReturn

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


# TODO(VitalyFedyunin): Add error when two different traces get combined

__all__ = [
    "Capture",
    "CaptureA",
    "CaptureAdd",
    "CaptureCall",
    "CaptureControl",
    "CaptureDataFrame",
    "CaptureDataFrameWithDataPipeOps",
    "CaptureF",
    "CaptureGetAttr",
    "CaptureGetItem",
    "CaptureInitial",
    "CaptureLikeMock",
    "CaptureMul",
    "CaptureSetItem",
    "CaptureSub",
    "CaptureVariable",
    "CaptureVariableAssign",
    "DataFrameTracer",
    "DataFrameTracedOps",
    "disable_capture",
    "get_val",
]


def disable_capture() -> None:
    CaptureControl.disabled = True


class CaptureControl:
    disabled = False


class DataFrameTracedOps(DFIterDataPipe):
    def __init__(self, source_datapipe, output_var) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        for item in self.source_datapipe:
            yield self.output_var.apply_ops(item)


#  TODO(VitalyFedyunin): Extract this list from the DFIterDataPipe registered functions
DATAPIPES_OPS = [
    "_dataframes_as_tuples",
    "groupby",
    "_dataframes_filter",
    "map",
    "to_datapipe",
    "shuffle",
    "concat",
    "batch",
    "_dataframes_per_row",
    "_dataframes_concat",
    "_dataframes_shuffle",
]

UNIMPLEMENTED_ATTR = ["__deepcopy__", "__setstate__", "is_shardable", "apply_sharding"]


class Capture:
    # TODO: All operations are shared across entire InitialCapture, need to figure out what if we join two captures

    def __init__(self, schema_df=None) -> None:
        self.ctx = {"operations": [], "variables": [], "schema_df": schema_df}

    def __str__(self) -> str:
        return self._ops_str()

    def _ops_str(self):
        res = ""
        for op in self.ctx["operations"]:
            if len(res) > 0:
                res += "\n"
            res += str(op)
        return res

    def __getstate__(self):
        # TODO(VitalyFedyunin): Currently can't pickle (why?)
        self.ctx["schema_df"] = None
        for var in self.ctx["variables"]:
            var.calculated_value = None
        state = {}
        for item in self.__dict__:
            state[item] = getattr(self, item)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __getattr__(self, attrname):
        if attrname == "kwarg" or attrname == "kwargs":
            raise RuntimeError("no kwargs!")
        if attrname == "__deepcopy__":
            raise AttributeError
        result = CaptureGetAttr(self, attrname, ctx=self.ctx)
        return result

    def __getitem__(self, key):
        return CaptureGetItem(self, key, ctx=self.ctx)

    def __setitem__(self, key, value) -> None:
        self.ctx["operations"].append(CaptureSetItem(self, key, value, ctx=self.ctx))

    def __add__(self, add_val):
        res = CaptureAdd(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx["operations"].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        )
        return var

    def __sub__(self, add_val):
        res = CaptureSub(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx["operations"].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        )
        return var

    def __mul__(self, add_val):
        res = CaptureMul(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        t = CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        self.ctx["operations"].append(t)
        return var

    def _is_context_empty(self):
        return len(self.ctx["operations"]) == 0 and len(self.ctx["variables"]) == 0

    def apply_ops_2(self, dataframe) -> None:
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        self.ctx["variables"][0].calculated_value = dataframe
        for op in self.ctx["operations"]:
            op.execute()

    @property
    def columns(self):
        self.apply_ops_2(self.ctx["schema_df"])
        value = self.execute()
        return value.columns

    # TODO(VitalyFedyunin): Add tests
    # TODO(VitalyFedyunin): Need to join context if one of them are empty because we used capture

    def __call__(self, *args, **kwargs):
        # TODO: Check if args or kwargs have more than one different context
        if self._is_context_empty():
            # TODO: Allow CaptureA to take context from mock
            for arg in args:
                if isinstance(arg, Capture) and not arg._is_context_empty():
                    self.ctx = arg.ctx
                    break
            if self._is_context_empty():
                for k, v in kwargs.items():
                    if isinstance(k, Capture) and not k._is_context_empty():
                        self.ctx = k.ctx
                        break
                    if isinstance(v, Capture) and not v._is_context_empty():
                        self.ctx = v.ctx
                        break

        res = CaptureCall(self, ctx=self.ctx, args=args, kwargs=kwargs)
        var = CaptureVariable(None, ctx=self.ctx)
        t = CaptureVariableAssign(ctx=self.ctx, variable=var, value=res)
        self.ctx["operations"].append(t)
        return var


class CaptureF(Capture):
    def __init__(self, ctx=None, **kwargs) -> None:
        super().__init__()
        if ctx is None:
            self.ctx = {"operations": [], "variables": []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs


class CaptureA(CaptureF):
    def __str__(self) -> str:
        return f"{self.kwargs['name']}"

    def execute(self):
        value = self.kwargs["real_attribute"]
        return value


class CaptureLikeMock:
    def __init__(self, name) -> None:
        import unittest.mock as mock

        # TODO(VitalyFedyunin): Do not use private function here, copy own implementation instead.
        get_target, attribute = mock._get_target(name)  # type: ignore[attr-defined]
        self.get_target = get_target
        self.attribute = attribute
        self.name = name

    def __enter__(self):
        self.save = getattr(self.get_target(), self.attribute)
        capt = CaptureA(name=self.name, real_attribute=self.save)
        setattr(self.get_target(), self.attribute, capt)

    def __exit__(self, *exc_info):
        setattr(self.get_target(), self.attribute, self.save)


class CaptureCall(Capture):
    def __init__(self, callable, ctx=None, **kwargs) -> None:
        super().__init__()
        if ctx is None:
            self.ctx = {"operations": [], "variables": []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs
        self.callable = callable

    def __str__(self) -> str:
        return "{callable}({args},{kwargs})".format(
            callable=self.callable, **self.kwargs
        )

    def execute(self):
        # TODO: VitalyFedyunin execute kwargs and maybe nested structures
        executed_args = []
        for arg in self.kwargs["args"]:
            if isinstance(arg, Capture):
                executed_args.append(arg.execute())
            else:
                executed_args.append(arg)
        left = get_val(self.callable)
        return left(*executed_args, **self.kwargs["kwargs"])


class CaptureVariableAssign(CaptureF):
    def __str__(self) -> str:
        variable = self.kwargs["variable"]
        value = self.kwargs["value"]
        return f"{variable} = {value}"

    def execute(self) -> None:
        self.kwargs["variable"].calculated_value = self.kwargs["value"].execute()


class CaptureVariable(Capture):
    # TODO(VitalyFedyunin): This should be atomic and thread safe
    names_idx = 0

    def __init__(self, value, ctx) -> None:
        super().__init__()
        if CaptureControl.disabled:
            raise RuntimeError("Attempting to create capture variable with capture off")
        self.ctx = ctx
        self.value = value
        self.name = f"var_{CaptureVariable.names_idx}"
        CaptureVariable.names_idx += 1
        self.ctx["variables"].append(self)

    def __str__(self) -> str:
        return self.name

    def execute(self):
        return self.calculated_value

    def apply_ops(self, dataframe):
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        self.ctx["variables"][0].calculated_value = dataframe
        for op in self.ctx["operations"]:
            op.execute()
        return self.calculated_value


class CaptureGetItem(Capture):
    def __init__(self, left, key, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.key = key

    def __str__(self) -> str:
        return f"{self.left}[{get_val(self.key)}]"

    def execute(self):
        left = self.left.execute()
        return left[self.key]


class CaptureSetItem(Capture):
    def __init__(self, left, key, value, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.key = key
        self.value = value

    def __str__(self) -> str:
        return f"{self.left}[{get_val(self.key)}] = {self.value}"

    def execute(self) -> None:
        left = self.left.execute()
        value = self.value.execute()
        left[self.key] = value


class CaptureAdd(Capture):
    def __init__(self, left, right, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} + {self.right}"

    def execute(self):
        return get_val(self.left) + get_val(self.right)


class CaptureMul(Capture):
    def __init__(self, left, right, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} * {self.right}"

    def execute(self):
        return get_val(self.left) * get_val(self.right)


class CaptureSub(Capture):
    def __init__(self, left, right, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} - {self.right}"

    def execute(self):
        return get_val(self.left) - get_val(self.right)


class CaptureGetAttr(Capture):
    def __init__(self, src, name, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.src = src
        self.name = name

    def __str__(self) -> str:
        return f"{self.src}.{self.name}"

    def execute(self):
        val = get_val(self.src)
        return getattr(val, self.name)


def get_val(capture):
    if isinstance(capture, Capture):
        return capture.execute()
    elif isinstance(capture, str):
        return f'"{capture}"'
    else:
        return capture


class CaptureInitial(CaptureVariable):
    def __init__(self, schema_df=None) -> None:
        # pyrefly: ignore [bad-assignment]
        new_ctx: dict[str, list[Any]] = {
            "operations": [],
            "variables": [],
            "schema_df": schema_df,
        }
        super().__init__(None, new_ctx)
        self.name = f"input_{self.name}"


class CaptureDataFrame(CaptureInitial):
    pass


class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):
    def as_datapipe(self):
        return DataFrameTracedOps(self.ctx["variables"][0].source_datapipe, self)

    def raw_iterator(self):
        return self.as_datapipe().__iter__()

    def __iter__(self):
        return iter(self._dataframes_as_tuples())

    def batch(self, batch_size=10, drop_last: bool = False, wrapper_class=DataChunkDF):
        dp = self._dataframes_per_row()._dataframes_concat(batch_size)
        dp = dp.as_datapipe().batch(1, drop_last=drop_last, wrapper_class=wrapper_class)
        dp._dp_contains_dataframe = True
        return dp

    def groupby(
        self,
        group_key_fn,
        *,
        buffer_size=10000,
        group_size=None,
        guaranteed_group_size=None,
        drop_remaining=False,
    ):
        dp = self._dataframes_per_row()
        dp = dp.as_datapipe().groupby(
            group_key_fn,
            buffer_size=buffer_size,
            group_size=group_size,
            guaranteed_group_size=guaranteed_group_size,
            drop_remaining=drop_remaining,
        )
        return dp

    def shuffle(self, *args, **kwargs):
        return self._dataframes_shuffle(*args, **kwargs)

    def filter(self, *args, **kwargs):
        return self._dataframes_filter(*args, **kwargs)

    def collate(self, *args, **kwargs) -> NoReturn:
        raise RuntimeError("Can't collate unbatched DataFrames stream")

    def __getattr__(self, attrname):  # ?
        if attrname in UNIMPLEMENTED_ATTR:
            raise AttributeError("Attempting to get ", attrname)
        if attrname in DATAPIPES_OPS:
            return (self.as_datapipe()).__getattr__(attrname)
        return super().__getattr__(attrname)


@functional_datapipe("trace_as_dataframe")
class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):  # type: ignore[misc]
    source_datapipe: Any | None = None

    # TODO(VitalyFedyunin): Must implement all special functions of datapipes

    def set_shuffle_settings(self, *args, **kwargs) -> None:
        pass

    def is_shardable(self) -> bool:
        return False

    def __init__(self, source_datapipe, schema_df=None) -> None:
        self.source_datapipe = source_datapipe
        if schema_df is None:
            schema_df = next(iter(self.source_datapipe))
        super().__init__(schema_df=schema_df)
<|endoftext|># mypy: allow-untyped-defs
from typing import Any


_pandas: Any = None
_WITH_PANDAS: bool | None = None


def _try_import_pandas() -> bool:
    try:
        import pandas  # type: ignore[import]

        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        return False


# pandas used only for prototyping, will be shortly replaced with TorchArrow
def _with_pandas() -> bool:
    global _WITH_PANDAS
    if _WITH_PANDAS is None:
        _WITH_PANDAS = _try_import_pandas()
    return _WITH_PANDAS


class PandasWrapper:
    @classmethod
    def create_dataframe(cls, data, columns):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return _pandas.DataFrame(data, columns=columns)  # type: ignore[union-attr]

    @classmethod
    def is_dataframe(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.frame.DataFrame)  # type: ignore[union-attr]

    @classmethod
    def is_column(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.series.Series)  # type: ignore[union-attr]

    @classmethod
    def iterate(cls, data):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        yield from data.itertuples(index=False)

    @classmethod
    def concat(cls, buffer):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return _pandas.concat(buffer)  # type: ignore[union-attr]

    @classmethod
    def get_item(cls, data, idx):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return data[idx : idx + 1]

    @classmethod
    def get_len(cls, df):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return len(df.index)

    @classmethod
    def get_columns(cls, df):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return list(df.columns.values.tolist())


# When you build own implementation just override it with dataframe_wrapper.set_df_wrapper(new_wrapper_class)
default_wrapper = PandasWrapper


def get_df_wrapper():
    return default_wrapper


def set_df_wrapper(wrapper) -> None:
    global default_wrapper
    default_wrapper = wrapper


def create_dataframe(data, columns=None):
    wrapper = get_df_wrapper()
    return wrapper.create_dataframe(data, columns)


def is_dataframe(data):
    wrapper = get_df_wrapper()
    return wrapper.is_dataframe(data)


def get_columns(data):
    wrapper = get_df_wrapper()
    return wrapper.get_columns(data)


def is_column(data):
    wrapper = get_df_wrapper()
    return wrapper.is_column(data)


def concat(buffer):
    wrapper = get_df_wrapper()
    return wrapper.concat(buffer)


def iterate(data):
    wrapper = get_df_wrapper()
    return wrapper.iterate(data)


def get_item(data, idx):
    wrapper = get_df_wrapper()
    return wrapper.get_item(data, idx)


def get_len(df):
    wrapper = get_df_wrapper()
    return wrapper.get_len(df)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe


__all__ = ["ConcaterMapDataPipe", "ZipperMapDataPipe"]

_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("concat")
class ConcaterMapDataPipe(MapDataPipe):
    r"""
    Concatenate multiple Map DataPipes (functional name: ``concat``).

    The new index is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
        datapipes: Map DataPipes being concatenated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """

    datapipes: tuple[MapDataPipe]

    def __init__(self, *datapipes: MapDataPipe) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        # pyrefly: ignore [unsafe-overlap]
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __getitem__(self, index) -> _T_co:  # type: ignore[type-var]
        offset = 0
        for dp in self.datapipes:
            # pyrefly: ignore [bad-argument-type]
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                # pyrefly: ignore [bad-argument-type]
                offset += len(dp)
        raise IndexError(f"Index {index} is out of range.")

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return sum(len(dp) for dp in self.datapipes)


@functional_datapipe("zip")
class ZipperMapDataPipe(MapDataPipe[tuple[_T_co, ...]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MapDataPipe is out of bounds as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """

    datapipes: tuple[MapDataPipe[_T_co], ...]

    def __init__(self, *datapipes: MapDataPipe[_T_co]) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        # pyrefly: ignore [unsafe-overlap]
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes

    def __getitem__(self, index) -> tuple[_T_co, ...]:
        res = []
        for dp in self.datapipes:
            try:
                res.append(dp[index])
            except IndexError as e:
                raise IndexError(
                    f"Index {index} is out of range for one of the input MapDataPipes {dp}."
                ) from e
        return tuple(res)

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return min(len(dp) for dp in self.datapipes)
<|endoftext|>import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from torch.utils.data.datapipes.datapipe import MapDataPipe


_T = TypeVar("_T")

__all__ = ["SequenceWrapperMapDataPipe"]


class SequenceWrapperMapDataPipe(MapDataPipe[_T]):
    r"""
    Wraps a sequence object into a MapDataPipe.

    Args:
        sequence: Sequence object to be wrapped into an MapDataPipe
        deepcopy: Option to deepcopy input sequence object

    .. note::
      If ``deepcopy`` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> dp = SequenceWrapper({"a": 100, "b": 200, "c": 300, "d": 400})
        >>> dp["a"]
        100
    """

    sequence: Sequence[_T] | Mapping[Any, _T]

    def __init__(
        self, sequence: Sequence[_T] | Mapping[Any, _T], deepcopy: bool = True
    ) -> None:
        if deepcopy:
            try:
                self.sequence = copy.deepcopy(sequence)
            except TypeError:
                warnings.warn(
                    "The input sequence can not be deepcopied, "
                    "please be aware of in-place modification would affect source data",
                    stacklevel=2,
                )
                self.sequence = sequence
        else:
            self.sequence = sequence

    def __getitem__(self, index: int) -> _T:
        return self.sequence[index]

    def __len__(self) -> int:
        return len(self.sequence)
<|endoftext|># Functional DataPipe
from torch.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from torch.utils.data.datapipes.map.combinatorics import (
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.map.combining import (
    ConcaterMapDataPipe as Concater,
    ZipperMapDataPipe as Zipper,
)
from torch.utils.data.datapipes.map.grouping import BatcherMapDataPipe as Batcher
from torch.utils.data.datapipes.map.utils import (
    SequenceWrapperMapDataPipe as SequenceWrapper,
)


__all__ = ["Batcher", "Concater", "Mapper", "SequenceWrapper", "Shuffler", "Zipper"]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe


__all__ = ["BatcherMapDataPipe"]


_T = TypeVar("_T")


@functional_datapipe("batch")
class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    r"""
    Create mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> batch_dp = dp.batch(batch_size=2)
        >>> list(batch_dp)
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """

    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        datapipe: MapDataPipe[_T],
        batch_size: int,
        drop_last: bool = False,
        wrapper_class: type[DataChunk] = DataChunk,
    ) -> None:
        if batch_size <= 0:
            raise AssertionError("Batch size is required to be larger than 0!")
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __getitem__(self, index) -> DataChunk:
        batch: list = []
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        try:
            batch.extend(self.datapipe[i] for i in indices)
            return self.wrapper_class(batch)
        except IndexError as e:
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.") from e

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


__all__ = ["MapperMapDataPipe", "default_fn"]


_T_co = TypeVar("_T_co", covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


@functional_datapipe("map")
class MapperMapDataPipe(MapDataPipe[_T_co]):
    r"""
    Apply the input function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source MapDataPipe
        fn: Function being applied to each item

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    datapipe: MapDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return len(self.datapipe)

    def __getitem__(self, index) -> _T_co:
        return self.fn(self.datapipe[index])
<|endoftext|># mypy: allow-untyped-defs
import random
from collections.abc import Iterator
from typing import TypeVar

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


__all__ = ["ShufflerIterDataPipe"]


_T_co = TypeVar("_T_co", covariant=True)


# @functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle().set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        >>> list(shuffle_dp)
        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
        >>> # Reset seed for Shuffler
        >>> shuffle_dp = shuffle_dp.set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    Note:
        Even though this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
        of data during data-processing.
    """

    datapipe: MapDataPipe[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random

    def __init__(
        self,
        datapipe: MapDataPipe[_T_co],
        *,
        indices: list | None = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # pyrefly: ignore [bad-argument-type]
        self.indices = list(range(len(datapipe))) if indices is None else indices
        self._enabled = True
        self._seed = None
        self._rng = random.Random()
        self._shuffled_indices: list = self.indices

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[_T_co]:
        if not self._enabled:
            for idx in self.indices:
                yield self.datapipe[idx]
        else:
            while self._shuffled_indices:
                idx = self._shuffled_indices.pop()
                yield self.datapipe[idx]

    def reset(self) -> None:
        if self._enabled and self._seed is None:
            self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._rng.seed(self._seed)
        self._seed = None
        self._shuffled_indices = self._rng.sample(self.indices, len(self.indices))

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return len(self.datapipe)

    def __getstate__(self):
        state = (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            rng_state,
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)


MapDataPipe.register_datapipe_as_function("shuffle", ShufflerIterDataPipe)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable, Iterator
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    StreamWrapper,
    validate_input_col,
)


__all__ = ["FilterIterDataPipe"]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("filter")
class FilterIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        input_col: Index or indices of data which ``filter_fn`` is applied, such as:

            - ``None`` as default to apply ``filter_fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """

    datapipe: IterDataPipe[_T_co]
    filter_fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        filter_fn: Callable,
        input_col=None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(filter_fn)
        self.filter_fn = filter_fn  # type: ignore[assignment]

        self.input_col = input_col
        validate_input_col(filter_fn, input_col)

    def _apply_filter_fn(self, data) -> bool:
        if self.input_col is None:
            return self.filter_fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.filter_fn(*args)
        else:
            return self.filter_fn(data[self.input_col])

    def __iter__(self) -> Iterator[_T_co]:
        for data in self.datapipe:
            condition, filtered = self._returnIfTrue(data)
            if condition:
                yield filtered
            else:
                StreamWrapper.close_streams(data)

    def _returnIfTrue(self, data: _T) -> tuple[bool, _T]:
        condition = self._apply_filter_fn(data)

        if df_wrapper.is_column(condition):
            # We are operating on DataFrames filter here
            result = []
            for idx, mask in enumerate(df_wrapper.iterate(condition)):
                if mask:
                    result.append(df_wrapper.get_item(data, idx))
            if result:
                return True, df_wrapper.concat(result)
            else:
                return False, None  # type: ignore[return-value]

        if not isinstance(condition, bool):
            raise ValueError(
                "Boolean output is required for `filter_fn` of FilterIterDataPipe, got",
                type(condition),
            )

        return condition, data
<|endoftext|># mypy: allow-untyped-defs
import copy as copymodule
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator, Sized
from typing import Any, Literal, TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, StreamWrapper


__all__ = [
    "ConcaterIterDataPipe",
    "DemultiplexerIterDataPipe",
    "ForkerIterDataPipe",
    "MultiplexerIterDataPipe",
    "ZipperIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("concat")
class ConcaterIterDataPipe(IterDataPipe):
    r"""
    Concatenates multiple Iterable DataPipes (functional name: ``concat``).

    The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """

    datapipes: tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            yield from dp

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            # pyrefly: ignore [bad-argument-type]
            return sum(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("fork")
class ForkerIterDataPipe(IterDataPipe):
    r"""
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
        copy: copy strategy to use for items yielded by each branch. Supported
            options are ``None`` for no copying, ``"shallow"`` for shallow object
            copies, and ``"deep"`` for deep object copies. Defaults to ``None``.

    Note:
        All branches of the forked pipeline return the identical object unless
        the copy parameter is supplied. If the object is mutable or contains
        mutable objects, changing them in one branch will affect all others.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.fork(num_instances=2)
        >>> list(dp1)
        [0, 1, 2, 3, 4]
        >>> list(dp2)
        [0, 1, 2, 3, 4]
    """

    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Literal["shallow", "deep"] | None = None,
    ):
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instances` larger than 0, but {num_instances} is found"
            )
        if num_instances == 1:
            return datapipe
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size, copy)  # type: ignore[abstract]
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _ContainerTemplate(ABC):
    r"""Abstract class for container ``DataPipes``. The following are three required methods."""

    @abstractmethod
    def get_next_element_by_instance(self, instance_id: int): ...

    @abstractmethod
    def is_every_instance_exhausted(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_length_by_instance(self, instance_id: int):
        r"""Raise TypeError if it's not supposed to be implemented to support `list(datapipe)`."""


def _no_op(x):
    return x


class _ForkerIterDataPipe(IterDataPipe, _ContainerTemplate):
    r"""
    Container to hold instance-specific information on behalf of ForkerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Literal["shallow", "deep"] | None = None,
    ) -> None:
        self.main_datapipe = datapipe
        self._datapipe_iterator: Iterator[Any] | None = None
        self.num_instances = num_instances
        self.buffer: deque = deque()
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `fork`, "
                "please be aware of OOM at random places",
                UserWarning,
                stacklevel=2,
            )
        if copy is None:
            self.copy_fn = _no_op
        elif copy == "shallow":
            self.copy_fn = copymodule.copy
        elif copy == "deep":
            self.copy_fn = copymodule.deepcopy
        else:
            raise ValueError(
                f"Unknown copy method `{copy}` requested, choose one of None, `shallow` or `deep`."
            )

        self.child_pointers: list[int] = [
            0
        ] * num_instances  # Indicate the indices of the next element to get
        self.slowest_ptr = 0  # The index to read by the slowest child
        self.leading_ptr = 0  # The index to read by the fastest child
        self.end_ptr: int | None = None  # The index to stop child
        self._child_stop: list[bool] = [True for _ in range(num_instances)]

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return len(self.main_datapipe)

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating
            for i in range(self.num_instances):
                self._child_stop[i] = False
        try:
            while not self._child_stop[instance_id]:
                self.child_pointers[instance_id] += 1
                if (
                    self.end_ptr is not None
                    and self.child_pointers[instance_id] == self.end_ptr
                ):
                    self._child_stop[instance_id] = True
                    break
                # Use buffer
                if self.buffer and self.child_pointers[instance_id] <= self.leading_ptr:
                    idx = self.child_pointers[instance_id] - self.slowest_ptr - 1
                    return_val = self.buffer[idx]
                else:  # Retrieve one element from main datapipe
                    self.leading_ptr = self.child_pointers[instance_id]
                    try:
                        return_val = next(self._datapipe_iterator)  # type: ignore[arg-type]
                        self.buffer.append(return_val)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self._datapipe_iterator = None
                        self.end_ptr = self.leading_ptr
                        continue
                if self.child_pointers[instance_id] == self.slowest_ptr + 1:
                    new_min = min(
                        self.child_pointers
                    )  # Can optimize by avoiding the call to min()
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                if (
                    self.buffer_size >= 0
                    and self.leading_ptr > self.buffer_size + self.slowest_ptr
                ):
                    raise BufferError(
                        "ForkerIterDataPipe buffer overflow,"
                        + f"buffer size {self.buffer_size} is insufficient."
                    )

                yield self.copy_fn(return_val)  # type: ignore[possibly-undefined]
        finally:
            self._child_stop[instance_id] = True
            # Cleanup _datapipe_iterator for the case that fork exits earlier
            if all(self._child_stop):
                self._datapipe_iterator = None
                self._cleanup()

    def is_every_instance_exhausted(self) -> bool:
        return self.end_ptr is not None and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        # pyrefly: ignore [bad-argument-type]
        return len(self.main_datapipe)

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def __getstate__(self):
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.copy_fn,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.copy_fn,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def _cleanup(self) -> None:
        while self.buffer:
            d = self.buffer.popleft()
            StreamWrapper.close_streams(d)

    def __del__(self) -> None:
        self._cleanup()


class _ChildDataPipe(IterDataPipe):
    r"""
    Iterable Datapipe that is a child of a main DataPipe.

    The instance of this class will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> # Single Iterator per IterDataPipe Invalidation
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        >>> # The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        >>> # The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        >>> # the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """

    _is_child_datapipe: bool = True

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int) -> None:
        if not isinstance(main_datapipe, _ContainerTemplate):
            raise AssertionError("main_datapipe must implement _ContainerTemplate")

        self.main_datapipe: IterDataPipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        # Note that the logic behind setting iterator ID and `reset` are handled within `hook_iterator`
        # We want to separate the code for reset and yield, so that 'reset' executes before __next__ is called
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)

    def __len__(self) -> int:
        return self.main_datapipe.get_length_by_instance(self.instance_id)

    # This method is called by `hook_iterator` in `_typing.py`.
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        r"""
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        # 1. First time any child iterator is created
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0  # type: ignore[attr-defined]
        # 2. This instance was already in the same generation as `main_datapipe`,
        #    we need to increment the ID further by 1
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:  # type: ignore[has-type]
            self.main_datapipe._valid_iterator_id += 1  # type: ignore[attr-defined]
            # Whenever a new generation of iterator is created, the `main_datapipe` must reset
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn(
                    "Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                    "the buffer and each child DataPipe will read from the start again.",
                    UserWarning,
                    stacklevel=2,
                )
            self.main_datapipe.reset()
        # 3. Otherwise, the iterator is behind the others, so it will just need to catch up by setting
        #    the instance's iterator to match that of `main_datapipe`
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    # This method is called by `hook_iterator` in `_typing.py`.
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        r"""Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""
        return (
            iterator_id == self._valid_iterator_id
            and iterator_id == self.main_datapipe._valid_iterator_id
        )


@functional_datapipe("demux")
class DemultiplexerIterDataPipe(IterDataPipe):
    r"""
    Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).

    A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(
        ...     num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True
        ... )
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """

    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        classifier_fn: Callable[[_T_co], int | None],
        drop_none: bool = False,
        buffer_size: int = 1000,
    ):
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instances` larger than 0, but {num_instances} is found"
            )

        _check_unpickable_fn(classifier_fn)

        # When num_instances == 1, demux can be replaced by filter,
        # but keep it as Demultiplexer for the sake of consistency
        # like throwing Error when classification result is out of range
        container = _DemultiplexerIterDataPipe(
            datapipe, num_instances, classifier_fn, drop_none, buffer_size
        )  # type: ignore[abstract]
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _DemultiplexerIterDataPipe(IterDataPipe, _ContainerTemplate):
    r"""
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        num_instances: int,
        classifier_fn: Callable[[_T_co], int | None],
        drop_none: bool,
        buffer_size: int,
    ) -> None:
        # pyrefly: ignore [invalid-type-var]
        self.main_datapipe = datapipe
        self._datapipe_iterator: Iterator[Any] | None = None
        self.num_instances = num_instances
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `demux`, "
                "please be aware of OOM at random places",
                UserWarning,
                stacklevel=2,
            )
        self.current_buffer_usage = 0
        # pyrefly: ignore [invalid-type-var]
        self.child_buffers: list[deque[_T_co]] = [deque() for _ in range(num_instances)]
        # pyrefly: ignore [invalid-type-var]
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.main_datapipe_exhausted = False
        self._child_stop: list[bool] = [True for _ in range(num_instances)]

    def _find_next(self, instance_id: int) -> _T_co:  # type: ignore[type-var]
        while True:
            if self.main_datapipe_exhausted or self._child_stop[instance_id]:
                raise StopIteration
            if self._datapipe_iterator is None:
                raise ValueError(
                    "_datapipe_iterator has not been set, likely because this private method is called directly "
                    "without invoking get_next_element_by_instance() first."
                )
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
                StreamWrapper.close_streams(value)
                continue
            if (
                classification is None
                or classification >= self.num_instances
                or classification < 0
            ):
                raise ValueError(
                    f"Output of the classification fn should be between 0 and {self.num_instances - 1}. "
                    + f"{classification} is returned."
                )
            if classification == instance_id:
                return value
            self.child_buffers[classification].append(value)
            self.current_buffer_usage += 1
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(
                    f"DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient."
                )

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = (
                _SnapshotState.Iterating
            )  # This is necessary for the DataPipe to reset properly.
            self.main_datapipe_exhausted = False
            for i in range(self.num_instances):
                self._child_stop[i] = False

        try:
            while not self._child_stop[instance_id]:
                if self.child_buffers[instance_id]:
                    self.current_buffer_usage -= 1
                    yield self.child_buffers[instance_id].popleft()
                else:
                    try:
                        yield self._find_next(instance_id)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self.main_datapipe_exhausted = True
                        self._datapipe_iterator = None
        finally:
            self._child_stop[instance_id] = True
            # Cleanup _datapipe_iterator for the case that demux exits earlier
            if all(self._child_stop):
                self._datapipe_iterator = None
            if self.child_buffers[instance_id]:
                self._cleanup(instance_id)

    def is_every_instance_exhausted(self) -> bool:
        return self.main_datapipe_exhausted and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        raise TypeError

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __getstate__(self):
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def _cleanup(self, instance_id: int | None = None) -> None:
        ids = (
            range(self.num_instances)
            if instance_id is None
            else [
                instance_id,
            ]
        )
        for i in ids:
            q = self.child_buffers[i]
            while q:
                d = q.popleft()
                StreamWrapper.close_streams(d)

    def __del__(self) -> None:
        self._cleanup()


@functional_datapipe("mux")
class MultiplexerIterDataPipe(IterDataPipe):
    r"""
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``).

    As in, one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = (
        ...     IterableWrapper(range(3)),
        ...     IterableWrapper(range(10, 15)),
        ...     IterableWrapper(range(20, 25)),
        ... )
        >>> list(dp1.mux(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
    """

    def __init__(self, *datapipes) -> None:
        self.datapipes = datapipes
        self.buffer: list = []  # Store values to be yielded only when every iterator provides one

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        while iterators:
            for it in iterators:
                try:
                    value = next(it)
                    self.buffer.append(value)
                except StopIteration:
                    self.buffer.clear()
                    return
            yield from self.buffer
            self.buffer.clear()

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return min(len(dp) for dp in self.datapipes) * len(self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def reset(self) -> None:
        self.buffer = []

    def __getstate__(self):
        state = (
            self.datapipes,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipes,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self.buffer = []

    def __del__(self) -> None:
        self.buffer.clear()


@functional_datapipe("zip")
class ZipperIterDataPipe(IterDataPipe[tuple[_T_co]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = (
        ...     IterableWrapper(range(5)),
        ...     IterableWrapper(range(10, 15)),
        ...     IterableWrapper(range(20, 25)),
        ... )
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """

    datapipes: tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe) -> None:
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` for `ZipIterDataPipe`."
            )
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator[tuple[_T_co]]:
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators, strict=False)

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            # pyrefly: ignore [bad-argument-type]
            return min(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
<|endoftext|>import copy
import warnings
from collections.abc import Iterable, Iterator, Sized
from typing import TypeVar

from torch.utils.data.datapipes.datapipe import IterDataPipe


_T = TypeVar("_T")

__all__ = ["IterableWrapperIterDataPipe"]


class IterableWrapperIterDataPipe(IterDataPipe[_T]):
    r"""
    Wraps an iterable object to create an IterDataPipe.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in ``iter()``.

    .. note::
        If ``deepcopy`` is explicitly set to ``False``, users should ensure
        that the data pipeline doesn't contain any in-place operations over
        the iterable instance to prevent data inconsistency across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, iterable: Iterable[_T], deepcopy: bool = True) -> None:
        self.iterable = iterable
        self.deepcopy = deepcopy

    def __iter__(self) -> Iterator[_T]:
        source_data = self.iterable
        if self.deepcopy:
            try:
                source_data = copy.deepcopy(self.iterable)
            # For the case that data cannot be deep-copied,
            # all in-place operations will affect iterable variable.
            # When this DataPipe is iterated second time, it will
            # yield modified items.
            except TypeError:
                warnings.warn(
                    "The input iterable can not be deepcopied, "
                    "please be aware of in-place modification would affect source data.",
                    stacklevel=2,
                )
        yield from source_data

    def __len__(self) -> int:
        if isinstance(self.iterable, Sized):
            return len(self.iterable)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
<|endoftext|>from torch.utils.data.datapipes.iter.callable import (
    CollatorIterDataPipe as Collator,
    MapperIterDataPipe as Mapper,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.iter.combining import (
    ConcaterIterDataPipe as Concater,
    DemultiplexerIterDataPipe as Demultiplexer,
    ForkerIterDataPipe as Forker,
    MultiplexerIterDataPipe as Multiplexer,
    ZipperIterDataPipe as Zipper,
)
from torch.utils.data.datapipes.iter.filelister import (
    FileListerIterDataPipe as FileLister,
)
from torch.utils.data.datapipes.iter.fileopener import (
    FileOpenerIterDataPipe as FileOpener,
)
from torch.utils.data.datapipes.iter.grouping import (
    BatcherIterDataPipe as Batcher,
    GrouperIterDataPipe as Grouper,
    UnBatcherIterDataPipe as UnBatcher,
)
from torch.utils.data.datapipes.iter.routeddecoder import (
    RoutedDecoderIterDataPipe as RoutedDecoder,
)
from torch.utils.data.datapipes.iter.selecting import FilterIterDataPipe as Filter
from torch.utils.data.datapipes.iter.sharding import (
    ShardingFilterIterDataPipe as ShardingFilter,
)
from torch.utils.data.datapipes.iter.streamreader import (
    StreamReaderIterDataPipe as StreamReader,
)
from torch.utils.data.datapipes.iter.utils import (
    IterableWrapperIterDataPipe as IterableWrapper,
)


__all__ = [
    "Batcher",
    "Collator",
    "Concater",
    "Demultiplexer",
    "FileLister",
    "FileOpener",
    "Filter",
    "Forker",
    "Grouper",
    "IterableWrapper",
    "Mapper",
    "Multiplexer",
    "RoutedDecoder",
    "Sampler",
    "ShardingFilter",
    "Shuffler",
    "StreamReader",
    "UnBatcher",
    "Zipper",
]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
<|endoftext|># mypy: allow-untyped-defs
from collections import defaultdict
from collections.abc import Callable, Iterator, Sized
from typing import Any, NoReturn, TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


__all__ = [
    "BatcherIterDataPipe",
    "GrouperIterDataPipe",
    "UnBatcherIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


def __getattr__(name: str) -> NoReturn:
    raise AttributeError(f"module {__name__} has no attribute {name}")


@functional_datapipe("batch")
class BatcherIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Creates mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
    last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
            defaults to ``DataChunk``

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> dp = dp.batch(batch_size=3, drop_last=True)
        >>> list(dp)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    datapipe: IterDataPipe
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        datapipe: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
        wrapper_class: type[DataChunk] = DataChunk,
    ) -> None:
        if batch_size <= 0:
            raise AssertionError("Batch size is required to be larger than 0!")
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __iter__(self) -> Iterator[DataChunk]:
        batch: list = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield self.wrapper_class(batch)
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield self.wrapper_class(batch)

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("unbatch")
class UnBatcherIterDataPipe(IterDataPipe):
    r"""
    Undoes batching of data (functional name: ``unbatch``).

    In other words, it flattens the data up to the specified level within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """

    def __init__(self, datapipe: IterDataPipe, unbatch_level: int = 1) -> None:
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            yield from self._dive(element, unbatch_level=self.unbatch_level)

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, (list, DataChunk)):
                for item in element:
                    yield from self._dive(item, unbatch_level=-1)
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if isinstance(element, (list, DataChunk)):
                for item in element:
                    yield from self._dive(item, unbatch_level=unbatch_level - 1)
            else:
                raise IndexError(
                    f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe"
                )


@functional_datapipe("groupby")
class GrouperIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Groups data from IterDataPipe by keys from ``group_key_fn``, yielding a ``DataChunk`` with batch size up to ``group_size``.

    (functional name: ``groupby``).

    The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
    will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
    the DataPipe will yield the largest batch with the same key, provided that its size is larger
    than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.

    After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
    will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.

    Args:
        datapipe: Iterable datapipe to be grouped
        group_key_fn: Function used to generate group key from the data of the source datapipe
        keep_key: Option to yield the matching key along with the items in a tuple,
            resulting in `(key, [items])` otherwise returning [items]
        buffer_size: The size of buffer for ungrouped data
        group_size: The max size of each group, a batch is yielded as soon as it reaches this size
        guaranteed_group_size: The guaranteed minimum group size to be yielded in case the buffer is full
        drop_remaining: Specifies if the group smaller than ``guaranteed_group_size`` will be dropped from buffer
            when the buffer is full

    Example:
        >>> import os
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def group_fn(file):
        ...     return os.path.basename(file).split(".")[0]
        >>> source_dp = IterableWrapper(
        ...     ["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"]
        ... )
        >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
        >>> list(dp0)
        [['a.png', 'a.json', 'a.jpg'], ['b.png', 'b.json'], ['c.json']]
        >>> # A group is yielded as soon as its size equals to `group_size`
        >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
        >>> list(dp1)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
        >>> # Scenario where `buffer` is full, and group 'a' needs to be yielded since its size > `guaranteed_group_size`
        >>> dp2 = source_dp.groupby(
        ...     group_key_fn=group_fn,
        ...     buffer_size=3,
        ...     group_size=3,
        ...     guaranteed_group_size=2,
        ... )
        >>> list(dp2)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
    """

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        group_key_fn: Callable[[_T_co], Any],
        *,
        keep_key: bool = False,
        buffer_size: int = 10000,
        group_size: int | None = None,
        guaranteed_group_size: int | None = None,
        drop_remaining: bool = False,
    ) -> None:
        _check_unpickable_fn(group_key_fn)
        # pyrefly: ignore [invalid-type-var]
        self.datapipe = datapipe
        # pyrefly: ignore [invalid-type-var]
        self.group_key_fn = group_key_fn

        self.keep_key = keep_key
        self.max_buffer_size = buffer_size
        self.buffer_elements: defaultdict[Any, list] = defaultdict(list)
        self.curr_buffer_size = 0
        self.group_size = group_size
        self.guaranteed_group_size = None
        if group_size is not None and buffer_size is not None:
            if not (0 < group_size <= buffer_size):
                raise AssertionError("group_size must be > 0 and <= buffer_size")
            self.guaranteed_group_size = group_size
        if guaranteed_group_size is not None:
            if group_size is None or not (0 < guaranteed_group_size <= group_size):
                raise AssertionError(
                    "guaranteed_group_size must be > 0 and <= group_size and group_size must be set"
                )
            self.guaranteed_group_size = guaranteed_group_size
        self.drop_remaining = drop_remaining
        self.wrapper_class = DataChunk

    def _remove_biggest_key(self):
        biggest_key = None
        biggest_size = 0
        result_to_yield = None
        for findkey in self.buffer_elements:
            if len(self.buffer_elements[findkey]) > biggest_size:
                biggest_size = len(self.buffer_elements[findkey])
                biggest_key = findkey

        if (
            self.guaranteed_group_size is not None
            and biggest_size < self.guaranteed_group_size
            and not self.drop_remaining
        ):
            raise RuntimeError(
                "Failed to group items", str(self.buffer_elements[biggest_key])
            )

        if (
            self.guaranteed_group_size is None
            or biggest_size >= self.guaranteed_group_size
        ):
            result_to_yield = self.buffer_elements[biggest_key]

        self.curr_buffer_size -= biggest_size
        del self.buffer_elements[biggest_key]

        return result_to_yield

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            if self.group_size is not None and self.group_size == len(
                self.buffer_elements[key]
            ):
                result: DataChunk[Any] = self.wrapper_class(self.buffer_elements[key])
                yield (key, result) if self.keep_key else result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield (key, result) if self.keep_key else result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield (key, result) if self.keep_key else result

    def reset(self) -> None:
        self.curr_buffer_size = 0
        self.buffer_elements = defaultdict(list)

    def __getstate__(self):
        state = (
            self.datapipe,
            self.group_key_fn,
            self.keep_key,
            self.max_buffer_size,
            self.group_size,
            self.guaranteed_group_size,
            self.drop_remaining,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.group_key_fn,
            self.keep_key,
            self.max_buffer_size,
            self.group_size,
            self.guaranteed_group_size,
            self.drop_remaining,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self.curr_buffer_size = 0
        self.buffer_elements = defaultdict(list)

    def __del__(self) -> None:
        self.buffer_elements.clear()
<|endoftext|>from collections.abc import Callable, Iterable, Iterator, Sized
from io import BufferedIOBase
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _deprecation_warning
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
    Decoder,
    extension_extract_fn,
    imagehandler as decoder_imagehandler,
)


__all__ = ["RoutedDecoderIterDataPipe"]


@functional_datapipe("routed_decode")
class RoutedDecoderIterDataPipe(IterDataPipe[tuple[str, Any]]):
    r"""
    Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.

    (functional name: ``routed_decode``)

    Args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
            handlers will be set as default. If multiple handlers are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname

    Note:
        When ``key_fn`` is specified returning anything other than extension, the default
        handler will not work and users need to specify custom handler. Custom handler
        could use regex to determine the eligibility to handle data.
    """

    def __init__(
        self,
        datapipe: Iterable[tuple[str, BufferedIOBase]],
        *handlers: Callable,
        key_fn: Callable = extension_extract_fn,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[tuple[str, BufferedIOBase]] = datapipe
        if not handlers:
            handlers = (decoder_basichandlers, decoder_imagehandler("torch"))
        self.decoder = Decoder(*handlers, key_fn=key_fn)
        _deprecation_warning(
            type(self).__name__,
            deprecation_version="1.12",
            removal_version="1.13",
            old_functional_name="routed_decode",
        )

    def add_handler(self, *handler: Callable) -> None:
        self.decoder.add_handler(*handler)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Sized
from enum import IntEnum
from typing import NoReturn

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = [
    "SHARDING_PRIORITIES",
    "ShardingFilterIterDataPipe",
]


class SHARDING_PRIORITIES(IntEnum):
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3


class _ShardingIterDataPipe(IterDataPipe):
    def apply_sharding(
        self,
        num_of_instances: int,
        instance_id: int,
        sharding_group: SHARDING_PRIORITIES,
    ) -> NoReturn:
        raise NotImplementedError


@functional_datapipe("sharding_filter")
class ShardingFilterIterDataPipe(_ShardingIterDataPipe):
    r"""
    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).

    After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
    """

    def __init__(
        self, source_datapipe: IterDataPipe, sharding_group_filter=None
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.sharding_group_filter = sharding_group_filter
        self.groups: dict[int, tuple[int, int]] = {}
        self.num_of_instances = 1
        self.instance_id = 0
        self._update_num_of_instances()

    def apply_sharding(
        self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT
    ):
        if instance_id >= num_of_instances:
            raise ValueError(
                f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})"
            )
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()

    def _update_num_of_instances(self) -> None:
        sorted_sharding_groups = [
            self.groups[key]
            for key in sorted(self.groups.keys())
            if self.sharding_group_filter is None or key == self.sharding_group_filter
        ]

        sorted_sharding_groups.reverse()

        self.num_of_instances = 1
        self.instance_id = 0

        for group_num_of_instances, group_instance_id in sorted_sharding_groups:
            self.instance_id += self.num_of_instances * group_instance_id
            self.num_of_instances *= group_num_of_instances

    def __iter__(self):
        for i, item in enumerate(self.source_datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self) -> int:
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe) // self.num_of_instances + (
                1
                if (
                    self.instance_id < len(self.source_datapipe) % self.num_of_instances
                )
                else 0
            )
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
<|endoftext|>from collections.abc import Iterable, Iterator
