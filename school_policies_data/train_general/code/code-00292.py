                            f_evt.time_range.end - f_evt.time_range.start,
                        )
                    elif f_evt.device_type == DeviceType.CPU:
                        # make sure that 'thread' of a CPU Kineto (e.g. Device Runtime) event is associated
                        # with the 'thread' of the corresponding linked PyTorch event to properly track
                        # parents and children
                        f_evt.thread = fe.thread

        def _create_function_event_for_memory_events(evt):
            rel_start_ns = evt.start_ns() - trace_start_ns
            fe = FunctionEvent(
                id=max_evt_id,
                name=evt.name(),
                overload_name="",
                trace_name=None,  # not outputting in the trace
                thread=evt.start_thread_id(),
                start_us=rel_start_ns / 1000,
                end_us=rel_start_ns / 1000,  # no duration
                fwd_thread=evt.start_thread_id(),
                input_shapes=[],
                stack=[],
                scope=0,  # RecordScope::FUNCTION
                use_device=self.use_device,
                cpu_memory_usage=_cpu_memory_usage(evt),
                device_memory_usage=_device_memory_usage(evt),
                is_async=False,
                sequence_nr=-1,
                device_type=DeviceType.CPU,
                device_index=0,
            )
            return fe

        # output top-level memory events
        for mem_record in mem_records:
            if _check_timeout():
                break

            if not mem_record[1]:
                max_evt_id += 1
                fe = _create_function_event_for_memory_events(mem_record[0])
                all_function_events.append(fe)

        for oom_record in oom_records:
            if _check_timeout():
                break

            max_evt_id += 1
            fe = _create_function_event_for_memory_events(oom_record)
            all_function_events.append(fe)

        if timed_out:
            log.warning(
                "Profiler _parse_kineto_results timed out after %.3f seconds, "
                "returning partial results with %d events",
                timeout_s,
                len(all_function_events),
            )

        all_function_events.sort(
            key=lambda evt: [evt.time_range.start, -evt.time_range.end]
        )
        return all_function_events


# Set by torch.profiler to the active cupti_monitor ProfilerObserver while a session is
# running (None otherwise). record_function routes regions to it via push/pop_annotation.
# Held as an opaque object -- NOT imported from the cupti package -- so record_function never
# pulls in the cupti chain on a non-cupti run, and there is a single "is a session active"
# signal (this reference) rather than a separate flag.
_active_cupti_profiler_observer: Any = None


def _set_active_cupti_profiler_observer(observer: Any) -> None:
    global _active_cupti_profiler_observer
    _active_cupti_profiler_observer = observer


class record_function(_ContextDecorator):  # pyrefly: ignore [invalid-inheritance]
    """Context manager/function decorator that adds a label to a code block/function when running autograd profiler.
    Label will only appear if CPU activity tracing is enabled.

    It is useful when tracing the code profile.

    Args:
        name (str): Label assigned to the block of code.
        node_id (int): ID of node, for distributed profiling. Unset in
        non-distributed cases.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x**2
        ...     with torch.autograd.profiler.record_function(
        ...         "label-z"
        ...     ):  # label the block
        ...         z = y**3
        ...     y.backward()
        >>> # xdoctest: +IGNORE_WANT
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us

    """

    def __init__(self, name: str, args: str | None = None):
        self.name: str = name
        self.args: str | None = args
        # Whether or not we should run record function's end callbacks when exiting.
        self.run_callbacks_on_exit: bool = True
        # TODO: TorchScript ignores standard type annotation here
        # self.record: Optional["torch.classes.profiler._RecordFunction"] = None
        self.record = torch.jit.annotate(
            # pyrefly: ignore [not-a-type]
            Optional["torch.classes.profiler._RecordFunction"],
            None,
        )
        self._cupti_monitor_external_id: int | None = None

    def __enter__(self):
        self.record = torch.ops.profiler._record_function_enter_new(
            self.name, self.args
        )
        # Route the region to the active cupti_monitor observer, if any. The reference is
        # None unless a cupti_monitor profile is running, so a non-cupti run never touches
        # the cupti chain. Guarded by is_scripting() (the global access doesn't compile under
        # TorchScript), and the global is read inside the guard so it is dead-code-eliminated.
        if not torch.jit.is_scripting():
            observer = _active_cupti_profiler_observer
            if observer is not None:
                self._cupti_monitor_external_id = observer.push_annotation(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if not torch.jit.is_scripting():
            if self._cupti_monitor_external_id is not None:
                observer = _active_cupti_profiler_observer
                if observer is not None:
                    observer.pop_annotation()
                self._cupti_monitor_external_id = None
        if not self.run_callbacks_on_exit:
            return

        # Local variable is needed by TorchScript to refine Optional[T] to T
        record = self.record
        if record is None:
            raise AssertionError("Expected record to be set")

        # TODO: Too slow with __torch_function__ handling enabled
        # See https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                torch.ops.profiler._record_function_exit._RecordFunction(record)
        else:
            torch.ops.profiler._record_function_exit(record)

    def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
        """Use for profiling async calls that return a future.

        Calling this function will extend recording beyond this scope, until the future is
        satisfied. It is useful for profiling the end to end time of asynchronous calls.
        This function should only be called once to attach the callback onto the future, and
        will throw if called multiple times.

        Args:
            fut: (torch._C.Future): future for which to schedule
            callback for.

        Returns:
            A future that completes with the value of the passed in future when
            the profiling callbacks have ran.

        """
        # Throw if we have already attached a callback onto the future.
        if not self.run_callbacks_on_exit:
            raise RuntimeError("_call_end_callbacks_on_future can only be called once.")

        # We are scheduling to run this RecordFunction's end callbacks when the
        # passed in future completes, so don't run end callbacks on exit.
        self.run_callbacks_on_exit = False

        # Local variable is needed by TorchScript to refine Optional[T] to T
        record = self.record
        if record is None:
            raise AssertionError("Expected record to be set")

        # TODO: Too slow with __torch_function__ handling enabled
        # See https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                profiled_future = (
                    torch.ops.profiler._call_end_callbacks_on_jit_fut._RecordFunction(
                        record, fut
                    )
                )
        else:
            profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(
                record, fut
            )
        return profiled_future


class emit_itt:
    """Context manager that makes every autograd operation emit an ITT range.

    It is useful when running the program under Intel(R) VTune Profiler::

        vtune <--vtune-flags> <regular command here>

    The Instrumentation and Tracing Technology (ITT) API enables your application to generate and
    control the collection of trace data during its execution across different Intel tools.
    This context manager is to annotate Intel(R) VTune Profiling trace. With help of this context manager,
    you will be able to see labeled ranges in Intel(R) VTune Profiler GUI.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args:
        enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
            Default: ``True``.
        record_shapes (bool, optional): If ``record_shapes=True``, the itt range wrapping
            each autograd op will append information about the sizes of Tensor arguments received
            by that op, in the following format:
            ``[[arg0.size(0), arg0.size(1), ...], [arg1.size(0), arg1.size(1), ...], ...]``
            Non-tensor arguments will be represented by ``[]``.
            Arguments will be listed in the order they are received by the backend op.
            Please note that this order may not match the order in which those arguments were passed
            on the Python side.  Also note that shape recording may increase the overhead of itt range creation.
            Default: ``False``

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.autograd.profiler.emit_itt():
        ...     model(x)

    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("ITT annotation context manager is not reentrant")
        self.entered = True
        _run_on_profiler_start()
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.ITT,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        _disable_profiler()
        _run_on_profiler_stop()
        return False


class emit_nvtx:
    """Context manager that makes every autograd operation emit an NVTX range.

    It is useful when running the program under nvprof::

        nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

    Unfortunately, there's no way to force nvprof to flush the data it collected
    to disk, so for CUDA profiling one has to use this context manager to annotate
    nvprof traces and wait for the process to exit before inspecting them.
    Then, either NVIDIA Visual Profiler (nvvp) can be used to visualize the timeline, or
    :func:`torch.autograd.profiler.load_nvprof` can load the results for inspection
    e.g. in Python REPL.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args:
        enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
            Default: ``True``.
        record_shapes (bool, optional): If ``record_shapes=True``, the nvtx range wrapping
            each autograd op will append information about the sizes of Tensor arguments received
            by that op, in the following format:
            ``[[arg0.size(0), arg0.size(1), ...], [arg1.size(0), arg1.size(1), ...], ...]``
            Non-tensor arguments will be represented by ``[]``.
            Arguments will be listed in the order they are received by the backend op.
            Please note that this order may not match the order in which those arguments were passed
            on the Python side.  Also note that shape recording may increase the overhead of nvtx range creation.
            Default: ``False``

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.cuda.profiler.profile():
        ...     model(x)  # Warmup CUDA memory allocator and profiler
        ...     with torch.autograd.profiler.emit_nvtx():
        ...         model(x)

    **Forward-backward correlation**

    When viewing a profile created using :class:`emit_nvtx` in the Nvidia Visual Profiler,
    correlating each backward-pass op with the corresponding forward-pass op can be difficult.
    To ease this task, :class:`emit_nvtx` appends sequence number information to the ranges it
    generates.

    During the forward pass, each function range is decorated with ``seq=<N>``.  ``seq`` is a running
    counter, incremented each time a new backward Function object is created and stashed for backward.
    Thus, the ``seq=<N>`` annotation associated with each forward function range tells you that
    if a backward Function object is created by this forward function,
    the backward object will receive sequence number N.
    During the backward pass, the top-level range wrapping each C++ backward Function's
    ``apply()`` call is decorated with ``stashed seq=<M>``.  ``M`` is the sequence number that
    the backward object was created with.  By comparing ``stashed seq`` numbers in backward with ``seq``
    numbers in forward, you can track down which forward op created each backward Function.

    Any functions executed during the backward pass are also decorated with ``seq=<N>``.  During
    default backward (with ``create_graph=False``) this information is irrelevant, and in fact,
    ``N`` may simply be 0 for all such functions.  Only the top-level ranges associated with
    backward Function objects' ``apply()`` methods are useful, as a way to correlate these Function
    objects with the earlier forward pass.

    **Double-backward**

    If, on the other hand, a backward pass with ``create_graph=True`` is underway (in other words,
    if you are setting up for a double-backward), each function's execution during backward
    is given a nonzero, useful ``seq=<N>``.  Those functions may themselves create Function objects
    to be executed later during double-backward, just as the original functions in the forward pass did.
    The relationship between backward and double-backward is conceptually the same as the relationship
    between forward and backward: The functions still emit current-sequence-number-tagged ranges,
    the Function objects they create still stash those sequence numbers, and during the eventual
    double-backward, the Function objects' ``apply()`` ranges are still tagged with ``stashed seq``
    numbers, which can be compared to `seq` numbers from the backward pass.

    .. warning:
        The sequence number is thread-local, and some forward functions don't create an associated
        backward Function object (instead delegating that to sub-functions further down the call chain).
        For these reasons, the correspondence of stashed sequence numbers in
        backward Function ``apply()`` ranges with `seq` numbers in forward-pass ranges is
        not guaranteed to be 1 to 1.  The sequence numbers alone may not be enough to fully
        disambiguate which forward function created which
        backward Function object.  You may need to make a judgment based on analytic knowledge of what
        the expected correspondence should be.
    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("NVTX annotation context manager is not reentrant")
        self.entered = True
        torch.cuda.synchronize()
        _run_on_profiler_start()
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.NVTX,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        _disable_profiler()
        _run_on_profiler_stop()
        return False


def load_nvprof(path):
    """Open an nvprof trace file and parses autograd annotations.

    Args:
        path (str): path to nvprof trace
    """
    return EventList(parse_nvprof_trace(path))


class EnforceUnique:
    """Raises an error if a key is seen more than once."""

    def __init__(self):
        self.seen = set()

    def see(self, *key):
        r"""
        Observe a key and raise an error if it is seen multiple times.
        """
        if key in self.seen:
            raise RuntimeError("duplicate key: " + str(key))
        self.seen.add(key)


def parse_nvprof_trace(path):
    import sqlite3

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Parse strings table
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = torch._C._demangle(r["value"])

    # First, find all functions and create FunctionEvents for them
    marker_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp AS start_time, end.timestamp AS end_time
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
        ON start.id = end.id
    WHERE
        start.name != 0 AND end.name = 0
    """
    functions = []
    functions_map = {}
    unique = EnforceUnique()
    for row in conn.execute(marker_query):
        unique.see(row["marker_id"])
        evt = FunctionEvent(
            id=row["marker_id"],
            node_id=0,  # missing a node_id when calling FunctionEvent. This is just to ensure
            # that pytorch doesn't crash when creating a FunctionEvent() object
            name=strings[row["name"]],
            start_us=row["start_time"],
            end_us=row["end_time"],
            thread=0,
        )  # TODO: find in sqlite database
        functions.append(evt)
        functions_map[evt.id] = evt

    # Now, correlate all kernels with FunctionEvents
    kernel_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp, end.timestamp,
        runtime._id_ AS runtime_id, runtime.cbid, runtime.start AS runtime_start, runtime.end AS runtime_end,
        kernel.start AS kernel_start, kernel.end AS kernel_end, kernel.name AS kernel_name
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start
        INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
            ON start.id = end.id
        INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME as runtime
            ON (start.timestamp < runtime.start AND runtime.end < end.timestamp)
        INNER JOIN CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS kernel
            ON kernel.correlationId = runtime.correlationId
    """
    unique = EnforceUnique()
    for row in conn.execute(kernel_query):
        unique.see(row["marker_id"], row["runtime_id"])
        # 211 is cudaKernelLaunch for cuda >= 9.2
        if row["cbid"] != 211:
            raise AssertionError(f"Expected cbid to be 211, but got {row['cbid']}")
        evt = functions_map[row["marker_id"]]
        evt.append_kernel(
            row["kernel_name"], 0, row["kernel_end"] - row["kernel_start"]
        )

    functions.sort(key=lambda evt: evt.time_range.start)
    return functions


class KinetoStepTracker:
    """Provides an abstraction for incrementing the step count globally.

    Previously, we only had one place to mark that a step() has occurred
    in the program via pytorch profiler step(). We will now add step hooks
    in the Optimizer class https://github.com/pytorch/pytorch/issues/88446

    - This could mean programs that already call profiler.step() every
      iteration can end up double incrementing step count.
    - If a model uses multiple optimizers we can also have double or more
      counting of the step.

    We fix this by adding a layer of abstraction before calling step()
    to the kineto library. The idea is to maintain steps per requester in a dict:

    .. code-block::

        {
           "ProfilerStep": 100,  # triggered by profiler step() call
           "Optimizer1Step": 100,   # Optimizer 1 or 2 are just examples, could be SGD, Adam etc
           "Optimizer2Step": 100,
        }

    To figure out the global step count just take the max of dict values (100).

    If one of the count increments the max will go up.

    .. code-block::

        {
           "ProfilerStep": 100,
           "Optimizer1Step": 101,   # Optimizer1 got incremented first say
           "Optimizer2Step": 100,
        }

    Then global step count is 101
    We only call the kineto step() function when global count increments.

    NOTE: Please do not use the KinetoStepTracker in modules beside the Optimizer
    for now. The result could be incorrect increments of the step count.
    """

    _current_step = 0
    _step_dict: dict[str, int] = defaultdict(int)

    @classmethod
    def init_step_count(cls, requester: str):
        r"""
        Initialize for a given requester.
        """
        cls._step_dict[requester] = cls._current_step

    @classmethod
    def erase_step_count(cls, requester: str) -> bool:
        r"""
        Remove a given requester.
        """
        return cls._step_dict.pop(requester, None) is not None

    @classmethod
    def increment_step(cls, requester: str) -> int:
        """Increments the step count for the requester.

        Additionally if the max over all step counts has incremented then
        trigger the _kineto_step() returns global step count
        """
        if requester not in cls._step_dict:
            cls.init_step_count(requester)
        cls._step_dict[requester] += 1

        new_step = max(cls._step_dict.values())
        if new_step > cls._current_step:
            delta = new_step - cls._current_step
            if delta > 1:
                warn(
                    "Profiler step count has increased more than 1 - "
                    f"current_step = {cls._current_step} step dict =  {cls._step_dict}",
                    stacklevel=2,
                )
            for _ in range(delta):
                _kineto_step()
            cls._current_step = new_step
        return cls._current_step

    @classmethod
    def current_step(cls) -> int:
        r"""
        Get the latest step for any requester
        """
        return cls._current_step
<|endoftext|># mypy: allow-untyped-defs
r"""Autograd anomaly mode."""

import warnings

import torch


__all__ = ["detect_anomaly", "set_detect_anomaly"]


class detect_anomaly:
    r"""Context-manager that enables anomaly detection for the autograd engine.

    This does two things:

    - Running the forward pass with detection enabled will allow the backward
      pass to print the traceback of the forward operation that created the failing
      backward function.
    - If ``check_nan`` is ``True``, any backward computation that generates "nan"
      value will raise an error. Default ``True``.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ANOMALY)
        >>> import torch
        >>> from torch import autograd
        >>> class MyFunc(autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, inp):
        ...         return inp.clone()
        ...
        ...     @staticmethod
        ...     def backward(ctx, gO):
        ...         # Error during the backward pass
        ...         raise RuntimeError("Some error in backward")
        ...         return gO.clone()
        >>> def run_fn(a):
        ...     out = MyFunc.apply(a)
        ...     return out.sum()
        >>> inp = torch.rand(10, 10, requires_grad=True)
        >>> out = run_fn(inp)
        >>> out.backward()
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward
        >>> with autograd.detect_anomaly():
        ...     inp = torch.rand(10, 10, requires_grad=True)
        ...     out = run_fn(inp)
        ...     out.backward()
            Traceback of forward call that caused the error:
              File "tmp.py", line 53, in <module>
                out = run_fn(inp)
              File "tmp.py", line 44, in run_fn
                out = MyFunc.apply(a)
            Traceback (most recent call last):
              File "<stdin>", line 4, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward

    """

    def __init__(self, check_nan=True) -> None:
        self.prev = torch.is_anomaly_enabled()
        self.check_nan = check_nan
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        warnings.warn(
            "Anomaly Detection has been enabled. "
            "This mode will increase the runtime "
            "and should only be enabled for debugging.",
            stacklevel=2,
        )

    def __enter__(self) -> None:
        torch.set_anomaly_enabled(True, self.check_nan)

    def __exit__(self, *args: object) -> None:
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)


class set_detect_anomaly:
    r"""Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
        check_nan (bool): Flag whether to raise an error when the backward
                          generate "nan"

    """

    def __init__(self, mode: bool, check_nan: bool = True) -> None:
        self.prev = torch.is_anomaly_enabled()
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        torch.set_anomaly_enabled(mode, check_nan)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)
<|endoftext|>import abc
import contextlib
import contextvars
import functools
import logging
import threading
from collections import defaultdict, deque
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    MutableMapping,
    Sequence,
)
from typing import (
    Any,
    cast,
    Literal,
    NamedTuple,
    Optional,
    TYPE_CHECKING,
    TypeAlias,
    Union,
)
from weakref import WeakKeyDictionary, WeakValueDictionary

import torch
from torch.autograd.variable import Variable
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle


if TYPE_CHECKING:
    from torch._ops import OpOverload


__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "node_creation_hook",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
    "Node",
    "GradientEdge",
    "get_gradient_edge",
    "increment_version",
    "region_activation_memory_budget",
    "set_warn_on_accumulate_grad_stream_mismatch",
    "set_override_stale_capture_stream",
]


log = logging.getLogger(__name__)


class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        r"""Return the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def next_functions(self) -> tuple[tuple[Optional["Node"], int], ...]:
        r"""Return the edges from this node to its input functions.

        Each entry is a ``(Node, int)`` pair. The node is ``None`` for an input
        that does not require gradients. The integer is the output index of the
        input function to which this edge connects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> dict:
        r"""Return the metadata."""
        raise NotImplementedError

    @abc.abstractmethod
    def _sequence_nr(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _input_metadata(self) -> list[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_inputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        .. note::
            In the rare case where the hook is registered while the Node has already
            begun execution, there is no longer any guarantee on :attr:`grad_outputs`
            content (it might be as usual or empty depending on other factors). The
            hook can still optionally return a new gradient to be used in place of
            :attr:`grad_inputs` independent of :attr:`grad_outputs`.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove() # Removes the hook
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if cls is Node and (
            (
                subclass is not None
                and subclass is getattr(torch._C._functions, subclass.__name__, None)
            )
            or issubclass(subclass, torch.autograd.function.BackwardCFunction)
        ):
            return True
        return NotImplemented


def _get_grad_fn_or_grad_acc(t: Union[torch.Tensor, "GradientEdge"]) -> Node:
    if isinstance(t, GradientEdge):
        return t.node
    if t.requires_grad and t.grad_fn is None:
        with torch.enable_grad():
            node = t.view_as(t).grad_fn.next_functions[0][0]  # type: ignore[union-attr]
    else:
        node = t.grad_fn
    if node is None:
        raise AssertionError("Expected gradient function to be set")
    return node


class GradientEdge(NamedTuple):
    """Object representing a given gradient edge within the autograd graph.

    To get the gradient edge where a given Tensor gradient will be computed,
    you can do ``edge = autograd.graph.get_gradient_edge(tensor)``.
    """

    node: Node
    output_nr: int
    # This token can be used to ensure the graph stays alive when it cannot be
    # done via the node field
    ownership_token: Node | None = None


def get_gradient_edge(tensor: torch.Tensor) -> GradientEdge:
    """Get the gradient edge for computing the gradient of the given Tensor.

    In particular, it is equivalent to call
    ``g = autograd.grad(loss, input)`` and ``g = autograd.grad(loss, get_gradient_edge(input))``.
    """
    if not tensor.requires_grad:
        raise RuntimeError(
            "It is not possible to get the gradient edge for a Tensor "
            "that does not require gradients",
        )
    grad_fn = _get_grad_fn_or_grad_acc(tensor)

    # Python-based Node are owned by the C++ side meaning the python grad_fn
    # object we hold here does NOT keep the C++ graph alive.
    # Create an ownership token by creating a new C++ node that own the graph
    # we care about here.
    token = None
    if isinstance(grad_fn, torch._C._FunctionBase):
        with torch.enable_grad():
            token = tensor.view_as(tensor).grad_fn

    # Note that output_nr default to 0 which is the right value
    # for the AccumulateGrad node.
    # pyrefly: ignore [bad-argument-type]
    return GradientEdge(grad_fn, tensor.output_nr, ownership_token=token)


def increment_version(tensor: torch.Tensor | Iterable[torch.Tensor]) -> None:
    """Update autograd metadata tracking whether the given Tensor was modified in place.

    This is to enable more accurate error checking within the autograd engine.
    It is already done automatically by PyTorch functions and within custom Function
    when mark_dirty() is called appropriately so you only need to call this explicitly
    if you are doing inplace operation on the Tensor data in a way that Pytorch doesn't
    know about. For example a custom kernel that reads the Tensor data_ptr and modifies
    the memory inplace based on this pointer. Can accept either a tensor, or a list of tensors.

    Note that incrementing the version counter multiple times for a single inplace operation
    is not problematic.

    Note that if you pass in tensor constructed under torch.inference_mode(),
    we will not bump its version counter (because your tensor does not have one).
    """
    if isinstance(tensor, torch.Tensor):
        tensor = (tensor,)
    torch._C._increment_version(tensor)


class saved_tensors_hooks:
    """Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called every time an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`torch.Tensor.backward()` or
    :func:`torch.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x.detach()
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.

    .. warning ::
        To avoid reference cycle, the return value of ``pack_hook`` cannot hold a
        reference to the input tensor. For example, use `lambda x: x.detach()`
        instead of `lambda x: x` as the pack hook.
    """

    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor],
    ) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_saved_tensors_default_hooks()


class save_on_cpu(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn't fit in GPU memory during training).

    .. warning::

        When ``pin_memory=True``, the GPU to CPU copy during packing is
        asynchronous. Accessing saved tensors on CPU (e.g. via
        ``grad_fn._saved_self``) before the CUDA stream has finished may
        yield incorrect data. Call :func:`torch.cuda.synchronize` first
        if you need to read them.

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during both
                           packing and unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
        >>> # all intermediary tensors are released (deleted) after the call to backward
    """

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            is_pinnable = device_module.is_available() and not tensor.is_sparse
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=is_pinnable,
            )
            packed.copy_(tensor, non_blocking=is_pinnable)
            return (tensor.device, packed)

        def unpack_from_cpu(packed: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message: str) -> Generator[None, None, None]:
    """Context-manager that disables the saved tensors default hooks feature.

    Useful for if you are creating a feature that does not work with saved
    tensors default hooks.

    Args:
        error_message (str): When saved tensors default hooks are used when they
                             have been disabled, a RuntimeError with this
                             error message gets raised.

    Example::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with torch.autograd.graph.disable_saved_tensors_hooks(message):
        ...     # Raises RuntimeError: saved tensors default hooks are disabled
        ...     with torch.autograd.graph.save_on_cpu():
        ...         pass
    """
    maybe_prev_message = None
    try:
        maybe_prev_message = (
            torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        )
        torch._C._autograd._saved_tensors_hooks_disable(error_message)
        yield
    finally:
        # See NOTE: [disabled_error_message invariant]
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(maybe_prev_message)


class node_creation_hook:
    """Context-manager that registers a hook called on each autograd Node created within it.

    In that context, ``hook`` is called once for every autograd graph node
    created by operations on tensors that require grad, with the freshly
    created :class:`torch.autograd.graph.Node` as its only argument. The
    intended use is to record the node, stash entries in ``node.metadata``,
    or register backward hooks on it via
    :meth:`~torch.autograd.graph.Node.register_hook` and
    :meth:`~torch.autograd.graph.Node.register_prehook`.

    The node is passed to the hook only once it is fully populated: its
    ``next_functions`` are wired, all outputs' metadata are bound, and the
    tensors saved for backward (``_saved_*``) have been stored, so a hook
    that inspects the node sees its complete state.

    The hook should have the following signature::

        hook(node: torch.autograd.graph.Node) -> None

    The registration is thread-local and propagates like other autograd
    thread-local state: it is active on autograd engine worker threads, so
    nodes created during backward (e.g. with ``create_graph=True`` or inside
    checkpoint recomputation) also fire the hook.

    When nesting this context-manager, every active hook is called for each
    node, in registration order (outermost context-manager first). Creating
    a new autograd node from inside a hook raises an error; hooks must only
    observe the node they are given.

    One motivating use case is attributing work done during backward to the
    forward region that created the graph, by capturing state at node
    creation time and restoring it around the node's backward execution::

        >>> # xdoctest: +SKIP
        >>> def creation_hook(node):
        ...     # ``current_region()``/``enter_region()`` are user-defined and
        ...     # stand in for whatever thread-local state you want to restore
        ...     # while this node runs in backward.
        ...     region = current_region()
        ...     node.register_prehook(lambda gO: enter_region(region))
        ...     node.register_hook(lambda gI, gO: enter_region(None))
        >>>
        >>> with torch.autograd.graph.node_creation_hook(creation_hook):
        ...     loss = model(inputs)

    Example::

        >>> a = torch.ones(5, requires_grad=True)
        >>> with torch.autograd.graph.node_creation_hook(lambda node: print(node)):
        ...     b = a * 2
        <AccumulateGrad object at ...>
        <MulBackward0 object at ...>

    .. note::
        An ``AccumulateGrad`` node fires this hook when it is created, but
        not when a previously created one is reused. The node is created on
        demand the first time a leaf tensor is wired into a graph and is then
        cached (via a weak reference) on the leaf, so it fires again only
        after the old node has been freed. Since freeing the autograd graph
        drops that node, code that frees the graph each iteration (the common
        case) fires the hook consistently on each leaf's first use within the
        context.
    """

    def __init__(self, hook: Callable[[Node], None]) -> None:
        self.hook = hook

    def __enter__(self) -> None:
        torch._C._autograd._push_node_creation_hook(self.hook)

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_node_creation_hook()


def region_activation_memory_budget(
    budget: float,
) -> contextlib.AbstractContextManager[None]:
    r"""Context-manager that sets the activation memory budget for the region of
    a compiled forward traced under it.

    .. warning::
        This is a prototype feature and is subject to change.

    Under :func:`torch.compile`, the min-cut partitioner chooses which
    activations to save versus recompute in the backward pass to stay under a
    memory budget. ``budget`` is a ratio in ``[0, 1]``: ``0.0`` corresponds to
    the activation memory from applying activation checkpointing to the full
    region, and ``1.0`` corresponds to the activation memory from the default
    runtime-optimized strategy. So ``0.4`` would result in a strategy that saves
    40% of the activations compared to the default strategy. It solves a 0-1
    knapsack to find the minimum recompute necessary to stay below the budget.
    This overrides the global ``torch._functorch.config.activation_memory_budget``
    for the annotated region.

    .. note::
        Today the partitioner only supports a single budget per compiled graph,
        so the annotation must cover every forward op in the graph (a partial
        annotation is rejected rather than silently applied graph-wide), and all
        annotated nodes must agree on the budget. To use different budgets for
        different parts of a model, separate them with a graph break (e.g.
        ``torch._dynamo.graph_break()``) so each part becomes its own graph.

    This only has an effect under :func:`torch.compile`; using it outside of a
    compiled region raises a ``RuntimeError``.

    Args:
        budget (float): Activation memory budget ratio in ``[0, 1]``.

    Example::

        >>> # xdoctest: +SKIP
        >>> with torch.autograd.graph.region_activation_memory_budget(0.0):
        ...     x = layer(x)  # recompute this region's activations in backward
    """
    import torch.fx.traceback as fx_traceback

    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise TypeError(
            f"torch.autograd.graph.region_activation_memory_budget: expects a "
            f"float, got {type(budget).__name__}"
        )
    if not 0.0 <= budget <= 1.0:
        raise ValueError(
            f"torch.autograd.graph.region_activation_memory_budget: must be in "
            f"[0, 1], got {budget}"
        )
    # The budget only takes effect when read back by the partitioner during
    # compilation. Dynamo folds torch.compiler.is_compiling() to True while
    # tracing this (inlined) call, so this guard only fires in eager mode.
    if not torch.compiler.is_compiling():
        raise RuntimeError(
            "torch.autograd.graph.region_activation_memory_budget can only be "
            "used inside a torch.compile region; it has no effect in eager mode."
        )
    return fx_traceback.annotate(
        {fx_traceback.MEMORY_BUDGET_ANNOTATION_KEY: float(budget)}
    )


def set_warn_on_accumulate_grad_stream_mismatch(enabled: bool) -> None:
    """Whether to warn when the AccumulateGrad node's stream does not match the stream
    of the node that produced the incoming gradient.
    """
    return torch._C._set_warn_on_accumulate_grad_stream_mismatch(enabled)


def set_override_stale_capture_stream(enabled: bool) -> None:
    """Control behavior when autograd detects a stale non-capturing stream during
    CUDA graph capture.

    During CUDA graph capture, autograd nodes may reference a stale stream
    that is not part of the capture. With the flag disabled (the
    process-initial state), autograd raises a ``RuntimeError`` when the stale
    stream is the default stream (stream 0), because this case always
    invalidates the capture: ``cudaStreamWaitEvent`` on the default stream
    pulls a non-capturing stream into the graph. For non-default stale streams
    the stream reference is left unchanged; the capture will succeed if the
    user has joined the stream into the capture (e.g. via
    ``capture_stream.wait_stream(stale_stream)``) and will otherwise fail with
    a CUDA runtime error.

    When ``enabled=True``, any stale non-capturing stream (default or
    non-default) is automatically overridden with the producer's capturing
    stream, allowing the capture to proceed. This is a process-global setting
    and is not thread-local.

    The flag also governs the end-of-backward sync between each leaf's stream
    and the caller's current stream. A leaf whose incoming gradients are all
    undefined (e.g. patterns that compute certain gradients out of band and
    return ``None`` from autograd) cannot be reconciled by the override, so
    when exactly one of the two streams is capturing, that sync would cross
    the capture boundary: with the flag enabled the sync is skipped (work on
    a non-capturing stream is not part of the capture, so the ordering has no
    effect on it); with the flag disabled a ``RuntimeError`` is raised.

    Args:
        enabled (bool): If ``True``, override stale non-capturing streams with
            the producer's capturing stream during CUDA graph capture, and
            skip end-of-backward leaf syncs that would cross the capture
            boundary. If ``False`` (the process-initial state), raise an error
            when the stale stream is the default stream (stream 0) or when a
            leaf sync would cross the capture boundary; other stale streams
            are left unchanged.
    """
    return torch._C._set_override_stale_capture_stream(enabled)


class _MultiHandle(RemovableHandle):
    handles: tuple[RemovableHandle, ...]

    def __init__(self, handles: tuple[RemovableHandle, ...]) -> None:
        self.handles = handles

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __getstate__(self) -> tuple[RemovableHandle, ...]:
        return self.handles

    def __setstate__(self, state: tuple[RemovableHandle, ...]) -> None:
        self.handles = state


def register_multi_grad_hook(
    tensors: Sequence[torch.Tensor],
    fn: Callable[[Sequence[torch.Tensor | None]], None]
    | Callable[[torch.Tensor], None],
    *,
    mode: Literal["all", "any"] = "all",
) -> RemovableHandle:
    r"""Register a multi-grad backward hook.

    There are two supported modes: ``"all"`` and ``"any"``.

    Under the ``"all"`` mode, the hook will be called after gradients with respect to every tensor in
    :attr:`tensors` have been computed. If a tensor is in :attr:`tensors` but
    is not part of the graph, or if a tensor is not needed to compute the gradients
    for any ``inputs`` specified for the current ``.backward()`` or ``.grad()`` call,
    this tensor will be ignored and the hook will not wait for its gradient to be
    computed.

    After every non-ignored tensor's gradient has been computed, :attr:`fn` will be
    called with those gradients. ``None`` will be passed for tensors that did not
    have their gradients computed.

    Under the ``"any"`` mode, the hook will be called after the first gradient
    with respect to a tensor in :attr:`tensors` has been computed. The hook
    will be called with that gradient as its argument.

    The hook should not modify its arguments.

    This function returns a handle with a method ``handle.remove()`` that removes the hook.

    .. note::
        See :ref:`backward-hooks-execution` for more information on how when this hook
        is executed, and how its execution is ordered relative to other hooks.

    Example::

        >>> import torch
        >>>
        >>> a = torch.rand(2, 3, requires_grad=True)
        >>> b = torch.rand(2, 3, requires_grad=True)
        >>> c = a * b
        >>> d = a * b
        >>>
        >>> def fn(grads):
        ...     print([g is not None for g in grads])
        ...
        >>> torch.autograd.graph.register_multi_grad_hook((a, b, c, d), fn)
        >>>
        >>> c.sum().backward(retain_graph=True)
        [True, True, True, False]
        >>> c.sum().backward(inputs=(a,), retain_graph=True)
        [True, False, True, False]
        >>>
    """
    supported_modes = ("all", "any")
    lock = threading.Lock()

    if mode not in supported_modes:
        raise ValueError(f"Expects mode to be one of {supported_modes} but got {mode}")

    if mode == "all":
        count: dict[int, int] = {}
        nb_calls = None
        buffer: dict[int, list[torch.Tensor | None]] = {}

        grad_fns = list(map(_get_grad_fn_or_grad_acc, tensors))
        len_tensors = len(tensors)

        def get_inner_hook(idx: int) -> Callable[[torch.Tensor], None]:
            def inner_hook(grad: torch.Tensor) -> None:
                nonlocal count, nb_calls, buffer, fn
                id = torch._C._current_graph_task_id()
                if id == -1:
                    raise AssertionError(
                        "expected this hook to be called inside a backward call"
                    )
                count[id] = count.get(id, 0)
                # pyrefly: ignore [unsupported-operation]
                buffer[id] = buffer.get(id, [None] * len_tensors)

                with lock:
                    curr_count, count[id] = count[id], count[id] + 1

                    if curr_count == 0:
                        # On the first call, compute the actual nb_calls and buffer
                        nb_calls = sum(
                            map(torch._C._will_engine_execute_node, grad_fns)
                        )

                buffer[id][idx] = grad

                if nb_calls is None:
                    raise AssertionError("Expected nb_calls to be set")
                if curr_count == nb_calls - 1:
                    fn = cast(Callable[[Sequence[torch.Tensor | None]], None], fn)
                    fn(buffer[id])
                    del count[id]
                    del buffer[id]

            return inner_hook

        handles = tuple(
            t.register_hook(get_inner_hook(i)) for i, t in enumerate(tensors)
        )
    elif mode == "any":
        fn = cast(Callable[[torch.Tensor], None], fn)
        ran_hook: dict[int, bool] = defaultdict(bool)

        @functools.wraps(fn)
        def wrapped_fn(grad: torch.Tensor) -> None:
            nonlocal ran_hook
            id = torch._C._current_graph_task_id()
            if id == -1:
                raise AssertionError(
                    "expected this hook to be called inside a backward call"
                )
            with lock:
                prev, ran_hook[id] = ran_hook[id], True
            if prev:
                return
            fn(grad)

        handles = tuple(
            tensor.register_hook(wrapped_fn)
            for tensor in tensors
            if tensor.requires_grad
        )

    return _MultiHandle(handles)  # type: ignore[possibly-undefined]


# NOTE [Allow mutation on tensors saved for backward]
#
# 1. Tensor gets saved for backward
#    - remember the python object id and the version of the tensor
#    - remember aliasing information (data_ptr of base + version)
#    - save the original so we control its lifetime
# 2. Any time a tensor gets in-placed
#    - for each tensor aliased to it:
#      - check using its object id and version to see if it has been saved
#      - if it has been saved, clone it
#      - delete the reference to the original
# 3. during backward
#    - if the clone exists, the tensor must've been modified in-place
_allow_mutation_on_saved_tensors_enabled: bool = False


_TID: TypeAlias = tuple[int, int, int]
_SID: TypeAlias = tuple[int, int]


def _get_tid(tensor: torch.Tensor) -> _TID:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        tensor,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        data_ptr = 0
    else:
        data_ptr = tensor.data_ptr()
    return (id(tensor), data_ptr, tensor._version)


def _get_sid(tensor: torch.Tensor) -> _SID:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        tensor,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        data_ptr = 0
    else:
        data_ptr = tensor.data_ptr()
    return (data_ptr, tensor._version)


class _Handle:
    pass


class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        def pack_hook(tensor: torch.Tensor) -> _Handle:
            tid = _get_tid(tensor)
            sid = _get_sid(tensor)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: _Handle | None = None

            # Save aliasing information
            ctx.sid_to_tid[sid].add(tid)

            # NB: The same tensor (of the same version) can be saved multiple times
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = tensor
            else:
                # Store an additional strong reference to the handle
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(handle: _Handle) -> torch.Tensor:
            error_msg = (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded."
            )
            if not _allow_mutation_on_saved_tensors_enabled:
                raise AssertionError(error_msg)
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                if handle not in ctx.original:
                    raise AssertionError(error_msg)
                res = ctx.original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)


class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        self.ctx = ctx

    def __torch_dispatch__(
        self,
        func: "OpOverload",
        types: Iterable[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[Any, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}

        def maybe_clone(t: torch.Tensor) -> None:
            tid = _get_tid(t)
            sid = _get_sid(t)
            ctx = self.ctx
            if sid in ctx.sid_to_tid:
                for tid in ctx.sid_to_tid[sid]:
                    if tid not in ctx.tid_to_weakhandle:
                        # We know that if tid is in sid_to_tid, then it must also be in
                        # tid_to_weakhandle. However, it is possible for the tensor to be
                        # saved at one point, but cleared by backward before it is modified
                        # in-place. Consider the following example:
                        #
                        # >>> a = torch.randn(2, 3, requires_grad=True).clone()
                        # >>> out = (a**2).sum()
                        # >>> out.backward()
                        # >>> a.sin_()
                        continue
                    handle = ctx.tid_to_weakhandle[tid]
                    if handle in ctx.cloned:
                        # The same exact tensor has been cloned already
                        continue
                    ctx.cloned[handle] = ctx.original[handle].clone()
                    del ctx.original[handle]

        for idx, arg in enumerate(func._schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                if arg.is_out:
                    maybe_clone(kwargs["out"])
                elif isinstance(args[idx], list):
                    # Foreach case. (Possible optimization: if most of the
                    # tensors need to be cloned, use a for each clone?)
                    for t in args[idx]:
                        maybe_clone(t)
                else:
                    maybe_clone(args[idx])

        return func(*args, **kwargs)


class _AllowMutationOnSavedContext:
    def __init__(self) -> None:
        self.cloned: MutableMapping[_Handle, torch.Tensor] = WeakKeyDictionary()
        self.original: MutableMapping[_Handle, torch.Tensor] = WeakKeyDictionary()
        self.tid_to_weakhandle: MutableMapping[_TID, _Handle] = WeakValueDictionary()
        self.sid_to_tid: dict[_SID, set[_TID]] = defaultdict(set)

    def clear(self) -> None:
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()


@contextlib.contextmanager
def allow_mutation_on_saved_tensors() -> Generator[
    _AllowMutationOnSavedContext, None, None
]:
    """Context manager under which mutating tensors saved for backward is allowed.

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.

    To ensure the correct behavior, both the forward and backward should be run under
    the same context manager.

    Returns:
        An _AllowMutationOnSavedContext object storing the state managed by this
        context manager. This object can be useful for debugging purposes. The state
        managed by the context manager is automatically cleared upon exiting.

    Example::

        >>> import torch
        >>> with torch.autograd.graph.allow_mutation_on_saved_tensors():
        ...     # forward
        ...     a = torch.ones(2, 3, requires_grad=True)
        ...     b = a.clone()
        ...     out = (b**2).sum()
        ...     b.sin_()
        ...     # backward
        ...     out.sum().backward()
        ...
        tensor([[0.8415, 0.8415, 0.8415],
                [0.8415, 0.8415, 0.8415]], grad_fn=<SinBackward0>)
    """
    global _allow_mutation_on_saved_tensors_enabled

    ctx = _AllowMutationOnSavedContext()

    with _swap_with_cloned(ctx), _CloneArgBeforeMutateMode(ctx):
        try:
            if _allow_mutation_on_saved_tensors_enabled:
                raise RuntimeError(
                    "allow_mutation_on_saved_tensors contexts cannot be nested"
                )
            _allow_mutation_on_saved_tensors_enabled = True
            yield ctx
        finally:
            ctx.clear()
            _allow_mutation_on_saved_tensors_enabled = False


def _register_logging_hooks_on_whole_graph(
    t_outputs: Sequence[torch.Tensor | GradientEdge],
) -> Callable[[], None]:
    grad_fns = list(map(_get_grad_fn_or_grad_acc, t_outputs))

    def iter_graph(roots: list[Node]) -> Iterator[Node]:
        if not roots:
            return
        seen: set[Node] = set()
        q: deque[Node] = deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _ in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def fmt(t: torch.Tensor | None) -> str:
        # Avoid circular import
        from torch.utils._dtype_abbrs import dtype_abbrs

        if t is None:
            return "None"
        return f"{dtype_abbrs[t.dtype]}[{', '.join(map(str, t.shape))}]"

    def prehook(grad_outputs: Sequence[torch.Tensor | None]) -> None:
        node = torch._C._current_autograd_node()
        grad_outputs_str = f"[{','.join(fmt(t) for t in grad_outputs)}]"
        log_str = f"Executing: {node} with grad_outputs: {grad_outputs_str}"
        log.debug(log_str)

    handles = [node.register_prehook(prehook) for node in iter_graph(grad_fns)]

    def unregister_hooks() -> None:
        for handle in handles:
            handle.remove()

    return unregister_hooks


def _engine_run_backward(
    t_outputs: Sequence[torch.Tensor | GradientEdge],
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    if attach_logging_hooks:
        unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)

    # Need to save the context so compiler config will be visible in device threads
    torch._C._stash_obj_in_tls("context", contextvars.copy_context())

    try:
        return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            t_outputs, *args, **kwargs
        )  # Calls into the C++ engine to run the backward pass
    finally:
        if attach_logging_hooks:
            unregister_hooks()  # type: ignore[possibly-undefined]
        # Erase rather than overwrite-with-None so the thread_local map is
        # truly empty.  SafePyObject's destructor needs the GIL; if a thread
        # exits while a SafePyObject is still in its thread_local,
        # __call_tls_dtors fires the destructor → take_gil → deadlock.
        torch._C._remove_obj_from_tls("context")
<|endoftext|># mypy: allow-untyped-defs
import bisect
import itertools
import json
import math
from collections import defaultdict, namedtuple
from collections.abc import Callable
from operator import attrgetter
from typing import Any, NamedTuple
from typing_extensions import deprecated

import torch
from torch.autograd import DeviceType


__all__ = [
    "EventList",
    "EventMetadata",
    "FormattedTimesMixin",
    "Interval",
    "Kernel",
    "FunctionEvent",
    "FunctionEventAvg",
    "StringTable",
    "MemRecordsAcc",
]


class EventList(list):
    """A list of profiling events with helper methods for analysis and visualization.

    EventList extends the standard Python list to provide specialized methods for
    working with profiling events (FunctionEvent or FunctionEventAvg objects).
    It includes utilities for aggregating statistics, formatting output tables,
    and exporting profiling data.

    This class is typically returned by profiler methods and should not be
    instantiated directly by users.

    Args:
        *args: Standard list arguments.
        use_device (str, optional): Device type for profiling ("cuda", "xpu", etc.).
        profile_memory (bool, optional): Whether memory profiling was enabled. Default: False.
        with_flops (bool, optional): Whether to include FLOP counts. Default: False.

    Attributes:
        _use_device (str): Device type being profiled.
        _profile_memory (bool): Whether memory profiling is enabled.
        _with_flops (bool): Whether FLOP counting is enabled.
        _tree_built (bool): Whether the event tree structure has been built.

    Key Methods:
        table(...): Format events as a table string for display.
        export_chrome_trace(path): Export to Chrome tracing format.
        export_stacks(path, metric): Export stack traces with metrics.
        key_averages(...): Compute averaged statistics grouped by operation name.
        total_average(): Compute aggregate totals across all events (sums, not averages).

    Properties:
        self_cpu_time_total: Sum of self CPU time across all events.

    Example::

        import torch
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)

        # EventList is returned by prof.events()
        events = prof.events()

        # Display as formatted table
        print(
            events.table(
                sort_by="cpu_time_total", row_limit=20, top_level_events_only=False
            )
        )

        # Export to Chrome tracing format
        events.export_chrome_trace("trace.json")

        # Get averaged statistics
        avg_events = events.key_averages()
        print(avg_events.table())

        # Export stack traces
        events.export_stacks("stacks.txt", "self_cpu_time_total")

    See Also:
        - :class:`FunctionEvent`: Individual profiling event
        - :class:`FunctionEventAvg`: Averaged profiling statistics
        - :meth:`table`: Format events as a readable table
        - :meth:`key_averages`: Aggregate events by operation name
    """

    def __init__(self, *args, **kwargs):
        use_device = kwargs.pop("use_device", None)
        profile_memory = kwargs.pop("profile_memory", False)
        with_flops = kwargs.pop("with_flops", False)
        # pyrefly: ignore [not-iterable]
        super().__init__(*args, **kwargs)
        self._use_device = use_device
        self._profile_memory = profile_memory
        self._tree_built = False
        self._with_flops = with_flops

    def _build_tree(self):
        self._populate_cpu_children()
        self._remove_dup_nodes()
        self._set_backward_stacktraces()
        self._tree_built = True

    def __str__(self):
        return self.table()

    def _remove_dup_nodes(self):
        while True:
            to_delete = set()

            for idx in range(len(self)):
                if (
                    self[idx].cpu_parent is not None
                    and self[idx].cpu_parent.name == self[idx].name
                    and len(self[idx].cpu_parent.cpu_children) == 1
                ):
                    self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                    self[idx].cpu_parent.kernels = self[idx].kernels  # lift kernels up
                    for ch in self[idx].cpu_children:
                        ch.cpu_parent = self[idx].cpu_parent
                    to_delete.add(idx)
            if len(to_delete) == 0:
                break

            new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]

            self.clear()

            self.extend(new_evts)

    def _populate_cpu_children(self):
        """Populate child events into each underlying FunctionEvent object.

        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
        # Some events can be async (i.e. start and end on different threads),
        # since it's generally undefined how to attribute children ranges to
        # async ranges, we do not use them when calculating nested ranges and stats
        sync_events = [
            evt
            for evt in self
            if not evt.is_async and evt.device_type == DeviceType.CPU
        ]
        events = sorted(
            sync_events,
            key=attrgetter("thread"),
        )
        # Group by both thread and node_id, so that events that happen to have
        # the same thread_id but are from different nodes aren't incorrectly
        # grouped together.
        threads = itertools.groupby(
            events, key=lambda event: (event.thread, event.node_id)
        )

        # For each thread we keep a stack of current nested parents.
        # We maintain the invariant that each interval is a subset of all other
        # intervals lower in the stack.
        #
        # First we sort the intervals by their start time. Then we iterate over them.
        # Every time we see a new interval we remove several parents from
        # the top until we restore the invariant. Then parent child relationship
        # if recorded if the stack is not empty.
        # Finally we add new interval to the list
        #
        # Algorithm has O(N * log(N)) complexity where N is number of
        # intervals
        for _thread_id, thread_events in threads:
            thread_events_ = sorted(
                thread_events,
                key=lambda event: [event.time_range.start, -event.time_range.end],
            )
            current_events: list[FunctionEvent] = []
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if (
                        event.time_range.start >= parent.time_range.end
                        or event.time_range.end > parent.time_range.end
                    ):
                        # this can't be a parent
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        if event.cpu_parent is not None:
                            raise AssertionError(
                                f"There is already a CPU parent event for {event.key}"
                            )
                        event.set_cpu_parent(parent)
                        break

                current_events.append(event)

    def _set_backward_stacktraces(self):
        def bw_parent(evt):
            if evt is None:
                return None
            elif evt.scope == 1:  # BACKWARD_FUNCTION
                return evt
            else:
                return bw_parent(evt.cpu_parent)

        fwd_stacks = {}
        for evt in self:
            if bw_parent(evt) is None and evt.stack is not None:
                t = (evt.sequence_nr, evt.thread)
                if t not in fwd_stacks:
                    fwd_stacks[t] = evt.stack

        for evt in self:
            p = bw_parent(evt)
            if p is not None:
                if p.fwd_thread is None:
                    raise AssertionError(
                        "Expected fwd_thread to be set for backward parent"
                    )
                t = (p.sequence_nr, p.fwd_thread)
                evt.stack = fwd_stacks.get(t, [])

    @property
    def self_cpu_time_total(self):
        return sum(event.self_cpu_time_total for event in self)

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
        time_unit=None,
    ):
        """Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``xpu_time``,
                ``cpu_time_total``, ``cuda_time_total``, ``xpu_time_total``,
                ``cpu_memory_usage``, ``cuda_memory_usage``, ``xpu_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``,
                ``self_xpu_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda/xpu ops events are omitted for profiler result readability.
            time_unit(str, optional): A time unit to be used for all values in the
                table. Valid options are: ``s``, ``ms`` and ``us``.

        Returns:
            A string containing the table.
        """
        return _build_table(
            self,
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            top_level_events_only=top_level_events_only,
            time_unit=time_unit,
        )

    def export_chrome_trace(self, path):
        """Export an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os

        device_name = "cuda" if not self._use_device else self._use_device
        with open(path, "w") as f:
            next_id = 0
            # Use file IO over using json.dump since JSON dumping is very slow and
            # this technique is proven to give a 4x speedup.
            f.write("[")
            for evt in self:
                if evt.trace_name is None:
                    continue
                name_json = json.dumps(evt.trace_name)
                f.write(
                    '{{"name": {}, '
                    '"ph": "X", '
                    '"ts": {}, '
                    '"dur": {}, '
                    '"tid": {}, '
                    '"pid": "CPU functions", '
                    '"args": {{}}}}, '.format(
                        name_json,
                        evt.time_range.start,
                        evt.time_range.elapsed_us(),
                        evt.thread
                        if not evt.is_remote
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "',
                    )
                )
                for _ in evt.kernels:
                    # 's' and 'f' draw Flow arrows from
                    # the CPU launch to the GPU kernel
                    f.write(
                        f'{{"name": {name_json}, '
                        '"ph": "s", '
                        f'"ts": {evt.time_range.start}, '
                        f'"tid": {evt.thread}, '
                        '"pid": "CPU functions", '
                        f'"id": {next_id}, '
                        f'"cat": "cpu_to_{device_name}", '
                        '"args": {}}, '
                    )
                    # Note: use torch.profiler to get device kernel trace
                    next_id += 1
            if len(self) > 0:
                # remove trailing whitespace and comma
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()
            f.write("]")

    def supported_export_stacks_metrics(self):
        return [
            "self_cpu_time_total",
            "self_cuda_time_total",
            "self_xpu_time_total",
            "self_privateuse1_time_total",
        ]

    def export_stacks(self, path: str, metric: str):
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError(
                "metric should be one of: "
                + str(self.supported_export_stacks_metrics())
            )
        translate_table = str.maketrans(" ;\t\n", "____")
        with open(path, "w") as f:
            for evt in self:
                if evt.stack and len(evt.stack) > 0:
                    metric_value = getattr(
                        evt,
                        metric.replace("cuda", "device")
                        .replace("xpu", "device")
                        .replace("privateuse1", "device"),
                    )
                    if int(metric_value) > 0:
                        stack_str = ""
                        for entry in reversed(evt.stack):
                            stack_str += entry.translate(translate_table)
                            stack_str += ";"
                        stack_str = stack_str[:-1] + " " + str(int(metric_value))
                        f.write(stack_str + "\n")

    def key_averages(
        self,
        group_by_input_shapes=False,
        group_by_stack_n=0,
        group_by_overload_name=False,
        include_python_functions=False,
    ):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

            group_by_overload_name: Differentiate operators by their overload name e.g. aten::add.Tensor
            and aten::add.out will be aggregated separately

            include_python_functions: include Python function events (e.g. individual
                Python callsite entries captured with ``with_stack=True``) in the
                averages. By default these are excluded because they tend to appear as
                misleading hotspots (e.g. ``threading.py: wait``) that obscure the
                real operator-level breakdown. Set to ``True`` to restore the raw
                per-callsite view.

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        if not self._tree_built:
            raise AssertionError(
                "Expected tree to be built before calling key_averages"
            )
        stats: dict[tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(
            event, group_by_input_shapes, group_by_stack_n, group_by_overload_name
        ) -> tuple[str, ...]:
            key = [
                str(event.key),
                str(event.node_id),
                str(event.device_type),
                str(event.is_legacy),
                str(event.is_user_annotation),
            ]
            if group_by_overload_name:
                key.append(evt.overload_name)
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)

        for evt in self:
            if evt.is_python_function and not include_python_functions:
                continue
            stats[
                get_key(
                    evt, group_by_input_shapes, group_by_stack_n, group_by_overload_name
                )
            ].add(evt)

        avg_list = EventList(
            stats.values(),
            use_device=self._use_device,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
        )
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                evt.input_shapes = ""
            if not group_by_overload_name:
                evt.overload_name = ""
        return avg_list

    def total_average(self):
        """Compute aggregate statistics across all events.

        Accumulates statistics from all events into a single FunctionEventAvg object.
        This is primarily useful for computing total metrics (total CPU time, total
        memory usage, etc.) across the entire profiling session, regardless of
        operation type.

        Note:
            This sums up times and counts across ALL different operations, so the
            "average" metrics (like cpu_time) represent the average time per operation
            call across the entire session, mixing all operation types together.
            For per-operation averages, use :meth:`key_averages` instead.

        Returns:
            FunctionEventAvg: A single aggregate object with key="Total" containing
                accumulated statistics.

        """
        total_stat = FunctionEventAvg()
        for evt in self:
            total_stat += evt
            total_stat.key = None
        total_stat.key = "Total"
        return total_stat


def _format_time(time_us):
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


def _format_time_share(time_us, total_time_us):
    """Define how to format time in FunctionEvent."""
    if total_time_us == 0:
        if time_us != 0:
            raise AssertionError(f"Expected time_us == 0 but got {time_us}")
        return "NaN"
    return f"{time_us * 100.0 / total_time_us:.2f}%"


def _format_memory(nbytes):
    """Return a formatted memory size string."""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} GB"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} MB"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} KB"
    else:
        return str(nbytes) + " B"


def _attr_formatter(name):
    return property(lambda self: _format_time(getattr(self, name)))


class FormattedTimesMixin:
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """

    cpu_time_str = _attr_formatter("cpu_time")
    device_time_str = _attr_formatter("device_time")
    cpu_time_total_str = _attr_formatter("cpu_time_total")
    device_time_total_str = _attr_formatter("device_time_total")
    self_cpu_time_total_str = _attr_formatter("self_cpu_time_total")
    self_device_time_total_str = _attr_formatter("self_device_time_total")

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count  # type: ignore[attr-defined]

    @property
    def device_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.device_time_total / self.count  # type: ignore[attr-defined]

    @property
    @deprecated(
        "`cuda_time` is deprecated, please use `device_time` instead.",
        category=FutureWarning,
    )
    def cuda_time(self):  # To be deprecated
        return self.device_time


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def elapsed_us(self):
        r"""
        Returns the length of the interval
        """
        return self.end - self.start


Kernel = namedtuple("Kernel", ["name", "device", "duration"])


class EventMetadata(NamedTuple):
    # Kernel fields
    registers_per_thread: int | None
    shared_memory: int | None
    grid: list[int] | None
    block: list[int] | None
    priority: int | None
    blocks_per_sm: float | None
    warps_per_sm: float | None
    occupancy: dict[str, Any] | None
    est_occupancy_pct: float | None
    queued: int | None
    graph_id: int | None
    graph_node_id: int | None
    stream: int | None
    context: int | None
    channel: int | None
    channel_type: int | None
    # Memory fields
    bytes: int | None
    bandwidth_gb_s: float | None
    # NCCL fields
    collective_name: str | None
    dtype: str | None
    in_msg_nelems: int | None
    out_msg_nelems: int | None
    in_split_size: str | None
    out_split_size: str | None
    global_rank_start: int | None
    global_rank_stride: int | None
    group_size: int | None
    process_group_name: str | None
    process_group_desc: str | None
    group_ranks: str | None
    rank: int | None
    src_rank: int | None
    dst_rank: int | None
    seq: int | None
    is_async: bool | None


def _to_str(v: str) -> str:
    return v.strip('"')


def _to_bool(v: str) -> bool:
    return v in ("1", "true")


# Kineto key → (EventMetadata field name, converter from string)
_EVENT_METADATA_KEYS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "registers per thread": ("registers_per_thread", int),
    "shared memory": ("shared_memory", int),
    "grid": ("grid", json.loads),  # list[int]
    "block": ("block", json.loads),  # list[int]
    "priority": ("priority", int),
    "blocks per SM": ("blocks_per_sm", float),
    "warps per SM": ("warps_per_sm", float),
    "est. achieved occupancy %": ("est_occupancy_pct", float),
    "occupancy": ("occupancy", json.loads),  # dict[str, Any]
    "queued": ("queued", int),
    "graph id": ("graph_id", int),
    "graph node id": ("graph_node_id", int),
    "stream": ("stream", int),
    "context": ("context", int),
    "channel": ("channel", int),
    "channel_type": ("channel_type", int),
    "bytes": ("bytes", int),
    "memory bandwidth (GB/s)": ("bandwidth_gb_s", float),
    "Collective name": ("collective_name", _to_str),
    "dtype": ("dtype", _to_str),
    "In msg nelems": ("in_msg_nelems", int),
    "Out msg nelems": ("out_msg_nelems", int),
    "In split size": ("in_split_size", _to_str),
    "Out split size": ("out_split_size", _to_str),
    "Global rank start": ("global_rank_start", int),
    "Global rank stride": ("global_rank_stride", int),
    "Group size": ("group_size", int),
    "Process Group Name": ("process_group_name", _to_str),
    "Process Group Description": ("process_group_desc", _to_str),
    "Process Group Ranks": ("group_ranks", _to_str),
    "Rank": ("rank", int),
    "Src Rank": ("src_rank", int),
    "Dst Rank": ("dst_rank", int),
    "Seq": ("seq", int),
    "Is asynchronized op": ("is_async", _to_bool),
}


def _build_metadata(extra_meta):
    fields: dict[str, Any] = {}
    any_populated = False
    for kineto_key, (field_name, convert) in _EVENT_METADATA_KEYS.items():
        v = extra_meta.get(kineto_key)
        if v is not None:
            fields[field_name] = convert(v)
            any_populated = True
        else:
            fields[field_name] = None
    return EventMetadata(**fields) if any_populated else None


class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function.

    FunctionEvent records the execution of a single operation during profiling.
    These events are obtained from the profiler/kineto and contain detailed
    timing and memory usage information.

    .. note::
        FunctionEvent objects are typically created by the profiler/kineto and should not
        be instantiated directly by users. Access them through the profiler's output.

    Attributes:
        id (int): Unique identifier for this event.
        node_id (int): Node identifier for distributed profiling (-1 if not applicable).
        name (str): Name of the profiled function/operator.
        overload_name (str): Overload name for the operator (requires _ExperimentalConfig(capture_overload_names=True) set).
        trace_name (str): Same as name, just changes ProfilerStep* to ProfilerStep#
        time_range (Interval): Time interval containing start and end timestamps in microseconds.
        thread (int): Thread ID where the operation started.
        fwd_thread (int): Thread ID of the corresponding forward operation.
        kernels (List[Kernel]): List of device kernels launched by this operation.
        count (int): Number of times this event was called (usually 1).
        cpu_children (List[FunctionEvent]): Direct CPU child operations.
        cpu_parent (FunctionEvent): Direct CPU parent operation.
        input_shapes (List[List[int]]): Shapes of input tensors (requires record_shapes=True).
            For plain tensor inputs, each entry is a list of dimensions (e.g. ``[16, 16]``).
            TensorList inputs are represented as an empty list ``[]``; use
            ``structured_input_shapes`` to get per-element shapes for TensorList inputs.
        concrete_inputs (List[Any]): Concrete input values (requires record_shapes=true).
        kwinputs (Dict[str, Any]): Keyword arguments (requires record_shapes=true).
        stack (List[str]): Python stack trace where the operation was called (requires with_stack=true).
        scope (int): at::RecordScope identifier (0=forward, 1=backward, etc.).
        use_device (str): Device type being profiled ("cuda", "xpu", etc.).
        cpu_memory_usage (int): CPU memory allocated in bytes.
        device_memory_usage (int): Device memory allocated in bytes.
        is_async (bool): Whether this is an asynchronous operation.
        is_remote (bool): Whether this operation occurred on a remote node.
        sequence_nr (int): Sequence number for autograd operations.
        device_type (DeviceType): Type of device (CPU, CUDA, XPU, PrivateUse1, etc.).
        device_index (int): Index of the device (e.g., GPU 0, 1, 2).
        device_resource_id (int): Resource ID on the device (ie. stream ID).
        is_legacy (bool): Whether this is from the legacy profiler.
        flops (int): Estimated floating point operations.
        is_user_annotation (bool): Whether this is a user-annotated region.
        metadata (Dict[str, Any]): Additional metadata keyed by the field names
            used in exported traces. Use
            ``_ExperimentalConfig(expose_kineto_event_metadata=True)`` to expose
            Kineto activity metadata. Available fields vary by activity and backend.
        metadata_json (str): Deprecated. Use event_metadata instead.
        event_metadata (EventMetadata): Additional metadata in structured format.
        structured_input_shapes (List[List[int] | List[List[int]]]): Like ``input_shapes``
            but distinguishes TensorList inputs.  Plain tensor inputs are ``List[int]``;
            TensorList inputs are ``List[List[int]]`` containing one shape per tensor in the list.
            Matches the ``"Input Dims"`` field in the Chrome trace JSON.
        structured_input_strides (List[List[int] | List[List[int]]]): Strides of input
            tensors in the same format as ``structured_input_shapes`` (requires
            record_shapes=True).

    Properties:
        cpu_time_total (float): Total CPU time in microseconds.
        device_time_total (float): Total device (CUDA/XPU/etc) time in microseconds.
        self_cpu_time_total (float): CPU time excluding child operations.
        self_device_time_total (float): Device time excluding child operations.
        self_cpu_memory_usage (int): CPU memory usage excluding child operations.
        self_device_memory_usage (int): Device memory usage excluding child operations.
        cpu_time (float): Average CPU time per call.
        device_time (float): Average device time per call.
        key (str): Key used for grouping events (usually same as name).

    See Also:
        - :class:`torch.profiler.profile`: Context manager for profiling
        - :class:`EventList`: List container for FunctionEvent objects with helper methods
        - :class:`FunctionEventAvg`: Averaged statistics over multiple FunctionEvent objects
    """

    def __init__(
        self,
        id,
        name,
        thread,
        start_us,
        end_us,
        overload_name=None,
        fwd_thread=None,
        input_shapes=None,
        stack=None,
        scope=0,
        use_device=None,
        cpu_memory_usage=0,
        device_memory_usage=0,
        is_async=False,
        is_remote=False,
        sequence_nr=-1,
        node_id=-1,
        device_type=DeviceType.CPU,
        device_index=0,
        device_resource_id=None,
        is_legacy=False,
        flops=None,
        trace_name=None,
        concrete_inputs=None,
        kwinputs=None,
        is_user_annotation=False,
        is_python_function=False,
        activity_type=None,
        metadata_json=None,
        flow_id=None,
        flow_type=None,
        flow_start=None,
        external_id=0,
        linked_correlation_id=0,
        extra_meta=None,
        structured_input_shapes=None,
        structured_input_strides=None,
        input_dtypes=None,
        python_id=-1,
        python_parent_id=-1,
        python_module_id=-1,
        typed_metadata=None,
    ):
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        # pyrefly: ignore [bad-assignment]
        self.overload_name: str = overload_name
        # pyrefly: ignore [bad-assignment]
        self.trace_name: str = trace_name
        self.time_range: Interval = Interval(start_us, end_us)
        self.thread: int = thread
        self.fwd_thread: int | None = fwd_thread
        self.kernels: list[Kernel] = []
        self.count: int = 1
        self.cpu_children: list[FunctionEvent] = []
        self.cpu_parent: FunctionEvent | None = None
        # pyrefly: ignore [bad-assignment]
        self.input_shapes: tuple[int, ...] = input_shapes
        # pyrefly: ignore [bad-assignment]
        self.concrete_inputs: list[Any] = concrete_inputs
        # pyrefly: ignore [bad-assignment]
        self.kwinputs: dict[str, Any] = kwinputs
        # pyrefly: ignore [bad-assignment]
        self.stack: list = stack
        self.scope: int = scope
        self.use_device: str | None = use_device
        self.cpu_memory_usage: int = cpu_memory_usage
        self.device_memory_usage: int = device_memory_usage
        self.is_async: bool = is_async
        self.is_remote: bool = is_remote
        self.sequence_nr: int = sequence_nr
        self.device_type: DeviceType = device_type
        self.device_index: int = device_index
        self.device_resource_id: int = (
            thread if device_resource_id is None else device_resource_id
        )
        self.is_legacy: bool = is_legacy
        self.flops: int | None = flops
        self.is_user_annotation: bool | None = is_user_annotation
        self.is_python_function: bool = is_python_function
        self.activity_type: str | None = activity_type
        self.self_cpu_percent = -1
        self.total_cpu_percent = -1
        self.total_device_percent = -1
        self._metadata_json = metadata_json
        self.flow_id: int | None = flow_id
        self.flow_type: int | None = flow_type
        self.flow_start: bool | None = flow_start
        self.external_id: int = external_id
        self.linked_correlation_id: int = linked_correlation_id
        self.metadata: dict[str, Any] | None = typed_metadata
        self.event_metadata: EventMetadata | None = (
            _build_metadata(extra_meta) if extra_meta else None
        )
        # pyrefly: ignore [bad-assignment]
        self.structured_input_shapes: list = structured_input_shapes
        # pyrefly: ignore [bad-assignment]
        self.structured_input_strides: list = structured_input_strides
        # pyrefly: ignore [bad-assignment]
        self.input_dtypes: list[str] = input_dtypes
        self.python_id: int = python_id
        self.python_parent_id: int = python_parent_id
        self.python_module_id: int = python_module_id

    def append_kernel(self, name, device, duration):
        if self.device_type != DeviceType.CPU:
            raise AssertionError("Expected device_type to be CPU")
        self.kernels.append(Kernel(name, device, duration))

    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        if self.device_type != DeviceType.CPU:
            raise AssertionError("Expected device_type to be CPU")
        if not isinstance(child, FunctionEvent):
            raise AssertionError("Expected child to be a FunctionEvent")
        if child.device_type != DeviceType.CPU:
            raise AssertionError("Expected child device_type to be CPU")
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        if self.device_type != DeviceType.CPU:
            raise AssertionError("Expected device_type to be CPU")
        if not isinstance(parent, FunctionEvent):
            raise AssertionError("Expected parent to be a FunctionEvent")
        if parent.device_type != DeviceType.CPU:
            raise AssertionError("Expected parent device_type to be CPU")
        self.cpu_parent = parent

    # Note: async events don't have children, are not used when computing 'self'
    # metrics of other events, have only total cpu time
    @property
    def self_cpu_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_memory_usage - sum(
            child.cpu_memory_usage for child in self.cpu_children
        )

    @property
    def self_device_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.device_memory_usage - sum(
            child.device_memory_usage for child in self.cpu_children
        )

    @property
    @deprecated(
        "`metadata_json` is deprecated. Use `event_metadata` instead.",
        category=FutureWarning,
    )
    def metadata_json(self):
        return self._metadata_json

    @property
    @deprecated(
        "`self_cuda_memory_usage` is deprecated. Use `self_device_memory_usage` instead.",
        category=FutureWarning,
    )
    def self_cuda_memory_usage(self):  # To be deprecated
        return self.self_device_memory_usage

    @property
    def cpu_time_total(self):
        if self.device_type == DeviceType.CPU:
            return self.time_range.elapsed_us()
        else:
            return 0

    @property
    def self_cpu_time_total(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_time_total - sum(
            child.cpu_time_total for child in self.cpu_children
        )

    @property
    def device_time_total(self):
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            if not self.is_legacy:
                # account for the kernels in the children ops
                return sum(kinfo.duration for kinfo in self.kernels) + sum(
                    ch.device_time_total for ch in self.cpu_children
                )
            else:
                # each legacy cpu events has a single (fake) kernel
                return sum(kinfo.duration for kinfo in self.kernels)
        else:
            if self.device_type not in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.HPU,
                DeviceType.XPU,
            ]:
                raise AssertionError(
                    f"Expected device_type to be CUDA, PrivateUse1, MTIA, HPU or XPU, but got {self.device_type}"
                )
            return self.time_range.elapsed_us()

    @property
    @deprecated(
        "`cuda_time_total` is deprecated. Use `device_time_total` instead.",
        category=FutureWarning,
    )
    def cuda_time_total(self):  # To be deprecated
        return self.device_time_total

    @property
    def self_device_time_total(self):
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            return self.device_time_total - sum(
                child.device_time_total for child in self.cpu_children
            )
        else:
            if self.device_type not in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.HPU,
                DeviceType.XPU,
            ]:
                raise AssertionError(
                    f"Expected device_type to be CUDA, PrivateUse1, MTIA, HPU or XPU, but got {self.device_type}"
                )
            return self.device_time_total

    @property
    @deprecated(
        "`self_cuda_time_total` is deprecated. Use `self_device_time_total` instead.",
        category=FutureWarning,
    )
    def self_cuda_time_total(self):  # To be deprecated
        return self.self_device_time_total

    @property
    def key(self):
        return self.name

    def __repr__(self):
        device_name = self.use_device
        device_time = self.device_time_str
        device_memory_usage = self.device_memory_usage
        return (
            f"<FunctionEvent id={self.id} name={self.name} overload_name={self.overload_name} "
            f"device_type={self.device_type} node_id={self.node_id} cpu_time={self.cpu_time_str} "
            f"start_us={self.time_range.start} end_us={self.time_range.end} "
            f"cpu_children={str([child.id for child in self.cpu_children])} {device_name}_time={device_time} "
            f"name={self.name} thread={self.thread} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory_usage} "
            f"is_async={self.is_async} is_remote={self.is_remote} seq_nr={self.sequence_nr} is_legacy={self.is_legacy}>"
        )


class FunctionEventAvg(FormattedTimesMixin):
    """Averaged profiling statistics over multiple FunctionEvent objects.

    FunctionEventAvg aggregates statistics from multiple FunctionEvent objects
    with the same key (typically same operation name). This is useful for getting
    average performance metrics across multiple invocations of the same operation.

    This class is typically created by calling :meth:`EventList.key_averages()` on
    a profiler's event list.

    Attributes:
        key (str): Grouping key for the events (typically operation name).
        count (int): Total number of events aggregated.
        node_id (int): Node identifier for distributed profiling (-1 if not applicable).
        is_async (bool): Whether the operations are asynchronous.
        is_remote (bool): Whether the operations occurred on a remote node.
        use_device (str): Device type being profiled ("cuda", "xpu", etc.).
        cpu_time_total (int): Accumulated total CPU time in microseconds.
        device_time_total (int): Accumulated total device time in microseconds.
        self_cpu_time_total (int): Accumulated self CPU time (excluding children) in microseconds.
        self_device_time_total (int): Accumulated self device time (excluding children) in microseconds.
        input_shapes (List[List[int]]): Input tensor shapes (requires record_shapes=true).
        overload_name (str): Operator overload name (requires _ExperimentalConfig(capture_overload_names=True) set).
        stack (List[str]): Python stack trace where the operation was called (requires with_stack=true).
        scope (int): at::RecordScope identifier (0=forward, 1=backward, etc.).
        cpu_memory_usage (int): Accumulated CPU memory usage in bytes.
        device_memory_usage (int): Accumulated device memory usage in bytes.
        self_cpu_memory_usage (int): Accumulated self CPU memory usage in bytes.
        self_device_memory_usage (int): Accumulated self device memory usage in bytes.
        cpu_children (List[FunctionEvent]): CPU child events.
        cpu_parent (FunctionEvent): CPU parent event.
        device_type (DeviceType): Type of device (CPU, CUDA, XPU, PrivateUse1, etc.).
        is_legacy (bool): Whether from legacy profiler.
        flops (int): Total floating point operations.
        is_user_annotation (bool): Whether this is a user-annotated region.

    Properties:
        cpu_time (float): Average CPU time per invocation.
        device_time (float): Average device time per invocation.

    See Also:
        - :class:`EventList.key_averages`: Method that creates FunctionEventAvg objects
        - :class:`FunctionEvent`: Individual profiling event
        - :class:`EventList`: Container for profiling events
    """

    def __init__(self) -> None:
        self.key: str | None = None
        self.count: int = 0
        self.node_id: int = 0
        self.is_async: bool = False
        self.is_remote: bool = False
        self.use_device: str | None = None
        self.cpu_time_total: int = 0
        self.device_time_total: int = 0
        self.self_cpu_time_total: int = 0
        self.self_device_time_total: int = 0
        self.input_shapes: list[list[int]] | None = None
        self.overload_name: str | None = None
        self.stack: list | None = None
        self.scope: int | None = None
        self.cpu_memory_usage: int = 0
        self.device_memory_usage: int = 0
        self.self_cpu_memory_usage: int = 0
        self.self_device_memory_usage: int = 0
        self.cpu_children: list[FunctionEvent] | None = None
        self.cpu_parent: FunctionEvent | None = None
        self.device_type: DeviceType = DeviceType.CPU
        self.is_legacy: bool = False
        self.flops: int = 0

    def add(self, other):
        if self.key is None:
            # First function being recorded as part of FunctionEventAvg, propagate
            # fields.
            self.key = other.key
            self.node_id = other.node_id
            self.is_async = other.is_async
            self.is_remote = other.is_remote
            self.cpu_parent = other.cpu_parent
            self.cpu_children = other.cpu_children

            self.overload_name = other.overload_name
            self.input_shapes = other.input_shapes
            self.stack = other.stack
            self.scope = other.scope
            self.device_type = other.device_type
            self.is_legacy = other.is_legacy
            self.use_device = other.use_device
            self.is_user_annotation = other.is_user_annotation

        if not isinstance(other, (FunctionEvent, FunctionEventAvg)):
            raise AssertionError(
                "Expected other to be a FunctionEvent or FunctionEventAvg"
            )
        if other.key != self.key:
            raise AssertionError(
                f"Expected keys to match, but got {other.key} vs {self.key}"
            )

        self.cpu_time_total += other.cpu_time_total
        self.device_time_total += other.device_time_total
        self.self_cpu_time_total += other.self_cpu_time_total
        self.self_device_time_total += other.self_device_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.device_memory_usage += other.device_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_device_memory_usage += other.self_device_memory_usage
        self.count += other.count
        if self.flops is None:
            # pyrefly: ignore [bad-assignment]
            self.flops = other.flops
        elif other.flops is not None:
            self.flops += other.flops
        return self

    def __iadd__(self, other):
        return self.add(other)

    def __repr__(self):
        device_name = "cuda" if not self.use_device else self.use_device
        self_device_time = self.self_device_time_total_str
        device_time = self.device_time_str
        device_memory = self.device_memory_usage
        return (
            f"<FunctionEventAvg key={self.key} self_cpu_time={self.self_cpu_time_total_str} cpu_time={self.cpu_time_str} "
            f" self_{device_name}_time={self_device_time} {device_name}_time={device_time} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory}>"
        )


class StringTable(defaultdict):
    def __missing__(self, key):
        # manage cases like 't' (demangled to 'unsigned short') separately,
        # for now simply check the length to avoid unexpected results for
        # the short sequences
        self[key] = torch._C._demangle(key) if len(key) > 1 else key
        return self[key]


class MemRecordsAcc:
    """Acceleration structure for accessing mem_records in interval."""

    def __init__(self, mem_records):
        self._mem_records = mem_records
        self._start_nses: list[int] = []
        self._indices: list[int] = []
        if len(mem_records) > 0:
            tmp = sorted([(r[0].start_ns(), i) for i, r in enumerate(mem_records)])
            self._start_nses, self._indices = zip(*tmp)  # type: ignore[assignment]

    def in_interval(self, start_ns, end_ns):
        r"""
        Return all records in the given interval
        """
        start_idx = bisect.bisect_left(self._start_nses, start_ns)
        end_idx = bisect.bisect_right(self._start_nses, end_ns)
        for i in range(start_idx, end_idx):
            yield self._mem_records[self._indices[i]]


def _filter_stack_entry(entry):
    filtered_entries = [
        ("autograd/__init__", "_make_grads"),
        ("autograd/__init__", "backward"),
        ("torch/tensor", "backward"),
        ("_internal/common_utils", "prof_callable"),
        ("_internal/common_utils", "prof_func_call"),
        ("_internal/common_utils", "prof_meth_call"),
    ]
    return all(not (f[0] in entry and f[1] in entry) for f in filtered_entries)


MEMORY_EVENT_NAME = "[memory]"
OUT_OF_MEMORY_EVENT_NAME = "[OutOfMemory]"


def _filter_name(name):
    # ignoring the following utility ops
    filtered_out_names = [
        MEMORY_EVENT_NAME,  # used only for the top-level memory events
        OUT_OF_MEMORY_EVENT_NAME,
        "profiler::_record_function_enter",
        "profiler::_record_function_enter_new",
        "profiler::_record_function_exit",
        "aten::is_leaf",
        "aten::output_nr",
        "aten::_version",
    ]
    return name in filtered_out_names


# Demangles and optionally rewrites the provided event name,
# with_wildcard - whether to replace certain numbered event names
# with a wildcard name to aggregate them together in the profiler table
# output
def _rewrite_name(name, with_wildcard=False):
    string_table = StringTable()
    name = string_table[name]
    if with_wildcard:
        if name.startswith("ProfilerStep#"):
            name = "ProfilerStep*"
    return name


def _build_table(
    events,
    sort_by=None,
    header=None,
    row_limit=100,
    max_src_column_width=75,
    max_name_column_width=55,
    max_shapes_column_width=80,
    with_flops=False,
    profile_memory=False,
    top_level_events_only=False,
    time_unit=None,
):
    """Print a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_device_time = any(event.self_device_time_total > 0 for event in events)
    has_device_mem = any(event.self_device_memory_usage > 0 for event in events)
    use_device = events[0].use_device
    # Running on PrivateUse1 device with profiler but not enable
    # ProfilerActivity.PrivateUse1 can also catch privateuse1 memory usage.
    # Here only need to check has_privateuse1_time if not use_device.
    if not use_device and has_device_time:
        raise RuntimeError("use_device is None, but there is device performance data.")

    has_input_shapes = any(
        (event.input_shapes is not None and len(event.input_shapes) > 0)
        for event in events
    )

    has_overload_names = any(
        (event.overload_name is not None and len(event.overload_name) > 0)
        for event in events
    )

    if sort_by is not None:
        events = EventList(
            sorted(
                events,
                key=lambda evt: getattr(
                    evt,
                    sort_by.replace("cuda", "device")
                    .replace("xpu", "device")
                    .replace("privateuse1", "device"),
                ),
                reverse=True,
            ),
            use_device=use_device,
            profile_memory=profile_memory,
            with_flops=with_flops,
        )

    name_column_width = max(len(evt.key) for evt in events) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)

    shapes_column_width = max(len(str(evt.input_shapes)) for evt in events) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)

    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = [
        evt.stack for evt in events if evt.stack is not None and len(evt.stack) > 0
    ]
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = (
            max(max(len(entry) for entry in stack) for stack in stacks) + 4
        )
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    headers = ["Name"]
    if has_overload_names:
        headers.append("Overload Name")
    headers += [
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
    ]

    device_name = use_device.upper() if use_device is not None else "None"
    if has_device_time:
        headers.extend(
            [
                f"Self {device_name}",
                f"Self {device_name} %",
                f"{device_name} total",
                f"{device_name} time avg",
            ]
        )
    if profile_memory:
        headers.extend(
            [
                "CPU Mem",
                "Self CPU Mem",
            ]
        )
        if use_device and has_device_mem:
            headers.extend(
                [
                    f"{device_name} Mem",
                    f"Self {device_name} Mem",
                ]
            )
    headers.append("# of Calls")
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any(evt.node_id != -1 for evt in events)
    if append_node_id:
        headers.append("Node ID")

    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]

    def add_column(padding, text_dir=">"):
        row_format_lst[0] += (
            "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
        )
        header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        flop_headers = [
            "FLOPs",
            "KFLOPs",
            "MFLOPs",
            "GFLOPs",
            "TFLOPs",
            "PFLOPs",
        ]
        if flops <= 0:
            raise AssertionError(f"Expected flops to be positive, but got {flops}")
        # pyrefly: ignore [no-matching-overload]
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        if not (log_flops >= 0 and log_flops < len(flop_headers)):
            raise AssertionError(
                f"Expected log_flops to be in range [0, {len(flop_headers)}), but got {log_flops}"
            )
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    if has_overload_names:
        add_column(name_column_width)
    for _ in headers[1 + has_overload_names :]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append("Input Shapes")
        add_column(shapes_column_width)

    if has_stack:
        headers.append("Source Location")
        add_column(src_column_width, text_dir="<")

    if with_flops:
        # Auto-scaling of flops header
        raw_flops = [evt.flops for evt in events if evt.flops > 0]
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append(f"Total {flops_header}")
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore[assignment]

    result = []

    def append(s):
        result.append(s)
        result.append("\n")  # Yes, newline after the end as well

    sum_self_cpu_time_total = 0
    sum_self_device_time_total = 0
    for evt in events:
        sum_self_cpu_time_total += evt.self_cpu_time_total
        if evt.device_type == DeviceType.CPU and evt.is_legacy:
            # in legacy profiler, kernel info is stored in cpu events
            sum_self_device_time_total += evt.self_device_time_total
        elif (
            evt.device_type
            in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.XPU,
            ]
            and not evt.is_user_annotation
        ):
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_device_time_total += evt.self_device_time_total

    # Actual printing
    if header is not None:
        append("=" * line_length)
        append(header)
    if top_level_events_only:
        append("=" * line_length)
        append("This report only display top-level ops statistics")
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)

    def trim_path(path, src_column_width):
        if len(path) > src_column_width:
            offset = len(path) - src_column_width
            path = path[offset:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    def override_time_unit(time_us, default_str, time_unit):
        US_IN_SECOND = 1000.0 * 1000.0
        US_IN_MS = 1000.0
        if time_unit == "s":
            return f"{time_us / US_IN_SECOND:.3f}s"
        elif time_unit == "ms":
            return f"{time_us / US_IN_MS:.3f}ms"
        elif time_unit == "us":
            return f"{time_us:.3f}us"
        else:
            return default_str

    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        name = evt.key
        if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
            name = name[: (max_name_column_width - 3)] + "..."

        evt.self_cpu_percent = _format_time_share(
            evt.self_cpu_time_total, sum_self_cpu_time_total
        )
        evt.total_cpu_percent = (
            _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total)
            if not evt.is_async
            else 0
        )

        row_values = [name]
        if has_overload_names:
            overload_name = evt.overload_name
            if (
                max_name_column_width is not None
                and len(overload_name) >= max_name_column_width - 3
            ):
                overload_name = overload_name[: (max_name_column_width - 3)] + "..."
            row_values += [overload_name]
        row_values += [
            # Self CPU total %, 0 for async events.
            evt.self_cpu_percent,
            override_time_unit(
                evt.self_cpu_time_total, evt.self_cpu_time_total_str, time_unit
            ),  # Self CPU total
            # CPU total %, 0 for async events.
            evt.total_cpu_percent,
            override_time_unit(
                evt.cpu_time_total, evt.cpu_time_total_str, time_unit
            ),  # CPU total
            override_time_unit(
                evt.cpu_time, evt.cpu_time_str, time_unit
            ),  # CPU time avg
        ]
        if has_device_time:
            evt.total_device_percent = _format_time_share(
                evt.self_device_time_total, sum_self_device_time_total
            )
            row_values.extend(
                [
                    override_time_unit(
                        evt.self_device_time_total,
                        evt.self_device_time_total_str,
                        time_unit,
                    ),
                    # device time total %
                    evt.total_device_percent,
                    override_time_unit(
                        evt.device_time_total, evt.device_time_total_str, time_unit
                    ),
                    override_time_unit(
                        evt.device_time, evt.device_time_str, time_unit
                    ),  # device time avg
                ]
            )
        if profile_memory:
            row_values.extend(
                [
                    # CPU Mem Total
                    _format_memory(evt.cpu_memory_usage),
                    # Self CPU Mem Total
                    _format_memory(evt.self_cpu_memory_usage),
                ]
            )
            if use_device and has_device_mem:
                row_values.extend(
                    [
                        # Device Mem Total
                        _format_memory(evt.device_memory_usage),
                        # Self Device Mem Total
                        _format_memory(evt.self_device_memory_usage),
                    ]
                )
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0:
                row_values.append("--")
            else:
                row_values.append(f"{evt.flops * flops_scale:8.3f}")  # type: ignore[possibly-undefined]
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:]:
                append(
                    row_format.format(
                        *(empty_headers + [trim_path(entry, src_column_width)])
                    )
                )
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append(
        f"Self CPU time total: {override_time_unit(sum_self_cpu_time_total, _format_time(sum_self_cpu_time_total), time_unit)}"
    )
    if has_device_time:
        append(
            f"Self {use_device.upper() if use_device is not None else 'None'} "
            f"time total: {override_time_unit(sum_self_device_time_total, _format_time(sum_self_device_time_total), time_unit)}"
        )
    return "".join(result)


# Collect all events with stack traces and format them canonically
def _canonicalize_profiler_events(events):
    """
    Extract and format all events with stack traces in a canonical way
    for deterministic testing.
    """
    events_with_traces = []

    for event in events:
        # Extract relevant fields
        event_name = event.get("name", "")
        node_name = event["args"].get("node_name", "")
        stack_trace = event["args"].get("stack_trace", "")

        # Get the last non-empty line of the stack trace that is actual source,
        # not a caret-marker line. Python 3.11+ appends "^^^^"/"~~~~" indicator
        # lines below the source when a FrameSummary has colno/end_colno set
        # (e.g. dynamo-generated stack traces); those lines must be skipped so
        # we still surface the source code line in canonicalized output.
        lines = [s.strip() for s in stack_trace.split("\n") if s.strip()]
        source_lines = [s for s in lines if not set(s).issubset({"^", "~", " "})]
        stack_trace = source_lines[-1] if source_lines else ""

        events_with_traces.append(
            {
                "event_name": event_name[:30],
                "node_name": node_name,
                "stack_trace": stack_trace,
                "start_time": event.get("ts", 0),
            }
        )

    # Sort by node_name for deterministic ordering
    events_with_traces.sort(key=lambda x: x["start_time"])

    # Format as a string
    lines: list[str] = []
    for evt in events_with_traces:
        lines.append(
            f"event={evt['event_name']} node={evt['node_name']} stack_trace={evt['stack_trace']}"
        )

    return "\n".join(lines)
<|endoftext|># mypy: allow-untyped-defs
"""
``torch.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import cast, overload

import torch
from torch import _vmap_internals
from torch.overrides import handle_torch_function, has_torch_function, is_tensor_like
from torch.types import _size, _TensorOrOptionalTensors, _TensorOrTensorsOrGradEdge

from . import forward_ad, functional, graph
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from .function import Function, NestedIOFunction
from .grad_mode import (
    _force_original_view_tracking,
    _unsafe_preserve_version_counter,
    enable_grad,
    enforce_grad_layout_policy,
    inference_mode,
    no_grad,
    set_grad_enabled,
    set_multithreading_enabled,
)
from .gradcheck import gradcheck, gradgradcheck
from .graph import _engine_run_backward
from .variable import Variable


__all__ = [
    "Variable",
    "Function",
    "backward",
    "grad_mode",
    "NestedIOFunction",
    "detect_anomaly",
    "enable_grad",
    "grad",
    "gradcheck",
    "gradgradcheck",
    "inference_mode",
    "no_grad",
    "set_detect_anomaly",
    "set_grad_enabled",
    "set_multithreading_enabled",
    "enforce_grad_layout_policy",
    "variable",
]

_OptionalTensor = torch.Tensor | None
_ShapeorNestedShape = _size | Sequence[_size] | torch.Tensor


def _calculate_shape(
    output: torch.Tensor | graph.GradientEdge,
    grad: torch.Tensor,
    is_grads_batched: bool,
) -> tuple[_ShapeorNestedShape, _ShapeorNestedShape]:
    # is_same_size ensures that both tensors are either nested or non nested
    # circular import
    from torch.nested._internal.nested_tensor import NestedTensor

    if isinstance(output, graph.GradientEdge):
        # We have already checked that we are not a C++ NestedTensor
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with GradientEdge")
        out_metadata = output.node._input_metadata[output.output_nr]
        return torch.Size(out_metadata.shape), grad.shape

    if output.is_nested and not isinstance(output, NestedTensor):
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with Nested Tensor.")
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()

        return out_shape, grad_shape

    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return reg_out_shape, reg_grad_shape


def _make_grads(
    outputs: Sequence[torch.Tensor] | Sequence[graph.GradientEdge],
    grads: Sequence[_OptionalTensor],
    is_grads_batched: bool,
) -> tuple[_OptionalTensor, ...]:
    new_grads: list[_OptionalTensor] = []

    for out, grad in zip(outputs, grads):
        # pyrefly: ignore [redundant-cast]
        out = cast(torch.Tensor | graph.GradientEdge, out)
        out_size = None
        out_device = None

        if isinstance(out, graph.GradientEdge):
            out_metadata = out.node._input_metadata[out.output_nr]
            out_size = torch.Size(out_metadata.shape)
            out_dtype = out_metadata.dtype
            out_device = out_metadata.device
            out_is_nested = out_metadata.is_nested_tensor
            if out_metadata.is_cpp_nested_tensor:
                raise RuntimeError(
                    "C++ NestedTensor are not supported with GradientEdge"
                )
            out_is_cpp_nested = False
        else:
            # circular import
            from torch.nested._internal.nested_tensor import NestedTensor

            if not isinstance(out, torch.Tensor):
                raise AssertionError("Expected output to be a torch.Tensor")
            out_dtype = out.dtype
            out_is_nested = out.is_nested
            out_is_cpp_nested = out_is_nested and not isinstance(out, NestedTensor)
            if not out_is_cpp_nested:
                out_size = out.shape

        if isinstance(grad, torch.Tensor):
            from torch.fx.experimental.symbolic_shapes import expect_true, sym_eq

            first_grad = grad if not is_grads_batched else grad[0]

            # TODO: We can remove this conditional once we uniformly use
            # singleton int to represent jagged dimension, so that size() call
            # on nested tensor works.
            if out_is_cpp_nested:
                if not isinstance(out, torch.Tensor):
                    raise AssertionError("Expected output to be a torch.Tensor.")
                shape_matches = torch.is_same_size(out, first_grad)
            else:
                # We need to do a regular size check, without going through
                # the operator, to be able to handle unbacked symints
                # (expect_true ensures we can deal with unbacked)
                if out_size is None:
                    raise AssertionError("Expected out_size to be set.")
                shape_matches = expect_true(sym_eq(out_size, first_grad.size()))

            if not shape_matches:
                out = cast(torch.Tensor | graph.GradientEdge, out)  # type: ignore[redundant-cast]
                out_shape, grad_shape = _calculate_shape(
                    out, first_grad, is_grads_batched
                )
                if is_grads_batched:
                    raise RuntimeError(
                        "If `is_grads_batched=True`, we interpret the first "
                        "dimension of each grad_output as the batch dimension. "
                        "The sizes of the remaining dimensions are expected to match "
                        "the shape of corresponding output, but a mismatch "
                        "was detected: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + ". "
                        "If you only want some tensors in `grad_output` to be considered "
                        "batched, consider using vmap."
                    )
                else:
                    raise RuntimeError(
                        "Mismatch in shape: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + "."
                    )
            if out_dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError(
                    "For complex Tensors, both grad_output and output"
                    " are required to have the same dtype."
                    " Mismatch in dtype: grad_output["
                    + str(grads.index(grad))
                    + "] has a dtype of "
                    + str(grad.dtype)
                    + " and output["
                    + str(outputs.index(out))
                    + "] has a dtype of "
                    + str(out_dtype)
                    + "."
                )
            new_grads.append(grad)
        elif grad is None:
            if isinstance(out, graph.GradientEdge) or out.requires_grad:  # type: ignore[attr-defined]
                if isinstance(out, graph.GradientEdge):
                    if out_size is None:
                        raise AssertionError("Expected out_size to be set.")
                    out_numel_is_1 = all(o == 1 for o in out_size)
                else:
                    if not isinstance(out, torch.Tensor):
                        raise AssertionError("Expected output to be a torch.Tensor")
                    out_numel_is_1 = out.numel() == 1
                if not out_numel_is_1:
                    raise RuntimeError(
                        "grad can be implicitly created only for scalar outputs"
                    )
                if not out_dtype.is_floating_point:
                    msg = (
                        "grad can be implicitly created only for real scalar outputs"
                        f" but got {out_dtype}"
                    )
                    raise RuntimeError(msg)
                if isinstance(out, graph.GradientEdge):
                    if out_size is None:
                        raise AssertionError("Expected out_size to be set.")
                    if out_device is None:
                        raise AssertionError("Expected out_device to be set.")
                    new_grads.append(
                        torch.ones(
                            out_size,
                            dtype=out_dtype,
                            device=out_device,
                        )
                    )
                else:
                    if not isinstance(out, torch.Tensor):
                        raise AssertionError("Expected output to be a torch.Tensor")
                    new_grads.append(
                        torch.ones_like(out, memory_format=torch.preserve_format)
                    )
            else:
                new_grads.append(None)
        else:
            raise TypeError(
                "gradients can be either Tensors or None, but got "
                + type(grad).__name__
            )
    return tuple(new_grads)


def _tensor_or_tensors_to_tuple(
    tensors: _TensorOrOptionalTensors | None, length: int
) -> tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    return tuple(tensors)


def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: _TensorOrOptionalTensors | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    grad_variables: _TensorOrOptionalTensors | None = None,
    inputs: _TensorOrTensorsOrGradEdge | dict[str, torch.Tensor] | None = None,
) -> None:
    r"""Compute the sum of gradients of given tensors with respect to graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the "vector"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding tensors (``None`` is an acceptable value for
    all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    ``.grad`` attributes or set them to ``None`` before calling it.
    See :ref:`Default gradient layouts<default-grad-layouts>`
    for details on the memory layout of accumulated gradients.

    .. note::
        Using this method with ``create_graph=True`` will create a reference cycle
        between the parameter and its gradient which can cause a memory leak.
        We recommend using ``autograd.grad`` when creating the graph to avoid this.
        If you have to use this function, make sure to reset the ``.grad`` fields of your
        parameters to ``None`` after use to break the cycle and avoid the leak.

    .. note::

        If you run any forward ops, create ``grad_tensors``, and/or call ``backward``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        When ``inputs`` are provided and a given input is not a leaf,
        the current implementation will call its grad_fn (even though it is not strictly needed to get these gradients).
        It is an implementation detail on which the user should not rely.
        See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

    Args:
        tensors (Sequence[Tensor] or Tensor or Sequence[GradientEdge] or GradientEdge): Tensors of which
            the derivative will be computed.
        grad_tensors (Sequence[Tensor or None] or Tensor, optional): The "vector" in
            the Jacobian-vector product, usually gradients w.r.t. each element of
            corresponding tensors. None values can be specified for scalar Tensors or
            ones that don't require grad. If a None value would be acceptable for all
            grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
        inputs (Sequence[Tensor] or Tensor or Sequence[GradientEdge] or dict[str, Tensor], optional):
            Inputs w.r.t. which the gradient will be accumulated into ``.grad``.
            All other Tensors will be ignored. If not provided, the gradient is
            accumulated into all the leaf Tensors that were used to compute the
            :attr:`tensors`. A dict of tensors (e.g.
            ``dict(model.named_parameters())``) is also accepted, in which case
            the values are used as the input tensors.
    """
    if torch._C._are_functorch_transforms_active():
        raise RuntimeError(
            "backward() called inside a functorch transform. This is not "
            "supported, please use functorch.grad or functorch.vjp instead "
            "or call backward() outside of functorch transforms."
        )

    if grad_variables is not None:
        warnings.warn(
            "`grad_variables` is deprecated. Use `grad_tensors` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError(
                "`grad_tensors` and `grad_variables` (deprecated) "
                "arguments both passed to `backward()`. Please only "
                "use `grad_tensors`."
            )

    inputs_tuple: tuple[torch.Tensor, ...] | tuple[graph.GradientEdge, ...]
    if inputs is None:
        inputs_tuple = ()
    elif isinstance(inputs, (torch.Tensor, graph.GradientEdge)):
        inputs_tuple = cast(
            tuple[torch.Tensor, ...] | tuple[graph.GradientEdge, ...], (inputs,)
        )
    elif type(inputs) is dict:
        # pyrefly: ignore [bad-argument-type]
        inputs_tuple = tuple(inputs.values())
    elif isinstance(inputs, Mapping):
        raise TypeError(
            f"`inputs` argument to `backward()` must be a dict, not {type(inputs).__name__}. "
            "Other Mapping types are not supported."
        )
    else:
        # pyrefly: ignore [bad-argument-type]
        inputs_tuple = tuple(inputs)
    if inputs is not None and len(inputs_tuple) == 0:
        raise RuntimeError("`inputs` argument to `backward()` cannot be empty.")

    if is_tensor_like(tensors) or isinstance(tensors, graph.GradientEdge):
        tensors = cast(tuple[torch.Tensor] | tuple[graph.GradientEdge], (tensors,))
    else:
        # pyrefly: ignore [bad-argument-type]
        tensors = tuple(tensors)

    # Check for __torch_function__ on tensors (similar to torch.autograd.grad)
    # This allows tensor subclasses to customize backward behavior
    t_tensors = tuple(t for t in tensors if is_tensor_like(t))
    t_inputs = tuple(t for t in inputs_tuple if is_tensor_like(t))
    overridable_args = t_tensors + t_inputs
    if has_torch_function(overridable_args):
        return handle_torch_function(
            backward,
            overridable_args,
            tensors,
            grad_tensors=grad_tensors,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs_tuple if inputs is not None else None,
        )

    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat the same comment below is that
    # some Python versions print out the first line of a multi-line function
    # calls in the traceback and some print out the last line
    _engine_run_backward(
        tensors,
        grad_tensors_,
        retain_graph,
        create_graph,
        inputs_tuple,
        allow_unreachable=True,
        accumulate_grad=True,
    )


@overload
def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: _TensorOrOptionalTensors | None = ...,
    retain_graph: bool | None = ...,
    create_graph: bool = ...,
    only_inputs: bool = ...,
    allow_unused: bool | None = ...,
    is_grads_batched: bool = ...,
    materialize_grads: bool = ...,
) -> tuple[torch.Tensor, ...]: ...


@overload
def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: dict[str, torch.Tensor],
    grad_outputs: _TensorOrOptionalTensors | None = ...,
    retain_graph: bool | None = ...,
    create_graph: bool = ...,
    only_inputs: bool = ...,
    allow_unused: bool | None = ...,
    is_grads_batched: bool = ...,
    materialize_grads: bool = ...,
) -> dict[str, torch.Tensor]: ...


def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge | dict[str, torch.Tensor],
    grad_outputs: _TensorOrOptionalTensors | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool | None = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
    r"""Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.

    Args:
        outputs (sequence of Tensor or GradientEdge): outputs of the differentiated function.
        inputs (sequence of Tensor or GradientEdge or dict[str, Tensor]): Inputs w.r.t. which
            the gradient will be returned (and not accumulated into ``.grad``).
            When a dict is provided (e.g. ``dict(model.named_parameters())``),
            the result is returned as a dict with matching keys.
        grad_outputs (sequence of [Tensor or None] or Tensor, optional): The "vector" in the
            vector-Jacobian product. Usually gradients w.r.t. each output. None values can be
            specified for scalar Tensors or ones that don't require grad. If a None value would be
            acceptable for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of each
            tensor in ``grad_outputs`` will be interpreted as the batch dimension.
            Instead of computing a single vector-Jacobian product, we compute a
            batch of vector-Jacobian products for each "vector" in the batch.
            We use the vmap prototype feature as the backend to vectorize calls
            to the autograd engine so that this computation can be performed in a
            single call. This should lead to performance improvements when compared
            to manually looping and performing backward multiple times. Note that
            due to this feature being experimental, there may be performance
            cliffs. Please use ``torch._C._debug_only_display_vmap_fallback_warnings(True)``
            to show any performance warnings and file an issue on github if warnings exist
            for your use case. Defaults to ``False``.
        materialize_grads (bool, optional): If ``True``, set the gradient for unused inputs
            to zero instead of None. This is useful when computing higher-order derivatives.
            If ``materialize_grads`` is ``True`` and ``allow_unused`` is ``False``, an error
            will be raised. Defaults to ``False``.

    """
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when materialize_grads=True, "
            "but got: allow_unused=False."
        )
    if allow_unused is None:
        allow_unused = materialize_grads
    if is_tensor_like(outputs) or isinstance(outputs, graph.GradientEdge):
        outputs = cast(
            Sequence[torch.Tensor] | Sequence[graph.GradientEdge], (outputs,)
        )
    else:
        # pyrefly: ignore [bad-argument-type]
        outputs = tuple(outputs)

    inputs_tuple: tuple[torch.Tensor, ...] | tuple[graph.GradientEdge, ...]
    if is_tensor_like(inputs) or isinstance(inputs, graph.GradientEdge):
        inputs_tuple = cast(
            tuple[torch.Tensor, ...] | tuple[graph.GradientEdge, ...], (inputs,)
        )
    elif type(inputs) is dict:
        # pyrefly: ignore [bad-argument-type]
        inputs_tuple = tuple(inputs.values())
    elif isinstance(inputs, Mapping):
        raise TypeError(
            f"`inputs` argument to `grad()` must be a dict, not {type(inputs).__name__}. "
            "Other Mapping types are not supported."
        )
    else:
        # pyrefly: ignore [bad-argument-type]
        inputs_tuple = tuple(inputs)
    if len(inputs_tuple) == 0:
        raise RuntimeError("`inputs` argument to `grad()` cannot be empty.")

    t_outputs = tuple(i for i in outputs if is_tensor_like(i))
    t_inputs = tuple(i for i in inputs_tuple if is_tensor_like(i))
    overridable_args = t_outputs + t_inputs
    if has_torch_function(overridable_args):
        result_tuple = handle_torch_function(
            grad,
            overridable_args,
            outputs,
            inputs_tuple,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
            is_grads_batched=is_grads_batched,
            materialize_grads=materialize_grads,
        )
        if type(inputs) is dict:
            return dict(zip(inputs.keys(), result_tuple, strict=True))
        return result_tuple

    if not only_inputs:
        warnings.warn(
            "only_inputs argument is deprecated and is ignored now "
            "(defaults to True). To accumulate gradient for other "
            "parts of the graph, please use torch.autograd.backward.",
            FutureWarning,
            stacklevel=2,
        )

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(outputs))
    grad_outputs_ = _make_grads(
        outputs, grad_outputs_, is_grads_batched=is_grads_batched
    )

    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat the same comment several times below is because
    # some Python versions print out the first line of multi-line function
    # calls in the traceback and some print out the last line
    if is_grads_batched:

        def vjp(gO):
            return _engine_run_backward(
                outputs,
                gO,
                retain_graph,
                create_graph,
                inputs_tuple,
                allow_unused,
                accumulate_grad=False,
            )

        result = _vmap_internals._vmap(vjp, 0, 0, allow_none_pass_through=True)(
            grad_outputs_
        )
    else:
        result = _engine_run_backward(
            outputs,
            grad_outputs_,
            retain_graph,
            create_graph,
            inputs_tuple,
            allow_unused,
            accumulate_grad=False,
        )
    if materialize_grads:
        if any(
            r is None and not is_tensor_like(inp)
            for r, inp in zip(result, inputs_tuple, strict=True)
        ):
            raise RuntimeError(
                "materialize_grads cannot be used when the given input is a GradientEdge"
            )
        result = tuple(
            r if r is not None else torch.zeros_like(inp, requires_grad=create_graph)
            for (r, inp) in zip(
                result,
                cast(tuple[torch.Tensor, ...], inputs_tuple),
                strict=True,
            )
        )
    if type(inputs) is dict:
        return dict(zip(inputs.keys(), result, strict=True))
    return result


# This function applies in case of gradient checkpointing for memory
# optimization. Currently, gradient checkpointing is supported only if the
# execution engine is invoked through torch.autograd.backward() and its
# inputs argument is not passed. It is not supported for torch.autograd.grad().
# This is because if inputs are specified, the gradient won't be calculated for
# anything else e.g. model parameters like weights, bias etc.
#
# This function returns whether the checkpointing is valid i.e. torch.autograd.backward
# or not i.e. torch.autograd.grad. The implementation works by maintaining a thread
# local variable in torch/csrc/autograd/engine.cpp which looks at the NodeTask
# in the stack and before a NodeTask is executed in evaluate_function, it
# checks for whether reentrant backwards is imperative or not.
# See https://github.com/pytorch/pytorch/pull/4594 for more discussion/context
def _is_checkpoint_valid():
    return Variable._execution_engine.is_checkpoint_valid()


def variable(*args, **kwargs):
    raise RuntimeError(
        "torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead"
    )


# Monkey patching variable.Variable to fix FX codegen. FX generates a call by roughly doing
# f"{fn.__module__}.{fn.__name__}(...). This yields torch.autograd.variable.Variable(...) in the
# output of an FX graph.  Unfortunately the module name torch.autograd.variable is shadowed by the
# deprecated function - variable(...).
variable.Variable = Variable  # type: ignore[attr-defined]

if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")

# Import all native method/classes
from torch._C._autograd import (
    _add_metadata_json,
    _disable_profiler,
    _disable_profiler_legacy,
    _enable_profiler,
    _enable_profiler_legacy,
    _enable_record_function,
    _get_sequence_nr,
    _is_kineto_stopped,
    _kineto_step,
    _KinetoEvent,
    _pop_saved_tensors_default_hooks,
    _prepare_profiler,
    _profiler_enabled,
    _ProfilerResult,
    _push_saved_tensors_default_hooks,
    _record_function_with_args_enter,
    _record_function_with_args_exit,
    _set_empty_test_observer,
    _supported_activities,
    _toggle_collection_dynamic,
    DeviceType,
    kineto_available,
    ProfilerEvent,
    SavedTensor,
)
from torch._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState

from . import profiler


def _register_py_tensor_class_for_device(device, cls):
    if not isinstance(cls, type):
        raise RuntimeError("cls isn't a typeinfo object")
    torch._C._register_py_class_for_device(device, cls)


is_multithreading_enabled = torch._C._is_multithreading_enabled
torch._C._add_docstr(
    is_multithreading_enabled, "Returns True if multithreading is currently enabled."
)

is_view_replay_enabled = torch._C._is_view_replay_enabled
torch._C._add_docstr(
    is_view_replay_enabled, "Returns True if view-replay is currently enabled."
)
<|endoftext|># mypy: allow-untyped-defs
import itertools
import warnings
from typing_extensions import deprecated

import torch
import torch.cuda
from torch.autograd import (
    _disable_profiler_legacy,
    _enable_profiler_legacy,
    DeviceType,
    ProfilerConfig,
    ProfilerState,
)
from torch.autograd.profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
)


__all__ = ["profile"]


@deprecated(
    "`torch.autograd.profiler_legacy.profile` is deprecated and will be removed in a future release. "
    "Please use `torch.profiler` instead.",
    category=None,  # TODO: change to `FutureWarning`
)
class profile:
    """DEPRECATED: use torch.profiler instead."""

    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        self.function_events = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules

        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available, disabling CUDA profiling",
                stacklevel=2,
            )
            self.use_cuda = False

        if self.use_cuda:
            self.profiler_kind = ProfilerState.CUDA
        else:
            self.profiler_kind = ProfilerState.CPU

    def config(self):
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            # avoid exposing _ExperimentalConfig this in legacy public API
            torch._C._profiler._ExperimentalConfig(),
        )

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self.entered = True
        self._start_trace()
        return self

    def _start_trace(self):
        _enable_profiler_legacy(self.config())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()

        records = _disable_profiler_legacy()
        parsed_results = _parse_legacy_records(records)
        # pyrefly: ignore [bad-assignment]
        self.function_events = EventList(
            parsed_results,
            use_device="cuda" if self.use_cuda else None,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        # pyrefly: ignore [missing-attribute]
        self.function_events._build_tree()
        return False

    def __repr__(self):
        if self.function_events is None:
            return "<unfinished profiler_legacy.profile>"
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return "<unfinished profile.profiler_legacy.profile>"
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.export_chrome_trace(path)

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        if not self.with_stack:
            raise AssertionError("export_stacks() requires with_stack=True")
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Return CPU time as the sum of self times across all events."""
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.self_cpu_time_total


def _parse_legacy_records(thread_records):
    def _get_record_key(record):
        """Return a tuple for correlating start and end records in `_parse_legacy_records`."""
        return (record.handle(), record.node_id())

    start_record = None
    functions = []

    # '__start_profile' is not guaranteed to be first, so we must find it here
    for record in itertools.chain.from_iterable(thread_records):
        name = record.name()
        if start_record is None and name == "__start_profile":
            start_record = record

    if start_record is None or start_record.is_remote():
        raise AssertionError("Expected a valid local start_record")

    for thread_record_list in thread_records:
        # accumulated memory allocations per handle
        cpu_memory_allocs = {}
        cuda_memory_allocs = {}
        # ranges per handle
        range_starts = {}

        filtered_handles = set()
        prev_record = None
        for record in thread_record_list:
            record_key = _get_record_key(record)
            if _filter_name(record.name()) or record_key in filtered_handles:
                filtered_handles.add(record_key)
                continue

            if record.kind() == "push":
                # workaround to reduce double logging from operator
                # wrappers and redispatch
                if prev_record is not None:
                    duplicate = (
                        prev_record.name() == record.name()
                        and prev_record.kind() == record.kind()
                        and prev_record.node_id() == record.node_id()
                    )
                    if duplicate:
                        filtered_handles.add(record_key)
                        continue

                range_starts[record_key] = record
                cpu_memory_allocs[record_key] = 0
                cuda_memory_allocs[record_key] = 0
            elif record.kind() == "pop":
                if record_key not in range_starts:
                    raise AssertionError(
                        f"Expected record with key {record_key} to exist in range_starts. "
                        "This means that the pop event did not have a corresponding push."
                    )

                start = range_starts[record_key]

                cpu_memory_usage = cpu_memory_allocs[record_key]
                cuda_memory_usage = cuda_memory_allocs[record_key]
                is_async = start.is_async() or (start.thread_id() != record.thread_id())
                is_remote_event = record.is_remote()
                start_flops = start.flops()

                fe = FunctionEvent(
                    id=record.handle(),
                    node_id=record.node_id(),
                    name=_rewrite_name(name=start.name(), with_wildcard=True),
                    trace_name=_rewrite_name(name=start.name(), with_wildcard=False),
                    thread=start.thread_id(),
                    start_us=start_record.cpu_elapsed_us(start),
                    end_us=start_record.cpu_elapsed_us(record),
                    fwd_thread=start.fwd_thread_id(),
                    input_shapes=start.shapes(),
                    stack=[
                        entry for entry in start.stack() if _filter_stack_entry(entry)
                    ],
                    scope=start.scope(),
                    use_device="cuda" if start.has_cuda() else None,
                    cpu_memory_usage=cpu_memory_usage,
                    device_memory_usage=cuda_memory_usage,
                    is_async=is_async,
                    is_remote=is_remote_event,
                    sequence_nr=start.sequence_nr(),
                    device_type=DeviceType.CPU,
                    is_legacy=True,
                    flops=start_flops,
                )
                # note: async events have only cpu total time
                if not is_async and start.has_cuda():
                    duration = start.cuda_elapsed_us(record)
                    if duration > 0:
                        fe.append_kernel(start.name(), start.device(), duration)
                functions.append(fe)
                del range_starts[record_key]
                del cpu_memory_allocs[record_key]
                del cuda_memory_allocs[record_key]
            elif record.kind() == "memory_alloc":
                num_open_handles_cpu = len(cpu_memory_allocs)
                num_open_handles_cuda = len(cuda_memory_allocs)
                if num_open_handles_cpu != num_open_handles_cuda:
                    raise AssertionError(
                        f"Expected CPU and CUDA memory allocation handles to match, "
                        f"but got {num_open_handles_cpu} CPU and {num_open_handles_cuda} CUDA"
                    )
                for handle in cpu_memory_allocs:
                    cpu_memory_allocs[handle] += record.cpu_memory_usage()
                for handle in cuda_memory_allocs:
                    cuda_memory_allocs[handle] += record.cuda_memory_usage()
                if num_open_handles_cpu == 0:
                    # output event as a top-level memory event
                    fe = FunctionEvent(
                        id=0,
                        name=MEMORY_EVENT_NAME,
                        trace_name=None,
                        thread=0,
                        start_us=0,
                        end_us=0,
                        stack=[],
                        cpu_memory_usage=record.cpu_memory_usage(),
                        device_memory_usage=record.cuda_memory_usage(),
                        is_legacy=True,
                    )
                    functions.append(fe)
            prev_record = record

    # Sort functions by start time then by end time ascending.
    # This ensures that--in the case of nested events which
    # have the same start time (which may happen due to the
    # granularity of the given clock tick)--we always show
    # the outermost nested call first. This adds stability
    # in how FunctionEvents appear
    functions.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
    return functions
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch._vmap_internals import _vmap

from . import forward_ad as fwAD


__all__ = ["vjp", "jvp", "jacobian", "hessian", "hvp", "vhp"]

# Utility functions


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _as_tuple(inp, arg_name=None, fn_name=None):
    # Ensures that inp is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not isinstance(el, torch.Tensor):
            if is_inp_tuple:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" value at index {i} has type {type(el)}."
                )
            else:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" given {arg_name} has type {type(el)}."
                )

    return is_inp_tuple, inp


def _tuple_postprocess(res, to_unpack):
    # Unpacks a potentially nested tuple of Tensors
    # to_unpack should be a single boolean or a tuple of two booleans.
    # It is used to:
    # - invert _as_tuple when res should match the inp given to _as_tuple
    # - optionally remove nesting of two tuples created by multiple calls to _as_tuple
    if isinstance(to_unpack, tuple):
        if len(to_unpack) != 2:
            raise AssertionError("Expected to_unpack tuple to have exactly 2 elements")
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        if not to_unpack[0]:
            res = res[0]
    else:
        if not to_unpack:
            res = res[0]
    return res


def _grad_preprocess(inputs, create_graph, need_graph):
    # Preprocess the inputs to make sure they require gradient
    # inputs is a tuple of Tensors to preprocess
    # create_graph specifies if the user wants gradients to flow back to the Tensors in inputs
    # need_graph specifies if we internally want gradients to flow back to the Tensors in res
    # Note that we *always* create a new Tensor object to be able to see the difference between
    # inputs given as arguments and the same Tensors automatically captured by the user function.
    # Check this issue for more details on how that can happen: https://github.com/pytorch/pytorch/issues/32576
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            # Create at least a new Tensor object in a differentiable way
            if not inp.is_sparse:
                # Use .view_as() to get a shallow copy
                res.append(inp.view_as(inp))
            else:
                # We cannot use view for sparse Tensors so we clone
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)


def _grad_postprocess(inputs, create_graph):
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
    # request it.
    if isinstance(inputs[0], torch.Tensor):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)


def _validate_v(v, other, is_other_tuple):
    # This assumes that other is the correct shape, and v should match
    # Both are assumed to be tuples of Tensors
    if len(other) != len(v):
        if is_other_tuple:
            raise RuntimeError(
                f"v is a tuple of invalid length: should be {len(other)} but got {len(v)}."
            )
        else:
            raise RuntimeError("The given v should contain a single Tensor.")

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = ""
            if is_other_tuple:
                prepend = f"Entry {idx} in "
            raise RuntimeError(
                f"{prepend}v has invalid size: should be {el_other.size()} but got {el_v.size()}."
            )


def _check_requires_grad(inputs, input_type, strict):
    # Used to make all the necessary checks to raise nice errors in strict mode.
    if not strict:
        return

    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            # This can only be reached for grad_inputs.
            raise RuntimeError(
                f"The output of the user-provided function is independent of input {i}."
                " This is not allowed in strict mode."
            )
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError(
                    f"The hessian of the user-provided function with respect to input {i}"
                    " is independent of the input. This is not allowed in strict mode."
                    " You should ensure that your function is thrice differentiable and that"
                    " the hessian depends on the inputs."
                )
            elif input_type == "jacobian":
                raise RuntimeError(
                    "While computing the hessian, found that the jacobian of the user-provided"
                    f" function with respect to input {i} is independent of the input. This is not"
                    " allowed in strict mode. You should ensure that your function is twice"
                    " differentiable and that the jacobian depends on the inputs (this would be"
                    " violated by a linear function for example)."
                )
            elif input_type == "grad_inputs":
                raise RuntimeError(
                    f"The gradient with respect to input {i} is independent of the inputs of the"
                    " user-provided function. This is not allowed in strict mode."
                )
            else:
                raise RuntimeError(
                    f"Output {i} of the user-provided function does not require gradients."
                    " The outputs must be computed in a differentiable manner from the input"
                    " when running in strict mode."
                )


def _autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    create_graph=False,
    retain_graph=None,
    is_grads_batched=False,
):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    if not isinstance(outputs, tuple):
        raise AssertionError("Expected outputs to be a tuple")
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    if not isinstance(grad_outputs, tuple):
        raise AssertionError("Expected grad_outputs to be a tuple")
    if len(outputs) != len(grad_outputs):
        raise AssertionError(
            f"Expected outputs and grad_outputs to have the same length, "
            f"but got {len(outputs)} and {len(grad_outputs)}"
        )

    new_outputs: tuple[torch.Tensor, ...] = ()
    new_grad_outputs: tuple[torch.Tensor, ...] = ()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            # pyrefly: ignore [bad-assignment]
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return torch.autograd.grad(
            new_outputs,
            inputs,
            new_grad_outputs,
            allow_unused=True,
            create_graph=create_graph,
            retain_graph=retain_graph,
            is_grads_batched=is_grads_batched,
        )


def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    # Used to detect None in the grads and depending on the flags, either replace them
    # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
    # strict and create graph allow us to detect when it is appropriate to raise an error
    # stage gives us information of which backward call we consider to give good error message
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")

    res: tuple[torch.Tensor, ...] = ()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError(
                        "The output of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                elif stage == "back_trick":
                    raise RuntimeError(
                        f"The gradient with respect to the input is independent of entry {i}"
                        " in the grad_outputs when using the double backward trick to compute"
                        " forward mode gradients. This is not allowed in strict mode."
                    )
                elif stage == "double_back":
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"entry {i} in the grad_jacobian. This is not allowed in strict "
                        "mode as it prevents from using the double backward trick to "
                        "replace forward mode AD."
                    )

            grads_i = torch.zeros_like(refs[i])
        else:
            if strict and create_graph and not grads_i.requires_grad:
                if "double" not in stage:
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )

        res += (grads_i,)

    return res


# Public API


def vjp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between a vector ``v`` and the Jacobian of the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vjp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vjp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the "
                    "user-provided function returns "
                    "a single Tensor with a single element."
                )

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vjp, is_inputs_tuple
    )


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the Jacobian of the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian
            vector product is computed. Must be the same size as the input of
            ``func``. This argument is optional when the input to ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            jvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the output.

    Note:
        ``autograd.functional.jvp`` computes the jvp by using the backward of
        the backward (sometimes called the double backwards trick). This is not
        the most performant way of computing the jvp. Please consider using
        :func:`torch.func.jvp` or the
        :ref:`low-level forward-mode AD API <forward-mode-ad>` instead.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4, 4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> jvp(exp_reducer, inputs, v, create_graph=True)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SumBackward1>),
         tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SqueezeBackward1>))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.ones(2), torch.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to "
                    "the user-provided function is a single Tensor "
                    "with a single element."
                )

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)
        # The backward is linear so the value of grad_outputs is not important as
        # it won't appear in the double backward graph. We only need to ensure that
        # it does not contain inf or nan.
        grad_outputs = tuple(
            torch.zeros_like(out, requires_grad=True) for out in outputs
        )

        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    if create_graph:
        with torch.enable_grad():
            grad_res = _autograd_grad(
                grad_inputs, grad_outputs, v, create_graph=create_graph
            )
            jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        grad_res = _autograd_grad(
            grad_inputs, grad_outputs, v, create_graph=create_graph
        )
        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        jvp, is_outputs_tuple
    )


def _construct_standard_basis_for(
    tensors: tuple[torch.Tensor, ...], tensor_numels: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    # This function:
    # - constructs a N=sum(tensor_numels) standard basis. i.e. an NxN identity matrix.
    # - Splits the identity matrix into chunks with each chunk size determined by `tensor_numels`.
    # - Each chunk corresponds to one tensor. The chunk has the same dtype and
    #   device as the tensor
    #
    # For example, with tensor_numels = [1, 2, 1], this function returns:
    # ( tensor([[1],     tensor([[0, 0],      tensor([[0],
    #           [0],             [1, 0],              [0],
    #           [0],             [0, 1],              [0],
    #           [0]])  ,         [0, 0]])  ,          [1]])  )
    #
    # Precondition: tensor_numels == tuple(tensor.numel() for tensor in tensors)
    # Precondition: tensors always has at least one element.
    #
    # See NOTE: [Computing jacobian with vmap and grad for multiple tensors]
    # for context behind this function. All the pre-conditions are guarded for
    # in torch.autograd.functional.jacobian.
    if len(tensors) != len(tensor_numels):
        raise AssertionError(
            f"Expected tensors and tensor_numels to have the same length, "
            f"but got {len(tensors)} and {len(tensor_numels)}"
        )
    if len(tensors) == 0:
        raise AssertionError("Expected at least one tensor")
    total_numel = sum(tensor_numels)
    chunks = tuple(
        tensor.new_zeros(total_numel, tensor_numel)
        for tensor, tensor_numel in zip(tensors, tensor_numels)
    )
    diag_start_idx = 0
    for chunk, numel in zip(chunks, tensor_numels):
        chunk.diagonal(diag_start_idx).fill_(1)
        diag_start_idx -= numel
    return chunks


def _jacfwd(func, inputs, strict=False, vectorize=False):
    if strict:
        raise RuntimeError(
            "torch.autograd.functional.jacobian: `strict=True` "
            'and `strategy="forward-mode"` are not supported together (yet). '
            "Please either set `strict=False` or "
            '`strategy="reverse-mode"`.'
        )
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    output_info = []

    if vectorize:
        # See NOTE: [Computing jacobian with vmap and grad for multiple outputs]
        input_numels = tuple(input.numel() for input in inputs)

        # Step 1: Prepare tangents
        tangents = _construct_standard_basis_for(inputs, input_numels)

        # Step 2: Compute vmap over computation with dual tensors
        def jvp(tangents):
            with fwAD.dual_level():
                dual_inputs = tuple(
                    fwAD.make_dual(input, tangent.view_as(input))
                    for input, tangent in zip(inputs, tangents)
                )
                _is_outputs_tuple, dual_outputs = _as_tuple(
                    func(*dual_inputs), "outputs"
                )
                output_info.append(_is_outputs_tuple)
                jv = []
                primal_outs = []
                for dual_out in dual_outputs:
                    primal, tangent = fwAD.unpack_dual(dual_out)
                    primal_outs.append(primal)
                    if tangent is not None:
                        jv.append(tangent)
                    else:
                        jv.append(torch.zeros_like(primal))
                output_info.append(primal_outs)
                return tuple(jv)

        outputs_before_split = _vmap(jvp)(tangents)
