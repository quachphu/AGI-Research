def deactivate(session: Optional[int] = None, flushing: bool = False) -> None:
    """
    Stop the specified session.
    The profiling session's data will still be in the memory, but no more data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to None (all sessions)
        flushing (bool): Whether to flush the profiling data before deactivating. Defaults to True.

    Returns:
        None
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be deactivated when running from the command line.")

    HookManager.deactivate(session)

    if session is None:
        libproton.deactivate_all(flushing)
    else:
        libproton.deactivate(session, flushing)


def finalize(session: Optional[int] = None, output_format: Optional[str] = "") -> None:
    """
    Finalizes a profiling session.
    Flush and write the profiling data to the file specified by the session name.

    Args:
        session (int, optional): The session ID to finalize. If None, all sessions are finalized. Defaults to None.
        output_format (str, optional): The output format for the profiling results.
                                       Available options are ["hatchet", "hatchet_msgpack", "chrome_trace"].

    Returns:
        None
    """
    HookManager.unregister(session)

    if session is None:
        flags.profiling_on = False
        libproton.finalize_all(output_format)
    else:
        if flags.command_line and session != 0:
            raise ValueError("Only one session can be finalized when running from the command line.")
        libproton.finalize(session, output_format)


def _profiling(
    func,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    hook: Optional[Union[str, Hook]] = None,
):
    """
    Context manager for profiling. Internally use only.

    Args:
        See start() for the arguments.

    Returns:
        wrapper (function): The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        session = start(name, context=context, data=data, backend=backend, mode=mode, hook=hook)
        ret = func(*args, **kwargs)
        deactivate(session)
        return ret

    return wrapper


def profile(
    func=None,
    *,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    hook: Optional[Union[str, Hook]] = None,
):
    """
    Decorator for profiling.

    Usage:

    ```python
    @proton.profile
    def foo():
        pass
    ```

    Args:
        See start() for the arguments.

    Returns:
        decorator (function): The decorator function.
    """
    if func is None:
        # It's being used with parentheses, so return a decorator
        def decorator(f):
            return _profiling(f, name=name, context=context, data=data, backend=backend, mode=mode, hook=hook)

        return decorator
    else:
        # It's being used without parentheses, so apply the decorator directly
        return _profiling(func, name=name, context=context, data=data, backend=backend, mode=mode, hook=hook)
<|endoftext|>from typing import Optional
from triton._C.libproton import proton as libproton  # type: ignore
import json as json
from .flags import flags


def get(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
    Returns:
        str: The profiling data in JSON format.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return json.loads(libproton.get_data(session, phase))


def get_msgpack(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        bytes: The profiling data encoded with MessagePack.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return libproton.get_data_msgpack(session, phase)


def advance_phase(session: Optional[int] = 0) -> Optional[int]:
    """
    Advances the profiling phase for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        Optional[int]: The next phase number after advancing.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can advance phase when running from the command line.")
    return libproton.advance_data_phase(session)


def is_phase_complete(session: Optional[int] = 0, phase: int = 0) -> bool:
    """
    Checks if the profiling data for a given session and phase is complete.

    A "complete" phase is safe to read/clear because all device-side records for
    the phase have been flushed to the host and the phase will no longer receive
    new records.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to check. Defaults to 0.

    Returns:
        bool: True if the phase data is complete, False otherwise.
    """
    if session is None:
        return False
    if flags.command_line and session != 0:
        raise ValueError("Only one session can check phase completion status when running from the command line.")
    return libproton.is_data_phase_complete(session, phase)


def clear(
    session: Optional[int] = 0,
    phase: int = 0,
    clear_up_to_phase: bool = False,
) -> None:
    """
    Clears profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to clear. Defaults to 0.
        clear_up_to_phase (bool): If True, clear all phases up to and including `phase`.
    """
    if session is None:
        return
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be cleared when running from the command line.")
    libproton.clear_data(session, phase, clear_up_to_phase)
<|endoftext|># ruff: noqa


# Point the C++ backend at custom rocprofiler-sdk include/lib paths before
# libproton.so is loaded. Explicit Proton knobs win; TheRock environments can
# still discover the SDK library through _rocm_sdk_core.
def _ensure_rocprofiler_sdk_env():
    import os
    import triton

    for key, value in (
        ("TRITON_ROCPROFILER_SDK_INCLUDE_PATH", triton.knobs.proton.rocprofiler_sdk_include_path),
        ("TRITON_ROCPROFILER_SDK_LIB_PATH", triton.knobs.proton.rocprofiler_sdk_lib_path),
    ):
        if not os.environ.get(key, None) and value is not None:
            triton.knobs.setenv(key, value)

    if not os.environ.get("TRITON_ROCPROFILER_SDK_LIB_PATH", None):
        try:
            import _rocm_sdk_core
            lib_dir = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib")
            if os.path.isdir(lib_dir):
                triton.knobs.proton.rocprofiler_sdk_lib_path = lib_dir
        except ImportError:
            pass


_ensure_rocprofiler_sdk_env()

from .scope import scope, cpu_timed_scope, enter_scope, exit_scope
from .state import state, enter_state, exit_state, metadata_state
from .profile import (
    start,
    activate,
    deactivate,
    finalize,
    profile,
    DEFAULT_PROFILE_NAME,
)
from . import context, specs, mode, data
<|endoftext|>import threading
import time
from functools import wraps
from typing import Optional, Union, Any, Sequence

from .flags import flags
from .metric import transform_tensor_metrics, set_metric_kernels
from .state import metadata_state
from triton._C.libproton import proton as libproton

thread_local_scopes = threading.local()

MetricValueType = Union[float, int, Sequence[float], Sequence[int]]


class scope:
    """
    A context manager and decorator for entering and exiting a scope.

    Usage:
        context manager:
        ```python
        with proton.scope("test0", {metric_name: metric_value}):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.scope("test0", {metric_name: metric_value})
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the scope.
        metrics (dict[str, float], optional): The metrics of the scope. Default is None.
    """

    def __init__(self, name: str, metrics: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.metrics = metrics
        self.id = None

    def _enter_scope(self):
        if not flags.profiling_on:
            return
        self.id = libproton.record_scope()
        libproton.enter_scope(self.id, self.name)
        if self.metrics:
            with metadata_state():
                set_metric_kernels()
                libproton.add_metrics(self.id, *transform_tensor_metrics(self.metrics))

    def _exit_scope(self):
        if not flags.profiling_on or self.id is None:
            return
        libproton.exit_scope(self.id, self.name)

    def __enter__(self):
        self._enter_scope()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_scope()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            self._enter_scope()
            try:
                return func(*args, **kwargs)
            finally:
                self._exit_scope()

        return wrapper


class cpu_timed_scope(scope):
    """
    A scope that measures elapsed time (cpu_time).

    Args:
        name (str): The name of the scope.
        metrics (dict[str, float], optional): Additional metrics to add. Default is None.
    """

    def __init__(self, name: str, metrics: Optional[dict[str, Any]] = None) -> None:
        super().__init__(name, metrics)
        self.start_time = None
        if metrics and "cpu_time" in metrics:
            raise ValueError("The metric name 'cpu_time' is reserved.")

    def _enter_scope(self):
        if not flags.profiling_on:
            return
        self.start_time = time.perf_counter_ns()
        super()._enter_scope()

    def _exit_scope(self):
        if not flags.profiling_on:
            return
        if self.start_time is not None:
            cpu_time = time.perf_counter_ns() - self.start_time
            libproton.add_metrics(self.id, {"cpu_time (ns)(exc)": cpu_time})
        super()._exit_scope()


def enter_scope(name: str, *, metrics: Optional[dict[str, Any]] = None) -> Optional[int]:
    if not flags.profiling_on:
        return None
    id = libproton.record_scope()
    thread_local_scopes.scopes = getattr(thread_local_scopes, "scopes", [])
    thread_local_scopes.scopes.append((id, name))
    libproton.enter_scope(id, name)
    if metrics:
        with metadata_state():
            set_metric_kernels()
            libproton.add_metrics(id, *transform_tensor_metrics(metrics))
    return id


def exit_scope(name: Optional[str] = None, *, metrics: Optional[dict[str, Any]] = None) -> Optional[int]:
    # `name` is an optional argument here, only to match the counterpart in enter_scope to make the API consistent with `proton.language.exit_scope`
    if not flags.profiling_on:
        return None
    id, popped_name = thread_local_scopes.scopes.pop()
    if name and name != popped_name:
        raise ValueError(f"Scope name mismatch: {name} != {popped_name}")
    elif not name:
        name = popped_name
    if metrics:
        with metadata_state():
            set_metric_kernels()
            libproton.add_metrics(id, *transform_tensor_metrics(metrics))
    libproton.exit_scope(id, name)
    return id
<|endoftext|>from typing import Any
from triton._C.libproton import proton as libproton
import triton.runtime.driver as driver
import triton.language as tl
import triton
from triton import MockTensor


@triton.jit
def tensor_metric_kernel(device_ptr, device_offset_ptr, size: tl.uint64, seq_id: tl.uint64, metric_value_ptr,
                         metric_value_size: tl.uint64):
    # Record layout is {seq_id, <metric_values>}.
    BLOCK_SIZE: tl.constexpr = 128
    record_size = metric_value_size + 1
    # Reserve the full record atomically so replayed graph streams can append
    # concurrently.
    device_offset = tl.atomic_add(device_offset_ptr, record_size, sem="relaxed") % size
    tl.store(device_ptr + device_offset, seq_id)
    device_offset = (device_offset + 1) % size
    num_iters = tl.cdiv(metric_value_size, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_iters):
        cur_offsets = offsets + i * BLOCK_SIZE
        mask = cur_offsets < metric_value_size
        metric_value = tl.load(metric_value_ptr + cur_offsets, mask=mask)
        tl.store(device_ptr + (device_offset + cur_offsets) % size, metric_value, mask=mask)


@triton.jit
def scalar_metric_kernel(device_ptr, device_offset_ptr, size: tl.uint64, seq_id: tl.uint64, metric_value: tl.uint64):
    # Record layout is {seq_id, metric_value}.
    device_offset = tl.atomic_add(device_offset_ptr, 2, sem="relaxed") % size
    tl.store(device_ptr + device_offset, seq_id)
    device_offset = (device_offset + 1) % size
    tl.store(device_ptr + device_offset, metric_value)


def _get_kernel(kernel_fn, *args):
    kernel = kernel_fn.warmup(*args, grid=(1, ), num_warps=1)
    kernel._init_handles()
    target = getattr(kernel.metadata, "target", None)
    warp_size = getattr(target, "warp_size", None)
    if warp_size is None:
        warp_size = driver.active.get_current_target().warp_size
    num_threads = kernel.metadata.num_warps * warp_size
    return kernel.function, num_threads, kernel.metadata.shared


def set_metric_kernels():
    mock_ptr = MockTensor(tl.uint64)
    mock_seq_id = 0
    mock_size = 1
    mock_metric_value_size = 1
    tensor_metric_kernel_fn, tensor_metric_kernel_num_threads, tensor_metric_kernel_shared = _get_kernel(
        tensor_metric_kernel,
        mock_ptr,
        mock_ptr,
        mock_size,
        mock_seq_id,
        mock_ptr,
        mock_metric_value_size,
    )
    scalar_metric_kernel_fn, scalar_metric_kernel_num_threads, scalar_metric_kernel_shared = _get_kernel(
        scalar_metric_kernel,
        mock_ptr,
        mock_ptr,
        mock_size,
        mock_seq_id,
        mock_metric_value_size,
    )
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    libproton.set_metric_kernels(
        tensor_metric_kernel_fn,
        scalar_metric_kernel_fn,
        stream,
        tensor_metric_kernel_num_threads,
        tensor_metric_kernel_shared,
        scalar_metric_kernel_num_threads,
        scalar_metric_kernel_shared,
    )


class _TensorMetric(libproton.TensorMetric):

    def __init__(self, value, metric_type_index):
        super().__init__(value.data_ptr(), metric_type_index, value.numel())
        # Hold a reference to the backing tensor so its device memory stays alive.
        self._value = value


FLOPS_WIDTHS = (8, 16, 32, 64)
FLOPS_DTYPES = {
    **{"flops": (float, libproton.metric_type_double_index, libproton.metric_type_vector_double_index)},
    **{
        f"flops{width}": (float, libproton.metric_type_double_index, libproton.metric_type_vector_double_index)
        for width in FLOPS_WIDTHS
    },
}
ROOFLINE_DTYPES = {
    **FLOPS_DTYPES,
    "bytes": (int, libproton.metric_type_int64_index, libproton.metric_type_vector_int64_index),
}


def _scalar_or_vector_value(value: Any, convert: type) -> Any:
    if hasattr(value, "data_ptr"):
        value = value.tolist() if value.numel() > 1 else value.item()
    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]
    return convert(value)


def _scalar_metric_value(key: str, value: Any) -> Any:
    # Proton's built-in roofline metrics should have stable dtypes regardless of
    # whether launch_metadata produced a Python scalar or a device tensor.
    if key in ROOFLINE_DTYPES:
        convert, _, _ = ROOFLINE_DTYPES[key]
        return _scalar_or_vector_value(value, convert)
    return value


def _tensor_metric_value_and_index(key: str, value: Any) -> tuple[Any, int]:
    if key in ROOFLINE_DTYPES:
        _, scalar_index, vector_index = ROOFLINE_DTYPES[key]
        value = value.long() if key == "bytes" else value.double()
        return value, vector_index if value.numel() > 1 else scalar_index

    # implicit casting to double or int64 tensors
    if value.is_floating_point():
        value = value.double()
        if value.numel() > 1:
            return value, libproton.metric_type_vector_double_index
        return value, libproton.metric_type_double_index

    value = value.long()
    if value.numel() > 1:
        return value, libproton.metric_type_vector_int64_index
    return value, libproton.metric_type_int64_index


def transform_tensor_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, libproton.TensorMetric]]:
    tensor_metrics = {}
    scalar_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "data_ptr"):  # tensor
            if value.device.type == "cpu":
                scalar_metrics[key] = _scalar_metric_value(key, value)
            else:  # device tensor
                value, metric_index = _tensor_metric_value_and_index(key, value)
                tensor_metrics[key] = _TensorMetric(value, metric_index)
        else:
            scalar_metrics[key] = _scalar_metric_value(key, value)
    return scalar_metrics, tensor_metrics
<|endoftext|>import argparse
from collections import namedtuple
import json
import pandas as pd

try:
    import hatchet as ht
    from hatchet.query import NegationQuery
except ImportError:
    raise ImportError("Failed to import hatchet. `pip install llnl-hatchet` to get the correct version.")
import numpy as np
from triton.profiler.state import COMPUTE_METADATA_SCOPE_NAME
from triton.profiler.metric import FLOPS_WIDTHS
from triton.profiler import specs


def match_available_metrics(metrics, inclusive_metrics, exclusive_metrics):
    ret = []
    if not isinstance(metrics, list):
        metrics = [metrics]
    if metrics:
        for metric in metrics:
            metric = metric.lower()
            for raw_metric in inclusive_metrics + exclusive_metrics:
                suffix = " (inc)" if raw_metric in inclusive_metrics else ""
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                if metric in (raw_metric, raw_metric_no_unit):
                    ret.append(raw_metric + suffix)
                    break
    if len(ret) == 0:
        raise RuntimeError(f"Metric {metric} is not found. Use the --list flag to list available metrics")
    return ret


def remove_frames(database: json):
    # We first fine frames that match either one of the two conditions:
    # 1. The frame name is a metadata frame
    # 2. The frame has no metrics and no children
    # Then we go up from the located nodes and remove the parents if all children were
    # metadata nodes
    def remove_frame_helper(node):
        if "frame" not in node:
            return node
        if node["frame"]["name"] == COMPUTE_METADATA_SCOPE_NAME or node["frame"]["name"].startswith(
                f"{COMPUTE_METADATA_SCOPE_NAME}:"):
            return None
        if len(node["metrics"]) == 0 and len(node["children"]) == 0:
            return None
        children = node.get("children", [])
        new_children = []
        for child in children:
            new_child = remove_frame_helper(child)
            if new_child is not None:
                new_children.append(new_child)
        if len(new_children) > 0 or len(children) == 0:
            node["children"] = new_children
            return node
        return None

    new_database = []
    for node in database:
        new_node = remove_frame_helper(node)
        if new_node is not None:
            new_database.append(new_node)
    return new_database


def get_raw_metrics(database) -> tuple[ht.GraphFrame, list[str], list[str], dict]:
    database = remove_frames(database)
    device_info = {} if len(database) < 2 else database.pop(1)
    gf = ht.GraphFrame.from_literal(database)
    inclusive_metrics = gf.show_metric_columns()
    exclusive_metrics = [metric for metric in gf.dataframe.columns if metric not in inclusive_metrics]
    return gf, inclusive_metrics, exclusive_metrics, device_info


def get_min_time_flops(df, device_info):
    min_time_flops = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            arch = device_info[device_type][device_index]["arch"]
            num_sms = device_info[device_type][device_index]["num_sms"]
            clock_rate = device_info[device_type][device_index]["clock_rate"]
            for width in FLOPS_WIDTHS:
                idx = df["device_id"] == device_index
                device_frames = df[idx]
                if f"flops{width}" not in device_frames.columns:
                    continue
                max_flops = specs.max_flops(device_type, arch, width, num_sms, clock_rate)
                min_time_flops.loc[idx, "min_time"] += device_frames[f"flops{width}"].fillna(0) / max_flops
    return min_time_flops


def get_min_time_bytes(df, device_info):
    min_time_bytes = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            idx = df["device_id"] == device_index
            device_frames = df[idx]
            device = device_info[device_type][device_index]
            memory_clock_rate = device["memory_clock_rate"]  # in khz
            bus_width = device["bus_width"]  # in bits
            peak_bandwidth = specs.max_bps(device_type, device['arch'], bus_width, memory_clock_rate)
            min_time_bytes.loc[idx, "min_time"] += device_frames["bytes"] / peak_bandwidth
    return min_time_bytes


FactorDict = namedtuple("FactorDict", ["name", "factor"])
time_factor_dict = FactorDict("time", {"time/s": 1, "time/ms": 1e-3, "time/us": 1e-6, "time/ns": 1e-9})
avg_time_factor_dict = FactorDict("avg_time", {f"avg_{key}": value for key, value in time_factor_dict.factor.items()})
cpu_time_factor_dict = FactorDict("cpu_time",
                                  {"cpu_time/s": 1, "cpu_time/ms": 1e-3, "cpu_time/us": 1e-6, "cpu_time/ns": 1e-9})
avg_cpu_time_factor_dict = FactorDict("avg_cpu_time",
                                      {f"avg_{key}": value
                                       for key, value in cpu_time_factor_dict.factor.items()})
bytes_factor_dict = FactorDict("bytes", {"byte/s": 1, "gbyte/s": 1e9, "tbyte/s": 1e12})

derivable_metrics = {
    **{key: bytes_factor_dict
       for key in bytes_factor_dict.factor.keys()},
}

# FLOPS have a specific width to their metric
default_flop_factor_dict = {"flop/s": 1, "gflop/s": 1e9, "tflop/s": 1e12}
derivable_metrics.update(
    {key: FactorDict("flops", default_flop_factor_dict)
     for key in default_flop_factor_dict.keys()})
for width in FLOPS_WIDTHS:
    factor_name = f"flops{width}"
    factor_dict = {f"flop{width}/s": 1, f"gflop{width}/s": 1e9, f"tflop{width}/s": 1e12}
    derivable_metrics.update({key: FactorDict(factor_name, factor_dict) for key in factor_dict.keys()})


def derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info):
    derived_metrics = []

    def get_time_seconds(df, metric, factor_dict):
        time_metric_name = match_available_metrics(metric, inclusive_metrics, exclusive_metrics)[0]
        time_unit = factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0]
        return df[time_metric_name] * factor_dict.factor[time_unit]

    for metric in metrics:
        if metric == "util":  # exclusive
            min_time_bytes = get_min_time_bytes(gf.dataframe, device_info)
            min_time_flops = get_min_time_flops(gf.dataframe, device_info)
            time_sec = get_time_seconds(gf.dataframe, "time", time_factor_dict)
            internal_frame_indices = gf.dataframe["device_id"].isna()
            gf.dataframe["util"] = min_time_flops["min_time"].combine(min_time_bytes["min_time"], max) / time_sec
            gf.dataframe.loc[internal_frame_indices, "util"] = np.nan
            derived_metrics.append("util")
        elif metric in derivable_metrics:  # flop<width>/s, <t/g>byte/s, inclusive
            derivable_metric = derivable_metrics[metric]
            metric_name = derivable_metric.name
            metric_factor_dict = derivable_metric.factor
            matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
            gf.dataframe[f"{metric} (inc)"] = (gf.dataframe[matched_metric_name] /
                                               (get_time_seconds(gf.dataframe, "time", time_factor_dict)) /
                                               metric_factor_dict[metric])
            derived_metrics.append(f"{metric} (inc)")
        elif (metric in time_factor_dict.factor or metric in cpu_time_factor_dict.factor
              or metric in avg_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor):  # inclusive
            is_cpu = metric in cpu_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor
            is_avg = metric in avg_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor

            factor_dict = ((avg_cpu_time_factor_dict if is_avg else cpu_time_factor_dict) if is_cpu else
                           (avg_time_factor_dict if is_avg else time_factor_dict))
            metric_name = "cpu_time" if is_cpu else "time"
            metric_time_unit = factor_dict.name + "/" + metric.split("/")[1]

            time_value = get_time_seconds(gf.dataframe, metric_name, factor_dict)
            if is_avg:
                time_value = time_value / gf.dataframe["count (inc)"]

            gf.dataframe[f"{metric} (inc)"] = time_value / factor_dict.factor[metric_time_unit]
            derived_metrics.append(f"{metric} (inc)")
        else:
            metric_name_and_unit = metric.split("/")
            metric_name = metric_name_and_unit[0]
            if len(metric_name_and_unit) > 1:  # percentage, exclusive or inclusive
                metric_unit = metric_name_and_unit[1]
                if metric_unit != "%":
                    raise ValueError(f"Unsupported unit {metric_unit}")
                matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
                single_frame = gf.dataframe[matched_metric_name]
                suffix = ""
                if "(inc)" in matched_metric_name:
                    suffix = " (inc)"
                    total = gf.dataframe[matched_metric_name].iloc[0]
                else:
                    total = gf.dataframe[matched_metric_name].sum()
                gf.dataframe[metric + suffix] = (single_frame / total) * 100.0
                derived_metrics.append(metric + suffix)
            else:
                matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
                derived_metrics.append(matched_metric_name)

    # Update derived metrics to the graph frame
    for derived_metric in derived_metrics:
        if derived_metric.endswith("(inc)"):
            gf.inc_metrics.append(derived_metric)
        else:
            gf.exc_metrics.append(derived_metric)

    return derived_metrics


def format_frames(gf, format):
    if format == "file_function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split("/")[-1])
    elif format == "function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split(":")[-1])
    elif format == "file_function":
        gf.dataframe["name"] = gf.dataframe["name"].apply(
            lambda x: f"{x.split('/')[-1].split(':')[0]}@{x.split('@')[-1].split(':')[0]}")
    return gf


def filter_frames(gf, include=None, exclude=None, threshold=None, metric=None):
    if include:
        query = f"""
MATCH ("*")->(".", p)->("*")
WHERE p."name" =~ "{include}"
"""
        gf = gf.filter(query, squash=True)
    if exclude:
        inclusion_query = f"""
MATCH (".", p)->("*")
WHERE p."name" =~ "{exclude}"
"""
        query = NegationQuery(inclusion_query)
        gf = gf.filter(query, squash=True)
    if threshold:
        query = ["*", {metric: f">= {threshold}"}]
        gf = gf.filter(query, squash=True)
    return gf


def emit_warnings(gf, metrics):
    if "bytes (inc)" in metrics:
        byte_values = gf.dataframe["bytes (inc)"].values
        min_byte_value = np.nanmin(byte_values)
        if min_byte_value < 0:
            print("Warning: Negative byte values detected, this is usually the result of a datatype overflow\n")


def print_tree(gf, metrics, depth=100, format=None, print_sorted=False):
    gf = format_frames(gf, format)
    print(gf.tree(metric_column=metrics, expand_name=True, depth=depth, render_header=False))

    if print_sorted:
        print("Sorted kernels by metric " + metrics[0])
        sorted_df = gf.dataframe.sort_values(by=[metrics[0]], ascending=False)
        for row in range(1, len(sorted_df)):
            kernel_name = (sorted_df.iloc[row]["name"][:100] +
                           "..." if len(sorted_df.iloc[row]["name"]) > 100 else sorted_df.iloc[row]["name"])
            print("{:105} {:.4}".format(kernel_name, sorted_df.iloc[row][metrics[0]]))
    emit_warnings(gf, metrics)


def read(filename):
    with open(filename, "r") as f:
        database = json.load(f)
        gf, inclusive_metrics, exclusive_metrics, device_info = get_raw_metrics(database)
        assert len(inclusive_metrics + exclusive_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        return gf, inclusive_metrics, exclusive_metrics, device_info


def parse(metrics, filename, include=None, exclude=None, threshold=None):
    gf, inclusive_metrics, exclusive_metrics, device_info = read(filename)
    metrics = derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info)
    # TODO: generalize to support multiple metrics, not just the first one
    gf = filter_frames(gf, include, exclude, threshold, metrics[0])
    return gf, metrics


def apply_diff_profile(gf, derived_metrics, diff_file, metrics, include, exclude, threshold):
    # Compute the diff against a secondary profile while keeping derived metrics consistent.
    gf2, _ = parse(metrics, diff_file, include, exclude, threshold)

    derived_inc_metrics = [metric for metric in derived_metrics if metric.endswith("(inc)")]
    derived_exc_metrics = [metric for metric in derived_metrics if not metric.endswith("(inc)")]

    gf.inc_metrics = derived_inc_metrics
    gf.exc_metrics = derived_exc_metrics
    gf2.inc_metrics = derived_inc_metrics
    gf2.exc_metrics = derived_exc_metrics
    return gf.sub(gf2)


def show_metrics(file_name):
    with open(file_name, "r") as f:
        database = json.load(f)
        _, inclusive_metrics, exclusive_metrics, _ = get_raw_metrics(database)
        print("Available inclusive metrics:")
        if inclusive_metrics:
            for raw_metric in inclusive_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")
        print("Available exclusive metrics:")
        if exclusive_metrics:
            for raw_metric in exclusive_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")


def main():
    argparser = argparse.ArgumentParser(
        description="Performance data viewer for proton profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="""List available metrics. Metric names are case insensitive and ignore units.
Derived metrics can be created when source metrics are available.
- time/s, time/ms, time/us, time/ns: time
- avg_time/s, avg_time/ms, avg_time/us, avg_time/ns: time / count
- flop[<8/16/32/64>]/s, gflop[<8/16/32/64>]/s, tflop[<8/16/32/64>]/s: flops / time
- byte/s, gbyte/s, tbyte/s: bytes / time
- util: max(sum(flops<width>) / peak_flops<width>_time, sum(bytes) / peak_bandwidth_time)
- <metric>/%%: frame(metric) / sum(metric). Only available for inclusive metrics (e.g. time)
""",
    )
    argparser.add_argument(
        "-m",
        "--metrics",
        type=str,
        default=None,
        help="""At maximum two metrics can be specified, separated by comma.
There are two modes:
1) Choose the output metric to display. It's case insensitive and ignore units.
2) Derive a new metric from existing metrics.
""",
    )
    argparser.add_argument(
        "-i",
        "--include",
        type=str,
        default=None,
        help=
        """Find frames that match the given regular expression and return all nodes in the paths that pass through the matching frames.
For example, the following command will display all paths that contain frames that contains "test":
```
proton-viewer -i ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default=None,
        help="""Exclude frames that match the given regular expression and their children.
For example, the following command will exclude all paths starting from frames that contains "test":
```
proton-viewer -e ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=
        "Exclude frames(kernels) whose metrics are below the given threshold. This filter only applies on the first metric.",
    )
    argparser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=100,
        help="The depth of the tree to display",
    )
    argparser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["full", "file_function_line", "function_line", "file_function"],
        default="full",
        help="""Formatting the frame name.
- full: include the path, file name, function name and line number.
- file_function_line: include the file name, function name and line number.
- function_line: include the function name and line number.
- file_function: include the file name and function name.
""",
    )
    argparser.add_argument(
        "--print-sorted",
        action="store_true",
        default=False,
        help="Sort output by metric value instead of chronologically",
    )
    argparser.add_argument(
        "--diff-profile",
        "-diff",
        type=str,
        default=None,
        help="Compare two profiles. When used as 'proton-viewer -m time -diff file1.log file2.log', "
        "computes the difference: file2['time'] - file1['time']",
    )

    args, target_args = argparser.parse_known_args()
    assert len(target_args) == 1, "Must specify a file to read"

    file_name = target_args[0]
    metrics = args.metrics.split(",") if args.metrics else None
    include = args.include
    exclude = args.exclude
    threshold = args.threshold
    depth = args.depth
    format = args.format
    diff = args.diff_profile
    print_sorted = args.print_sorted
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")
    if args.list:
        show_metrics(file_name)
    elif metrics:
        gf, derived_metrics = parse(metrics, file_name, include, exclude, threshold)
        if diff:
            gf = apply_diff_profile(gf, derived_metrics, diff, metrics, include, exclude, threshold)
        print_tree(gf, derived_metrics, depth, format, print_sorted)


if __name__ == "__main__":
    main()
<|endoftext|>from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton
from triton.language.semantic import TritonSemantic
from triton.experimental.gluon.language._semantic import GluonSemantic

from .flags import flags

_ALL_SEMANTICS = {
    "triton": TritonSemantic,
    "gluon": GluonSemantic,
}
"""
By default **only Gluon** semantic is enabled.
Instrumenting kernels written in Triton DSL is disable because Triton's higher-level IR undergoes
aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.).
These transformations can invalidate naïve instrumentation and lead to misleading results.
"""
_SEMANTICS = {_ALL_SEMANTICS["gluon"]}


def _check_supported_semantic(semantic):
    if not isinstance(semantic, tuple(_SEMANTICS)):
        raise TypeError(f"Unsupported semantic type: {type(semantic)}. "
                        f"Supported semantics are: {_SEMANTICS}")


def enable_semantic(semantic_name: str):
    _SEMANTICS.add(_ALL_SEMANTICS[semantic_name])


def disable_semantic(semantic_name: str):
    _SEMANTICS.remove(_ALL_SEMANTICS[semantic_name])


def record(is_start: tl.constexpr, scope_name: tl.constexpr, semantic):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(semantic)
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    return tl.tensor(triton_proton.create_proton_record(semantic.builder, is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _semantic=None):
    record(is_start=True, scope_name=name, semantic=_semantic)


@builtin
def exit_scope(name: tl.constexpr, _semantic=None):
    record(is_start=False, scope_name=name, semantic=_semantic)


class scope:

    def __init__(self, name: str, _semantic=None):
        self.name = name
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, _semantic=self.semantic)
<|endoftext|>"""
Centralized, process-local flags with a minimal interface (no environment variables).

Usage:
    from triton.profiler.flags import flags

    # Toggle
    flags.profiling_on = True
    flags.instrumentation_on = False

    # Check
    if flags.command_line:
            ...
"""
from dataclasses import dataclass


@dataclass
class ProfilerFlags:
    # Whether profiling is enabled. Default is False.
    profiling_on: bool = False
    # Whether instrumentation is enabled. Default is False.
    instrumentation_on: bool = False
    # Whether the script is run from the command line. Default is False.
    command_line: bool = False


flags = ProfilerFlags()
<|endoftext|>from ..state import metadata_state
from ..metric import transform_tensor_metrics, set_metric_kernels
from triton.compiler import LazyDict
from .hook import Hook
from triton._C.libproton import proton as libproton
from contextvars import ContextVar
from numbers import Number
import re
from typing import Optional

op_name = ContextVar("op_name", default=None)
id = ContextVar("id", default=None)
enabled = ContextVar("enabled", default=False)
in_hook = ContextVar("in_hook", default=False)


class LaunchHook(Hook):
    # Highest priority
    priority = 100

    # Reserved keys that Triton's runtime always attaches to launch_metadata.
    # We never treat these as metrics.
    _reserved_metadata_keys = {"name", "function", "stream"}

    # LaunchHook is intended to be a process-wide singleton. HookManager dedupes
    # by identity (object instance), so we must ensure repeated LaunchHook()
    # constructions return the same instance to avoid double registration.
    _instance = None

    def configure(self, *, include: Optional[str] = None, exclude: Optional[str] = None) -> None:
        # Regexes over the compiled kernel name (metadata.data["name"]).
        self._include_re = re.compile(include) if include else None
        self._exclude_re = re.compile(exclude) if exclude else None

    def _matches_kernel_name(self, kernel_name: str) -> bool:
        if kernel_name is None:
            return True
        if self._include_re is not None and self._include_re.match(kernel_name) is None:
            return False
        if self._exclude_re is not None and self._exclude_re.match(kernel_name) is not None:
            return False
        return True

    @staticmethod
    def _is_supported_metric_value(value) -> bool:
        # Supported scalar: Python/numpy number-like (bools are allowed but not very useful).
        # Supported tensor: objects with a data_ptr() method (e.g., torch.Tensor).
        if value is None:
            return False
        if hasattr(value, "data_ptr"):
            return True
        return isinstance(value, Number)

    @staticmethod
    def _extract_metrics(lazy_metadata: dict) -> dict:
        # Accept arbitrary metrics from launch_metadata while filtering out reserved fields
        # and unsupported values (e.g., objects/functions).
        return {
            k: v
            for k, v in lazy_metadata.items()
            if k not in LaunchHook._reserved_metadata_keys and LaunchHook._is_supported_metric_value(v)
        }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Singleton: __init__ is invoked on every construction even when __new__
        # returns an existing instance.
        if getattr(self, "_initialized", False):
            return
        # Ensure filter state is always initialized even if configure() isn't called.
        self.configure(include=None, exclude=None)
        self._initialized = True

    def init_handle(self, module, function, name: str, metadata_group: dict, hash: str) -> None:
        pass

    def activate(self):
        pass

    def deactivate(self):
        pass

    def enter(self, metadata: LazyDict) -> None:
        if enabled.get() or in_hook.get():
            return
        enabled.set(True)
        in_hook.set(True)

        # Early return path 1: skip metadata evaluation when the raw kernel
        # name is enough to filter.
        raw_kernel_name = metadata.data.get("name")
        if not self._matches_kernel_name(raw_kernel_name):
            enabled.set(False)
            in_hook.set(False)
            return

        with metadata_state(raw_kernel_name):
            lazy_metadata = metadata.get()
        kernel_name = lazy_metadata["name"]

        # Early return path 2: if the evaluated kernel name doesn't match
        # filters, skip recording.
        if not self._matches_kernel_name(kernel_name):
            enabled.set(False)
            in_hook.set(False)
            return

        fn_metrics = LaunchHook._extract_metrics(lazy_metadata)
        with metadata_state(raw_kernel_name):
            if fn_metrics:
                set_metric_kernels()
            scalar_metrics, tensor_metrics = transform_tensor_metrics(fn_metrics)

        scope_id = libproton.record_scope()
        libproton.enter_op(scope_id, kernel_name)
        with metadata_state(raw_kernel_name):
            libproton.add_metrics(scope_id, scalar_metrics, tensor_metrics)
        id.set(scope_id)
        op_name.set(kernel_name)
        in_hook.set(False)

    def exit(self, metadata: LazyDict) -> None:
        if in_hook.get():
            return
        if not enabled.get():
            return
        libproton.exit_op(id.get(), op_name.get())
        enabled.set(False)
<|endoftext|>from triton.compiler import LazyDict
from abc import abstractmethod
from typing import Dict, Any, Optional
from collections import defaultdict
import triton.knobs as knobs


class Hook:
    priority: int = 0

    @abstractmethod
    def init_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str],
                    hash: str) -> None:  # noqa: D401
        raise NotImplementedError

    def destroy_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        pass

    @abstractmethod
    def enter(self, metadata: LazyDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def exit(self, metadata: LazyDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def activate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def deactivate(self) -> None:
        raise NotImplementedError


class HookManager:
    # active hooks
    active_hooks: list[Hook] = []
    # session_id -> (hook_type -> active)
    session_hooks: Dict[int, Dict[Hook, bool]] = defaultdict(lambda: defaultdict(bool))

    @staticmethod
    def init_handle(module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        for hook in HookManager.active_hooks:
            hook.init_handle(module, function, name, metadata_group, hash)

    @staticmethod
    def destroy_handle(module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        for hook in reversed(HookManager.active_hooks):
            hook.destroy_handle(module, function, name, metadata_group, hash)

    @staticmethod
    def enter(metadata: LazyDict) -> None:
        for hook in HookManager.active_hooks:
            hook.enter(metadata)

    @staticmethod
    def exit(metadata: LazyDict) -> None:
        # It's important to reverse the order of hooks so that we keep the first in last out order
        for hook in reversed(HookManager.active_hooks):
            hook.exit(metadata)

    @staticmethod
    def activate(session: Optional[int] = None) -> None:
        if session is None:
            sessions = HookManager.session_hooks.keys()
        else:
            sessions = [session]

        for session in sessions:
            for hook in HookManager.session_hooks[session]:
                if hook not in HookManager.active_hooks:
                    hook.activate()
                    HookManager.active_hooks.append(hook)
                HookManager.session_hooks[session][hook] = True
        # Sort active_hooks by priority
        HookManager.active_hooks.sort(key=lambda x: x.priority, reverse=True)

    @staticmethod
    def deactivate(session: Optional[int] = None) -> None:
        if session is None:
            sessions = HookManager.session_hooks.keys()
        else:
            sessions = [session]

        deactivated_hooks = set()
        for session in sessions:
            for hook in HookManager.session_hooks[session]:
                if hook in HookManager.active_hooks:
                    deactivated_hooks.add(hook)
                HookManager.session_hooks[session][hook] = False

        # Check if any other sessions rely on this hook
        for hook in deactivated_hooks:
            if not any(session_hooks[hook] for session_hooks in HookManager.session_hooks.values()):
                hook.deactivate()
                HookManager.active_hooks.remove(hook)

    @staticmethod
    def register(hook: Hook, session: int) -> None:
        HookManager.session_hooks[session][hook] = True
        if hook not in HookManager.active_hooks:
            hook.activate()
            HookManager.active_hooks.append(hook)
        # Sort active_hooks by priority
        HookManager.active_hooks.sort(key=lambda x: x.priority, reverse=True)

        # Register the heads
        knobs.runtime.kernel_load_end_hook.add(HookManager.init_handle)
        knobs.runtime.kernel_unload_hook.add(HookManager.destroy_handle)
        knobs.runtime.launch_enter_hook.add(HookManager.enter)
        knobs.runtime.launch_exit_hook.add(HookManager.exit)

    @staticmethod
    def unregister(session: Optional[int] = None) -> None:
        if session is not None and session not in HookManager.session_hooks:
            return

        if session is None:
            for hook in HookManager.active_hooks:
                hook.deactivate()
            HookManager.active_hooks.clear()
            HookManager.session_hooks.clear()
        else:
            popped_hooks = HookManager.session_hooks.pop(session)
            # Deactivate hooks that are not used by any other session
            for hook, active in popped_hooks.items():
                if not active:
                    continue
                if not any(session_hooks[hook] for session_hooks in HookManager.session_hooks.values()):
                    hook.deactivate()
                    HookManager.active_hooks.remove(hook)
        # Unregister the heads
        if not HookManager.active_hooks:
            knobs.runtime.kernel_load_end_hook.remove(HookManager.init_handle)
            knobs.runtime.kernel_unload_hook.remove(HookManager.destroy_handle)
            knobs.runtime.launch_enter_hook.remove(HookManager.enter)
            knobs.runtime.launch_exit_hook.remove(HookManager.exit)
<|endoftext|>from typing import Dict, Optional, Union, Any

import triton
from triton._C.libtriton import ir as triton_ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libtriton import amd as triton_amd
from triton._C.libtriton import nvidia as triton_nvidia
from triton._C.libtriton import passes as triton_passes
from triton._C.libproton import proton as libproton
from triton.compiler import LazyDict
from triton.runtime._allocation import set_profile_allocator, NullAllocator
from triton.backends import backends

from .hook import Hook
from ..flags import flags
from ..state import metadata_state
from .. import mode

# TODO(fywkevin): add support for major.minor
VERSION = 1


class CudaAllocator:

    def __init__(self, instrumentation_hook):
        self.instrumentation_hook = instrumentation_hook

    def __call__(self, size: int, alignment: int, stream: Optional[int]):
        if alignment != self.instrumentation_hook.profile_buffer_alignment:
            raise RuntimeError(
                f"Alignment mismatch: {alignment} != {self.instrumentation_hook.profile_buffer_alignment}")
        aligned_size = (size + alignment - 1) // alignment * alignment
        # Note: profile_buffer_size may be smaller than the aligned size if the kernel launches many blocks
        # and the host CPU cannot store all profiling data in memory. This streaming mode is not yet implemented.
        # In the future, we should support copying data incrementally from device to host to enable
        # more efficient profiling data processing, rather than relying solely on post-processing.
        aligned_size = max(aligned_size, self.instrumentation_hook.profile_buffer_size)

        # Create the buffer
        import torch
        with metadata_state():
            buffer = torch.zeros((aligned_size, ), dtype=torch.uint8, device="cuda")
        self.instrumentation_hook.buffer = buffer
        return buffer


class Instrumentation:

    def __init__(self, ir_map: Dict[str, Any]):
        self.manager = ir_map

    def register(self, ir: str, func):
        if ir in self.manager:
            raise RuntimeError(f"IR already registered: {ir}")
        self.manager[ir] = func

    def patch(self, ir: str, pm, context):
        self.load_dialects(context)
        if ir in self.manager:
            self.manager[ir](pm)

    def load_dialects(self, ctx):
        triton_proton.load_dialects(ctx)


def _interpret_mode(mode_obj: Union[str, mode.InstrumentationMode]) -> mode.InstrumentationMode:
    if isinstance(mode_obj, mode.InstrumentationMode):
        return mode_obj
    elif not mode_obj:
        mode_obj = "default"

    parts = mode_obj.split(":")
    mode_name = parts[0]
    opts: Dict[str, str] = {}
    for opt in parts[1:]:
        if "=" in opt:
            key, val = opt.split("=", 1)
            opts[key] = val
        else:
            raise ValueError(f"Malformed instrumentation option: '{opt}'")

    # Get option values or empty strings
    options = {
        "metric_type": opts.get("metric_type", "cycle"), "buffer_type": opts.get("buffer_type", "shared"),
        "buffer_strategy": opts.get("buffer_strategy", "circular"), "buffer_size": int(opts.get("buffer_size", "0")),
        "granularity": opts.get("granularity", "warp"), "sampling_strategy": opts.get("sampling_strategy", "none"),
        "sampling_options": opts.get("sampling_options", ""), "optimizations": opts.get("optimizations", "")
    }

    # Helper function to validate and map options to their enum values
    def get_option_value(opt_name, mapping):
        value = options[opt_name]
        if value and value not in mapping:
            raise ValueError(f"Unknown {opt_name}: {value}")
        return mapping[value] if value else value

    # Look up enum values for each option
    options["metric_type"] = get_option_value("metric_type", mode.metric_types)
    options["buffer_type"] = get_option_value("buffer_type", mode.buffer_types)
    options["buffer_strategy"] = get_option_value("buffer_strategy", mode.buffer_strategies)
    options["granularity"] = get_option_value("granularity", mode.granularities)
    options["sampling_strategy"] = get_option_value("sampling_strategy", mode.sampling_strategies)

    values = ([value.strip()
               for value in options["optimizations"].split(",")] if len(options["optimizations"]) > 0 else [])
    for value in values:
        if value not in mode.optimizations:
            raise ValueError(f"Unknown optimization: {value}")
    options["optimizations"] = [mode.optimizations[value] for value in values]

    # Create the appropriate mode instance
    if mode_name == "default":
        return mode.Default(**options)
    elif mode_name == "mma":
        return mode.MMA(**options)
    else:
        raise ValueError(f"Unknown mode: {mode_obj}")


def _get_backend_name() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "nvidia"
    elif backend == "hip":
        return "amd"
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")


class InstrumentationHook(Hook):
    priority: int = 0
    # It's important to note that only one instance of the instrumentation hook can be active at a time.
    active_count: int = 0
    enable_host_buffer: bool = False
    host_buffer: Optional[Any] = None
    # FIXME(fywkevin): change to a more reasonable value after we have support for periodic buffer dumping.
    profile_buffer_size: int = 1
    profile_buffer_alignment: int = 128

    def __init__(self, mode_obj: Union[None, str, mode.InstrumentationMode]):
        # Mapping of function objects to their scope ID pairs
        self.mode: mode.InstrumentationMode = _interpret_mode(mode_obj)

        self.allocator = CudaAllocator(self)
        self.buffer = None
        self.metadata_path: Dict[Any, Optional[str]] = {}

    def activate(self):
        if InstrumentationHook.active_count > 0:
            raise RuntimeError("Only one instance of the instrumentation hook can be active at a time.")

        InstrumentationHook.active_count += 1

        flags.instrumentation_on = True

        device = triton.runtime.driver.active.get_current_device()
        max_shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
        backend_name = _get_backend_name()

        def to_llvmir_passes(pm):
            is_long_clk = False if mode.Optimize.CLOCK32 in self.mode.optimizations else True
            triton_proton.add_convert_proton_to_protongpu(pm, self.mode.metric_type, self.mode.sampling_strategy,
                                                          self.mode.sampling_options, self.mode.granularity,
                                                          self.mode.buffer_strategy, self.mode.buffer_type,
                                                          self.mode.buffer_size, max_shared_mem,
                                                          self.profile_buffer_size, self.profile_buffer_alignment,
                                                          is_long_clk)
            triton_passes.common.add_cse(pm)

            if mode.Optimize.SCHED_STORES in self.mode.optimizations:
                triton_proton.add_schedule_buffer_store(pm)

            triton_proton.add_allocate_proton_shared_memory(pm)

            if mode.Optimize.SCHED_BARRIERS in self.mode.optimizations and backend_name == "amd":
                triton_proton.add_sched_barriers(pm)

        def to_llvm_passes(pm):
            if backend_name == "nvidia":
                triton_proton.add_convert_proton_nvidia_gpu_to_llvm(pm)
            elif backend_name == "amd":
                arch = triton.runtime.driver.active.utils.get_device_properties(device)["arch"].split(":")[0]
                triton_proton.add_convert_proton_amd_gpu_to_llvm(pm, arch)

        backends[backend_name].compiler.instrumentation = Instrumentation({
            "ttgpuir_to_llvmir":
            lambda pm: to_llvmir_passes(pm),
            "llvmir_to_llvm":
            lambda pm: to_llvm_passes(pm),
        })

        # Set up the profiling allocator
        set_profile_allocator(self.allocator)

        # Set the instrumentation mode
        triton.knobs.compilation.instrumentation_mode = str(self.mode)

    def deactivate(self):
        if InstrumentationHook.active_count == 0:
            return

        InstrumentationHook.active_count -= 1

        backend_name = _get_backend_name()

        # No instrumentation passes are registered anymore
        backends[backend_name].compiler.instrumentation = {}

        # No runtime instrumentation hook is active anymore
        flags.instrumentation_on = False

        # Restore the instrumentation mode
        triton.knobs.compilation.instrumentation_mode = ""

        # Reset profile allocator
        set_profile_allocator(NullAllocator())

        # Reset host memory for external processing
        InstrumentationHook.host_buffer = None

        # Reset the buffer reference
        self.buffer = None

    def init_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        if not function:
            return

        # Find the IR path in metadata
        ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir"))), None)
        metadata_path = next((path for key, path in metadata_group.items() if key.endswith(("json"))), None)
        self.metadata_path[function] = metadata_path

        if ir_path:
            context = triton_ir.context()
            triton_ir.load_dialects(context)
            backend_name = _get_backend_name()
            if backend_name == "nvidia":
                triton_nvidia.load_dialects(context)
            elif backend_name == "amd":
                triton_amd.load_dialects(context)
            triton_proton.load_dialects(context)
            module = triton_ir.parse_mlir_module(ir_path, context)
            module.context = context

            scope_id_names = triton_proton.get_scope_id_names(module)
            scope_id_parents = triton_proton.get_scope_id_parents(module)
            libproton.init_function_metadata(function, name, scope_id_names, scope_id_parents, metadata_path)
        else:
            raise RuntimeError(f"IR path not found in metadata for function {function}")

    def destroy_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        if not function:
            return
        self.metadata_path.pop(function, None)
        libproton.destroy_function_metadata(function)

    def _data_ptr(self) -> int:
        return 0 if self.buffer is None else self.buffer.data_ptr()

    def enter(self, metadata: LazyDict) -> None:
        func = metadata.data.get("function")
        stream = metadata.data.get("stream")
        alloc_size = 0 if self.buffer is None else self.buffer.element_size() * self.buffer.numel()
        libproton.enter_instrumented_op(stream, func, self._data_ptr(), alloc_size)
        if InstrumentationHook.enable_host_buffer:
            InstrumentationHook.host_buffer = None

    def exit(self, metadata: LazyDict) -> None:
        func = metadata.data.get("function")
        stream = metadata.data.get("stream")
        alloc_size = 0 if self.buffer is None else self.buffer.element_size() * self.buffer.numel()
        libproton.exit_instrumented_op(stream, func, self._data_ptr(), alloc_size)

        if InstrumentationHook.enable_host_buffer:
            self._populate_host_buffer(func)

    def _populate_host_buffer(self, function: Any) -> None:
        if function and self.metadata_path[function]:
            import torch
            import struct
            import json

            def encode_target(target: Dict[str, Any]) -> int:
                #TODO(fywkevin): also account for `arch`
                if target["backend"] == "cuda":
                    return 1
                elif target["backend"] == "hip":
                    return 2
                return 0

            alloc_size = 0 if self.buffer is None else self.buffer.element_size() * self.buffer.numel()
            sampled_warps = self.mode.sampling_options.strip().split(",")
            data = {}
            with open(self.metadata_path[function], 'r') as file:
                data = json.load(file)

            device_type = encode_target(data["target"])
            scratch_mem_size = data["profile_scratch_size"]
            total_unit = data["num_warps"]
            uid_num = total_unit if self.mode.sampling_strategy == triton_proton.SAMPLING_STRATEGY.NONE else len(
                sampled_warps)
            block_num = int(alloc_size / scratch_mem_size)

            # Binary trace layout:
            # +------------------+
            # |     version      |  4 bytes
            # +------------------+
            # |  header_offset   |  4 bytes
            # +------------------+
            # |   header_size    |  4 bytes
            # +------------------+
            # |  payload_offset  |  4 bytes
            # +------------------+
            # |   payload_size   |  4 bytes
            # +------------------+
            # |   device_type    |  4 bytes
            # +------------------+
            # |    block_num     |  4 bytes
            # +------------------+
            # |   total_unit     |  4 bytes
            # +------------------+
            # | scratch_mem_size |  4 bytes
            # +------------------+
            # |     uid_num      |  4 bytes
            # +------------------+
            # |                  |
            # |     uid_vec      |  uid_num * 4 bytes
            # |                  |
            # +------------------+
            # |                  |
            # |     payload      |  size_payload bytes
            # |                  |
            # +------------------+

            is_all_warps = self.mode.sampling_options == "" and self.mode.granularity == triton_proton.GRANULARITY.WARP
            if is_all_warps:
                uid_vec = [i for i in range(total_unit)]
            else:
                uid_vec = [int(i) for i in sampled_warps]

            header_size = 40 + uid_num * 4
            header_offset = 4
            payload_offset = header_size
            payload_size = alloc_size
            header_values = [
                VERSION, header_offset, header_size, payload_offset, payload_size, device_type, block_num, total_unit,
                scratch_mem_size, uid_num, *uid_vec
            ]
            header_bytes = struct.pack("I" * len(header_values), *header_values)

            InstrumentationHook.host_buffer = torch.empty(header_size + alloc_size, dtype=torch.uint8, device="cpu")
            config_portion = InstrumentationHook.host_buffer[:header_size]
            config_portion.copy_(torch.tensor(list(header_bytes), dtype=torch.uint8))
            data_portion = InstrumentationHook.host_buffer[header_size:].view_as(self.buffer)
            data_portion.copy_(self.buffer.cpu())
<|endoftext|>import ast
import builtins
import contextlib
import copy
import functools
import inspect
import re
import warnings
import textwrap
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, Iterable, List

from .. import knobs, language
from .._C.libtriton import ir, gluon_ir
from ..language import constexpr, str_to_ty, tensor, tuple as tl_tuple
from ..language.core import _unwrap_if_constexpr, base_value, base_type
# ideally we wouldn't need any runtime component
from ..runtime.jit import get_full_name, JITCallable, BoundConstexprFunction, ConstexprFunction, JITFunction
from .._utils import apply_with_path, set_iterable_path, is_namedtuple

from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)


def check_identifier_legality(name, type):
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    if not re.match(pattern, name):
        raise CompilationError(f"invalid {type} identifier: {name}", name)
    return name


def mangle_fn(name, arg_tys, caller_context):
    # doesn't mangle ret type, which must be a function of arg tys
    mangled_args = '_'.join([ty.mangle() for ty in arg_tys])
    mangled_args = mangled_args.replace("'", '_sq_')
    # [ and ] are not allowed in LLVM identifiers
    mangled_args = mangled_args.replace('[', '_').replace(']', '_')
    ret = f'{name}__{mangled_args}'
    if caller_context is not None:
        ret += caller_context.mangle()
    return ret


def _is_triton_value(o: Any) -> bool:
    return isinstance(o, base_value)


def _is_triton_tensor(o: Any) -> bool:
    return isinstance(o, tensor)


def _is_constexpr(o: Any) -> bool:
    return o is None or isinstance(o, (constexpr, language.core.dtype, JITCallable))


def _is_non_scalar_tensor(o: Any) -> bool:
    return _is_triton_tensor(o) and (o.type.is_block() and o.type.numel != 1)


def _is_list_like(o: Any) -> bool:
    return isinstance(o, (list, tuple))


def _check_fn_args(node, fn, args):
    if fn.noinline:
        for idx, arg in enumerate(args):
            if not _is_constexpr(arg) and _is_non_scalar_tensor(arg):
                raise UnsupportedLanguageConstruct(
                    fn.src, node,
                    f'Function {fn.__name__} is marked noinline, but was called with non-scalar argument {fn.arg_names[idx]}:{arg}'
                )


def _check(cond, msg_fn, category=TypeError):
    if not cond:
        raise category(msg_fn())


def _apply_to_tuple_values(value, fn):
    if is_namedtuple(type(value)):
        fields = value._fields
    elif isinstance(value, builtins.tuple):
        fields = None
    elif isinstance(value, language.tuple):
        fields = value.type.fields
    else:
        assert False, f"Unsupported type {type(value)}"

    vals = [fn(v) for v in value]
    vals = [constexpr(v) if v is None else v for v in vals]
    types = [v.type for v in vals]
    return language.tuple(vals, language.tuple_type(types, fields))


def normalize_value(value):
    if isinstance(value, (builtins.tuple, language.tuple)):
        return _apply_to_tuple_values(value, normalize_value)
    if _is_triton_value(value):
        return value
    return constexpr(value)


def flatten_values_to_ir(values: Iterable[base_value]):
    handles = []
    for v in values:
        v._flatten_ir(handles)
    return handles


def unflatten_ir_values(handles: List[ir.value], types: List[base_type]):
    cursor = 0
    for ty in types:
        value, cursor = ty._unflatten_ir(handles, cursor)
        yield value
    assert cursor == len(handles)


_condition_types = {bool, int, type(None)}  # Python types accepted for conditionals inside kernels


class enter_sub_region:

    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        # record lscope & local_defs in the parent scope
        self.liveins = dict(self.generator.lscope)
        self.prev_defs = dict(self.generator.local_defs)
        self.prev_pending_annotations = dict(self.generator.pending_annotations)
        self.generator.local_defs = {}
        self.insert_block = self.generator.builder.get_insertion_block()
        self.insert_point = self.generator.builder.get_insertion_point()
        return self.liveins, self.insert_block

    def __exit__(self, *args, **kwargs):
        self.generator.builder.restore_insertion_point(self.insert_point)
        self.generator.lscope = self.liveins
        self.generator.local_defs = self.prev_defs
        self.generator.pending_annotations = self.prev_pending_annotations


# Check if the given syntax node has an "early" return
class ContainsReturnChecker(ast.NodeVisitor):

    def __init__(self, gscope):
        self.gscope = gscope

    def _visit_stmts(self, body) -> bool:
        return any(self.visit(s) for s in body)

    def _visit_function(self, fn) -> bool:
        # No need to check within the function as it won't cause an early return.
        # If the function itself has unstructured control flow we may not be able to inline it causing poor performance,
        # we should check for this and emit a warning.
        return False

    def generic_visit(self, node) -> bool:
        ret = False
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ret = ret or self.visit(item)
            elif isinstance(value, ast.AST):
                ret = ret or self.visit(value)
        return ret

    def visit_Attribute(self, node: ast.Attribute) -> bool:
        # If the left part is a name, it's possible that
        # we call triton native function or a jit function from another module.
        # If the left part is not a name, it must return a tensor or a constexpr
        # whose methods do not contain return statements
        # e.g., (tl.load(x)).to(y)
        # So we only check if the expressions within value have return or not
        if isinstance(node.value, ast.Name):
            if node.value.id in self.gscope:
                value = self.gscope[node.value.id]
                fn = getattr(value, node.attr)
                return self._visit_function(fn)
            return False
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> bool:
        if type(node.ctx) is ast.Store:
            return False
        if node.id in self.gscope:
            fn = self.gscope[node.id]
            return self._visit_function(fn)
        return False

    def visit_Return(self, node: ast.Return) -> bool:
        return True

    def visit_Assign(self, node: ast.Assign) -> bool:
        # There couldn't be an early return
        # x = ...
        return False

    def visit_AugAssign(self, node: ast.AugAssign) -> bool:
        # There couldn't be an early return
        # x += ...
        return False

    def visit_Module(self, node: ast.Module) -> bool:
        return self._visit_stmts(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> bool:
        return self._visit_stmts(node.body)

    def visit_If(self, node: ast.If) -> bool:
        # TODO: optimize the following case in which we actually don't have
        # a return when static_cond is false:
        # if dynamic_cond
        #   if static_cond
        #     func_with_return
        #   else
        #     func_without_return
        ret = self._visit_stmts(node.body)
        if node.orelse:
            ret = ret or self._visit_stmts(node.orelse)
        return ret

    def visit_IfExp(self, node: ast.IfExp) -> bool:
        return self.visit(node.body) or self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> bool:
        return self.visit(node.func)


class ASTFunction:

    def __init__(self, ret_types, arg_types, attrs):
        self.ret_types = ret_types
        self.arg_types = arg_types
        self.attrs = attrs

    def flatten_ir_types(self, builder: ir.builder, types: List[base_type]) -> List[ir.type]:
        ir_types = []
        for ty in types:
            if ty is None:
                continue
            ty._flatten_ir_types(builder, ir_types)
        return ir_types

    def return_types_ir(self, builder: ir.builder) -> List[ir.type]:
        return self.flatten_ir_types(builder, self.ret_types)

    def serialize(self, builder: ir.builder):
        # > build mlir function type
        arg_types_ir = self.flatten_ir_types(builder, self.arg_types)
        ret_types_ir = self.return_types_ir(builder)
        return builder.get_function_ty(arg_types_ir, ret_types_ir)

    def deserialize(self, fn):
        # create "template"
        def make_template(ty):
            if isinstance(ty, (list, tuple, language.tuple_type)):
                return language.tuple([make_template(x) for x in ty], ty)
            return language.constexpr(None)

        vals = make_template(self.arg_types)
        handles = [fn.args(i) for i in range(fn.get_num_args())]
        cursor = 0

        def build_value(path, ty):
            nonlocal cursor, handles
            # > set attributes
            attr_specs = self.attrs.get(path, [])
            for attr_name, attr_val in attr_specs:
                fn.set_arg_attr(cursor, attr_name, attr_val)
            # > build frontend value
            val, cursor = ty._unflatten_ir(handles, cursor)
            set_iterable_path(vals, path, val)

        apply_with_path(self.arg_types, build_value)
        return vals


@dataclass(frozen=True)
class BoundJITMethod:
    __self__: base_value
    __func__: JITFunction


class CodeGenerator(ast.NodeVisitor):

    def __init__(self, context, prototype, gscope, function_name, jit_fn: JITFunction, *, options, codegen_fns,
                 module_map, is_gluon, module=None, is_kernel=False, function_types: Optional[Dict] = None,
                 noinline=False, caller_context=None, file_name: Optional[str] = None, begin_line=0, begin_col=1):
        self.context = context
        self.is_gluon = is_gluon
        if is_gluon:
            from triton.experimental.gluon.language._semantic import GluonSemantic
            self.builder = gluon_ir.GluonOpBuilder(context, options.arch)
            self.semantic = GluonSemantic(self.builder)
        else:
            from triton.language.semantic import TritonSemantic
            self.builder = ir.builder(context)
            self.semantic = TritonSemantic(self.builder)

        self.name_loc_as_prefix = None
        self.file_name = file_name
        # node.lineno starts from 1, so we need to subtract 1
        self.begin_line = begin_line - 1
        self.begin_col = begin_col
        self.builder.set_loc(file_name, begin_line, begin_col)
        self.builder.options = options
        # dict of functions provided by the backend. Below are the list of possible functions:
        # Convert custom types not natively supported on HW.
        # convert_custom_types(input_tensor, dtype, fp_downcast_rounding=None, _builder=None)
        self.builder.codegen_fns = codegen_fns
        self.builder.module_map = {} if module_map is None else module_map
        self.module = self.builder.create_module() if module is None else module
        self.function_ret_types = {} if function_types is None else function_types
        self.prototype = prototype

        self.return_vals: List[base_value | None] = []
        self.return_ips: List[Tuple[ir.InsertPoint, ir.Loc]] = []

        self.gscope = {}
        for k, v in gscope.items():
            if isinstance(v, ModuleType):
                self.gscope[k] = module_map.get(v.__name__, v)
                continue

            module_name = getattr(v, "__module__", "")
            if module_name in module_map:
                self.gscope[k] = getattr(module_map[module_name], v.__name__)
            else:
                self.gscope[k] = v

        self.lscope = {}
        self.jit_fn = jit_fn
        # TODO: we currently generate illegal names for non-kernel functions involving constexprs!
        if is_kernel:
            function_name = function_name[function_name.rfind('.') + 1:]
            function_name = check_identifier_legality(function_name, "function")
        self.function_name = function_name
        self.is_kernel = is_kernel
        self.cur_node = None
        self.noinline = noinline
        self.caller_context = caller_context
        self.scf_stack = []
        self.ret_type = None
        # SSA-construction
        # name => language.tensor
        self.local_defs: Dict[str, tensor] = {}
        # Bare local annotations such as `x: tl.constexpr`
        self.pending_annotations: Dict[str, Any] = {}
        self.dereference_name: Callable[[str], Any] = self._define_name_lookup()
        self.fn = None
        # Are we currently visiting an ast.arg's default value?  These have some
        # special handling.
        self.visiting_arg_default_value = False

    builtin_namespace: Dict[str, Any] = {
        _.__name__: _
        for _ in (len, list, range, float, int, isinstance, getattr, hasattr)
    }
    builtin_namespace.update((
        ('print', language.core.device_print),
        ('min', language.core.builtin_min),
        ('max', language.core.builtin_max),
    ))

    def _unsupported(self, node, message):
        return UnsupportedLanguageConstruct(self.jit_fn.src, node, message)

    def _is_constexpr_global(self, name):
        absent_marker = object()
        val = self.gscope.get(name, absent_marker)
        if val is absent_marker:
            return False

        if _is_constexpr(val):
            return True

        return False

    def _define_name_lookup(self):

        def local_lookup(name: str, absent):
            # this needs to be re-fetched from `self` every time, because it gets switched occasionally
            return self.lscope.get(name, absent)

        def global_lookup(name: str, absent):
            val = self.gscope.get(name, absent)
            # The high-level rule is that only constexpr globals are allowed.
            # But actually a bunch of other things, such as module imports, are
            # technically Python globals. We have to allow these too!
            if any([
                    val is absent,
                    name in self.builtin_namespace,  #
                    type(val) is ModuleType,  #
                    isinstance(val, JITCallable),  #
                    getattr(val, "__triton_builtin__", False),  #
                    getattr(val, "__triton_aggregate__", False),  #
                    getattr(val, "__module__", "").startswith("triton.language"),  #
                    getattr(val, "__module__", "").startswith("triton.experimental.gluon.language"),  #
                    isinstance(val, language.dtype),  #
                    is_namedtuple(val),
                    self._is_constexpr_global(name),  #
                    # Allow accesses to globals while visiting an ast.arg
                    # because you should be able to do
                    #   @triton.jit def fn(x: tl.constexpr = GLOBAL): ...
                    self.visiting_arg_default_value,  #
                    knobs.compilation.allow_non_constexpr_globals,
            ]):
                return val
            raise NameError(
                textwrap.dedent(f"""\
                Cannot access global variable {name} from within @jit'ed
                function. Triton kernels can only access global variables that
                are instanstiated as constexpr (`x = triton.language.constexpr(42)`). Note that this is different from
                annotating a variable as constexpr (`x: triton.language.constexpr = 42`), which is not supported.  Alternatively, set the
                envvar TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1, but we do not
                promise to support this forever.""").replace("\n", " "))

        absent_marker = object()

        def name_lookup(name: str) -> Any:
            absent = absent_marker
            for lookup_function in local_lookup, global_lookup, self.builtin_namespace.get:
                value = lookup_function(name, absent)
                if value is not absent:
                    return value
            raise NameError(f'{name} is not defined')

        return name_lookup

    @contextlib.contextmanager
    def _name_loc_prefix(self, prefix):
        self.name_loc_as_prefix = prefix
        yield
        self.name_loc_as_prefix = None

    def _maybe_set_loc_to_name(self, val, name):
        if isinstance(val, (ir.value, ir.block_argument)):
            val.set_loc(self.builder.create_name_loc(name, val.get_loc()))
        elif _is_triton_value(val):
            val._set_name(self.builder, name)

    def set_value(self, name: str, value: Union[base_value, constexpr]) -> None:
        ''' This function:
            called by visit_Assign() & visit_FunctionDef() to store left value (lvalue)
        1. record local defined name (FIXME: should consider control flow)
        2. store tensor in self.lvalue
        '''
        self.lscope[name] = value
        self.local_defs[name] = value

    def _get_insertion_point_and_loc(self):
        # XXX: this is a hack to get the location of the insertion point.
        # The insertion point's location could be invalid sometimes,
        # so we need to explicitly set the location
        loc = self.builder.get_loc()
        ip = self.builder.get_insertion_point()
        return ip, loc

    def _set_insertion_point_and_loc(self, ip, loc):
        self.builder.restore_insertion_point(ip)
        self.builder.set_loc(loc)

    def _find_carries(self, node, liveins, ignore: set[str] = set()):
        # create loop body block
        block = self.builder.create_block()
        self.builder.set_insertion_point_to_start(block)
        # dry visit loop body
        self.scf_stack.append(node)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        block.erase()

        # If a variable (name) has changed value within the loop, then it's
        # a loop-carried variable. (The new and old value must be of the
        # same type)
        init_tys = []
        init_handles = []
        names = []

        for name, live_val in liveins.items():
            if name in ignore:
                continue

            if _is_triton_value(live_val):
                loop_val = self.lscope[name]
                self._verify_loop_carried_variable(name, loop_val, live_val)

                live_handles = flatten_values_to_ir([live_val])
                loop_handles = flatten_values_to_ir([loop_val])
                if live_handles != loop_handles:
                    names.append(name)
                    init_tys.append(live_val.type)
                    init_handles.extend(live_handles)
            else:
                assert name not in self.local_defs, f'Loop carried variable {name} is not a triton value'

        # reset local scope to not pick up local defs from the dry run.
        self.lscope = liveins.copy()
        self.local_defs = {}

        return names, init_handles, init_tys

    #
    # AST visitor
    #
    def visit_compound_statement(self, stmts):
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]
        for stmt in stmts:
            self.visit(stmt)
            # Stop parsing as soon as we hit a `return` statement; everything
            # after this is dead code.
            if isinstance(stmt, ast.Return):
                break

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_List(self, node):
        ctx = self.visit(node.ctx)
        assert ctx is None
        elts = language.tuple([self.visit(elt) for elt in node.elts])
        return elts

    def visit_ListComp(self, node: ast.ListComp):
        if len(node.generators) != 1:
            raise ValueError("nested comprehensions are not supported")

        comp = node.generators[0]
        iter = self.visit(comp.iter)
        if not isinstance(iter, tl_tuple):
            raise NotImplementedError("only tuple comprehensions are supported")

        results = []
        for item in iter:
            self.set_value(comp.target.id, item)
            results.append(self.visit(node.elt))
        return tl_tuple(results)

    # By design, only non-kernel functions can return
    def visit_Return(self, node):
        ret_value = self.visit(node.value)
        if ret_value is None:
            ret_value = language.constexpr(None)
        self.return_vals.append(ret_value)
        self.return_ips.append(self._get_insertion_point_and_loc())

        # A return op must always terminate the basic block, so we create a dead
        # basic block in case there are any ops after the return.
        post_ret_block = self.builder.create_block()
        self.builder.set_insertion_point_to_end(post_ret_block)

    def decide_return_type(self):
        assert len(self.return_vals) == len(self.return_ips)
        if not self.return_vals:
            return language.constexpr_type(None)

        tl = language.core

        def error_msg(a, b):
            err = f"Return type mismatch: {a} and {b}. "
            err += f"Note all return types were: {return_types}"
            return err

        def common_type(a, b):
            if isinstance(a, tl.tuple_type):
                _check(isinstance(b, tl.tuple_type), lambda: error_msg(a, b))
                _check(a.fields == b.fields, lambda: error_msg(a, b))
                return tl.tuple_type([common_type(ai, bi) for ai, bi in zip(a, b)], fields=a.fields)
            if isinstance(a, tl.constexpr_type):
                if a == b:
                    return a
                a = self.semantic.to_tensor_type(a)
                b = self.semantic.to_tensor_type(b)
            elif isinstance(b, tl.constexpr_type):
                a = self.semantic.to_tensor_type(a)
                b = self.semantic.to_tensor_type(b)
            _check(a == b, lambda: error_msg(a, b))
            return a

        return_types = [x.type for x in self.return_vals]
        return functools.reduce(common_type, return_types)

    def cast_to(self, value, ty):
        if value.type == ty:
            return value

        tl = language.core
        if isinstance(value, tl.tuple):
            assert isinstance(ty, tl.tuple_type)
            return tl.tuple(
                [self.cast_to(v, t) for v, t in zip(value.values, ty.types)],
                ty,
            )
        if isinstance(value, tl.constexpr):
            if isinstance(ty, tl.constexpr_type):
                _check(value.type == ty, lambda: f"Return type mismatch {value.type} and {ty}")
                return value
            return self.semantic.scalar_constant(value.value, ty)
        _check(value.type == ty, lambda: f"Return type mismatch {value.type} and {ty}")
        return value

    def handle_returns(self):
        return_type = self.decide_return_type()
        ip, loc = self._get_insertion_point_and_loc()

        assert len(self.return_vals) == len(self.return_ips)
        for ret, ret_ip in zip(self.return_vals, self.return_ips):
            self._set_insertion_point_and_loc(*ret_ip)
            assert not self.builder.get_insertion_block().has_terminator()
            ret = self.cast_to(ret, return_type)
            ret_handles = flatten_values_to_ir([ret])
            self.builder.ret(ret_handles)

        self._set_insertion_point_and_loc(ip, loc)
        self.ret_type = return_type
        assert not self.builder.get_insertion_block().has_terminator()
        if isinstance(self.ret_type, language.tuple_type):
            self.prototype.ret_types = list(self.ret_type.types)
        else:
            self.prototype.ret_types = [self.ret_type]
        self.fn.reset_type(self.prototype.serialize(self.builder))
        self.builder.ret([self.builder.create_poison(ty) for ty in self.prototype.return_types_ir(self.builder)])

    def visit_FunctionDef(self, node):
        if self.fn:
            raise self._unsupported(
                node, "nested function definitions are not allowed inside a @triton.jit kernel. "
                "Move the helper function to module level and decorate it with @triton.jit.")
        arg_names, kwarg_names = self.visit(node.args)
        # initialize defaults
        for i, default_value in enumerate(node.args.defaults[::-1]):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
            st_target = ast.Name(id=name, ctx=ast.Store())
            if annotation is None:
                init_node = ast.Assign(targets=[st_target], value=default_value)
            else:
                init_node = ast.AnnAssign(target=st_target, value=default_value, annotation=annotation)
            try:
                assert not self.visiting_arg_default_value
                self.visiting_arg_default_value = True
                self.visit(init_node)
            finally:
                self.visiting_arg_default_value = False

        # initialize function
        visibility = "public" if self.is_kernel else "private"
        fn_ty = self.prototype.serialize(self.builder)
        self.fn = self.builder.get_or_insert_function(self.module, self.function_name, fn_ty, visibility, self.noinline)
        self.module.push_back(self.fn)
        entry = self.fn.add_entry_block()
        arg_values = self.prototype.deserialize(self.fn)
        if self.caller_context is not None:
            self.caller_context.initialize_callee(self.fn, self.builder)
        # bind arguments to symbols
        for arg_name, arg_value in zip(arg_names, arg_values):
            self._maybe_set_loc_to_name(arg_value, arg_name)
            self.set_value(arg_name, arg_value)
        insert_pt = self.builder.get_insertion_block()
        self.builder.set_insertion_point_to_start(entry)
        # visit function body
        self.visit_compound_statement(node.body)

        # finalize function
        self.handle_returns()

        if insert_pt:
            self.builder.set_insertion_point_to_end(insert_pt)

    def visit_arguments(self, node):
        arg_names = []
        for arg in node.args:
            arg_names += [self.visit(arg)]
        if node.vararg is not None:
            arg_names += [self.visit(node.vararg)]
        kwarg_names = self.visit(node.kwarg)
        return arg_names, kwarg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        param = next(p for p in self.jit_fn.params if p.name == node.arg)
        if param.is_constexpr and (param.do_not_specialize or param.do_not_specialize_on_alignment):
            raise CompilationError(
                self.jit_fn.src, node,
                f"{node.arg} marked as constexpr and listed in do_not_specialize/do_not_specialize_on_alignment. "
                "Remove constexpr designation to skip specialization.")
        return node.arg

    def visit_AnnAssign(self, node):
        # extract attributes
        annotation = self.visit(node.annotation)
        target = self.visit(node.target)
        value = self.visit(node.value)
        # Bare annotation, without assigment
        if node.value is None:
            self.pending_annotations[target] = annotation
            return None
        # constexpr
        if annotation == constexpr:
            if target in self.lscope:
                raise ValueError(f'{target} is already defined.'
                                 f' constexpr cannot be reassigned.')
            value = normalize_value(value)
            self.lscope[target] = value
            return self.lscope[target]
        # default: call visit_Assign
        return self.visit_Assign(node)

    def assignTarget(self, target, value):
        assert isinstance(target.ctx, ast.Store)
        if isinstance(target, ast.Subscript):
            return self.visit_Subscript_Store(target, value)
        if isinstance(target, ast.Tuple):
            for i, target in enumerate(target.elts):
                self.assignTarget(target, value.values[i])
            return
        if isinstance(target, ast.Attribute):
            raise NotImplementedError("Attribute assignment is not supported in triton")
        assert isinstance(target, ast.Name)
        self.set_value(self.visit(target), value)

    def visit_Assign(self, node):
        # construct values to assign
        def _sanitize_value(value):
            if isinstance(value, language.tuple):
                return _apply_to_tuple_values(value, _sanitize_value)
            native_nontensor_types = (language.dtype, language.tuple)
            value = _unwrap_if_constexpr(value)
            if value is not None and \
                not _is_triton_value(value) and \
                not isinstance(value, native_nontensor_types):
                value = self.semantic.to_tensor(value)
            return value

        def _sanitize_target_value(target, value):
            if isinstance(target, ast.Tuple) and isinstance(value, language.tuple):
                vals = [_sanitize_target_value(elt, val) for elt, val in zip(target.elts, value.values)]
                vals = [constexpr(val) if val is None else val for val in vals]
                types = [val.type for val in vals]
                return language.tuple(vals, language.tuple_type(types, value.type.fields))
            if isinstance(target, ast.Name):
                annotation = self.pending_annotations.pop(target.id, None)
                if annotation == constexpr:
                    return normalize_value(value)
            return _sanitize_value(value)

        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        assert len(targets) == 1
        target = targets[0]
        if isinstance(target, ast.Name):
            with self._name_loc_prefix(target.id):
                values = _sanitize_target_value(target, self.visit(node.value))
        else:
            values = _sanitize_target_value(target, self.visit(node.value))
        self.assignTarget(target, values)

    def visit_AugAssign(self, node):
        lhs = copy.deepcopy(node.target)
        lhs.ctx = ast.Load()
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        for x in ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']:
            if hasattr(node, x):
                y = getattr(node, x)
                setattr(rhs, x, y)
                setattr(assign, x, y)
        self.visit(assign)
        return self.visit(lhs)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        # Named expressions are simple and can only be of the form x := value
        self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))
        return self.dereference_name(node.target.id)

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id
        return self.dereference_name(node.id)

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return language.tuple(args)

    def _apply_binary_method(self, node, method_name, lhs, rhs):
        # TODO: raise something meaningful if getattr fails below, esp for reverse method
        if _is_triton_tensor(lhs):
            return getattr(lhs, method_name)(rhs, _semantic=self.semantic)
        if _is_triton_tensor(rhs):
            reverse_method_name = re.sub(r"__(.*)__", r"__r\1__", method_name)
            return getattr(rhs, reverse_method_name)(lhs, _semantic=self.semantic)
        if not isinstance(lhs, (constexpr, language.tuple)) and isinstance(rhs, constexpr):
            lhs = constexpr(lhs)
        if isinstance(lhs, constexpr):
            fn = getattr(lhs, method_name)
        else:
            fn = self.get_Attribute(lhs, method_name)
        return self.call_Function(node, fn, [rhs], {})

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self._method_name_for_bin_op.get(type(node.op))
        if method_name is None:
            raise self._unsupported(node,
                                    "AST binary operator '{}' is not (currently) implemented.".format(node.op.__name__))
        return self._apply_binary_method(node, method_name, lhs, rhs)

    _method_name_for_bin_op: Dict[Type[ast.operator], str] = {
        ast.Add: '__add__',
        ast.Sub: '__sub__',
        ast.Mult: '__mul__',
        ast.Div: '__truediv__',
        ast.FloorDiv: '__floordiv__',
        ast.Mod: '__mod__',
        ast.Pow: '__pow__',
        ast.LShift: '__lshift__',
        ast.RShift: '__rshift__',
        ast.BitAnd: '__and__',
        ast.BitOr: '__or__',
        ast.BitXor: '__xor__',
    }

    def visit_then_else_blocks(self, node, liveins, then_block, else_block):
        pending_annotations = self.pending_annotations.copy()
        # then block
        self.builder.set_insertion_point_to_start(then_block)
        self.visit_compound_statement(node.body)
        then_block = self.builder.get_insertion_block()
        then_defs = self.local_defs.copy()
        then_vals = self.lscope.copy()
        # else block
        else_defs = {}
        else_vals = liveins.copy()
        if node.orelse:
            self.builder.set_insertion_point_to_start(else_block)
            self.lscope = liveins.copy()
            self.local_defs = {}
            self.pending_annotations = pending_annotations.copy()
            self.visit_compound_statement(node.orelse)
            else_defs = self.local_defs.copy()
            else_block = self.builder.get_insertion_block()
            else_vals = self.lscope.copy()

        # update block arguments
        names = []
        # variables in livein whose value is updated in `if`
        for name, value in liveins.items():
            # livein variable changed value in either then or else
            if not _is_triton_value(value):
                continue
            then_handles = flatten_values_to_ir([then_vals[name]])
            else_handles = flatten_values_to_ir([else_vals[name]])
            if then_handles == else_handles:
                continue
            names.append(name)
            then_defs[name] = then_vals[name]
            else_defs[name] = else_vals[name]
            # check type
            for defs, block_name in [(then_defs, 'then'), (else_defs, 'else')]:
                type_equal = type(defs[name]) == type(value)  # noqa: E721
                assert type_equal and defs[name].type == value.type, \
                    f'initial value for `{name}` is of type {value}, '\
                    f'but the {block_name} block redefines it as {defs[name]}'

        # variables that are both in then and else but not in liveins
        # TODO: could probably be cleaned up
        for name in sorted(then_defs.keys() & else_defs.keys()):
            if name in names:
                continue
            then_val = then_defs[name]
            then_ty = then_val.type
            else_val = else_defs[name]
            else_ty = else_val.type
            type_equal = type(then_val) == type(else_val)  # noqa: E721
            assert type_equal and then_ty == else_ty, \
                f'Mismatched type for {name} between then block ({then_ty}) '\
                f'and else block ({else_ty})'
            names.append(name)

        return then_defs, else_defs, then_block, else_block, names

    def visit_if_top_level(self, cond, node):
        with enter_sub_region(self) as sr:
            liveins, ip_block = sr
            then_block = self.builder.create_block()
            else_block = self.builder.create_block()
            # create branch
            self.builder.set_insertion_point_to_end(ip_block)
            self.builder.create_cond_branch(cond.handle, then_block, else_block)
            # visit then and else blocks
            then_defs, else_defs, then_block, else_block, names = \
                self.visit_then_else_blocks(node, liveins, then_block, else_block)
            # create basic-block after conditional
            endif_block = self.builder.create_block()
            # then terminator
            self.builder.set_insertion_point_to_end(then_block)
            assert not then_block.has_terminator(), f"{then_block}"
            then_handles = flatten_values_to_ir(then_defs[name] for name in names)
            self.builder.create_branch(endif_block, then_handles)
            # else terminator
            self.builder.set_insertion_point_to_end(else_block)
            assert not else_block.has_terminator(), f"{else_block}"
            else_handles = flatten_values_to_ir(else_defs[name] for name in names)
            self.builder.create_branch(endif_block, else_handles)
            assert len(then_handles) == len(else_handles)
            for then_h, else_h in zip(then_handles, else_handles):
                ty = then_h.get_type()
                assert ty == else_h.get_type()
                endif_block.add_argument(ty)

        # change block
        self.builder.set_insertion_point_to_start(endif_block)
        # update value
        res_handles = [endif_block.arg(i) for i in range(len(then_handles))]
        types = [then_defs[name].type for name in names]
        new_values = unflatten_ir_values(res_handles, types)
        for name, new_value in zip(names, new_values):
            self.set_value(name, new_value)

    # TODO: refactor
    def visit_if_scf(self, cond, node):
        with enter_sub_region(self) as sr:
            liveins, _ = sr
            ip, last_loc = self._get_insertion_point_and_loc()
            then_block = self.builder.create_block()
            else_block = self.builder.create_block() if node.orelse else None
            then_defs, else_defs, then_block, else_block, names = \
                self.visit_then_else_blocks(node, liveins, then_block, else_block)
            # create if op
            then_handles = flatten_values_to_ir(then_defs[name] for name in names)
            for name, val in zip(names, then_handles):
                self._maybe_set_loc_to_name(val, name)
            self._set_insertion_point_and_loc(ip, last_loc)
            if_op = self.builder.create_if_op([h.get_type() for h in then_handles], cond.handle, True)
            then_block.merge_block_before(if_op.get_then_block())
            self.builder.set_insertion_point_to_end(if_op.get_then_block())
            if len(names) > 0:
                self.builder.create_yield_op(then_handles)
            if not node.orelse:
                else_block = if_op.get_else_block()
            else:
                else_block.merge_block_before(if_op.get_else_block())
            self.builder.set_insertion_point_to_end(if_op.get_else_block())
            if len(names) > 0:
                else_handles = flatten_values_to_ir(else_defs[name] for name in names)
                for name, val in zip(names, else_handles):
                    self._maybe_set_loc_to_name(val, name)
                self.builder.create_yield_op(else_handles)
        # update values
        res_handles = [if_op.get_result(i) for i in range(len(then_handles))]
        types = [then_defs[name].type for name in names]
        new_values = unflatten_ir_values(res_handles, types)
        for name, new_value in zip(names, new_values):
            self.set_value(name, new_value)

    def visit_If(self, node):
        cond = self.visit(node.test)

        if _is_triton_tensor(cond):
            if _is_non_scalar_tensor(cond):
                raise self._unsupported(node, "Boolean value of Tensor with more than one value is ambiguous")
            if cond.type.is_block():
                warnings.warn(
                    "If conditional called with multidimensional Tensor instead of scalar; please use \"if (%s).item()\" instead"
                    % ast.unparse(node.test))
                cond = language.core._unsplat(cond, _semantic=self.semantic, _generator=self)
            cond = cond.to(language.int1, _semantic=self.semantic)
            if ContainsReturnChecker(self.gscope).visit(node):
                if self.scf_stack:
                    raise self._unsupported(
                        node, "Cannot have `return` statements inside `while` or `for` statements in triton.")
                self.visit_if_top_level(cond, node)
            else:
                self.visit_if_scf(cond, node)
        else:
            cond = _unwrap_if_constexpr(cond)
            # not isinstance - we insist the real thing, no subclasses and no ducks
            if type(cond) not in _condition_types:
                raise self._unsupported(
                    node, "`if` conditionals can only accept values of type {{{}}}, not objects of type {}".format(
                        ', '.join(_.__name__ for _ in _condition_types),
                        type(cond).__name__))

            active_block = node.body if cond else node.orelse
            self.visit_compound_statement(active_block)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if _is_triton_tensor(cond):
            cond = cond.to(language.int1, _semantic=self.semantic)
            # TODO: Deal w/ more complicated return types (e.g tuple)
            with enter_sub_region(self):
                ip, last_loc = self._get_insertion_point_and_loc()

                then_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(then_block)
                then_val = self.semantic.to_tensor(self.visit(node.body))
                then_block = self.builder.get_insertion_block()

                else_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(else_block)
                # do not need to reset lscope since
                # ternary expressions cannot define new variables
                else_val = self.semantic.to_tensor(self.visit(node.orelse))
                else_block = self.builder.get_insertion_block()

                self._set_insertion_point_and_loc(ip, last_loc)

                assert then_val.type == else_val.type, \
                    f'Ternary expression with dynamic condition has inconsistent types {then_val.type} and {else_val.type}'
                ret_type = then_val.type

                ret_type_ir = [ret_type.to_ir(self.builder)] if ret_type != language.void else []
                if_op = self.builder.create_if_op(ret_type_ir, cond.handle, True)
                then_block.merge_block_before(if_op.get_then_block())
                if ret_type_ir:
                    self.builder.set_insertion_point_to_end(if_op.get_then_block())
                    self.builder.create_yield_op([then_val.handle])

                self.builder.set_insertion_point_to_end(if_op.get_then_block())
                else_block.merge_block_before(if_op.get_else_block())
                if ret_type_ir:
                    self.builder.set_insertion_point_to_end(if_op.get_else_block())
                    self.builder.create_yield_op([else_val.handle])
                return language.core.tensor(if_op.get_result(0), ret_type) if ret_type_ir else None
        else:
            cond = _unwrap_if_constexpr(cond)

            # not isinstance - we insist the real thing, no subclasses and no ducks
            if type(cond) not in _condition_types:
                raise self._unsupported(
                    node, "`if` conditionals can only accept values of type {{{}}}, not objects of type {}".format(
                        ', '.join(_.__name__ for _ in _condition_types),
                        type(cond).__name__))
            if cond:
                return self.visit(node.body)
            else:
                return self.visit(node.orelse)

    def visit_With(self, node):
        # Lower `with` statements by constructing context managers and calling their enter/exit hooks
        # Instantiate each context manager with builder injection
        cm_list = []
        for item in node.items:
            call = item.context_expr
            fn = self.visit(call.func)
            args = [self.visit(arg) for arg in call.args]
            kws = dict(self.visit(kw) for kw in call.keywords)
            cm = fn(*args, _semantic=self.semantic, **kws)
            cm_list.append(cm)
        for cm, item in zip(cm_list, node.items):
            res = cm.__enter__()
            if item.optional_vars is not None:
                var_name = self.visit(item.optional_vars)
                self.set_value(var_name, res)
        if ContainsReturnChecker(self.gscope).visit(node):
            raise self._unsupported(node, "Cannot have `return` statements inside `with` statements in triton ")
        self.visit_compound_statement(node.body)
        for cm in reversed(cm_list):
            cm.__exit__(None, None, None)

    def visit_Pass(self, node):
        pass

    def visit_Compare(self, node):
        if not (len(node.comparators) == 1 and len(node.ops) == 1):
            raise self._unsupported(node, "simultaneous multiple comparison is not supported")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        lhs_value = _unwrap_if_constexpr(lhs)
        rhs_value = _unwrap_if_constexpr(rhs)
        if type(node.ops[0]) is ast.Is:
            return constexpr(lhs_value is rhs_value)
        if type(node.ops[0]) is ast.IsNot:
            return constexpr(lhs_value is not rhs_value)
        method_name = self._method_name_for_comp_op.get(type(node.ops[0]))
        if method_name is None:
            raise self._unsupported(
                node, "AST comparison operator '{}' is not (currently) implemented.".format(node.ops[0].__name__))
        return self._apply_binary_method(node, method_name, lhs, rhs)

    _method_name_for_comp_op: Dict[Type[ast.cmpop], str] = {
        ast.Eq: '__eq__', ast.NotEq: '__ne__', ast.Lt: '__lt__', ast.LtE: '__le__', ast.Gt: '__gt__', ast.GtE: '__ge__'
    }

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        fn = self._method_name_for_unary_op.get(type(node.op))
        if fn is None:
            raise self._unsupported(node, f"AST unary operator '{node.op.__name__}' is not (currently) implemented.")
        if _is_triton_tensor(operand):
            return getattr(operand, fn)(_semantic=self.semantic)
        try:
            return getattr(operand, fn)()
        except AttributeError:
            if fn == "__not__":
                return constexpr(not operand)
            raise self._unsupported(
                node, f"AST unary operator '{fn}' is not (currently) implemented on type {type(operand).__name__}")

    _method_name_for_unary_op: Dict[Type[ast.unaryop], str] = {
        ast.USub: '__neg__', ast.UAdd: '__pos__', ast.Not: '__not__', ast.Invert: '__invert__'
    }

    def _verify_loop_carried_variable(self, name, loop_val, live_val):
        assert _is_triton_value(loop_val), f'cannot reassign constexpr {name} in the loop'
        assert _is_triton_value(live_val), f'cannot reassign constexpr {name} in the loop'
        assert type(loop_val) is type(live_val), (
            f'Loop carried variable {name} changed type, was {type(loop_val)} but is now {type(live_val)}')
        assert not _is_triton_tensor(loop_val) or loop_val.type == live_val.type, \
            f'Loop-carried variable {name} has initial type {live_val.type} '\
            f'but is re-assigned to {loop_val.type} in loop! '\
            f'Please make sure that the type stays consistent.'

    def visit_While(self, node):
        with enter_sub_region(self) as sr:
            liveins, insert_block = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            names, init_handles, init_fe_tys = self._find_carries(node, liveins)

            init_tys = [h.get_type() for h in init_handles]
            self._set_insertion_point_and_loc(ip, last_loc)
            while_op = self.builder.create_while_op(init_tys, init_handles)
            # merge the condition region
            before_block = self.builder.create_block_with_parent(while_op.get_before(), init_tys)
            self.builder.set_insertion_point_to_start(before_block)
            block_args = [before_block.arg(i) for i in range(len(init_handles))]
            for arg, init in zip(block_args, init_handles):
                arg.set_loc(init.get_loc())
            condition_args = unflatten_ir_values(block_args, init_fe_tys)
            for name, val in zip(names, condition_args):
                self.lscope[name] = val
                self.local_defs[name] = val
                self._maybe_set_loc_to_name(val, name)
            cond = self.visit(node.test)
            if isinstance(cond, language.condition):
                if cond.disable_licm:
                    while_op.set_attr("llvm.loop_annotation", self.builder.get_disable_loop_licm_attr())
                cond = cond.condition
            self.builder.set_insertion_point_to_end(before_block)
            # create ConditionOp: e.g., scf.condition(%cond) %arg0, %arg1, ...
            self.builder.create_condition_op(cond.handle, block_args)
            # merge the loop body
            after_block = self.builder.create_block_with_parent(while_op.get_after(), init_tys)

            # generate loop body
            self.builder.set_insertion_point_to_start(after_block)
            body_handles = [after_block.arg(i) for i in range(len(init_handles))]
            for arg, init in zip(body_handles, init_handles):
                arg.set_loc(init.get_loc())
            body_args = unflatten_ir_values(body_handles, init_fe_tys)
            for name, val in zip(names, body_args):
                self.lscope[name] = val
                self.local_defs[name] = val
                self._maybe_set_loc_to_name(val, name)
            self.scf_stack.append(node)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()

            yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)
            self.builder.create_yield_op(yield_handles)

        # WhileOp defines new values, update the symbol table (lscope, local_defs)
        result_handles = [while_op.get_result(i) for i in range(len(init_handles))]
        result_vals = unflatten_ir_values(result_handles, init_fe_tys)
        for name, new_def in zip(names, result_vals):
            self.lscope[name] = new_def
            self.local_defs[name] = new_def
            self._maybe_set_loc_to_name(new_def, name)

        for stmt in node.orelse:
            assert False, "Not implemented"
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Subscript_Load(self, node):
        assert isinstance(node.ctx, ast.Load)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if _is_triton_value(lhs):
            return self.call_Method(node, lhs.__getitem__, lhs, [slices], {})
        return lhs[slices]

    def visit_Subscript_Store(self, node, value):
        raise NotImplementedError("__setitem__ is not supported in triton")

    def visit_Subscript(self, node):
        return self.visit_Subscript_Load(node)

    def visit_ExtSlice(self, node):
        return [self.visit(dim) for dim in node.dims]

    def visit_For(self, node):
        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = dict(self.visit(keyword) for keyword in node.iter.keywords)
        if IteratorClass == language.static_range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            static_range = range(iterator.start.value, iterator.end.value, iterator.step.value)
            for i in static_range:
                self.lscope[node.target.id] = constexpr(i)
                self.visit_compound_statement(node.body)
                for stmt in node.orelse:
                    ast.NodeVisitor.generic_visit(self, stmt)
            return
        num_stages = None
        loop_unroll_factor = None
        disallow_acc_multi_buffer = False
        flatten = False
        warp_specialize = False
        disable_licm = False
        if IteratorClass is language.range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            # visit iterator arguments
            # note: only `range` iterator is supported now
            # collect lower bound (lb), upper bound (ub), and step
            lb = iterator.start
            ub = iterator.end
            step = iterator.step
            num_stages = iterator.num_stages
            loop_unroll_factor = iterator.loop_unroll_factor
            disallow_acc_multi_buffer = iterator.disallow_acc_multi_buffer
            flatten = iterator.flatten
            warp_specialize = iterator.warp_specialize
            disable_licm = iterator.disable_licm
        elif IteratorClass is range:
            # visit iterator arguments
            # note: only `range` iterator is supported now
            # collect lower bound (lb), upper bound (ub), and step
            lb = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Constant(0))
            ub = iter_args[1] if len(iter_args) > 1 else self.visit(node.iter.args[0])
            step = iter_args[2] if len(iter_args) > 2 else self.visit(ast.Constant(1))
        else:
            raise RuntimeError('Only `range` and `static_range` iterators are currently supported')
        # handle negative constant step (not supported by scf.for in MLIR)
        negative_step = False
        if _is_constexpr(step) and step.value < 0:
            step = constexpr(-step.value)
            negative_step = True
            lb, ub = ub, lb
        lb = self.semantic.to_tensor(lb)
        ub = self.semantic.to_tensor(ub)
        step = self.semantic.to_tensor(step)
        # induction variable type
        if not lb.dtype.is_int() or not ub.dtype.is_int() or not step.dtype.is_int():
            raise TypeError(f"For loop bounds and step must all be ints, are ({lb.dtype}, {ub.dtype}, {step.dtype})")
        if _is_non_scalar_tensor(lb):
            raise TypeError(f"For lower bound must be a scalar, got {lb.type}")
        if _is_non_scalar_tensor(ub):
            raise TypeError(f"For upper bound must be a scalar, got {ub.type}")
        if _is_non_scalar_tensor(step):
            raise TypeError(f"For step must be a scalar, got {step.type}")
        iv_type = self.semantic.integer_promote_impl(lb.dtype, ub.dtype)
        iv_type = self.semantic.integer_promote_impl(iv_type, step.dtype)
        iv_ir_type = iv_type.to_ir(self.builder)
        iv_is_signed = iv_type.int_signedness == language.core.dtype.SIGNEDNESS.SIGNED
        # lb/ub/step might be constexpr, we need to cast them to tensor
        lb = lb.handle
        ub = ub.handle
        step = step.handle
        # ForOp can only accept IndexType as lb/ub/step. Cast integer to Index
        lb = self.builder.create_int_cast(lb, iv_ir_type, iv_is_signed)
        ub = self.builder.create_int_cast(ub, iv_ir_type, iv_is_signed)
        step = self.builder.create_int_cast(step, iv_ir_type, iv_is_signed)
        # Create placeholder for the loop induction variable
        iv_placeholder = self.builder.create_poison(iv_ir_type)
        self.set_value(node.target.id, language.core.tensor(iv_placeholder, iv_type))

        with enter_sub_region(self) as sr:
            liveins, insert_block = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            names, init_handles, init_tys = self._find_carries(node, liveins, ignore={node.target.id})

            # create ForOp
            self._set_insertion_point_and_loc(ip, last_loc)
            for_op = self.builder.create_for_op(lb, ub, step, init_handles)
            if _unwrap_if_constexpr(num_stages) is not None:
                for_op.set_attr("tt.num_stages", self.builder.get_int32_attr(num_stages))
            if _unwrap_if_constexpr(loop_unroll_factor) is not None:
                for_op.set_attr("tt.loop_unroll_factor", self.builder.get_int32_attr(loop_unroll_factor))
            if disallow_acc_multi_buffer:
                for_op.set_attr("tt.disallow_acc_multi_buffer", self.builder.get_unit_attr())
            if flatten:
                for_op.set_attr("tt.flatten", self.builder.get_unit_attr())
            if warp_specialize:
                for_op.set_attr("tt.warp_specialize", self.builder.get_unit_attr())
            if disable_licm:
                for_op.set_attr("llvm.loop_annotation", self.builder.get_disable_loop_licm_attr())

            self.scf_stack.append(node)
            for_op_body = for_op.get_body(0)
            self.builder.set_insertion_point_to_start(for_op_body)
            block_handles = [for_op_body.arg(i + 1) for i in range(len(init_handles))]
            block_args = unflatten_ir_values(block_handles, init_tys)
            for name, val in zip(names, block_args):
                self._maybe_set_loc_to_name(val, name)
                self.set_value(name, val)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)

            # create YieldOp
            if len(yield_handles) > 0:
                self.builder.create_yield_op(yield_handles)
            for_op_region = for_op_body.get_parent()
            assert for_op_region.size() == 1, "We use SCF, so the loop body should only have one block"

            # update induction variable with actual value, and replace all uses
            self.builder.set_insertion_point_to_start(for_op_body)
            iv = for_op.get_induction_var()
            if negative_step:
                iv = self.builder.create_sub(ub, iv)
                iv = self.builder.create_add(iv, lb)
            iv_placeholder.replace_all_uses_with(iv)
            self.set_value(node.target.id, language.core.tensor(iv, iv_type))
            self._maybe_set_loc_to_name(iv, node.target.id)

        # update lscope & local_defs (ForOp defines new values)
        result_handles = [for_op.get_result(i) for i in range(len(init_handles))]
        result_values = unflatten_ir_values(result_handles, init_tys)
        for name, val in zip(names, result_values):
            self.set_value(name, val)
            self._maybe_set_loc_to_name(val, name)

        for stmt in node.orelse:
            assert False, "Don't know what to do with else after for"
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Slice(self, node):
        lower = self.visit(node.lower)
        upper = self.visit(node.upper)
        step = self.visit(node.step)
        return language.slice(lower, upper, step)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_keyword(self, node) -> Tuple[str, Any]:
        return node.arg, self.visit(node.value)

    def visit_Assert(self, node) -> Any:
        test = self.visit(node.test)
        msg = self.visit(node.msg) if node.msg is not None else ""
        return language.core.device_assert(test, msg, _semantic=self.semantic)

    def call_JitFunction(self, fn: JITFunction, args, kwargs, caller_context=None):
        bound_args = fn.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        args = bound_args.arguments
        args = [args[name] for name in fn.arg_names]

        for i, arg in enumerate(args):
            args[i] = normalize_value(arg)
        # mangle
        caller_context = caller_context or self.caller_context
        arg_types = [arg.type for arg in args]
        fn_name = mangle_fn(get_full_name(fn), arg_types, caller_context)
        # generate function def if necessary
        if not self.module.has_function(fn_name):
            # If the callee is not set, we use the same debug setting as the caller
            prototype = ASTFunction([], arg_types, dict())
            generator = CodeGenerator(self.context, prototype, fn.get_capture_scope(), module=self.module, jit_fn=fn,
                                      function_name=fn_name, function_types=self.function_ret_types,
                                      noinline=fn.noinline, file_name=fn.file_name, begin_line=fn.def_file_line_number,
                                      begin_col=fn.def_file_col_number, options=self.builder.options,
                                      codegen_fns=self.builder.codegen_fns, module_map=self.builder.module_map,
                                      caller_context=caller_context, is_gluon=self.is_gluon)
            try:
                generator.visit(fn.parse())
            except Exception as e:
                # Wrap the error in the callee with the location of the call.
                if knobs.compilation.front_end_debugging:
                    raise
                raise CompilationError(self.jit_fn.src, self.cur_node, None) from e

            callee_ret_type = generator.ret_type
            self.function_ret_types[fn_name] = callee_ret_type
        else:
            callee_ret_type = self.function_ret_types[fn_name]
        symbol = self.module.get_function(fn_name)
        args_val = flatten_values_to_ir(args)
        call_op = self.builder.call(symbol, args_val)
        handles = [call_op.get_result(i) for i in range(call_op.get_num_results())]
        return next(unflatten_ir_values(handles, [callee_ret_type]))

    def call_Function(self, node, fn, args, kws):
        if isinstance(fn, (BoundJITMethod, BoundConstexprFunction)):
            args.insert(0, fn.__self__)
            fn = fn.__func__

        mur = getattr(fn, '_must_use_result', False)
        if mur and getattr(node, '_is_unused', False):
            error_message = ["The result of %s is not being used." % ast.unparse(node.func)]
            if isinstance(mur, str):
                error_message.append(mur)
            raise CompilationError(self.jit_fn.src, node, " ".join(error_message))

        if isinstance(fn, JITFunction):
            _check_fn_args(node, fn, args)
            return self.call_JitFunction(fn, args, kws)
        if (hasattr(fn, '__self__') and _is_triton_value(fn.__self__)) or language.core.is_builtin(fn) or isinstance(
                fn, ConstexprFunction):
            extra_kwargs = dict()

            sig = getattr(fn, "signature", None)
            if isinstance(fn, ConstexprFunction):
                extra_kwargs["_semantic"] = self.semantic
            else:
                if sig is None:
                    sig = inspect.signature(fn)
                if '_semantic' in sig.parameters:
                    extra_kwargs["_semantic"] = self.semantic
                if '_generator' in sig.parameters:
                    extra_kwargs['_generator'] = self
            try:
                ret = fn(*args, **extra_kwargs, **kws)
                # builtin functions return plain tuples for readability
                if isinstance(ret, tuple):
                    ret = language.tuple(ret)
                return ret
            except Exception as e:
                if knobs.compilation.front_end_debugging:
                    raise
                # Normally when we raise a CompilationError, we raise it as
                # `from None`, because the original fileline from the exception
                # is not relevant (and often points into code_generator.py
                # itself).  But when calling a function, we raise as `from e` to
                # preserve the traceback of the original error, which may e.g.
                # be in core.py.
                raise CompilationError(self.jit_fn.src, node, str(e)) from e

        if fn in self.builtin_namespace.values() or (hasattr(fn, '__self__') and not _is_triton_value(fn.__self__)):
            args = map(_unwrap_if_constexpr, args)
        ret = fn(*args, **kws)

        return normalize_value(ret)

    def call_Method(self, node, fn, fn_self, args, kws):
        if isinstance(fn, JITFunction):
            args.insert(0, fn_self)
        return self.call_Function(node, fn, args, kws)

    def visit_Call(self, node):
        fn = _unwrap_if_constexpr(self.visit(node.func))
        if not isinstance(fn, BoundJITMethod):
            static_implementation = self.statically_implemented_functions.get(fn)
            if static_implementation is not None:
                return static_implementation(self, node)

        kws = dict(self.visit(keyword) for keyword in node.keywords)
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                arg = self.visit(arg.value)
                if isinstance(arg, constexpr):
                    arg = arg.value
                if isinstance(arg, tuple):
                    arg = language.core.tuple(arg)
                assert isinstance(arg, language.core.tuple)
                args.extend(arg.values)
            else:
                args.append(self.visit(arg))

        return self.call_Function(node, fn, args, kws)

    def visit_Constant(self, node):
        return constexpr(node.value)

    def visit_BoolOp(self, node: ast.BoolOp):
        method_name = self._method_name_for_bool_op.get(type(node.op))
        if method_name is None:
            raise self._unsupported(
                node, "AST boolean operator '{}' is not (currently) implemented.".format(node.op.__name__))

        nontrivial_values = []

        for subnode in node.values:
            # we visit the values in order, executing their side-effects
            # and possibly early-exiting:
            value = self.visit(subnode)
            if not _is_triton_tensor(value):
                # this is a constexpr, so we might be able to short-circuit:
                bv = bool(value)
                if (bv is False) and (method_name == "logical_and"):
                    # value is falsey so return that:
                    return value
                if (bv is True) and (method_name == "logical_or"):
                    # value is truthy so return that:
                    return value
                # otherwise, our constexpr has no effect on the output of the
                # expression so we do not append it to nontrivial_values.
            else:
                if value.type.is_block():
                    lineno = getattr(node, "lineno", None)
                    if lineno is not None:
                        lineno += self.begin_line
                    warnings.warn_explicit(
                        "Logical operators 'and' and 'or' are deprecated for non-scalar tensors; please use '&' or '|' instead",
                        category=UserWarning,
                        filename=self.file_name,
                        lineno=lineno,
                        source=ast.unparse(node),
                    )
                # not a constexpr so we must append it:
                nontrivial_values.append(value)

        if len(nontrivial_values) == 0:
            # the semantics of a disjunction of falsey values or conjunction
            # of truthy values is to return the final value:
            nontrivial_values.append(value)

        while len(nontrivial_values) >= 2:
            rhs = nontrivial_values.pop()
            lhs = nontrivial_values.pop()
            res = self._apply_binary_method(node, method_name, lhs, rhs)
            nontrivial_values.append(res)

        assert len(nontrivial_values) == 1
        return nontrivial_values[0]

    _method_name_for_bool_op: Dict[Type[ast.boolop], str] = {ast.And: 'logical_and', ast.Or: 'logical_or'}

    def get_Attribute(self, lhs, attr):
        if _is_triton_tensor(lhs) and attr == "T":
            return self.semantic.permute(lhs, (1, 0))
        # NOTE: special case ".value" for BC
        if isinstance(lhs, constexpr) and attr not in ("value", "type"):
            lhs = lhs.value
        attr = getattr(lhs, attr)
        if _is_triton_value(lhs) and isinstance(attr, JITFunction):
            return BoundJITMethod(lhs, attr)
        return attr

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if isinstance(lhs, ModuleType):
            # follow module_map until reaching fixed-point:
            while (name := lhs.__name__) in self.builder.module_map:
                lhs = self.builder.module_map[name]
                if lhs.__name__ == name:
                    break
        return self.get_Attribute(lhs, node.attr)

    def visit_Expr(self, node):
        node.value._is_unused = True
        ast.NodeVisitor.generic_visit(self, node)

    def visit_NoneType(self, node):
        return None

    def visit_JoinedStr(self, node):
        values = list(node.values)
        for i, value in enumerate(values):
            if isinstance(value, ast.Constant):
                values[i] = str(value.value)
            elif isinstance(value, ast.FormattedValue):
                conversion_code = value.conversion
                evaluated = self.visit(value.value)
                if not _is_constexpr(evaluated):
                    raise self._unsupported(
                        node,
                        "Cannot evaluate f-string containing non-constexpr conversion values, found conversion of type "
                        + str(type(evaluated)))
                values[i] = ("{}" if conversion_code < 0 else "{!" + chr(conversion_code) + "}").format(evaluated.value)
            else:
                raise AssertionError("encountered unexpected node of type {} in a JoinedStr node".format(type(value)))
        return ''.join(values)

    def visit(self, node):
        if node is None:
            return
        last_node = self.cur_node
        last_loc = self.builder.get_loc()
        self.cur_node = node
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            here_loc = self.builder.create_loc(self.file_name, self.begin_line + node.lineno,
                                               self.begin_col + node.col_offset)
            if self.name_loc_as_prefix is not None:
                self.builder.set_loc(self.builder.create_name_loc(self.name_loc_as_prefix, here_loc))
            else:
                self.builder.set_loc(here_loc)
        try:
            ret = super().visit(node)
        except CompilationError:
            raise
        except Exception as e:
            if knobs.compilation.front_end_debugging:
                raise
            # Wrap the error in a CompilationError which contains the source
            # of the @jit function.
            raise CompilationError(self.jit_fn.src, self.cur_node, repr(e)) from None

        # Reset the location to the last one before the visit
        if last_loc:
            # If last_loc doesn't already have a name, set its name.
            if self.name_loc_as_prefix is not None and last_loc.get_name() is None:
                last_loc.set_name(self.name_loc_as_prefix)
            self.cur_node = last_node
            self.builder.set_loc(last_loc)
        return ret

    def generic_visit(self, node):
        raise self._unsupported(node, "unsupported AST node type: {}".format(type(node).__name__))

    def execute_static_assert(self, node: ast.Call) -> None:
        arg_count = len(node.args)
        if not (0 < arg_count <= 2) or len(node.keywords):
            raise TypeError("`static_assert` requires one or two positional arguments only")

        passed = _unwrap_if_constexpr(self.visit(node.args[0]))
        if not isinstance(passed, bool):
            raise NotImplementedError(
                "Assertion condition could not be determined at compile-time. Make sure that it depends only on `constexpr` values"
            )
        if not passed:
            if arg_count == 1:
                message = ""
            else:
                try:
                    message = self.visit(node.args[1])
                except Exception as e:
                    message = "<failed to evaluate assertion message: " + repr(e) + ">"

            raise CompileTimeAssertionFailure(self.jit_fn.src, node, _unwrap_if_constexpr(message))
        return None

    def static_executor(python_fn):

        def ret(self, node: ast.Call):
            kws = {
                name: _unwrap_if_constexpr(value)
                for name, value in (self.visit(keyword) for keyword in node.keywords)
            }
            args = [_unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
            return constexpr(python_fn(*args, **kws))

        return ret

    from ..experimental.gluon import language as ttgl
    statically_implemented_functions: Dict[object, Callable[[ast.Call], Any]] = {
        language.core.static_assert: execute_static_assert,
        language.core.static_print: static_executor(print),
        ttgl.static_assert: execute_static_assert,
        ttgl.static_print: static_executor(print),
        int: static_executor(int),
        len: static_executor(len),
    }


def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    arg_types = [None] * len(fn.arg_names)

    for k, v in src.signature.items():
        idx = fn.arg_names.index(k)
        arg_types[idx] = str_to_ty(v, None)

    def constexpr_type(value):
        if isinstance(value, builtins.tuple):
            fields = value._fields if is_namedtuple(type(value)) else None
            return language.tuple_type([constexpr_type(v) for v in value], fields)
        return constexpr(value).type

    def apply_constexpr_types(argument, indices, value):
        index = indices.pop()
        if len(indices) == 0:
            if isinstance(argument, list):
                argument[index] = constexpr_type(value)
            else:
                argument.types[index] = constexpr_type(value)
        else:
            apply_constexpr_types(argument[index], indices, value)

    for path, value in src.constants.items():
        apply_constexpr_types(arg_types, list(path)[::-1], value)

    prototype = ASTFunction([], arg_types, src.attrs)
    # query function representation
    from collections import namedtuple
    leaves = filter(lambda v: len(v) == 1, src.constants)
    constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
    signature = src.signature
    proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, signature)
    generator = CodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                              jit_fn=fn, is_kernel=True, file_name=fn.file_name, begin_line=fn.def_file_line_number,
                              begin_col=fn.def_file_col_number, options=options, codegen_fns=codegen_fns,
                              module_map=module_map, module=module, is_gluon=fn.is_gluon())
    generator.visit(fn.parse())
    module = generator.module
    # module takes ownership of the context
    module.context = context
    if not module.verify():
        if not fn.is_gluon():
            print(module)
        raise RuntimeError("error encountered during parsing")
    return module
<|endoftext|>from __future__ import annotations
import hashlib
import json
from .._C.libtriton import get_cache_invalidating_env_vars, ir
from ..backends import backends
from ..backends.compiler import Language
from ..backends.compiler import BaseBackend, GPUTarget
from .. import __version__, knobs
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager, get_cache_key
from ..runtime.driver import driver
from ..tools.disasm import get_sass
from pathlib import Path
import re
import functools
import os
import time
import copy

# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
# - (attributes \{[\S\s]+\})? : optionally match attributes enclosed in braces and capture it as group 3
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ptx": ptx_prototype_pattern,
}

ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ptx": ptx_arg_type_pattern,
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    tma = re.search(r'tt.nv_tma_desc = 1', x)
    if tma is not None:
        return 'nvTmaDesc'
    x = re.sub(r' {[^}]+}', '', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


class ASTSource:

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        self.fn = fn
        self.language = Language.TRITON
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = signature
        self.constants = dict()
        if constexprs is not None:
            for k, v in constexprs.items():
                k = (fn.arg_names.index(k), ) if isinstance(k, str) else k
                assert isinstance(k, tuple)
                self.constants[k] = v
        self.attrs = attrs or dict()
        for k in self.signature.keys():
            if not isinstance(k, str):
                raise TypeError("Signature keys must be string")

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]
        get_key = lambda x: x.cache_key if hasattr(x, 'cache_key') else str(x)
        constants_key = '-'.join([get_key(v) for k, v in sorted(self.constants.items())])
        key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                           module_map=module_map)

    def parse_options(self):
        return dict()


class IRSource:

    def __init__(self, path, context, backend):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.language = Language.TRITON
        self.src = path.read_text()
        ir.load_dialects(context)
        backend.load_dialects(context)

        # We don't have a easy-to-use PTX parser that we can use, so keep that regex for now.
        # TODO - replace with a proper parser
        if self.ext == "ptx":
            match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
            self.name = match.group(1)
            signature = match.group(2)
            types = re.findall(arg_type_pattern[self.ext], signature)
            self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}
        else:
            self.module = ir.parse_mlir_module(self.path, context)
            fn_name = self.module.get_entry_func_name()
            self.name = "@" + fn_name
            funcOp = self.module.get_function(fn_name)
            func_ty = self.module.get_function_signature(funcOp)
            self.signature = {k: ty for k, ty in enumerate(func_ty)}

    def hash(self):
        return hashlib.sha256(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        self.module.context = context
        return self.module

    def parse_options(self):
        if self.ext == "ttgir":
            num_warps = self.module.get_int_attr("ttg.num-warps")
            assert num_warps is not None, "Unable to parse ttg.num-warps attribute"
            options = {'num_warps': num_warps}
            num_ctas = self.module.get_int_attr("ttg.num-ctas")
            if num_ctas is not None:
                options['num_ctas'] = num_ctas
            return options
        return dict()


@functools.lru_cache()
def max_shared_mem(device):
    return driver.active.utils.get_device_properties(device)["max_shared_mem"]


def parse(full_name, ext, context):
    if ext == "ttir" or ext == "ttgir":
        module = ir.parse_mlir_module(full_name, context)
        module.context = context
        return module
    if ext == "llir" or ext == "ptx" or ext == "amdgcn":
        return Path(full_name).read_text()
    if ext == "cubin" or ext == "hsaco":
        return Path(full_name).read_bytes()


def filter_traceback(e: BaseException):
    """
    Removes code_generator.py and related files from tracebacks.

    These are uninteresting to the user -- "just show me *my* code!"
    """
    if knobs.compilation.front_end_debugging:
        return

    if e.__cause__ is not None:
        filter_traceback(e.__cause__)
    if e.__context__ is not None:
        filter_traceback(e.__context__)

    # If a user has a file that matches one of these, they're out of luck.
    BAD_FILES = [
        "/triton/compiler/code_generator.py",
        "/ast.py",
    ]
    BAD_FILES = [bad_file.replace("/", os.sep) for bad_file in BAD_FILES]

    tb = e.__traceback__
    frames = []
    while tb is not None:
        if not any(f for f in BAD_FILES if tb.tb_frame.f_code.co_filename.endswith(f)):
            frames.append(tb)
        tb = tb.tb_next

    for (cur_frame, next_frame) in zip(frames, frames[1:]):
        cur_frame.tb_next = next_frame

    if not frames:
        e.__traceback__ = None
    else:
        frames[-1].tb_next = None
        e.__traceback__ = frames[0]


class CompileTimer:

    def __init__(self) -> None:
        self.start: float = time.time()
        self.ir_initialization_end: float | None = None
        self.lowering_stage_ends: list[tuple[str, float]] = []
        self.store_results_end: float | None = None

    def finished_ir_initialization(self) -> None:
        self.ir_initialization_end = time.time()

    def stage_finished(self, stage_name: str) -> None:
        self.lowering_stage_ends.append((stage_name, time.time()))

    def end(self) -> knobs.CompileTimes:
        timestamp = time.time()
        if self.ir_initialization_end is None:
            self.ir_initialization_end = timestamp
        else:
            self.store_results_end = timestamp

        def delta(start: float, end: float | None) -> int:
            if end is None:
                return 0
            return int((end - start) * 1000000)

        lowering_stage_durations = []
        stage_start = self.ir_initialization_end
        for stage_name, stage_end in self.lowering_stage_ends:
            lowering_stage_durations.append((stage_name, delta(stage_start, stage_end)))
            stage_start = stage_end

        return knobs.CompileTimes(
            ir_initialization=delta(self.start, self.ir_initialization_end),
            lowering_stages=lowering_stage_durations,
            store_results=delta(stage_start, self.store_results_end),
        )


def compile(src, target=None, options=None, _env_vars=None):
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        context = ir.context()
        src = IRSource(src, context, backend)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    if knobs.runtime.add_stages_inspection_hook is not None:
        inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
        key += inspect_stages_key
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = knobs.compilation.override
    enable_ir_dump = knobs.compilation.dump_ir
    store_only_binary = knobs.compilation.store_binary_only
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res

    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options, src.language)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource):
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    try:
        module = src.make_ir(target, options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise

    if ir_source:
        ir_filename = f"{file_name}.{src.ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    use_ir_loc = knobs.compilation.use_ir_loc
    if ir_source and use_ir_loc:
        module.create_location_snapshot(src.path)
        print(f"Creating new locations for {src.path}")

    if compilation_listener:
        timer.finished_ir_initialization()
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if fn_override_manager is None:
            # Users can override kernels at scale by setting `ir_override` in autotune config
            # without TRITON_KERNEL_OVERRIDE
            if (ir_override := metadata.get("ir_override", None)) and ir_override.endswith(f".{ext}"):
                next_module = parse(ir_override, ext, context)
        elif full_name := fn_override_manager.get_file(ir_filename):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
            if ext == "cubin":
                sass = get_sass(next_module)
                fn_dump_manager.put(sass, file_name + ".sass")
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)

    # notify any listener
    if compilation_listener:
        compilation_listener(src=src, metadata=metadata, metadata_group=metadata_group, times=timer.end(),
                             cache_hit=False)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)


def make_backend(target: GPUTarget) -> BaseBackend:
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
    return actives[0](target)


class LazyDict:

    def __init__(self, data):
        self.data = data
        self.extras = []

    def get(self):
        for func, args in self.extras:
            self.data = self.data | func(*args)
        self.extras.clear()
        return self.data

    def add(self, func, args):
        self.extras.append((func, args))


class AsmDict(dict):

    def __missing__(self, key):

        if key == "sass":
            value = get_sass(self["cubin"])
        else:
            raise KeyError("Unknown key: '%s'" % key)

        self[key] = value
        return value


def _raise_error(err, *args, **kwargs):
    raise copy.deepcopy(err)


class CompiledKernel:

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None
        self._run = None

    def __del__(self):

        if self.module is not None:
            if knobs.runtime.kernel_unload_hook is not None:
                knobs.runtime.kernel_unload_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

            driver.active.utils.unload_module(self.module)
            self.module = None

    def _init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            # clone the exception object so that the one saved in the closure
            # of the partial function below doesn't get assigned a stack trace
            # after the subsequent raise. otherwise, the CompiledKernel instance
            # saved in the (global) kernel cache will keep references to all the
            # locals in the traceback via the exception instance in the closure.
            cloned_err = copy.deepcopy(err)
            self._run = functools.partial(_raise_error, cloned_err)
            raise err

        device = driver.active.get_current_device()
        # create launcher
        self._run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.target.arch == 107:
                max_tmem_size = 576
            if self.metadata.tmem_size > max_tmem_size:
                raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        if knobs.runtime.kernel_load_start_hook is not None:
            knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        if knobs.runtime.kernel_load_end_hook is not None:
            knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run

    def launch_metadata(self, grid, stream, *args):
        if knobs.runtime.launch_enter_hook is None:
            return None
        self._init_handles()
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_dict = {name: arg for name, arg in zip(self.src.fn.arg_names, args)}
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

        return runner
<|endoftext|>import ast
from typing import Optional
from ..errors import TritonError


class CompilationError(TritonError):
    """Base class for all errors raised during compilation"""
    source_line_count_max_in_message = 12

    def _format_message(self) -> str:
        node = self.node
        if self.src is None:
            source_excerpt = " <source unavailable>"
        else:
            if hasattr(node, 'lineno'):
                source_excerpt = self.src.split('\n')[:node.lineno][-self.source_line_count_max_in_message:]
                if source_excerpt:
                    source_excerpt.append(' ' * node.col_offset + '^')
                    source_excerpt = '\n'.join(source_excerpt)
                else:
                    source_excerpt = " <source empty>"
            else:
                source_excerpt = self.src

        message = "at {}:{}:\n{}".format(node.lineno, node.col_offset, source_excerpt) if hasattr(
            node, 'lineno') else source_excerpt
        if self.error_message:
            message += '\n' + self.error_message
        return message

    def __init__(self, src: Optional[str], node: ast.AST, error_message: Optional[str] = None):
        self.src = src
        self.node = node
        self.error_message = error_message
        self.message = self._format_message()

    def __str__(self):
        return self.message

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return type(self), (self.src, self.node, self.error_message)


class CompileTimeAssertionFailure(CompilationError):
    """Specific exception for failed tests in `static_assert` invocations"""
    pass


class UnsupportedLanguageConstruct(CompilationError):
    pass
<|endoftext|>from __future__ import annotations

import functools
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import Literal, overload

from triton.runtime import driver as runtime_driver
from triton.runtime.build import compile_module_from_file

_THIS_DIR = Path(__file__).resolve().parent
_GSAN_SOURCE_PATH = _THIS_DIR / "src" / "GSanAllocator.cc"


class ShareableHandleType(IntEnum):
    POSIX_FILE_DESCRIPTOR = 0x1
    FABRIC = 0x8


@functools.lru_cache()
def _load_gsan_module() -> ModuleType:
    if runtime_driver.active.get_current_target().backend != "cuda":
        raise RuntimeError("GSan allocator requires the CUDA backend.")

    from triton.backends.nvidia.driver import library_dirs, include_dirs

    return compile_module_from_file(
        src_path=str(_GSAN_SOURCE_PATH),
        name="gsan_allocator",
        library_dirs=library_dirs(),
        include_dirs=include_dirs,
        libraries=["libcuda.so.1"],
    )


@functools.lru_cache()
def _compile_gsan_allocator() -> str:
    # __file__ for a compiled module is the so file
    return _load_gsan_module().__file__


@functools.lru_cache()
def get_allocator():
    from torch.cuda.memory import CUDAPluggableAllocator
    so_name = _compile_gsan_allocator()
    return CUDAPluggableAllocator(so_name, "gsanMalloc", "gsanFree")


def configure(
    *,
    device_ranks: dict[int, int] | None = None,
    num_devices: int | None = None,
    rng_seed: int | None = None,
    clock_buffer_size: int | None = None,
    handle_type: ShareableHandleType | None = None,
) -> None:
    """Configures the process-local GSan state.

    GSan keeps one allocator configuration per process. Call this before the
    allocator initializes runtime state, or before calling :func:`freeze_config`.
    Once frozen, later calls to :func:`configure` raise ``RuntimeError``.

    Args:
        device_ranks (dict[int, int], optional): Mapping from local CUDA device index to the
            logical GSan device id. This enables gsan to be used with multi-node nvlink domains,
            or simply processes with different CUDA_VISIBLE_DEVICES settings. Each value must be
            unique and in ``[0, num_devices)``. If None, defaults to a 1:1 mapping from device
            index to device id.
        num_devices (int, optional): Total number of logical GSan devices in the topology. If
            None, defaults to the number of visible CUDA devices.
        rng_seed (int, optional): Optional seed for GSan's stochastic read-clock sampling. Use this
            to make sampling decisions reproducible across runs when debugging sanitizer behavior.
            If omitted, GSan first checks ``TRITON_GSAN_SEED`` and otherwise generates a random
            seed when the allocator runtime state is initialized.
        clock_buffer_size (int, optional): When doing an atomic release operation, GSan uses a
            circular buffer to record what memory accesses have been released. If the writing CTA
            has done more release writes than there are circular buffer entries, then the atomic
            flag cannot be read and you will need to increase the buffer size. If omitted, GSan
            first checks ``TRITON_GSAN_CLOCK_BUFFER_SIZE`` and otherwise defaults to 1024.
        handle_type (ShareableHandleType, optional): Type of shareable handle requested for GSan
            allocations. If omitted, GSan uses fabric handles when ``PYTORCH_CUDA_ALLOC_CONF``
            contains ``fabric_handles:True`` and otherwise uses POSIX file descriptors.
    """
    _load_gsan_module().configure(device_ranks, num_devices, rng_seed, clock_buffer_size, handle_type)


def freeze_config() -> None:
    """Prevents later `configure(...)` calls from changing allocator configuration."""
    _load_gsan_module().freeze_config()


def create_mem_pool():
    from torch.cuda.memory import MemPool
    return MemPool(get_allocator().allocator())


def gsan_malloc(size: int, device: int, stream: int = 0) -> int:
    module = _load_gsan_module()
    return module.malloc(size, device, stream)


def gsan_free(ptr: int, device: int, size: int = 0, stream: int = 0) -> None:
    module = _load_gsan_module()
    module.free(ptr, device, size, stream)


def get_reserve_pointer() -> int:
    return _load_gsan_module().get_reserve_pointer()


def get_reserve_size() -> int:
    return _load_gsan_module().get_reserve_size()


def get_global_state_pointer() -> int:
    return _load_gsan_module().get_global_state_pointer()


def get_device_rank(device: int) -> int:
    return _load_gsan_module().get_device_rank(device)


def get_runtime_state_layout(device: int) -> dict[str, int]:
    module = _load_gsan_module()
    return module.get_runtime_state_layout(device)


@overload
def export_allocation_handles(
    ptr: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> tuple[int, int, int]:
    ...


@overload
def export_allocation_handles(
    ptr: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> tuple[bytes, bytes, int]:
    ...


def export_allocation_handles(ptr, handle_type):
    module = _load_gsan_module()
    return module.export_allocation_handles(ptr, handle_type)


def export_allocation_memhandle_regions(ptr: int) -> tuple[int, int, int, int]:
    module = _load_gsan_module()
    return module.export_allocation_memhandle_regions(ptr)


@overload
def import_allocation_handles(
    real_handle: int,
    shadow_handle: int,
    alloc_size: int,
    device: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> int:
    ...


@overload
def import_allocation_handles(
    real_handle: bytes,
    shadow_handle: bytes,
    alloc_size: int,
    device: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> int:
    ...


def import_allocation_handles(
    real_handle,
    shadow_handle,
    alloc_size,
    device,
    handle_type,
):
    module = _load_gsan_module()
    return module.import_allocation_handles(
        real_handle,
        shadow_handle,
        alloc_size,
        device,
        handle_type,
    )


@overload
def export_runtime_state_handle(
    device: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> tuple[int, int]:
    ...


@overload
def export_runtime_state_handle(
    device: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> tuple[bytes, int]:
    ...


def export_runtime_state_handle(device, handle_type):
    module = _load_gsan_module()
    return module.export_runtime_state_handle(device, handle_type)


@overload
def import_runtime_state_handle(
    handle: int,
    alloc_size: int,
    peer_device: int,
    device: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> None:
    ...


@overload
def import_runtime_state_handle(
    handle: bytes,
    alloc_size: int,
    peer_device: int,
    device: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> None:
    ...


def import_runtime_state_handle(
    handle,
    alloc_size,
    peer_device,
    device,
    handle_type,
):
    module = _load_gsan_module()
    module.import_runtime_state_handle(
        handle,
        alloc_size,
        peer_device,
        device,
        handle_type,
    )


def free_allocation(ptr: int, device: int) -> None:
    gsan_free(ptr, device, size=0, stream=0)
<|endoftext|>from ._allocator import ShareableHandleType, configure, create_mem_pool, freeze_config, get_allocator

__all__ = ["ShareableHandleType", "configure", "create_mem_pool", "freeze_config", "get_allocator"]

_LAZY_LOAD_MODULES = {"symmetric_memory"}


def __getattr__(name):
    if name in _LAZY_LOAD_MODULES:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
<|endoftext|>from __future__ import annotations

import ctypes

import torch

_DLPACK_CAPSULE_NAME = b"dltensor"
_DL_UINT = 1
_DL_BITS_UINT8 = 8
_DL_LANES = 1
_DL_CUDA = 2


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorHandle = ctypes.POINTER(_DLManagedTensor)
_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, _DLManagedTensorHandle)

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

_PyCapsuleNew = ctypes.PYFUNCTYPE(
    ctypes.py_object,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
)
_PyCapsule_New = _PyCapsuleNew(("PyCapsule_New", ctypes.pythonapi))

# Keep the ctypes-owned metadata alive until PyTorch drops the imported tensor.
_DLPACK_STATE: dict[int, object] = {}


@_DLManagedTensorDeleter
def _dl_managed_tensor_deleter(dl_managed_tensor: _DLManagedTensorHandle) -> None:
    if not dl_managed_tensor:
        return
    _DLPACK_STATE.pop(ctypes.addressof(dl_managed_tensor.contents), None)


class _DLPackCudaPtrView:

    def __init__(self, data_ptr: int, numel: int, device_index: int):
        self._managed_tensor = _DLManagedTensor()
        self._shape = (ctypes.c_int64 * 1)(numel)
        self._strides = (ctypes.c_int64 * 1)(1)

        self._managed_tensor.dl_tensor.data = ctypes.c_void_p(data_ptr)
        self._managed_tensor.dl_tensor.device = _DLDevice(_DL_CUDA, device_index)
        self._managed_tensor.dl_tensor.ndim = 1
        self._managed_tensor.dl_tensor.dtype = _DLDataType(_DL_UINT, _DL_BITS_UINT8, _DL_LANES)
        self._managed_tensor.dl_tensor.shape = ctypes.cast(self._shape, ctypes.POINTER(ctypes.c_int64))
        self._managed_tensor.dl_tensor.strides = ctypes.cast(self._strides, ctypes.POINTER(ctypes.c_int64))
        self._managed_tensor.dl_tensor.byte_offset = 0
        self._managed_tensor.manager_ctx = None
        self._managed_tensor.deleter = _dl_managed_tensor_deleter

    def __dlpack_device__(self) -> tuple[int, int]:
        device = self._managed_tensor.dl_tensor.device
        return int(device.device_type), int(device.device_id)

    def __dlpack__(self, stream: int | None = None):
        # These pointer views do not carry producer-stream semantics. Callers are
        # responsible for any synchronization before exposing the pointer.
        _ = stream
        dl_managed_tensor_ptr = self.managed_tensor_ptr
        _DLPACK_STATE[dl_managed_tensor_ptr] = self
        return _PyCapsule_New(
            ctypes.c_void_p(dl_managed_tensor_ptr),
            _DLPACK_CAPSULE_NAME,
            None,
        )

    @property
    def managed_tensor_ptr(self) -> int:
        return ctypes.addressof(self._managed_tensor)


def uint8_cuda_tensor_from_ptr(data_ptr: int, numel: int, device_index: int) -> torch.Tensor:
    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be >= 0, got {numel}")
    if numel == 0:
        return torch.empty((0, ), dtype=torch.uint8, device=f"cuda:{device_index}")
    view = _DLPackCudaPtrView(int(data_ptr), numel, int(device_index))
    try:
        return torch.from_dlpack(view)
    except Exception:
        _DLPACK_STATE.pop(view.managed_tensor_ptr, None)
        raise
<|endoftext|>from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass

import triton
import triton.language as tl

from ._allocator import get_device_rank, get_runtime_state_layout
from ._utils import uint8_cuda_tensor_from_ptr


@dataclass(frozen=True)
class _RuntimeStateLayout:
    thread_state_region: object
    thread_state_stride_bytes: int
    thread_state_header_size_bytes: int
    num_sms: int
    num_threads: int


@functools.lru_cache()
def _runtime_state_layout(runtime_state_device: int, access_device: int) -> _RuntimeStateLayout:
    layout = get_runtime_state_layout(runtime_state_device)
    region_size = layout["thread_state_stride_bytes"] * layout["num_sms"]
    thread_state_region = uint8_cuda_tensor_from_ptr(layout["thread_state_base_ptr"], region_size, access_device)
    return _RuntimeStateLayout(
        thread_state_region=thread_state_region,
        thread_state_stride_bytes=layout["thread_state_stride_bytes"],
        thread_state_header_size_bytes=layout["thread_state_header_size_bytes"],
        num_sms=layout["num_sms"],
        num_threads=layout["num_threads"],
    )


@contextmanager
def _compile_without_gsan():
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = ""
        yield


@functools.lru_cache()
def _compiled_sync_kernel(device: int, stride_bytes: int, num_sms: int, num_threads: int, header_bytes: int,
                          BLOCK_SIZE):
    with _compile_without_gsan():
        return _synchronize_vector_clocks_kernel.warmup(
            triton.MockTensor(tl.uint8),
            stride_bytes,
            num_sms,
            num_threads,
            header_bytes,
            BLOCK_SIZE=BLOCK_SIZE,
            grid=(1, ),
            num_warps=1,
        )


@triton.jit
def _synchronize_vector_clocks_kernel(thread_state_region, stride_bytes, num_sms, num_threads, header_bytes,
                                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_threads
    max_clocks = tl.full([BLOCK_SIZE], 0, tl.uint16)

    for sm in range(num_sms):
        vector_clock_ptr = (thread_state_region + sm * stride_bytes + header_bytes).to(tl.pointer_type(tl.uint16))
        thread_clocks = tl.load(vector_clock_ptr + offsets, mask=mask, other=0)
        max_clocks = tl.maximum(thread_clocks, max_clocks)

    for sm in range(num_sms):
        vector_clock_ptr = (thread_state_region + sm * stride_bytes + header_bytes).to(tl.pointer_type(tl.uint16))
        tl.store(vector_clock_ptr + offsets, max_clocks, mask=mask)


def _check_compatible_runtime_state_layout(lhs: _RuntimeStateLayout, rhs: _RuntimeStateLayout) -> None:
    if (lhs.thread_state_stride_bytes != rhs.thread_state_stride_bytes
            or lhs.thread_state_header_size_bytes != rhs.thread_state_header_size_bytes or lhs.num_sms != rhs.num_sms
            or lhs.num_threads != rhs.num_threads):
        raise RuntimeError("GSan runtime state layout mismatch across synchronized devices.")


@triton.jit
def _synchronize_process_group_barrier_kernel(local_thread_state_region, peer_thread_state_regions, stride_bytes,
                                              num_sms, num_threads, header_bytes, BLOCK_SIZE: tl.constexpr,
                                              N_PEERS: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_threads
    max_clocks = tl.full([BLOCK_SIZE], 0, tl.uint16)

    for peer in tl.static_range(N_PEERS):
        peer_thread_state_region = peer_thread_state_regions[peer]
        # Each rank first collapses its local per-SM clocks, so every SM on a peer
        # carries the same vector clock at this point. Reading SM0 is sufficient.
        vector_clock_ptr = (peer_thread_state_region + header_bytes).to(tl.pointer_type(tl.uint16))
        peer_clocks = tl.load(vector_clock_ptr + offsets, mask=mask, other=0)
        max_clocks = tl.maximum(peer_clocks, max_clocks)

    for sm in range(num_sms):
        vector_clock_ptr = (local_thread_state_region + sm * stride_bytes + header_bytes).to(tl.pointer_type(tl.uint16))
        tl.store(vector_clock_ptr + offsets, max_clocks, mask=mask)


def synchronize_launch_stream(device: int) -> None:
    """This models the implicit synchronization between kernel launches.

    To do this, we compute the elementwise maximum of all SMs' vector clocks, and set every SM's clock to this starting point.

    This makes all reads and writes transitively visible to other threads.
    """
    layout = _runtime_state_layout(get_device_rank(device), device)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(layout.num_threads, BLOCK_SIZE), 1, 1)
    kernel = _compiled_sync_kernel(
        device,
        layout.thread_state_stride_bytes,
        layout.num_sms,
        layout.num_threads,
        layout.thread_state_header_size_bytes,
        BLOCK_SIZE,
    )
    kernel[grid](
        layout.thread_state_region,
        layout.thread_state_stride_bytes,
        layout.num_sms,
        layout.num_threads,
        layout.thread_state_header_size_bytes,
        BLOCK_SIZE,
    )


def synchronize_process_group_barrier(device: int, peer_devices: tuple[int, ...]) -> None:
    """Join vector clocks across the devices participating in a process-group barrier.

    The peer runtime-state mappings are already imported by symmetric-memory rendezvous.
    This helper computes the elementwise maximum of all participating devices' per-SM
    vector clocks and writes that join back into the local device's per-SM state.
    """
    if not peer_devices:
        return

    local_layout = _runtime_state_layout(get_device_rank(device), device)
    peer_regions = []
    for peer_device in peer_devices:
        peer_layout = _runtime_state_layout(peer_device, device)
        _check_compatible_runtime_state_layout(local_layout, peer_layout)
        peer_regions.append(peer_layout.thread_state_region)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(local_layout.num_threads, BLOCK_SIZE), 1, 1)
    with _compile_without_gsan():
        _synchronize_process_group_barrier_kernel[grid](
            local_layout.thread_state_region,
            tuple(peer_regions),
            local_layout.thread_state_stride_bytes,
            local_layout.num_sms,
            local_layout.num_threads,
            local_layout.thread_state_header_size_bytes,
            BLOCK_SIZE=BLOCK_SIZE,
            N_PEERS=len(peer_regions),
            num_warps=1,
        )
<|endoftext|>from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._allocator import get_global_state_pointer
from triton._C.libtriton.gsan_testing import thread_state_address, SHADOW_GRANULARITY_BYTES, PER_DEVICE_STATE_STRIDE_BYTES, GLOBAL_STATE_SIZE_BYTES, shadow_cell_address, thread_state_stride_bytes, SHADOW_CELL_SIZE_BYTES
from ._testing import (decode_global_state_tensor, decode_shadow_cell_tensor, decode_thread_state_tensor)
from ._utils import uint8_cuda_tensor_from_ptr


@triton.jit
def nanosleep(duration):
    duration = tl.to_tensor(duration)
    tl.inline_asm_elementwise("nanosleep.u32 $1; mov.b32 $0, 0;", "=r, r", [duration], tl.int32, is_pure=False, pack=1)


@triton.jit
def atomic_poll(ptr, expect, sem: tl.constexpr = "relaxed", scope: tl.constexpr = "gpu"):
    while tl.atomic_add(ptr, 0, sem=sem, scope=scope) != expect:
        nanosleep(100)


def shadow_cell_tensor_from_address(real_address: int, *, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    shadow_ptr = shadow_cell_address(real_address)
    return uint8_cuda_tensor_from_ptr(shadow_ptr, SHADOW_CELL_SIZE_BYTES, device_index)


def shadow_cell_from_address(real_address: int, *, device_index: int | None = None):
    return decode_shadow_cell_tensor(shadow_cell_tensor_from_address(real_address, device_index=device_index))


def global_state_tensor(*, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    ptr = get_global_state_pointer() + device_index * PER_DEVICE_STATE_STRIDE_BYTES
    return uint8_cuda_tensor_from_ptr(ptr, GLOBAL_STATE_SIZE_BYTES, device_index)


def global_state(*, device_index: int | None = None):
    return decode_global_state_tensor(global_state_tensor(device_index=device_index))


def thread_state_tensor(smid: int, *, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    gs = global_state(device_index=device_index)
    ptr = thread_state_address(
        get_global_state_pointer() + device_index * PER_DEVICE_STATE_STRIDE_BYTES,
        gs.num_threads,
        gs.clock_buffer_size,
        smid,
    )
    size = thread_state_stride_bytes(gs.num_threads, gs.clock_buffer_size)
    return uint8_cuda_tensor_from_ptr(ptr, size, device_index)


def thread_state_from_smid(smid: int, *, device_index: int | None = None):
    if device_index is None:
        device_index = torch.cuda.current_device()
    gs = global_state(device_index=device_index)
    return decode_thread_state_tensor(
        thread_state_tensor(smid, device_index=device_index),
        gs.num_threads,
        gs.clock_buffer_size,
    )


def shadow_tensor_for(real: torch.Tensor) -> torch.Tensor:
    shadow_ptr = shadow_cell_address(real.data_ptr())
    nbytes = real.untyped_storage().nbytes()
    num_cells = triton.cdiv(nbytes, SHADOW_GRANULARITY_BYTES)
    shadow_size = num_cells * SHADOW_CELL_SIZE_BYTES
    return uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, real.device.index)


@triton.jit
def store_one_i32(ptr):
    tl.store(ptr, 1)


@triton.jit
def load_one_i32(ptr, out_ptr):
    value = tl.load(ptr)
    tl.store(out_ptr, value)
<|endoftext|>from __future__ import annotations

import atexit
import array
import contextlib
import errno
import functools
import math
import os
import socket
import struct
import tempfile
import time
import uuid
import weakref
from collections.abc import Sequence
from typing import TypeAlias
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from . import _stream_sync
from ._allocator import (ShareableHandleType, configure as _configure, create_mem_pool, export_allocation_handles,
                         export_runtime_state_handle, free_allocation, import_allocation_handles,
                         import_runtime_state_handle)
from ._utils import uint8_cuda_tensor_from_ptr

_RendezvousCacheKey: TypeAlias = tuple[int, int, int]
_RuntimeBootstrapCacheKey: TypeAlias = tuple[int, int]

_RENDEZVOUS_FLAG_RUNTIME_STATE_FD = 1 << 0
_RETRYABLE_UNIX_CONNECT_ERRNOS = frozenset((errno.ENOENT, errno.ENOTSOCK, errno.ECONNREFUSED))


def _normalize_size(size: tuple[object, ...]) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = size[0]
    return tuple(int(v) for v in size)


@functools.lru_cache()
def _get_mem_pool():
    return create_mem_pool()


@atexit.register
def _clear_mem_pool_cache() -> None:
    _get_mem_pool.cache_clear()


def empty(*size, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    shape = _normalize_size(size)
    if dtype is None:
        dtype = torch.get_default_dtype()

    dev: torch.device
    if device is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    else:
        dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("GSan symmetric memory requires CUDA tensors.")
    device_index = torch.cuda.current_device() if dev.index is None else dev.index
    dev = torch.device("cuda", device_index)

    with torch.cuda.use_mem_pool(_get_mem_pool()):
        return torch.empty(shape, dtype=dtype, device=dev)


def _resolve_group(group) -> tuple[dist.ProcessGroup, str]:
    if isinstance(group, dist.ProcessGroup):
        return group, str(group.group_name)
    if isinstance(group, str):
        return c10d._resolve_process_group(group), group
    raise TypeError(f"rendezvous: unsupported group type: {type(group)}")


def configure(group, *, rng_seed: int | None = None, clock_buffer_size: int | None = None):
    process_group, _ = _resolve_group(group)
    _configure(
        device_ranks={torch.cuda.current_device(): dist.get_rank(process_group)},
        num_devices=dist.get_world_size(process_group),
        rng_seed=rng_seed,
        clock_buffer_size=clock_buffer_size,
        handle_type=ShareableHandleType.POSIX_FILE_DESCRIPTOR,
    )


def _send_fds(
    sock: socket.socket,
    *,
    src_rank: int,
    src_device_index: int,
    real_fd: int,
    shadow_fd: int,
    runtime_state_fd: int | None = None,
) -> None:
    flags = _RENDEZVOUS_FLAG_RUNTIME_STATE_FD if runtime_state_fd is not None else 0
    msg = struct.pack("iii", src_rank, src_device_index, flags)
    fds_list = [real_fd, shadow_fd]
    if runtime_state_fd is not None:
        fds_list.append(runtime_state_fd)
    fds = array.array("i", fds_list)
    sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())])


def _recv_fds(sock: socket.socket) -> tuple[int, int, int, int, int | None]:
    int_size = array.array("i").itemsize
    msg, ancdata, _, _ = sock.recvmsg(12, socket.CMSG_SPACE(3 * int_size))
    if len(msg) != 12:
        raise RuntimeError("Failed to receive rank metadata during rendezvous.")
    src_rank, src_device_index, flags = struct.unpack("iii", msg)
    recv_fds: list[int] = []
    for level, ctype, data in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            arr = array.array("i")
            arr.frombytes(data[:len(data) - (len(data) % int_size)])
            recv_fds.extend(arr.tolist())
    needs_runtime_fd = (flags & _RENDEZVOUS_FLAG_RUNTIME_STATE_FD) != 0
    expected_fd_count = 3 if needs_runtime_fd else 2
    if len(recv_fds) < expected_fd_count:
        raise RuntimeError("Failed to receive file descriptors during rendezvous.")
    runtime_state_fd = recv_fds[2] if needs_runtime_fd else None
    return src_rank, src_device_index, recv_fds[0], recv_fds[1], runtime_state_fd


def _import_peer_ptrs(
    *,
    received_fds: dict[int, tuple[int, int, int | None]],
    metas: list[dict],
    rank: int,
    world_size: int,
    device_index: int,
    base_ptr: int,
) -> tuple[int, ...]:
    peer_ptrs = [0] * world_size
    peer_ptrs[rank] = base_ptr
    try:
        for peer in range(world_size):
            if peer == rank:
                continue
            peer_real_fd, peer_shadow_fd, peer_runtime_state_fd = received_fds[peer]
            ptr = import_allocation_handles(
                peer_real_fd,
                peer_shadow_fd,
                int(metas[peer]["alloc_size"]),
                device_index,
                ShareableHandleType.POSIX_FILE_DESCRIPTOR,
            )
            peer_ptrs[peer] = ptr
            if peer_runtime_state_fd is not None:
                peer_runtime_state_alloc_size = metas[peer]["runtime_state_alloc_size"]
                if peer_runtime_state_alloc_size is None:
                    raise RuntimeError(
                        f"rendezvous: rank {peer} sent a runtime-state FD but did not publish runtime_state_alloc_size")
                import_runtime_state_handle(
                    peer_runtime_state_fd,
                    int(peer_runtime_state_alloc_size),
                    peer,
                    device_index,
                    ShareableHandleType.POSIX_FILE_DESCRIPTOR,
                )
    except Exception:
        for peer, ptr in enumerate(peer_ptrs):
            if peer != rank:
                free_allocation(ptr, device_index)
        raise
    finally:
        pending_fds = tuple(received_fds.values())
        for peer_real_fd, peer_shadow_fd, peer_runtime_state_fd in pending_fds:
            os.close(peer_real_fd)
            os.close(peer_shadow_fd)
            if peer_runtime_state_fd is not None:
                os.close(peer_runtime_state_fd)
    return tuple(peer_ptrs)


class GSanSymmetricMemoryHandle:

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        rank: int,
        world_size: int,
        device_index: int,
        buffer_size: int,
        peer_ptrs: tuple[int, ...],
        peer_device_indices: tuple[int, ...],
        cache_key: _RendezvousCacheKey | None = None,
    ):
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._device_index = device_index
        self._buffer_size = buffer_size
        self._peer_ptrs = tuple(peer_ptrs)
        self._peer_device_indices = tuple(int(v) for v in peer_device_indices)
        self._cache_key = cache_key
        self._closed = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None:
        if self._closed:
            raise RuntimeError("GSanSymmetricMemoryHandle has been closed.")
        if channel != 0:
            raise NotImplementedError("Only channel=0 is supported in GSan symmetric memory.")
        _ = timeout_ms
        if self._world_size > 1:
            torch.cuda.synchronize(self._device_index)
            dist.barrier(group=self._group)
            _stream_sync.synchronize_process_group_barrier(self._device_index, self._peer_device_indices)
            torch.cuda.synchronize(self._device_index)
            dist.barrier(group=self._group)

    def get_buffer(
        self,
        rank: int,
        sizes: Sequence[int],
        dtype: torch.dtype,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("GSanSymmetricMemoryHandle has been closed.")
        if rank < 0 or rank >= self._world_size:
            raise ValueError(f"Invalid peer rank: {rank}")
        if storage_offset < 0:
            raise ValueError(f"storage_offset must be >= 0, got {storage_offset}")

        shape = tuple(int(v) for v in sizes)
        element_size = torch.empty((), dtype=dtype).element_size()
        offset_bytes = storage_offset * element_size
        req_bytes = math.prod(shape) * element_size
        if offset_bytes + req_bytes > self._buffer_size:
            raise ValueError(
                f"Requested slice ({offset_bytes + req_bytes} bytes) exceeds buffer size {self._buffer_size} bytes.")

        base_ptr = self._peer_ptrs[rank]
        if base_ptr == 0:
            raise RuntimeError(f"Peer rank {rank} has no mapped buffer.")

        byte_tensor = uint8_cuda_tensor_from_ptr(base_ptr + offset_bytes, req_bytes, self._device_index)
        return byte_tensor.view(dtype=dtype).reshape(shape)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rank, ptr in enumerate(self._peer_ptrs):
            if rank != self._rank:
                free_allocation(ptr, self._device_index)
        if self._cache_key is not None:
            _RENDEZVOUS_CACHE.pop(self._cache_key)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _connect(path: Path, timeout: float = 10) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(remaining)
            sock.connect(str(path))
            sock.settimeout(timeout)
            return sock
        except OSError as err:
            sock.close()
            if err.errno not in _RETRYABLE_UNIX_CONNECT_ERRNOS:
                raise
            last_error = err

        time.sleep(.001)
    raise TimeoutError(f"Timed out connecting to Unix socket at {path}") from last_error


_RENDEZVOUS_CACHE: weakref.WeakValueDictionary[_RendezvousCacheKey,
                                               GSanSymmetricMemoryHandle] = weakref.WeakValueDictionary()
_RUNTIME_BOOTSTRAP_CACHE: dict[_RuntimeBootstrapCacheKey, set[int]] = {}


def rendezvous(tensor: torch.Tensor, group) -> GSanSymmetricMemoryHandle:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("rendezvous: tensor must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise RuntimeError("rendezvous: tensor must be on CUDA device")
    if tensor.storage_offset() != 0:
        raise RuntimeError("rendezvous: tensor must have storage_offset() == 0")
    if not tensor.is_contiguous():
        raise RuntimeError("rendezvous: tensor must be contiguous")

    process_group, _ = _resolve_group(group)
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    device_index = tensor.device.index
    if device_index is None:
        raise RuntimeError("rendezvous: CUDA tensor must have concrete device index")

    storage = tensor.untyped_storage()
    base_ptr = storage.data_ptr()
    storage_key = int(getattr(storage, "_cdata", base_ptr))
    cache_key = (base_ptr, storage_key, id(process_group))
    cached = _RENDEZVOUS_CACHE.get(cache_key)
    if cached is not None and not cached._closed:
        return cached
    _RENDEZVOUS_CACHE.pop(cache_key, None)

    buffer_size = tensor.untyped_storage().nbytes()

    if world_size == 1:
        handle = GSanSymmetricMemoryHandle(
            group=process_group,
            rank=rank,
            world_size=world_size,
            device_index=device_index,
            buffer_size=buffer_size,
            peer_ptrs=(base_ptr, ),
            peer_device_indices=(rank, ),
            cache_key=cache_key,
        )
        _RENDEZVOUS_CACHE[cache_key] = handle
        return handle

    runtime_bootstrap_cache_key = (id(process_group), int(device_index))
    runtime_bootstrapped_peers = _RUNTIME_BOOTSTRAP_CACHE.setdefault(runtime_bootstrap_cache_key, set())
    peers_needing_runtime_bootstrap = {
        peer
        for peer in range(world_size)
        if peer != rank and peer not in runtime_bootstrapped_peers
    }

    with contextlib.ExitStack() as stack:
        real_fd, shadow_fd, alloc_size = export_allocation_handles(base_ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR)

        stack.callback(os.close, real_fd)
        stack.callback(os.close, shadow_fd)
        runtime_state_fd: int | None = None
        runtime_state_alloc_size: int | None = None
        if peers_needing_runtime_bootstrap:
            runtime_state_fd, runtime_state_alloc_size = export_runtime_state_handle(
                int(device_index), ShareableHandleType.POSIX_FILE_DESCRIPTOR)
            stack.callback(os.close, runtime_state_fd)

        local_meta = {
            "hostname": socket.gethostname(),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device_type": tensor.device.type,
            "device_index": int(device_index),
            "nbytes": buffer_size,
            "alloc_size": int(alloc_size),
            "runtime_state_alloc_size": (None if runtime_state_alloc_size is None else int(runtime_state_alloc_size)),
        }

        metas: list[dict] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(metas, local_meta, group=process_group)

        first = metas[0]
        for i, meta in enumerate(metas):
            if meta["hostname"] != first["hostname"]:
                raise RuntimeError(
                    f"rendezvous: rank {i} is on host {meta['hostname']}, expected single-node host {first['hostname']}"
                )
            if meta["shape"] != first["shape"] or meta["dtype"] != first["dtype"]:
                raise RuntimeError("rendezvous: all ranks must use tensors with identical shape and dtype.")
            if meta["device_type"] != "cuda":
                raise RuntimeError("rendezvous: all ranks must use CUDA tensors.")
            if meta["nbytes"] != first["nbytes"]:
                raise RuntimeError("rendezvous: all ranks must use tensors with identical byte size.")
            if meta["alloc_size"] != first["alloc_size"]:
                raise RuntimeError("rendezvous: all ranks must use identical GSan allocation sizes.")
            runtime_meta_size = meta["runtime_state_alloc_size"]
            if runtime_meta_size is not None and int(runtime_meta_size) <= 0:
                raise RuntimeError("rendezvous: runtime_state_alloc_size must be > 0 when provided.")
        peer_device_indices = tuple(range(world_size))

        token_holder = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(token_holder, group=process_group, group_src=0)
        token = str(token_holder[0])

        def make_socket_path(rank):
